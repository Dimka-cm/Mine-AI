"""
Тесты пиксельного цикла обучения.

Главное, что они стерегут: связка «обученные глаза -> RL на пикселях» не
должна разойтись молча. Раньше её не существовало вовсе — train.py учил
символьную модель, а vision.pt относился к пиксельной, и подключить одно
к другому было нечем.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

import train_pixel as TP
from brain.algos.pixel_buffer import PixelReplayBuffer, pixels_to_tensor
from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.env.pixel_env import PixelVisionEnv
from brain.pixel_model import PixelBrain, PixelConfig
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed as seed_rules
from brain.spaces import DENSE_DIM, N_ACTIONS


def _env(w=96, h=72, n=2):
    tmp = tempfile.mkdtemp()
    db = RewardDB(str(Path(tmp) / "p.db"))
    seed_rules(db)
    base = MinecraftCraftEnv(RewardEngine(db), max_steps=12, seed=2, mobs=False)
    return PixelVisionEnv(base, width=w, height=h, n_frames=n)


def test_goals_is_list_of_names():
    """
    GOALS — список строк, не словарей. Обращение GOALS[i]['name'] роняло
    цикл на первом же логировании; тест фиксирует реальную форму данных.
    """
    assert isinstance(GOALS, list) and GOALS
    assert all(isinstance(g, str) for g in GOALS)


def test_single_obs_to_torch_shapes():
    """Наблюдение из среды превращается в батч размера 1 нужных типов."""
    env = _env()
    obs = env.reset()
    t = TP.single_obs_to_torch(obs, "cpu")
    assert t["pixels"].shape[0] == 1
    assert t["pixels"].dtype == torch.float32
    assert t["pixels"].max() <= 1.0 + 1e-6      # нормировано в [0,1]
    assert t["dense"].shape == (1, DENSE_DIM)
    assert t["grid"].dtype == torch.int64
    assert t["held"].shape == (1, 1)


def test_model_accepts_env_observation():
    """Среда и модель стыкуются без ручных переделок формы."""
    env = _env(96, 72, 2)
    obs = env.reset()
    m = PixelBrain(PixelConfig(width=96, height=72, n_frames=2, head="dqn"))
    q = m.q_values(TP.single_obs_to_torch(obs, "cpu"))
    assert q.shape == (1, N_ACTIONS)
    assert torch.isfinite(q).all()


def test_buffer_roundtrip_matches_model():
    """
    Батч из буфера скармливается модели как есть. Буфер отдаёт кадры
    каналом ПОСЛЕДНИМ (B,N,H,W,3) — перестановку делает pixels_to_tensor.
    """
    env = _env(96, 72, 2)
    obs = env.reset()
    buf = PixelReplayBuffer(40, 72, 96, 2, dense_dim=DENSE_DIM,
                            grid_size=9, n_actions=N_ACTIONS)
    buf.new_episode()
    for _ in range(12):
        mask = env.action_mask()
        a = int(np.random.choice(np.flatnonzero(mask)))
        frame = env.render_rgb()
        nobs, r, done, _ = env.step(a)
        buf.push(frame, obs, a, r, done, env.action_mask())
        obs = nobs
        if done:
            break
    b_obs, b_act, b_rew, b_next, b_done, b_nmask = buf.sample(4)
    t = TP.obs_batch_to_torch(b_obs, "cpu")
    assert t["pixels"].shape == (4, 2 * 3, 72, 96)
    m = PixelBrain(PixelConfig(width=96, height=72, n_frames=2, head="dqn"))
    q = m.q_values(t)
    assert q.shape == (4, N_ACTIONS)
    assert b_nmask.shape == (4, N_ACTIONS)


def test_action_mask_blocks_illegal_moves():
    """
    Маска запрещённых ходов — главный щит от галлюцинаций. Выбор действия
    обязан оставаться внутри разрешённых.
    """
    env = _env()
    env.reset()
    mask = env.action_mask()
    q = np.random.randn(N_ACTIONS)
    q[~mask] = -np.inf
    assert mask[int(np.argmax(q))]


def test_vision_weights_reach_pixel_brain(tmp_path):
    """
    Ради этого файл и появился: обученные глаза должны доезжать до боевой
    модели. Проверяем на честно сохранённом чекпоинте, а не на заглушке.
    """
    import pretrain_vision as PV
    net = PV.VisionNet(72, 96)
    with torch.no_grad():
        for p in net.conv.parameters():
            p.add_(0.3)
    ck = tmp_path / "v.pt"
    torch.save({"conv": net.conv.state_dict(), "width": 96, "height": 72,
                "targets": PV.TARGETS, "report": {"facing_acc": 0.8}}, ck)

    brain = PixelBrain(PixelConfig(width=96, height=72, n_frames=2, head="dqn"))
    before = brain.conv[0].weight.clone()
    assert PV.load_into_brain(brain, str(ck), verbose=False)
    assert not torch.allclose(before, brain.conv[0].weight)


def test_first_layer_scaled_for_frame_stack(tmp_path):
    """
    Pretrain видит 3 канала, агент — стек из N кадров. Первый слой
    размножается по стеку и делится на N, иначе активации на старте
    выходят в N раз выше, чем при обучении глаз.
    """
    import pretrain_vision as PV
    n_frames = 4
    net = PV.VisionNet(72, 96)
    ck = tmp_path / "v.pt"
    torch.save({"conv": net.conv.state_dict(), "width": 96, "height": 72,
                "targets": PV.TARGETS, "report": {}}, ck)
    brain = PixelBrain(PixelConfig(width=96, height=72, n_frames=n_frames,
                                   head="dqn"))
    PV.load_into_brain(brain, str(ck), verbose=False)
    src = net.conv[0].weight
    dst = brain.conv[0].weight
    assert dst.shape[1] == src.shape[1] * n_frames
    # сумма по входным каналам сохраняется -> средний отклик тот же
    assert torch.allclose(dst.sum(1), src.sum(1), atol=1e-5)


def test_double_dqn_target_uses_both_nets():
    """
    Double DQN: действие выбирает обучаемая сеть, оценивает целевая.
    Если перепутать, Q систематически завышается — это один из пяти
    каналов галлюцинаций, за которыми мы следим.
    """
    cfg = PixelConfig(width=96, height=72, n_frames=2, head="dqn")
    online, target = PixelBrain(cfg), PixelBrain(cfg)
    with torch.no_grad():
        for p in target.parameters():
            p.mul_(0.5)
    x = {"pixels": torch.rand(3, 6, 72, 96), "dense": torch.zeros(3, DENSE_DIM),
         "grid": torch.zeros(3, 9, dtype=torch.long),
         "held": torch.zeros(3, 1, dtype=torch.long)}
    nm = torch.ones(3, N_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        qo = online.q_values(x).masked_fill(~nm, -float("inf"))
        best = qo.argmax(1, keepdim=True)
        qt = target.q_values(x).gather(1, best).squeeze(1)
        naive = target.q_values(x).max(1).values
    # Оценка по выбору online не должна совпадать с «максимум по target»
    assert not torch.allclose(qt, naive)


def test_pixels_to_tensor_normalises():
    """uint8 -> float [0,1] и канал переезжает на место 1."""
    arr = np.full((2, 3, 8, 10, 3), 255, dtype=np.uint8)
    t = pixels_to_tensor(arr, "cpu")
    assert t.shape == (2, 9, 8, 10)
    assert t.dtype == torch.float32
    assert abs(float(t.max()) - 1.0) < 1e-6


def test_minimum_frame_size():
    """
    Ниже 96x72 свёртки не складываются: после четырёх слоёв остаётся 2x4,
    и ядро 3x3 туда не влезает. Это предел архитектуры, а не случайность —
    фиксируем, чтобы никто не поставил 64x48 и не получил загадочный
    RuntimeError уже в середине обучения.
    """
    import pytest
    ok = PixelBrain(PixelConfig(width=96, height=72, n_frames=1, head="dqn"))
    x = {"pixels": torch.zeros(1, 3, 72, 96), "dense": torch.zeros(1, DENSE_DIM),
         "grid": torch.zeros(1, 9, dtype=torch.long),
         "held": torch.zeros(1, 1, dtype=torch.long)}
    assert ok.q_values(x).shape == (1, N_ACTIONS)

    # 64x48 роняет уже КОНСТРУКТОР: он делает пробный проход, чтобы узнать
    # размер flatten. Падает сразу при создании, а не при первом обучении —
    # это удачно, ошибка видна мгновенно.
    with pytest.raises(RuntimeError):
        PixelBrain(PixelConfig(width=64, height=48, n_frames=1, head="dqn"))

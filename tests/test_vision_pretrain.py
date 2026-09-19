"""
Тесты обучения зрения.

Проверяем не «сеть выучилась» (это долго и шумно), а что задача поставлена
корректно: разметка честная, стены видимые и разные, веса переносятся
в боевую модель без потерь.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import pretrain_vision as PV
from brain.env.mc_env import WORLD_D, WORLD_W, MinecraftCraftEnv
from brain.env.render import Raycaster
from brain.pixel_model import PixelBrain, PixelConfig
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed as seed_rules
from brain.spaces import ID_BLOCK, PASSABLE


@pytest.fixture(scope="module")
def env():
    tmp = tempfile.mkdtemp()
    db = RewardDB(str(Path(tmp) / "v.db"))
    seed_rules(db)
    e = MinecraftCraftEnv(RewardEngine(db), max_steps=90, seed=4, mobs=False)
    e.reset()
    return e


# --- граница мира ---------------------------------------------------------
def test_walls_are_solid(env):
    """Сквозь стену мира нельзя пройти — иначе это не граница."""
    w = env.world
    probes = [w[0, 2, 5], w[WORLD_W - 1, 2, 5], w[5, 2, 0], w[5, 2, WORLD_D - 1]]
    for v in probes:
        assert ID_BLOCK[int(v)] not in PASSABLE, ID_BLOCK[int(v)]


def test_walls_are_distinguishable(env):
    """
    Четыре стороны — четыре РАЗНЫХ материала.
    Если стены одинаковы, вид на север и на юг совпадает, и сторону света
    по картинке определить физически невозможно (замер давал 27% при
    случайных 25%). Этот тест защищает компас от случайной унификации.
    """
    w = env.world
    mats = {
        "-X": ID_BLOCK[int(w[0, 2, 5])],
        "+X": ID_BLOCK[int(w[WORLD_W - 1, 2, 5])],
        "-Z": ID_BLOCK[int(w[5, 2, 0])],
        "+Z": ID_BLOCK[int(w[5, 2, WORLD_D - 1])],
    }
    assert len(set(mats.values())) == 4, mats


def test_walls_are_visible_on_frame(env):
    """Стена должна отличаться от пола на картинке, а не только в числах."""
    rc = Raycaster(128, 96, fov=75.0, max_dist=12.0)
    # встаём вплотную к стене -X и смотрим на неё (facing WEST=1)
    near = rc.render(env.world, 1, 2, 6, 1, 0, [])
    # и в центре мира в ту же сторону
    far = rc.render(env.world, 6, 2, 6, 1, 0, [])
    assert np.abs(near.astype(int) - far.astype(int)).mean() > 5.0


def test_resources_not_inside_walls(env):
    """Руда не должна попадать в стену — иначе она недостижима."""
    w = env.world
    border = np.concatenate([w[0, 2, :], w[WORLD_W - 1, 2, :],
                             w[:, 2, 0], w[:, 2, WORLD_D - 1]])
    names = {ID_BLOCK[int(v)] for v in border}
    for ore in ("oak_log", "iron_ore", "diamond_ore", "stone"):
        assert ore not in names, f"{ore} оказался в стене"


# --- разметка -------------------------------------------------------------
def test_wall_distances_are_relative(env):
    """
    Расстояния до стен считаются в системе взгляда: повернулся — значения
    переехали, а не остались привязаны к сторонам света.
    """
    a = PV.wall_distances(2, 6, 0)
    b = PV.wall_distances(2, 6, 1)
    assert a != b
    for v in a + b:
        assert 0.0 <= v <= 1.0


def test_wall_distance_zero_at_wall():
    """У самой стены расстояние до неё минимально, у дальней — максимально."""
    d = PV.wall_distances(1, 6, 1)          # почти вплотную к -X, смотрим туда
    assert d[0] < 0.15, d                    # вперёд — стена рядом
    assert d[2] > 0.8, d                     # назад — далеко


def test_targets_in_unit_range(env):
    """Вся разметка нормирована в [0,1] — иначе MSE перекосит на одну цель."""
    rc = Raycaster(64, 48, fov=75.0, max_dist=12.0)
    rng = np.random.default_rng(0)
    X, Y, FA = PV.collect(12, rc, env, rng, 48, 64, verbose=False)
    assert X.shape == (12, 3, 48, 64)
    assert Y.shape == (12, len(PV.TARGETS))
    assert Y.min() >= -1e-6 and Y.max() <= 1.0 + 1e-6
    assert set(np.unique(FA)) <= {0, 1, 2, 3}


def test_collect_varies_positions(env):
    """Выборка не должна стоять на месте: иначе учить нечему."""
    rc = Raycaster(64, 48, fov=75.0, max_dist=12.0)
    rng = np.random.default_rng(1)
    _, Y, _ = PV.collect(40, rc, env, rng, 48, 64, verbose=False)
    assert Y[:, 0].std() > 0.05 and Y[:, 1].std() > 0.05


# --- модель и перенос весов ----------------------------------------------
def test_vision_encoder_keeps_position():
    """
    Ключевое свойство: свёртки сохраняют ПОЗИЦИЮ признака на экране.
    Усредняющий пулинг её стирает, и ориентация перестаёт выучиваться
    (замер: 27% против 84%). Проверяем, что сдвиг картинки меняет выход.
    """
    net = PV.VisionNet(96, 128)
    x = torch.zeros(1, 3, 96, 128)
    x[:, :, 40:60, 10:30] = 1.0            # пятно слева
    y = torch.zeros(1, 3, 96, 128)
    y[:, :, 40:60, 95:115] = 1.0           # то же пятно справа
    with torch.no_grad():
        a, b = net.conv(x), net.conv(y)
    assert (a - b).abs().mean() > 1e-4


def test_encoder_matches_pixelbrain():
    """
    Архитектура глаз в pretrain и в боевой модели обязана совпадать,
    иначе обученные веса некуда переносить.
    """
    cfg = PixelConfig(width=128, height=96, n_frames=1)
    brain = PixelBrain(cfg)
    net = PV.VisionNet(96, 128, c_in=3)
    bk = [k for k, _ in brain.conv.state_dict().items()]
    nk = [k for k, _ in net.conv.state_dict().items()]
    assert bk == nk, (bk, nk)
    for k in bk:
        assert brain.conv.state_dict()[k].shape == net.conv.state_dict()[k].shape, k


def test_weight_transfer_changes_brain():
    """Перенос весов действительно меняет боевую модель."""
    cfg = PixelConfig(width=128, height=96, n_frames=1)
    brain = PixelBrain(cfg)
    net = PV.VisionNet(96, 128, c_in=3)
    with torch.no_grad():
        for p in net.conv.parameters():
            p.add_(0.5)
    before = brain.conv[0].weight.clone()
    brain.conv.load_state_dict(net.conv.state_dict())
    assert not torch.allclose(before, brain.conv[0].weight)


def test_forward_shapes():
    net = PV.VisionNet(96, 128)
    reg, fac = net(torch.zeros(2, 3, 96, 128))
    assert reg.shape == (2, len(PV.TARGETS))
    assert fac.shape == (2, 4)


# --- качество картинки ----------------------------------------------------
def test_frame_quality_thresholds(env):
    """
    Приёмочные пороги кадра: детализация >= 7, цветов >= 53 000.

    Детализация меряется средним градиентом яркости — той же мерой, что и
    референсные изображения (их значения 4.68 / 11.26 / 7.88 она
    воспроизводит с точностью до десятых).

    Важно, КАК порог достигнут. Поднять цифру амплитудой шума легко:
    разброс x2.6 даёт 16.35, но камень превращается в телевизионный снег.
    Честный путь — мельче тексель (TEX=192): рисунок тот же, он просто не
    растягивается по крупному блоку. Этот тест ловит падение качества,
    но не мешает поднимать его по делу.
    """
    from PIL import Image

    from brain.env.mc_env import WORLD_D, WORLD_W

    # Меряем по многим ракурсам, а не по паре. Детализация сильно зависит
    # от того, куда смотрит агент: упёрся носом в стену — 3.8 (там нечего
    # детализировать, и это не дефект), видит поле с блоками — 14.9.
    # Порог проверяем по среднему, иначе тест меряет удачу выбора ракурса.
    rc = Raycaster(640, 360, fov=75.0, max_dist=12.0)
    rng = np.random.default_rng(0)
    dets, cols = [], []
    for i in range(24):
        if i % 8 == 0:
            env.reset()
        for _ in range(50):
            x = int(rng.integers(1, WORLD_W - 1))
            z = int(rng.integers(1, WORLD_D - 1))
            if env.world[x, 2, z] == 0:
                break
        im = rc.render(env.world, x, 2, z, int(rng.integers(0, 4)), 0, [])
        g = np.asarray(Image.fromarray(im).convert("L"), np.float32)
        dets.append((np.abs(np.diff(g, axis=1)).mean()
                     + np.abs(np.diff(g, axis=0)).mean()) / 2)
        cols.append(len(np.unique(im.reshape(-1, 3), axis=0)))
    assert np.mean(dets) >= 7.0, f"детализация {np.mean(dets):.2f} < 7"
    assert np.mean(cols) >= 53_000, f"цветов {int(np.mean(cols))} < 53000"


def test_texture_amplitude_not_inflated():
    """
    Зерно не должно быть пережато. Атлас — множители яркости; если разброс
    уполз далеко за 0.70..1.30, значит кто-то поднимал детализацию шумом.
    """
    from brain.env.render import _ATLAS
    assert _ATLAS.min() >= 0.60, f"слишком тёмное зерно: {_ATLAS.min():.2f}"
    assert _ATLAS.max() <= 1.40, f"слишком яркое зерно: {_ATLAS.max():.2f}"

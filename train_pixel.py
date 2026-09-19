#!/usr/bin/env python3
"""
ОБУЧЕНИЕ НА ПИКСЕЛЯХ — недостающее звено между зрением и наградами.

Зачем отдельный файл. train.py обучает символьную модель CraftBrain: она
получает мир числами и глаз не имеет вовсе. Обученные глаза из
pretrain_vision.py относятся к пиксельной PixelBrain, и подключить их было
некуда — цикла на пикселях просто не существовало. Этот файл его добавляет.

Что здесь соединяется:
    PixelVisionEnv     — мир картинкой 426x240, инвентарь названиями
    PixelBrain         — свёртки (глаза) + эмбеддинги предметов
    PixelReplayBuffer  — хранит ОДИНОЧНЫЕ кадры, стек собирает при выборке
    vision.pt          — обученные глаза, старт не со случайных весов

Порядок работы:
    python3 pretrain_vision.py --samples 6000 --epochs 25   # сначала глаза
    python3 train_pixel.py --episodes 5000 --vision vision.pt

Проверить, что всё сходится, без долгого прогона:
    python3 train_pixel.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brain.algos.pixel_buffer import PixelReplayBuffer, pixels_to_tensor
from brain.cpu_limit import apply_torch_limits, reserve_cores
from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.env.pixel_env import PixelVisionEnv
from brain.pixel_model import PixelBrain, PixelConfig
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed as seed_rules
from brain.spaces import N_ACTIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Обучение пиксельного агента на системе наград")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=90)
    p.add_argument("--goal", type=int, default=0, help="стартовая цель 0..13")
    p.add_argument("--curriculum", action="store_true",
                   help="вести по целям от простых к сложным")
    p.add_argument("--vision", default="vision.pt",
                   help="обученные глаза; пусто — начать со случайных")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer", type=int, default=20_000)
    p.add_argument("--buffer-on-disk", action="store_true",
                   help="держать кадры в файле, а не в ОЗУ")
    p.add_argument("--warmup", type=int, default=1000,
                   help="шагов случайной игры до начала обучения")
    p.add_argument("--train-every", type=int, default=4,
                   help="батч обучения раз в N шагов (4 — стандарт DQN)")
    p.add_argument("--target-sync", type=int, default=1000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=50_000)
    p.add_argument("--freeze-vision", type=int, default=0,
                   help="первые N шагов не трогать обученные свёртки")
    p.add_argument("--cores", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--db", default="data/pixel_rewards.db")
    p.add_argument("--save", default="checkpoints/pixel_brain.pt")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--progress", default="data/pixel_progress.jsonl")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="короткая проверка, что цикл работает")
    return p.parse_args()


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def obs_batch_to_torch(obs: dict, device: str) -> dict:
    """Батч из буфера -> тензоры на устройстве. Пиксели переводит GPU."""
    return {
        "pixels": pixels_to_tensor(obs["pixels"], device),
        "dense": torch.from_numpy(obs["dense"]).to(device),
        "grid": torch.from_numpy(obs["grid"]).to(device),
        "held": torch.from_numpy(obs["held"]).to(device),
    }


def single_obs_to_torch(obs: dict, device: str) -> dict:
    """Одно наблюдение из среды -> батч размера 1."""
    px = obs["pixels"]
    if px.ndim == 3:                      # (N*3, H, W) — уже сложенный стек
        t = torch.from_numpy(px).unsqueeze(0).to(device)
        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
    else:
        t = pixels_to_tensor(px[None], device)
    return {
        "pixels": t,
        "dense": torch.from_numpy(obs["dense"][None]).float().to(device),
        "grid": torch.from_numpy(obs["grid"][None]).long().to(device),
        "held": torch.from_numpy(np.asarray(obs["held"]).reshape(1, 1)).long().to(device),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        # Быстрая проверка механики: маленький кадр, десяток эпизодов.
        args.episodes, args.max_steps = 6, 20
        args.width, args.height = 128, 96
        args.buffer, args.warmup, args.batch = 400, 40, 8
        args.train_every, args.log_every, args.save_every = 2, 2, 10_000

    reserve_cores(args.cores)
    apply_torch_limits()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.progress).parent.mkdir(parents=True, exist_ok=True)

    db = RewardDB(args.db)
    seed_rules(db)
    engine = RewardEngine(db)
    base = MinecraftCraftEnv(engine, max_steps=args.max_steps, seed=args.seed)
    env = PixelVisionEnv(base, width=args.width, height=args.height,
                         n_frames=args.n_frames)

    cfg = PixelConfig(width=args.width, height=args.height,
                      n_frames=args.n_frames, head="dqn")
    model = PixelBrain(cfg).to(device)
    target = PixelBrain(cfg).to(device)

    # --- обученные глаза ---------------------------------------------------
    if args.vision and Path(args.vision).exists():
        from pretrain_vision import load_into_brain
        load_into_brain(model, args.vision)
    elif args.vision:
        print(f"[зрение] {args.vision} не найден — глаза случайные. "
              f"Сначала: python3 pretrain_vision.py")

    target.load_state_dict(model.state_dict())
    target.eval()

    spec = PixelReplayBuffer.estimate(args.buffer, args.height, args.width,
                                      args.n_frames)
    print(f"[модель]  {model.n_params():,} параметров, устройство {device}")
    print(f"[буфер]   {spec.describe() if hasattr(spec, 'describe') else spec}")
    buf = PixelReplayBuffer(args.buffer, args.height, args.width,
                            args.n_frames, dense_dim=model.cfg_dense_dim()
                            if hasattr(model, "cfg_dense_dim") else
                            env.reset()["dense"].shape[0],
                            grid_size=9, n_actions=N_ACTIONS,
                            on_disk=args.buffer_on_disk)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    goal = args.goal
    step_count = 0
    history: list[float] = []
    t0 = time.perf_counter()
    prog = open(args.progress, "a", buffering=1)

    for ep in range(1, args.episodes + 1):
        obs = env.reset(goal_index=goal)
        buf.new_episode()
        total_r = 0.0
        done = False

        while not done:
            # --- выбор действия (epsilon-greedy с маской) ------------------
            eps = max(args.eps_end, args.eps_start -
                      (args.eps_start - args.eps_end) * step_count / max(args.eps_decay, 1))
            mask = env.action_mask()
            if np.random.rand() < eps or step_count < args.warmup:
                legal = np.flatnonzero(mask)
                action = int(np.random.choice(legal)) if len(legal) else 0
            else:
                model.eval()
                with torch.no_grad():
                    q = model.q_values(single_obs_to_torch(obs, device))[0]
                q = q.cpu().numpy()
                q[~mask] = -np.inf          # запрещённые ходы не выбираем
                action = int(np.argmax(q))

            frame = env.render_rgb()        # (H, W, 3) uint8 — свежий кадр
            nobs, r, done, info = env.step(action)
            buf.push(frame, obs, action, r, done, env.action_mask())
            obs = nobs
            total_r += r
            step_count += 1

            # --- шаг обучения ---------------------------------------------
            if (step_count >= args.warmup and len(buf) > args.batch
                    and step_count % args.train_every == 0):
                model.train()
                b_obs, b_act, b_rew, b_next, b_done, b_nmask = buf.sample(args.batch)
                to = obs_batch_to_torch(b_obs, device)
                tn = obs_batch_to_torch(b_next, device)
                a = torch.from_numpy(b_act).long().to(device)
                rw = torch.from_numpy(b_rew).float().to(device)
                dn = torch.from_numpy(b_done).float().to(device)
                nm = torch.from_numpy(b_nmask).to(device)

                q = model.q_values(to).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    # Double DQN: действие выбирает обучаемая сеть,
                    # оценивает целевая. Иначе Q систематически завышается.
                    qn_online = model.q_values(tn)
                    qn_online = qn_online.masked_fill(~nm, -float("inf"))
                    best = qn_online.argmax(1, keepdim=True)
                    qn = target.q_values(tn).gather(1, best).squeeze(1)
                    tgt = rw + args.gamma * qn * (1.0 - dn)

                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if step_count < args.freeze_vision:
                    # Обученные глаза не портим, пока головы не осмыслятся.
                    for p in model.conv.parameters():
                        p.grad = None
                opt.step()

                if step_count % args.target_sync == 0:
                    target.load_state_dict(model.state_dict())

        history.append(total_r)
        if ep % args.log_every == 0:
            avg = float(np.mean(history[-args.log_every:]))
            el = time.perf_counter() - t0
            print(f"эп {ep:5d} | цель {GOALS[goal]:16s} | "
                  f"награда {avg:7.2f} | eps {eps:.3f} | "
                  f"шагов {step_count:6d} | {el / 60:.1f} мин", flush=True)
            prog.write(json.dumps({"episode": ep, "goal": GOALS[goal],
                                   "reward": round(avg, 3), "eps": round(eps, 4),
                                   "steps": step_count}, ensure_ascii=False) + "\n")

        if ep % args.save_every == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                        "episode": ep}, args.save)

        # --- переход к следующей цели -------------------------------------
        if args.curriculum and len(history) >= 50:
            if float(np.mean(history[-50:])) > 0 and goal < len(GOALS) - 1:
                goal += 1
                history.clear()
                print(f"[цель] перехожу к: {GOALS[goal]}")

    torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                "episode": args.episodes}, args.save)
    buf.close()
    prog.close()
    print(f"\nГотово. Веса: {args.save}")
    print(f"Заняло: {(time.perf_counter() - t0) / 60:.1f} мин, "
          f"шагов среды: {step_count:,}")


if __name__ == "__main__":
    main()

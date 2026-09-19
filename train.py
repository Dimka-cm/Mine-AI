#!/usr/bin/env python3
"""
Цикл обучения агента.

Примеры:
    python train.py --algo dqn --episodes 300
    python train.py --algo ppo --episodes 300 --curriculum
    python train.py --algo dqn --episodes 50 --reset-db --verbose
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from brain.algos.dqn import DQNConfig, DQNTrainer
from brain.algos.ppo import PPOConfig, PPOTrainer
from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.model import CraftBrain, ModelConfig
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение Minecraft-агента на наградах")
    p.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--db", default="data/rewards.db")
    p.add_argument("--reset-db", action="store_true", help="забыть историю наград")
    p.add_argument("--curriculum", action="store_true",
                   help="автоматически повышать цель при успехах")
    p.add_argument("--goal", type=int, default=0, help="стартовая цель (индекс)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="checkpoints/brain.pt")
    p.add_argument("--load", default="")
    p.add_argument("--metrics", default="data/metrics.json")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--train-every", type=int, default=1,
                   help="делать градиентный шаг раз в N шагов среды (ускорение)")
    p.add_argument("--promote-window", type=int, default=20,
                   help="окно эпизодов ТЕКУЩЕЙ цели для повышения curriculum")
    p.add_argument("--promote-rate", type=float, default=0.7,
                   help="доля успехов в окне, нужная для перехода к след. цели")
    p.add_argument("--promote-patience", type=int, default=900,
                   help="макс. эпизодов на одной цели; потом переходим дальше "
                        "принудительно, чтобы увидеть верхние тиры")
    p.add_argument("--revisit", type=float, default=0.15,
                   help="доля эпизодов со случайной ПРОЙДЕННОЙ целью "
                        "(защита от забывания)")
    p.add_argument("--eps-restart", type=float, default=0.35,
                   help="до какого eps подскочить при смене цели (0 = не трогать)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--progress", default="data/progress.jsonl",
                   help="построчный лог прогресса для мониторинга")
    p.add_argument("--eps-decay", type=int, default=0,
                   help="шагов до минимального eps (0 = авто, 80k)")
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    db = RewardDB(args.db)
    if args.reset_db:
        db.reset_progress(keep_rules=True)
        print("[db] история наград очищена")
    seed(db)
    print(f"[db] правил в базе: {len(db.all_rules())}")

    engine = RewardEngine(db)
    env = MinecraftCraftEnv(engine, max_steps=args.max_steps, seed=args.seed,
                            goal_index=args.goal)

    model = CraftBrain(ModelConfig(head="both"))
    if args.load and Path(args.load).exists():
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
        print(f"[model] загружены веса {args.load}")
    else:
        print("[model] новая ПУСТАЯ модель (случайная инициализация)")
    print(f"[model] параметров: {model.n_params():,}")

    if args.algo == "dqn":
        # Длинный прогон требует длинного расписания исследования: иначе
        # epsilon упрётся в минимум на первых процентах обучения.
        # Расписание исследования калибруется по ЦЕЛИ curriculum, а не по
        # длине всего прогона: иначе на 15k эпизодов eps застревает около 0.9
        # и агент тысячи эпизодов играет вслепую. При смене цели eps частично
        # восстанавливается (--eps-restart), так что общий запас исследования
        # всё равно растянут на весь прогон.
        est_steps = args.episodes * args.max_steps
        dcfg = DQNConfig(
            eps_decay_steps=args.eps_decay or 80_000,
            buffer_size=max(50_000, min(200_000, est_steps // 6)),
        )
        trainer = DQNTrainer(model, dcfg)
        print(f"[dqn] eps_decay={dcfg.eps_decay_steps:,} "
              f"buffer={dcfg.buffer_size:,} train_every={args.train_every}")
    else:
        trainer = PPOTrainer(model, PPOConfig())

    history, successes, t0 = [], 0, time.time()
    goal_idx = args.goal
    # Успехи считаем ОТДЕЛЬНО для каждой цели: иначе окно смешивает эпизоды
    # старой и новой цели и curriculum скачет вперёд по чужим заслугам.
    goal_hist: dict = {}
    goal_first_ep: dict = {goal_idx: 1}
    env_step_counter = 0
    prog_path = Path(args.progress)
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    prog_f = prog_path.open("w", buffering=1)

    rng = np.random.default_rng(args.seed)
    for ep in range(1, args.episodes + 1):
        # Иногда возвращаемся к уже пройденной цели, иначе сеть забывает
        # старые рецепты, пока учит новые (catastrophic forgetting).
        ep_goal = goal_idx
        if args.revisit > 0 and goal_idx > 0 and rng.random() < args.revisit:
            ep_goal = int(rng.integers(0, goal_idx))
        obs = env.reset(goal_index=ep_goal)
        total, steps, done = 0.0, 0, False

        while not done:
            mask = env.action_mask()
            if args.algo == "dqn":
                a = trainer.act(obs, mask)
                nobs, r, done, info = env.step(a)
                trainer.buffer.push(obs, a, r, nobs, done, env.action_mask())
                env_step_counter += 1
                if env_step_counter % args.train_every == 0:
                    trainer.learn()
            else:
                a, lp, v = trainer.act(obs, mask)
                nobs, r, done, info = env.step(a)
                trainer.buf.add(obs, a, lp, r, v, done)
                if len(trainer.buf) >= trainer.cfg.rollout:
                    trainer.update(0.0 if done else v)
            obs = nobs
            total += r
            steps += 1

        if args.algo == "ppo" and len(trainer.buf) > 0:
            trainer.update(0.0)

        reached = "goal_reached" in info
        successes += int(reached)
        gname = GOALS[min(ep_goal, len(GOALS) - 1)]
        history.append({"episode": ep, "reward": round(total, 2), "steps": steps,
                        "goal": gname, "success": reached})
        goal_hist.setdefault(ep_goal, []).append(
            {"reward": total, "success": reached})
        is_revisit = ep_goal != goal_idx

        # --- повышение цели: только по статистике ТЕКУЩЕЙ цели
        if args.curriculum and goal_idx < len(GOALS) - 1 and not is_revisit:
            gh = goal_hist[goal_idx]
            win = gh[-args.promote_window:]
            spent = ep - goal_first_ep.get(goal_idx, 1)
            forced = spent >= args.promote_patience
            if len(win) >= args.promote_window or forced:
                rate = sum(w["success"] for w in win) / max(1, len(win))
                if rate >= args.promote_rate or forced:
                    if forced and rate < args.promote_rate:
                        print(f"  ⏭ PATIENCE (эп {ep}): {spent} эп на "
                              f"{GOALS[goal_idx]} при {rate:.0%} — идём дальше")
                    goal_idx += 1
                    goal_first_ep[goal_idx] = ep + 1
                    if args.algo == "dqn" and args.eps_restart > 0:
                        # Новая цель = новая задача: возвращаем немного
                        # исследования, иначе агент застревает в старой привычке.
                        target = args.eps_restart
                        c = trainer.cfg
                        frac = (c.eps_start - target) / (c.eps_start - c.eps_end)
                        trainer.steps = min(trainer.steps,
                                            int(frac * c.eps_decay_steps))
                    print(f"  ↑ CURRICULUM (эп {ep}): {rate:.0%} успеха -> "
                          f"новая цель {GOALS[goal_idx]}")

        if ep % args.log_every == 0 or args.verbose:
            gh = goal_hist.get(goal_idx, [{"reward": 0, "success": False}])[-args.log_every:]
            avg = sum(h["reward"] for h in gh) / max(1, len(gh))
            sr = sum(h["success"] for h in gh) / max(1, len(gh))
            extra = f" eps={trainer.epsilon():.3f}" if args.algo == "dqn" else ""
            el = time.time() - t0
            eta = el / ep * (args.episodes - ep)
            cur = GOALS[min(goal_idx, len(GOALS) - 1)]
            print(f"эп {ep:5d}/{args.episodes} | цель={cur:<15} "
                  f"| ср.награда={avg:9.2f} | успех={sr:4.0%}{extra} "
                  f"| {el/60:.0f}м прошло, ~{eta/60:.0f}м осталось", flush=True)
            prog_f.write(json.dumps({
                "ep": ep, "goal": gname, "goal_idx": goal_idx,
                "avg_reward": round(avg, 2), "success_rate": round(sr, 3),
                "eps": round(trainer.epsilon(), 3) if args.algo == "dqn" else None,
                "elapsed_s": round(el), "total_successes": successes,
            }, ensure_ascii=False) + "\n")

        if ep % args.save_every == 0:
            torch.save(model.state_dict(), args.save)

    dt = time.time() - t0
    prog_f.close()
    print(f"\nГотово за {dt/60:.1f} мин. Успешных эпизодов: {successes}/{args.episodes}")
    print(f"Достигнутая цель curriculum: {GOALS[min(goal_idx, len(GOALS)-1)]} "
          f"({goal_idx + 1}/{len(GOALS)})")
    print("\nПрогресс по целям:")
    for gi in sorted(goal_hist):
        gh = goal_hist[gi]
        sr = sum(h["success"] for h in gh) / max(1, len(gh))
        best = max((h["reward"] for h in gh), default=0)
        print(f"  {GOALS[gi]:<16} эпизодов={len(gh):<6} успех={sr:5.1%} "
              f"лучшая награда={best:+8.1f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"[model] сохранено -> {args.save}")

    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(history, ensure_ascii=False, indent=1))
    print(f"[metrics] -> {args.metrics}")

    print("\nТоп правил по накопленной награде:")
    for row in db.top_rules(12):
        print(f"  {row['ctx_key']:<42} x{row['times']:<4} = {row['total_value']:+8.2f}")
    db.close()


if __name__ == "__main__":
    main()

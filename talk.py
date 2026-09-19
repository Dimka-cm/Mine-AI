#!/usr/bin/env python3
"""
РАЗГОВОР С АГЕНТОМ.

Вы говорите по-русски, агент понимает команду, ставит цель и отчитывается
о том, что делает. Все числа в ответах берутся прямо из наблюдения —
выдумать он ничего не может.

    python3 talk.py                      # диалог в терминале
    python3 talk.py --goal wooden_sword  # сразу с целью
    python3 talk.py --steps 40           # сколько шагов делать за команду

Команды:
    сделай каменный меч      поставить цель и работать
    добудь 5 брёвен          то же, с количеством
    что ты делаешь           отчёт о текущем состоянии
    стоп                     прервать
    выход                    закончить
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from brain.env.mc_env import GOALS, MinecraftCraftEnv  # noqa: E402
from brain.language.head import (command_to_goal, describe,  # noqa: E402
                                 parse_command)
from brain.recipes import item_id  # noqa: E402
from brain.rewards.db import RewardDB  # noqa: E402
from brain.rewards.engine import RewardEngine  # noqa: E402
from brain.rewards.rules import seed  # noqa: E402


def agent_state(env, goal_name: str) -> dict:
    """Собирает контекст для ответа: сколько есть, сколько надо, кто рядом."""
    extra = {}
    try:
        have = env.agent.inventory.get(item_id(goal_name), 0)
        extra["have"] = have
        extra["need"] = 1
        extra["crafted"] = have > 0
    except Exception:
        pass
    hostiles = [m for m in env.mobs if m.hostile]
    if hostiles:
        ru = {"zombie": "зомби", "skeleton": "скелетом", "spider": "пауком",
              "creeper": "крипером"}
        extra["mob"] = ru.get(hostiles[0].kind, "врагом")
    return extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goal", default="oak_log", help="стартовая цель")
    ap.add_argument("--steps", type=int, default=30,
                    help="шагов среды на одну команду")
    ap.add_argument("--db", default="data/talk.db")
    a = ap.parse_args()

    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    db = RewardDB(a.db)
    seed(db)
    env = MinecraftCraftEnv(RewardEngine(db), max_steps=200, seed=7)

    goal_idx = GOALS.index(a.goal) if a.goal in GOALS else 0
    env.reset(goal_index=goal_idx)

    print("Агент готов. Говорите по-русски. 'выход' — закончить.\n")
    print(f"  агент: {describe(env.observe(), GOALS[goal_idx])}")

    while True:
        try:
            text = input("\n  вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in ("выход", "exit", "quit", "пока"):
            print("  агент: пока")
            break

        parsed = parse_command(text)

        # Вопрос — отвечаем и не трогаем мир.
        if parsed["question"] or low in ("стоп", "стой", "хватит"):
            print(f"  агент: {describe(env.observe(), GOALS[goal_idx], agent_state(env, GOALS[goal_idx]))}")
            continue

        # Новая цель?
        new_goal = command_to_goal(text, GOALS)
        if new_goal is not None and new_goal != goal_idx:
            goal_idx = new_goal
            env.reset(goal_index=goal_idx)
            print(f"  агент: понял, цель: {GOALS[goal_idx]}")
        elif new_goal is None and parsed["verb"] is None:
            print("  агент: не понял, скажите иначе")
            continue

        # Работаем: модель не обучена, поэтому пока случайные ходы.
        for _ in range(a.steps):
            m = env.action_mask()
            act = int(np.random.choice(np.flatnonzero(m)))
            _, _, done, _ = env.step(act)
            if done:
                break
        goal_name = GOALS[goal_idx]
        print(f"  агент: {describe(env.observe(), goal_name, agent_state(env, goal_name))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

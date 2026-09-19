#!/usr/bin/env python3
"""
Демонстрация системы наград — ровно тот сценарий, который описан в задаче.

    палка в нужный слот меча        -> награда
    доска НАД палкой                -> награда больше (пространственная связь)
    форма собрана                   -> ещё награда
    меч скрафчен                    -> крупная награда
    ТОТ ЖЕ меч второй/третий раз    -> награда затухает
    каменный / железный меч         -> считается заново, с полной наградой
    предмет не в ту клетку          -> ШТРАФ
    крафт несобранного рецепта      -> ШТРАФ
    разрушил верную форму           -> ШТРАФ

Запуск: python demo_rewards.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from brain.recipes import RECIPE_BY_NAME
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import EMPTY, item_id

E = EMPTY
STICK = item_id("stick")
PLANK = item_id("oak_planks")
COBBLE = item_id("cobblestone")
IRON = item_id("iron_ingot")
DIRT = item_id("dirt")

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"


def hdr(t: str) -> None:
    print(f"\n{B}{C}{'═' * 74}\n  {t}\n{'═' * 74}{X}")


def show(br, label: str) -> None:
    col = G if br.total > 0 else (R if br.total < 0 else Y)
    print(f"  {label:<44} {col}{B}{br.total:+8.2f}{X}")
    for key, val, reason in br.parts:
        c = G if val > 0 else R
        print(f"      {c}{val:+7.2f}{X}  {key:<34} {reason[:60]}")


def draw(grid) -> None:
    names = {E: "·", STICK: "палка", PLANK: "доска", COBBLE: "камень",
             IRON: "железо", DIRT: "грязь"}
    print(f"      {Y}┌────────┬────────┬────────┐{X}")
    for r in range(3):
        cells = "│".join(names.get(grid[r * 3 + c], "?").center(8) for c in range(3))
        print(f"      {Y}│{X}{cells}{Y}│{X}")
        if r < 2:
            print(f"      {Y}├────────┼────────┼────────┤{X}")
    print(f"      {Y}└────────┴────────┴────────┘{X}")


def main() -> None:
    path = Path("data/demo.db")
    if path.exists():
        path.unlink()
    db = RewardDB(path)
    seed(db)
    eng = RewardEngine(db)
    eng.begin_episode(1)

    # ---------------------------------------------------------------
    hdr("1. СБОРКА ДЕРЕВЯННОГО МЕЧА ПО ШАГАМ")
    grid = [E] * 9
    print(f"\n  {B}Шаг 1: палка в нижнюю центральную клетку (рукоять){X}")
    nxt = list(grid); nxt[7] = STICK
    show(eng.on_place_in_grid(grid, nxt, 7, STICK), "палка -> слот 7"); grid = nxt
    draw(grid)

    print(f"\n  {B}Шаг 2: доска НАД палкой — пространственная связь{X}")
    nxt = list(grid); nxt[4] = PLANK
    show(eng.on_place_in_grid(grid, nxt, 4, PLANK), "доска -> слот 4 (над палкой)")
    grid = nxt
    draw(grid)

    print(f"\n  {B}Шаг 3: вторая доска сверху — форма меча собрана{X}")
    nxt = list(grid); nxt[1] = PLANK
    show(eng.on_place_in_grid(grid, nxt, 1, PLANK), "доска -> слот 1 (остриё)")
    grid = nxt
    draw(grid)

    print(f"\n  {B}Шаг 4: крафт{X}")
    show(eng.on_craft(RECIPE_BY_NAME["wooden_sword"], True), "КРАФТ деревянного меча")

    # ---------------------------------------------------------------
    hdr("2. ШТРАФЫ ЗА НЕПРАВИЛЬНЫЕ ДЕЙСТВИЯ")
    g = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    print(f"\n  {B}Грязь в угол — этой клетки в рецепте нет{X}")
    n = list(g); n[0] = DIRT
    show(eng.on_place_in_grid(g, n, 0, DIRT), "грязь -> слот 0")

    print(f"\n  {B}Убрал рукоять из собранного меча{X}")
    n2 = list(g); n2[7] = E
    show(eng.on_take_from_grid(g, n2, 7), "забрал палку из слота 7")

    print(f"\n  {B}Нажал 'крафт' на несобранном рецепте{X}")
    show(eng.on_craft(None, False), "крафт впустую")

    print(f"\n  {B}Не тот предмет в правильную клетку{X}")
    g3 = [E, E, E, E, E, E, E, STICK, E]
    n3 = list(g3); n3[4] = DIRT
    show(eng.on_place_in_grid(g3, n3, 4, DIRT), "грязь вместо доски -> слот 4")

    # ---------------------------------------------------------------
    hdr("3. ЗАТУХАНИЕ: ОДНО И ТО ЖЕ БОЛЬШЕ НЕ ФАРМИТСЯ")
    print()
    for i in range(1, 7):
        br = eng.on_craft(RECIPE_BY_NAME["wooden_sword"], True)
        bar = "█" * max(0, int(br.total / 1.5))
        col = G if br.total > 5 else Y if br.total > 0 else R
        print(f"  деревянный меч #{i}: {col}{br.total:+8.2f}{X}  {col}{bar}{X}")
    print(f"\n  {Y}→ база помнит счётчик, награда падает. Спам больше не выгоден.{X}")

    # ---------------------------------------------------------------
    hdr("4. НОВЫЙ ТИР — СЧЁТЧИК НАЧИНАЕТСЯ ЗАНОВО")
    print()
    for name in ["stone_sword", "iron_sword", "diamond_sword"]:
        br = eng.on_craft(RECIPE_BY_NAME[name], True)
        bar = "█" * max(0, int(br.total / 3))
        print(f"  {name:<16} первый раз: {G}{br.total:+8.2f}{X}  {G}{bar}{X}")
    print(f"\n  {Y}→ у каждого тира свой счётчик: деревянный выдохся, "
          f"каменный даёт полную награду.{X}")

    print(f"\n  {B}А вот повтор каменного меча уже дешевеет:{X}")
    for i in range(2, 5):
        br = eng.on_craft(RECIPE_BY_NAME["stone_sword"], True)
        print(f"  каменный меч #{i}: {Y}{br.total:+8.2f}{X}")

    # ---------------------------------------------------------------
    hdr("5. ЧТО НАКОПИЛОСЬ В БАЗЕ НАГРАД")
    print()
    print(f"  {B}{'ключ':<44}{'раз':>5}{'итого':>11}{'дальше':>10}{X}")
    print(f"  {'─' * 70}")
    for row in db.top_rules(16):
        ctx = row["ctx_key"].split("|", 1)[1] if "|" in row["ctx_key"] else ""
        _, nxt_val = db.peek(row["rule_key"], ctx)
        col = G if row["total_value"] > 0 else R
        print(f"  {row['ctx_key']:<44}{row['times']:>5}"
              f"{col}{row['total_value']:>11.2f}{X}{nxt_val:>10.2f}")

    print(f"\n  {C}Файл базы: {path}  (SQLite — можно открыть любым клиентом){X}")
    print(f"  {C}Таблицы: reward_rules, reward_state, reward_log, episodes{X}\n")
    db.close()


if __name__ == "__main__":
    main()

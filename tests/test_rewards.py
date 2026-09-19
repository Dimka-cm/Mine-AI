"""Тесты логики наград — то, ради чего всё затевалось."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from brain.recipes import RECIPE_BY_NAME, best_partial, find_completed, match_grid
from brain.rewards.db import RewardDB, Rule
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import EMPTY, item_id


@pytest.fixture()
def engine():
    tmp = Path(tempfile.mkdtemp()) / "t.db"
    db = RewardDB(tmp)
    seed(db)
    eng = RewardEngine(db)
    eng.begin_episode(1)
    yield eng
    db.close()


STICK = item_id("stick")
PLANK = item_id("oak_planks")
COBBLE = item_id("cobblestone")
IRON = item_id("iron_ingot")
DIRT = item_id("dirt")
E = EMPTY


# --------------------------------------------------------------------------
# 1. Базовый сценарий из ТЗ: палка -> доска над палкой -> меч
# --------------------------------------------------------------------------
def test_stick_in_correct_slot_gives_reward(engine):
    before = [E] * 9
    after = [E] * 9
    after[7] = STICK  # низ-центр — рукоять меча
    br = engine.on_place_in_grid(before, after, 7, STICK)
    assert br.total > 0, "палка в нужном слоте должна давать награду"


def test_plank_above_stick_gives_more(engine):
    # шаг 1: палка
    g0 = [E] * 9
    g1 = [E] * 9; g1[7] = STICK
    r1 = engine.on_place_in_grid(g0, g1, 7, STICK).total

    # шаг 2: доска НАД палкой -> должна сработать пространственная награда
    g2 = list(g1); g2[4] = PLANK
    br2 = engine.on_place_in_grid(g1, g2, 4, PLANK)
    keys = [k for k, _, _ in br2.parts]
    assert any("place_adjacent" in k for k in keys), \
        "доска над палкой должна давать награду за пространственную связь"
    assert br2.total > 0


def test_full_sword_shape_completes(engine):
    grid = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    assert find_completed(grid) is not None
    assert find_completed(grid).name == "wooden_sword"


def test_craft_sword_big_reward(engine):
    recipe = RECIPE_BY_NAME["wooden_sword"]
    br = engine.on_craft(recipe, True)
    assert br.total > 20, f"крафт меча должен давать много, получили {br.total}"


# --------------------------------------------------------------------------
# 2. Затухание: повтор того же не должен фармиться
# --------------------------------------------------------------------------
def test_repeat_craft_decays(engine):
    recipe = RECIPE_BY_NAME["wooden_sword"]
    rewards = [engine.on_craft(recipe, True).total for _ in range(5)]
    assert rewards[0] > rewards[1] > rewards[2], \
        f"награда должна затухать: {rewards}"
    assert rewards[-1] < rewards[0] * 0.3


def test_repeat_grid_placement_decays(engine):
    vals = []
    for _ in range(4):
        g0 = [E] * 9
        g1 = [E] * 9; g1[7] = STICK
        vals.append(engine.on_place_in_grid(g0, g1, 7, STICK).total)
    assert vals[0] > vals[-1], f"повтор должен дешеветь: {vals}"


# --------------------------------------------------------------------------
# 3. Тиры: каменный/железный награждаются заново
# --------------------------------------------------------------------------
def test_tiers_have_separate_counters(engine):
    wood = engine.on_craft(RECIPE_BY_NAME["wooden_sword"], True).total
    for _ in range(4):
        engine.on_craft(RECIPE_BY_NAME["wooden_sword"], True)
    stone = engine.on_craft(RECIPE_BY_NAME["stone_sword"], True).total
    assert stone > wood * 0.8, \
        f"каменный меч должен награждаться заново: wood={wood} stone={stone}"


def test_tier_up_milestone_once(engine):
    engine.on_craft(RECIPE_BY_NAME["stone_sword"], True)
    br2 = engine.on_craft(RECIPE_BY_NAME["stone_pickaxe"], True)
    keys = [k for k, _, _ in br2.parts]
    assert not any("tier_up" in k and "stone" in k for k in keys), \
        "веха тира выдаётся только один раз"


# --------------------------------------------------------------------------
# 4. Штрафы за неправильные действия
# --------------------------------------------------------------------------
def test_wrong_cell_penalised(engine):
    g0 = [E, PLANK, E, E, PLANK, E, E, STICK, E]  # почти меч
    g1 = list(g0); g1[0] = DIRT                   # грязь в угол — лишнее
    br = engine.on_place_in_grid(g0, g1, 0, DIRT)
    assert br.total < 0, f"лишний предмет должен штрафоваться, было {br.total}"


def test_craft_fail_penalised(engine):
    br = engine.on_craft(None, False)
    assert br.total < 0


def test_breaking_shape_penalised(engine):
    g0 = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    g1 = list(g0); g1[7] = E
    br = engine.on_take_from_grid(g0, g1, 7)
    assert br.total < 0, "разрушение верной формы должно штрафоваться"


def test_penalties_do_not_decay(engine):
    vals = []
    for _ in range(5):
        g0 = [E] * 9
        g1 = [E] * 9; g1[0] = DIRT
        # кладём мусор в пустую сетку — сравним со сценарием "мешает рецепту"
        vals.append(engine.on_craft(None, False).total)
    assert all(abs(v - vals[0]) < 1e-6 for v in vals), \
        f"штрафы не должны слабеть: {vals}"


# --------------------------------------------------------------------------
# 5. Пространственная логика рецептов
# --------------------------------------------------------------------------
def test_shape_matters_not_just_items(engine):
    right = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    wrong = [PLANK, PLANK, STICK, E, E, E, E, E, E]  # те же предметы, форма иная
    assert find_completed(right) is not None
    assert find_completed(wrong) is None


def test_recipe_shift_is_accepted(engine):
    # меч, сдвинутый в левую колонку — Minecraft это принимает
    shifted = [PLANK, E, E, PLANK, E, E, STICK, E, E]
    r = find_completed(shifted)
    assert r is not None and r.name == "wooden_sword"


def test_partial_progress_tracked(engine):
    # Меч наполовину: две доски есть, палки-рукояти ещё нет.
    grid = [E, PLANK, E, E, PLANK, E, E, E, E]
    info = match_grid(grid, RECIPE_BY_NAME["wooden_sword"])
    assert 0 < info.progress < 1.0, f"прогресс меча должен быть частичным: {info}"
    assert not info.complete

    # А best_partial на этой же сетке узнаёт полностью собранный рецепт палок.
    auto = best_partial(grid)
    assert auto is not None and auto.recipe.name == "stick" and auto.complete


# --------------------------------------------------------------------------
# 6. База данных
# --------------------------------------------------------------------------
def test_db_persists_and_decays_across_sessions():
    tmp = Path(tempfile.mkdtemp()) / "p.db"
    db1 = RewardDB(tmp); seed(db1)
    e1 = RewardEngine(db1); e1.begin_episode(1)
    first = e1.on_craft(RECIPE_BY_NAME["iron_sword"], True).total
    db1.close()

    db2 = RewardDB(tmp); seed(db2)
    e2 = RewardEngine(db2); e2.begin_episode(2)
    second = e2.on_craft(RECIPE_BY_NAME["iron_sword"], True).total
    db2.close()
    assert second < first, "БД должна помнить прогресс между запусками"


def test_one_shot_rule():
    tmp = Path(tempfile.mkdtemp()) / "o.db"
    db = RewardDB(tmp)
    db.upsert_rule(Rule("t.once", "milestone", 10.0, one_shot=True))
    a = db.grant("t.once", episode=1, step=1).value
    b = db.grant("t.once", episode=1, step=2).value
    assert a == 10.0 and b == 0.0
    db.close()


def test_reset_progress_restores_rewards():
    tmp = Path(tempfile.mkdtemp()) / "r.db"
    db = RewardDB(tmp); seed(db)
    eng = RewardEngine(db); eng.begin_episode(1)
    a = eng.on_craft(RECIPE_BY_NAME["wooden_sword"], True).total
    for _ in range(3):
        eng.on_craft(RECIPE_BY_NAME["wooden_sword"], True)
    db.reset_progress()
    eng2 = RewardEngine(db); eng2.begin_episode(1)
    c = eng2.on_craft(RECIPE_BY_NAME["wooden_sword"], True).total
    assert abs(c - a) < 1e-6, "после сброса награда возвращается к базовой"
    db.close()


# --------------------------------------------------------------------------
# 7. Защита от фарма (регрессия на реальный баг из прогона 15001)
# --------------------------------------------------------------------------
def test_shape_complete_paid_once_per_episode(engine):
    """
    Цикл 'собрал форму -> разобрал -> собрал' не должен приносить деньги.

    Реальный баг: за 800 эпизодов grid.shape_complete выплатил +9392, при
    этом целые эпизоды заканчивались вообще без единого крафта.
    """
    full = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    almost = list(full); almost[1] = E

    first = engine.on_place_in_grid(almost, full, 1, PLANK)
    assert any("shape_complete" in k for k, _, _ in first.parts)

    # Полный цикл: разобрать (штраф) + собрать заново (уже не платят).
    total_cycle = 0.0
    for _ in range(5):
        total_cycle += engine.on_take_from_grid(full, almost, 1).total
        br = engine.on_place_in_grid(almost, full, 1, PLANK)
        assert not any("shape_complete" in k for k, _, _ in br.parts), \
            "форма не должна оплачиваться повторно в том же эпизоде"
        assert br.total <= 0, \
            f"пересборка не должна приносить доход, получили {br.total:+.2f}"
        total_cycle += br.total

    assert total_cycle < 0, \
        f"цикл пересборки должен быть убыточным, получили {total_cycle:+.2f}"


def test_farming_loop_is_unprofitable(engine):
    """Полный цикл разбор-сборка в сумме со штрафом break_shape = минус."""
    full = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    almost = list(full); almost[1] = E
    engine.on_place_in_grid(almost, full, 1, PLANK)

    cycle = 0.0
    cycle += engine.on_take_from_grid(full, almost, 1).total
    cycle += engine.on_place_in_grid(almost, full, 1, PLANK).total
    assert cycle < 0, f"цикл фарма должен быть в минус, получили {cycle:+.2f}"


def test_approach_rewards_only_new_record(engine):
    """Ходьба туда-сюда не должна оплачиваться — только новый рекорд."""
    engine.on_approach("crafting_table", 5)
    gain = engine.on_approach("crafting_table", 3).total
    assert gain > 0, "приближение к новому рекорду награждается"

    again = engine.on_approach("crafting_table", 4).total
    assert again == 0.0, "возврат на прежнюю дистанцию не оплачивается"
    again2 = engine.on_approach("crafting_table", 3).total
    assert again2 == 0.0, "повтор прежнего рекорда не оплачивается"


def test_episode_reset_restores_shape_reward(engine):
    """Новый эпизод — новая попытка: веха снова доступна."""
    full = [E, PLANK, E, E, PLANK, E, E, STICK, E]
    almost = list(full); almost[1] = E
    engine.on_place_in_grid(almost, full, 1, PLANK)

    engine.begin_episode(2)
    br = engine.on_place_in_grid(almost, full, 1, PLANK)
    assert any("shape_complete" in k for k, _, _ in br.parts), \
        "в новом эпизоде веха должна снова оплачиваться"

"""
Контракт моста «симулятор <-> реальный Minecraft».

Смысл: веса, обученные в симуляторе, должны читать наблюдение из реального
мира ТОЧНО так же. Один сдвинутый элемент вектора — и модель видит мусор.
Эти тесты ловят расхождение без запуска Minecraft-сервера.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import (ACTIONS, DENSE_DIM, GRID_2X2_SLOTS, N_ACTIONS,
                          N_MAP, N_VOXELS, ActionType, SL_CRAFT_STATE,
                                SL_SURVIVAL, SL_COORDS)

import play
from brain.spaces import ITEMS

BRIDGE_JS = Path(__file__).resolve().parents[1] / "bot" / "bridge.js"


@pytest.fixture()
def engine(tmp_path):
    db = RewardDB(str(tmp_path / "t.db"))
    seed(db)
    eng = RewardEngine(db)
    eng.begin_episode(1)
    yield eng
    db.close()


def _fake_raw(**over):
    """Наблюдение, какое прислал бы mineflayer-мост."""
    raw = {
        "grid": [0] * 9,
        "voxels": [0] * N_VOXELS,
        "blockmap": [0] * N_MAP,
        "entmap": [0] * N_MAP,
        "inventory": [0.0] * len(ITEMS),
        "held": 0,
        "facing": 0,
        "pitch": 0,
        "near_table": 0,
        "table_dist": 99,
        "position": [0, 64, 0],
        "health": 20,
        "food": 20,
        "table_open": 0,
        "table_reachable": 0,
        "has_3x3": 0,
        "active_slots": list(GRID_2X2_SLOTS),
    }
    raw.update(over)
    return raw


# ------------------------------------------------- размерность наблюдения
def test_bridge_obs_has_exact_model_shape(engine):
    """Главный тест: мост отдаёт ровно DENSE_DIM чисел, а не 62."""
    obs = play.bridge_obs_to_model(_fake_raw(), 0, engine, set(), 0, 100)
    assert obs["dense"].shape[0] == DENSE_DIM
    assert obs["grid"].shape[0] == 9
    assert obs["voxels"].shape[0] == N_VOXELS
    assert obs["blockmap"].shape[0] == N_MAP
    assert obs["entmap"].shape[0] == N_MAP


def test_bridge_obs_matches_simulator_layout(engine):
    """Формат из моста совпадает с форматом симулятора ключ-в-ключ."""
    env = MinecraftCraftEnv(engine=engine, seed=0)
    sim = env.reset()
    real = play.bridge_obs_to_model(_fake_raw(), 0, engine, set(), 0, 100)
    assert sim.keys() == real.keys()
    for k in sim:
        assert sim[k].shape == real[k].shape, f"{k}: {sim[k].shape} != {real[k].shape}"
        assert sim[k].dtype == real[k].dtype, k


def test_craft_state_tail_is_wired(engine):
    """
    craft_state = table_open / has_3x3 / table_reachable.

    Он идёт перед блоком survival (6 чисел), поэтому смотрим срез
    SL_CRAFT_STATE, а не хардкод хвоста.
    """
    raw = _fake_raw(table_open=1, has_3x3=1, table_reachable=1,
                    active_slots=list(range(9)))
    obs = play.bridge_obs_to_model(raw, 0, engine, set(), 0, 100)
    assert list(obs["dense"][SL_CRAFT_STATE]) == [1.0, 1.0, 1.0]

    obs0 = play.bridge_obs_to_model(_fake_raw(), 0, engine, set(), 0, 100)
    assert list(obs0["dense"][SL_CRAFT_STATE]) == [0.0, 0.0, 0.0]


def test_survival_tail_is_wired(engine):
    """Последние 6 чисел — здоровье, голод, броня, враг, дистанция, еда."""
    raw = _fake_raw(health=10, food=5)
    obs = play.bridge_obs_to_model(raw, 0, engine, set(), 0, 100)
    surv = obs["dense"][SL_SURVIVAL]
    assert abs(surv[0] - 0.5) < 1e-6, "здоровье 10/20"
    assert abs(surv[1] - 0.25) < 1e-6, "голод 5/20"
    assert surv[3] == 0.0, "врагов рядом нет"


def test_hostile_on_entmap_is_seen(engine):
    """Зомби на карте сущностей -> флаг опасности поднимается."""
    from brain.spaces import ENTITY_ID
    ent = [0] * N_MAP
    W = int(N_MAP ** 0.5)
    ent[(W // 2) * W + (W // 2 + 2)] = ENTITY_ID["zombie"]
    obs = play.bridge_obs_to_model(_fake_raw(entmap=ent), 0, engine,
                                   set(), 0, 100)
    assert obs["dense"][-3] == 1.0, "враг рядом должен быть виден"


def test_attack_and_eat_actions_exist():
    """Действия боя и еды есть в пространстве действий и в мосте."""
    names = {a.name for a in ACTIONS}
    assert "attack_front" in names
    assert "eat_food" in names
    src = _js()
    assert "case 'attack'" in src
    assert "case 'eat'" in src


def test_bridge_js_sends_vision_maps():
    """Мост обязан слать оба слоя дальнего зрения."""
    src = _js()
    assert "blockmap" in src, "мост не шлёт карту блоков"
    assert "entmap" in src, "мост не шлёт карту сущностей"


def test_bridge_js_entity_list_matches_python():
    from brain.spaces import ENTITIES
    src = _js()
    block = re.search(r"const ENTITIES = \[(.*?)\];", src, re.S).group(1)
    js_ents = re.findall(r"'([a-z_]+)'", block)
    assert js_ents == list(ENTITIES), "порядок сущностей разошёлся"


# ------------------------------------------------- маска действий
def test_mask_blocks_3x3_slots_without_table():
    """Без верстака слоты 2,5,6,7,8 недоступны — как в инвентаре 2x2."""
    inv = [0.0] * len(ITEMS)
    inv[2] = 0.5                       # есть доски
    raw = _fake_raw(inventory=inv, held=2)
    m = play.mask_from_raw(raw)
    for act in ACTIONS:
        if act.type == ActionType.PLACE_IN_SLOT:
            if act.arg in (2, 5, 6, 7, 8):
                assert not m[act.index], f"слот {act.arg} должен быть закрыт"
            else:
                assert m[act.index], f"слот {act.arg} должен быть открыт"


def test_mask_opens_3x3_with_table():
    inv = [0.0] * len(ITEMS)
    inv[2] = 0.5
    raw = _fake_raw(inventory=inv, held=2, table_open=1, has_3x3=1,
                    table_reachable=1, active_slots=list(range(9)))
    m = play.mask_from_raw(raw)
    for act in ACTIONS:
        if act.type == ActionType.PLACE_IN_SLOT:
            assert m[act.index], f"слот {act.arg} должен быть открыт"


def test_mask_use_table_requires_table_near():
    m = play.mask_from_raw(_fake_raw())
    idx = [a.index for a in ACTIONS if a.type == ActionType.USE_TABLE][0]
    assert not m[idx], "открывать нечего — верстака рядом нет"

    m2 = play.mask_from_raw(_fake_raw(table_reachable=1))
    assert m2[idx], "верстак рядом — открыть можно"


def test_mask_size_matches_action_space():
    assert play.mask_from_raw(_fake_raw()).shape[0] == N_ACTIONS


# ------------------------------------------------- Node-сторона (статически)
def _js() -> str:
    return BRIDGE_JS.read_text(encoding="utf-8")


def test_bridge_js_handles_use_table():
    """Мост обязан понимать действие open_close_table (индекс 57)."""
    assert "case 'use_table'" in _js()


def test_bridge_js_knows_tool_levels():
    """Мост обязан знать правила добычи, иначе штрафа wrong_tool не будет."""
    src = _js()
    assert "canHarvest" in src
    assert "NEEDS_PICKAXE" in src
    assert "wrong_tool" in src


def test_bridge_js_item_list_matches_python():
    """Словарь предметов в JS и Python — один и тот же порядок."""
    from brain.spaces import ITEMS
    src = _js()
    block = re.search(r"const ITEMS = \[(.*?)\];", src, re.S).group(1)
    js_items = re.findall(r"'([a-z_]+)'", block)
    assert js_items == list(ITEMS), "порядок предметов разошёлся — id поедут"


def test_bridge_js_block_list_matches_python():
    from brain.spaces import BLOCKS
    src = _js()
    block = re.search(r"const BLOCKS = \[(.*?)\];", src, re.S).group(1)
    js_blocks = re.findall(r"'([a-z_]+)'", block)
    assert js_blocks == list(BLOCKS), "порядок блоков разошёлся"


def test_bridge_js_reports_grid_state():
    """Мост шлёт поля, из которых Python строит craft_state."""
    src = _js()
    for field in ("table_open", "has_3x3", "table_reachable", "active_slots"):
        assert field in src, f"мост не шлёт {field}"


def test_bridge_js_enforces_2x2():
    """Слот вне активной сетки мост обязан отклонять сам."""
    assert "activeSlots().includes(a)" in _js()

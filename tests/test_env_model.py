"""Тесты среды и модели."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.model import CraftBrain, ModelConfig, batch_obs, obs_to_tensor
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import (ACTION_BY_NAME, BLOCK_ID, N_ACTIONS, N_MAP, N_VOXELS,
                          item_id)


@pytest.fixture()
def env():
    tmp = Path(tempfile.mkdtemp()) / "e.db"
    db = RewardDB(tmp); seed(db)
    e = MinecraftCraftEnv(RewardEngine(db), max_steps=60, seed=42)
    yield e
    db.close()


@pytest.fixture()
def engine():
    """Чистый движок наград — для тестов выживания."""
    tmp = Path(tempfile.mkdtemp()) / "s.db"
    db = RewardDB(tmp); seed(db)
    yield RewardEngine(db)
    db.close()


def test_env_reset_shapes(env):
    obs = env.reset()
    assert obs["grid"].shape == (9,)
    assert obs["voxels"].shape == (N_VOXELS,)
    assert obs["blockmap"].shape == (N_MAP,)
    assert obs["entmap"].shape == (N_MAP,)
    assert obs["held"].shape == (1,)
    assert obs["dense"].ndim == 1


def test_env_random_rollout(env):
    obs = env.reset()
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(60):
        a = int(rng.integers(N_ACTIONS))
        obs, r, done, info = env.step(a)
        total += r
        assert np.isfinite(r)
        if done:
            break
    assert isinstance(total, float)


def test_scripted_wooden_sword(env):
    """Сценарий из ТЗ целиком: собрать деревянный меч и получить награду."""
    env.reset(goal_index=GOALS.index("wooden_sword"))
    env.agent.inventory[item_id("oak_planks")] = 8
    env.agent.inventory[item_id("stick")] = 4
    env.spawn_table_adjacent()
    # Меч — рецепт 3x3, значит верстак надо ОТКРЫТЬ, иначе слоты недоступны.
    env.step(ACTION_BY_NAME["open_close_table"].index)
    assert env.has_3x3(), "после открытия верстака должна быть сетка 3x3"

    seq = [
        "hold_stick", "place_slot_7",
        "hold_oak_planks", "place_slot_4", "place_slot_1",
        "craft",
    ]
    total = 0.0
    for name in seq:
        _, r, done, info = env.step(ACTION_BY_NAME[name].index)
        total += r
        if done:
            break
    assert env.agent.inventory.get(item_id("wooden_sword"), 0) >= 1, \
        "меч должен быть скрафчен"
    assert total > 20, f"за сборку меча ожидаем крупную награду, было {total}"


def test_wrong_placement_is_punished(env):
    env.reset(goal_index=GOALS.index("wooden_sword"))
    env.agent.inventory[item_id("dirt")] = 5
    env.spawn_table_adjacent()
    env.step(ACTION_BY_NAME["hold_stick"].index)
    env.step(ACTION_BY_NAME["place_slot_7"].index)
    env.step(ACTION_BY_NAME["hold_dirt"].index)
    _, r, _, info = env.step(ACTION_BY_NAME["place_slot_0"].index)
    assert r < 0, f"грязь в угол должна штрафоваться, r={r}"


def test_craft_without_recipe_punished(env):
    env.reset()
    _, r, _, _ = env.step(ACTION_BY_NAME["craft"].index)
    assert r < 0


# --------------------------------------------------------------------------
# модель
# --------------------------------------------------------------------------
def test_model_forward_shapes(env):
    obs = env.reset()
    m = CraftBrain(ModelConfig(head="both"))
    t = obs_to_tensor(obs)
    q = m.q_values(t)
    logits, v = m.policy_value(t)
    assert q.shape == (1, N_ACTIONS)
    assert logits.shape == (1, N_ACTIONS)
    assert v.shape == (1,)


def test_model_is_untrained():
    """Модель действительно 'пустая': политика близка к равномерной."""
    m = CraftBrain(ModelConfig(head="both"))
    dummy = {
        "grid": torch.zeros(4, 9, dtype=torch.long),
        "voxels": torch.zeros(4, N_VOXELS, dtype=torch.long),
        "blockmap": torch.zeros(4, N_MAP, dtype=torch.long),
        "entmap": torch.zeros(4, N_MAP, dtype=torch.long),
        "dense": torch.zeros(4, m.dense_proj.in_features),
        "held": torch.zeros(4, 1, dtype=torch.long),
    }
    logits, _ = m.policy_value(dummy)
    probs = torch.softmax(logits, dim=-1)
    uniform = 1.0 / N_ACTIONS
    assert (probs - uniform).abs().max().item() < 0.02, \
        "необученная политика должна быть почти равномерной"


def test_model_learns_signal():
    """Градиентный шаг реально уменьшает лосс — сеть обучаема."""
    m = CraftBrain(ModelConfig(head="dqn"))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    obs = {
        "grid": torch.randint(0, 5, (16, 9)),
        "voxels": torch.randint(0, 5, (16, N_VOXELS)),
        "blockmap": torch.randint(0, 5, (16, N_MAP)),
        "entmap": torch.randint(0, 5, (16, N_MAP)),
        "dense": torch.randn(16, m.dense_proj.in_features),
        "held": torch.randint(0, 5, (16, 1)),
    }
    target = torch.randn(16, N_ACTIONS)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.mse_loss(m.q_values(obs), target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5, f"лосс должен падать: {losses[0]}->{losses[-1]}"


def test_dqn_trainer_step(env):
    from brain.algos.dqn import DQNConfig, DQNTrainer
    m = CraftBrain(ModelConfig(head="both"))
    tr = DQNTrainer(m, DQNConfig(batch_size=8, warmup=8))
    obs = env.reset()
    for _ in range(40):
        a = tr.act(obs)
        nobs, r, done, _ = env.step(a)
        tr.buffer.push(obs, a, r, nobs, done)
        obs = env.reset() if done else nobs
    loss = tr.learn()
    assert loss is None or np.isfinite(loss)


def test_ppo_trainer_step(env):
    from brain.algos.ppo import PPOConfig, PPOTrainer
    m = CraftBrain(ModelConfig(head="both"))
    tr = PPOTrainer(m, PPOConfig(rollout=32, minibatch=8, epochs=2))
    obs = env.reset()
    for _ in range(32):
        a, lp, v = tr.act(obs)
        nobs, r, done, _ = env.step(a)
        tr.buf.add(obs, a, lp, r, v, done)
        obs = env.reset() if done else nobs
    stats = tr.update(0.0)
    assert "pi_loss" in stats and np.isfinite(stats["pi_loss"])


def test_no_reward_for_already_sufficient_item(env):
    """
    Регрессия: агент 37 эпизодов подряд крафтил доски, хотя цель — палки,
    а досок в инвентаре уже было 8. Простой промежуточный рецепт не должен
    оплачиваться, если его продукта уже достаточно.
    """
    env.reset(goal_index=GOALS.index("stick"))
    env.agent.inventory[item_id("oak_planks")] = 8
    env.agent.inventory[item_id("oak_log")] = 4
    env.spawn_table_adjacent()

    env.step(ACTION_BY_NAME["hold_oak_log"].index)
    env.step(ACTION_BY_NAME["place_slot_0"].index)
    _, r, _, info = env.step(ACTION_BY_NAME["craft"].index)
    assert r <= 0.5, f"крафт лишних досок не должен кормить агента, r={r}"


def test_needed_intermediate_still_rewarded(env):
    """Обратная проверка: если досок НЕТ, их крафт по-прежнему награждается."""
    env.reset(goal_index=GOALS.index("stick"))
    env.agent.inventory.pop(item_id("oak_planks"), None)
    env.agent.inventory[item_id("oak_log")] = 4
    env.spawn_table_adjacent()

    env.step(ACTION_BY_NAME["hold_oak_log"].index)
    env.step(ACTION_BY_NAME["place_slot_0"].index)
    _, r, _, _ = env.step(ACTION_BY_NAME["craft"].index)
    assert r > 0, f"нужный промежуточный рецепт должен награждаться, r={r}"


def test_observation_is_markovian(env):
    """
    Регрессия на реальный баг: агент крутил place_slot -> clear_grid по кругу.

    Причина была в том, что движок наград помнил "за эту клетку уже платили",
    а агент этого НЕ видел. Одинаковое наблюдение давало то +11, то -0.02 —
    среда переставала быть марковской, и выучить правило было невозможно.
    Теперь память движка входит в наблюдение: разная награда => разное obs.
    """
    import numpy as np
    env.reset(goal_index=GOALS.index("oak_planks"))
    env.agent.inventory[item_id("oak_log")] = 4
    env.step(ACTION_BY_NAME["hold_oak_log"].index)

    # Слот 0 входит в инвентарную сетку 2x2 — доступен без верстака.
    obs_a = env.observe()["dense"].copy()
    _, r_first, _, _ = env.step(ACTION_BY_NAME["place_slot_0"].index)
    env.step(ACTION_BY_NAME["clear_grid"].index)

    obs_b = env.observe()["dense"].copy()
    _, r_second, _, _ = env.step(ACTION_BY_NAME["place_slot_0"].index)

    assert r_first > r_second, "повтор должен стоить дешевле"
    assert not np.array_equal(obs_a, obs_b), (
        "наблюдения ОБЯЗАНЫ отличаться, раз награда отличается — "
        "иначе сеть не сможет выучить разницу")


# --------------------------------------------------------------------------
# Механика 2x2 -> 3x3 и старт с пустыми руками
# --------------------------------------------------------------------------
def test_starts_empty_handed(env):
    """Новая игра: инвентарь пуст, добывать всё придётся самому."""
    env.reset(goal_index=0)
    assert sum(env.agent.inventory.values()) == 0, \
        f"инвентарь должен быть пуст, а там {env.agent.inventory}"
    assert env.agent.held == 0


def test_only_2x2_without_table(env):
    """Без верстака доступна только инвентарная сетка 2x2."""
    env.reset(goal_index=0)
    assert env.active_slots() == (0, 1, 3, 4)
    assert not env.has_3x3()

    env.agent.inventory[item_id("oak_log")] = 4
    env.step(ACTION_BY_NAME["hold_oak_log"].index)
    mask = env.action_mask()
    for slot in (2, 5, 6, 7, 8):
        idx = ACTION_BY_NAME[f"place_slot_{slot}"].index
        assert not mask[idx], f"слот {slot} не должен быть доступен без верстака"
    for slot in (0, 1, 3, 4):
        idx = ACTION_BY_NAME[f"place_slot_{slot}"].index
        assert mask[idx], f"слот {slot} обязан быть доступен в сетке 2x2"


def test_table_unlocks_3x3(env):
    """Верстак рядом + открыть -> становится доступна полная сетка 3x3."""
    env.reset(goal_index=0)
    env.spawn_table_adjacent()
    assert not env.has_3x3(), "верстак стоит, но ещё не открыт"

    env.step(ACTION_BY_NAME["open_close_table"].index)
    assert env.has_3x3(), "после открытия должна быть сетка 3x3"
    assert env.active_slots() == tuple(range(9))

    env.agent.inventory[item_id("oak_planks")] = 4
    env.step(ACTION_BY_NAME["hold_oak_planks"].index)
    assert env.action_mask()[ACTION_BY_NAME["place_slot_8"].index], \
        "слот 8 должен открыться вместе с верстаком"


def test_cannot_open_table_without_table(env):
    """Нельзя открыть верстак, которого нет рядом."""
    env.reset(goal_index=0)
    _, r, _, _ = env.step(ACTION_BY_NAME["open_close_table"].index)
    assert r < 0, "попытка открыть несуществующий верстак должна штрафоваться"
    assert not env.has_3x3()


def test_3x3_recipe_blocked_without_table(env):
    """Кирку (3x3) нельзя скрафтить в инвентарной сетке 2x2."""
    env.reset(goal_index=0)
    env.agent.inventory[item_id("oak_planks")] = 8
    env.agent.inventory[item_id("stick")] = 4
    # выкладываем кирку в доступные слоты — форма всё равно не соберётся
    env.step(ACTION_BY_NAME["hold_oak_planks"].index)
    for sl in (0, 1):
        env.step(ACTION_BY_NAME[f"place_slot_{sl}"].index)
    _, r, _, _ = env.step(ACTION_BY_NAME["craft"].index)
    assert env.agent.inventory.get(item_id("wooden_pickaxe"), 0) == 0, \
        "кирка не должна крафтиться без верстака"


def test_walking_away_closes_table(env):
    """Отошёл от верстака — сетка схлопывается обратно в 2x2."""
    env.reset(goal_index=0)
    env.spawn_table_adjacent()
    env.step(ACTION_BY_NAME["open_close_table"].index)
    assert env.has_3x3()

    for _ in range(6):
        env.step(ACTION_BY_NAME["move_forward"].index)
        if not env._table_reachable():
            break
    assert not env.has_3x3(), "уйдя от верстака, агент теряет сетку 3x3"


def test_can_gather_by_breaking(env):
    """Голыми руками можно сломать дерево и получить бревно."""
    env.reset(goal_index=0)
    assert env.agent.inventory.get(item_id("oak_log"), 0) == 0
    tx, ty, tz = env._nearest_block(BLOCK_ID["oak_log"])
    env.agent.x, env.agent.y, env.agent.z = tx, ty, tz - 1
    env.agent.facing = 0          # смотрим на +Z
    env.agent.pitch = 0
    _, r, _, _ = env.step(ACTION_BY_NAME["break_block_front"].index)
    assert env.agent.inventory.get(item_id("oak_log"), 0) >= 1, \
        "сломанное дерево должно попасть в инвентарь"
    assert r > 0, "добыча нужного ресурса награждается"


# --------------------------------------------------------------------------
# Уровни инструментов: камень->дерев.кирка, железо->камен., алмаз->желез.
# Регрессия на реальный баг: агент добывал железо и камень ГОЛОЙ РУКОЙ.
# --------------------------------------------------------------------------
def _mine(env, block_name, tool_name):
    """Поставить блок перед агентом, взять инструмент, сломать. Вернуть дроп."""
    from brain.spaces import BLOCK_ID as BID
    env.reset(goal_index=0)
    env.agent.inventory.clear()
    if tool_name != "empty":
        env.agent.inventory[item_id(tool_name)] = 1
        env.agent.held = item_id(tool_name)
    else:
        env.agent.held = 0
    env.agent.facing = 0
    fx, fy, fz = env._front()
    env.world[fx, fy, fz] = BID[block_name]
    before = dict(env.agent.inventory)
    env.step(ACTION_BY_NAME["break_block_front"].index)
    gained = {k: v for k, v in env.agent.inventory.items()
              if v > before.get(k, 0)}
    return gained


def test_bare_hand_cannot_mine_stone(env):
    """Рукой камень не добыть — блок ломается, булыжник не падает."""
    got = _mine(env, "stone", "empty")
    assert item_id("cobblestone") not in got, \
        f"рукой не должно выпадать ничего, выпало {got}"


def test_bare_hand_cannot_mine_iron(env):
    """Главный баг из отчёта: железо голой рукой."""
    got = _mine(env, "iron_ore", "empty")
    assert item_id("raw_iron") not in got, \
        f"железо рукой добывать нельзя, выпало {got}"


def test_wooden_pickaxe_mines_stone(env):
    """Деревянная кирка — минимальный уровень для камня."""
    got = _mine(env, "stone", "wooden_pickaxe")
    assert got.get(item_id("cobblestone"), 0) >= 1, \
        f"деревянная кирка обязана давать булыжник, выпало {got}"


def test_wooden_pickaxe_cannot_mine_iron(env):
    """Железо деревянной киркой не берётся — нужна каменная."""
    got = _mine(env, "iron_ore", "wooden_pickaxe")
    assert item_id("raw_iron") not in got, \
        f"деревянной киркой железо добывать нельзя, выпало {got}"


def test_stone_pickaxe_mines_iron(env):
    got = _mine(env, "iron_ore", "stone_pickaxe")
    assert got.get(item_id("raw_iron"), 0) >= 1, \
        f"каменная кирка обязана давать железо, выпало {got}"


def test_stone_pickaxe_cannot_mine_diamond(env):
    """Алмаз каменной киркой не берётся — нужна железная."""
    got = _mine(env, "diamond_ore", "stone_pickaxe")
    assert item_id("diamond") not in got, \
        f"каменной киркой алмаз добывать нельзя, выпало {got}"


def test_iron_pickaxe_mines_diamond(env):
    got = _mine(env, "diamond_ore", "iron_pickaxe")
    assert got.get(item_id("diamond"), 0) >= 1, \
        f"железная кирка обязана давать алмаз, выпало {got}"


def test_sword_is_not_a_pickaxe(env):
    """Каменный меч не заменяет каменную кирку — нужен именно тот инструмент."""
    got = _mine(env, "stone", "stone_sword")
    assert item_id("cobblestone") not in got, \
        f"мечом камень не добывается, выпало {got}"


def test_wrong_tool_is_punished(env):
    """Сломать руду без нужной кирки = потерять её, за это штраф."""
    from brain.spaces import BLOCK_ID as BID
    env.reset(goal_index=0)
    env.agent.inventory.clear()
    env.agent.held = 0
    env.agent.facing = 0
    fx, fy, fz = env._front()
    env.world[fx, fy, fz] = BID["iron_ore"]
    env.mask_level = "physical"      # smart-маска такой ход запрещает
    _, r, _, info = env.step(ACTION_BY_NAME["break_block_front"].index)
    keys = [p["key"] for p in info["breakdown"]["parts"]]
    assert any("wrong_tool" in k for k in keys), \
        f"должен быть штраф за неверный инструмент, получили {keys}"
    assert r < 0


def test_wood_still_minable_by_hand(env):
    """Дерево рукой ломается — как в ванили."""
    got = _mine(env, "oak_log", "empty")
    assert got.get(item_id("oak_log"), 0) >= 1, \
        f"дерево рукой добываться должно, выпало {got}"


def test_starting_inventory_has_no_free_ore(env):
    """Верхние цели получают ИНСТРУМЕНТ, а не бесплатную руду."""
    env.reset(goal_index=GOALS.index("diamond_sword"))
    inv = env.agent.inventory
    assert inv.get(item_id("iron_pickaxe"), 0) >= 1, \
        "для алмазной цели нужна железная кирка"
    assert inv.get(item_id("diamond"), 0) == 0, \
        "алмазы даром не выдаём — их надо добыть железной киркой"


# ==========================================================================
# ЗРЕНИЕ И ВЫЖИВАНИЕ
# ==========================================================================
def _survival_env(engine, **kw):
    from brain.env.mc_env import MinecraftCraftEnv
    e = MinecraftCraftEnv(engine, max_steps=200, seed=5,
                          mask_level="physical", **kw)
    e.reset()
    return e


def test_vision_is_bigger_than_before(env):
    """3D-зрение выросло с 3x3x3 до 5x5x5 — иначе мобов не заметить."""
    from brain.spaces import VIEW
    obs = env.reset()
    assert VIEW == 5
    assert obs["voxels"].shape[0] == 125


def test_2d_map_sees_far(env):
    """2D-карта охватывает больше, чем ближний куб."""
    from brain.spaces import MAP_W, N_MAP, N_VOXELS
    obs = env.reset()
    assert MAP_W == 13
    assert obs["blockmap"].shape[0] == N_MAP
    assert N_MAP > 0
    # карта реально что-то содержит, а не одни нули
    assert int((obs["blockmap"] > 0).sum()) > 0


def test_entmap_shows_mobs(engine):
    """Мобы видны на карте сущностей."""
    from brain.env.mc_env import Mob
    from brain.spaces import ENTITY_ID
    e = _survival_env(engine)
    e.mobs = [Mob("zombie", e.agent.x, e.agent.y, e.agent.z + 2, 20.0)]
    e.agent.facing = 0
    obs = e.observe()
    assert int((obs["entmap"] == ENTITY_ID["zombie"]).sum()) == 1


def test_mob_damages_agent(engine):
    """
    Зомби вплотную отнимает здоровье.

    Бьёт не каждый шаг (есть перезарядка, как в игре), поэтому крутим
    несколько шагов, а не один.
    """
    from brain.env.mc_env import Mob
    e = _survival_env(engine)
    hp = e.agent.health
    for _ in range(6):
        e.mobs = [Mob("zombie", e.agent.x, e.agent.y, e.agent.z + 1, 10.0)]
        e.step(ACTION_BY_NAME["noop"].index)
        if e.agent.health < hp:
            break
    assert e.agent.health < hp, "моб обязан наносить урон"


def test_death_ends_episode_and_hurts(engine):
    """Смерть обрывает эпизод и стоит дорого."""
    from brain.env.mc_env import Mob
    e = _survival_env(engine)
    e.agent.health = 1.0
    done = False
    info = {}
    r = 0.0
    # Мобы бьют раз в 3 шага, поэтому крутим цикл и пересоздаём моба.
    for _ in range(6):
        e.mobs = [Mob("creeper", e.agent.x, e.agent.y, e.agent.z + 1, 10.0)]
        _, r, done, info = e.step(ACTION_BY_NAME["noop"].index)
        if done:
            break
    assert done and info.get("died")
    # Смерть дороже одной цели (-25 против +60 за goal.completed), но не
    # катастрофа: замер показал, что при -80 она давала 63% всех штрафов,
    # и градиент учил агента просто не двигаться. Сам обрыв эпизода уже
    # наказывает — агент теряет всю будущую награду.
    assert r < -20, f"смерть должна быть дорогой, а не {r}"


def test_sword_beats_bare_hand(engine):
    """Меч наносит больше урона, чем кулак."""
    from brain.env.mc_env import Mob
    from brain.spaces import item_id
    dmg = {}
    for weapon in ("empty", "iron_sword"):
        e = _survival_env(engine)
        if weapon != "empty":
            e.agent.inventory[item_id(weapon)] = 1
            e.agent.held = item_id(weapon)
        e.agent.facing = 0
        e.mobs = [Mob("zombie", e.agent.x, e.agent.y, e.agent.z + 1, 20.0)]
        e.step(ACTION_BY_NAME["attack_front"].index)
        dmg[weapon] = 20.0 - e.mobs[0].health
    assert dmg["iron_sword"] > dmg["empty"], dmg


def test_killing_passive_mob_gives_food(engine):
    """С коровы падает мясо — источник еды."""
    from brain.env.mc_env import Mob
    from brain.spaces import item_id
    e = _survival_env(engine)
    e.agent.inventory[item_id("iron_sword")] = 1
    e.agent.held = item_id("iron_sword")
    e.agent.facing = 0
    e.mobs = [Mob("cow", e.agent.x, e.agent.y, e.agent.z + 1, 1.0)]
    e.step(ACTION_BY_NAME["attack_front"].index)
    assert e.agent.inventory.get(item_id("beef"), 0) > 0


def test_eating_restores_food(engine):
    """Еда восстанавливает голод и вознаграждается."""
    from brain.spaces import item_id
    e = _survival_env(engine, mobs=False)
    e.agent.food = 10.0
    e.agent.inventory[item_id("bread")] = 1
    _, r, _, _ = e.step(ACTION_BY_NAME["eat_food"].index)
    assert e.agent.food > 10.0
    assert r > 0, "за своевременную еду должна быть награда"


def test_eating_when_full_is_punished(engine):
    """Жрать будучи сытым — трата ресурса, за это штраф."""
    from brain.spaces import item_id
    e = _survival_env(engine, mobs=False)
    e.agent.food = 20.0
    e.agent.inventory[item_id("bread")] = 1
    _, r, _, _ = e.step(ACTION_BY_NAME["eat_food"].index)
    assert r < 0


def test_mobs_can_be_disabled(engine):
    """Мобов можно выключить — тесты на крафт остаются чистыми."""
    e = _survival_env(engine, mobs=False)
    assert e.mobs == []


def test_armor_reduces_damage(engine):
    """Броня уменьшает входящий урон."""
    from brain.env.mc_env import Mob
    from brain.spaces import item_id
    lost = {}
    for armor in (None, "diamond_chestplate"):
        e = _survival_env(engine)
        if armor:
            e.agent.inventory[item_id(armor)] = 1
        hp = e.agent.health
        # крутим шаги, пока моб не пробьёт перезарядку удара
        for _ in range(6):
            e.mobs = [Mob("zombie", e.agent.x, e.agent.y, e.agent.z + 1, 10.0)]
            e.step(ACTION_BY_NAME["noop"].index)
            if e.agent.health < hp:
                break
        lost[armor] = hp - e.agent.health
    assert lost["diamond_chestplate"] < lost[None], lost

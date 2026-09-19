#!/usr/bin/env python3
"""
Python-сторона моста: мозг управляет реальным ботом в Minecraft.

Поток данных:
    play.py  --TCP JSON-->  bot/bridge.js  --protocol-->  Minecraft server

Награды считает тот же RewardEngine и та же база, что и в симуляторе, —
то есть агент продолжает учиться на живом сервере, а не только в песочнице.

Запуск (в двух терминалах):
    node bot/bridge.js --host localhost --port 25565
    python play.py --checkpoint checkpoints/brain.pt --learn
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from brain.algos.dqn import DQNConfig, DQNTrainer
from brain.model import CraftBrain, ModelConfig
from brain.env.mc_env import GOALS as GOALS_LIST
from brain.recipes import (RECIPE_BY_NAME, best_partial, find_completed,
                           prerequisites)
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardBreakdown, RewardEngine
from brain.rewards.rules import seed
from brain.spaces import (ACTIONS, ARMOR_POINTS, DENSE_DIM, EMPTY, ENTITIES,
                          FOOD_VALUE, GRID_2X2_SLOTS, HOSTILE, ID_ITEM, ITEMS,
                          N_ACTIONS, N_GOALS, N_ITEMS, N_MAP, ActionType)


class BridgeClient:
    """Тонкий JSON-line клиент к mineflayer-мосту."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5599,
                 timeout: float = 30.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.fp = self.sock.makefile("rwb")

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.fp.write((json.dumps(payload) + "\n").encode())
        self.fp.flush()
        line = self.fp.readline()
        if not line:
            raise ConnectionError("мост закрыл соединение")
        return json.loads(line.decode())

    def observe(self) -> Dict[str, Any]:
        return self.call({"cmd": "observe"})["obs"]

    def reset(self) -> Dict[str, Any]:
        return self.call({"cmd": "reset"})["obs"]

    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self.call({"cmd": "act", "action": action})

    def chat(self, text: str) -> None:
        self.call({"cmd": "chat", "text": text})

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def bridge_obs_to_model(raw: Dict[str, Any], goal_index: int,
                        engine: "RewardEngine",
                        goal_recipes, step: int = 0,
                        max_steps: int = 500) -> Dict[str, np.ndarray]:
    """
    Наблюдение из Minecraft -> тот же формат, что у симулятора.

    ВАЖНО: вектор обязан совпадать с MinecraftEnv.observe() до последнего
    числа, иначе веса, обученные в симуляторе, читают мусор. Порядок:
        inv | held(4) | orient(6) | goal | paid(11) | craft_state(3)
    """
    inv = np.array(raw["inventory"], dtype=np.float32)
    held = np.zeros(4, dtype=np.float32)
    held[0] = raw["held"] / max(1, N_ITEMS - 1)
    held[1] = float(raw["held"] != 0)
    held[2] = float(raw["near_table"])
    held[3] = min(step / max(1, max_steps), 1.0)
    orient = np.zeros(6, dtype=np.float32)
    orient[int(raw["facing"]) % 4] = 1.0
    orient[4] = (int(raw["pitch"]) + 1) / 2.0
    orient[5] = 1.0 - min(float(raw["table_dist"]), 8.0) / 8.0
    goal = np.zeros(N_GOALS, dtype=np.float32)
    goal[min(goal_index, N_GOALS - 1)] = 1.0

    # Память антифарма — без неё среда немарковская и агент залипает
    # в цикле place <-> clear (эта ошибка уже была поймана в симуляторе).
    ps = engine.paid_state(goal_recipes)
    paid = np.array(ps["slots"] + [ps["shape"], ps["crafts"]], dtype=np.float32)

    # Состояние крафта: агент должен видеть, какая сетка ему доступна.
    craft_state = np.array([
        float(raw.get("table_open", 0)),
        float(raw.get("has_3x3", 0)),
        float(raw.get("table_reachable", 0)),
    ], dtype=np.float32)

    # Выживание: здоровье, голод, броня, враг рядом, дистанция, есть ли еда.
    ent = raw.get("entmap", [])
    hostile_near = 0.0
    ndist = 99.0
    if ent:
        W = int(len(ent) ** 0.5)
        for idx, e in enumerate(ent):
            nm = ENTITIES[e] if e < len(ENTITIES) else "none"
            if nm in HOSTILE:
                i, k = divmod(idx, W)
                d = abs(i - W // 2) + abs(k - W // 2)
                if d < ndist:
                    ndist = d
                    hostile_near = 1.0
    inv_names = {ITEMS[i] for i, v in enumerate(raw["inventory"])
                 if v and i < len(ITEMS)}
    survival = np.array([
        float(raw.get("health", 20)) / 20.0,
        float(raw.get("food", 20)) / 20.0,
        min(sum(ARMOR_POINTS.get(n, 0.0) for n in inv_names), 20.0) / 20.0,
        hostile_near,
        1.0 - min(ndist, 10) / 10.0,
        float(any(n in FOOD_VALUE for n in inv_names)),
    ], dtype=np.float32)

    # Координаты: те же 14 чисел, что и в симуляторе. Живой бот получает их
    # от моста (bot.entity.position), симулятор считает сам — но формат
    # обязан совпадать до числа, иначе веса не перенесутся.
    ax = float(raw.get("x", 0.0))
    ay = float(raw.get("y", 64.0))
    az = float(raw.get("z", 0.0))
    sx = float(raw.get("spawn_x", ax))
    sz = float(raw.get("spawn_z", az))
    tx = float(raw.get("table_x", ax))
    tz = float(raw.get("table_z", az))
    span = float(raw.get("world_span", 64.0)) or 64.0

    dxt, dzt = tx - ax, tz - az
    dxs, dzs = sx - ax, sz - az
    dist_t = float(np.hypot(dxt, dzt))
    dist_s = float(np.hypot(dxs, dzs))
    ang_t = float(np.arctan2(dxt, dzt))
    ang_s = float(np.arctan2(dxs, dzs))
    maxd = span * 1.4142

    def _wrap(v: float) -> float:
        """Позиция внутри условного квадрата мира, 0..1."""
        return float((v % span) / span)

    coords = np.array([
        _wrap(ax),
        min(max(ay, 0.0), 255.0) / 255.0,
        _wrap(az),
        np.sin(ang_t), np.cos(ang_t),
        1.0 - min(dist_t, maxd) / maxd,
        np.sin(ang_s), np.cos(ang_s),
        1.0 - min(dist_s, maxd) / maxd,
        float(raw.get("step_frac", 0.0)),
        _wrap(ax), 1.0 - _wrap(ax),
        _wrap(az), 1.0 - _wrap(az),
    ], dtype=np.float32)

    dense = np.concatenate(
        [inv, held, orient, goal, paid, craft_state, survival, coords]
    ).astype(np.float32)
    assert dense.shape[0] == DENSE_DIM, (dense.shape, DENSE_DIM)
    return {
        "grid": np.array(raw["grid"], dtype=np.int64),
        "voxels": np.array(raw["voxels"], dtype=np.int64),
        "blockmap": np.array(raw.get("blockmap", [0] * N_MAP), dtype=np.int64),
        "entmap": np.array(raw.get("entmap", [0] * N_MAP), dtype=np.int64),
        "dense": dense,
        "held": np.array([raw["held"]], dtype=np.int64),
    }


def action_to_bridge(index: int, grid, has_table: bool = False) -> Dict[str, Any]:
    """
    Действие модели -> команда для mineflayer.

    has_table берётся из реального состояния бота: рецепты 3x3 доступны
    только при открытом верстаке, ровно как в симуляторе.
    """
    a = ACTIONS[index]
    payload: Dict[str, Any] = {"type": a.type.name.lower(), "arg": a.arg}
    if a.type == ActionType.CRAFT:
        recipe = find_completed(list(grid), has_table=has_table)
        payload["result_name"] = recipe.name if recipe else "stick"
    return payload


def mask_from_raw(raw: Dict[str, Any]) -> np.ndarray:
    """
    Маска допустимых действий по состоянию реального бота.

    Зеркало MinecraftEnv.action_mask(): маскируем только физически
    невозможное. Смысловые ошибки оставляем доступными — на их штрафах
    агент и учится.
    """
    m = np.ones(N_ACTIONS, dtype=bool)
    inv = raw["inventory"]
    grid = raw["grid"]
    held = raw["held"]
    held_ok = held != 0 and inv[held] > 0
    # Какие слоты реально доступны: без верстака только 2x2.
    active = set(raw.get("active_slots", GRID_2X2_SLOTS))
    for act in ACTIONS:
        i = act.index
        if act.type == ActionType.SELECT_ITEM:
            m[i] = inv[act.arg] > 0
        elif act.type == ActionType.PLACE_IN_SLOT:
            m[i] = held_ok and grid[act.arg] == 0 and act.arg in active
        elif act.type == ActionType.TAKE_FROM_SLOT:
            m[i] = grid[act.arg] != 0 and act.arg in active
        elif act.type == ActionType.USE_TABLE:
            # Открыть — только если верстак рядом; закрыть — всегда.
            m[i] = bool(raw.get("table_open", 0)) or bool(
                raw.get("table_reachable", 0))
        elif act.type == ActionType.NOOP:
            m[i] = False
    if not m.any():
        m[:] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-host", default="127.0.0.1")
    ap.add_argument("--brain-port", type=int, default=5599)
    ap.add_argument("--checkpoint", default="checkpoints/brain.pt")
    ap.add_argument("--db", default="data/rewards.db")
    ap.add_argument("--goal", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--learn", action="store_true",
                    help="продолжать обучение прямо на сервере")
    ap.add_argument("--delay", type=float, default=0.35)
    args = ap.parse_args()

    db = RewardDB(args.db)
    seed(db)
    engine = RewardEngine(db)
    engine.begin_episode(9000 + int(time.time()) % 1000)

    model = CraftBrain(ModelConfig(head="both"))
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"[model] загружен {args.checkpoint}")
    else:
        print("[model] чекпоинт не найден — играем пустой моделью")
    trainer = DQNTrainer(model, DQNConfig())
    if not args.learn:
        trainer.cfg.eps_start = trainer.cfg.eps_end = 0.02

    cli = BridgeClient(args.brain_host, args.brain_port)
    cli.chat("RL-агент подключился. Учусь крафтить!")
    raw = cli.reset()

    # Какие рецепты ведут к цели — тем же способом, что и в симуляторе.
    goal_name = GOALS_LIST[min(args.goal, len(GOALS_LIST) - 1)]
    goal_recipes = set(prerequisites(goal_name)) if goal_name in RECIPE_BY_NAME else set()
    engine.set_goal_recipes(goal_recipes)
    print(f"[цель] {goal_name} | рецепты: {sorted(goal_recipes) or '— добыча'}")

    obs = bridge_obs_to_model(raw, args.goal, engine, goal_recipes,
                              0, args.steps)
    total = 0.0

    try:
        for step in range(1, args.steps + 1):
            engine.step = step
            mask = mask_from_raw(raw)
            a = trainer.act(obs, mask)
            act_payload = action_to_bridge(a, raw["grid"],
                                           bool(raw.get("has_3x3", 0)))

            grid_before = list(raw["grid"])
            resp = cli.act(act_payload)
            raw = resp["obs"]
            grid_after = list(raw["grid"])
            ok = resp.get("result", {}).get("ok", True)

            # --- награда считается ровно теми же правилами, что и в симуляторе
            br = RewardBreakdown()
            spec = ACTIONS[a]
            if spec.type == ActionType.PLACE_IN_SLOT and ok:
                sub = engine.on_place_in_grid(grid_before, grid_after, spec.arg,
                                              grid_after[spec.arg],
                                              bool(raw["near_table"]))
                br.total += sub.total; br.parts += sub.parts
            elif spec.type == ActionType.TAKE_FROM_SLOT and ok:
                sub = engine.on_take_from_grid(grid_before, grid_after, spec.arg,
                                               bool(raw["near_table"]))
                br.total += sub.total; br.parts += sub.parts
            elif spec.type == ActionType.CRAFT:
                name = act_payload.get("result_name")
                recipe = RECIPE_BY_NAME.get(name) if ok else None
                sub = engine.on_craft(recipe, ok and recipe is not None)
                br.total += sub.total; br.parts += sub.parts
            elif spec.type == ActionType.BREAK_BLOCK:
                # Не тот инструмент — блок сломан, а дропа нет. Тот же
                # штраф, что и в симуляторе.
                res = resp.get("result", {})
                if res.get("wrong_tool"):
                    sub = engine.on_world(
                        "world.wrong_tool", ctx=str(res.get("block", "")),
                        reason="неподходящий инструмент — дроп не выпал")
                    br.total += sub.total; br.parts += sub.parts
                elif ok:
                    sub = engine.on_world(
                        "world.gather", ctx=str(res.get("block", "")),
                        reason="добыл ресурс")
                    br.total += sub.total; br.parts += sub.parts
            elif spec.type == ActionType.USE_TABLE and ok:
                sub = engine.on_world("world.use_table",
                                      reason="открыл/закрыл верстак")
                br.total += sub.total; br.parts += sub.parts
            sub = engine.on_step_overhead(a, ok, spec.type == ActionType.NOOP)
            br.total += sub.total; br.parts += sub.parts

            nobs = bridge_obs_to_model(raw, args.goal, engine, goal_recipes,
                                       step, args.steps)
            if args.learn:
                trainer.buffer.push(obs, a, br.total, nobs, False, mask_from_raw(raw))
                trainer.learn()
            obs = nobs
            total += br.total

            flag = "" if ok else " ✗"
            print(f"ш{step:4d} {spec.name:<22} r={br.total:+7.2f} "
                  f"Σ={total:+9.2f}{flag}")
            for k, v, why in br.parts[:3]:
                print(f"        {v:+6.2f} {k} — {why[:50]}")
            if args.delay:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nостановлено пользователем")
    finally:
        engine.end_episode(total, max(total, 0), min(total, 0))
        if args.learn:
            torch.save(model.state_dict(), args.checkpoint)
            print(f"[model] сохранено -> {args.checkpoint}")
        cli.close()
        db.close()


if __name__ == "__main__":
    main()

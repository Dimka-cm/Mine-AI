"""
Локальный симулятор Minecraft-подобной среды.

Нужен, чтобы модель можно было обучать без запущенного сервера Minecraft.
Интерфейс намеренно повторяет то, что отдаёт мост mineflayer, поэтому одни и
те же веса работают и тут, и в реальной игре.

Что моделируем:
  * воксельный мир (сетка блоков) + позиция/поворот/наклон агента;
  * инвентарь и предмет "в руке";
  * сетку крафта 3x3 и верстак;
  * curriculum целей: доски -> палки -> верстак -> деревянный меч -> ... -> железный.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..recipes import (RECIPE_BY_NAME, Recipe, best_partial,
                       find_completed, prerequisites)
from ..rewards.engine import RewardBreakdown, RewardEngine
from ..spaces import (
    N_COORDS,ACTIONS, AIR, ARMOR_POINTS, BLOCK_DROP, BLOCK_ID,
                      BLOCK_REQUIRED_LEVEL, DAMAGING, DENSE_DIM, EMPTY,
                      ENTITY_ID, FOOD_VALUE, GRID_2X2_SLOTS, GRID_3X3_SLOTS,
                      HOSTILE, ID_ENTITY, MAP_R, MAP_W, N_ENT_MAP, N_MAP,
                      PASSABLE, PASSIVE_DROP, can_harvest, tool_level,
                      weapon_damage,
                      FACING_DELTA, GRID_SIZE, ID_BLOCK, ID_ITEM, ITEM_TO_BLOCK,
                      N_ACTIONS, N_BLOCKS, N_GOALS, N_ITEMS, N_VOXELS, PITCHES,
                      VIEW, ActionType, Facing, item_id)

# Curriculum: агент проходит цели по порядку, каждая открывает следующую.
# Curriculum теперь начинается с ДОБЫЧИ: у агента пустой инвентарь, он обязан
# сам найти дерево, сломать его, подобрать дроп и только потом крафтить.
# Верстак — отдельная веха: пока он не поставлен и не открыт, доступна
# только инвентарная сетка 2x2, как в настоящем Minecraft.
GOALS: List[str] = [
    "oak_log",          # сломать дерево голыми руками
    "oak_planks",       # 2x2: бревно -> доски
    "stick",            # 2x2: доски -> палки
    "crafting_table",   # 2x2: доски -> верстак
    "wooden_pickaxe",   # 3x3! требует открытого верстака
    "wooden_sword",
    "cobblestone",      # добыть камень киркой
    "stone_pickaxe",
    "stone_sword",
    "raw_iron",
    "iron_pickaxe",
    "iron_sword",
    "diamond",
    "diamond_sword",
]

WORLD_W, WORLD_H, WORLD_D = 12, 5, 12


@dataclass
class AgentState:
    x: int = 6
    y: int = 1
    z: int = 6
    facing: int = int(Facing.SOUTH)
    pitch: int = 0            # -1 вниз, 0 прямо, +1 вверх
    held: int = EMPTY         # предмет в руке
    inventory: Dict[int, int] = field(default_factory=dict)
    grid: List[int] = field(default_factory=lambda: [EMPTY] * GRID_SIZE)
    near_table: bool = False
    # Верстак ОТКРЫТ -> доступна сетка 3x3. Иначе только 2x2 из инвентаря.
    table_open: bool = False
    # --- выживание ---
    health: float = 20.0
    food: float = 20.0


@dataclass
class Mob:
    """Моб в мире. Свой класс, чтобы среда умела ими управлять."""

    kind: str                 # 'zombie', 'cow', ...
    x: int
    y: int
    z: int
    health: float = 20.0

    @property
    def hostile(self) -> bool:
        return self.kind in HOSTILE


class MinecraftCraftEnv:
    """Gym-подобная среда. reset() / step(action) -> obs, reward, done, info."""

    def __init__(self, engine: RewardEngine, *, max_steps: int = 300,
                 seed: Optional[int] = None, generous_inventory: bool = True,
                 goal_index: int = 0, mask_level: str = "smart",
                 mobs: bool = True, n_hostile: int = 1) -> None:
        assert mask_level in ("off", "physical", "smart"), mask_level
        # Мобов можно выключить — так старые тесты на крафт остаются честными.
        self.mobs_enabled = mobs
        self.n_hostile = n_hostile
        self.mobs: List["Mob"] = []
        self.engine = engine
        self.mask_level = mask_level
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.generous = generous_inventory
        self.goal_index = goal_index
        self.episode = 0

        self.agent = AgentState()
        self.world = np.zeros((WORLD_W, WORLD_H, WORLD_D), dtype=np.int16)
        self.steps = 0
        self.done = False
        self._pos_reward = 0.0
        self._neg_reward = 0.0
        self._prev_dist: Optional[int] = None
        self._faced_target_once = False
        self._current_goal_recipes: set = set()

    # ---------------- мир ----------------
    def _generate_world(self) -> None:
        w = np.zeros((WORLD_W, WORLD_H, WORLD_D), dtype=np.int16)
        w[:, 0, :] = BLOCK_ID["bedrock"]
        w[:, 1, :] = BLOCK_ID["grass_block"]
        # ГРАНИЦА МИРА — видимая стена по периметру.
        # Раньше край мира был невидим: пол просто обрывался, и агент не мог
        # глазами понять, что впереди стена. Замер это подтвердил — признак
        # "расстояние до стены" не выучивался вообще (ошибка как у монетки).
        # Двухблочная ограда даёт зримый ориентир и делает границу честной:
        # её видно на картинке, а не только в служебных числах.
        # У каждой стороны СВОЙ материал — это компас, видимый глазами.
        # Если все четыре стены одинаковые, стороны света в принципе не
        # различить по картинке: вид на север и на юг совпадает до пикселя.
        # Замер: одинаковые стены -> 27% угадывания стороны (случайно 25%),
        # разные -> 54%. Материалы взяты недобываемые и непохожие по цвету,
        # чтобы они не превращались в добычу и не путались с рудой.
        WALL_MARKS = {
            "-X": BLOCK_ID["bedrock"],      # тёмно-серый
            "+X": BLOCK_ID["sand_block"],   # песочный
            "-Z": BLOCK_ID["gravel"],       # крапчатый серый
            "+Z": BLOCK_ID["dirt"],         # коричневый (листва не годится:
            #                                  сквозь неё можно пройти,
            #                                  и в стене была бы дыра)
        }
        for yy in (2, 3):
            w[0, yy, :] = WALL_MARKS["-X"]
            w[WORLD_W - 1, yy, :] = WALL_MARKS["+X"]
            w[:, yy, 0] = WALL_MARKS["-Z"]
            w[:, yy, WORLD_D - 1] = WALL_MARKS["+Z"]
        # Углы возвращаем к bedrock: так угол читается как угол, а не как
        # обрыв одной стены в другую.
        for yy in (2, 3):
            for cx in (0, WORLD_W - 1):
                for cz in (0, WORLD_D - 1):
                    w[cx, yy, cz] = BLOCK_ID["bedrock"]
        # немного ресурсов вокруг
        def _free_xz():
            """Случайная клетка ВНУТРИ ограды — чтобы руда не легла в стену."""
            return (self.rng.randrange(1, WORLD_W - 1),
                    self.rng.randrange(1, WORLD_D - 1))

        for _ in range(10):
            x, z = _free_xz()
            w[x, 2, z] = BLOCK_ID["oak_log"]
        for _ in range(14):
            x, z = _free_xz()
            w[x, 2, z] = BLOCK_ID["stone"]
        for _ in range(5):
            x, z = _free_xz()
            w[x, 2, z] = BLOCK_ID["iron_ore"]
        for _ in range(2):
            x, z = _free_xz()
            w[x, 2, z] = BLOCK_ID["diamond_ore"]
        self.world = w
        self.agent.x, self.agent.y, self.agent.z = WORLD_W // 2, 2, WORLD_D // 2
        self.world[self.agent.x, self.agent.y, self.agent.z] = AIR
        self._spawn_mobs()
        # Верстака в мире НЕТ — агент обязан скрафтить его сам и поставить.
        # table_pos указывает на ближайшее дерево: это цель навигации на
        # старте игры (надо дойти до дерева и сломать его).
        self.table_pos = self._nearest_block(BLOCK_ID["oak_log"])

    def _spawn_mobs(self) -> None:
        """
        Расставить мобов. Враждебных — подальше от агента, чтобы он не умирал
        на первом же шаге, но достаточно близко, чтобы они были угрозой.
        """
        self.mobs = []
        if not self.mobs_enabled:
            return
        ax, az = self.agent.x, self.agent.z
        # Опасность НАРАСТАЕТ вместе с curriculum: на первых целях (найти
        # дерево) враждебных мобов нет вовсе, иначе агент гибнет раньше, чем
        # успевает понять правила. Чем выше тир цели — тем опаснее мир.
        goal = GOALS[min(self.goal_index, len(GOALS) - 1)]
        tier = self._goal_tier(goal)
        n_hostile = 0 if tier == 0 else min(self.n_hostile, tier)
        # Враждебные спавнятся ДАЛЕКО: иначе случайный агент погибает за
        # несколько шагов и обучаться просто не на чем (проверено: 6 смертей
        # из 6 эпизодов при min_d=4). Мирных можно ставить ближе.
        for kind, n, min_d in (("zombie", n_hostile, 7),
                               ("cow", 2, 2), ("pig", 1, 2)):
            for _ in range(n):
                for _try in range(30):
                    x = self.rng.randrange(WORLD_W)
                    z = self.rng.randrange(WORLD_D)
                    if abs(x - ax) + abs(z - az) < min_d:
                        continue
                    if int(self.world[x, 2, z]) != AIR:
                        continue
                    hp = HOSTILE.get(kind, {}).get("health", 10.0)
                    self.mobs.append(Mob(kind, x, 2, z, hp))
                    break

    def _mobs_act(self) -> RewardBreakdown:
        """
        Ход мобов: враждебные идут к агенту и бьют, если дотянулись.

        Это и есть причина, по которой агенту нужно зрение и оружие.
        """
        br = RewardBreakdown()
        if not self.mobs_enabled:
            return br
        for m in self.mobs:
            if m.health <= 0 or not m.hostile:
                continue
            spec = HOSTILE[m.kind]
            d = abs(m.x - self.agent.x) + abs(m.z - self.agent.z)
            # Мобы медленнее агента: шаг раз в три тика. Иначе убежать
            # невозможно в принципе, зрение бесполезно, и единственная
            # выученная стратегия — умереть побыстрее.
            if d > spec["range"] and self.steps % 3 != 0:
                continue
            # Далёкий моб агента ещё не заметил (как радиус агро в игре).
            if d > 8:
                continue
            if d <= spec["range"]:
                # Мобы бьют не каждый шаг — в игре есть перезарядка удара.
                # Без неё зомби выносит агента за 8 шагов подряд и учиться
                # не на чем (проверено: 100% смертей у случайной политики).
                if self.steps % 3 != 0:
                    continue
                dmg = spec["damage"] * (1.0 - min(self._armor_points(), 16.0) / 32.0)
                self.agent.health = max(0.0, self.agent.health - dmg)
                self._merge(br, self.engine.on_world(
                    "survival.hurt", ctx=m.kind,
                    reason=f"{m.kind} ударил на {dmg:.1f}"))
            else:
                # шаг в сторону агента по той оси, где дальше
                if abs(m.x - self.agent.x) >= abs(m.z - self.agent.z):
                    m.x += 1 if self.agent.x > m.x else -1
                else:
                    m.z += 1 if self.agent.z > m.z else -1
                m.x = max(0, min(WORLD_W - 1, m.x))
                m.z = max(0, min(WORLD_D - 1, m.z))
        return br

    def _mob_in_front(self) -> Optional["Mob"]:
        """Моб в клетке прямо перед агентом (или вплотную)."""
        fx, fy, fz = self._front()
        for m in self.mobs:
            if m.health > 0 and m.x == fx and m.z == fz:
                return m
        return None

    def _do_attack(self) -> Tuple[bool, RewardBreakdown]:
        """Ударить моба перед собой."""
        br = RewardBreakdown()
        m = self._mob_in_front()
        if m is None:
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason="бить некого"))
            return False, br
        held = ID_ITEM.get(self.agent.held, "empty")
        dmg = weapon_damage(held)
        m.health -= dmg
        if m.health <= 0:
            drop = PASSIVE_DROP.get(m.kind)
            if drop:
                self._give(item_id(drop), 1)
            self._merge(br, self.engine.on_world(
                "survival.kill", ctx=m.kind, reason=f"убил {m.kind}"))
        else:
            self._merge(br, self.engine.on_world(
                "survival.hit", ctx=m.kind, reason=f"попал по {m.kind}"))
        return True, br

    def _do_eat(self) -> Tuple[bool, RewardBreakdown]:
        """Съесть самую питательную еду из инвентаря."""
        br = RewardBreakdown()
        best, best_val = None, 0.0
        for item, cnt in self.agent.inventory.items():
            if cnt <= 0:
                continue
            val = FOOD_VALUE.get(ID_ITEM.get(item, ""), 0.0)
            if val > best_val:
                best, best_val = item, val
        if best is None:
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason="еды нет"))
            return False, br
        if self.agent.food >= 20.0:
            self._merge(br, self.engine.on_world(
                "survival.eat_full", reason="сыт, еда потрачена зря"))
            return False, br
        self._take(best, 1)
        self.agent.food = min(20.0, self.agent.food + best_val)
        self._merge(br, self.engine.on_world(
            "survival.eat", ctx=ID_ITEM.get(best, ""), reason="поел"))
        return True, br

    def _nearest_block(self, block_id: int) -> Tuple[int, int, int]:
        """Ближайший блок такого типа — служит целью навигации."""
        best = None
        bd = 10 ** 9
        for x in range(WORLD_W):
            for y in range(1, WORLD_H):
                for z in range(WORLD_D):
                    if int(self.world[x, y, z]) == block_id:
                        d = abs(x - self.agent.x) + abs(z - self.agent.z)
                        if d < bd:
                            bd, best = d, (x, y, z)
        return best or (self.agent.x, self.agent.y, self.agent.z)

    def _starting_inventory(self) -> Dict[int, int]:
        """
        Стартовый инвентарь.

        По умолчанию — ПУСТОЙ: агент начинает с голыми руками, как в новой
        игре, и обязан сам добыть всё дерево/камень/руду. Это делает задачу
        честной, но заметно длиннее.

        generous_inventory=True выдаёт минимальный набор для ВЕРХНИХ целей
        (железо, алмазы), иначе один эпизод должен был бы вместить всю игру
        от первого бревна до алмазной кирки. Для нижних целей набор всё равно
        пустой — их агент проходит с нуля.
        """
        inv: Dict[int, int] = {}
        if not self.generous:
            return inv
        goal = GOALS[min(self.goal_index, len(GOALS) - 1)]
        tier = self._goal_tier(goal)
        # Выдаём ИНСТРУМЕНТ предыдущего тира, а не готовое сырьё: иначе
        # нарушается сама цепочка "кирка открывает следующий материал".
        if tier >= 2:          # каменные цели: дерево + деревянная кирка
            inv[item_id("oak_planks")] = 4
            inv[item_id("stick")] = 4
            inv[item_id("crafting_table")] = 1
            inv[item_id("wooden_pickaxe")] = 1
        if tier >= 3:          # железные цели: нужна КАМЕННАЯ кирка
            inv[item_id("stone_pickaxe")] = 1
            inv[item_id("cobblestone")] = 4
        if tier >= 4:          # алмазные цели: нужна ЖЕЛЕЗНАЯ кирка
            inv[item_id("iron_pickaxe")] = 1
            inv[item_id("iron_ingot")] = 3
        return inv

    @staticmethod
    def _goal_tier(goal: str) -> int:
        """0 - добыча дерева, 1 - деревянные, 2 - каменные, 3 - железо, 4 - алмаз."""
        if goal.startswith("diamond"):
            return 4
        if goal.startswith("iron") or goal == "raw_iron":
            return 3
        if goal.startswith("stone") or goal == "cobblestone":
            return 2
        if goal.startswith("wooden"):
            return 1
        return 0

    # ---------------- reset ----------------
    def reset(self, goal_index: Optional[int] = None) -> np.ndarray:
        if goal_index is not None:
            self.goal_index = goal_index
        self.episode += 1
        self.steps = 0
        self.done = False
        self._pos_reward = 0.0
        self._neg_reward = 0.0
        self._faced_target_once = False
        self.agent = AgentState()
        self.mobs = []
        self._generate_world()
        self.agent.inventory = self._starting_inventory()
        self.agent.held = EMPTY
        self.agent.grid = [EMPTY] * GRID_SIZE
        self.agent.table_open = False
        self.agent.near_table = self._is_near_table()
        self._spawn_xz = (self.agent.x, self.agent.z)
        self._prev_dist = self._dist_to_table()
        self.engine.begin_episode(self.episode)
        # Плотные награды — только за рецепты, ведущие к текущей цели.
        self._sync_goal_recipes()
        return self.observe()

    # ---------------- наблюдение ----------------
    def _voxels(self) -> np.ndarray:
        """
        БЛИЖНЕЕ 3D-ЗРЕНИЕ: куб 5x5x5 вокруг агента, в системе его взгляда.

        Было 3x3x3 — агент видел буквально вплотную и погибал от мобов,
        которых физически не мог заметить. 5x5x5 даёт два блока запаса
        во все стороны: этого хватает, чтобы среагировать.
        """
        out = np.zeros((VIEW, VIEW, VIEW), dtype=np.int64)
        half = VIEW // 2
        for i, dx in enumerate(range(-half, half + 1)):
            for j, dy in enumerate(range(-half, half + 1)):
                for k, dz in enumerate(range(-half, half + 1)):
                    # поворачиваем смещения так, чтобы "вперёд" всегда был +z
                    fx, fz = self._rotate(dx, dz, self.agent.facing)
                    x, y, z = self.agent.x + fx, self.agent.y + dy, self.agent.z + fz
                    if 0 <= x < WORLD_W and 0 <= y < WORLD_H and 0 <= z < WORLD_D:
                        out[i, j, k] = int(self.world[x, y, z])
                    else:
                        out[i, j, k] = BLOCK_ID["bedrock"]
        return out.reshape(-1)

    def _blockmap(self) -> np.ndarray:
        """
        ДАЛЬНЕЕ 2D-ЗРЕНИЕ: карта сверху 13x13, повёрнутая по взгляду.

        Это «миникарта»: агент видит, ГДЕ лес и руда, а не только что у него
        под носом. Берём верхний непустой блок в каждой колонке — как если бы
        смотрели на местность сверху.
        """
        out = np.zeros((MAP_W, MAP_W), dtype=np.int64)
        for i, dx in enumerate(range(-MAP_R, MAP_R + 1)):
            for k, dz in enumerate(range(-MAP_R, MAP_R + 1)):
                fx, fz = self._rotate(dx, dz, self.agent.facing)
                x, z = self.agent.x + fx, self.agent.z + fz
                if not (0 <= x < WORLD_W and 0 <= z < WORLD_D):
                    out[i, k] = BLOCK_ID["bedrock"]
                    continue
                top = AIR
                for y in range(WORLD_H - 1, 0, -1):
                    b = int(self.world[x, y, z])
                    if b != AIR and ID_BLOCK.get(b) not in PASSABLE:
                        top = b
                        break
                out[i, k] = top
        return out.reshape(-1)

    def _entmap(self) -> np.ndarray:
        """
        2D-КАРТА СУЩНОСТЕЙ: кто где стоит, в тех же координатах, что blockmap.

        Отдельный слой, а не вперемешку с блоками: моб и блок — принципиально
        разные вещи, и сеть должна их различать, а не угадывать по id.
        """
        out = np.zeros((MAP_W, MAP_W), dtype=np.int64)
        for m in self.mobs:
            if m.health <= 0:
                continue
            dx, dz = m.x - self.agent.x, m.z - self.agent.z
            # обратный поворот: мир -> система взгляда агента
            rx, rz = self._unrotate(dx, dz, self.agent.facing)
            i, k = rx + MAP_R, rz + MAP_R
            if 0 <= i < MAP_W and 0 <= k < MAP_W:
                out[i, k] = ENTITY_ID.get(m.kind, 0)
        return out.reshape(-1)

    @staticmethod
    def _unrotate(dx: int, dz: int, facing: int) -> Tuple[int, int]:
        """Обратное к _rotate: из координат мира в систему взгляда."""
        if facing == Facing.SOUTH:
            return dx, dz
        if facing == Facing.WEST:
            return dz, -dx
        if facing == Facing.NORTH:
            return -dx, -dz
        return -dz, dx  # EAST

    @staticmethod
    def _rotate(dx: int, dz: int, facing: int) -> Tuple[int, int]:
        if facing == Facing.SOUTH:
            return dx, dz
        if facing == Facing.WEST:
            return -dz, dx
        if facing == Facing.NORTH:
            return -dx, -dz
        return dz, -dx  # EAST

    def observe(self) -> Dict[str, np.ndarray]:
        inv = np.zeros(N_ITEMS, dtype=np.float32)
        for k, v in self.agent.inventory.items():
            inv[k] = min(v, 64) / 64.0
        held = np.zeros(4, dtype=np.float32)
        held[0] = self.agent.held / max(1, N_ITEMS - 1)
        held[1] = float(self.agent.held != EMPTY)
        held[2] = float(self.agent.near_table)
        held[3] = self.steps / self.max_steps

        # ориентация: 4 стороны + 3 наклона, one-hot
        orient = np.zeros(6, dtype=np.float32)
        orient[self.agent.facing] = 1.0
        orient[4] = (self.agent.pitch + 1) / 2.0
        d = self._dist_to_table()
        orient[5] = 1.0 - min(d, 8) / 8.0

        goal = np.zeros(N_GOALS, dtype=np.float32)
        goal[min(self.goal_index, N_GOALS - 1)] = 1.0

        # Память движка наград — агент обязан её видеть, иначе одинаковые
        # состояния дают разную награду и учиться не на чем.
        ps = self.engine.paid_state(self._current_goal_recipes)
        paid = np.array(ps["slots"] + [ps["shape"], ps["crafts"]],
                        dtype=np.float32)

        # Состояние крафта — агент должен ВИДЕТЬ, какая сетка ему доступна.
        craft_state = np.array([
            float(self.agent.table_open),
            float(self.has_3x3()),
            float(self._table_reachable()),
        ], dtype=np.float32)

        # Состояние выживания — без него агент не научится убегать и есть.
        nearest, ndist = self._nearest_hostile()
        survival = np.array([
            self.agent.health / 20.0,
            self.agent.food / 20.0,
            min(self._armor_points(), 20.0) / 20.0,
            1.0 if nearest is not None else 0.0,
            1.0 - min(ndist, 10) / 10.0,        # 1 = враг вплотную
            float(self._has_food()),
        ], dtype=np.float32)

        # --- КООРДИНАТЫ -------------------------------------------------
        # Абсолютные ("где я") + относительные ("куда идти"). Углы даём
        # через sin/cos: иначе на переходе 359->0 градусов получается
        # скачок, который сеть воспринимает как телепортацию.
        W, H, D = self.world.shape
        ax, ay, az = self.agent.x, self.agent.y, self.agent.z

        tx, tz = self._table_xz()
        dxt, dzt = (tx - ax), (tz - az)
        dist_t = float(np.hypot(dxt, dzt))
        ang_t = np.arctan2(dxt, dzt)

        sx, sz = self._spawn_xz
        dxs, dzs = (sx - ax), (sz - az)
        dist_s = float(np.hypot(dxs, dzs))
        ang_s = np.arctan2(dxs, dzs)

        maxd = float(np.hypot(W, D))
        coords = np.array([
            ax / max(W - 1, 1),
            ay / max(H - 1, 1),
            az / max(D - 1, 1),
            np.sin(ang_t), np.cos(ang_t),
            1.0 - min(dist_t, maxd) / maxd,
            np.sin(ang_s), np.cos(ang_s),
            1.0 - min(dist_s, maxd) / maxd,
            self.steps / max(self.max_steps, 1),
            ax / max(W - 1, 1),                    # близость к стене -X
            1.0 - ax / max(W - 1, 1),              # к стене +X
            az / max(D - 1, 1),                    # к стене -Z
            1.0 - az / max(D - 1, 1),              # к стене +Z
        ], dtype=np.float32)

        dense = np.concatenate(
            [inv, held, orient, goal, paid, craft_state, survival, coords]
        ).astype(np.float32)
        assert dense.shape[0] == DENSE_DIM, (dense.shape, DENSE_DIM)
        return {
            "grid": np.array(self.agent.grid, dtype=np.int64),
            "voxels": self._voxels(),
            "blockmap": self._blockmap(),
            "entmap": self._entmap(),
            "dense": dense,
            "held": np.array([self.agent.held], dtype=np.int64),
        }

    # ---------------- выживание ----------------
    def _armor_points(self) -> float:
        """Сколько брони надето (в симуляторе — просто наличие в инвентаре)."""
        total = 0.0
        for item, cnt in self.agent.inventory.items():
            if cnt > 0:
                total += ARMOR_POINTS.get(ID_ITEM.get(item, ""), 0.0)
        return total

    def _has_food(self) -> bool:
        return any(ID_ITEM.get(i, "") in FOOD_VALUE and c > 0
                   for i, c in self.agent.inventory.items())

    def _nearest_hostile(self) -> Tuple[Optional["Mob"], int]:
        """Ближайший враг и расстояние до него."""
        best, bd = None, 10 ** 9
        for m in self.mobs:
            if m.health <= 0 or not m.hostile:
                continue
            d = abs(m.x - self.agent.x) + abs(m.z - self.agent.z)
            if d < bd:
                best, bd = m, d
        return best, (bd if best else 99)

    # ---------------- сетка крафта: 2x2 или 3x3 ----------------
    def active_slots(self) -> Tuple[int, ...]:
        """
        Какие слоты сетки сейчас доступны.

        Как в Minecraft: в инвентаре у игрока сетка 2x2 (слоты 0,1,3,4).
        Полная 3x3 открывается ТОЛЬКО когда агент стоит у верстака и
        открыл его действием open_close_table.
        """
        if self.agent.table_open and self._table_reachable():
            return GRID_3X3_SLOTS
        return GRID_2X2_SLOTS

    def _table_reachable(self) -> bool:
        """Верстак прямо перед агентом или вплотную рядом."""
        if self._block_at(self._front()) == BLOCK_ID["crafting_table"]:
            return True
        x, y, z = self.agent.x, self.agent.y, self.agent.z
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if self._block_at((x + dx, y, z + dz)) == BLOCK_ID["crafting_table"]:
                return True
        return False

    def has_3x3(self) -> bool:
        return len(self.active_slots()) == 9

    # ---------------- геометрия ----------------
    def _front(self) -> Tuple[int, int, int]:
        dx, dy, dz = FACING_DELTA[self.agent.facing]
        return (self.agent.x + dx, self.agent.y + self.agent.pitch, self.agent.z + dz)

    def _block_at(self, pos: Tuple[int, int, int]) -> int:
        x, y, z = pos
        if 0 <= x < WORLD_W and 0 <= y < WORLD_H and 0 <= z < WORLD_D:
            return int(self.world[x, y, z])
        return BLOCK_ID["bedrock"]

    def _dist_to_table(self) -> int:
        tx, _, tz = self.table_pos
        return abs(self.agent.x - tx) + abs(self.agent.z - tz)

    def _is_near_table(self) -> bool:
        return self._dist_to_table() <= 1

    def _facing_table(self) -> bool:
        return self._block_at(self._front()) == BLOCK_ID["crafting_table"]

    # ---------------- инвентарь ----------------
    def _take(self, item: int, n: int = 1) -> bool:
        have = self.agent.inventory.get(item, 0)
        if have < n:
            return False
        if have == n:
            del self.agent.inventory[item]
        else:
            self.agent.inventory[item] = have - n
        return True

    def _give(self, item: int, n: int = 1) -> None:
        if item == EMPTY:
            return
        self.agent.inventory[item] = self.agent.inventory.get(item, 0) + n

    # ---------------- шаг ----------------
    def step(self, action_index: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        assert 0 <= action_index < N_ACTIONS
        act = ACTIONS[action_index]
        self.steps += 1
        self.engine.step = self.steps

        br = RewardBreakdown()
        valid = True
        info: Dict[str, object] = {"action": act.name}
        self._sync_goal_recipes()

        if act.type == ActionType.SELECT_ITEM:
            if self.agent.inventory.get(act.arg, 0) > 0:
                self.agent.held = act.arg
            else:
                valid = False

        elif act.type == ActionType.PLACE_IN_SLOT:
            valid, sub = self._do_place_slot(act.arg)
            self._merge(br, sub)

        elif act.type == ActionType.TAKE_FROM_SLOT:
            valid, sub = self._do_take_slot(act.arg)
            self._merge(br, sub)

        elif act.type == ActionType.CRAFT:
            valid, sub = self._do_craft()
            self._merge(br, sub)
            info["crafted"] = sub.parts[0][0] if sub.parts else None

        elif act.type == ActionType.CLEAR_GRID:
            for i, v in enumerate(self.agent.grid):
                if v != EMPTY:
                    self._give(v, 1)
            self.agent.grid = [EMPTY] * GRID_SIZE

        elif act.type == ActionType.TURN:
            self._merge(br, self._do_turn(act.arg))

        elif act.type == ActionType.MOVE:
            valid, sub = self._do_move(act.arg)
            self._merge(br, sub)

        elif act.type == ActionType.PLACE_BLOCK:
            valid, sub = self._do_place_block()
            self._merge(br, sub)

        elif act.type == ActionType.BREAK_BLOCK:
            valid, sub = self._do_break_block()
            self._merge(br, sub)

        elif act.type == ActionType.USE_TABLE:
            valid, sub = self._do_use_table()
            self._merge(br, sub)

        elif act.type == ActionType.ATTACK:
            valid, sub = self._do_attack()
            self._merge(br, sub)

        elif act.type == ActionType.EAT:
            valid, sub = self._do_eat()
            self._merge(br, sub)

        # Общие издержки. Важно: если действие уже получило собственный
        # содержательный штраф (краф впустую, шаг в стену, попытка положить
        # пустоту), второй раз бить за "invalid_action" нельзя — иначе одна
        # ошибка стоит двойную цену и агент переучивается бояться действия,
        # а не ситуации.
        already_punished = act.type in (
            ActionType.CRAFT, ActionType.MOVE, ActionType.PLACE_IN_SLOT,
            ActionType.PLACE_BLOCK, ActionType.ATTACK, ActionType.EAT,
        )
        self._merge(br, self.engine.on_step_overhead(
            action_index, valid or already_punished,
            act.type == ActionType.NOOP))

        # --- мир живёт своей жизнью: мобы ходят и бьют ---
        self._merge(br, self._mobs_act())

        # Голод медленно убывает; на нуле начинает грызть здоровье.
        self.agent.food = max(0.0, self.agent.food - 0.04)
        if self.agent.food <= 0.0:
            self.agent.health = max(0.0, self.agent.health - 0.25)

        # Лава и прочие опасные блоки под ногами.
        here = ID_BLOCK.get(self._block_at(
            (self.agent.x, self.agent.y, self.agent.z)), "air")
        if here in DAMAGING:
            self.agent.health = max(0.0, self.agent.health - DAMAGING[here])
            self._merge(br, self.engine.on_world(
                "survival.hurt", ctx=here, reason=f"стоит в {here}"))

        # проверка цели curriculum
        goal_name = GOALS[min(self.goal_index, len(GOALS) - 1)]
        if self.agent.inventory.get(item_id(goal_name), 0) > 0:
            self._merge(br, self.engine.on_goal(goal_name))
            self.done = True
            info["goal_reached"] = goal_name

        # СМЕРТЬ — эпизод обрывается, и это должно быть больно.
        if self.agent.health <= 0.0:
            self._merge(br, self.engine.on_world(
                "survival.death", reason="агент погиб"))
            self.done = True
            info["died"] = True

        self.agent.near_table = self._is_near_table()
        if self.steps >= self.max_steps:
            self.done = True
            info["timeout"] = True

        reward = br.total
        if reward > 0:
            self._pos_reward += reward
        else:
            self._neg_reward += reward
        info["breakdown"] = br.as_dict()

        if self.done:
            self.engine.end_episode(self._pos_reward + self._neg_reward,
                                    self._pos_reward, self._neg_reward)
        return self.observe(), reward, self.done, info

    @staticmethod
    def _merge(dst: RewardBreakdown, src: RewardBreakdown) -> None:
        dst.total += src.total
        dst.parts.extend(src.parts)

    # ---------------- реализация действий ----------------
    def _do_place_slot(self, slot: int) -> Tuple[bool, RewardBreakdown]:
        br = RewardBreakdown()
        item = self.agent.held
        if item == EMPTY or self.agent.inventory.get(item, 0) <= 0:
            self._merge(br, self.engine.on_world(
                "grid.place_no_item", reason="в руке пусто"))
            return False, br
        if self.agent.grid[slot] != EMPTY:
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason=f"слот {slot} занят"))
            return False, br

        before = list(self.agent.grid)
        self._take(item, 1)
        self.agent.grid[slot] = item
        has_table = self.has_3x3()
        self._merge(br, self.engine.on_place_in_grid(
            before, list(self.agent.grid), slot, item, has_table))
        if self.agent.inventory.get(item, 0) == 0:
            self.agent.held = EMPTY
        return True, br

    def _do_take_slot(self, slot: int) -> Tuple[bool, RewardBreakdown]:
        br = RewardBreakdown()
        item = self.agent.grid[slot]
        if item == EMPTY:
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason=f"слот {slot} пуст"))
            return False, br
        before = list(self.agent.grid)
        self.agent.grid[slot] = EMPTY
        self._give(item, 1)
        has_table = self.has_3x3()
        self._merge(br, self.engine.on_take_from_grid(
            before, list(self.agent.grid), slot, has_table))
        return True, br

    def _do_craft(self) -> Tuple[bool, RewardBreakdown]:
        # Рецепты 3x3 доступны только при ОТКРЫТОМ верстаке.
        has_table = self.has_3x3()
        recipe = find_completed(self.agent.grid, has_table)
        if recipe is None:
            return False, self.engine.on_craft(None, False)
        # расходуем сетку, выдаём результат
        self.agent.grid = [EMPTY] * GRID_SIZE
        self._give(recipe.result, recipe.count)
        return True, self.engine.on_craft(recipe, True)

    def _do_turn(self, arg: int) -> RewardBreakdown:
        if arg == 0:
            self.agent.facing = (self.agent.facing + 1) % 4
        elif arg == 1:
            self.agent.facing = (self.agent.facing - 1) % 4
        elif arg == 2:
            self.agent.pitch = min(1, self.agent.pitch + 1)
        else:
            self.agent.pitch = max(-1, self.agent.pitch - 1)

        br = RewardBreakdown()
        if self._facing_table() and not self._faced_target_once:
            self._faced_target_once = True
            self._merge(br, self.engine.on_world(
                "spatial.face_target", ctx="crafting_table",
                reason="повернулся лицом к верстаку"))
        return br

    def _do_move(self, arg: int) -> Tuple[bool, RewardBreakdown]:
        br = RewardBreakdown()
        fx, _, fz = FACING_DELTA[self.agent.facing]
        if arg == 0:
            dx, dz = fx, fz
        elif arg == 1:
            dx, dz = -fx, -fz
        elif arg == 2:
            dx, dz = self._rotate(-1, 0, self.agent.facing)
        else:
            dx, dz = self._rotate(1, 0, self.agent.facing)

        nx, nz = self.agent.x + dx, self.agent.z + dz
        if not (0 <= nx < WORLD_W and 0 <= nz < WORLD_D) or \
                self._block_at((nx, self.agent.y, nz)) != AIR:
            self._merge(br, self.engine.on_world(
                "spatial.wall_bump", reason="путь заблокирован"))
            return False, br

        self.agent.x, self.agent.z = nx, nz
        if self.agent.table_open and not self._table_reachable():
            self.agent.table_open = False   # отошёл — верстак закрылся
        d = self._dist_to_table()
        if self._prev_dist is not None:
            if d < self._prev_dist:
                self._merge(br, self.engine.on_approach("crafting_table", d))
            elif d > self._prev_dist:
                self._merge(br, self.engine.on_world(
                    "spatial.retreat", reason="удалился от верстака"))
        self._prev_dist = d
        return True, br

    def _do_use_table(self) -> Tuple[bool, RewardBreakdown]:
        """
        Открыть/закрыть верстак — именно это переключает 2x2 <-> 3x3.

        Требует, чтобы верстак реально стоял рядом: нельзя "открыть" его,
        держа в инвентаре, ровно как в игре.
        """
        br = RewardBreakdown()
        if self.agent.table_open:
            self.agent.table_open = False
            # при закрытии содержимое сетки возвращается в инвентарь
            for i, v in enumerate(self.agent.grid):
                if v != EMPTY:
                    self._give(v, 1)
            self.agent.grid = [EMPTY] * GRID_SIZE
            return True, br
        if not self._table_reachable():
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason="верстака рядом нет"))
            return False, br
        self.agent.table_open = True
        self._merge(br, self.engine.on_world(
            "world.use_table", ctx="crafting_table",
            reason="открыл верстак -> доступна сетка 3x3"))
        return True, br

    def _do_place_block(self) -> Tuple[bool, RewardBreakdown]:
        br = RewardBreakdown()
        item = self.agent.held
        name = ID_ITEM.get(item, "")
        if item == EMPTY or name not in ITEM_TO_BLOCK or \
                self.agent.inventory.get(item, 0) <= 0:
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason="этот предмет не ставится"))
            return False, br
        pos = self._front()
        if self._block_at(pos) != AIR:
            self._merge(br, self.engine.on_world(
                "spatial.wall_bump", reason="место занято"))
            return False, br
        x, y, z = pos
        self._take(item, 1)
        self.world[x, y, z] = BLOCK_ID[ITEM_TO_BLOCK[name]]
        if name == "crafting_table":
            self.table_pos = (x, y, z)
            self._prev_dist = 0
            self._prev_dist = self._dist_to_table()
            self._merge(br, self.engine.on_world(
                "spatial.place_block_correct", ctx="crafting_table",
                reason="поставил верстак прямо перед собой"))
        return True, br

    def _do_break_block(self) -> Tuple[bool, RewardBreakdown]:
        br = RewardBreakdown()
        pos = self._front()
        b = self._block_at(pos)
        bname = ID_BLOCK.get(b, "air")
        if b == AIR or bname == "bedrock":
            self._merge(br, self.engine.on_world(
                "behavior.invalid_action", reason="ломать нечего"))
            return False, br
        held_name = ID_ITEM.get(self.agent.held, "empty")
        # ПРАВИЛА MINECRAFT: неподходящий инструмент ломает блок, но дроп
        # НЕ выпадает. Камень нужна деревянная кирка, железо - каменная,
        # алмаз - железная. Рукой руду не добыть в принципе.
        harvestable = can_harvest(bname, held_name)
        drop = BLOCK_DROP.get(bname) if harvestable else None
        x, y, z = pos
        self.world[x, y, z] = AIR

        if not harvestable:
            need = BLOCK_REQUIRED_LEVEL.get(bname, 0)
            tiers = {1: "деревянная", 2: "каменная", 3: "железная", 4: "алмазная"}
            self._merge(br, self.engine.on_world(
                "world.wrong_tool",
                reason=f"{bname} сломан впустую: нужна {tiers.get(need, '?')} "
                       f"кирка, в руке {held_name}"))
            return True, br

        if drop:
            self._give(item_id(drop), 1)
            if drop in ("dirt", "sand"):
                self._merge(br, self.engine.on_world(
                    "world.gather_useless", reason=f"добыл {drop}"))
            else:
                self._merge(br, self.engine.on_world(
                    "world.gather", ctx=drop, reason=f"добыл {drop}"))
        return True, br

    def _sync_goal_recipes(self) -> None:
        """
        Пересчитать, какие рецепты сейчас ещё имеет смысл награждать.

        Правило простое: рецепт полезен, если он ведёт к цели И нужного
        предмета в инвентаре пока не хватает. Как только досок хватает на
        палки — за доски больше не платим, иначе агент залипает на самом
        лёгком промежуточном рецепте вместо настоящей цели.
        """
        goal_name = GOALS[min(self.goal_index, len(GOALS) - 1)]
        # Цели-ресурсы (бревно, камень, руда, алмаз) не крафтятся — их
        # добывают. Плотных крафт-наград для них нет, работает world.gather.
        if goal_name not in RECIPE_BY_NAME:
            self._current_goal_recipes = set()
            self.engine.set_goal_recipes(set())
            return
        useful = set()
        for name in prerequisites(goal_name):
            if name == goal_name:
                useful.add(name)
                continue
            r = RECIPE_BY_NAME.get(name)
            if r is None:
                continue
            # сколько штук этого предмета требует цель
            goal_recipe = RECIPE_BY_NAME.get(goal_name)
            need = 1
            if goal_recipe is not None:
                need = max(1, sum(1 for c in goal_recipe.shape if c == r.result))
            if self.agent.inventory.get(r.result, 0) < need:
                useful.add(name)
        self._current_goal_recipes = useful
        self.engine.set_goal_recipes(useful)

    # ---------------- маска допустимых действий ----------------
    def _table_xz(self):
        """Координаты ближайшего верстака (или позиция агента, если его нет)."""
        best = None
        bd = 1e9
        W, H, D = self.world.shape
        tid = BLOCK_ID.get("crafting_table", -1)
        for x in range(W):
            for z in range(D):
                for y in range(H):
                    if int(self.world[x, y, z]) == tid:
                        d = abs(x - self.agent.x) + abs(z - self.agent.z)
                        if d < bd:
                            bd, best = d, (x, z)
        return best if best else (self.agent.x, self.agent.z)

    def action_mask(self) -> np.ndarray:
        """
        Булев вектор допустимых действий.

        ГЛАВНЫЙ ПРИНЦИП (важно для системы наград):
        маскируются ТОЛЬКО физически невозможные ходы и чистые no-op'ы —
        то, что в настоящем Minecraft просто не даст никакого эффекта.
        Смысловые ошибки НЕ маскируются никогда: положить предмет не в ту
        клетку, положить не тот предмет, разобрать уже верную форму, нажать
        крафт на неполном рецепте — всё это остаётся доступным, потому что
        именно на этих штрафах агент и должен учиться.

        Уровни (`mask_level`):
          "off"      — ничего не маскируем, чистый эксперимент;
          "physical" — только невозможное (нет предмета, слот занят, стена);
          "smart"    — physical + бессмысленные no-op'ы (по умолчанию).
        """
        m = np.ones(N_ACTIONS, dtype=bool)
        if self.mask_level == "off":
            return m

        smart = self.mask_level == "smart"
        inv = self.agent.inventory
        grid = self.agent.grid
        held = self.agent.held
        held_ok = held != EMPTY and inv.get(held, 0) > 0
        grid_empty = all(v == EMPTY for v in grid)
        # Слоты вне активной сетки физически недоступны: без верстака у
        # игрока в инвентаре только 2x2.
        active = set(self.active_slots())

        for act in ACTIONS:
            i = act.index
            t = act.type

            if t == ActionType.SELECT_ITEM:
                # физически: предмета нет в инвентаре
                ok = inv.get(act.arg, 0) > 0
                # no-op: он уже и так в руке
                if smart and act.arg == held:
                    ok = False
                m[i] = ok

            elif t == ActionType.PLACE_IN_SLOT:
                # Не туда положить — МОЖНО (за это штраф). Нельзя только
                # положить пустоту, в занятый слот или в недоступную клетку
                # сетки 3x3, когда верстак не открыт.
                m[i] = (held_ok and grid[act.arg] == EMPTY
                        and act.arg in active)

            elif t == ActionType.TAKE_FROM_SLOT:
                # Разобрать верную форму — МОЖНО (за это штраф break_shape).
                m[i] = grid[act.arg] != EMPTY and act.arg in active

            elif t == ActionType.CLEAR_GRID:
                m[i] = not grid_empty

            elif t == ActionType.CRAFT:
                # Крафт неполного рецепта — МОЖНО (штраф craft_fail).
                # Но по пустой сетке кнопка крафта в Minecraft вообще
                # ничего не делает — это no-op, а не ошибка.
                m[i] = (not grid_empty) if smart else True

            elif t == ActionType.TURN:
                if smart and act.arg == 2:      # look_up на максимуме
                    m[i] = self.agent.pitch < 1
                elif smart and act.arg == 3:    # look_down на минимуме
                    m[i] = self.agent.pitch > -1
                else:
                    m[i] = True

            elif t == ActionType.MOVE:
                if smart:
                    # Шаг в стену в Minecraft не двигает игрока — no-op.
                    # (В мосте к реальному серверу коллизии не так надёжны,
                    #  поэтому там правило spatial.wall_bump остаётся живым.)
                    m[i] = self._can_move(act.arg)
                else:
                    m[i] = True

            elif t == ActionType.PLACE_BLOCK:
                name = ID_ITEM.get(held, "")
                m[i] = (held_ok and name in ITEM_TO_BLOCK
                        and self._block_at(self._front()) == AIR)

            elif t == ActionType.BREAK_BLOCK:
                b = self._block_at(self._front())
                bn = ID_BLOCK.get(b, "")
                ok = b != AIR and bn != "bedrock"
                if ok and smart:
                    # Ломать руду без нужной кирки бессмысленно: блок
                    # исчезнет, дропа не будет. В режиме physical оставляем
                    # такую возможность, чтобы агент мог на этом учиться.
                    ok = can_harvest(bn, ID_ITEM.get(held, "empty"))
                m[i] = ok

            elif t == ActionType.USE_TABLE:
                # Открыть можно только если верстак рядом; закрыть — всегда.
                m[i] = self.agent.table_open or self._table_reachable()

            elif t == ActionType.ATTACK:
                # физически: бить можно всегда, но в smart — только если
                # перед агентом реально кто-то есть.
                m[i] = (not smart) or (self._mob_in_front() is not None)

            elif t == ActionType.EAT:
                # smart: есть, только если еда есть и агент не сыт
                if smart:
                    m[i] = self._has_food() and self.agent.food < 20.0
                else:
                    m[i] = True

            elif t == ActionType.NOOP:
                m[i] = not smart  # простой полезен только в режиме physical

        if not m.any():                       # страховка от полного тупика
            m[ACTIONS[-1].index] = True
        return m

    def _can_move(self, arg: int) -> bool:
        """Можно ли реально сделать шаг в эту сторону (без изменения мира)."""
        fx, _, fz = FACING_DELTA[self.agent.facing]
        if arg == 0:
            dx, dz = fx, fz
        elif arg == 1:
            dx, dz = -fx, -fz
        elif arg == 2:
            dx, dz = self._rotate(-1, 0, self.agent.facing)
        else:
            dx, dz = self._rotate(1, 0, self.agent.facing)
        nx, nz = self.agent.x + dx, self.agent.z + dz
        if not (0 <= nx < WORLD_W and 0 <= nz < WORLD_D):
            return False
        return self._block_at((nx, self.agent.y, nz)) == AIR

    def mask_stats(self) -> Dict[str, int]:
        """Сколько действий сейчас разрешено — для отладки и дашборда."""
        m = self.action_mask()
        return {"allowed": int(m.sum()), "total": int(m.size)}

    # ---------------- служебное ----------------
    def spawn_table_adjacent(self) -> None:
        """
        Поставить верстак вплотную к агенту.

        Нужно для скриптовых демо и тестов, где интересен именно крафт, а не
        путь до верстака: мир остаётся источником правды для near_table,
        поэтому вручную выставлять флаг бесполезно — он перезапишется.
        """
        old = getattr(self, "table_pos", None)
        if old is not None:
            ox, oy, oz = old
            if self._block_at(old) == BLOCK_ID["crafting_table"]:
                self.world[ox, oy, oz] = AIR
        x, y, z = self.agent.x, self.agent.y, self.agent.z
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, nz = x + dx, z + dz
            if 0 <= nx < WORLD_W and 0 <= nz < WORLD_D:
                self.world[nx, y, nz] = BLOCK_ID["crafting_table"]
                self.table_pos = (nx, y, nz)
                break
        self.agent.near_table = self._is_near_table()
        self._prev_dist = self._dist_to_table()

    def render_text(self) -> str:
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                v = self.agent.grid[r * 3 + c]
                cells.append("·" if v == EMPTY else ID_ITEM[v][:6].center(6))
            rows.append(" | ".join(x.center(6) for x in cells))
        inv = ", ".join(f"{ID_ITEM[k]}x{v}" for k, v in
                        sorted(self.agent.inventory.items()) if v)
        mode = "3x3 (верстак открыт)" if self.has_3x3() else "2x2 (инвентарь)"
        return (f"[{mode}] pos=({self.agent.x},{self.agent.y},{self.agent.z}) "
                f"facing={Facing(self.agent.facing).name} pitch={self.agent.pitch} "
                f"held={ID_ITEM.get(self.agent.held, '-')}\n"
                + "\n".join(rows) + f"\ninv: {inv}")

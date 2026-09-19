"""
Словарь предметов/блоков, пространство действий и утилиты кодирования.

Всё, что видит модель, проходит через этот файл. Он общий и для локального
симулятора, и для моста в реальный Minecraft (mineflayer), чтобы веса модели
были совместимы между средами.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# ПРЕДМЕТЫ
# --------------------------------------------------------------------------
# id 0 всегда зарезервирован под "пусто". Имена совпадают с minecraft-data,
# чтобы мост в mineflayer мапился один-в-один без таблиц перевода.
ITEMS: List[str] = [
    "empty",
    # дерево
    "oak_log",
    "oak_planks",
    "stick",
    # камень / руды
    "cobblestone",
    "raw_iron",
    "iron_ingot",
    "diamond",
    "coal",
    # верстак и печь
    "crafting_table",
    "furnace",
    # инструменты: деревянные
    "wooden_sword",
    "wooden_pickaxe",
    "wooden_axe",
    "wooden_shovel",
    # каменные
    "stone_sword",
    "stone_pickaxe",
    "stone_axe",
    "stone_shovel",
    # железные
    "iron_sword",
    "iron_pickaxe",
    "iron_axe",
    "iron_shovel",
    # алмазные
    "diamond_sword",
    "diamond_pickaxe",
    # мусор / отвлекающие предметы (нужны, чтобы было за что штрафовать)
    "dirt",
    "sand",
    "rotten_flesh",
    # --- расширение под полную базу крафтов ---
    # золото и прочее сырьё
    "raw_gold",
    "gold_ingot",
    "redstone",
    "flint",
    "leather",
    "string",
    "feather",
    "gunpowder",
    "charcoal",
    # еда (нужна: голод убивает не хуже мобов)
    "bread",
    "wheat",
    "apple",
    "cooked_beef",
    "beef",
    "porkchop",
    "cooked_porkchop",
    # золотые и кожаные инструменты/броня
    "golden_sword",
    "golden_pickaxe",
    "golden_axe",
    "golden_shovel",
    "diamond_axe",
    "diamond_shovel",
    # броня — главное средство не умирать
    "leather_helmet",
    "leather_chestplate",
    "leather_leggings",
    "leather_boots",
    "iron_helmet",
    "iron_chestplate",
    "iron_leggings",
    "iron_boots",
    "diamond_helmet",
    "diamond_chestplate",
    "diamond_leggings",
    "diamond_boots",
    # оружие дальнего боя и утиль
    "bow",
    "arrow",
    "shield",
    "torch",
    "chest",
    "ladder",
    "stone_slab",
    "oak_slab",
    "oak_stairs",
    "cobblestone_stairs",
    "oak_door",
    "bucket",
    "shears",
    "flint_and_steel",
    "fishing_rod",
    "iron_nugget",
    "stick_bundle",
]

ITEM_ID: Dict[str, int] = {name: i for i, name in enumerate(ITEMS)}
ID_ITEM: Dict[int, str] = {i: name for name, i in ITEM_ID.items()}
N_ITEMS: int = len(ITEMS)
EMPTY: int = 0


def item_id(name: str) -> int:
    """Имя -> id. Неизвестные предметы считаем 'мусором' (dirt), а не падаем."""
    return ITEM_ID.get(name, ITEM_ID["dirt"])


# Материальные "тиры" — используются деревом наград, чтобы каменный меч
# награждался отдельно от деревянного, но по той же схеме затухания.
TIERS: List[str] = ["wooden", "stone", "iron", "diamond"]
TIER_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TIERS)}
TIER_MATERIAL: Dict[str, str] = {
    "wooden": "oak_planks",
    "stone": "cobblestone",
    "iron": "iron_ingot",
    "diamond": "diamond",
}
TOOL_KINDS: List[str] = ["sword", "pickaxe", "axe", "shovel"]


# --------------------------------------------------------------------------
# БЛОКИ МИРА (воксели)
# --------------------------------------------------------------------------
BLOCKS: List[str] = [
    "air",
    "stone",
    "dirt",
    "grass_block",
    "oak_log",
    "oak_planks",
    "crafting_table",
    "furnace",
    "iron_ore",
    "diamond_ore",
    "bedrock",
    "water",
    # расширение под полную базу крафтов и выживание
    "gold_ore",
    "redstone_ore",
    "coal_ore",
    "gravel",
    "sand_block",
    "oak_leaves",
    "torch_block",
    "chest_block",
    "lava",
]
BLOCK_ID: Dict[str, int] = {name: i for i, name in enumerate(BLOCKS)}
ID_BLOCK: Dict[int, str] = {i: n for n, i in BLOCK_ID.items()}
N_BLOCKS: int = len(BLOCKS)
AIR: int = 0

# Блоки, сквозь которые видно и можно пройти.
PASSABLE = {"air", "water", "torch_block", "oak_leaves"}
# Блоки, которые наносят урон при контакте.
DAMAGING = {"lava": 4.0}


# --------------------------------------------------------------------------
# СУЩНОСТИ (мобы) — агент должен их ВИДЕТЬ, иначе будет умирать вслепую
# --------------------------------------------------------------------------
ENTITIES: List[str] = [
    "none",
    "zombie",
    "skeleton",
    "spider",
    "creeper",
    "cow",
    "pig",
    "sheep",
    "chicken",
]
ENTITY_ID: Dict[str, int] = {n: i for i, n in enumerate(ENTITIES)}
ID_ENTITY: Dict[int, str] = {i: n for n, i in ENTITY_ID.items()}
N_ENTITIES: int = len(ENTITIES)

# Кто нападает, сколько бьёт и сколько у него здоровья.
# Здоровье занижено против ванили сознательно: в симуляторе один "шаг" —
# это целое действие, а не тик. С ванильными 20 HP зомби переживает 5 ударов
# мечом, за которые успевает убить агента, и обучаться бою не на чем.
HOSTILE: Dict[str, Dict[str, float]] = {
    "zombie":   {"damage": 2.0, "health": 10.0, "range": 1},
    "skeleton": {"damage": 2.0, "health": 8.0, "range": 3},
    "spider":   {"damage": 1.5, "health": 8.0, "range": 1},
    "creeper":  {"damage": 9.0, "health": 10.0, "range": 1},
}
# Мирные мобы — источник еды и ресурсов.
PASSIVE_DROP: Dict[str, str] = {
    "cow": "beef",
    "pig": "porkchop",
    "sheep": "string",
    "chicken": "feather",
}
# Урон оружием в руке (голая рука бьёт слабо).
WEAPON_DAMAGE: Dict[str, float] = {
    "empty": 1.0,
    "wooden_sword": 4.0, "stone_sword": 5.0, "iron_sword": 6.0,
    "golden_sword": 4.0, "diamond_sword": 7.0,
    "wooden_axe": 3.0, "stone_axe": 4.0, "iron_axe": 5.0, "diamond_axe": 6.0,
    "wooden_pickaxe": 2.0, "stone_pickaxe": 3.0, "iron_pickaxe": 4.0,
    "diamond_pickaxe": 5.0,
}
# Сколько брони даёт предмет (уменьшает входящий урон).
ARMOR_POINTS: Dict[str, float] = {
    "leather_helmet": 1, "leather_chestplate": 3,
    "leather_leggings": 2, "leather_boots": 1,
    "iron_helmet": 2, "iron_chestplate": 6,
    "iron_leggings": 5, "iron_boots": 2,
    "diamond_helmet": 3, "diamond_chestplate": 8,
    "diamond_leggings": 6, "diamond_boots": 3,
}
# Сколько голода восстанавливает еда.
FOOD_VALUE: Dict[str, float] = {
    "apple": 4.0, "bread": 5.0, "wheat": 1.0,
    "beef": 3.0, "cooked_beef": 8.0,
    "porkchop": 3.0, "cooked_porkchop": 8.0,
    "rotten_flesh": 4.0,          # но с риском — за него штраф
}


def weapon_damage(item_name: str) -> float:
    """Урон предметом в руке."""
    return WEAPON_DAMAGE.get(item_name, 1.0)

# --------------------------------------------------------------------------
# УРОВНИ ИНСТРУМЕНТОВ (как в Minecraft)
# --------------------------------------------------------------------------
# 0 = рука, 1 = дерево, 2 = камень, 3 = железо, 4 = алмаз
TOOL_LEVEL: Dict[str, int] = {
    "empty": 0,
    "wooden_pickaxe": 1, "wooden_axe": 1, "wooden_shovel": 1, "wooden_sword": 1,
    "stone_pickaxe": 2, "stone_axe": 2, "stone_shovel": 2, "stone_sword": 2,
    "iron_pickaxe": 3, "iron_axe": 3, "iron_shovel": 3, "iron_sword": 3,
    "diamond_pickaxe": 4, "diamond_sword": 4,
}

# Какой МИНИМАЛЬНЫЙ уровень КИРКИ нужен, чтобы блок вообще что-то уронил.
# Ровно правила ванильного Minecraft:
#   камень      -> деревянная кирка (1)
#   железо      -> каменная кирка  (2)
#   алмаз/золото-> железная кирка  (3)
# Дерево и земля ломаются рукой, но топор/лопата ускоряют (скорость не
# моделируем, только сам факт дропа).
BLOCK_REQUIRED_LEVEL: Dict[str, int] = {
    "stone": 1,
    "iron_ore": 2,
    "diamond_ore": 3,
}
# Блоки, для которых нужна именно КИРКА, а не любой инструмент того же уровня.
NEEDS_PICKAXE = {"stone", "iron_ore", "diamond_ore", "furnace"}

PICKAXES = {"wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"}


def tool_level(item_name: str) -> int:
    """Уровень инструмента в руке (0 = голая рука)."""
    return TOOL_LEVEL.get(item_name, 0)


def can_harvest(block_name: str, held_item: str) -> bool:
    """
    Выпадет ли из блока дроп при текущем инструменте.

    Неправильный инструмент = блок ломается, но НИЧЕГО не падает — именно так
    работает ваниль. Отдельно проверяем, что для руды нужна именно кирка:
    каменным мечом камень не добудешь.
    """
    need = BLOCK_REQUIRED_LEVEL.get(block_name, 0)
    if need == 0:
        return True
    if block_name in NEEDS_PICKAXE and held_item not in PICKAXES:
        return False
    return tool_level(held_item) >= need


# Что падает из блока при разрушении (для симулятора).
BLOCK_DROP: Dict[str, str] = {
    "oak_log": "oak_log",
    "stone": "cobblestone",
    "dirt": "dirt",
    "grass_block": "dirt",
    "iron_ore": "raw_iron",
    "diamond_ore": "diamond",
    "oak_planks": "oak_planks",
    "crafting_table": "crafting_table",
    "furnace": "furnace",
}
# Блок, который ставится из предмета (если предмет вообще ставится).
ITEM_TO_BLOCK: Dict[str, str] = {
    "oak_log": "oak_log",
    "oak_planks": "oak_planks",
    "cobblestone": "stone",
    "dirt": "dirt",
    "sand": "dirt",
    "crafting_table": "crafting_table",
    "furnace": "furnace",
}


# --------------------------------------------------------------------------
# ОРИЕНТАЦИЯ
# --------------------------------------------------------------------------
class Facing(IntEnum):
    """Стороны света. Смещения совпадают с координатной системой Minecraft."""

    SOUTH = 0  # +Z
    WEST = 1   # -X
    NORTH = 2  # -Z
    EAST = 3   # +X


FACING_DELTA: Dict[int, Tuple[int, int, int]] = {
    Facing.SOUTH: (0, 0, 1),
    Facing.WEST: (-1, 0, 0),
    Facing.NORTH: (0, 0, -1),
    Facing.EAST: (1, 0, 0),
}
# Наклон взгляда: -1 вниз, 0 прямо, +1 вверх
PITCHES: Tuple[int, int, int] = (-1, 0, 1)


# --------------------------------------------------------------------------
# ПРОСТРАНСТВО ДЕЙСТВИЙ
# --------------------------------------------------------------------------
GRID_SIZE = 9  # физически всегда 9 слотов, но доступна не вся сетка

# Инвентарная сетка 2x2 — это левый верхний угол сетки 3x3.
# Слоты 2,5,6,7,8 недоступны, пока агент не откроет верстак.
GRID_2X2_SLOTS = (0, 1, 3, 4)
GRID_3X3_SLOTS = tuple(range(9))


class ActionType(IntEnum):
    PLACE_IN_SLOT = 0   # положить предмет "в руке" в слот крафта
    TAKE_FROM_SLOT = 1  # забрать предмет из слота обратно в инвентарь
    SELECT_ITEM = 2     # взять предмет из инвентаря в руку
    CRAFT = 3           # нажать "скрафтить"
    CLEAR_GRID = 4      # очистить сетку
    TURN = 5            # повернуться / поднять-опустить взгляд
    MOVE = 6            # шаг в мире
    PLACE_BLOCK = 7     # поставить блок перед собой
    BREAK_BLOCK = 8     # сломать блок перед собой
    USE_TABLE = 9       # ОТКРЫТЬ/ЗАКРЫТЬ верстак -> переключает 2x2 <-> 3x3
    ATTACK = 10         # ударить моба перед собой
    EAT = 11            # съесть еду из инвентаря
    NOOP = 12           # ничего не делать


@dataclass(frozen=True)
class Action:
    """Одно дискретное действие агента."""

    index: int
    type: ActionType
    arg: int = 0
    name: str = ""

    def __repr__(self) -> str:  # pragma: no cover - только для логов
        return f"<{self.index}:{self.name}>"


def _build_actions() -> List[Action]:
    acts: List[Action] = []

    def add(t: ActionType, arg: int, name: str) -> None:
        acts.append(Action(len(acts), t, arg, name))

    for s in range(GRID_SIZE):
        add(ActionType.PLACE_IN_SLOT, s, f"place_slot_{s}")
    for s in range(GRID_SIZE):
        add(ActionType.TAKE_FROM_SLOT, s, f"take_slot_{s}")
    for i in range(1, N_ITEMS):  # 0 = empty, брать в руку нечего
        add(ActionType.SELECT_ITEM, i, f"hold_{ITEMS[i]}")
    add(ActionType.CRAFT, 0, "craft")
    add(ActionType.CLEAR_GRID, 0, "clear_grid")
    for i, nm in enumerate(["turn_left", "turn_right", "look_up", "look_down"]):
        add(ActionType.TURN, i, nm)
    for i, nm in enumerate(["move_forward", "move_back", "strafe_left", "strafe_right"]):
        add(ActionType.MOVE, i, nm)
    add(ActionType.PLACE_BLOCK, 0, "place_block_front")
    add(ActionType.BREAK_BLOCK, 0, "break_block_front")
    add(ActionType.USE_TABLE, 0, "open_close_table")
    add(ActionType.ATTACK, 0, "attack_front")
    add(ActionType.EAT, 0, "eat_food")
    add(ActionType.NOOP, 0, "noop")
    return acts


ACTIONS: List[Action] = _build_actions()
N_ACTIONS: int = len(ACTIONS)
ACTION_BY_NAME: Dict[str, Action] = {a.name: a for a in ACTIONS}


# --------------------------------------------------------------------------
# ФОРМА НАБЛЮДЕНИЯ
# --------------------------------------------------------------------------
# --- ЗРЕНИЕ ---------------------------------------------------------------
# 3D: куб вокруг агента. Было 3x3x3 (агент видел буквально вплотную и
# умирал от мобов, которых физически не мог заметить). Стало 5x5x5.
VIEW = 5
N_VOXELS = VIEW ** 3

# 2D: карта сверху дальнего обзора — как миникарта. Даёт агенту понять,
# КУДА идти (где лес, где руда), а не только что у него под носом.
MAP_R = 6                      # радиус в блоках
MAP_W = MAP_R * 2 + 1          # 13x13
N_MAP = MAP_W * MAP_W

# 2D: сущности вокруг — отдельный слой той же карты. Без него агент не
# видит мобов и не может ни убежать, ни ударить.
N_ENT_MAP = N_MAP
N_GOALS = 24        # сколько целей поддерживает curriculum-вектор
# Память движка наград, видимая агенту (см. ниже).
N_PAID = GRID_SIZE + 2
# Состояние крафта: [верстак открыт, доступна ли 3x3, верстак рядом]
N_CRAFT_STATE = 3
# Плотная часть наблюдения: инвентарь + ориентация + прогресс + цель + "уже оплачено"
#
# ПОЧЕМУ ЕСТЬ N_PAID:
# движок наград помнит, за какие клетки он уже заплатил в этом эпизоде
# (защита от фарма). Если не показать эту память агенту, то одно и то же
# наблюдение даёт то +11, то -0.02 — среда перестаёт быть марковской, и
# сеть физически не может выучить "положил один раз - дальше не надо".
# Именно из-за этого агент залипал в цикле place -> clear -> place.
# Состояние выживания: [здоровье, голод, уровень брони, есть ли враг рядом,
# дистанция до ближайшего врага, светло ли]. Без этого агент не может
# научиться убегать, есть и надевать броню.
N_SURVIVAL = 6

# ----------------------------------------------------------------------
# КООРДИНАТЫ. Раньше агент знал только направление взгляда и расстояние до
# верстака — где он находится в мире, он не знал вообще. Без этого нельзя
# ни вернуться на место, ни понять "я в углу", ни построить маршрут.
#
# Даём и абсолютные, и относительные: абсолютные отвечают на вопрос "где я",
# относительные — "куда идти". Человек с F3 видит ровно это же.
#   0-2  x/y/z, нормированные размером мира
#   3-4  sin/cos угла на верстак — направление без разрыва на 360 градусов
#   5    расстояние до верстака
#   6-7  sin/cos угла на точку спавна
#   8    расстояние до спавна
#   9    доля пройденных шагов эпизода (сколько времени осталось)
#   10-13 близость к четырём стенам мира — чтобы не утыкался в край
# ----------------------------------------------------------------------
N_COORDS = 14

DENSE_DIM = (N_ITEMS + 4 + 6 + N_GOALS + N_PAID + N_CRAFT_STATE + N_SURVIVAL
             + N_COORDS)

# ----------------------------------------------------------------------
# ИМЕНОВАННЫЕ СРЕЗЫ dense-вектора.
#
# Раньше тесты и мониторы резали хвост руками: dense[-6:] для выживания,
# dense[-9:-6] для состояния крафта. Любое новое поле молча ломало ВСЕ
# такие места — именно так мост однажды и протух. Теперь границы считаются
# из размеров, и добавление поля правит их автоматически.
# ----------------------------------------------------------------------
_o = 0
SL_INV = slice(_o, _o + N_ITEMS);           _o += N_ITEMS
SL_HELD = slice(_o, _o + 4);                _o += 4
SL_ORIENT = slice(_o, _o + 6);              _o += 6
SL_GOAL = slice(_o, _o + N_GOALS);          _o += N_GOALS
SL_PAID = slice(_o, _o + N_PAID);           _o += N_PAID
SL_CRAFT_STATE = slice(_o, _o + N_CRAFT_STATE); _o += N_CRAFT_STATE
SL_SURVIVAL = slice(_o, _o + N_SURVIVAL);   _o += N_SURVIVAL
SL_COORDS = slice(_o, _o + N_COORDS);       _o += N_COORDS
assert _o == DENSE_DIM, (_o, DENSE_DIM)
del _o


@dataclass(frozen=True)
class ObsSpec:
    """Описание наблюдения — модель строится ровно по нему."""

    n_items: int = N_ITEMS
    n_blocks: int = N_BLOCKS
    n_entities: int = N_ENTITIES
    grid_size: int = GRID_SIZE
    n_voxels: int = N_VOXELS
    n_map: int = N_MAP
    n_ent_map: int = N_ENT_MAP
    dense_dim: int = DENSE_DIM
    n_actions: int = N_ACTIONS


OBS_SPEC = ObsSpec()

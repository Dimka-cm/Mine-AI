"""
Рецепты крафта с учётом ПРОСТРАНСТВЕННОЙ ориентации.

Ключевая идея для наград: рецепт — это не просто "мешок предметов", а
конкретная форма на сетке 3x3. Поэтому мы умеем считать не только "совпало /
не совпало", но и ЧАСТИЧНОЕ совпадение — сколько клеток формы уже выложено
правильно. Именно это даёт промежуточные награды:

    палка в нужном слоте          -> +
    доска НАД палкой (для меча)   -> ++
    форма собрана целиком         -> +++
    предмет скрафчен              -> +++++

и штраф, если предмет положен в клетку, которой в рецепте быть не должно.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .spaces import (EMPTY, ITEM_ID, TIER_MATERIAL, TIERS, TOOL_KINDS,
                     item_id)

# Символьные шаблоны инструментов. '#' — материал тира, '/' — палка, '.' — пусто.
TOOL_PATTERNS: Dict[str, List[str]] = {
    "sword":   ["·#·", "·#·", "·/·"],
    "pickaxe": ["###", "·/·", "·/·"],
    "axe":     ["##·", "#/·", "·/·"],
    "shovel":  ["·#·", "·/·", "·/·"],
}

# Броня — те же формы, что в ванили.
ARMOR_PATTERNS: Dict[str, List[str]] = {
    "helmet":     ["###", "#·#", "···"],
    "chestplate": ["#·#", "###", "###"],
    "leggings":   ["###", "#·#", "#·#"],
    "boots":      ["···", "#·#", "#·#"],
}


def _shape_from(rows: Sequence[str], mapping: Dict[str, str]) -> Tuple[int, ...]:
    """
    Читаемая запись рецепта: ["p i p", "p p p", ". p ."] + {"p": "oak_planks"}.

    Точка и пробел — пустая клетка. Так рецепты видно глазом, и опечатку
    поймать проще, чем в плоском списке из девяти id.
    """
    cells: List[int] = []
    for row in rows:
        for ch in row.replace(" ", ""):
            if ch in (".", "·"):
                cells.append(EMPTY)
            else:
                cells.append(item_id(mapping[ch]))
    assert len(cells) == 9, f"плохая форма: {rows} -> {len(cells)} клеток"
    return tuple(cells)


@dataclass(frozen=True)
class Recipe:
    """Рецепт с жёсткой формой на сетке 3x3."""

    name: str              # что получаем, напр. "stone_sword"
    result: int            # item id результата
    count: int             # сколько штук выдаёт
    shape: Tuple[int, ...]  # 9 item id, EMPTY = клетка обязана быть пустой
    shapeless: bool = False  # если True — важен только набор предметов
    tier: Optional[str] = None
    kind: Optional[str] = None
    needs_table: bool = True  # нужен ли верстак (3x3) или хватит 2x2
    needs_furnace: bool = False  # переплавка: нужна печь, а не верстак

    # Клетки, которые обязаны быть заполнены (кэш для скорости).
    filled_cells: Tuple[int, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "filled_cells",
            tuple(i for i, v in enumerate(self.shape) if v != EMPTY),
        )

    @property
    def n_filled(self) -> int:
        return len(self.filled_cells)


def _pattern_to_shape(pattern: Sequence[str], material: str) -> Tuple[int, ...]:
    """'·#·' -> [EMPTY, material_id, EMPTY]"""
    mapping = {"#": item_id(material), "/": item_id("stick"), "·": EMPTY, ".": EMPTY}
    cells: List[int] = []
    for row in pattern:
        for ch in row:
            cells.append(mapping[ch])
    assert len(cells) == 9, f"плохой шаблон: {pattern}"
    return tuple(cells)


def _build_recipes() -> List[Recipe]:
    rs: List[Recipe] = []

    # --- базовая цепочка ---
    rs.append(Recipe(
        name="oak_planks", result=item_id("oak_planks"), count=4,
        shape=(item_id("oak_log"),) + (EMPTY,) * 8,
        shapeless=True, needs_table=False,
    ))
    rs.append(Recipe(
        name="stick", result=item_id("stick"), count=4,
        shape=(
            EMPTY, item_id("oak_planks"), EMPTY,
            EMPTY, item_id("oak_planks"), EMPTY,
            EMPTY, EMPTY, EMPTY,
        ),
        needs_table=False,
    ))
    rs.append(Recipe(
        name="crafting_table", result=item_id("crafting_table"), count=1,
        shape=(
            item_id("oak_planks"), item_id("oak_planks"), EMPTY,
            item_id("oak_planks"), item_id("oak_planks"), EMPTY,
            EMPTY, EMPTY, EMPTY,
        ),
        needs_table=False,
    ))
    rs.append(Recipe(
        name="furnace", result=item_id("furnace"), count=1,
        shape=(
            item_id("cobblestone"), item_id("cobblestone"), item_id("cobblestone"),
            item_id("cobblestone"), EMPTY, item_id("cobblestone"),
            item_id("cobblestone"), item_id("cobblestone"), item_id("cobblestone"),
        ),
    ))

    # --- инструменты всех тиров (теперь и алмазные топор/лопата, и золото) ---
    for tier in TIERS:
        material = TIER_MATERIAL[tier]
        for kind in TOOL_KINDS:
            name = f"{tier}_{kind}"
            rs.append(Recipe(
                name=name, result=item_id(name), count=1,
                shape=_pattern_to_shape(TOOL_PATTERNS[kind], material),
                tier=tier, kind=kind,
            ))

    # --- БРОНЯ: главное средство не умирать от мобов ---
    for tier, material in (("leather", "leather"), ("iron", "iron_ingot"),
                           ("diamond", "diamond")):
        for kind, pattern in ARMOR_PATTERNS.items():
            name = f"{tier}_{kind}"
            if name not in ITEM_ID:
                continue
            rs.append(Recipe(
                name=name, result=item_id(name), count=1,
                shape=_pattern_to_shape(pattern, material),
                tier=tier, kind=kind,
            ))

    # --- бой и защита ---
    rs.append(Recipe(
        name="shield", result=item_id("shield"), count=1,
        shape=_shape_from(["p i p", "p p p", ". p ."],
                          {"p": "oak_planks", "i": "iron_ingot"}),
    ))
    rs.append(Recipe(
        name="bow", result=item_id("bow"), count=1,
        shape=_shape_from([". s t", "s . t", ". s t"],
                          {"s": "string", "t": "stick"}),
    ))
    rs.append(Recipe(
        name="arrow", result=item_id("arrow"), count=4,
        shape=_shape_from([". f .", ". t .", ". e ."],
                          {"f": "flint", "t": "stick", "e": "feather"}),
    ))

    # --- свет и выживание ночью ---
    rs.append(Recipe(
        name="torch", result=item_id("torch"), count=4,
        shape=_shape_from([". c .", ". t .", ". . ."],
                          {"c": "coal", "t": "stick"}),
        needs_table=False,
    ))
    rs.append(Recipe(
        name="chest", result=item_id("chest"), count=1,
        shape=_shape_from(["p p p", "p . p", "p p p"], {"p": "oak_planks"}),
    ))
    rs.append(Recipe(
        name="ladder", result=item_id("ladder"), count=3,
        shape=_shape_from(["t . t", "t t t", "t . t"], {"t": "stick"}),
    ))

    # --- еда: голод убивает не хуже мобов ---
    rs.append(Recipe(
        name="bread", result=item_id("bread"), count=1,
        shape=_shape_from(["w w w", ". . .", ". . ."], {"w": "wheat"}),
    ))

    # --- утварь ---
    rs.append(Recipe(
        name="bucket", result=item_id("bucket"), count=1,
        shape=_shape_from(["i . i", ". i .", ". . ."], {"i": "iron_ingot"}),
    ))
    rs.append(Recipe(
        name="shears", result=item_id("shears"), count=1,
        shape=_shape_from([". i .", "i . .", ". . ."], {"i": "iron_ingot"}),
    ))
    rs.append(Recipe(
        name="flint_and_steel", result=item_id("flint_and_steel"), count=1,
        shape=_shape_from(["i . .", ". f .", ". . ."],
                          {"i": "iron_ingot", "f": "flint"}),
        needs_table=False,
    ))
    rs.append(Recipe(
        name="fishing_rod", result=item_id("fishing_rod"), count=1,
        shape=_shape_from([". . t", ". t s", "t . s"],
                          {"t": "stick", "s": "string"}),
    ))

    # --- строительные блоки ---
    rs.append(Recipe(
        name="stone_slab", result=item_id("stone_slab"), count=6,
        shape=_shape_from(["c c c", ". . .", ". . ."], {"c": "cobblestone"}),
    ))
    rs.append(Recipe(
        name="oak_slab", result=item_id("oak_slab"), count=6,
        shape=_shape_from(["p p p", ". . .", ". . ."], {"p": "oak_planks"}),
    ))
    rs.append(Recipe(
        name="oak_stairs", result=item_id("oak_stairs"), count=4,
        shape=_shape_from(["p . .", "p p .", "p p p"], {"p": "oak_planks"}),
    ))
    rs.append(Recipe(
        name="cobblestone_stairs", result=item_id("cobblestone_stairs"), count=4,
        shape=_shape_from(["c . .", "c c .", "c c c"], {"c": "cobblestone"}),
    ))
    rs.append(Recipe(
        name="oak_door", result=item_id("oak_door"), count=3,
        shape=_shape_from(["p p .", "p p .", "p p ."], {"p": "oak_planks"}),
    ))

    # --- переплавка (печь) как «рецепты» 1-в-1 ---
    for src, dst in (("raw_iron", "iron_ingot"), ("raw_gold", "gold_ingot"),
                     ("beef", "cooked_beef"), ("porkchop", "cooked_porkchop"),
                     ("oak_log", "charcoal")):
        rs.append(Recipe(
            name=f"smelt_{dst}", result=item_id(dst), count=1,
            shape=(item_id(src),) + (EMPTY,) * 8,
            shapeless=True, needs_furnace=True, needs_table=False,
        ))

    # --- обратимые/мелкие рецепты ---
    rs.append(Recipe(
        name="iron_nugget", result=item_id("iron_nugget"), count=9,
        shape=(item_id("iron_ingot"),) + (EMPTY,) * 8,
        shapeless=True, needs_table=False,
    ))
    return rs


RECIPES: List[Recipe] = _build_recipes()
RECIPE_BY_NAME: Dict[str, Recipe] = {r.name: r for r in RECIPES}
RECIPE_NAMES: List[str] = [r.name for r in RECIPES]


# --------------------------------------------------------------------------
# СОПОСТАВЛЕНИЕ СЕТКИ С РЕЦЕПТАМИ
# --------------------------------------------------------------------------
@dataclass
class MatchInfo:
    """Насколько текущая сетка похожа на данный рецепт."""

    recipe: Recipe
    correct: int        # клеток рецепта выложено верно
    wrong: int          # клеток, которые рецепту противоречат
    complete: bool      # форма собрана полностью и лишнего нет
    progress: float     # correct / n_filled, 0..1


def match_grid(grid: Sequence[int], recipe: Recipe) -> MatchInfo:
    """Сравнивает сетку 3x3 с рецептом (учитывает сдвиги формы)."""
    if recipe.shapeless:
        need: Dict[int, int] = {}
        for v in recipe.shape:
            if v != EMPTY:
                need[v] = need.get(v, 0) + 1
        have: Dict[int, int] = {}
        for v in grid:
            if v != EMPTY:
                have[v] = have.get(v, 0) + 1
        correct = sum(min(c, have.get(k, 0)) for k, c in need.items())
        wrong = sum(c for k, c in have.items() if k not in need)
        wrong += sum(max(0, have.get(k, 0) - c) for k, c in need.items())
        total = sum(need.values())
        return MatchInfo(recipe, correct, wrong, have == need,
                         correct / max(1, total))

    best: Optional[MatchInfo] = None
    # Пробуем все сдвиги формы по сетке — так "меч слева" тоже засчитается.
    rows = [recipe.shape[i * 3:(i + 1) * 3] for i in range(3)]
    used_rows = [i for i, r in enumerate(rows) if any(v != EMPTY for v in r)]
    used_cols = [c for c in range(3) if any(rows[r][c] != EMPTY for r in range(3))]
    if not used_rows:
        used_rows, used_cols = [0], [0]
    h = max(used_rows) - min(used_rows) + 1
    w = max(used_cols) - min(used_cols) + 1
    r0, c0 = min(used_rows), min(used_cols)

    for dr in range(0, 3 - h + 1):
        for dc in range(0, 3 - w + 1):
            placed: Dict[int, int] = {}
            for rr in range(h):
                for cc in range(w):
                    v = rows[r0 + rr][c0 + cc]
                    if v != EMPTY:
                        placed[(dr + rr) * 3 + (dc + cc)] = v
            correct = sum(1 for idx, v in placed.items() if grid[idx] == v)
            wrong = 0
            for idx in range(9):
                g = grid[idx]
                if g == EMPTY:
                    continue
                if idx not in placed:
                    wrong += 1          # положил туда, где должно быть пусто
                elif placed[idx] != g:
                    wrong += 1          # положил не тот предмет
            complete = correct == len(placed) and wrong == 0
            info = MatchInfo(recipe, correct, wrong, complete,
                             correct / max(1, len(placed)))
            if best is None or (info.correct - info.wrong) > (best.correct - best.wrong):
                best = info
    assert best is not None
    return best


def find_completed(grid: Sequence[int], has_table: bool = True) -> Optional[Recipe]:
    """Возвращает рецепт, который полностью собран на сетке, иначе None."""
    for r in RECIPES:
        if r.needs_table and not has_table:
            continue
        if match_grid(grid, r).complete:
            return r
    return None


def best_partial(grid: Sequence[int], has_table: bool = True) -> Optional[MatchInfo]:
    """Рецепт, к которому сетка ближе всего (для промежуточных наград)."""
    if all(v == EMPTY for v in grid):
        return None
    best: Optional[MatchInfo] = None
    for r in RECIPES:
        if r.needs_table and not has_table:
            continue
        info = match_grid(grid, r)
        score = info.correct - 1.5 * info.wrong
        if best is None or score > (best.correct - 1.5 * best.wrong):
            best = info
    return best


# --------------------------------------------------------------------------
# ЦЕПОЧКА ПРЕДПОСЫЛОК
# --------------------------------------------------------------------------
def needs_table(goal: str) -> bool:
    """Требует ли цель открытого верстака (рецепт 3x3)."""
    r = RECIPE_BY_NAME.get(goal)
    return bool(r and r.needs_table)


def prerequisites(goal: str) -> set:
    """
    Все рецепты, которые реально нужны на пути к цели.

    Нужно для того, чтобы плотные награды за сборку формы выдавались ТОЛЬКО
    по делу. Иначе агент получает деньги за случайную выкладку постороннего
    рецепта (положил бревно -> "форма досок собрана") и перестаёт стремиться
    к настоящей цели.
    """
    need = {goal}
    frontier = [goal]
    # Если цель требует верстака, то верстак сам становится предпосылкой.
    if needs_table(goal):
        need.add("crafting_table")
        frontier.append("crafting_table")
    while frontier:
        cur = frontier.pop()
        r = RECIPE_BY_NAME.get(cur)
        if r is None:
            continue
        for cell in r.shape:
            if cell == EMPTY:
                continue
            name = None
            for cand, cr in RECIPE_BY_NAME.items():
                if cr.result == cell:
                    name = cand
                    break
            if name and name not in need:
                need.add(name)
                frontier.append(name)
    return need

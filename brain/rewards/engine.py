"""
Движок наград: превращает "что произошло в мире" в число.

Работает поверх RewardDB, поэтому:
  * повторение уже награждённого действия автоматически стоит дешевле;
  * у каждого тира (wooden/stone/iron/diamond) свой счётчик — прогресс
    по каменному мечу не обесценивает железный;
  * ошибки всегда стоят полную цену (штрафы не затухают).

Движок ничего не знает про нейросеть — его можно дёргать и из симулятора,
и из моста в реальный Minecraft.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..recipes import Recipe, best_partial, match_grid
from ..spaces import EMPTY, ID_ITEM, TIER_INDEX, TIERS
from .db import RewardDB


@dataclass
class RewardBreakdown:
    """Из чего сложилась награда за один шаг — для логов и дашборда."""

    total: float = 0.0
    parts: List[Tuple[str, float, str]] = field(default_factory=list)

    def add(self, key: str, value: float, reason: str = "") -> None:
        if value == 0.0:
            return
        self.total += value
        self.parts.append((key, value, reason))

    def as_dict(self) -> Dict[str, object]:
        return {
            "total": round(self.total, 3),
            "parts": [{"key": k, "value": round(v, 3), "reason": r}
                      for k, v, r in self.parts],
        }


class RewardEngine:
    """Считает награду за переход состояния."""

    def __init__(self, db: RewardDB) -> None:
        self.db = db
        self.episode = 0
        self.step = 0
        # Максимум прогресса по каждому рецепту в текущем эпизоде,
        # чтобы награждать только за НОВЫЙ прогресс, а не за топтание.
        self._best_progress: Dict[str, int] = {}
        self._tiers_reached: Set[str] = set()
        self._recent_actions: List[int] = []
        # Вехи, уже оплаченные в этом эпизоде. Без них агент фармит цикл
        # "собрал форму -> разобрал -> собрал": каждая пересборка снова
        # платила shape_complete, и крафтить становилось необязательно.
        self._shape_paid: Set[str] = set()
        # Клетки (рецепт, слот, предмет), за которые уже платили в эпизоде.
        # Повторная укладка того же в тот же слот не оплачивается — иначе
        # цикл "вынул-положил" остаётся прибыльным даже без shape_complete.
        self._cells_paid: Set[tuple] = set()
        self._best_dist: Dict[str, int] = {}
        # Рецепты, за которые сейчас имеет смысл платить плотную награду.
        # Пусто = платим за любые (режим совместимости).
        self._goal_recipes: Set[str] = set()
        self._crafted_this_episode: List[str] = []
        self._last_craft: Optional[str] = None
        self._same_craft_streak = 0

    # ---------------- жизненный цикл ----------------
    def set_goal_recipes(self, names) -> None:
        """
        Какие рецепты сейчас ведут к цели И ещё нужны.

        Среда пересчитывает этот набор каждый шаг: как только нужного предмета
        в инвентаре достаточно, его рецепт из набора выпадает. Без этого агент
        находит локальный оптимум "фармить самый простой промежуточный рецепт":
        имея 8 досок и цель "палки", он 37 эпизодов подряд крафтил доски,
        потому что это проще, а награда всё равно капала.
        """
        self._goal_recipes = set(names or ())

    def begin_episode(self, episode: int) -> None:
        self.episode = episode
        self.step = 0
        self._best_progress.clear()
        self._shape_paid.clear()
        self._cells_paid.clear()
        self._best_dist.clear()
        self._recent_actions.clear()
        self._crafted_this_episode.clear()
        self._last_craft = None
        self._same_craft_streak = 0
        self.db.start_episode(episode)

    def end_episode(self, total: float, positive: float, negative: float) -> None:
        self.db.finish_episode(self.episode, self.step, total, positive, negative,
                               list(self._crafted_this_episode))

    # ---------------- вспомогательное ----------------
    def _grant(self, br: RewardBreakdown, key: str, ctx: str = "",
               reason: str = "", scale: float = 1.0) -> float:
        g = self.db.grant(key, episode=self.episode, step=self.step,
                          ctx=ctx, reason=reason)
        val = g.value * scale
        if val:
            br.add(g.ctx_key, val, g.reason)
        return val

    @staticmethod
    def _neighbours(slot: int) -> List[int]:
        """Соседние клетки сетки 3x3 (сверху/снизу/слева/справа)."""
        r, c = divmod(slot, 3)
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 3 and 0 <= cc < 3:
                out.append(rr * 3 + cc)
        return out

    # ---------------- основные события ----------------
    def on_place_in_grid(self, grid_before: Sequence[int], grid_after: Sequence[int],
                         slot: int, item: int, has_table: bool = True) -> RewardBreakdown:
        """
        Агент положил предмет в слот сетки.

        Здесь живёт вся суть ТЗ: палка в нужный слот -> +, доска НАД палкой -> ++,
        предмет не туда -> минус.
        """
        br = RewardBreakdown()
        after = best_partial(grid_after, has_table)
        if after is None:
            br.add("grid.nothing", 0.0)
            return br

        recipe = after.recipe
        # Побочный рецепт, не ведущий к цели, плотной награды не приносит —
        # но ошибки по нему всё равно наказываются ниже.
        off_path = bool(self._goal_recipes) and recipe.name not in self._goal_recipes
        before = match_grid(grid_before, recipe)
        item_name = ID_ITEM.get(item, "?")
        ctx_tool = recipe.name

        # Верна ли конкретно эта клетка в лучшем совпадении рецепта?
        cell_ok = after.correct > before.correct
        cell_bad = after.wrong > before.wrong

        if cell_ok and off_path:
            return br          # полезного прогресса к цели нет — молчим
        if cell_ok:
            cell_key = (recipe.name, slot, item)
            repeat = cell_key in self._cells_paid
            self._cells_paid.add(cell_key)
            if not repeat:
                self._grant(br, "grid.place_correct", ctx=f"{ctx_tool}:{slot}",
                            reason=f"{item_name} -> слот {slot} рецепта {recipe.name}")
            # Пространственная связь: рядом уже стоит верная деталь.
            touching = any(
                grid_before[n] != EMPTY and grid_after[n] == grid_before[n]
                for n in self._neighbours(slot)
            )
            if touching and before.correct > 0 and not repeat:
                self._grant(br, "grid.place_adjacent", ctx=f"{ctx_tool}:{slot}",
                            reason=f"{item_name} примыкает к собранной части "
                                   f"{recipe.name}")
            # Новый максимум формы за эпизод.
            prev_best = self._best_progress.get(recipe.name, 0)
            if after.correct > prev_best and not repeat:
                self._best_progress[recipe.name] = after.correct
                self._grant(br, "grid.progress", ctx=ctx_tool,
                            reason=f"прогресс {recipe.name}: "
                                   f"{after.correct}/{recipe.n_filled}",
                            scale=after.correct / max(1, recipe.n_filled))
            # Веха "форма собрана" оплачивается ОДИН раз за эпизод на рецепт.
            # Разобрать и собрать заново — уже бесплатно, так что единственный
            # способ продолжить зарабатывать это нажать крафт.
            if after.complete and recipe.name not in self._shape_paid:
                self._shape_paid.add(recipe.name)
                self._grant(br, "grid.shape_complete", ctx=ctx_tool,
                            reason=f"форма {recipe.name} собрана целиком")
        elif cell_bad:
            # Отличаем "не та клетка" от "не тот предмет".
            key = ("grid.place_wrong_item"
                   if any(grid_after[s] != EMPTY and grid_before[s] == EMPTY
                          and s in recipe.filled_cells for s in [slot])
                   else "grid.place_wrong_cell")
            self._grant(br, key, ctx="",
                        reason=f"{item_name} -> слот {slot}: рецепту "
                               f"{recipe.name} это мешает")
        return br

    def on_take_from_grid(self, grid_before: Sequence[int], grid_after: Sequence[int],
                          slot: int, has_table: bool = True) -> RewardBreakdown:
        """Забрал предмет из сетки — если сломал правильную часть, штраф."""
        br = RewardBreakdown()
        info_before = best_partial(grid_before, has_table)
        if info_before is None:
            return br
        info_after = match_grid(grid_after, info_before.recipe)
        if info_after.correct < info_before.correct:
            self._grant(br, "grid.break_shape", ctx="",
                        reason=f"убрал верную деталь из слота {slot} "
                               f"({info_before.recipe.name})")
        return br

    def on_craft(self, recipe: Optional[Recipe], success: bool) -> RewardBreakdown:
        """Нажата кнопка крафта."""
        br = RewardBreakdown()
        if not success or recipe is None:
            self._grant(br, "grid.craft_fail", reason="рецепт не собран")
            return br

        name = recipe.name
        self._crafted_this_episode.append(name)

        # Антифарм: крафтит одно и то же подряд — штраф поверх затухания.
        if self._last_craft == name:
            self._same_craft_streak += 1
        else:
            self._same_craft_streak = 0
        self._last_craft = name
        if self._same_craft_streak >= 3:
            self._grant(br, "craft.duplicate_spam", reason=f"{name} подряд "
                                                           f"{self._same_craft_streak + 1} раз")

        # Крафт того, что уже не нужно для цели, почти не оплачивается:
        # ошибкой это не назовёшь, но и кормить за это нельзя.
        if self._goal_recipes and name not in self._goal_recipes:
            br.add("craft.not_needed", 0.0,
                   f"{name} сейчас не нужен для цели — награды нет")
            return br

        if recipe.kind:  # это инструмент
            self._grant(br, "craft.tool", ctx=name, reason=f"скрафтил {name}")
            self._grant(br, "craft.tool_first_ever", ctx=name,
                        reason=f"первый в жизни {name}")
            tier = recipe.tier or ""
            if tier and tier not in self._tiers_reached:
                self._tiers_reached.add(tier)
                self._grant(br, "craft.tier_up", ctx=tier,
                            reason=f"новый тир: {tier}",
                            scale=1.0 + 0.5 * TIER_INDEX.get(tier, 0))
        else:
            key = f"craft.{name}"
            if self.db.rule(key) is None:
                key = "craft.planks"
            self._grant(br, key, ctx=name, reason=f"скрафтил {name}")
        return br

    def on_world(self, event: str, ctx: str = "", reason: str = "",
                 scale: float = 1.0) -> RewardBreakdown:
        """Универсальная точка для событий мира/ориентации."""
        br = RewardBreakdown()
        self._grant(br, event, ctx=ctx, reason=reason, scale=scale)
        return br

    def on_approach(self, target: str, dist: int) -> RewardBreakdown:
        """
        Награда за приближение к цели — только за НОВЫЙ рекорд близости.

        Классическое potential-based shaping: ходить туда-сюда бессмысленно,
        платят лишь за то, что агент оказался ближе, чем когда-либо в этом
        эпизоде. Отход от цели по-прежнему штрафуется отдельно.
        """
        br = RewardBreakdown()
        best = self._best_dist.get(target)
        if best is None or dist < best:
            self._best_dist[target] = dist
            if best is not None:
                self._grant(br, "spatial.approach", ctx=target,
                            reason=f"новый рекорд близости к {target}: {dist}")
        return br

    def on_step_overhead(self, action_index: int, was_valid: bool,
                         is_noop: bool) -> RewardBreakdown:
        """Постоянные мелкие издержки: время, простой, зацикливание."""
        br = RewardBreakdown()
        self._grant(br, "behavior.step_cost")
        if is_noop:
            self._grant(br, "behavior.noop")
        if not was_valid:
            self._grant(br, "behavior.invalid_action")

        self._recent_actions.append(action_index)
        if len(self._recent_actions) > 8:
            self._recent_actions.pop(0)
        # A B A B A B — явный цикл.
        ra = self._recent_actions
        if len(ra) >= 6 and ra[-1] == ra[-3] == ra[-5] and ra[-2] == ra[-4] == ra[-6] \
                and ra[-1] != ra[-2]:
            self._grant(br, "behavior.loop", reason="повторяющийся цикл действий")
        return br

    def on_goal(self, goal_name: str) -> RewardBreakdown:
        br = RewardBreakdown()
        self._grant(br, "goal.completed", ctx=goal_name,
                    reason=f"цель выполнена: {goal_name}")
        return br

    # ---------------- память движка для наблюдения ----------------
    def paid_state(self, recipe_names) -> Dict[str, object]:
        """
        Что уже оплачено в этом эпизоде — чтобы агент это ВИДЕЛ.

        Без этого среда неявно меняет правила между одинаковыми состояниями,
        и обучение на таких данных невозможно (частично наблюдаемая среда).
        """
        slots = [0.0] * 9
        for (rname, slot, _item) in self._cells_paid:
            if not recipe_names or rname in recipe_names:
                if 0 <= slot < 9:
                    slots[slot] = 1.0
        shape_done = 1.0 if (recipe_names and
                             any(r in self._shape_paid for r in recipe_names)) else 0.0
        return {"slots": slots, "shape": shape_done,
                "crafts": min(len(self._crafted_this_episode), 5) / 5.0}

    # ---------------- подсказка для модели ----------------
    def potential_for(self, recipe_name: str) -> float:
        """Сколько ещё можно выжать из этого рецепта (для curriculum)."""
        times, nxt = self.db.peek("craft.tool", ctx=recipe_name)
        return nxt

"""
ГИБРИДНОЕ ЗРЕНИЕ: пиксели + названия предметов.

Мир агент видит глазами — RGB-картинкой, как человек на экране. Блоки, мобов,
расстояние, перспективу он распознаёт сам по цветам и формам.

А вот инвентарь, сетку крафта и предмет в руке он читает символьно, по
названиям. Так и у человека: сундук ты видишь глазами, но что в нём лежит —
читаешь подписями, а не угадываешь по пикселям иконки.

Что стало пикселями:  voxels (125) + blockmap (169) + entmap (169) = 463 числа
Что осталось словами: grid (9) + dense (133) + held (1) = 143 числа
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .mc_env import MinecraftCraftEnv
from .render import FrameStack, Raycaster


class PixelVisionEnv:
    """
    Обёртка над MinecraftCraftEnv, добавляющая зрение-картинку.

    Наружу выдаёт словарь:
        pixels — (N*3, H, W) float32 в [0,1], стек последних N кадров
        grid   — (9,)   int64,   что лежит в сетке крафта
        dense  — (133,) float32, инвентарь, здоровье, цель, прогресс
        held   — (1,)   int64,   предмет в руке

    frame_skip — сколько кадров мира на одно решение агента. Кадры внутри
    пропуска всё равно рисуются и попадают в стек: агент думает реже, но
    видит всё движение. Ровно этот приём даёт DQN понимание скорости.
    """

    def __init__(self, env: MinecraftCraftEnv, width: int = 426,
                 height: int = 240, n_frames: int = 4, frame_skip: int = 1,
                 fov: float = 75.0, max_dist: float = 12.0,
                 keep_symbolic_world: bool = False):
        self.env = env
        self.rc = Raycaster(width, height, fov=fov, max_dist=max_dist)
        self.stack = FrameStack(n_frames, height, width, 3)
        self.frame_skip = max(1, int(frame_skip))
        self.n_frames = int(n_frames)
        self.width, self.height = int(width), int(height)
        # Если True — рядом с пикселями остаются и старые символьные карты.
        # По умолчанию выключено: мир агент должен читать глазами.
        self.keep_symbolic_world = bool(keep_symbolic_world)

    # -- служебное ---------------------------------------------------------
    def _frame(self) -> np.ndarray:
        """Рисует текущий кадр от лица агента."""
        a = self.env.agent
        return self.rc.render(self.env.world, a.x, a.y, a.z,
                              a.facing, a.pitch, self.env.mobs)

    def _pack(self, raw: Dict, pixels: np.ndarray) -> Dict:
        """Собирает итоговое наблюдение: картинка + символьная часть."""
        out = {
            "pixels": pixels,
            "grid": raw["grid"],
            "dense": raw["dense"],
            "held": raw["held"],
        }
        if self.keep_symbolic_world:
            for k in ("voxels", "blockmap", "entmap"):
                if k in raw:
                    out[k] = raw[k]
        return out

    # -- API среды ---------------------------------------------------------
    def reset(self, goal_index: Optional[int] = None) -> Dict:
        raw = (self.env.reset() if goal_index is None
               else self.env.reset(goal_index))
        pixels = self.stack.reset(self._frame())
        return self._pack(raw, pixels)

    def step(self, action: int):
        """
        Делает frame_skip шагов мира с одним и тем же действием.

        Награды суммируются, кадры все до одного попадают в стек.
        """
        total = 0.0
        raw, done, info = None, False, {}
        for i in range(self.frame_skip):
            raw, r, done, info = self.env.step(action)
            total += float(r)
            # Каждый промежуточный кадр тоже виден агенту — иначе движение
            # между решениями теряется и стек перестаёт нести скорость.
            pixels = self.stack.push(self._frame())
            if done:
                break
        return self._pack(raw, pixels), total, done, info

    # -- прозрачный доступ к исходной среде --------------------------------
    def action_mask(self) -> np.ndarray:
        return self.env.action_mask()

    def render_rgb(self) -> np.ndarray:
        """Один кадр (H, W, 3) uint8 — для просмотра человеком."""
        return self._frame()

    @property
    def agent(self):
        return self.env.agent

    @property
    def mobs(self) -> List:
        return self.env.mobs

    @property
    def world(self) -> np.ndarray:
        return self.env.world

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def done(self) -> bool:
        return self.env.done

    def __getattr__(self, name):
        # Всё, чего нет у обёртки, спрашиваем у среды.
        return getattr(self.__dict__["env"], name)

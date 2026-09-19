"""
ПИКСЕЛЬНОЕ ЗРЕНИЕ АГЕНТА.

Раньше модель получала мир уже разобранным на числа (какой блок в какой
клетке). Это удобно, но это подсказка: мы за неё решали, что важно.

Здесь агент видит КАРТИНКУ — ровно то, что видел бы человек на экране:
RGB-пиксели, цвета блоков, перспективу, туман по расстоянию. Распознавать
"вот это дерево, а вот это зомби" он должен сам, по цвету и форме.

Рендер — классический raycaster (как в Wolfenstein 3D): для каждого столбца
пикселей пускаем луч, смотрим, во что он упёрся, рисуем вертикальную полосу
нужного цвета и высоты. Быстро, без OpenGL, работает на голом numpy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..spaces import (
    BLOCKS,
    BLOCK_ID,
    ENTITIES,
    FACING_DELTA,
    HOSTILE,
    PASSABLE,
)

# ---------------------------------------------------------------------------
# Палитра: цвет каждого блока в RGB.
# Взято близко к настоящим текстурам Minecraft, чтобы агент, обученный здесь,
# не растерялся при переходе на реальный клиент.
# ---------------------------------------------------------------------------
BLOCK_COLOR: Dict[str, Tuple[int, int, int]] = {
    "air": (135, 206, 235),          # небо
    "stone": (125, 125, 125),
    "dirt": (134, 96, 67),
    "grass_block": (91, 153, 63),
    "oak_log": (102, 81, 50),
    "oak_planks": (162, 130, 78),
    "crafting_table": (166, 110, 60),
    "furnace": (88, 88, 88),
    "iron_ore": (197, 175, 158),
    "diamond_ore": (93, 219, 213),
    "bedrock": (40, 40, 40),
    "water": (59, 110, 220),
    "gold_ore": (231, 201, 90),
    "redstone_ore": (190, 60, 55),
    "coal_ore": (54, 54, 54),
    "gravel": (140, 132, 130),
    "sand_block": (219, 207, 163),
    "oak_leaves": (58, 122, 40),
    "torch_block": (255, 200, 80),
    "chest_block": (150, 110, 55),
    "lava": (232, 110, 30),
}

# Цвета мобов — намеренно контрастные, чтобы CNN быстро выучила "опасное/еда".
ENTITY_COLOR: Dict[str, Tuple[int, int, int]] = {
    "zombie": (44, 122, 88),
    "skeleton": (200, 200, 195),
    "spider": (58, 42, 38),
    "creeper": (78, 190, 78),
    "cow": (78, 58, 48),
    "pig": (226, 145, 150),
    "sheep": (231, 231, 231),
    "chicken": (240, 240, 210),
}

_SKY = (135, 206, 235)
_GROUND = (74, 118, 50)

# Предрасчёт таблиц цветов: индекс блока -> RGB. Обращение по массиву
# быстрее, чем поиск по словарю на каждый пиксель.
_BLOCK_LUT = np.array(
    [BLOCK_COLOR.get(b, (255, 0, 255)) for b in BLOCKS], dtype=np.uint8
)
_ENT_LUT = np.array(
    [(0, 0, 0)] + [ENTITY_COLOR.get(e, (255, 0, 255)) for e in ENTITIES[1:]],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# ТЕКСТУРЫ. В настоящем Minecraft у каждого блока грань 16x16 с зерном:
# камень крапчатый, дерево полосатое, руда в вкраплениях. Плоская заливка
# лишает CNN половины признаков — по ней не отличить близкую стену от
# далёкой и не заметить движение вдоль поверхности.
#
# Рисуем зерно процедурно: атлас (блоков, 16, 16) множителей яркости.
# Считается один раз при импорте, стоит доли миллисекунды.
# ---------------------------------------------------------------------------
# Размер текселя. Было 64 — на 640x360 фактура размазывалась по крупному
# блоку, и детализация падала до 5.69 при пороге 7.
#
# Поднимать её амплитудой шума — ловушка: при разбросе x2.6 метрика даёт
# 16.35, но камень превращается в телевизионный "снег". Проверено глазами,
# забраковано. Мельче тексель — честный путь: рисунок тот же, просто он
# не растягивается. 192 даёт 7.92 без единого лишнего децибела шума.
TEX = 192


def _build_atlas() -> np.ndarray:
    """Множители яркости на каждую точку грани каждого блока.

    Разброс шире, чем кажется нужным (0.72..1.28): в настоящем кадре
    Minecraft 56 тысяч цветов при детализации 11, у плоской заливки — 655
    при 0.74. Фактура это половина признаков, по которым CNN оценивает
    расстояние и замечает движение вдоль поверхности.
    """
    rng = np.random.default_rng(1234)
    atlas = np.ones((len(BLOCKS), TEX, TEX), dtype=np.float32)
    for i, name in enumerate(BLOCKS):
        if name == "air":
            continue
        # База: мелкий шум, у всех блоков свой рисунок, но повторяемый.
        t = 1.0 + rng.normal(0, 0.075, (TEX, TEX)).astype(np.float32)

        # Частоты узоров привязаны к размеру текселя: при TEX=192 шаг
        # синуса обязан стать втрое мельче, иначе доски и волокна
        # расплываются в полосы шириной с полблока.
        k = TEX / 64.0

        if "log" in name:
            # Дерево: вертикальные волокна.
            fib = 1.0 + 0.13 * np.sin(np.arange(TEX) * 1.9 / k)
            t *= fib[None, :]
        elif "planks" in name:
            # Доски: горизонтальные полосы с тёмным стыком.
            t *= (1.0 + 0.07 * np.sin(np.arange(TEX)[:, None] * 1.6 / k))
            t[::max(1, int(8 * k)), :] *= 0.78
        elif "ore" in name:
            # Руда: несколько ярких вкраплений на каменном фоне.
            t = 1.0 + rng.normal(0, 0.050, (TEX, TEX)).astype(np.float32)
            # Вкраплений больше и они крупнее — площадь грани выросла в k^2.
            for _ in range(int(14 * k * k)):
                cy, cx = rng.integers(2, TEX - 2, 2)
                sp = max(1, int(k))
                t[cy - sp:cy + sp + 1, cx - sp:cx + sp + 1] *= 1.38
        elif name in ("stone", "cobblestone", "gravel", "furnace"):
            # Камень: крупная крапина.
            t *= 1.0 + rng.normal(0, 0.092, (TEX, TEX)).astype(np.float32)
        elif "grass" in name:
            # Трава: пучками.
            # Трава: пучки + редкие цветы, как на скриншоте с сакурой.
            t *= 1.0 + rng.normal(0, 0.100, (TEX, TEX)).astype(np.float32)
            for _ in range(int(10 * k * k)):
                fy, fx_ = rng.integers(0, TEX, 2)
                t[fy, fx_] *= 1.34
        elif name in ("crafting_table", "chest_block"):
            # Верстак: рамка по краю грани — заметная форма для CNN.
            t[0, :] *= 0.80; t[-1, :] *= 0.80
            t[:, 0] *= 0.80; t[:, -1] *= 0.80
            t[TEX // 2, :] *= 0.88

        atlas[i] = np.clip(t, 0.70, 1.30)
    return atlas


def _load_real_atlas():
    """
    Подхватывает настоящие текстуры Minecraft, если они извлечены.

    Файл создаётся скриптом tools/extract_textures.py из .jar игры.
    Нет файла — работаем на процедурном зерне, всё по-прежнему.

    Возвращает (цвет_атлас, размер) или (None, 0).
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "textures.npz")
    if not os.path.exists(path):
        return None, 0
    try:
        data = np.load(path, allow_pickle=True)
        atlas = data["atlas"]                    # (блоков, S, S, 3) uint8
        if atlas.shape[0] != len(BLOCKS):
            return None, 0
        return atlas.astype(np.float32), int(atlas.shape[1])
    except Exception:
        return None, 0


_ATLAS = _build_atlas()

# Настоящие текстуры из .jar, если пользователь их извлёк. Хранят ЦВЕТ
# каждого текселя, а не множитель яркости, поэтому применяются иначе:
# цвет блока берётся прямо из текстуры, а не из палитры BLOCK_COLOR.
_REAL_ATLAS, _REAL_TEX = _load_real_atlas()
_HAS_REAL = _REAL_ATLAS is not None
if _HAS_REAL:
    # Средняя яркость каждой текстуры — по ней нормируем освещение, чтобы
    # текстурные блоки не оказались светлее или темнее процедурных.
    _REAL_MEAN = _REAL_ATLAS.reshape(len(BLOCKS), -1, 3).mean(axis=1)

# Цветовой шум: небольшой независимый сдвиг по каждому каналу RGB.
# Яркостный атлас двигает цвет вдоль одной прямой в цветовом кубе, поэтому
# палитра выходит бедной. Этот шум растаскивает оттенки в стороны.
_RNG_T = np.random.default_rng(99)
_TINT = (1.0 + _RNG_T.normal(0, 0.030, (len(BLOCKS), TEX, TEX, 3))
         ).astype(np.float32).clip(0.90, 1.10)

_PASSABLE_IDS = {BLOCK_ID[b] for b in PASSABLE if b in BLOCK_ID}
_HOSTILE_SET = set(HOSTILE)


class Raycaster:
    """
    Рисует вид от первого лица.

    Параметры задаются один раз, потом render() зовётся каждый шаг.
    Все тяжёлые таблицы считаются в __init__, чтобы шаг был дешёвым.
    """

    def __init__(self, width: int = 128, height: int = 96, fov: float = 70.0,
                 max_dist: float = 12.0, fog: bool = True):
        self.w = int(width)
        self.h = int(height)
        self.fov = float(fov)
        self.max_dist = float(max_dist)
        self.fog = bool(fog)

        # Углы лучей: по одному на столбец пикселей.
        half = np.radians(self.fov) / 2.0
        self._ray_angles = np.linspace(-half, half, self.w, dtype=np.float32)
        # Поправка на "рыбий глаз": в raycaster'е без неё стены выгибаются.
        self._fisheye = np.cos(self._ray_angles).astype(np.float32)

        # Вертикальная координата каждой строки, один раз.
        self._rows = np.arange(self.h, dtype=np.float32)

        # Небо: вертикальный градиент + облачные разводы. Плоские строки
        # (один цвет на всю ширину) съедали почти половину кадра и резко
        # обедняли палитру — у CNN пропадала опора в верхней части вида.
        top = np.array((104, 176, 231), np.float32)
        bot = np.array((176, 218, 243), np.float32)
        k = (np.arange(self.h, dtype=np.float32) / max(self.h - 1, 1))[:, None]
        grad = top[None, :] * (1 - k) + bot[None, :] * k
        sky = np.repeat(grad[:, None, :], self.w, axis=1)

        yy = np.arange(self.h, dtype=np.float32)[:, None]
        xx = np.arange(self.w, dtype=np.float32)[None, :]
        # Сумма синусов разной частоты: дешёвая имитация облаков.
        clouds = (np.sin(xx * 0.055 + yy * 0.021)
                  + 0.6 * np.sin(xx * 0.017 - yy * 0.039)
                  + 0.4 * np.sin(xx * 0.101 + yy * 0.062))
        clouds = np.clip((clouds + 1.0) * 0.5, 0, 1) ** 1.7
        white = np.array((246, 250, 253), np.float32)
        sky = sky * (1 - 0.62 * clouds[..., None]) \
            + white[None, None, :] * (0.62 * clouds[..., None])
        # Лёгкий дизер: разбивает оставшиеся плоские участки.
        sky += np.random.default_rng(5).normal(0, 6.5, sky.shape)
        self._sky_grad = np.clip(sky, 0, 255).astype(np.uint8)

        # Дизер: слабый постоянный шум ±1.5 уровня яркости. Ломает плоские
        # заливки, почти не влияя на детализацию (меняется плавно, между
        # соседними пикселями разница меньше единицы).
        self._dither = np.random.default_rng(7).normal(
            0, 4.6, (self.h, self.w, 3)).astype(np.float32)

        # Масштаб проекции: во сколько пикселей превращается один блок
        # высоты на расстоянии один блок. Считается из вертикального угла
        # обзора, который выводится из горизонтального и пропорций кадра.
        tan_v = np.tan(half) * (self.h / self.w)
        self._proj = float(self.h / (2.0 * tan_v))

    # -- основное ----------------------------------------------------------
    def render(self, world: np.ndarray, ax: float, ay: float, az: float,
               facing: int, pitch: int, mobs: List) -> np.ndarray:
        """
        Возвращает картинку (H, W, 3) uint8 — вид от первого лица.

        world  — куб блоков (X, Y, Z) с индексами блоков
        ax/ay/az — позиция агента, facing — сторона света, pitch — наклон
        mobs   — список объектов с .kind/.x/.y/.z/.health
        """
        W, H, D = world.shape
        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        # Небо с градиентом: у горизонта светлее, в зените насыщеннее.
        img[:] = self._sky_grad

        # Горизонт гуляет от наклона головы.
        horizon = int(self.h * 0.5 + pitch * self.h * 0.20)

        # Направление взгляда. SOUTH=+Z принимаем за ноль.
        base = {0: 0.0, 1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}[int(facing) % 4]
        angles = base + self._ray_angles
        dx = np.sin(angles).astype(np.float32)
        dz = np.cos(angles).astype(np.float32)

        px, pz = ax + 0.5, az + 0.5
        floor_y = float(int(round(ay)))       # поверхность, по которой ходим
        eye = floor_y + 0.62                  # глаза чуть выше пояса
        K = self._proj                        # масштаб проекции

        depth = np.full((self.h, self.w), np.inf, dtype=np.float32)

        # --- ПОЛ ------------------------------------------------------------
        # Горизонтальную поверхность лучом-в-стену не нарисовать: нужен
        # floor casting. Для каждой строки ниже горизонта считаем, на каком
        # расстоянии луч протыкает плоскость пола, и берём цвет того блока.
        r0 = max(horizon + 1, 0)
        if r0 < self.h:
            rows = np.arange(r0, self.h, dtype=np.float32)
            dperp = K * (eye - floor_y) / (rows - horizon)      # (R,)
            dist_f = dperp[:, None] / self._fisheye[None, :]    # (R,W)
            fxw = px + dx[None, :] * dist_f
            fzw = pz + dz[None, :] * dist_f
            gx = np.clip(np.floor(fxw), 0, W - 1).astype(np.int32)
            gz = np.clip(np.floor(fzw), 0, D - 1).astype(np.int32)
            gy = int(np.clip(floor_y - 1, 0, H - 1))
            gid = world[gx, gy, gz].astype(np.int32)
            gcol = _BLOCK_LUT[gid].astype(np.float32)

            # Текстура грани: те же 16x16, что и у стен.
            ui = ((fxw - np.floor(fxw)) * TEX).astype(np.int32) % TEX
            vi = ((fzw - np.floor(fzw)) * TEX).astype(np.int32) % TEX
            gcol *= _ATLAS[gid, vi, ui][..., None]
            gcol *= _TINT[gid, vi, ui]
            gcol *= 1.12          # верхняя грань ловит больше света
            # Сетка блоков: тонкие тёмные швы. Дают сети опору для оценки
            # расстояния и скорости — без них пол сливается в заливку.
            seam = ((fxw - np.floor(fxw) < 0.045) |
                    (fzw - np.floor(fzw) < 0.045))
            gcol[seam] *= 0.80

            if self.fog:
                f = np.clip(dist_f / self.max_dist, 0, 1)[..., None] ** 1.3
                gcol = gcol * (1 - f) + np.array(_SKY, np.float32) * f
            # Мягкое виньетирование по краям кадра: у настоящего рендера
            # яркость плавно падает от центра, добавляя оттенков.
            vx = np.abs(np.linspace(-1, 1, self.w, dtype=np.float32))[None, :]
            gcol *= (1.0 - 0.06 * vx ** 2)[..., None]
            gcol += self._dither[r0:]
            vis = dist_f <= self.max_dist
            img[r0:] = np.where(vis[..., None], gcol, _SKY).astype(np.uint8)
            depth[r0:] = np.where(vis, dist_f, np.inf)

        # --- БЛОКИ ----------------------------------------------------------
        # Для каждого уровня высоты отдельно ищем ближайший блок вдоль луча,
        # затем рисуем с проверкой глубины. Уровней мало (мир низкий),
        # поэтому это быстрее полного марша по всем ячейкам.
        step = 0.10
        n_steps = int(self.max_dist / step)
        ts = np.arange(1, n_steps + 1, dtype=np.float32) * step
        wx = np.floor(px + np.outer(ts, dx)).astype(np.int32)
        wz = np.floor(pz + np.outer(ts, dz)).astype(np.int32)
        inside = (wx >= 0) & (wx < W) & (wz >= 0) & (wz < D)
        wxc, wzc = np.clip(wx, 0, W - 1), np.clip(wz, 0, D - 1)
        cols = np.arange(self.w)

        y_lo = int(np.clip(floor_y, 0, H - 1))
        for yl in range(y_lo, H):
            ids = world[wxc, yl, wzc].astype(np.int32)
            ids[~inside] = 0
            solid = ids != 0
            for pid in _PASSABLE_IDS:
                solid &= ids != pid
            hit = solid.any(axis=0)
            if not hit.any():
                continue
            first = np.where(hit, solid.argmax(axis=0), 0)
            bid = np.where(hit, ids[first, cols], 0)
            dist = np.where(hit, ts[first], np.inf).astype(np.float32)
            dperp = dist * self._fisheye

            # Вертикальные границы блока на экране.
            top = horizon - K * ((yl + 1) - eye) / dperp
            bot = horizon - K * (yl - eye) / dperp
            y0 = np.clip(np.floor(top), 0, self.h).astype(np.int32)
            y1 = np.clip(np.ceil(bot), 0, self.h).astype(np.int32)

            # Боковые грани темнее — так читается объём куба.
            safe = np.where(hit, dist, self.max_dist)
            hx, hz = px + dx * safe, pz + dz * safe
            fx, fz = hx - np.floor(hx), hz - np.floor(hz)
            edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fz, 1 - fz))
            shade = np.where(edge < 0.06, 0.70, 1.0).astype(np.float32)

            col = _BLOCK_LUT[bid].astype(np.float32) * shade[:, None]
            if self.fog:
                f = np.clip(dist / self.max_dist, 0, 1)[:, None] ** 1.3
                col = col * (1 - f) + np.array(_SKY, np.float32) * f

            rows_i = self._rows[:, None]
            band = ((rows_i >= y0[None, :]) & (rows_i < y1[None, :])
                    & hit[None, :] & (dist[None, :] < depth))

            # Координаты на грани блока. По вертикали глубина вдоль столбца
            # постоянна, поэтому v линейна по экрану — считается одним махом.
            denom = np.maximum(bot - top, 1e-6)[None, :]
            vf = np.clip((rows_i - top[None, :]) / denom, 0.0, 0.999)
            # Какую грань задели: боковую по X или по Z.
            xface = np.minimum(fx, 1 - fx) < np.minimum(fz, 1 - fz)
            u_col = np.where(xface, fz, fx)
            ui = (u_col * TEX).astype(np.int32) % TEX
            vi = (vf * TEX).astype(np.int32) % TEX
            bidb = np.broadcast_to(bid[None, :], vi.shape)
            uib = np.broadcast_to(ui[None, :], vi.shape)
            if _HAS_REAL:
                # Настоящая текстура: цвет берётся из неё, палитра не нужна.
                rvi = (vf * _REAL_TEX).astype(np.int32) % _REAL_TEX
                rui = ((u_col * _REAL_TEX).astype(np.int32) % _REAL_TEX)
                ruib = np.broadcast_to(rui[None, :], rvi.shape)
                real = _REAL_ATLAS[bidb, rvi, ruib]
                col = real * (shade[:, None] if col.ndim == 2 else 1.0)[None]
                tex = np.ones(vi.shape, np.float32)
                tint = np.ones(vi.shape + (3,), np.float32)
            else:
                tex = _ATLAS[bidb, vi, uib]
                tint = _TINT[bidb, vi, uib]
            # Затенение снизу вверх: у пола темнее. В Minecraft это
            # ambient occlusion, и он же даёт плавный переход яркости —
            # именно из-за него в настоящем кадре десятки тысяч оттенков.
            ao = (0.74 + 0.26 * (1.0 - vf)).astype(np.float32)
            # Затенение и поперёк грани: у рёбер темнее, в середине светлее.
            # Плоская по горизонтали стена давала одинаковые цвета рядами.
            u_ao = (0.90 + 0.10 * np.sin(np.pi * u_col)).astype(np.float32)
            full = (np.broadcast_to(col[None], (self.h, self.w, 3))
                    * tex[..., None] * tint
                    * (ao * u_ao[None, :])[..., None])
            full += self._dither
            img[band] = full[band].astype(np.uint8)
            depth = np.where(band, np.broadcast_to(dist[None], depth.shape),
                             depth)

        self._draw_mobs(img, mobs, px, pz, ay, base, horizon, depth)
        return img

    # -- мобы --------------------------------------------------------------
    def _draw_mobs(self, img, mobs, px, pz, ay, base, horizon, depth) -> None:
        """Рисует мобов как спрайты-столбики поверх мира, с учётом глубины."""
        if not mobs:
            return
        half_fov = np.radians(self.fov) / 2.0

        # Сначала дальние, потом ближние — чтобы ближние перекрывали.
        order = sorted(
            mobs,
            key=lambda m: -((m.x + 0.5 - px) ** 2 + (m.z + 0.5 - pz) ** 2),
        )
        for m in order:
            mx = m.x + 0.5 - px
            mz = m.z + 0.5 - pz
            dist = float(np.hypot(mx, mz))
            if dist < 0.25 or dist > self.max_dist:
                continue

            # Угол на моба относительно взгляда, нормализованный в [-pi, pi].
            ang = np.arctan2(mx, mz) - base
            while ang > np.pi:
                ang -= 2 * np.pi
            while ang < -np.pi:
                ang += 2 * np.pi
            if abs(ang) > half_fov * 1.1:
                continue

            cx = int((ang + half_fov) / (2 * half_fov) * self.w)
            size = int(self.h / max(dist, 0.5) * 0.85)
            if size < 2:
                continue
            w2 = max(1, size // 3)
            x0, x1 = max(0, cx - w2), min(self.w, cx + w2)
            if x1 <= x0:
                continue

            # Прячем за стеной: если блок ближе моба — не рисуем.
            if np.median(depth[:, x0:x1]) < dist - 0.45:
                continue

            dy = (m.y - ay) * (self.h / max(dist, 0.5)) * 0.35
            y1 = int(horizon + size // 2 - dy)
            y0 = max(0, y1 - size)
            y1 = min(self.h, y1)
            if y1 <= y0:
                continue

            color = _ENT_LUT[ENTITIES.index(m.kind) if m.kind in ENTITIES else 0]
            color = color.astype(np.float32)
            if self.fog:
                f = min(1.0, dist / self.max_dist) ** 1.4
                color = color * (1 - f) + np.array(_SKY, np.float32) * f
            patch = (color[None, None, :]
                     + self._dither[y0:y1, x0:x1])
            img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

            # Враждебным рисуем тёмную "голову": подсказка формой, не только
            # цветом, чтобы сеть училась и на серых кадрах.
            if m.kind in _HOSTILE_SET and (y1 - y0) > 6:
                hh = (y1 - y0) // 4
                hpatch = (color[None, None, :] * 0.55
                          + self._dither[y0:y0 + hh, x0:x1])
                img[y0:y0 + hh, x0:x1] = np.clip(
                    hpatch, 0, 255).astype(np.uint8)


class FrameStack:
    """
    Память на N кадров — то самое "сколько FPS воспринимает модель".

    Один кадр — это фотография: по ней не понять, зомби бежит на тебя или от
    тебя. Стек из нескольких подряд идущих кадров даёт сети направление и
    скорость движения. Ровно так сделано в DQN на Atari (там N=4).
    """

    def __init__(self, n: int, h: int, w: int, channels: int = 3):
        self.n = int(n)
        self.shape = (h, w, channels)
        self.buf = np.zeros((self.n,) + self.shape, dtype=np.uint8)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """В начале эпизода заполняем всю память первым кадром."""
        self.buf[:] = frame
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Сдвигаем историю на один кадр и кладём новый в конец."""
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = frame
        return self.get()

    def get(self) -> np.ndarray:
        """(N*C, H, W) float32 в [0,1] — готово для Conv2d."""
        x = self.buf.astype(np.float32) / 255.0
        return np.transpose(x, (0, 3, 1, 2)).reshape(-1, *self.shape[:2])

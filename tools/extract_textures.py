#!/usr/bin/env python3
"""
ИЗВЛЕЧЕНИЕ ТЕКСТУР БЛОКОВ ИЗ .jar MINECRAFT.

Сейчас рендер рисует блоки процедурным зерном: похоже на Minecraft, но не
он. Настоящие текстуры лежат прямо в .jar игры и достаются без всякой
декомпиляции — это обычный zip-архив.

    .minecraft/versions/26.1/26.1.jar
        assets/minecraft/textures/block/*.png

ВАЖНО: берём ТОЛЬКО текстуры блоков мира (камень, трава, брёвна). Иконки
предметов НЕ трогаем сознательно: инвентарь агент читает названиями, и
подсовывать ему картинки предметов — значит дать угадывать крафт по виду
иконки, минуя чтение. Это была бы поблажка, а не обучение.

Использование:
    python3 tools/extract_textures.py --jar "C:/.../26.1.jar"
    python3 tools/extract_textures.py --auto        # ищет сам

Результат: brain/env/textures.npz — атлас, который подхватит render.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.spaces import BLOCKS  # noqa: E402

# Какой файл текстуры соответствует нашему блоку. Где у блока разные грани
# (трава сверху зелёная, сбоку землистая) — берём характерную.
TEXTURE_MAP: Dict[str, List[str]] = {
    "stone": ["stone.png"],
    "dirt": ["dirt.png"],
    "grass_block": ["grass_block_top.png", "grass_block_side.png"],
    "oak_log": ["oak_log.png", "oak_log_top.png"],
    "oak_planks": ["oak_planks.png"],
    "crafting_table": ["crafting_table_front.png", "crafting_table_top.png"],
    "furnace": ["furnace_front.png", "furnace_side.png"],
    "iron_ore": ["iron_ore.png"],
    "diamond_ore": ["diamond_ore.png"],
    "bedrock": ["bedrock.png"],
    "water": ["water_still.png"],
    "gold_ore": ["gold_ore.png"],
    "redstone_ore": ["redstone_ore.png"],
    "coal_ore": ["coal_ore.png"],
    "gravel": ["gravel.png"],
    "sand_block": ["sand.png"],
    "oak_leaves": ["oak_leaves.png"],
    "torch_block": ["torch.png"],
    "chest_block": ["chest.png", "oak_planks.png"],
    "lava": ["lava_still.png"],
}

# Трава и листва в игре серые — цвет им даёт биом. Без этого они выйдут
# блёклыми, поэтому красим вручную под умеренный биом.
BIOME_TINT: Dict[str, tuple] = {
    "grass_block": (0.48, 0.74, 0.35),
    "oak_leaves": (0.30, 0.66, 0.24),
}

DEFAULT_PATHS = [
    Path.home() / "AppData/Roaming/.minecraft/versions",
    Path.home() / ".minecraft/versions",
    Path.home() / "Library/Application Support/minecraft/versions",
]


def find_jar(version: str = "26.1") -> Optional[Path]:
    """Ищет .jar нужной версии в стандартных местах установки."""
    for base in DEFAULT_PATHS:
        if not base.exists():
            continue
        exact = base / version / f"{version}.jar"
        if exact.exists():
            return exact
        # Версия могла быть установлена под другим именем папки.
        for d in sorted(base.iterdir(), reverse=True):
            cand = d / f"{d.name}.jar"
            if cand.exists() and version in d.name:
                return cand
    return None


def load_png(data: bytes) -> np.ndarray:
    """PNG в массив RGB. Пытается через PIL, иначе — встроенным декодером."""
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data)).convert("RGBA")
        return np.asarray(im)
    except ImportError:
        raise SystemExit("нужен pillow: pip install pillow")


def extract(jar: Path, size: int = 64, verbose: bool = True) -> dict:
    """
    Достаёт текстуры блоков и приводит к одному размеру.

    Возвращает словарь: имя блока -> (size, size, 3) uint8.
    """
    from PIL import Image
    import io

    out: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    with zipfile.ZipFile(jar) as z:
        names = set(z.namelist())
        base = "assets/minecraft/textures/block/"
        if not any(n.startswith(base) for n in names):
            # В очень старых версиях папка называлась blocks/
            base = "assets/minecraft/textures/blocks/"

        for block in BLOCKS:
            if block == "air":
                continue
            found = None
            for fname in TEXTURE_MAP.get(block, [f"{block}.png"]):
                path = base + fname
                if path in names:
                    found = path
                    break
            if not found:
                missing.append(block)
                continue

            im = Image.open(io.BytesIO(z.read(found))).convert("RGBA")
            # Анимированные текстуры (вода, лава) хранятся вертикальной
            # лентой кадров — берём первый квадратный кадр.
            if im.height > im.width:
                im = im.crop((0, 0, im.width, im.width))
            im = im.resize((size, size), Image.NEAREST)
            arr = np.asarray(im).astype(np.float32)

            rgb = arr[..., :3]
            alpha = arr[..., 3:4] / 255.0
            # Прозрачное (листва, факел) смешиваем с серым, чтобы не было
            # чёрных дыр — в рендере эти блоки и так полупрозрачны.
            rgb = rgb * alpha + 128.0 * (1 - alpha)

            if block in BIOME_TINT:
                rgb = rgb * np.array(BIOME_TINT[block], np.float32)

            out[block] = np.clip(rgb, 0, 255).astype(np.uint8)

    if verbose:
        print(f"извлечено текстур: {len(out)} из {len(BLOCKS) - 1}")
        if missing:
            print(f"не найдены (останутся процедурными): {', '.join(missing)}")
    return out


def build_atlas(tex: dict, size: int = 64) -> np.ndarray:
    """
    Собирает атлас (блоков, size, size, 3) в порядке BLOCKS.

    Блоки без текстуры остаются нулями — рендер для них возьмёт
    процедурное зерно.
    """
    atlas = np.zeros((len(BLOCKS), size, size, 3), dtype=np.uint8)
    for i, name in enumerate(BLOCKS):
        if name in tex:
            atlas[i] = tex[name]
    return atlas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jar", type=str, help="путь к .jar Minecraft")
    ap.add_argument("--version", default="26.1", help="версия для авто-поиска")
    ap.add_argument("--auto", action="store_true", help="искать .jar самому")
    ap.add_argument("--size", type=int, default=64, help="размер грани")
    ap.add_argument("--out", default="brain/env/textures.npz")
    a = ap.parse_args()

    jar = Path(a.jar) if a.jar else find_jar(a.version)
    if not jar or not jar.exists():
        print("Не нашёл .jar. Укажите путь явно:")
        print('  python3 tools/extract_textures.py --jar "C:/Users/ВЫ/'
              'AppData/Roaming/.minecraft/versions/26.1/26.1.jar"')
        print("\nИскал в:")
        for p in DEFAULT_PATHS:
            print(f"  {p}")
        return 1

    print(f"читаю {jar}")
    tex = extract(jar, size=a.size)
    if not tex:
        print("текстуры не найдены — структура .jar изменилась?")
        return 1

    atlas = build_atlas(tex, size=a.size)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, atlas=atlas,
                        blocks=np.array(BLOCKS, dtype=object),
                        size=a.size)
    mb = out.stat().st_size / 2 ** 20
    print(f"\nсохранено: {out}  ({mb:.1f} МБ)")
    print("рендер подхватит автоматически при следующем запуске")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

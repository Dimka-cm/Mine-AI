#!/usr/bin/env python3
"""
Скачивает client.jar нужной версии с серверов Mojang и извлекает текстуры.

Зачем отдельно от extract_textures.py: тот ищет .jar в локальной установке
Minecraft. На машине без игры (CI, сервер) искать нечего — файл надо взять
с официального зеркала. Ассеты Mojang в репозиторий не кладутся, поэтому
каждый, кто клонирует проект, собирает атлас сам этой командой.

    python3 tools/fetch_textures.py --version 26.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MANIFEST = "https://launchermeta.mojang.com/mc/game/version_manifest_v2.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать .jar и извлечь текстуры")
    ap.add_argument("--version", default="26.1")
    ap.add_argument("--size", type=int, default=16,
                    help="размер грани; 16 — нативный для Minecraft")
    ap.add_argument("--keep-jar", action="store_true",
                    help="не удалять скачанный .jar")
    args = ap.parse_args()

    print(f"манифест версий Mojang...")
    man = json.load(urllib.request.urlopen(MANIFEST, timeout=60))
    hit = [v for v in man["versions"] if v["id"] == args.version]
    if not hit:
        print(f"версии {args.version} нет. Последняя: {man['latest']['release']}")
        return 1

    meta = json.load(urllib.request.urlopen(hit[0]["url"], timeout=60))
    url = meta["downloads"]["client"]["url"]
    size_mb = meta["downloads"]["client"]["size"] / 1e6
    print(f"качаю client.jar {args.version} ({size_mb:.0f} МБ)...")

    tmp = Path(tempfile.gettempdir()) / f"mc{args.version}.jar"
    urllib.request.urlretrieve(url, tmp)
    print(f"скачан: {tmp}")

    root = Path(__file__).resolve().parent.parent
    rc = subprocess.call([sys.executable, str(root / "tools" / "extract_textures.py"),
                          "--jar", str(tmp), "--size", str(args.size)], cwd=root)
    if not args.keep_jar:
        tmp.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Фейковый mineflayer-мост: говорит тем же JSON-протоколом, что и bridge.js,
но без Minecraft-сервера. Нужен, чтобы проверить play.py целиком.

Запуск:
    python tests/mock_bridge.py --port 5599
"""
from __future__ import annotations

import argparse
import json
import socketserver
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.spaces import GRID_2X2_SLOTS, ITEMS

N_ITEMS = len(ITEMS)
ITEM_ID = {n: i for i, n in enumerate(ITEMS)}


class World:
    """Минимальная модель мира — ровно то, что нужно протоколу."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid = [0] * 9
        self.held = 0
        self.inv = [0] * N_ITEMS
        self.facing = 0
        self.pitch = 0
        self.table_open = False
        self.table_near = True          # верстак «рядом» для теста
        self.log_count = 0

    def active_slots(self):
        if self.table_open and self.table_near:
            return list(range(9))
        return list(GRID_2X2_SLOTS)

    def obs(self):
        return {
            "grid": list(self.grid),
            "voxels": [4] + [0] * 26,
            "inventory": [min(v, 64) / 64.0 for v in self.inv],
            "held": self.held,
            "facing": self.facing,
            "pitch": self.pitch,
            "near_table": 1 if self.table_near else 0,
            "table_dist": 1 if self.table_near else 99,
            "position": [0, 64, 0],
            "health": 20,
            "food": 20,
            "table_open": 1 if self.table_open else 0,
            "table_reachable": 1 if self.table_near else 0,
            "has_3x3": 1 if len(self.active_slots()) == 9 else 0,
            "active_slots": self.active_slots(),
        }

    def act(self, a):
        t, arg = a.get("type"), a.get("arg", 0)
        res = {"ok": True, "note": ""}
        if t == "select_item":
            if self.inv[arg] <= 0:
                res.update(ok=False, note="нет предмета")
            else:
                self.held = arg
        elif t == "place_in_slot":
            if self.held == 0:
                res.update(ok=False, note="рука пуста")
            elif arg not in self.active_slots():
                res.update(ok=False, note="слот недоступен (2x2)")
            elif self.grid[arg] != 0:
                res.update(ok=False, note="слот занят")
            else:
                self.grid[arg] = self.held
        elif t == "take_from_slot":
            if self.grid[arg] == 0:
                res.update(ok=False, note="слот пуст")
            else:
                self.grid[arg] = 0
        elif t == "clear_grid":
            self.grid = [0] * 9
        elif t == "craft":
            name = a.get("result_name", "")
            if name in ITEM_ID and any(self.grid):
                self.inv[ITEM_ID[name]] += 1
                self.grid = [0] * 9
                res["note"] = f"скрафтил {name}"
            else:
                res.update(ok=False, note="рецепт недоступен")
        elif t == "break_block":
            # первое дерево ломается рукой, камень — нет
            self.inv[ITEM_ID["oak_log"]] += 1
            self.log_count += 1
            res.update(block="oak_log", wrong_tool=0, note="сломал oak_log")
        elif t == "use_table":
            if self.table_open:
                self.table_open = False
                self.grid = [0] * 9
                res["note"] = "верстак закрыт"
            elif self.table_near:
                self.table_open = True
                res["note"] = "верстак открыт (3x3)"
            else:
                res.update(ok=False, note="верстака нет")
        return res


WORLD = World()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line.decode())
            cmd = msg.get("cmd")
            if cmd == "observe":
                reply = {"obs": WORLD.obs()}
            elif cmd == "reset":
                WORLD.reset()
                reply = {"obs": WORLD.obs()}
            elif cmd == "act":
                r = WORLD.act(msg.get("action", {}))
                reply = {"result": r, "obs": WORLD.obs()}
            elif cmd == "chat":
                reply = {"ok": True}
            else:
                reply = {"error": "?"}
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5599)
    args = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as s:
        print(f"[mock] фейковый мост на 127.0.0.1:{args.port}")
        s.serve_forever()

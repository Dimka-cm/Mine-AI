#!/usr/bin/env python3
"""
Пульт живого бота (4-й монитор).

Показывает состояние подключения к реальному Minecraft 26.1: жив ли мост,
что бот видит вокруг, чем занят, сколько здоровья и еды, какие награды
капают прямо сейчас. В отличие от monitor.py (обучение) и viewer3d.py
(симулятор) — это окно именно в ЖИВУЮ игру.

Работает и без запущенного моста: тогда честно показывает "мост не отвечает"
и инструкцию по запуску, а не пустую страницу.

Запуск:  python botmonitor.py --port 7863
"""
from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.spaces import BLOCKS, ENTITIES, HOSTILE, ITEMS, TOOL_LEVEL

# Целевая версия игры. 26.1 — последняя, которую понимает mineflayer,
# и первая деобфусцированная версия Minecraft.
TARGET_VERSION = "26.1"

STATE: Dict[str, Any] = {
    "connected": False,
    "error": "мост ещё не опрошен",
    "obs": None,
    "last_ok": 0.0,
    "polls": 0,
    "fails": 0,
    "history": [],          # последние наблюдения для мини-графика
}
LOCK = threading.RLock()


# --------------------------------------------------------------- опрос моста
def poll_bridge(host: str, port: int, period: float) -> None:
    """Фоновый опрос mineflayer-моста по тому же JSON-протоколу, что и play.py."""
    while True:
        try:
            with socket.create_connection((host, port), timeout=3.0) as s:
                fp = s.makefile("rwb")
                fp.write(b'{"cmd":"observe"}\n')
                fp.flush()
                line = fp.readline()
                if not line:
                    raise ConnectionError("мост закрыл соединение")
                reply = json.loads(line.decode())
                obs = reply.get("obs")
                if obs is None:
                    raise ValueError(reply.get("error", "нет поля obs"))
                with LOCK:
                    STATE["connected"] = True
                    STATE["error"] = ""
                    STATE["obs"] = obs
                    STATE["last_ok"] = time.time()
                    STATE["polls"] += 1
                    h = STATE["history"]
                    h.append({
                        "t": time.time(),
                        "health": obs.get("health", 0),
                        "food": obs.get("food", 0),
                    })
                    if len(h) > 120:
                        del h[:-120]
        except Exception as e:                      # мост не запущен — это норма
            with LOCK:
                STATE["connected"] = False
                STATE["error"] = str(e)
                STATE["fails"] += 1
        time.sleep(period)


# --------------------------------------------------------------- разбор мира
def describe(obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Человекочитаемая сводка из сырого наблюдения."""
    if not obs:
        return {}
    inv_raw = obs.get("inventory", [])
    inv = {}
    for i, v in enumerate(inv_raw):
        if v and i < len(ITEMS) and i != 0:
            inv[ITEMS[i]] = round(v * 64)

    held_id = obs.get("held", 0)
    held = ITEMS[held_id] if held_id < len(ITEMS) else "?"
    lvl = TOOL_LEVEL.get(held, 0)
    lvl_name = ["рука", "дерево", "камень", "железо", "алмаз"][min(lvl, 4)]

    # воксели 3x3x3 вокруг агента
    vox = obs.get("voxels", [])
    around = {}
    for b in vox:
        if b and b < len(BLOCKS):
            around[BLOCKS[b]] = around.get(BLOCKS[b], 0) + 1

    grid = obs.get("grid", [0] * 9)
    grid_names = [ITEMS[g] if g < len(ITEMS) else "?" for g in grid]

    # Кто рядом по карте сущностей — главный индикатор опасности.
    ent = obs.get("entmap", [])
    mobs: Dict[str, int] = {}
    danger = 0
    if ent:
        W = int(len(ent) ** 0.5) or 1
        for idx, e in enumerate(ent):
            if not e:
                continue
            nm = ENTITIES[e] if e < len(ENTITIES) else "?"
            mobs[nm] = mobs.get(nm, 0) + 1
            if nm in HOSTILE:
                i, k = divmod(idx, W)
                d = abs(i - W // 2) + abs(k - W // 2)
                if danger == 0 or d < danger:
                    danger = d

    return {
        "inv": inv,
        "held": held,
        "tool_level": lvl,
        "tool_level_name": lvl_name,
        "around": around,
        "grid": grid_names,
        "has_3x3": bool(obs.get("has_3x3", 0)),
        "table_open": bool(obs.get("table_open", 0)),
        "table_reachable": bool(obs.get("table_reachable", 0)),
        "position": obs.get("position", [0, 0, 0]),
        "health": obs.get("health", 0),
        "food": obs.get("food", 0),
        "facing": ["юг", "запад", "север", "восток"][obs.get("facing", 0) % 4],
        "mobs": mobs,
        "danger": danger,
        "hostile_count": sum(v for k, v in mobs.items() if k in HOSTILE),
    }


def read_rewards(db_path: str) -> Dict[str, Any]:
    """Последние награды из живой игры (ту же БД пишет play.py)."""
    p = Path(db_path)
    if not p.exists():
        return {"events": [], "top": [], "total": 0.0}
    try:
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        events = [dict(r) for r in con.execute(
            """SELECT episode, step, ctx_key, value, reason FROM reward_log
               ORDER BY id DESC LIMIT 25""")]
        top = [dict(r) for r in con.execute(
            """SELECT ctx_key, times, total_value FROM reward_state
               ORDER BY total_value DESC LIMIT 12""")]
        tot = con.execute(
            "SELECT COALESCE(SUM(value),0) FROM reward_log").fetchone()[0]
        con.close()
        return {"events": events, "top": top, "total": round(tot or 0.0, 2)}
    except sqlite3.Error as e:
        return {"events": [], "top": [], "total": 0.0, "err": str(e)}


PAGE = """<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Пульт живого бота — Minecraft 26.1</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#e6edf3;
     font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}
header{padding:14px 20px;background:#161b22;border-bottom:1px solid #30363d;
       display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.pill{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.on{background:#1a7f37;color:#fff}.off{background:#a40e26;color:#fff}
.ver{background:#1f6feb;color:#fff}
main{padding:18px;display:grid;gap:14px;
     grid-template-columns:repeat(auto-fit,minmax(310px,1fr));max-width:1500px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
.card h2{margin:0 0 10px;font-size:13px;color:#7d8590;text-transform:uppercase;
         letter-spacing:.5px;font-weight:600}
.big{font-size:26px;font-weight:700}
.row{display:flex;justify-content:space-between;padding:3px 0;
     border-bottom:1px solid #21262d}
.row:last-child{border:0}
.muted{color:#7d8590}
.bar{height:9px;background:#21262d;border-radius:5px;overflow:hidden;margin-top:5px}
.bar>i{display:block;height:100%}
.hp>i{background:#f85149}.fd>i{background:#d29922}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:6px}
.cell{aspect-ratio:1;background:#0d1117;border:1px solid #30363d;border-radius:6px;
      display:flex;align-items:center;justify-content:center;font-size:9px;
      text-align:center;padding:2px;word-break:break-all}
.cell.lock{opacity:.32}
.cell.full{border-color:#1f6feb;background:#0f1c33}
.ev{font-size:12px;padding:2px 0;border-bottom:1px solid #21262d}
.pos{color:#3fb950}.neg{color:#f85149}
.help{background:#0d1117;border:1px dashed #30363d;border-radius:8px;padding:12px;
      font-size:12px;color:#7d8590;white-space:pre-wrap;line-height:1.7}
code{background:#21262d;padding:1px 6px;border-radius:4px;color:#79c0ff}
</style>
<header>
  <h1>🎮 Пульт живого бота</h1>
  <span class="pill ver">Minecraft __VER__</span>
  <span id="st" class="pill off">нет связи</span>
  <span class="muted" id="upd"></span>
</header>
<main id="app"></main>
<script>
const E=(s)=>document.getElementById(s);
function bar(v,max,cls){const p=Math.max(0,Math.min(100,100*v/max));
  return `<div class="bar ${cls}"><i style="width:${p}%"></i></div>`}

async function tick(){
 let s;
 try{ s=await (await fetch('/api/s')).json(); }catch(e){ return; }
 E('st').className='pill '+(s.connected?'on':'off');
 E('st').textContent=s.connected?'мост на связи':'мост не отвечает';
 E('upd').textContent='опросов: '+s.polls+' · сбоев: '+s.fails;

 if(!s.connected){
   E('app').innerHTML=`<div class="card" style="grid-column:1/-1">
     <h2>Мост не запущен</h2>
     <div class="help">Бот ещё не подключён к игре. Чтобы увидеть его здесь:

<b>1.</b> Откройте мир в Minecraft <b>__VER__</b> и нажмите
   Esc → «Открыть для сети» → запомните порт.

<b>2.</b> Поставьте зависимости моста (один раз):
   <code>cd bot && npm install</code>

<b>3.</b> Запустите мост, подставив свой порт:
   <code>node bot/bridge.js --host 127.0.0.1 --port ВАШ_ПОРТ --mc-version __VER__</code>

<b>4.</b> Запустите мозг:
   <code>python play.py --goal 1 --steps 500</code>

Причина последней ошибки: <span class="neg">${s.error||'—'}</span></div></div>`;
   return;
 }

 const d=s.world, r=s.rewards;
 const slots=[...Array(9).keys()];
 const act=d.has_3x3?slots:[0,1,3,4];
 E('app').innerHTML=`
 <div class="card"><h2>Состояние бота</h2>
   <div class="row"><span>Здоровье</span><b>${d.health}/20</b></div>
   ${bar(d.health,20,'hp')}
   <div class="row" style="margin-top:8px"><span>Еда</span><b>${d.food}/20</b></div>
   ${bar(d.food,20,'fd')}
   <div class="row" style="margin-top:8px"><span>Позиция</span>
     <b>${(d.position||[]).map(Math.round).join(', ')}</b></div>
   <div class="row"><span>Смотрит на</span><b>${d.facing}</b></div>
 </div>

 <div class="card"><h2>В руке</h2>
   <div class="big">${d.held==='empty'?'—':d.held}</div>
   <div class="muted">уровень: ${d.tool_level_name} (${d.tool_level})</div>
   <div class="row" style="margin-top:10px"><span>Верстак рядом</span>
     <b>${d.table_reachable?'да':'нет'}</b></div>
   <div class="row"><span>Верстак открыт</span><b>${d.table_open?'да':'нет'}</b></div>
 </div>

 <div class="card"><h2>Сетка крафта ${d.has_3x3?'3×3 — верстак':'2×2 — инвентарь'}</h2>
   <div class="grid3">${slots.map(i=>{
     const on=act.includes(i), v=d.grid[i];
     const full=v&&v!=='empty';
     return `<div class="cell ${on?'':'lock'} ${full?'full':''}">${
       on?(full?v:''):'🔒'}</div>`}).join('')}</div>
 </div>

 <div class="card"><h2>Инвентарь</h2>
   ${Object.keys(d.inv).length?Object.entries(d.inv).map(([k,v])=>
     `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('')
     :'<div class="muted">пусто — всё добывает сам</div>'}
 </div>

 <div class="card"><h2>Кто рядом ${d.hostile_count?'⚠':''}</h2>
   ${Object.keys(d.mobs||{}).length?Object.entries(d.mobs).map(([k,v])=>
     `<div class="row"><span class="${k in {zombie:1,skeleton:1,spider:1,creeper:1}?'neg':'pos'}">${k}</span><b>${v}</b></div>`).join('')
     :'<div class="muted">никого не видно</div>'}
   ${d.danger?`<div class="row"><span>ближайший враг</span><b class="neg">${d.danger} бл.</b></div>`:''}
 </div>

 <div class="card"><h2>Блоки вокруг (5×5×5)</h2>
   ${Object.keys(d.around).length?Object.entries(d.around)
     .sort((a,b)=>b[1]-a[1]).map(([k,v])=>
     `<div class="row"><span>${k}</span><b>${v}</b></div>`).join('')
     :'<div class="muted">только воздух</div>'}
 </div>

 <div class="card"><h2>Награды · всего ${r.total}</h2>
   ${r.events.length?r.events.slice(0,14).map(e=>
     `<div class="ev"><span class="${e.value>=0?'pos':'neg'}">${
       e.value>=0?'+':''}${(+e.value).toFixed(2)}</span>
      <span class="muted">${e.ctx_key}</span></div>`).join('')
     :'<div class="muted">наград пока нет — запустите play.py</div>'}
 </div>`;
}
tick();setInterval(tick,1000);
</script></html>"""


class H(BaseHTTPRequestHandler):
    db_path = "data/rewards.db"

    def log_message(self, *a):        # тишина в консоли
        pass

    def do_GET(self):
        if self.path.startswith("/api/s"):
            with LOCK:
                snap = {
                    "connected": STATE["connected"],
                    "error": STATE["error"],
                    "polls": STATE["polls"],
                    "fails": STATE["fails"],
                    "world": describe(STATE["obs"]),
                    "version": TARGET_VERSION,
                }
            snap["rewards"] = read_rewards(self.db_path)
            body = json.dumps(snap).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            body = PAGE.replace("__VER__", TARGET_VERSION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--bridge-port", type=int, default=5599)
    ap.add_argument("--db", default="data/rewards.db")
    ap.add_argument("--period", type=float, default=1.0)
    a = ap.parse_args()

    H.db_path = a.db
    threading.Thread(target=poll_bridge,
                     args=(a.bridge_host, a.bridge_port, a.period),
                     daemon=True).start()
    print(f"[бот-пульт] http://0.0.0.0:{a.port} (игра {TARGET_VERSION})")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

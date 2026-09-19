#!/usr/bin/env python3
"""
Живой веб-дашборд обучения.

Показывает:
  * график награды по эпизодам и долю успехов;
  * текущую сетку крафта агента и его инвентарь;
  * содержимое базы наград: правила, счётчики, затухание;
  * поток последних начислений (за что дали / за что отняли).

Запуск:  python dashboard.py --port 7860
Только стандартная библиотека — никаких внешних веб-фреймворков.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from brain.algos.dqn import DQNConfig, DQNTrainer
from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.model import CraftBrain, ModelConfig
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import ACTIONS, ID_ITEM

STATE: Dict[str, Any] = {
    "running": False, "episode": 0, "step": 0, "goal": GOALS[0],
    "reward": 0.0, "history": [], "grid": [0] * 9, "inventory": {},
    "last_events": [], "action": "-", "epsilon": 1.0, "total_steps": 0,
    "facing": "SOUTH", "pos": [0, 0, 0], "held": "-", "successes": 0,
}
LOCK = threading.Lock()


def training_loop(db_path: str, max_steps: int, delay: float) -> None:
    db = RewardDB(db_path)
    seed(db)
    engine = RewardEngine(db)
    env = MinecraftCraftEnv(engine, max_steps=max_steps, seed=7)
    model = CraftBrain(ModelConfig(head="both"))
    trainer = DQNTrainer(model, DQNConfig())
    goal_idx, successes = 0, 0

    while True:
        obs = env.reset(goal_index=goal_idx)
        done, total, steps = False, 0.0, 0
        while not done:
            mask = env.action_mask()
            a = trainer.act(obs, mask)
            nobs, r, done, info = env.step(a)
            trainer.buffer.push(obs, a, r, nobs, done, env.action_mask())
            trainer.learn()
            obs = nobs
            total += r
            steps += 1

            with LOCK:
                STATE.update({
                    "running": True, "episode": env.episode, "step": steps,
                    "goal": GOALS[min(goal_idx, len(GOALS) - 1)],
                    "reward": round(total, 2),
                    "grid": list(env.agent.grid),
                    "inventory": {ID_ITEM[k]: v for k, v in
                                  sorted(env.agent.inventory.items()) if v},
                    "action": ACTIONS[a].name,
                    "epsilon": round(trainer.epsilon(), 3),
                    "total_steps": trainer.steps,
                    "facing": ["SOUTH", "WEST", "NORTH", "EAST"][env.agent.facing],
                    "pos": [env.agent.x, env.agent.y, env.agent.z],
                    "held": ID_ITEM.get(env.agent.held, "-"),
                    "successes": successes,
                })
                for part in info.get("breakdown", {}).get("parts", []):
                    STATE["last_events"].insert(0, {
                        "ep": env.episode, "step": steps, **part})
                STATE["last_events"] = STATE["last_events"][:80]
            if delay:
                time.sleep(delay)

        reached = "goal_reached" in info
        successes += int(reached)
        with LOCK:
            STATE["history"].append({
                "episode": env.episode, "reward": round(total, 2),
                "success": reached, "goal": GOALS[min(goal_idx, len(GOALS) - 1)]})
            STATE["history"] = STATE["history"][-400:]
            recent = [h["success"] for h in STATE["history"][-10:]]
        if len(recent) >= 10 and sum(recent) >= 7 and goal_idx < len(GOALS) - 1:
            goal_idx += 1


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Обучение Minecraft-агента</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#12131a;color:#e6e6ef;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:18px}
h1{font-size:19px;margin-bottom:4px;color:#7ee787}
.sub{color:#8b8ba0;font-size:12px;margin-bottom:16px}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:#1b1d27;border:1px solid #2a2d3a;border-radius:10px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#8b8ba0;margin-bottom:10px}
.stat{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #23262f}
.stat:last-child{border:0}
.stat b{color:#79c0ff;font-weight:600}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:6px}
.cell{aspect-ratio:1;background:#0e0f15;border:1px solid #2f3342;border-radius:6px;
display:flex;align-items:center;justify-content:center;font-size:9px;text-align:center;
padding:2px;color:#7ee787;word-break:break-all}
.cell.empty{color:#3a3d4a}
ul{list-style:none;max-height:260px;overflow-y:auto}
li{display:flex;justify-content:space-between;gap:8px;padding:3px 0;
border-bottom:1px solid #23262f;font-size:11.5px}
.pos{color:#7ee787}.neg{color:#ff7b72}
.k{color:#8b8ba0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
canvas{width:100%;height:170px;display:block}
.tag{display:inline-block;background:#252a38;border-radius:4px;padding:1px 7px;
margin:2px 3px 0 0;font-size:11px}
.dot{width:8px;height:8px;border-radius:50%;background:#7ee787;display:inline-block;
margin-right:6px;animation:p 1.4s infinite}
@keyframes p{50%{opacity:.25}}
</style></head><body>
<h1><span class="dot"></span>Обучение Minecraft-агента</h1>
<div class="sub">Пустая нейросеть учится крафтить на системе наград с базой SQLite</div>
<div class="wrap">
  <div class="card"><h2>Состояние обучения</h2><div id="stats"></div></div>
  <div class="card"><h2>Сетка крафта 3×3</h2><div class="grid3" id="grid"></div>
    <div style="margin-top:10px" id="world"></div></div>
  <div class="card"><h2>Награда по эпизодам</h2><canvas id="chart"></canvas></div>
  <div class="card"><h2>Инвентарь</h2><div id="inv"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Поток наград (последние события)</h2>
    <ul id="events"></ul></div>
  <div class="card" style="grid-column:1/-1"><h2>База наград — счётчики и затухание</h2>
    <ul id="rules"></ul></div>
</div>
<script>
const $=id=>document.getElementById(id);
function drawChart(h){
  const c=$('chart'),x=c.getContext('2d'),W=c.width=c.offsetWidth*2,H=c.height=340;
  x.clearRect(0,0,W,H); if(!h.length)return;
  const v=h.map(d=>d.reward),mn=Math.min(...v),mx=Math.max(...v),rg=(mx-mn)||1;
  x.strokeStyle='#2a2d3a';x.lineWidth=2;
  for(let i=0;i<=4;i++){const y=H*i/4;x.beginPath();x.moveTo(0,y);x.lineTo(W,y);x.stroke();}
  x.beginPath();x.strokeStyle='#7ee787';x.lineWidth=3;
  v.forEach((d,i)=>{const px=W*i/Math.max(1,v.length-1),py=H-((d-mn)/rg)*(H-30)-15;
    i?x.lineTo(px,py):x.moveTo(px,py);});
  x.stroke();
  // скользящее среднее
  const w=10,ma=v.map((_,i)=>{const s=v.slice(Math.max(0,i-w),i+1);
    return s.reduce((a,b)=>a+b,0)/s.length;});
  x.beginPath();x.strokeStyle='#79c0ff';x.lineWidth=3;
  ma.forEach((d,i)=>{const px=W*i/Math.max(1,ma.length-1),py=H-((d-mn)/rg)*(H-30)-15;
    i?x.lineTo(px,py):x.moveTo(px,py);});
  x.stroke();
  x.fillStyle='#8b8ba0';x.font='20px monospace';
  x.fillText(mx.toFixed(0),6,22);x.fillText(mn.toFixed(0),6,H-6);
}
async function tick(){
  const s=await (await fetch('/api/state')).json();
  const sr=s.history.slice(-20).filter(d=>d.success).length;
  $('stats').innerHTML=[
    ['Эпизод',s.episode],['Шаг',s.step+' / '+s.max_steps],
    ['Текущая цель',s.goal],['Награда эпизода',s.reward],
    ['Успехов (посл. 20)',sr+'/20'],['Всего успехов',s.successes],
    ['Действие',s.action],['ε (исследование)',s.epsilon],
    ['Шагов всего',s.total_steps]
  ].map(([k,v])=>`<div class="stat"><span>${k}</span><b>${v}</b></div>`).join('');
  $('grid').innerHTML=s.grid.map(g=>g==='empty'
    ?'<div class="cell empty">·</div>'
    :`<div class="cell">${g.replace(/_/g,' ')}</div>`).join('');
  $('world').innerHTML=`<div class="stat"><span>Позиция</span><b>${s.pos.join(', ')}</b></div>
    <div class="stat"><span>Взгляд</span><b>${s.facing}</b></div>
    <div class="stat"><span>В руке</span><b>${s.held}</b></div>`;
  $('inv').innerHTML=Object.entries(s.inventory).length
    ? Object.entries(s.inventory).map(([k,v])=>
        `<span class="tag">${k.replace(/_/g,' ')} ×${v}</span>`).join('')
    : '<span style="color:#8b8ba0">пусто</span>';
  $('events').innerHTML=s.last_events.map(e=>
    `<li><span class="k">эп${e.ep}·ш${e.step} ${e.key}</span>
     <b class="${e.value>=0?'pos':'neg'}">${e.value>=0?'+':''}${e.value}</b></li>`).join('');
  $('rules').innerHTML=s.rules.map(r=>
    `<li><span class="k">${r.ctx_key} — выдано ${r.times}×, следующая ${r.next}</span>
     <b class="${r.total>=0?'pos':'neg'}">${r.total>=0?'+':''}${r.total}</b></li>`).join('');
  drawChart(s.history);
}
setInterval(tick,700);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    db: RewardDB = None  # type: ignore
    max_steps: int = 250

    def log_message(self, *a) -> None:  # тише в консоли
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            with LOCK:
                payload = dict(STATE)
                payload["grid"] = [ID_ITEM.get(v, "empty") for v in payload["grid"]]
                payload["max_steps"] = self.max_steps
            rows = []
            for r in self.db.top_rules(24):
                rule_key = r["rule_key"]
                ctx = r["ctx_key"].split("|", 1)[1] if "|" in r["ctx_key"] else ""
                _, nxt = self.db.peek(rule_key, ctx)
                rows.append({"ctx_key": r["ctx_key"], "times": r["times"],
                             "total": round(r["total_value"], 2),
                             "next": round(nxt, 3)})
            payload["rules"] = rows
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--db", default="data/dashboard.db")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="пауза между шагами, чтобы обучение было видно глазом")
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=training_loop,
                         args=(args.db, args.max_steps, args.delay), daemon=True)
    t.start()
    time.sleep(1.0)

    Handler.db = RewardDB(args.db)
    Handler.max_steps = args.max_steps
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[dashboard] http://0.0.0.0:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()

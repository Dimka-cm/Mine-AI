#!/usr/bin/env python3
"""
3D-вид мира агента (изометрический воксельный рендер).

Показывает не графики, а саму игру: блоки мира, фигурку агента, куда он
смотрит, что держит в руке, сетку крафта 3x3 и инвентарь — в реальном
времени, пока агент играет.

Как это работает:
  * в фоне крутится ОЦЕНОЧНЫЙ цикл (без обучения) на текущем чекпоинте;
  * чекпоинт пишется тренировкой каждые N эпизодов, вьювер сам его
    перечитывает — значит в 3D видно, как агент умнеет по ходу обучения;
  * своя база наград (data/viewer.db), чтобы не мешать основному прогону.

Рендер написан вручную на canvas 2D (изометрия), без внешних библиотек —
работает и в песочнице без интернета.

Запуск:  python viewer3d.py --port 7862
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

from brain.env.mc_env import GOALS, MinecraftCraftEnv
from brain.model import CraftBrain, ModelConfig, batch_obs
from brain.rewards.db import RewardDB
from brain.rewards.engine import RewardEngine
from brain.rewards.rules import seed
from brain.spaces import (ACTIONS, ID_BLOCK, ID_ITEM, N_ACTIONS,
                          tool_level)

STATE: Dict[str, Any] = {
    "world": [], "agent": {"x": 6, "y": 2, "z": 6, "facing": 0, "pitch": 0},
    "grid": ["empty"] * 9, "inventory": {}, "held": "-",
    "episode": 0, "step": 0, "reward": 0.0, "action": "-",
    "goal": GOALS[0], "events": [], "crafted": [], "ckpt_age": "-",
    "has3x3": False, "active": [0, 1, 3, 4], "table_open": False,
    "table_near": False, "tool_level": 0,
    "health": 20.0, "food": 20.0, "mobs": [], "died": False,
    "last_reward": 0.0, "success": False, "speed": 1.0,
}
LOCK = threading.Lock()
CONTROL = {"delay": 0.22, "paused": False}


def world_payload(env: MinecraftCraftEnv) -> List[List[int]]:
    """Непустые блоки мира. y=0 (бедрок) пропускаем — он всё равно скрыт."""
    out: List[List[int]] = []
    w = env.world
    for x in range(w.shape[0]):
        for y in range(1, w.shape[1]):
            for z in range(w.shape[2]):
                b = int(w[x, y, z])
                if b:
                    out.append([x, y, z, b])
    return out


def agent_loop(ckpt: str, db_path: str, max_steps: int) -> None:
    torch.set_num_threads(1)          # не отнимаем CPU у обучения
    db = RewardDB(db_path)
    seed(db)
    env = MinecraftCraftEnv(RewardEngine(db), max_steps=max_steps, seed=None)
    model = CraftBrain(ModelConfig(head="both"))
    model.eval()

    ckpt_path = Path(ckpt)
    loaded_mtime = 0.0
    goal_idx = 0
    rng = np.random.default_rng()

    while True:
        # Перечитываем чекпоинт, если тренировка его обновила.
        try:
            if ckpt_path.exists():
                m = ckpt_path.stat().st_mtime
                if m > loaded_mtime:
                    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    model.eval()
                    loaded_mtime = m
        except Exception:
            pass

        # Цель берём из прогресса обучения, чтобы 3D показывал актуальный этап.
        try:
            pl = Path("data/progress.jsonl")
            if pl.exists():
                last = pl.read_text().strip().split("\n")[-1]
                goal_idx = int(json.loads(last).get("goal_idx", goal_idx))
        except Exception:
            pass

        obs = env.reset(goal_index=goal_idx)
        done = False
        total = 0.0
        with LOCK:
            STATE["world"] = world_payload(env)

        while not done:
            while CONTROL["paused"]:
                time.sleep(0.1)
            mask = env.action_mask()
            with torch.no_grad():
                q = model.q_values(batch_obs([obs], "cpu")).squeeze(0)
            q[~torch.as_tensor(mask)] = -1e9
            a = int(q.argmax().item())
            if rng.random() < 0.05:                    # немного живости
                a = int(rng.choice(np.flatnonzero(mask)))

            obs, r, done, info = env.step(a)
            total += r
            ag = env.agent
            age = time.time() - loaded_mtime if loaded_mtime else None
            with LOCK:
                STATE.update({
                    "world": world_payload(env),
                    "agent": {"x": ag.x, "y": ag.y, "z": ag.z,
                              "facing": ag.facing, "pitch": ag.pitch},
                    "grid": [ID_ITEM.get(v, "empty") for v in ag.grid],
                    "inventory": {ID_ITEM[k]: v for k, v in
                                  sorted(ag.inventory.items()) if v},
                    "held": ID_ITEM.get(ag.held, "-"),
                    "episode": env.episode, "step": env.steps,
                    "reward": round(total, 1), "last_reward": round(r, 2),
                    "action": ACTIONS[a].name,
                    "goal": GOALS[min(goal_idx, len(GOALS) - 1)],
                    "crafted": list(env.engine._crafted_this_episode),
                    "success": "goal_reached" in info,
                    "has3x3": env.has_3x3(),
                    "active": list(env.active_slots()),
                    "table_open": env.agent.table_open,
                    "table_near": env._table_reachable(),
                    "tool_level": tool_level(ID_ITEM.get(ag.held, "empty")),
                    # выживание: без этого не видно, почему эпизод оборвался
                    "health": round(ag.health, 1),
                    "food": round(ag.food, 1),
                    "died": bool(info.get("died")),
                    "mobs": [{"kind": m.kind, "x": m.x, "y": m.y, "z": m.z,
                              "hp": round(m.health, 1),
                              "hostile": m.hostile}
                             for m in env.mobs if m.health > 0],
                    "ckpt_age": (f"{age/60:.0f} мин назад" if age and age > 60
                                 else "только что" if age is not None else "нет"),
                })
                for part in info.get("breakdown", {}).get("parts", []):
                    STATE["events"].insert(0, {
                        "step": env.steps, "key": part["key"],
                        "value": part["value"], "reason": part["reason"]})
                STATE["events"] = STATE["events"][:24]
            time.sleep(CONTROL["delay"])
        time.sleep(0.7)


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D мир агента</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0b0d13;color:#e8e8f2;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;
padding:14px;overflow-x:hidden}
h1{font-size:17px;color:#7ee787;margin-bottom:2px}
.sub{color:#8b8ba0;font-size:11.5px;margin-bottom:12px}
.main{display:grid;grid-template-columns:1fr 310px;gap:12px}
@media(max-width:900px){.main{grid-template-columns:1fr}}
.stage{background:linear-gradient(180deg,#141b2e 0%,#0d1018 100%);
border:1px solid #262a38;border-radius:12px;position:relative;overflow:hidden;
min-height:430px}
canvas{display:block;width:100%;height:430px}
.hud{position:absolute;left:12px;top:12px;background:rgba(10,12,18,.82);
border:1px solid #2a2f3e;border-radius:8px;padding:9px 12px;font-size:11.5px;
backdrop-filter:blur(4px)}
.hud b{color:#79c0ff}
.badge{position:absolute;right:12px;top:12px;background:rgba(10,12,18,.82);
border:1px solid #2a2f3e;border-radius:8px;padding:9px 12px;font-size:11.5px}
.rw{position:absolute;right:12px;bottom:12px;font-size:25px;font-weight:700;
text-shadow:0 2px 10px #000}
.side{display:flex;flex-direction:column;gap:12px}
.card{background:#151823;border:1px solid #262a38;border-radius:10px;padding:12px}
.card h2{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;
color:#8b8ba0;margin-bottom:9px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.cell{aspect-ratio:1;background:#0c0e15;border:2px solid #2f3546;border-radius:7px;
display:flex;align-items:center;justify-content:center;font-size:8.5px;
text-align:center;padding:2px;color:#7ee787;line-height:1.15;word-break:break-word}
.cell.e{color:#343a4a;border-style:dashed}
.cell.f{border-color:#3fb950;background:#101c13;box-shadow:0 0 9px #3fb95033 inset}
.cell.lock{background:#0a0b10;border-color:#1c1f29;border-style:solid;color:#2e3442;
font-size:14px}
.hot{display:flex;flex-wrap:wrap;gap:5px}
.slot{background:#0c0e15;border:1px solid #2f3546;border-radius:6px;padding:4px 8px;
font-size:10.5px;color:#c9d1d9}.slot b{color:#7ee787}
.slot.held{border-color:#d29922;background:#1a1408}
ul{list-style:none;max-height:190px;overflow-y:auto}
li{display:flex;justify-content:space-between;gap:7px;padding:2.5px 0;
border-bottom:1px solid #1e2230;font-size:10.5px}
.pos{color:#7ee787}.neg{color:#ff7b72}
.k{color:#8b8ba0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row{display:flex;justify-content:space-between;padding:2.5px 0;
border-bottom:1px solid #1e2230}.row:last-child{border:0}.row b{color:#79c0ff}
button{background:#1d2130;color:#e8e8f2;border:1px solid #333a4d;border-radius:6px;
padding:5px 11px;font:inherit;font-size:11px;cursor:pointer;margin-right:5px}
button:hover{background:#262c3d;border-color:#3f4760}
button.on{background:#132a17;border-color:#2ea043;color:#7ee787}
.tag{display:inline-block;background:#1c2333;border:1px solid #2c3446;border-radius:4px;
padding:1px 6px;margin:2px 3px 0 0;font-size:10px;color:#7ee787}
</style></head><body>
<h1>3D мир агента</h1>
<div class="sub">Изометрический вид. Агент играет на текущих весах обучения —
что видно здесь, то он уже умеет.</div>
<div class="main">
 <div class="stage">
  <canvas id="cv"></canvas>
  <div class="hud" id="hud"></div>
  <div class="badge" id="badge"></div>
  <div class="rw" id="rw"></div>
 </div>
 <div class="side">
  <div class="card"><h2 id="gh">Сетка крафта</h2><div class="g3" id="grid"></div>
    <div style="margin-top:9px" id="crafted"></div></div>
  <div class="card"><h2>Инвентарь</h2><div class="hot" id="inv"></div></div>
  <div class="card"><h2>Состояние</h2><div id="st"></div>
   <div style="margin-top:9px">
    <button id="bp">⏸ Пауза</button>
    <button id="bs">🐢 Медленно</button>
    <button id="bf">⚡ Быстро</button></div></div>
  <div class="card"><h2>Награды прямо сейчас</h2><ul id="ev"></ul></div>
 </div>
</div>
<script>
const $=i=>document.getElementById(i),cv=$('cv'),cx=cv.getContext('2d');
const NAMES={0:'air',1:'stone',2:'dirt',3:'grass_block',4:'oak_log',5:'oak_planks',
6:'crafting_table',7:'furnace',8:'iron_ore',9:'diamond_ore',10:'bedrock',11:'water'};
// [верх, левая грань, правая грань]
const COL={1:['#9aa0a6','#6f757b','#585e64'],2:['#a9764a','#7d5636','#634428'],
3:['#6cc24a','#7d5636','#634428'],4:['#a9813f','#6f5228','#56401f'],
5:['#d0a15a','#9b7540','#7c5d33'],6:['#c08a4e','#7d5a33','#654928'],
7:['#9098a0','#666d75','#51575e'],8:['#c9b295','#9c8871','#7d6c59'],
9:['#6fe3d6','#3f9f95','#317f77'],10:['#4a4a52','#35353b','#2a2a2f'],
11:['#4a86e8','#3465b0','#2a518c']};
const TW=30,TH=15,BH=22;   // ширина/высота ромба, высота блока
let ROT=0;                  // поворот сцены (кнопками не управляем, но заложен)

function iso(x,y,z,ox,oy){
  return [ox+(x-z)*TW, oy+(x+z)*TH-y*BH];
}
function cube(x,y,z,c,ox,oy,alpha){
  const [sx,sy]=iso(x,y,z,ox,oy);
  cx.globalAlpha=alpha===undefined?1:alpha;
  // верх
  cx.fillStyle=c[0];cx.beginPath();
  cx.moveTo(sx,sy-TH);cx.lineTo(sx+TW,sy);cx.lineTo(sx,sy+TH);cx.lineTo(sx-TW,sy);
  cx.closePath();cx.fill();
  // левая
  cx.fillStyle=c[1];cx.beginPath();
  cx.moveTo(sx-TW,sy);cx.lineTo(sx,sy+TH);cx.lineTo(sx,sy+TH+BH);cx.lineTo(sx-TW,sy+BH);
  cx.closePath();cx.fill();
  // правая
  cx.fillStyle=c[2];cx.beginPath();
  cx.moveTo(sx+TW,sy);cx.lineTo(sx,sy+TH);cx.lineTo(sx,sy+TH+BH);cx.lineTo(sx+TW,sy+BH);
  cx.closePath();cx.fill();
  cx.globalAlpha=1;
}
function mob(m,ox,oy){
  const [sx,sy]=iso(m.x,m.y,m.z,ox,oy);
  const by=sy-BH*0.15;
  cx.fillStyle='rgba(0,0,0,.3)';cx.beginPath();
  cx.ellipse(sx,sy+TH*0.55,TW*0.5,TH*0.5,0,0,7);cx.fill();
  const skin={zombie:'#3f7d4f',skeleton:'#d6d6d6',spider:'#3a2a2a',
              creeper:'#4fc04f',cow:'#6b4a33',pig:'#e29a9a',
              sheep:'#e8e8e8',chicken:'#f0e6c8'}[m.kind]||'#888';
  const body={zombie:'#2f6b8f',skeleton:'#b9b9b9',spider:'#2a1e1e',
              creeper:'#3ea83e',cow:'#4a3222',pig:'#c97f7f',
              sheep:'#d8d8d8',chicken:'#d8cba8'}[m.kind]||'#666';
  cx.fillStyle=body;cx.fillRect(sx-7,by-22,14,18);
  cx.fillStyle=skin;cx.fillRect(sx-8,by-38,16,16);
  cx.fillStyle='#16233a';
  cx.fillRect(sx-5,by-32,3,4);cx.fillRect(sx+2,by-32,3,4);
  if(m.hostile){
    cx.fillStyle='#ff4d4d';cx.font='bold 11px system-ui';
    cx.fillText('!',sx-2,by-42);
    const w=18,hp=Math.max(0,Math.min(1,m.hp/10));
    cx.fillStyle='#3a1416';cx.fillRect(sx-w/2,by-48,w,3);
    cx.fillStyle='#ff4d4d';cx.fillRect(sx-w/2,by-48,w*hp,3);
  }
}
function steve(x,y,z,facing,ox,oy){
  const [sx,sy]=iso(x,y,z,ox,oy);
  const by=sy-BH*0.15;
  // тень
  cx.fillStyle='rgba(0,0,0,.33)';cx.beginPath();
  cx.ellipse(sx,sy+TH*0.55,TW*0.55,TH*0.55,0,0,7);cx.fill();
  // тело
  cx.fillStyle='#2f7fd6';cx.fillRect(sx-8,by-26,16,22);
  cx.fillStyle='#265f9e';cx.fillRect(sx-8,by-26,5,22);
  // руки
  cx.fillStyle='#d8a07a';cx.fillRect(sx-12,by-24,5,16);cx.fillRect(sx+7,by-24,5,16);
  // голова
  cx.fillStyle='#d8a07a';cx.fillRect(sx-9,by-44,18,18);
  cx.fillStyle='#8d5524';cx.fillRect(sx-9,by-44,18,6);
  // глаза смотрят по направлению
  cx.fillStyle='#16233a';
  const eo=[0,-3,0,3][facing]||0;
  cx.fillRect(sx-5+eo,by-35,3,4);cx.fillRect(sx+2+eo,by-35,3,4);
  // стрелка направления
  const dir=[[0,1],[-1,0],[0,-1],[1,0]][facing]||[0,1];
  const [ax,ay]=iso(x+dir[0]*0.85,y,z+dir[1]*0.85,ox,oy);
  cx.strokeStyle='#ffd33d';cx.lineWidth=3;cx.globalAlpha=.9;
  cx.beginPath();cx.moveTo(sx,sy+2);cx.lineTo(ax,ay+2);cx.stroke();
  cx.fillStyle='#ffd33d';cx.beginPath();
  cx.arc(ax,ay+2,4,0,7);cx.fill();cx.globalAlpha=1;
}
function draw(s){
  const W=cv.width=cv.offsetWidth*2, H=cv.height=430*2;
  cx.setTransform(2,0,0,2,0,0);
  cx.clearRect(0,0,W,H);
  const ox=cv.offsetWidth/2, oy=90;
  const blocks=(s.world||[]).slice();
  const a=s.agent||{x:6,y:2,z:6,facing:0};
  // painter's algorithm: дальние сначала
  blocks.sort((p,q)=>(p[0]+p[2]+p[1])-(q[0]+q[2]+q[1]));
  const akey=a.x+a.z+a.y;
  let drawn=false;
  for(const b of blocks){
    const key=b[0]+b[2]+b[1];
    if(!drawn && key>akey){ steve(a.x,a.y,a.z,a.facing,ox,oy); drawn=true; }
    const c=COL[b[3]]||COL[1];
    cube(b[0],b[1],b[2],c,ox,oy, b[1]===1?0.92:1);
  }
  if(!drawn) steve(a.x,a.y,a.z,a.facing,ox,oy);
  for(const m of (s.mobs||[])) mob(m,ox,oy);
}
async function tick(){
  const s=await (await fetch('/api/w')).json();
  draw(s);
  const F=['ЮГ','ЗАПАД','СЕВЕР','ВОСТОК'][s.agent.facing]||'?';
  $('hud').innerHTML=`эпизод <b>${s.episode}</b> · шаг <b>${s.step}</b><br>
   цель <b>${s.goal}</b><br>смотрит <b>${F}</b> · в руке <b>${s.held}</b><br>
   крафт <b style="color:${s.has3x3?'#7ee787':'#d29922'}">${s.has3x3?'3×3':'2×2'}</b>
   · уровень <b>${['рука','дерево','камень','железо','алмаз'][s.tool_level||0]}</b>
   ${s.table_near?'· верстак рядом':''}<br>
   ❤ <b style="color:${(s.health||0)>8?'#7ee787':'#ff7b72'}">${s.health}</b>/20
   · 🍗 <b style="color:${(s.food||0)>6?'#7ee787':'#d29922'}">${s.food}</b>/20
   ${(s.mobs||[]).filter(m=>m.hostile).length
      ? `· <b style="color:#ff7b72">⚠ врагов: ${(s.mobs||[]).filter(m=>m.hostile).length}</b>`
      : '· <span style="color:#7ee787">спокойно</span>'}
   ${s.died?'<br><b style="color:#ff7b72">☠ ПОГИБ</b>':''}`;
  $('badge').innerHTML=`действие<br><b>${s.action}</b>`;
  const col=s.reward>=0?'#7ee787':'#ff7b72';
  $('rw').innerHTML=`<span style="color:${col}">${s.reward>=0?'+':''}${s.reward}</span>`;
  const act=new Set(s.active||[0,1,3,4]);
 $('gh').innerHTML=s.has3x3
   ? 'Сетка крафта <span style="color:#7ee787">3×3 — верстак открыт</span>'
   : 'Сетка крафта <span style="color:#d29922">2×2 — инвентарь</span>';
 $('grid').innerHTML=s.grid.map((g,i)=>{
   if(!act.has(i)) return '<div class="cell lock" title="нужен верстак">🔒</div>';
   return g==='empty'?'<div class="cell e">·</div>'
     :`<div class="cell f">${g.replace(/_/g,'<br>')}</div>`;}).join('');
  $('crafted').innerHTML=(s.crafted||[]).length
    ?'<span style="color:#8b8ba0;font-size:10.5px">скрафтил: </span>'+
     s.crafted.map(c=>`<span class="tag">${c.replace(/_/g,' ')}</span>`).join('')
    :'<span style="color:#444b5d;font-size:10.5px">пока ничего не скрафтил</span>';
  $('inv').innerHTML=Object.entries(s.inventory).map(([k,v])=>
    `<span class="slot ${k===s.held?'held':''}">${k.replace(/_/g,' ')} <b>${v}</b></span>`
    ).join('')||'<span style="color:#444b5d">пусто</span>';
  $('st').innerHTML=[['Позиция',`${s.agent.x}, ${s.agent.y}, ${s.agent.z}`],
   ['Наклон',s.agent.pitch],['Награда за шаг',s.last_reward],
   ['Цель достигнута',s.success?'ДА 🎉':'—'],['Веса обновлены',s.ckpt_age]]
   .map(([k,v])=>`<div class="row"><span>${k}</span><b>${v}</b></div>`).join('');
  $('ev').innerHTML=(s.events||[]).map(e=>
    `<li><span class="k">ш${e.step} ${e.key}</span>
     <b class="${e.value>=0?'pos':'neg'}">${e.value>=0?'+':''}${e.value}</b></li>`
    ).join('')||'<li style="color:#444b5d">ждём…</li>';
}
$('bp').onclick=async()=>{const r=await(await fetch('/api/ctl?toggle=1')).json();
  $('bp').textContent=r.paused?'▶ Продолжить':'⏸ Пауза';
  $('bp').className=r.paused?'on':'';};
$('bs').onclick=()=>fetch('/api/ctl?delay=0.6');
$('bf').onclick=()=>fetch('/api/ctl?delay=0.05');
setInterval(tick,330);tick();
window.addEventListener('resize',()=>tick());
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api/w"):
            with LOCK:
                body = json.dumps(STATE, ensure_ascii=False).encode()
            ct = "application/json; charset=utf-8"
        elif self.path.startswith("/api/ctl"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            if "toggle" in q:
                CONTROL["paused"] = not CONTROL["paused"]
            if "delay" in q:
                CONTROL["delay"] = max(0.0, min(2.0, float(q["delay"][0])))
                CONTROL["paused"] = False
            body = json.dumps(CONTROL).encode()
            ct = "application/json"
        else:
            body = PAGE.encode()
            ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--checkpoint", default="checkpoints/long.pt")
    ap.add_argument("--db", default="data/viewer.db")
    ap.add_argument("--max-steps", type=int, default=50)
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=agent_loop,
                     args=(a.checkpoint, a.db, a.max_steps), daemon=True).start()
    print(f"[3d] http://0.0.0.0:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

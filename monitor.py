#!/usr/bin/env python3
"""
Монитор длинного прогона.

Читает data/progress.jsonl (его пишет train.py) и базу наград, показывает
живой график наград, продвижение по curriculum и какие тиры уже взяты.
Не мешает обучению: только читает файлы.

Запуск:  python monitor.py --port 7861
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from brain.env.mc_env import GOALS

PROGRESS = Path("data/progress.jsonl")
DB = Path("data/long.db")
LOG = Path("data/train_long.log")


def read_progress() -> List[Dict[str, Any]]:
    if not PROGRESS.exists():
        return []
    rows = []
    for line in PROGRESS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def read_db() -> Dict[str, Any]:
    if not DB.exists():
        return {"tools": [], "rules": [], "episodes": 0, "last_eps": [],
                "events": []}
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
        tools = [dict(r) for r in con.execute(
            """SELECT ctx_key, times, total_value FROM reward_state
               WHERE rule_key IN ('craft.tool','craft.tool_first_ever','craft.tier_up')
               ORDER BY total_value DESC LIMIT 30""")]
        eps_rows = [dict(r) for r in con.execute(
            """SELECT episode, steps, total_reward, crafted FROM episodes
               WHERE ended_ts > 0 ORDER BY episode DESC LIMIT 12""")]
        events = [dict(r) for r in con.execute(
            """SELECT episode, step, ctx_key, value, reason FROM reward_log
               ORDER BY id DESC LIMIT 40""")]
        rules = [dict(r) for r in con.execute(
            """SELECT ctx_key, times, total_value FROM reward_state
               ORDER BY ABS(total_value) DESC LIMIT 18""")]
        n = con.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
        con.close()
        return {"tools": tools, "rules": rules, "episodes": n,
                "last_eps": eps_rows, "events": events}
    except sqlite3.Error as e:
        return {"tools": [], "rules": [], "episodes": 0, "last_eps": [],
                "events": [], "error": str(e)}


def promotions() -> List[str]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        if "CURRICULUM" in line or "PATIENCE" in line:
            out.append(line.strip())
    return out[-14:]


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Прогон 15001 эпизод</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1018;color:#e8e8f2;font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;padding:20px}
h1{font-size:20px;color:#7ee787}.sub{color:#8b8ba0;font-size:12px;margin:4px 0 18px}
.wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.card{background:#181a24;border:1px solid #282b38;border-radius:10px;padding:15px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#8b8ba0;margin-bottom:11px}
.stat{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #21242e}
.stat:last-child{border:0}.stat b{color:#79c0ff}
canvas{width:100%;height:190px;display:block}
.steps{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
.step{padding:5px 10px;border-radius:6px;font-size:11.5px;border:1px solid #2c303d;
background:#13151d;color:#5a5f70}
.step.done{background:#132a17;border-color:#2ea043;color:#7ee787}
.step.cur{background:#1a2740;border-color:#388bfd;color:#79c0ff;
box-shadow:0 0 0 1px #388bfd inset}
ul{list-style:none;max-height:250px;overflow-y:auto}
li{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
border-bottom:1px solid #21242e;font-size:11.5px}
.pos{color:#7ee787}.neg{color:#ff7b72}.k{color:#8b8ba0;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:#7ee787;display:inline-block;
margin-right:7px;animation:p 1.4s infinite}@keyframes p{50%{opacity:.2}}
.big{font-size:27px;color:#7ee787;font-weight:700}
pre{font-size:11px;color:#8b8ba0;white-space:pre-wrap;max-height:180px;overflow-y:auto}
</style></head><body>
<h1><span class="dot"></span>Обучение: 15001 эпизод</h1>
<div class="sub">Дойдёт ли пустая нейросеть до алмазного меча?</div>
<div class="wrap">
 <div class="card"><h2>Прогресс</h2><div id="stats"></div></div>
 <div class="card"><h2>Лестница curriculum</h2><div class="steps" id="ladder"></div>
   <div style="margin-top:12px"><div class="big" id="pct">0%</div>
   <div style="color:#8b8ba0;font-size:11px">пройдено целей</div></div></div>
 <div class="card"><h2>Средняя награда</h2><canvas id="c1"></canvas></div>
 <div class="card"><h2>Доля успехов / epsilon</h2><canvas id="c2"></canvas></div>
 <div class="card"><h2>Последние эпизоды — что скрафтил</h2><ul id="eps"></ul></div>
 <div class="card"><h2>Поток наград (живой)</h2><ul id="events"></ul></div>
 <div class="card"><h2>Скрафченные инструменты</h2><ul id="tools"></ul></div>
 <div class="card"><h2>Топ правил базы наград</h2><ul id="rules"></ul></div>
 <div class="card" style="grid-column:1/-1"><h2>События curriculum</h2>
   <pre id="promo"></pre></div>
</div>
<script>
const $=i=>document.getElementById(i);
function chart(id,series,colors,fmt){
 const c=$(id),x=c.getContext('2d'),W=c.width=c.offsetWidth*2,H=c.height=380;
 x.clearRect(0,0,W,H);
 const all=series.flat(); if(!all.length)return;
 const mn=Math.min(...all),mx=Math.max(...all),rg=(mx-mn)||1;
 x.strokeStyle='#242833';x.lineWidth=2;
 for(let i=0;i<=4;i++){const y=H*i/4;x.beginPath();x.moveTo(0,y);x.lineTo(W,y);x.stroke();}
 series.forEach((s,si)=>{if(!s.length)return;
  x.beginPath();x.strokeStyle=colors[si];x.lineWidth=3;
  s.forEach((d,i)=>{const px=W*i/Math.max(1,s.length-1),py=H-((d-mn)/rg)*(H-34)-17;
   i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();});
 x.fillStyle='#8b8ba0';x.font='19px monospace';
 x.fillText(fmt(mx),8,24);x.fillText(fmt(mn),8,H-8);
}
async function tick(){
 const s=await (await fetch('/api/p')).json();
 const p=s.progress,last=p[p.length-1]||{};
 const gi=last.goal_idx??0;
 $('stats').innerHTML=[
  ['Эпизод',(last.ep||0)+' / 15001'],
  ['Текущая цель',last.goal||'-'],
  ['Ср. награда',last.avg_reward??'-'],
  ['Успех (окно)',((last.success_rate??0)*100).toFixed(0)+'%'],
  ['epsilon',last.eps??'-'],
  ['Всего успехов',last.total_successes??0],
  ['Прошло',Math.round((last.elapsed_s||0)/60)+' мин'],
  ['Эпизодов в БД',s.db.episodes]
 ].map(([k,v])=>`<div class="stat"><span>${k}</span><b>${v}</b></div>`).join('');
 $('ladder').innerHTML=s.goals.map((g,i)=>
  `<span class="step ${i<gi?'done':i===gi?'cur':''}">${i<gi?'✓ ':''}${g}</span>`).join('');
 $('pct').textContent=Math.round(gi/(s.goals.length-1)*100)+'%';
 chart('c1',[p.map(d=>d.avg_reward)],['#7ee787'],v=>v.toFixed(0));
 chart('c2',[p.map(d=>d.success_rate),p.map(d=>d.eps??0)],['#79c0ff','#d29922'],
   v=>v.toFixed(2));
 $('eps').innerHTML=(s.db.last_eps||[]).length?s.db.last_eps.map(e=>{
   let c={};try{(JSON.parse(e.crafted)||[]).forEach(x=>c[x]=(c[x]||0)+1)}catch(_){}
   const items=Object.entries(c).map(([k,v])=>k.replace(/_/g,' ')+'×'+v).join(', ');
   return `<li><span class="k">эп${e.episode} · ${e.steps} шагов · ${
     items||'<i style="color:#5a5f70">ничего</i>'}</span>
    <b class="${e.total_reward>=0?'pos':'neg'}">${e.total_reward>=0?'+':''}${
     e.total_reward.toFixed(0)}</b></li>`}).join('')
  :'<li style="color:#5a5f70">ждём первые эпизоды…</li>';
 $('events').innerHTML=(s.db.events||[]).length?s.db.events.map(e=>
  `<li><span class="k">эп${e.episode}·ш${e.step} ${e.ctx_key}</span>
   <b class="${e.value>=0?'pos':'neg'}">${e.value>=0?'+':''}${e.value.toFixed(2)}</b></li>`
  ).join(''):'<li style="color:#5a5f70">пока пусто</li>';
 $('tools').innerHTML=s.db.tools.length?s.db.tools.map(t=>
  `<li><span class="k">${t.ctx_key}</span><b class="pos">×${t.times}
   (${t.total_value>=0?'+':''}${t.total_value.toFixed(0)})</b></li>`).join('')
  :'<li style="color:#5a5f70">пока ни одного инструмента</li>';
 $('rules').innerHTML=s.db.rules.map(r=>
  `<li><span class="k">${r.ctx_key} ×${r.times}</span>
   <b class="${r.total_value>=0?'pos':'neg'}">${r.total_value>=0?'+':''}${r.total_value.toFixed(0)}</b></li>`).join('');
 $('promo').textContent=s.promotions.join('\\n')||'пока нет переходов';
}
setInterval(tick,3000);tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/api/p"):
            body = json.dumps({
                "progress": read_progress(), "db": read_db(),
                "promotions": promotions(), "goals": GOALS,
            }, ensure_ascii=False).encode()
            ct = "application/json; charset=utf-8"
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
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    print(f"[monitor] http://0.0.0.0:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

"""Web UI — the brain's face.

A zero-dependency, local-only interactive dashboard served straight from the
stdlib (`python3 -m brain.webui`, default http://127.0.0.1:8800):

  - live brain state (wake/sleep, affect, job queue, capture),
  - YOUR mood over time — the wellness webcam readings charted,
  - computer-usage analytics — hours active, you-vs-automation, top apps,
  - the nightly journal entries,
  - an ASK box: questions about your own activity ("which app did I use
    most?") answered from real aggregates + Recall matches + the executive
    model,
  - an ADD-TASK box: free-text tasks forwarded to the daemon's webhook
    (channel=direct), so the brain picks them up as first-class input.

Charts are hand-rolled SVG (no CDN, nothing loaded from the network — the
same local-only spine as everything else). Reads the SQLite stores read-only;
the only writes happen in the brain itself.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime
from typing import Any, Optional

_FATIGUE_RE = re.compile(r"fatigue=([0-9.]+)")
_TENSION_RE = re.compile(r"tension=([0-9.]+)")
_MOOD_RE = re.compile(r"mood=([a-z\-]+)")


def _ro_conn(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── aggregates (pure: conn in, JSON-able out) ────────────────────────────────
def wellness_series(conn, since_ts: float) -> list[dict[str, Any]]:
    """Wellness readings → [{ts, fatigue, tension, mood, summary}]."""
    rows = conn.execute(
        "SELECT ts, content FROM episodes WHERE mem_type='observation' "
        "AND tags LIKE '%app:wellness%' AND ts >= ? ORDER BY ts",
        (since_ts,)).fetchall()
    out = []
    for r in rows:
        c = r["content"] or ""
        f = _FATIGUE_RE.search(c)
        t = _TENSION_RE.search(c)
        m = _MOOD_RE.search(c)
        out.append({
            "ts": float(r["ts"]),
            "fatigue": float(f.group(1)) if f else None,
            "tension": float(t.group(1)) if t else None,
            "mood": m.group(1) if m else None,
            "summary": c.split(";")[0].replace("[wellness]", "").strip()[:160],
        })
    return out


def activity_summary(conn, since_ts: float, *, bucket_s: int = 3600,
                     now: Optional[float] = None) -> dict[str, Any]:
    """Computer-usage analytics from the observation stream (wellness rows
    excluded): hourly active/passive buckets, top apps, ~hours active
    (15-minute granularity)."""
    now = time.time() if now is None else now
    rows = conn.execute(
        "SELECT ts, tags FROM episodes WHERE mem_type='observation' "
        "AND ts >= ? AND (tags IS NULL OR tags NOT LIKE '%app:wellness%') "
        "ORDER BY ts", (since_ts,)).fetchall()
    buckets: dict[int, dict[str, int]] = {}
    quarter_hours: set[int] = set()
    apps: Counter = Counter()
    active = passive = 0
    for r in rows:
        tags = r["tags"] or ""
        ts = float(r["ts"])
        is_passive = "agency:passive" in tags
        b = int(ts // bucket_s) * bucket_s
        slot = buckets.setdefault(b, {"active": 0, "passive": 0})
        slot["passive" if is_passive else "active"] += 1
        if not is_passive:
            active += 1
            quarter_hours.add(int(ts // 900))
        else:
            passive += 1
        for t in tags.split(","):
            t = t.strip()
            if t.startswith("app:"):
                apps[t[4:]] += 1
    top_apps = [{"app": a, "n": n} for a, n in apps.most_common(8)]
    series = [{"ts": b, **v} for b, v in sorted(buckets.items())]
    return {"series": series, "top_apps": top_apps,
            "active_obs": active, "passive_obs": passive,
            "hours_active": round(len(quarter_hours) * 0.25, 2)}


def journal_entries(conn, k: int = 5) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT ts, content FROM episodes WHERE mem_type='semantic' "
        "AND tags LIKE '%journal%' ORDER BY ts DESC LIMIT ?", (k,)).fetchall()
    return [{"ts": float(r["ts"]), "content": r["content"]} for r in rows]


def build_ask_context(conn, question: str, days: float,
                      now: Optional[float] = None) -> str:
    """Ground an arbitrary question in REAL aggregates + matched observations
    — so 'which app did I use the most?' is answered from counts, not vibes."""
    now = time.time() if now is None else now
    since = now - days * 86400.0
    act = activity_summary(conn, since, now=now)
    wl = wellness_series(conn, since)
    apps_line = ", ".join(f"{a['app']} ({a['n']})" for a in act["top_apps"]) or "n/a"
    moods = [w["mood"] for w in wl if w.get("mood")]
    fatigue_vals = [w["fatigue"] for w in wl if w.get("fatigue") is not None]
    wl_line = (f"{len(wl)} self-checks; moods: {', '.join(moods[-8:]) or 'n/a'}; "
               f"avg fatigue {round(sum(fatigue_vals)/len(fatigue_vals), 2) if fatigue_vals else 'n/a'}")
    stats = (
        f"WINDOW: last {days:g} day(s).\n"
        f"USAGE STATS (from {act['active_obs'] + act['passive_obs']} screen "
        f"observations): ~{act['hours_active']}h actively used; "
        f"{act['active_obs']} user-driven vs {act['passive_obs']} autonomous "
        f"events; top apps by observations: {apps_line}.\n"
        f"WELLNESS: {wl_line}.")
    # Matched observations via Recall's ranking.
    from .recall import render_matches, search_observations

    class _Shim:  # search_observations only needs .conn
        pass
    shim = _Shim()
    shim.conn = conn
    matches = search_observations(shim, question, since_ts=since, k=18, now=now)
    return stats + "\n\nMATCHED OBSERVATIONS:\n" + render_matches(matches)


def ask(conn, llm, model: str, question: str, days: float) -> str:
    context = build_ask_context(conn, question, days)
    prompt = (
        "You are answering a question about the USER'S OWN computer activity "
        "and wellbeing, from their private records. ((you)=they acted; "
        "(auto)=the screen changed on its own.)\n\n"
        f"{context}\n\nQUESTION: {question}\n\n"
        "Answer in 1-5 sentences, first person to the user, citing the actual "
        "numbers/times when relevant. If the records genuinely can't answer "
        "it, say what's missing. Return JSON exactly like: {\"answer\": \"...\"}")
    out = llm.chat_json(model, "You are a precise personal-analytics assistant.",
                        prompt, temperature=0.2, max_tokens=1200)
    return (out.get("answer") or "").strip() or "(no answer produced)"


# ── HTTP server ──────────────────────────────────────────────────────────────
def serve(cfg, host: str = "127.0.0.1", port: int = 8800,
          webhook_url: str = "http://127.0.0.1:8765") -> None:
    import http.server

    import httpx

    from .embeddings import make_backend  # noqa: F401  (kept light; ask uses raw conn)
    from .llm import LLM
    from .status import gather_status

    llm = LLM(cfg)
    db_path = str(cfg.db_path)
    exec_model = cfg.models.get("executive", "")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence access noise
            pass

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _qs(self) -> dict:
            from urllib.parse import parse_qs, urlparse
            return {k: v[0] for k, v in
                    parse_qs(urlparse(self.path).query).items()}

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                if path == "/":
                    body = _PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/overview":
                    days = float(self._qs().get("days", 7))
                    since = time.time() - days * 86400.0
                    conn = _ro_conn(db_path)
                    try:
                        self._json({
                            "status": gather_status(cfg),
                            "wellness": wellness_series(conn, since),
                            "activity": activity_summary(conn, since),
                            "journal": journal_entries(conn, 4),
                        })
                    finally:
                        conn.close()
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                data = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                data = {}
            try:
                if path == "/api/ask":
                    q = str(data.get("question") or "").strip()
                    days = float(data.get("days") or 7)
                    if not q:
                        return self._json({"error": "empty question"}, 400)
                    conn = _ro_conn(db_path)
                    try:
                        answer = ask(conn, llm, exec_model, q, days)
                    finally:
                        conn.close()
                    self._json({"answer": answer})
                elif path == "/api/task":
                    content = str(data.get("content") or "").strip()
                    if not content:
                        return self._json({"error": "empty task"}, 400)
                    try:
                        r = httpx.post(
                            f"{webhook_url}/task?channel=direct&kind=task",
                            json={"content": content, "sender": "webui"},
                            timeout=5)
                        okk = 200 <= r.status_code < 300
                        self._json({"sent": okk,
                                    "detail": f"daemon webhook {r.status_code}"})
                    except Exception as e:  # noqa: BLE001
                        self._json({"sent": False,
                                    "detail": "daemon webhook unreachable — "
                                              "start it with --webhook-port 8765 "
                                              f"({type(e).__name__})"}, 502)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    httpd = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"🧠 brain web UI → http://{host}:{port}   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        llm.close()


# ── the page (single file, no CDN) ───────────────────────────────────────────
_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>brain</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
<style>
:root{--bg:#070a0f;--card:rgba(255,255,255,.045);--line:rgba(255,255,255,.09);
--tx:#e8edf4;--mut:#8b95a6;--cy:#22d3ee;--vi:#a78bfa;--am:#fbbf24;--rd:#fb7185;
--gr:#34d399}
*{box-sizing:border-box;margin:0}
body{background:radial-gradient(1200px 600px at 80% -10%,rgba(167,139,250,.14),transparent),
radial-gradient(900px 500px at -10% 110%,rgba(34,211,238,.10),transparent),var(--bg);
color:var(--tx);font:14px/1.5 -apple-system,Inter,Segoe UI,sans-serif;min-height:100vh;padding:22px}
.wrap{max-width:1180px;margin:0 auto}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:22px;font-weight:700;background:linear-gradient(90deg,var(--cy),var(--vi));
-webkit-background-clip:text;background-clip:text;color:transparent}
.pill{padding:3px 12px;border-radius:999px;border:1px solid var(--line);
background:var(--card);font-size:12px;color:var(--mut)}
.pill b{color:var(--tx)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{grid-column:span 6;background:var(--card);border:1px solid var(--line);
border-radius:16px;padding:16px;backdrop-filter:blur(10px)}
.card.w4{grid-column:span 4}.card.w8{grid-column:span 8}.card.w12{grid-column:span 12}
@media(max-width:900px){.card,.card.w4,.card.w8{grid-column:span 12}}
h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);margin-bottom:10px}
.big{font-size:30px;font-weight:700}
.sub{color:var(--mut);font-size:12px}
.gauges{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.g .bar{height:6px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:4px}
.g .fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--cy),var(--vi))}
.g label{font-size:11px;color:var(--mut)}
svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:14px;font-size:11px;color:var(--mut);margin-top:6px}
.legend i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.apps .row{display:flex;align-items:center;gap:10px;margin:7px 0}
.apps .name{width:130px;font-size:12px;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.apps .bar{flex:1;height:10px;border-radius:5px;background:rgba(255,255,255,.06);overflow:hidden}
.apps .fill{height:100%;background:linear-gradient(90deg,var(--vi),var(--cy));border-radius:5px}
.apps .n{width:42px;text-align:right;font-size:11px;color:var(--mut)}
.journal p{margin:8px 0;color:#c6cfdb;font-size:13px}
.journal .d{color:var(--cy);font-size:11px;letter-spacing:.06em}
textarea,input[type=text]{width:100%;background:rgba(0,0,0,.3);border:1px solid var(--line);
border-radius:10px;color:var(--tx);padding:10px 12px;font:inherit;resize:vertical}
button{background:linear-gradient(90deg,var(--cy),var(--vi));border:0;border-radius:10px;
color:#06121a;font-weight:700;padding:9px 18px;cursor:pointer;font:inherit;font-weight:700}
button:disabled{opacity:.5;cursor:wait}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.chip{font-size:11px;color:var(--cy);border:1px solid rgba(34,211,238,.35);border-radius:999px;
padding:3px 10px;cursor:pointer;background:transparent}
.ans{margin-top:10px;padding:12px;border-radius:10px;background:rgba(34,211,238,.07);
border:1px solid rgba(34,211,238,.2);font-size:13px;display:none;white-space:pre-wrap}
.note{font-size:12px;color:var(--mut);margin-top:8px}
.row2{display:flex;gap:10px;margin-top:10px;align-items:center}
.range{margin-left:auto;display:flex;gap:6px}
.range button{background:transparent;border:1px solid var(--line);color:var(--mut);
font-weight:600;padding:3px 10px;border-radius:8px;font-size:11px}
.range button.on{color:var(--cy);border-color:rgba(34,211,238,.5)}
.moodlbl{fill:#8b95a6;font-size:9px}
</style></head><body><div class="wrap">
<header>
  <h1>🧠 brain</h1>
  <span class="pill" id="state"><span class="dot" style="background:var(--mut)"></span>…</span>
  <span class="pill" id="presence">presence …</span>
  <span class="pill" id="queue">queue …</span>
  <span class="pill" id="counts">…</span>
  <span class="range"><button data-d="1">24h</button><button data-d="7" class="on">7d</button><button data-d="30">30d</button></span>
</header>
<div class="grid">
  <div class="card w4"><h2>Brain affect (now)</h2><div class="gauges" id="gauges"></div>
    <div class="note" id="affmood"></div></div>
  <div class="card w4"><h2>Computer time</h2><div class="big" id="hours">–</div>
    <div class="sub" id="agency"></div><svg id="actsvg" viewBox="0 0 320 90"></svg>
    <div class="legend"><span><i style="background:var(--cy)"></i>you</span>
    <span><i style="background:#475569"></i>automation</span></div></div>
  <div class="card w4"><h2>Top apps</h2><div class="apps" id="apps"></div></div>
  <div class="card w8"><h2>Your mood over time <span class="sub">(webcam self-checks)</span></h2>
    <svg id="moodsvg" viewBox="0 0 640 150"></svg>
    <div class="legend"><span><i style="background:var(--rd)"></i>fatigue</span>
    <span><i style="background:var(--am)"></i>tension</span></div>
    <div class="note" id="lastwl"></div></div>
  <div class="card w4 journal"><h2>Journal</h2><div id="journal" class="sub">…</div></div>
  <div class="card"><h2>Ask the brain about yourself</h2>
    <input type="text" id="q" placeholder="e.g. which app did I use the most this week?">
    <div class="chips">
      <span class="chip">Which app did I use the most?</span>
      <span class="chip">How was my mood today?</span>
      <span class="chip">How many hours was I at the computer?</span>
      <span class="chip">What ran without me?</span></div>
    <div class="row2"><button id="askbtn">Ask</button><span class="sub" id="askst"></span></div>
    <div class="ans" id="ans"></div></div>
  <div class="card"><h2>Give the brain a task</h2>
    <textarea id="task" rows="3" placeholder="e.g. observe whether anyone else uses this computer and note it in the journal"></textarea>
    <div class="row2"><button id="taskbtn">Send to brain</button><span class="sub" id="taskst"></span></div>
    <div class="note">Tasks enter the brain's input stream (channel=direct) and are processed
    in its next waking cycle — watch daemon.log.</div></div>
</div></div>
<script>
let DAYS=7;
const $=id=>document.getElementById(id);
const fmt=ts=>new Date(ts*1000).toLocaleString([],{weekday:'short',hour:'2-digit',minute:'2-digit'});
function gauge(label,v,lo,hi){const pct=Math.max(0,Math.min(1,(v-lo)/(hi-lo)))*100;
return `<div class="g"><label>${label} <b style="color:var(--tx)">${(+v).toFixed(2)}</b></label>
<div class="bar"><div class="fill" style="width:${pct}%"></div></div></div>`}
function line(svg,series,key,color,ymax){if(series.length<1)return'';
const xs=series.map(p=>p.ts),x0=Math.min(...xs),x1=Math.max(...xs)||x0+1;
const W=640,H=150,P=16;const X=t=>P+(W-2*P)*((t-x0)/Math.max(1,(x1-x0)));
const Y=v=>H-P-(H-2*P)*(v/ymax);
const pts=series.filter(p=>p[key]!=null).map(p=>`${X(p.ts).toFixed(1)},${Y(p[key]).toFixed(1)}`);
if(!pts.length)return'';
return `<polyline fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" points="${pts.join(' ')}"/>`+
 series.filter(p=>p[key]!=null).map(p=>`<circle cx="${X(p.ts).toFixed(1)}" cy="${Y(p[key]).toFixed(1)}" r="2.5" fill="${color}"/>`).join('')}
function moodLabels(series){if(!series.length)return'';
const xs=series.map(p=>p.ts),x0=Math.min(...xs),x1=Math.max(...xs)||x0+1;
const W=640,P=16;const X=t=>P+(W-2*P)*((t-x0)/Math.max(1,(x1-x0)));
return series.filter(p=>p.mood).map(p=>`<text class="moodlbl" x="${X(p.ts).toFixed(1)}" y="146" text-anchor="middle">${p.mood}</text>`).join('')}
function bars(svg,series){if(!series.length){$(svg).innerHTML='';return}
const W=320,H=90,P=4;const max=Math.max(...series.map(s=>s.active+s.passive),1);
const bw=Math.max(2,(W-2*P)/series.length-1);
$(svg).innerHTML=series.map((s,i)=>{const x=P+i*((W-2*P)/series.length);
const ha=(H-2*P)*(s.active/max),hp=(H-2*P)*(s.passive/max);
return `<rect x="${x}" y="${H-P-ha}" width="${bw}" height="${ha}" rx="1" fill="var(--cy)" opacity=".85"/>
<rect x="${x}" y="${H-P-ha-hp}" width="${bw}" height="${hp}" rx="1" fill="#475569"/>`}).join('')}
async function refresh(){try{
const r=await fetch('/api/overview?days='+DAYS);const d=await r.json();
const st=d.status||{},live=(st.daemon||{});const a=live.affect||{};
const stateColors={wake:'var(--gr)',drowsy:'var(--am)',nrem:'var(--vi)',rem:'var(--cy)'};
const s=live.state||'offline';
$('state').innerHTML=`<span class="dot" style="background:${stateColors[s]||'var(--rd)'}"></span><b>${s}</b>`;
$('presence').innerHTML='presence: <b>'+(live.user_present===true?'here':live.user_present===false?'away':'?')+'</b>';
const q=live.queue||{};$('queue').innerHTML=`queue <b>${q.depth??'–'}</b> · done <b>${q.done??0}</b>`;
const mem=(st.memory_by_type||{});$('counts').innerHTML=`obs <b>${mem.observation??'–'}</b> · facts <b>${mem.semantic??'–'}</b> · skills <b>${st.skills??'–'}</b>`;
$('gauges').innerHTML=gauge('valence',a.valence??0,-1,1)+gauge('arousal',a.arousal??0,0,1)+
 gauge('stress',a.stress??0,0,1)+gauge('fatigue',a.fatigue??0,0,1);
$('affmood').textContent=a.mood_label?('feels '+a.mood_label):'';
const act=d.activity||{series:[],top_apps:[]};
$('hours').textContent=(act.hours_active??0)+'h';
$('agency').textContent=`${act.active_obs??0} your actions · ${act.passive_obs??0} automated · last ${DAYS}d`;
bars('actsvg',act.series||[]);
const apps=act.top_apps||[];const mx=Math.max(...apps.map(x=>x.n),1);
$('apps').innerHTML=apps.map(x=>`<div class="row"><span class="name">${x.app}</span>
<div class="bar"><div class="fill" style="width:${100*x.n/mx}%"></div></div><span class="n">${x.n}</span></div>`).join('')||'<span class="sub">no data yet</span>';
const wl=d.wellness||[];
$('moodsvg').innerHTML=line('moodsvg',wl,'fatigue','var(--rd)',1)+line('moodsvg',wl,'tension','var(--am)',1)+moodLabels(wl);
const last=wl[wl.length-1];
$('lastwl').textContent=last?`latest (${fmt(last.ts)}): ${last.mood??''} — ${last.summary}`:'no self-checks in window yet';
$('journal').innerHTML=(d.journal||[]).map(j=>{const m=j.content.match(/^Journal (\d{4}-\d{2}-\d{2}):\s*(.*)$/s);
return `<p><span class="d">${m?m[1]:fmt(j.ts)}</span><br>${(m?m[2]:j.content).slice(0,360)}…</p>`}).join('')||'no entries yet';
}catch(e){$('state').innerHTML='<span class="dot" style="background:var(--rd)"></span>ui error'}}
document.querySelectorAll('.range button').forEach(b=>b.onclick=()=>{
document.querySelectorAll('.range button').forEach(x=>x.classList.remove('on'));
b.classList.add('on');DAYS=+b.dataset.d;refresh()});
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{$('q').value=c.textContent;doAsk()});
async function doAsk(){const q=$('q').value.trim();if(!q)return;
$('askbtn').disabled=true;$('askst').textContent='thinking (strong model — can take ~30s)…';$('ans').style.display='none';
try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({question:q,days:DAYS})});const d=await r.json();
$('ans').textContent=d.answer||d.error||'(no answer)';$('ans').style.display='block';}
catch(e){$('ans').textContent='error: '+e;$('ans').style.display='block'}
$('askbtn').disabled=false;$('askst').textContent=''}
$('askbtn').onclick=doAsk;$('q').addEventListener('keydown',e=>{if(e.key==='Enter')doAsk()});
$('taskbtn').onclick=async()=>{const c=$('task').value.trim();if(!c)return;
$('taskbtn').disabled=true;$('taskst').textContent='sending…';
try{const r=await fetch('/api/task',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({content:c})});const d=await r.json();
$('taskst').textContent=d.sent?'✓ delivered to the brain':'✗ '+(d.detail||'failed');
if(d.sent)$('task').value='';}catch(e){$('taskst').textContent='✗ '+e}
$('taskbtn').disabled=false};
refresh();setInterval(refresh,10000);
</script></body></html>
"""


def main() -> int:
    import argparse
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config

    ap = argparse.ArgumentParser(description="brain web UI")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from other devices on your LAN")
    ap.add_argument("--webhook", default="http://127.0.0.1:8765",
                    help="daemon webhook base URL (for the Add-Task box)")
    args = ap.parse_args()
    serve(load_config(), host=args.host, port=args.port,
          webhook_url=args.webhook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

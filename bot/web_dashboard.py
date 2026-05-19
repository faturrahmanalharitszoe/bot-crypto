"""
bot/web_dashboard.py — Real-time web monitoring dashboard
Runs Flask in a background daemon thread.
Access at: http://localhost:5050
"""

import csv
import logging
import os
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, request

from bot.logger import TRADE_LOG_PATH

logger = logging.getLogger("bot")

# ── Shared state (written by bot loop, read by Flask) ────────
_lock  = threading.Lock()
_state = {
    "running": False, "testnet": True, "loop": 0,
    "last_update": "", "usdt_balance": 0.0,
    "has_position": False,
    "positions": [],       # list of position dicts (multi-pos)
    # backward-compat single-pos fields
    "symbol": None, "entry_price": 0.0, "current_price": 0.0,
    "quantity": 0.0, "pnl_usdt": 0.0, "pnl_pct": 0.0,
    "take_profit": 0.0, "stop_loss": 0.0, "duration_min": 0.0,
    "side": "BUY", "cost_usdt": 0.0,
    "watchlist": [], "sentiment": {}, "optimizer_last_run": "Never",
}

def update_state(new_data: dict) -> None:
    with _lock:
        _state.update(new_data)
        _state["running"] = True
        _state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_state() -> dict:
    with _lock:
        return dict(_state)

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.logger.disabled = True
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)  # Suppress Flask request logs

@app.route("/")
@app.route("/bot-monitor/")
@app.route("/bot-monitor")
def dashboard():
    return Response(DASHBOARD_HTML, content_type="text/html; charset=utf-8")

@app.route("/api/status")
@app.route("/bot-monitor/api/status")
def api_status():
    return jsonify(get_state())

@app.route("/api/trades")
@app.route("/bot-monitor/api/trades")
def api_trades():
    trades = []
    # Use absolute path to avoid CWD issues in Flask threads
    abs_path = os.path.abspath(TRADE_LOG_PATH)
    
    if os.path.exists(abs_path):
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                trades = [row for row in reader if row]
            # logger.debug(f"Dashboard read {len(trades)} trades from {abs_path}")
        except Exception as e:
            logger.error(f"❌ Dashboard failed to read trades.csv at {abs_path}: {e}")
    else:
        # This is the most likely culprit!
        logger.warning(f"⚠️  Dashboard could not find trades.csv at: {abs_path}")
        
    # Reverse to show most recent first
    trades = list(reversed(trades))
    
    # Pagination
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
    except ValueError:
        page = 1
        per_page = 50
    
    start = (page - 1) * per_page
    end = start + per_page
    paginated_trades = trades[start:end]
    
    # Calculate global statistics (over all trades, not just paginated)
    total_pnl = 0.0
    wins = 0
    for t in trades:
        try:
            pnl_val = float(t.get("pnl_usdt", 0) or 0)
            total_pnl += pnl_val
            if pnl_val >= 0:
                wins += 1
        except (ValueError, TypeError):
            pass
            
    return jsonify({
        "trades": paginated_trades,
        "total": len(trades),
        "page": page,
        "per_page": per_page,
        "total_pnl": total_pnl,
        "wins": wins
    })

def start_web_server(port: int = 5050) -> None:
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port,
                               debug=False, use_reloader=False),
        daemon=True,
        name="WebDashboard",
    )
    t.start()
    logger.info(f"Web dashboard → http://localhost:{port}")


# ── Embedded Dashboard HTML ───────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚡ Scalping Bot Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#060a12;--bg2:#0c1220;--card:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.14);
  --text:#e2e8f0;--muted:#64748b;--green:#00e5a0;--red:#ff4757;
  --yellow:#fbbf24;--cyan:#38bdf8;--purple:#a78bfa;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
.wrap{max-width:1320px;margin:0 auto;padding:20px 16px}

/* Header */
header{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap}
.brand{font-size:20px;font-weight:700;letter-spacing:-.3px;display:flex;align-items:center;gap:8px}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.testnet{background:rgba(251,191,36,.15);color:var(--yellow);border:1px solid rgba(251,191,36,.3)}
.live{background:rgba(255,71,87,.15);color:var(--red);border:1px solid rgba(255,71,87,.3)}
.live-dot{display:flex;align-items:center;gap:6px;margin-left:auto;font-size:12px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,229,160,.4)}50%{opacity:.7;box-shadow:0 0 0 6px rgba(0,229,160,0)}}
.clock{font-family:var(--mono);font-size:13px;color:var(--muted)}

/* Top bar */
.topbar{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
@media(max-width:800px){.topbar{grid-template-columns:repeat(2,1fr)}}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.stat-v{font-family:var(--mono);font-size:22px;font-weight:700;margin-bottom:3px}
.stat-l{font-size:11px;color:var(--muted);font-weight:500}
.green{color:var(--green)}.red{color:var(--red)}.yellow{color:var(--yellow)}.cyan{color:var(--cyan)}.purple{color:var(--purple)}

/* Sentiment bar */
.senti-bar{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.senti-label{font-size:13px;color:var(--muted);white-space:nowrap}
.senti-gauge{flex:1;min-width:120px;max-width:260px;height:8px;border-radius:4px;background:linear-gradient(to right,#ff4757,#fbbf24,#00e5a0);position:relative}
.senti-thumb{position:absolute;top:50%;transform:translate(-50%,-50%);width:14px;height:14px;border-radius:50%;background:#fff;border:2px solid var(--bg);box-shadow:0 0 6px rgba(255,255,255,.4);transition:left .5s ease}
.senti-val{font-weight:700;font-size:14px;white-space:nowrap}
.senti-sep{width:1px;height:20px;background:var(--border)}
.senti-opt{font-size:11px;color:var(--muted);margin-left:auto}
.senti-opt span{color:var(--cyan)}

/* Main layout */
.main-grid{display:grid;grid-template-columns:1fr 300px;gap:12px;margin-bottom:14px}
@media(max-width:960px){.main-grid{grid-template-columns:1fr}}

/* Card */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;backdrop-filter:blur(10px)}
.card-title{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px}

/* Positions */
.no-pos{text-align:center;padding:40px 20px;color:var(--muted)}
.no-pos .ico{font-size:36px;margin-bottom:10px}
.pos-list{display:flex;flex-direction:column;gap:12px}
.pos-item{background:rgba(255,255,255,0.03);border:1px solid var(--border2);border-radius:10px;padding:14px}
.pos-head{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.pos-sym{font-family:var(--mono);font-size:17px;font-weight:700}
.side-badge{padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700}
.side-long{background:rgba(0,229,160,.15);color:var(--green);border:1px solid rgba(0,229,160,.3)}
.side-short{background:rgba(255,71,87,.15);color:var(--red);border:1px solid rgba(255,71,87,.3)}
.pos-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}
.pf{font-size:10px;color:var(--muted);margin-bottom:2px}
.pv{font-family:var(--mono);font-size:13px;font-weight:500}
.pnl-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.pnl-big{font-family:var(--mono);font-size:22px;font-weight:700}
.pnl-sub{font-size:11px;color:var(--muted)}
/* Price track */
.track-wrap{margin-bottom:8px}
.track-labels{display:flex;justify-content:space-between;font-size:10px;font-family:var(--mono);margin-bottom:4px}
.track{position:relative;height:5px;border-radius:3px;background:linear-gradient(to right,var(--red),rgba(255,255,255,.2),var(--green))}
.track-thumb{position:absolute;top:50%;transform:translate(-50%,-50%);width:12px;height:12px;border-radius:50%;background:#fff;border:2px solid var(--bg);box-shadow:0 0 6px rgba(255,255,255,.4);transition:left .5s}
/* Funding pill */
.fund-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:6px;font-size:10px;font-family:var(--mono);margin-top:6px}
.fund-ok{background:rgba(0,229,160,.1);color:var(--green);border:1px solid rgba(0,229,160,.25)}
.fund-warn{background:rgba(251,191,36,.1);color:var(--yellow);border:1px solid rgba(251,191,36,.3)}
.fund-bad{background:rgba(255,71,87,.1);color:var(--red);border:1px solid rgba(255,71,87,.3)}

/* Watchlist */
.wl-item{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid var(--border)}
.wl-item:last-child{border:none}
.wl-sym{font-family:var(--mono);font-size:13px;font-weight:600}
.wl-sub{font-size:9px;color:var(--muted)}
.mbadge{font-size:8px;padding:1px 4px;border-radius:3px;font-weight:700;margin-right:4px}
.mf{background:rgba(56,189,248,.15);color:var(--cyan);border:1px solid rgba(56,189,248,.3)}
.ms{background:rgba(255,255,255,.05);color:var(--muted);border:1px solid var(--border)}
.conf-bar{height:3px;border-radius:2px;margin-top:3px;transition:width .4s}
.bias-long{color:var(--green);font-size:11px;font-weight:700}
.bias-short{color:var(--red);font-size:11px;font-weight:700}
.bias-hold{color:var(--muted);font-size:11px}
.wl-rsi{font-family:var(--mono);font-size:12px;text-align:right}

/* Trades table */
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;font-size:10px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.6px;padding-bottom:8px;border-bottom:1px solid var(--border)}
td{padding:9px 6px 9px 0;border-bottom:1px solid rgba(255,255,255,.03);font-family:var(--mono)}
tr:hover td{background:rgba(255,255,255,.02)}
.tp{color:var(--green)}.sl{color:var(--red)}.trail{color:var(--cyan)}.rev{color:var(--yellow)}.mh{color:var(--muted)}
.no-trades{text-align:center;padding:30px;color:var(--muted);font-family:var(--font)}

/* Pagination Styling */
.trades-ctrls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
  gap: 12px;
}
.per-page-select {
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-family: var(--font);
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s;
}
.per-page-select:focus {
  border-color: var(--cyan);
}
.pg-btn {
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.pg-btn:hover:not(:disabled) {
  background: rgba(255,255,255,0.08);
  border-color: var(--cyan);
  color: #fff;
}
.pg-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}
.pg-info {
  font-size: 12px;
  color: var(--muted);
  font-family: var(--mono);
}

/* Footer */
.footer{display:flex;gap:16px;font-size:11px;color:var(--muted);padding-top:10px;border-top:1px solid var(--border);flex-wrap:wrap}
.footer b{color:var(--text)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">⚡ Scalping Bot</div>
    <div id="mode-badge" class="badge testnet">TESTNET</div>
    <div class="live-dot"><div class="dot"></div><span id="run-txt">Connecting...</span></div>
    <div class="clock" id="clock"></div>
  </header>

  <!-- Stats row -->
  <div class="topbar">
    <div class="stat"><div class="stat-v cyan" id="s-bal">$0.00</div><div class="stat-l">USDT Balance</div></div>
    <div class="stat"><div class="stat-v" id="s-pnl">$0.00</div><div class="stat-l">Session P&L</div></div>
    <div class="stat"><div class="stat-v" id="s-wr">—</div><div class="stat-l">Win Rate</div></div>
    <div class="stat"><div class="stat-v cyan" id="s-tc">0</div><div class="stat-l">Total Trades</div></div>
    <div class="stat"><div class="stat-v purple" id="s-pos">0</div><div class="stat-l">Open Positions</div></div>
  </div>

  <!-- Sentiment bar -->
  <div class="senti-bar">
    <span class="senti-label">Fear &amp; Greed:</span>
    <div class="senti-gauge"><div class="senti-thumb" id="sg-thumb" style="left:50%"></div></div>
    <span class="senti-val" id="sg-val">—</span>
    <span id="sg-label" style="font-size:13px;color:var(--muted)">Loading...</span>
    <div class="senti-sep"></div>
    <div class="senti-opt">ML Threshold: <span id="opt-thr">—</span> &nbsp;|&nbsp; Optimizer: <span id="opt-last">Never</span></div>
  </div>

  <!-- Main: Positions + Watchlist -->
  <div class="main-grid">
    <div class="card">
      <div class="card-title">Open Positions (<span id="pos-count">0</span>/<span id="pos-max">3</span>)</div>
      <div id="pos-area"><div class="no-pos"><div class="ico">📡</div><div>Scanning for signals...</div></div></div>
    </div>
    <div class="card">
      <div class="card-title">Watchlist Breakdown</div>
      <div id="watchlist" style="max-height:420px;overflow-y:auto">Loading...</div>
    </div>
  </div>

  <!-- Trades -->
  <div class="card" style="margin-bottom:14px">
    <div class="card-title">Recent Trades</div>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Time</th><th>Pair</th><th>Side</th><th>Invested</th>
          <th>Entry</th><th>Exit</th><th>PnL (USDT)</th><th>PnL %</th><th>Reason</th><th>Duration</th>
        </tr></thead>
        <tbody id="t-body"><tr><td colspan="10" class="no-trades">No trades yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    <span>Loop <b id="f-loop">—</b></span>
    <span>Last update <b id="f-upd">—</b></span>
    <span style="margin-left:auto">⚡ Binance Scalping Bot</span>
  </div>
</div>

<script>
// Clock
setInterval(()=>document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false}),1000);

function fmt(n,d=6){return parseFloat(n||0).toFixed(d);}
function sign(n){return n>=0?'+':'';}

// Funding pill helper
function fundPill(pos){
  if(!pos.funding) return '';
  const r=pos.funding.rate||0, m=pos.funding.minutes_until??999;
  const rp=(r*100).toFixed(4);
  const isLong=pos.side==='BUY';
  const paying=(isLong&&r>0)||((!isLong)&&r<0);
  const cls=Math.abs(r)>0.0005?'fund-bad':(m<=30&&paying?'fund-warn':'fund-ok');
  const arrow=paying?'💸':'💰';
  return `<div class="fund-pill ${cls}">${arrow} Rate: ${r>=0?'+':''}${rp}% in ${Math.round(m)}min</div>`;
}

// Render positions
function renderPositions(positions){
  const area=document.getElementById('pos-area');
  document.getElementById('pos-count').textContent=positions.length;
  if(!positions||!positions.length){
    area.innerHTML='<div class="no-pos"><div class="ico">📡</div><div>Scanning for signals...</div></div>';
    return;
  }
  area.innerHTML='<div class="pos-list">'+positions.map(p=>{
    const pnl=parseFloat(p.pnl_usdt||0);
    const pnlPct=parseFloat(p.pnl_pct||0);
    const pnlCls=pnl>=0?'green':'red';
    const sl=parseFloat(p.stop_loss||0), tp=parseFloat(p.take_profit||0), cp=parseFloat(p.current_price||0);
    const range=tp-sl; const pct=range>0?((cp-sl)/range)*100:50;
    const sideCls=p.side==='BUY'?'side-long':'side-short';
    const sideLabel=p.side==='BUY'?'LONG':'SHORT';
    const isFut=p.is_futures!==false;
    return `<div class="pos-item">
      <div class="pos-head">
        <span class="pos-sym">${p.symbol||'—'}</span>
        <span class="side-badge ${sideCls}">${sideLabel}</span>
        ${isFut?'<span class="mbadge mf">F</span>':'<span class="mbadge ms">S</span>'}
        <span style="margin-left:auto;font-size:11px;color:var(--muted)">${fmt(p.duration_min,1)} min</span>
      </div>
      <div class="pos-grid">
        <div><div class="pf">Entry</div><div class="pv">${fmt(p.entry_price)}</div></div>
        <div><div class="pf">Current</div><div class="pv">${fmt(p.current_price)}</div></div>
        <div><div class="pf">Invested</div><div class="pv cyan">${fmt(p.cost_usdt,2)} USDT</div></div>
        <div><div class="pf">TP</div><div class="pv green">${fmt(p.take_profit)}</div></div>
        <div><div class="pf">SL</div><div class="pv red">${fmt(p.stop_loss)}</div></div>
        <div><div class="pf">Qty</div><div class="pv">${p.quantity}</div></div>
      </div>
      <div class="pnl-row">
        <div>
          <div class="pnl-big ${pnlCls}">${sign(pnl)}${pnl.toFixed(4)} USDT</div>
          <div class="pnl-sub">${sign(pnlPct)}${pnlPct.toFixed(2)}% unrealized</div>
        </div>
      </div>
      <div class="track-wrap">
        <div class="track-labels">
          <span class="red">SL ${fmt(p.stop_loss)}</span>
          <span class="green">TP ${fmt(p.take_profit)}</span>
        </div>
        <div class="track"><div class="track-thumb" style="left:${Math.max(2,Math.min(98,pct))}%"></div></div>
      </div>
      ${fundPill(p)}
    </div>`;
  }).join('')+'</div>';
}

// Render watchlist
function renderWatchlist(wl){
  const el=document.getElementById('watchlist');
  if(!wl||!wl.length){el.innerHTML='<div style="color:var(--muted);font-size:13px;padding:20px 0;text-align:center">No data</div>';return;}
  el.innerHTML=wl.map(p=>{
    const mc=p.ml_class||0, conf=p.ml_confidence||0, rsi=p.rsi||50;
    let bias='<span class="bias-hold">HOLD ◆</span>';
    let barClr='var(--muted)';
    if(mc===1){bias='<span class="bias-long">LONG ▲</span>';barClr='var(--green)';}
    if(mc===2){bias='<span class="bias-short">SHORT ▼</span>';barClr='var(--red)';}
    const mBadge=p.is_futures?'<span class="mbadge mf">F</span>':'<span class="mbadge ms">S</span>';
    const rsiCls=rsi>70?'red':(rsi<30?'green':'');
    const confPct=Math.round(conf*100);
    return `<div class="wl-item">
      <div>
        ${mBadge}
        <span class="wl-sym">${p.symbol}</span>
        <div class="conf-bar" style="width:${confPct}%;background:${barClr};max-width:60px"></div>
        <div class="wl-sub">AI: ${confPct}%</div>
      </div>
      <div style="padding-left:4px">${bias}</div>
      <div class="wl-rsi ${rsiCls}">${rsi.toFixed(1)}<div style="font-size:9px;color:var(--muted)">RSI</div></div>
    </div>`;
  }).join('');
}

async function refreshStatus(){
  try{
    const d=await fetch('api/status').then(r=>r.json());
    // Header
    const mb=document.getElementById('mode-badge');
    mb.textContent=d.testnet?'TESTNET':'LIVE';
    mb.className='badge '+(d.testnet?'testnet':'live');
    document.getElementById('run-txt').textContent=d.running?'Live':'Stopped';
    document.getElementById('s-bal').textContent='$'+parseFloat(d.usdt_balance||0).toFixed(2);
    document.getElementById('f-loop').textContent=d.loop||'—';
    document.getElementById('f-upd').textContent=d.last_update||'—';
    document.getElementById('opt-last').textContent=d.optimizer_last_run||'Never';

    // Positions
    const positions=d.positions&&d.positions.length?d.positions
      :(d.has_position&&d.symbol?[d]:[]); // backward compat
    renderPositions(positions);
    document.getElementById('s-pos').textContent=positions.length;

    // Sentiment gauge
    if(d.sentiment&&d.sentiment.fng_value!=null){
      const fv=parseInt(d.sentiment.fng_value||50);
      const score=d.sentiment.score||0;
      document.getElementById('sg-val').textContent=fv+'/100';
      document.getElementById('sg-label').textContent=d.sentiment.fng_label||'';
      const lEl=document.getElementById('sg-label');
      lEl.style.color=score<-0.5?'var(--red)':(score<0?'var(--yellow)':(score>0.5?'var(--green)':'var(--cyan)'));
      document.getElementById('sg-thumb').style.left=(fv)+'%';
    }
    if(d.ml_confidence_threshold!=null)
      document.getElementById('opt-thr').textContent=(d.ml_confidence_threshold*100).toFixed(0)+'%';

    // Watchlist
    renderWatchlist(d.watchlist||[]);

  }catch(e){document.getElementById('run-txt').textContent='Offline';}
}

// Global variables for pagination
let currentTradesPage = 1;
let tradesPerPage = 50;

async function refreshTrades(){
    try{
        const response = await fetch(`api/trades?page=${currentTradesPage}&per_page=${tradesPerPage}`);
        const data = await response.json();
        const trades = data.trades;
        const total = data.total;
        const page = data.page;
        const per_page = data.per_page;
        const totalPages = Math.ceil(total / per_page);
        const totalPnl = parseFloat(data.total_pnl || 0);
        const wins = parseInt(data.wins || 0);
        const n = trades.length;
        document.getElementById('s-tc').textContent=total;
        const pe=document.getElementById('s-pnl');
        pe.textContent=(totalPnl>=0?'+':'')+totalPnl.toFixed(4)+' USDT';
        pe.className='stat-v '+(totalPnl>=0?'green':'red');
        document.getElementById('s-wr').textContent=total>0?Math.round(wins/total*100)+'%':'—';
        document.getElementById('s-wr').className='stat-v '+(total>0?(wins/total>=0.5?'green':'red'):'');

        const rMap={'TAKE_PROFIT':['Take Profit','tp'],'STOP_LOSS':['Stop Loss','sl'],
          'TRAILING_STOP':['Trailing','trail'],'REVERSAL_SIGNAL':['Reversal','rev'],
          'MAX_HOLD':['Timeout','mh']};
        const tbody=document.getElementById('t-body');
        if(!n){tbody.innerHTML='<tr><td colspan="10" class="no-trades">No trades yet — waiting for signals...</td></tr>';return;}
        tbody.innerHTML=trades.map(t=>{
            const pnl=parseFloat(t.pnl_usdt||0);
            const cls=pnl>=0?'green':'red';
            const [rLabel,rCls]=rMap[t.exit_reason]||[t.exit_reason||'—',''];
            const side=t.side||'SELL';
            const sideCls=side==='BUY'||side==='LONG'?'green':'red';
            // Format timestamp to show date and time
            const timestamp = t.timestamp||'—';
            const datePart = timestamp.split(' ')[0];
            const timePart = timestamp.split(' ')[1]||'';
            const displayTime = datePart + '<br><small>' + timePart + '</small>';
            return `<tr>
                <td style="color:var(--muted);white-space:nowrap;">${displayTime}</td>
                <td><b>${t.symbol||'—'}</b></td>
                <td class="${sideCls}">${side}</td>
                <td>${fmt(t.cost_usdt,2)}</td>
                <td>${fmt(t.entry_price)}</td>
                <td>${fmt(t.exit_price)}</td>
                <td class="${cls}">${sign(pnl)}${pnl.toFixed(4)}</td>
                <td class="${cls}">${sign(parseFloat(t.pnl_pct||0))}${fmt(t.pnl_pct,2)}%</td>
                <td class="${rCls}">${rLabel}</td>
                <td style="color:var(--muted)">${fmt(t.duration_min,1)}m</td>
            </tr>`;
        }).join('');
        
        // Add pagination controls
        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'trades-pagination';
        paginationDiv.innerHTML = `
            <div class="trades-ctrls">
                <div class="pg-info">Showing ${total > 0 ? (page - 1) * per_page + 1 : 0} to ${Math.min(page * per_page, total)} of ${total} entries</div>
                <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <label style="font-size: 11px; color: var(--muted)">Show</label>
                        <select onchange="changeTradesPerPage(this.value)" class="per-page-select">
                            <option value="10" ${per_page === 10 ? 'selected' : ''}>10</option>
                            <option value="25" ${per_page === 25 ? 'selected' : ''}>25</option>
                            <option value="50" ${per_page === 50 ? 'selected' : ''}>50</option>
                            <option value="100" ${per_page === 100 ? 'selected' : ''}>100</option>
                        </select>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <button onclick="changeTradesPage(${page - 1})" class="pg-btn" ${page <= 1 ? 'disabled' : ''}>
                            &#9664; Prev
                        </button>
                        <span class="pg-info">Page ${page} of ${totalPages || 1}</span>
                        <button onclick="changeTradesPage(${page + 1})" class="pg-btn" ${page >= totalPages ? 'disabled' : ''}>
                            Next &#9654;
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Insert pagination after the table
        const tableContainer = document.querySelector('.card:has(#t-body)');
        let existingPagination = tableContainer.querySelector('.trades-pagination');
        if (existingPagination) {
            existingPagination.replaceWith(paginationDiv);
        } else {
            tableContainer.appendChild(paginationDiv);
        }
    }catch(e){console.error('Error refreshing trades:', e);}
}

function changeTradesPage(newPage){
    currentTradesPage = newPage;
    refreshTrades();
}

function changeTradesPerPage(newVal){
    tradesPerPage = parseInt(newVal);
    currentTradesPage = 1;
    refreshTrades();
}

refreshStatus(); refreshTrades();
setInterval(refreshStatus,5000);
setInterval(refreshTrades,10000);
</script>
</body></html>"""

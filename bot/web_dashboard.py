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
from flask import Flask, Response, jsonify

from bot.logger import TRADE_LOG_PATH

logger = logging.getLogger("bot")

# ── Shared state (written by bot loop, read by Flask) ────────
_lock  = threading.Lock()
_state = {
    "running": False, "testnet": True, "loop": 0,
    "last_update": "", "usdt_balance": 0.0,
    "has_position": False, "symbol": None,
    "entry_price": 0.0, "current_price": 0.0, "quantity": 0.0,
    "pnl_usdt": 0.0, "pnl_pct": 0.0,
    "take_profit": 0.0, "stop_loss": 0.0,
    "duration_min": 0.0, "watchlist": [],
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
            
    return jsonify(list(reversed(trades))[:50])

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
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>⚡ Scalping Bot Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#080c14;--bg2:#0d1321;--card:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.14);
  --text:#e2e8f0;--muted:#64748b;--green:#00e5a0;--red:#ff4757;
  --yellow:#fbbf24;--cyan:#38bdf8;--font:'Inter',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
.container{max-width:1280px;margin:0 auto;padding:20px 16px}

/* Header */
header{display:flex;align-items:center;gap:12px;margin-bottom:24px;flex-wrap:wrap}
.brand{font-size:20px;font-weight:700;letter-spacing:-0.3px}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.badge-testnet{background:rgba(251,191,36,.15);color:var(--yellow);border:1px solid rgba(251,191,36,.3)}
.badge-live{background:rgba(255,71,87,.15);color:var(--red);border:1px solid rgba(255,71,87,.3)}
.live-dot{display:flex;align-items:center;gap:6px;margin-left:auto;font-size:12px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,229,160,.4)}50%{opacity:.8;box-shadow:0 0 0 6px rgba(0,229,160,0)}}
.clock{font-family:var(--mono);font-size:13px;color:var(--muted)}
.sentiment-bar{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,0.03);padding:8px 15px;border-radius:10px;margin-bottom:20px;font-size:13px;border:1px solid var(--border)}
.ml-badge{font-size:10px;padding:2px 6px;border-radius:4px;background:rgba(0,229,160,0.1);color:var(--green);font-weight:700;margin-left:5px}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px;backdrop-filter:blur(12px)}
.card-title{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px}

/* Stat grid */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat-val{font-family:var(--mono);font-size:26px;font-weight:600;margin-bottom:4px}
.stat-lbl{font-size:12px;color:var(--muted)}
.green{color:var(--green)} .red{color:var(--red)} .yellow{color:var(--yellow)} .cyan{color:var(--cyan)}

/* Main grid */
.main{display:grid;grid-template-columns:1fr 280px;gap:12px;margin-bottom:16px}
@media(max-width:900px){.main{grid-template-columns:1fr}}

/* Position card */
.no-pos{text-align:center;padding:40px 20px;color:var(--muted)}
.no-pos .icon{font-size:40px;margin-bottom:12px}
.pos-header{display:flex;align-items:center;gap:10px;margin-bottom:20px}
.pos-symbol{font-size:22px;font-weight:700;font-family:var(--mono)}
.pos-rows{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}
.pos-field label{font-size:11px;color:var(--muted);margin-bottom:3px;display:block}
.pos-field span{font-family:var(--mono);font-size:15px;font-weight:500}
.pnl-big{font-family:var(--mono);font-size:28px;font-weight:700;margin-bottom:4px}
.pnl-pct{font-size:13px;color:var(--muted)}

/* Progress bar SL → Price → TP */
.price-bar-wrap{margin-top:20px}
.price-bar-labels{display:flex;justify-content:space-between;font-size:11px;font-family:var(--mono);margin-bottom:6px}
.price-bar-track{position:relative;height:6px;border-radius:3px;background:linear-gradient(to right,var(--red) 0%,#888 40%,var(--green) 100%)}
.price-bar-thumb{position:absolute;top:50%;transform:translate(-50%,-50%);width:14px;height:14px;border-radius:50%;background:#fff;border:2px solid var(--bg);box-shadow:0 0 8px rgba(255,255,255,.5);transition:left .5s ease}

/* Watchlist */
.watch-list{max-height:400px;overflow-y:auto;padding-right:4px}
.watch-list::-webkit-scrollbar{width:4px}
.watch-list::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:4px}
.watch-item{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border)}
.watch-item:last-child{border:none}
.watch-sym{font-family:var(--mono);font-weight:500;font-size:14px}
.watch-chg{font-family:var(--mono);font-size:13px;font-weight:600}

/* Trades table */
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.6px;padding-bottom:10px;border-bottom:1px solid var(--border)}
td{padding:10px 8px 10px 0;border-bottom:1px solid rgba(255,255,255,.04);font-family:var(--mono)}
tr:hover td{background:rgba(255,255,255,.02)}
.reason-TP{color:var(--green)} .reason-SL{color:var(--red)} .reason-EMA{color:var(--yellow)} .reason-MAX{color:var(--muted)}
.no-trades{text-align:center;padding:30px;color:var(--muted)}

/* Status bar */
.statusbar{display:flex;gap:20px;font-size:12px;color:var(--muted);margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.statusbar span b{color:var(--text)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="brand">⚡ Scalping Bot</div>
    <div id="mode-badge" class="badge badge-testnet">TESTNET</div>
    <div class="live-dot"><div class="dot"></div><span id="running-txt">Connecting...</span></div>
    <div class="clock" id="clock"></div>
  </header>

  <!-- AI & Sentiment Bar -->
  <div class="sentiment-bar" id="sentiment-wrap">
    <span style="color:var(--muted)">Market Sentiment:</span>
    <span id="fng-label" style="font-weight:700">Loading...</span>
    <span id="fng-value" class="badge" style="background:rgba(255,255,255,0.1)">—</span>
    <div style="margin-left:auto;font-size:11px;color:var(--muted)">
      Optimizer: <span id="opt-last" style="color:var(--cyan)">Never</span>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="card"><div class="stat-val cyan" id="s-balance">$0.00</div><div class="stat-lbl">USDT Balance</div></div>
    <div class="card"><div class="stat-val" id="s-pnl">$0.00</div><div class="stat-lbl">Session P&L</div></div>
    <div class="card"><div class="stat-val" id="s-winrate">—</div><div class="stat-lbl">Win Rate</div></div>
    <div class="card"><div class="stat-val cyan" id="s-trades">0</div><div class="stat-lbl">Total Trades</div></div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="card" id="pos-card">
      <div class="card-title">Open Position</div>
      <div class="no-pos" id="no-pos"><div class="icon">📡</div><div>Scanning for signals...</div></div>
      <div id="pos-data" style="display:none">
        <div class="pos-header">
          <span class="pos-symbol" id="p-symbol">—</span>
          <span class="badge" id="p-status-badge">OPEN</span>
        </div>
        <div class="pos-rows">
          <div class="pos-field"><label>Entry Price</label><span id="p-entry">—</span></div>
          <div class="pos-field"><label>Current Price</label><span id="p-current">—</span></div>
          <div class="pos-field"><label>Invested</label><span id="p-cost" class="cyan">—</span></div>
          <div class="pos-field"><label>Quantity</label><span id="p-qty">—</span></div>
          <div class="pos-field"><label>Duration</label><span id="p-dur">—</span></div>
        </div>
        <div class="pnl-big" id="p-pnl">—</div>
        <div class="pnl-pct" id="p-pnl-pct">unrealized P&L</div>
        <div class="price-bar-wrap">
          <div class="price-bar-labels">
            <span class="red" id="p-sl">SL</span>
            <span style="color:var(--muted)">Price Position</span>
            <span class="green" id="p-tp">TP</span>
          </div>
          <div class="price-bar-track"><div class="price-bar-thumb" id="p-thumb" style="left:50%"></div></div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Watchlist</div>
      <div id="watchlist" class="watch-list"><div style="color:var(--muted);font-size:13px">Loading...</div></div>
    </div>
  </div>

  <!-- Trades -->
  <div class="card">
    <div class="card-title">Recent Trades</div>
    <div id="trades-wrap">
      <table>
        <thead><tr>
          <th>Time</th><th>Pair</th><th>Invested</th><th>Entry</th><th>Exit</th>
          <th>PnL (USDT)</th><th>PnL %</th><th>Reason</th><th>Duration</th>
        </tr></thead>
        <tbody id="trades-body"><tr><td colspan="8" class="no-trades">No trades yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="statusbar">
    <span>Loop <b id="sb-loop">—</b></span>
    <span>Last update <b id="sb-upd">—</b></span>
  </div>
</div>

<script>
function updateClock(){
  const now=new Date();
  document.getElementById('clock').textContent=now.toLocaleTimeString('en-GB',{hour12:false});
}
setInterval(updateClock,1000); updateClock();

async function refreshStatus(){
  try{
    // Use relative path so it works on localhost and /bot-monitor/
    const r=await fetch('api/status'); const d=await r.json();
    // Mode badge
    const mb=document.getElementById('mode-badge');
    if(d.testnet){mb.textContent='TESTNET';mb.className='badge badge-testnet';}
    else{mb.textContent='LIVE';mb.className='badge badge-live';}
    document.getElementById('running-txt').textContent=d.running?'Live':'Stopped';
    document.getElementById('s-balance').textContent='$'+d.usdt_balance.toFixed(2);
    document.getElementById('sb-loop').textContent=d.loop;
    document.getElementById('sb-upd').textContent=d.last_update||'—';

    // Position
    if(d.has_position){
      document.getElementById('no-pos').style.display='none';
      document.getElementById('pos-data').style.display='block';
      document.getElementById('p-symbol').textContent=d.symbol;
      document.getElementById('p-entry').textContent=d.entry_price.toFixed(6);
      document.getElementById('p-current').textContent=d.current_price.toFixed(6);
      document.getElementById('p-cost').textContent=d.cost_usdt.toFixed(2)+' USDT';
      document.getElementById('p-qty').textContent=d.quantity;
      document.getElementById('p-dur').textContent=d.duration_min.toFixed(1)+' min';
      document.getElementById('p-sl').textContent='SL: '+d.stop_loss.toFixed(6);
      document.getElementById('p-tp').textContent='TP: '+d.take_profit.toFixed(6);
      // PnL
      const pnlEl=document.getElementById('p-pnl');
      const sign=d.pnl_usdt>=0?'+':'';
      pnlEl.textContent=sign+d.pnl_usdt.toFixed(4)+' USDT';
      pnlEl.className='pnl-big '+(d.pnl_usdt>=0?'green':'red');
      document.getElementById('p-pnl-pct').textContent=sign+d.pnl_pct.toFixed(2)+'% unrealized';
      // Price bar thumb position
      const range=d.take_profit-d.stop_loss;
      const pos=range>0?((d.current_price-d.stop_loss)/range)*100:50;
      document.getElementById('p-thumb').style.left=Math.max(2,Math.min(98,pos))+'%';
    } else {
      document.getElementById('no-pos').style.display='block';
      document.getElementById('pos-data').style.display='none';
      document.getElementById('s-trades').textContent=d.trades_count||0;
    }
    
    // AI Metrics
    if(d.sentiment){
      document.getElementById('fng-label').textContent = d.sentiment.fng_label;
      document.getElementById('fng-value').textContent = d.sentiment.fng_value + '/100';
      const s = d.sentiment.score;
      document.getElementById('fng-label').style.color = s < -0.3 ? 'var(--red)' : (s > 0.3 ? 'var(--green)' : 'var(--yellow)');
    }
    if(d.optimizer_last_run){
        document.getElementById('opt-last').textContent = d.optimizer_last_run;
    }

    // Watchlist
    const wl=document.getElementById('watchlist');
    if(d.watchlist && d.watchlist.length){
      wl.innerHTML=d.watchlist.map(p=>{
        const chg = p.change_pct || 0;
        const ml = p.ml_confidence || 0;
        const color = chg >= 0 ? 'green' : 'red';
        const sign = chg >= 0 ? '+' : '';
        const mlHtml = ml > 0 ? `<span class="ml-badge">AI:${Math.round(ml*100)}%</span>` : '';
        return `
          <div class="watch-item">
            <div style="display:flex;flex-direction:column">
                <span class="watch-sym">${p.symbol}${mlHtml}</span>
            </div>
            <span class="watch-chg ${color}">${sign}${chg.toFixed(2)}%</span>
          </div>`;
      }).join('');
    }
  }catch(e){document.getElementById('running-txt').textContent='Offline';}
}

async function refreshTrades(){
  try{
    // Use relative path so it works on localhost and /bot-monitor/
    const r=await fetch('api/trades'); const trades=await r.json();
    let totalPnl=0, wins=0;
    trades.forEach(t=>{totalPnl+=parseFloat(t.pnl_usdt||0); if(parseFloat(t.pnl_usdt||0)>=0)wins++;});
    const n=trades.length;
    document.getElementById('s-trades').textContent=n;
    const pnlEl=document.getElementById('s-pnl');
    const sign=totalPnl>=0?'+':'';
    pnlEl.textContent=sign+totalPnl.toFixed(4)+' USDT';
    pnlEl.className='stat-val '+(totalPnl>=0?'green':'red');
    document.getElementById('s-winrate').textContent=n>0?Math.round(wins/n*100)+'%':'—';
    const tbody=document.getElementById('trades-body');
    if(!n){tbody.innerHTML='<tr><td colspan="9" class="no-trades">No trades yet — waiting for signals...</td></tr>';return;}
    const reasonClass={'TAKE_PROFIT':'reason-TP','STOP_LOSS':'reason-SL','EMA_REVERSAL':'reason-EMA','MAX_HOLD':'reason-MAX'};
    tbody.innerHTML=trades.map(t=>{
      const pnl=parseFloat(t.pnl_usdt||0);
      const sign=pnl>=0?'+':'';
      const cls=pnl>=0?'green':'red';
      const rKey=t.exit_reason||'';
      const rLabel={'TAKE_PROFIT':'Take Profit','STOP_LOSS':'Stop Loss','EMA_REVERSAL':'EMA Exit','MAX_HOLD':'Timeout'}[rKey]||rKey;
      
      // Safe parsing for numerical fields
      const entry = parseFloat(t.entry_price || 0).toFixed(6);
      const exit  = parseFloat(t.exit_price || 0).toFixed(6);
      const cost  = parseFloat(t.cost_usdt || 0).toFixed(2);
      const pnl_pct = parseFloat(t.pnl_pct || 0).toFixed(2);

      return `<tr>
        <td style="color:var(--muted)">${t.timestamp || '—'}</td>
        <td><b>${t.symbol || '—'}</b></td>
        <td>${cost}</td>
        <td>${entry}</td>
        <td>${exit}</td>
        <td class="${cls}">${sign}${pnl.toFixed(4)}</td>
        <td class="${cls}">${sign}${pnl_pct}%</td>
        <td class="${reasonClass[rKey]||''}">${rLabel}</td>
        <td style="color:var(--muted)">${parseFloat(t.duration_min||0).toFixed(1)} min</td>
      </tr>`;
    }).join('');
  }catch(e){}
}

refreshStatus(); refreshTrades();
setInterval(refreshStatus,5000);
setInterval(refreshTrades,10000);
</script>
</body></html>"""

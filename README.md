# 🚀 AI-Powered Binance Scalping Bot (Spot & Futures)

A high-performance, intelligent crypto scalping bot built with Python and Scikit-Learn. This bot doesn't just follow simple indicators; it uses a **RandomForest AI Engine** to analyze market conditions across multiple timeframes simultaneously, supporting both **Binance Spot** and **USDⓈ-M Futures** (Long & Short positions).

---

## 🧠 Core Intelligence: The AI Engine
Unlike standard bots, this bot uses **Machine Learning** to make decisions:
- **22 Technical Indicators:** Analyzing Trend (EMA, ADX), Momentum (RSI, MACD), Volatility (ATR, BB), and Volume Momentum.
- **Multi-Timeframe Analysis:** Synchronously monitors **1m (Micro)**, **5m (Execution)**, and **15m (Macro)** intervals.
- **Self-Optimizing:** Includes an **Auto-Optimizer** that retrains and tunes the model dynamically based on historical market performance.

### 📈 Latest Training Results
- **ROC-AUC Score:** `0.7387` (High Predictive Accuracy)
- **Top 5 Predictive Indicators:**
  1. `ema_cross_gap` (Trend Strength)
  2. `atr_norm` (Volatility/Noise Filter)
  3. `hour` (Time-of-Day Seasonality)
  4. `dist_ema_slow` (Mean Reversion)
  5. `rsi_15m` (Macro Trend Confirmation)

---

## 🛡️ Advanced Risk Management & Capital Protection
Protecting your capital is the top priority:
- **Binance USDⓈ-M Futures Integration:** Supports leveraged Long/Short positions, automatically configuring cross margin, leverage (default 5x), and size calculations.
- **Enhanced TP/SL Mechanics:** Futures positions feature optimized Take Profit (up to +10%) and trailing SL.
- **Break-Even Protection:** Moves Stop Loss to entry price after hitting a small target (+0.3%) for risk-free runs.
- **Funding Fee Avoidance:** Blocks entry when funding rate is excessively high or when settlement is imminent (within 30 minutes) to avoid funding bleed.
- **Trade Cooldown:** Prevents "whipsawing" by enforcing a rest period for a symbol after every exit.

---

## 📊 Monitoring & Web Dashboard
- **Real-Time Web Dashboard:** Access at `http://localhost:5050` (or VPS IP) featuring a premium glassmorphic dark-theme UI.
- **Server-Side Global Stats:** Track accurate total Session P&L and Win Rates computed from the entire trade log history, not just the active page.
- **Interactive Pagination:** Advanced datatable pagination with a customizable rows-per-page dropdown (10, 25, 50, or 100 entries).
- **Telegram Notifications:** Receives instant messages with P&L details and reasons for entry/exit for every transaction.
- **Historical Recalculator (`repair_trades.py`):** Dedicated utility to repair and audit corrupted `trades.csv` entries, recalculating true entry prices and real-world P&L.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Binance API Keys (Testnet or Live)

### 2. Install
```bash
git clone https://github.com/faturrahmanalharitszoe/bot-crypto.git
cd bot-crypto
pip install -r requirements.txt
```

### 3. Training the AI
Train the model with fresh market data before running:
```bash
python train_model.py
```

### 4. Configuration
Create a `.env` file based on `.env.example`:
```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TESTNET=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Configure strategy parameters in `config.py`:
- `TRADE_FUTURES`: Toggle between Spot and USDⓈ-M Futures.
- `LEVERAGE`: Choose leverage tier (e.g. 5x).
- `MAX_POSITIONS`: Maximum simultaneous open positions.
- `FUTURES_TAKE_PROFIT_PCT`: Customize target TP percentage (e.g., 10%).

### 5. Running the Bot
For local testing:
```bash
python main.py
```

To run in background mode on a production VPS using PM2:
```bash
pm2 start main.py --name bot_crypto --interpreter python3
```

---

## ⚙️ Strategy Parameters (`config.py`)
| Parameter | Default | Description |
|---|---|---|
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | Min AI confidence to allow a trade entry |
| `TRAILING_STOP_ENABLED` | `True` | Protect profits dynamically |
| `BREAK_EVEN_ENABLED` | `True` | Move SL to entry after +0.3% |
| `TRADE_FUTURES` | `True` | Enable USDⓈ-M Futures mode |
| `LEVERAGE` | `5` | Leverage multiplier for futures trades |
| `FUTURES_TAKE_PROFIT_PCT`| `0.10` | 10% Take Profit for futures |

---

## ⚠️ Disclaimer
Trading cryptocurrencies involves significant risk. This bot is for **educational purposes only**. The author is not responsible for any financial losses. Always test thoroughly on **Testnet** before using real funds.

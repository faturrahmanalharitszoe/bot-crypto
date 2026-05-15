# 🚀 AI-Powered Binance Scalping Bot (Multi-Timeframe)

A high-performance, intelligent crypto scalping bot built with Python and Scikit-Learn. This bot doesn't just follow simple indicators; it uses a **RandomForest AI Engine** to analyze market conditions across multiple timeframes simultaneously.

---

## 🧠 Core Intelligence: The AI Engine
Unlike standard bots, this bot uses **Machine Learning** to make decisions:
- **22 Technical Indicators:** Analyzing Trend (EMA, ADX), Momentum (RSI, MACD), Volatility (ATR, BB), and Volume Momentum.
- **Multi-Timeframe Analysis:** Synchronously monitors **1m (Micro)**, **5m (Execution)**, and **15m (Macro)** intervals.
- **Self-Optimizing:** Includes an **Auto-Optimizer** that retrains and tunes the model every hour based on the last 24h of market performance.

---

## 🛡️ Advanced Risk Management
Protecting your capital is the top priority:
- **Trailing Stop Loss:** Automatically locks in profits by trailing the peak price by a specific percentage.
- **Break-Even Protection:** Once a small profit target is hit (+0.3%), the SL is moved to the entry price to ensure a risk-free trade.
- **Trade Cooldown:** Prevents "whipsawing" by enforcing a 15-minute rest period for a symbol after every exit.
- **Dynamic Exit Logic:** Combines AI signals, EMA reversals, and hard TP/SL caps for the best possible exit.

---

## 📊 Monitoring & Dashboard
- **Web Dashboard:** Real-time monitoring of your balance, active positions, ML scores, and market watchlist via `http://localhost:5050`.
- **Telegram Notifications:** Get instant alerts on your phone for every BUY and SELL operation.
- **Granular Logging:** 
    - `logs/trades.csv`: Complete history of all closed trades.
    - `logs/position_history.csv`: Per-minute "journey" data of every open trade for post-trade analysis.

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
Before running, train the model using current market data:
```bash
python train_model.py
```

### 4. Configuration
Edit `.env` for API keys and `config.py` for strategy tuning:
```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TESTNET=true
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### 5. Start the Bot
```bash
python main.py
```

---

## ⚙️ Strategy Parameters (`config.py`)
| Parameter | Default | Description |
|---|---|---|
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | Min AI confidence to allow a BUY |
| `TRAILING_STOP_ENABLED` | `True` | Protect profits dynamically |
| `BREAK_EVEN_ENABLED` | `True` | Move SL to entry after +0.3% |
| `TOP_PAIRS_COUNT` | `20` | Number of high-volume pairs to scan |
| `OPTIMIZER_INTERVAL_HOURS`| `1` | How often the AI retrains itself |

---

## ⚠️ Disclaimer
Trading cryptocurrencies involves significant risk. This bot is for **educational purposes only**. The author is not responsible for any financial losses. Always test thoroughly on **Testnet** before using real funds.

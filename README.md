# Binance Scalping Bot 🤖

A fully automatic crypto scalping bot built with Python and the Binance API.
Runs in **Testnet (paper trading) mode** to safely validate the strategy before going live.

---

## Strategy

**EMA 9/21 Crossover + RSI 14 + Volume Spike** on 5-minute candles

| Signal | Condition |
|---|---|
| 🟢 **BUY**  | EMA9 crosses above EMA21 **AND** RSI < 65 **AND** volume > 1.5× avg |
| 🔴 **SELL (TP)** | Price rises +1.5% from entry |
| 🛑 **SELL (SL)** | Price drops -0.8% from entry |
| 🔁 **SELL (Reversal)** | EMA9 crosses back below EMA21 |
| ⏰ **SELL (Timeout)** | Position held for 48 candles (~4 hours) |

**Risk/Reward ratio: ~1.87:1**  
**Max position size: 40% of available USDT balance**  
**One trade at a time** to protect small capital

---

## Setup Guide

### 1. Prerequisites

- Python 3.10 or newer
- A Binance account

### 2. Install Dependencies

```bash
cd bot-crypto
pip install -r requirements.txt
```

### 3. Get Testnet API Keys

1. Go to [https://testnet.binance.vision](https://testnet.binance.vision)
2. Log in with your GitHub account
3. Click **"Generate HMAC_SHA256 Key"**
4. Copy your **API Key** and **Secret Key**

> ⚠️ Testnet keys are different from your real Binance keys!

### 4. Configure Environment

```bash
# Copy the template
copy .env.example .env
```

Edit `.env`:
```env
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here
TESTNET=true
```

### 5. Run the Bot

```bash
python main.py
```

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CANDLE_INTERVAL` | `5m` | Scalping timeframe |
| `EMA_FAST` | `9` | Fast EMA period |
| `EMA_SLOW` | `21` | Slow EMA period |
| `RSI_PERIOD` | `14` | RSI lookback |
| `RSI_OVERBOUGHT` | `65` | Max RSI to allow BUY |
| `VOLUME_SPIKE_MULTIPLIER` | `1.5` | Volume spike threshold |
| `MAX_POSITION_PCT` | `0.40` | Max 40% of balance per trade |
| `TAKE_PROFIT_PCT` | `0.015` | +1.5% take profit |
| `STOP_LOSS_PCT` | `0.008` | -0.8% stop loss |
| `TOP_PAIRS_COUNT` | `5` | Number of pairs to watch |
| `MIN_VOLUME_USDT_24H` | `5,000,000` | Minimum 24h liquidity |
| `LOOP_INTERVAL_SECONDS` | `30` | Main loop sleep time |

---

## Output

### Console Status Banner (every 30 seconds)
```
══════════════════════════════════════════════════════════════
  ⚡ BINANCE SCALPING BOT  🧪 TESTNET  │  Loop #47
══════════════════════════════════════════════════════════════
  ⏰ Time       : 2026-05-11 15:45:00
  💰 USDT Bal   : $24.87
  📈 Position   : DOGEUSDT
     Entry      : 0.123400
     Current    : 0.124800
     PnL        : +0.1400 USDT  (+1.13%)
     TP / SL    : 0.125271 / 0.122413
     Held       : 12.5 min
══════════════════════════════════════════════════════════════
```

### Trade Journal (`logs/trades.csv`)
All trades are automatically saved with:
- Symbol, entry/exit price, quantity
- P&L in USDT and percentage
- Exit reason (TAKE_PROFIT / STOP_LOSS / EMA_REVERSAL / MAX_HOLD)
- Trade duration in minutes

---

## File Structure

```
bot-crypto/
├── .env                  # Your API keys (git-ignored)
├── .env.example          # Key template
├── .gitignore
├── requirements.txt
├── config.py             # All tunable parameters
├── main.py               # Entry point
├── bot/
│   ├── exchange.py       # Binance API wrapper
│   ├── strategy.py       # EMA + RSI + Volume signals
│   ├── risk_manager.py   # Position sizing & TP/SL logic
│   ├── pair_selector.py  # Dynamic pair ranking
│   ├── trader.py         # Trade orchestration
│   └── logger.py         # Console + CSV logging
└── logs/
    ├── bot.log           # Full debug log
    └── trades.csv        # Trade history
```

---

## Switching to Live Trading

When you're confident in the strategy results on testnet:

1. Get real API keys from [Binance](https://www.binance.com/en/my/settings/api-management)
2. Enable **Spot Trading** permission on the key (disable withdrawal!)
3. Update `.env`:
   ```env
   BINANCE_API_KEY=your_REAL_api_key
   BINANCE_API_SECRET=your_REAL_api_secret
   TESTNET=false
   ```
4. Start with `$25` and monitor closely

---

## ⚠️ Risk Disclaimer

Trading cryptocurrencies involves significant risk of financial loss.
This bot is provided for **educational purposes only**. Past performance on testnet
does not guarantee future results on live markets. Never trade money you cannot
afford to lose. The author is not responsible for any financial losses.

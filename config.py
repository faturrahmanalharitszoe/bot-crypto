"""
config.py — Central configuration for Binance Scalping Bot
All tunable parameters live here. Secrets are loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Enable programmatic DNS bypass for regions where Binance is blocked
from bot import dns_bypass

# ════════════════════════════════════════════════════════════
#  Binance API Credentials
# ════════════════════════════════════════════════════════════
API_KEY: str = os.getenv("BINANCE_API_KEY", "")
API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
TESTNET: bool = os.getenv("TESTNET", "true").lower() == "true"

# ── Futures Settings ──────────────────────────────────────
FUTURES_ENABLED: bool = True       # ⚠️ ENABLE FUTURES (USDT-M)
#FUTURES_LEVERAGE: int = 10          # Default leverage (1x to 20x recommended)

FUTURES_LEVERAGE: int = 10
FUTURES_MARGIN_TYPE: str = "ISOLATED" # ISOLATED or CROSS

# ════════════════════════════════════════════════════════════
#  Candle / Timeframe Settings
# ════════════════════════════════════════════════════════════
CANDLE_INTERVAL: str = "15m"      # Main candle interval for day trading
LOWER_INTERVAL: str = "5m"        # Lower interval for trigger confirmation
HIGHER_INTERVAL: str = "1h"       # Higher interval for trend bias confirmation
CANDLE_LIMIT: int = 150           # Historical candles to fetch per request

# ════════════════════════════════════════════════════════════
#  Technical Indicator Parameters
# ════════════════════════════════════════════════════════════
EMA_FAST: int = 9                 # Fast EMA period
EMA_SLOW: int = 21                # Slow EMA period
RSI_PERIOD: int = 14              # RSI lookback period
RSI_OVERBOUGHT: float = 70.0      # Max RSI to allow a BUY signal (was 65, relaxed for scalping)
RSI_OVERSOLD: float = 35.0        # Min RSI for SELL signal (was 30, tighter for shorts)
VOLUME_SPIKE_MULTIPLIER: float = 1.1   # Volume must be > 1.1x rolling average (was 1.5, relaxed)
VOLUME_AVG_PERIOD: int = 20       # Rolling window for volume average

# ════════════════════════════════════════════════════════════
#  Risk Management
# ════════════════════════════════════════════════════════════
MAX_POSITION_PCT: float = 4.00    # Use at most 40% of available USDT per trade

#MAX_POSITION_PCT: float = 0.40

# ── Spot TP / SL ─────────────────────────────────────────────
TAKE_PROFIT_PCT: float = 0.015   # Spot: Close trade at +1.5% profit
STOP_LOSS_PCT: float = 0.008      # Spot: Close trade at -0.8% loss

# ── Futures TP / SL (higher because price swings are larger on perp) ─
FUTURES_TAKE_PROFIT_PCT: float = 0.020   # Futures: Close at +2.0% price move (10% profit @ 5x leverage)
FUTURES_STOP_LOSS_PCT: float = 0.008     # Futures: Close at -0.8% price move (4% loss @ 5x leverage)

MAX_HOLD_CANDLES: int = 48        # Force-exit after 48 candles (~4 hrs on 5m)

# ── Spot Trailing Stop Settings ──────────────────────────────
TRAILING_STOP_ENABLED: bool = True
TRAILING_STOP_ACTIVATION_PCT: float = 0.006  # Spot: Activate after +0.6% profit
TRAILING_STOP_CALLBACK_PCT: float = 0.003    # Spot: Trail by 0.3% from the peak

# ── Futures Trailing Stop Settings ───────────────────────────
FUTURES_TRAILING_STOP_ENABLED: bool = True
FUTURES_TRAILING_STOP_ACTIVATION_PCT: float = 0.008  # Futures: Activate after +0.8% price move (8% profit @ 10x leverage)
FUTURES_TRAILING_STOP_CALLBACK_PCT: float = 0.004    # Futures: Trail by 0.4% from the peak

# ── Spot Break-Even Settings ─────────────────────────────────
BREAK_EVEN_ENABLED: bool = True
BREAK_EVEN_ACTIVATION_PCT: float = 0.003     # Spot: Move SL to entry after +0.3%

# ── Futures Break-Even Settings ──────────────────────────────
FUTURES_BREAK_EVEN_ENABLED: bool = True
FUTURES_BREAK_EVEN_ACTIVATION_PCT: float = 0.005     # Futures: Move SL to entry after +0.5% price move

# ════════════════════════════════════════════════════════════
#  Pair Selection
# ════════════════════════════════════════════════════════════
TOP_PAIRS_COUNT: int = 15          # Number of top-scored pairs to watch
MIN_VOLUME_USDT_24H: float = 5_000_000   # Minimum 24h quote volume in USDT
PAIR_REFRESH_LOOPS: int = 60      # Re-rank pairs every N loops (~30 min)

# Tokens/patterns to exclude from trading
EXCLUDE_KEYWORDS: list = [
    "UP", "DOWN", "BULL", "BEAR",       # Leveraged tokens (e.g. BTCUP)
    "3L", "3S", "5L", "5S",             # 3x/5x leveraged tokens
    "BUSD", "USDC", "TUSD", "USDP",     # Stablecoins vs USDT
    "DAI", "FDUSD", "AEUR", "BVND",     # More stablecoins
]

# ════════════════════════════════════════════════════════════
#  Bot Loop
# ════════════════════════════════════════════════════════════
LOOP_INTERVAL_SECONDS: int = 30   # Sleep between each main loop iteration when scanning only
LOOP_INTERVAL_FAST: int = 5       # Sleep between each main loop iteration when holding positions (faster TP/SL check)

# ════════════════════════════════════════════════════════════
#  Telegram Notifications (optional)
# ════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# ════════════════════════════════════════════════════════════
#  Web Dashboard
# ════════════════════════════════════════════════════════════
WEB_DASHBOARD_ENABLED: bool = os.getenv("WEB_DASHBOARD", "true").lower() == "true"
WEB_PORT: int = int(os.getenv("WEB_PORT", "5050"))

# ════════════════════════════════════════════════════════════
#  Machine Learning (Strategy Signal Confirmation Mode)
# ════════════════════════════════════════════════════════════
ML_ENABLED: bool = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_MODEL_PATH: str = "models/dl_model.pth"       # Path to trained Deep Learning model
ML_SCALER_PATH: str = "models/scaler.pkl"        # Path to fitted feature scaler
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.55"))
# In Confirmation Mode, strategy (EMA/RSI) acts as primary trigger, and ML confirms direction.
# Since HOLD (0) is filtered from training, ML does not discover new trades.
ML_FULL_AI_MODE: bool = True

# ── ML Exit Settings ─────────────────────────────────────────
ML_EXIT_ENABLED: bool = True               # Enable ML-based exit decisions
ML_REVERSAL_EXIT_ACTIVE: bool = True       # Exit if ML predicts opposite trend (LONG -> SHORT, SHORT -> LONG)
ML_EXHAUSTION_EXIT_ACTIVE: bool = False    # Exit if ML predicts HOLD (momentum exhausted)
TECH_REVERSAL_EXIT_ENABLED: bool = False   # If False, traditional EMA crossover reversal exits are disabled

# ════════════════════════════════════════════════════════════
#  Sentiment Analysis (Fear & Greed Index — No API Key needed)
# ════════════════════════════════════════════════════════════
SENTIMENT_ENABLED: bool = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
SENTIMENT_BLOCK_THRESHOLD: float = -0.9   # Block BUY only in Extreme Fear (score=-1.0)
                                           # Fear (score=-0.5) is allowed — SHORT can profit
SENTIMENT_REFRESH_MINUTES: int = 360      # Refresh every 6 hours (F&G updates daily)

# ════════════════════════════════════════════════════════════
#  Auto-Optimizer
# ════════════════════════════════════════════════════════════
OPTIMIZER_ENABLED: bool = True
OPTIMIZER_INTERVAL_HOURS: int = 1         # Re-optimize every 1 hour

# ════════════════════════════════════════════════════════════
#  Funding Fee Management (Futures only)
# ════════════════════════════════════════════════════════════
# Binance charges funding fee every 8 hours (00:00, 08:00, 16:00 UTC)
# Formula: Funding Fee = Qty × Mark Price × Funding Rate
FUNDING_FEE_ENABLED: bool = True
FUNDING_FEE_BLOCK_MINUTES: int = 30      # Block NEW entry if funding < 30 min away
                                          # AND rate is unfavorable for our direction
FUNDING_FEE_MAX_RATE: float = 0.0005     # Block entry if |funding rate| > 0.05%
                                          # (extreme rates = market imbalance, risky)

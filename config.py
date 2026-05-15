"""
config.py — Central configuration for Binance Scalping Bot
All tunable parameters live here. Secrets are loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════
#  Binance API Credentials
# ════════════════════════════════════════════════════════════
API_KEY: str = os.getenv("BINANCE_API_KEY", "")
API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
TESTNET: bool = os.getenv("TESTNET", "true").lower() == "true"

# ════════════════════════════════════════════════════════════
#  Candle / Timeframe Settings
# ════════════════════════════════════════════════════════════
CANDLE_INTERVAL: str = "5m"       # 5-minute candles for scalping
CANDLE_LIMIT: int = 150           # Historical candles to fetch per request

# ════════════════════════════════════════════════════════════
#  Technical Indicator Parameters
# ════════════════════════════════════════════════════════════
EMA_FAST: int = 9                 # Fast EMA period
EMA_SLOW: int = 21                # Slow EMA period
RSI_PERIOD: int = 14              # RSI lookback period
RSI_OVERBOUGHT: float = 65.0      # Max RSI to allow a BUY signal
RSI_OVERSOLD: float = 30.0        # RSI floor (informational)
VOLUME_SPIKE_MULTIPLIER: float = 1.5   # Volume must be > 1.5x rolling average
VOLUME_AVG_PERIOD: int = 20       # Rolling window for volume average

# ════════════════════════════════════════════════════════════
#  Risk Management
# ════════════════════════════════════════════════════════════
MAX_POSITION_PCT: float = 0.40    # Use at most 40% of available USDT per trade
TAKE_PROFIT_PCT: float = 0.015    # Close trade at +1.5% profit
STOP_LOSS_PCT: float = 0.008      # Close trade at -0.8% loss
MAX_HOLD_CANDLES: int = 48        # Force-exit after 48 candles (~4 hrs on 5m)

# ── Trailing Stop Settings ───────────────────────────────────
TRAILING_STOP_ENABLED: bool = True
TRAILING_STOP_ACTIVATION_PCT: float = 0.006  # Activate after +0.6% profit
TRAILING_STOP_CALLBACK_PCT: float = 0.003    # Trail by 0.3% from the peak

# ── Break-Even Settings ──────────────────────────────────────
BREAK_EVEN_ENABLED: bool = True
BREAK_EVEN_ACTIVATION_PCT: float = 0.003     # Move SL to entry after +0.3%

# ════════════════════════════════════════════════════════════
#  Pair Selection
# ════════════════════════════════════════════════════════════
TOP_PAIRS_COUNT: int = 20          # Number of top-scored pairs to watch
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
LOOP_INTERVAL_SECONDS: int = 30   # Sleep between each main loop iteration

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
#  Machine Learning (Full AI Mode)
# ════════════════════════════════════════════════════════════
ML_ENABLED: bool = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_MODEL_PATH: str = "models/rf_model.pkl"       # Path to trained RandomForest model
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.60"))
# In Full AI Mode: ML is the primary trigger, EMA/RSI is the confirmation filter
ML_FULL_AI_MODE: bool = True

# ════════════════════════════════════════════════════════════
#  Sentiment Analysis (Fear & Greed Index — No API Key needed)
# ════════════════════════════════════════════════════════════
SENTIMENT_ENABLED: bool = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
SENTIMENT_BLOCK_THRESHOLD: float = -0.3   # Block BUY if score < this (Fear zone)
SENTIMENT_REFRESH_MINUTES: int = 360      # Refresh every 6 hours (F&G updates daily)

# ════════════════════════════════════════════════════════════
#  Auto-Optimizer
# ════════════════════════════════════════════════════════════
OPTIMIZER_ENABLED: bool = True
OPTIMIZER_INTERVAL_HOURS: int = 1         # Re-optimize every 1 hour

"""
config.py — Central configuration for Binance Scalping Bot
All tunable parameters live here. Secrets are loaded from .env

[REVAMP v2] Major changes to fix low winrate:
  - Fixed MAX_POSITION_PCT from 4.00 (bug!) to 0.30 (30%)
  - Raised ML_CONFIDENCE_THRESHOLD from 0.55 to 0.68
  - Disabled ML_REVERSAL_EXIT (was killing winners prematurely)
  - Widened trailing stop to let winners run
  - Added ADX minimum filter for trend confirmation
  - Tightened pair selection to favor trending (not just volatile) pairs
  - Reduced leverage from 10x to 5x for better risk management
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
MOCK_TESTNET_BALANCE: float = 27.51

# ── Futures Settings ──────────────────────────────────────
FUTURES_ENABLED: bool = True
FUTURES_LEVERAGE: int = 5              # [REVAMP] Reduced from 10x to 5x — less risk per trade
FUTURES_MARGIN_TYPE: str = "ISOLATED"

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
RSI_OVERBOUGHT: float = 68.0     # [REVAMP] Tightened from 70 — avoid buying near tops
RSI_OVERSOLD: float = 32.0       # [REVAMP] Tightened from 35 — avoid shorting near bottoms
VOLUME_SPIKE_MULTIPLIER: float = 1.2   # [REVAMP] Moderate volume confirmation (1.2x avg)
VOLUME_AVG_PERIOD: int = 20

# ── ADX Trend Strength Filter ────────────────────────────────
# [REVAMP] NEW: Minimum ADX to confirm a trend exists before entering
# ADX < 20 = no trend (choppy market, EMA crossovers whipsaw)
# ADX 20-25 = weak trend (acceptable)
# ADX > 25 = strong trend (ideal)
ADX_PERIOD: int = 14
ADX_MIN_THRESHOLD: float = 20.0   # Don't enter if ADX < 20 (ranging market)

# ════════════════════════════════════════════════════════════
#  Risk Management
# ════════════════════════════════════════════════════════════
MAX_POSITION_PCT: float = 0.80    # [REVAMP] 80% of balance per trade (only 1 position at a time, 5x leverage)

# ── Spot TP / SL ─────────────────────────────────────────────
TAKE_PROFIT_PCT: float = 0.020    # [REVAMP] Spot: +2.0% (was 1.8%)
STOP_LOSS_PCT: float = 0.010      # [REVAMP] Spot: -1.0% (was 0.7% — too tight, noise hits it)

# ── Futures TP / SL ──────────────────────────────────────────
FUTURES_TAKE_PROFIT_PCT: float = 0.025   # [REVAMP] Futures: +2.5% (was 2.2%)
FUTURES_STOP_LOSS_PCT: float = 0.012     # [REVAMP] Futures: -1.2% (was 0.7% — way too tight for 15m)

# ── Dynamic ATR-based SL / TP ────────────────────────────────
DYNAMIC_ATR_SL_TP: bool = True
ATR_SL_MULTIPLE: float = 2.0             # [REVAMP] SL = 2.0x ATR (was 1.5 — too tight)
ATR_TP_MULTIPLE: float = 3.0             # TP = 3.0x ATR (1.5:1 R/R ratio)
MIN_STOP_LOSS_PCT: float = 0.008         # Floor SL at 0.8%
MAX_STOP_LOSS_PCT: float = 0.030         # [REVAMP] Cap SL at 3.0% (was 2.5%)
MIN_TAKE_PROFIT_PCT: float = 0.015       # Floor TP at 1.5%
MAX_TAKE_PROFIT_PCT: float = 0.060       # Cap TP at 6.0%

MAX_HOLD_CANDLES: int = 32        # [REVAMP] Reduced from 48 to 32 (~8 hrs on 15m) — cut losers faster

# ── Spot Trailing Stop Settings ──────────────────────────────
# [REVAMP] Widened significantly — old settings were too tight and
# exited winners at +0.78% avg while TP was at +2.26%.
TRAILING_STOP_ENABLED: bool = True
TRAILING_STOP_ACTIVATION_PCT: float = 0.012   # [REVAMP] Activate after +1.2% (was 0.8%)
TRAILING_STOP_CALLBACK_PCT: float = 0.007     # [REVAMP] Trail by 0.7% from peak (was 0.5%)

# ── Futures Trailing Stop Settings ───────────────────────────
FUTURES_TRAILING_STOP_ENABLED: bool = True
FUTURES_TRAILING_STOP_ACTIVATION_PCT: float = 0.015  # [REVAMP] Activate after +1.5% (was 1.0%)
FUTURES_TRAILING_STOP_CALLBACK_PCT: float = 0.008    # [REVAMP] Trail by 0.8% from peak (was 0.6%)

# ── Spot Break-Even Settings ─────────────────────────────────
BREAK_EVEN_ENABLED: bool = True
BREAK_EVEN_ACTIVATION_PCT: float = 0.006     # [REVAMP] Move SL to entry after +0.6% (was 0.4%)

# ── Futures Break-Even Settings ──────────────────────────────
FUTURES_BREAK_EVEN_ENABLED: bool = True
FUTURES_BREAK_EVEN_ACTIVATION_PCT: float = 0.008     # [REVAMP] Move SL to entry after +0.8% (was 0.6%)

# ════════════════════════════════════════════════════════════
#  Pair Selection
# ════════════════════════════════════════════════════════════
TOP_PAIRS_COUNT: int = 10          # [REVAMP] Reduced from 15 to 10 — focus on fewer, better pairs
MIN_VOLUME_USDT_24H: float = 10_000_000   # [REVAMP] Raised from 5M to 10M — more liquid pairs only
PAIR_REFRESH_LOOPS: int = 60

# Tokens/patterns to exclude from trading
EXCLUDE_KEYWORDS: list = [
    "UP", "DOWN", "BULL", "BEAR",
    "3L", "3S", "5L", "5S",
    "BUSD", "USDC", "TUSD", "USDP",
    "DAI", "FDUSD", "AEUR", "BVND",
]

# ════════════════════════════════════════════════════════════
#  Bot Loop
# ════════════════════════════════════════════════════════════
LOOP_INTERVAL_SECONDS: int = 30
LOOP_INTERVAL_FAST: int = 2

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
ML_MODEL_PATH: str = "models/dl_model.pth"
ML_SCALER_PATH: str = "models/scaler.pkl"
ML_SEQUENCE_LENGTH: int = 10
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.60"))
# [REVAMP] Changed from Full AI Mode to Confirmation Mode
# In Full AI Mode, ML alone decides entries — with 55% accuracy this is barely better than random.
# In Confirmation Mode, technical strategy must trigger first, then ML confirms direction.
# This dramatically reduces false entries.
ML_FULL_AI_MODE: bool = False      # [REVAMP] Disabled — use Confirmation Mode instead

# ── ML Exit Settings ─────────────────────────────────────────
ML_EXIT_ENABLED: bool = True
ML_REVERSAL_EXIT_ACTIVE: bool = False    # [REVAMP] Disabled — was killing winners prematurely
ML_EXHAUSTION_EXIT_ACTIVE: bool = False
TECH_REVERSAL_EXIT_ENABLED: bool = False

# ════════════════════════════════════════════════════════════
#  Sentiment Analysis (Fear & Greed Index)
# ════════════════════════════════════════════════════════════
SENTIMENT_ENABLED: bool = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
SENTIMENT_BLOCK_THRESHOLD: float = -0.9
SENTIMENT_REFRESH_MINUTES: int = 360

# ════════════════════════════════════════════════════════════
#  [FIX-DRIFT] Re-Entry Price Drift Protection
# ════════════════════════════════════════════════════════════
DRIFT_PROTECTION_ENABLED: bool = True
DRIFT_MAX_PCT: float = 0.020  # [REVAMP] Raised from 1.5% to 2.0% — slightly more room

# ════════════════════════════════════════════════════════════
#  [FIX-B] Candle Confirmation Filter
# ════════════════════════════════════════════════════════════
CANDLE_CONFIRM_ENABLED: bool = True
CANDLE_CONFIRM_LOOKBACK: int = 3   # [REVAMP] Reduced to 3 — faster confirmation in strong trends
CANDLE_CONFIRM_MIN: int = 2        # At least 2 of 3 must agree with direction

# ════════════════════════════════════════════════════════════
#  [FIX-C] Symbol Blacklist (Consecutive Loss Protection)
# ════════════════════════════════════════════════════════════
BLACKLIST_ENABLED: bool = True
BLACKLIST_CONSECUTIVE_LOSSES: int = 2    # [REVAMP] Reduced from 3 to 2 — blacklist faster
BLACKLIST_WINDOW_HOURS: float = 4.0
BLACKLIST_COOLDOWN_HOURS: float = 3.0    # [REVAMP] Raised from 2h to 3h — longer ban

# ── Cooldown & Protection Settings ──────────────────────────
COOLDOWN_MINUTES: int = 10               # [REVAMP] Raised from 5 to 10 min — avoid whipsaw re-entry
DOUBLE_COOLDOWN_ON_LOSS: bool = True
LONG_ONLY: bool = False

# ════════════════════════════════════════════════════════════
#  Auto-Optimizer
# ════════════════════════════════════════════════════════════
OPTIMIZER_ENABLED: bool = True
OPTIMIZER_INTERVAL_HOURS: int = 2         # [REVAMP] Changed from 1h to 2h — less frequent adjustments

# ════════════════════════════════════════════════════════════
#  Funding Fee Management (Futures only)
# ════════════════════════════════════════════════════════════
FUNDING_FEE_ENABLED: bool = True
FUNDING_FEE_BLOCK_MINUTES: int = 30
FUNDING_FEE_MAX_RATE: float = 0.0003     # [REVAMP] Tightened from 0.0005 to 0.0003 — more conservative

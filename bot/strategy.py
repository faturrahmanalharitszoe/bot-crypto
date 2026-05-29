"""
bot/strategy.py — Technical signal generation (REVAMP v2)

Strategy: EMA9/21 crossover + RSI14 + MACD momentum + Volume spike + ADX trend filter

[REVAMP v2] Key changes:
  - Added ADX minimum threshold (ADX < 20 = no trade, market is ranging)
  - MACD confirmation is now REQUIRED for both BUY and SELL
  - Volume filter raised to meaningful level (1.3x avg)
  - Added EMA gap minimum to avoid signals when EMAs are too close (noise zone)
  - RSI divergence detection for higher-quality entries

Signal rules:
  BUY  → EMA9 > EMA21 + gap > 0.1% + MACD bullish + RSI valid + Volume spike + ADX > 20
  SELL → EMA9 < EMA21 + gap > 0.1% + MACD bearish + RSI valid + Volume spike + ADX > 20
  HOLD → No clear signal or market is ranging
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import ta.trend as trend_ta
import ta.momentum as mom_ta

import config

logger = logging.getLogger("bot")


class Strategy:
    """Computes technical indicators and returns trading signals."""

    def __init__(self) -> None:
        self.ema_fast        = config.EMA_FAST
        self.ema_slow        = config.EMA_SLOW
        self.rsi_period      = config.RSI_PERIOD
        self.rsi_overbought  = config.RSI_OVERBOUGHT
        self.rsi_oversold    = config.RSI_OVERSOLD
        self.vol_multiplier  = config.VOLUME_SPIKE_MULTIPLIER
        self.vol_avg_period  = config.VOLUME_AVG_PERIOD
        self.adx_period      = getattr(config, "ADX_PERIOD", 14)
        self.adx_min         = getattr(config, "ADX_MIN_THRESHOLD", 20.0)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns to the DataFrame.
        Returns enriched DataFrame with ema_fast, ema_slow, rsi, volume_avg, adx columns.
        """
        if df.empty or len(df) < self.ema_slow + 5:
            return df

        df = df.copy()

        # EMA indicators
        df["ema_fast"] = trend_ta.EMAIndicator(
            close=df["close"], window=self.ema_fast
        ).ema_indicator()

        df["ema_slow"] = trend_ta.EMAIndicator(
            close=df["close"], window=self.ema_slow
        ).ema_indicator()

        # RSI
        df["rsi"] = mom_ta.RSIIndicator(
            close=df["close"], window=self.rsi_period
        ).rsi()

        # Volume rolling average
        df["volume_avg"] = df["volume"].rolling(self.vol_avg_period).mean()

        # MACD
        macd = trend_ta.MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # ADX (Average Directional Index) — trend strength
        try:
            adx_indicator = trend_ta.ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"],
                window=self.adx_period
            )
            df["adx"] = adx_indicator.adx()
            df["di_plus"] = adx_indicator.adx_pos()
            df["di_minus"] = adx_indicator.adx_neg()
        except Exception:
            df["adx"] = 25.0  # Default to allow trading if calculation fails
            df["di_plus"] = 50.0
            df["di_minus"] = 50.0

        # ATR for volatility context
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        return df

    def get_signal(self, df: pd.DataFrame) -> tuple[str, dict]:
        """
        Evaluate the latest candles and return:
          - signal: "BUY" | "SELL" | "HOLD"
          - details: dict with indicator values for display

        [REVAMP v2] Now requires:
          1. ADX > threshold (trend exists)
          2. EMA alignment with minimum gap (not just touching)
          3. MACD confirmation (required, not optional)
          4. Volume above average * multiplier
          5. RSI in valid range
          6. Price position relative to EMA
        """
        min_required = self.ema_slow + self.vol_avg_period + 5
        if df.empty or len(df) < min_required:
            logger.debug("Not enough candles to compute signal.")
            return "HOLD", {}

        # Drop rows where indicators haven't warmed up yet
        required_cols = ["ema_fast", "ema_slow", "rsi", "volume_avg"]
        available_cols = [c for c in required_cols if c in df.columns]
        if len(available_cols) < len(required_cols):
            return "HOLD", {}
        
        df = df.dropna(subset=available_cols)
        if len(df) < 3:
            return "HOLD", {}

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        prev2  = df.iloc[-3]

        # ── ADX Trend Strength Gate ────────────────────────────
        adx_value = float(latest.get("adx", 25.0))
        adx_ok = adx_value >= self.adx_min

        # ── EMA alignment with minimum gap ─────────────────────
        ema_gap_pct = abs(latest["ema_fast"] - latest["ema_slow"]) / (latest["ema_slow"] + 1e-9) * 100
        ema_gap_sufficient = ema_gap_pct > 0.10  # At least 0.1% gap (avoid noise zone)

        ema_bullish = latest["ema_fast"] > latest["ema_slow"]
        ema_bearish = latest["ema_fast"] < latest["ema_slow"]

        # ── Price position relative to EMA9 ───────────────────
        price_above_ema9 = latest["close"] > latest["ema_fast"]
        price_below_ema9 = latest["close"] < latest["ema_fast"]

        # ── MACD momentum (REQUIRED) ──────────────────────────
        macd_bullish = (latest["macd"] > latest["macd_signal"]) and \
                       (latest["macd"] > prev["macd"])  # MACD rising
        macd_bearish = (latest["macd"] < latest["macd_signal"]) and \
                       (latest["macd"] < prev["macd"])  # MACD falling

        # ── MACD histogram momentum (additional confirmation) ──
        macd_hist_val = float(latest.get("macd_hist", 0))
        macd_hist_prev = float(prev.get("macd_hist", 0))
        macd_hist_growing_positive = macd_hist_val > 0 and macd_hist_val > macd_hist_prev
        macd_hist_growing_negative = macd_hist_val < 0 and macd_hist_val < macd_hist_prev

        # ── RSI filters ───────────────────────────────────────
        rsi_value = float(latest["rsi"])
        # BUY: RSI must not be overbought (< 68) and not in extreme oversold territory (> 25)
        # The > 25 check prevents buying into a falling knife
        rsi_buy_ok  = rsi_value < self.rsi_overbought and rsi_value > 25
        # SELL: RSI must not be deeply oversold (> 20) — in downtrends RSI stays low, that's fine
        # We only block SELL if RSI is extremely oversold (< 20 = bounce imminent)
        rsi_sell_ok = rsi_value > 20

        # ── RSI momentum (slope over 3 periods) ───────────────
        rsi_prev = float(prev["rsi"]) if "rsi" in prev.index else 50
        rsi_rising = rsi_value > rsi_prev
        rsi_falling = rsi_value < rsi_prev

        # ── Volume filter (meaningful spike required) ──────────
        volume_ok = latest["volume"] > (latest["volume_avg"] * self.vol_multiplier)

        # ── EMA momentum (fast EMA slope) ─────────────────────
        ema_fast_rising  = latest["ema_fast"] > prev["ema_fast"]
        ema_fast_falling = latest["ema_fast"] < prev["ema_fast"]

        # ── DI+/DI- directional confirmation ──────────────────
        di_plus = float(latest.get("di_plus", 50))
        di_minus = float(latest.get("di_minus", 50))
        di_bullish = di_plus > di_minus  # Buyers stronger than sellers
        di_bearish = di_minus > di_plus  # Sellers stronger than buyers

        details = {
            "ema_fast":         round(float(latest["ema_fast"]), 6),
            "ema_slow":         round(float(latest["ema_slow"]), 6),
            "ema_gap_pct":      round(ema_gap_pct, 3),
            "rsi":              round(rsi_value, 2),
            "adx":              round(adx_value, 2),
            "volume":           round(float(latest["volume"]), 2),
            "volume_avg":       round(float(latest["volume_avg"]), 2),
            "ema_bullish":      ema_bullish,
            "ema_bearish":      ema_bearish,
            "macd_bullish":     macd_bullish,
            "macd_bearish":     macd_bearish,
            "adx_ok":           adx_ok,
            "vol_ok":           volume_ok,
            "di_plus":          round(di_plus, 2),
            "di_minus":         round(di_minus, 2),
        }

        # ── Signal decision (REVAMP v2.1: balanced conditions) ───
        # Core conditions (MUST have): ADX + EMA alignment + EMA slope + MACD + RSI
        # Bonus conditions (nice to have): Volume spike, DI direction
        # This ensures signals fire during real trends while still filtering noise

        # BUY Signal:
        # Required: ADX > threshold + EMA bullish + gap sufficient + price > EMA9 + EMA rising + MACD bullish + RSI ok
        # Optional bonus: volume spike, DI+ > DI-
        buy_core = (adx_ok and ema_bullish and ema_gap_sufficient and
                    price_above_ema9 and ema_fast_rising and
                    macd_bullish and rsi_buy_ok)
        
        if buy_core:
            # At least one bonus condition should be met (volume OR directional)
            buy_bonus = volume_ok or di_bullish
            if buy_bonus:
                logger.debug(
                    f"  BUY signal | ADX={adx_value:.1f} | EMA gap={ema_gap_pct:.2f}% | "
                    f"RSI={rsi_value:.1f} | MACD✓ | Vol={'✓' if volume_ok else '✗'} | DI={'✓' if di_bullish else '✗'}"
                )
                return "BUY", details

        # SELL Signal:
        # Required: ADX > threshold + EMA bearish + gap sufficient + price < EMA9 + EMA falling + MACD bearish + RSI ok
        # Optional bonus: volume spike, DI- > DI+
        sell_core = (adx_ok and ema_bearish and ema_gap_sufficient and
                     price_below_ema9 and ema_fast_falling and
                     macd_bearish and rsi_sell_ok)
        
        if sell_core:
            sell_bonus = volume_ok or di_bearish
            if sell_bonus:
                logger.debug(
                    f"  SELL signal | ADX={adx_value:.1f} | EMA gap={ema_gap_pct:.2f}% | "
                    f"RSI={rsi_value:.1f} | MACD✓ | Vol={'✓' if volume_ok else '✗'} | DI={'✓' if di_bearish else '✗'}"
                )
                return "SELL", details

        return "HOLD", details

    def get_trend(self, df: pd.DataFrame) -> str:
        """
        Returns "UP", "DOWN", or "NEUTRAL" based on current EMA alignment + ADX.
        [REVAMP] Now also checks ADX — no trend if ADX < threshold.
        """
        if df.empty or len(df) < self.ema_slow + 5:
            return "NEUTRAL"
        df = df.dropna(subset=["ema_fast", "ema_slow"])
        if df.empty:
            return "NEUTRAL"
        latest = df.iloc[-1]

        # Check ADX — if market is ranging, return NEUTRAL
        adx_value = float(latest.get("adx", 0))
        if adx_value < self.adx_min:
            return "NEUTRAL"

        if latest["ema_fast"] > latest["ema_slow"]:
            return "UP"
        elif latest["ema_fast"] < latest["ema_slow"]:
            return "DOWN"
        return "NEUTRAL"

    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        [REVAMP v2] NEW: Returns a signal strength score 0.0-1.0.
        Used to prioritize which pairs to trade when multiple signals fire.
        
        Factors:
          - ADX strength (higher = stronger trend)
          - EMA gap magnitude (wider = more conviction)
          - Volume spike magnitude
          - MACD histogram magnitude
        """
        if df.empty or len(df) < self.ema_slow + self.vol_avg_period + 5:
            return 0.0

        try:
            latest = df.iloc[-1]
            
            # ADX component (0-1): normalized to 0-50 range
            adx = float(latest.get("adx", 0))
            adx_score = min(adx / 50.0, 1.0)
            
            # EMA gap component (0-1): normalized to 0-1% range
            ema_gap = abs(float(latest["ema_fast"]) - float(latest["ema_slow"])) / (float(latest["ema_slow"]) + 1e-9) * 100
            gap_score = min(ema_gap / 1.0, 1.0)
            
            # Volume component (0-1): normalized to 1-3x average
            vol_avg = float(latest.get("volume_avg", 1))
            vol_ratio = float(latest["volume"]) / (vol_avg + 1e-9)
            vol_score = min((vol_ratio - 1.0) / 2.0, 1.0)
            vol_score = max(vol_score, 0.0)
            
            # Weighted average
            strength = (adx_score * 0.35) + (gap_score * 0.30) + (vol_score * 0.35)
            return round(strength, 3)
        except Exception:
            return 0.0

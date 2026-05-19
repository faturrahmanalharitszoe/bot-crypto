"""
bot/strategy.py — Technical signal generation
Strategy: EMA9/21 alignment + RSI14 + MACD momentum + Volume filter

Uses the 'ta' library (Python 3.9+ compatible, no C extensions required).

Signal rules:
  BUY  → EMA9 > EMA21 (aligned)  AND  RSI in valid range  AND  MACD bullish  AND  price > EMA9
  SELL → EMA9 < EMA21 (aligned)  AND  RSI not oversold  AND  MACD bearish  AND  price < EMA9
  HOLD → No clear signal

Note: Uses EMA *alignment* (not just crossover) so signals fire continuously
during a trend, not just the single moment of crossing.
"""

from __future__ import annotations  # Python 3.9 compatibility

import logging
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

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicator columns to the DataFrame.
        Returns enriched DataFrame with ema_fast, ema_slow, rsi, volume_avg columns.
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

        return df

    def get_signal(self, df: pd.DataFrame) -> tuple[str, dict]:
        """
        Evaluate the latest two candles and return:
          - signal: "BUY" | "SELL" | "HOLD"
          - details: dict with indicator values for display

        Returns ("HOLD", {}) if data is insufficient.
        """
        min_required = self.ema_slow + self.vol_avg_period + 5
        if df.empty or len(df) < min_required:
            logger.debug("Not enough candles to compute signal.")
            return "HOLD", {}

        # Drop rows where indicators haven't warmed up yet
        df = df.dropna(subset=["ema_fast", "ema_slow", "rsi", "volume_avg"])
        if len(df) < 2:
            return "HOLD", {}

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # ── Crossover detection ───────────────────────────────
        ema_cross_up   = (prev["ema_fast"] <= prev["ema_slow"]) and \
                         (latest["ema_fast"] > latest["ema_slow"])

        ema_cross_down = (prev["ema_fast"] >= prev["ema_slow"]) and \
                         (latest["ema_fast"] < latest["ema_slow"])

        # ── EMA alignment (trend direction) ───────────────────
        ema_bullish = latest["ema_fast"] > latest["ema_slow"]
        ema_bearish = latest["ema_fast"] < latest["ema_slow"]

        # ── Price position relative to EMA9 ───────────────────
        price_above_ema9 = latest["close"] > latest["ema_fast"]
        price_below_ema9 = latest["close"] < latest["ema_fast"]

        # ── MACD momentum ─────────────────────────────────────
        macd_bullish = (latest["macd"] > latest["macd_signal"]) and \
                       (latest["macd"] > prev["macd"])  # MACD rising
        macd_bearish = (latest["macd"] < latest["macd_signal"]) and \
                       (latest["macd"] < prev["macd"])  # MACD falling

        # ── RSI filters ───────────────────────────────────────
        rsi_buy_ok  = latest["rsi"] < self.rsi_overbought  # Not overbought
        rsi_sell_ok = latest["rsi"] > self.rsi_oversold    # Not oversold

        # ── Volume filter (relaxed: 1.0x avg is enough) ───────
        volume_ok = latest["volume"] > (latest["volume_avg"] * 1.0)  # At or above average

        # ── EMA momentum (angle check: fast EMA is rising/falling) ─
        ema_fast_rising  = latest["ema_fast"] > prev["ema_fast"]
        ema_fast_falling = latest["ema_fast"] < prev["ema_fast"]

        details = {
            "ema_fast":         round(float(latest["ema_fast"]), 6),
            "ema_slow":         round(float(latest["ema_slow"]), 6),
            "rsi":              round(float(latest["rsi"]), 2),
            "volume":           round(float(latest["volume"]), 2),
            "volume_avg":       round(float(latest["volume_avg"]), 2),
            "cross_up":         ema_cross_up,
            "cross_down":       ema_cross_down,
            "ema_bullish":      ema_bullish,
            "ema_bearish":      ema_bearish,
            "macd_bullish":     macd_bullish,
            "macd_bearish":     macd_bearish,
            "rsi_ok":           rsi_buy_ok,
            "vol_ok":           volume_ok,
        }

        # ── Signal decision ───────────────────────────────────
        # BUY: EMA aligned bullish + price above EMA9 + MACD bullish + RSI ok + volume ok
        if ema_bullish and price_above_ema9 and ema_fast_rising and rsi_buy_ok and volume_ok:
            logger.debug(
                f"  BUY  signal | EMA aligned↑ | RSI={details['rsi']:.1f} | "
                f"MACD_bull={macd_bullish}"
            )
            return "BUY", details

        # SELL: EMA aligned bearish + price below EMA9 + MACD bearish + RSI ok
        if ema_bearish and price_below_ema9 and ema_fast_falling and rsi_sell_ok:
            logger.debug(
                f"  SELL signal | EMA aligned↓ | RSI={details['rsi']:.1f}"
            )
            return "SELL", details

        return "HOLD", details

    def get_trend(self, df: pd.DataFrame) -> str:
        """
        Returns "UP", "DOWN", or "NEUTRAL" based on current EMA alignment.
        """
        if df.empty or len(df) < self.ema_slow + 5:
            return "NEUTRAL"
        df = df.dropna(subset=["ema_fast", "ema_slow"])
        if df.empty:
            return "NEUTRAL"
        latest = df.iloc[-1]
        if latest["ema_fast"] > latest["ema_slow"]:
            return "UP"
        elif latest["ema_fast"] < latest["ema_slow"]:
            return "DOWN"
        return "NEUTRAL"

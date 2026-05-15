"""
bot/strategy.py — Technical signal generation
Strategy: EMA9/21 crossover + RSI14 + Volume spike filter

Uses the 'ta' library (Python 3.9+ compatible, no C extensions required).

Signal rules:
  BUY  → EMA9 crosses above EMA21  AND  RSI < RSI_OVERBOUGHT  AND  volume spike
  SELL → EMA9 crosses below EMA21  (trend reversal exit signal)
  HOLD → No clear signal
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

        # ── Filters ───────────────────────────────────────────
        rsi_ok       = latest["rsi"] < self.rsi_overbought
        volume_spike = latest["volume"] > (latest["volume_avg"] * self.vol_multiplier)

        details = {
            "ema_fast":   round(float(latest["ema_fast"]), 6),
            "ema_slow":   round(float(latest["ema_slow"]), 6),
            "rsi":        round(float(latest["rsi"]), 2),
            "volume":     round(float(latest["volume"]), 2),
            "volume_avg": round(float(latest["volume_avg"]), 2),
            "cross_up":   ema_cross_up,
            "cross_down": ema_cross_down,
            "rsi_ok":     rsi_ok,
            "vol_spike":  volume_spike,
        }

        # ── Signal decision ───────────────────────────────────
        if ema_cross_up and rsi_ok and volume_spike:
            logger.debug(
                f"  BUY  signal | EMA cross↑ | RSI={details['rsi']:.1f} | "
                f"VolSpike={volume_spike}"
            )
            return "BUY", details

        if ema_cross_down:
            logger.debug(f"  SELL signal | EMA cross↓")
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

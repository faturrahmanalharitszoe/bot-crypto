"""
bot/ml_model.py — Machine Learning Predictor (Full AI Mode)

Loads a pre-trained RandomForest model and generates BUY confidence scores.
Training is done separately via train_model.py (run locally).

Feature Engineering:
  - EMA cross angle & gap
  - RSI, MACD, MACD signal
  - Volume spike ratio
  - Volatility (high-low range %)
  - Time of day (hour), day of week
"""

from __future__ import annotations

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger("bot")

# Standard feature set for the model (22 features with MTF)
FEATURE_COLS = [
    "ema_cross_gap", "ema_cross_angle", "macd_hist", "rsi", "volatility_pct",
    "volume_spike", "price_change_5", "price_change_3", "hour",
    "adx", "bb_pct", "atr_norm", "volume_momentum", "dist_ema_fast",
    "dist_ema_slow", "rsi_slope", "high_low_gap", "is_bullish_candle",
    "rsi_1m", "rsi_15m", "trend_15m", "vol_1m_spike"
]

class MLPredictor:
    def __init__(self, model_path: str = "models/rf_model.pkl"):
        self.model_path = model_path
        self.model = None
        self._is_ready = False
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                # Use absolute path to ensure reliability on VPS
                abs_path = os.path.abspath(self.model_path)
                with open(abs_path, 'rb') as f:
                    self.model = pickle.load(f)
                self._is_ready = True
                logger.info(f"🤖 ML Model loaded successfully | {abs_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load ML model: {e}")
        else:
            logger.warning(f"⚠️  ML Model file not found at {self.model_path}. Bot will use default logic.")

    def reload(self) -> None:
        """Reload model from disk (for hot-reload after re-training)."""
        self.model = None
        self._is_ready = False
        self._load_model()

    def prepare_features(self, df_5m: pd.DataFrame, df_1m: pd.DataFrame = None, df_15m: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """Convert multi-timeframe kline data into model features (22 indicators)."""
        try:
            if len(df_5m) < 30:
                return None

            df = df_5m.copy()
            # ── 5m Features (Core) ───────────────────────────
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_cross_gap'] = (df['ema_9'] - df['ema_21']) / (df['ema_21'] + 1e-9) * 100
            df['ema_cross_angle'] = df['ema_cross_gap'].diff()
            
            # MACD
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd_line'] = ema_12 - ema_26
            df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd_line'] - df['signal_line']
            
            # RSI
            def calc_rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / (loss + 1e-9)
                return 100 - (100 / (1 + rs))

            df['rsi'] = calc_rsi(df['close'])
            df['rsi_slope'] = df['rsi'].diff(3)
            
            # Volatility & Volume
            df['volatility_pct'] = (df['high'] - df['low']) / (df['low'] + 1e-9) * 100
            df['volume_mean'] = df['volume'].rolling(window=20).mean()
            df['volume_spike'] = df['volume'] / (df['volume_mean'] + 1e-9)
            df['volume_momentum'] = df['volume'].diff(3) / (df['volume_mean'] + 1e-9)
            
            # Price Momentum
            df['price_change_5'] = df['close'].pct_change(5) * 100
            df['price_change_3'] = df['close'].pct_change(3) * 100
            df['is_bullish_candle'] = (df['close'] > df['open']).astype(int)
            df['high_low_gap'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
            
            # ADX (Trend Strength)
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = pd.concat([df['high'] - df['low'], 
                           (df['high'] - df['close'].shift()).abs(), 
                           (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            df['adx'] = (plus_dm.rolling(14).mean() / (atr + 1e-9)).rolling(14).mean() * 100
            df['atr_norm'] = atr / (df['close'] + 1e-9) * 100
            
            # Bollinger Bands
            df['bb_mid'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_pct'] = (df['close'] - (df['bb_mid'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-9)
            
            # Distance to EMA
            df['dist_ema_fast'] = (df['close'] - df['ema_9']) / (df['ema_9'] + 1e-9) * 100
            df['dist_ema_slow'] = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-9) * 100

            # ── 1m Features (Execution) ──────────────────────
            if df_1m is not None and not df_1m.empty:
                rsi_1m = calc_rsi(df_1m['close']).iloc[-1]
                vol_1m_mean = df_1m['volume'].rolling(20).mean().iloc[-1]
                vol_1m_spike = df_1m['volume'].iloc[-1] / (vol_1m_mean + 1e-9)
            else:
                rsi_1m, vol_1m_spike = 50.0, 1.0
            
            df['rsi_1m'] = rsi_1m
            df['vol_1m_spike'] = vol_1m_spike

            # ── 15m Features (Context) ───────────────────────
            if df_15m is not None and not df_15m.empty:
                rsi_15m = calc_rsi(df_15m['close']).iloc[-1]
                ema_15m_fast = df_15m['close'].ewm(span=9).mean().iloc[-1]
                ema_15m_slow = df_15m['close'].ewm(span=21).mean().iloc[-1]
                trend_15m = 1 if ema_15m_fast > ema_15m_slow else -1
            else:
                rsi_15m, trend_15m = 50.0, 0
                
            df['rsi_15m'] = rsi_15m
            df['trend_15m'] = trend_15m

            # Time feature
            df['hour'] = df.index.hour
            
            return df[FEATURE_COLS].dropna().iloc[[-1]]
            
        except Exception as e:
            logger.error(f"Error preparing MTF features: {e}")
            return None

    def predict(self, df_5m: pd.DataFrame, df_1m: pd.DataFrame = None, df_15m: pd.DataFrame = None) -> float:
        """Predict BUY probability (0.0 to 1.0) using Multi-Timeframe data."""
        if not self.is_ready:
            return 0.0
            
        features_df = self.prepare_features(df_5m, df_1m, df_15m)
        if features_df is None or features_df.empty:
            return 0.0
            
        try:
            # Predict probability of class 1 (BUY)
            probs = self.model.predict_proba(features_df)
            prob = float(probs[0][1]) # Cast to native float for JSON
            return prob
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0

    @property
    def is_ready(self) -> bool:
        """True if model is loaded and ready for predictions."""
        return self._is_ready

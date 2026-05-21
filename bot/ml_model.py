"""
bot/ml_model.py — Machine Learning Predictor (Full AI Mode)

Loads a pre-trained Deep Learning (PyTorch MLP) model and generates BUY/SELL confidence scores.
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

# Standard feature set for the model (25 features with MTF)
FEATURE_COLS = [
    "ema_cross_gap", "ema_cross_angle", "macd_hist", "rsi", "volatility_pct",
    "volume_spike", "price_change_5", "price_change_3", "hour", "day_of_week",
    "adx", "bb_pct", "atr_norm", "volume_momentum", "dist_ema_fast",
    "dist_ema_slow", "rsi_slope", "high_low_gap", "is_bullish_candle",
    "rsi_lower", "rsi_higher", "trend_higher", "vol_lower_spike", "range_pct", "rsi_div"
]

class MLPredictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.ML_MODEL_PATH
        self.scaler_path = getattr(config, 'ML_SCALER_PATH', 'models/scaler.pkl')
        self.model = None
        self.scaler = None
        self._is_ready = False
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model and scaler from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                import torch
                from bot.nn_model import DayTradingMLP

                # Use absolute paths to ensure reliability on VPS
                abs_model_path = os.path.abspath(self.model_path)
                abs_scaler_path = os.path.abspath(self.scaler_path)

                # Load fitted feature scaler
                with open(abs_scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

                # Initialize model architecture and load weights
                checkpoint = torch.load(abs_model_path, map_location=torch.device('cpu'))
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    hidden_dims = checkpoint.get('hidden_dims', [64, 32])
                    dropout = checkpoint.get('dropout', 0.2)
                    activation = checkpoint.get('activation', 'relu')
                    logger.info(f"📦 Loaded model metadata: hidden_dims={hidden_dims}, dropout={dropout}, activation={activation}")
                    self.model = DayTradingMLP(hidden_dims=hidden_dims, dropout=dropout, activation=activation)
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    logger.info("📦 Loaded model (legacy raw state_dict without metadata)")
                    self.model = DayTradingMLP()
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval() # Set to evaluation mode for inference

                self._is_ready = True
                logger.info(f"🤖 DL Model loaded successfully | {abs_model_path}")
                logger.info(f"📐 RobustScaler loaded successfully | {abs_scaler_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load DL model/scaler: {e}")
        else:
            logger.warning(f"⚠️  DL Model or Scaler file not found. Paths:\n"
                           f"   Model: {self.model_path}\n"
                           f"   Scaler: {self.scaler_path}\n"
                           f"   Bot will use default logic.")

    def reload(self) -> None:
        """Reload model and scaler from disk (for hot-reload after re-training)."""
        self.model = None
        self.scaler = None
        self._is_ready = False
        self._load_model()

    def prepare_features(self, df_main: pd.DataFrame, df_lower: pd.DataFrame = None, df_higher: pd.DataFrame = None) -> Optional[pd.DataFrame]:
        """Convert multi-timeframe kline data into model features (25 indicators)."""
        try:
            if len(df_main) < 30:
                return None

            df = df_main.copy()
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

            # ── Lower & Higher Timeframe Sync ────────────────
            if df_lower is not None and not df_lower.empty:
                df_lower_feat = pd.DataFrame(index=df.index)
                df_lower_feat['rsi_lower'] = calc_rsi(df_lower['close']).reindex(df.index, method='ffill')
                vol_lower_mean = df_lower['volume'].rolling(20).mean()
                df_lower_feat['vol_lower_spike'] = (df_lower['volume'] / (vol_lower_mean + 1e-9)).reindex(df.index, method='ffill')
            else:
                df_lower_feat = pd.DataFrame(index=df.index)
                df_lower_feat['rsi_lower'] = 50.0
                df_lower_feat['vol_lower_spike'] = 1.0

            if df_higher is not None and not df_higher.empty:
                df_higher_feat = pd.DataFrame(index=df.index)
                df_higher_feat['rsi_higher'] = calc_rsi(df_higher['close']).reindex(df.index, method='ffill')
                ema_higher_f = df_higher['close'].ewm(span=9).mean()
                ema_higher_s = df_higher['close'].ewm(span=21).mean()
                df_higher_feat['trend_higher'] = (ema_higher_f > ema_higher_s).astype(int).replace(0, -1).reindex(df.index, method='ffill')
            else:
                df_higher_feat = pd.DataFrame(index=df.index)
                df_higher_feat['rsi_higher'] = 50.0
                df_higher_feat['trend_higher'] = 0

            df = pd.concat([df, df_lower_feat, df_higher_feat], axis=1)

            # Additional features to match train_model.py
            df['rsi_div'] = df['rsi'].diff() - df['close'].diff() / df['close']
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
            
            valid_df = df[FEATURE_COLS].dropna()
            if len(valid_df) < 2:
                # Fallback to last row if history is very short
                return valid_df.iloc[[-1]] if not valid_df.empty else None
            # Return the last fully completed candle (index -2) to avoid unclosed live candle noise
            return valid_df.iloc[[-2]]
            
        except Exception as e:
            logger.error(f"Error preparing MTF features: {e}")
            return None

    def predict(self, df_main: pd.DataFrame, df_lower: pd.DataFrame = None, df_higher: pd.DataFrame = None) -> tuple[int, float]:
        """
        Predict market direction using Multi-Timeframe data.
        Returns: (signal_class, confidence)
        signal_class: 0=HOLD, 1=LONG, 2=SHORT
        """
        if not self.is_ready or self.model is None or self.scaler is None:
            return 0, 0.0
            
        features_df = self.prepare_features(df_main, df_lower, df_higher)
        if features_df is None or features_df.empty:
            return 0, 0.0
            
        try:
            import torch
            
            # Scale features
            features_scaled = self.scaler.transform(features_df.values)
            
            # Convert to PyTorch tensor
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()
                
            # Class mapping:
            # Output class index 0 maps to Class 1 (LONG)
            # Output class index 1 maps to Class 2 (SHORT)
            best_idx = int(np.argmax(probs))
            best_class = 1 if best_idx == 0 else 2
            confidence = float(probs[best_idx])
            
            return best_class, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0

    @property
    def is_ready(self) -> bool:
        """True if model and scaler are loaded and ready for predictions."""
        return self._is_ready

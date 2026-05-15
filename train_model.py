import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Import local bot components
from bot.strategy import Strategy

# ── Optional: load .env for API keys ──────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET            = os.getenv("TESTNET", "true").lower() == "true"
MODEL_OUTPUT       = "models/rf_model.pkl"

# Training configuration
CANDLE_INTERVAL    = "5m"
CANDLE_LIMIT       = 1000      # Max allowed by Binance per request
LABEL_HORIZON      = 5         # Predict price change N candles ahead
LABEL_THRESHOLD    = 0.005     # +0.5% = BUY label
MIN_CANDLES        = 200       # Skip pairs with less data

TRAINING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "POLUSDT", "DOTUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "TIAUSDT",
    "SEIUSDT", "FETUSDT", "WLDUSDT", "PENDLEUSDT", "ORDIUSDT",
]


def fetch_klines(client, symbol: str, interval: str = "5m") -> pd.DataFrame:
    """Fetch OHLCV klines from Binance."""
    try:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=CANDLE_LIMIT,
        )
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "tb_base_vol",
            "tb_quote_vol", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        print(f"  ⚠️  Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def build_features(df_5m: pd.DataFrame, df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using MTF logic exactly like ml_model.py."""
    df = df_5m.copy()
    
    # ── 5m Features ──────────────────────────────────
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_cross_gap'] = (df['ema_9'] - df['ema_21']) / (df['ema_21'] + 1e-9) * 100
    df['ema_cross_angle'] = df['ema_cross_gap'].diff()
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['signal_line']
    
    def calc_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['rsi'] = calc_rsi(df['close'])
    df['rsi_slope'] = df['rsi'].diff(3)
    df['volatility_pct'] = (df['high'] - df['low']) / (df['low'] + 1e-9) * 100
    df['volume_mean'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] / (df['volume_mean'] + 1e-9)
    df['volume_momentum'] = df['volume'].diff(3) / (df['volume_mean'] + 1e-9)
    df['price_change_5'] = df['close'].pct_change(5) * 100
    df['price_change_3'] = df['close'].pct_change(3) * 100
    df['is_bullish_candle'] = (df['close'] > df['open']).astype(int)
    df['high_low_gap'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-9)
    
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df['adx'] = (plus_dm := df['high'].diff().clip(lower=0)).rolling(14).mean() / (atr + 1e-9) * 100
    df['atr_norm'] = atr / (df['close'] + 1e-9) * 100
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_pct'] = (df['close'] - (df['bb_mid'] - 2*df['bb_std'])) / (4*df['bb_std'] + 1e-9)
    df['dist_ema_fast'] = (df['close'] - df['ema_9']) / (df['ema_9'] + 1e-9) * 100
    df['dist_ema_slow'] = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-9) * 100

    # ── 1m & 15m Sync ────────────────────────────────
    # For training, we align the timestamps
    df_1m_feat = pd.DataFrame(index=df.index)
    df_1m_feat['rsi_1m'] = calc_rsi(df_1m['close']).reindex(df.index, method='ffill')
    vol_1m_mean = df_1m['volume'].rolling(20).mean()
    df_1m_feat['vol_1m_spike'] = (df_1m['volume'] / (vol_1m_mean + 1e-9)).reindex(df.index, method='ffill')
    
    df_15m_feat = pd.DataFrame(index=df.index)
    df_15m_feat['rsi_15m'] = calc_rsi(df_15m['close']).reindex(df.index, method='ffill')
    ema_15m_f = df_15m['close'].ewm(span=9).mean()
    ema_15m_s = df_15m['close'].ewm(span=21).mean()
    df_15m_feat['trend_15m'] = (ema_15m_f > ema_15m_s).astype(int).replace(0, -1).reindex(df.index, method='ffill')

    df = pd.concat([df, df_1m_feat, df_15m_feat], axis=1)
    df['hour'] = df.index.hour
    
    return df


# Standard feature set for the model (22 features now)
FEATURE_COLS = [
    "ema_cross_gap", "ema_cross_angle", "macd_hist", "rsi", "volatility_pct",
    "volume_spike", "price_change_5", "price_change_3", "hour",
    "adx", "bb_pct", "atr_norm", "volume_momentum", "dist_ema_fast",
    "dist_ema_slow", "rsi_slope", "high_low_gap", "is_bullish_candle",
    "rsi_1m", "rsi_15m", "trend_15m", "vol_1m_spike"
]


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create binary label: 1 if price rises > LABEL_THRESHOLD in next N candles.
    """
    future_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1
    return (future_return > LABEL_THRESHOLD).astype(int)


def main():
    print("=" * 60)
    print("🤖 Binance Scalping Bot — AI Auto-Tuner")
    print("=" * 60)

    try:
        from binance.client import Client
    except ImportError:
        print("❌ Install python-binance: pip install python-binance")
        return

    print(f"\n📡 Connecting to Binance...")
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=TESTNET)

    # ── Collect training data ─────────────────────────────────
    all_X, all_y = [], []
    print(f"\n📊 Fetching data for {len(TRAINING_PAIRS)} pairs...")

    for symbol in TRAINING_PAIRS:
        print(f"   📥 {symbol}...", end="\r")
        df_5m = fetch_klines(client, symbol, interval="5m")
        df_1m = fetch_klines(client, symbol, interval="1m")
        df_15m = fetch_klines(client, symbol, interval="15m")
        
        if len(df_5m) < MIN_CANDLES or len(df_1m) < MIN_CANDLES or len(df_15m) < MIN_CANDLES:
            continue
            
        df = build_features(df_5m, df_1m, df_15m)
        df["label"] = build_labels(df)
        df.dropna(inplace=True)
        
        X = df[FEATURE_COLS].values
        y = df["label"].values
        X = X[:-LABEL_HORIZON]
        y = y[:-LABEL_HORIZON]
        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"📈 Total samples: {len(X_all):,} | BUY rate: {y_all.mean():.1%}")

    # ── Hyperparameter Tuning Loop ────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    best_score = -1
    best_pipe = None
    
    # Grid of parameters to try
    param_grid = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 8},
        {"n_estimators": 150, "max_depth": 12},
        {"n_estimators": 300, "max_depth": 15},
    ]

    print("\n🔍 Tuning model for best performance...")
    for params in param_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # We use ROC-AUC as it's better for imbalanced trading data
        scores = cross_val_score(pipe, X_all, y_all, cv=3, scoring="roc_auc")
        avg_score = scores.mean()
        
        print(f"   Testing: {params} → ROC-AUC: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_pipe = pipe

    # ── Final Report ──────────────────────────────────────────
    print(f"\n🏆 Best Model Found! ROC-AUC: {best_score:.4f}")
    best_pipe.fit(X_all, y_all)
    
    print("\n🧠 AI Logic (Top Features):")
    importances = best_pipe.named_steps["clf"].feature_importances_
    feat_importances = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for feat, val in feat_importances[:8]:
        print(f"   {feat:<18}: {'█' * int(val*50)} {val:.1%}")

    # ── Save model ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(best_pipe, f)

    print(f"\n✅ Best model saved to: {MODEL_OUTPUT}")
    print(f"🚀 Sekarang upload ke VPS dan restart bot!")
    print("=" * 60)

if __name__ == "__main__":
    main()

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import glob

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
    "PEPEUSDT", "SHIBUSDT", "RENDERUSDT", "STXUSDT", "FILUSDT",
    "NEARUSDT", "GRTUSDT", "ICPUSDT", "GALAUSDT", "LDOUSDT",
]


def fetch_klines(client, symbol: str, interval: str = "5m", days: int = 14, end_time: str = None) -> pd.DataFrame:
    """Fetch historical OHLCV klines from Binance (multi-page)."""
    try:
        # Calculate start time
        start_str = f"{days} days ago UTC"
        if end_time is not None:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_time
            )
        else:
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )
         
        if not klines: return pd.DataFrame()
 
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
    
    # RSI Divergence (Simple)
    df['rsi_div'] = df['rsi'].diff() - df['close'].diff() / df['close']
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    return df


# Standard feature set for the model (22 features now)
FEATURE_COLS = [
    "ema_cross_gap", "ema_cross_angle", "macd_hist", "rsi", "volatility_pct",
    "volume_spike", "price_change_5", "price_change_3", "hour", "day_of_week",
    "adx", "bb_pct", "atr_norm", "volume_momentum", "dist_ema_fast",
    "dist_ema_slow", "rsi_slope", "high_low_gap", "is_bullish_candle",
    "rsi_1m", "rsi_15m", "trend_15m", "vol_1m_spike", "range_pct", "rsi_div"
]


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create multi-class labels:
    1: LONG (price rises > threshold)
    2: SHORT (price falls > threshold)
    0: HOLD (otherwise)
    """
    future_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1
    
    labels = np.zeros(len(df))
    labels[future_return > LABEL_THRESHOLD] = 1
    labels[future_return < -LABEL_THRESHOLD] = 2
    return pd.Series(labels, index=df.index)


def load_trade_data(client):
    """Load and process trade data from logs directory."""
    import glob
    trade_files = glob.glob("bot-crypto/logs/trades*.csv")
    all_X_trade = []
    all_y_trade = []
    
    for file in trade_files:
        try:
            df_trades = pd.read_csv(file)
        except Exception as e:
            print(f"  ⚠️  Failed to read {file}: {e}")
            continue
            
        for _, row in df_trades.iterrows():
            try:
                # Parse the timestamp
                trade_time = pd.to_datetime(row['timestamp'])
                symbol = row['symbol']
                # Fetch klines up to the trade_time
                df_5m = fetch_klines(client, symbol, interval="5m", days=14, end_time=trade_time.strftime("%Y-%m-%d %H:%M:%S"))
                df_1m = fetch_klines(client, symbol, interval="1m", days=14, end_time=trade_time.strftime("%Y-%m-%d %H:%M:%S"))
                df_15m = fetch_klines(client, symbol, interval="15m", days=14, end_time=trade_time.strftime("%Y-%m-%d %H:%M:%S"))

                # Check if we have enough data
                if len(df_5m) < MIN_CANDLES or len(df_1m) < MIN_CANDLES or len(df_15m) < MIN_CANDLES:
                    continue

                # Build features for the entire dataframes (we need the last row)
                df_features = build_features(df_5m, df_1m, df_15m)
                # We want the last row (which corresponds to the trade_time or the last kline before)
                if df_features.empty:
                    continue
                last_features = df_features.iloc[[-1]]  # This is a DataFrame with one row

                # Determine the label
                pnl = row['pnl_usdt']
                side = row['side']
                if side == 'BUY' and pnl > 0:
                    label = 1   # LONG
                elif side == 'SELL' and pnl > 0:
                    label = 2   # SHORT
                else:
                    label = 0   # HOLD (unprofitable trade)

                # Append
                all_X_trade.append(last_features[FEATURE_COLS].values)
                all_y_trade.append(label)
            except Exception as e:
                print(f"  ⚠️  Failed to process trade {row['timestamp']} {row['symbol']}: {e}")
                continue

    if all_X_trade:
        X_trade = np.vstack(all_X_trade)
        y_trade = np.concatenate(all_y_trade)
        return X_trade, y_trade
    else:
        return np.array([]), np.array([])


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
    
    # Load trade data
    print(f"\n📈 Loading trade data from logs...")
    X_trade, y_trade = load_trade_data(client)
    if len(X_trade) > 0:
        print(f"   Loaded {len(X_trade)} trade samples")
        # Combine
        X_all = np.vstack([X_all, X_trade])
        y_all = np.concatenate([y_all, y_trade])
    else:
        print("   No trade data loaded")
    
    long_rate = (y_all == 1).mean()
    short_rate = (y_all == 2).mean()
    print(f"📈 Total samples: {len(X_all):,} | LONG: {long_rate:.1%} | SHORT: {short_rate:.1%}")

    # ── Hyperparameter Tuning Loop ────────────────────────────
    # ExtraTrees is often better for trading data noise
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    
    # Check for SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        print("\n🪄 SMOTE balancing enabled...")
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_all, y_all)
        print(f"   Samples after SMOTE: {len(X_res):,} (Balanced)")
    except ImportError:
        print("\n⚠️  INSTALL 'imbalanced-learn' SEKARANG: pip install imbalanced-learn")
        X_res, y_res = X_all, y_all

    best_score = -1
    best_pipe = None
    
    # More aggressive grid
    param_grid = [
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 15},
        {"n_estimators": 400, "max_depth": 20},
        {"n_estimators": 500, "max_depth": 25},
        {"n_estimators": 600, "max_depth": 15},
    ]

    print("\n🔍 Tuning model for best performance...")
    for params in param_grid:
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", ExtraTreesClassifier(
                **params,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Cross-validation on balanced data
        scores = cross_val_score(pipe, X_res, y_res, cv=3, scoring="f1_macro")
        avg_score = scores.mean()
        
        print(f"   Testing: {params} → F1-Score: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_pipe = pipe

    # ── Final Report ──────────────────────────────────────────
    print(f"\n🏆 Best Model Found! F1-Score: {best_score:.4f}")
    best_pipe.fit(X_res, y_res) # Fit on balanced data
    
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

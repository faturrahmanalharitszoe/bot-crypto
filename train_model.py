import os
import sys

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

# Enable programmatic DNS bypass for regions where Binance is blocked
from bot import dns_bypass

# Import PyTorch and network architecture
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from bot.nn_model import DayTradingMLP

# Import local bot components
from bot.strategy import Strategy
import config

# ── Optional: load .env for API keys ──────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET            = os.getenv("TESTNET", "true").lower() == "true"

MODEL_OUTPUT       = config.ML_MODEL_PATH
SCALER_OUTPUT      = getattr(config, 'ML_SCALER_PATH', 'models/scaler.pkl')

# Training configuration
CANDLE_INTERVAL    = config.CANDLE_INTERVAL
LOWER_INTERVAL     = config.LOWER_INTERVAL
HIGHER_INTERVAL    = config.HIGHER_INTERVAL
TRAINING_DAYS      = 60
CANDLE_LIMIT       = 1000
# [MTF] Scalping-oriented horizon: 2 candles × 15m = 30m lookahead
# (was 4 × 15m = 1h, too long for scalp trades that last 1-20m)
LABEL_HORIZON      = 2
LABEL_THRESHOLD    = 0.004     # +0.4% = signal (was 0.5%, tighter for scalping)
MIN_CANDLES        = 200       # Skip pairs with less data

# [MTF] Extended feature set: 25 original + 6 MTF confluence features = 31 total
FEATURE_COLS = [
    # ── Original 25 features ──
    "ema_cross_gap", "ema_cross_angle", "macd_hist", "rsi", "volatility_pct",
    "volume_spike", "price_change_5", "price_change_3", "hour", "day_of_week",
    "adx", "bb_pct", "atr_norm", "volume_momentum", "dist_ema_fast",
    "dist_ema_slow", "rsi_slope", "high_low_gap", "is_bullish_candle",
    "rsi_lower", "rsi_higher", "trend_higher", "vol_lower_spike", "range_pct", "rsi_div",
    # ── 6 NEW MTF Confluence features ──
    "mtf_alignment_score",   # Sum of TF directions: -3=all SHORT, 0=mixed, +3=all LONG
    "tf_1h_15m_agree",       # 1 if 1h and 15m trend same direction
    "tf_15m_5m_agree",       # 1 if 15m and 5m (lower) same direction
    "tf_all_bearish",        # 1 if all 3 TFs bearish (strong SHORT confluence)
    "tf_all_bullish",        # 1 if all 3 TFs bullish (strong LONG confluence)
    "higher_trend_strength", # EMA9/21 gap magnitude on 1h (how strong the HTF trend is)
    # ── 7 Fibonacci Retracement features (50-candle window) ──
    "fib_dist_0_0",
    "fib_dist_236",
    "fib_dist_382",
    "fib_dist_500",
    "fib_dist_618",
    "fib_dist_786",
    "fib_dist_1_0",
]

TRAINING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "POLUSDT", "DOTUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "TIAUSDT",
    "SEIUSDT", "FETUSDT", "WLDUSDT", "PENDLEUSDT", "ORDIUSDT",
    "PEPEUSDT", "SHIBUSDT", "RENDERUSDT", "STXUSDT", "FILUSDT",
    "GRTUSDT", "ICPUSDT", "GALAUSDT", "LDOUSDT",
]


def fetch_klines(client, symbol: str, interval: str = "5m", days: int = None, end_time: str = None) -> pd.DataFrame:
    """Fetch historical OHLCV klines from Binance (multi-page)."""
    if days is None:
        days = TRAINING_DAYS
    try:
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

        if not klines:
            return pd.DataFrame()

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


def build_features(df_main: pd.DataFrame, df_lower: pd.DataFrame, df_higher: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix using MTF logic exactly like ml_model.py."""
    df = df_main.copy()

    # ── Main Features ──────────────────────────────────
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

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    df['adx'] = plus_dm.rolling(14).mean() / (atr + 1e-9) * 100
    df['atr_norm'] = atr / (df['close'] + 1e-9) * 100
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_pct'] = (df['close'] - (df['bb_mid'] - 2 * df['bb_std'])) / (4 * df['bb_std'] + 1e-9)
    df['dist_ema_fast'] = (df['close'] - df['ema_9']) / (df['ema_9'] + 1e-9) * 100
    df['dist_ema_slow'] = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-9) * 100

    # ── Lower & Higher Timeframe Sync ────────────────────────────────
    df_lower_feat = pd.DataFrame(index=df.index)
    df_lower_feat['rsi_lower'] = calc_rsi(df_lower['close']).reindex(df.index, method='ffill')
    vol_lower_mean = df_lower['volume'].rolling(20).mean()
    df_lower_feat['vol_lower_spike'] = (df_lower['volume'] / (vol_lower_mean + 1e-9)).reindex(df.index, method='ffill')

    df_higher_feat = pd.DataFrame(index=df.index)
    df_higher_feat['rsi_higher'] = calc_rsi(df_higher['close']).reindex(df.index, method='ffill')
    ema_higher_f = df_higher['close'].ewm(span=9).mean()
    ema_higher_s = df_higher['close'].ewm(span=21).mean()
    df_higher_feat['trend_higher'] = (ema_higher_f > ema_higher_s).astype(int).replace(0, -1).reindex(df.index, method='ffill')

    df = pd.concat([df, df_lower_feat, df_higher_feat], axis=1)

    df['rsi_div'] = df['rsi'].diff() - df['close'].diff() / df['close']
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

    # ── [MTF] 6 Confluence Features ──────────────────────────
    # Direction of each TF: +1 = bullish, -1 = bearish
    # 15m (main): EMA9 vs EMA21
    dir_15m = np.where(df['ema_9'] > df['ema_21'], 1, -1)

    # 5m (lower): EMA9 vs EMA21 on lower TF, reindexed to main
    ema_lower_f = df_lower['close'].ewm(span=9, adjust=False).mean()
    ema_lower_s = df_lower['close'].ewm(span=21, adjust=False).mean()
    dir_5m_series = np.where(ema_lower_f > ema_lower_s, 1, -1)
    dir_5m_series = pd.Series(dir_5m_series, index=df_lower.index).reindex(df.index, method='ffill').fillna(0)

    # 1h (higher): already have trend_higher (+1/-1)
    dir_1h = df_higher_feat['trend_higher'].fillna(0)

    # mtf_alignment_score: -3 = all SHORT, +3 = all LONG
    df['mtf_alignment_score'] = dir_15m + dir_5m_series.values + dir_1h.values

    # tf_1h_15m_agree: 1 if 1h and 15m same direction
    df['tf_1h_15m_agree'] = (dir_1h.values == dir_15m).astype(int)

    # tf_15m_5m_agree: 1 if 15m and 5m same direction
    df['tf_15m_5m_agree'] = (dir_15m == dir_5m_series.values).astype(int)

    # tf_all_bearish / tf_all_bullish: all 3 TFs aligned
    df['tf_all_bearish'] = ((df['mtf_alignment_score'] == -3)).astype(int)
    df['tf_all_bullish'] = ((df['mtf_alignment_score'] ==  3)).astype(int)

    # higher_trend_strength: absolute EMA gap on 1h (how strong the HTF trend is, %)
    ema_h9  = df_higher['close'].ewm(span=9,  adjust=False).mean()
    ema_h21 = df_higher['close'].ewm(span=21, adjust=False).mean()
    htf_strength = ((ema_h9 - ema_h21) / (ema_h21 + 1e-9) * 100).abs()
    df['higher_trend_strength'] = htf_strength.reindex(df.index, method='ffill').fillna(0)

    # ── [FIX-FIB] Fibonacci Retracement Levels (50-candle window) ──
    rolling_high = df['high'].rolling(window=50, min_periods=1).max()
    rolling_low = df['low'].rolling(window=50, min_periods=1).min()
    range_diff = rolling_high - rolling_low
    
    fib_0_0 = rolling_low
    fib_236 = rolling_low + 0.236 * range_diff
    fib_382 = rolling_low + 0.382 * range_diff
    fib_500 = rolling_low + 0.500 * range_diff
    fib_618 = rolling_low + 0.618 * range_diff
    fib_786 = rolling_low + 0.786 * range_diff
    fib_1_0 = rolling_high
    
    df['fib_dist_0_0'] = (df['close'] - fib_0_0) / (df['close'] + 1e-9)
    df['fib_dist_236'] = (df['close'] - fib_236) / (df['close'] + 1e-9)
    df['fib_dist_382'] = (df['close'] - fib_382) / (df['close'] + 1e-9)
    df['fib_dist_500'] = (df['close'] - fib_500) / (df['close'] + 1e-9)
    df['fib_dist_618'] = (df['close'] - fib_618) / (df['close'] + 1e-9)
    df['fib_dist_786'] = (df['close'] - fib_786) / (df['close'] + 1e-9)
    df['fib_dist_1_0'] = (df['close'] - fib_1_0) / (df['close'] + 1e-9)

    return df


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create multi-class labels based on FUTURE price movement.
    1: LONG  (price rises > threshold N candles ahead)
    2: SHORT (price falls > threshold N candles ahead)
    0: HOLD  (otherwise)
    """
    future_return = df["close"].shift(-LABEL_HORIZON) / df["close"] - 1

    labels = np.zeros(len(df))
    labels[future_return > LABEL_THRESHOLD] = 1
    labels[future_return < -LABEL_THRESHOLD] = 2
    return pd.Series(labels, index=df.index)


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Create sequence windows from a 2D feature matrix and 1D label array."""
    X_seq = []
    y_seq = []
    for i in range(seq_len - 1, len(X)):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_pytorch_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, 
                        epochs: int = 60, batch_size: int = 128, device: str = "cpu",
                        cnn_channels: int = 32, lstm_hidden: int = 64, dropout: float = 0.2,
                        lr: float = 0.002, weight_decay: float = 1e-4) -> tuple[nn.Module, float]:
    """
    Train the DayTradingCLSTM neural network.
    Maps classes: Class 1 (LONG) -> 0, Class 2 (SHORT) -> 1.
    If val data is provided, performs early stopping/model checkpointing based on best macro F1-score.
    """
    from bot.nn_model import DayTradingCLSTM
    # Map classes from [1, 2] to [0, 1] for CrossEntropyLoss
    y_train_mapped = (y_train - 1).astype(int)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_mapped, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]

    model = DayTradingCLSTM(
        input_dim=input_dim,
        seq_len=seq_len,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        dropout=dropout
    ).to(device)
    
    # Calculate class weights for CrossEntropyLoss
    class_counts = np.bincount(y_train_mapped)
    total_samples = len(y_train_mapped)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-9)
    weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                val_outputs = model(X_val_t)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_preds_orig = val_preds + 1 # map back to 1 and 2
                
                from sklearn.metrics import f1_score
                val_f1 = f1_score(y_val, val_preds_orig, average="macro", zero_division=0)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if X_val is not None and y_val is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        return model, best_val_f1
    else:
        return model, 0.0


def predict_pytorch(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predict classes using the PyTorch model. Returns original labels [1, 2]."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_t)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds + 1



def main():
    print("=" * 60)
    print("🤖 Binance Scalping Bot — Deep Learning Auto-Tuner (PyTorch)")
    print("=" * 60)

    try:
        from binance.client import Client
    except ImportError:
        print("❌ Install python-binance: pip install python-binance")
        return

    # Check SMOTE availability
    try:
        from imblearn.over_sampling import SMOTE
        smote_available = True
        print("✅ imbalanced-learn found — SMOTE will be used for training balance")
    except ImportError:
        print("⚠️  imbalanced-learn not found. Install: pip install imbalanced-learn")
        print("   Continuing without SMOTE (weighted CrossEntropyLoss will compensate)")
        smote_available = False

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler

    # Configure hardware device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Training device: {device.upper()}")

    print(f"\n📡 Connecting to Binance...")
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=TESTNET)

    # ── Collect training data ─────────────────────────────────
    all_X, all_y = [], []
    print(f"\n📊 Fetching {TRAINING_DAYS} days of data for {len(TRAINING_PAIRS)} pairs...")

    for symbol in TRAINING_PAIRS:
        print(f"   📥 {symbol}...", end="\r")
        df_main   = fetch_klines(client, symbol, interval=CANDLE_INTERVAL,  days=TRAINING_DAYS)
        df_lower  = fetch_klines(client, symbol, interval=LOWER_INTERVAL,   days=TRAINING_DAYS)
        df_higher = fetch_klines(client, symbol, interval=HIGHER_INTERVAL,  days=TRAINING_DAYS)

        if len(df_main) < MIN_CANDLES or len(df_lower) < MIN_CANDLES or len(df_higher) < MIN_CANDLES:
            print(f"   ⚠️  {symbol}: insufficient data, skipping")
            continue

        df = build_features(df_main, df_lower, df_higher)
        df["label"] = build_labels(df)
        df.dropna(inplace=True)

        X = df[FEATURE_COLS].values
        y = df["label"].values
        # Remove last LABEL_HORIZON rows — their labels use future data not yet available
        X = X[:-LABEL_HORIZON]
        y = y[:-LABEL_HORIZON]

        if len(X) < MIN_CANDLES:
            continue

        all_X.append(X)
        all_y.append(y)
        print(f"   ✅ {symbol}: {len(X):,} samples")

    if not all_X:
        print("❌ No training data collected. Check API keys and pairs.")
        return

    # Fit final scaler globally on stacked 2D feature matrices
    print("\n📐 Fitting global RobustScaler on all 2D features...")
    final_scaler = RobustScaler()
    temp_X_stacked = np.vstack(all_X)
    final_scaler.fit(temp_X_stacked)

    # Build sequence windows per symbol (to avoid boundary crossing)
    seq_len = getattr(config, 'ML_SEQUENCE_LENGTH', 10)
    print(f"🔄 Building overlapping sequence windows (seq_len={seq_len}) per symbol...")
    all_seq_X, all_seq_y = [], []
    for i, X in enumerate(all_X):
        y = all_y[i]
        X_scaled = final_scaler.transform(X)
        X_seq, y_seq = create_sequences(X_scaled, y, seq_len=seq_len)
        if len(X_seq) > 0:
            all_seq_X.append(X_seq)
            all_seq_y.append(y_seq)

    X_all = np.vstack(all_seq_X)      # shape: [Total_Seqs, seq_len, feature_dim]
    y_all = np.concatenate(all_seq_y)  # shape: [Total_Seqs]

    # Filter out HOLD (0) samples for binary classification
    print(f"\n⚡ Filtering out HOLD (0) class samples for binary classification...")
    mask = y_all != 0
    X_all = X_all[mask]
    y_all = y_all[mask]

    long_rate  = (y_all == 1).mean()
    short_rate = (y_all == 2).mean()
    print(f"   LONG (1): {long_rate:.1%} | SHORT (2): {short_rate:.1%}")
    print(f"📈 Total Signal-only sequence samples: {len(X_all):,}")

    # ── Hyperparameter Tuning Grid Search (C-LSTM version) ────
    param_grid = [
        {"cnn_channels": 16, "lstm_hidden": 32, "dropout": 0.15, "lr": 0.002, "weight_decay": 1e-4},
        {"cnn_channels": 32, "lstm_hidden": 64, "dropout": 0.20, "lr": 0.001, "weight_decay": 1e-4},
        {"cnn_channels": 32, "lstm_hidden": 64, "dropout": 0.25, "lr": 0.002, "weight_decay": 1e-3},
        {"cnn_channels": 64, "lstm_hidden": 128, "dropout": 0.20, "lr": 0.001, "weight_decay": 1e-4},
    ]

    print(f"\n🔍 Tuning hyperparameters across {len(param_grid)} C-LSTM configurations...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    best_config = None
    best_cv_score = -1.0

    for idx, params in enumerate(param_grid):
        print(f"\n⚙️ Config {idx+1}/{len(param_grid)}: cnn_channels={params['cnn_channels']}, lstm_hidden={params['lstm_hidden']}, drop={params['dropout']}, lr={params['lr']}, wd={params['weight_decay']}")
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
            X_train_cv, X_val_cv = X_all[train_idx], X_all[val_idx]
            y_train_cv, y_val_cv = y_all[train_idx], y_all[val_idx]

            _, fold_f1 = train_pytorch_model(
                X_train_cv, y_train_cv, 
                X_val_cv, y_val_cv,
                epochs=25, device=device,
                cnn_channels=params['cnn_channels'],
                lstm_hidden=params['lstm_hidden'],
                dropout=params['dropout'],
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            cv_scores.append(fold_f1)
            
        mean_f1 = np.mean(cv_scores)
        print(f"👉 Mean CV F1-score: {mean_f1:.4f} (Folds: {[f'{score:.4f}' for score in cv_scores]})")
        
        if mean_f1 > best_cv_score:
            best_cv_score = mean_f1
            best_config = params

    print(f"\n🏆 Best Hyperparameters Found:")
    for k, v in best_config.items():
        print(f"   • {k}: {v}")
    print(f"   • Best CV F1-score: {best_cv_score:.4f}")

    # ── Walk-forward validation check (last 20%) using the best hyperparameters ──────
    split_idx = int(len(X_all) * 0.80)
    X_train_final, X_test_final = X_all[:split_idx], X_all[split_idx:]
    y_train_final, y_test_final = y_all[:split_idx], y_all[split_idx:]

    print(f"\n📊 Running final validation with best config (train: {len(X_train_final):,} | test: {len(X_test_final):,} most recent)")
    
    wf_model, wf_f1 = train_pytorch_model(
        X_train_final, y_train_final,
        X_test_final, y_test_final,
        epochs=40, device=device,
        cnn_channels=best_config['cnn_channels'],
        lstm_hidden=best_config['lstm_hidden'],
        dropout=best_config['dropout'],
        lr=best_config['lr'],
        weight_decay=best_config['weight_decay']
    )

    from sklearn.metrics import classification_report, f1_score
    y_pred = predict_pytorch(wf_model, X_test_final, device=device)
    wf_f1_val = f1_score(y_test_final, y_pred, average='macro', zero_division=0)
    print(f"   Walk-forward F1: {wf_f1_val:.4f}")
    print("\n" + classification_report(y_test_final, y_pred, target_names=["LONG", "SHORT"], zero_division=0))

    # ── Fit final model on ALL data using the best hyperparameters ───────────
    print("🏆 Fitting final deep learning model on complete dataset...")

    # Train on full data. We run for 40 epochs.
    best_model, _ = train_pytorch_model(
        X_all, y_all, 
        epochs=40, device=device,
        cnn_channels=best_config['cnn_channels'],
        lstm_hidden=best_config['lstm_hidden'],
        dropout=best_config['dropout'],
        lr=best_config['lr'],
        weight_decay=best_config['weight_decay']
    )

    # ── Save model & scaler ───────────────────────────────────
    os.makedirs("models", exist_ok=True)
    
    # Save RobustScaler
    with open(SCALER_OUTPUT, "wb") as f:
        pickle.dump(final_scaler, f)
    
    # Save PyTorch Model with metadata dict
    checkpoint = {
        "state_dict": best_model.state_dict(),
        "cnn_channels": best_config["cnn_channels"],
        "lstm_hidden": best_config["lstm_hidden"],
        "dropout": best_config["dropout"],
        "seq_len": seq_len,
        "input_dim": X_all.shape[2]
    }
    torch.save(checkpoint, MODEL_OUTPUT)

    print(f"\n✅ DL Model saved to: {MODEL_OUTPUT}")
    print(f"✅ RobustScaler saved to: {SCALER_OUTPUT}")
    print(f"🚀 Upload the above two files and bot code to the VPS, then run 'pm2 restart bot_crypto'!")
    print("=" * 60)


if __name__ == "__main__":
    main()
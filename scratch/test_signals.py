import os
import sys
import pandas as pd
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.exchange import Exchange
from bot.strategy import Strategy
from bot.ml_model import MLPredictor
import config

def main():
    print("=" * 60)
    print("🔍 Testing Technical Signals and ML Confirmation")
    print("=" * 60)

    # Enable Testnet config so we don't need valid live API keys for public data
    config.TESTNET = True
    exchange = Exchange()
    strategy = Strategy()
    predictor = MLPredictor()

    if not predictor.is_ready:
        print("❌ ML Predictor is not ready!")
        return

    # Watchlist pairs from the user's log
    watchlist = [
        "UTKUSDT", "MITOUSDT", "FIDAUSDT", "NILUSDT", "1000CHEEMSUSDT", 
        "JTOUSDT", "AVNTUSDT", "ALTUSDT", "ZECUSDT", "2ZUSDT", 
        "BBUSDT", "DASHUSDT", "SAPIENUSDT", "LRCUSDT", "SAHARAUSDT"
    ]

    print(f"ML Confidence Threshold: {config.ML_CONFIDENCE_THRESHOLD}")
    print(f"Futures Enabled: {config.FUTURES_ENABLED}")
    print("-" * 60)
    print(f"{'Symbol':<15} | {'Tech Signal':<12} | {'ML Pred':<10} | {'ML Conf':<7} | {'Confirmed?':<10} | {'Reason if blocked'}")
    print("-" * 60)

    for symbol in watchlist:
        try:
            # 1. Fetch main, lower, higher klines
            df_main = exchange.get_klines(symbol, interval=config.CANDLE_INTERVAL)
            if df_main.empty:
                print(f"{symbol:<15} | Empty main klines")
                continue

            # Calculate technical signal
            df_feat = strategy.calculate(df_main)
            tech_signal, details = strategy.get_signal(df_feat)
            tech_signal = tech_signal or "HOLD"

            # Predict ML if tech signal is BUY/SELL
            ml_class, ml_conf = 0, 0.0
            if tech_signal in ("BUY", "SELL"):
                df_lower = exchange.get_klines(symbol, interval=config.LOWER_INTERVAL)
                df_higher = exchange.get_klines(symbol, interval=config.HIGHER_INTERVAL)
                if not df_lower.empty and not df_higher.empty:
                    ml_class, ml_conf = predictor.predict(df_main, df_lower, df_higher)

            pred_text = "HOLD ⚪"
            if ml_class == 1: pred_text = "LONG 🟢"
            if ml_class == 2: pred_text = "SHORT 🔴"

            # Check confirmation
            is_confirmed = (tech_signal == "BUY" and ml_class == 1) or (tech_signal == "SELL" and ml_class == 2)
            
            # Determine status/block reason
            status = "NO"
            reason = "No Tech Signal"
            
            if tech_signal in ("BUY", "SELL"):
                if not is_confirmed:
                    reason = f"Mismatch (Tech: {tech_signal}, ML: {pred_text})"
                elif ml_conf < config.ML_CONFIDENCE_THRESHOLD:
                    reason = f"Low Confidence ({ml_conf:.2f} < {config.ML_CONFIDENCE_THRESHOLD})"
                else:
                    is_f = exchange.is_futures(symbol)
                    if not is_f and tech_signal == "SELL":
                        reason = "Spot cannot Short"
                    else:
                        status = "YES ✅"
                        reason = "Ready"

            print(f"{symbol:<15} | {tech_signal:<12} | {pred_text:<10} | {ml_conf:.2f} | {status:<10} | {reason}")

        except Exception as e:
            print(f"{symbol:<15} | Error: {e}")

if __name__ == "__main__":
    main()

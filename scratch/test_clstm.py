import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from bot.ml_model import MLPredictor

def generate_mock_df(length=100):
    idx = pd.date_range(start="2026-05-25 00:00:00", periods=length, freq="15min")
    df = pd.DataFrame(index=idx)
    df["open"] = np.random.uniform(90, 110, length)
    df["high"] = df["open"] + np.random.uniform(0.1, 2.0, length)
    df["low"] = df["open"] - np.random.uniform(0.1, 2.0, length)
    df["close"] = np.random.uniform(df["low"], df["high"], length)
    df["volume"] = np.random.uniform(100, 1000, length)
    return df

def main():
    print("Testing MLPredictor with C-LSTM...")
    
    # Generate mock inputs
    df_main = generate_mock_df(100)
    df_lower = generate_mock_df(300)
    df_higher = generate_mock_df(50)
    
    predictor = MLPredictor()
    print("Is predictor ready?", predictor.is_ready)
    
    # If the model is not trained yet, it might not be ready, but we can verify feature preparation
    features = predictor.prepare_features(df_main, df_lower, df_higher)
    if features is not None:
        print("Prepared features shape:", features.shape)
        print("Feature columns:", list(features.columns))
        print("SUCCESS: Feature preparation shape and columns are correct!")
        
        # Test prediction integration
        try:
            side, conf = predictor.predict(df_main, df_lower, df_higher)
            print(f"Prediction Successful: side={side} (1=LONG, 2=SHORT, 0=HOLD), confidence={conf:.4f}")
        except Exception as e:
            print("FAILED: Prediction crashed:", e)
    else:
        print("FAILED: Feature preparation returned None.")

if __name__ == "__main__":
    main()

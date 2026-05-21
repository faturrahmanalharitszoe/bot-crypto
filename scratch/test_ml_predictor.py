import os
import sys
import pandas as pd
import numpy as np

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.ml_model import MLPredictor, FEATURE_COLS

def main():
    print("=" * 60)
    print("🤖 MLPredictor Verification Test")
    print("=" * 60)
    
    # 1. Initialize MLPredictor
    predictor = MLPredictor()
    
    # 2. Check if ready
    print(f"Is predictor ready? {predictor.is_ready}")
    if not predictor.is_ready:
        print("❌ Predictor is not ready. Check logs and paths.")
        return
        
    # 3. Check model architecture metadata
    model = predictor.model
    print("\n📦 Loaded Model Architecture:")
    print(f"  • Hidden Dimensions: {getattr(model, 'hidden_dims', 'Default')}")
    print(f"  • Dropout: {getattr(model, 'dropout', 'Default')}")
    print(f"  • Activation: {getattr(model, 'activation', 'Default')}")
    print(f"  • Underlyng Network:\n{model.network}")
    
    # 4. Perform a dry-run prediction with dummy input
    print("\n⚡ Running mock prediction inference...")
    # Generate 100 rows of dummy data matching standard OHLCV
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5m')
    dummy_data = {
        'open': np.random.uniform(100, 105, 100),
        'high': np.random.uniform(105, 110, 100),
        'low': np.random.uniform(95, 100, 100),
        'close': np.random.uniform(99, 104, 100),
        'volume': np.random.uniform(10, 100, 100),
    }
    df_main = pd.DataFrame(dummy_data, index=dates)
    
    # Run predict
    signal, conf = predictor.predict(df_main)
    print(f"👉 Inference Result:")
    print(f"  • Signal: {signal} (0=HOLD, 1=LONG, 2=SHORT)")
    print(f"  • Confidence: {conf:.4f}")
    
    print("\n✅ Verification Test Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

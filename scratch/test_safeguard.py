import sys
import pandas as pd
import numpy as np
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bot")

# Import real config first
import config
config.ML_ENABLED = True
config.ML_FULL_AI_MODE = True
config.ML_CONFIDENCE_THRESHOLD = 0.55
config.FUNDING_FEE_ENABLED = False
config.SENTIMENT_ENABLED = False
config.CANDLE_CONFIRM_ENABLED = False
config.DRIFT_PROTECTION_ENABLED = False
config.BLACKLIST_ENABLED = False

# Now import Strategy and Trader
from bot.strategy import Strategy
from bot.trader import Trader

class MockExchange:
    def is_futures(self, symbol):
        return True
    
    def get_symbol_info(self, symbol):
        return {
            "step_size": 0.001,
            "min_qty": 0.001,
            "min_notional": 5.0,
            "base_precision": 3,
        }

    def get_current_price(self, symbol):
        return 101.0

    def place_market_order(self, symbol, side, quantity):
        return {"status": "FILLED"}

    def get_filled_price(self, order):
        return 101.0

class MockSentiment:
    def is_blocked(self, symbol):
        return False

class MockRiskManager:
    def __init__(self):
        self.positions = {}
        
    def has_position_for(self, symbol):
        return False
        
    def calculate_quantity(self, usdt_balance, current_price, symbol_info, *args, **kwargs):
        return 1.0

    def open_position(self, symbol, entry_price, quantity, side, is_futures, *args, **kwargs):
        class MockPosition:
            take_profit = entry_price * 1.015
            stop_loss = entry_price * 0.992
        return MockPosition()

def make_mock_df(prices, rsi_val):
    # Create mock candles
    dates = pd.date_range(end="2026-05-23", periods=len(prices), freq="15min")
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * len(prices)
    }, index=dates)
    
    # We will manually assign indicators to match the mock
    df['rsi'] = rsi_val
    df['ema_fast'] = prices
    df['ema_slow'] = prices
    df['volume_avg'] = 1000.0
    df['macd'] = 0.0
    df['macd_signal'] = 0.0
    return df

class TestStrategy(Strategy):
    def calculate(self, df):
        # Return df directly since we pre-assigned indicators
        return df

def test_safeguards():
    print("Initializing Mock Trader...")
    trader = Trader.__new__(Trader)  # Create instance without running __init__
    trader.exchange = MockExchange()
    trader.sentiment = MockSentiment()
    trader.risk_manager = MockRiskManager()
    trader.cooldowns = {}
    trader.cooldown_minutes = 15
    trader.max_positions = 2
    trader.strategy = TestStrategy()
    trader.notifier = None
    trader._blacklist = {}
    trader._last_exit_price = {}
    trader._last_exit_reason = {}
    
    # Enable ML
    trader._ml_scores = {"BTCUSDT": (1, 0.60)}  # AI predicts LONG (class 1) with 60% confidence
    
    print("\n--- TEST CASE 1: Falling Knife (RSI = 30, prices decreasing) ---")
    # Last completed candle (index -2) close is 100. Live candle (index -1) close is 99 (falling)
    df_falling = make_mock_df(prices=[105, 104, 103, 102, 101, 100, 99], rsi_val=30.0)
    
    # Evaluate
    result = trader._evaluate_pair_with_data(
        symbol="BTCUSDT",
        usdt_balance=100.0,
        df_main=df_falling,
        df_lower=None,
        df_higher=None
    )
    print(f"Result (should be False): {result}")
    assert result is False, "Failed: Falling knife should be blocked!"
    print("✅ Case 1 passed: Falling knife successfully blocked.")

    print("\n--- TEST CASE 2: Bouncing Price (RSI = 30, live price bounces to 101) ---")
    # Last completed candle (index -2) close is 100. Live candle (index -1) close is 101 (bouncing)
    df_bouncing = make_mock_df(prices=[105, 104, 103, 102, 101, 100, 101], rsi_val=30.0)
    
    # Evaluate
    result = trader._evaluate_pair_with_data(
        symbol="BTCUSDT",
        usdt_balance=100.0,
        df_main=df_bouncing,
        df_lower=None,
        df_higher=None
    )
    print(f"Result (should be True): {result}")
    assert result is True, "Failed: Bouncing price should trigger entry!"
    print("✅ Case 2 passed: Bouncing price successfully triggered entry.")

    print("\n--- TEST CASE 3: Overbought Rising (RSI = 70, prices increasing) ---")
    trader._ml_scores = {"BTCUSDT": (2, 0.60)}  # AI predicts SHORT (class 2) with 60% confidence
    # Last completed candle close is 100. Live candle close is 101 (still rising)
    df_rising = make_mock_df(prices=[95, 96, 97, 98, 99, 100, 101], rsi_val=70.0)
    
    # Evaluate
    result = trader._evaluate_pair_with_data(
        symbol="BTCUSDT",
        usdt_balance=100.0,
        df_main=df_rising,
        df_lower=None,
        df_higher=None
    )
    print(f"Result (should be False): {result}")
    assert result is False, "Failed: Overbought rising price should be blocked!"
    print("✅ Case 3 passed: Overbought rising price successfully blocked.")

    print("\n--- TEST CASE 4: Overbought Rejection (RSI = 70, live price rejects to 99) ---")
    # Last completed candle close is 100. Live candle close is 99 (rejecting)
    df_rejection = make_mock_df(prices=[95, 96, 97, 98, 99, 100, 99], rsi_val=70.0)
    
    # Evaluate
    result = trader._evaluate_pair_with_data(
        symbol="BTCUSDT",
        usdt_balance=100.0,
        df_main=df_rejection,
        df_lower=None,
        df_higher=None
    )
    print(f"Result (should be True): {result}")
    assert result is True, "Failed: Overbought rejection should trigger entry!"
    print("✅ Case 4 passed: Overbought rejection successfully triggered entry.")

if __name__ == "__main__":
    test_safeguards()

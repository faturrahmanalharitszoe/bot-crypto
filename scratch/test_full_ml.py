import os
import sys
import logging

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from bot.exchange import Exchange
from bot.trader import Trader
import config

# Set config to Full ML Mode
config.ML_ENABLED = True
config.ML_FULL_AI_MODE = True
config.TESTNET = True # force testnet for safety check

print("Initializing Exchange...")
exchange = Exchange()

# Mock place_market_order so no trades actually occur during test
def mock_place_market_order(symbol, side, quantity):
    print(f"\n[DRY RUN MOCK] Would place market order: {side} {quantity} {symbol}")
    return None

def mock_get_usdt_balance():
    return 100.0

exchange.place_market_order = mock_place_market_order
exchange.get_usdt_balance = mock_get_usdt_balance

print("Initializing Trader...")
trader = Trader(exchange=exchange)

print(f"ML Predictor Ready: {trader.ml_predictor.is_ready}")
print(f"ML Confidence Threshold: {config.ML_CONFIDENCE_THRESHOLD}")
print(f"Watchlist: {trader.watchlist}")

print("\n--- Running Single Scan (Full ML Mode) ---")
trader._scan_for_entry()
print("--- Scan Finished ---")

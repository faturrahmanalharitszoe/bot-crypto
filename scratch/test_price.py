import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()

from bot.exchange import Exchange
import config

config.FUTURES_ENABLED = True
exchange = Exchange()

symbol = "LRCUSDT"
try:
    ticker = exchange.client.futures_symbol_ticker(symbol=symbol)
    print(f"futures_symbol_ticker({symbol}):", ticker)
except Exception as e:
    print(f"futures_symbol_ticker({symbol}) failed:", e)

try:
    ticker2 = exchange.client.get_symbol_ticker(symbol=symbol)
    print(f"get_symbol_ticker({symbol}):", ticker2)
except Exception as e:
    print(f"get_symbol_ticker({symbol}) failed:", e)

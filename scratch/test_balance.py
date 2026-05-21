import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()

from bot.exchange import Exchange
import config

config.FUTURES_ENABLED = True
exchange = Exchange()

try:
    info = exchange.client.futures_account_balance()
    usdt_asset = None
    for asset in info:
        if asset["asset"] == "USDT":
            usdt_asset = asset
            break
    print("USDT Asset Info:", usdt_asset)
    print("get_usdt_balance():", exchange.get_usdt_balance())
except Exception as e:
    print("Error:", e)

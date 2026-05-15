"""
bot/exchange.py — Thin Binance API wrapper
Handles client init, balance fetching, kline data, and order placement.
Supports both Testnet and Mainnet via config.TESTNET flag.
"""

import math
import time
import logging
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

import config

logger = logging.getLogger("bot")


class Exchange:
    """Wrapper around the python-binance Client."""

    def __init__(self) -> None:
        self.testnet = config.TESTNET

        if not config.API_KEY or not config.API_SECRET:
            raise ValueError(
                "❌ BINANCE_API_KEY and BINANCE_API_SECRET must be set in your .env file!\n"
                "   For testnet keys, visit: https://testnet.binance.vision"
            )

        self.client = Client(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            testnet=self.testnet,
        )

        # ── Sync clock offset with Binance server ─────────
        # Fixes error -1021 "Timestamp outside recvWindow"
        # which happens when PC clock drifts from server time.
        self._sync_server_time()

        mode = "🧪 TESTNET" if self.testnet else "🔴 LIVE"
        logger.info(f"Exchange initialized — Mode: {mode}")
        self._symbol_info_cache: dict = {}

    # ── Server Time Sync ─────────────────────────────────────

    def _sync_server_time(self) -> None:
        """
        Compute the offset between local time and Binance server time,
        then apply it to all outgoing requests.
        This permanently fixes error -1021 (Timestamp outside recvWindow).
        """
        try:
            local_before = int(time.time() * 1000)
            server_time  = self.client.get_server_time()["serverTime"]
            local_after  = int(time.time() * 1000)
            # Round-trip midpoint approximates the server time at request moment
            local_mid    = (local_before + local_after) // 2
            offset_ms    = server_time - local_mid
            self.client.timestamp_offset = offset_ms
            logger.info(f"Clock sync OK — offset: {offset_ms:+d} ms")
        except Exception as e:
            logger.warning(f"Clock sync failed (non-fatal): {e}")

    # ── Balance ──────────────────────────────────────────────

    def get_usdt_balance(self) -> float:
        """Return available USDT balance."""
        try:
            info = self.client.get_asset_balance(asset="USDT")
            return float(info["free"]) if info else 0.0
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch USDT balance: {e}")
            return 0.0

    def get_asset_balance(self, asset: str) -> float:
        """Return available balance for a given asset (e.g. 'BTC')."""
        try:
            info = self.client.get_asset_balance(asset=asset)
            return float(info["free"]) if info else 0.0
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch {asset} balance: {e}")
            return 0.0

    # ── Market Data ──────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol.
        Returns a DataFrame with columns: open, high, low, close, volume.
        """
        target_interval = interval if interval else config.CANDLE_INTERVAL
        try:
            raw = self.client.get_klines(
                symbol=symbol,
                interval=target_interval,
                limit=config.CANDLE_LIMIT,
            )
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        return df[["open", "high", "low", "close", "volume"]]

    def get_current_price(self, symbol: str) -> float:
        """Return the latest price for a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            return 0.0

    def get_24h_tickers(self) -> list:
        """Return 24h ticker stats for all symbols."""
        try:
            return self.client.get_ticker()
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch 24h tickers: {e}")
            return []

    # ── Symbol Info ──────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Return exchange filters for a symbol (cached).
        Extracts: step_size, min_qty, min_notional, price_precision.
        """
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                return None

            parsed = {
                "step_size":      0.001,
                "min_qty":        0.001,
                "min_notional":   5.0,
                "base_precision": 3,
            }

            for f in info.get("filters", []):
                ftype = f.get("filterType")
                if ftype == "LOT_SIZE":
                    parsed["step_size"] = float(f["stepSize"])
                    parsed["min_qty"]   = float(f["minQty"])
                    # Calculate decimal precision from stepSize
                    step = float(f["stepSize"])
                    if step < 1:
                        parsed["base_precision"] = int(round(-math.log10(step)))
                    else:
                        parsed["base_precision"] = 0

                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    parsed["min_notional"] = float(
                        f.get("minNotional", f.get("minNotional", 5.0))
                    )

            self._symbol_info_cache[symbol] = parsed
            return parsed

        except BinanceAPIException as e:
            logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
            return None

    # ── Orders ───────────────────────────────────────────────

    def place_market_buy(self, symbol: str, quantity: float) -> Optional[dict]:
        """Place a market BUY order. Returns the order dict or None on failure."""
        try:
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=quantity,
            )
            logger.info(
                f"🟢 BUY  | {symbol} | Qty: {quantity} | "
                f"Status: {order.get('status')}"
            )
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Market BUY failed for {symbol}: {e}")
            return None

    def place_market_sell(self, symbol: str, quantity: float) -> Optional[dict]:
        """Place a market SELL order. Returns the order dict or None on failure."""
        try:
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=quantity,
            )
            logger.info(
                f"🔴 SELL | {symbol} | Qty: {quantity} | "
                f"Status: {order.get('status')}"
            )
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"Market SELL failed for {symbol}: {e}")
            return None

    def get_filled_price(self, order: dict) -> float:
        """Extract average fill price from an order dict."""
        try:
            fills = order.get("fills", [])
            if fills:
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty  = sum(float(f["qty"]) for f in fills)
                return total_cost / total_qty if total_qty > 0 else 0.0
            # Fallback: cummulative quote qty / executed qty
            exec_qty  = float(order.get("executedQty", 0))
            cum_quote = float(order.get("cummulativeQuoteQty", 0))
            return cum_quote / exec_qty if exec_qty > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

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

        # ── [FIX] Force Futures DEMO URL when in demo/testnet mode ────────
        # The user is using Binance Spot Demo / Futures Demo, which requires
        # the demo futures base endpoint (not the normal futures testnet URL).
        # Using the wrong futures endpoint causes -1109 Invalid account.
        if self.testnet and config.FUTURES_ENABLED:
            # Binance Futures Demo API base endpoint.
            # Docs: https://developers.binance.com/docs/derivatives/
            #
            # ⚠️ CRITICAL: python-binance's _create_futures_api_uri() does:
            #     if self.testnet: url = self.FUTURES_TESTNET_URL
            # So we MUST override FUTURES_TESTNET_URL (not FUTURES_URL) for
            # demo Futures API keys (from demo.binance.com / demo-fapi.binance.com)
            # to work. The default points to testnet.binancefuture.com which
            # is a *different* environment that rejects demo keys with -1109.
            self.client.FUTURES_TESTNET_URL = "https://demo-fapi.binance.com/fapi"
            self.client.FUTURES_DATA_TESTNET_URL = "https://demo-fapi.binance.com/futures/data"
            self.client.FUTURES_COIN_TESTNET_URL = "https://demo-dapi.binance.com/dapi"
            self.client.FUTURES_COIN_DATA_TESTNET_URL = "https://demo-dapi.binance.com/futures/data"
            logger.info(f"Futures demo URL override: {self.client.FUTURES_TESTNET_URL}")

        # ── Sync clock offset with Binance server ─────────
        self._sync_server_time()

        # Cache futures symbols for hybrid mode
        self.futures_symbols = set()
        try:
            f_info = self.client.futures_exchange_info()
            self.futures_symbols = {s['symbol'] for s in f_info['symbols']}
            logger.info(f"✅ Hybrid Engine: {len(self.futures_symbols)} Futures symbols cached.")
        except Exception as e:
            logger.warning(f"Could not fetch Futures symbols: {e}. Falling back to Spot only.")

        # Cache initial testnet wallet balance baseline (persistent across restarts)
        self._initial_testnet_wallet_balance = 5000.0
        if self.testnet:
            current_config_mock = float(getattr(config, "MOCK_TESTNET_BALANCE", 0.0))
            if current_config_mock > 0.0:
                import json
                import os
                state_file = os.path.join("logs", "mock_balance_state.json")
                
                # Fetch current raw collateral balance as a fallback/starting value
                current_raw = 5000.0
                try:
                    if config.FUTURES_ENABLED:
                        info = self.client.futures_account_balance()
                        for asset in info:
                            if asset["asset"] == "USDT":
                                current_raw = float(asset.get("balance") or 5000.0)
                                break
                    else:
                        info = self.client.get_asset_balance(asset="USDT")
                        if info:
                            current_raw = float(info.get("free", 0.0)) + float(info.get("locked", 0.0))
                except Exception as e:
                    logger.warning(f"Could not fetch current raw balance from Binance: {e}")
                    
                # Try to load existing state
                loaded_initial_raw = None
                if os.path.exists(state_file):
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            state_data = json.load(f)
                        # Only reuse the raw baseline if the mock starting balance hasn't changed in the config
                        if state_data.get("mock_starting_balance") == current_config_mock:
                            loaded_initial_raw = state_data.get("initial_raw_balance")
                            logger.info(
                                f"ℹ️ Loaded persistent mock balance baseline. "
                                f"Starting: ${current_config_mock:.2f} USDT | "
                                f"Raw baseline: ${loaded_initial_raw:.2f} USDT"
                            )
                        else:
                            logger.info(
                                f"ℹ️ Mock starting balance in config changed "
                                f"({state_data.get('mock_starting_balance')} -> {current_config_mock}). "
                                f"Resetting baseline."
                            )
                    except Exception as e:
                        logger.warning(f"Could not read mock balance state file: {e}. Resetting baseline.")
                
                if loaded_initial_raw is not None:
                    self._initial_testnet_wallet_balance = loaded_initial_raw
                else:
                    # Save new state
                    self._initial_testnet_wallet_balance = current_raw
                    try:
                        os.makedirs("logs", exist_ok=True)
                        with open(state_file, "w", encoding="utf-8") as f:
                            json.dump({
                                "initial_raw_balance": current_raw,
                                "mock_starting_balance": current_config_mock
                            }, f, indent=4)
                        logger.info(f"ℹ️ Saved new mock balance baseline: starting={current_config_mock:.2f}, raw={current_raw:.2f}")
                    except Exception as e:
                        logger.error(f"Could not write mock balance state: {e}")
            else:
                # Mocking is disabled, just fetch raw balance for baseline
                try:
                    if config.FUTURES_ENABLED:
                        info = self.client.futures_account_balance()
                        for asset in info:
                            if asset["asset"] == "USDT":
                                self._initial_testnet_wallet_balance = float(asset.get("balance") or 5000.0)
                                break
                    else:
                        info = self.client.get_asset_balance(asset="USDT")
                        if info:
                            self._initial_testnet_wallet_balance = float(info.get("free", 0.0)) + float(info.get("locked", 0.0))
                    logger.info(f"ℹ️ Testnet initial wallet balance baseline: ${self._initial_testnet_wallet_balance:.2f} USDT")
                except Exception as e:
                    logger.warning(f"Could not fetch initial testnet balance: {e}. Using 5000.0 as baseline.")

        mode = "🧪 TESTNET" if self.testnet else "🔴 LIVE"
        market = "📈 FUTURES (Hybrid)" if config.FUTURES_ENABLED else "💰 SPOT"
        logger.info(f"Exchange initialized — Mode: {mode} | Market: {market}")
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
        """Return available USDT balance (Spot or Futures)."""
        try:
            raw_balance = 0.0
            if config.FUTURES_ENABLED:
                info = self.client.futures_account_balance()
                # Find USDT in the list of assets
                for asset in info:
                    if asset["asset"] == "USDT":
                        # Try multiple possible keys (withdrawAvailable, availableBalance, balance)
                        raw_balance = float(asset.get("withdrawAvailable") or 
                                            asset.get("availableBalance") or 
                                            asset.get("balance") or 0.0)
                        break
            else:
                info = self.client.get_asset_balance(asset="USDT")
                raw_balance = float(info["free"]) if info else 0.0

            if self.testnet and getattr(config, "MOCK_TESTNET_BALANCE", 0.0) > 0.0:
                initial = getattr(self, "_initial_testnet_wallet_balance", 5000.0)
                # Mock balance fluctuates relative to starting wallet balance
                mocked = float(config.MOCK_TESTNET_BALANCE) + (raw_balance - initial)
                return max(0.0, mocked)

            return raw_balance
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch USDT balance: {e}")
            if self.testnet and getattr(config, "MOCK_TESTNET_BALANCE", 0.0) > 0.0:
                return float(config.MOCK_TESTNET_BALANCE)
            return 0.0


    def get_asset_balance(self, asset: str) -> float:
        """Return available balance for a given asset (e.g. 'BTC')."""
        try:
            info = self.client.get_asset_balance(asset=asset)
            return float(info["free"]) if info else 0.0
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch {asset} balance: {e}")
            return 0.0

    # ── Funding Fee ───────────────────────────────────────────

    def get_funding_info(self, symbol: str) -> dict:
        """
        Fetch current funding rate and time until next funding for a Futures symbol.

        Returns:
            {
                "rate":           float   — current funding rate (e.g. 0.0001 = 0.01%)
                "next_funding_ms": int    — next funding timestamp in milliseconds
                "minutes_until":  float  — minutes until next funding event
                "available":      bool   — False if symbol is Spot-only or fetch failed
            }
        """
        default = {"rate": 0.0, "next_funding_ms": 0, "minutes_until": 999.0, "available": False}
        if not self.is_futures(symbol):
            return default
        try:
            data = self.client.futures_mark_price(symbol=symbol)
            rate            = float(data.get("lastFundingRate", 0))
            next_ms         = int(data.get("nextFundingTime", 0))
            now_ms          = int(time.time() * 1000)
            minutes_until   = max(0.0, (next_ms - now_ms) / 60_000)
            return {
                "rate":            rate,
                "next_funding_ms": next_ms,
                "minutes_until":   round(minutes_until, 1),
                "available":       True,
            }
        except Exception as e:
            logger.debug(f"Funding info fetch failed for {symbol}: {e}")
            return default

    def get_mark_price(self, symbol: str) -> float:
        """Return the mark price for a Futures symbol (used for funding fee calculation)."""
        if not self.is_futures(symbol):
            return self.get_current_price(symbol)
        try:
            data = self.client.futures_mark_price(symbol=symbol)
            return float(data.get("markPrice", 0))
        except Exception:
            return self.get_current_price(symbol)

    # ── Market Data ──────────────────────────────────────────

    def get_klines(self, symbol: str, interval: str = None) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a symbol.
        Returns a DataFrame with columns: open, high, low, close, volume.
        """
        target_interval = interval if interval else config.CANDLE_INTERVAL
        try:
            # Smart check: Use futures if enabled AND symbol exists there
            use_f = config.FUTURES_ENABLED and self.is_futures(symbol)
            
            if use_f:
                raw = self.client.futures_klines(
                    symbol=symbol,
                    interval=target_interval,
                    limit=config.CANDLE_LIMIT,
                )
            else:
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
        """Return the latest price for a symbol (Hybrid)."""
        try:
            use_f = config.FUTURES_ENABLED and self.is_futures(symbol)
            ticker = None
            if use_f:
                try:
                    ticker = self.client.futures_symbol_ticker(symbol=symbol)
                except Exception as e:
                    logger.warning(f"Futures price fetch failed for {symbol}: {e}. Trying Spot fallback.")
            
            # Fallback to Spot ticker if Futures was not fetched or returned empty/invalid response
            if not ticker or not isinstance(ticker, dict) or "price" not in ticker:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
            
            if not isinstance(ticker, dict) or "price" not in ticker:
                logger.error(f"Failed to fetch price for {symbol}: ticker response invalid or missing 'price': {ticker}")
                return 0.0
                
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            return 0.0

    def get_24h_tickers(self) -> list:
        """Return 24h ticker stats from Spot (which includes most coins)."""
        try:
            # We use Spot tickers as the base because it covers almost everything
            return self.client.get_ticker()
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch 24h tickers: {e}")
            return []

    def is_futures(self, symbol: str) -> bool:
        """Check if a symbol is available on the Futures market."""
        return symbol in self.futures_symbols

    def get_active_symbols(self) -> Optional[set]:
        """Return a set of all symbols that are currently in TRADING status on Spot and/or Futures."""
        active = set()
        spot_success = False
        futures_success = False
        try:
            spot_info = self.client.get_exchange_info()
            for s in spot_info.get('symbols', []):
                if s.get('status') == 'TRADING':
                    active.add(s['symbol'])
            spot_success = True
        except Exception as e:
            logger.warning(f"Failed to fetch active Spot symbols: {e}")

        try:
            if config.FUTURES_ENABLED:
                futures_info = self.client.futures_exchange_info()
                for s in futures_info.get('symbols', []):
                    if s.get('status') == 'TRADING':
                        active.add(s['symbol'])
                futures_success = True
            else:
                futures_success = True
        except Exception as e:
            logger.warning(f"Failed to fetch active Futures symbols: {e}")
            
        if not spot_success and not futures_success:
            return None
        if not active:
            return None
        return active


    # ── Symbol Info ──────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Return exchange filters for a symbol (cached).
        """
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        try:
            use_f = config.FUTURES_ENABLED and self.is_futures(symbol)
            if use_f:
                try:
                    exchange_info = self.client.futures_exchange_info(symbol=symbol)
                except Exception:
                    exchange_info = self.client.futures_exchange_info()
            else:
                try:
                    exchange_info = self.client.get_exchange_info(symbol=symbol)
                except Exception:
                    exchange_info = self.client.get_exchange_info()
                
            info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if not info:
                self._symbol_info_cache[symbol] = None
                return None

            if info.get("status", "TRADING") != "TRADING":
                logger.warning(f"⚠️  Symbol {symbol} is not in TRADING status (current status: {info.get('status')}). Skipping.")
                self._symbol_info_cache[symbol] = None
                return None

            parsed = {
                "step_size":      0.001,
                "min_qty":        0.001,
                "min_notional":   5.0,
                "base_precision": 3,
            }

            for f in info.get("filters", []):
                ftype = f.get("filterType")
                if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    parsed["step_size"] = float(f["stepSize"])
                    parsed["min_qty"]   = float(f["minQty"])
                    step = float(f["stepSize"])
                    if 0 < step < 1:
                        try:
                            parsed["base_precision"] = int(round(-math.log10(step)))
                        except ValueError:
                            parsed["base_precision"] = 0
                    else:
                        parsed["base_precision"] = 0
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    parsed["min_notional"] = float(f.get("minNotional", f.get("notional", 5.0)))


            self._symbol_info_cache[symbol] = parsed
            return parsed

        except Exception as e:
            logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
            return None

    # ── Futures Specific ─────────────────────────────────────

    def setup_futures_symbol(self, symbol: str) -> bool:
        """Set leverage and margin type for a futures symbol.

        On Binance Futures Demo, margin type / leverage changes via API often
        return -1109 ("Invalid account") even when auth is valid. In that case
        we treat the call as a soft failure and proceed with the order using
        whatever defaults the demo account has configured (set in the UI).
        """
        if not config.FUTURES_ENABLED: return True

        soft_fail = False  # True if API rejected config but auth is fine

        try:
            # 1. Set Margin Type
            try:
                self.client.futures_change_margin_type(
                    symbol=symbol,
                    marginType=config.FUTURES_MARGIN_TYPE
                )
            except BinanceAPIException as e:
                msg = str(e)
                if "No need to change margin type" in msg:
                    pass  # Already set — fine
                elif "-1109" in msg or "Invalid account" in msg:
                    # Demo accounts often reject this — proceed anyway.
                    logger.warning(
                        f"⚠️  {symbol} margin type change rejected (-1109). "
                        f"Using account default. (Common on Futures Demo)"
                    )
                    soft_fail = True
                else:
                    logger.warning(f"Could not set margin type for {symbol}: {e}")

            # 2. Set Leverage (skip if margin already soft-failed to avoid spam)
            if not soft_fail:
                try:
                    self.client.futures_change_leverage(
                        symbol=symbol,
                        leverage=config.FUTURES_LEVERAGE
                    )
                except BinanceAPIException as e:
                    msg = str(e)
                    if "-1109" in msg or "Invalid account" in msg:
                        logger.warning(
                            f"⚠️  {symbol} leverage change rejected (-1109). "
                            f"Using account default."
                        )
                        soft_fail = True
                    else:
                        logger.warning(f"Could not set leverage for {symbol}: {e}")

            # Even if config calls failed, return True so the order is attempted.
            # The actual order endpoint may still succeed using account defaults.
            return True
        except BinanceAPIException as e:
            logger.error(f"Failed to setup futures for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to setup futures for {symbol}: {e}")
            return False

    # ── Orders ───────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[dict]:
        """Place a market BUY/SELL order (Auto-routes to Spot or Futures)."""
        try:
            use_f = config.FUTURES_ENABLED and self.is_futures(symbol)
            
            if use_f:
                # Ensure leverage/margin is set first — abort if setup fails
                if not self.setup_futures_symbol(symbol):
                    logger.error(f"Futures setup failed for {symbol} — cannot place order. Check API key permissions.")
                    return None
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            else:
                # Spot market execution
                if side == "BUY":
                    order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
                else:
                    order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
            
            emoji = "🟢" if side == "BUY" else "🔴"
            market_label = "[F]" if use_f else "[S]"
            logger.info(f"{emoji} {side} {market_label} | {symbol} | Qty: {quantity} | Status: {order.get('status')}")
            return order
        except Exception as e:
            logger.error(f"Market {side} failed for {symbol}: {e}")
            return None

    def get_filled_price(self, order: dict) -> float:
        """Extract average fill price from an order dict (supports both Spot and Futures)."""
        if not order or not isinstance(order, dict):
            return 0.0
        try:
            # 1. Direct average price (Futures)
            if "avgPrice" in order and float(order["avgPrice"]) > 0:
                return float(order["avgPrice"])

            # 2. Fills list (Spot)
            fills = order.get("fills", [])
            if fills:
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty  = sum(float(f["qty"]) for f in fills)
                return total_cost / total_qty if total_qty > 0 else 0.0

            # 3. Cumulative quote qty / executed qty (Spot fallback or Futures)
            exec_qty = float(order.get("executedQty", order.get("cumQty", 0)))
            if exec_qty > 0:
                cum_quote = float(order.get("cummulativeQuoteQty", order.get("cumQuote", 0)))
                return cum_quote / exec_qty
            
            # 4. Standard price field as last fallback
            if "price" in order and float(order["price"]) > 0:
                return float(order["price"])
                
            return 0.0
        except (ValueError, ZeroDivisionError, KeyError, TypeError):
            return 0.0

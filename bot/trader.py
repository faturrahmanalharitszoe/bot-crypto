"""
bot/trader.py — Core trading orchestration

Each call to Trader.tick() represents one loop iteration:
  1. If a position is open  → monitor TP / SL / EMA exit / max-hold
  2. If no position is open → scan watchlist for BUY signals
"""

from __future__ import annotations  # Python 3.9 compatibility

import os
import csv
import logging
import time
from datetime import datetime
from typing import Callable, List, Optional, Dict

from bot.exchange      import Exchange
from bot.strategy      import Strategy
from bot.risk_manager  import RiskManager
from bot.logger        import log_trade
from bot.pair_selector import select_best_pairs
from bot.notifier      import TelegramNotifier
from bot.ml_model      import MLPredictor
from bot.sentiment     import SentimentScorer
from bot.optimizer     import AutoOptimizer
import config

logger = logging.getLogger("bot")


class Trader:
    """Orchestrates pair scanning, signal checking, and order execution."""

    def __init__(self, exchange: Exchange,
                 notifier: Optional[TelegramNotifier] = None,
                 on_state_update: Optional[Callable] = None) -> None:
        self.exchange         = exchange
        self.strategy         = Strategy()
        self.risk_manager     = RiskManager()
        self.notifier         = notifier
        self.on_state_update  = on_state_update

        # AI modules
        self.ml_predictor  = MLPredictor()
        self.sentiment     = SentimentScorer()
        self.optimizer     = AutoOptimizer()
        self.optimizer.start()

        self.watchlist: List[str] = []
        self.watchlist_data: List[dict] = []
        self._ml_scores: Dict[str, tuple] = {} # Cache for dashboard
        self.dashboard_watchlist: List[dict] = []
        self.loop_count: int = 0
        
        # Cooldown management: { symbol: timestamp_of_last_exit }
        self.cooldowns: Dict[str, float] = {}
        self.cooldown_minutes: int = 5  # 5 min cooldown per pair (scalping-friendly)
        
        # Multi-position: max concurrent open positions
        self.max_positions: int = 3
        
        # Position History Logging
        self.history_log_path = "logs/position_history.csv"
        self._ensure_history_log()
        
        self._refresh_watchlist()

    # ── Watchlist Management ─────────────────────────────────

    def _refresh_watchlist(self) -> None:
        """Re-select the best pairs from the market."""
        logger.info("🔄 Refreshing pair watchlist...")
        data = select_best_pairs(self.exchange)
        # Store full data for dashboard
        self.watchlist_data = data
        # Extract just symbols for scanning
        self.watchlist = [item["symbol"] for item in data] if data else []

    # ── Main Tick ────────────────────────────────────────────

    def tick(self) -> None:
        """Single iteration of the trading loop."""
        self.loop_count += 1

        # Periodically refresh the watchlist
        if self.loop_count % config.PAIR_REFRESH_LOOPS == 0:
            self._refresh_watchlist()

        # Manage all open positions
        if self.risk_manager.has_positions:
            self._manage_open_positions()

        # Scan for new entries (up to max_positions)
        self._scan_for_entry()

        # Push state to web dashboard after every tick
        if self.on_state_update:
            try:
                self.on_state_update(self.get_status_dict())
            except Exception:
                pass

    def _ensure_history_log(self) -> None:
        """Create headers for position history log if not exists."""
        if not os.path.exists(self.history_log_path):
            os.makedirs("logs", exist_ok=True)
            with open(self.history_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "entry_price", "current_price", 
                    "pnl_pct", "max_pnl_pct", "trailing_active", "be_active", "duration_min"
                ])

    def _log_position_history(self) -> None:
        """Log the current state of all open positions."""
        for pos in list(self.risk_manager.positions.values()):
            try:
                curr_price = self.exchange.get_current_price(pos.symbol)
                pnl_pct = pos.unrealized_pnl_pct(curr_price)
                if pos.side == "BUY":
                    max_pnl_pct = ((pos.peak_price - pos.entry_price) / pos.entry_price) * 100
                else:
                    max_pnl_pct = ((pos.entry_price - pos.peak_price) / pos.entry_price) * 100

                with open(self.history_log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        pos.symbol,
                        round(pos.entry_price, 8),
                        round(curr_price, 8),
                        round(pnl_pct, 4),
                        round(max_pnl_pct, 4),
                        int(pos.trailing_active),
                        int(pos.be_active),
                        round(pos.duration_minutes, 1)
                    ])
            except Exception as e:
                logger.error(f"Failed to log position history for {pos.symbol}: {e}")

    # ── Position Management ──────────────────────────────────

    def _manage_open_positions(self) -> None:
        """Monitor all open positions for exit conditions."""
        for pos in list(self.risk_manager.positions.values()):
            self._manage_single_position(pos)
        self._log_position_history()

    def _manage_single_position(self, pos) -> None:
        """Monitor a single open position for exit conditions."""

        # Get current price
        current_price = self.exchange.get_current_price(pos.symbol)
        if current_price <= 0:
            logger.warning(f"Could not fetch price for {pos.symbol}. Skipping tick.")
            return

        # Get latest signal for EMA-reversal detection (Spot or Futures)
        df = self.exchange.get_klines(pos.symbol)
        df = self.strategy.calculate(df)
        signal, _ = self.strategy.get_signal(df)

        # Check exit conditions (reversal logic)
        reversal_signal = False
        if pos.side == "BUY":
            reversal_signal = (signal == "SELL") # Long reversal
        else:
            reversal_signal = (signal == "BUY")  # Short reversal

        exit_reason = self.risk_manager.check_exit(pos.symbol, current_price, reversal_signal)

        pnl = pos.unrealized_pnl(current_price)
        pnl_pct = pos.unrealized_pnl_pct(current_price)

        # ── Funding Fee Display (Futures only) ───────────────────
        funding_str = ""
        if config.FUNDING_FEE_ENABLED and self.exchange.is_futures(pos.symbol):
            fi = self.exchange.get_funding_info(pos.symbol)
            if fi["available"]:
                mark = self.exchange.get_mark_price(pos.symbol)
                fee_usdt = pos.estimate_funding_fee(mark, fi["rate"])
                fee_pct  = pos.funding_fee_pct(mark, fi["rate"])
                mins     = fi["minutes_until"]
                rate_pct = fi["rate"] * 100
                funding_str = f" | 💸 Funding: {rate_pct:+.4f}% in {mins:.0f}min (impact: {fee_usdt:+.4f} USDT / {fee_pct:+.4f}%)"

                # Warn if funding is soon and unfavorable
                if mins <= config.FUNDING_FEE_BLOCK_MINUTES:
                    if (pos.side == "BUY" and fi["rate"] > 0.0001) or \
                       (pos.side == "SELL" and fi["rate"] < -0.0001):
                        logger.warning(
                            f"⚠️  FUNDING IMMINENT | {pos.symbol} | Rate: {rate_pct:+.4f}% "
                            f"in {mins:.0f}min | Cost: {fee_usdt:+.4f} USDT — Consider early exit!"
                        )

        logger.info(
            f"📍 Holding {pos.symbol} | "
            f"Entry: {pos.entry_price:.6f} | "
            f"Current: {current_price:.6f} | "
            f"PnL: {pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
            f"{funding_str}"
        )

        if exit_reason:
            self._execute_exit(pos, exit_reason, current_price)

    def _execute_exit(self, pos, reason: str, current_price: float) -> None:
        """Place a market SELL and record the trade."""

        # Get actual current balance of the base asset
        base_asset = pos.symbol.replace("USDT", "")
        is_futures_pos = self.exchange.is_futures(pos.symbol)

        # For Futures: use position quantity directly (no spot balance to fetch)
        # For Spot: fetch actual asset balance to avoid partial-fill issues
        if is_futures_pos:
            actual_qty = pos.quantity
        else:
            base_asset = pos.symbol.replace("USDT", "")
            actual_qty = self.exchange.get_asset_balance(base_asset)
            if actual_qty <= 0:
                logger.error(
                    f"No {base_asset} spot balance to sell! "
                    f"Clearing position state without order."
                )
                self.risk_manager.close_position(pos.symbol)
                return

        # Respect LOT_SIZE constraints
        sym_info = self.exchange.get_symbol_info(pos.symbol)
        sell_qty = actual_qty

        if sym_info:
            import math
            step = sym_info.get("step_size", 0.001)
            prec = sym_info.get("base_precision", 3)
            sell_qty = math.floor(actual_qty / step) * step
            sell_qty = round(sell_qty, prec)

        if sell_qty <= 0:
            logger.error(f"Sell quantity rounded to 0 for {pos.symbol}. Clearing state.")
            self.risk_manager.close_position(pos.symbol)
            return

        # Exit side is always opposite of entry side
        exit_side = "SELL" if pos.side == "BUY" else "BUY"
        order = self.exchange.place_market_order(pos.symbol, exit_side, sell_qty)

        exit_price = current_price
        if order:
            filled = self.exchange.get_filled_price(order)
            if filled > 0:
                exit_price = filled

        # ── PnL Calculation (side-aware) ────────────────────────
        # LONG: profit when price goes UP  → pnl = (exit - entry) × qty
        # SHORT: profit when price goes DOWN → pnl = (entry - exit) × qty
        if pos.side == "BUY":  # Long
            pnl_usdt = (exit_price - pos.entry_price) * sell_qty
            pnl_pct  = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        else:  # Short
            pnl_usdt = (pos.entry_price - exit_price) * sell_qty
            pnl_pct  = ((pos.entry_price - exit_price) / pos.entry_price) * 100

        log_trade(
            symbol=pos.symbol,
            side=pos.side,          # Log ENTRY side (BUY=Long, SELL=Short) not exit order side
            cost_usdt=pos.cost_usdt,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=sell_qty,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_min=pos.duration_minutes,
        )

        # Telegram notification
        if self.notifier:
            self.notifier.notify_sell(
                symbol=pos.symbol, reason=reason,
                entry=pos.entry_price, exit_price=exit_price,
                pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                duration_min=pos.duration_minutes,
            )

        # Set cooldown for this symbol
        self.cooldowns[pos.symbol] = time.time()
        logger.info(f"⏳ Cooldown started for {pos.symbol} ({self.cooldown_minutes} min)")

        self.risk_manager.close_position(pos.symbol)

    # ── Entry Scanning ───────────────────────────────────────

    def _scan_for_entry(self) -> None:
        """Scan all watchlist pairs for a BUY/SELL signal."""
        # NOTE: loop_count is already incremented in tick(), do NOT re-increment here
        usdt_balance = self.exchange.get_usdt_balance()
        if usdt_balance < 1.0:
            logger.warning(f"USDT balance too low ({usdt_balance:.2f} USDT).")
            return

        # Cache data for this tick to avoid multiple API calls
        tick_data = {}

        for symbol in self.watchlist:
            # Fetch all timeframes once
            df_5m = self.exchange.get_klines(symbol, interval="5m")
            df_1m = self.exchange.get_klines(symbol, interval="1m")
            df_15m = self.exchange.get_klines(symbol, interval="15m")
            
            if df_5m.empty: continue
            
            ml_class, ml_conf = self.ml_predictor.predict(df_5m, df_1m, df_15m)
            self._ml_scores[symbol] = (ml_class, ml_conf)
            tick_data[symbol] = (df_5m, df_1m, df_15m)
            
        # Print Breakdown Table
        logger.info("📊 WATCHLIST BREAKDOWN:")
        logger.info(f"  {'Pair':<12} | {'Prediction':<10} | {'Conf':<6} | {'Status'}")
        logger.info("  " + "─" * 45)
        
        # Prepare dashboard data
        self.dashboard_watchlist = []
        
        for symbol in self.watchlist:
            if symbol not in self._ml_scores: continue
            ml_class, ml_conf = self._ml_scores[symbol]
            pred_text = "HOLD ⚪"
            if ml_class == 1: pred_text = "LONG 🟢"
            if ml_class == 2: pred_text = "SHORT 🔴"
            
            # Market label
            is_f = self.exchange.is_futures(symbol)
            m_label = "[F]" if is_f else "[S]"
            
            df_5m, _, _ = tick_data[symbol]
            df_feat = self.strategy.calculate(df_5m)
            rsi = df_feat['rsi'].iloc[-1]
            
            logger.info(f"  {symbol:<12} {m_label} | {pred_text:<10} | {ml_conf:.2f} | RSI: {rsi:.1f}")
            
            self.dashboard_watchlist.append({
                "symbol": symbol,
                "is_futures": bool(is_f),
                "ml_class": int(ml_class),
                "ml_confidence": float(ml_conf),
                "rsi": float(rsi)
            })
        
        logger.info("  " + "─" * 45)

        open_count = len(self.risk_manager.positions)
        for symbol in self.watchlist:
            # Stop scanning if we've hit max concurrent positions
            if open_count >= self.max_positions:
                break
            if symbol not in tick_data:
                continue
            # Skip if already in a position for this symbol
            if self.risk_manager.has_position_for(symbol):
                continue

            # Pass cached data to evaluate
            dfs = tick_data[symbol]
            entered = self._evaluate_pair_with_data(symbol, usdt_balance, *dfs)
            if entered:
                open_count += 1

    def _evaluate_pair_with_data(self, symbol: str, usdt_balance: float, 
                                 df_5m: pd.DataFrame, df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> bool:
        """Evaluate a pair using provided (cached) data. Returns True if position was opened."""
        # ── Cooldown Check ────────────────────────────────────
        last_exit = self.cooldowns.get(symbol, 0)
        elapsed = (time.time() - last_exit) / 60
        if elapsed < self.cooldown_minutes:
            return False

        df_5m = self.strategy.calculate(df_5m)
        signal, _ = self.strategy.get_signal(df_5m)

        # ── Full AI Mode Decision (MTF Enabled) ───────────────
        if config.ML_ENABLED and config.ML_FULL_AI_MODE:
            ml_class, ml_confidence = self._ml_scores.get(symbol, (0, 0.0))
            if ml_class == 0: return False

            side = "BUY" if ml_class == 1 else "SELL"
            is_f = self.exchange.is_futures(symbol)
            
            # Safeguard: Spot cannot SHORT
            if not is_f and side == "SELL":
                return False

            if side == "BUY" and self.sentiment.is_blocked(symbol):
                return False

            if ml_confidence < config.ML_CONFIDENCE_THRESHOLD:
                return False

            market_text = "FUTURES" if is_f else "SPOT"
            logger.info(f"🚀 AI SIGNAL | {symbol} ({market_text}) | Side: {side} | Conf: {ml_confidence:.2f} → EXECUTING")
        else:
            if signal not in ("BUY", "SELL"): return False
            side = signal

        # ── Funding Fee Guard (Futures only) ─────────────────────
        if config.FUNDING_FEE_ENABLED and self.exchange.is_futures(symbol):
            fi = self.exchange.get_funding_info(symbol)
            if fi["available"]:
                rate      = fi["rate"]
                mins      = fi["minutes_until"]
                rate_pct  = rate * 100
                # Block if: funding very soon AND rate is unfavorable for our direction
                #   LONG  + positive rate = we'll pay → unfavorable
                #   SHORT + negative rate = we'll pay → unfavorable
                # Also block if rate is extremely extreme regardless of timing
                rate_unfavorable = (
                    (side == "BUY"  and rate > 0.0001) or   # Long pays
                    (side == "SELL" and rate < -0.0001)      # Short pays
                )
                rate_extreme = abs(rate) > config.FUNDING_FEE_MAX_RATE

                if rate_extreme:
                    logger.info(
                        f"💸 FUNDING SKIP | {symbol} | Rate {rate_pct:+.4f}% is extreme "
                        f"(>{config.FUNDING_FEE_MAX_RATE*100:.3f}%) — Too risky to enter"
                    )
                    return False

                if mins <= config.FUNDING_FEE_BLOCK_MINUTES and rate_unfavorable:
                    logger.info(
                        f"💸 FUNDING SKIP | {symbol} | {side} blocked — "
                        f"Funding in {mins:.0f}min at {rate_pct:+.4f}% (unfavorable). "
                        f"Will re-evaluate after funding."
                    )
                    return False

        # Get symbol constraints
        sym_info = self.exchange.get_symbol_info(symbol)
        if not sym_info:
            logger.warning(f"Could not get symbol info for {symbol}. Skipping.")
            return False

        current_price = self.exchange.get_current_price(symbol)
        if current_price <= 0:
            return False

        # Divide usdt balance by max_positions for per-trade allocation
        usdt_per_trade = usdt_balance / self.max_positions

        # Calculate position size
        quantity = self.risk_manager.calculate_quantity(
            usdt_balance=usdt_per_trade,
            current_price=current_price,
            symbol_info=sym_info,
        )

        if quantity <= 0:
            logger.warning(
                f"Cannot open position on {symbol} — "
                f"quantity calculation returned 0 "
                f"(balance: {usdt_per_trade:.2f} USDT @ {current_price:.6f})"
            )
            return False

        # Execute market order
        order = self.exchange.place_market_order(symbol, side, quantity)
        if not order:
            return False

        entry_price = self.exchange.get_filled_price(order)
        if entry_price <= 0:
            entry_price = current_price  # Fallback to ticker price

        pos = self.risk_manager.open_position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            side=side,
            is_futures=is_f,
        )

        # Telegram notification
        if self.notifier:
            self.notifier.notify_buy(
                symbol=symbol, price=entry_price, qty=quantity,
                take_profit=pos.take_profit, stop_loss=pos.stop_loss,
            )

        return True

    # ── Status Summary ───────────────────────────────────────

    def get_status_dict(self) -> dict:
        """Return current bot state as a dict for the status banner."""
        usdt = self.exchange.get_usdt_balance()
        positions = self.risk_manager.positions  # Dict[symbol, Position]

        positions_list = []
        for sym, pos in positions.items():
            price = self.exchange.get_current_price(sym)
            pnl   = pos.unrealized_pnl(price)
            pnl_p = pos.unrealized_pnl_pct(price)
            is_fut = self.exchange.is_futures(sym)
            positions_list.append({
                "symbol":        sym,
                "entry_price":   pos.entry_price,
                "quantity":      pos.quantity,
                "cost_usdt":     pos.cost_usdt,
                "current_price": price,
                "pnl_usdt":      pnl,
                "pnl_pct":       pnl_p,
                "take_profit":   pos.take_profit,
                "stop_loss":     pos.stop_loss,
                "duration_min":  pos.duration_minutes,
                "side":          pos.side,
                "is_futures":    is_fut,
                "funding":       self.exchange.get_funding_info(sym) if is_fut else None,
            })

        # Backward compat: expose first position as top-level fields
        first = positions_list[0] if positions_list else None

        base = {
            "usdt_balance": usdt,
            "has_position": bool(positions_list),
            "positions":    positions_list,
            "watchlist":    self.dashboard_watchlist,
            "loop":         self.loop_count,
            "testnet":      config.TESTNET,
            "sentiment":    self.sentiment.all_scores(),
            "optimizer_last_run": self.optimizer.last_run,
        }

        if first:
            base.update(first)

        return base

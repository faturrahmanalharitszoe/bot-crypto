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
        self._ml_scores: Dict[str, float] = {} # Cache for dashboard
        self.loop_count: int = 0
        
        # Cooldown management: { symbol: timestamp_of_last_exit }
        self.cooldowns: Dict[str, float] = {}
        self.cooldown_minutes: int = 15 
        
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

        if self.risk_manager.has_position:
            self._manage_open_position()
            self._log_position_history()
        else:
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
        """Log the current state of the open position."""
        if not self.risk_manager.has_position:
            return
            
        pos = self.risk_manager.position
        try:
            curr_price = self.exchange.get_current_price(pos.symbol)
            pnl_pct = pos.unrealized_pnl_pct(curr_price)
            # Use max_price recorded in position
            max_pnl_pct = ((pos.max_price - pos.entry_price) / pos.entry_price) * 100
            
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
            logger.error(f"Failed to log position history: {e}")

    # ── Position Management ──────────────────────────────────

    def _manage_open_position(self) -> None:
        """Monitor an open position for exit conditions."""
        pos = self.risk_manager.position

        # Get current price
        current_price = self.exchange.get_current_price(pos.symbol)
        if current_price <= 0:
            logger.warning(f"Could not fetch price for {pos.symbol}. Skipping tick.")
            return

        # Get latest signal for EMA-reversal detection
        df = self.exchange.get_klines(pos.symbol)
        df = self.strategy.calculate(df)
        signal, _ = self.strategy.get_signal(df)
        sell_signal = (signal == "SELL")

        # Check exit conditions
        exit_reason = self.risk_manager.check_exit(current_price, sell_signal)

        pnl = pos.unrealized_pnl(current_price)
        pnl_pct = pos.unrealized_pnl_pct(current_price)
        logger.info(
            f"📍 Holding {pos.symbol} | "
            f"Entry: {pos.entry_price:.6f} | "
            f"Current: {current_price:.6f} | "
            f"PnL: {pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
        )

        if exit_reason:
            self._execute_exit(exit_reason, current_price)

    def _execute_exit(self, reason: str, current_price: float) -> None:
        """Place a market SELL and record the trade."""
        pos = self.risk_manager.position

        # Get actual current balance of the base asset
        base_asset = pos.symbol.replace("USDT", "")
        actual_qty = self.exchange.get_asset_balance(base_asset)

        if actual_qty <= 0:
            logger.error(
                f"No {base_asset} balance to sell! "
                f"Clearing position state without order."
            )
            self.risk_manager.close_position()
            return

        # Get symbol info to respect LOT_SIZE on the sell
        sym_info = self.exchange.get_symbol_info(pos.symbol)
        sell_qty = pos.quantity

        if sym_info:
            import math
            step = sym_info.get("step_size", 0.001)
            prec = sym_info.get("base_precision", 3)
            sell_qty = math.floor(actual_qty / step) * step
            sell_qty = round(sell_qty, prec)

        if sell_qty <= 0:
            logger.error("Sell quantity rounded to 0. Skipping order.")
            self.risk_manager.close_position()
            return

        order = self.exchange.place_market_sell(pos.symbol, sell_qty)

        exit_price = current_price
        if order:
            filled = self.exchange.get_filled_price(order)
            if filled > 0:
                exit_price = filled

        pnl_usdt = (exit_price - pos.entry_price) * sell_qty
        pnl_pct  = ((exit_price - pos.entry_price) / pos.entry_price) * 100

        log_trade(
            symbol=pos.symbol,
            side="SELL",
            cost_usdt=pos.cost_usdt,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=sell_qty,
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

        self.risk_manager.close_position()

    # ── Entry Scanning ───────────────────────────────────────

    def _scan_for_entry(self) -> None:
        """Scan all watchlist pairs for a BUY signal and enter the best one."""
        logger.info(
            f"🔍 Scanning {len(self.watchlist)} pairs for BUY signal..."
        )

        usdt_balance = self.exchange.get_usdt_balance()
        if usdt_balance < 1.0:
            logger.warning(
                f"USDT balance too low ({usdt_balance:.2f} USDT). "
                f"Minimum ~$1 required."
            )
            return

        for symbol in self.watchlist:
            # Don't open another position once one is found this tick
            if self.risk_manager.has_position:
                break

            self._evaluate_pair(symbol, usdt_balance)

    def _evaluate_pair(self, symbol: str, usdt_balance: float) -> None:
        """Check a single pair for a valid entry signal using Multi-Timeframe data."""
        # ── Cooldown Check ────────────────────────────────────
        last_exit = self.cooldowns.get(symbol, 0)
        elapsed = (time.time() - last_exit) / 60
        if elapsed < self.cooldown_minutes:
            return

        # Fetch multi-timeframe candles
        df_5m = self.exchange.get_klines(symbol, interval="5m")
        if df_5m.empty: return
        
        df_1m = self.exchange.get_klines(symbol, interval="1m")
        df_15m = self.exchange.get_klines(symbol, interval="15m")

        df_5m = self.strategy.calculate(df_5m)
        signal, details = self.strategy.get_signal(df_5m)

        # ── Full AI Mode Decision (MTF Enabled) ───────────────
        if config.ML_ENABLED and config.ML_FULL_AI_MODE:
            # 1. Get ML confidence score (Multi-Timeframe Analysis)
            ml_confidence = self.ml_predictor.predict(df_5m, df_1m, df_15m)
            self._ml_scores[symbol] = ml_confidence 

            # 2. Check sentiment (blocker)
            if self.sentiment.is_blocked(symbol):
                logger.info(f"  {symbol:<12} | ❌ BLOCKED by negative sentiment")
                return

            # 3. ML confidence must exceed threshold
            if ml_confidence < config.ML_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"  {symbol:<12} | Signal: {signal:<4} | "
                    f"ML: {ml_confidence:.2f} < {config.ML_CONFIDENCE_THRESHOLD:.2f} → SKIP"
                )
                return

            # 4. EMA/RSI acts as confirmation (not rejection if no model)
            if self.ml_predictor.is_ready and signal == "SELL":
                logger.info(
                    f"  {symbol:<12} | ML: {ml_confidence:.2f} ✅ | "
                    f"EMA says SELL — skipping (conflicting signals)"
                )
                return

            logger.info(
                f"  {symbol:<12} | ML: {ml_confidence:.2f} ✅ | "
                f"Sentiment: OK | EMA: {signal} | → ENTRY"
            )

        else:
            # Legacy Rule-Based Mode
            logger.info(
                f"  {symbol:<12} | Signal: {signal:<4} | "
                f"EMA9: {details.get('ema_fast', 0):.4f} | "
                f"RSI: {details.get('rsi', 0):.1f}"
            )
            if signal != "BUY":
                return

        # Get symbol constraints
        sym_info = self.exchange.get_symbol_info(symbol)
        if not sym_info:
            logger.warning(f"Could not get symbol info for {symbol}. Skipping.")
            return

        current_price = self.exchange.get_current_price(symbol)
        if current_price <= 0:
            return

        # Calculate position size
        quantity = self.risk_manager.calculate_quantity(
            usdt_balance=usdt_balance,
            current_price=current_price,
            symbol_info=sym_info,
        )

        if quantity <= 0:
            logger.warning(
                f"Cannot open position on {symbol} — "
                f"quantity calculation returned 0 "
                f"(balance: {usdt_balance:.2f} USDT @ {current_price:.6f})"
            )
            return

        # Execute market buy
        order = self.exchange.place_market_buy(symbol, quantity)
        if not order:
            return

        entry_price = self.exchange.get_filled_price(order)
        if entry_price <= 0:
            entry_price = current_price  # Fallback to ticker price

        pos = self.risk_manager.open_position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
        )

        # Telegram notification
        if self.notifier:
            self.notifier.notify_buy(
                symbol=symbol, price=entry_price, qty=quantity,
                take_profit=pos.take_profit, stop_loss=pos.stop_loss,
            )

    # ── Status Summary ───────────────────────────────────────

    def get_status_dict(self) -> dict:
        """Return current bot state as a dict for the status banner."""
        usdt = self.exchange.get_usdt_balance()
        pos  = self.risk_manager.position

        # Inject ML scores into watchlist data for dashboard
        enriched_watchlist = []
        for item in self.watchlist_data:
            new_item = item.copy()
            new_item["ml_confidence"] = self._ml_scores.get(item["symbol"], 0.0)
            enriched_watchlist.append(new_item)

        base = {
            "usdt_balance": usdt,
            "has_position": False,
            "watchlist":    enriched_watchlist,
            "watchlist_count": len(enriched_watchlist),
            "trades_count": len(self.risk_manager.trade_history) if hasattr(self.risk_manager, "trade_history") else 0,
            "loop":         self.loop_count,
            "testnet":      config.TESTNET,
            "sentiment":    self.sentiment.all_scores(),
            "optimizer_last_run": self.optimizer.last_run,
        }

        if pos:
            price = self.exchange.get_current_price(pos.symbol)
            pnl   = pos.unrealized_pnl(price)
            pnl_p = pos.unrealized_pnl_pct(price)
            base.update({
                "has_position":  True,
                "symbol":        pos.symbol,
                "entry_price":   pos.entry_price,
                "quantity":      pos.quantity,
                "cost_usdt":     pos.cost_usdt,
                "current_price": price,
                "pnl_usdt":      pnl,
                "pnl_pct":       pnl_p,
                "take_profit":   pos.take_profit,
                "stop_loss":     pos.stop_loss,
                "duration_min":  pos.duration_minutes,
            })

        return base

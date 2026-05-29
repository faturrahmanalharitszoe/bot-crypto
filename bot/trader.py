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

import pandas as pd

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
        self.cooldown_minutes: int = getattr(config, "COOLDOWN_MINUTES", 5)
        # Reason tracking for dynamic cooldown
        self._last_exit_reason: Dict[str, str] = {}

        # [FIX-C] Consecutive loss blacklist tracking
        # { symbol: [timestamp_of_loss, ...] }  — keeps last N loss timestamps
        self._loss_history: Dict[str, list] = {}
        # { symbol: blacklist_expiry_timestamp }
        self._blacklist: Dict[str, float] = {}

        # [FIX-DRIFT] Last exit price per symbol for drift protection
        # { symbol: exit_price }
        self._last_exit_price: Dict[str, float] = {}
        
        # Multi-position: max concurrent open positions
        # [REVAMP] Reduced from 2 to 1 — focus on ONE high-quality trade at a time
        self.max_positions: int = 1
        
        # Position History Logging
        self.history_log_path = "logs/position_history.csv"
        self._ensure_history_log()
        
        # Rate-limiting scanner loop
        self.last_scan_time = 0.0
        
        self._refresh_watchlist()

    # ── Watchlist Management ─────────────────────────────────

    def _refresh_watchlist(self) -> None:
        """Re-select the best pairs from the market."""
        logger.info("🔄 Refreshing pair watchlist...")
        data = select_best_pairs(self.exchange)
        
        valid_data = []
        for item in data:
            symbol = item["symbol"]
            try:
                info = self.exchange.get_symbol_info(symbol)
                if info is not None:
                    valid_data.append(item)
                else:
                    logger.debug(f"Filtering out non-TRADING or invalid symbol: {symbol}")
            except Exception as e:
                logger.warning(f"Error checking status for {symbol}: {e}")
            
            if len(valid_data) >= config.TOP_PAIRS_COUNT:
                break

        # Store full data for dashboard
        self.watchlist_data = valid_data
        # Extract just symbols for scanning
        self.watchlist = [item["symbol"] for item in valid_data]

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

        now = time.time()
        # Only run heavy MTF/ML and strategy kline calls once every 30 seconds
        if now - getattr(pos, "last_signal_check_time", 0.0) >= 30.0:
            reversal_signal = False

            # 1. Machine Learning Exit (if enabled and model is loaded)
            # [REVAMP v2] ML exit now requires:
            #   - Higher confidence (0.75+) to override position
            #   - Position must be in profit OR held > 10 candles to allow ML exit
            #   - This prevents ML from killing winners prematurely
            if config.ML_ENABLED and config.ML_EXIT_ENABLED and self.ml_predictor.is_ready:
                try:
                    df_main = self.exchange.get_klines(pos.symbol, interval=config.CANDLE_INTERVAL)
                    df_lower = self.exchange.get_klines(pos.symbol, interval=config.LOWER_INTERVAL)
                    df_higher = self.exchange.get_klines(pos.symbol, interval=config.HIGHER_INTERVAL)

                    if not df_main.empty:
                        ml_class, ml_conf = self.ml_predictor.predict(df_main, df_lower, df_higher)
                        
                        # [REVAMP] Only allow ML exit if confidence is HIGH (0.75+)
                        # AND position is either in loss or has been held long enough
                        pnl_pct_now = pos.unrealized_pnl_pct(current_price)
                        ml_exit_confidence = 0.75  # Higher bar for exits than entries
                        allow_ml_exit = (
                            ml_conf >= ml_exit_confidence and
                            (pnl_pct_now < -0.3 or pos.duration_minutes > 60)
                        )

                        if allow_ml_exit:
                            if pos.side == "BUY" and config.ML_REVERSAL_EXIT_ACTIVE and ml_class == 2:
                                reversal_signal = "ML_REVERSAL"
                                logger.info(f"🔮 AI EXIT | {pos.symbol} LONG -> SHORT (Conf: {ml_conf:.2f}, PnL: {pnl_pct_now:+.2f}%) → EXITING")
                            elif pos.side == "SELL" and config.ML_REVERSAL_EXIT_ACTIVE and ml_class == 1:
                                reversal_signal = "ML_REVERSAL"
                                logger.info(f"🔮 AI EXIT | {pos.symbol} SHORT -> LONG (Conf: {ml_conf:.2f}, PnL: {pnl_pct_now:+.2f}%) → EXITING")
                except Exception as e:
                    logger.error(f"Failed to check ML exit for {pos.symbol}: {e}")

            # 2. Traditional technical signal exit (fallback or overlay)
            # We only check this if ML exit hasn't triggered AND config allows it
            if not reversal_signal and config.TECH_REVERSAL_EXIT_ENABLED:
                df = self.exchange.get_klines(pos.symbol)
                df = self.strategy.calculate(df)
                signal, _ = self.strategy.get_signal(df)

                if pos.side == "BUY":
                    if signal == "SELL":
                        reversal_signal = "REVERSAL_SIGNAL"
                else:
                    if signal == "BUY":
                        reversal_signal = "REVERSAL_SIGNAL"

            pos.last_signal_check_time = now
            pos.cached_reversal_signal = reversal_signal
        else:
            reversal_signal = getattr(pos, "cached_reversal_signal", False)

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

        # Calculate dynamic cooldown duration based on exit reason
        cooldown_duration = self.cooldown_minutes
        if getattr(config, "DOUBLE_COOLDOWN_ON_LOSS", True) and reason == "STOP_LOSS":
            cooldown_duration *= 2
        elif reason == "ML_REVERSAL":
            cooldown_duration *= 1.5

        # Set cooldown expiry timestamp directly
        self.cooldowns[pos.symbol] = time.time() + (cooldown_duration * 60)
        self._last_exit_reason[pos.symbol] = reason
        
        logger.info(f"⏳ Cooldown started for {pos.symbol} ({cooldown_duration:.1f} min, reason: {reason})")

        # [FIX-DRIFT] Remember last exit price for drift protection
        self._last_exit_price[pos.symbol] = exit_price

        # [FIX-C] Track consecutive stop-losses for blacklist
        if reason == "STOP_LOSS" and config.BLACKLIST_ENABLED:
            self._record_loss(pos.symbol)

        self.risk_manager.close_position(pos.symbol)

    # ── [FIX-C] Consecutive Loss Blacklist ───────────────────

    # ── [FIX-HTF] Higher Timeframe Trend Helper ──────────────

    def _get_htf_trend(self, df_higher: pd.DataFrame) -> str:
        """
        Determine the 1h trend direction using EMA9/21 alignment + slope.
        Returns: 'UP', 'DOWN', or 'NEUTRAL'

        Logic (two conditions must agree):
          1. EMA9 vs EMA21 position (alignment)
          2. EMA9 slope over last 3 candles (momentum confirmation)

        NEUTRAL is returned when trend is ambiguous (EMAs are crossing or flat),
        which lets the bot trade in both directions during consolidation.
        """
        if df_higher is None or df_higher.empty or len(df_higher) < 25:
            return "NEUTRAL"

        try:
            df = df_higher.copy()
            df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
            df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

            latest = df.iloc[-1]
            prev3  = df.iloc[-4]  # 3 candles ago for slope check

            ema_aligned_up   = latest["ema9"] > latest["ema21"]
            ema_aligned_down = latest["ema9"] < latest["ema21"]

            # Slope: is EMA9 consistently moving in one direction?
            ema9_rising  = latest["ema9"] > prev3["ema9"]
            ema9_falling = latest["ema9"] < prev3["ema9"]

            # Gap magnitude: require at least 0.1% gap to avoid false signals
            gap_pct = abs(latest["ema9"] - latest["ema21"]) / (latest["ema21"] + 1e-9) * 100
            gap_significant = gap_pct > 0.10

            if ema_aligned_up and ema9_rising and gap_significant:
                return "UP"
            elif ema_aligned_down and ema9_falling and gap_significant:
                return "DOWN"
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.debug(f"HTF trend calc error: {e}")
            return "NEUTRAL"

    # ── [FIX-B] Candle Direction Confirmation ────────────────

    def _candle_direction_confirmed(self, df: pd.DataFrame, side: str) -> bool:
        """
        Check that at least CANDLE_CONFIRM_MIN of the last CANDLE_CONFIRM_LOOKBACK
        closed candles agree with the intended trade direction.

        For SELL (Short): candle is bearish  → close < open
        For BUY  (Long) : candle is bullish  → close > open

        Uses the completed candles only (excludes the live/current candle, i.e. iloc[-1]).
        Returns True if confirmation threshold is met, False otherwise.
        """
        lookback = config.CANDLE_CONFIRM_LOOKBACK
        minimum  = config.CANDLE_CONFIRM_MIN

        # Need at least lookback + 1 rows (last row = live candle, excluded)
        if df is None or len(df) < lookback + 1:
            return True  # Not enough data — don't block, let other filters decide

        # Closed candles: skip the last (live) candle
        closed = df.iloc[-(lookback + 1):-1]

        if side == "SELL":
            agreeing = int((closed["close"] < closed["open"]).sum())
        else:  # BUY
            agreeing = int((closed["close"] > closed["open"]).sum())

        confirmed = agreeing >= minimum
        if not confirmed:
            logger.info(
                f"🕯️  [CANDLE CONFIRM] {side} blocked — only {agreeing}/{lookback} "
                f"candles bearish/bullish (need {minimum})"
            )
        else:
            logger.debug(
                f"🕯️  [CANDLE CONFIRM] {side} passed — {agreeing}/{lookback} candles aligned"
            )
        return confirmed

    def _record_loss(self, symbol: str) -> None:
        """Record a stop-loss for symbol and blacklist if threshold is crossed."""
        now = time.time()
        window_secs = config.BLACKLIST_WINDOW_HOURS * 3600

        if symbol not in self._loss_history:
            self._loss_history[symbol] = []

        # Append loss and prune entries outside the window
        self._loss_history[symbol].append(now)
        self._loss_history[symbol] = [
            t for t in self._loss_history[symbol] if now - t <= window_secs
        ]

        recent_count = len(self._loss_history[symbol])
        logger.info(
            f"📉 Loss recorded for {symbol} | "
            f"{recent_count}/{config.BLACKLIST_CONSECUTIVE_LOSSES} stop-losses "
            f"in last {config.BLACKLIST_WINDOW_HOURS}h"
        )

        if recent_count >= config.BLACKLIST_CONSECUTIVE_LOSSES:
            expiry = now + config.BLACKLIST_COOLDOWN_HOURS * 3600
            self._blacklist[symbol] = expiry
            self._loss_history[symbol] = []  # Reset counter after blacklisting
            logger.warning(
                f"🚫 [BLACKLIST] {symbol} hit {config.BLACKLIST_CONSECUTIVE_LOSSES} "
                f"stop-losses in {config.BLACKLIST_WINDOW_HOURS}h — "
                f"banned for {config.BLACKLIST_COOLDOWN_HOURS}h"
            )

    def _is_blacklisted(self, symbol: str) -> bool:
        """Return True if symbol is currently blacklisted."""
        if not config.BLACKLIST_ENABLED:
            return False
        expiry = self._blacklist.get(symbol, 0)
        if time.time() < expiry:
            remaining = (expiry - time.time()) / 60
            logger.debug(f"🚫 {symbol} blacklisted ({remaining:.0f}m remaining)")
            return True
        # Expired — remove from blacklist
        if symbol in self._blacklist:
            del self._blacklist[symbol]
            logger.info(f"✅ {symbol} removed from blacklist")
        return False

    # ── Entry Scanning ───────────────────────────────────────

    def _scan_for_entry(self) -> None:
        """Scan all watchlist pairs for a BUY/SELL signal."""
        # Rate-limit scanning to prevent API spam (especially when monitoring fast in position mode)
        now = time.time()
        if now - self.last_scan_time < config.LOOP_INTERVAL_SECONDS:
            return
        self.last_scan_time = now

        # NOTE: loop_count is already incremented in tick(), do NOT re-increment here
        usdt_balance = self.exchange.get_usdt_balance()
        if usdt_balance < 1.0:
            logger.warning(f"USDT balance too low ({usdt_balance:.2f} USDT).")
            return

        # Cache data for this tick to avoid multiple API calls
        tick_data = {}
        tech_signals = {}

        for symbol in self.watchlist:
            # Ensure symbol is active/trading on Binance
            sym_info = self.exchange.get_symbol_info(symbol)
            if not sym_info:
                self._ml_scores[symbol] = (0, 0.0)
                tick_data[symbol] = (None, None, None)
                tech_signals[symbol] = "HOLD"
                continue

            # Fetch main timeframe first
            df_main = self.exchange.get_klines(symbol, interval=config.CANDLE_INTERVAL)
            if df_main.empty: continue
            
            # Check technical strategy signal first (still tracked for RSI/dashboard)
            df_feat = self.strategy.calculate(df_main)
            tech_signal, _ = self.strategy.get_signal(df_feat)
            tech_signals[symbol] = tech_signal or "HOLD"
            
            ml_class, ml_conf = 0, 0.0
            df_lower, df_higher = None, None
            
            # Fetch lower and higher timeframes and run predictor in Full AI Mode,
            # or in Confirmation Mode when tech signal is present.
            should_predict = (config.ML_ENABLED and config.ML_FULL_AI_MODE) or (tech_signal in ("BUY", "SELL"))
            
            if should_predict:
                df_lower = self.exchange.get_klines(symbol, interval=config.LOWER_INTERVAL)
                df_higher = self.exchange.get_klines(symbol, interval=config.HIGHER_INTERVAL)
                if not df_lower.empty and not df_higher.empty:
                    ml_class, ml_conf = self.ml_predictor.predict(df_main, df_lower, df_higher)
            
            self._ml_scores[symbol] = (ml_class, ml_conf)
            tick_data[symbol] = (df_main, df_lower, df_higher)
            
        # Print Breakdown Table
        logger.info("📊 WATCHLIST BREAKDOWN:")
        logger.info(f"  {'Pair':<16} | {'Tech':<8} | {'ML Pred':<9} | {'Conf':<4} | {'RSI':<6} | {'ADX':<6} | {'Status'}")
        logger.info("  " + "─" * 85)
        
        # Prepare dashboard data
        self.dashboard_watchlist = []
        
        for symbol in self.watchlist:
            if symbol not in self._ml_scores: continue
            ml_class, ml_conf = self._ml_scores[symbol]
            tech_signal = tech_signals.get(symbol, "HOLD")
            
            # Formatting signals with emojis
            tech_text = "HOLD ⚪"
            if tech_signal == "BUY": tech_text = "BUY 🟢"
            if tech_signal == "SELL": tech_text = "SELL 🔴"

            pred_text = "HOLD ⚪"
            if ml_class == 1: pred_text = "LONG 🟢"
            if ml_class == 2: pred_text = "SHORT 🔴"
            
            # Market label
            is_f = self.exchange.is_futures(symbol)
            m_label = "[F]" if is_f else "[S]"
            
            df_main, _, _ = tick_data[symbol]
            adx = 0.0
            if df_main is not None and not df_main.empty:
                df_feat = self.strategy.calculate(df_main)
                rsi = df_feat['rsi'].iloc[-1] if 'rsi' in df_feat.columns else 0.0
                adx = float(df_feat['adx'].iloc[-1]) if 'adx' in df_feat.columns else 0.0
            else:
                rsi = 0.0
            
            # Cooldown check
            expiry = self.cooldowns.get(symbol, 0.0)
            remaining_min = (expiry - time.time()) / 60
            
            # Determine status text and block reason
            sym_info = self.exchange.get_symbol_info(symbol)
            if remaining_min > 0:
                status_text = f"COOLDOWN ({remaining_min:.1f}m)"
            elif self._is_blacklisted(symbol):
                # [FIX-C] Show blacklist remaining time
                remaining_min = (self._blacklist.get(symbol, 0) - time.time()) / 60
                status_text = f"BLACKLISTED 🚫 ({remaining_min:.0f}m)"
            elif not sym_info:
                status_text = "BLOCKED ❌ (Closed)"
            elif config.ML_ENABLED and config.ML_FULL_AI_MODE:
                # Full AI Mode status determination
                if ml_class not in (1, 2):
                    status_text = "HOLD ⚪"
                elif ml_conf < config.ML_CONFIDENCE_THRESHOLD:
                    status_text = f"BLOCKED ❌ (Low Conf)"
                else:
                    side = "BUY" if ml_class == 1 else "SELL"
                    if not is_f and side == "SELL":
                        status_text = "BLOCKED ❌ (Spot cannot Short)"
                    elif side == "BUY" and self.sentiment.is_blocked(symbol):
                        status_text = "BLOCKED ❌ (Sentiment Blocked)"
                    else:
                        status_text = "READY ✅"
            else:
                # Confirmation / Pure Technical Mode status determination
                if tech_signal not in ("BUY", "SELL"):
                    status_text = "HOLD ⚪"
                else:
                    if config.ML_ENABLED:
                        is_confirmed = (tech_signal == "BUY" and ml_class == 1) or (tech_signal == "SELL" and ml_class == 2)
                        if not is_confirmed:
                            status_text = f"BLOCKED ❌ (Mismatch)"
                        elif ml_conf < config.ML_CONFIDENCE_THRESHOLD:
                            status_text = f"BLOCKED ❌ (Low Conf)"
                        elif not is_f and tech_signal == "SELL":
                            status_text = "BLOCKED ❌ (Spot cannot Short)"
                        elif tech_signal == "BUY" and self.sentiment.is_blocked(symbol):
                            status_text = "BLOCKED ❌ (Sentiment Blocked)"
                        else:
                            status_text = "READY ✅"
                    else:
                        if not is_f and tech_signal == "SELL":
                            status_text = "BLOCKED ❌ (Spot cannot Short)"
                        elif tech_signal == "BUY" and self.sentiment.is_blocked(symbol):
                            status_text = "BLOCKED ❌ (Sentiment Blocked)"
                        else:
                            status_text = "READY ✅"

            logger.info(f"  {symbol:<12} {m_label} | {tech_text:<8} | {pred_text:<9} | {ml_conf:.2f} | {rsi:<5.1f} | {adx:<5.1f} | {status_text}")
            
            self.dashboard_watchlist.append({
                "symbol": symbol,
                "is_futures": bool(is_f),
                "tech_signal": tech_signal,
                "ml_class": int(ml_class),
                "ml_confidence": float(ml_conf),
                "rsi": float(rsi),
                "adx": float(adx),
                "adx_ok": adx >= getattr(config, "ADX_MIN_THRESHOLD", 20.0),
                "status_text": status_text
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
                                 df_main: pd.DataFrame, df_lower: pd.DataFrame, df_higher: pd.DataFrame) -> bool:
        """Evaluate a pair using provided (cached) data. Returns True if position was opened."""
        # ── Cooldown Check ────────────────────────────────────
        expiry = self.cooldowns.get(symbol, 0.0)
        if time.time() < expiry:
            return False

        # ── [FIX-C] Blacklist Check ───────────────────────────
        if self._is_blacklisted(symbol):
            return False

        is_f = self.exchange.is_futures(symbol)

        # ── [FIX-DRIFT] Re-Entry Price Drift Protection ───────
        # If we already traded this symbol and price has drifted significantly
        # in the direction we'd be chasing, skip the entry.
        # Example: shorted at 0.641, price pumped to 0.661 (+3%) → don't short again
        if config.DRIFT_PROTECTION_ENABLED and symbol in self._last_exit_price:
            last_exit = self._last_exit_price[symbol]
            if last_exit > 0 and not df_main.empty:
                current_price = float(df_main["close"].iloc[-1])
                drift = (current_price - last_exit) / last_exit  # positive = price went up

                # If trying to SHORT but price already pumped > DRIFT_MAX_PCT from last exit
                # OR trying to LONG but price already dumped > DRIFT_MAX_PCT from last exit,
                # we need to know the intended side first — check ML scores cache
                cached = self._ml_scores.get(symbol, (0, 0.0))
                intended_side = "BUY" if cached[0] == 1 else ("SELL" if cached[0] == 2 else None)

                if intended_side == "SELL" and drift > config.DRIFT_MAX_PCT:
                    logger.info(
                        f"🚫 [DRIFT] {symbol} | Price drifted +{drift*100:.2f}% UP from last exit "
                        f"({last_exit:.6f} → {current_price:.6f}). Blocking SHORT re-entry."
                    )
                    return False
                elif intended_side == "BUY" and drift < -config.DRIFT_MAX_PCT:
                    logger.info(
                        f"🚫 [DRIFT] {symbol} | Price drifted {drift*100:.2f}% DOWN from last exit "
                        f"({last_exit:.6f} → {current_price:.6f}). Blocking LONG re-entry."
                    )
                    return False

        # ── ML Full AI Mode / Confirmation Mode / Pure Technical ──
        if config.ML_ENABLED and config.ML_FULL_AI_MODE:
            ml_class, ml_confidence = self._ml_scores.get(symbol, (0, 0.0))
            if ml_class not in (1, 2) or ml_confidence < config.ML_CONFIDENCE_THRESHOLD:
                return False
            
            side = "BUY" if ml_class == 1 else "SELL"
            
            # Safeguard: Spot cannot SHORT
            if not is_f and side == "SELL":
                return False

            if side == "BUY" and self.sentiment.is_blocked(symbol):
                return False

            # ── [FIX-HTF] Higher Timeframe Hard Gate ─────────────
            # If 1h trend is clearly UP   → block ALL SHORT (SELL) entries
            # If 1h trend is clearly DOWN → block ALL LONG  (BUY)  entries
            # This prevents the #1 loss pattern: AI predicts short on 15m
            # while the 1h chart is in a strong uptrend (price keeps pumping).
            htf_bias = self._get_htf_trend(df_higher)
            if htf_bias == "UP" and side == "SELL":
                logger.info(
                    f"🚫 [HTF GATE] {symbol} | 1h trend=UP — blocking SHORT entry. "
                    f"Trade WITH the higher timeframe trend."
                )
                return False
            if htf_bias == "DOWN" and side == "BUY":
                logger.info(
                    f"🚫 [HTF GATE] {symbol} | 1h trend=DOWN — blocking LONG entry. "
                    f"Trade WITH the higher timeframe trend."
                )
                return False

            # ── Counter-Trend / Falling Knife Safeguard ──
            # Calculate technical indicators on main timeframe to get RSI
            df_feat = self.strategy.calculate(df_main)
            if df_feat is not None and len(df_feat) >= 2:
                last_closed = df_feat.iloc[-2]
                rsi = last_closed.get("rsi", 50.0)
                current_price = df_feat['close'].iloc[-1]
                last_close_val = last_closed['close']
                
                # For BUY: If RSI is oversold (< 35), ensure price is showing a short-term bounce
                if side == "BUY" and rsi < 35.0:
                    if current_price <= last_close_val:
                        logger.info(
                            f"⏳ AI LONG PENDING | {symbol} | RSI {rsi:.1f} is oversold but price is still falling "
                            f"(Current: {current_price:.6f} <= Last Close: {last_close_val:.6f}). Waiting for bounce."
                        )
                        return False
                
                # For SELL: If RSI is overbought (> 65), ensure price is showing a short-term rejection
                if side == "SELL" and rsi > 65.0:
                    if current_price >= last_close_val:
                        logger.info(
                            f"⏳ AI SHORT PENDING | {symbol} | RSI {rsi:.1f} is overbought but price is still rising "
                            f"(Current: {current_price:.6f} >= Last Close: {last_close_val:.6f}). Waiting for reversal."
                        )
                        return False

            # ── [FIX-B] Candle Confirmation Filter ───────────────
            # Require at least CANDLE_CONFIRM_MIN of the last CANDLE_CONFIRM_LOOKBACK
            # candles to agree with the trade direction before entering.
            if config.CANDLE_CONFIRM_ENABLED and not self._candle_direction_confirmed(df_main, side):
                return False

            market_text = "FUTURES" if is_f else "SPOT"
            logger.info(f"🚀 AI TRIGGERED | {symbol} ({market_text}) | Side: {side} | ML Conf: {ml_confidence:.2f} → EXECUTING")
        else:
            df_main = self.strategy.calculate(df_main)
            signal, _ = self.strategy.get_signal(df_main)

            if config.ML_ENABLED:
                if signal not in ("BUY", "SELL"): return False
                
                ml_class, ml_confidence = self._ml_scores.get(symbol, (0, 0.0))
                
                # Verify if ML confirms the strategy direction:
                # signal "BUY" is confirmed by ml_class 1 (LONG)
                # signal "SELL" is confirmed by ml_class 2 (SHORT)
                is_confirmed = (signal == "BUY" and ml_class == 1) or (signal == "SELL" and ml_class == 2)
                
                if not is_confirmed or ml_confidence < config.ML_CONFIDENCE_THRESHOLD:
                    return False
                    
                side = signal
                
                # Safeguard: Spot cannot SHORT
                if not is_f and side == "SELL":
                    return False

                if side == "BUY" and self.sentiment.is_blocked(symbol):
                    return False

                # ── [FIX-HTF] Higher Timeframe Hard Gate ─────────
                htf_bias = self._get_htf_trend(df_higher)
                if htf_bias == "UP" and side == "SELL":
                    logger.info(f"🚫 [HTF GATE] {symbol} | 1h trend=UP — blocking SHORT (Confirmation Mode)")
                    return False
                if htf_bias == "DOWN" and side == "BUY":
                    logger.info(f"🚫 [HTF GATE] {symbol} | 1h trend=DOWN — blocking LONG (Confirmation Mode)")
                    return False

                # ── [FIX-B] Candle Confirmation Filter ───────────
                if config.CANDLE_CONFIRM_ENABLED and not self._candle_direction_confirmed(df_main, side):
                    return False

                market_text = "FUTURES" if is_f else "SPOT"
                logger.info(f"🚀 AI CONFIRMED | {symbol} ({market_text}) | Side: {side} | ML Conf: {ml_confidence:.2f} → EXECUTING")
            else:
                if signal not in ("BUY", "SELL"): return False
                side = signal

                # ── [FIX-B] Candle Confirmation Filter (pure tech mode) ──
                if config.CANDLE_CONFIRM_ENABLED and not self._candle_direction_confirmed(df_main, side):
                    return False

        # ── [LONG_ONLY] Safeguard ──────────────────────────────
        if getattr(config, "LONG_ONLY", False) and side == "SELL":
            logger.info(f"🚫 [LONG_ONLY] {symbol} | Trade blocked — LONG_ONLY is active in config.py")
            return False

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
                    # Set cooldown to prevent immediate re-triggering
                    self.cooldowns[symbol] = time.time() + (self.cooldown_minutes * 60)
                    self._last_exit_reason[symbol] = "FUNDING_SKIP"
                    return False

                if mins <= config.FUNDING_FEE_BLOCK_MINUTES and rate_unfavorable:
                    logger.info(
                        f"💸 FUNDING SKIP | {symbol} | {side} blocked — "
                        f"Funding in {mins:.0f}min at {rate_pct:+.4f}% (unfavorable). "
                        f"Will re-evaluate after funding."
                    )
                    # Set cooldown to prevent immediate re-triggering
                    self.cooldowns[symbol] = time.time() + (self.cooldown_minutes * 60)
                    self._last_exit_reason[symbol] = "FUNDING_SKIP"
                    return False

        # Get symbol constraints
        sym_info = self.exchange.get_symbol_info(symbol)
        if not sym_info:
            logger.warning(f"Could not get symbol info for {symbol}. Skipping.")
            return False

        current_price = self.exchange.get_current_price(symbol)
        if current_price <= 0:
            return False

        # ── [FIX] Pre-filter: skip pairs where balance can't meet minNotional ──
        # This prevents log spam from expensive pairs like BTC where step_size
        # rounds quantity to 0 or the notional value is below exchange minimum.
        min_notional = sym_info.get("min_notional", 5.0)
        step_size = sym_info.get("step_size", 0.001)
        leverage = config.FUTURES_LEVERAGE if is_f else 1
        max_notional = (usdt_balance / self.max_positions) * config.MAX_POSITION_PCT * leverage

        # Check 1: Can we even afford 1 step of this asset?
        one_step_value = step_size * current_price
        if one_step_value > max_notional:
            logger.debug(
                f"SKIP {symbol} | 1 step = {step_size} x ${current_price:.2f} = "
                f"${one_step_value:.2f} > max notional ${max_notional:.2f}"
            )
            return False

        # Check 2: Will the floored quantity meet minNotional?
        if max_notional < min_notional:
            logger.debug(
                f"SKIP {symbol} | Max notional ${max_notional:.2f} < "
                f"minNotional ${min_notional:.2f}"
            )
            return False

        # Calculate ATR and dynamic SL/TP percentages
        sl_pct = config.FUTURES_STOP_LOSS_PCT if is_f else config.STOP_LOSS_PCT
        tp_pct = config.FUTURES_TAKE_PROFIT_PCT if is_f else config.TAKE_PROFIT_PCT

        if getattr(config, "DYNAMIC_ATR_SL_TP", True) and not df_main.empty and len(df_main) >= 15:
            try:
                # 14-period ATR calculation
                high_low = df_main['high'] - df_main['low']
                high_close = (df_main['high'] - df_main['close'].shift()).abs()
                low_close = (df_main['low'] - df_main['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                close_val = df_main['close'].iloc[-1]
                
                if close_val > 0:
                    atr_pct = atr / close_val
                    sl_pct = atr_pct * getattr(config, "ATR_SL_MULTIPLE", 1.5)
                    sl_pct = max(getattr(config, "MIN_STOP_LOSS_PCT", 0.008), 
                                 min(getattr(config, "MAX_STOP_LOSS_PCT", 0.025), sl_pct))

                    tp_pct = atr_pct * getattr(config, "ATR_TP_MULTIPLE", 3.0)
                    tp_pct = max(getattr(config, "MIN_TAKE_PROFIT_PCT", 0.015), 
                                 min(getattr(config, "MAX_TAKE_PROFIT_PCT", 0.060), tp_pct))
            except Exception as e:
                logger.warning(f"Failed to calculate ATR for dynamic SL/TP on {symbol}: {e}")

        # Divide usdt balance by max_positions for per-trade allocation
        usdt_per_trade = usdt_balance / self.max_positions

        # Calculate position size
        quantity = self.risk_manager.calculate_quantity(
            usdt_balance=usdt_per_trade,
            current_price=current_price,
            symbol_info=sym_info,
            sl_pct=sl_pct,
            is_futures=is_f
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
            tp_pct=tp_pct,
            sl_pct=sl_pct
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
                "opened_at":     pos.opened_at.isoformat() + "Z",
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
            "ml_confidence_threshold": config.ML_CONFIDENCE_THRESHOLD,
        }

        if first:
            base.update(first)

        return base

"""
bot/risk_manager.py — Position sizing, stop-loss/take-profit, and trade state

Responsibilities:
  - Calculate safe buy quantity (respects exchange LOT_SIZE and MIN_NOTIONAL)
  - Track multiple concurrent open positions in memory (multi-position support)
  - Determine if TP, SL, or max-hold exit conditions are met
"""

from __future__ import annotations  # Python 3.9 compatibility

import json
import os
import math
import logging
from datetime import datetime
from typing import Optional, Dict

import config

logger = logging.getLogger("bot")

# Binance maker/taker fee (0.1% per side, 0.2% round-trip on spot)
FEE_RATE = 0.001


class Position:
    """Represents a single open trade."""

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        take_profit: float,
        stop_loss: float,
        side: str = "BUY", # BUY for Long, SELL for Short
        is_futures: bool = False
    ) -> None:
        self.symbol      = symbol
        self.entry_price = entry_price
        self.quantity    = quantity
        self.take_profit = take_profit
        self.stop_loss   = stop_loss
        self.side        = side
        self.is_futures  = is_futures
        self.opened_at   = datetime.utcnow()
        self.candles_held: int = 0
        
        # Protection tracking
        self.peak_price: float = entry_price # Best price reached (High for Long, Low for Short)
        self.trailing_active: bool = False
        self.be_active: bool = False

    def to_dict(self) -> dict:
        """Convert position to serializable dict."""
        return {
            "symbol":      self.symbol,
            "entry_price": self.entry_price,
            "quantity":    self.quantity,
            "take_profit": self.take_profit,
            "stop_loss":   self.stop_loss,
            "side":        self.side,
            "opened_at":   self.opened_at.isoformat(),
            "peak_price":   self.peak_price,
            "trailing_active": self.trailing_active,
            "be_active":   self.be_active,
            "is_futures":  self.is_futures
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Create position from dict."""
        pos = cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            take_profit=data["take_profit"],
            stop_loss=data["stop_loss"],
            side=data.get("side", "BUY"),
            is_futures=data.get("is_futures", False)
        )
        pos.opened_at = datetime.fromisoformat(data["opened_at"])
        pos.peak_price = data.get("peak_price", data.get("max_price", pos.entry_price))
        pos.trailing_active = data.get("trailing_active", False)
        pos.be_active = data.get("be_active", False)
        return pos

    @property
    def cost_usdt(self) -> float:
        return self.entry_price * self.quantity

    @property
    def duration_minutes(self) -> float:
        delta = datetime.utcnow() - self.opened_at
        return delta.total_seconds() / 60.0

    def unrealized_pnl(self, current_price: float) -> float:
        """Gross P&L in USDT (before fees)."""
        if self.side == "BUY":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.side == "BUY":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100

    def estimate_funding_fee(self, mark_price: float, funding_rate: float) -> float:
        """
        Estimate the USDT cost of the next funding payment.

        Binance formula: Funding Fee = Position Size (contracts) × Mark Price × Funding Rate
        For LONG (BUY)  : positive rate → we PAY (negative for us)
        For SHORT (SELL): positive rate → we RECEIVE (positive for us)

        Returns the net impact in USDT (negative = cost, positive = income).
        """
        notional = self.quantity * mark_price
        fee = notional * funding_rate  # raw fee (positive = longs pay)
        if self.side == "BUY":
            return -fee   # long pays when rate > 0, receives when rate < 0
        else:
            return fee    # short receives when rate > 0, pays when rate < 0

    def funding_fee_pct(self, mark_price: float, funding_rate: float) -> float:
        """Funding fee as a % of position cost."""
        cost = self.quantity * self.entry_price
        if cost == 0:
            return 0.0
        return (self.estimate_funding_fee(mark_price, funding_rate) / cost) * 100

    def __repr__(self) -> str:
        return (
            f"Position({self.side} {self.symbol} | entry={self.entry_price} | "
            f"qty={self.quantity} | TP={self.take_profit:.6f} | SL={self.stop_loss:.6f})"
        )


class RiskManager:
    """Manages trading state and exit conditions. Supports multiple concurrent positions."""

    def __init__(self) -> None:
        # Multi-position: keyed by symbol
        self.positions: Dict[str, Position] = {}
        self.state_file = "logs/active_position.json"
        self._load_state()

    def _save_state(self) -> None:
        """Save all current positions to file."""
        try:
            os.makedirs("logs", exist_ok=True)
            if self.positions:
                data = {sym: pos.to_dict() for sym, pos in self.positions.items()}
                with open(self.state_file, "w") as f:
                    json.dump(data, f)
            else:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load positions from file if it exists (backward compat with old single-pos format)."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            # Detect old single-position format (has "symbol" at top level)
            if "symbol" in data and isinstance(data.get("symbol"), str):
                pos = Position.from_dict(data)
                self.positions[pos.symbol] = pos
                logger.info(f"💾 Restored single position for {pos.symbol} (migrated to multi-pos)")
            else:
                # New multi-position format
                for sym, pos_data in data.items():
                    pos = Position.from_dict(pos_data)
                    self.positions[sym] = pos
                logger.info(f"💾 Restored {len(self.positions)} position(s): {list(self.positions.keys())}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # ── Position State ────────────────────────────────────────

    @property
    def has_positions(self) -> bool:
        """True if any position is open."""
        return bool(self.positions)

    @property
    def has_position(self) -> bool:
        """Backward compat: True if any position is open."""
        return bool(self.positions)

    @property
    def position(self) -> Optional[Position]:
        """Backward compat: return first open position (or None)."""
        if self.positions:
            return next(iter(self.positions.values()))
        return None

    def has_position_for(self, symbol: str) -> bool:
        """True if a position is already open for this symbol."""
        return symbol in self.positions

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str = "BUY",
        is_futures: bool = False,
    ) -> Position:
        """Record a newly opened position."""
        # Use futures-specific TP/SL if available, otherwise spot
        tp_pct = config.FUTURES_TAKE_PROFIT_PCT if is_futures else config.TAKE_PROFIT_PCT
        sl_pct = config.FUTURES_STOP_LOSS_PCT   if is_futures else config.STOP_LOSS_PCT

        if side == "BUY":
            take_profit = entry_price * (1 + tp_pct)
            stop_loss   = entry_price * (1 - sl_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - tp_pct)
            stop_loss   = entry_price * (1 + sl_pct)

        pos = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            side=side,
            is_futures=is_futures
        )
        self.positions[symbol] = pos
        self._save_state()

        market_label = "Futures" if is_futures else "Spot"
        logger.info(
            f"📂 Position opened | {symbol} ({market_label}) | {side} | Entry: {entry_price:.6f} | "
            f"Qty: {quantity} | TP: {take_profit:.6f} ({tp_pct*100:.1f}%) | SL: {stop_loss:.6f} ({sl_pct*100:.1f}%)"
        )
        return pos

    def close_position(self, symbol: str = None) -> None:
        """Close an open position by symbol. If no symbol given, close all (backward compat)."""
        if symbol is None:
            # Backward compat: close first/only position
            if self.positions:
                sym = next(iter(self.positions))
                del self.positions[sym]
        elif symbol in self.positions:
            del self.positions[symbol]
        self._save_state()

    # ── Quantity Calculation ──────────────────────────────────

    def calculate_quantity(
        self,
        usdt_balance: float,
        current_price: float,
        symbol_info: dict,
    ) -> float:
        """
        Calculate how many units to buy given:
          - Available USDT balance
          - Current price
          - Exchange LOT_SIZE and MIN_NOTIONAL constraints

        Returns 0.0 if a valid quantity cannot be computed.
        """
        if current_price <= 0 or usdt_balance <= 0:
            return 0.0

        # Apply max position percentage
        usdt_to_spend = usdt_balance * config.MAX_POSITION_PCT

        # Account for estimated fee in the spend amount
        usdt_to_spend *= (1 - FEE_RATE)

        raw_qty  = usdt_to_spend / current_price
        step     = symbol_info.get("step_size", 0.001)
        min_qty  = symbol_info.get("min_qty", 0.001)
        min_not  = symbol_info.get("min_notional", 5.0)
        prec     = symbol_info.get("base_precision", 3)

        # Floor to valid step size
        if step > 0:
            qty = math.floor(raw_qty / step) * step
        else:
            qty = raw_qty

        qty = round(qty, prec)

        # Validate constraints
        if qty < min_qty:
            logger.warning(
                f"Quantity {qty} < minQty {min_qty} for "
                f"spend={usdt_to_spend:.2f} USDT @ {current_price}"
            )
            return 0.0

        if qty * current_price < min_not:
            logger.warning(
                f"Order value {qty * current_price:.2f} < minNotional {min_not}"
            )
            return 0.0

        return qty

    # ── Exit Conditions ───────────────────────────────────────

    def check_exit(self, symbol: str, current_price: float, reversal_signal: bool | str) -> Optional[str]:
        """
        Check whether an open position should be closed.
        Returns the exit reason string, or None.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        is_long = (pos.side == "BUY")

        # ── Update Peak Price ──────────────────────────────
        if is_long:
            if current_price > pos.peak_price:
                pos.peak_price = current_price
        else: # Short
            if current_price < pos.peak_price or pos.peak_price == 0:
                pos.peak_price = current_price

        pnl_pct = pos.unrealized_pnl_pct(current_price)

        # Determine settings based on Spot vs Futures
        if getattr(pos, "is_futures", False):
            ts_enabled = config.FUTURES_TRAILING_STOP_ENABLED if hasattr(config, "FUTURES_TRAILING_STOP_ENABLED") else config.TRAILING_STOP_ENABLED
            ts_activation = config.FUTURES_TRAILING_STOP_ACTIVATION_PCT if hasattr(config, "FUTURES_TRAILING_STOP_ACTIVATION_PCT") else config.TRAILING_STOP_ACTIVATION_PCT
            ts_callback = config.FUTURES_TRAILING_STOP_CALLBACK_PCT if hasattr(config, "FUTURES_TRAILING_STOP_CALLBACK_PCT") else config.TRAILING_STOP_CALLBACK_PCT
            
            be_enabled = config.FUTURES_BREAK_EVEN_ENABLED if hasattr(config, "FUTURES_BREAK_EVEN_ENABLED") else config.BREAK_EVEN_ENABLED
            be_activation = config.FUTURES_BREAK_EVEN_ACTIVATION_PCT if hasattr(config, "FUTURES_BREAK_EVEN_ACTIVATION_PCT") else config.BREAK_EVEN_ACTIVATION_PCT
        else:
            ts_enabled = config.TRAILING_STOP_ENABLED
            ts_activation = config.TRAILING_STOP_ACTIVATION_PCT
            ts_callback = config.TRAILING_STOP_CALLBACK_PCT
            
            be_enabled = config.BREAK_EVEN_ENABLED
            be_activation = config.BREAK_EVEN_ACTIVATION_PCT

        # 1. Break-Even Check
        if be_enabled and not pos.be_active:
            if pnl_pct >= (be_activation * 100):
                pos.be_active = True
                # Move SL to entry
                if is_long:
                    if pos.entry_price > pos.stop_loss:
                        pos.stop_loss = pos.entry_price
                        logger.info(f"🛡️  BREAK-EVEN (LONG) | {pos.symbol} SL -> {pos.entry_price:.6f}")
                else: # Short
                    if pos.entry_price < pos.stop_loss:
                        pos.stop_loss = pos.entry_price
                        logger.info(f"🛡️  BREAK-EVEN (SHORT) | {pos.symbol} SL -> {pos.entry_price:.6f}")

        # 2. Trailing Stop Check
        if ts_enabled:
            if not pos.trailing_active and pnl_pct >= (ts_activation * 100):
                pos.trailing_active = True
                logger.info(f"🛡️  TRAILING STOP ACTIVATED for {pos.symbol} at {current_price:.6f}")

            if pos.trailing_active:
                if is_long:
                    new_sl = pos.peak_price * (1 - ts_callback)
                    if new_sl > pos.stop_loss:
                        pos.stop_loss = new_sl
                else: # Short
                    new_sl = pos.peak_price * (1 + ts_callback)
                    if new_sl < pos.stop_loss:
                        pos.stop_loss = new_sl

                # Check hit
                if is_long and current_price <= pos.stop_loss:
                    logger.info(f"🛡️  TRAILING STOP hit | Price: {current_price:.6f} ≤ SL: {pos.stop_loss:.6f}")
                    return "TRAILING_STOP"
                elif not is_long and current_price >= pos.stop_loss:
                    logger.info(f"🛡️  TRAILING STOP hit | Price: {current_price:.6f} ≥ SL: {pos.stop_loss:.6f}")
                    return "TRAILING_STOP"

        # ── Standard TP / SL ──────────────────────────────
        if is_long:
            if current_price >= pos.take_profit:
                return "TAKE_PROFIT"
            if current_price <= pos.stop_loss:
                return "STOP_LOSS"
        else: # Short
            if current_price <= pos.take_profit:
                return "TAKE_PROFIT"
            if current_price >= pos.stop_loss:
                return "STOP_LOSS"

        # ── Reversal (strategy / ML signal) ────────────────
        if reversal_signal:
            if isinstance(reversal_signal, str):
                return reversal_signal
            return "REVERSAL_SIGNAL"

        # ── Max hold time ─────────────────────────────────
        max_minutes = config.MAX_HOLD_CANDLES * 5
        if pos.duration_minutes >= max_minutes:
            return "MAX_HOLD"

        return None

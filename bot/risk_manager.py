"""
bot/risk_manager.py — Position sizing, stop-loss/take-profit, and trade state

Responsibilities:
  - Calculate safe buy quantity (respects exchange LOT_SIZE and MIN_NOTIONAL)
  - Track the current open position in memory
  - Determine if TP, SL, or max-hold exit conditions are met
"""

from __future__ import annotations  # Python 3.9 compatibility

import json
import os
import math
import logging
from datetime import datetime
from typing import Optional

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
    ) -> None:
        self.symbol      = symbol
        self.entry_price = entry_price
        self.quantity    = quantity
        self.take_profit = take_profit
        self.stop_loss   = stop_loss
        self.opened_at   = datetime.utcnow()
        self.candles_held: int = 0
        
        # Protection tracking
        self.max_price: float = entry_price
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
            "opened_at":   self.opened_at.isoformat(),
            "max_price":   self.max_price,
            "trailing_active": self.trailing_active,
            "be_active":   self.be_active
        }

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        """Create position from dict."""
        pos = cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            take_profit=data["take_profit"],
            stop_loss=data["stop_loss"]
        )
        pos.opened_at = datetime.fromisoformat(data["opened_at"])
        pos.max_price = data.get("max_price", pos.entry_price)
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
        return (current_price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        return ((current_price - self.entry_price) / self.entry_price) * 100

    def __repr__(self) -> str:
        return (
            f"Position({self.symbol} | entry={self.entry_price} | "
            f"qty={self.quantity} | TP={self.take_profit:.6f} | SL={self.stop_loss:.6f})"
        )


class RiskManager:
    """Manages trading state and exit conditions."""

    def __init__(self) -> None:
        self.position: Optional[Position] = None
        self.state_file = "logs/active_position.json"
        self._load_state()

    def _save_state(self) -> None:
        """Save current position to file."""
        try:
            os.makedirs("logs", exist_ok=True)
            if self.position:
                with open(self.state_file, "w") as f:
                    json.dump(self.position.to_dict(), f)
            else:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load position from file if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.position = Position.from_dict(data)
                    logger.info(f"💾 Restored active position for {self.position.symbol}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    # ── Position State ────────────────────────────────────────

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
    ) -> Position:
        """Record a newly opened position."""
        take_profit = entry_price * (1 + config.TAKE_PROFIT_PCT)
        stop_loss   = entry_price * (1 - config.STOP_LOSS_PCT)

        self.position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        self._save_state()

        logger.info(
            f"📂 Position opened | {symbol} | Entry: {entry_price:.6f} | "
            f"Qty: {quantity} | TP: {take_profit:.6f} | SL: {stop_loss:.6f}"
        )
        return self.position

    def close_position(self) -> None:
        """Clear the current position state."""
        self.position = None
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

    def check_exit(self, current_price: float, sell_signal: bool) -> Optional[str]:
        """
        Check whether the current position should be closed.

        Returns:
          "TAKE_PROFIT"  — price hit TP level
          "STOP_LOSS"    — price hit SL level
          "EMA_REVERSAL" — strategy issued a SELL signal
          "MAX_HOLD"     — position held too many candles
          None           — keep holding
        """
        if not self.has_position:
            return None

        pos = self.position

        # ── Trailing & Break-Even Logic ────────────────────
        # Always update max_price reached so far
        if current_price > pos.max_price:
            pos.max_price = current_price

        pnl_pct = pos.unrealized_pnl_pct(current_price)

        # 1. Break-Even Check
        if config.BREAK_EVEN_ENABLED and not pos.be_active:
            if pnl_pct >= (config.BREAK_EVEN_ACTIVATION_PCT * 100):
                pos.be_active = True
                if pos.entry_price > pos.stop_loss:
                    pos.stop_loss = pos.entry_price
                    logger.info(f"🛡️  BREAK-EVEN ACTIVATED for {pos.symbol} (SL moved to {pos.entry_price:.6f})")

        # 2. Trailing Stop Check
        if config.TRAILING_STOP_ENABLED:
            # Activate trailing if threshold hit
            if not pos.trailing_active and pnl_pct >= (config.TRAILING_STOP_ACTIVATION_PCT * 100):
                pos.trailing_active = True
                logger.info(f"🛡️  TRAILING STOP ACTIVATED for {pos.symbol} at {current_price:.6f}")

            # If active, trail the stop loss behind the peak price
            if pos.trailing_active:
                new_sl = pos.max_price * (1 - config.TRAILING_STOP_CALLBACK_PCT)
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl

                # Check if price hit the trailed SL
                if current_price <= pos.stop_loss:
                    logger.info(
                        f"🛡️  TRAILING STOP hit | {pos.symbol} | "
                        f"Price: {current_price:.6f} ≤ Trailed SL: {pos.stop_loss:.6f}"
                    )
                    return "TRAILING_STOP"

        # ── Standard Take Profit ──────────────────────────
        # Only check if trailing isn't handling it, or as a hard cap
        if current_price >= pos.take_profit:
            logger.info(f"🎯 TAKE PROFIT hit | {pos.symbol} | Price: {current_price:.6f}")
            return "TAKE_PROFIT"

        # ── Standard Stop Loss (Hard Floor) ───────────────
        if current_price <= pos.stop_loss:
            logger.info(f"🛑 STOP LOSS hit | {pos.symbol} | Price: {current_price:.6f}")
            return "STOP_LOSS"

        # ── EMA Reversal (strategy signal) ────────────────
        if sell_signal:
            logger.info(f"🔁 EMA REVERSAL exit | {pos.symbol} | Price: {current_price:.6f}")
            return "EMA_REVERSAL"

        # ── Max hold time ─────────────────────────────────
        max_minutes = config.MAX_HOLD_CANDLES * 5
        if pos.duration_minutes >= max_minutes:
            logger.warning(f"⏰ MAX HOLD reached | {pos.symbol} — forcing exit")
            return "MAX_HOLD"

        return None

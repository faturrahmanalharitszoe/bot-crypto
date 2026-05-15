"""
bot/optimizer.py — Auto-Optimizer

Runs periodically in a background thread.
Reads trade history from logs/trades.csv and adjusts:
  - ML_CONFIDENCE_THRESHOLD (based on recent win-rate)

Runs every OPTIMIZER_INTERVAL_HOURS (default: 72 hours).
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict

import config

logger = logging.getLogger("bot")

_TRADE_LOG = "logs/trades.csv"


def _read_recent_trades(days: int = 1) -> List[Dict]:
    """Read trades from the last N days (default 1 day for scalping)."""
    abs_trade_log = os.path.abspath(os.path.join("logs", "trades.csv"))
    if not os.path.exists(abs_trade_log):
        return []

    cutoff = datetime.utcnow() - timedelta(days=days)
    trades = []

    try:
        with open(abs_trade_log, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        trades.append(row)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        logger.error(f"Optimizer: Failed to read trades: {e}")

    return trades


def _calculate_win_rate(trades: List[Dict]) -> float:
    """Return win rate as a float 0.0-1.0."""
    if not trades:
        return 0.5  # Neutral if no data
    wins = sum(1 for t in trades if float(t.get("pnl_usdt", 0)) >= 0)
    return wins / len(trades)


def _calculate_avg_pnl(trades: List[Dict]) -> float:
    """Return average PnL per trade in USDT."""
    if not trades:
        return 0.0
    total = sum(float(t.get("pnl_usdt", 0)) for t in trades)
    return total / len(trades)


def run_optimization() -> Dict:
    """
    Analyze recent trade history and suggest config adjustments.
    Returns a dict with the optimization results.
    """
    trades = _read_recent_trades(days=7)
    n = len(trades)

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trades_analyzed": n,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "old_threshold": config.ML_CONFIDENCE_THRESHOLD,
        "new_threshold": config.ML_CONFIDENCE_THRESHOLD,
        "action": "No change (insufficient data)",
    }

    if n < 5:
        logger.info(f"🔧 Optimizer: Only {n} trades available — need at least 5 to optimize.")
        return result

    win_rate = _calculate_win_rate(trades)
    avg_pnl  = _calculate_avg_pnl(trades)
    old_threshold = config.ML_CONFIDENCE_THRESHOLD

    result["win_rate"] = round(win_rate, 4)
    result["avg_pnl"] = round(avg_pnl, 4)

    # ── Adjustment Logic ─────────────────────────────────────
    new_threshold = old_threshold

    if win_rate < 0.40:
        # Losing too much → be more selective (raise threshold)
        new_threshold = min(0.80, old_threshold + 0.05)
        action = f"↑ Raised threshold (win rate {win_rate:.0%} < 40%)"
    elif win_rate > 0.70 and avg_pnl > 0:
        # Winning a lot → can be a bit more aggressive (lower threshold)
        new_threshold = max(0.50, old_threshold - 0.03)
        action = f"↓ Lowered threshold (win rate {win_rate:.0%} > 70%)"
    else:
        action = f"✓ No change (win rate {win_rate:.0%} — within healthy range)"

    # Apply if changed
    if new_threshold != old_threshold:
        config.ML_CONFIDENCE_THRESHOLD = new_threshold
        logger.info(
            f"🔧 Optimizer: Threshold {old_threshold:.2f} → {new_threshold:.2f} | {action}"
        )
    else:
        logger.info(f"🔧 Optimizer: {action}")

    result["new_threshold"] = round(new_threshold, 4)
    result["action"] = action

    # ── Save log ─────────────────────────────────────────────
    _save_optimization_log(result)
    return result


def _save_optimization_log(result: Dict) -> None:
    """Append optimization result to logs/optimization_log.csv."""
    log_path = "logs/optimization_log.csv"
    fieldnames = [
        "timestamp", "trades_analyzed", "win_rate", "avg_pnl",
        "old_threshold", "new_threshold", "action"
    ]
    write_header = not os.path.exists(log_path)

    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)
    except Exception as e:
        logger.error(f"Optimizer: Failed to save log: {e}")


class AutoOptimizer:
    """
    Background thread that runs optimization periodically.
    """

    def __init__(self) -> None:
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_run: datetime | None = None
        self._last_result: Dict = {}

    def start(self) -> None:
        """Start the optimizer background thread."""
        if not config.OPTIMIZER_ENABLED:
            logger.info("🔧 Auto-Optimizer is disabled.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="optimizer"
        )
        self._thread.start()
        logger.info(
            f"🔧 Auto-Optimizer started | "
            f"Interval: every {config.OPTIMIZER_INTERVAL_HOURS}h"
        )

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        interval_secs = config.OPTIMIZER_INTERVAL_HOURS * 3600
        while self._running:
            self._last_result = run_optimization()
            self._last_run = datetime.utcnow()
            time.sleep(interval_secs)

    @property
    def last_result(self) -> Dict:
        return self._last_result

    @property
    def last_run(self) -> str:
        if self._last_run:
            return self._last_run.strftime("%Y-%m-%d %H:%M:%S")
        return "Never"

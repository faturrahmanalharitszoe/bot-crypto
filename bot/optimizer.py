"""
bot/optimizer.py — Auto-Optimizer (REVAMP v2)

[REVAMP v2] Key changes:
  - Smarter threshold adjustment with smaller steps
  - Analyzes win rate BY exit reason (TP vs SL vs Trailing)
  - Adjusts trailing stop parameters if trailing exits are underperforming
  - Longer analysis window (7 days) for more stable decisions
  - Won't lower threshold below 0.60 (safety floor)

Runs every OPTIMIZER_INTERVAL_HOURS (default: 2 hours).
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


def _read_recent_trades(days: int = 7) -> List[Dict]:
    """Read trades from the last N days."""
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
        return 0.5
    wins = sum(1 for t in trades if float(t.get("pnl_usdt", 0)) > 0)
    return wins / len(trades)


def _calculate_avg_pnl(trades: List[Dict]) -> float:
    """Return average PnL per trade in USDT."""
    if not trades:
        return 0.0
    total = sum(float(t.get("pnl_usdt", 0)) for t in trades)
    return total / len(trades)


def _analyze_by_exit_reason(trades: List[Dict]) -> Dict[str, Dict]:
    """
    [REVAMP v2] Analyze performance broken down by exit reason.
    Returns: { reason: { count, win_rate, avg_pnl } }
    """
    by_reason = {}
    for t in trades:
        reason = t.get("exit_reason", "UNKNOWN")
        if reason not in by_reason:
            by_reason[reason] = {"trades": [], "count": 0}
        by_reason[reason]["trades"].append(t)
        by_reason[reason]["count"] += 1

    result = {}
    for reason, data in by_reason.items():
        trades_list = data["trades"]
        wins = sum(1 for t in trades_list if float(t.get("pnl_usdt", 0)) > 0)
        total_pnl = sum(float(t.get("pnl_usdt", 0)) for t in trades_list)
        result[reason] = {
            "count": data["count"],
            "win_rate": wins / len(trades_list) if trades_list else 0,
            "avg_pnl": total_pnl / len(trades_list) if trades_list else 0,
            "total_pnl": total_pnl,
        }
    return result


def _calculate_profit_factor(trades: List[Dict]) -> float:
    """
    [REVAMP v2] Profit Factor = Gross Wins / Gross Losses.
    PF > 1.0 = profitable, PF < 1.0 = losing money.
    """
    gross_wins = sum(float(t.get("pnl_usdt", 0)) for t in trades if float(t.get("pnl_usdt", 0)) > 0)
    gross_losses = abs(sum(float(t.get("pnl_usdt", 0)) for t in trades if float(t.get("pnl_usdt", 0)) < 0))
    if gross_losses == 0:
        return 10.0  # No losses = excellent
    return gross_wins / gross_losses


def run_optimization() -> Dict:
    """
    [REVAMP v2] Smarter optimization with multi-factor analysis.
    """
    trades = _read_recent_trades(days=7)
    n = len(trades)

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trades_analyzed": n,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "profit_factor": 0.0,
        "old_threshold": config.ML_CONFIDENCE_THRESHOLD,
        "new_threshold": config.ML_CONFIDENCE_THRESHOLD,
        "action": "No change (insufficient data)",
    }

    if n < 8:
        logger.info(f"🔧 Optimizer: Only {n} trades available — need at least 8 to optimize.")
        return result

    win_rate = _calculate_win_rate(trades)
    avg_pnl  = _calculate_avg_pnl(trades)
    profit_factor = _calculate_profit_factor(trades)
    by_reason = _analyze_by_exit_reason(trades)
    old_threshold = config.ML_CONFIDENCE_THRESHOLD

    result["win_rate"] = round(win_rate, 4)
    result["avg_pnl"] = round(avg_pnl, 4)
    result["profit_factor"] = round(profit_factor, 3)

    # Log breakdown by exit reason
    logger.info(f"🔧 Optimizer Analysis | {n} trades | WR: {win_rate:.0%} | "
                f"Avg PnL: {avg_pnl:+.4f} | PF: {profit_factor:.2f}")
    for reason, stats in by_reason.items():
        logger.info(f"   └─ {reason}: {stats['count']} trades | "
                    f"WR: {stats['win_rate']:.0%} | Avg: {stats['avg_pnl']:+.4f}")

    # ── Adjustment Logic (REVAMP v2: smaller, smarter steps) ──
    new_threshold = old_threshold
    actions = []

    # Primary: Adjust ML confidence threshold based on win rate
    if win_rate < 0.35:
        # Severely losing → raise threshold significantly
        new_threshold = min(0.80, old_threshold + 0.04)
        actions.append(f"↑ Threshold +0.04 (WR {win_rate:.0%} < 35%)")
    elif win_rate < 0.45:
        # Losing → raise threshold moderately
        new_threshold = min(0.78, old_threshold + 0.02)
        actions.append(f"↑ Threshold +0.02 (WR {win_rate:.0%} < 45%)")
    elif win_rate > 0.65 and profit_factor > 1.5:
        # Winning well AND profitable → can be slightly more aggressive
        new_threshold = max(0.60, old_threshold - 0.02)
        actions.append(f"↓ Threshold -0.02 (WR {win_rate:.0%} > 65%, PF {profit_factor:.1f})")
    else:
        actions.append(f"✓ Threshold stable (WR {win_rate:.0%}, PF {profit_factor:.2f})")

    # Secondary: Check if STOP_LOSS exits dominate (SL too tight)
    sl_stats = by_reason.get("STOP_LOSS", {})
    sl_count = sl_stats.get("count", 0)
    sl_ratio = sl_count / n if n > 0 else 0

    if sl_ratio > 0.50:
        # More than 50% of exits are stop losses — SL is probably too tight
        actions.append(f"⚠️ SL exits = {sl_ratio:.0%} of trades — consider widening SL")
        logger.warning(
            f"🔧 Optimizer: {sl_ratio:.0%} of exits are STOP_LOSS. "
            f"Current SL may be too tight for market conditions."
        )

    # Check if trailing stops are cutting winners short
    ts_stats = by_reason.get("TRAILING_STOP", {})
    tp_stats = by_reason.get("TAKE_PROFIT", {})
    if ts_stats.get("count", 0) > 3:
        ts_avg = ts_stats.get("avg_pnl", 0)
        tp_avg = tp_stats.get("avg_pnl", 0) if tp_stats.get("count", 0) > 0 else 0
        if tp_avg > 0 and ts_avg > 0 and ts_avg < tp_avg * 0.4:
            actions.append(f"⚠️ Trailing avg ({ts_avg:+.4f}) << TP avg ({tp_avg:+.4f}) — trail too tight")

    # Apply threshold change
    if new_threshold != old_threshold:
        config.ML_CONFIDENCE_THRESHOLD = new_threshold
        logger.info(
            f"🔧 Optimizer: Threshold {old_threshold:.2f} → {new_threshold:.2f}"
        )

    action_str = " | ".join(actions)
    result["new_threshold"] = round(new_threshold, 4)
    result["action"] = action_str
    logger.info(f"🔧 Actions: {action_str}")

    _save_optimization_log(result)
    return result


def _save_optimization_log(result: Dict) -> None:
    """Append optimization result to logs/optimization_log.csv."""
    log_path = "logs/optimization_log.csv"
    fieldnames = [
        "timestamp", "trades_analyzed", "win_rate", "avg_pnl", "profit_factor",
        "old_threshold", "new_threshold", "action"
    ]
    write_header = not os.path.exists(log_path)

    try:
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)
    except Exception as e:
        logger.error(f"Optimizer: Failed to save log: {e}")


class AutoOptimizer:
    """Background thread that runs optimization periodically."""

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
        # Wait 5 minutes before first run to let bot accumulate some data
        time.sleep(300)
        while self._running:
            try:
                self._last_result = run_optimization()
                self._last_run = datetime.utcnow()
            except Exception as e:
                logger.error(f"Optimizer loop error: {e}")
            time.sleep(interval_secs)

    @property
    def last_result(self) -> Dict:
        return self._last_result

    @property
    def last_run(self) -> str:
        if self._last_run:
            return self._last_run.strftime("%Y-%m-%d %H:%M:%S")
        return "Never"

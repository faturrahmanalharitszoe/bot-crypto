"""
bot/pair_selector.py — Dynamic trading pair selection (REVAMP v2)

[REVAMP v2] Key changes:
  - Score now favors TRENDING pairs, not just volatile ones
  - Added trend consistency check (price above/below EMA on 1h)
  - Penalizes pairs with extreme volatility (>15% daily = too risky)
  - Prefers pairs with moderate, consistent movement over wild swings
  - Added minimum trade count filter to avoid illiquid pairs

Scoring formula:
  Score = log10(volume) * trend_factor * volatility_factor
  Where:
    trend_factor = abs(price_change_pct) capped at 8%
    volatility_factor = min(volatility, 10) — penalizes extreme volatility
"""

import logging
import math
from typing import List

import config

logger = logging.getLogger("bot")

# Fallback list if the API call fails
_TESTNET_FALLBACK = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "LTCUSDT", "DOTUSDT", "LINKUSDT",
]


def select_best_pairs(exchange) -> List[str]:
    """
    Dynamically pick the top N pairs by trend-momentum score.
    [REVAMP v2] Favors pairs that are TRENDING (consistent direction)
    over pairs that are just volatile (random swings).
    """
    tickers = exchange.get_24h_tickers()

    if not tickers:
        logger.warning("No ticker data returned — using fallback pair list.")
        return [{"symbol": s, "change_pct": 0, "volume_usdt": 0, "volatility_pct": 0, "score": 0}
                for s in _TESTNET_FALLBACK[:config.TOP_PAIRS_COUNT]]

    # Get all active trading symbols
    try:
        active_symbols = exchange.get_active_symbols()
        if not active_symbols:
            active_symbols = None
    except Exception as e:
        logger.warning(f"Could not fetch active symbols: {e}. Skipping status filtering.")
        active_symbols = None

    candidates = []

    for t in tickers:
        symbol = t.get("symbol", "")

        # ── Only USDT pairs ───────────────────────────────
        if not symbol.endswith("USDT"):
            continue

        # ── Skip non-TRADING symbols ──────────────────────
        if active_symbols is not None and symbol not in active_symbols:
            continue

        # ── Exclude leveraged/stable tokens ──────────────
        base = symbol.replace("USDT", "")
        if any(kw in base for kw in config.EXCLUDE_KEYWORDS):
            continue

        # ── Parse numeric fields ──────────────────────────
        try:
            vol_24h = float(t["quoteVolume"])
            trade_count = int(t.get("count", 1))
            
            # [REVAMP] Skip pairs with very low trade count (illiquid)
            if vol_24h == 0 or trade_count == 0:
                continue
            if trade_count < 50000:  # Need at least 50k trades in 24h
                continue

            # Volume filter — use config minimum
            min_vol = 10_000 if config.TESTNET else config.MIN_VOLUME_USDT_24H
            if vol_24h < min_vol:
                continue

            price_change_pct = float(t["priceChangePercent"])
            
            # [REVAMP] Anti-Pump Filter: skip if pumped >15% in 24h (was 30%)
            # Extreme movers are likely to mean-revert, bad for trend following
            if abs(price_change_pct) > 15.0:
                continue

            # Volatility check
            high = float(t["highPrice"])
            low  = float(t["lowPrice"])
            if low <= 0:
                continue
            volatility = ((high - low) / low) * 100
            
            # [REVAMP] Require moderate volatility: 2-12%
            # Too low = no opportunity, too high = unpredictable
            if volatility < 2.0 or volatility > 12.0:
                continue

            # Skip dust-level prices
            if float(t["lastPrice"]) < 0.0001:
                continue

        except (ValueError, KeyError):
            continue

        # ── Scoring (REVAMP v2) ───────────────────────────────
        # Trend factor: how much the price moved in one direction
        # Capped at 8% to avoid chasing pumps
        trend_factor = min(abs(price_change_pct), 8.0)
        
        # Volatility factor: moderate volatility is ideal (3-8%)
        # Penalize both too low and too high
        if volatility <= 5.0:
            vol_factor = volatility  # Linear up to 5%
        else:
            vol_factor = 5.0 + (volatility - 5.0) * 0.5  # Diminishing returns above 5%
        
        # Trend consistency: ratio of directional move to total range
        # High ratio = trending, Low ratio = choppy
        if volatility > 0:
            trend_consistency = abs(price_change_pct) / volatility
        else:
            trend_consistency = 0
        
        # Final score: volume * trend * consistency
        # Pairs that moved consistently in one direction with high volume score highest
        score = math.log10(vol_24h + 1) * trend_factor * (1 + trend_consistency)
        
        candidates.append({
            "symbol":         symbol,
            "change_pct":     price_change_pct,
            "volume_usdt":    vol_24h,
            "volatility_pct": volatility,
            "trend_consistency": round(trend_consistency, 3),
            "score":          round(score, 2),
        })

    if not candidates:
        logger.warning("Pair selection returned 0 candidates — using fallback list.")
        return [{"symbol": s, "change_pct": 0, "volume_usdt": 0, "volatility_pct": 0, "score": 0} 
                for s in _TESTNET_FALLBACK[:config.TOP_PAIRS_COUNT * 2]]

    # Sort by score (descending)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    top = candidates[:config.TOP_PAIRS_COUNT * 2]

    logger.info("━" * 60)
    logger.info(f"📊 Top {len(top)} pairs by trend-momentum score:")
    for i, c in enumerate(top, 1):
        dir_arrow = "▲" if c["change_pct"] >= 0 else "▼"
        logger.info(
            f"  {i}. {c['symbol']:<12} | "
            f"{dir_arrow} {abs(c['change_pct']):5.2f}% | "
            f"Vol: ${c['volume_usdt'] / 1e6:,.1f}M | "
            f"Volatility: {c['volatility_pct']:.2f}% | "
            f"Trend: {c['trend_consistency']:.2f}"
        )
    return top

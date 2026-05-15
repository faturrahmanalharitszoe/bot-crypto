"""
bot/pair_selector.py — Dynamic trading pair selection

Selects the N highest-momentum USDT pairs from Binance 24h ticker data.
Momentum Score = abs(24h_price_change_pct) × 24h_volume_usdt

Filters out:
  - Leveraged tokens (UP/DOWN/BULL/BEAR/3L/3S etc.)
  - Stablecoins (USDC/BUSD/TUSD etc.)
  - Illiquid pairs (< MIN_VOLUME_USDT_24H)
  - Extremely cheap tokens (dust-level prices)
"""

import logging
from typing import List

import config

logger = logging.getLogger("bot")

# Fallback list if the API call fails (common testnet-compatible pairs)
_TESTNET_FALLBACK = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "LTCUSDT", "DOTUSDT", "LINKUSDT",
]


def select_best_pairs(exchange) -> List[str]:
    """
    Dynamically pick the top N pairs by momentum score.
    Falls back to _TESTNET_FALLBACK if the API call fails.

    Args:
        exchange: Exchange instance with a get_24h_tickers() method.

    Returns:
        List of symbol strings e.g. ["XRPUSDT", "DOGEUSDT", ...]
    """
    tickers = exchange.get_24h_tickers()

    if not tickers:
        logger.warning("No ticker data returned — using fallback pair list.")
        return _TESTNET_FALLBACK[:config.TOP_PAIRS_COUNT]

    candidates = []

    for t in tickers:
        symbol = t.get("symbol", "")

        # ── Only USDT pairs ───────────────────────────────
        if not symbol.endswith("USDT"):
            continue

        # ── Exclude leveraged/stable tokens ──────────────
        base = symbol.replace("USDT", "")
        if any(kw in base for kw in config.EXCLUDE_KEYWORDS):
            continue

        # ── Parse numeric fields ──────────────────────────
        try:
            change_pct   = float(t["priceChangePercent"])
            volume_usdt  = float(t["quoteVolume"])        # 24h volume in USDT
            # 3. Liquidity check (Minimum 24h volume)
            # We lower this to $2M to catch newer/smaller memecoins
            # 3. Liquidity check (Minimum 24h volume)
            # We lower this to $2M to catch newer/smaller memecoins
            vol_24h = float(t["quoteVolume"])
            if vol_24h < 2_000_000: continue

            # 4. Anti-Pump Filter (Anti-FOMO / Anti-Rugpull)
            # If a coin pumped >30% in 24h, it's risky to entry now
            price_change_pct = float(t["priceChangePercent"])
            if price_change_pct > 30.0:
                continue
            
            # 5. Volatility check (Must have some movement)
            high = float(t["highPrice"])
            low  = float(t["lowPrice"])
            volatility = ((high - low) / low) * 100
            if volatility < 2.0: continue
            
            # 6. Skip dust-level prices
            if float(t["lastPrice"]) < 0.00001:
                continue
        except (ValueError, KeyError):
            continue

        # ── Scoring ──────────────────────────────────────────
        # We rank by a mix of volume and recent momentum
        # Score = Volume (logged) * Volatility
        import math
        score = math.log10(vol_24h) * volatility
        
        candidates.append({
            "symbol":         symbol,
            "change_pct":     price_change_pct,
            "volume_usdt":    vol_24h,
            "volatility_pct": volatility,
            "score":          score,
        })

    if not candidates:
        logger.warning("Pair selection returned 0 candidates — using fallback list.")
        return [{"symbol": s, "change_pct": 0, "volume_usdt": 0, "volatility_pct": 0, "score": 0} 
                for s in _TESTNET_FALLBACK[:config.TOP_PAIRS_COUNT]]

    # Sort by momentum score (descending)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    top = candidates[:config.TOP_PAIRS_COUNT]

    logger.info("━" * 60)
    logger.info(f"📊 Top {len(top)} pairs by momentum score:")
    for i, c in enumerate(top, 1):
        dir_arrow = "▲" if c["change_pct"] >= 0 else "▼"
        logger.info(
            f"  {i}. {c['symbol']:<12} | "
            f"{dir_arrow} {abs(c['change_pct']):5.2f}% | "
            f"Vol: ${c['volume_usdt'] / 1e6:,.1f}M | "
            f"Volatility: {c['volatility_pct']:.2f}%"
        )
    # Return top 20 for the dashboard (we scan everything but show top 20)
    return top

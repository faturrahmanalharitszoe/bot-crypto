"""
bot/sentiment.py — Crypto Sentiment via Fear & Greed Index

Menggunakan Alternative.me Fear & Greed Index API.
✅ 100% Free
✅ Tidak perlu API Key / daftar akun
✅ Data diupdate harian oleh Alternative.me

API: https://api.alternative.me/fng/
Score: 0 = Extreme Fear, 100 = Extreme Greed
Bot mapping: 0-25 = Sangat Negatif (-1.0), 75-100 = Sangat Positif (+1.0)

Note: Fear & Greed adalah indikator MARKET-WIDE, bukan per-koin.
Bot akan menggunakan ini sebagai filter global (kalau market sangat ketakutan,
bot tidak membuka posisi baru).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests
import config

logger = logging.getLogger("bot")

# Fear & Greed Index API (no key needed)
FNG_API_URL = "https://api.alternative.me/fng/?limit=1&format=json"

# Classification thresholds
# Score 0-100 → our scale -1.0 to +1.0
# 0-24: Extreme Fear     → -1.0
# 25-44: Fear            → -0.5
# 45-55: Neutral         → 0.0
# 56-74: Greed           → +0.5
# 75-100: Extreme Greed  → +1.0


def _fng_to_score(fng_value: int) -> float:
    """Convert Fear & Greed 0-100 to our -1.0 to +1.0 scale."""
    if fng_value <= 24:
        return -1.0      # Extreme Fear → Don't buy
    elif fng_value <= 44:
        return -0.5      # Fear → Cautious
    elif fng_value <= 55:
        return 0.0       # Neutral
    elif fng_value <= 74:
        return 0.5       # Greed → Good entry
    else:
        return 1.0       # Extreme Greed → FOMO risk, be careful


class SentimentScorer:
    """
    Fetches the Crypto Fear & Greed Index and exposes a market-wide
    sentiment score. Thread-safe, updates in background every 6 hours.
    """

    def __init__(self) -> None:
        self._fng_value: int = 50        # Default: Neutral
        self._fng_label: str = "Neutral"
        self._score: float = 0.0
        self._last_fetch: Optional[datetime] = None
        self._is_fetching: bool = False  # Prevent thread spam
        self._lock = threading.Lock()
        self._enabled = config.SENTIMENT_ENABLED

        if not self._enabled:
            logger.info("📰 Sentiment Analysis is disabled (SENTIMENT_ENABLED=false)")
        else:
            logger.info("📰 Sentiment: Fear & Greed Index enabled")
            self._maybe_refresh()

    def _fetch(self) -> None:
        """Fetch current Fear & Greed index from Alternative.me."""
        with self._lock:
            if self._is_fetching: return
            self._is_fetching = True

        try:
            resp = requests.get(FNG_API_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            entry = data["data"][0]
            
            fng_value = int(entry["value"])
            fng_label = entry["value_classification"]
            score = _fng_to_score(fng_value)

            with self._lock:
                self._fng_value = fng_value
                self._fng_label = fng_label
                self._score = score
                self._last_fetch = datetime.utcnow()

            emoji = "😨" if score < -0.3 else ("😐" if score < 0.3 else "🤑")
            logger.info(f"📰 Fear & Greed Index: {fng_value}/100 — {fng_label} {emoji}")

        except Exception as e:
            logger.debug(f"Sentiment fetch failed: {e}")
        finally:
            with self._lock:
                self._is_fetching = False

    def _maybe_refresh(self) -> None:
        """Trigger a background refresh if cache is stale."""
        with self._lock:
            if self._is_fetching: return
            
        refresh_interval = timedelta(hours=6)
        if self._last_fetch is None or datetime.utcnow() - self._last_fetch > refresh_interval:
            threading.Thread(target=self._fetch, daemon=True).start()

    def get_score(self, symbol: str = "") -> float:
        """
        Returns market-wide sentiment score.
        -1.0 (Extreme Fear) → +1.0 (Extreme Greed)
        symbol param is ignored (F&G is market-wide).
        """
        if not self._enabled:
            return 0.0
        self._maybe_refresh()
        with self._lock:
            return self._score

    def is_blocked(self, symbol: str = "") -> bool:
        """
        Returns True if market is in Extreme Fear → block new BUYs.
        Uses config.SENTIMENT_BLOCK_THRESHOLD (default: -0.3 = Fear zone).
        """
        if not self._enabled:
            return False

        score = self.get_score(symbol)
        blocked = score < config.SENTIMENT_BLOCK_THRESHOLD

        if blocked:
            with self._lock:
                logger.info(
                    f"📰 Sentiment BLOCK | Market: {self._fng_label} "
                    f"({self._fng_value}/100, score={score:.1f}) — "
                    f"No new positions in Extreme Fear"
                )
        return blocked

    def all_scores(self) -> Dict:
        """Return current F&G data for dashboard display."""
        with self._lock:
            return {
                "fng_value": self._fng_value,
                "fng_label": self._fng_label,
                "score": self._score,
                "last_fetch": self._last_fetch.strftime("%Y-%m-%d %H:%M:%S") if self._last_fetch else "N/A"
            }

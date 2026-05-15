"""
bot/notifier.py — Telegram push notifications for key bot events.
Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env to enable.
If either is missing, all notifications are silently skipped.

How to get your token & chat_id:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Send any message to your new bot
  3. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Copy the "id" value from "chat" object
"""

import logging
import requests

logger = logging.getLogger("bot")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token   = (token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.enabled = bool(self.token and self.chat_id)
        self._url    = f"https://api.telegram.org/bot{self.token}/sendMessage"

        status = "ENABLED" if self.enabled else "DISABLED (add TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to .env)"
        logger.info(f"Telegram: {status}")

    def _send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(self._url, json={
                "chat_id": self.chat_id, "text": text, "parse_mode": "HTML"
            }, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")

    def notify_start(self, mode: str, balance: float) -> None:
        self._send(
            f"🚀 <b>Bot Started</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Balance: <code>${balance:.2f} USDT</code>"
        )

    def notify_stop(self) -> None:
        self._send("🛑 <b>Bot Stopped</b>")

    def notify_buy(self, symbol: str, price: float, qty: float,
                   take_profit: float, stop_loss: float) -> None:
        self._send(
            f"🟢 <b>BUY ORDER PLACED</b>\n"
            f"Pair: <code>{symbol}</code>\n"
            f"Entry: <code>{price:.6f}</code>  Qty: <code>{qty}</code>\n"
            f"🎯 TP: <code>{take_profit:.6f}</code>  (+1.5%)\n"
            f"🛑 SL: <code>{stop_loss:.6f}</code>  (-0.8%)"
        )

    def notify_sell(self, symbol: str, reason: str,
                    entry: float, exit_price: float,
                    pnl_usdt: float, pnl_pct: float,
                    duration_min: float) -> None:
        emoji = "✅" if pnl_usdt >= 0 else "❌"
        sign  = "+" if pnl_usdt >= 0 else ""
        labels = {
            "TAKE_PROFIT":  "🎯 Take Profit",
            "STOP_LOSS":    "🛑 Stop Loss",
            "EMA_REVERSAL": "🔁 EMA Reversal",
            "MAX_HOLD":     "⏰ Max Hold Time",
        }
        self._send(
            f"{emoji} <b>TRADE CLOSED</b> — {labels.get(reason, reason)}\n"
            f"Pair: <code>{symbol}</code>\n"
            f"<code>{entry:.6f}</code> → <code>{exit_price:.6f}</code>\n"
            f"PnL: <code>{sign}{pnl_usdt:.4f} USDT ({sign}{pnl_pct:.2f}%)</code>\n"
            f"Duration: {duration_min:.1f} min"
        )

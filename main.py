"""
main.py — Binance Scalping Bot Entry Point

Usage:
    python main.py

Make sure you have:
  1. Copied .env.example to .env and filled in your keys
  2. Installed dependencies: pip install -r requirements.txt
  3. (Testnet) Got keys from https://testnet.binance.vision
"""

import sys

# ── Fix Windows console encoding for emoji/unicode ───────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import time
import signal
import logging
from datetime import datetime

from colorama import Fore, Style, init

import config
from bot.exchange       import Exchange
from bot.trader         import Trader
from bot.logger         import logger
from bot.notifier       import TelegramNotifier
from bot.web_dashboard  import start_web_server, update_state

init(autoreset=True)

# ── Graceful shutdown flag ────────────────────────────────────
_running = True

def _handle_exit(signum, frame):
    global _running
    logger.info("🛑 Shutdown signal received. Finishing current loop...")
    _running = False

signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


# ────────────────────────────────────────────────────────────
#  Status Banner
# ────────────────────────────────────────────────────────────
def print_banner(status: dict) -> None:
    """Print a colored status banner to the console each tick."""
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = f"{Fore.YELLOW}🧪 TESTNET{Style.RESET_ALL}" if config.TESTNET else f"{Fore.RED}🔴 LIVE{Style.RESET_ALL}"

    print()
    print(f"{Fore.CYAN}{'═' * 62}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  ⚡ BINANCE SCALPING BOT  {mode}  │  Loop #{status['loop']}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═' * 62}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}⏰ Time       :{Style.RESET_ALL} {now}")
    print(f"  {Fore.WHITE}💰 USDT Bal   :{Style.RESET_ALL} {Fore.GREEN}${status['usdt_balance']:.2f}{Style.RESET_ALL}")

    if status.get("has_position"):
        pnl     = status["pnl_usdt"]
        pnl_pct = status["pnl_pct"]
        pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
        pnl_sym   = "+" if pnl >= 0 else ""

        print(f"  {Fore.WHITE}📈 Position   :{Style.RESET_ALL} {Fore.YELLOW}{status['symbol']}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}   Entry      :{Style.RESET_ALL} {status['entry_price']:.6f}")
        print(f"  {Fore.WHITE}   Current    :{Style.RESET_ALL} {status['current_price']:.6f}")
        print(
            f"  {Fore.WHITE}   PnL        :{Style.RESET_ALL} "
            f"{pnl_color}{pnl_sym}{pnl:.4f} USDT  ({pnl_sym}{pnl_pct:.2f}%){Style.RESET_ALL}"
        )
        print(f"  {Fore.WHITE}   TP / SL    :{Style.RESET_ALL} "
              f"{Fore.GREEN}{status['take_profit']:.6f}{Style.RESET_ALL} / "
              f"{Fore.RED}{status['stop_loss']:.6f}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}   Held       :{Style.RESET_ALL} {status['duration_min']:.1f} min")
    else:
        # Extract symbols for display in terminal
        wl_list = [p["symbol"] for p in status.get("watchlist", [])]
        pairs = ", ".join(wl_list)
        print(f"  {Fore.WHITE}🔍 Watching   :{Style.RESET_ALL} {Fore.CYAN}{pairs}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}📂 Position   :{Style.RESET_ALL} No open position")

    print(f"{Fore.CYAN}{'═' * 62}{Style.RESET_ALL}")


# ────────────────────────────────────────────────────────────
#  Startup Checks
# ────────────────────────────────────────────────────────────
def pre_flight_check(exchange: Exchange) -> bool:
    """Validate connectivity and API keys before starting the loop."""
    logger.info("🚀 Running pre-flight checks...")

    # Check connectivity
    try:
        balance = exchange.get_usdt_balance()
        logger.info(f"✅ API connected. USDT Balance: ${balance:.2f}")

        if balance < 1.0 and not config.TESTNET:
            logger.error(
                "❌ USDT balance too low for live trading ($1 minimum). "
                "Check your account."
            )
            return False

        if balance < 1.0 and config.TESTNET:
            logger.warning(
                "⚠️  Testnet balance is very low. "
                "Go to https://testnet.binance.vision and request test funds."
            )

    except Exception as e:
        logger.error(f"❌ Pre-flight failed: {e}")
        return False

    return True


# ────────────────────────────────────────────────────────────
#  Main Loop
# ────────────────────────────────────────────────────────────
def main() -> None:
    global _running

    # ── Startup banner ────────────────────────────────────
    mode_str = "TESTNET (Paper Trading)" if config.TESTNET else "⚠️  LIVE TRADING ⚠️"
    print(f"\n{Fore.CYAN}{'█' * 62}")
    print(f"  BINANCE SCALPING BOT  —  {mode_str}")
    print(f"  Strategy: EMA9/21 Crossover + RSI14 + Volume Spike")
    print(f"  TP: +{config.TAKE_PROFIT_PCT*100:.1f}%  |  SL: -{config.STOP_LOSS_PCT*100:.1f}%  |  "
          f"Max Pos: {config.MAX_POSITION_PCT*100:.0f}%  |  Interval: {config.CANDLE_INTERVAL}")
    print(f"{'█' * 62}{Style.RESET_ALL}\n")

    # ── Initialize components ─────────────────────────────
    try:
        exchange = Exchange()
    except ValueError as e:
        print(f"\n{Fore.RED}{e}{Style.RESET_ALL}\n")
        sys.exit(1)

    if not pre_flight_check(exchange):
        sys.exit(1)

    # ── Telegram notifier ─────────────────────────────────
    notifier = TelegramNotifier(
        token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
    )

    # ── Web dashboard ─────────────────────────────────────
    if config.WEB_DASHBOARD_ENABLED:
        start_web_server(port=config.WEB_PORT)
        logger.info(f"   Open browser → http://localhost:{config.WEB_PORT}")

    # ── Trader ────────────────────────────────────────────
    trader = Trader(exchange, notifier=notifier, on_state_update=update_state)
    logger.info(f"Bot initialized. Loop interval: {config.LOOP_INTERVAL_SECONDS}s")
    logger.info("   Press Ctrl+C to stop gracefully.\n")

    # Send startup notification
    notifier.notify_start(
        mode="TESTNET" if config.TESTNET else "LIVE",
        balance=exchange.get_usdt_balance(),
    )

    # ── Main trading loop ─────────────────────────────────
    while _running:
        try:
            trader.tick()
            status = trader.get_status_dict()
            print_banner(status)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"⚠️  Unexpected error in main loop: {e}", exc_info=True)
            logger.info(f"   Continuing in {config.LOOP_INTERVAL_SECONDS}s...")

        if _running:
            # Dynamic sleep: monitor open positions fast, check watchlist normally
            sleep_time = getattr(config, "LOOP_INTERVAL_FAST", 5) if trader.risk_manager.has_positions else config.LOOP_INTERVAL_SECONDS
            time.sleep(sleep_time)

    # ── Shutdown ──────────────────────────────────────────
    notifier.notify_stop()
    logger.info("=" * 60)
    logger.info("Bot stopped. Check logs/trades.csv for trade history.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

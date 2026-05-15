"""
bot/logger.py — Colored console logging + CSV trade journal
"""

import sys
import logging
import os
import csv
from datetime import datetime
from colorama import Fore, Style, init

# Force UTF-8 output on Windows PowerShell so emoji render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7 fallback

# Initialize colorama (required on Windows)
init(autoreset=True)

# ── Ensure logs/ directory exists ────────────────────────────
os.makedirs("logs", exist_ok=True)
TRADE_LOG_PATH = os.path.join("logs", "trades.csv")
BOT_LOG_PATH   = os.path.join("logs", "bot.log")


# ────────────────────────────────────────────────────────────
#  Colored Formatter for console output
# ────────────────────────────────────────────────────────────
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG:    Fore.CYAN,
        logging.INFO:     Fore.WHITE,
        logging.WARNING:  Fore.YELLOW,
        logging.ERROR:    Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, "")
        ts    = datetime.now().strftime("%H:%M:%S")
        level = f"{record.levelname:<8}"
        msg   = record.getMessage()

        # Inject emoji colors for specific keywords
        if "BUY" in msg:
            color = Fore.GREEN
        elif "SELL" in msg or "EXIT" in msg:
            color = Fore.RED + Style.BRIGHT
        elif "PROFIT" in msg or "✅" in msg:
            color = Fore.GREEN + Style.BRIGHT
        elif "LOSS" in msg or "❌" in msg:
            color = Fore.RED

        return f"{Fore.CYAN}[{ts}]{Style.RESET_ALL} {color}{level}{Style.RESET_ALL} {msg}"


def setup_logger(name: str = "bot") -> logging.Logger:
    """Configure and return the root bot logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    # ── Console handler (colored) ─────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)

    # ── File handler (plain text) ─────────────────────────
    fh = logging.FileHandler(BOT_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Singleton logger ─────────────────────────────────────────
logger = setup_logger("bot")


# ────────────────────────────────────────────────────────────
#  CSV Trade Journal
# ────────────────────────────────────────────────────────────
_TRADE_HEADERS = [
    "timestamp", "symbol", "side", "cost_usdt", "entry_price", "exit_price",
    "quantity", "pnl_usdt", "pnl_pct", "exit_reason", "duration_min"
]


def _ensure_trade_log() -> None:
    """
    Create the CSV file with headers if it doesn't exist.
    If it exists but has an old format, migrate the data to the new format.
    """
    if os.path.exists(TRADE_LOG_PATH):
        try:
            abs_path = os.path.abspath(TRADE_LOG_PATH)
            needs_repair = False
            rows = []
            with open(abs_path, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                if not reader: return
                header = reader[0]
                
                # Check 1: Header length mismatch
                if len(header) != len(_TRADE_HEADERS):
                    needs_repair = True
                
                # Check 2: Content check (is 'exit_reason' a number?)
                for r in reader[1:]:
                    if not r: continue
                    # If the 10th column (exit_reason) looks like a float, it's shifted!
                    try:
                        float(r[9]) 
                        needs_repair = True
                        break
                    except (ValueError, IndexError):
                        continue

            if needs_repair:
                backup_path = TRADE_LOG_PATH.replace(".csv", f"_repair_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                logger.warning(f"⚠️  Shifted data detected in trades.csv. Repairing and backing up to {backup_path}")
                
                # Perform the repair
                repaired_rows = []
                with open(abs_path, "r", encoding="utf-8") as f:
                    raw = list(csv.reader(f))
                    for r in raw[1:]: # skip header
                        if not r: continue
                        
                        # CASE A: Row is shifted because of a '0.0' inserted at index 3
                        # Pattern: TS, Sym, Side, 0.0, COST, ENTRY, EXIT, QTY, PNL_U, PNL_P, REASON
                        if len(r) == 11 and r[3] == "0.0":
                            repaired = {
                                "timestamp": r[0], "symbol": r[1], "side": r[2],
                                "cost_usdt": r[4], "entry_price": r[5], "exit_price": r[6],
                                "quantity": r[7], "pnl_usdt": r[8], "pnl_pct": r[9],
                                "exit_reason": r[10], "duration_min": "0.0" # duration lost in shift
                            }
                            repaired_rows.append(repaired)
                        # CASE B: Standard old 10-column format
                        elif len(r) == 10:
                            repaired_rows.append({
                                "timestamp": r[0], "symbol": r[1], "side": r[2],
                                "cost_usdt": "0.0", "entry_price": r[3], "exit_price": r[4],
                                "quantity": r[5], "pnl_usdt": r[6], "pnl_pct": r[7],
                                "exit_reason": r[8], "duration_min": r[9]
                            })
                        else:
                            # Keep as is or pad
                            repaired_rows.append({h: (r[i] if i < len(r) else "0.0") for i, h in enumerate(_TRADE_HEADERS)})

                os.rename(abs_path, backup_path)
                with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as nf:
                    writer = csv.DictWriter(nf, fieldnames=_TRADE_HEADERS)
                    writer.writeheader()
                    writer.writerows(repaired_rows)
                logger.info(f"✅ Deep repair complete. Fixed {len(repaired_rows)} rows.")
                
        except Exception as e:
            logger.error(f"Error during deep repair: {e}")

    if not os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_HEADERS)
            writer.writeheader()
            logger.info("📄 Created new trade journal with updated headers.")


def log_trade(
    symbol: str,
    side: str,
    cost_usdt: float,
    entry_price: float,
    exit_price: float,
    quantity: float,
    exit_reason: str,
    duration_min: float,
) -> None:
    """Append a completed trade to the CSV trade journal."""
    _ensure_trade_log()

    pnl_usdt = (exit_price - entry_price) * quantity
    pnl_pct  = ((exit_price - entry_price) / entry_price) * 100

    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":       symbol,
        "side":         side,
        "cost_usdt":    round(cost_usdt, 4),
        "entry_price":  round(entry_price, 8),
        "exit_price":   round(exit_price, 8),
        "quantity":     round(quantity, 6),
        "pnl_usdt":     round(pnl_usdt, 4),
        "pnl_pct":      round(pnl_pct, 4),
        "exit_reason":  exit_reason,
        "duration_min": round(duration_min, 1),
    }

    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_HEADERS)
        writer.writerow(row)

    emoji = "✅" if pnl_usdt >= 0 else "❌"
    logger.info(
        f"{emoji} TRADE CLOSED | {symbol} | {exit_reason} | "
        f"PnL: {pnl_usdt:+.4f} USDT ({pnl_pct:+.2f}%) | "
        f"Duration: {duration_min:.1f}m"
    )
# ── Initialize Trade Journal on startup ──────────────────────
_ensure_trade_log()

import sys, csv, os, shutil
from datetime import datetime
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
"""
repair_trades.py — Fix historical trade CSV with wrong PnL for SHORT positions

Old format: 'side' column = EXIT order side
  - "BUY"  = was SHORT position (closing with BUY) -> PnL = (exit-entry)*qty  WRONG
  - "SELL" = was LONG  position (closing with SELL) -> PnL = (exit-entry)*qty  CORRECT

New format: 'side' column = ENTRY side
  - "BUY"  = LONG  -> PnL = (exit-entry)*qty
  - "SELL" = SHORT -> PnL = (entry-exit)*qty
"""

TRADE_LOG = "logs/trades.csv"
BACKUP    = f"logs/trades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

HEADERS = [
    "timestamp", "symbol", "side", "cost_usdt", "entry_price",
    "exit_price", "quantity", "pnl_usdt", "pnl_pct", "exit_reason", "duration_min"
]


def detect_old_format(row: dict) -> bool:
    """
    A row is in OLD format if the stored pnl_usdt matches (exit-entry)*qty.
    i.e. the formula was always applied as LONG regardless of side.
    """
    try:
        entry = float(row["entry_price"])
        exit_ = float(row["exit_price"])
        qty   = float(row["quantity"])
        stored_pnl = float(row["pnl_usdt"])
        long_formula = (exit_ - entry) * qty
        # Allow 1% tolerance for rounding
        if entry == 0 or qty == 0:
            return False
        return abs(long_formula - stored_pnl) / (abs(entry) * qty + 1e-9) < 0.02
    except (ValueError, KeyError, ZeroDivisionError):
        return False


def repair_row(row: dict) -> dict:
    """
    Convert from old format (exit-side) to new format (entry-side) with correct PnL.
    """
    entry = float(row["entry_price"])
    exit_ = float(row["exit_price"])
    qty   = float(row["quantity"])
    old_side = row["side"]

    if old_side == "BUY":
        # Was SHORT position: entry side = SELL, correct PnL = (entry - exit) * qty
        new_side = "SELL"
        pnl_usdt = (entry - exit_) * qty
        pnl_pct  = ((entry - exit_) / entry) * 100 if entry else 0
    else:
        # Was LONG position: entry side = BUY, PnL formula is the same (exit - entry)
        new_side = "BUY"
        pnl_usdt = (exit_ - entry) * qty
        pnl_pct  = ((exit_ - entry) / entry) * 100 if entry else 0

    row["side"]     = new_side
    row["pnl_usdt"] = round(pnl_usdt, 4)
    row["pnl_pct"]  = round(pnl_pct, 4)
    return row


def main():
    if not os.path.exists(TRADE_LOG):
        print(f"❌ {TRADE_LOG} not found!")
        return

    # Backup first
    shutil.copy2(TRADE_LOG, BACKUP)
    print(f"💾 Backup saved: {BACKUP}")

    with open(TRADE_LOG, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    repaired = 0
    skipped  = 0

    fixed_rows = []
    for row in rows:
        # Pad missing columns
        for h in HEADERS:
            if h not in row:
                row[h] = "0"

        if detect_old_format(row):
            old_pnl  = row["pnl_usdt"]
            old_side = row["side"]
            row = repair_row(row)
            print(
                f"  ✅ {row['timestamp']} | {row['symbol']} | "
                f"{old_side}→{row['side']} | "
                f"PnL: {old_pnl} → {row['pnl_usdt']} USDT | {row['exit_reason']}"
            )
            repaired += 1
        else:
            skipped += 1

        fixed_rows.append(row)

    # Write repaired file
    with open(TRADE_LOG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"\n📊 Done: {repaired} repaired, {skipped} unchanged (already correct / new format)")
    print(f"📄 File: {TRADE_LOG}")
    print(f"💾 Backup: {BACKUP}")


if __name__ == "__main__":
    main()

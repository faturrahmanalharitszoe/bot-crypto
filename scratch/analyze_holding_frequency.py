import re
from datetime import datetime

def analyze():
    log_path = r"c:\Personal Project\bot-crypto\logs\bot.log"
    # Matches logs like: 2026-05-22 04:47:19,252 | INFO     | 📍 Holding FIDAUSDT ...
    pattern = re.compile(r"^(2026-05-22 \d{2}:\d{2}:\d{2}),\d+ \| INFO\s+\| 📍 Holding (\w+)")
    
    holdings = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                dt_str, symbol = m.groups()
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                holdings.append((dt, symbol))
                
    if not holdings:
        print("No holding logs found for 2026-05-22")
        return
        
    print(f"Total holding logs on 2026-05-22: {len(holdings)}")
    print("Sample intervals on 2026-05-22:")
    # Group by symbol to see intervals
    symbol_logs = {}
    for dt, symbol in holdings:
        if symbol not in symbol_logs:
            symbol_logs[symbol] = []
        symbol_logs[symbol].append(dt)
        
    for symbol, dts in list(symbol_logs.items())[:3]:
        print(f"\nSymbol: {symbol} (Total logs: {len(dts)})")
        for i in range(1, min(10, len(dts))):
            diff = (dts[i] - dts[i-1]).total_seconds()
            print(f"  {dts[i-1]} -> {dts[i]}: {diff} seconds")

if __name__ == "__main__":
    analyze()

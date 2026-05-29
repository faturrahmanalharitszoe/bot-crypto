import logging
import sys
from bot.exchange import Exchange
from bot.pair_selector import select_best_pairs
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def test_watchlist():
    print("Initializing Exchange...")
    ex = Exchange()
    
    print("\nFetching active symbols from Exchange...")
    active_syms = ex.get_active_symbols()
    if active_syms is not None:
        print(f"Success! Found {len(active_syms)} active trading symbols on Binance.")
        print(f"Is UTKUSDT in active symbols? {'UTKUSDT' in active_syms}")
    else:
        print("Warning: Active symbols returned None (API issue or empty)")

    print("\nRunning select_best_pairs...")
    candidates = select_best_pairs(ex)
    print(f"Number of candidates returned: {len(candidates)}")
    
    print("\nFiltering and checking watchlist refresh simulation...")
    valid_data = []
    for item in candidates:
        symbol = item["symbol"]
        info = ex.get_symbol_info(symbol)
        if info is not None:
            valid_data.append(item)
        else:
            print(f"Skipped invalid/inactive symbol: {symbol}")
            
        if len(valid_data) >= config.TOP_PAIRS_COUNT:
            break

    print(f"\nFinal watchlist size: {len(valid_data)}")
    print("Watchlist symbols:")
    for idx, item in enumerate(valid_data, 1):
        print(f"  {idx}. {item['symbol']} (Score: {item['score']:.2f})")

    # Verify that UTKUSDT is not in the final watchlist
    final_symbols = [item["symbol"] for item in valid_data]
    if "UTKUSDT" in final_symbols:
        print("\n[FAILED]: UTKUSDT is in the watchlist!")
        sys.exit(1)
    else:
        print("\n[SUCCESS]: UTKUSDT is not in the watchlist.")
        
if __name__ == "__main__":
    test_watchlist()

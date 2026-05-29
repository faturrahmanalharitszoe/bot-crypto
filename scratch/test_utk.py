from bot.exchange import Exchange

def test():
    ex = Exchange()
    print("Exchange initialized")
    
    # Check Spot
    try:
        spot_info = ex.client.get_exchange_info()
        utk_spot = next((s for s in spot_info.get('symbols', []) if s['symbol'] == 'UTKUSDT'), None)
        if utk_spot:
            print("Spot UTKUSDT status:", utk_spot.get('status'))
            print("Spot UTKUSDT filters:", utk_spot.get('filters'))
        else:
            print("Spot UTKUSDT not found")
    except Exception as e:
        print("Spot error:", e)

    # Check Futures
    try:
        futures_info = ex.client.futures_exchange_info()
        utk_fut = next((s for s in futures_info.get('symbols', []) if s['symbol'] == 'UTKUSDT'), None)
        if utk_fut:
            print("Futures UTKUSDT status:", utk_fut.get('status'))
        else:
            print("Futures UTKUSDT not found")
    except Exception as e:
        print("Futures error:", e)

if __name__ == "__main__":
    test()

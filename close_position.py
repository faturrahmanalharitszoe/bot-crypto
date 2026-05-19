"""
close_position.py — Emergency manual close script
Run: python close_position.py ALICEUSDT
"""
import sys
from bot.exchange import Exchange

symbol = sys.argv[1] if len(sys.argv) > 1 else "ALICEUSDT"

ex = Exchange()

# Check current futures position
try:
    positions = ex.client.futures_position_information(symbol=symbol)
    for p in positions:
        amt = float(p["positionAmt"])
        if amt == 0:
            print(f"✅ {symbol}: No open position (already closed or was never opened)")
            continue
        side = "SELL" if amt > 0 else "BUY"   # Close long=sell, close short=buy
        qty  = abs(amt)
        print(f"⚠️  {symbol}: Found open position | Qty: {amt} | Closing with {side} {qty}...")
        order = ex.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty,
            reduceOnly=True   # Safety: only reduce, never open new
        )
        print(f"✅ Closed! Order: {order.get('orderId')} | Status: {order.get('status')}")
except Exception as e:
    print(f"❌ Error: {e}")

from binance.client import Client
client = Client()
info = client.get_exchange_info()
utk = [s for s in info['symbols'] if s['symbol'] == 'UTKUSDT']
if utk:
    print("UTKUSDT Info:")
    import pprint
    pprint.pprint(utk[0])
else:
    print("UTKUSDT not found in exchange info")

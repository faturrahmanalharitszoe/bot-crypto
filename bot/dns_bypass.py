"""
bot/dns_bypass.py — Programmatic bypass for DNS-based ISP blocking of Binance API.
Overrides socket.getaddrinfo to resolve Binance endpoints using Cloudflare/Google DNS-over-HTTPS.
"""
import socket
import logging

logger = logging.getLogger("bot")

DOMAINS = [
    "api.binance.com",
    "api-gcp.binance.com",
    "api1.binance.com",
    "api2.binance.com",
    "api3.binance.com",
    "api4.binance.com",
    "fapi.binance.com",
    "dapi.binance.com",
    "api.binancefuture.com",
    "testnet.binancefuture.com"
]

_patched = False

def init_bypass():
    global _patched
    if _patched:
        return
    
    # We delay requests import to avoid circular dependencies
    import requests
    
    ip_map = {}
    
    # Try resolving via Cloudflare DoH, fallback to Google DoH
    for domain in DOMAINS:
        ips = []
        # 1. Try Cloudflare
        try:
            r = requests.get(
                "https://cloudflare-dns.com/dns-query",
                headers={"accept": "application/dns-json"},
                params={"name": domain, "type": "A"},
                timeout=3
            )
            if r.status_code == 200:
                data = r.json()
                if "Answer" in data:
                    for ans in data["Answer"]:
                        if ans.get("type") == 1: # A record
                            ips.append(ans.get("data"))
        except Exception:
            pass
            
        # 2. Try Google
        if not ips:
            try:
                r = requests.get(
                    "https://dns.google/resolve",
                    params={"name": domain, "type": "A"},
                    timeout=3
                )
                if r.status_code == 200:
                    data = r.json()
                    if "Answer" in data:
                        for ans in data["Answer"]:
                            if ans.get("type") == 1: # A record
                                ips.append(ans.get("data"))
            except Exception:
                pass
                
        if ips:
            # We strip trailing dots if any (e.g. from Google DNS response)
            ips = [ip.rstrip('.') for ip in ips if ip]
            ip_map[domain] = ips
            # Log bypass using print if logger isn't initialized or has no handlers
            msg = f"[DNS Bypass] Resolved {domain} -> {ips[0]}"
            if logger.handlers:
                logger.info(msg)
            else:
                print(msg)
            
    if not ip_map:
        return
        
    original_getaddrinfo = socket.getaddrinfo
    
    def custom_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if host in ip_map:
            ip = ip_map[host][0]
            try:
                results = original_getaddrinfo(ip, port, family, type, proto, flags)
                new_results = []
                for res in results:
                    # sockaddr format is (ip, port, ...)
                    sockaddr = res[4]
                    new_sockaddr = (ip, sockaddr[1]) + sockaddr[2:]
                    new_results.append((res[0], res[1], res[2], res[3], new_sockaddr))
                return new_results
            except Exception:
                pass
        return original_getaddrinfo(host, port, family, type, proto, flags)
        
    socket.getaddrinfo = custom_getaddrinfo
    _patched = True

try:
    init_bypass()
except Exception as e:
    msg = f"[DNS Bypass] Warning: Failed to initialize DNS bypass: {e}"
    if logger.handlers:
        logger.warning(msg)
    else:
        print(msg)

import ccxt
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize exchange
exchange = ccxt.krakenfutures({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
})

exchange.set_sandbox_mode(config.get('use_testnet', True))

print("✅ Connected to Kraken Futures" + (" Testnet/Demo" if config.get('use_testnet') else " LIVE"))

# Load markets once (required before checking symbols)
print("\nLoading markets...")
markets = exchange.load_markets()
print(f"✅ Loaded {len(markets)} markets successfully.\n")

# Check the symbols you specified in config
print("=== Checking your configured symbols ===")
for symbol in config.get("symbols", []):
    try:
        market = exchange.market(symbol)
        print(f"✅ {symbol} is valid")
        print(f"   Type: {market.get('type')} | Swap/Future: {market.get('swap') or market.get('future')}")
        print(f"   Precision (price): {market.get('precision', {}).get('price')}")
        print(f"   Min amount: {market.get('limits', {}).get('amount', {}).get('min')}")
        print("   ---")
    except Exception as e:
        print(f"❌ {symbol} → Error: {e}")
        print("   Tip: Try 'PI_XBTUSD' or 'PF_ETHUSD' instead.\n")

print("\nYour current demo balance (USD):")
try:
    balance = exchange.fetch_balance()
    usd_total = balance.get('total', {}).get('USD', 0)
    print(f"USD: ${usd_total:,.2f}")
except Exception as e:
    print("Error fetching balance:", e)
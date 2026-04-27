import ccxt
import json
import os
import asyncio
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

# ========================= CONFIG =========================
with open('config.json', 'r') as f:
    config = json.load(f)

RISK_USD = config.get("risk_usd", 1.0)
LEVERAGE = config.get("leverage", 5)
SYMBOLS = config.get("symbols", ["PI_XBTUSD"])
TIMEFRAME = config.get("timeframe", "1m")
HTF_TIMEFRAME = "5m"
WARMUP_MINUTES = config.get("warmup_minutes", 10)
MAX_TRADES_PER_DAY = config.get("max_trades_per_day", 8)
MAX_DAILY_LOSS_USD = config.get("max_daily_loss_usd", 5.0)
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RR_RATIO = config.get("rr_ratio", 4.0)
USE_TESTNET = config.get("use_testnet", True)

exchange = ccxt.krakenfutures({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(USE_TESTNET)

log_file = f"trading_log_{date.today()}.txt"
stats_file = "bot_stats.json"

def log(message, to_file=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if to_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# Load persistent stats
def load_stats():
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"wins": 0, "losses": 0, "total_pnl": 0.0}

stats = load_stats()
wins = stats.get("wins", 0)
losses = stats.get("losses", 0)
daily_pnl = 0.0
daily_trades = 0
current_day = date.today()

log(f"Bot started | Risk: ${RISK_USD} | Max Trades/Day: {MAX_TRADES_PER_DAY} | Max Daily Loss: ${MAX_DAILY_LOSS_USD} | Warm-up: {WARMUP_MINUTES} min", to_file=True)
log(f"Symbols: {SYMBOLS}\n", to_file=True)

candle_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}
active_trades = {}

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1] if len(atr) > period else 0.001

def detect_swing_highs_lows(df, strength=5):
    highs = df['high'].rolling(window=strength*2+1, center=True).max()
    lows = df['low'].rolling(window=strength*2+1, center=True).min()
    df['swing_high'] = np.where(df['high'] == highs, df['high'], np.nan)
    df['swing_low'] = np.where(df['low'] == lows, df['low'], np.nan)
    return df

def detect_choch(df):
    df = detect_swing_highs_lows(df)
    signals = []
    for i in range(30, len(df)):
        recent_highs = df['swing_high'].iloc[i-40:i].dropna()
        recent_lows = df['swing_low'].iloc[i-40:i].dropna()
        if len(recent_lows) < 2 or len(recent_highs) < 2:
            continue
        if df['close'].iloc[i] > recent_highs.iloc[-1] and df['close'].iloc[i-1] <= recent_highs.iloc[-1]:
            signals.append(('bullish', i, recent_highs.iloc[-1]))
        if df['close'].iloc[i] < recent_lows.iloc[-1] and df['close'].iloc[i-1] >= recent_lows.iloc[-1]:
            signals.append(('bearish', i, recent_lows.iloc[-1]))
    return signals

def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            fvgs.append(('bullish', (df['low'].iloc[i] + df['high'].iloc[i-2])/2, df['low'].iloc[i-1]))
        if df['low'].iloc[i-2] > df['high'].iloc[i]:
            fvgs.append(('bearish', (df['high'].iloc[i] + df['low'].iloc[i-2])/2, df['high'].iloc[i-1]))
    return fvgs

async def fetch_latest_candles(symbol, tf, limit=300):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log(f"❌ Candle error ({tf}) for {symbol}: {e}", to_file=False)
        return pd.DataFrame()

async def place_trade(symbol, side, current_price, stop_price, tp_price, risk_usd):
    try:
        m = exchange.market(symbol)
        min_amount = m.get('limits', {}).get('amount', {}).get('min') or 1.0
        quantity = max(min_amount, round((risk_usd * LEVERAGE) / abs(current_price - stop_price), 4))
        quantity = min(quantity, 2000)

        limit_price = current_price * (1.003 if side == 'buy' else 0.997)

        params = {
            'leverage': LEVERAGE,
            'stopLoss': {'type': 'stop', 'price': stop_price, 'trigger': 'mark'},
            'takeProfit': {'type': 'takeProfit', 'price': tp_price, 'trigger': 'mark'}
        }

        log(f"📊 [{symbol}] Placing {side.upper()} LIMIT | Qty: {quantity:.2f} | Limit: {limit_price:.4f}")

        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=quantity,
            price=limit_price,
            params=params
        )

        active_trades[symbol] = {
            'side': side,
            'entry_price': float(current_price),
            'stop_price': float(stop_price),
            'tp_price': float(tp_price),
            'quantity': float(quantity)
        }

        log(f"✅ [{symbol}] {side.upper()} POSITION + REAL SL/TP PLACED | Qty: {quantity:.2f} | SL: {stop_price:.4f} | TP: {tp_price:.4f}")
        return order
    except Exception as e:
        log(f"❌ [{symbol}] Order failed: {e}")
        return None

async def manage_open_trade(symbol, df):
    if symbol not in active_trades:
        return
    trade = active_trades[symbol]
    current_price = df['close'].iloc[-1]

    hit = False
    pnl = 0.0
    result = ""

    if trade['side'] == 'sell':
        if current_price >= trade['stop_price']:
            pnl = RISK_USD * -1
            result = "LOSS"
            hit = True
        elif current_price <= trade['tp_price']:
            pnl = RISK_USD * RR_RATIO
            result = "WIN"
            hit = True
    else:
        if current_price <= trade['stop_price']:
            pnl = RISK_USD * -1
            result = "LOSS"
            hit = True
        elif current_price >= trade['tp_price']:
            pnl = RISK_USD * RR_RATIO
            result = "WIN"
            hit = True

    if hit:
        global daily_pnl, wins, losses
        daily_pnl += pnl
        if result == "WIN":
            wins += 1
        else:
            losses += 1

        log(f"   ✅ [{symbol}] {trade['side'].upper()} {result} | PNL: ${pnl:.2f} | Total PNL today: ${daily_pnl:.2f} | Win Rate: {wins/(wins+losses)*100:.1f}% ({wins}W / {losses}L)")
        del active_trades[symbol]

async def main():
    global daily_trades, daily_pnl, wins, losses, current_day
    log("🚀 Final Bot with Clean Trading Log + Persistent Stats + Startup Position Check\n")

    # Startup position check
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                symbol = pos['symbol']
                side = 'buy' if float(pos['contracts']) > 0 else 'sell'
                log(f"🔄 Found existing {side.upper()} position on {symbol} - added to management", to_file=True)
                active_trades[symbol] = {
                    'side': side,
                    'entry_price': float(pos.get('entryPrice', 0)),
                    'stop_price': 0,
                    'tp_price': 0,
                    'quantity': abs(float(pos['contracts']))
                }
    except Exception as e:
        log(f"Warning: Could not load existing positions: {e}", to_file=True)

    warmup_end = datetime.now() + timedelta(minutes=WARMUP_MINUTES)
    in_warmup = WARMUP_MINUTES > 0

    while True:
        try:
            today = date.today()
            if today != current_day:
                win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                log(f"\n📅 === DAILY SUMMARY ({current_day}) === Trades: {daily_trades} | PNL: ${daily_pnl:.2f} | Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)\n", to_file=True)
                daily_trades = 0
                daily_pnl = 0.0
                wins = 0
                losses = 0
                current_day = today

            if daily_pnl <= -MAX_DAILY_LOSS_USD:
                log(f"🚫 Daily max loss of ${MAX_DAILY_LOSS_USD} reached. Bot paused for today.", to_file=True)
                await asyncio.sleep(3600)
                continue

            if in_warmup and datetime.now() < warmup_end:
                log(f"   ⏳ Warm-up active ({(warmup_end - datetime.now()).seconds//60} min left)", to_file=False)
                await asyncio.sleep(25)
                continue
            elif in_warmup:
                log("✅ Warm-up finished. Starting live trading.", to_file=True)
                in_warmup = False

            best_setup = None
            best_score = -999

            for symbol in SYMBOLS:
                df1m = await fetch_latest_candles(symbol, TIMEFRAME)
                df5m = await fetch_latest_candles(symbol, HTF_TIMEFRAME, limit=200)

                if len(df1m) < 100 or len(df5m) < 50:
                    continue

                candle_data[symbol] = df1m
                choch_5m = detect_choch(df5m)
                fvg_1m = detect_fvg(df1m)
                current_price = df1m['close'].iloc[-1]
                atr = calculate_atr(df1m)

                # Verbose console only
                log(f"[{datetime.now().strftime('%H:%M:%S')}] [{symbol}] Price: {current_price:.4f} | ATR: {atr:.4f} | 5m CHOCH: {len(choch_5m)} | 1m FVGs: {len(fvg_1m)}", to_file=False)

                await manage_open_trade(symbol, df1m)

                if symbol in active_trades:
                    continue

                for signal in choch_5m[-3:]:
                    direction, _, _ = signal
                    for fvg in fvg_1m[-6:]:
                        fvg_type, midpoint, fvg_extreme = fvg
                        if direction != fvg_type:
                            continue
                        if abs(current_price - midpoint) / midpoint > 0.005:
                            continue

                        score = 100 - (abs(current_price - midpoint) / midpoint * 1000)
                        if score > best_score:
                            best_score = score
                            best_setup = (symbol, direction, current_price, fvg_extreme, ATR_MULTIPLIER * atr)

            if best_setup and daily_trades < MAX_TRADES_PER_DAY:
                symbol, direction, current_price, fvg_extreme, dynamic_buffer = best_setup
                if direction == 'bullish':
                    stop_price = fvg_extreme - dynamic_buffer
                    tp_price = current_price + (current_price - stop_price) * RR_RATIO
                    await place_trade(symbol, 'buy', current_price, stop_price, tp_price, RISK_USD)
                else:
                    stop_price = fvg_extreme + dynamic_buffer
                    tp_price = current_price - (stop_price - current_price) * RR_RATIO
                    await place_trade(symbol, 'sell', current_price, stop_price, tp_price, RISK_USD)
                daily_trades += 1

            await asyncio.sleep(25)
        except Exception as e:
            log(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("\n\nBot stopped by user.")
    except Exception as e:
        log(f"Unexpected error: {e}")
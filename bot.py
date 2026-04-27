import ccxt
import json
import os
import asyncio
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from flask import Flask, Response
import threading
import logging
from collections import deque

load_dotenv()

# ========================= CONFIG =========================
with open('config.json', 'r') as f:
    config = json.load(f)

RISK_USD = config.get("risk_usd", 1.0)
LEVERAGE = config.get("leverage", 5)
SYMBOLS = config.get("symbols", ["PI_XBTUSD"])
TIMEFRAME = config.get("timeframe", "1m")
HTF_TIMEFRAME = "15m"
WARMUP_MINUTES = config.get("warmup_minutes", 10)
MAX_TRADES_PER_DAY = config.get("max_trades_per_day", 8)
MAX_DAILY_LOSS_USD = config.get("max_daily_loss_usd", 5.0)
ATR_MULTIPLIER = 1.5
RR_RATIO = config.get("rr_ratio", 4.0)
USE_TESTNET = config.get("use_testnet", True)

exchange = ccxt.krakenfutures({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(USE_TESTNET)

# ====================== LOGGING SETUP ======================
log_file = f"trading_log_{date.today()}.txt"
recent_logs = deque(maxlen=2000)

def log(message, to_file=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    recent_logs.append(line)
    if to_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[LOG ERROR] {e}")

# ====================== FLASK LIVE VIEWER ======================
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.route('/')
def show_log():
    try:
        display_content = "\n".join(recent_logs)
        html = f"""
        <html>
        <head>
            <title>Kraken ICT Bot - Live Log</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; line-height: 1.4; margin: 0; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 13px; max-height: 88vh; overflow-y: auto; background: #252526; padding: 15px; border-radius: 6px; border: 1px solid #3c3c3c; }}
                .header {{ color: #569cd6; }}
                .info {{ font-size: 13px; color: #888; margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            <h2 class="header">🚀 Kraken ICT Bot - LIVE Trading Log (15m Bias + Cooldown)</h2>
            <div class="info">
                ✅ 1m CHOCH + Fresh FVG + Strong 15m Bias + Auto Cooldown<br>
                Last updated: <span id="timestamp"></span>
            </div>
            <pre id="logpre">{display_content}</pre>
            <script>
                async function refreshLog() {{
                    try {{
                        const response = await fetch('/raw?' + Date.now(), {{ cache: 'no-store' }});
                        const text = await response.text();
                        const pre = document.getElementById('logpre');
                        pre.textContent = text;
                        pre.scrollTop = pre.scrollHeight;
                        document.getElementById('timestamp').textContent = new Date().toLocaleString();
                    }} catch(e) {{}}
                }}
                window.onload = () => {{
                    document.getElementById('timestamp').textContent = new Date().toLocaleString();
                    const pre = document.getElementById('logpre');
                    pre.scrollTop = pre.scrollHeight;
                    setInterval(refreshLog, 3000);
                }};
            </script>
        </body>
        </html>
        """
        response = Response(html)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception:
        return "<h2>Error reading log</h2>"

@app.route('/raw')
def raw_log():
    response = Response("\n".join(recent_logs))
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ====================== BOT LOGIC ======================
candle_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}
active_trades = {}
last_trade_time = None  # Global cooldown for quality control

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
            fvgs.append(('bullish', (df['low'].iloc[i] + df['high'].iloc[i-2])/2, df['low'].iloc[i-1], i))
        if df['low'].iloc[i-2] > df['high'].iloc[i]:
            fvgs.append(('bearish', (df['high'].iloc[i] + df['low'].iloc[i-2])/2, df['high'].iloc[i-1], i))
    return fvgs

async def fetch_latest_candles(symbol, tf, limit=500):
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

        log(f"📊 [{symbol}] Placing {side.upper()} LIMIT | Qty: {quantity:.2f}", to_file=True)

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
            'quantity': float(quantity),
            'open_time': datetime.now()
        }

        log(f"✅ [{symbol}] {side.upper()} POSITION OPENED", to_file=True)
        return order
    except Exception as e:
        log(f"❌ [{symbol}] Order failed: {e}", to_file=True)
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
        duration = (datetime.now() - trade['open_time']).seconds // 60
        log(f"   ✅ [{symbol}] {trade['side'].upper()} {result} CLOSED | PNL: ${pnl:.2f} | Duration: {duration} min", to_file=True)
        del active_trades[symbol]

async def main():
    global daily_trades, daily_pnl, wins, losses, current_day, log_file, last_trade_time
    log("🚀 1m CHOCH + Fresh FVG + Strong 15m Bias + Auto Cooldown (Video Strategy Enhanced)\n", to_file=True)

    # Startup checks (unchanged)
    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                symbol = pos['symbol']
                side = 'buy' if float(pos['contracts']) > 0 else 'sell'
                log(f"🔄 Found existing {side.upper()} position on {symbol}", to_file=True)
                active_trades[symbol] = {'side': side, 'entry_price': float(pos.get('entryPrice', 0)), 'stop_price': 0, 'tp_price': 0, 'quantity': abs(float(pos['contracts'])), 'open_time': datetime.now()}
    except Exception as e:
        log(f"Warning: Could not load positions: {e}", to_file=True)

    warmup_end = datetime.now() + timedelta(minutes=WARMUP_MINUTES)
    in_warmup = WARMUP_MINUTES > 0

    # Calculate cooldown (e.g. 8 trades/day = ~3 hours between trades)
    cooldown_minutes = 1440 / MAX_TRADES_PER_DAY
    log(f"Cooldown set to {cooldown_minutes:.1f} minutes between trades (based on {MAX_TRADES_PER_DAY} max/day)", to_file=True)

    while True:
        try:
            today = date.today()
            if today != current_day:
                win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                log(f"\n📅 === DAILY SUMMARY ({current_day}) === Trades: {daily_trades} | PNL: ${daily_pnl:.2f} | Win Rate: {win_rate:.1f}%", to_file=True)
                log_file = f"trading_log_{today}.txt"
                daily_trades = 0
                daily_pnl = 0.0
                wins = 0
                losses = 0
                current_day = today
                last_trade_time = None  # Reset cooldown on new day

            if daily_pnl <= -MAX_DAILY_LOSS_USD:
                log(f"🚫 Daily max loss reached. Paused for today.", to_file=True)
                await asyncio.sleep(3600)
                continue

            if in_warmup and datetime.now() < warmup_end:
                log(f"   ⏳ Warm-up active ({(warmup_end - datetime.now()).seconds//60} min left)", to_file=False)
                await asyncio.sleep(25)
                continue
            elif in_warmup:
                log("✅ Warm-up finished. Starting live trading.", to_file=True)
                in_warmup = False

            # Cooldown check
            can_trade = True
            if last_trade_time is not None:
                minutes_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
                if minutes_since_last < cooldown_minutes:
                    can_trade = False
                    log(f"⏳ Cooldown active: {minutes_since_last:.1f}/{cooldown_minutes:.1f} min since last trade", to_file=False)

            best_setup = None
            best_score = -999

            for symbol in SYMBOLS:
                df1m = await fetch_latest_candles(symbol, TIMEFRAME)
                df15m = await fetch_latest_candles(symbol, HTF_TIMEFRAME)

                if len(df1m) < 100 or len(df15m) < 100:
                    continue

                choch_15m = detect_choch(df15m)
                choch_1m = detect_choch(df1m)
                fvg_1m = detect_fvg(df1m)

                current_price = df1m['close'].iloc[-1]
                atr = calculate_atr(df1m)

                latest_15m_bias = choch_15m[-1][0] if choch_15m else None

                log(f"[{symbol}] Price: {current_price:.4f} | 15m Bias: {latest_15m_bias or 'NONE'} | 15m CHOCH: {len(choch_15m)} | 1m CHOCH: {len(choch_1m)} | 1m FVGs: {len(fvg_1m)}", to_file=False)

                await manage_open_trade(symbol, df1m)
                if symbol in active_trades or not can_trade:
                    continue

                latest_choch_1m_idx = choch_1m[-1][1] if choch_1m else -1

                for fvg in fvg_1m[-10:]:
                    fvg_type, midpoint, fvg_extreme, fvg_idx = fvg

                    # Must be fresh after 1m CHOCH
                    if latest_choch_1m_idx != -1 and fvg_idx < latest_choch_1m_idx:
                        continue

                    # Must match current 15m bias (strong filter)
                    if latest_15m_bias and fvg_type != latest_15m_bias:
                        continue

                    if abs(current_price - midpoint) / midpoint > 0.003:
                        continue

                    recent_candle = df1m.iloc[-1]
                    if (fvg_type == 'bullish' and recent_candle['close'] <= recent_candle['open']) or \
                       (fvg_type == 'bearish' and recent_candle['close'] >= recent_candle['open']):
                        continue

                    score = 100 - (abs(current_price - midpoint) / midpoint * 1000)
                    if score > best_score:
                        best_score = score
                        best_setup = (symbol, fvg_type, current_price, fvg_extreme, ATR_MULTIPLIER * atr)

            if best_setup and daily_trades < MAX_TRADES_PER_DAY and can_trade:
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
                last_trade_time = datetime.now()

            await asyncio.sleep(25)
        except Exception as e:
            log(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    def load_stats():
        if os.path.exists("bot_stats.json"):
            try:
                with open("bot_stats.json", 'r') as f:
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
    last_trade_time = None

    log(f"Bot started | Risk: ${RISK_USD} | Max Trades/Day: {MAX_TRADES_PER_DAY} | Cooldown: {1440/MAX_TRADES_PER_DAY:.1f} min")
    log(f"Symbols: {SYMBOLS} | 1m CHOCH + Fresh FVG + Strong 15m Bias + Spacing\n", to_file=True)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    log(f"🌐 Live log viewer running at http://0.0.0.0:{os.environ.get('PORT', 8080)}", to_file=True)

    asyncio.run(main())
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
import traceback

load_dotenv()

# ========================= CONFIG =========================
with open('config.json', 'r') as f:
    config = json.load(f)

RISK_PERCENT = config.get("risk_percent", 1.0)
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
daily_setups = deque(maxlen=300)

# Cumulative stats (never reset)
total_wins = 0
total_losses = 0
total_pnl = 0.0
total_win_usd = 0.0
total_loss_usd = 0.0

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
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"
        pnl_sign = "+" if total_pnl >= 0 else ""

        html = f"""
        <html>
        <head>
            <title>Kraken ICT Bot - Live Log</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; line-height: 1.4; margin: 0; }}
                .stats-header {{ background: #252526; border: 2px solid #3c3c3c; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; font-size: 18px; display: flex; flex-wrap: wrap; gap: 25px; align-items: center; }}
                .stat-item {{ display: flex; align-items: center; gap: 8px; }}
                .stat-label {{ color: #888; font-size: 14px; }}
                .pnl {{ font-weight: bold; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 13px; max-height: 78vh; overflow-y: auto; background: #252526; padding: 15px; border-radius: 6px; border: 1px solid #3c3c3c; }}
                .header {{ color: #569cd6; }}
            </style>
        </head>
        <body>
            <h2 class="header">🚀 Kraken ICT Bot - LIVE Trading Log (Manual Refresh)</h2>
            
            <div class="stats-header">
                <div class="stat-item"><span class="stat-label">WINS</span><strong style="color:#4ade80">{total_wins}</strong></div>
                <div class="stat-item"><span class="stat-label">LOSSES</span><strong style="color:#f87171">{total_losses}</strong></div>
                <div class="stat-item"><span class="stat-label">WIN RATE</span><strong>{win_rate:.1f}%</strong></div>
                <div class="stat-item"><span class="stat-label">TRADES TODAY</span><strong>{daily_trades}</strong></div>
                <div class="stat-item"><span class="stat-label">PNL TODAY</span><strong class="pnl" style="color:{pnl_color}">{pnl_sign}${total_pnl:.2f}</strong></div>
                <div class="stat-item"><span class="stat-label">WON USD</span><strong style="color:#4ade80">${total_win_usd:.2f}</strong></div>
                <div class="stat-item"><span class="stat-label">LOST USD</span><strong style="color:#f87171">${total_loss_usd:.2f}</strong></div>
            </div>

            <pre id="logpre">{display_content}</pre>
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

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ====================== BOT LOGIC ======================
candle_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}
active_trades = {}
last_trade_time = None

def get_account_balance():
    try:
        balance = exchange.fetch_balance()
        return float(balance.get('USDT', {}).get('free', 100.0))
    except:
        return 100.0

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
            fvgs.append(('bullish', (df['low'].iloc[i] + df['high'].iloc[i-2])/2, df['low'].iloc[i-1], i, df['high'].iloc[i-2], df['low'].iloc[i]))
        if df['low'].iloc[i-2] > df['high'].iloc[i]:
            fvgs.append(('bearish', (df['high'].iloc[i] + df['low'].iloc[i-2])/2, df['high'].iloc[i-1], i, df['low'].iloc[i-2], df['high'].iloc[i]))
    return fvgs

def score_fvg(fvg, current_price, df):
    fvg_type, midpoint, fvg_extreme, fvg_idx, fvg_start, fvg_end = fvg
    age = len(df) - fvg_idx
    size = abs(fvg_end - fvg_start)
    proximity = abs(current_price - midpoint) / midpoint
    momentum = 1 if (fvg_type == 'bullish' and df['close'].iloc[-1] > df['open'].iloc[-1]) else 1 if (fvg_type == 'bearish' and df['close'].iloc[-1] < df['open'].iloc[-1]) else 0.6
    freshness = max(0.3, 1 - age / 30)
    score = (1 - proximity) * 40 + freshness * 30 + (size / df['close'].iloc[-1]) * 20 + momentum * 10
    return round(score, 2)

async def fetch_latest_candles(symbol, tf, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log(f"❌ Candle error ({tf}) for {symbol}: {e}", to_file=False)
        return pd.DataFrame()

async def place_trade(symbol, side, current_price, stop_price, tp_price, risk_usd, score, choch_level):
    log(f"=== ATTEMPTING ORDER === [{symbol}] {side.upper()} | Score: {score:.1f} | CHOCH Level: {choch_level:.4f}", to_file=True)
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

        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=side,
            amount=quantity,
            price=limit_price,
            params=params
        )

        log(f"✅ ORDER CREATED | ID: {order.get('id')}", to_file=True)

        log(f"""
=== POSITION OPENED SUCCESSFULLY ===
Symbol     : {symbol}
Side       : {side.upper()}
Entry      : {current_price:.4f}
Stop Loss  : {stop_price:.4f}
Take Profit: {tp_price:.4f} ({RR_RATIO:.1f}R)
Quantity   : {quantity:.4f}
Risk       : ${risk_usd:.2f} ({RISK_PERCENT}% of account)
Leverage   : {LEVERAGE}x
Score      : {score:.1f}
CHOCH Level: {choch_level:.4f}
""".strip(), to_file=True)

        await asyncio.sleep(2)
        positions = exchange.fetch_positions()
        symbol_pos = next((p for p in positions if p['symbol'] == symbol and float(p.get('contracts', 0)) != 0), None)
        if symbol_pos:
            log(f"✅ REAL POSITION CONFIRMED | Contracts: {symbol_pos.get('contracts')}", to_file=True)

        active_trades[symbol] = {
            'side': side,
            'entry_price': float(current_price),
            'stop_price': float(stop_price),
            'tp_price': float(tp_price),
            'quantity': float(quantity),
            'open_time': datetime.now(),
            'choch_level': float(choch_level),
            'breakeven_moved': False
        }
        return order
    except Exception as e:
        log(f"❌ ORDER FAILED: {e}", to_file=True)
        return None

async def manage_open_trade(symbol, df):
    if symbol not in active_trades:
        return
    trade = active_trades[symbol]
    current_price = df['close'].iloc[-1]

    if not trade.get('breakeven_moved', False) and trade.get('choch_level'):
        choch_level = trade['choch_level']
        if (trade['side'] == 'buy' and current_price > choch_level) or \
           (trade['side'] == 'sell' and current_price < choch_level):
            try:
                new_sl = trade['entry_price']
                side_for_sl = 'sell' if trade['side'] == 'buy' else 'buy'
                exchange.create_order(
                    symbol=symbol,
                    type='stopMarket',
                    side=side_for_sl,
                    amount=trade['quantity'],
                    params={'stopPrice': new_sl, 'reduceOnly': True, 'trigger': 'mark'}
                )
                trade['breakeven_moved'] = True
                log(f"🔄 [{symbol}] BREAKEVEN TRIGGERED → SL moved to entry {new_sl:.4f}", to_file=True)
            except Exception as e:
                log(f"⚠️ Breakeven update failed for {symbol}: {e}", to_file=True)

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
        global daily_pnl, daily_trades, total_wins, total_losses, total_pnl, total_win_usd, total_loss_usd
        daily_pnl += pnl
        total_pnl += pnl

        if result == "WIN":
            total_wins += 1
            total_win_usd += abs(pnl)
        else:
            total_losses += 1
            total_loss_usd += abs(pnl)

        duration = (datetime.now() - trade['open_time']).seconds // 60
        log(f"   ✅ [{symbol}] {trade['side'].upper()} {result} CLOSED | PNL: ${pnl:.2f} | Duration: {duration} min", to_file=True)
        del active_trades[symbol]

async def main():
    global daily_trades, daily_pnl, total_wins, total_losses, total_pnl, total_win_usd, total_loss_usd, current_day, log_file, last_trade_time
    log("🚀 Real Money Ready - Cumulative Stats + Dynamic Risk + Improved Stop Distance\n", to_file=True)

    try:
        positions = exchange.fetch_positions()
        for pos in positions:
            if float(pos.get('contracts', 0)) != 0:
                symbol = pos['symbol']
                side = 'buy' if float(pos['contracts']) > 0 else 'sell'
                log(f"🔄 Found existing {side.upper()} position on {symbol}", to_file=True)
                active_trades[symbol] = {'side': side, 'entry_price': float(pos.get('entryPrice', 0)), 'stop_price': 0, 'tp_price': 0, 'quantity': abs(float(pos['contracts'])), 'open_time': datetime.now(), 'choch_level': 0, 'breakeven_moved': False}
    except Exception as e:
        log(f"Warning: Could not load positions: {e}", to_file=True)

    warmup_end = datetime.now() + timedelta(minutes=WARMUP_MINUTES)
    in_warmup = WARMUP_MINUTES > 0
    cooldown_minutes = 1440 / MAX_TRADES_PER_DAY
    wiggle_minutes = cooldown_minutes * 0.25

    while True:
        try:
            # Dynamic risk calculation
            account_balance = get_account_balance()
            RISK_USD = account_balance * (RISK_PERCENT / 100.0)

            today = date.today()
            if today != current_day:
                log_file = f"trading_log_{today}.txt"
                daily_trades = 0
                daily_pnl = 0.0
                current_day = today
                daily_setups.clear()
                last_trade_time = None

            if daily_pnl <= -MAX_DAILY_LOSS_USD:
                log(f"🚫 Daily max loss reached. Paused.", to_file=True)
                await asyncio.sleep(3600)
                continue

            if in_warmup and datetime.now() < warmup_end:
                log(f"   ⏳ Warm-up active ({(warmup_end - datetime.now()).seconds//60} min left)", to_file=False)
                await asyncio.sleep(25)
                continue
            elif in_warmup:
                log("✅ Warm-up finished. Starting live trading.", to_file=True)
                in_warmup = False

            can_trade = True
            minutes_since_last = 9999
            if last_trade_time:
                minutes_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
                if minutes_since_last < cooldown_minutes - wiggle_minutes:
                    can_trade = False

            best_setup = None
            best_score = -999
            best_choch_level = None

            for symbol in SYMBOLS:
                df1m = await fetch_latest_candles(symbol, TIMEFRAME)
                df15m = await fetch_latest_candles(symbol, HTF_TIMEFRAME)

                if len(df1m) < 100 or len(df15m) < 100:
                    continue

                choch_15m = detect_choch(df15m)
                choch_1m = detect_choch(df1m)
                fvg_1m = detect_fvg(df1m)

                current_price = df1m['close'].iloc[-1]
                latest_15m_bias = choch_15m[-1][0] if choch_15m else None

                log(f"[{symbol}] Price: {current_price:.4f} | 15m Bias: {latest_15m_bias or 'NONE'} | 1m CHOCH: {len(choch_1m)} | 1m FVGs: {len(fvg_1m)}", to_file=False)

                await manage_open_trade(symbol, df1m)
                if symbol in active_trades:
                    continue

                latest_choch_1m_idx = choch_1m[-1][1] if choch_1m else -1
                latest_choch_level = choch_1m[-1][2] if choch_1m else current_price

                for fvg in fvg_1m[-12:]:
                    fvg_type, midpoint, fvg_extreme, fvg_idx, _, _ = fvg
                    if latest_choch_1m_idx != -1 and fvg_idx < latest_choch_1m_idx:
                        continue
                    if latest_15m_bias and fvg_type != latest_15m_bias:
                        continue

                    score = score_fvg(fvg, current_price, df1m)
                    daily_setups.append({'time': datetime.now(), 'symbol': symbol, 'score': score, 'fvg_type': fvg_type, 'price': current_price})

                    if abs(current_price - midpoint) / midpoint > 0.003:
                        continue
                    recent_candle = df1m.iloc[-1]
                    if (fvg_type == 'bullish' and recent_candle['close'] <= recent_candle['open']) or \
                       (fvg_type == 'bearish' and recent_candle['close'] >= recent_candle['open']):
                        continue

                    if score > best_score:
                        best_score = score
                        best_setup = (symbol, fvg_type, current_price, fvg_extreme, ATR_MULTIPLIER * calculate_atr(df1m))
                        best_choch_level = latest_choch_level

            is_elite = False
            if daily_setups and best_score > 0:
                all_scores = [s['score'] for s in daily_setups]
                percentile = np.percentile(all_scores, 90) if all_scores else 80
                is_elite = best_score >= percentile or best_score > 92

            allow_by_wiggle = is_elite and minutes_since_last < cooldown_minutes + wiggle_minutes and minutes_since_last > cooldown_minutes - wiggle_minutes

            if best_setup and daily_trades < MAX_TRADES_PER_DAY and (can_trade or allow_by_wiggle):
                symbol, direction, current_price, fvg_extreme, dynamic_buffer = best_setup

                # Improved minimum stop distance
                if direction == 'bullish':
                    stop_price = fvg_extreme - dynamic_buffer
                    min_distance = max(current_price * 0.002, 0.0001, dynamic_buffer * 1.5)  # 0.2% or 2× ATR
                    if stop_price >= current_price - min_distance:
                        log(f"⚠️ [{symbol}] Skipped bad setup (stop too close to entry)", to_file=False)
                        continue
                    tp_price = current_price + (current_price - stop_price) * RR_RATIO
                    await place_trade(symbol, 'buy', current_price, stop_price, tp_price, RISK_USD, best_score, best_choch_level)
                else:
                    stop_price = fvg_extreme + dynamic_buffer
                    min_distance = max(current_price * 0.002, 0.0001, dynamic_buffer * 1.5)
                    if stop_price <= current_price + min_distance:
                        log(f"⚠️ [{symbol}] Skipped bad setup (stop too close to entry)", to_file=False)
                        continue
                    tp_price = current_price - (stop_price - current_price) * RR_RATIO
                    await place_trade(symbol, 'sell', current_price, stop_price, tp_price, RISK_USD, best_score, best_choch_level)
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
    global wins, losses, daily_pnl, daily_trades
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    daily_pnl = 0.0
    daily_trades = 0
    current_day = date.today()
    last_trade_time = None

    log(f"Bot started | Risk: {RISK_PERCENT}% of balance | Cumulative stats + Improved Stop Distance")
    log(f"Symbols: {SYMBOLS} | Real Money Ready\n", to_file=True)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    log(f"🌐 Live log viewer running at http://0.0.0.0:{os.environ.get('PORT', 8080)} (manual refresh)", to_file=True)

    asyncio.run(main())
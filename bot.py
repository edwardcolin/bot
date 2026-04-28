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
import time

load_dotenv()

# ========================= CONFIG =========================
with open('config.json', 'r') as f:
    config = json.load(f)

RISK_PERCENT = config.get("risk_percent", 1.0)
LEVERAGE = config.get("leverage", 5)
SYMBOLS = config.get("symbols", ["PI_XBTUSD"])
TIMEFRAME = config.get("timeframe", "1m")
WARMUP_MINUTES = config.get("warmup_minutes", 0)
MAX_TRADES_PER_DAY = config.get("max_trades_per_day", 9)
MAX_DAILY_LOSS_USD = config.get("max_daily_loss_usd", 5.0)
RR_RATIO = config.get("rr_ratio", 3.7)
USE_TESTNET = config.get("use_testnet", True)

exchange = ccxt.krakenfutures({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
    'timeout': 15000,
})
exchange.set_sandbox_mode(USE_TESTNET)

# ====================== LOGGING SETUP ======================
log_file = f"trading_log_{date.today()}.txt"
recent_logs = deque(maxlen=2000)
daily_setups = deque(maxlen=300)

total_wins = 0
total_losses = 0
total_pnl = 0.0
total_win_usd = 0.0
total_loss_usd = 0.0

def log(message, to_file=True, to_recent=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if to_recent:
        recent_logs.append(line)
    if to_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"[LOG ERROR] {e}")

# ====================== ROBUST MARKET LOAD ======================
def load_markets_robust():
    for attempt in range(3):
        try:
            log(f"📡 Loading markets (attempt {attempt+1}/3)...", to_file=False)
            markets = exchange.load_markets()
            log(f"✅ Loaded {len(markets)} markets successfully", to_file=False)
            return markets
        except Exception as e:
            log(f"⚠️ Market load failed (attempt {attempt+1}): {e}", to_file=False)
            if attempt < 2:
                time.sleep(5)
    log("⚠️ Could not load markets from Kraken. Continuing with safe defaults...", to_file=False)
    return {}

load_markets_robust()

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
        active_summary = ""
        for sym, trade in active_trades.items():
            active_summary += f"<div class='trade-item'>[{sym}] {trade['side'].upper()} @ {trade['entry_price']:.4f} | SL: {trade['stop_price']:.4f} | TP: {trade['tp_price']:.4f}</div>"

        html = f"""
        <html>
        <head>
            <title>Kraken ICT Bot - LIVE</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; line-height: 1.4; margin: 0; }}
                .stats-header {{ background: #252526; border: 2px solid #3c3c3c; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; font-size: 18px; display: flex; flex-wrap: wrap; gap: 25px; align-items: center; }}
                .stat-item {{ display: flex; align-items: center; gap: 8px; }}
                .stat-label {{ color: #888; font-size: 14px; }}
                .pnl {{ font-weight: bold; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 13px; max-height: 65vh; overflow-y: auto; background: #252526; padding: 15px; border-radius: 6px; border: 1px solid #3c3c3c; }}
                .header {{ color: #569cd6; }}
                .buttons {{ margin-bottom: 15px; display: flex; gap: 15px; flex-wrap: wrap; }}
                button {{ padding: 10px 20px; font-size: 14px; background: #3c3c3c; color: #d4d4d4; border: 1px solid #569cd6; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background: #569cd6; color: #1e1e1e; }}
                .trade-item {{ background: #2a2a2a; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #569cd6; }}
            </style>
        </head>
        <body>
            <h2 class="header">🚀 Kraken ICT Bot - LIVE (1m CHOCH 90min Pattern + FVG)</h2>
            <div class="buttons">
                <button onclick="clearLogs()">🗑️ Clear Logs</button>
                <button onclick="resetDailyStats()">🔄 Reset Daily Stats</button>
                <button onclick="toggleAutoRefresh()">🔄 Toggle Auto-Refresh (15s)</button>
            </div>
            <div class="stats-header">
                <div class="stat-item"><span class="stat-label">WINS</span><strong style="color:#4ade80">{total_wins}</strong></div>
                <div class="stat-item"><span class="stat-label">LOSSES</span><strong style="color:#f87171">{total_losses}</strong></div>
                <div class="stat-item"><span class="stat-label">WIN RATE</span><strong>{win_rate:.1f}%</strong></div>
                <div class="stat-item"><span class="stat-label">TRADES TODAY</span><strong>{daily_trades}</strong></div>
                <div class="stat-item"><span class="stat-label">PNL TODAY</span><strong class="pnl" style="color:{pnl_color}">{pnl_sign}${daily_pnl:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">TOTAL PNL</span><strong class="pnl" style="color:{pnl_color}">{pnl_sign}${total_pnl:.5f}</strong></div>
            </div>
            <h3 style="color:#569cd6; margin: 10px 0 5px;">Active Trades</h3>
            <div style="margin-bottom: 20px; max-height: 150px; overflow-y: auto;">
                {active_summary or '<em style="color:#888;">No active trades</em>'}
            </div>
            <pre id="logpre">{display_content}</pre>
            <script>
                let autoRefreshInterval = null;
                function clearLogs() {{ fetch('/clear_logs').then(() => location.reload()); }}
                function resetDailyStats() {{ fetch('/reset_daily_stats').then(() => location.reload()); }}
                function toggleAutoRefresh() {{
                    if (autoRefreshInterval) {{ clearInterval(autoRefreshInterval); autoRefreshInterval = null; alert('Auto-refresh disabled'); }}
                    else {{ autoRefreshInterval = setInterval(() => location.reload(), 15000); alert('Auto-refresh enabled (15s)'); }}
                }}
            </script>
        </body>
        </html>
        """
        response = Response(html)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception:
        return "<h2>Error reading log</h2>"

@app.route('/clear_logs')
def clear_logs():
    recent_logs.clear()
    return "Console logs cleared", 200

@app.route('/reset_daily_stats')
def reset_daily_stats():
    global daily_trades, daily_pnl
    daily_trades = 0
    daily_pnl = 0.0
    return "Daily stats reset", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ====================== BOT LOGIC ======================
active_trades = {}
last_trade_time = None
daily_trades = 0
daily_pnl = 0.0
current_day = date.today()

def get_account_balance():
    try:
        balance = exchange.fetch_balance()
        usd_free = float(balance.get('USD', {}).get('free', 0) or balance.get('info', {}).get('marginAvailable', 100.0))
        return max(usd_free, 100.0)
    except:
        return 100.0

def detect_swing_highs_lows(df, strength=5):
    highs = df['high'].rolling(window=strength*2+1, center=True).max()
    lows = df['low'].rolling(window=strength*2+1, center=True).min()
    df['swing_high'] = np.where(df['high'] == highs, df['high'], np.nan)
    df['swing_low'] = np.where(df['low'] == lows, df['low'], np.nan)
    return df

def detect_choch(df):
    if len(df) < 100:
        return []
    recent_df = df.iloc[-90:].copy()
    recent_df = detect_swing_highs_lows(recent_df)
    signals = []
    for i in range(20, len(recent_df)):
        recent_highs = recent_df['swing_high'].iloc[i-20:i].dropna()
        recent_lows = recent_df['swing_low'].iloc[i-20:i].dropna()
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            continue
        if (recent_df['close'].iloc[i] > recent_highs.iloc[-1] and 
            recent_df['close'].iloc[i-1] <= recent_highs.iloc[-1] and
            recent_highs.iloc[-1] > recent_highs.iloc[-2]):
            signals.append(('bullish', len(df)-90+i, recent_highs.iloc[-1]))
        if (recent_df['close'].iloc[i] < recent_lows.iloc[-1] and 
            recent_df['close'].iloc[i-1] >= recent_lows.iloc[-1] and
            recent_lows.iloc[-1] < recent_lows.iloc[-2]):
            signals.append(('bearish', len(df)-90+i, recent_lows.iloc[-1]))
    return signals

def detect_fvg(df):
    fvgs = []
    for i in range(3, len(df)):
        c1 = df.iloc[i-3]
        c2 = df.iloc[i-2]
        c3 = df.iloc[i-1]
        if c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and c1['high'] < c3['low']:
            midpoint = (c1['high'] + c3['low']) / 2
            fvgs.append({'type': 'bullish', 'midpoint': midpoint, 'extreme': c3['low'], 'idx': i, 'candle_idx': i-1})
        if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] < c3['open'] and c1['low'] > c3['high']:
            midpoint = (c1['low'] + c3['high']) / 2
            fvgs.append({'type': 'bearish', 'midpoint': midpoint, 'extreme': c3['high'], 'idx': i, 'candle_idx': i-1})
    return fvgs

async def fetch_latest_candles(symbol, tf, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log(f"❌ Candle error for {symbol}: {e}", to_file=False, to_recent=False)
        return pd.DataFrame()

async def place_trade(symbol, side, current_price, stop_price, tp_price, risk_usd, score, choch_level, fvg_extreme):
    log(f"=== ATTEMPTING ORDER === [{symbol}] {side.upper()} | Score: {score:.1f} | CHOCH Level: {choch_level:.4f}", to_file=True)
    try:
        m = exchange.market(symbol)
        distance = abs(current_price - stop_price)
        if distance <= 0:
            return None

        min_amount = m.get('limits', {}).get('amount', {}).get('min') or 1
        quantity = risk_usd / distance
        # FIX: Kraken Futures requires integer contracts → safe integer rounding
        quantity = round(quantity)
        quantity = max(int(min_amount), int(quantity))
        quantity = min(quantity, 2000)

        log(f"📊 Calculated quantity: {quantity} (integer contracts for {symbol})", to_file=True)

        exchange.set_leverage(LEVERAGE, symbol)
        limit_price = current_price * (1.002 if side == 'buy' else 0.998)

        order = exchange.create_order(symbol=symbol, type='limit', side=side, amount=quantity, price=limit_price, params={'leverage': LEVERAGE})
        log(f"✅ ENTRY ORDER CREATED | ID: {order.get('id')}", to_file=True)

        await asyncio.sleep(2)
        positions = exchange.fetch_positions()
        if not any(p['symbol'] == symbol and float(p.get('contracts', 0)) != 0 for p in positions):
            log("❌ Position not confirmed", to_file=True)
            return None

        opposite = 'sell' if side == 'buy' else 'buy'
        exchange.create_order(symbol=symbol, type='stopMarket', side=opposite, amount=quantity, params={'stopPrice': stop_price, 'reduceOnly': True, 'trigger': 'mark'})
        log(f"✅ SL placed @ {stop_price:.4f} (comfortably outside FVG)", to_file=True)
        exchange.create_order(symbol=symbol, type='takeProfit', side=opposite, amount=quantity, price=tp_price, params={'reduceOnly': True, 'trigger': 'mark'})
        log(f"✅ TP placed @ {tp_price:.4f} ({RR_RATIO:.1f}R)", to_file=True)

        active_trades[symbol] = {
            'side': side,
            'entry_price': float(current_price),
            'stop_price': float(stop_price),
            'tp_price': float(tp_price),
            'quantity': float(quantity),
            'open_time': datetime.now(),
            'choch_level': float(choch_level),
            'fvg_extreme': float(fvg_extreme),
            'breakeven_moved': False
        }
        log(f"=== POSITION OPENED SUCCESSFULLY (1m CHOCH+FVG) ===", to_file=True)
        return order
    except Exception as e:
        log(f"❌ ORDER FAILED: {e}", to_file=True)
        return None

async def manage_open_trade(symbol, df):
    if symbol not in active_trades:
        return
    trade = active_trades[symbol]
    current_price = df['close'].iloc[-1]
    fvg_extreme = trade['fvg_extreme']

    if not trade.get('breakeven_moved', False):
        if (trade['side'] == 'buy' and current_price > fvg_extreme) or \
           (trade['side'] == 'sell' and current_price < fvg_extreme):
            try:
                new_sl = trade['entry_price']
                side_for_sl = 'sell' if trade['side'] == 'buy' else 'buy'
                exchange.create_order(symbol=symbol, type='stopMarket', side=side_for_sl, amount=trade['quantity'],
                                      params={'stopPrice': new_sl, 'reduceOnly': True, 'trigger': 'mark'})
                trade['stop_price'] = new_sl
                trade['breakeven_moved'] = True
                log(f"🔄 [{symbol}] BREAKEVEN TRIGGERED (BOS of FVG candle)", to_file=True)
            except Exception as e:
                log(f"⚠️ Breakeven failed: {e}", to_file=True)

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
        del active_trades[symbol]
        log(f"   ✅ [{symbol}] {trade['side'].upper()} {result} CLOSED (simulated)", to_file=True)

async def main():
    global daily_trades, daily_pnl, total_wins, total_losses, total_pnl, total_win_usd, total_loss_usd, current_day, log_file, last_trade_time, RISK_USD
    log("🚀 Kraken ICT Bot Started - 1m CHOCH (90min pattern) + FVG + ALL errors fixed", to_file=True)

    warmup_end = datetime.now() + timedelta(minutes=WARMUP_MINUTES)
    in_warmup = WARMUP_MINUTES > 0
    cooldown_minutes = 1440 / MAX_TRADES_PER_DAY
    wiggle_minutes = cooldown_minutes * 0.25

    while True:
        try:
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
                log("🚫 Daily max loss reached. Paused.", to_file=True)
                await asyncio.sleep(3600)
                continue

            if in_warmup and datetime.now() < warmup_end:
                log("   ⏳ Warm-up active", to_file=False, to_recent=False)
                await asyncio.sleep(25)
                continue
            elif in_warmup:
                log("✅ Warm-up finished.", to_file=True)
                in_warmup = False

            best_setup = None

            for symbol in SYMBOLS:
                df1m = await fetch_latest_candles(symbol, TIMEFRAME)
                if len(df1m) < 100:
                    continue

                choch_1m = detect_choch(df1m)
                fvg_1m = detect_fvg(df1m)
                current_price = df1m['close'].iloc[-1]
                latest_choch_idx = choch_1m[-1][1] if choch_1m else -1

                for fvg in fvg_1m[-20:]:
                    fvg_type = fvg['type']
                    midpoint = fvg['midpoint']
                    fvg_extreme = fvg['extreme']
                    fvg_idx = fvg['idx']
                    fvg_candle_idx = fvg['candle_idx']

                    if latest_choch_idx != -1 and fvg_idx < latest_choch_idx:
                        continue

                    confirmation_window = df1m.iloc[-12:-2]
                    if fvg_type == 'bullish':
                        confirmed = (confirmation_window['close'] > midpoint).mean() >= 0.7
                    else:
                        confirmed = (confirmation_window['close'] < midpoint).mean() >= 0.7
                    if not confirmed:
                        continue

                    score = 90.0
                    if best_setup is None or score > best_setup.get('score', 0):
                        best_setup = {
                            'symbol': symbol,
                            'direction': fvg_type,
                            'current_price': current_price,
                            'fvg_extreme': fvg_extreme,
                            'fvg_candle_idx': fvg_candle_idx,
                            'score': score,
                            'choch_level': choch_1m[-1][2] if choch_1m else current_price
                        }

                await manage_open_trade(symbol, df1m)

            if best_setup and daily_trades < MAX_TRADES_PER_DAY:
                s = best_setup
                buffer = 0.0005 * s['current_price']
                if s['direction'] == 'bullish':
                    stop_price = s['fvg_extreme'] - buffer
                    tp_price = s['current_price'] + (s['current_price'] - stop_price) * RR_RATIO
                    await place_trade(s['symbol'], 'buy', s['current_price'], stop_price, tp_price, RISK_USD, s['score'], s['choch_level'], s['fvg_extreme'])
                else:
                    stop_price = s['fvg_extreme'] + buffer
                    tp_price = s['current_price'] - (stop_price - s['current_price']) * RR_RATIO
                    await place_trade(s['symbol'], 'sell', s['current_price'], stop_price, tp_price, RISK_USD, s['score'], s['choch_level'], s['fvg_extreme'])
                daily_trades += 1
                last_trade_time = datetime.now()

            await asyncio.sleep(25)
        except Exception as e:
            log(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log(f"🌐 Live viewer running at http://0.0.0.0:{os.environ.get('PORT', 8080)}", to_file=True)
    asyncio.run(main())
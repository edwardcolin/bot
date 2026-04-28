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
from uuid import uuid4
import difflib

load_dotenv()

# ========================= CONFIG =========================
with open('config.json', 'r') as f:
    config = json.load(f)

RISK_PERCENT = config.get("risk_percent", 1.0)
LEVERAGE = config.get("leverage", 1)
SYMBOLS = config.get("symbols", ["BTC/USD"])
TIMEFRAME = config.get("timeframe", "1m")
WARMUP_MINUTES = config.get("warmup_minutes", 0)
MAX_TRADES_PER_DAY = config.get("max_trades_per_day", 9)
MAX_DAILY_LOSS_USD = config.get("max_daily_loss_usd", 5.0)
RR_RATIO = config.get("rr_ratio", 3.7)
USE_TESTNET = config.get("use_testnet", True)
DRY_RUN_MODE = config.get("dry_run_mode", False)
try:
    DRY_RUN_STARTING_WALLET_USD = float(config.get("dry_run_starting_wallet_usd", 100.0))
except (TypeError, ValueError):
    DRY_RUN_STARTING_WALLET_USD = 100.0
PRECHECK_DISABLE_AFTER = int(config.get("precheck_disable_after", 3))
try:
    SIMULATED_FEE_BPS = float(config.get("simulated_fee_bps", 26.0))
except (TypeError, ValueError):
    SIMULATED_FEE_BPS = 26.0
try:
    SIMULATED_SLIPPAGE_BPS = float(config.get("simulated_slippage_bps", 5.0))
except (TypeError, ValueError):
    SIMULATED_SLIPPAGE_BPS = 5.0

exchange = ccxt.kraken({
    'apiKey': os.getenv('KRAKEN_API_KEY'),
    'secret': os.getenv('KRAKEN_API_SECRET'),
    'enableRateLimit': True,
    'timeout': 15000,
})
exchange.set_sandbox_mode(USE_TESTNET)
MODE_LABEL = "TESTNET" if USE_TESTNET else "LIVE"
EXECUTION_LABEL = "DRY-RUN" if DRY_RUN_MODE else "REAL-ORDERS"

# ====================== LOGGING SETUP ======================
log_file = f"trading_log_{date.today()}.txt"
audit_file = f"trade_audit_{date.today()}.jsonl"
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

def log_trade_audit(event, payload):
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "event": event,
        "payload": payload,
    }
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        log(f"⚠️ Audit log write failed: {e}", to_file=False, to_recent=False)

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

def resolve_configured_symbols(configured_symbols, markets):
    if not configured_symbols:
        return []
    if not markets:
        log("⚠️ Market cache is empty; symbol validation skipped.", to_file=True)
        return configured_symbols

    symbol_keys = set(markets.keys())
    id_to_symbol = {}
    for symbol_key, market in markets.items():
        market_id = (market or {}).get("id")
        if market_id:
            id_to_symbol[str(market_id).upper()] = symbol_key

    resolved = []
    for raw_symbol in configured_symbols:
        candidate = str(raw_symbol).strip()
        if not candidate:
            continue
        if candidate in symbol_keys:
            resolved.append(candidate)
            continue
        by_id = id_to_symbol.get(candidate.upper())
        if by_id:
            resolved.append(by_id)
            log(f"ℹ️ Symbol '{candidate}' resolved via market id -> '{by_id}'", to_file=True)
            continue

        # Help with common separator mistakes, e.g. spaces vs underscores.
        compact = candidate.replace(" ", "")
        normalized = candidate.replace(" ", "_")
        by_compact = id_to_symbol.get(compact.upper()) or id_to_symbol.get(normalized.upper())
        if by_compact:
            resolved.append(by_compact)
            log(f"ℹ️ Symbol '{candidate}' normalized -> '{by_compact}'", to_file=True)
            continue

        universe = list(symbol_keys) + list(id_to_symbol.keys())
        suggestions = difflib.get_close_matches(candidate.upper(), [u.upper() for u in universe], n=3, cutoff=0.45)
        if suggestions:
            log(
                f"❌ Unsupported Kraken spot symbol '{candidate}'. Closest matches: {', '.join(suggestions)}",
                to_file=True
            )
        else:
            log(
                f"❌ Unsupported Kraken spot symbol '{candidate}'. Use Kraken spot symbols like BTC/USD, ETH/USD.",
                to_file=True
            )

    deduped = []
    seen = set()
    for sym in resolved:
        if sym not in seen:
            deduped.append(sym)
            seen.add(sym)
    return deduped

MARKETS_CACHE = load_markets_robust()
SYMBOLS = resolve_configured_symbols(SYMBOLS, MARKETS_CACHE)
if not SYMBOLS:
    log("🚫 No valid Kraken spot symbols configured after validation. Bot will idle until symbols are fixed.", to_file=True)

# ====================== FLASK LIVE VIEWER ======================
app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

@app.route('/')
def show_log():
    try:
        balance_snapshot = live_balance.copy()
        display_content = "\n".join(recent_logs)
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        pnl_color = "#4ade80" if total_pnl >= 0 else "#f87171"
        pnl_sign = "+" if total_pnl >= 0 else ""
        risk_usd_preview = _safe_float(balance_snapshot.get("available_usd"), 0.0) * (RISK_PERCENT / 100.0)
        simulated_active_count = sum(1 for t in active_trades.values() if t.get("simulated"))
        simulation_notice_html = (
            f"<div style='margin-bottom:12px;padding:10px 14px;background:#3b2f07;border:1px solid #d4a017;border-radius:6px;color:#f7d774;'>"
            f"🧪 SIMULATION MODE ACTIVE (NO REAL ORDERS) | Active simulated trades: {simulated_active_count}"
            f"</div>"
            if DRY_RUN_MODE else ""
        )
        active_summary = ""
        for sym, trade in active_trades.items():
            active_summary += f"<div class='trade-item'>[{sym}] {trade['side'].upper()} @ {trade['entry_price']:.4f} | SL: {trade['stop_price']:.4f} | TP: {trade['tp_price']:.4f}</div>"

        html = f"""
        <html>
        <head>
            <title>Kraken ICT Bot - {MODE_LABEL}</title>
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
            <h2 class="header">🚀 Kraken ICT Bot - {MODE_LABEL} | {EXECUTION_LABEL} (1m CHOCH 90min Pattern + FVG)</h2>
            {simulation_notice_html}
            <div class="buttons">
                <button onclick="clearLogs()">🗑️ Clear Logs</button>
                <button onclick="resetDailyStats()">🔄 Reset Daily Stats</button>
                <button onclick="toggleAutoRefresh()">🔄 Toggle Auto-Refresh (15s)</button>
            </div>
            <div class="stats-header">
                <div class="stat-item"><span class="stat-label">WINS</span><strong style="color:#4ade80">{total_wins}</strong></div>
                <div class="stat-item"><span class="stat-label">LOSSES</span><strong style="color:#f87171">{total_losses}</strong></div>
                <div class="stat-item"><span class="stat-label">WIN AMOUNT</span><strong style="color:#4ade80">${total_win_usd:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">LOSS AMOUNT</span><strong style="color:#f87171">${total_loss_usd:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">WIN RATE</span><strong>{win_rate:.1f}%</strong></div>
                <div class="stat-item"><span class="stat-label">TRADES TODAY</span><strong>{daily_trades}</strong></div>
                <div class="stat-item"><span class="stat-label">PNL TODAY</span><strong class="pnl" style="color:{pnl_color}">{pnl_sign}${daily_pnl:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">TOTAL PNL</span><strong class="pnl" style="color:{pnl_color}">{pnl_sign}${total_pnl:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">WALLET</span><strong>${balance_snapshot.get('wallet_usd', 0.0):.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">AVAILABLE</span><strong>${balance_snapshot.get('available_usd', 0.0):.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">USED</span><strong>${balance_snapshot.get('used_usd', 0.0):.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">RISK/TRADE</span><strong>${risk_usd_preview:.5f}</strong></div>
                <div class="stat-item"><span class="stat-label">BALANCE TS</span><strong>{balance_snapshot.get('fetched_at') or '--:--:--'}</strong></div>
                <div class="stat-item"><span class="stat-label">DISABLED SYMBOLS</span><strong>{len(disabled_symbols)}</strong></div>
            </div>
            <h3 style="color:#569cd6; margin: 10px 0 5px;">Active Trades</h3>
            <div style="margin-bottom: 20px; max-height: 150px; overflow-y: auto;">
                {active_summary or '<em style="color:#888;">No active trades</em>'}
            </div>
            <h3 style="color:#569cd6; margin: 10px 0 5px;">Diagnostics (1m)</h3>
            <pre style="max-height: 140px; margin-bottom: 20px;">{latest_diagnostics}</pre>
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
live_balance = {
    "wallet_usd": 0.0,
    "equity_usd": 0.0,
    "available_usd": 0.0,
    "used_usd": 0.0,
    "fetched_at": None,
}
simulated_wallet_usd = max(DRY_RUN_STARTING_WALLET_USD, 0.0)
symbol_precheck_failures = {}
disabled_symbols = {}
latest_diagnostics = "No diagnostics yet."

def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _extract_fee_cost(trade):
    fee = trade.get('fee') or {}
    fee_cost = _safe_float(fee.get('cost'), 0.0)
    if fee_cost != 0:
        return abs(fee_cost)
    info = trade.get('info') or {}
    for key in ('fee', 'fees', 'execFee', 'commission'):
        value = info.get(key)
        parsed = _safe_float(value, 0.0)
        if parsed != 0:
            return abs(parsed)
    return 0.0

def _fetch_order_safe(order_id, symbol):
    if not order_id:
        return None
    try:
        return exchange.fetch_order(order_id, symbol)
    except Exception:
        return None

def _determine_close_reason_and_price(symbol, trade):
    tp_order = _fetch_order_safe(trade.get('tp_order_id'), symbol)
    sl_order = _fetch_order_safe(trade.get('sl_order_id'), symbol)

    close_reason = "MANUAL_OR_UNKNOWN"
    close_order = None
    if tp_order and tp_order.get('status') == 'closed':
        close_reason = "TP_HIT"
        close_order = tp_order
    elif sl_order and sl_order.get('status') == 'closed':
        close_reason = "SL_HIT"
        close_order = sl_order
    elif tp_order and _safe_float(tp_order.get('filled'), 0.0) > 0:
        close_reason = "TP_PARTIAL_OR_FILLED"
        close_order = tp_order
    elif sl_order and _safe_float(sl_order.get('filled'), 0.0) > 0:
        close_reason = "SL_PARTIAL_OR_FILLED"
        close_order = sl_order

    exit_price = None
    if close_order:
        exit_price = (
            _safe_float(close_order.get('average'), 0.0)
            or _safe_float(close_order.get('price'), 0.0)
            or _safe_float((close_order.get('info') or {}).get('stopPrice'), 0.0)
            or _safe_float((close_order.get('info') or {}).get('triggerPrice'), 0.0)
            or _safe_float((close_order.get('info') or {}).get('limitPrice'), 0.0)
        )
        if exit_price == 0:
            exit_price = None
    if exit_price is None:
        exit_price = _safe_float(trade.get('tp_price' if close_reason.startswith('TP') else 'stop_price'), 0.0)
    return close_reason, exit_price

def _compute_realized_pnl(symbol, trade, exit_price):
    entry_price = _safe_float(trade.get('entry_fill_price'), 0.0) or _safe_float(trade.get('entry_price'), 0.0)
    qty = _safe_float(trade.get('quantity'), 0.0)
    side = trade.get('side')
    if qty <= 0 or entry_price <= 0 or exit_price <= 0:
        return 0.0, 0.0

    gross = (exit_price - entry_price) * qty if side == 'buy' else (entry_price - exit_price) * qty
    fee_total = 0.0
    try:
        trades = exchange.fetch_my_trades(symbol, limit=100)
        tracked_ids = {
            trade.get('entry_order_id'),
            trade.get('sl_order_id'),
            trade.get('tp_order_id'),
        }
        for t in trades:
            if t.get('order') in tracked_ids:
                fee_total += _extract_fee_cost(t)
    except Exception:
        pass
    realized = gross - fee_total
    return realized, fee_total

def _compute_simulated_pnl(trade, exit_price):
    entry_price = _safe_float(trade.get('entry_fill_price'), 0.0) or _safe_float(trade.get('entry_price'), 0.0)
    qty = _safe_float(trade.get('quantity'), 0.0)
    side = trade.get('side')
    if qty <= 0 or entry_price <= 0 or exit_price <= 0:
        return 0.0
    return (exit_price - entry_price) * qty if side == 'buy' else (entry_price - exit_price) * qty

def _floor_to_step(value, step):
    if step <= 0:
        return max(value, 0.0)
    units = int(value / step)
    return max(units * step, 0.0)

def _apply_slippage(price, side, slippage_bps):
    factor = slippage_bps / 10000.0
    if side == 'buy':
        return price * (1.0 + factor)
    return price * (1.0 - factor)

def _estimate_fee(amount_quote, fee_bps):
    return max(amount_quote, 0.0) * (max(fee_bps, 0.0) / 10000.0)

def _estimate_stop_loss_usd(market, contracts, entry_price, stop_price):
    contracts = _safe_float(contracts, 0.0)
    entry_price = _safe_float(entry_price, 0.0)
    stop_price = _safe_float(stop_price, 0.0)
    if contracts <= 0 or entry_price <= 0 or stop_price <= 0:
        return 0.0
    price_distance = abs(entry_price - stop_price)
    contract_size = _safe_float(market.get('contractSize'), 1.0) or 1.0

    # Approximate USD risk from stop distance by contract type.
    if market.get('linear'):
        return contracts * contract_size * price_distance
    if market.get('inverse'):
        denom = max(min(entry_price, stop_price), 1e-8)
        return contracts * contract_size * (price_distance / denom)
    return contracts * price_distance

def _estimate_initial_margin_usd(market, contracts, entry_price, leverage):
    contracts = _safe_float(contracts, 0.0)
    entry_price = _safe_float(entry_price, 0.0)
    leverage = max(_safe_float(leverage, 1.0), 1.0)
    if contracts <= 0 or entry_price <= 0:
        return 0.0
    contract_size = _safe_float(market.get('contractSize'), 1.0) or 1.0
    if market.get('linear'):
        notional = contracts * contract_size * entry_price
    elif market.get('inverse'):
        notional = contracts * contract_size
    else:
        notional = contracts * entry_price
    # Keep a small buffer for fees/slippage.
    return (notional / leverage) * 1.05

def precheck_symbol_tradeability(symbol, current_price, stop_price, risk_usd):
    market = exchange.market(symbol)
    min_amount = _safe_float(market.get('limits', {}).get('amount', {}).get('min'), 1.0) or 1.0
    precision_amount = market.get('precision', {}).get('amount')
    step = 1.0
    if precision_amount is not None:
        try:
            step = 10 ** (-int(precision_amount))
        except Exception:
            step = 1.0
    min_qty = max(min_amount, step)
    available_usd = _safe_float(live_balance.get("available_usd"), 0.0)
    min_lot_risk = abs(current_price - stop_price) * min_qty
    min_lot_cost = current_price * min_qty

    if available_usd <= 0:
        return {
            "ok": False,
            "reason": "NO_AVAILABLE_BALANCE",
            "available_usd": available_usd,
            "min_lot_risk": min_lot_risk,
            "min_lot_cost": min_lot_cost,
        }
    if min_lot_cost > available_usd:
        return {
            "ok": False,
            "reason": "MIN_LOT_COST_EXCEEDS_AVAILABLE",
            "available_usd": available_usd,
            "min_lot_risk": min_lot_risk,
            "min_lot_cost": min_lot_cost,
        }
    if min_lot_risk > risk_usd:
        return {
            "ok": False,
            "reason": "MIN_LOT_RISK_EXCEEDS_TARGET",
            "available_usd": available_usd,
            "min_lot_risk": min_lot_risk,
            "min_lot_cost": min_lot_cost,
        }
    return {
        "ok": True,
        "reason": "OK",
        "available_usd": available_usd,
        "min_lot_risk": min_lot_risk,
        "min_lot_cost": min_lot_cost,
    }

def refresh_live_balance():
    global simulated_wallet_usd
    if DRY_RUN_MODE:
        simulated_used_usd = 0.0
        for trade in active_trades.values():
            if trade.get("simulated"):
                simulated_used_usd += max(_safe_float(trade.get("required_margin_estimate_usd"), 0.0), 0.0)
        simulated_used_usd = min(simulated_used_usd, simulated_wallet_usd)
        simulated_available_usd = max(simulated_wallet_usd - simulated_used_usd, 0.0)
        live_balance.update({
            "wallet_usd": max(simulated_wallet_usd, 0.0),
            "equity_usd": max(simulated_wallet_usd, 0.0),
            "available_usd": simulated_available_usd,
            "used_usd": simulated_used_usd,
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        })
        return live_balance.copy()
    try:
        balance = exchange.fetch_balance()
        usd_bucket = balance.get('USD', {}) or {}
        wallet = (
            _safe_float(usd_bucket.get('total'), 0.0)
        )
        available = (
            _safe_float(usd_bucket.get('free'), 0.0)
        )
        used = _safe_float(usd_bucket.get('used'), 0.0) or max(wallet - available, 0.0)
        live_balance.update({
            "wallet_usd": max(wallet, 0.0),
            "equity_usd": max(wallet, 0.0),
            "available_usd": max(available, 0.0),
            "used_usd": max(used, 0.0),
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        })
        return live_balance.copy()
    except Exception:
        return live_balance.copy()

def get_account_balance():
    snap = refresh_live_balance()
    return max(_safe_float(snap.get("available_usd"), 0.0), 0.0)

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
        if side != 'buy':
            log(f"⚠️ [{symbol}] Spot mode only supports long entries; skipping non-buy signal", to_file=True)
            return None
        m = exchange.market(symbol)
        distance = abs(current_price - stop_price)
        if distance <= 0:
            return None

        min_amount = _safe_float(m.get('limits', {}).get('amount', {}).get('min'), 1.0) or 1.0
        precision_amount = m.get('precision', {}).get('amount')
        step = 1.0
        if precision_amount is not None:
            try:
                step = 10 ** (-int(precision_amount))
            except Exception:
                step = 1.0

        balance_snapshot = refresh_live_balance()
        available_usd = _safe_float(balance_snapshot.get("available_usd"), 0.0)
        if available_usd <= 0:
            log(f"⚠️ [{symbol}] No available balance; skipping order", to_file=True)
            return None

        per_unit_risk = distance
        raw_quantity = max(risk_usd / per_unit_risk, 0.0)
        quantity = _floor_to_step(raw_quantity, step)
        if quantity < min_amount:
            log(
                f"⚠️ [{symbol}] Risk target too small for min size. "
                f"risk=${risk_usd:.5f}, per_unit_risk=${per_unit_risk:.5f}, min_amount={min_amount}",
                to_file=True
            )
            return None

        max_by_cash = _floor_to_step(available_usd / max(current_price, 1e-8), step)
        if max_by_cash <= 0:
            log(f"⚠️ [{symbol}] Available balance too low for spot entry", to_file=True)
            return None
        quantity = min(quantity, max_by_cash)

        est_risk = distance * quantity
        while quantity >= min_amount and est_risk > (risk_usd * 1.001):
            quantity = _floor_to_step(quantity - step, step)
            est_risk = distance * quantity

        if quantity < min_amount:
            log(
                f"⚠️ [{symbol}] Could not fit min lot within risk target. "
                f"Target=${risk_usd:.5f}, min_lot_risk=${(distance * min_amount):.5f}",
                to_file=True
            )
            return None

        required_cash = current_price * quantity
        if required_cash > available_usd:
            log(
                f"⚠️ [{symbol}] Required cash too high. Need~${required_cash:.5f}, available=${available_usd:.5f}",
                to_file=True
            )
            return None
        log(
            f"📐 [{symbol}] Sized order | qty={quantity} | target_risk=${risk_usd:.5f} | est_risk=${est_risk:.5f} "
            f"| req_cash~${required_cash:.5f} | avail=${available_usd:.5f}",
            to_file=True
        )

        opposite = 'sell'

        if DRY_RUN_MODE:
            trade_id = str(uuid4())
            entry_fill_price = _apply_slippage(current_price, 'buy', SIMULATED_SLIPPAGE_BPS)
            entry_notional = entry_fill_price * quantity
            entry_fee = _estimate_fee(entry_notional, SIMULATED_FEE_BPS)
            dry_run_payload = {
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "entry_type": "simulated_market",
                "entry_price": float(entry_fill_price),
                "entry_reference_price": float(current_price),
                "quantity": float(quantity),
                "stop_price": float(stop_price),
                "stop_side": opposite,
                "take_profit_price": float(tp_price),
                "take_profit_side": opposite,
                "risk_usd_target": float(risk_usd),
                "estimated_stop_loss_usd": float(est_risk),
                "required_cash_estimate_usd": float(required_cash),
                "available_usd_at_signal": float(available_usd),
                "simulated_fee_bps": float(SIMULATED_FEE_BPS),
                "simulated_slippage_bps": float(SIMULATED_SLIPPAGE_BPS),
                "entry_fee_usd": float(entry_fee),
                "score": float(score),
                "choch_level": float(choch_level),
                "fvg_extreme": float(fvg_extreme),
            }
            log(
                f"🧪 DRY RUN | WOULD CREATE ORDER [{symbol}] {side.upper()} "
                f"| qty={quantity} | entry~{entry_fill_price:.4f} | SL={stop_price:.4f} | TP={tp_price:.4f}",
                to_file=True
            )

            active_trades[symbol] = {
                'trade_id': trade_id,
                'side': side,
                'entry_price': float(current_price),
                'stop_price': float(stop_price),
                'tp_price': float(tp_price),
                'quantity': float(quantity),
                'open_time': datetime.now(),
                'choch_level': float(choch_level),
                'fvg_extreme': float(fvg_extreme),
                'breakeven_moved': False,
                'entry_order_id': f"dryrun-entry-{uuid4()}",
                'sl_order_id': f"dryrun-sl-{uuid4()}",
                'tp_order_id': f"dryrun-tp-{uuid4()}",
                'entry_fill_price': float(entry_fill_price),
                'entry_filled': float(quantity),
                'simulated': True,
                'required_margin_estimate_usd': float(required_cash),
                'entry_fee_usd': float(entry_fee),
            }
            log_trade_audit("dry_run_trade_signal", dry_run_payload)
            return {"id": f"dryrun-{uuid4()}", "dry_run": True, "info": dry_run_payload}

        order = exchange.create_order(symbol=symbol, type='market', side='buy', amount=quantity)
        log(f"✅ ENTRY ORDER CREATED | ID: {order.get('id')}", to_file=True)

        trade_id = str(uuid4())
        entry_fill_price = _safe_float(order.get('average'), float(current_price))
        entry_filled_qty = _safe_float(order.get('filled'), float(quantity))
        entry_fee = _estimate_fee(entry_fill_price * entry_filled_qty, SIMULATED_FEE_BPS)
        active_trades[symbol] = {
            'trade_id': trade_id,
            'side': 'buy',
            'entry_price': float(current_price),
            'stop_price': float(stop_price),
            'tp_price': float(tp_price),
            'quantity': float(entry_filled_qty),
            'open_time': datetime.now(),
            'choch_level': float(choch_level),
            'fvg_extreme': float(fvg_extreme),
            'breakeven_moved': False,
            'entry_order_id': order.get('id'),
            'sl_order_id': None,
            'tp_order_id': None,
            'entry_fill_price': float(entry_fill_price),
            'entry_filled': float(entry_filled_qty),
            'entry_fee_usd': float(entry_fee),
        }
        log_trade_audit("trade_opened", {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "score": float(score),
            "entry_price": float(current_price),
            "entry_fill_price": float(entry_fill_price),
            "stop_price": float(stop_price),
            "tp_price": float(tp_price),
            "quantity": float(entry_filled_qty),
            "risk_usd": float(risk_usd),
            "estimated_stop_loss_usd": float(est_risk),
            "required_cash_estimate_usd": float(required_cash),
            "available_usd_at_entry": float(available_usd),
            "entry_fee_usd": float(entry_fee),
            "order_ids": {
                "entry_order_id": order.get('id'),
                "sl_order_id": None,
                "tp_order_id": None,
            },
        })
        log(f"=== SPOT POSITION OPENED SUCCESSFULLY (1m CHOCH+FVG) ===", to_file=True)
        return order
    except Exception as e:
        if "insufficientAvailableFunds" in str(e):
            log(f"⚠️ insufficientAvailableFunds - skipping trade (balance too low for this size)", to_file=True)
        else:
            log(f"❌ ORDER FAILED: {e}", to_file=True)
        return None

# (manage_open_trade, main loop, etc. are the same as the previous working version)

async def manage_open_trade(symbol, df):
    global daily_pnl, daily_trades, total_wins, total_losses, total_pnl, total_win_usd, total_loss_usd, simulated_wallet_usd
    if symbol not in active_trades:
        return
    trade = active_trades[symbol]
    current_price = df['close'].iloc[-1]
    fvg_extreme = trade['fvg_extreme']
    latest_high = _safe_float(df['high'].iloc[-1], 0.0)
    latest_low = _safe_float(df['low'].iloc[-1], 0.0)
    both_touched = (
        latest_low <= trade['stop_price'] and latest_high >= trade['tp_price']
        if trade['side'] == 'buy'
        else latest_high >= trade['stop_price'] and latest_low <= trade['tp_price']
    )

    if both_touched:
        mode_tag = "SIM" if trade.get('simulated') else "LIVE"
        log(
            f"⚖️ [{symbol}] {mode_tag} candle touched both SL and TP; no action this cycle.",
            to_file=True
        )
        return

    if not trade.get('breakeven_moved', False):
        if (trade['side'] == 'buy' and current_price > fvg_extreme) or \
           (trade['side'] == 'sell' and current_price < fvg_extreme):
            try:
                new_sl = trade['entry_price']
                trade['stop_price'] = new_sl
                trade['breakeven_moved'] = True
                trade['sl_order_id'] = f"{'dryrun' if trade.get('simulated') else 'live'}-sl-be-{uuid4()}"
                log_trade_audit("breakeven_moved", {
                    "trade_id": trade.get('trade_id'),
                    "symbol": symbol,
                    "new_stop_price": float(new_sl),
                    "new_sl_order_id": trade.get('sl_order_id'),
                    "side": trade.get('side'),
                })
                log(f"🔄 [{symbol}] BREAKEVEN TRIGGERED (BOS of FVG candle)", to_file=True)
            except Exception as e:
                log(f"⚠️ Breakeven failed: {e}", to_file=True)

    close_reason = None
    trigger_price = None
    if latest_low <= trade['stop_price']:
        close_reason = "SL_HIT_SIMULATED" if trade.get('simulated') else "SL_HIT"
        trigger_price = _safe_float(trade['stop_price'], 0.0)
    elif latest_high >= trade['tp_price']:
        close_reason = "TP_HIT_SIMULATED" if trade.get('simulated') else "TP_HIT"
        trigger_price = _safe_float(trade['tp_price'], 0.0)

    if not close_reason or trigger_price <= 0:
        return

    exit_qty = _safe_float(trade.get('quantity'), 0.0)
    if exit_qty <= 0:
        return

    if trade.get('simulated'):
        exit_price = _apply_slippage(trigger_price, 'sell', SIMULATED_SLIPPAGE_BPS)
        exit_fee = _estimate_fee(exit_price * exit_qty, SIMULATED_FEE_BPS)
        entry_price = _safe_float(trade.get('entry_fill_price') or trade.get('entry_price'))
        entry_fee = _safe_float(trade.get('entry_fee_usd'), 0.0)
        pnl = ((exit_price - entry_price) * exit_qty) - entry_fee - exit_fee
        fee_total = entry_fee + exit_fee
        simulated_wallet_usd = max(simulated_wallet_usd + pnl, 0.0)
        refresh_live_balance()
    else:
        try:
            exit_order = exchange.create_order(symbol=symbol, type='market', side='sell', amount=exit_qty)
        except Exception as e:
            log(f"⚠️ [{symbol}] Exit order failed ({close_reason}): {e}", to_file=True)
            return
        exit_price = _safe_float(exit_order.get('average'), trigger_price)
        filled_qty = _safe_float(exit_order.get('filled'), exit_qty) or exit_qty
        entry_price = _safe_float(trade.get('entry_fill_price') or trade.get('entry_price'))
        entry_fee = _safe_float(trade.get('entry_fee_usd'), 0.0)
        exit_fee = _extract_fee_cost(exit_order)
        if exit_fee == 0:
            exit_fee = _estimate_fee(exit_price * filled_qty, SIMULATED_FEE_BPS)
        fee_total = entry_fee + exit_fee
        pnl = ((exit_price - entry_price) * filled_qty) - fee_total
        trade['tp_order_id'] = exit_order.get('id')

    daily_pnl += pnl
    total_pnl += pnl
    if pnl >= 0:
        total_wins += 1
        total_win_usd += abs(pnl)
        result = "WIN"
    else:
        total_losses += 1
        total_loss_usd += abs(pnl)
        result = "LOSS"
    log_trade_audit("trade_closed", {
        "trade_id": trade.get('trade_id'),
        "symbol": symbol,
        "side": trade.get('side'),
        "result": result,
        "close_reason": close_reason,
        "entry_price": _safe_float(trade.get('entry_fill_price') or trade.get('entry_price')),
        "exit_price": float(exit_price),
        "quantity": _safe_float(trade.get('quantity')),
        "fees_usd": float(fee_total),
        "realized_pnl_usd": float(pnl),
        "order_ids": {
            "entry_order_id": trade.get('entry_order_id'),
            "sl_order_id": trade.get('sl_order_id'),
            "tp_order_id": trade.get('tp_order_id'),
        },
        "duration_seconds": max((datetime.now() - trade.get('open_time', datetime.now())).total_seconds(), 0.0),
        "simulated": bool(trade.get('simulated')),
    })
    del active_trades[symbol]
    log_prefix = "🧪" if trade.get('simulated') else "✅"
    log(
        f"   {log_prefix} [{symbol}] {trade['side'].upper()} {result} CLOSED ({close_reason}) | Exit @{exit_price:.4f} | Fees ${fee_total:.4f} | PnL ${pnl:.4f}",
        to_file=True
    )

async def main():
    global daily_trades, daily_pnl, total_wins, total_losses, total_pnl, total_win_usd, total_loss_usd, current_day, log_file, audit_file, last_trade_time, RISK_USD, latest_diagnostics
    log("🚀 Kraken ICT Bot Started - 1m CHOCH (90min pattern) + FVG + improved tracking", to_file=True)
    log(f"🔐 Mode: {MODE_LABEL} | Execution: {EXECUTION_LABEL}", to_file=True)
    if DRY_RUN_MODE:
        log(f"🧪 Dry-run simulated wallet start: ${simulated_wallet_usd:.5f}", to_file=True)

    warmup_end = datetime.now() + timedelta(minutes=WARMUP_MINUTES)
    in_warmup = WARMUP_MINUTES > 0
    cooldown_minutes = 1440 / MAX_TRADES_PER_DAY
    wiggle_minutes = cooldown_minutes * 0.25
    last_diag_minute = None

    while True:
        try:
            balance_snapshot = refresh_live_balance()
            account_balance = _safe_float(balance_snapshot.get("available_usd"), 0.0)
            RISK_USD = account_balance * (RISK_PERCENT / 100.0)
            if RISK_USD <= 0:
                log("⚠️ No available USD balance yet; waiting...", to_file=False, to_recent=False)
                await asyncio.sleep(20)
                continue
            log(
                f"💰 Balance check | Wallet=${balance_snapshot.get('wallet_usd', 0.0):.5f} "
                f"| Available=${balance_snapshot.get('available_usd', 0.0):.5f} "
                f"| Risk/Trade={RISK_PERCENT:.3f}% => ${RISK_USD:.5f}",
                to_file=False,
                to_recent=False
            )

            today = date.today()
            if today != current_day:
                log_file = f"trading_log_{today}.txt"
                audit_file = f"trade_audit_{today}.jsonl"
                daily_trades = 0
                daily_pnl = 0.0
                current_day = today
                daily_setups.clear()
                last_trade_time = None
                symbol_precheck_failures.clear()
                disabled_symbols.clear()

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
            diagnostics = []

            for symbol in SYMBOLS:
                if symbol in disabled_symbols:
                    diagnostics.append(f"{symbol}:DISABLED({disabled_symbols.get(symbol)})")
                    continue
                df1m = await fetch_latest_candles(symbol, TIMEFRAME)
                if len(df1m) < 100:
                    diagnostics.append(f"{symbol}:candles<{100}")
                    continue

                choch_1m = detect_choch(df1m)
                fvg_1m = detect_fvg(df1m)
                current_price = df1m['close'].iloc[-1]
                latest_choch_idx = choch_1m[-1][1] if choch_1m else -1
                considered_fvgs = 0
                confirmed_fvgs = 0

                for fvg in fvg_1m[-20:]:
                    fvg_type = fvg['type']
                    midpoint = fvg['midpoint']
                    fvg_extreme = fvg['extreme']
                    fvg_idx = fvg['idx']
                    fvg_candle_idx = fvg['candle_idx']

                    if latest_choch_idx != -1 and fvg_idx < latest_choch_idx:
                        continue

                    considered_fvgs += 1
                    confirmation_window = df1m.iloc[-12:-2]
                    if fvg_type == 'bullish':
                        confirmed = (confirmation_window['close'] > midpoint).mean() >= 0.7
                    else:
                        confirmed = (confirmation_window['close'] < midpoint).mean() >= 0.7
                    if not confirmed:
                        continue

                    confirmed_fvgs += 1
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

                diagnostics.append(
                    f"{symbol}:choch={len(choch_1m)} fvg={len(fvg_1m)} considered={considered_fvgs} confirmed={confirmed_fvgs}"
                )
                await manage_open_trade(symbol, df1m)

            now_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
            if now_minute != last_diag_minute:
                last_diag_minute = now_minute
                diag_line = " | ".join(diagnostics) if diagnostics else "no symbols processed"
                best_line = (
                    f"best={best_setup['symbol']}:{best_setup['direction']} score={best_setup['score']:.1f}"
                    if best_setup else "best=None"
                )
                latest_diagnostics = f"{now_minute} | {best_line} | {diag_line}"
                log(f"📊 1m diagnostics | {best_line} | {diag_line}", to_file=True, to_recent=False)

            if best_setup and daily_trades < MAX_TRADES_PER_DAY and best_setup['symbol'] not in active_trades:
                s = best_setup
                buffer = 0.0005 * s['current_price']
                created_order = None
                if s['direction'] == 'bullish':
                    stop_price = s['fvg_extreme'] - buffer
                    tp_price = s['current_price'] + (s['current_price'] - stop_price) * RR_RATIO
                    precheck = precheck_symbol_tradeability(s['symbol'], s['current_price'], stop_price, RISK_USD)
                    if precheck.get("ok"):
                        symbol_precheck_failures.pop(s['symbol'], None)
                        created_order = await place_trade(s['symbol'], 'buy', s['current_price'], stop_price, tp_price, RISK_USD, s['score'], s['choch_level'], s['fvg_extreme'])
                    else:
                        fail_count = symbol_precheck_failures.get(s['symbol'], 0) + 1
                        symbol_precheck_failures[s['symbol']] = fail_count
                        log(
                            f"🧪 Precheck fail [{s['symbol']}] {precheck.get('reason')} "
                            f"| min_lot_risk=${precheck.get('min_lot_risk', 0.0):.5f} "
                            f"| min_lot_cost=${precheck.get('min_lot_cost', 0.0):.5f} "
                            f"| available=${precheck.get('available_usd', 0.0):.5f} "
                            f"| fail_count={fail_count}/{PRECHECK_DISABLE_AFTER}",
                            to_file=True
                        )
                        if fail_count >= PRECHECK_DISABLE_AFTER:
                            disabled_symbols[s['symbol']] = precheck.get('reason')
                            log(f"🚫 Disabling {s['symbol']} from tracking (precheck failures: {precheck.get('reason')})", to_file=True)
                else:
                    log(f"ℹ️ [{s['symbol']}] Bearish setup ignored in spot mode (long-only)", to_file=False, to_recent=False)
                if created_order:
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
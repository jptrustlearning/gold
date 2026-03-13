#!/usr/bin/env python3
"""
Gold Momentum Backtest — Daily Edition
JP Trust Learning

Loops through each trading day (1 year by default), computes:
  1. Buy Score (same logic as gold_momentum_v2.py)
  2. Sell Score (same logic as gold_momentum_sell.py)
  3. Net Bias + Scenario (same logic as gold_momentum_net.py)

Output: 3 CSV files with 1 row per day
  - backtest_daily_buy.csv
  - backtest_daily_sell.csv
  - backtest_daily_net.csv

Usage:
  python3 gold_backtest_daily.py              # backtest 1 year (~252 days)
  python3 gold_backtest_daily.py --days 3     # backtest last 3 days (test)
  python3 gold_backtest_daily.py --days 60    # backtest last 60 days
"""

import pandas as pd
import numpy as np
import os, sys
from datetime import datetime, timezone

# ── CONFIG ──
ROLLING_WINDOW = 252
LOOKBACK = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
WEIGHTS = {'1Y': 0.30, '6M': 0.25, '3M': 0.20, '1M': 0.15, '1W': 0.10}
WEIGHT_ORDER = ['1Y', '6M', '3M', '1M', '1W']
BD_GAP = 5  # BD1 = BD2 - 5 days (same as daily pipeline)

RUN_TS = datetime.now(timezone.utc)
AS_OF = RUN_TS.strftime("%d/%m/%Y %H:%M UTC")
TS_FILE = RUN_TS.strftime("%d%m%Y_%H%M")

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Parse args ──
backtest_days = 252  # default 1 year
for i, arg in enumerate(sys.argv):
    if arg == '--days' and i + 1 < len(sys.argv):
        backtest_days = int(sys.argv[i + 1])

# ── LOAD DATA ──
def load_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found")
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_csv('gold_prices.csv')
df_dxy = load_csv('dxy_prices.csv')
df_vix = load_csv('vix_prices.csv')

if df is None:
    print("❌ gold_prices.csv not found"); sys.exit(1)

print(f"Gold Momentum Backtest — Daily Edition")
print(f"{'='*55}")
print(f"Data: {len(df)} rows ({df['Date'].min().date()} → {df['Date'].max().date()})")
print(f"Backtest: last {backtest_days} trading days")
print(f"DXY: {'✅' if df_dxy is not None else '❌'} | VIX: {'✅' if df_vix is not None else '❌'}")

# ══════════════════════════════════════════════════════
# SHARED FUNCTIONS (from gold_momentum_v2.py)
# ══════════════════════════════════════════════════════

def compute_return(closes, end_idx, period_days):
    start_idx = end_idx - period_days
    if start_idx < 0: return None
    return (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100

def rolling_percentile(series_values, current_val):
    valid = series_values[~np.isnan(series_values)]
    if len(valid) < 10: return 50.0
    count_below = np.sum(valid < current_val)
    return count_below / (len(valid) - 1) * 100 if len(valid) > 1 else 50.0

def calc_return_percentiles(df, base_idx):
    closes = df['Close'].values
    results = {}
    for period, days in LOOKBACK.items():
        current_ret = compute_return(closes, base_idx, days)
        if current_ret is None:
            results[period] = {'return': 0, 'percentile': 50}
            continue
        rolling_rets = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            r = compute_return(closes, i, days)
            if r is not None: rolling_rets.append(r)
        pctl = rolling_percentile(np.array(rolling_rets), current_ret) if rolling_rets else 50
        results[period] = {'return': current_ret, 'percentile': pctl}
    return results

def calc_volume_percentiles(df, base_idx):
    volumes = df['Volume'].values
    results = {}
    for period, days in LOOKBACK.items():
        end = base_idx + 1
        start = end - days
        if start < 0:
            results[period] = {'volume': 0, 'percentile': 50}
            continue
        current_vol = np.sum(volumes[start:end])
        rolling_vols = []
        roll_start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(roll_start, base_idx):
            s = i + 1 - days
            if s < 0: continue
            rolling_vols.append(np.sum(volumes[s:i+1]))
        pctl = rolling_percentile(np.array(rolling_vols), current_vol) if rolling_vols else 50
        results[period] = {'volume': current_vol, 'percentile': pctl}
    return results

def weighted_percentile(pctl_dict):
    return sum(pctl_dict[p]['percentile'] * WEIGHTS[p] for p in WEIGHT_ORDER)

def calc_rsi(df, base_idx, period=14):
    start = base_idx - 29
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2: return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    last_n = min(period, len(gains))
    avg_gain = np.mean(gains[-last_n:])
    avg_loss = np.mean(losses[-last_n:])
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_ma(df, base_idx, window):
    start = base_idx + 1 - window
    if start < 0: return None
    return np.mean(df['Close'].values[start:base_idx+1])

def calc_volatility(df, base_idx):
    start = base_idx - 20
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2: return 0
    rets = np.diff(closes) / closes[:-1]
    return np.std(rets) * np.sqrt(252) * 100

def find_closest_idx(ext_df, target_date, max_gap_days=5):
    """Find closest date ON or BEFORE target_date (no look-ahead bias)."""
    if ext_df is None: return None
    mask = ext_df['Date'] <= target_date
    if not mask.any(): return None
    past = ext_df.loc[mask]
    diff = (target_date - past['Date']).dt.days
    if diff.min() > max_gap_days: return None
    return diff.idxmin()

def calc_ext_return(ext_df, end_idx, period_days):
    if ext_df is None or end_idx is None: return None
    start_idx = end_idx - period_days
    if start_idx < 0: return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[start_idx]) / ext_df['Close'].values[start_idx] * 100

# ══════════════════════════════════════════════════════
# BUY SCORING
# ══════════════════════════════════════════════════════

def d3_buy(rsi):
    if 50 <= rsi <= 70: return 100
    if 40 <= rsi < 50: return 80
    if 70 < rsi <= 80: return 70
    if 30 <= rsi < 40: return 60
    if rsi > 80: return 50
    return 30

def d4_buy(price, ma50, ma200):
    pts = 0
    if ma50 is not None and price > ma50: pts += 35
    if ma200 is not None and price > ma200: pts += 35
    if ma50 is not None and ma200 is not None and ma50 > ma200: pts += 30
    return min(pts, 100)

def d5_buy(vol):
    if vol <= 20: return 100
    if vol <= 30: return 90
    if vol <= 40: return 70
    if vol <= 50: return 55
    if vol <= 60: return 40
    if vol <= 80: return 25
    return 10

def calc_d6_buy(df_gold, idx, df_dxy, df_vix):
    gold_date = df_gold.iloc[idx]['Date']
    gold_1m = compute_return(df_gold['Close'].values, idx, 21) or 0
    gold_up = gold_1m >= 0
    dxy_score, vix_score = 0, 0
    dxy_1m, vix_level = None, None

    if df_dxy is not None:
        dxy_idx = find_closest_idx(df_dxy, gold_date)
        if dxy_idx is not None:
            dxy_1m = calc_ext_return(df_dxy, dxy_idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if gold_up and dxy_up: dxy_score = +5
                elif gold_up and not dxy_up: dxy_score = +2
                elif not gold_up and not dxy_up: dxy_score = 0
                else: dxy_score = -5

    if df_vix is not None:
        vix_idx = find_closest_idx(df_vix, gold_date)
        if vix_idx is not None:
            vix_level = df_vix['Close'].values[vix_idx]
            if gold_up:
                if vix_level > 30: vix_score = +5
                elif vix_level >= 20: vix_score = +3
                else: vix_score = +1
            else:
                if vix_level > 30: vix_score = -3
                elif vix_level >= 20: vix_score = -2
                else: vix_score = 0

    total = max(min(dxy_score + vix_score, 10), -10)
    return {'d6_total': total, 'd6_scaled': (total + 10) / 20 * 100,
            'dxy_1m': dxy_1m, 'vix_level': vix_level}

def calc_buy_penalties(df, idx):
    closes = df['Close'].values
    r1y = compute_return(closes, idx, 252) or 0
    r6m = compute_return(closes, idx, 126) or 0
    r1m = compute_return(closes, idx, 21) or 0
    r1w = compute_return(closes, idx, 5) or 0
    rev = 0; flags = ""
    if r1y > 20 and r1m < -5 and r1w < -3: rev = -10; flags = "Strong Reversal"
    elif (r1y > 0 or r6m > 0) and r1m < 0 and r1w < 0: rev = -5; flags = "Mild Reversal"
    ma50 = calc_ma(df, idx, 50); ma200 = calc_ma(df, idx, 200)
    dc = 0
    if ma50 is not None and ma200 is not None and ma50 < ma200: dc = -5; flags += (" | " if flags else "") + "Death Cross"
    return {'total': max(rev + dc, -15), 'reversal': rev, 'death_cross': dc, 'flags': flags,
            'ret_1y': r1y, 'ret_6m': r6m, 'ret_1m': r1m, 'ret_1w': r1w}

def full_buy_score(df, idx, df_dxy, df_vix):
    ret_p = calc_return_percentiles(df, idx)
    vol_p = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_p)
    wp_vol = weighted_percentile(vol_p)
    d1 = wp_ret; d2 = wp_vol
    rsi = calc_rsi(df, idx); d3 = d3_buy(rsi)
    price = df['Close'].values[idx]
    ma50 = calc_ma(df, idx, 50); ma200 = calc_ma(df, idx, 200)
    d4 = d4_buy(price, ma50, ma200)
    vol = calc_volatility(df, idx); d5 = d5_buy(vol)
    ext = calc_d6_buy(df, idx, df_dxy, df_vix); d6 = ext['d6_scaled']
    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    pen = calc_buy_penalties(df, idx)
    pen_sc = pen['total'] * (100 / 110)
    net = gross + pen_sc
    gc = ma50 is not None and ma200 is not None and ma50 > ma200
    return {'price': price, 'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
            'd6_raw': ext['d6_total'], 'rsi': rsi, 'ma50': ma50, 'ma200': ma200,
            'golden_cross': gc, 'volatility': vol, 'gross': gross, 'pen': pen,
            'pen_sc': pen_sc, 'net': net, 'wp_ret': wp_ret, 'wp_vol': wp_vol,
            'ret_p': ret_p, 'ext': ext}

# ══════════════════════════════════════════════════════
# SELL SCORING
# ══════════════════════════════════════════════════════

def d1_sell(wp): return 100 - wp

def d2_sell(wp_vol, ret_1m):
    if wp_vol >= 70 and ret_1m < 0: return 80 + (wp_vol - 70) / 30 * 20
    elif wp_vol >= 70: return 30 - (wp_vol - 70) / 30 * 10
    elif ret_1m < 0: return 40 + (70 - wp_vol) / 70 * 30
    return 10

def d3_sell(rsi):
    if rsi < 30: return 100
    if rsi < 40: return 85
    if rsi < 50: return 65
    if rsi > 80: return 65
    if rsi > 70: return 50
    if rsi >= 60: return 20
    return 40

def d4_sell(price, ma50, ma200):
    pts = 0
    if ma50 is not None and price < ma50: pts += 35
    if ma200 is not None and price < ma200: pts += 35
    if ma50 is not None and ma200 is not None and ma50 < ma200: pts += 30
    return min(pts, 100)

def d5_sell(vol):
    if vol > 80: return 100
    if vol > 60: return 90
    if vol > 50: return 75
    if vol > 40: return 60
    if vol > 30: return 40
    if vol > 20: return 20
    return 5

def calc_d6_sell(df_gold, idx, df_dxy, df_vix):
    gold_date = df_gold.iloc[idx]['Date']
    gold_1m = compute_return(df_gold['Close'].values, idx, 21) or 0
    gold_down = gold_1m < 0
    dxy_score, vix_score = 0, 0
    dxy_1m, vix_level = None, None

    if df_dxy is not None:
        dxy_idx = find_closest_idx(df_dxy, gold_date)
        if dxy_idx is not None:
            dxy_1m = calc_ext_return(df_dxy, dxy_idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if gold_down and dxy_up: dxy_score = +5
                elif gold_down and not dxy_up: dxy_score = +2
                elif not gold_down and dxy_up: dxy_score = 0
                else: dxy_score = -5

    if df_vix is not None:
        vix_idx = find_closest_idx(df_vix, gold_date)
        if vix_idx is not None:
            vix_level = df_vix['Close'].values[vix_idx]
            if gold_down:
                if vix_level < 20: vix_score = +5
                elif vix_level <= 30: vix_score = +3
                else: vix_score = +1
            else:
                if vix_level < 20: vix_score = -3
                elif vix_level <= 30: vix_score = -2
                else: vix_score = 0

    total = max(min(dxy_score + vix_score, 10), -10)
    return {'d6_total': total, 'd6_scaled': (total + 10) / 20 * 100,
            'dxy_1m': dxy_1m, 'vix_level': vix_level}

def calc_sell_penalties(df, idx):
    closes = df['Close'].values
    r1y = compute_return(closes, idx, 252) or 0
    r6m = compute_return(closes, idx, 126) or 0
    r1m = compute_return(closes, idx, 21) or 0
    r1w = compute_return(closes, idx, 5) or 0
    rev = 0; flags = ""
    if r1y < -20 and r1m > 5 and r1w > 3: rev = -10; flags = "Strong Bullish Reversal"
    elif (r1y < 0 or r6m < 0) and r1m > 0 and r1w > 0: rev = -5; flags = "Mild Bullish Reversal"
    ma50 = calc_ma(df, idx, 50); ma200 = calc_ma(df, idx, 200)
    gc = 0
    if ma50 is not None and ma200 is not None and ma50 > ma200: gc = -5; flags += (" | " if flags else "") + "Golden Cross"
    return {'total': max(rev + gc, -15), 'reversal': rev, 'golden_cross': gc, 'flags': flags,
            'ret_1y': r1y, 'ret_6m': r6m, 'ret_1m': r1m, 'ret_1w': r1w}

def full_sell_score(df, idx, df_dxy, df_vix):
    ret_p = calc_return_percentiles(df, idx)
    vol_p = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_p)
    wp_vol = weighted_percentile(vol_p)
    d1 = d1_sell(wp_ret)
    ret_1m = compute_return(df['Close'].values, idx, 21) or 0
    d2 = d2_sell(wp_vol, ret_1m)
    rsi = calc_rsi(df, idx); d3 = d3_sell(rsi)
    price = df['Close'].values[idx]
    ma50 = calc_ma(df, idx, 50); ma200 = calc_ma(df, idx, 200)
    d4 = d4_sell(price, ma50, ma200)
    vol = calc_volatility(df, idx); d5 = d5_sell(vol)
    ext = calc_d6_sell(df, idx, df_dxy, df_vix); d6 = ext['d6_scaled']
    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    pen = calc_sell_penalties(df, idx)
    pen_sc = pen['total'] * (100 / 110)
    net = gross + pen_sc
    dc = ma50 is not None and ma200 is not None and ma50 < ma200
    return {'price': price, 'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
            'd6_raw': ext['d6_total'], 'rsi': rsi, 'ma50': ma50, 'ma200': ma200,
            'death_cross': dc, 'volatility': vol, 'gross': gross, 'pen': pen,
            'pen_sc': pen_sc, 'net': net, 'wp_ret': wp_ret, 'wp_vol': wp_vol,
            'ret_p': ret_p, 'ext': ext}

# ══════════════════════════════════════════════════════
# TIER + SCENARIO (from net script)
# ══════════════════════════════════════════════════════

def tier_buy(s):
    s = max(0, min(100, s))
    if s >= 85: return "Very Strong ↑↑"
    if s >= 75: return "Strong ↑"
    if s >= 60: return "Moderate ↑"
    if s >= 45: return "Neutral →"
    if s >= 30: return "Weak ↓"
    return "Very Weak ↓↓"

def tier_sell(s):
    s = max(0, min(100, s))
    if s >= 85: return "Very Strong Sell ↓↓"
    if s >= 75: return "Strong Sell ↓"
    if s >= 60: return "Moderate Sell ↓"
    if s >= 45: return "Neutral →"
    if s >= 30: return "Weak Sell"
    return "No Sell Signal"

def net_bias_tier(nb):
    if nb >= 50: return "Strong Bullish ↑↑"
    if nb >= 25: return "Bullish ↑"
    if nb >= 10: return "Lean Bullish ↗"
    if nb >= -10: return "Neutral →"
    if nb >= -25: return "Lean Bearish ↘"
    if nb >= -50: return "Bearish ↓"
    return "Strong Bearish ↓↓"

def calc_pivot_pp(df, idx):
    """D1 pivot from current day's H/L/C (matches pipeline calc_multi_tf_pivots)."""
    row = df.iloc[idx]
    h, l, c = row['High'], row['Low'], row['Close']
    pp = (h + l + c) / 3
    r1 = 2 * pp - l; r2 = pp + (h - l)
    s1 = 2 * pp - h; s2 = pp - (h - l)
    return {'pp': pp, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2}

def assign_scenario(buy_sc, sell_sc, net_b, buy_d, sell_d, price, pv):
    pp = pv.get('pp', 0); r1 = pv.get('r1', 0); s1 = pv.get('s1', 0)
    above_r1 = price >= r1 if r1 else False
    above_pp = price >= pp if pp else False
    below_s1 = price < s1 if s1 else False
    between_pp_r1 = (pp <= price < r1) if (pp and r1) else False
    between_s1_pp = (s1 <= price < pp) if (s1 and pp) else False

    if buy_sc >= 60 and sell_sc < 45 and net_b >= 20:
        if above_r1 and buy_sc >= 75 and buy_d > 0: return 1, 'Bullish Breakout', 'BUY', 'BULLISH'
        if above_r1 and buy_sc >= 75: return 2, 'Bullish but Cooling', 'HOLD', 'BULLISH'
        if between_pp_r1: return 3, 'Bullish Accumulation', 'BUY', 'BULLISH'
        if between_s1_pp: return 4, 'Support Bounce', 'BUY', 'BULLISH'
        if below_s1: return 5, 'Oversold Recovery', 'BUY', 'BULLISH'
        return 3, 'Bullish Accumulation', 'BUY', 'BULLISH'

    if sell_sc >= 60 and buy_sc < 45 and net_b <= -20:
        if below_s1 and sell_sc >= 75 and sell_d > 0: return 6, 'Bearish Breakdown', 'SELL', 'BEARISH'
        if below_s1: return 7, 'Bearish Continuation', 'SELL', 'BEARISH'
        if between_s1_pp: return 8, 'Bearish Distribution', 'SELL', 'BEARISH'
        if above_pp: return 9, 'Resistance Rejection', 'SELL', 'BEARISH'
        return 7, 'Bearish Continuation', 'SELL', 'BEARISH'

    if buy_sc >= 55 and sell_sc >= 55 and abs(net_b) < 20:
        if above_r1: return 10, 'Volatile Breakout', 'HOLD', 'CONFLICT'
        return 11, 'Tug of War', 'HOLD', 'CONFLICT'

    if buy_sc < 55 and sell_sc < 55:
        return 12, 'Dead Zone', 'HOLD', 'NEUTRAL'

    if buy_sc >= 55 and sell_sc < 55 and net_b > 0:
        return 13, 'Cautious Bullish', 'HOLD', 'LEAN BULLISH'
    if sell_sc >= 55 and buy_sc < 55 and net_b < 0:
        return 14, 'Cautious Bearish', 'HOLD', 'LEAN BEARISH'

    return 12, 'Monitoring', 'HOLD', 'NEUTRAL'

COMBINED_MAP = {1:'Full Bullish Confirmed',2:'Full Bullish Confirmed',
    3:'Bullish Dominant',4:'Bullish Dominant',5:'Bullish Dominant',
    6:'Full Bearish Confirmed',7:'Bearish Dominant',8:'Bearish Dominant',9:'Bearish Dominant',
    10:'Tug of War',11:'Tug of War',12:'Dead Zone',13:'Lean Bullish',14:'Lean Bearish'}

# ══════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════

# Need minimum data: 252 (rolling) + 252 (1Y lookback) + BD_GAP
min_idx = 252 + 252 + BD_GAP
end_idx = len(df) - 1
start_idx = max(min_idx, end_idx - backtest_days + 1)

print(f"Backtest range: idx {start_idx} → {end_idx} ({end_idx - start_idx + 1} days)")
print(f"Date range: {df.iloc[start_idx]['Date'].date()} → {df.iloc[end_idx]['Date'].date()}")
print(f"{'='*55}")

buy_rows = []
sell_rows = []
net_rows = []

for bd2_idx in range(start_idx, end_idx + 1):
    bd1_idx = bd2_idx - BD_GAP
    if bd1_idx < min_idx - BD_GAP:
        continue

    date_str = df.iloc[bd2_idx]['Date'].strftime('%Y-%m-%d')

    # Buy scores for BD1 and BD2
    b1 = full_buy_score(df, bd1_idx, df_dxy, df_vix)
    b2 = full_buy_score(df, bd2_idx, df_dxy, df_vix)
    buy_net_avg = (b1['net'] + b2['net']) / 2
    buy_gross_avg = (b1['gross'] + b2['gross']) / 2
    buy_delta = b2['net'] - b1['net']

    # Sell scores for BD1 and BD2
    s1_sc = full_sell_score(df, bd1_idx, df_dxy, df_vix)
    s2_sc = full_sell_score(df, bd2_idx, df_dxy, df_vix)
    sell_net_avg = (s1_sc['net'] + s2_sc['net']) / 2
    sell_gross_avg = (s1_sc['gross'] + s2_sc['gross']) / 2
    sell_delta = s2_sc['net'] - s1_sc['net']

    # Net Bias
    net_bias = buy_net_avg - sell_net_avg
    bias_t = net_bias_tier(net_bias)

    # Pivot + Scenario
    pv = calc_pivot_pp(df, bd2_idx)
    sc_num, sc_signal, sc_action, sc_zone = assign_scenario(
        buy_net_avg, sell_net_avg, net_bias, buy_delta, sell_delta,
        b2['price'], pv)
    combined = COMBINED_MAP.get(sc_num, 'Mixed Signal')

    # Buy CSV row
    buy_rows.append({
        'Date': date_str, 'Price': round(b2['price'], 2),
        'Net_Score_Avg': round(buy_net_avg, 2), 'Gross_Score_Avg': round(buy_gross_avg, 2),
        'Net_Score_BD1': round(b1['net'], 2), 'Net_Score_BD2': round(b2['net'], 2),
        'Score_Delta': round(buy_delta, 2), 'Tier': tier_buy(buy_net_avg),
        'D1_ReturnRank': round(b2['d1'], 2), 'D2_VolumeRank': round(b2['d2'], 2),
        'D3_RSI': round(b2['d3'], 2), 'D4_MA': round(b2['d4'], 2),
        'D5_Volatility': round(b2['d5'], 2), 'D6_External': round(b2['d6'], 2),
        'D6_Raw': b2['d6_raw'], 'Penalty_Scaled': round(b2['pen_sc'], 2),
        'WP_Return_Pct': round(b2['wp_ret'], 2), 'WP_Volume_Pct': round(b2['wp_vol'], 2),
        'RSI_Value': round(b2['rsi'], 2),
        'MA50': round(b2['ma50'], 2) if b2['ma50'] else '',
        'MA200': round(b2['ma200'], 2) if b2['ma200'] else '',
        'Golden_Cross': str(b2['golden_cross']),
        'Volatility_Pct': round(b2['volatility'], 2),
        'Penalty_Total': b2['pen']['total'],
        'Warning_Flags': b2['pen']['flags'] or 'None',
    })

    # Sell CSV row
    sell_rows.append({
        'Date': date_str, 'Price': round(s2_sc['price'], 2),
        'Net_Score_Avg': round(sell_net_avg, 2), 'Gross_Score_Avg': round(sell_gross_avg, 2),
        'Net_Score_BD1': round(s1_sc['net'], 2), 'Net_Score_BD2': round(s2_sc['net'], 2),
        'Score_Delta': round(sell_delta, 2), 'Tier': tier_sell(sell_net_avg),
        'D1_ReturnRank': round(s2_sc['d1'], 2), 'D2_VolumeRank': round(s2_sc['d2'], 2),
        'D3_RSI': round(s2_sc['d3'], 2), 'D4_MA': round(s2_sc['d4'], 2),
        'D5_Volatility': round(s2_sc['d5'], 2), 'D6_External': round(s2_sc['d6'], 2),
        'D6_Raw': s2_sc['d6_raw'], 'Penalty_Scaled': round(s2_sc['pen_sc'], 2),
        'WP_Return_Pct': round(s2_sc['wp_ret'], 2), 'WP_Volume_Pct': round(s2_sc['wp_vol'], 2),
        'RSI_Value': round(s2_sc['rsi'], 2),
        'MA50': round(s2_sc['ma50'], 2) if s2_sc['ma50'] else '',
        'MA200': round(s2_sc['ma200'], 2) if s2_sc['ma200'] else '',
        'Death_Cross': str(s2_sc['death_cross']),
        'Volatility_Pct': round(s2_sc['volatility'], 2),
        'Penalty_Total': s2_sc['pen']['total'],
        'Warning_Flags': s2_sc['pen']['flags'] or 'None',
    })

    # Net CSV row
    net_rows.append({
        'Date': date_str, 'Price': round(b2['price'], 2),
        'Buy_Score': round(buy_net_avg, 2), 'Sell_Score': round(sell_net_avg, 2),
        'Net_Bias': round(net_bias, 2), 'Bias_Tier': bias_t,
        'Combined_Signal': combined,
        'Buy_Tier': tier_buy(buy_net_avg), 'Sell_Tier': tier_sell(sell_net_avg),
        'Buy_Delta': round(buy_delta, 2), 'Sell_Delta': round(sell_delta, 2),
        'Scenario_Num': sc_num, 'Scenario_Signal': sc_signal,
        'Scenario_Action': sc_action, 'Scenario_Zone': sc_zone,
        'Buy_D1': round(b2['d1'], 2), 'Sell_D1': round(s2_sc['d1'], 2),
        'Buy_D2': round(b2['d2'], 2), 'Sell_D2': round(s2_sc['d2'], 2),
        'Buy_D3': round(b2['d3'], 2), 'Sell_D3': round(s2_sc['d3'], 2),
        'Buy_D4': round(b2['d4'], 2), 'Sell_D4': round(s2_sc['d4'], 2),
        'Buy_D5': round(b2['d5'], 2), 'Sell_D5': round(s2_sc['d5'], 2),
        'Buy_D6': round(b2['d6'], 2), 'Sell_D6': round(s2_sc['d6'], 2),
        'Pivot_PP': round(pv.get('pp', 0), 2),
        'Pivot_R1': round(pv.get('r1', 0), 2),
        'Pivot_S1': round(pv.get('s1', 0), 2),
    })

    # Progress
    done = bd2_idx - start_idx + 1
    total = end_idx - start_idx + 1
    if done % 50 == 0 or done == total:
        print(f"  [{done}/{total}] {date_str} | Buy={buy_net_avg:.1f} Sell={sell_net_avg:.1f} Net={net_bias:+.1f} → {sc_action} ({sc_signal})")

# ══════════════════════════════════════════════════════
# SAVE CSVs
# ══════════════════════════════════════════════════════

buy_df = pd.DataFrame(buy_rows)
sell_df = pd.DataFrame(sell_rows)
net_df = pd.DataFrame(net_rows)

buy_path = os.path.join(base_dir, 'backtest_daily_buy.csv')
sell_path = os.path.join(base_dir, 'backtest_daily_sell.csv')
net_path = os.path.join(base_dir, 'backtest_daily_net.csv')

buy_df.to_csv(buy_path, index=False, encoding='utf-8')
sell_df.to_csv(sell_path, index=False, encoding='utf-8')
net_df.to_csv(net_path, index=False, encoding='utf-8')

print(f"\n{'='*55}")
print(f"✅ Backtest Complete!")
print(f"{'='*55}")
print(f"📊 Buy:  {buy_path} ({len(buy_df)} rows)")
print(f"📊 Sell: {sell_path} ({len(sell_df)} rows)")
print(f"📊 Net:  {net_path} ({len(net_df)} rows)")
print(f"📅 Range: {buy_df['Date'].iloc[0]} → {buy_df['Date'].iloc[-1]}")

# Quick summary
if len(net_df) > 0:
    actions = net_df['Scenario_Action'].value_counts()
    print(f"\n📈 Action Distribution:")
    for action, count in actions.items():
        print(f"  {action}: {count} days ({count/len(net_df)*100:.1f}%)")
    print(f"\n📊 Net Bias Stats:")
    print(f"  Mean: {net_df['Net_Bias'].mean():+.2f}")
    print(f"  Min:  {net_df['Net_Bias'].min():+.2f}")
    print(f"  Max:  {net_df['Net_Bias'].max():+.2f}")

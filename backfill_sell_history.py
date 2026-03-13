#!/usr/bin/env python3
"""
Backfill score_history_sell.csv
Reads dates from score_history.csv (buy), runs sell scoring for each date,
builds sell history with exhaustion detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os, sys

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── Import sell scoring functions from sell script ──
# We'll copy the essential functions inline to avoid import issues

ROLLING_WINDOW = 252
LOOKBACK = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
WEIGHTS = {'1Y': 0.30, '6M': 0.25, '3M': 0.20, '1M': 0.15, '1W': 0.10}
WEIGHT_ORDER = ['1Y', '6M', '3M', '1M', '1W']

def load_price_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_price_csv('gold_prices.csv')
df_dxy = load_price_csv('dxy_prices.csv')
df_vix = load_price_csv('vix_prices.csv')

print(f"Gold: {len(df)} rows | DXY: {len(df_dxy) if df_dxy is not None else 0} | VIX: {len(df_vix) if df_vix is not None else 0}")

# ── Core functions (copied from sell script) ──
def compute_return(closes, end_idx, period):
    start_idx = end_idx - period
    if start_idx < 0: return None
    if closes[start_idx] == 0: return None
    return (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100

def calc_return_percentiles(df, base_idx):
    closes = df['Close'].values
    results = {}
    for period_name in WEIGHT_ORDER:
        days = LOOKBACK[period_name]
        current_ret = compute_return(closes, base_idx, days)
        if current_ret is None:
            results[period_name] = {'return': 0, 'percentile': 50}
            continue
        rolling_rets = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            r = compute_return(closes, i, days)
            if r is not None: rolling_rets.append(r)
        if len(rolling_rets) < 10:
            results[period_name] = {'return': current_ret, 'percentile': 50}
            continue
        count_below = sum(1 for r in rolling_rets if r < current_ret)
        pctl = count_below / len(rolling_rets) * 100
        results[period_name] = {'return': current_ret, 'percentile': pctl}
    return results

def calc_volume_percentiles(df, base_idx):
    results = {}
    for period_name in WEIGHT_ORDER:
        days = LOOKBACK[period_name]
        end = base_idx + 1
        start = end - days
        if start < 0:
            results[period_name] = {'volume': 0, 'percentile': 50}
            continue
        current_vol = df['Volume'].values[start:end].sum()
        rolling_vols = []
        roll_start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(roll_start, base_idx):
            s = i + 1 - days
            if s < 0: continue
            v = df['Volume'].values[s:i+1].sum()
            rolling_vols.append(v)
        if len(rolling_vols) < 10:
            results[period_name] = {'volume': current_vol, 'percentile': 50}
            continue
        count_below = sum(1 for v in rolling_vols if v < current_vol)
        pctl = count_below / len(rolling_vols) * 100
        results[period_name] = {'volume': current_vol, 'percentile': pctl}
    return results

def weighted_percentile(pctls):
    total = 0
    for p in WEIGHT_ORDER:
        total += pctls[p]['percentile'] * WEIGHTS[p]
    return total

def calc_rsi(df, base_idx, period=14):
    start = base_idx - 29
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2: return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_ma(df, base_idx, period):
    start = base_idx - period + 1
    if start < 0: return None
    return df['Close'].values[start:base_idx+1].mean()

def calc_volatility(df, base_idx):
    start = base_idx - 20
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2: return 0
    rets = np.diff(closes) / closes[:-1]
    return np.std(rets) * np.sqrt(252) * 100

def d1_sell_score(wp_ret):
    flipped = 100 - wp_ret
    return flipped

def d2_sell_score(wp_volume, ret_1m):
    if ret_1m < 0:
        return min(100, wp_volume * 1.2)
    elif ret_1m < 2:
        return wp_volume * 0.5
    else:
        return max(0, wp_volume * 0.2)

def d3_sell_score(rsi):
    if rsi < 30: return 100
    if rsi < 40: return 80
    if rsi < 50: return 60
    if rsi <= 70: return 30
    if rsi <= 80: return 50
    return 40

def d4_sell_score(price, ma50, ma200):
    score = 0
    if ma50 is not None and price < ma50: score += 35
    if ma200 is not None and price < ma200: score += 35
    if ma50 is not None and ma200 is not None and ma50 < ma200: score += 30
    return min(100, score)

def d5_sell_score(vol):
    if vol <= 20: return 10
    if vol <= 30: return 25
    if vol <= 40: return 55
    if vol <= 50: return 70
    if vol <= 60: return 85
    return 100

def find_closest_idx(ext_df, target_date, max_gap_days=5):
    if ext_df is None: return None
    diffs = (ext_df['Date'] - target_date).abs()
    min_diff = diffs.min()
    if min_diff.days > max_gap_days: return None
    return diffs.idxmin()

def calc_external_return(ext_df, end_idx, period_days):
    if ext_df is None or end_idx is None: return None
    start_idx = end_idx - period_days
    if start_idx < 0: return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[start_idx]) / ext_df['Close'].values[start_idx] * 100

def calc_d6_sell_external(df_gold, gold_idx, df_dxy_arg, df_vix_arg):
    gold_date = df_gold.iloc[gold_idx]['Date']
    gold_closes = df_gold['Close'].values
    gold_1m = compute_return(gold_closes, gold_idx, 21) or 0
    gold_up = gold_1m >= 0
    dxy_score = 0; dxy_signal = "N/A"; dxy_1m = None
    if df_dxy_arg is not None:
        dxy_idx = find_closest_idx(df_dxy_arg, gold_date)
        if dxy_idx is not None:
            dxy_1m = calc_external_return(df_dxy_arg, dxy_idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if not gold_up and dxy_up: dxy_score = +5
                elif not gold_up and not dxy_up: dxy_score = +2
                elif gold_up and not dxy_up: dxy_score = 0
                elif gold_up and dxy_up: dxy_score = -5
    vix_score = 0; vix_signal = "N/A"; vix_level = None
    if df_vix_arg is not None:
        vix_idx = find_closest_idx(df_vix_arg, gold_date)
        if vix_idx is not None:
            vix_level = df_vix_arg['Close'].values[vix_idx]
            if not gold_up:
                if vix_level > 30: vix_score = +5
                elif vix_level > 20: vix_score = +3
                else: vix_score = +1
            else:
                if vix_level > 30: vix_score = -3
                elif vix_level > 20: vix_score = -2
                else: vix_score = 0
    d6_total = dxy_score + vix_score
    d6_scaled = ((d6_total + 10) / 20) * 100
    d6_scaled = max(0, min(100, d6_scaled))
    return {'d6_total': d6_total, 'd6_scaled': d6_scaled, 'dxy_score': dxy_score, 'vix_score': vix_score,
            'dxy_1m': dxy_1m, 'vix_level': vix_level, 'dxy_signal': dxy_signal, 'vix_signal': vix_signal}

def calc_sell_penalties(df_arg, idx):
    closes = df_arg['Close'].values
    ret_1y = compute_return(closes, idx, 252) or 0
    ret_6m = compute_return(closes, idx, 126) or 0
    ret_1m = compute_return(closes, idx, 21) or 0
    ret_1w = compute_return(closes, idx, 5) or 0
    reversal_pen = 0; reversal_flag = ""
    strong = (ret_1y < -20 and ret_1m > 5 and ret_1w > 3)
    mild = ((ret_1y < 0 or ret_6m < 0) and ret_1m > 0 and ret_1w > 0)
    if strong: reversal_pen = -10; reversal_flag = "🔴 Strong Bullish Reversal"
    elif mild: reversal_pen = -5; reversal_flag = "⚠️ Mild Bullish Reversal"
    ma50 = calc_ma(df_arg, idx, 50); ma200 = calc_ma(df_arg, idx, 200)
    gc_pen = 0; gc_flag = ""
    if ma50 is not None and ma200 is not None and ma50 > ma200:
        gc_pen = -5; gc_flag = "✨ Golden Cross (sell penalty)"
    total = max(reversal_pen + gc_pen, -15)
    flags = " | ".join(f for f in [reversal_flag, gc_flag] if f)
    return {'total': total, 'reversal': reversal_pen, 'golden_cross_pen': gc_pen, 'flags': flags,
            'ret_1y': ret_1y, 'ret_6m': ret_6m, 'ret_1m': ret_1m, 'ret_1w': ret_1w}

def full_sell_score(df_arg, idx, df_dxy_arg, df_vix_arg):
    ret_pctls = calc_return_percentiles(df_arg, idx)
    vol_pctls = calc_volume_percentiles(df_arg, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)
    d1 = d1_sell_score(wp_ret)
    ret_1m = compute_return(df_arg['Close'].values, idx, 21) or 0
    d2 = d2_sell_score(wp_vol, ret_1m)
    rsi = calc_rsi(df_arg, idx)
    d3 = d3_sell_score(rsi)
    price = df_arg['Close'].values[idx]
    ma50 = calc_ma(df_arg, idx, 50)
    ma200 = calc_ma(df_arg, idx, 200)
    d4 = d4_sell_score(price, ma50, ma200)
    vol = calc_volatility(df_arg, idx)
    d5 = d5_sell_score(vol)
    ext = calc_d6_sell_external(df_arg, idx, df_dxy_arg, df_vix_arg)
    d6 = ext['d6_scaled']
    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    penalties = calc_sell_penalties(df_arg, idx)
    penalty_scaled = penalties['total'] * (100 / 110)
    net = gross + penalty_scaled
    return {
        'date': df_arg.iloc[idx]['Date'], 'price': price,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'rsi': rsi, 'volatility': vol,
        'gross': gross, 'penalty_scaled': penalty_scaled, 'net': net,
        'penalties': penalties
    }

# ── Read target dates from buy history ──
buy_history = pd.read_csv(os.path.join(base_dir, 'score_history.csv'), encoding='utf-8')
target_dates = pd.to_datetime(buy_history['Date']).tolist()
print(f"Target dates: {len(target_dates)} ({target_dates[0].strftime('%Y-%m-%d')} → {target_dates[-1].strftime('%Y-%m-%d')})")

# ── Build sell history ──
rows = []
for i, target_date in enumerate(target_dates):
    # Find closest index in gold_prices for this date
    diffs = (df['Date'] - target_date).abs()
    idx = diffs.idxmin()
    if diffs[idx].days > 3:
        continue  # skip if no close match
    
    s = full_sell_score(df, idx, df_dxy, df_vix)
    
    def tier_sell(sc):
        c = max(0, min(100, sc))
        if c >= 75: return "Strong Sell ↓↓"
        if c >= 60: return "Sell Signal ↓"
        if c >= 45: return "Moderate"
        return "No Sell Signal"
    
    rows.append({
        'Date': target_date.strftime('%Y-%m-%d'),
        'Price': round(s['price'], 2),
        'Net_Score': round(s['net'], 2),
        'Gross_Score': round(s['gross'], 2),
        'D1_Return': round(s['d1'], 2),
        'D2_Volume': round(s['d2'], 2),
        'D3_RSI': round(s['d3'], 2),
        'D4_MA': round(s['d4'], 2),
        'D5_Volatility': round(s['d5'], 2),
        'D6_External': round(s['d6'], 2),
        'RSI': round(s['rsi'], 2),
        'Volatility_Pct': round(s['volatility'], 2),
        'Penalty_Scaled': round(s['penalty_scaled'], 2),
        'Warning_Flags': s['penalties']['flags'] if s['penalties']['flags'] else 'None',
        'Tier': tier_sell(s['net']),
        'As_Of_Running': 'backfill',
        'Exhaust_Scenario': '',
    })
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(target_dates)}...")

history_df = pd.DataFrame(rows)
print(f"\nBackfill complete: {len(history_df)} rows")

# ── Run exhaustion detection on the full history ──
h = history_df.copy()
h['Net_Score'] = pd.to_numeric(h['Net_Score'], errors='coerce')
h['D5_Volatility'] = pd.to_numeric(h['D5_Volatility'], errors='coerce')

for i in range(len(h)):
    if i < 5:
        continue
    net_now = h.iloc[i]['Net_Score']
    d5_now = h.iloc[i]['D5_Volatility']
    net_5d_ago = h.iloc[i-5]['Net_Score']
    net_5d_change = net_now - net_5d_ago
    last10 = h['Net_Score'].iloc[max(0,i-9):i+1]
    max_10d = last10.max()
    min_10d = last10.min()
    d5_5d_ago = h.iloc[i-5]['D5_Volatility']
    d5_shift = d5_now - d5_5d_ago
    
    se1 = net_now >= 70 and net_5d_change < 0
    se2 = max_10d >= 80 and net_5d_change < -8 and not se1
    se3 = min_10d < 50 and net_5d_change > 3
    se4 = abs(d5_shift) >= 50 and not se1 and not se2 and not se3
    
    if se1: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Exhaustion'
    elif se3: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Recovery'
    elif se2: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Topping'
    elif se4: h.at[h.index[i], 'Exhaust_Scenario'] = 'Vol Shift'

# Count signals
exhaust_counts = h['Exhaust_Scenario'].value_counts()
print(f"\nExhaustion signals found:")
for k, v in exhaust_counts.items():
    if k: print(f"  {k}: {v}")

# Save
output_path = os.path.join(base_dir, 'score_history_sell.csv')
h.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Saved: {output_path} ({len(h)} rows)")
print(f"Date range: {h['Date'].iloc[0]} → {h['Date'].iloc[-1]}")

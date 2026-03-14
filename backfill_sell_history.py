#!/usr/bin/env python3
"""
Backfill score_history_sell.csv — CLEAN VERSION
Reads dates from score_history.csv (buy), computes sell scoring for each date.
All functions inline (no import from sell script).
Output columns match buy history: D5_DirVol, Vol_Ratio, Ret_1W/1M/3M, Golden_Cross, Z_Score, Exhaust
"""

import pandas as pd
import numpy as np
import os, sys

base_dir = os.path.dirname(os.path.abspath(__file__))

ROLLING_WINDOW = 252
LOOKBACK = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
WEIGHTS = {'1Y': 0.30, '6M': 0.25, '3M': 0.20, '1M': 0.15, '1W': 0.10}
WEIGHT_ORDER = ['1Y', '6M', '3M', '1M', '1W']


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════

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

if df is None:
    print("ERROR: gold_prices.csv not found"); sys.exit(1)

print(f"Gold: {len(df)} rows | DXY: {len(df_dxy) if df_dxy is not None else 0} | VIX: {len(df_vix) if df_vix is not None else 0}")


# ══════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════

def compute_return(closes, end_idx, period):
    start_idx = end_idx - period
    if start_idx < 0 or closes[start_idx] == 0: return None
    return (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100

def calc_return_percentiles(df, base_idx):
    closes = df['Close'].values
    results = {}
    for p in WEIGHT_ORDER:
        days = LOOKBACK[p]
        cur = compute_return(closes, base_idx, days)
        if cur is None:
            results[p] = {'return': 0, 'percentile': 50}; continue
        rolling = []
        for i in range(max(0, base_idx - ROLLING_WINDOW), base_idx):
            r = compute_return(closes, i, days)
            if r is not None: rolling.append(r)
        if len(rolling) < 10:
            results[p] = {'return': cur, 'percentile': 50}; continue
        pctl = sum(1 for r in rolling if r < cur) / len(rolling) * 100
        results[p] = {'return': cur, 'percentile': pctl}
    return results

def calc_volume_percentiles(df, base_idx):
    results = {}
    for p in WEIGHT_ORDER:
        days = LOOKBACK[p]
        end = base_idx + 1; start = end - days
        if start < 0:
            results[p] = {'volume': 0, 'percentile': 50}; continue
        cur = df['Volume'].values[start:end].sum()
        rolling = []
        for i in range(max(0, base_idx - ROLLING_WINDOW), base_idx):
            s = i + 1 - days
            if s < 0: continue
            rolling.append(df['Volume'].values[s:i+1].sum())
        if len(rolling) < 10:
            results[p] = {'volume': cur, 'percentile': 50}; continue
        pctl = sum(1 for v in rolling if v < cur) / len(rolling) * 100
        results[p] = {'volume': cur, 'percentile': pctl}
    return results

def weighted_percentile(pctls):
    return sum(pctls[p]['percentile'] * WEIGHTS[p] for p in WEIGHT_ORDER)

def calc_rsi(df, base_idx, period=14):
    start = max(0, base_idx - 29)
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2: return 50
    deltas = np.diff(closes)
    avg_gain = np.mean(np.where(deltas > 0, deltas, 0)[-period:])
    avg_loss = np.mean(np.where(deltas < 0, -deltas, 0)[-period:])
    if avg_loss == 0: return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_ma(df, base_idx, period):
    start = base_idx - period + 1
    if start < 0: return None
    return float(df['Close'].values[start:base_idx+1].mean())

def calc_volatility(df, base_idx):
    start = max(0, base_idx - 20)
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2:
        return {'abs_vol': 0, 'up_vol': 0, 'down_vol': 0, 'vol_ratio': 1.0}
    rets = np.diff(closes) / closes[:-1]
    abs_vol = float(np.std(rets) * np.sqrt(252) * 100)
    up_rets = rets[rets > 0]; down_rets = rets[rets < 0]
    up_vol = float(np.std(up_rets) * np.sqrt(252) * 100) if len(up_rets) >= 2 else 0
    down_vol = float(np.std(down_rets) * np.sqrt(252) * 100) if len(down_rets) >= 2 else 0
    if up_vol > 0: vol_ratio = down_vol / up_vol
    elif down_vol > 0: vol_ratio = 999
    else: vol_ratio = 1.0
    return {'abs_vol': abs_vol, 'up_vol': up_vol, 'down_vol': down_vol, 'vol_ratio': vol_ratio}


# ══════════════════════════════════════════════════════
# SELL SCORING FUNCTIONS (0-100)
# ══════════════════════════════════════════════════════

def d1_sell_score(wp_ret): return 100 - wp_ret

def d2_sell_score(wp_vol, ret_1m):
    if wp_vol >= 70 and ret_1m < 0: return 80 + (wp_vol - 70) / 30 * 20
    elif wp_vol >= 70: return 30 - (wp_vol - 70) / 30 * 10
    elif ret_1m < 0: return 40 + (70 - wp_vol) / 70 * 30
    else: return 10

def d3_sell_score(rsi):
    if rsi < 30: return 100
    if rsi < 40: return 85
    if rsi < 50: return 65
    if rsi > 80: return 65
    if rsi > 70: return 50
    if rsi >= 60: return 20
    return 40

def d4_sell_score(price, ma50, ma200):
    pts = 0
    if ma50 is not None and price < ma50: pts += 35
    if ma200 is not None and price < ma200: pts += 35
    if ma50 is not None and ma200 is not None and ma50 < ma200: pts += 30
    return min(pts, 100)

def d5_sell_score(vol_data):
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio >= 2.0: return 100
    if ratio >= 1.5: return 85
    if ratio >= 1.2: return 70
    if ratio >= 1.0: return 55
    if ratio >= 0.8: return 40
    if ratio >= 0.6: return 20
    return 10


# ══════════════════════════════════════════════════════
# EXTERNAL (D6) + PENALTIES + Z-SCORE
# ══════════════════════════════════════════════════════

def find_closest_idx(ext_df, target_date, max_gap=5):
    if ext_df is None: return None
    diffs = (ext_df['Date'] - target_date).abs()
    if diffs.min().days > max_gap: return None
    return diffs.idxmin()

def calc_ext_return(ext_df, end_idx, period):
    if ext_df is None or end_idx is None: return None
    start = end_idx - period
    if start < 0: return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[start]) / ext_df['Close'].values[start] * 100

def calc_d6_sell(df_gold, gold_idx, df_dxy_arg, df_vix_arg):
    gold_date = df_gold.iloc[gold_idx]['Date']
    gold_1m = compute_return(df_gold['Close'].values, gold_idx, 21) or 0
    gold_up = gold_1m >= 0

    dxy_score = 0; dxy_1m = None
    if df_dxy_arg is not None:
        di = find_closest_idx(df_dxy_arg, gold_date)
        if di is not None:
            dxy_1m = calc_ext_return(df_dxy_arg, di, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if not gold_up and dxy_up: dxy_score = +5
                elif not gold_up and not dxy_up: dxy_score = +2
                elif gold_up and not dxy_up: dxy_score = 0
                elif gold_up and dxy_up: dxy_score = -5

    vix_score = 0; vix_level = None
    if df_vix_arg is not None:
        vi = find_closest_idx(df_vix_arg, gold_date)
        if vi is not None:
            vix_level = df_vix_arg['Close'].values[vi]
            if not gold_up:
                if vix_level > 30: vix_score = +5
                elif vix_level > 20: vix_score = +3
                else: vix_score = +1
            else:
                if vix_level > 30: vix_score = -3
                elif vix_level > 20: vix_score = -2
                else: vix_score = 0

    raw = dxy_score + vix_score
    scaled = max(0, min(100, ((raw + 10) / 20) * 100))
    return {'d6_scaled': scaled, 'dxy_1m': dxy_1m, 'vix_level': vix_level}

def calc_sell_penalties(df_arg, idx):
    closes = df_arg['Close'].values
    ret_1y = compute_return(closes, idx, 252) or 0
    ret_6m = compute_return(closes, idx, 126) or 0
    ret_1m = compute_return(closes, idx, 21) or 0
    ret_1w = compute_return(closes, idx, 5) or 0
    rev_pen = 0; rev_flag = ""
    if ret_1y < -20 and ret_1m > 5 and ret_1w > 3:
        rev_pen = -10; rev_flag = "🔴 Strong Bullish Reversal"
    elif (ret_1y < 0 or ret_6m < 0) and ret_1m > 0 and ret_1w > 0:
        rev_pen = -5; rev_flag = "⚠️ Mild Bullish Reversal"
    ma50 = calc_ma(df_arg, idx, 50); ma200 = calc_ma(df_arg, idx, 200)
    gc_pen = 0; gc_flag = ""
    if ma50 is not None and ma200 is not None and ma50 > ma200:
        gc_pen = -5; gc_flag = "✨ Golden Cross (sell penalty)"
    total = max(rev_pen + gc_pen, -15)
    flags = " | ".join(f for f in [rev_flag, gc_flag] if f)
    return {'total': total, 'reversal': rev_pen, 'gc_pen': gc_pen, 'flags': flags}

def calc_zscore(df, base_idx):
    closes = df['Close'].values[:base_idx + 1]
    result = {}
    for period, label in [(50, '50d'), (100, '100d'), (200, '200d')]:
        if len(closes) < period:
            result[f'z_{label}'] = None; continue
        window = closes[-period:]
        mean = np.mean(window); std = np.std(window, ddof=1)
        result[f'z_{label}'] = (closes[-1] - mean) / std if std > 0 else 0.0

    z = result.get('z_50d')
    if z is None: result['zone'] = 'N/A'
    elif z >= 2.5: result['zone'] = 'Extreme Extended'
    elif z >= 2.0: result['zone'] = 'Extended'
    elif z <= -2.0: result['zone'] = 'Extreme Depressed'
    elif z <= -1.5: result['zone'] = 'Depressed'
    else: result['zone'] = 'Normal'

    if len(closes) >= 55:
        c5 = closes[:-5]; w5 = c5[-50:]
        m5 = np.mean(w5); s5 = np.std(w5, ddof=1)
        result['z_delta_5d'] = (result['z_50d'] - ((c5[-1] - m5) / s5)) if s5 > 0 else 0.0
    else:
        result['z_delta_5d'] = None
    return result


# ══════════════════════════════════════════════════════
# FULL SELL SCORE FOR ONE DATE
# ══════════════════════════════════════════════════════

def full_sell(idx):
    ret_pctls = calc_return_percentiles(df, idx)
    vol_pctls = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)

    d1 = d1_sell_score(wp_ret)
    ret_1m = compute_return(df['Close'].values, idx, 21) or 0
    d2 = d2_sell_score(wp_vol, ret_1m)
    rsi = calc_rsi(df, idx)
    d3 = d3_sell_score(rsi)
    price = df['Close'].values[idx]
    ma50 = calc_ma(df, idx, 50); ma200 = calc_ma(df, idx, 200)
    d4 = d4_sell_score(price, ma50, ma200)
    vol_data = calc_volatility(df, idx)
    d5 = d5_sell_score(vol_data)
    ext = calc_d6_sell(df, idx, df_dxy, df_vix)
    d6 = ext['d6_scaled']

    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    pen = calc_sell_penalties(df, idx)
    pen_scaled = pen['total'] * (100 / 110)
    net = gross + pen_scaled

    golden_cross = (ma50 is not None and ma200 is not None and ma50 > ma200)
    zscore = calc_zscore(df, idx)

    ret_1w = compute_return(df['Close'].values, idx, 5) or 0
    ret_3m = compute_return(df['Close'].values, idx, 63) or 0

    return {
        'price': price, 'net': net, 'gross': gross,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'vol_ratio': vol_data['vol_ratio'], 'rsi': rsi,
        'volatility': vol_data['abs_vol'], 'pen_scaled': pen_scaled,
        'ret_1w': ret_1w, 'ret_1m': ret_1m, 'ret_3m': ret_3m,
        'golden_cross': golden_cross, 'penalties': pen,
        'zscore': zscore,
    }

def tier_sell(sc):
    c = max(0, min(100, sc))
    if c >= 85: return "Very Strong Sell ↓↓"
    if c >= 75: return "Strong Sell ↓"
    if c >= 60: return "Moderate Sell ↓"
    if c >= 45: return "Neutral →"
    if c >= 30: return "Weak Sell"
    return "No Sell Signal"


# ══════════════════════════════════════════════════════
# BACKFILL LOOP
# ══════════════════════════════════════════════════════

buy_path = os.path.join(base_dir, 'score_history.csv')
if not os.path.exists(buy_path):
    print("ERROR: score_history.csv not found"); sys.exit(1)

buy_hist = pd.read_csv(buy_path, encoding='utf-8')
target_dates = pd.to_datetime(buy_hist['Date']).tolist()
print(f"Target dates: {len(target_dates)} ({target_dates[0].strftime('%Y-%m-%d')} → {target_dates[-1].strftime('%Y-%m-%d')})")

rows = []
for i, td in enumerate(target_dates):
    diffs = (df['Date'] - td).abs()
    idx = diffs.idxmin()
    if diffs[idx].days > 3: continue

    s = full_sell(idx)
    rows.append({
        'Date': td.strftime('%Y-%m-%d'),
        'Price': round(s['price'], 2),
        'Net_Score': round(s['net'], 2),
        'Gross_Score': round(s['gross'], 2),
        'D1_Return': round(s['d1'], 2),
        'D2_Volume': round(s['d2'], 2),
        'D3_RSI': round(s['d3'], 2),
        'D4_MA': round(s['d4'], 2),
        'D5_DirVol': round(s['d5'], 2),
        'D6_External': round(s['d6'], 2),
        'Vol_Ratio': round(s['vol_ratio'], 3),
        'RSI': round(s['rsi'], 2),
        'Volatility_Pct': round(s['volatility'], 2),
        'Penalty_Scaled': round(s['pen_scaled'], 2),
        'Ret_1W': round(s['ret_1w'], 2),
        'Ret_1M': round(s['ret_1m'], 2),
        'Ret_3M': round(s['ret_3m'], 2),
        'Golden_Cross': str(s['golden_cross']),
        'Z_Score_50d': round(s['zscore']['z_50d'], 3) if s['zscore']['z_50d'] is not None else '',
        'Z_Zone': s['zscore']['zone'],
        'Z_Delta_5d': round(s['zscore']['z_delta_5d'], 3) if s['zscore'].get('z_delta_5d') is not None else '',
        'Warning_Flags': s['penalties']['flags'] if s['penalties']['flags'] else 'None',
        'Tier': tier_sell(s['net']),
        'As_Of_Running': 'backfill',
        'Exhaust_Scenario': '',
    })

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(target_dates)}...")

h = pd.DataFrame(rows)
print(f"\nBackfill complete: {len(h)} rows")


# ══════════════════════════════════════════════════════
# EXHAUSTION DETECTION
# ══════════════════════════════════════════════════════

h['Net_Score_num'] = pd.to_numeric(h['Net_Score'], errors='coerce')
h['D5_num'] = pd.to_numeric(h['D5_DirVol'], errors='coerce')

for i in range(len(h)):
    if i < 5: continue
    net_now = h.iloc[i]['Net_Score_num']
    d5_now = h.iloc[i]['D5_num']
    net_5d = h.iloc[i-5]['Net_Score_num']
    delta = net_now - net_5d
    last10 = h['Net_Score_num'].iloc[max(0,i-9):i+1]
    mx = last10.max(); mn = last10.min()
    d5_5d = h.iloc[i-5]['D5_num']
    d5_shift = d5_now - d5_5d

    se1 = net_now >= 70 and delta < 0
    se2 = mx >= 80 and delta < -8 and not se1
    se3 = mn < 50 and delta > 3
    se4 = abs(d5_shift) >= 50 and not se1 and not se2 and not se3

    if se1: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Exhaustion'
    elif se3: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Recovery'
    elif se2: h.at[h.index[i], 'Exhaust_Scenario'] = 'Sell Topping'
    elif se4: h.at[h.index[i], 'Exhaust_Scenario'] = 'Vol Shift'

h.drop(columns=['Net_Score_num', 'D5_num'], inplace=True)

# Print exhaust stats
ec = h['Exhaust_Scenario'].value_counts()
print(f"\nExhaustion signals:")
for k, v in ec.items():
    if k: print(f"  {k}: {v}")

# Save
out_path = os.path.join(base_dir, 'score_history_sell.csv')
h.to_csv(out_path, index=False, encoding='utf-8')
print(f"\n✅ Saved: {out_path} ({len(h)} rows)")
print(f"Columns: {list(h.columns)}")
print(f"Date range: {h['Date'].iloc[0]} → {h['Date'].iloc[-1]}")

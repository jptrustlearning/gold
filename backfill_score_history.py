#!/usr/bin/env python3
"""
Backfill score_history.csv — คำนวณ score ย้อนหลังทุกวันทำการ
One-time script: รันครั้งเดียวเพื่อสร้าง historical data สำหรับ exhaustion analysis

Usage: python3 backfill_score_history.py [days]
  days = จำนวนวันย้อนหลัง (default: 252 = 1 ปี)
"""

import pandas as pd
import numpy as np
import os, sys

# ── CONFIG (same as gold_momentum_v2.py) ──
ROLLING_WINDOW = 252
LOOKBACK = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
WEIGHTS = {'1Y': 0.30, '6M': 0.25, '3M': 0.20, '1M': 0.15, '1W': 0.10}
WEIGHT_ORDER = ['1Y', '6M', '3M', '1M', '1W']

base_dir = os.path.dirname(os.path.abspath(__file__))

# ── LOAD DATA ──
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
    print("❌ gold_prices.csv not found")
    sys.exit(1)

# ══════════════════════════════════════════════════════
# SCORING FUNCTIONS (copied from gold_momentum_v2.py)
# ══════════════════════════════════════════════════════

def compute_return(closes, end_idx, period_days):
    start_idx = end_idx - period_days
    if start_idx < 0:
        return None
    return (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100

def rolling_percentile(series_values, current_val, window=ROLLING_WINDOW):
    valid = series_values[~np.isnan(series_values)]
    if len(valid) < 10:
        return 50.0
    count_below = np.sum(valid < current_val)
    return count_below / (len(valid) - 1) * 100 if len(valid) > 1 else 50.0

def calc_return_percentiles(df, base_idx):
    closes = df['Close'].values
    results = {}
    for period, days in LOOKBACK.items():
        ret = compute_return(closes, base_idx, days)
        if ret is None:
            results[period] = {'return': 0, 'percentile': 50.0}
            continue
        rolling_rets = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            r = compute_return(closes, i, days)
            if r is not None:
                rolling_rets.append(r)
        pctl = rolling_percentile(np.array(rolling_rets), ret) if rolling_rets else 50.0
        results[period] = {'return': ret, 'percentile': pctl}
    return results

def calc_volume_percentiles(df, base_idx):
    volumes = df['Volume'].values
    results = {}
    for period, days in LOOKBACK.items():
        start = base_idx - days + 1
        if start < 0:
            results[period] = {'volume': 0, 'percentile': 50.0}
            continue
        cum_vol = np.sum(volumes[start:base_idx + 1])
        rolling_vols = []
        roll_start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(roll_start, base_idx):
            s = i - days + 1
            if s >= 0:
                rv = np.sum(volumes[s:i + 1])
                rolling_vols.append(rv)
        pctl = rolling_percentile(np.array(rolling_vols), cum_vol) if rolling_vols else 50.0
        results[period] = {'volume': cum_vol, 'percentile': pctl}
    return results

def weighted_percentile(pctls):
    total = 0
    for period in WEIGHT_ORDER:
        total += pctls[period]['percentile'] * WEIGHTS[period]
    return total

def d1_score(wp): return wp
def d2_score(wp): return wp

def calc_rsi(df, base_idx):
    start = base_idx - 29
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx + 1]
    if len(closes) < 15:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)[-14:]
    losses = np.where(deltas < 0, -deltas, 0)[-14:]
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def d3_score(rsi):
    if 50 <= rsi <= 70: return 100
    if 40 <= rsi < 50: return 80
    if 70 < rsi <= 80: return 70
    if 30 <= rsi < 40: return 60
    if rsi > 80: return 50
    if rsi < 30: return 30
    return 50

def calc_ma(df, base_idx, period):
    start = base_idx - period + 1
    if start < 0: return None
    return np.mean(df['Close'].values[start:base_idx + 1])

def d4_score(price, ma50, ma200):
    pts = 0
    if ma50 is not None and price > ma50: pts += 35
    if ma200 is not None and price > ma200: pts += 35
    if ma50 is not None and ma200 is not None and ma50 > ma200: pts += 30
    return min(pts, 100)

def calc_volatility(df, base_idx):
    start = base_idx - 20
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2:
        return {'abs_vol': 0, 'up_vol': 0, 'down_vol': 0, 'vol_ratio': 1.0}
    rets = np.diff(closes) / closes[:-1]
    abs_vol = np.std(rets) * np.sqrt(252) * 100
    up_rets = rets[rets > 0]
    down_rets = rets[rets < 0]
    up_vol = np.std(up_rets) * np.sqrt(252) * 100 if len(up_rets) >= 2 else 0
    down_vol = np.std(down_rets) * np.sqrt(252) * 100 if len(down_rets) >= 2 else 0
    if up_vol > 0:
        vol_ratio = down_vol / up_vol
    elif down_vol > 0:
        vol_ratio = 999
    else:
        vol_ratio = 1.0
    return {'abs_vol': abs_vol, 'up_vol': up_vol, 'down_vol': down_vol, 'vol_ratio': vol_ratio}

def d5_score(vol_data):
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio <= 0.6:  return 100
    if ratio <= 0.8:  return 85
    if ratio <= 1.0:  return 70
    if ratio <= 1.2:  return 55
    if ratio <= 1.5:  return 40
    if ratio <= 2.0:  return 20
    return 10

def calc_penalties(df, base_idx):
    closes = df['Close'].values
    ret_1y = compute_return(closes, base_idx, 252) or 0
    ret_6m = compute_return(closes, base_idx, 126) or 0
    ret_1m = compute_return(closes, base_idx, 21) or 0
    ret_1w = compute_return(closes, base_idx, 5) or 0
    reversal_pen = 0
    reversal_flag = ""
    strong = (ret_1y > 20 and ret_1m < -5 and ret_1w < -3)
    mild = ((ret_1y > 0 or ret_6m > 0) and ret_1m < 0 and ret_1w < 0)
    if strong:
        reversal_pen = -10
        reversal_flag = "🔴 Strong Reversal"
    elif mild:
        reversal_pen = -5
        reversal_flag = "⚠️ Mild Reversal"
    ma50 = calc_ma(df, base_idx, 50)
    ma200 = calc_ma(df, base_idx, 200)
    dc_pen = 0
    dc_flag = ""
    if ma50 is not None and ma200 is not None and ma50 < ma200:
        dc_pen = -5
        price = closes[base_idx]
        if price < ma50 and price < ma200:
            dc_flag = "💀💀 Death Cross + Below MAs"
        else:
            dc_flag = "💀 Death Cross"
    total = max(reversal_pen + dc_pen, -15)
    flags_list = [f for f in [reversal_flag, dc_flag] if f]
    return {
        'total': total, 'reversal': reversal_pen, 'death_cross': dc_pen,
        'flags': ' | '.join(flags_list) if flags_list else '',
        'ret_1y': ret_1y, 'ret_6m': ret_6m, 'ret_1m': ret_1m, 'ret_1w': ret_1w
    }

def calc_d6_external(df, idx, df_dxy, df_vix):
    dxy_score, vix_score = 0, 0
    dxy_1m, vix_level = None, None
    dxy_signal, vix_signal = '⚪ N/A', '⚪ N/A'
    gold_close = df['Close'].values[idx]
    gold_date = df.iloc[idx]['Date']
    gold_1m = compute_return(df['Close'].values, idx, 21) or 0
    gold_up = gold_1m >= 0
    if df_dxy is not None and len(df_dxy) > 21:
        dxy_mask = df_dxy['Date'] <= gold_date
        dxy_sub = df_dxy[dxy_mask]
        if len(dxy_sub) > 21:
            dxy_1m = (dxy_sub['Close'].values[-1] - dxy_sub['Close'].values[-22]) / dxy_sub['Close'].values[-22] * 100
            if gold_up:
                if dxy_1m <= -2: dxy_score, dxy_signal = 3, "🟢 Weak Dollar + Gold Up"
                elif dxy_1m <= 0: dxy_score, dxy_signal = 2, "🟢 Mild Dollar Weak + Gold Up"
                elif dxy_1m <= 2: dxy_score, dxy_signal = 0, "⚪ Dollar Mild + Gold Up"
                else: dxy_score, dxy_signal = -1, "🟡 Dollar Strong but Gold Up"
            else:
                if dxy_1m > 2: dxy_score, dxy_signal = -3, "🔴 Strong Dollar + Gold Down"
                elif dxy_1m > 0: dxy_score, dxy_signal = -2, "🟠 Dollar Up + Gold Down"
                elif dxy_1m > -2: dxy_score, dxy_signal = 0, "⚪ Dollar Mild + Gold Down"
                else: dxy_score, dxy_signal = 1, "🟡 Dollar Weak but Gold Down"
    if df_vix is not None:
        vix_mask = df_vix['Date'] <= gold_date
        vix_sub = df_vix[vix_mask]
        if len(vix_sub) > 0:
            vix_level = vix_sub['Close'].values[-1]
            if gold_up:
                if vix_level >= 30: vix_score, vix_signal = 2, "🟢 Fear + Gold Up (safe haven)"
                elif vix_level >= 20: vix_score, vix_signal = 1, "🟢 Elevated VIX + Gold Up"
                else: vix_score, vix_signal = 0, "⚪ Calm (VIX<20 + Gold Up)"
            else:
                if vix_level > 30: vix_score, vix_signal = -3, "🔴 Panic Selling"
                elif vix_level >= 20: vix_score, vix_signal = -2, "🟠 Fear Not Helping"
                else: vix_score, vix_signal = 0, "⚪ Calm Drift"
    total_d6 = max(min(dxy_score + vix_score, 10), -10)
    d6_scaled = (total_d6 + 10) / 20 * 100
    return {
        'd6_total': total_d6, 'd6_scaled': d6_scaled,
        'dxy_score': dxy_score, 'vix_score': vix_score,
        'dxy_1m': dxy_1m, 'vix_level': vix_level,
        'dxy_signal': dxy_signal, 'vix_signal': vix_signal, 'gold_1m': gold_1m
    }

def full_score(df, idx, df_dxy, df_vix):
    ret_pctls = calc_return_percentiles(df, idx)
    vol_pctls = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)
    d1 = d1_score(wp_ret)
    d2 = d2_score(wp_vol)
    rsi = calc_rsi(df, idx)
    d3 = d3_score(rsi)
    price = df['Close'].values[idx]
    ma50 = calc_ma(df, idx, 50)
    ma200 = calc_ma(df, idx, 200)
    d4 = d4_score(price, ma50, ma200)
    vol_data = calc_volatility(df, idx)
    d5 = d5_score(vol_data)
    vol = vol_data['abs_vol']
    penalties = calc_penalties(df, idx)
    ext = calc_d6_external(df, idx, df_dxy, df_vix)
    d6 = ext['d6_scaled']
    gross_avg_dims = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    penalty_scaled = penalties['total'] * (100 / 110)
    net = gross_avg_dims + penalty_scaled
    golden_cross = (ma50 is not None and ma200 is not None and ma50 > ma200)
    return {
        'date': df.iloc[idx]['Date'], 'price': price,
        'ret_pctls': ret_pctls, 'vol_pctls': vol_pctls,
        'wp_ret': wp_ret, 'wp_vol': wp_vol,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'rsi': rsi, 'ma50': ma50, 'ma200': ma200,
        'golden_cross': golden_cross, 'volatility': vol,
        'vol_ratio': vol_data['vol_ratio'],
        'gross': gross_avg_dims, 'penalties': penalties,
        'penalty_scaled': penalty_scaled, 'net': net,
    }

def calc_zscore_regime(df, base_idx):
    closes = df['Close'].values[:base_idx + 1]
    result = {}
    for period, label in [(50, '50d'), (100, '100d'), (200, '200d')]:
        if len(closes) < period:
            result[f'z_{label}'] = None
            continue
        window = closes[-period:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        result[f'z_{label}'] = (closes[-1] - mean) / std if std > 0 else 0.0
    z_primary = result.get('z_50d')
    if z_primary is None:
        result['zone'] = 'N/A'
    elif z_primary >= 2.5:
        result['zone'] = 'Extreme Extended'
    elif z_primary >= 2.0:
        result['zone'] = 'Extended'
    elif z_primary <= -2.0:
        result['zone'] = 'Extreme Depressed'
    elif z_primary <= -1.5:
        result['zone'] = 'Depressed'
    else:
        result['zone'] = 'Normal'
    if len(closes) >= 55:
        closes_5d_ago = closes[:-5]
        window_5d_ago = closes_5d_ago[-50:]
        mean_ago = np.mean(window_5d_ago)
        std_ago = np.std(window_5d_ago, ddof=1)
        if std_ago > 0:
            z_5d_ago = (closes_5d_ago[-1] - mean_ago) / std_ago
            result['z_delta_5d'] = result['z_50d'] - z_5d_ago
        else:
            result['z_delta_5d'] = 0.0
    else:
        result['z_delta_5d'] = None
    return result

def tier(score):
    clamped = max(0, min(100, score))
    if clamped >= 85: return "Very Strong ↑↑"
    if clamped >= 75: return "Strong ↑"
    if clamped >= 60: return "Moderate ↑"
    if clamped >= 45: return "Neutral →"
    if clamped >= 30: return "Weak ↓"
    return "Very Weak ↓↓"


# ══════════════════════════════════════════════════════
# BACKFILL LOOP
# ══════════════════════════════════════════════════════

backfill_days = int(sys.argv[1]) if len(sys.argv) > 1 else 252
total_rows = len(df)

# Need at least 252 (rolling window) + 252 (lookback) days before first backfill point
min_idx = 252 + 252  # need enough history for rolling percentile
start_idx = max(min_idx, total_rows - backfill_days)
end_idx = total_rows - 1

print(f"Backfill score_history.csv")
print(f"{'='*50}")
print(f"Gold data: {total_rows} rows ({df.iloc[0]['Date'].strftime('%Y-%m-%d')} → {df.iloc[-1]['Date'].strftime('%Y-%m-%d')})")
print(f"Backfill range: idx {start_idx} → {end_idx} ({end_idx - start_idx + 1} trading days)")
print(f"Date range: {df.iloc[start_idx]['Date'].strftime('%Y-%m-%d')} → {df.iloc[end_idx]['Date'].strftime('%Y-%m-%d')}")
print()

rows = []
for i in range(start_idx, end_idx + 1):
    s = full_score(df, i, df_dxy, df_vix)
    z = calc_zscore_regime(df, i)
    t = tier(s['net'])

    row = {
        'Date': s['date'].strftime('%Y-%m-%d'),
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
        'Penalty_Scaled': round(s['penalty_scaled'], 2),
        'Ret_1W': round(s['ret_pctls']['1W']['return'], 2),
        'Ret_1M': round(s['ret_pctls']['1M']['return'], 2),
        'Ret_3M': round(s['ret_pctls']['3M']['return'], 2),
        'Golden_Cross': str(s['golden_cross']),
        'Z_Score_50d': round(z['z_50d'], 3) if z.get('z_50d') is not None else '',
        'Z_Zone': z['zone'],
        'Z_Delta_5d': round(z['z_delta_5d'], 3) if z.get('z_delta_5d') is not None else '',
        'Warning_Flags': s['penalties']['flags'] if s['penalties']['flags'] else 'None',
        'Tier': t,
        'As_Of_Running': 'backfill',
    }
    rows.append(row)

    # Progress
    done = i - start_idx + 1
    total = end_idx - start_idx + 1
    if done % 50 == 0 or done == total:
        print(f"  [{done}/{total}] {row['Date']}  Net={row['Net_Score']:.1f}  D5={row['D5_DirVol']}  VR={row['Vol_Ratio']}  Z={row.get('Z_Score_50d', 'N/A')}  {t}")

# Save
history_df = pd.DataFrame(rows)
history_path = os.path.join(base_dir, 'score_history.csv')
history_df.to_csv(history_path, index=False, encoding='utf-8')

print(f"\n{'='*50}")
print(f"✅ score_history.csv — {len(history_df)} rows")
print(f"   {history_df.iloc[0]['Date']} → {history_df.iloc[-1]['Date']}")
print(f"   Net Score range: {history_df['Net_Score'].min():.1f} → {history_df['Net_Score'].max():.1f}")
print(f"   D5 DirVol range: {history_df['D5_DirVol'].min()} → {history_df['D5_DirVol'].max()}")
print(f"   Vol Ratio range: {history_df['Vol_Ratio'].min():.3f} → {history_df['Vol_Ratio'].max():.3f}")

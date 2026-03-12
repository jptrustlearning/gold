#!/usr/bin/env python3
"""
Gold Momentum Scoring — H1 SELL Side
JP Trust Learning

Mirror of H1 Buy score: gives HIGH scores when bearish conditions are strong.
Adapted from daily gold_momentum_sell.py for H1 timeframe.

Differences from daily:
- H1 data input (gold_prices_h1.csv)
- Lookback: 4H/1D/1W/2W/1M
- MA: 120/480 instead of 50/200
- DXY/VIX: uses daily data (latest available)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os, sys

# ── CONFIG (H1 adapted) ──
ROLLING_WINDOW = 504
LOOKBACK = {'4H': 4, '1D': 24, '1W': 120, '2W': 240, '1M': 504}
WEIGHTS = {'1M': 0.30, '2W': 0.25, '1W': 0.20, '1D': 0.15, '4H': 0.10}
WEIGHT_ORDER = ['1M', '2W', '1W', '1D', '4H']
MA_SHORT = 120
MA_LONG = 480

RUN_TS = datetime.now(timezone.utc)
AS_OF = RUN_TS.strftime("%d/%m/%Y %H:%M UTC")
TS_FILE = RUN_TS.strftime("%d%m%Y_%H%M")

base_dir = os.path.dirname(os.path.abspath(__file__))

def load_h1_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found"); return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

def load_daily_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found"); return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

df = load_h1_csv('gold_prices_h1.csv')
df_dxy = load_daily_csv('dxy_prices.csv')
df_vix = load_daily_csv('vix_prices.csv')

if df is None:
    print("❌ gold_prices_h1.csv not found"); sys.exit(1)

BD2_idx = len(df) - 1
BD1_idx = max(0, len(df) - 25)
BD1_date = df.iloc[BD1_idx]['Datetime']
BD2_date = df.iloc[BD2_idx]['Datetime']

print(f"Gold Momentum Scoring — H1 SELL Side")
print(f"{'='*55}")
print(f"BD1: {BD1_date.strftime('%Y-%m-%d %H:%M')} | BD2: {BD2_date.strftime('%Y-%m-%d %H:%M')}")
print(f"H1 rows: {len(df)}")


# ══════════════════════════════════════════════════════
# SHARED CALCULATIONS
# ══════════════════════════════════════════════════════

def compute_return(closes, end_idx, period):
    si = end_idx - period
    if si < 0: return None
    return (closes[end_idx] - closes[si]) / closes[si] * 100

def rolling_percentile(vals, cur, window=ROLLING_WINDOW):
    v = vals[~np.isnan(vals)]
    if len(v) < 10: return 50.0
    return np.sum(v < cur) / (len(v) - 1) * 100 if len(v) > 1 else 50.0

def calc_return_percentiles(df, idx):
    closes = df['Close'].values
    results = {}
    for period, bars in LOOKBACK.items():
        cur = compute_return(closes, idx, bars)
        if cur is None:
            results[period] = {'return': 0, 'percentile': 50}; continue
        rolling = []
        start = max(0, idx - ROLLING_WINDOW)
        for i in range(start, idx):
            r = compute_return(closes, i, bars)
            if r is not None: rolling.append(r)
        pctl = rolling_percentile(np.array(rolling), cur) if rolling else 50
        results[period] = {'return': cur, 'percentile': pctl}
    return results

def calc_volume_percentiles(df, idx):
    vols = df['Volume'].values
    results = {}
    for period, bars in LOOKBACK.items():
        si = idx - bars
        if si < 0:
            results[period] = {'volume': 0, 'percentile': 50}; continue
        cur = float(np.sum(vols[si:idx+1]))
        rolling = []
        start = max(0, idx - ROLLING_WINDOW)
        for i in range(start, idx):
            s = i - bars
            if s < 0: continue
            rolling.append(float(np.sum(vols[s:i+1])))
        pctl = rolling_percentile(np.array(rolling), cur) if rolling else 50
        results[period] = {'volume': cur, 'percentile': pctl}
    return results

def weighted_percentile(p):
    return sum(p[k]['percentile'] * WEIGHTS[k] for k in WEIGHT_ORDER)

def calc_rsi(df, idx, period=14):
    if idx < period + 1: return 50.0
    closes = df['Close'].values[max(0, idx-30):idx+1]
    if len(closes) < period+1: return 50.0
    d = np.diff(closes)
    ag = np.mean(np.where(d > 0, d, 0)[-period:])
    al = np.mean(np.where(d < 0, -d, 0)[-period:])
    if al == 0: return 100.0
    return 100 - (100 / (1 + ag / al))

def calc_ma(df, idx, window):
    if idx < window - 1: return None
    return float(np.mean(df['Close'].values[idx-window+1:idx+1]))

def calc_volatility(df, idx, period=24):
    if idx < period: return 0
    closes = df['Close'].values[idx-period:idx+1]
    rets = np.diff(closes) / closes[:-1]
    return float(np.std(rets) * np.sqrt(24 * 252) * 100)

def find_closest_daily_idx(daily_df, target_dt, max_gap=5):
    if daily_df is None: return None
    target_date = pd.Timestamp(target_dt.date()) if hasattr(target_dt, 'date') else pd.Timestamp(target_dt)
    diffs = (daily_df['Date'] - target_date).abs()
    if diffs.min().days > max_gap: return None
    return diffs.idxmin()

def calc_ext_return(ext_df, end_idx, period):
    if ext_df is None or end_idx is None: return None
    si = end_idx - period
    if si < 0: return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[si]) / ext_df['Close'].values[si] * 100


# ══════════════════════════════════════════════════════
# SELL-SIDE SCORING (0-100)
# ══════════════════════════════════════════════════════

def d1_sell_score(wp): return 100 - wp

def d2_sell_score(wp_vol, ret_1m):
    if wp_vol >= 70 and ret_1m < 0: return 80 + (wp_vol - 70) / 30 * 20
    if wp_vol >= 70: return 30 - (wp_vol - 70) / 30 * 10
    if ret_1m < 0: return 40 + (70 - wp_vol) / 70 * 30
    return 10

def d3_sell_score(rsi):
    if rsi < 30: return 100
    if rsi < 40: return 85
    if rsi < 50: return 65
    if rsi > 80: return 65
    if rsi > 70: return 50
    if rsi >= 60: return 20
    return 40

def d4_sell_score(price, ma_s, ma_l):
    pts = 0
    if ma_s is not None and price < ma_s: pts += 35
    if ma_l is not None and price < ma_l: pts += 35
    if ma_s is not None and ma_l is not None and ma_s < ma_l: pts += 30
    return min(pts, 100)

def d5_sell_score(vol):
    if vol > 80: return 100
    if vol > 60: return 90
    if vol > 50: return 75
    if vol > 40: return 60
    if vol > 30: return 40
    if vol > 20: return 20
    return 5

def calc_d6_sell_external(df_h1, h1_idx, df_dxy, df_vix):
    h1_dt = df_h1.iloc[h1_idx]['Datetime']
    gold_1m = compute_return(df_h1['Close'].values, h1_idx, min(504, h1_idx))
    if gold_1m is None: gold_1m = 0
    gold_down = gold_1m < 0

    dxy_score, dxy_1m, dxy_signal = 0, None, "N/A"
    if df_dxy is not None:
        idx = find_closest_daily_idx(df_dxy, h1_dt)
        if idx is not None:
            dxy_1m = calc_ext_return(df_dxy, idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if gold_down and dxy_up: dxy_score, dxy_signal = +5, "🔴 Bearish Confirmed (gold down + strong $)"
                elif gold_down: dxy_score, dxy_signal = +2, "🟠 Gold Weakness (gold down despite weak $)"
                elif dxy_up: dxy_score, dxy_signal = 0, "⚪ Mixed (gold up + strong $)"
                else: dxy_score, dxy_signal = -5, "🟢 Not Bearish (gold up + weak $)"

    vix_score, vix_level, vix_signal = 0, None, "N/A"
    if df_vix is not None:
        idx = find_closest_daily_idx(df_vix, h1_dt)
        if idx is not None:
            vix_level = df_vix['Close'].values[idx]
            if gold_down:
                if vix_level < 20: vix_score, vix_signal = +5, "🔴 No Safe-Haven (VIX<20 + gold down)"
                elif vix_level <= 30: vix_score, vix_signal = +3, "🟠 Fear Not Saving Gold (VIX 20-30 + gold down)"
                else: vix_score, vix_signal = +1, "⚪ Panic Selling Gold Too (VIX>30 + gold down)"
            else:
                if vix_level < 20: vix_score, vix_signal = -3, "🟢 Calm Rally (VIX<20 + gold up)"
                elif vix_level <= 30: vix_score, vix_signal = -2, "🟢 Fear + Gold Up (VIX 20-30)"
                else: vix_score, vix_signal = 0, "⚪ Safe-Haven Rally (VIX>30 + gold up)"

    total = max(min(dxy_score + vix_score, 10), -10)
    return {'d6_total': total, 'd6_scaled': (total + 10) / 20 * 100,
            'dxy_score': dxy_score, 'vix_score': vix_score, 'dxy_1m': dxy_1m,
            'vix_level': vix_level, 'dxy_signal': dxy_signal, 'vix_signal': vix_signal}


# ══════════════════════════════════════════════════════
# SELL PENALTIES (punishes bullish reversals)
# ══════════════════════════════════════════════════════

def calc_sell_penalties(df, idx):
    closes = df['Close'].values
    ret_1m = compute_return(closes, idx, 504) or 0
    ret_2w = compute_return(closes, idx, 240) or 0
    ret_1w = compute_return(closes, idx, 120) or 0
    ret_1d = compute_return(closes, idx, 24) or 0

    rev_pen, rev_flag = 0, ""
    if ret_1m < -10 and ret_1w > 3 and ret_1d > 1:
        rev_pen, rev_flag = -10, "🟢 Strong Bullish Reversal (bad for sell)"
    elif (ret_1m < 0 or ret_2w < 0) and ret_1w > 0 and ret_1d > 0:
        rev_pen, rev_flag = -5, "⚠️ Mild Bullish Reversal (bad for sell)"

    ma_s = calc_ma(df, idx, MA_SHORT)
    ma_l = calc_ma(df, idx, MA_LONG)
    price = closes[idx]
    gc_pen, gc_flag = 0, ""
    if ma_s is not None and ma_l is not None and ma_s > ma_l:
        gc_pen = -5
        gc_flag = "✨✨ Golden Cross + Above MAs (bad for sell)" if price > ma_s and price > ma_l else "✨ Golden Cross (bad for sell)"

    total = max(rev_pen + gc_pen, -15)
    flags = " | ".join(f for f in [rev_flag, gc_flag] if f)
    return {'reversal': rev_pen, 'golden_cross_pen': gc_pen, 'total': total, 'flags': flags,
            'ret_1m': ret_1m, 'ret_2w': ret_2w, 'ret_1w': ret_1w, 'ret_1d': ret_1d}


# ══════════════════════════════════════════════════════
# FULL SELL SCORE
# ══════════════════════════════════════════════════════

def full_sell_score(df, idx, df_dxy, df_vix):
    ret_pctls = calc_return_percentiles(df, idx)
    vol_pctls = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)
    d1 = d1_sell_score(wp_ret)
    ret_1m = compute_return(df['Close'].values, idx, min(504, idx)) or 0
    d2 = d2_sell_score(wp_vol, ret_1m)
    rsi = calc_rsi(df, idx)
    d3 = d3_sell_score(rsi)
    price = df['Close'].values[idx]
    ma_s = calc_ma(df, idx, MA_SHORT)
    ma_l = calc_ma(df, idx, MA_LONG)
    d4 = d4_sell_score(price, ma_s, ma_l)
    vol = calc_volatility(df, idx)
    d5 = d5_sell_score(vol)
    ext = calc_d6_sell_external(df, idx, df_dxy, df_vix)
    d6 = ext['d6_scaled']
    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    penalties = calc_sell_penalties(df, idx)
    penalty_scaled = penalties['total'] * (100 / 110)
    net = gross + penalty_scaled
    golden_cross = (ma_s is not None and ma_l is not None and ma_s > ma_l)
    death_cross = (ma_s is not None and ma_l is not None and ma_s < ma_l)
    return {
        'datetime': df.iloc[idx]['Datetime'], 'price': price,
        'ret_pctls': ret_pctls, 'vol_pctls': vol_pctls,
        'wp_ret': wp_ret, 'wp_vol': wp_vol,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'd6_raw': ext['d6_total'],
        'rsi': rsi, 'ma_short': ma_s, 'ma_long': ma_l,
        'death_cross': death_cross, 'golden_cross': golden_cross, 'volatility': vol,
        'gross': gross, 'penalties': penalties, 'penalty_scaled': penalty_scaled,
        'net': net, 'external': ext
    }

s1 = full_sell_score(df, BD1_idx, df_dxy, df_vix)
s2 = full_sell_score(df, BD2_idx, df_dxy, df_vix)
net_avg = (s1['net'] + s2['net']) / 2
gross_avg = (s1['gross'] + s2['gross']) / 2
delta = s2['net'] - s1['net']

def tier_sell(sc):
    c = max(0, min(100, sc))
    if c >= 85: return "Very Strong Sell ↓↓"
    if c >= 75: return "Strong Sell ↓"
    if c >= 60: return "Moderate Sell ↓"
    if c >= 45: return "Neutral →"
    if c >= 30: return "Weak Sell"
    return "No Sell Signal"

sell_tier = tier_sell(net_avg)

print(f"\nSell Score Avg: {net_avg:.2f} ({sell_tier})")
print(f"BD1: Net={s1['net']:.2f} | BD2: Net={s2['net']:.2f} | Delta: {delta:+.2f}")
print(f"D1s={s2['d1']:.1f} D2s={s2['d2']:.1f} D3s={s2['d3']:.1f} D4s={s2['d4']:.1f} D5s={s2['d5']:.1f} D6s={s2['d6']:.1f}")
print(f"Penalties: {s2['penalties']['total']} ({s2['penalties']['flags'] or 'None'})")

# ══════════════════════════════════════════════════════
# CSV OUTPUT
# ══════════════════════════════════════════════════════

csv_row = {
    'Rank': 1, 'Ticker': 'GOLD_H1', 'Side': 'SELL', 'Timeframe': 'H1',
    'Net_Score_Avg': round(net_avg, 2), 'Gross_Score_Avg': round(gross_avg, 2),
    'Net_Score_BD1': round(s1['net'], 2), 'Net_Score_BD2': round(s2['net'], 2),
    'Score_Delta': round(delta, 2), 'Tier': sell_tier,
    'D1_ReturnRank': round(s2['d1'], 2), 'D2_VolumeRank': round(s2['d2'], 2),
    'D3_RSI': round(s2['d3'], 2), 'D4_MA': round(s2['d4'], 2),
    'D5_Volatility': round(s2['d5'], 2), 'D6_External': round(s2['d6'], 2),
    'D6_Raw': s2['d6_raw'], 'Penalty_Scaled': round(s2['penalty_scaled'], 2),
    'WP_Return_Pct': round(s2['wp_ret'], 2), 'WP_Volume_Pct': round(s2['wp_vol'], 2),
    'Ret_1M_Pct': round(s2['ret_pctls']['1M']['return'], 2),
    'Ret_2W_Pct': round(s2['ret_pctls']['2W']['return'], 2),
    'Ret_1W_Pct': round(s2['ret_pctls']['1W']['return'], 2),
    'Ret_1D_Pct': round(s2['ret_pctls']['1D']['return'], 2),
    'Ret_4H_Pct': round(s2['ret_pctls']['4H']['return'], 2),
    'RSI_Value': round(s2['rsi'], 2),
    'MA_Short': round(s2['ma_short'], 2) if s2['ma_short'] else '',
    'MA_Long': round(s2['ma_long'], 2) if s2['ma_long'] else '',
    'Price': round(s2['price'], 2),
    'Death_Cross': str(s2['death_cross']), 'Golden_Cross': str(s2['golden_cross']),
    'Volatility_Pct': round(s2['volatility'], 2),
    'Penalty_Total': s2['penalties']['total'], 'Penalty_Reversal': s2['penalties']['reversal'],
    'Penalty_GoldenCross': s2['penalties']['golden_cross_pen'],
    'Warning_Flags': s2['penalties']['flags'] if s2['penalties']['flags'] else 'None',
    'DXY_1M_Pct': round(s2['external']['dxy_1m'], 2) if s2['external']['dxy_1m'] is not None else '',
    'VIX_Level': round(s2['external']['vix_level'], 2) if s2['external']['vix_level'] is not None else '',
    'DXY_Signal': s2['external']['dxy_signal'], 'VIX_Signal': s2['external']['vix_signal'],
    'Base_Date_1': s1['datetime'].strftime('%Y-%m-%d %H:%M'),
    'Base_Date_2': s2['datetime'].strftime('%Y-%m-%d %H:%M'),
    'As_Of_Running': AS_OF,
}

csv_df = pd.DataFrame([csv_row])
csv_fixed = os.path.join(base_dir, 'output_momentum_gold_h1_sell.csv')
csv_ts = os.path.join(base_dir, f'output_momentum_gold_h1_sell_{TS_FILE}.csv')
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV saved: {csv_fixed}")
print(f"CSV saved: {csv_ts}")
print(f"\n✅ H1 SELL score complete!")

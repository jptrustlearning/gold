"""
Backfill score_history_h1.csv and score_history_h1_sell.csv
Runs scoring at regular intervals through H1 data to create history for exhaustion detection.

Usage: python3 backfill_h1_history.py [--interval N]
  --interval N : score every N bars (default: 24 = ~1 day)
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

INTERVAL = 24  # score every 24 bars ≈ daily snapshots
for i, arg in enumerate(sys.argv):
    if arg == '--interval' and i + 1 < len(sys.argv):
        INTERVAL = int(sys.argv[i + 1])

base_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Load Data ───
df = pd.read_csv(os.path.join(base_dir, 'gold_prices_h1.csv'), encoding='utf-8-sig')
df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# External data
df_dxy, df_vix = None, None
for fname, target in [('dxy_prices.csv', 'dxy'), ('vix_prices.csv', 'vix')]:
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        tmp = pd.read_csv(fpath, encoding='utf-8-sig')
        tmp.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        tmp['Date'] = pd.to_datetime(tmp['Date'])
        tmp = tmp.sort_values('Date').reset_index(drop=True)
        if target == 'dxy': df_dxy = tmp
        else: df_vix = tmp

print(f"H1 data: {len(df)} rows ({df['Datetime'].min()} → {df['Datetime'].max()})")
print(f"Interval: every {INTERVAL} bars")

# ─── Import scoring functions from buy and sell scripts ───
# We'll define them inline to avoid import side-effects

def compute_return(closes, idx, period):
    if idx < period or period <= 0: return None
    return (closes[idx] - closes[idx - period]) / closes[idx - period] * 100

def calc_rsi(df, idx, period=14):
    start = max(0, idx - 30)
    closes = df['Close'].values[start:idx + 1]
    if len(closes) < period + 1: return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)[-period:]
    losses = np.where(deltas < 0, -deltas, 0)[-period:]
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_ma(df, idx, period):
    if idx < period - 1: return None
    return float(np.mean(df['Close'].values[idx - period + 1:idx + 1]))

def calc_volatility(df, idx, period=24):
    if idx < period:
        return {'abs_vol': 0, 'up_vol': 0, 'down_vol': 0, 'vol_ratio': 1.0}
    closes = df['Close'].values[idx - period:idx + 1]
    rets = np.diff(closes) / closes[:-1]
    abs_vol = float(np.std(rets) * np.sqrt(24 * 252) * 100)
    up_rets = rets[rets > 0]
    down_rets = rets[rets < 0]
    up_vol = float(np.std(up_rets) * np.sqrt(24 * 252) * 100) if len(up_rets) >= 2 else 0
    down_vol = float(np.std(down_rets) * np.sqrt(24 * 252) * 100) if len(down_rets) >= 2 else 0
    if up_vol > 0: vol_ratio = down_vol / up_vol
    elif down_vol > 0: vol_ratio = 999
    else: vol_ratio = 1.0
    return {'abs_vol': abs_vol, 'up_vol': up_vol, 'down_vol': down_vol, 'vol_ratio': vol_ratio}

# Buy D5
def d5_score_buy(vol_data):
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio <= 0.6: return 100
    if ratio <= 0.8: return 85
    if ratio <= 1.0: return 70
    if ratio <= 1.2: return 55
    if ratio <= 1.5: return 40
    if ratio <= 2.0: return 20
    return 10

# Sell D5 (flipped)
def d5_score_sell(vol_data):
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio >= 2.0: return 100
    if ratio >= 1.5: return 85
    if ratio >= 1.2: return 70
    if ratio >= 1.0: return 55
    if ratio >= 0.8: return 40
    if ratio >= 0.6: return 20
    return 10

# H1 periods & weights
PERIODS = {'1M': 504, '2W': 240, '1W': 120, '1D': 24, '4H': 4}
WEIGHTS = {'1M': 0.30, '2W': 0.25, '1W': 0.20, '1D': 0.15, '4H': 0.10}
ROLLING_WINDOW = 504
MA_SHORT, MA_LONG = 120, 480

def calc_return_percentiles(df, idx):
    closes = df['Close'].values
    result = {}
    for label, period in PERIODS.items():
        ret = compute_return(closes, idx, period)
        if ret is None:
            result[label] = {'return': 0, 'percentile': 50}
            continue
        window_start = max(0, idx - ROLLING_WINDOW)
        rets = []
        for i in range(window_start, idx + 1):
            r = compute_return(closes, i, period)
            if r is not None: rets.append(r)
        if len(rets) < 10:
            result[label] = {'return': ret, 'percentile': 50}
            continue
        rank = sum(1 for r in rets if r < ret)
        pctile = rank / (len(rets) - 1) * 100 if len(rets) > 1 else 50
        result[label] = {'return': ret, 'percentile': pctile}
    return result

def calc_volume_percentiles(df, idx):
    vols = df['Volume'].values
    result = {}
    for label, period in PERIODS.items():
        if idx < period:
            result[label] = {'volume': 0, 'percentile': 50}
            continue
        cum_vol = float(np.sum(vols[idx - period + 1:idx + 1]))
        window_start = max(0, idx - ROLLING_WINDOW)
        cvols = []
        for i in range(window_start, idx + 1):
            if i >= period:
                cv = float(np.sum(vols[i - period + 1:i + 1]))
                cvols.append(cv)
        if len(cvols) < 10:
            result[label] = {'volume': cum_vol, 'percentile': 50}
            continue
        rank = sum(1 for v in cvols if v < cum_vol)
        pctile = rank / (len(cvols) - 1) * 100 if len(cvols) > 1 else 50
        result[label] = {'volume': cum_vol, 'percentile': pctile}
    return result

def weighted_percentile(pctls):
    return sum(pctls[k]['percentile'] * WEIGHTS[k] for k in WEIGHTS)

# Buy scoring functions
def d1_buy(wp): return wp
def d2_buy(wp): return wp
def d3_buy(rsi):
    if 50 <= rsi <= 70: return 100
    if 40 <= rsi < 50: return 80
    if 70 < rsi <= 80: return 70
    if 30 <= rsi < 40: return 60
    if rsi > 80: return 50
    return 30

def d4_buy(price, ma_s, ma_l):
    pts = 0
    if ma_s is not None and price > ma_s: pts += 35
    if ma_l is not None and price > ma_l: pts += 35
    if ma_s is not None and ma_l is not None and ma_s > ma_l: pts += 30
    return min(pts, 100)

# Sell scoring functions
def d1_sell(wp): return 100 - wp
def d2_sell(wp_vol, ret_1m):
    if wp_vol >= 70 and ret_1m < 0: return 80 + (wp_vol - 70) / 30 * 20
    if wp_vol >= 70: return 30 - (wp_vol - 70) / 30 * 10
    if ret_1m < 0: return 40 + (70 - wp_vol) / 70 * 30
    return 10
def d3_sell(rsi):
    if rsi < 30: return 100
    if rsi < 40: return 85
    if rsi < 50: return 65
    if rsi > 80: return 65
    if rsi > 70: return 50
    if rsi >= 60: return 20
    return 40
def d4_sell(price, ma_s, ma_l):
    pts = 0
    if ma_s is not None and price < ma_s: pts += 35
    if ma_l is not None and price < ma_l: pts += 35
    if ma_s is not None and ma_l is not None and ma_s < ma_l: pts += 30
    return min(pts, 100)

def find_closest_daily_idx(daily_df, target_dt, max_gap=5):
    """Find closest date ON or BEFORE the target — no look-ahead."""
    if daily_df is None: return None
    target_date = pd.Timestamp(target_dt.date()) if hasattr(target_dt, 'date') else pd.Timestamp(target_dt)
    mask = daily_df['Date'] <= target_date
    if mask.sum() == 0: return None
    candidate_idx = daily_df.loc[mask, 'Date'].idxmax()
    gap = (target_date - daily_df.loc[candidate_idx, 'Date']).days
    if gap > max_gap: return None
    return candidate_idx

def calc_ext_return(ext_df, end_idx, period):
    if ext_df is None or end_idx is None: return None
    si = end_idx - period
    if si < 0: return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[si]) / ext_df['Close'].values[si] * 100

def calc_d6_buy(df_h1, h1_idx, dxy_df, vix_df):
    h1_dt = df_h1.iloc[h1_idx]['Datetime']
    gold_1m = compute_return(df_h1['Close'].values, h1_idx, min(504, h1_idx)) or 0
    gold_up = gold_1m > 0
    score, dxy_1m_val, vix_level = 0, None, None

    if dxy_df is not None:
        idx = find_closest_daily_idx(dxy_df, h1_dt)
        if idx is not None:
            dxy_1m_val = calc_ext_return(dxy_df, idx, 21)
            if dxy_1m_val is not None:
                if gold_up and dxy_1m_val > 0: score += 5
                elif not gold_up and dxy_1m_val > 0: score -= 5
                elif gold_up and dxy_1m_val <= 0: score += 3
                else: score -= 2

    if vix_df is not None:
        idx = find_closest_daily_idx(vix_df, h1_dt)
        if idx is not None:
            vix_level = vix_df['Close'].values[idx]
            if vix_level >= 30:
                score += 5 if gold_up else -3
            elif vix_level >= 20:
                score += 3 if gold_up else -1
            else:
                score += 1 if gold_up else 0

    scaled = max(0, min(100, 50 + score * (50 / 10)))
    return {'d6_scaled': scaled, 'd6_total': score, 'dxy_1m': dxy_1m_val, 'vix_level': vix_level}

def calc_d6_sell(df_h1, h1_idx, dxy_df, vix_df):
    h1_dt = df_h1.iloc[h1_idx]['Datetime']
    gold_1m = compute_return(df_h1['Close'].values, h1_idx, min(504, h1_idx)) or 0
    gold_down = gold_1m < 0
    score, dxy_1m_val, vix_level = 0, None, None

    if dxy_df is not None:
        idx = find_closest_daily_idx(dxy_df, h1_dt)
        if idx is not None:
            dxy_1m_val = calc_ext_return(dxy_df, idx, 21)
            if dxy_1m_val is not None:
                dxy_up = dxy_1m_val > 0
                if gold_down and dxy_up: score += 5
                elif gold_down: score += 2
                elif dxy_up: score += 0
                else: score -= 3

    if vix_df is not None:
        idx = find_closest_daily_idx(vix_df, h1_dt)
        if idx is not None:
            vix_level = vix_df['Close'].values[idx]
            if vix_level >= 30:
                score += 5 if gold_down else 3
            elif vix_level >= 20:
                score += 3 if gold_down else 1
            else:
                score += 0

    scaled = max(0, min(100, 50 + score * (50 / 10)))
    return {'d6_scaled': scaled, 'd6_total': score, 'dxy_1m': dxy_1m_val, 'vix_level': vix_level}

def calc_zscore(closes, idx, bars=1200):
    if idx < bars: return None
    window = closes[idx - bars + 1:idx + 1]
    mean = np.mean(window)
    std = np.std(window, ddof=1)
    return (closes[idx] - mean) / std if std > 0 else 0.0

def z_zone(z):
    if z is None: return 'N/A'
    if z >= 2.5: return 'Extreme Extended'
    if z >= 2.0: return 'Extended'
    if z <= -2.0: return 'Extreme Depressed'
    if z <= -1.5: return 'Depressed'
    return 'Normal'

# Simple penalty for buy
def calc_penalties_buy(df, idx):
    closes = df['Close'].values
    ret_1m = compute_return(closes, idx, 504) or 0
    ret_2w = compute_return(closes, idx, 240) or 0
    ret_1w = compute_return(closes, idx, 120) or 0
    ret_1d = compute_return(closes, idx, 24) or 0

    reversal = 0
    flags = ""
    strong = (ret_1m > 20 and ret_1w < -3 and ret_1d < -1)
    mild = ((ret_1m > 0 or ret_2w > 0) and ret_1w < 0 and ret_1d < 0)
    if strong: reversal, flags = -10, "🔴 Strong Reversal"
    elif mild: reversal, flags = -5, "⚠️ Mild Reversal"

    ma_s = calc_ma(df, idx, MA_SHORT)
    ma_l = calc_ma(df, idx, MA_LONG)
    price = closes[idx]
    dc_pen = 0
    if ma_s is not None and ma_l is not None and ma_s < ma_l:
        dc_pen = -5
        if price < ma_s and price < ma_l:
            flags = (flags + " | " if flags else "") + "💀💀 Death Cross + Below MAs"
        else:
            flags = (flags + " | " if flags else "") + "💀 Death Cross"

    total = max(reversal + dc_pen, -15)
    return {'total': total, 'reversal': reversal, 'death_cross': dc_pen, 'flags': flags}

def calc_penalties_sell(df, idx):
    closes = df['Close'].values
    ma_s = calc_ma(df, idx, MA_SHORT)
    ma_l = calc_ma(df, idx, MA_LONG)
    price = closes[idx]

    gc_pen = 0
    flags = ""
    if ma_s is not None and ma_l is not None and ma_s > ma_l:
        gc_pen = -5
        if price > ma_s and price > ma_l:
            flags = "✨✨ Golden Cross + Above MAs (bad for sell)"
        else:
            flags = "✨ Golden Cross (bad for sell)"

    ret_1m = compute_return(closes, idx, 504) or 0
    ret_1w = compute_return(closes, idx, 120) or 0
    reversal = 0
    if ret_1m < -10 and ret_1w > 3:
        reversal = -5
        flags = (flags + " | " if flags else "") + "⚠️ Sell Reversal (bounce)"

    total = max(gc_pen + reversal, -15)
    return {'total': total, 'golden_cross_pen': gc_pen, 'reversal': reversal, 'flags': flags}


# ─── Main Backfill Loop ───
min_start = 1200  # need at least 1200 bars for Z-Score
total = len(df)
indices = list(range(min_start, total, INTERVAL))
if indices[-1] != total - 1:
    indices.append(total - 1)

buy_rows = []
sell_rows = []
closes = df['Close'].values

print(f"Scoring {len(indices)} points...")

for count, idx in enumerate(indices):
    dt = df.iloc[idx]['Datetime']
    price = closes[idx]

    # Common calcs
    ret_pctls = calc_return_percentiles(df, idx)
    vol_pctls = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)
    rsi = calc_rsi(df, idx)
    ma_s = calc_ma(df, idx, MA_SHORT)
    ma_l = calc_ma(df, idx, MA_LONG)
    vol_data = calc_volatility(df, idx)
    vol = vol_data['abs_vol']

    # Buy scoring
    b_d1 = d1_buy(wp_ret)
    b_d2 = d2_buy(wp_vol)
    b_d3 = d3_buy(rsi)
    b_d4 = d4_buy(price, ma_s, ma_l)
    b_d5 = d5_score_buy(vol_data)
    b_ext = calc_d6_buy(df, idx, df_dxy, df_vix)
    b_d6 = b_ext['d6_scaled']
    b_gross = (b_d1 + b_d2 + b_d3 + b_d4 + b_d5 + b_d6) / 6
    b_pen = calc_penalties_buy(df, idx)
    b_pen_scaled = b_pen['total'] * (100 / 110)
    b_net = b_gross + b_pen_scaled
    golden_cross = (ma_s is not None and ma_l is not None and ma_s > ma_l)

    # Sell scoring
    ret_1m = compute_return(closes, idx, min(504, idx)) or 0
    s_d1 = d1_sell(wp_ret)
    s_d2 = d2_sell(wp_vol, ret_1m)
    s_d3 = d3_sell(rsi)
    s_d4 = d4_sell(price, ma_s, ma_l)
    s_d5 = d5_score_sell(vol_data)
    s_ext = calc_d6_sell(df, idx, df_dxy, df_vix)
    s_d6 = s_ext['d6_scaled']
    s_gross = (s_d1 + s_d2 + s_d3 + s_d4 + s_d5 + s_d6) / 6
    s_pen = calc_penalties_sell(df, idx)
    s_pen_scaled = s_pen['total'] * (100 / 110)
    s_net = s_gross + s_pen_scaled
    death_cross = (ma_s is not None and ma_l is not None and ma_s < ma_l)

    # Z-Score
    z50 = calc_zscore(closes, idx, 1200)
    z_zn = z_zone(z50)

    # Buy history row
    buy_rows.append({
        'Date': dt.strftime('%Y-%m-%d %H:%M'),
        'Price': round(price, 2),
        'Net_Score': round(b_net, 2),
        'Gross_Score': round(b_gross, 2),
        'D1_Return': round(b_d1, 2), 'D2_Volume': round(b_d2, 2),
        'D3_RSI': round(b_d3, 2), 'D4_MA': round(b_d4, 2),
        'D5_DirVol': round(b_d5, 2), 'D6_External': round(b_d6, 2),
        'Vol_Ratio': round(vol_data['vol_ratio'], 3),
        'RSI': round(rsi, 2),
        'Volatility_Pct': round(vol, 2),
        'Penalty_Scaled': round(b_pen_scaled, 2),
        'Ret_1W': round(ret_pctls['1W']['return'], 2),
        'Ret_1M': round(ret_pctls['1M']['return'], 2),
        'Golden_Cross': str(golden_cross),
        'Z_Score_50d': round(z50, 3) if z50 is not None else '',
        'Z_Zone': z_zn,
        'Z_Delta_5d': '',
        'Warning_Flags': b_pen['flags'] if b_pen['flags'] else 'None',
        'Tier': '',
        'As_Of_Running': 'backfill',
        'Exhaust_Scenario': '',
    })

    # Sell history row
    sell_rows.append({
        'Date': dt.strftime('%Y-%m-%d %H:%M'),
        'Price': round(price, 2),
        'Net_Score': round(s_net, 2),
        'Gross_Score': round(s_gross, 2),
        'D1_Return': round(s_d1, 2), 'D2_Volume': round(s_d2, 2),
        'D3_RSI': round(s_d3, 2), 'D4_MA': round(s_d4, 2),
        'D5_DirVol': round(s_d5, 2), 'D6_External': round(s_d6, 2),
        'Vol_Ratio': round(vol_data['vol_ratio'], 3),
        'RSI': round(rsi, 2),
        'Volatility_Pct': round(vol, 2),
        'Penalty_Scaled': round(s_pen_scaled, 2),
        'Ret_1W': round(ret_pctls['1W']['return'], 2),
        'Ret_1M': round(ret_pctls['1M']['return'], 2),
        'Death_Cross': str(death_cross),
        'Golden_Cross': str(golden_cross),
        'Z_Score_50d': round(z50, 3) if z50 is not None else '',
        'Z_Zone': z_zn,
        'Z_Delta_5d': '',
        'Warning_Flags': s_pen['flags'] if s_pen['flags'] else 'None',
        'Tier': '',
        'As_Of_Running': 'backfill',
        'Exhaust_Scenario': '',
    })

    if (count + 1) % 100 == 0:
        print(f"  {count + 1}/{len(indices)} ({dt.strftime('%Y-%m-%d %H:%M')})")

# Save
buy_df = pd.DataFrame(buy_rows)
sell_df = pd.DataFrame(sell_rows)

buy_path = os.path.join(base_dir, 'score_history_h1.csv')
sell_path = os.path.join(base_dir, 'score_history_h1_sell.csv')

buy_df.to_csv(buy_path, index=False, encoding='utf-8')
sell_df.to_csv(sell_path, index=False, encoding='utf-8')

print(f"\n✅ Backfill complete!")
print(f"  Buy history:  {buy_path} ({len(buy_df)} rows)")
print(f"  Sell history: {sell_path} ({len(sell_df)} rows)")
print(f"  Date range:   {buy_df['Date'].iloc[0]} → {buy_df['Date'].iloc[-1]}")

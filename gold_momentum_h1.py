#!/usr/bin/env python3
"""
Gold Momentum Scoring System — H1 (Hourly) Edition
JP Trust Learning

Adapted from daily Gold Momentum Scoring v3.0 (100-Scale Edition)
Uses H1 OHLCV data with intraday-appropriate lookback periods.

Key differences from daily version:
- Lookback periods: 4H, 1D(24H), 1W(120H), 2W(240H), 1M(504H)
- Rolling window: 504 H1 bars (~1 month of hourly data)
- MA periods: MA120 (≈5 days) and MA480 (≈20 days) — intraday equivalents
- RSI: 14-period on H1 bars
- D6 External: uses daily DXY/VIX data (latest available value)
- Base Dates: BD1 ≈ 24 bars before BD2 (≈1 day apart)
- Pivots: H4, D1, W1 timeframes from H1 data

Output files (separate from daily — no conflicts):
- output_momentum_gold_h1.csv (fixed name, overwrite)
- output_momentum_gold_h1_DDMMYYYYHHmm.csv (timestamped log)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import os, sys

# ── CONFIG ──
ROLLING_WINDOW = 504       # ~21 trading days × 24H bars
LOOKBACK = {'4H': 4, '1D': 24, '1W': 120, '2W': 240, '1M': 504}
WEIGHTS = {'1M': 0.30, '2W': 0.25, '1W': 0.20, '1D': 0.15, '4H': 0.10}
WEIGHT_ORDER = ['1M', '2W', '1W', '1D', '4H']

# MA periods for H1 timeframe
MA_SHORT = 120    # ≈5 trading days
MA_LONG = 480     # ≈20 trading days

RUN_TS = datetime.now(timezone.utc)
AS_OF = RUN_TS.strftime("%d/%m/%Y %H:%M UTC")
TS_FILE = RUN_TS.strftime("%d%m%Y_%H%M")

# ── LOAD DATA ──
base_dir = os.path.dirname(os.path.abspath(__file__))


def load_h1_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found — skipping")
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df


def load_daily_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found — skipping")
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


df = load_h1_csv('gold_prices_h1.csv')
df_dxy = load_daily_csv('dxy_prices.csv')
df_vix = load_daily_csv('vix_prices.csv')
df_gold_daily = load_daily_csv('gold_prices.csv')  # for D1/W1 pivots (exact same as daily dashboard)

if df is None:
    print("❌ gold_prices_h1.csv not found — cannot continue")
    sys.exit(1)

# ── BASE DATES ──
# BD2 = latest bar, BD1 = ~24 bars earlier (~1 day)
BD2_idx = len(df) - 1
BD1_idx = max(0, len(df) - 25)
BD1_date = df.iloc[BD1_idx]['Datetime']
BD2_date = df.iloc[BD2_idx]['Datetime']

print(f"Gold Momentum Scoring — H1 (Hourly) Edition")
print(f"{'='*55}")
print(f"Base Date 1: {BD1_date.strftime('%Y-%m-%d %H:%M')} (idx={BD1_idx})")
print(f"Base Date 2: {BD2_date.strftime('%Y-%m-%d %H:%M')} (idx={BD2_idx})")
print(f"Total H1 rows: {len(df)}")
if df_dxy is not None:
    print(f"DXY rows: {len(df_dxy)} (latest: {df_dxy['Date'].max().strftime('%Y-%m-%d')})")
if df_vix is not None:
    print(f"VIX rows: {len(df_vix)} (latest: {df_vix['Date'].max().strftime('%Y-%m-%d')})")


# ══════════════════════════════════════════════════════
# DIMENSION 1: RETURN RANK (Rolling Percentile)
# ══════════════════════════════════════════════════════

def compute_return(closes, end_idx, period_bars):
    start_idx = end_idx - period_bars
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
    for period, bars in LOOKBACK.items():
        current_ret = compute_return(closes, base_idx, bars)
        if current_ret is None:
            results[period] = {'return': 0, 'percentile': 50}
            continue
        rolling_rets = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            r = compute_return(closes, i, bars)
            if r is not None:
                rolling_rets.append(r)
        if len(rolling_rets) < 10:
            results[period] = {'return': current_ret, 'percentile': 50}
            continue
        pctl = rolling_percentile(np.array(rolling_rets), current_ret)
        results[period] = {'return': current_ret, 'percentile': pctl}
    return results


# ══════════════════════════════════════════════════════
# DIMENSION 2: VOLUME RANK (Rolling Percentile)
# ══════════════════════════════════════════════════════

def calc_volume_percentiles(df, base_idx):
    volumes = df['Volume'].values
    results = {}
    for period, bars in LOOKBACK.items():
        start_idx = base_idx - bars
        if start_idx < 0:
            results[period] = {'volume': 0, 'percentile': 50}
            continue
        current_vol = float(np.sum(volumes[start_idx:base_idx + 1]))
        rolling_vols = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            si = i - bars
            if si < 0:
                continue
            v = float(np.sum(volumes[si:i + 1]))
            rolling_vols.append(v)
        if len(rolling_vols) < 10:
            results[period] = {'volume': current_vol, 'percentile': 50}
            continue
        pctl = rolling_percentile(np.array(rolling_vols), current_vol)
        results[period] = {'volume': current_vol, 'percentile': pctl}
    return results


# ══════════════════════════════════════════════════════
# SCORING FUNCTIONS (all 0-100 scale)
# ══════════════════════════════════════════════════════

def weighted_percentile(pctls):
    total = 0
    for period in WEIGHT_ORDER:
        total += pctls[period]['percentile'] * WEIGHTS[period]
    return total


def d1_score(wp):
    """Return Rank: WP directly as 0-100."""
    return wp


def d2_score(wp):
    """Volume Rank: WP directly as 0-100."""
    return wp


def calc_rsi(df, idx, period=14):
    if idx < period + 1:
        return 50.0
    closes = df['Close'].values[max(0, idx - 30):idx + 1]
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def d3_score(rsi):
    """RSI scoring (0-100 scale)."""
    if 50 <= rsi <= 70:
        return 100
    elif 40 <= rsi < 50:
        return 80
    elif 70 < rsi <= 80:
        return 70
    elif 30 <= rsi < 40:
        return 60
    elif rsi > 80:
        return 50
    else:  # < 30
        return 30


def calc_ma(df, idx, period):
    if idx < period - 1:
        return None
    return float(np.mean(df['Close'].values[idx - period + 1:idx + 1]))


def d4_score(price, ma_short, ma_long):
    """MA Trend scoring (0-100 scale). Uses MA_SHORT and MA_LONG."""
    score = 0
    if ma_short is not None and price > ma_short:
        score += 35
    if ma_long is not None and price > ma_long:
        score += 35
    if ma_short is not None and ma_long is not None and ma_short > ma_long:
        score += 30
    return score


def calc_volatility(df, idx, period=24):
    """Directional Volatility — แยก upside/downside vol แล้วคำนวณ ratio (H1 version)."""
    if idx < period:
        return {'abs_vol': 0, 'up_vol': 0, 'down_vol': 0, 'vol_ratio': 1.0}
    closes = df['Close'].values[idx - period:idx + 1]
    rets = np.diff(closes) / closes[:-1]
    abs_vol = float(np.std(rets) * np.sqrt(24 * 252) * 100)

    up_rets = rets[rets > 0]
    down_rets = rets[rets < 0]

    up_vol = float(np.std(up_rets) * np.sqrt(24 * 252) * 100) if len(up_rets) >= 2 else 0
    down_vol = float(np.std(down_rets) * np.sqrt(24 * 252) * 100) if len(down_rets) >= 2 else 0

    if up_vol > 0:
        vol_ratio = down_vol / up_vol
    elif down_vol > 0:
        vol_ratio = 999
    else:
        vol_ratio = 1.0

    return {'abs_vol': abs_vol, 'up_vol': up_vol, 'down_vol': down_vol, 'vol_ratio': vol_ratio}


def d5_score(vol_data):
    """D5 Directional Volatility Score (0-100) — ใช้ vol_ratio (H1 Buy)."""
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio <= 0.6:  return 100
    if ratio <= 0.8:  return 85
    if ratio <= 1.0:  return 70
    if ratio <= 1.2:  return 55
    if ratio <= 1.5:  return 40
    if ratio <= 2.0:  return 20
    return 10


# ══════════════════════════════════════════════════════
# PENALTY SYSTEM
# ══════════════════════════════════════════════════════

def calc_penalties(df, idx):
    closes = df['Close'].values
    # Adapted lookback for H1: use bar counts
    # 1Y equivalent not available in H1 (~6000 bars may not exist)
    # Use 1M(504), 2W(240), 1W(120), 1D(24), 4H(4)
    ret_1m = compute_return(closes, idx, 504)  # ~1 month
    ret_2w = compute_return(closes, idx, 240)  # ~2 weeks
    ret_1w = compute_return(closes, idx, 120)  # ~1 week
    ret_1d = compute_return(closes, idx, 24)   # ~1 day

    if ret_1m is None: ret_1m = 0
    if ret_2w is None: ret_2w = 0
    if ret_1w is None: ret_1w = 0
    if ret_1d is None: ret_1d = 0

    # Penalty 1: Momentum Reversal
    reversal = 0
    reversal_flag = ""

    # Strong Reversal: long-term up >10% but short-term dropping
    if ret_1m > 10 and ret_1w < -3 and ret_1d < -1:
        reversal = -10
        reversal_flag = "🔴 Strong Reversal"
    # Mild Reversal: medium-term up but short-term down
    elif (ret_1m > 0 or ret_2w > 0) and ret_1w < 0 and ret_1d < 0:
        reversal = -5
        reversal_flag = "⚠️ Mild Reversal"

    # Penalty 2: Death Cross (MA_SHORT < MA_LONG)
    ma_short = calc_ma(df, idx, MA_SHORT)
    ma_long = calc_ma(df, idx, MA_LONG)
    price = closes[idx]
    dc_penalty = 0
    dc_flag = ""

    if ma_short is not None and ma_long is not None and ma_short < ma_long:
        dc_penalty = -5
        if price < ma_short and price < ma_long:
            dc_flag = "💀💀 Death Cross + Below MAs"
        else:
            dc_flag = "💀 Death Cross"

    total = max(reversal + dc_penalty, -15)
    flags_list = [f for f in [reversal_flag, dc_flag] if f]

    return {
        'total': total,
        'reversal': reversal,
        'death_cross': dc_penalty,
        'flags': ' | '.join(flags_list) if flags_list else '',
        'ret_1m': ret_1m,
        'ret_2w': ret_2w,
        'ret_1w': ret_1w,
        'ret_1d': ret_1d
    }


# ══════════════════════════════════════════════════════
# DIMENSION 6: EXTERNAL CONTEXT (DXY + VIX) — uses daily data
# ══════════════════════════════════════════════════════

def find_closest_daily_idx(daily_df, target_datetime, max_gap_days=5):
    """Find closest date in daily data that is ON or BEFORE the target datetime.
    No look-ahead: never returns a date after the target."""
    if daily_df is None:
        return None
    target_date = target_datetime.normalize() if hasattr(target_datetime, 'normalize') else pd.Timestamp(target_datetime.date())
    mask = daily_df['Date'] <= target_date
    if mask.sum() == 0:
        return None
    candidate_idx = daily_df.loc[mask, 'Date'].idxmax()
    gap = (target_date - daily_df.loc[candidate_idx, 'Date']).days
    if gap > max_gap_days:
        return None
    return candidate_idx


def calc_external_return(ext_df, end_idx, period_days):
    if ext_df is None or end_idx is None:
        return None
    start_idx = end_idx - period_days
    if start_idx < 0:
        return None
    return (ext_df['Close'].values[end_idx] - ext_df['Close'].values[start_idx]) / ext_df['Close'].values[start_idx] * 100


def calc_d6_external(df_h1, h1_idx, df_dxy, df_vix):
    """
    D6 External Context — uses daily DXY/VIX values.
    Same logic as daily version.
    """
    h1_datetime = df_h1.iloc[h1_idx]['Datetime']
    gold_closes = df_h1['Close'].values
    # Use ~504 bars (1M) for gold trend, converted to approximate daily return
    gold_1m = compute_return(gold_closes, h1_idx, min(504, h1_idx))
    if gold_1m is None:
        gold_1m = 0
    gold_up = gold_1m >= 0

    # Part A: DXY Divergence
    dxy_score = 0
    dxy_1m = None
    dxy_signal = "N/A"

    if df_dxy is not None:
        dxy_idx = find_closest_daily_idx(df_dxy, h1_datetime)
        if dxy_idx is not None:
            dxy_1m = calc_external_return(df_dxy, dxy_idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if gold_up and dxy_up:
                    dxy_score = +5
                    dxy_signal = "🟢 Bullish Divergence (gold up despite strong $)"
                elif gold_up and not dxy_up:
                    dxy_score = +2
                    dxy_signal = "🔵 Normal (gold up + weak $)"
                elif not gold_up and not dxy_up:
                    dxy_score = 0
                    dxy_signal = "⚪ Neutral (both down)"
                else:
                    dxy_score = -5
                    dxy_signal = "🔴 Headwind (gold down + strong $)"

    # Part B: VIX Regime
    vix_score = 0
    vix_level = None
    vix_signal = "N/A"

    if df_vix is not None:
        vix_idx = find_closest_daily_idx(df_vix, h1_datetime)
        if vix_idx is not None:
            vix_level = df_vix['Close'].values[vix_idx]
            if gold_up:
                if vix_level > 30:
                    vix_score = +5
                    vix_signal = "🟢 Safe-Haven Confirmed (VIX>30 + gold up)"
                elif vix_level >= 20:
                    vix_score = +3
                    vix_signal = "🔵 Elevated Fear (VIX 20-30 + gold up)"
                else:
                    vix_score = +1
                    vix_signal = "⚪ Calm Rally (VIX<20 + gold up)"
            else:
                if vix_level > 30:
                    vix_score = -3
                    vix_signal = "🔴 Panic Selling (VIX>30 + gold down)"
                elif vix_level >= 20:
                    vix_score = -2
                    vix_signal = "🟠 Fear Not Helping (VIX 20-30 + gold down)"
                else:
                    vix_score = 0
                    vix_signal = "⚪ Calm Drift (VIX<20 + gold down)"

    total_d6 = max(min(dxy_score + vix_score, 10), -10)
    d6_scaled = (total_d6 + 10) / 20 * 100

    return {
        'd6_total': total_d6,
        'd6_scaled': d6_scaled,
        'dxy_score': dxy_score,
        'vix_score': vix_score,
        'dxy_1m': dxy_1m,
        'vix_level': vix_level,
        'dxy_signal': dxy_signal,
        'vix_signal': vix_signal,
        'gold_1m': gold_1m
    }


# ══════════════════════════════════════════════════════
# MULTI-TF PIVOT POINTS (from H1 data)
# ══════════════════════════════════════════════════════

def calc_pivot_levels(high, low, close):
    PP = (high + low + close) / 3
    R1 = 2 * PP - low
    S1 = 2 * PP - high
    R2 = PP + (high - low)
    S2 = PP - (high - low)
    R3 = high + 2 * (PP - low)
    S3 = low - 2 * (high - PP)
    return {'PP': round(PP, 2), 'R1': round(R1, 2), 'R2': round(R2, 2), 'R3': round(R3, 2),
            'S1': round(S1, 2), 'S2': round(S2, 2), 'S3': round(S3, 2)}


def calc_h1_pivots(df_h1, df_daily=None):
    """
    Calculate pivot points for H4, D1, W1 timeframes.
    - H4: resample from H1 data (same data source, accurate)
    - D1: from daily CSV directly (matches daily dashboard exactly)
    - W1: from daily CSV aggregated weekly (matches daily dashboard exactly)
    """
    df = df_h1.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    result = {}
    current_price = df['Close'].values[-1]

    # ── H4: Resample H1 → 4H, take previous completed bar ──
    df_temp = df.set_index('Datetime')
    h4 = df_temp.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Open'])

    if len(h4) >= 2:
        prev_h4 = h4.iloc[-2]
        levels = calc_pivot_levels(prev_h4['High'], prev_h4['Low'], prev_h4['Close'])
        result['H4'] = {**levels,
                        'H': round(prev_h4['High'], 2), 'L': round(prev_h4['Low'], 2),
                        'C': round(prev_h4['Close'], 2), 'date': str(h4.index[-2])}

    # ── D1 + W1: from daily CSV (exact same source as daily dashboard) ──
    if df_daily is not None and len(df_daily) >= 2:
        df_d = df_daily.copy()
        df_d['Date'] = pd.to_datetime(df_d['Date'])
        df_d = df_d.sort_values('Date').reset_index(drop=True)

        # D1: last completed trading day
        prev = df_d.iloc[-1]
        levels = calc_pivot_levels(prev['High'], prev['Low'], prev['Close'])
        result['D1'] = {**levels,
                        'H': round(prev['High'], 2), 'L': round(prev['Low'], 2),
                        'C': round(prev['Close'], 2), 'date': prev['Date'].strftime('%Y-%m-%d')}

        # W1: previous completed week
        df_d['iso_year'] = df_d['Date'].dt.isocalendar().year.astype(int)
        df_d['iso_week'] = df_d['Date'].dt.isocalendar().week.astype(int)
        df_d['yw_key'] = df_d['iso_year'] * 100 + df_d['iso_week']
        latest_yw = df_d['Date'].max().isocalendar()
        current_yw_key = latest_yw.year * 100 + latest_yw.week
        weekly = df_d[df_d['yw_key'] < current_yw_key].groupby('yw_key').agg(
            High=('High', 'max'), Low=('Low', 'min'), Close=('Close', 'last'),
            Date_last=('Date', 'max')
        ).reset_index().sort_values('yw_key')

        if len(weekly) >= 1:
            prev_w = weekly.iloc[-1]
            levels = calc_pivot_levels(prev_w['High'], prev_w['Low'], prev_w['Close'])
            result['W1'] = {**levels,
                            'H': round(prev_w['High'], 2), 'L': round(prev_w['Low'], 2),
                            'C': round(prev_w['Close'], 2), 'date': prev_w['Date_last'].strftime('%Y-%m-%d')}
    else:
        # Fallback: resample H1 if daily CSV not available
        d1 = df_temp.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna(subset=['Open'])
        if len(d1) >= 2:
            prev_d1 = d1.iloc[-2]
            levels = calc_pivot_levels(prev_d1['High'], prev_d1['Low'], prev_d1['Close'])
            result['D1'] = {**levels,
                            'H': round(prev_d1['High'], 2), 'L': round(prev_d1['Low'], 2),
                            'C': round(prev_d1['Close'], 2), 'date': str(d1.index[-2].date())}

    # ── Confluence Zones ──
    all_levels = []
    for tf in ['H4', 'D1', 'W1']:
        if tf in result:
            for lv in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']:
                all_levels.append({'tf': tf, 'level': lv, 'value': result[tf][lv]})

    confluence_zones = []
    CLUSTER_THRESHOLD = 15  # $15 cluster for intraday

    used = set()
    sorted_levels = sorted(all_levels, key=lambda x: x['value'])
    for i, a in enumerate(sorted_levels):
        if i in used:
            continue
        cluster = [a]
        used.add(i)
        for j, b in enumerate(sorted_levels):
            if j in used or j == i:
                continue
            if abs(a['value'] - b['value']) <= CLUSTER_THRESHOLD:
                cluster.append(b)
                used.add(j)
        if len(cluster) >= 2:
            avg_val = round(np.mean([c['value'] for c in cluster]), 2)
            labels = '+'.join([f"{c['tf']} {c['level']}({c['value']})" for c in cluster])
            zone_type = 'Resistance' if avg_val > current_price else 'Support'
            confluence_zones.append({'avg': avg_val, 'labels': labels, 'type': zone_type, 'count': len(cluster)})

    return result, confluence_zones


def flatten_pivots_for_csv(pivots, confluence_zones, current_price):
    flat = {}
    for tf in ['H4', 'D1', 'W1']:
        if tf in pivots:
            for lv in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']:
                flat[f'Pivot_{tf}_{lv}'] = pivots[tf][lv]
            flat[f'Pivot_{tf}_H'] = pivots[tf]['H']
            flat[f'Pivot_{tf}_L'] = pivots[tf]['L']
            flat[f'Pivot_{tf}_C'] = pivots[tf]['C']
            flat[f'Pivot_{tf}_Date'] = pivots[tf]['date']
        else:
            for lv in ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']:
                flat[f'Pivot_{tf}_{lv}'] = ''
            flat[f'Pivot_{tf}_H'] = ''
            flat[f'Pivot_{tf}_L'] = ''
            flat[f'Pivot_{tf}_C'] = ''
            flat[f'Pivot_{tf}_Date'] = ''

    if confluence_zones:
        zone_strs = [f"~{z['avg']}|{z['labels']}|{z['type']}" for z in confluence_zones[:3]]
        flat['Confluence_Zones'] = ';'.join(zone_strs)
        flat['Confluence_Count'] = len(confluence_zones)
    else:
        flat['Confluence_Zones'] = ''
        flat['Confluence_Count'] = 0

    # Pivot Position (relative to D1 pivot if available)
    if 'D1' in pivots:
        pp = pivots['D1']
        if current_price > pp['R3']:
            flat['Pivot_Position'] = 'Above R3'
        elif current_price > pp['R2']:
            flat['Pivot_Position'] = 'R2-R3'
        elif current_price > pp['R1']:
            flat['Pivot_Position'] = 'R1-R2'
        elif current_price > pp['PP']:
            flat['Pivot_Position'] = 'PP-R1'
        elif current_price > pp['S1']:
            flat['Pivot_Position'] = 'S1-PP'
        elif current_price > pp['S2']:
            flat['Pivot_Position'] = 'S2-S1'
        elif current_price > pp['S3']:
            flat['Pivot_Position'] = 'S3-S2'
        else:
            flat['Pivot_Position'] = 'Below S3'
    else:
        flat['Pivot_Position'] = ''

    return flat


# ══════════════════════════════════════════════════════
# COMPUTE FULL SCORES
# ══════════════════════════════════════════════════════

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
    ma_short = calc_ma(df, idx, MA_SHORT)
    ma_long = calc_ma(df, idx, MA_LONG)
    d4 = d4_score(price, ma_short, ma_long)

    vol_data = calc_volatility(df, idx)
    vol = vol_data['abs_vol']
    d5 = d5_score(vol_data)

    penalties = calc_penalties(df, idx)

    ext = calc_d6_external(df, idx, df_dxy, df_vix)
    d6 = ext['d6_scaled']

    gross_avg_dims = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    penalty_scaled = penalties['total'] * (100 / 110)
    net = gross_avg_dims + penalty_scaled
    golden_cross = (ma_short is not None and ma_long is not None and ma_short > ma_long)

    return {
        'datetime': df.iloc[idx]['Datetime'],
        'price': price,
        'ret_pctls': ret_pctls, 'vol_pctls': vol_pctls,
        'wp_ret': wp_ret, 'wp_vol': wp_vol,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'd6_raw': ext['d6_total'],
        'rsi': rsi, 'ma_short': ma_short, 'ma_long': ma_long,
        'golden_cross': golden_cross, 'volatility': vol,
        'vol_ratio': vol_data['vol_ratio'],
        'gross': gross_avg_dims, 'penalties': penalties,
        'penalty_scaled': penalty_scaled,
        'net': net,
        'external': ext
    }


s1 = full_score(df, BD1_idx, df_dxy, df_vix)
s2 = full_score(df, BD2_idx, df_dxy, df_vix)

net_avg = (s1['net'] + s2['net']) / 2
gross_avg = (s1['gross'] + s2['gross']) / 2
delta = s2['net'] - s1['net']


def tier(score):
    clamped = max(0, min(100, score))
    if clamped >= 85: return "Very Strong ↑↑"
    if clamped >= 75: return "Strong ↑"
    if clamped >= 60: return "Moderate ↑"
    if clamped >= 45: return "Neutral →"
    if clamped >= 30: return "Weak ↓"
    return "Very Weak ↓↓"


momentum_tier = tier(net_avg)

print(f"\n{'='*55}")
print(f"Gold Momentum Score — H1 Edition")
print(f"{'='*55}")
print(f"Net Score Avg:  {net_avg:.2f}  ({momentum_tier})")
print(f"Gross Score Avg: {gross_avg:.2f}")
print(f"BD1 ({s1['datetime'].strftime('%Y-%m-%d %H:%M')}): Net={s1['net']:.2f}  D6={s1['d6']:.1f}/100 (raw {s1['d6_raw']:+d})")
print(f"BD2 ({s2['datetime'].strftime('%Y-%m-%d %H:%M')}): Net={s2['net']:.2f}  D6={s2['d6']:.1f}/100 (raw {s2['d6_raw']:+d})")
print(f"Delta: {delta:+.2f}")
print(f"Price: ${s2['price']:.1f}")
print(f"RSI: {s2['rsi']:.1f} | Volatility: {s2['volatility']:.1f}%")
print(f"MA{MA_SHORT}: {s2['ma_short']:.2f} | MA{MA_LONG}: {s2['ma_long']:.2f}" if s2['ma_long'] else "MA data insufficient")
print(f"Penalties: {s2['penalties']['total']} (scaled: {s2['penalty_scaled']:.1f}) ({s2['penalties']['flags'] or 'None'})")
print(f"\n── External Context (BD2) ──")
print(f"D6 Raw: {s2['d6_raw']:+d} → Scaled: {s2['d6']:.1f}/100")
print(f"  DXY: {s2['external']['dxy_score']:+d}  ({s2['external']['dxy_signal']})")
print(f"  VIX: {s2['external']['vix_score']:+d}  ({s2['external']['vix_signal']})")

# ══════════════════════════════════════════════════════
# PIVOT POINTS
# ══════════════════════════════════════════════════════

pivots, confluence = calc_h1_pivots(df, df_gold_daily)
pivot_csv = flatten_pivots_for_csv(pivots, confluence, s2['price'])

print(f"\nMulti-TF Pivot Points (H4/D1/W1)")
for tf in ['H4', 'D1', 'W1']:
    if tf in pivots:
        p = pivots[tf]
        print(f"  {tf}: PP={p['PP']} | R1={p['R1']} R2={p['R2']} R3={p['R3']} | S1={p['S1']} S2={p['S2']} S3={p['S3']}")

# ══════════════════════════════════════════════════════
# UNIFIED PRICE ZONES (ATR + Pivot + Mean confluence)
# ══════════════════════════════════════════════════════

def calc_atr_h1(df_daily, period=14):
    """Calculate ATR from daily data (same as daily script)."""
    if df_daily is None or len(df_daily) < period + 1:
        return None
    highs = df_daily['High'].values
    lows = df_daily['Low'].values
    closes = df_daily['Close'].values
    n = len(df_daily)
    trs = []
    for i in range(n - period, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    return sum(trs) / len(trs)


def calc_unified_price_zones_h1(df_h1, df_daily, pivots, confluences):
    """
    Unified Price Zones for H1 — merges Pivot Points + ATR targets + Mean levels
    into actionable price zones with confluence scoring.
    ATR uses daily data (same timeframe as daily dashboard).
    MA50/MA200 from daily data. Pivot levels from H4/D1/W1.
    """
    price = df_h1['Close'].values[-1]
    atr14 = calc_atr_h1(df_daily, 14)
    if atr14 is None:
        return [], 0, price

    # MA levels from daily data
    ma50 = float(np.mean(df_daily['Close'].values[-50:])) if df_daily is not None and len(df_daily) >= 50 else None
    ma200 = float(np.mean(df_daily['Close'].values[-200:])) if df_daily is not None and len(df_daily) >= 200 else None

    # ── Candidate levels ──
    candidates = []

    # ATR-based targets
    candidates.append({'label': 'ATR_BULL', 'price': round(price + atr14, 2),
                        'source': 'ATR +1', 'category': 'bull_target'})
    candidates.append({'label': 'ATR_BULL2', 'price': round(price + 2 * atr14, 2),
                        'source': 'ATR +2', 'category': 'breakout'})
    candidates.append({'label': 'ATR_BEAR', 'price': round(price - atr14, 2),
                        'source': 'ATR -1', 'category': 'bear_target'})
    candidates.append({'label': 'ATR_BEAR2', 'price': round(price - 2 * atr14, 2),
                        'source': 'ATR -2', 'category': 'deep_bear'})

    # Mean reversion targets
    if ma50:
        candidates.append({'label': 'MA50', 'price': round(ma50, 2),
                            'source': 'MA50', 'category': 'mean'})
    if ma200:
        candidates.append({'label': 'MA200', 'price': round(ma200, 2),
                            'source': 'MA200', 'category': 'deep_mean'})

    # Key pivot levels from all TFs (H4, D1, W1)
    key_levels = ['R2', 'R1', 'PP', 'S1', 'S2']
    for tf in ['H4', 'D1', 'W1']:
        if tf not in pivots:
            continue
        for lv in key_levels:
            if lv in pivots[tf]:
                cat = 'resistance' if lv.startswith('R') else ('pivot' if lv == 'PP' else 'support')
                candidates.append({'label': f'{tf}_{lv}', 'price': pivots[tf][lv],
                                    'source': f'{tf} {lv}', 'category': cat})

    # ── Merge nearby levels (within 0.5 × ATR) ──
    merge_threshold = atr14 * 0.5
    candidates.sort(key=lambda x: x['price'], reverse=True)

    zones = []
    used = set()
    for i, c in enumerate(candidates):
        if i in used:
            continue
        group = [c]
        for j, d in enumerate(candidates):
            if j <= i or j in used:
                continue
            if abs(c['price'] - d['price']) < merge_threshold:
                group.append(d)
                used.add(j)
        used.add(i)

        avg_price = round(sum(g['price'] for g in group) / len(group), 2)
        sources = [g['source'] for g in group]
        categories = [g['category'] for g in group]

        # Determine zone type
        if avg_price > price:
            if any('breakout' in c for c in categories):
                zone_type = 'BREAKOUT'
            elif any('bull' in c for c in categories):
                zone_type = 'BULL TARGET'
            else:
                zone_type = 'RESISTANCE'
        elif avg_price < price:
            if any('deep' in c for c in categories):
                zone_type = 'DEEP MEAN'
            elif any('bear' in c for c in categories):
                zone_type = 'BEAR TARGET'
            elif any('support' in c for c in categories):
                zone_type = 'SUPPORT'
            else:
                zone_type = 'PULLBACK'
        else:
            zone_type = 'CURRENT'

        conf_score = len(sources)

        # Check confluence cluster overlap
        for cluster in confluences:
            if isinstance(cluster, dict):
                cluster_avg = cluster.get('avg', 0)
            else:
                cluster_avg = sum(x['price'] for x in cluster) / len(cluster) if cluster else 0
            if abs(cluster_avg - avg_price) < merge_threshold:
                conf_score += cluster.get('count', len(cluster)) if isinstance(cluster, dict) else len(cluster)
                break

        dist_pct = round((avg_price - price) / price * 100, 2)
        dist_pts = round(avg_price - price, 2)

        zones.append({
            'label': zone_type,
            'price': avg_price,
            'sources': sources,
            'source_str': ' + '.join(sources),
            'zone_type': zone_type,
            'confluence_score': conf_score,
            'distance_pts': dist_pts,
            'distance_pct': dist_pct,
        })

    zones.sort(key=lambda z: z['price'], reverse=True)

    # Limit to 7 most meaningful zones
    if len(zones) > 7:
        for z in zones:
            z['_priority'] = z['confluence_score'] * 10 + max(0, 5 - abs(z['distance_pct']))
        zones.sort(key=lambda z: z['_priority'], reverse=True)
        zones = zones[:7]
        for z in zones:
            del z['_priority']
        zones.sort(key=lambda z: z['price'], reverse=True)

    return zones, round(atr14, 2), price


def flatten_zones_for_csv(zones, atr14, current_price):
    """Create flat dict of Unified Price Zone data for CSV columns."""
    flat = {
        'ATR_14d': atr14,
        'UPZ_Count': len(zones),
    }
    for i, z in enumerate(zones):
        flat[f'UPZ_{i+1}_Label'] = z['label']
        flat[f'UPZ_{i+1}_Price'] = z['price']
        flat[f'UPZ_{i+1}_Sources'] = z['source_str']
        flat[f'UPZ_{i+1}_Confluence'] = z['confluence_score']
        flat[f'UPZ_{i+1}_DistPts'] = z['distance_pts']
        flat[f'UPZ_{i+1}_DistPct'] = z['distance_pct']
    for i in range(len(zones), 7):
        flat[f'UPZ_{i+1}_Label'] = ''
        flat[f'UPZ_{i+1}_Price'] = ''
        flat[f'UPZ_{i+1}_Sources'] = ''
        flat[f'UPZ_{i+1}_Confluence'] = ''
        flat[f'UPZ_{i+1}_DistPts'] = ''
        flat[f'UPZ_{i+1}_DistPct'] = ''
    return flat


upz_zones, atr_14d, upz_price = calc_unified_price_zones_h1(df, df_gold_daily, pivots, confluence)

print(f"\n{'='*55}")
print(f"Unified Price Zones (ATR 14d = {atr_14d:.2f})")
print(f"{'='*55}")
for z in upz_zones:
    conf_stars = '★' * min(z['confluence_score'], 5)
    arrow = '▲' if z['distance_pts'] > 0 else ('▼' if z['distance_pts'] < 0 else '●')
    print(f"  {arrow} {z['label']:<14} {z['price']:>9.2f}  {z['distance_pts']:>+8.0f} pts ({z['distance_pct']:>+.2f}%)  {conf_stars}  [{z['source_str']}]")
print(f"  ● {'NOW':<14} {upz_price:>9.2f}")

upz_csv = flatten_zones_for_csv(upz_zones, atr_14d, upz_price)

# ══════════════════════════════════════════════════════
# Z-SCORE REGIME FILTER (H1 version — uses H1 close prices)
# ══════════════════════════════════════════════════════

def calc_zscore_regime(df, base_idx):
    """Z-Score regime for H1 data. Uses bar-count equivalent lookbacks.
    50d daily ≈ 50*24=1200 bars, 100d≈2400, 200d≈4800."""
    closes = df['Close'].values[:base_idx + 1]
    result = {}

    for bars, label in [(1200, '50d'), (2400, '100d'), (4800, '200d')]:
        if len(closes) < bars:
            result[f'z_{label}'] = None
            continue
        window = closes[-bars:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        result[f'z_{label}'] = (closes[-1] - mean) / std if std > 0 else 0.0

    z_primary = result.get('z_50d')

    if z_primary is None:
        result['zone'] = 'N/A'
        result['regime'] = 'Insufficient data for Z-Score'
        result['signal'] = '⚪ N/A'
    elif z_primary >= 2.5:
        result['zone'] = 'Extreme Extended'
        result['regime'] = 'ราคาวิ่งเกิน +2.5σ — pullback risk สูงมาก ควรระวังการเปิด Long ใหม่'
        result['signal'] = '🔴 Extreme Extended (Z≥+2.5)'
    elif z_primary >= 2.0:
        result['zone'] = 'Extended'
        result['regime'] = 'ราคาเหนือ +2.0σ — momentum อาจแรงจริง แต่ pullback risk เพิ่มขึ้น'
        result['signal'] = '🟡 Extended (Z≥+2.0)'
    elif z_primary <= -2.0:
        result['zone'] = 'Extreme Depressed'
        result['regime'] = 'ราคาตกเกิน -2.0σ — oversold สุดโต่ง bounce potential สูง'
        result['signal'] = '🟢 Extreme Depressed (Z≤-2.0)'
    elif z_primary <= -1.5:
        result['zone'] = 'Depressed'
        result['regime'] = 'ราคาต่ำกว่า -1.5σ — oversold zone อาจเป็นจุด mean-revert'
        result['signal'] = '🔵 Depressed (Z≤-1.5)'
    else:
        result['zone'] = 'Normal'
        result['regime'] = 'ราคาอยู่ในกรอบปกติ (-1.5σ ถึง +2.0σ) — momentum score ใช้ได้ตามปกติ'
        result['signal'] = '🟢 Normal'

    # Z delta: compare current 50d Z vs approx 120 bars (5 days) ago
    if len(closes) >= 1320:  # 1200 + 120
        closes_ago = closes[:-120]
        w_ago = closes_ago[-1200:]
        m_ago = np.mean(w_ago)
        s_ago = np.std(w_ago, ddof=1)
        if s_ago > 0:
            z_ago = (closes_ago[-1] - m_ago) / s_ago
            result['z_delta_5d'] = result['z_50d'] - z_ago
        else:
            result['z_delta_5d'] = 0.0
    else:
        result['z_delta_5d'] = None

    return result

zscore = calc_zscore_regime(df, BD2_idx)

print(f"\n{'='*55}")
print(f"Z-Score Regime Filter (H1)")
print(f"{'='*55}")
print(f"  Z-Score 50d:  {zscore['z_50d']:.3f}" if zscore['z_50d'] is not None else "  Z-Score 50d:  N/A")
print(f"  Z-Score 100d: {zscore['z_100d']:.3f}" if zscore['z_100d'] is not None else "  Z-Score 100d: N/A")
print(f"  Z-Score 200d: {zscore['z_200d']:.3f}" if zscore['z_200d'] is not None else "  Z-Score 200d: N/A")
print(f"  Zone:         {zscore['zone']}")
print(f"  Signal:       {zscore['signal']}")
if zscore.get('z_delta_5d') is not None:
    zd = zscore['z_delta_5d']
    print(f"  Z Delta 5d:   {zd:+.3f} ({'Z rising' if zd > 0 else 'Z falling' if zd < 0 else 'flat'})")

# ══════════════════════════════════════════════════════
# CSV OUTPUT
# ══════════════════════════════════════════════════════

csv_row = {
    'Rank': 1,
    'Ticker': 'GOLD_H1',
    'Timeframe': 'H1',
    'Net_Score_Avg': round(net_avg, 2),
    'Gross_Score_Avg': round(gross_avg, 2),
    'Net_Score_BD1': round(s1['net'], 2),
    'Net_Score_BD2': round(s2['net'], 2),
    'Score_Delta': round(delta, 2),
    'Tier': momentum_tier,
    'D1_ReturnRank': round(s2['d1'], 2),
    'D2_VolumeRank': round(s2['d2'], 2),
    'D3_RSI': round(s2['d3'], 2),
    'D4_MA': round(s2['d4'], 2),
    'D5_DirVol': round(s2['d5'], 2),
    'D6_External': round(s2['d6'], 2),
    'D6_Raw': s2['d6_raw'],
    'Penalty_Scaled': round(s2['penalty_scaled'], 2),
    'WP_Return_Pct': round(s2['wp_ret'], 2),
    'WP_Volume_Pct': round(s2['wp_vol'], 2),
    'Ret_1M_Pct': round(s2['ret_pctls']['1M']['return'], 2),
    'Ret_2W_Pct': round(s2['ret_pctls']['2W']['return'], 2),
    'Ret_1W_Pct': round(s2['ret_pctls']['1W']['return'], 2),
    'Ret_1D_Pct': round(s2['ret_pctls']['1D']['return'], 2),
    'Ret_4H_Pct': round(s2['ret_pctls']['4H']['return'], 2),
    'RSI_Value': round(s2['rsi'], 2),
    'MA_Short': round(s2['ma_short'], 2) if s2['ma_short'] else '',
    'MA_Long': round(s2['ma_long'], 2) if s2['ma_long'] else '',
    'MA_Short_Period': MA_SHORT,
    'MA_Long_Period': MA_LONG,
    'Price': round(s2['price'], 2),
    'Golden_Cross': str(s2['golden_cross']),
    'Volatility_Pct': round(s2['volatility'], 2),
    'Vol_Ratio': round(s2['vol_ratio'], 3),
    'Penalty_Total': s2['penalties']['total'],
    'Penalty_Reversal': s2['penalties']['reversal'],
    'Penalty_DeathCross': s2['penalties']['death_cross'],
    'Warning_Flags': s2['penalties']['flags'] if s2['penalties']['flags'] else 'None',
    'DXY_1M_Pct': round(s2['external']['dxy_1m'], 2) if s2['external']['dxy_1m'] is not None else '',
    'VIX_Level': round(s2['external']['vix_level'], 2) if s2['external']['vix_level'] is not None else '',
    'DXY_Signal': s2['external']['dxy_signal'],
    'VIX_Signal': s2['external']['vix_signal'],
    'Base_Date_1': s1['datetime'].strftime('%Y-%m-%d %H:%M'),
    'Base_Date_2': s2['datetime'].strftime('%Y-%m-%d %H:%M'),
    'As_Of_Running': AS_OF,
    **pivot_csv,
    **upz_csv,
    # Z-Score Regime Filter
    'Z_Score_50d': round(zscore['z_50d'], 3) if zscore['z_50d'] is not None else '',
    'Z_Score_100d': round(zscore['z_100d'], 3) if zscore['z_100d'] is not None else '',
    'Z_Score_200d': round(zscore['z_200d'], 3) if zscore['z_200d'] is not None else '',
    'Z_Zone': zscore['zone'],
    'Z_Signal': zscore['signal'],
    'Z_Regime': zscore['regime'],
    'Z_Delta_5d': round(zscore['z_delta_5d'], 3) if zscore.get('z_delta_5d') is not None else '',
}

csv_df = pd.DataFrame([csv_row])
csv_fixed = os.path.join(base_dir, 'output_momentum_gold_h1.csv')
csv_ts = os.path.join(base_dir, f'output_momentum_gold_h1_{TS_FILE}.csv')
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV saved: {csv_fixed}")
print(f"CSV saved: {csv_ts}")

# ══════════════════════════════════════════════════════
# SCORE HISTORY — append per-run (for exhaustion detection)
# ══════════════════════════════════════════════════════

history_row = {
    'Date': s2['datetime'].strftime('%Y-%m-%d %H:%M'),
    'Price': round(s2['price'], 2),
    'Net_Score': round(s2['net'], 2),
    'Gross_Score': round(s2['gross'], 2),
    'D1_Return': round(s2['d1'], 2),
    'D2_Volume': round(s2['d2'], 2),
    'D3_RSI': round(s2['d3'], 2),
    'D4_MA': round(s2['d4'], 2),
    'D5_DirVol': round(s2['d5'], 2),
    'D6_External': round(s2['d6'], 2),
    'Vol_Ratio': round(s2['vol_ratio'], 3),
    'RSI': round(s2['rsi'], 2),
    'Volatility_Pct': round(s2['volatility'], 2),
    'Penalty_Scaled': round(s2['penalty_scaled'], 2),
    'Ret_1W': round(s2['ret_pctls']['1W']['return'], 2),
    'Ret_1M': round(s2['ret_pctls']['1M']['return'], 2),
    'Golden_Cross': str(s2['golden_cross']),
    'Z_Score_50d': round(zscore['z_50d'], 3) if zscore['z_50d'] is not None else '',
    'Z_Zone': zscore['zone'],
    'Z_Delta_5d': round(zscore['z_delta_5d'], 3) if zscore.get('z_delta_5d') is not None else '',
    'Warning_Flags': s2['penalties']['flags'] if s2['penalties']['flags'] else 'None',
    'Tier': momentum_tier,
    'As_Of_Running': AS_OF,
}

history_path = os.path.join(base_dir, 'score_history_h1.csv')
history_df = pd.DataFrame([history_row])

if os.path.exists(history_path):
    existing = pd.read_csv(history_path, encoding='utf-8')
    for col in ['Exhaust_Scenario', 'Warning_Flags', 'Tier', 'Z_Zone', 'Golden_Cross', 'As_Of_Running']:
        if col in existing.columns:
            existing[col] = existing[col].fillna('').astype(str)
    existing = existing[existing['Date'] != history_row['Date']]
    history_df = pd.concat([existing, history_df], ignore_index=True)
    history_df = history_df.sort_values('Date').reset_index(drop=True)

history_df.to_csv(history_path, index=False, encoding='utf-8')
print(f"Score history: {history_path} ({len(history_df)} rows)")

# ══════════════════════════════════════════════════════
# EXHAUSTION DETECTION (from score_history_h1)
# ══════════════════════════════════════════════════════

exhaust_result = {
    'scenario': 'None', 'label': '', 'action_override': '',
    'net_5d_change': '', 'max_10d': '', 'min_10d': '', 'd5_shift_5d': '',
}

# H1 runs hourly → "5d" ≈ last 120 rows, "10d" ≈ last 240 rows
# But history has 1 row per run (hourly), so 5d ≈ ~120 rows weekday
# Use index-based: 6 rows back ≈ 6 hours (short), better use date-based
# For simplicity: use last 120 rows for "5d equivalent" and last 240 for "10d"
# BUT history deduplicates by datetime, so each run = 1 row
# With hourly runs, 5 days ≈ 5*24 = 120 rows
H1_5D_ROWS = 120
H1_10D_ROWS = 240

if len(history_df) >= 6:
    h = history_df.copy()
    h['Net_Score'] = pd.to_numeric(h['Net_Score'], errors='coerce')
    h['D5_DirVol'] = pd.to_numeric(h.get('D5_DirVol', h.get('D5_Volatility', 0)), errors='coerce')
    h['Z_Score_50d'] = pd.to_numeric(h['Z_Score_50d'], errors='coerce')

    current = h.iloc[-1]
    net_now = current['Net_Score']
    d5_now = current['D5_DirVol'] if pd.notna(current.get('D5_DirVol')) else 0
    z_now = current['Z_Score_50d'] if pd.notna(current.get('Z_Score_50d')) else 0

    # Use min of available rows vs target
    n5 = min(H1_5D_ROWS, len(h) - 1)
    n10 = min(H1_10D_ROWS, len(h))

    net_5d_ago = h.iloc[-(n5+1)]['Net_Score'] if n5 > 0 else net_now
    net_5d_change = net_now - net_5d_ago

    last_10d = h['Net_Score'].tail(n10)
    max_10d = last_10d.max()
    min_10d = last_10d.min()

    d5_5d_ago_val = h.iloc[-(n5+1)].get('D5_DirVol', h.iloc[-(n5+1)].get('D5_Volatility', d5_now))
    d5_5d_ago = pd.to_numeric(d5_5d_ago_val, errors='coerce')
    if pd.isna(d5_5d_ago): d5_5d_ago = d5_now
    d5_shift = d5_now - d5_5d_ago

    exhaust_result['net_5d_change'] = round(net_5d_change, 2)
    exhaust_result['max_10d'] = round(max_10d, 2)
    exhaust_result['min_10d'] = round(min_10d, 2)
    exhaust_result['d5_shift_5d'] = round(d5_shift, 2)

    sc13 = z_now >= 2.0 and net_now >= 70 and net_5d_change < 0
    sc14 = max_10d >= 80 and net_5d_change < -8 and not sc13
    sc15 = min_10d < 50 and net_5d_change > 3
    sc16 = abs(d5_shift) >= 50 and not sc13 and not sc14 and not sc15

    if sc13:
        exhaust_result['scenario'] = 'Bull Exhaustion'
        exhaust_result['label'] = '🔥 Bull Exhaustion: Z extended + momentum fading → HOLD'
        exhaust_result['action_override'] = 'HOLD'
    elif sc15:
        exhaust_result['scenario'] = 'Bear Exhaustion'
        exhaust_result['label'] = '🔋 Bear Exhaustion: Selling exhausted, bounce likely → BUY'
        exhaust_result['action_override'] = 'BUY'
    elif sc14:
        exhaust_result['scenario'] = 'Topping'
        exhaust_result['label'] = '🏔️ Topping: Score collapsed from recent high → HOLD'
        exhaust_result['action_override'] = 'HOLD'
    elif sc16:
        exhaust_result['scenario'] = 'Vol Shift'
        exhaust_result['label'] = '⚡ Vol Regime Shift: D5 changed ' + str(round(d5_shift)) + 'pts → HOLD'
        exhaust_result['action_override'] = 'HOLD'

    print(f"\n── Exhaustion Detection (H1) ──")
    print(f"  Net {n5}-bar Δ: {net_5d_change:+.2f}")
    print(f"  Max {n10}-bar:  {max_10d:.2f}  |  Min: {min_10d:.2f}")
    print(f"  D5 shift:    {d5_shift:+.0f}")
    print(f"  Z-Score:     {z_now:.3f}")
    if exhaust_result['scenario'] != 'None':
        print(f"  >>> {exhaust_result['label']}")
    else:
        print(f"  >>> No exhaustion signal")

# Add exhaustion columns to main CSV
csv_row['Exhaust_Scenario'] = exhaust_result['scenario']
csv_row['Exhaust_Action'] = exhaust_result['action_override']
csv_row['Net_5d_Change'] = exhaust_result['net_5d_change']
csv_row['Max_10d'] = exhaust_result['max_10d']
csv_row['Min_10d'] = exhaust_result['min_10d']
csv_row['D5_Shift_5d'] = exhaust_result['d5_shift_5d']

# Re-save CSVs with exhaustion columns
csv_df = pd.DataFrame([csv_row])
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV updated with exhaustion: {csv_fixed}")

# Update history with exhaustion info
history_df.loc[history_df['Date'] == history_row['Date'], 'Exhaust_Scenario'] = exhaust_result['scenario']
history_df.to_csv(history_path, index=False, encoding='utf-8')

print(f"\n{'='*55}")
print(f"✅ H1 Momentum Scoring Complete")
print(f"{'='*55}")

# ══════════════════════════════════════════════════════
# PRICE TAIL — last 100 H1 bars for dashboard chart (lightweight)
# ══════════════════════════════════════════════════════
tail_n = min(100, len(df))
df_tail = df.iloc[-tail_n:].copy()
df_tail['Datetime'] = df_tail['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
df_tail_out = df_tail[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_tail_out.columns = ['วันที่', 'ราคาเปิด', 'ราคาสูงสุด', 'ราคาต่ำสุด', 'ราคาปิด', 'ปริมาณซื้อขาย']
tail_path = os.path.join(base_dir, 'gold_prices_h1_tail.csv')
df_tail_out.to_csv(tail_path, index=False, encoding='utf-8-sig')
print(f"Price tail saved: {tail_path} ({tail_n} rows)")


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
    """Annualized volatility from H1 returns. period=24 (1 day of bars)."""
    if idx < period:
        return 0
    closes = df['Close'].values[idx - period:idx + 1]
    rets = np.diff(closes) / closes[:-1]
    # Annualize: sqrt(24 bars/day * 252 trading days)
    return float(np.std(rets) * np.sqrt(24 * 252) * 100)


def d5_score(vol):
    """Volatility scoring (0-100 scale)."""
    if vol <= 20:
        return 100
    elif vol <= 30:
        return 90
    elif vol <= 40:
        return 70
    elif vol <= 50:
        return 55
    elif vol <= 60:
        return 40
    elif vol <= 80:
        return 25
    else:
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
    """Find closest date in daily data to H1 target datetime."""
    if daily_df is None:
        return None
    target_date = target_datetime.normalize() if hasattr(target_datetime, 'normalize') else pd.Timestamp(target_datetime.date())
    diffs = (daily_df['Date'] - target_date).abs()
    min_diff = diffs.min()
    if min_diff.days > max_gap_days:
        return None
    return diffs.idxmin()


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


def calc_h1_pivots(df):
    """
    Calculate pivot points for H4, D1, W1 timeframes from H1 data.
    - H4: previous completed 4-hour block
    - D1: previous completed day
    - W1: previous completed week
    """
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    result = {}
    current_price = df['Close'].values[-1]
    latest_dt = df['Datetime'].max()

    # ── H4: Resample to 4H, take previous completed bar ──
    df_temp = df.set_index('Datetime')
    h4 = df_temp.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Open'])

    if len(h4) >= 2:
        prev_h4 = h4.iloc[-2]  # previous completed H4 bar
        levels = calc_pivot_levels(prev_h4['High'], prev_h4['Low'], prev_h4['Close'])
        result['H4'] = {**levels,
                        'H': round(prev_h4['High'], 2), 'L': round(prev_h4['Low'], 2),
                        'C': round(prev_h4['Close'], 2), 'date': str(h4.index[-2])}

    # ── D1: Resample to daily, take previous completed day ──
    d1 = df_temp.resample('1D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Open'])

    if len(d1) >= 2:
        prev_d1 = d1.iloc[-2]
        levels = calc_pivot_levels(prev_d1['High'], prev_d1['Low'], prev_d1['Close'])
        result['D1'] = {**levels,
                        'H': round(prev_d1['High'], 2), 'L': round(prev_d1['Low'], 2),
                        'C': round(prev_d1['Close'], 2), 'date': str(d1.index[-2].date())}

    # ── W1: Resample to weekly, take previous completed week ──
    w1 = df_temp.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Open'])

    if len(w1) >= 2:
        prev_w1 = w1.iloc[-2]
        levels = calc_pivot_levels(prev_w1['High'], prev_w1['Low'], prev_w1['Close'])
        result['W1'] = {**levels,
                        'H': round(prev_w1['High'], 2), 'L': round(prev_w1['Low'], 2),
                        'C': round(prev_w1['Close'], 2), 'date': str(w1.index[-2].date())}

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

    vol = calc_volatility(df, idx)
    d5 = d5_score(vol)

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

pivots, confluence = calc_h1_pivots(df)
pivot_csv = flatten_pivots_for_csv(pivots, confluence, s2['price'])

print(f"\nMulti-TF Pivot Points (H4/D1/W1)")
for tf in ['H4', 'D1', 'W1']:
    if tf in pivots:
        p = pivots[tf]
        print(f"  {tf}: PP={p['PP']} | R1={p['R1']} R2={p['R2']} R3={p['R3']} | S1={p['S1']} S2={p['S2']} S3={p['S3']}")

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
    'D5_Volatility': round(s2['d5'], 2),
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
    **pivot_csv
}

csv_df = pd.DataFrame([csv_row])
csv_fixed = os.path.join(base_dir, 'output_momentum_gold_h1.csv')
csv_ts = os.path.join(base_dir, f'output_momentum_gold_h1_{TS_FILE}.csv')
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV saved: {csv_fixed}")
print(f"CSV saved: {csv_ts}")

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


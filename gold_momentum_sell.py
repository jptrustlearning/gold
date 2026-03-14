#!/usr/bin/env python3
"""
Gold Momentum Scoring System v3.0 — SELL Side
JP Trust Learning

Mirror of the Buy score: gives HIGH scores when bearish conditions are strong.
Each dimension 0-100, total = average of D1-D6, equal weight (1/6).

Logic summary (vs Buy):
  D1s Return Rank:  return LOW percentile = high score (dropping hard vs history)
  D2s Volume Rank:  volume HIGH + price dropping = high score (selling pressure)
  D3s RSI:          RSI<30 = 100 (oversold = strong bearish), RSI 50-70 = low
  D4s MA Trend:     Price<MA50, Price<MA200, Death Cross = high score
  D5s Volatility:   HIGH vol = high score (panic/fear = sell opportunity)
  D6s External:     DXY strong + VIX low (no safe-haven) = bearish for gold

Penalty (sell side): punishes bullish reversal signals
  - Bullish Reversal: price was falling but short-term bouncing back
  - Golden Cross: MA50 crossing above MA200
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os, sys

# ── CONFIG (same as buy) ──
ROLLING_WINDOW = 252
LOOKBACK = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}
WEIGHTS = {'1Y': 0.30, '6M': 0.25, '3M': 0.20, '1M': 0.15, '1W': 0.10}
WEIGHT_ORDER = ['1Y', '6M', '3M', '1M', '1W']

RUN_TS = datetime.now(timezone.utc)
AS_OF = RUN_TS.strftime("%d/%m/%Y %H:%M UTC")
TS_FILE = RUN_TS.strftime("%d%m%Y_%H%M")

# ── LOAD DATA ──
base_dir = os.path.dirname(os.path.abspath(__file__))

def load_price_csv(filename):
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ {filename} not found — skipping")
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
    print("❌ gold_prices.csv not found — cannot continue")
    sys.exit(1)

# ── BASE DATES ──
BD2_idx = len(df) - 1
BD1_idx = len(df) - 6
BD1_date = df.iloc[BD1_idx]['Date']
BD2_date = df.iloc[BD2_idx]['Date']

print(f"Gold Momentum Scoring v3.0 — SELL Side")
print(f"{'='*55}")
print(f"Base Date 1: {BD1_date.strftime('%Y-%m-%d')} (idx={BD1_idx})")
print(f"Base Date 2: {BD2_date.strftime('%Y-%m-%d')} (idx={BD2_idx})")
print(f"Total gold rows: {len(df)}")
if df_dxy is not None:
    print(f"DXY rows: {len(df_dxy)} (latest: {df_dxy['Date'].max().strftime('%Y-%m-%d')})")
if df_vix is not None:
    print(f"VIX rows: {len(df_vix)} (latest: {df_vix['Date'].max().strftime('%Y-%m-%d')})")


# ══════════════════════════════════════════════════════
# SHARED CALCULATIONS (same as buy)
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
        current_ret = compute_return(closes, base_idx, days)
        if current_ret is None:
            results[period] = {'return': 0, 'percentile': 50}
            continue
        rolling_rets = []
        start = max(0, base_idx - ROLLING_WINDOW)
        for i in range(start, base_idx):
            r = compute_return(closes, i, days)
            if r is not None:
                rolling_rets.append(r)
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
            if s < 0:
                continue
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
    if len(closes) < 2:
        return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    last_n = min(period, len(gains))
    avg_gain = np.mean(gains[-last_n:])
    avg_loss = np.mean(losses[-last_n:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_ma(df, base_idx, window):
    start = base_idx + 1 - window
    if start < 0: return None
    return np.mean(df['Close'].values[start:base_idx+1])

def calc_volatility(df, base_idx):
    """Directional Volatility — แยก upside/downside vol แล้วคำนวณ ratio"""
    start = base_idx - 20
    if start < 0: start = 0
    closes = df['Close'].values[start:base_idx+1]
    if len(closes) < 2:
        return {'abs_vol': 0, 'up_vol': 0, 'down_vol': 0, 'vol_ratio': 1.0}
    rets = np.diff(closes) / closes[:-1]
    abs_vol = float(np.std(rets) * np.sqrt(252) * 100)

    up_rets = rets[rets > 0]
    down_rets = rets[rets < 0]

    up_vol = float(np.std(up_rets) * np.sqrt(252) * 100) if len(up_rets) >= 2 else 0
    down_vol = float(np.std(down_rets) * np.sqrt(252) * 100) if len(down_rets) >= 2 else 0

    if up_vol > 0:
        vol_ratio = down_vol / up_vol
    elif down_vol > 0:
        vol_ratio = 999
    else:
        vol_ratio = 1.0

    return {'abs_vol': abs_vol, 'up_vol': up_vol, 'down_vol': down_vol, 'vol_ratio': vol_ratio}

def find_closest_idx(ext_df, target_date, max_gap_days=5):
    if ext_df is None:
        return None
    diffs = (ext_df['Date'] - target_date).abs()
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


# ══════════════════════════════════════════════════════
# SELL-SIDE SCORING FUNCTIONS (0-100 each)
# ══════════════════════════════════════════════════════

def d1_sell_score(wp_return):
    """
    D1s Return Rank (Sell): FLIP the percentile.
    Low return percentile = price dropping harder than usual = strong sell signal.
    wp_return is 0-100 where 100 = returns are high vs history.
    For sell: invert it → 100 - wp = score.
    """
    return 100 - wp_return

def d2_sell_score(wp_volume, ret_1m):
    """
    D2s Volume Rank (Sell): High volume + negative returns = selling pressure.
    High volume alone is ambiguous (could be buying), so we weight by direction.
    
    - Volume high (pctl>70) + returns negative → 100 (distribution/selling)
    - Volume high + returns positive → 30 (accumulation, not sell signal)
    - Volume low + returns negative → 60 (quiet decline, moderate sell)
    - Volume low + returns positive → 10 (quiet rally, no sell signal)
    """
    if wp_volume >= 70 and ret_1m < 0:
        # Strong selling pressure: scale 70-100 pctl → 80-100 score
        return 80 + (wp_volume - 70) / 30 * 20
    elif wp_volume >= 70 and ret_1m >= 0:
        # High volume but price up = not a sell signal
        return 30 - (wp_volume - 70) / 30 * 10  # 30→20
    elif wp_volume < 70 and ret_1m < 0:
        # Quiet decline: moderate sell signal
        # Deeper decline (lower pctl ≠ volume, but return is negative) → 40-70
        return 40 + (70 - wp_volume) / 70 * 30
    else:
        # Low volume + positive return: no sell signal
        return 10

def d3_sell_score(rsi):
    """
    D3s RSI (Sell): Low RSI = strong bearish momentum.
    
    RSI < 30   → 100 (oversold = strong selling pressure, bearish momentum)
    RSI 30-40  → 85  (approaching oversold, bearish)
    RSI 40-50  → 65  (below midline, mildly bearish)
    RSI 50-60  → 40  (neutral, not bearish)
    RSI 60-70  → 20  (bullish zone, not a sell)
    RSI > 70   → 50  (overbought = potential reversal, somewhat sell-worthy)
    RSI > 80   → 65  (extremely overbought = likely reversal down)
    """
    if rsi < 30:   return 100
    if rsi < 40:   return 85
    if rsi < 50:   return 65
    if rsi > 80:   return 65   # overbought → reversal likely
    if rsi > 70:   return 50   # overbought zone
    if rsi >= 60:  return 20   # bullish, not a sell
    return 40                  # 50-60 neutral

def d4_sell_score(price, ma50, ma200):
    """
    D4s MA Trend (Sell): Price below MAs + Death Cross = bearish trend confirmed.
    
    Price < MA50  → +35
    Price < MA200 → +35
    Death Cross (MA50 < MA200) → +30
    Max: 100
    """
    pts = 0
    if ma50 is not None and price < ma50: pts += 35
    if ma200 is not None and price < ma200: pts += 35
    if ma50 is not None and ma200 is not None and ma50 < ma200: pts += 30
    return min(pts, 100)

def d5_sell_score(vol_data):
    """
    D5s Directional Volatility (Sell): FLIPPED — high vol_ratio = downside dominant = strong sell.
    vol_ratio = down_vol / up_vol
    High ratio means downside volatility dominates → panic/fear → sell signal strong.
    """
    ratio = vol_data['vol_ratio'] if isinstance(vol_data, dict) else 1.0
    if ratio >= 2.0:  return 100   # downside dominant มาก
    if ratio >= 1.5:  return 85
    if ratio >= 1.2:  return 70
    if ratio >= 1.0:  return 55    # balanced
    if ratio >= 0.8:  return 40
    if ratio >= 0.6:  return 20
    return 10                       # upside dominant → no sell pressure

def calc_d6_sell_external(df_gold, gold_idx, df_dxy, df_vix):
    """
    D6s External Context — SELL Side
    
    Bearish for gold when:
    - DXY strong (dollar up) = gold headwind
    - VIX low (no fear) = no safe-haven demand for gold
    
    Part A — DXY (±5 pts, INVERTED from buy):
      Gold down + DXY up = +5 (strong headwind confirmed)
      Gold down + DXY down = +2 (gold weak even with weak dollar)
      Gold up + DXY up = 0 (mixed)
      Gold up + DXY down = -5 (gold bullish, not a sell)
    
    Part B — VIX Regime (±5 pts, INVERTED):
      VIX < 20 + Gold down = +5 (no fear, gold just weak on its own)
      VIX 20-30 + Gold down = +3 (some fear but gold still dropping)
      VIX > 30 + Gold down = +1 (panic but gold still selling = extreme)
      VIX < 20 + Gold up = -3 (calm + gold rising = not sell)
      VIX 20-30 + Gold up = -2
      VIX > 30 + Gold up = 0 (safe-haven rally, neutral for sell)
    """
    gold_date = df_gold.iloc[gold_idx]['Date']
    gold_closes = df_gold['Close'].values
    gold_1m = compute_return(gold_closes, gold_idx, 21)
    if gold_1m is None:
        gold_1m = 0
    gold_down = gold_1m < 0

    # ── Part A: DXY ──
    dxy_score = 0
    dxy_1m = None
    dxy_signal = "N/A"
    
    if df_dxy is not None:
        dxy_idx = find_closest_idx(df_dxy, gold_date)
        if dxy_idx is not None:
            dxy_1m = calc_external_return(df_dxy, dxy_idx, 21)
            if dxy_1m is not None:
                dxy_up = dxy_1m > 0
                if gold_down and dxy_up:
                    dxy_score = +5
                    dxy_signal = "🔴 Bearish Confirmed (gold down + strong $)"
                elif gold_down and not dxy_up:
                    dxy_score = +2
                    dxy_signal = "🟠 Gold Weakness (gold down despite weak $)"
                elif not gold_down and dxy_up:
                    dxy_score = 0
                    dxy_signal = "⚪ Mixed (gold up + strong $)"
                else:  # gold up, dxy down
                    dxy_score = -5
                    dxy_signal = "🟢 Not Bearish (gold up + weak $)"

    # ── Part B: VIX ──
    vix_score = 0
    vix_level = None
    vix_signal = "N/A"
    
    if df_vix is not None:
        vix_idx = find_closest_idx(df_vix, gold_date)
        if vix_idx is not None:
            vix_level = df_vix['Close'].values[vix_idx]
            if gold_down:
                if vix_level < 20:
                    vix_score = +5
                    vix_signal = "🔴 No Safe-Haven (VIX<20 + gold down)"
                elif vix_level <= 30:
                    vix_score = +3
                    vix_signal = "🟠 Fear Not Saving Gold (VIX 20-30 + gold down)"
                else:
                    vix_score = +1
                    vix_signal = "⚪ Panic Selling Gold Too (VIX>30 + gold down)"
            else:
                if vix_level < 20:
                    vix_score = -3
                    vix_signal = "🟢 Calm Rally (VIX<20 + gold up)"
                elif vix_level <= 30:
                    vix_score = -2
                    vix_signal = "🟢 Fear + Gold Up (VIX 20-30)"
                else:
                    vix_score = 0
                    vix_signal = "⚪ Safe-Haven Rally (VIX>30 + gold up)"

    total_d6 = max(min(dxy_score + vix_score, 10), -10)
    d6_scaled = (total_d6 + 10) / 20 * 100  # ±10 → 0-100

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
# SELL-SIDE PENALTY: Punishes BULLISH reversals
# ══════════════════════════════════════════════════════

def calc_sell_penalties(df, base_idx):
    """
    Mirror of buy penalties. Punishes when bearish trend shows bullish reversal.
    
    Bullish Reversal (bad for sell position):
      Mild:   1Y<0 or 6M<0 (was bearish) AND 1M>0 AND 1W>0 (bouncing) → -5
      Strong: 1Y<-20 AND 1M>+5 AND 1W>+3 (was deep bearish, now reversing hard) → -10
    
    Golden Cross (bad for sell):
      MA50 > MA200 → -5
      Aggravated: MA50 > MA200 AND price > both → -5 (same pts, worse flag)
    """
    closes = df['Close'].values
    ret_1y = compute_return(closes, base_idx, 252) or 0
    ret_6m = compute_return(closes, base_idx, 126) or 0
    ret_1m = compute_return(closes, base_idx, 21) or 0
    ret_1w = compute_return(closes, base_idx, 5) or 0

    reversal_pen = 0
    reversal_flag = ""
    # Strong bullish reversal: was deep bearish, now bouncing hard
    strong = (ret_1y < -20 and ret_1m > 5 and ret_1w > 3)
    # Mild bullish reversal: was bearish, now short-term positive
    mild = ((ret_1y < 0 or ret_6m < 0) and ret_1m > 0 and ret_1w > 0)
    if strong:
        reversal_pen = -10
        reversal_flag = "🟢 Strong Bullish Reversal (bad for sell)"
    elif mild:
        reversal_pen = -5
        reversal_flag = "⚠️ Mild Bullish Reversal (bad for sell)"

    ma50 = calc_ma(df, base_idx, 50)
    ma200 = calc_ma(df, base_idx, 200)
    price = closes[base_idx]
    gc_pen = 0
    gc_flag = ""
    if ma50 is not None and ma200 is not None and ma50 > ma200:
        gc_pen = -5
        if price > ma50 and price > ma200:
            gc_flag = "✨✨ Golden Cross + Above MAs (bad for sell)"
        else:
            gc_flag = "✨ Golden Cross (bad for sell)"

    total = max(reversal_pen + gc_pen, -15)
    flags = " | ".join(f for f in [reversal_flag, gc_flag] if f)
    return {
        'reversal': reversal_pen, 'golden_cross_pen': gc_pen,
        'total': total, 'flags': flags,
        'ret_1y': ret_1y, 'ret_6m': ret_6m, 'ret_1m': ret_1m, 'ret_1w': ret_1w
    }


# ══════════════════════════════════════════════════════
# COMPUTE FULL SELL SCORES
# ══════════════════════════════════════════════════════

def full_sell_score(df, idx, df_dxy, df_vix):
    ret_pctls = calc_return_percentiles(df, idx)
    vol_pctls = calc_volume_percentiles(df, idx)
    wp_ret = weighted_percentile(ret_pctls)
    wp_vol = weighted_percentile(vol_pctls)
    
    # Sell-side D1: flip return percentile
    d1 = d1_sell_score(wp_ret)
    
    # Sell-side D2: volume + direction
    ret_1m = compute_return(df['Close'].values, idx, 21) or 0
    d2 = d2_sell_score(wp_vol, ret_1m)
    
    # Sell-side D3: RSI
    rsi = calc_rsi(df, idx)
    d3 = d3_sell_score(rsi)
    
    # Sell-side D4: MA (flipped)
    price = df['Close'].values[idx]
    ma50 = calc_ma(df, idx, 50)
    ma200 = calc_ma(df, idx, 200)
    d4 = d4_sell_score(price, ma50, ma200)
    
    # Sell-side D5: Volatility (flipped — uses vol_ratio)
    vol_data = calc_volatility(df, idx)
    vol = vol_data['abs_vol']
    d5 = d5_sell_score(vol_data)
    
    # Sell-side D6: External (inverted)
    ext = calc_d6_sell_external(df, idx, df_dxy, df_vix)
    d6 = ext['d6_scaled']
    
    # Gross = average of all 6 dims (equal weight)
    gross = (d1 + d2 + d3 + d4 + d5 + d6) / 6
    
    # Sell penalties (punish bullish reversals)
    penalties = calc_sell_penalties(df, idx)
    penalty_scaled = penalties['total'] * (100 / 110)
    
    # Net Score
    net = gross + penalty_scaled
    
    # Death Cross (sell-side positive indicator, but track for reference)
    death_cross = (ma50 is not None and ma200 is not None and ma50 < ma200)
    golden_cross = (ma50 is not None and ma200 is not None and ma50 > ma200)
    
    return {
        'date': df.iloc[idx]['Date'],
        'price': price,
        'ret_pctls': ret_pctls, 'vol_pctls': vol_pctls,
        'wp_ret': wp_ret, 'wp_vol': wp_vol,
        'd1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5, 'd6': d6,
        'd6_raw': ext['d6_total'],
        'rsi': rsi, 'ma50': ma50, 'ma200': ma200,
        'death_cross': death_cross, 'golden_cross': golden_cross,
        'volatility': vol, 'vol_ratio': vol_data['vol_ratio'],
        'gross': gross, 'penalties': penalties,
        'penalty_scaled': penalty_scaled,
        'net': net,
        'external': ext
    }


# ── Compute ──
s1 = full_sell_score(df, BD1_idx, df_dxy, df_vix)
s2 = full_sell_score(df, BD2_idx, df_dxy, df_vix)

net_avg = (s1['net'] + s2['net']) / 2
gross_avg = (s1['gross'] + s2['gross']) / 2
delta = s2['net'] - s1['net']

def tier_sell(score):
    clamped = max(0, min(100, score))
    if clamped >= 85: return "Very Strong Sell ↓↓"
    if clamped >= 75: return "Strong Sell ↓"
    if clamped >= 60: return "Moderate Sell ↓"
    if clamped >= 45: return "Neutral →"
    if clamped >= 30: return "Weak Sell"
    return "No Sell Signal"

sell_tier = tier_sell(net_avg)

print(f"\n{'='*55}")
print(f"Gold SELL Momentum Score v3.0")
print(f"{'='*55}")
print(f"Sell Score Avg:  {net_avg:.2f}  ({sell_tier})")
print(f"Gross Score Avg: {gross_avg:.2f}")
print(f"BD1 ({s1['date'].strftime('%Y-%m-%d')}): Net={s1['net']:.2f}")
print(f"BD2 ({s2['date'].strftime('%Y-%m-%d')}): Net={s2['net']:.2f}")
print(f"Delta: {delta:+.2f}")
print(f"Price: ${s2['price']:.1f}")
print(f"D1s={s2['d1']:.1f}  D2s={s2['d2']:.1f}  D3s={s2['d3']:.1f}  D4s={s2['d4']:.1f}  D5s={s2['d5']:.1f}  D6s={s2['d6']:.1f}")
print(f"Penalties: {s2['penalties']['total']} (scaled: {s2['penalty_scaled']:.1f}) ({s2['penalties']['flags'] or 'None'})")
print(f"\n── External Context SELL (BD2) ──")
print(f"D6s Raw: {s2['d6_raw']:+d} → Scaled: {s2['d6']:.1f}/100")
print(f"  DXY: {s2['external']['dxy_score']:+d}  ({s2['external']['dxy_signal']})")
print(f"  VIX: {s2['external']['vix_score']:+d}  ({s2['external']['vix_signal']})")


# ══════════════════════════════════════════════════════
# Z-SCORE REGIME FILTER
# ══════════════════════════════════════════════════════

def calc_zscore_regime(df, base_idx):
    """
    Z-Score regime — same as buy script.
    Asymmetric thresholds calibrated for gold's positive drift bias.
    """
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
        result['regime'] = 'Insufficient data for Z-Score'
        result['signal'] = '⚪ N/A'
    elif z_primary >= 2.5:
        result['zone'] = 'Extreme Extended'
        result['regime'] = 'ราคาวิ่งเกิน +2.5σ — pullback risk สูงมาก'
        result['signal'] = '🔴 Extreme Extended (Z≥+2.5)'
    elif z_primary >= 2.0:
        result['zone'] = 'Extended'
        result['regime'] = 'ราคาเหนือ +2.0σ — pullback risk เพิ่มขึ้น'
        result['signal'] = '🟡 Extended (Z≥+2.0)'
    elif z_primary <= -2.0:
        result['zone'] = 'Extreme Depressed'
        result['regime'] = 'ราคาตกเกิน -2.0σ — oversold สุดโต่ง'
        result['signal'] = '🟢 Extreme Depressed (Z≤-2.0)'
    elif z_primary <= -1.5:
        result['zone'] = 'Depressed'
        result['regime'] = 'ราคาต่ำกว่า -1.5σ — oversold zone'
        result['signal'] = '🔵 Depressed (Z≤-1.5)'
    else:
        result['zone'] = 'Normal'
        result['regime'] = 'ราคาอยู่ในกรอบปกติ (-1.5σ ถึง +2.0σ)'
        result['signal'] = '🟢 Normal'

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

# Compute Z-Score for BD2 (latest)
zscore = calc_zscore_regime(df, BD2_idx)

print(f"\n{'='*55}")
print(f"Z-Score Regime Filter (SELL)")
print(f"{'='*55}")
print(f"  Z-Score 50d:  {zscore['z_50d']:.3f}" if zscore['z_50d'] is not None else "  Z-Score 50d:  N/A")
print(f"  Z-Score 100d: {zscore['z_100d']:.3f}" if zscore['z_100d'] is not None else "  Z-Score 100d: N/A")
print(f"  Z-Score 200d: {zscore['z_200d']:.3f}" if zscore['z_200d'] is not None else "  Z-Score 200d: N/A")
print(f"  Zone:         {zscore['zone']}")
print(f"  Signal:       {zscore['signal']}")
if zscore.get('z_delta_5d') is not None:
    zd5 = zscore['z_delta_5d']
    print(f"  Z Delta 5d:   {zd5:+.3f} ({'Z rising — extending' if zd5 > 0 else 'Z falling — reverting' if zd5 < 0 else 'flat'})")


# ══════════════════════════════════════════════════════
# CSV OUTPUT — SELL
# ══════════════════════════════════════════════════════

csv_row = {
    'Rank': 1,
    'Ticker': 'GOLD',
    'Side': 'SELL',
    'Net_Score_Avg': round(net_avg, 2),
    'Gross_Score_Avg': round(gross_avg, 2),
    'Net_Score_BD1': round(s1['net'], 2),
    'Net_Score_BD2': round(s2['net'], 2),
    'Score_Delta': round(delta, 2),
    'Tier': sell_tier,
    'D1_ReturnRank': round(s2['d1'], 2),
    'D2_VolumeRank': round(s2['d2'], 2),
    'D3_RSI': round(s2['d3'], 2),
    'D4_MA': round(s2['d4'], 2),
    'D5_DirVol': round(s2['d5'], 2),
    'D6_External': round(s2['d6'], 2),
    'D6_Raw': s2['d6_raw'],
    'Vol_Ratio': round(s2['vol_ratio'], 3),
    'Penalty_Scaled': round(s2['penalty_scaled'], 2),
    'WP_Return_Pct': round(s2['wp_ret'], 2),
    'WP_Volume_Pct': round(s2['wp_vol'], 2),
    'Ret_1Y_Pct': round(s2['penalties']['ret_1y'], 2),
    'Ret_6M_Pct': round(s2['penalties']['ret_6m'], 2),
    'Ret_3M_Pct': round(s2['ret_pctls']['3M']['return'], 2),
    'Ret_1M_Pct': round(s2['ret_pctls']['1M']['return'], 2),
    'Ret_1W_Pct': round(s2['ret_pctls']['1W']['return'], 2),
    'RSI_Value': round(s2['rsi'], 2),
    'MA50': round(s2['ma50'], 2) if s2['ma50'] else '',
    'MA200': round(s2['ma200'], 2) if s2['ma200'] else '',
    'Price': round(s2['price'], 2),
    'Death_Cross': str(s2['death_cross']),
    'Golden_Cross': str(s2['golden_cross']),
    'Volatility_Pct': round(s2['volatility'], 2),
    'Penalty_Total': s2['penalties']['total'],
    'Penalty_Reversal': s2['penalties']['reversal'],
    'Penalty_GoldenCross': s2['penalties']['golden_cross_pen'],
    'Warning_Flags': s2['penalties']['flags'] if s2['penalties']['flags'] else 'None',
    'DXY_1M_Pct': round(s2['external']['dxy_1m'], 2) if s2['external']['dxy_1m'] is not None else '',
    'VIX_Level': round(s2['external']['vix_level'], 2) if s2['external']['vix_level'] is not None else '',
    'DXY_Signal': s2['external']['dxy_signal'],
    'VIX_Signal': s2['external']['vix_signal'],
    'Z_Score_50d': round(zscore['z_50d'], 3) if zscore['z_50d'] is not None else '',
    'Z_Score_100d': round(zscore['z_100d'], 3) if zscore['z_100d'] is not None else '',
    'Z_Score_200d': round(zscore['z_200d'], 3) if zscore['z_200d'] is not None else '',
    'Z_Zone': zscore['zone'],
    'Z_Signal': zscore['signal'],
    'Z_Regime': zscore['regime'],
    'Z_Delta_5d': round(zscore['z_delta_5d'], 3) if zscore.get('z_delta_5d') is not None else '',
    'Base_Date_1': s1['date'].strftime('%Y-%m-%d'),
    'Base_Date_2': s2['date'].strftime('%Y-%m-%d'),
    'As_Of_Running': AS_OF,
}

csv_df = pd.DataFrame([csv_row])
csv_fixed = os.path.join(base_dir, 'output_momentum_gold_sell.csv')
csv_ts = os.path.join(base_dir, f'output_momentum_gold_sell_{TS_FILE}.csv')
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV saved: {csv_fixed}")
print(f"CSV saved: {csv_ts}")

# ══════════════════════════════════════════════════════
# SELL SCORE HISTORY — append daily (for exhaustion detection)
# ══════════════════════════════════════════════════════

history_row = {
    'Date': s2['date'].strftime('%Y-%m-%d'),
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
    'Ret_3M': round(s2['ret_pctls']['3M']['return'], 2),
    'Golden_Cross': str(s2['golden_cross']),
    'Z_Score_50d': round(zscore['z_50d'], 3) if zscore['z_50d'] is not None else '',
    'Z_Zone': zscore['zone'],
    'Z_Delta_5d': round(zscore['z_delta_5d'], 3) if zscore.get('z_delta_5d') is not None else '',
    'Warning_Flags': s2['penalties']['flags'] if s2['penalties']['flags'] else 'None',
    'Tier': sell_tier,
    'As_Of_Running': AS_OF,
    'Exhaust_Scenario': '',
}

history_path = os.path.join(base_dir, 'score_history_sell.csv')
history_df = pd.DataFrame([history_row])

if os.path.exists(history_path):
    existing = pd.read_csv(history_path, encoding='utf-8')
    # Ensure string columns don't become float64 from empty values
    for col in ['Exhaust_Scenario', 'Warning_Flags', 'Tier', 'Z_Zone', 'Golden_Cross', 'As_Of_Running']:
        if col in existing.columns:
            existing[col] = existing[col].fillna('').astype(str)
    existing = existing[existing['Date'] != history_row['Date']]
    history_df = pd.concat([existing, history_df], ignore_index=True)
    history_df = history_df.sort_values('Date').reset_index(drop=True)

history_df.to_csv(history_path, index=False, encoding='utf-8')
print(f"Sell score history: {history_path} ({len(history_df)} rows)")

# ══════════════════════════════════════════════════════
# SELL EXHAUSTION DETECTION (mirror of buy-side logic)
# ══════════════════════════════════════════════════════

exhaust_result = {
    'scenario': 'None',
    'label': '',
    'action_override': '',
    'net_5d_change': '',
    'max_10d': '',
    'min_10d': '',
    'd5_shift_5d': '',
}

if len(history_df) >= 6:
    h = history_df.copy()
    h['Net_Score'] = pd.to_numeric(h['Net_Score'], errors='coerce')
    h['D5_DirVol'] = pd.to_numeric(h['D5_DirVol'], errors='coerce')

    current = h.iloc[-1]
    net_now = current['Net_Score']
    d5_now = current['D5_DirVol']

    # Net score 5d ago
    net_5d_ago = h.iloc[-6]['Net_Score'] if len(h) >= 6 else net_now
    net_5d_change = net_now - net_5d_ago

    # Max/Min Net in last 10 days
    last10 = h['Net_Score'].tail(min(10, len(h)))
    max_10d = last10.max()
    min_10d = last10.min()

    # D5 shift in 5 days
    d5_5d_ago = h.iloc[-6]['D5_DirVol'] if len(h) >= 6 else d5_now
    d5_shift = d5_now - d5_5d_ago

    exhaust_result['net_5d_change'] = round(net_5d_change, 2)
    exhaust_result['max_10d'] = round(max_10d, 2)
    exhaust_result['min_10d'] = round(min_10d, 2)
    exhaust_result['d5_shift_5d'] = round(d5_shift, 2)

    # ── SE1: Sell Exhaustion (mirror of Bull Exhaustion) ──
    # Sell score high + starting to fade → selling pressure exhausted
    se1 = net_now >= 70 and net_5d_change < 0

    # ── SE2: Sell Topping (mirror of Topping) ──
    # Sell score was very high recently but dropped hard → panic selling may be over
    se2 = max_10d >= 80 and net_5d_change < -8 and not se1

    # ── SE3: Sell Recovery (mirror of Bear Exhaustion) ──
    # Sell score was low (no selling) + now rising → sell pressure building
    se3 = min_10d < 50 and net_5d_change > 3

    # ── SE4: D5s Volatility Regime Shift ──
    se4 = abs(d5_shift) >= 50 and not se1 and not se2 and not se3

    if se1:
        exhaust_result['scenario'] = 'Sell Exhaustion'
        exhaust_result['label'] = '🔥 Sell Exhaustion: Sell momentum fading → selling pressure หมดแรง'
        exhaust_result['action_override'] = 'HOLD'
    elif se3:
        exhaust_result['scenario'] = 'Sell Recovery'
        exhaust_result['label'] = '🔋 Sell Recovery: Sell score rising from low → sell pressure building → SELL'
        exhaust_result['action_override'] = 'SELL'
    elif se2:
        exhaust_result['scenario'] = 'Sell Topping'
        exhaust_result['label'] = '🏔️ Sell Topping: Sell score collapsed from peak → panic may be over'
        exhaust_result['action_override'] = 'HOLD'
    elif se4:
        exhaust_result['scenario'] = 'Vol Shift'
        exhaust_result['label'] = '⚡ Vol Regime Shift: D5s changed ' + str(round(d5_shift)) + 'pts → HOLD'
        exhaust_result['action_override'] = 'HOLD'

    print(f"\n── Sell Exhaustion Detection ──")
    print(f"  Net 5d Δ:    {net_5d_change:+.2f}")
    print(f"  Max 10d:     {max_10d:.2f}  |  Min 10d: {min_10d:.2f}")
    print(f"  D5s shift 5d: {d5_shift:+.0f}  (was {d5_5d_ago:.0f} → now {d5_now:.0f})")
    if exhaust_result['scenario'] != 'None':
        print(f"  >>> {exhaust_result['label']}")
    else:
        print(f"  >>> No sell exhaustion signal")

# Add exhaustion columns to sell CSV
csv_row['Exhaust_Scenario'] = exhaust_result['scenario']
csv_row['Exhaust_Action'] = exhaust_result['action_override']
csv_row['Net_5d_Change'] = exhaust_result['net_5d_change']
csv_row['Max_10d'] = exhaust_result['max_10d']
csv_row['Min_10d'] = exhaust_result['min_10d']
csv_row['D5_Shift_5d'] = exhaust_result['d5_shift_5d']

# Update exhaustion in history
if len(history_df) >= 1:
    history_df.loc[history_df['Date'] == history_row['Date'], 'Exhaust_Scenario'] = exhaust_result['scenario']
    history_df.to_csv(history_path, index=False, encoding='utf-8')

# Re-save CSVs with exhaustion columns
csv_df = pd.DataFrame([csv_row])
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')

print(f"\n✅ SELL score outputs generated successfully!")
print(f"   CSV: output_momentum_gold_sell.csv + output_momentum_gold_sell_{TS_FILE}.csv")

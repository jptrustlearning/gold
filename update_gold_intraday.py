#!/usr/bin/env python3
"""
🥇 Gold Intraday Price Updater (H1 + H4)
JP Trust Learning

ดึงข้อมูลราคาทอง GC=F แบบ H1 จาก yfinance
แล้ว resample เป็น H4 อัตโนมัติ
สะสมข้อมูลใน CSV บน GitHub — ยิ่งรันยิ่งมีข้อมูลมากขึ้น

Usage:
    python3 update_gold_intraday.py              # ดึงข้อมูลใหม่ merge กับเดิม
    python3 update_gold_intraday.py --backfill   # ดึงย้อนหลัง 730 วัน (ครั้งแรก)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================
SYMBOL = 'GC=F'
H1_CSV = 'gold_prices_h1.csv'
H4_CSV = 'gold_prices_h4.csv'
ENCODING = 'utf-8-sig'

# Thai column headers (consistent with daily pipeline)
THAI_COLS = ['วันที่', 'ราคาเปิด', 'ราคาสูงสุด', 'ราคาต่ำสุด', 'ราคาปิด', 'ปริมาณซื้อขาย']
ENG_COLS = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']


def fetch_h1_data(start=None, end=None, period=None):
    """ดึงข้อมูล H1 จาก yfinance"""
    gold = yf.Ticker(SYMBOL)

    if period:
        print(f'📥 Fetching {SYMBOL} H1 data: period={period}')
        df = gold.history(period=period, interval='1h')
    else:
        print(f'📥 Fetching {SYMBOL} H1 data: {start} → {end}')
        df = gold.history(start=start, end=end, interval='1h')

    if df.empty:
        print('ℹ️  No new H1 data available')
        return pd.DataFrame(columns=ENG_COLS)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index.name = 'Datetime'
    df = df.reset_index()

    # Remove timezone info for consistent storage
    if df['Datetime'].dt.tz is not None:
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)

    # Round prices, int volume
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].round(1)
    df['Volume'] = df['Volume'].fillna(0).astype(int)

    print(f'📊 Fetched {len(df)} H1 rows: {df["Datetime"].min()} → {df["Datetime"].max()}')
    return df


def read_existing_csv(filepath):
    """อ่าน CSV เดิม (Thai headers) → English columns"""
    if not os.path.exists(filepath):
        print(f'📂 No existing file: {filepath}')
        return pd.DataFrame(columns=ENG_COLS)

    df = pd.read_csv(filepath, encoding=ENCODING)
    df.columns = ENG_COLS
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    print(f'📂 Existing {filepath}: {len(df)} rows, last: {df["Datetime"].max()}')
    return df


def merge_data(df_existing, df_new):
    """Merge ข้อมูลเก่า + ใหม่ ไม่ซ้ำ"""
    if df_new.empty:
        return df_existing

    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all['Datetime'] = pd.to_datetime(df_all['Datetime'])
    df_all = df_all.drop_duplicates(subset='Datetime', keep='last')
    df_all = df_all.sort_values('Datetime').reset_index(drop=True)

    new_count = len(df_all) - len(df_existing)
    print(f'🔄 Merged: {len(df_existing)} + {len(df_new)} → {len(df_all)} rows (+{new_count} net new)')
    return df_all


def resample_h1_to_h4(df_h1):
    """Resample H1 → H4 using OHLCV aggregation

    H4 bars: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    Each bar covers 4 hours ending at its timestamp.
    """
    if df_h1.empty:
        return pd.DataFrame(columns=ENG_COLS)

    df = df_h1.copy()
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    # Resample to 4H bars
    df_h4 = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna(subset=['Open'])

    df_h4 = df_h4.reset_index()
    df_h4.rename(columns={'index': 'Datetime'}, inplace=True)
    if 'Datetime' not in df_h4.columns:
        df_h4 = df_h4.rename_axis(None).reset_index()
        if df_h4.columns[0] != 'Datetime':
            df_h4 = df_h4.rename(columns={df_h4.columns[0]: 'Datetime'})

    for col in ['Open', 'High', 'Low', 'Close']:
        df_h4[col] = df_h4[col].round(1)
    df_h4['Volume'] = df_h4['Volume'].fillna(0).astype(int)

    print(f'📐 Resampled H1→H4: {len(df_h1)} → {len(df_h4)} rows')
    return df_h4


def save_csv(df, filepath):
    """Save DataFrame → CSV with Thai headers"""
    df_out = df.copy()
    df_out['Datetime'] = pd.to_datetime(df_out['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_out.columns = THAI_COLS
    df_out.to_csv(filepath, index=False, encoding=ENCODING)
    print(f'💾 Saved {filepath}: {len(df_out)} rows')


def main():
    backfill = '--backfill' in sys.argv

    print('=' * 60)
    print('🥇 Gold Intraday Price Updater (H1 + H4)')
    print(f'📅 Run time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print(f'📌 Mode: {"BACKFILL (730 days)" if backfill else "INCREMENTAL"}')
    print('=' * 60)

    # --- Step 1: Read existing H1 data ---
    df_h1_existing = read_existing_csv(H1_CSV)

    # --- Step 2: Fetch new H1 data ---
    if backfill or df_h1_existing.empty:
        # First run or backfill: fetch max available (~730 days)
        df_h1_new = fetch_h1_data(period='730d')
    else:
        # Incremental: fetch from last date
        last_dt = df_h1_existing['Datetime'].max()
        start = (last_dt - timedelta(hours=4)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        df_h1_new = fetch_h1_data(start=start, end=end)

    if df_h1_new.empty and df_h1_existing.empty:
        print('\n⚠️  No data at all — nothing to save')
        return

    # --- Step 3: Merge H1 ---
    df_h1_all = merge_data(df_h1_existing, df_h1_new)

    # --- Step 4: Save H1 CSV ---
    save_csv(df_h1_all, H1_CSV)

    # --- Step 5: Resample H1 → H4 and save ---
    df_h4 = resample_h1_to_h4(df_h1_all)
    save_csv(df_h4, H4_CSV)

    # --- Summary ---
    print('\n' + '=' * 60)
    print('✅ DONE!')
    print(f'📊 H1: {len(df_h1_all)} rows | {df_h1_all["Datetime"].min()} → {df_h1_all["Datetime"].max()}')
    print(f'📊 H4: {len(df_h4)} rows | {df_h4["Datetime"].min()} → {df_h4["Datetime"].max()}')
    if df_h1_all["Close"].iloc[-1]:
        print(f'🏷️  Latest: ${df_h1_all["Close"].iloc[-1]:.1f}')
    print('=' * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Gold Price Backfill — JP Trust Learning
ดึงราคาทอง GC=F ย้อนหลังตั้งแต่ 2013 → prepend เข้า gold_prices.csv

ใช้ครั้งเดียว (one-time) — ไม่กระทบ daily pipeline
รัน manual จาก GitHub Actions workflow_dispatch
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os, time

SYMBOL = 'GC=F'
BACKFILL_START = '2013-01-01'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'gold_prices.csv')

print(f'🥇 Gold Price Backfill (One-Time)')
print(f'{"="*60}')

# ── Read existing data ──
if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    df_existing.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_existing['Date'] = pd.to_datetime(df_existing['Date'])
    first_date = df_existing['Date'].min()
    print(f'📂 Existing: {len(df_existing)} rows')
    print(f'   Range: {first_date.date()} → {df_existing["Date"].max().date()}')
else:
    print('❌ gold_prices.csv not found!')
    exit(1)

# ── Check if backfill needed ──
target_start = pd.Timestamp(BACKFILL_START)
if first_date <= target_start + timedelta(days=7):
    print(f'\n✅ Already have data from {first_date.date()} — no backfill needed')
    exit(0)

# ── Fetch older data in chunks (yfinance may limit range) ──
end_fetch = (first_date + timedelta(days=1)).strftime('%Y-%m-%d')
print(f'\n🔄 Fetching {SYMBOL}: {BACKFILL_START} → {end_fetch}')

all_new = []
chunk_start = pd.Timestamp(BACKFILL_START)
chunk_size = timedelta(days=365 * 2)  # 2 years per chunk

while chunk_start < first_date:
    chunk_end = min(chunk_start + chunk_size, first_date + timedelta(days=1))
    s = chunk_start.strftime('%Y-%m-%d')
    e = chunk_end.strftime('%Y-%m-%d')
    print(f'  📥 Chunk: {s} → {e}...', end=' ')

    for attempt in range(3):
        try:
            gold = yf.Ticker(SYMBOL)
            df_chunk = gold.history(start=s, end=e)
            break
        except Exception as ex:
            print(f'retry {attempt+1}...', end=' ')
            time.sleep(30 * (attempt + 1))
            if attempt == 2:
                print(f'❌ Failed')
                df_chunk = pd.DataFrame()

    if df_chunk.empty:
        print(f'no data')
    else:
        df_chunk = df_chunk[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        df_chunk.rename(columns={'Date': 'Date_raw'}, inplace=True)
        if df_chunk['Date_raw'].dt.tz is not None:
            df_chunk['Date_raw'] = df_chunk['Date_raw'].dt.tz_localize(None)
        df_chunk['Date'] = df_chunk['Date_raw']
        df_chunk = df_chunk[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        all_new.append(df_chunk)
        print(f'{len(df_chunk)} rows')

    chunk_start = chunk_end
    time.sleep(5)  # rate limit

if not all_new:
    print('\n❌ No historical data fetched — yfinance may not have GC=F data before 2015')
    print('   If needed, manually download from Investing.com and merge')
    exit(1)

df_new = pd.concat(all_new, ignore_index=True)
for col in ['Open', 'High', 'Low', 'Close']:
    df_new[col] = df_new[col].round(1)
df_new['Volume'] = df_new['Volume'].fillna(0).astype(int)

print(f'\n📊 Fetched total: {len(df_new)} rows')
print(f'   Range: {df_new["Date"].min().date()} → {df_new["Date"].max().date()}')

# ── Merge: prepend new data before existing ──
df_all = pd.concat([df_new, df_existing], ignore_index=True)
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all = df_all.drop_duplicates(subset='Date', keep='last')
df_all = df_all.sort_values('Date').reset_index(drop=True)

# ── Save with Thai headers ──
df_out = df_all.copy()
df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')
df_out.columns = ['วันที่', 'ราคาเปิด', 'ราคาสูงสุด', 'ราคาต่ำสุด', 'ราคาปิด', 'ปริมาณซื้อขาย']
df_out.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

print(f'\n{"="*60}')
print(f'✅ gold_prices.csv updated: {len(df_all)} rows')
print(f'   Range: {df_all["Date"].min().date()} → {df_all["Date"].max().date()}')
print(f'   Added: {len(df_all) - len(df_existing)} new rows before {first_date.date()}')
print(f'{"="*60}')

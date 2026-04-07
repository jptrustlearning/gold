#!/usr/bin/env python3
"""
🥇 Gold H1 Backfill — JP Trust Learning
ดึง GC=F H1 data ย้อนหลังถึง 2013 แบบ chunked (730 วัน/chunk)
merge กับ gold_prices_h1.csv เดิม

Usage:
    python3 backfill_gold_h1.py              # default: backfill to 2013
    python3 backfill_gold_h1.py 2015         # backfill to 2015
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os, sys, time
from datetime import datetime, timedelta

SYMBOL = 'GC=F'
H1_CSV = 'gold_prices_h1.csv'
ENCODING = 'utf-8-sig'
ENG_COLS = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
CHUNK_DAYS = 729  # yfinance max for interval='1h'

# Target start year from arg or default 2013
target_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2013
target_start = datetime(target_year, 1, 1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, H1_CSV)

print(f'🥇 Gold H1 Backfill')
print(f'{"="*60}')
print(f'Target: {target_start.date()} → existing data start')

# ── Read existing H1 data ──
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path, encoding=ENCODING)
    df_existing.columns = ENG_COLS
    df_existing['Datetime'] = pd.to_datetime(df_existing['Datetime'])
    first_date = df_existing['Datetime'].min()
    print(f'📂 Existing: {len(df_existing)} rows, starts {first_date}')
else:
    df_existing = pd.DataFrame(columns=ENG_COLS)
    first_date = datetime.now()
    print(f'📂 No existing H1 file — will create from scratch')

if first_date <= target_start + timedelta(days=7):
    print(f'\n✅ Already have data from {first_date.date()} — no backfill needed')
    sys.exit(0)

# ── Fetch in chunks ──
all_chunks = []
chunk_start = target_start
fetch_end = first_date + timedelta(days=1)
chunk_num = 0
empty_count = 0

while chunk_start < fetch_end:
    chunk_num += 1
    chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), fetch_end)
    s = chunk_start.strftime('%Y-%m-%d')
    e = chunk_end.strftime('%Y-%m-%d')
    print(f'\n  📥 Chunk {chunk_num}: {s} → {e}...', end=' ', flush=True)

    df_chunk = pd.DataFrame()
    for attempt in range(3):
        try:
            ticker = yf.Ticker(SYMBOL)
            df_chunk = ticker.history(start=s, end=e, interval='1h')
            break
        except Exception as ex:
            print(f'retry {attempt+1}...', end=' ', flush=True)
            time.sleep(30 * (attempt + 1))

    if df_chunk.empty:
        print(f'❌ no data')
        empty_count += 1
        if empty_count >= 3:
            print(f'\n⚠️  3 consecutive empty chunks — likely reached yfinance H1 limit')
            print(f'   yfinance typically only has ~2 years of H1 data for GC=F')
            break
    else:
        empty_count = 0  # reset
        df_chunk = df_chunk[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_chunk = df_chunk.reset_index()

        # Handle column name (could be 'Date' or 'Datetime' or 'index')
        dt_col = df_chunk.columns[0]
        df_chunk.rename(columns={dt_col: 'Datetime'}, inplace=True)

        # Strip timezone
        if df_chunk['Datetime'].dt.tz is not None:
            df_chunk['Datetime'] = df_chunk['Datetime'].dt.tz_localize(None)

        # Round & clean
        for col in ['Open', 'High', 'Low', 'Close']:
            df_chunk[col] = df_chunk[col].round(2)
        df_chunk['Volume'] = df_chunk['Volume'].fillna(0).astype(int)

        all_chunks.append(df_chunk)
        print(f'✅ {len(df_chunk)} bars ({df_chunk["Datetime"].min()} → {df_chunk["Datetime"].max()})')

    chunk_start = chunk_end
    time.sleep(5)  # rate limit between chunks

if not all_chunks:
    print(f'\n❌ No H1 data fetched')
    print(f'   yfinance likely does not have H1 data for GC=F before ~{(datetime.now()-timedelta(days=730)).date()}')
    print(f'   For older H1 data, export from MT5 or use a paid data provider')
    sys.exit(1)

# ── Merge ──
df_new = pd.concat(all_chunks, ignore_index=True)
print(f'\n📊 Fetched total: {len(df_new)} new bars')
print(f'   Range: {df_new["Datetime"].min()} → {df_new["Datetime"].max()}')

df_all = pd.concat([df_new, df_existing], ignore_index=True)
df_all['Datetime'] = pd.to_datetime(df_all['Datetime'])
df_all = df_all.drop_duplicates(subset='Datetime', keep='last')
df_all = df_all.sort_values('Datetime').reset_index(drop=True)

# ── Save ──
df_out = df_all.copy()
df_out['Datetime'] = df_out['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_out.to_csv(csv_path, index=False, encoding=ENCODING)

added = len(df_all) - len(df_existing)
print(f'\n{"="*60}')
print(f'✅ {H1_CSV} updated: {len(df_all)} rows (+{added} new)')
print(f'   Range: {df_all["Datetime"].min()} → {df_all["Datetime"].max()}')
print(f'{"="*60}')

# ── Also update tail file ──
tail_path = os.path.join(BASE_DIR, 'gold_prices_h1_tail.csv')
tail_n = min(200, len(df_all))
df_tail = df_all.tail(tail_n).copy()
df_tail['Datetime'] = df_tail['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_tail.to_csv(tail_path, index=False, encoding=ENCODING)
print(f'📄 gold_prices_h1_tail.csv updated ({tail_n} rows)')

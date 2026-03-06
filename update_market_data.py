#!/usr/bin/env python3
"""
Market Data Updater — JP Trust Learning
ดึงข้อมูล DXY, US10Y, Brent, WTI, VIX จาก Yahoo Finance
ใช้เป็น Step 1 ของ Full Pipeline
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

FALLBACK_START = '2020-01-01'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS = [
    {"symbol": "DX-Y.NYB", "csv": "dxy_prices.csv",   "name": "DXY",          "decimals": 2},
    {"symbol": "^TNX",     "csv": "us10y_prices.csv", "name": "US 10Y Yield", "decimals": 3},
    {"symbol": "BZ=F",     "csv": "brent_prices.csv", "name": "Brent Crude",  "decimals": 2},
    {"symbol": "CL=F",     "csv": "wti_prices.csv",   "name": "WTI Crude",    "decimals": 2},
    {"symbol": "^VIX",     "csv": "vix_prices.csv",   "name": "VIX",          "decimals": 2},
]

results = []

for asset in ASSETS:
    symbol = asset["symbol"]
    csv_file = os.path.join(BASE_DIR, asset["csv"])
    name = asset["name"]
    decimals = asset["decimals"]

    print(f'\n{"="*50}')
    print(f'📊 {name} ({symbol})')
    print(f'{"="*50}')

    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file, encoding='utf-8-sig')
        df_existing.columns = ['Date','Open','High','Low','Close','Volume']
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        last_date = df_existing['Date'].max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'📂 Existing: {len(df_existing)} rows, last: {last_date.date()}')
    else:
        df_existing = pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume'])
        start_date = FALLBACK_START
        print(f'📂 No existing file — starting from {FALLBACK_START}')

    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f'🔄 Fetching: {start_date} → {end_date}')

    try:
        ticker = yf.Ticker(symbol)
        df_new = ticker.history(start=start_date, end=end_date)
    except Exception as e:
        print(f'❌ Error fetching {symbol}: {e}')
        results.append({"name": name, "status": "error", "new": 0})
        continue

    if df_new.empty:
        print(f'ℹ️  No new data available')
        results.append({"name": name, "status": "skip", "new": 0})
        continue

    try:
        df_new = df_new[['Open','High','Low','Close','Volume']].reset_index()
        df_new.rename(columns={'Date': 'Date_raw'}, inplace=True)
        if df_new['Date_raw'].dt.tz is not None:
            df_new['Date_raw'] = df_new['Date_raw'].dt.tz_localize(None)
        df_new['Date'] = df_new['Date_raw']
        df_new = df_new[['Date','Open','High','Low','Close','Volume']]
        for col in ['Open','High','Low','Close']:
            df_new[col] = df_new[col].round(decimals)
        df_new['Volume'] = df_new['Volume'].fillna(0).astype(int)
        new_count = len(df_new)

        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.drop_duplicates(subset='Date', keep='last')
        df_all = df_all.sort_values('Date').reset_index(drop=True)

        df_out = df_all.copy()
        df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')
        df_out.columns = ['วันที่','ราคาเปิด','ราคาสูงสุด','ราคาต่ำสุด','ราคาปิด','ปริมาณซื้อขาย']
        df_out.to_csv(csv_file, index=False, encoding='utf-8-sig')

        latest = df_all['Close'].iloc[-1]
        latest_date = df_all['Date'].max().strftime('%Y-%m-%d')
        print(f'✅ Total: {len(df_all)} rows (+{new_count} new) | Latest: {latest:.{decimals}f} @ {latest_date}')
        results.append({"name": name, "status": "ok", "new": new_count, "latest": latest, "decimals": decimals})
    except Exception as e:
        print(f'❌ Error processing {name}: {e}')
        results.append({"name": name, "status": "error", "new": 0})

# Summary
print(f'\n{"="*50}')
print('📋 MARKET DATA SUMMARY')
print(f'{"="*50}')
for r in results:
    if r["status"] == "ok":
        print(f'  ✅ {r["name"]}: +{r["new"]} rows → {r["latest"]:.{r["decimals"]}f}')
    elif r["status"] == "skip":
        print(f'  ⏭️  {r["name"]}: no new data')
    else:
        print(f'  ❌ {r["name"]}: error')

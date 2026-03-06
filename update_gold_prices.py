#!/usr/bin/env python3
"""
Gold Price Updater — JP Trust Learning
ดึงราคาทอง GC=F จาก Yahoo Finance
ใช้เป็น Step 2 ของ Full Pipeline
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

SYMBOL = 'GC=F'
FALLBACK_START = '2020-01-01'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, 'gold_prices.csv')

print(f'🥇 Gold Price Updater')
print(f'{"="*50}')

if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
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
print(f'🔄 Fetching {SYMBOL}: {start_date} → {end_date}')

gold = yf.Ticker(SYMBOL)
df_new = gold.history(start=start_date, end=end_date)

if df_new.empty:
    print('ℹ️  No new gold data available (market closed / holiday / already up to date)')
else:
    df_new = df_new[['Open','High','Low','Close','Volume']].reset_index()
    df_new.rename(columns={'Date': 'Date_raw'}, inplace=True)
    if df_new['Date_raw'].dt.tz is not None:
        df_new['Date_raw'] = df_new['Date_raw'].dt.tz_localize(None)
    df_new['Date'] = df_new['Date_raw']
    df_new = df_new[['Date','Open','High','Low','Close','Volume']]
    for col in ['Open','High','Low','Close']:
        df_new[col] = df_new[col].round(1)
    df_new['Volume'] = df_new['Volume'].astype(int)
    new_count = len(df_new)

    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all = df_all.drop_duplicates(subset='Date', keep='last')
    df_all = df_all.sort_values('Date').reset_index(drop=True)

    df_out = df_all.copy()
    df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')
    df_out.columns = ['วันที่','ราคาเปิด','ราคาสูงสุด','ราคาต่ำสุด','ราคาปิด','ปริมาณซื้อขาย']
    df_out.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    latest_price = df_all['Close'].iloc[-1]
    latest_date = df_all['Date'].max().strftime('%Y-%m-%d')
    print(f'✅ Total: {len(df_all)} rows (+{new_count} new)')
    print(f'🏷️  Latest: ${latest_price:.1f} @ {latest_date}')

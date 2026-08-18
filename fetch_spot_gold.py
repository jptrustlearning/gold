#!/usr/bin/env python3
"""
Gold SPOT Pipeline — ดึงราคา spot XAU/USD → gold_prices_spot.csv
═══════════════════════════════════════════════════════════════════
แยกไฟล์จาก gold_prices.csv (GC=F ฟิวเจอร์ส) โดยสิ้นเชิง — ห้ามปนกัน
(มติ Joon 18 ส.ค. 2026: ของเก่าคง GC=F เหมือนเดิม · spot เก็บไฟล์ใหม่)

แหล่งข้อมูล (เรียงลำดับ):
  1. OANDA v3  — feed เดียวกับ OANDA:XAUUSD บน TradingView (ต้องมี OANDA_TOKEN)
     dailyAlignment default = 17:00 America/New_York = เส้นปิดแท่งเดียวกับจอเทรด
     รองรับ backfill ย้อนถึง 2013 (แบ่งหน้า หน้าละ 5000 แท่ง)
  2. stooq xauusd — สำรอง (ไม่ต้องใช้ key แต่บางช่วงโดน JS challenge)
  3. Yahoo XAUUSD=X — สำรองสุดท้าย (ข้อมูลรายวันมีบ้างไม่มีบ้าง)

หลักการเดียวกับฝั่ง GC=F ฉบับแก้บั๊กแล้ว:
  - เก็บเฉพาะแท่งที่ "ปิดเซสชันแล้ว" เท่านั้น (OANDA มี flag complete บอกตรงๆ)
  - ติดป้ายวันด้วย trade date (แท่งเปิด 17:00 ET วันก่อน = วันเทรดถัดไป)
  - ดึงทับย้อน OVERLAP_DAYS วัน เพื่อซ่อมแท่งที่เคยเก็บไม่ครบ (ฟีดเดียวกัน ทับได้)
  - volume: OANDA มี tick volume ของตัวเอง · แหล่งสำรองยืม volume จาก GC=F
"""
import os
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

CSV_FILE = 'gold_prices_spot.csv'
FALLBACK_START = '2013-01-01'   # backfill รอบแรก (OANDA XAU_USD มีข้อมูลลึกพอ)
OVERLAP_DAYS = 10               # ดึงทับย้อนหลัง — ปลอดภัยเพราะไฟล์นี้ฟีดเดียวล้วน
OANDA_PAGE = 5000               # เพดาน candles ต่อ 1 request ของ OANDA v3
ET = ZoneInfo('America/New_York')
THAI_COLS = ['วันที่', 'ราคาเปิด', 'ราคาสูงสุด', 'ราคาต่ำสุด', 'ราคาปิด', 'ปริมาณซื้อขาย']


def to_trade_date(raw):
    """แปลง timestamp เปิดแท่ง → วันเทรด (เปิด ≥17:00 ET = วันเทรดถัดไป)"""
    idx = pd.DatetimeIndex(raw)
    if idx.tz is not None:
        idx = idx.tz_convert(ET).tz_localize(None)
    shift = pd.to_timedelta((idx.hour >= 17).astype(int), unit='D')
    return (idx + shift).normalize()


def session_cutoff(now_et=None):
    """วันเทรดล่าสุดที่ปิดเซสชันแล้ว (เซสชันวัน D ปิด 17:00 ET ของวัน D)"""
    now_et = now_et or datetime.now(ET)
    d = now_et.date()
    if now_et.hour < 17:
        d = d - timedelta(days=1)
    return pd.Timestamp(d)


def fetch_oanda(start_date):
    """ดึงจาก OANDA v3 แบบแบ่งหน้า — คืน (DataFrame[Date_raw,O,H,L,C,V], src) หรือ (empty, None)"""
    token = (os.environ.get('OANDA_TOKEN') or '').strip()
    if not token:
        print('ℹ️  ไม่มี OANDA_TOKEN — ข้ามไปแหล่งสำรอง')
        return pd.DataFrame(), None

    _h = (os.environ.get('OANDA_HOST') or '').strip().rstrip('/')
    hosts = [_h] if _h else ['https://api-fxpractice.oanda.com',
                             'https://api-fxtrade.oanda.com']
    for host in hosts:
        rows, cursor, pages = [], f'{start_date}T00:00:00Z', 0
        try:
            while True:
                qs = urllib.parse.urlencode({
                    'price': 'M', 'granularity': 'D',
                    'from': cursor, 'count': OANDA_PAGE, 'includeFirst': 'true',
                })
                req = urllib.request.Request(
                    f'{host}/v3/instruments/XAU_USD/candles?{qs}',
                    headers={'Authorization': f'Bearer {token}',
                             'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    cands = json.load(r).get('candles', [])
                pages += 1
                rows.extend(c for c in cands if c.get('complete'))
                # หน้าไม่เต็ม = ถึงปัจจุบันแล้ว · กัน loop ค้างที่ 40 หน้า (~700 ปี)
                if len(cands) < OANDA_PAGE or pages >= 40:
                    break
                cursor = cands[-1]['time']  # includeFirst=true ครั้งแรกเท่านั้น — หน้าถัดไปเริ่มหลังแท่งนี้
            if rows:
                # dedupe กันแท่งซ้ำตรงรอยต่อหน้า
                seen, uniq = set(), []
                for c in rows:
                    if c['time'] not in seen:
                        seen.add(c['time'])
                        uniq.append(c)
                df = pd.DataFrame([{
                    'Date_raw': c['time'],
                    'Open': float(c['mid']['o']), 'High': float(c['mid']['h']),
                    'Low': float(c['mid']['l']), 'Close': float(c['mid']['c']),
                    'Volume': int(c.get('volume') or 0),
                } for c in uniq])
                df['Date_raw'] = pd.to_datetime(df['Date_raw'], utc=True)
                print(f'✅ OANDA: {len(df)} แท่งที่ปิดแล้ว ({pages} หน้า)')
                return df, f'OANDA XAU_USD ({host.split("//")[-1]})'
            print(f'⚠️  {host}: ตอบกลับแต่ไม่มีแท่งที่ปิดแล้ว')
        except Exception as e:
            print(f'⚠️  {host} ล้มเหลว: {e!r}')
    return pd.DataFrame(), None


def fetch_stooq(start_date):
    d1 = start_date.replace('-', '')
    d2 = (datetime.now() + timedelta(days=1)).strftime('%Y%m%d')
    url = f'https://stooq.com/q/d/l/?s=xauusd&i=d&d1={d1}&d2={d2}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'jptrust-pipeline/1.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode('utf-8', 'replace')
        if not txt.lstrip().lower().startswith('date'):
            print(f'⚠️  stooq ตอบกลับไม่ใช่ CSV: {txt[:100]!r}')
            return pd.DataFrame(), None
        t = pd.read_csv(io.StringIO(txt))
        t.columns = [c.strip().title() for c in t.columns]
        if 'Volume' not in t.columns:
            t['Volume'] = 0
        t = t[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna(
            subset=['Open', 'High', 'Low', 'Close'])
        if t.empty:
            return pd.DataFrame(), None
        t = t.rename(columns={'Date': 'Date_raw'})
        t['Date_raw'] = pd.to_datetime(t['Date_raw'])
        print(f'✅ stooq: {len(t)} แถว')
        return t.reset_index(drop=True), 'stooq xauusd'
    except Exception as e:
        print(f'⚠️  stooq ล้มเหลว: {e!r}')
        return pd.DataFrame(), None


def fetch_yahoo(start_date, end_date):
    import yfinance as yf
    print('⏳ รอ 30 วิ เลี่ยง rate limit Yahoo...')
    time.sleep(30)
    for attempt in range(3):
        try:
            h = yf.Ticker('XAUUSD=X').history(start=start_date, end=end_date)
            if h.empty:
                return pd.DataFrame(), None
            h = h[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
            h = h.rename(columns={h.columns[0]: 'Date_raw'})
            return h, 'Yahoo XAUUSD=X'
        except Exception as e:
            print(f'⚠️  Yahoo attempt {attempt + 1}/3: {e}')
            if attempt < 2:
                time.sleep(30 * (attempt + 1))
    return pd.DataFrame(), None


def borrow_gcf_volume(df_new, start_date, end_date):
    """แหล่งสำรองไม่มี volume — ยืมของ COMEX GC=F มาให้คอลัมน์ต่อเนื่อง"""
    import yfinance as yf
    try:
        time.sleep(10)
        gv = yf.Ticker('GC=F').history(start=start_date, end=end_date)
        if gv.empty:
            return df_new
        gv = gv[['Volume']].reset_index()
        gv = gv.rename(columns={gv.columns[0]: 'Date_raw'})
        vmap = dict(zip(to_trade_date(gv['Date_raw']), gv['Volume']))
        spot_dates = to_trade_date(df_new['Date_raw'])
        df_new['Volume'] = [int(vmap.get(d, 0)) for d in spot_dates]
        hit = int((df_new['Volume'] > 0).sum())
        print(f'🔗 ยืม volume GC=F ได้ {hit}/{len(df_new)} แถว')
    except Exception as e:
        print(f'⚠️  ยืม volume GC=F ไม่ได้: {e!r} — volume = 0')
    return df_new


def merge_and_write(df_existing, df_new, src):
    df_new = df_new.copy()
    df_new['Date'] = to_trade_date(df_new['Date_raw'])
    df_new = df_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    before = len(df_new)
    df_new = df_new[df_new['Date'] <= session_cutoff()]
    if before != len(df_new):
        print(f'✂️  ตัดแท่งที่เซสชันยังไม่ปิด {before - len(df_new)} แถว')
    if df_new.empty:
        raise SystemExit('ไม่มีแท่งที่ปิดแล้วให้เขียน')
    for col in ['Open', 'High', 'Low', 'Close']:
        df_new[col] = df_new[col].round(1)
    df_new['Volume'] = df_new['Volume'].fillna(0).astype(int)

    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all = df_all[df_all['Date'].dt.dayofweek < 5]      # แท่งคืนวันอาทิตย์ถูกจัดเป็นวันจันทร์แล้ว
    df_all = df_all.drop_duplicates(subset='Date', keep='last')
    df_all = df_all.sort_values('Date').reset_index(drop=True)

    df_out = df_all.copy()
    df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')
    df_out.columns = THAI_COLS
    df_out.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    latest_price = df_all['Close'].iloc[-1]
    latest_date = df_all['Date'].max().strftime('%Y-%m-%d')
    print(f'✅ Spot: {len(df_all)} rows (+{len(df_new)} fetched) | ${latest_price:.1f} @ {latest_date} | src={src}')
    with open('SPOT_INFO', 'w') as f:
        f.write(f'+{len(df_new)} rows | ${latest_price:.1f} @ {latest_date} | src={src}')


def main():
    first_run = not os.path.exists(CSV_FILE)
    if first_run:
        df_existing = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        start_date = FALLBACK_START
        print(f'🆕 ยังไม่มี {CSV_FILE} — backfill ตั้งแต่ {FALLBACK_START}')
    else:
        df_existing = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        df_existing.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_existing['Date'] = pd.to_datetime(df_existing['Date'])
        last_date = df_existing['Date'].max()
        start_date = (last_date - timedelta(days=OVERLAP_DAYS)).strftime('%Y-%m-%d')
        print(f'📂 Existing spot: {len(df_existing)} rows, last {last_date.date()} (ทับย้อน {OVERLAP_DAYS} วัน)')

    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    print(f'🕓 cutoff วันเทรดที่ปิดแล้ว: {session_cutoff().date()} (ET now {datetime.now(ET):%Y-%m-%d %H:%M})')

    df_new, src = fetch_oanda(start_date)
    if df_new.empty:
        df_new, src = fetch_stooq(start_date)
    if df_new.empty:
        df_new, src = fetch_yahoo(start_date, end_date)
    if not df_new.empty and not str(src).startswith('OANDA'):
        df_new = borrow_gcf_volume(df_new, start_date, end_date)

    if df_new.empty:
        # ห้าม fallback ไป GC=F เด็ดขาด — ไฟล์นี้ต้องเป็น spot ล้วนเท่านั้น
        print('❌ ทุกแหล่ง spot ไม่มีข้อมูล — ไม่เขียนไฟล์รอบนี้')
        with open('SPOT_INFO', 'w') as f:
            f.write('ERROR: spot XAU/USD ไม่มีข้อมูล — ไม่เขียนไฟล์')
        # รอบ backfill แรกต้องสำเร็จเท่านั้น (ให้ workflow ขึ้นแดงเตือน) · รอบ incremental พลาดได้
        raise SystemExit(1 if first_run else 0)

    merge_and_write(df_existing, df_new, src)


if __name__ == '__main__':
    main()

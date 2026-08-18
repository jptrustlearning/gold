# PROBE RESULT

- run UTC: `2026-08-18T13:23:09.658053+00:00`
- yfinance: `1.6.0`  ·  pandas: `3.0.5`

## 1) stooq
### `https://stooq.com/q/d/l/?s=xauusd&i=d&d1=20260801&d2=20260819`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="iRoiGm9PDI6cEgB7UrYoQA">
(async()=>{const c="AAAAAGqEXL549d4aFk7jo3mcU1GodwiFNKX7oIB1yJwFh-E7rpK9UYHRTyQ",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
```

### `https://stooq.com/q/d/l/?s=xauusd&i=d`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="XJktgYhj2q7i6S6YxXeTEg">
(async()=>{const c="AAAAAGqEXMLTQRuVB8wBRIFs1aVdV35bNKX7oNUVVvUCm-F6lTuKNaMZpcU",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
```

### `https://stooq.pl/q/d/l/?s=xauusd&i=d`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="t_9EgMG4IlROSwvg0vnAjQ">
(async()=>{const c="AAAAAGqEXMYQWPKL9yL3urCSUxYPIZvJNKX7oJtoC2gsYkXl5iB6qanvOdM",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
```

## 2) Yahoo (yfinance)
### `XAUUSD=X`
- EMPTY (ไม่มีข้อมูลรายวัน)

### `XAU=X`
- EMPTY (ไม่มีข้อมูลรายวัน)

### `GC=F`
- 22 rows · index tz `America/New_York`
- ชั่วโมงของ timestamp: `{0: 22}`
```
                                  Open         High          Low        Close  Volume
Date                                                                                 
2026-08-11 00:00:00-04:00  4408.600098  4408.600098  4365.100098  4383.000000     204
2026-08-12 00:00:00-04:00  4406.500000  4434.000000  4406.299805  4408.899902     660
2026-08-13 00:00:00-04:00  4403.500000  4445.000000  4350.000000  4363.600098    1491
2026-08-14 00:00:00-04:00  4322.100098  4397.100098  4315.000000  4380.399902     673
2026-08-17 00:00:00-04:00  4394.799805  4428.500000  4386.500000  4417.799805     673
2026-08-18 00:00:00-04:00  4473.399902  4493.100098  4441.399902  4449.899902   55041
```

### `GLD`
- 21 rows · index tz `America/New_York`
- ชั่วโมงของ timestamp: `{0: 21}`
```
                                 Open        High         Low       Close    Volume
Date                                                                               
2026-08-10 00:00:00-04:00  397.709991  403.239990  395.929993  402.540009  11057000
2026-08-11 00:00:00-04:00  403.440002  404.029999  399.890015  400.959991   7439100
2026-08-12 00:00:00-04:00  405.309998  407.359985  403.220001  404.920013  10383100
2026-08-13 00:00:00-04:00  402.170013  402.579987  398.279999  398.959991   8695200
2026-08-14 00:00:00-04:00  402.179993  403.329987  400.940002  401.480011   6749800
2026-08-17 00:00:00-04:00         NaN         NaN         NaN         NaN   9351756
```

## 3) gold_prices.csv ปัจจุบัน
- 3450 rows · first `2013-01-02` · last `2026-08-17`
```
      Date   Open   High    Low  Close  Volume
2026-08-11 4430.0 4437.8 4421.4 4429.8    6723
2026-08-12 4468.8 4471.0 4456.5 4470.6    2634
2026-08-13 4408.2 4419.4 4405.1 4414.5    7005
2026-08-14 4408.2 4454.6 4365.5 4437.3  124670
2026-08-16 4440.0 4473.2 4422.3 4451.8   17148
2026-08-17 4473.4 4493.1 4461.1 4463.6   13322
```

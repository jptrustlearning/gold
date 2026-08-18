# PROBE RESULT

- run UTC: `2026-08-18T13:28:49.739721+00:00`
- yfinance: `1.6.0`  ·  pandas: `3.0.5`

## 1) stooq
### `https://stooq.com/q/d/l/?s=xauusd&i=d&d1=20260801&d2=20260819`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="2MF9VeULMEabgqYEq7tymg">
(async()=>{const c="AAAAAGqEXhLYblg-FWLNRISVTQ1uD1LEFFBWxjPDBYYiEYGloQ5SADyyI7k",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
```

### `https://stooq.com/q/d/l/?s=xauusd&i=d`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="5FMpmAqCqmFAakLoYaQX7w">
(async()=>{const c="AAAAAGqEXhbdMli1YEBpyHjEy5cvPPENFFBWxiexCErU4DIlLRDLkimiLt0",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
```

### `https://stooq.pl/q/d/l/?s=xauusd&i=d`
- HTTP `200` · 796 bytes
```
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="robots" content="noindex,nofollow"></head><body><noscript>This site requires JavaScript to verify your browser. Please enable JavaScript and reload.</noscript><script nonce="2HkC965Knk1WJv9HyQ0t_Q">
(async()=>{const c="AAAAAGqEXho5El9BmUfReVISAHUdNwR5FFBWxlrW_9vE7iCWXvP090DYK44",d=4,t="0".repeat(d),e=new TextEncoder;let n=0;while(1){cons
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
2026-08-18 00:00:00-04:00  4473.399902  4493.100098  4441.399902  4447.500000   56335
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

## 2.5) candidate spot quote APIs (ไม่ต้องใช้ key)
### `https://data-asg.goldprice.org/dbXRates/USD`
- HTTPError `403` — Forbidden

### `https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD`
- HTTP `200` · 1129 bytes · CORS `None`
```
[{"topo":{"platform":"SwissquoteLtd","server":"Live5"},"spreadProfilePrices":[{"spreadProfile":"premium","bidSpread":25.60,"askSpread":25.60,"bid":4389.049,"ask":4389.711},{"spreadProfile":"prime","bidSpread":24.20,"askSpread":24.20,"bid":4389.063,"ask":4389.697},{"spreadProfile":"elite","bidSpread":17.70,"askSpread":17.70,"bid":4389.128,"ask":4389.632}],"ts":1787059781075},{"topo":{"platform":"AT","server":"AT"},"spreadProfilePrices":[{"spreadProfile":"standard","bidSpread":27.00,"askSpread":27
```

### `https://api.gold-api.com/price/XAU`
- HTTP `200` · 177 bytes · CORS `*`
```
{"currency":"USD","currencySymbol":"$","exchangeRate":1.0,"name":"Gold","price":4392.0,"symbol":"XAU","updatedAt":"2026-08-18T13:29:35Z","updatedAtReadable":"a few seconds ago"}
```

### `https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=5d&interval=1d`
- HTTP `200` · 1391 bytes · CORS `None`
```
{"chart":{"result":[{"meta":{"currency":"USD","symbol":"GC=F","exchangeName":"CMX","fullExchangeName":"COMEX","instrumentType":"FUTURE","firstTradeDate":967608000,"regularMarketTime":1787059182,"hasPrePostMarketData":false,"gmtoffset":-14400,"timezone":"EDT","exchangeTimezoneName":"America/New_York","regularMarketPrice":4448.4,"fiftyTwoWeekHigh":5586.2,"fiftyTwoWeekLow":3310.1,"regularMarketDayHigh":4493.1,"regularMarketDayLow":4441.4,"regularMarketVolume":56384,"shortName":"Gold Dec 26","chartP
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

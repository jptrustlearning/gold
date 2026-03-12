#!/usr/bin/env python3
"""
Gold Momentum — H1 Net Bias Calculator
JP Trust Learning

Reads H1 Buy CSV + H1 Sell CSV → computes Net Bias → assigns Scenario → writes net CSV.
Run AFTER both gold_momentum_h1.py and gold_momentum_h1_sell.py.
"""

import pandas as pd
import os, sys
from datetime import datetime, timezone

RUN_TS = datetime.now(timezone.utc)
AS_OF = RUN_TS.strftime("%d/%m/%Y %H:%M UTC")
TS_FILE = RUN_TS.strftime("%d%m%Y_%H%M")

base_dir = os.path.dirname(os.path.abspath(__file__))

buy_path = os.path.join(base_dir, 'output_momentum_gold_h1.csv')
sell_path = os.path.join(base_dir, 'output_momentum_gold_h1_sell.csv')

if not os.path.exists(buy_path):
    print("❌ output_momentum_gold_h1.csv not found"); sys.exit(1)
if not os.path.exists(sell_path):
    print("❌ output_momentum_gold_h1_sell.csv not found"); sys.exit(1)

buy = pd.read_csv(buy_path).iloc[0]
sell = pd.read_csv(sell_path).iloc[0]

print(f"Gold Momentum — H1 Net Bias Calculator")
print(f"{'='*55}")

buy_score = float(buy['Net_Score_Avg'])
sell_score = float(sell['Net_Score_Avg'])
buy_delta = float(buy['Score_Delta'])
sell_delta = float(sell['Score_Delta'])
net_bias = buy_score - sell_score
price = float(buy['Price'])

# Pivot data from buy CSV (D1 pivots)
pivot_pos = str(buy.get('Pivot_Position', ''))
pp = float(buy.get('Pivot_D1_PP', 0)) if buy.get('Pivot_D1_PP', '') != '' else 0
r1 = float(buy.get('Pivot_D1_R1', 0)) if buy.get('Pivot_D1_R1', '') != '' else 0
r2 = float(buy.get('Pivot_D1_R2', 0)) if buy.get('Pivot_D1_R2', '') != '' else 0
s1 = float(buy.get('Pivot_D1_S1', 0)) if buy.get('Pivot_D1_S1', '') != '' else 0
s2_pv = float(buy.get('Pivot_D1_S2', 0)) if buy.get('Pivot_D1_S2', '') != '' else 0

buy_dims = {f'Buy_D{i}': float(buy[col]) for i, col in enumerate(
    ['D1_ReturnRank','D2_VolumeRank','D3_RSI','D4_MA','D5_Volatility','D6_External'], 1)}
sell_dims = {f'Sell_D{i}': float(sell[col]) for i, col in enumerate(
    ['D1_ReturnRank','D2_VolumeRank','D3_RSI','D4_MA','D5_Volatility','D6_External'], 1)}

print(f"Buy Score:  {buy_score:.2f} ({buy['Tier']})")
print(f"Sell Score: {sell_score:.2f} ({sell['Tier']})")
print(f"Net Bias:   {net_bias:+.2f}")
print(f"Price:      ${price:.1f}")
print(f"Pivot Pos:  {pivot_pos}")


def net_bias_tier(nb):
    if nb >= 50:  return "Strong Bullish ↑↑"
    if nb >= 25:  return "Bullish ↑"
    if nb >= 10:  return "Lean Bullish ↗"
    if nb >= -10: return "Neutral →"
    if nb >= -25: return "Lean Bearish ↘"
    if nb >= -50: return "Bearish ↓"
    return "Strong Bearish ↓↓"

bias_tier = net_bias_tier(net_bias)
print(f"Bias Tier:  {bias_tier}")


# ══════════════════════════════════════════════════════
# SCENARIO ASSIGNMENT (same logic as daily)
# ══════════════════════════════════════════════════════

def assign_scenario(buy_sc, sell_sc, net_b, buy_d, sell_d, price, pp, r1, r2, s1, s2):
    above_r1 = price >= r1 if r1 else False
    above_pp = price >= pp if pp else False
    below_s1 = price < s1 if s1 else False
    between_pp_r1 = (price >= pp and price < r1) if (pp and r1) else False
    between_s1_pp = (price >= s1 and price < pp) if (s1 and pp) else False

    # BULLISH ZONE
    if buy_sc >= 60 and sell_sc < 45 and net_b >= 20:
        if above_r1 and buy_sc >= 75 and buy_d > 0:
            return {'num':1,'signal':'Bullish Breakout','thai':'ทะลุแนวต้าน','icon':'🚀','action':'BUY','zone':'BULLISH',
                    'cond':'Price≥R1 & Buy≥75 & Sell<45 & Net≥+20 & BuyΔ>0',
                    'detail':'ราคาทะลุ R1 + Buy momentum แข็ง + Sell อ่อน + ยังเร่งตัว → Breakout ยืนยัน'}
        if above_r1 and buy_sc >= 75:
            return {'num':2,'signal':'Bullish but Cooling','thai':'ขาขึ้นแต่ชะลอ','icon':'📈','action':'HOLD','zone':'BULLISH',
                    'cond':'Price≥R1 & Buy≥75 & Sell<45 & BuyΔ≤0',
                    'detail':'ทะลุ R1 + Buy ยังแข็ง แต่ delta เริ่มชะลอ → พิจารณา take-profit'}
        if between_pp_r1:
            return {'num':3,'signal':'Bullish Accumulation','thai':'สะสมแรงขาขึ้น','icon':'💪','action':'BUY','zone':'BULLISH',
                    'cond':'PP≤Price<R1 & Buy≥60 & Sell<45 & Net≥+20',
                    'detail':'ราคาเหนือ PP กำลังวิ่งหา R1 + Buy แข็ง Sell อ่อน → สะสมแรงเพื่อ breakout'}
        if between_s1_pp:
            return {'num':4,'signal':'Support Bounce','thai':'เด้งจากแนวรับ','icon':'🔄','action':'BUY','zone':'BULLISH',
                    'cond':'S1≤Price<PP & Buy≥60 & Sell<45 & Net≥+20',
                    'detail':'ราคาย่อลงมาใต้ PP แต่ Buy ยังแข็ง Sell อ่อน → มีโอกาสเด้งกลับ'}
        if below_s1:
            return {'num':5,'signal':'Oversold Recovery','thai':'ร่วงแรงแต่มีแรงฟื้น','icon':'⚡','action':'BUY','zone':'BULLISH',
                    'cond':'Price<S1 & Buy≥60 & Sell<45 & Net≥+20',
                    'detail':'ราคาหลุด S1 แต่ Buy momentum ยังแข็ง + Sell ไม่มีแรง → Divergence อาจ recover'}
        return {'num':3,'signal':'Bullish Accumulation','thai':'สะสมแรง','icon':'💪','action':'BUY','zone':'BULLISH',
                'cond':'Buy≥60 & Sell<45 & Net≥+20','detail':'Buy momentum แข็ง + Sell อ่อน → แนวโน้ม bullish'}

    # BEARISH ZONE
    if sell_sc >= 60 and buy_sc < 45 and net_b <= -20:
        if below_s1 and sell_sc >= 75 and sell_d > 0:
            return {'num':6,'signal':'Bearish Breakdown','thai':'หลุดแนวรับ','icon':'📉','action':'SELL','zone':'BEARISH',
                    'cond':'Price<S1 & Sell≥75 & Buy<45 & Net≤-20 & SellΔ>0',
                    'detail':'ราคาหลุด S1 + Sell แข็ง + Buy หมดแรง → เสี่ยงลงต่อ S2/S3'}
        if below_s1:
            return {'num':7,'signal':'Bearish Continuation','thai':'ขาลงต่อเนื่อง','icon':'🔻','action':'SELL','zone':'BEARISH',
                    'cond':'Price<S1 & Sell≥60 & Buy<45 & Net≤-20','detail':'ราคาใต้ S1 + Sell แข็ง → ขาลงยังไม่จบ'}
        if between_s1_pp:
            return {'num':8,'signal':'Bearish Distribution','thai':'กระจายของขาลง','icon':'📊','action':'SELL','zone':'BEARISH',
                    'cond':'S1≤Price<PP & Sell≥60 & Buy<45 & Net≤-20','detail':'กำลังกระจายก่อนลงต่อ'}
        if above_pp:
            return {'num':9,'signal':'Resistance Rejection','thai':'โดนแนวต้านตีกลับ','icon':'⚠️','action':'SELL','zone':'BEARISH',
                    'cond':'Price≥PP & Sell≥60 & Buy<45 & Net≤-20','detail':'ราคาเหนือ PP แต่ Sell แข็ง → เสี่ยงโดนตีกลับ'}
        return {'num':7,'signal':'Bearish Continuation','thai':'ขาลง','icon':'🔻','action':'SELL','zone':'BEARISH',
                'cond':'Sell≥60 & Buy<45 & Net≤-20','detail':'Sell momentum แข็ง → bearish'}

    # CONFLICT ZONE
    if buy_sc >= 55 and sell_sc >= 55 and abs(net_b) < 20:
        if above_r1:
            return {'num':10,'signal':'Volatile Breakout','thai':'ทะลุแต่ผันผวน','icon':'🌪️','action':'HOLD','zone':'CONFLICT',
                    'cond':'Price≥R1 & Buy≥55 & Sell≥55 & |Net|<20','detail':'ผันผวนสูง รอยืนยัน'}
        return {'num':11,'signal':'Tug of War','thai':'ชักเย่อ','icon':'⚔️','action':'HOLD','zone':'CONFLICT',
                'cond':'Buy≥55 & Sell≥55 & |Net|<20','detail':'ทั้งสองฝั่งมีแรง → รอให้ฝั่งใดชนะ'}

    # NEUTRAL ZONE
    if buy_sc < 55 and sell_sc < 55:
        return {'num':12,'signal':'Dead Zone','thai':'ไม่มีทิศทาง','icon':'💤','action':'HOLD','zone':'NEUTRAL',
                'cond':'Buy<55 & Sell<55','detail':'ไม่มีแรงทั้งสองฝั่ง → รอ breakout'}

    # LEAN
    if buy_sc >= 55 and sell_sc < 55 and net_b > 0:
        return {'num':13,'signal':'Cautious Bullish','thai':'เอียง bullish','icon':'📊','action':'HOLD','zone':'LEAN BULLISH',
                'cond':'Buy≥55 & Sell<55 & Net>0','detail':'Buy แข็งกว่า Sell แต่ยังไม่ชัดเจน → รอยืนยัน'}
    if sell_sc >= 55 and buy_sc < 55 and net_b < 0:
        return {'num':14,'signal':'Cautious Bearish','thai':'เอียง bearish','icon':'📉','action':'HOLD','zone':'LEAN BEARISH',
                'cond':'Sell≥55 & Buy<55 & Net<0','detail':'Sell แข็งกว่า Buy แต่ยังไม่ชัดเจน → ระวัง'}

    return {'num':12,'signal':'Monitoring','thai':'ติดตาม','icon':'👁️','action':'HOLD','zone':'NEUTRAL',
            'cond':'Mixed signals','detail':'สัญญาณไม่ชัดเจน'}

scenario = assign_scenario(buy_score, sell_score, net_bias, buy_delta, sell_delta, price, pp, r1, r2, s1, s2_pv)

print(f"\nScenario: #{scenario['num']} {scenario['icon']} {scenario['signal']}")
print(f"Zone: {scenario['zone']} | Action: {scenario['action']}")


# ══════════════════════════════════════════════════════
# CSV OUTPUT
# ══════════════════════════════════════════════════════

csv_row = {
    'Ticker': 'GOLD_H1', 'Timeframe': 'H1',
    'Buy_Score': round(buy_score, 2), 'Sell_Score': round(sell_score, 2),
    'Net_Bias': round(net_bias, 2), 'Bias_Tier': bias_tier,
    'Buy_Tier': buy['Tier'], 'Sell_Tier': sell['Tier'],
    'Buy_Delta': round(buy_delta, 2), 'Sell_Delta': round(sell_delta, 2),
    'Scenario_Num': scenario['num'], 'Scenario_Signal': scenario['signal'],
    'Scenario_Thai': scenario['thai'], 'Scenario_Icon': scenario['icon'],
    'Scenario_Action': scenario['action'], 'Scenario_Zone': scenario['zone'],
    'Scenario_Cond': scenario['cond'], 'Scenario_Detail': scenario['detail'],
    'Buy_D1': buy_dims['Buy_D1'], 'Sell_D1': sell_dims['Sell_D1'],
    'Buy_D2': buy_dims['Buy_D2'], 'Sell_D2': sell_dims['Sell_D2'],
    'Buy_D3': buy_dims['Buy_D3'], 'Sell_D3': sell_dims['Sell_D3'],
    'Buy_D4': buy_dims['Buy_D4'], 'Sell_D4': sell_dims['Sell_D4'],
    'Buy_D5': buy_dims['Buy_D5'], 'Sell_D5': sell_dims['Sell_D5'],
    'Buy_D6': buy_dims['Buy_D6'], 'Sell_D6': sell_dims['Sell_D6'],
    'Price': round(price, 2), 'Pivot_Position': pivot_pos,
    'Pivot_D1_PP': pp, 'Pivot_D1_R1': r1, 'Pivot_D1_R2': r2,
    'Pivot_D1_S1': s1, 'Pivot_D1_S2': s2_pv,
    'Base_Date_1': buy['Base_Date_1'], 'Base_Date_2': buy['Base_Date_2'],
    'As_Of_Running': AS_OF,
}

csv_df = pd.DataFrame([csv_row])
csv_fixed = os.path.join(base_dir, 'output_momentum_gold_h1_net.csv')
csv_ts = os.path.join(base_dir, f'output_momentum_gold_h1_net_{TS_FILE}.csv')
csv_df.to_csv(csv_fixed, index=False, encoding='utf-8')
csv_df.to_csv(csv_ts, index=False, encoding='utf-8')
print(f"\nCSV saved: {csv_fixed}")
print(f"CSV saved: {csv_ts}")
print(f"\n✅ H1 Net Bias complete!")

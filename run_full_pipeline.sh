#!/bin/bash
# Gold Full Pipeline v2.0 — Runner Script
# ลำดับ: Market Data → Gold Prices → Momentum Score V2
# Usage: cd gold && bash run_full_pipeline.sh

set -e

echo "══════════════════════════════════════════════"
echo "  Gold Full Pipeline v2.0"
echo "  Market Data → Gold Price → Momentum Score"
echo "══════════════════════════════════════════════"

# Setup git credentials
git config user.email "jptrustlearning@users.noreply.github.com"
git config user.name "JP Trust Learning"

# Pull latest
echo ""
echo "📥 Pulling latest data..."
git pull origin main --rebase

# ── STEP 1: Update Market Data ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Update Market Data (DXY, VIX, etc.)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 update_market_data.py

# ── STEP 2: Update Gold Prices ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Update Gold Prices"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 update_gold_prices.py

# ── STEP 3: Run Momentum Score V2 ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Momentum Score V2 (with DXY + VIX)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 gold_momentum_v2.py

# ── STEP 4: Git Push ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: Push to GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Add all data files
git add dxy_prices.csv us10y_prices.csv brent_prices.csv wti_prices.csv vix_prices.csv 2>/dev/null || true
git add gold_prices.csv 2>/dev/null || true
git add output_momentum_gold.csv 2>/dev/null || true

# Find timestamped CSV
TS_CSV=$(ls -t output_momentum_gold_*.csv 2>/dev/null | head -1)
if [ -n "$TS_CSV" ]; then
    git add "$TS_CSV"
    echo "📄 Timestamped CSV: $TS_CSV"
fi

DISPLAY_TS=$(date -u +"%d/%m/%Y %H:%M UTC")

if git diff --cached --quiet; then
    echo "✅ No changes to push"
else
    git commit -m "🔄 Full Pipeline v2.0 — $DISPLAY_TS"
    git push origin main
    echo "✅ Pushed to GitHub"
fi

echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ Full Pipeline Complete!"
echo "══════════════════════════════════════════════"

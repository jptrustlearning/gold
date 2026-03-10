#!/bin/bash
# Gold Momentum Score v2.0 — Runner Script (D1-D5 + D6 External Context)
# Usage: cd gold && bash run_momentum.sh

set -e

echo "═══════════════════════════════════════════"
echo "  Gold Momentum Score v2.0 (D6 External)"
echo "═══════════════════════════════════════════"

# Setup git credentials
git config user.email "jptrustlearning@users.noreply.github.com"
git config user.name "JP Trust Learning"
# Token is set by Claude at runtime or via environment variable GH_TOKEN
# git remote set-url origin https://jptrustlearning:$GH_TOKEN@github.com/jptrustlearning/gold.git

# Pull latest data
echo "📥 Pulling latest data..."
git pull origin main --rebase

# Run scoring (unified script with D6 External Context)
echo "📊 Running Momentum Score v2.0 (D1-D5 + D6 External)..."
python3 gold_momentum_v2.py

# Find timestamped CSV
TS_CSV=$(ls -t output_momentum_gold_*.csv | head -1)
echo "📄 Timestamped CSV: $TS_CSV"

# Git push
echo "🚀 Pushing to GitHub..."
git add output_momentum_gold.csv "$TS_CSV"
DISPLAY_TS=$(date -u +"%d/%m/%Y %H:%M UTC")
git commit -m "Gold Momentum Score v2.0 — $DISPLAY_TS"
git push origin main

echo ""
echo "✅ Done! Dashboard will auto-update."
echo "   https://raw.githubusercontent.com/jptrustlearning/gold/main/output_momentum_gold.csv"

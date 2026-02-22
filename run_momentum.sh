#!/bin/bash
# Gold Momentum Score v2.2 — Runner Script
# Usage: cd gold && bash run_momentum.sh

set -e

echo "═══════════════════════════════════════════"
echo "  Gold Momentum Score v2.2 — Runner"
echo "═══════════════════════════════════════════"

# Setup git credentials
git config user.email "jptrustlearning@users.noreply.github.com"
git config user.name "JP Trust Learning"
# Token is set by Claude at runtime or via environment variable GH_TOKEN
# git remote set-url origin https://jptrustlearning:$GH_TOKEN@github.com/jptrustlearning/gold.git

# Pull latest data
echo "📥 Pulling latest data..."
git pull origin main --rebase

# Run scoring
echo "📊 Running Momentum Score v2.2..."
python3 gold_momentum_v22.py

# Find timestamped CSV
TS_CSV=$(ls -t output_momentum_gold_*.csv | head -1)
echo "📄 Timestamped CSV: $TS_CSV"

# Git push
echo "🚀 Pushing to GitHub..."
git add output_momentum_gold.csv "$TS_CSV"
DISPLAY_TS=$(date -u +"%d/%m/%Y %H:%M UTC")
git commit -m "Gold Momentum Score v2.2 — $DISPLAY_TS"
git push origin main

echo ""
echo "✅ Done! Dashboard will auto-update."
echo "   https://raw.githubusercontent.com/jptrustlearning/gold/main/output_momentum_gold.csv"

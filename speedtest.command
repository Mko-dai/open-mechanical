#!/bin/bash
# ソルバーの速度比較を実行し、結果を logs/speed.txt に保存
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u -m openmech.speedtest 2>&1 | tee logs/speed.txt
echo ""
echo "結果は logs/speed.txt に保存しました。"

#!/bin/bash
# iPhone Air 簡易モデルの落下解析（結果は logs/iphone_air/）
cd "$(dirname "$0")"
mkdir -p logs/iphone_air
source ./sync.sh
source .venv/bin/activate
python -u tools/iphone_air_drop.py 2>&1 | tee logs/iphone_air/console.txt | tail -60

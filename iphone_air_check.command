#!/bin/bash
# iPhone Air モデルのメッシュだけ作って品質を確認（logs/iphone_air/log.txt）
cd "$(dirname "$0")"
mkdir -p logs/iphone_air
source ./sync.sh
source .venv/bin/activate
CHECK=1 python -u tools/iphone_air_drop.py 2>&1 | tail -30

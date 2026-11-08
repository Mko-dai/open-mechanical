#!/bin/bash
# iPhone Air の落下を陽解法（LS-DYNA 方式）で解く（結果は logs/iphone_air_explicit/）
cd "$(dirname "$0")"
mkdir -p logs/iphone_air_explicit
source ./sync.sh
source .venv/bin/activate
python -u tools/iphone_air_explicit.py corner back 2>&1 | tee logs/iphone_air_explicit/console.txt | tail -60

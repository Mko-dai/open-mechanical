#!/bin/bash
# 全テストを実行して logs/selftest.txt に保存
cd "$(dirname "$0")"
./sync.sh
mkdir -p logs
source .venv/bin/activate 2>/dev/null
python -m pytest -q 2>&1 | tee logs/selftest.txt

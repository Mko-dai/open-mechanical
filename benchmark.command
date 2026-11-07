#!/bin/bash
# 精度検証ベンチマークを実行し、結果を logs/benchmark.txt に保存
cd "$(dirname "$0")"
mkdir -p logs
source .venv/bin/activate
python -u -m openmech.benchmarks --converge 2>&1 | tee logs/benchmark.txt
echo ""
echo "結果は logs/benchmark.txt に保存しました。"

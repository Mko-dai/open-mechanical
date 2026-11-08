#!/bin/bash
# 歪な形状 5 モデルのメッシュ品質（Ansys と同じ指標）を logs/mesh_compare/ に保存
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u tools/mesh_compare.py 2>&1 | tee logs/mesh_compare.txt

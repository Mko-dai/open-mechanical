#!/bin/bash
# Ansys Mechanical との比較検証（5モデル）を実行し、logs/verification.json と logs/verification.txt に保存
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u -m openmech.verification 2>&1 | tee logs/verification.txt

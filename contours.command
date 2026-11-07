#!/bin/bash
# 検証5モデルのコンター図を logs/contours/ に保存
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u -m openmech.contours 2>&1 | tee logs/contours.txt

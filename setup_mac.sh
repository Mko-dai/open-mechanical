#!/bin/bash
# open-mechanical 初回セットアップ（Mac）
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "使用する Python: $PY ($($PY --version))"
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "セットアップ完了。 ./run.sh で起動できます。"

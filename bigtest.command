#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u tools/bigtest.py 30 40 50 2>&1 | tee logs/bigtest.txt

#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs
source ./sync.sh
source .venv/bin/activate
python -u tools/meshcheck.py 2>&1 | tee logs/meshcheck.txt

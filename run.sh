#!/bin/bash
# open-mechanical を起動
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "先に ./setup_mac.sh を実行してください"; exit 1
fi
source .venv/bin/activate
python -m openmech "$@"

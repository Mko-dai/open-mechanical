#!/bin/bash
# open-mechanical をダブルクリックで起動（Mac）
# 初回は GitHub との同期と、必要なライブラリのインストールも自動で行います。
cd "$(dirname "$0")"

# GitHub から最新版を取得
source ./sync.sh

# 初回セットアップ
if [ ! -x .venv/bin/python ]; then
  echo "=== 初回セットアップ（数分かかります） ==="
  ./setup_mac.sh || { echo "セットアップに失敗しました。上のエラーを確認してください。"; read -r; exit 1; }
fi

echo "=== open-mechanical を起動します ==="
source .venv/bin/activate
python -m openmech

#!/bin/bash
# open-mechanical をダブルクリックで起動（Mac）
# 初回は GitHub との同期と、必要なライブラリのインストールも自動で行います。
cd "$(dirname "$0")"

# GitHub との初回同期（まだコミットが無いフォルダの場合だけ）
if ! git rev-parse HEAD >/dev/null 2>&1 && git remote get-url origin >/dev/null 2>&1; then
  echo "=== GitHub と同期しています ==="
  git fetch origin && git reset origin/main && git branch -u origin/main
fi

# 初回セットアップ
if [ ! -x .venv/bin/python ]; then
  echo "=== 初回セットアップ（数分かかります） ==="
  ./setup_mac.sh || { echo "セットアップに失敗しました。上のエラーを確認してください。"; read -r; exit 1; }
fi

echo "=== open-mechanical を起動します ==="
source .venv/bin/activate
python -m openmech

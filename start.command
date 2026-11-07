#!/bin/bash
# open-mechanical をダブルクリックで起動（Mac）
# 初回は GitHub との同期と、必要なライブラリのインストールも自動で行います。
cd "$(dirname "$0")"

# GitHub から最新版を取得
if git remote get-url origin >/dev/null 2>&1; then
  echo "=== GitHub と同期しています ==="
  if ! git rev-parse HEAD >/dev/null 2>&1; then
    git fetch origin && git reset origin/main && git branch -u origin/main
  else
    git fetch -q origin
    if git diff --quiet origin/main 2>/dev/null; then
      git reset -q origin/main                 # ファイルは既に最新
    elif git diff --quiet HEAD 2>/dev/null; then
      git merge -q --ff-only origin/main       # 手元に変更なし → 最新版に更新
    else
      echo "（手元で変更されたファイルがあるため自動更新はスキップしました）"
    fi
  fi
fi

# 初回セットアップ
if [ ! -x .venv/bin/python ]; then
  echo "=== 初回セットアップ（数分かかります） ==="
  ./setup_mac.sh || { echo "セットアップに失敗しました。上のエラーを確認してください。"; read -r; exit 1; }
fi

echo "=== open-mechanical を起動します ==="
source .venv/bin/activate
python -m openmech

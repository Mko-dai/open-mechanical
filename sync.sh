#!/bin/bash
# GitHub から最新版を取得（start.command / benchmark.command から呼ばれる）
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


# 必要なライブラリが増えていたら追加インストール
if [ -x .venv/bin/python ]; then
  H=$(cat requirements.txt requirements-fast.txt 2>/dev/null | shasum | cut -c1-12)
  if [ "$(cat .venv/.req_hash 2>/dev/null)" != "$H" ]; then
    echo "=== ライブラリを更新しています ==="
    .venv/bin/python -m pip install -q -r requirements.txt
    # 高速化用（入らなくても動作します）
    while read -r pkg; do
      [ -z "$pkg" ] || [ "${pkg:0:1}" = "#" ] && continue
      .venv/bin/python -m pip install -q "$pkg" || echo "（$pkg はインストールできませんでした。標準のソルバーを使います）"
    done < requirements-fast.txt
    echo "$H" > .venv/.req_hash
  fi
fi

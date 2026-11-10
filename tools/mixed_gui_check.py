"""GUI の動作確認（ソリッド＋シェルだけ）: guismoke_analyses.py の準備部分と、最後のソリッド＋シェルの部分だけを実行する"""
import os

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "guismoke_analyses.py"), encoding="utf-8").read()
head = src[:src.index('fix = BC("fixed", left)')]
tail = src[src.index("# ソリッド＋シェル: 直方体"):]
exec(compile(head + "\n" + tail, os.path.join(here, "guismoke_analyses.py"), "exec"))

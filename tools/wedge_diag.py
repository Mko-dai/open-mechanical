"""iPhone の多層スイープ（三角柱）で静解析が止まる原因を調べる（Mac で実行）

  python tools/wedge_diag.py     結果: logs/wedge_diag.txt

各段階の時刻・メモリ・行列の大きさを書き、4 分ごとに実行中の場所（スタック）を書き出す。
"""
from __future__ import annotations

import faulthandler
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

out = open(os.path.join(ROOT, "logs", "wedge_diag.txt"), "w", buffering=1)
faulthandler.dump_traceback_later(240, repeat=True, file=out)
T0 = time.time()


def P(*a):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)     # macOS: バイト
    s = f"[{time.time() - T0:7.1f}s / 最大メモリ {rss:5.2f} GB] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")


os.environ.setdefault("MESH", "2.0")
import iphone_air_model as ip  # noqa: E402
from openmech.geometry import build_mesh  # noqa: E402
from openmech.solver import BC, choose_method, stiffness_pattern  # noqa: E402

configs = [("四面体（既定）", []),
           ("三角柱（層の分割は既定）", [{"type": "sweep", "axis": "Z", "element": "wedge"}]),
           ("三角柱（厚さ方向 1 mm）", [{"type": "sweep", "axis": "Z", "element": "wedge", "layer_size": 1.0}])]
only = sys.argv[1:]
for name, ctrl in configs:
    if only and not any(o in name for o in only):
        continue
    P(f"=== {name}")
    m = build_mesh(ip.p.geometry, 2.0, 2, log=lambda s: None, local_sizes=ctrl, share=ip.p.share_map())
    ip_, ix = stiffness_pattern(m.elems, m.n_nodes)
    nnz = 9 * len(ix)
    P(f"要素 {m.n_elems:,}（{m.elems.shape[1]} 節点）/ 節点 {m.n_nodes:,} / 自由度 {3 * m.n_nodes:,} / "
      f"剛性行列の非ゼロ {nnz / 1e6:.0f} 百万（約 {nnz * 12 / 1e9:.1f} GB）/ 解法 {choose_method(3 * m.n_nodes)}")
    p = ip.p
    info = m.face_info()
    hi = m.nodes[:, 2].max()
    # 落下解析（iphone_air_drop.py の「画面から落下」）と同じ: 前面ガラスの面を固定して 1 G
    f0 = [f["id"] for f in info if f["flat"] and abs(f["centroid"][2] - hi) < 0.05 and f["normal"][2] > 0.99]
    p.bcs = [BC("fixed", f0), BC("gravity", [], [0, 0, 9810.0])]
    P(f"固定する面 {len(f0)} / 非線形ソルバーが必要か {p.needs_nonlinear_solver(m)}")
    t1 = time.time()
    try:
        r = p.run(m, log=lambda s: P("  " + str(s)))
        P(f"完了 {time.time() - t1:.1f} s / 最大変位 {r.disp_mag.max():.4g} mm / 最大ミーゼス {r.von_mises.max():.4g} MPa")
    except Exception as e:  # noqa: BLE001
        import traceback
        P("エラー: " + "".join(traceback.format_exception(e))[-1500:])
P("完了")
faulthandler.cancel_dump_traceback_later()

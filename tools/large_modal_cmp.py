"""大規模モーダル: LOBPCG（AMG 前処理・ブロック）とシフト逆反復（AMG-CG で 1 本ずつ）の時間を比べる"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech import modal  # noqa: E402
from openmech.materials import LIBRARY  # noqa: E402
from openmech.mesh import structured_box  # noqa: E402
from openmech.solver import BC  # noqa: E402

S = LIBRARY["構造用鋼 (Structural Steel)"]
args = sys.argv[1:] or ["16:lobpcg,shift_invert", "22:lobpcg"]
for a in args:
    n, methods = a.split(":")
    n = int(n)
    m = structured_box(400, 40, 40, n * 5, n, n).to_quadratic()
    m.group_faces()
    f0 = [x["id"] for x in m.face_info() if abs(x["centroid"][0]) < 1e-6][0]
    print(f"=== 節点 {m.n_nodes:,}（自由度 {3 * m.n_nodes:,}）", flush=True)
    ref = None
    for meth in methods.split(","):
        modal.EIG_METHOD = meth
        t = time.time()
        md = modal.solve_modal(m, S, [BC("fixed", [f0])], n_modes=10,
                               log=lambda *x: print("   ", *x, flush=True) if "モード" not in x[0] else None)
        dt = time.time() - t
        diff = "" if ref is None else f" / 前の方法との差 最大 {np.abs(md.freqs / ref - 1).max():.1e}"
        ref = md.freqs if ref is None else ref
        print(f"  {meth}: 10 モード {dt:.1f} s / 1 次 {md.freqs[0]:.4f} Hz / 10 次 {md.freqs[9]:.2f} Hz{diff}", flush=True)

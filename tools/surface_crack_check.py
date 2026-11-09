"""半楕円の表面き裂（有限幅の板の引張）: 前縁に沿った K を Raju–Newman の式（精度 ±5 %）と比べる"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.benchmarks import face_where  # noqa: E402
from openmech.fracture import evaluate_cracks, raju_newman  # noqa: E402
from openmech.geometry import build_mesh  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402

LX, LY, T, SIG = 100.0, 200.0, 20.0, 100.0
mat = Material("s", 200000, 0.3, 7850)
for a, c in [(8.0, 16.0), (6.0, 12.0)]:
    crack = {"shape": "ellipse", "center": [LX / 2, LY / 2, T], "normal": [0, 1, 0], "axis_a": [0, 0, -1],
             "a": a, "c": c, "front_size": 0.6, "stations": 9}
    t0 = time.time()
    m = build_mesh({"kind": "box", "lx": LX, "ly": LY, "lz": T}, size=6.0, order=2, log=print,
                   settings={"cracks": [crack]})
    tm = time.time() - t0
    X = m.nodes
    y0, y1 = face_where(m, 1, 0.0, -1), face_where(m, 1, LY, 1)
    A = int(np.argmin(np.linalg.norm(X - [0, LY / 2, 0], axis=1)))
    B = int(np.argmin(np.linalg.norm(X - [LX, LY / 2, 0], axis=1)))
    C = int(np.argmin(np.linalg.norm(X - [0, 0, 0], axis=1)))
    r = solve(m, mat, [BC("pressure", [y0], [-SIG]), BC("pressure", [y1], [-SIG]),
                       BC("displacement", [], [0.0, 0.0, 0.0], nodes=[A]),
                       BC("displacement", [], [None, 0.0, 0.0], nodes=[B]),
                       BC("displacement", [], [None, None, 0.0], nodes=[C])], log=lambda *x: None)
    t1 = time.time()
    rows = evaluate_cracks(m, r, [crack])
    print(f"a = {a}, c = {c}（a/c = {a / c:.2f}, a/t = {a / T:.2f}）: 節点 {m.n_nodes:,} / メッシュ {tm:.1f}s / "
          f"解析 {t1 - t0 - tm:.1f}s / 評価 {time.time() - t1:.1f}s / 1/4 点要素 {getattr(m, 'quarter_point', False)}")
    for row in rows:
        phi = np.radians(row["位置 φ [°]"])
        ref = raju_newman(SIG, a, c, T, LX / 2, phi)
        K = row["K（J から）[MPa√mm]"]
        print(f"   φ {row['位置 φ [°]']:7.2f}°  K {K:8.1f}  Raju-Newman {ref:8.1f}  差 {K / ref - 1:+.2%}  "
              f"（積分路 J {row['J 積分路 1']:.4g} / {row['J 積分路 2']:.4g}）")

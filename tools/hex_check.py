"""六面体スイープ（Hex20）の確認: 片持ちはり（四面体と比較）と円柱の引張・ねじり"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.benchmarks import face_where  # noqa: E402
from openmech.geometry import build_mesh, mesh_quality  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402

mat = Material("s", 200000, 0.3, 7850)
L, b, h, P = 200.0, 10.0, 20.0, 100.0
I = b * h ** 3 / 12
G = 200000 / 2.6
exact = P * L ** 3 / (3 * 200000 * I) + P * L / (5 / 6 * G * b * h)
spec = {"kind": "box", "lx": L, "ly": b, "lz": h}
for name, ls in [("四面体 Tet10", []), ("六面体スイープ Hex20", [{"type": "sweep", "axis": "X", "element": "hex"}])]:
    t0 = time.time()
    m = build_mesh(spec, size=5.0, order=2, log=lambda *a: None, local_sizes=ls)
    tm = time.time() - t0
    r = solve(m, mat, [BC("fixed", [face_where(m, 0, 0.0, -1)]), BC("force", [face_where(m, 0, L, 1)], [0, 0, -P])],
              log=lambda *a: None)
    q = mesh_quality(m)
    print(f"{name}: 要素 {m.n_elems:,}（{m.elems.shape[1]} 節点）/ 節点 {m.n_nodes:,} / 先端たわみ {-r.u[:, 2].min():.5f} "
          f"（理論 {exact:.5f}、{-r.u[:, 2].min() / exact - 1:+.2%}）/ 品質 平均 {q['要素品質']['平均']:.3f} / "
          f"メッシュ {tm:.2f}s・合計 {time.time() - t0:.2f}s")
# 円柱（Z 方向スイープ、断面の円を四角形で切る）: 引張の一様応力と体積
R, H = 10.0, 50.0
m = build_mesh({"kind": "cylinder", "r": R, "h": H}, size=2.0, order=2, log=lambda *a: None,
               local_sizes=[{"type": "sweep", "axis": "Z", "element": "hex"}])
z0, z1 = face_where(m, 2, 0.0, -1), face_where(m, 2, H, 1)
c0 = int(np.argmin(np.linalg.norm(m.nodes - [0, 0, 0], axis=1)))
c1 = int(np.argmin(np.linalg.norm(m.nodes - [R, 0, 0], axis=1)))     # z 軸まわりの回転を止める
r = solve(m, mat, [BC("symmetry", [z0]), BC("pressure", [z1], [-100.0]),
                   BC("displacement", [], [0.0, 0.0, None], nodes=[c0]), BC("displacement", [], [None, 0.0, None], nodes=[c1])],
          log=lambda *a: None)
print(f"円柱 Hex20: 要素 {m.n_elems:,} / 体積 {m.volume():.3f}（理論 {np.pi * R * R * H:.3f}）/ σz 平均 {r.stress[:, 2].mean():.3f}"
      f"（理論 100）/ 伸び {r.u[:, 2].max():.6f}（理論 {100 * H / 200000:.6f}）")

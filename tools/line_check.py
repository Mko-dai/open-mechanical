"""三角柱の積層メッシュで、反復法の線平滑化（厚さ方向の列ごとに解く）の効果を測る（Mac で実行）

  python tools/line_check.py      結果: logs/line_check.txt
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402

from openmech.materials import Material  # noqa: E402
from openmech.mesh import Mesh  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402

out = open(os.path.join(ROOT, "logs", "line_check.txt"), "w", buffering=1)
faulthandler.dump_traceback_later(300, repeat=True, file=out)
T0 = time.time()


def P(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")


def laminate(lx, ly, h2=2.0):
    """iPhone に似た 8 層（ガラス・磁石・電池・接着 0.17 mm・チタン・クッション・表示・ガラス）"""
    th = [0.55, 0.45, 2.72, 0.17, 0.30, 0.20, 0.55, 0.70]
    E = [70000, 10000, 5000, 5, 110000, 1.0, 3000, 70000]
    nx, ny = int(lx / h2), int(ly / h2)
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Pn = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: i * (ny + 1) + j  # noqa: E731
    tris = np.array([t for i in range(nx) for j in range(ny) for t in
                     [(nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)), (nid(i, j), nid(i + 1, j + 1), nid(i, j + 1))]])
    zs = np.concatenate([[0], np.cumsum(th)])
    n2 = len(Pn)
    nodes = np.vstack([np.column_stack([Pn, np.full(n2, z)]) for z in zs])
    el = np.vstack([np.column_stack([tris + k * n2, tris + (k + 1) * n2]) for k in range(len(th))])
    body = np.repeat(np.arange(len(th)), len(tris))
    m = Mesh(nodes, el).to_quadratic()
    m.elem_body = body
    mats = [Material(f"m{i}", e, 0.3, 2000).state() for i, e in enumerate(E)]
    return m, (mats, body)


def run(m, em, use_lines, robust=False):
    info = m.face_info()
    top = [f["id"] for f in info if f["normal"][2] > 0.99 and abs(f["centroid"][2] - m.nodes[:, 2].max()) < 1e-6]
    bcs = [BC("fixed", top), BC("gravity", [], [0, 0, 9810.0])]
    orig = Mesh.sweep_lines
    os.environ["OPENMECH_AMG"] = "robust" if robust else ""
    if not use_lines:
        Mesh.sweep_lines = lambda self: None
    try:
        t = time.time()
        r = solve(m, em[0][0], bcs, log=lambda s: P("    " + str(s)), elem_materials=em, method="amg")
        return time.time() - t, float(r.disp_mag.max())
    finally:
        Mesh.sweep_lines = orig


CASES = [(80, 40, False, False), (80, 40, True, False), (80, 40, False, True), (80, 40, True, True)]
if "--big" in sys.argv:
    CASES = [(156, 75, a, b) for a, b in ((False, False), (True, False), (True, True))]
for lx, ly, use, rob in CASES:
    m, em = laminate(lx, ly)
    P(f"=== 積層板 {lx}×{ly} mm / 自由度 {3 * m.n_nodes:,} / 線平滑化 {'あり' if use else 'なし'} / "
      f"AMG {'平滑化集約（頑健）' if rob else '平滑化なし集約（既定）'}")
    dt, d = run(m, em, use, rob)
    P(f"  → {dt:.1f} s / 最大変位 {d:.6e} mm")
if "--iphone" not in sys.argv:
    P("完了")
    sys.exit(0)

os.environ.setdefault("MESH", "2.0")
import iphone_air_model as ip  # noqa: E402
from openmech.geometry import build_mesh  # noqa: E402
m = build_mesh(ip.p.geometry, 2.0, 2, log=lambda s: None, local_sizes=[{"type": "sweep", "axis": "Z", "element": "wedge"}],
               share=ip.p.share_map())
P(f"=== iPhone 三角柱 / 自由度 {3 * m.n_nodes:,}")
p = ip.p
info = m.face_info()
hi = m.nodes[:, 2].max()
f0 = [f["id"] for f in info if f["flat"] and abs(f["centroid"][2] - hi) < 0.05 and f["normal"][2] > 0.99]
p.bcs = [BC("fixed", f0), BC("gravity", [], [0, 0, 9810.0])]
os.environ["OPENMECH_AMG"] = "robust" if "--robust" in sys.argv else ""
t = time.time()
r = p.run(m, log=lambda s: P("    " + str(s)))
P(f"  → {time.time() - t:.1f} s / 最大変位 {r.disp_mag.max():.4g} mm / 最大ミーゼス {r.von_mises.max():.4g} MPa")
P("完了")
faulthandler.cancel_dump_traceback_later()

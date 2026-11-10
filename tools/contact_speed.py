"""接触ありの非線形解析の速度: 直接法と自動（大きなモデルは反復法 AMG）を比べる

    python tools/contact_speed.py [分割数（既定 14）]

3 つのブロックを重ねる（下: 結合、上: 摩擦なし）。上面に圧力、下面を固定。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["OPENMECH_SOLVER_LOG"] = "1"
import numpy as np  # noqa: E402

from openmech.contact import detect_contacts  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.mesh import Mesh, structured_box  # noqa: E402
from openmech.nonlinear import AnalysisSettings, solve_static  # noqa: E402
from openmech.solver import BC, _direct_backend  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
parts = [structured_box(60, 60, 20, n, n, max(2, n // 3), origin=(0, 0, 20 * k)).to_quadratic() for k in range(3)]
off, nodes, elems, body = 0, [], [], []
for k, p in enumerate(parts):
    nodes.append(p.nodes)
    elems.append(p.elems + off)
    body.append(np.full(p.n_elems, k))
    off += p.n_nodes
m = Mesh(np.vstack(nodes), np.vstack(elems))
m.elem_body = np.concatenate(body)
m.group_faces()
cs = detect_contacts(m)
cs[-1].type = "frictionless"
info = m.face_info()
bot = [f["id"] for f in info if abs(f["centroid"][2]) < 1e-6]
top = [f["id"] for f in info if abs(f["centroid"][2] - 60) < 1e-6]
xs = [f["id"] for f in info if abs(f["centroid"][0]) < 1e-6]
ys = [f["id"] for f in info if abs(f["centroid"][1]) < 1e-6]
bcs = [BC("fixed", bot), BC("pressure", top, [10.0]), BC("displacement", xs, [0.0, None, None]),
       BC("displacement", ys, [None, 0.0, None])]
print(f"節点 {m.n_nodes:,} / 自由度 {3 * m.n_nodes:,} / 接触 {[c.type for c in cs]} / 直接法 = {_direct_backend()}", flush=True)
res = {}
for st in (["auto", "direct"] if "--auto-only" not in sys.argv else ["auto"]):
    lines = []
    t = time.time()
    r = solve_static(m, S, bcs, cs, AnalysisSettings(solver_type=st), log=lines.append)[-1]
    res[st] = (time.time() - t, float(np.nanmedian(r.contact["接触圧 [MPa]"])), float(-r.u[:, 2].min()))
    print(f"{st}: {res[st][0]:.1f} s / 接触圧 中央値 {res[st][1]:.4f} MPa / 最大沈み {res[st][2]:.6f} mm / "
          f"反復 {len(r.history)}", flush=True)
    for ln in lines:
        if "反復回数" in str(ln) or "切り替え" in str(ln) or "ソルバー" in str(ln):
            print("   ", str(ln).strip(), flush=True)
print("完了")

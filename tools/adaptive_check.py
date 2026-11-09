"""適応メッシュ（収束ツール）の確認: 穴あき板の引張を粗いメッシュから始めて、穴縁の σx が
Peterson の応力集中係数に収束するかを見る（gmsh が必要）"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.adaptive import converge  # noqa: E402
from openmech.benchmarks import face_where  # noqa: E402
from openmech.geometry import build_mesh  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.project import Project  # noqa: E402
from openmech.solver import BC  # noqa: E402

L, W, t, d, F = 200.0, 50.0, 5.0, 10.0, 5000.0
spec = {"kind": "plate_hole", "lx": L, "ly": W, "t": t, "d": d}
p = Project(geometry=spec, mesh_size=5.0, mesh_order=2, material=Material("steel", 200000.0, 0.3, 7850.0))
m0 = build_mesh(spec, size=5.0, order=2, log=lambda *a: None)
# 左端は法線方向だけ拘束（固定にすると角が特異点になり、最大値が収束しない）、剛体移動は弱いばねで止める
p.bcs = [BC("frictionless", [face_where(m0, 0, 0.0, -1)]), BC("force", [face_where(m0, 0, L, 1)], [F, 0.0, 0.0])]
p.settings.weak_springs = True
q = d / W
kt = 3 - 3.14 * q + 3.667 * q ** 2 - 1.527 * q ** 3
target = kt * F / ((W - d) * t)
t0 = time.time()
mesh, res, hist = converge(p, field="応力 σx [MPa]", stat="max", allowable=1.0, max_loops=5,
                           log=lambda m: print(m, flush=True))
print(f"Peterson 理論値 {target:.2f} MPa（平面応力）。3D の板厚中央は数 % 高い")
for h in hist:
    v = h["最大 応力 σx [MPa]"] if "最大 応力 σx [MPa]" in h else [x for k, x in h.items() if "σx" in k][0]
    print(f"{h['回']}  要素 {h['要素数']:>8,}  σx max {v:8.2f}  理論比 {v / target:6.3f}  "
          f"推定誤差 {h['推定誤差 [%]']:.2f} %  {h['時間 [s]']} s")
print(f"合計 {time.time() - t0:.1f} s")

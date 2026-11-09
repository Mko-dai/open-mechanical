"""破壊力学の確認（gmsh の自由メッシュに き裂を埋め込む）: 片側き裂の帯板の引張（平面ひずみ）を Tada の式と比べる"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.benchmarks import face_where  # noqa: E402
from openmech.fracture import evaluate_cracks  # noqa: E402
from openmech.geometry import build_mesh  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402

W, H2, t, SIG = 50.0, 100.0, 2.0, 100.0
mat = Material("s", 200000, 0.3, 7850)
for a in (10.0, 15.0, 25.0):
    q = a / W
    K = SIG * np.sqrt(np.pi * a) * (1.12 - 0.231 * q + 10.55 * q ** 2 - 21.72 * q ** 3 + 30.39 * q ** 4)
    crack = {"front0": [a, H2 / 2, 0], "front1": [a, H2 / 2, t], "normal": [0, 1, 0], "direction": [1, 0, 0],
             "depth": a, "to_surface": True, "front_size": 0.4}
    t0 = time.time()
    m = build_mesh({"kind": "box", "lx": W, "ly": H2, "lz": t}, size=4.0, order=2, log=lambda *x: None,
                   settings={"cracks": [crack]})
    tm = time.time() - t0
    X = m.nodes
    bot, top = face_where(m, 1, 0.0, -1), face_where(m, 1, H2, 1)
    z0, z1 = face_where(m, 2, 0.0, -1), face_where(m, 2, t, 1)
    # 面内の剛体移動を静定に止める（反力 0）: 右端中央の x・y と 右上の角の x
    c = int(np.argmin(np.linalg.norm(X - [W, H2 / 2, t / 2], axis=1)))
    c2 = int(np.argmin(np.linalg.norm(X - [W, H2, t / 2], axis=1)))
    r = solve(m, mat, [BC("displacement", [z0, z1], [None, None, 0.0]),
                       BC("displacement", [], [0.0, 0.0, None], nodes=[c]),
                       BC("displacement", [], [0.0, None, None], nodes=[c2]),
                       BC("pressure", [top], [-SIG]), BC("pressure", [bot], [-SIG])], log=lambda *x: None)
    row = evaluate_cracks(m, r, [crack])[0]
    print(f"a/W={q:.2f}  節点 {m.n_nodes:,}  面 {m.n_faces}（き裂面 {m.crack_faces}）  メッシュ {tm:.1f}s  合計 {time.time() - t0:.1f}s")
    print(f"   Tada K = {K:.1f}  J から {row['K（J から）[MPa√mm]']:.1f} ({row['K（J から）[MPa√mm]'] / K - 1:+.2%})  "
          f"変位から K_I {row['K_I（変位）[MPa√mm]']:.1f} ({row['K_I（変位）[MPa√mm]'] / K - 1:+.2%})  "
          f"K_II {row['K_II（変位）[MPa√mm]']:.2f}  積分路 " + " ".join(f"{row[f'J 積分路 {k}']:.4g}" for k in range(1, 6)))

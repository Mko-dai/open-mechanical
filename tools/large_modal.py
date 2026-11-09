"""大きなモデルのモーダル解析の時間（メモリ削減後の確認）。logs/large_modal.txt に書く"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.materials import LIBRARY  # noqa: E402
from openmech.mesh import structured_box  # noqa: E402
from openmech.modal import solve_modal  # noqa: E402
from openmech.solver import BC  # noqa: E402

out = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "large_modal.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")
    out.flush()


S = LIBRARY["構造用鋼 (Structural Steel)"]
for n in [int(x) for x in (sys.argv[1:] or ["16", "22"])]:
    m = structured_box(400, 40, 40, n * 5, n, n).to_quadratic()
    m.group_faces()
    f0 = [x["id"] for x in m.face_info() if abs(x["centroid"][0]) < 1e-6][0]
    P(f"=== 節点 {m.n_nodes:,}（自由度 {3 * m.n_nodes:,}）")
    t = time.time()
    md = solve_modal(m, S, [BC("fixed", [f0])], n_modes=10, log=lambda *a: P("   ", *a) if "モード" not in a[0] else None)
    P(f"  モーダル 10 モード {time.time() - t:.1f} s / 1 次 {md.freqs[0]:.4f} Hz")
P("完了")

"""大きなモデルでの解析時間（静解析・モーダル・定常熱）。Mac で実行して logs/large_check.txt に書く"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openmech.materials import LIBRARY  # noqa: E402
from openmech.mesh import structured_box  # noqa: E402
from openmech.modal import solve_modal  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402
from openmech.thermal import solve_thermal  # noqa: E402

out = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "large_check.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")
    out.flush()


S = LIBRARY["構造用鋼 (Structural Steel)"]
for n in (10, 16, 22):
    m = structured_box(400, 40, 40, n * 5, n, n).to_quadratic()
    m.group_faces()
    f0 = [x["id"] for x in m.face_info() if abs(x["centroid"][0]) < 1e-6][0]
    others = [x["id"] for x in m.face_info() if x["id"] != f0]
    P(f"=== 節点 {m.n_nodes:,}（構造の自由度 {3 * m.n_nodes:,}）")
    t = time.time()
    r = solve(m, S, [BC("fixed", [f0]), BC("gravity", [], [0, 0, -9806.65])], log=lambda *a: None)
    P(f"  静解析 {time.time() - t:.1f} s（{r.info.get('ソルバー')}）")
    t = time.time()
    md = solve_modal(m, S, [BC("fixed", [f0])], n_modes=10, log=lambda *a: P("   ", *a))
    P(f"  モーダル 10 モード {time.time() - t:.1f} s / 1 次 {md.freqs[0]:.4f} Hz")
    t = time.time()
    th = solve_thermal(m, S, [BC("temperature", [f0], [100.0]), BC("convection", others, [10.0, 22.0])],
                       log=lambda *a: P("   ", *a))
    P(f"  定常熱 {time.time() - t:.1f} s / 最低温度 {th[-1].T.min():.4f} °C")
P("完了")

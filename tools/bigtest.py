"""大規模モデルの解析時間とメモリ（開発用）: 立方体 Tet10 を 約 50 万 / 100 万節点で解く"""
import os, sys, time, resource, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from openmech.mesh import structured_box
from openmech.materials import Material
from openmech.solver import BC, solve
from openmech.parallel import set_threads, available_cores

try:
    mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30
except Exception:
    mem = float("nan")
print(f"メモリ {mem:.1f} GB / コア {available_cores()}", flush=True)
mat = Material("鋼", 200000, 0.3, 7850)
set_threads(0)
for n in [int(a) for a in (sys.argv[1:] or ["30", "40", "50"])]:
    t0 = time.time()
    m = structured_box(50, 50, 50, n, n, n).to_quadratic()
    tm = time.time() - t0
    print(f"--- n={n}: 節点 {m.n_nodes:,} / 要素 {m.n_elems:,} / 自由度 {3*m.n_nodes:,}（メッシュ作成 {tm:.1f} s）", flush=True)
    r = solve(m, mat, [BC("fixed", [4]), BC("pressure", [5], [10.0])], log=lambda s: print("  ", s, flush=True))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # macOS は bytes
    i = r.info
    print(f"   組立 {i['組立時間 [s]']} s / 求解 {i['求解時間 [s]']} s / 合計 {i['合計時間 [s]']} s / 最大メモリ {rss:.1f} GB / 最大変位 {r.disp_mag.max():.6g}", flush=True)
    try:
        sw = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode().strip()
        print(f"   スワップ: {sw}", flush=True)
    except Exception:
        pass
    del m, r

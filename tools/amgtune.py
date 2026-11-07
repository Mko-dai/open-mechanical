"""AMG 設定の比較（開発用）"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np, scipy.sparse as sp, pyamg
from openmech.speedtest import cantilever, block
from openmech.solver import assemble_stiffness, elasticity_matrix, build_load_and_constraints, _rigid_modes, _pcg
from openmech.materials import Material
from openmech.parallel import set_threads
set_threads(0)
mat = Material("鋼", 200000, 0.3, 7850, 250, 400)

def system(m, bcs):
    K = assemble_stiffness(m, elasticity_matrix(mat.E, mat.nu))
    F, fixed = build_load_and_constraints(m, mat, bcs)
    free = np.ones(K.shape[0], bool); free[list(fixed)] = False
    # 節点単位で自由度がそろうように: 固定は節点全体（ここでは固定支持のみ）
    A = K[free][:, free].tocsr(); b = F[free]
    ids = np.nonzero(free)[0]
    coords = np.column_stack([m.nodes[ids // 3], ids % 3])
    return A, b, coords

from openmech.amg import ChebyshevAMG
cfgs = {
 "A pyamg標準":        dict(bsr=False, kw=dict(strength="symmetric")),
 "L cheb3 mc3000":     dict(bsr=False, cheb=3, kw=dict(strength="symmetric", max_coarse=3000)),
 "M cheb2 mc3000":     dict(bsr=False, cheb=2, kw=dict(strength="symmetric", max_coarse=3000)),
 "N cheb4 mc3000":     dict(bsr=False, cheb=4, kw=dict(strength="symmetric", max_coarse=3000)),
 "O cheb3 mc1000":     dict(bsr=False, cheb=3, kw=dict(strength="symmetric", max_coarse=1000)),
}
for name, fn, n in [("片持ち梁", cantilever, 40), ("立方体", block, 18), ("立方体", block, 26)]:
    m, bcs = fn(n)
    A, b, coords = system(m, bcs)
    print(f"--- {name} dof {A.shape[0]:,}", flush=True)
    B = _rigid_modes(coords, A.shape[0])
    for cn, c in cfgs.items():
        try:
            t = time.time()
            AA = A.tobsr(blocksize=(3, 3)) if c["bsr"] else A
            kw = dict(max_coarse=1000); kw.update(c["kw"])
            ml = pyamg.smoothed_aggregation_solver(AA, B=B, symmetry="hermitian", **kw)
            M = ChebyshevAMG(ml, c["cheb"]) if c.get("cheb") else ml.aspreconditioner(cycle="V")
            ts = time.time() - t
            t = time.time()
            x, info, it = _pcg(A, b, M, tol=1e-10)
            tc = time.time() - t
            r = np.random.rand(A.shape[0])
            t = time.time(); M(r); tv = time.time() - t
            t = time.time(); A @ r; tm = time.time() - t
            sizes = "/".join(str(l.A.shape[0]) for l in ml.levels)
            print(f"{cn:22s} Vcycle {tv:.3f} matvec {tm:.3f} sizes {sizes}")
            res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            print(f"{cn:22s} res {res:.1e} setup {ts:6.2f}  solve {tc:6.2f}  it {it:4d}  levels {len(ml.levels)} cx {ml.operator_complexity():.2f}", flush=True)
        except Exception as e:
            print(cn, "失敗", e, flush=True)

"""シェル・2次元要素の非線形（塑性）用のニュートン・ラフソン法（荷重増分＋自動時間ステップ）"""
from __future__ import annotations

import numpy as np

from .solver import _linear_solve


class _NotConverged(RuntimeError):
    pass


def solve_incremental(internal, commit, F_ext, fixed, ndof, settings, log=print, free_mask=None, follower=None):
    """internal(u) -> (f_int, K)。commit() で積分点の状態を確定。

    荷重は 0 → F_ext、強制変位は 0 → 値 に直線で増やす（サブステップ）。収束しなければ増分を半分に。
    follower(u) -> 変形後の形で決まる荷重（面に垂直なままの圧力など。全荷重のときの値）。 接線には入れない
    戻り値: u
    """
    cd = np.array(sorted(fixed), dtype=np.int64)
    cv = np.array([fixed[int(d)] for d in cd])
    free = np.ones(ndof, bool)
    free[cd] = False
    if free_mask is not None:
        free &= free_mask
    nsub = max(1, int(getattr(settings, "initial_substeps", 1) or 1), 5)
    max_sub = max(nsub, int(getattr(settings, "max_substeps", 10) or 10) * 4)
    tol = float(getattr(settings, "force_convergence", 0.005) or 0.005)
    max_it = int(getattr(settings, "max_iterations", 50) or 50)
    u = np.zeros(ndof)
    t, dt = 0.0, 1.0 / nsub

    def equilibrium(t1, u0):
        u = u0.copy()
        F = t1 * F_ext
        vals = t1 * cv
        for it in range(1, max_it + 1):
            pred = it == 1
            ue = u0 if pred else u
            fint, K = internal(ue)
            R = F - fint
            if follower is not None:
                R = R + t1 * follower(ue)
            if pred and len(cd):
                R = R - K[:, cd] @ (vals - u0[cd])
            Kff = K[free][:, free].tocsc()
            du = _linear_solve(Kff, R[free], lambda *a: None, method="direct")
            if not np.all(np.isfinite(du)):
                raise RuntimeError("剛性マトリクスが特異です。拘束が不足していないか確認してください。")
            if pred:
                u[cd] = vals
            u[free] += du
            if it > 1 or True:
                fint2, _ = internal(u)
                R2 = F - fint2
                if follower is not None:
                    R2 = R2 + t1 * follower(u)
                ref = max(np.linalg.norm(F[free]), np.linalg.norm(R2[~free]), 0.01)
                crit = np.linalg.norm(R2[free]) / ref
                if crit <= tol:
                    log(f"  時間 {t1:.4g}: {it} 回で収束（力の残差比 {crit:.2e}）")
                    return u
        raise _NotConverged(f"時間 {t1:.4g} で収束しませんでした")

    while t < 1.0 - 1e-12:
        t1 = min(1.0, t + dt)
        try:
            u_new = equilibrium(t1, u)
        except _NotConverged as e:
            if dt / 2 >= 1.0 / max_sub:
                dt /= 2
                log(f"  {e} → 増分を半分に（{dt:.4g}）")
                continue
            raise RuntimeError(f"{e}。解析設定でサブステップ数を増やしてください。")
        internal(u_new)
        commit()
        u, t = u_new, t1
    return u


def plane_stress_return(model, e_inplane, ep, alpha, pbar, ez0, iters=12):
    """平面応力（σz = 0）の弾塑性: 3次元のリターンマッピングで εz を反復して決める。

    e_inplane: (n,3) [εx, εy, γxy]（工学せん断）
    戻り値: σ (n,3) [σx, σy, τxy], C (n,3,3) 縮約接線, ep, alpha, pbar, εz
    """
    from .plasticity import W_EPS, W_SIG
    n = len(e_inplane)
    ez = ez0.copy()
    for _ in range(iters):
        ev = np.zeros((n, 6))
        ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3] = e_inplane[:, 0], e_inplane[:, 1], ez, e_inplane[:, 2]
        sM, epn, aln, pbn, CM = model.update(ev * W_EPS, ep, alpha, pbar)
        sv = sM / W_SIG
        Cv = CM / W_SIG[None, :, None] * W_EPS[None, None, :]
        r = sv[:, 2]
        if np.all(np.abs(r) <= 1e-9 * max(model.sy0, 1.0)):
            break
        ez = ez - r / Cv[:, 2, 2]
    ip = [0, 1, 3]
    Caa = Cv[:, ip][:, :, ip]
    Ca2 = Cv[:, ip, 2]
    C2a = Cv[:, 2, ip]
    C = Caa - np.einsum("ei,ej->eij", Ca2, C2a) / Cv[:, 2, 2][:, None, None]
    return sv[:, ip], C, epn, aln, pbn, ez

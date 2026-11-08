"""過渡（動的）構造解析

  陰解法（Ansys の「過渡構造」と同じ）: Newmark-β / HHT-α 法。整合質量。大きな時間増分でも安定。
      振動・衝撃応答・ゆっくりした動的荷重向き。剛体壁との接触はペナルティ法（座標軸に垂直な壁）。
  陽解法（LS-DYNA と同じ）: 中心差分法。集中質量（HRZ）。時間増分は安定限界から自動決定。
      落下・衝突など短時間の現象向き。質量スケーリング（LS-DYNA の DT2MS 相当）、剛体壁（任意の向き・摩擦）。

単位: mm, t, s, N, MPa（速度 mm/s、加速度 mm/s²）。いずれも線形弾性・微小変形（剛体壁の接触は非線形）。
荷重（力・圧力・重力）は「ステップ（t=0 から一定）」または「ランプ（ramp_time までに直線で増加）」。
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np
import scipy.sparse as sp

from .solver import (BC, Result, _element_K, _linear_solve, _quadrature, assemble_stiffness,
                     build_load_and_constraints, material_D, nodal_stress, shape_dN_dxi)

METHODS = {"implicit": "陰解法（Newmark / HHT-α）", "explicit": "陽解法（中心差分・LS-DYNA 方式）"}


@dataclass
class DynamicSettings:
    method: str = "implicit"           # implicit / explicit
    end_time: float = 1e-3             # 終了時間 [s]
    dt: float = 0.0                    # 時間増分 [s]（0 = 自動）
    n_frames: int = 20                 # 結果を保存する時刻の数
    init_velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # 初速度 [mm/s]（全節点）
    ramp_time: float = 0.0             # 荷重を直線で増やす時間 [s]（0 = ステップ荷重）
    alpha_damping: float = 0.0         # レイリー減衰 質量比例 α [1/s]
    beta_damping: float = 0.0          # レイリー減衰 剛性比例 β [s]
    hht_alpha: float = 0.0             # HHT-α（0 = Newmark 平均加速度法、-0.1 程度で高周波を減衰）
    mass_scaling_dt: float = 0.0       # 陽解法の目標時間増分 [s]（0 = 質量スケーリングなし）
    dt_safety: float = 0.9             # 陽解法の安定限界に掛ける係数（LS-DYNA の TSSFAC）
    wall: dict = field(default_factory=dict)  # 剛体壁 {"normal": [nx,ny,nz]（壁から物体側）, "point": [x,y,z], "friction": μ}

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        s = DynamicSettings()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


# ---------------------------------------------------------------------------- 質量マトリクス

def _poly_mul(a, b):
    out = {}
    for (ea, ca), (eb, cb) in product(a.items(), b.items()):
        e = tuple(x + y for x, y in zip(ea, eb))
        out[e] = out.get(e, 0.0) + ca * cb
    return out


def _tet_mass_ref(nn):
    """要素の整合質量 = ρ V C（C は体積座標の多項式を厳密に積分した係数。合計 1）"""
    if nn == 4:
        return (np.ones((4, 4)) + np.eye(4)) / 20.0
    e = lambda *idx: tuple(1 if i in idx else 0 for i in range(4))  # noqa: E731
    N = []
    for i in range(4):
        N.append({tuple(2 if k == i else 0 for k in range(4)): 2.0, e(i): -1.0})
    from .solver import _EDGES10
    for i, j in _EDGES10:
        N.append({e(i, j): 4.0})
    C = np.zeros((10, 10))
    for a in range(10):
        for b in range(10):
            s = 0.0
            for ex, c in _poly_mul(N[a], N[b]).items():
                s += c * 6.0 * math.prod(math.factorial(x) for x in ex) / math.factorial(sum(ex) + 3)
            C[a, b] = s
    return C


def element_rho(mesh, mats, elem_mat):
    return np.array([m.density_t_mm3 for m in mats])[elem_mat]


def lumped_mass(mesh, rho_e):
    """集中質量（HRZ: 整合質量の対角を、要素の質量に合うように拡大）→ 節点ごとの質量 (N,)"""
    nn = mesh.elems.shape[1]
    d = np.diag(_tet_mass_ref(nn))
    w = d / d.sum()
    me = rho_e * np.abs(mesh.element_volumes())
    return np.bincount(mesh.elems.ravel(), (me[:, None] * w[None, :]).ravel(), mesh.n_nodes)


def consistent_mass(mesh, rho_e):
    """整合質量（自由度 3N の CSR）"""
    nn = mesh.elems.shape[1]
    C = _tet_mass_ref(nn)
    me = rho_e * np.abs(mesh.element_volumes())
    r = np.repeat(mesh.elems, nn, axis=1).ravel()
    c = np.tile(mesh.elems, (1, nn)).ravel()
    v = (me[:, None, None] * C[None]).ravel()
    Mn = sp.csr_matrix((v, (r, c)), shape=(mesh.n_nodes, mesh.n_nodes))
    return sp.kron(Mn, sp.identity(3), format="csr")


# ---------------------------------------------------------------------------- 安定時間増分

def element_dt(mesh, D, elem_mat, rho_e, chunk=8000):
    """要素ごとの安定時間増分 2/ω_max（集中質量の要素固有値。全体の安定限界はこの最小値以上）"""
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    d = np.diag(_tet_mass_ref(nn))
    w = d / d.sum()
    vol = np.abs(mesh.element_volumes())
    out = np.empty(mesh.n_elems)
    for a in range(0, mesh.n_elems, chunk):
        b = min(mesh.n_elems, a + chunk)
        X = mesh.nodes[mesh.elems[a:b]]
        De = D[elem_mat[a:b]] if D.ndim == 3 else np.broadcast_to(D, (b - a, 6, 6))
        Ke = _element_K(X, De, dN, wts)
        m = np.repeat((rho_e[a:b] * vol[a:b])[:, None] * w[None, :], 3, axis=1)      # (E,3nn)
        s = 1.0 / np.sqrt(m)
        A = Ke * s[:, :, None] * s[:, None, :]
        lam = np.linalg.eigvalsh(A)[:, -1]
        out[a:b] = 2.0 / np.sqrt(np.maximum(lam, 1e-300))
    return out


# ---------------------------------------------------------------------------- 共通の準備

def _prepare(mesh, material, bcs, elem_materials):
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    D = np.stack([material_D(m) for m in mats])
    rho_e = element_rho(mesh, mats, elem_mat)
    F, fixed = build_load_and_constraints(mesh, mats[0], bcs, rho_e=rho_e)
    return mats, elem_mat, D, rho_e, F, fixed


def _ramp(t, ramp):
    return 1.0 if ramp <= 0 else min(1.0, t / ramp)


def _wall(s, mesh):
    w = s.wall or {}
    if not w or not w.get("normal"):
        return None
    n = np.asarray(w["normal"], float)
    n = n / np.linalg.norm(n)
    p = np.asarray(w.get("point", [0, 0, 0]), float)
    cand = np.unique(mesh.boundary_tris_full) if hasattr(mesh, "boundary_tris_full") else np.arange(mesh.n_nodes)
    return {"n": n, "p": p, "mu": float(w.get("friction", 0.0)), "nodes": cand}


def _frame_result(mesh, mats, elem_mat, D, u, v, t, info, base_material):
    stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat)
    r = Result(mesh, base_material, u.reshape(-1, 3).copy(), stress, strain.astype(np.float32), {}, dict(info),
               mats, elem_mat)
    r.time = float(t)
    vv = v.reshape(-1, 3)
    r.extra = {"速度 [mm/s]": np.linalg.norm(vv, axis=1), "速度 X [mm/s]": vv[:, 0].copy(),
               "速度 Y [mm/s]": vv[:, 1].copy(), "速度 Z [mm/s]": vv[:, 2].copy()}
    return r


def _frame_times(s):
    n = max(1, int(s.n_frames))
    return np.linspace(s.end_time / n, s.end_time, n)


# ---------------------------------------------------------------------------- 陽解法

def solve_explicit(mesh, material, bcs, settings: DynamicSettings, elem_materials=None, log=print, progress=None):
    """中心差分法（LS-DYNA と同じ陽解法）。集中質量・自動時間増分・質量スケーリング・剛体壁"""
    from .parallel import ParallelMatvec
    t0 = time.time()
    s = settings
    mats, elem_mat, D, rho_e, F0, fixed = _prepare(mesh, material, bcs, elem_materials)
    log("陽解法: 剛性マトリクスを組み立て中…")
    K = assemble_stiffness(mesh, D, elem_mat=elem_mat)
    Kmv = ParallelMatvec(K)
    nn = mesh.elems.shape[1]
    d = np.diag(_tet_mass_ref(nn))
    wloc = d / d.sum()
    m_e = rho_e * np.abs(mesh.element_volumes())
    mass0 = float(m_e.sum())
    log("安定時間増分を計算中（要素ごとの固有値）…")
    dte = element_dt(mesh, D, elem_mat, rho_e) * s.dt_safety
    dt_min = float(dte.min())
    added = 0.0
    if s.mass_scaling_dt and s.mass_scaling_dt > dt_min:
        scale = np.maximum(1.0, (s.mass_scaling_dt / dte) ** 2)
        added = float((m_e * (scale - 1)).sum())
        m_e = m_e * scale
        dt = float(s.mass_scaling_dt)
        log(f"  質量スケーリング: 時間増分 {dt_min:.3e} → {dt:.3e} s、追加質量 {added / mass0 * 100:.2f} %"
            f"（{int((scale > 1).sum()):,} 要素）")
    else:
        dt = dt_min
    if s.dt and 0 < s.dt < dt:
        dt = float(s.dt)
    m = np.bincount(mesh.elems.ravel(), (m_e[:, None] * wloc[None, :]).ravel(), mesh.n_nodes)
    m3 = np.repeat(m, 3)
    minv = 1.0 / m3
    ndof = 3 * mesh.n_nodes
    cd = np.fromiter(fixed.keys(), dtype=np.int64)
    cv = np.fromiter(fixed.values(), dtype=float)
    nsteps = int(math.ceil(s.end_time / dt))
    dt = s.end_time / nsteps
    log(f"  時間増分 {dt:.3e} s × {nsteps:,} ステップ（終了時間 {s.end_time:g} s、自由度 {ndof:,}）")
    u = np.zeros(ndof)
    v = np.tile(np.asarray(s.init_velocity, float), mesh.n_nodes)
    v[cd] = 0.0
    u[cd] = cv * _ramp(0.0, s.ramp_time)
    wall = _wall(s, mesh)
    if wall is not None:
        xw = mesh.nodes[wall["nodes"]]
        gap0 = (xw - wall["p"]) @ wall["n"]
        wdofs = (3 * wall["nodes"][:, None] + np.arange(3)).ravel()
        mw = m[wall["nodes"]]
    aR, bR = s.alpha_damping, s.beta_damping

    def accel(u, v, t):
        ku = Kmv(u)
        f = F0 * _ramp(t, s.ramp_time) - ku
        if aR:
            f -= aR * m3 * v
        if bR:
            f -= bR * Kmv(v)
        a = f * minv
        a[cd] = 0.0
        return a, ku

    a, ku = accel(u, v, 0.0)
    frames, ft = [], list(_frame_times(s))
    hist = {"時間 [s]": [], "運動エネルギー [mJ]": [], "ひずみエネルギー [mJ]": [], "接触力 [N]": [],
            "最大変位 [mm]": []}
    every = max(1, nsteps // 400)
    info = {"解析": METHODS["explicit"], "時間増分 [s]": dt, "ステップ数": nsteps, "安定限界 [s]": dt_min,
            "追加質量 [%]": added / mass0 * 100, "質量 [kg]": (mass0 + added) * 1000}
    t_last = time.time()
    imp, t_rec = 0.0, 0.0       # 記録の間に壁から受けた力積（接触力は「力積 ÷ 時間」の平均で記録）
    for k in range(1, nsteps + 1):
        t = k * dt
        vh = v + 0.5 * dt * a
        u += dt * vh
        if len(cd):
            u[cd] = cv * _ramp(t, s.ramp_time)
            vh[cd] = 0.0
        if wall is not None:
            nidx = wall["nodes"]
            U = u.reshape(-1, 3)
            g = gap0 + U[nidx] @ wall["n"]
            hit = g < 0
            if np.any(hit):
                hn = nidx[hit]
                U[hn] -= g[hit, None] * wall["n"][None, :]           # 壁の上に戻す
                V = vh.reshape(-1, 3)
                vn = V[hn] @ wall["n"]
                dvn = np.where(vn < 0, -vn, 0.0)                     # 壁に向かう速度を止める
                V[hn] += dvn[:, None] * wall["n"][None, :]
                if wall["mu"] > 0:
                    vt = V[hn] - (V[hn] @ wall["n"])[:, None] * wall["n"][None, :]
                    vtn = np.linalg.norm(vt, axis=1)
                    red = np.minimum(1.0, wall["mu"] * (dvn + np.abs(g[hit]) / dt) / np.maximum(vtn, 1e-300))
                    V[hn] -= vt * red[:, None]
                imp += float((mw[hit] * dvn).sum())          # 壁が止めた運動量 = 力積
        a, ku = accel(u, vh, t)
        v = vh + 0.5 * dt * a
        if k % every == 0 or k == nsteps:
            hist["時間 [s]"].append(t)
            hist["運動エネルギー [mJ]"].append(0.5 * float((m3 * v * v).sum()))
            hist["ひずみエネルギー [mJ]"].append(0.5 * float(u @ ku))
            hist["接触力 [N]"].append(imp / max(t - t_rec, dt))
            imp, t_rec = 0.0, t
            hist["最大変位 [mm]"].append(float(np.abs(u).max()))
        while ft and t >= ft[0] - 0.5 * dt:
            frames.append(_frame_result(mesh, mats, elem_mat, D, u, v, ft.pop(0), info, material))
        if progress and k % max(1, nsteps // 200) == 0:
            progress(k / nsteps, f"陽解法 {k:,}/{nsteps:,} ステップ（t = {t:.3e} s）")
        if time.time() - t_last > 10:
            t_last = time.time()
            log(f"  {k:,}/{nsteps:,} ステップ（{k / nsteps * 100:.0f}%、経過 {time.time() - t0:.0f} s）")
    info["計算時間 [s]"] = round(time.time() - t0, 1)
    info["最大接触力 [N]"] = max(hist["接触力 [N]"] or [0.0])
    for r in frames:
        r.info.update(info)
        r.history = hist
    log(f"陽解法 完了: {nsteps:,} ステップ、{info['計算時間 [s]']} s")
    return frames


# ---------------------------------------------------------------------------- 陰解法

def solve_implicit(mesh, material, bcs, settings: DynamicSettings, elem_materials=None, log=print, progress=None):
    """HHT-α 法（α=0 で Newmark 平均加速度法）。整合質量・レイリー減衰・剛体壁（ペナルティ）"""
    t0 = time.time()
    s = settings
    mats, elem_mat, D, rho_e, F0, fixed = _prepare(mesh, material, bcs, elem_materials)
    log("陰解法: 剛性・質量マトリクスを組み立て中…")
    K = assemble_stiffness(mesh, D, elem_mat=elem_mat)
    M = consistent_mass(mesh, rho_e)
    ml = np.asarray(M.sum(axis=1)).ravel()
    ndof = K.shape[0]
    al = float(s.hht_alpha)
    if not (-1 / 3 - 1e-12 <= al <= 0):
        raise ValueError("HHT-α は -1/3 〜 0 の範囲で指定してください")
    beta = (1 - al) ** 2 / 4
    gamma = 0.5 - al
    dt = float(s.dt) if s.dt else s.end_time / max(20, int(s.n_frames) * 5)
    nsteps = int(math.ceil(s.end_time / dt - 1e-9))
    dt = s.end_time / nsteps
    c0, c1, c2, c3 = 1 / (beta * dt * dt), gamma / (beta * dt), 1 / (beta * dt), 1 / (2 * beta) - 1
    aR, bR = s.alpha_damping, s.beta_damping
    C = (aR * M if aR else None)
    if bR:
        C = bR * K if C is None else C + bR * K
    Keff = c0 * M + (1 + al) * K
    if C is not None:
        Keff = Keff + (1 + al) * c1 * C
    Keff = Keff.tocsr()
    cd = np.fromiter(fixed.keys(), dtype=np.int64)
    cv = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(ndof, bool)
    free[cd] = False
    fidx = np.nonzero(free)[0]
    Kff = Keff[free][:, free].tocsr()
    log(f"  時間増分 {dt:.3e} s × {nsteps:,} ステップ（HHT α = {al:g}、自由度 {free.sum():,}）")
    # 剛体壁（座標軸に垂直な壁のみ）: ペナルティばね
    wall = _wall(s, mesh)
    if wall is not None:
        ax = int(np.argmax(np.abs(wall["n"])))
        if abs(abs(wall["n"][ax]) - 1) > 1e-6:
            raise ValueError("陰解法の剛体壁は座標軸に垂直な向きにしてください（斜めの壁は陽解法で）")
        sgn = np.sign(wall["n"][ax])
        wn = wall["nodes"]
        gap0 = (mesh.nodes[wn, ax] - wall["p"][ax]) * sgn
        wdof = 3 * wn + ax
        kpen = 50.0 * float(np.median(Keff.diagonal()))
        gpos = -np.ones(ndof, dtype=np.int64)
        gpos[fidx] = np.arange(len(fidx))
    direct = len(fidx) <= 150_000
    lu_cache = {}

    def solve_ff(A_extra_idx, A_extra_val, rhs, x0=None):
        if direct:
            key = tuple(A_extra_idx.tolist())
            if key not in lu_cache:
                from scipy.sparse.linalg import splu
                A = Kff
                if len(A_extra_idx):
                    A = (Kff + sp.csr_matrix((A_extra_val, (A_extra_idx, A_extra_idx)), shape=Kff.shape)).tocsc()
                lu_cache.clear()
                lu_cache[key] = splu(A.tocsc())
            return lu_cache[key].solve(rhs)
        A = Kff
        if len(A_extra_idx):
            A = (Kff + sp.csr_matrix((A_extra_val, (A_extra_idx, A_extra_idx)), shape=Kff.shape)).tocsr()
        coords = np.column_stack([mesh.nodes[fidx // 3], fidx % 3])
        return _linear_solve(A, rhs, lambda *a: None, coords=coords, method="amg")

    u = np.zeros(ndof)
    v = np.tile(np.asarray(s.init_velocity, float), mesh.n_nodes)
    v[cd] = 0.0
    u[cd] = cv * _ramp(0.0, s.ramp_time)
    Fprev = F0 * _ramp(0.0, s.ramp_time)
    f0 = Fprev - K @ u - (C @ v if C is not None else 0)
    a = f0 / ml
    a[cd] = 0.0
    frames, ft = [], list(_frame_times(s))
    hist = {"時間 [s]": [], "運動エネルギー [mJ]": [], "ひずみエネルギー [mJ]": [], "接触力 [N]": [],
            "最大変位 [mm]": []}
    info = {"解析": METHODS["implicit"], "時間増分 [s]": dt, "ステップ数": nsteps, "HHT α": al}
    active = np.zeros(0, dtype=np.int64)
    for k in range(1, nsteps + 1):
        t = k * dt
        Fn = F0 * _ramp(t, s.ramp_time)
        r = (1 + al) * Fn - al * Fprev - K @ u + M @ (c2 * v + c3 * a)
        if C is not None:
            r -= (1 + al) * (C @ ((1 - gamma / beta) * v + dt * (1 - gamma / (2 * beta)) * a)) - al * (C @ v)
        du = np.zeros(ndof)
        du[cd] = cv * _ramp(t, s.ramp_time) - u[cd]
        rf = r[free] - (Keff[free][:, cd] @ du[cd] if len(cd) else 0)
        fc = 0.0
        for _it in range(15):
            if wall is not None and len(active):
                idx = gpos[wdof[active]]
                ok = idx >= 0
                idx, act = idx[ok], active[ok]
                # 壁の位置まで押し戻すばね: kpen (gap + u_n + du_n) = 0
                un = u[wdof[act]] * sgn
                rhs = rf.copy()
                np.add.at(rhs, idx, -kpen * (gap0[act] + un) * sgn)
                duf = solve_ff(idx, np.full(len(idx), kpen), rhs)
            else:
                duf = solve_ff(np.zeros(0, dtype=np.int64), np.zeros(0), rf)
            du[free] = duf
            if wall is None:
                break
            g = gap0 + (u[wdof] + du[wdof]) * sgn
            new_active = np.union1d(active, np.nonzero(g < -1e-9)[0])
            if len(active):
                # 引っ張られている（壁から離れようとしている）点は外す
                fpen = -kpen * g[active]
                new_active = np.setdiff1d(new_active, active[fpen < 0])
            if np.array_equal(new_active, active):
                fc = float(np.maximum(-kpen * g[active], 0).sum()) if len(active) else 0.0
                break
            active = new_active
        a_new = c0 * du - c2 * v - c3 * a
        v = v + dt * ((1 - gamma) * a + gamma * a_new)
        a = a_new
        u = u + du
        Fprev = Fn
        hist["時間 [s]"].append(t)
        hist["運動エネルギー [mJ]"].append(0.5 * float(v @ (M @ v)))
        hist["ひずみエネルギー [mJ]"].append(0.5 * float(u @ (K @ u)))
        hist["接触力 [N]"].append(fc)
        hist["最大変位 [mm]"].append(float(np.abs(u).max()))
        while ft and t >= ft[0] - 0.5 * dt:
            frames.append(_frame_result(mesh, mats, elem_mat, D, u, v, ft.pop(0), info, material))
        if progress:
            progress(k / nsteps, f"陰解法 {k}/{nsteps} ステップ（t = {t:.3e} s）")
    info["計算時間 [s]"] = round(time.time() - t0, 1)
    info["最大接触力 [N]"] = max(hist["接触力 [N]"] or [0.0])
    for fr in frames:
        fr.info.update(info)
        fr.history = hist
    log(f"陰解法 完了: {nsteps:,} ステップ、{info['計算時間 [s]']} s")
    return frames


def solve_dynamic(mesh, material, bcs, settings: DynamicSettings, elem_materials=None, log=print, progress=None):
    if getattr(mesh, "is_shell", False) or getattr(mesh, "is_2d", False):
        raise ValueError("過渡解析は現在ソリッド（四面体）要素に対応しています")
    f = solve_explicit if settings.method == "explicit" else solve_implicit
    return f(mesh, material, bcs, settings, elem_materials=elem_materials, log=log, progress=progress)


__all__ = ["DynamicSettings", "METHODS", "solve_dynamic", "solve_explicit", "solve_implicit",
           "lumped_mass", "consistent_mass", "element_dt", "BC"]

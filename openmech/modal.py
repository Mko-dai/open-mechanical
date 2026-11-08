"""モーダル解析（固有振動数・モード形）と線形座屈解析。Ansys Mechanical の「モーダル」「固有値座屈」に相当。

モーダル:  K φ = ω² M φ        （支持なしでも可。剛体モードは 0 Hz 付近に出る）
座屈:      (K + λ K_σ) φ = 0   （K_σ は静解析の応力から作る幾何剛性。λ = 座屈荷重係数）

モード形は Ansys と同じく質量で正規化（φᵀ M φ = 1）。刺激係数・有効質量も Ansys の「刺激係数の要約」と同じ定義:
  Γ_i = φ_iᵀ M r（r: 方向ごとの剛体並進・回転）、 有効質量 = Γ_i²
単位系: mm, N, t, s（振動数は Hz）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import Material
from .mesh import Mesh
from .solver import (BC, Result, _B_matrix, _D_of, _dN_dx, _quadrature, assemble_stiffness,
                     build_load_and_constraints, factorize, material_D, nodal_stress, shape_dN_dxi)


SUPPORT_KINDS = ("fixed", "displacement", "symmetry", "frictionless")


def _materials(mesh, material, elem_materials):
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    D = np.stack([material_D(m) for m in mats])
    rho_e = np.array([m.density_t_mm3 for m in mats])[elem_mat]
    return mats, elem_mat, D, rho_e


def _supports_only(bcs):
    """モーダルでは荷重は使わず、支持だけを使う（強制変位は 0 として扱う＝Ansys と同じ）"""
    out = []
    for b in bcs:
        if b.kind == "displacement":
            out.append(BC("displacement", b.faces, [None if v is None else 0.0 for v in b.values], b.name, b.nodes))
        elif b.kind in SUPPORT_KINDS:
            out.append(b)
    return out


def _free_dofs(mesh, material, bcs, extra_fixed=None):
    _, fixed = build_load_and_constraints(mesh, material, _supports_only(bcs))
    if extra_fixed:
        fixed.update(extra_fixed)
    free = np.ones(3 * mesh.n_nodes, bool)
    if fixed:
        free[np.fromiter(fixed.keys(), dtype=np.int64)] = False
    return free


def _eig(Kff, Mff, n, sigma, log, label="固有値"):
    """一般化固有値問題 K x = λ M x の、sigma に近い n 個（シフト逆反復＋ランチョス）"""
    nd = Kff.shape[0]
    n = max(1, min(n, nd - 2))
    A = (Kff - sigma * Mff).tocsc() if sigma != 0 else Kff.tocsc()
    t0 = time.time()
    solve = factorize(A, log)
    OPinv = spla.LinearOperator((nd, nd), matvec=solve, dtype=float)
    ncv = min(nd - 1, max(3 * n + 1, n + 30))
    lam, vec = spla.eigsh(Kff, k=n, M=Mff, sigma=sigma, which="LM", OPinv=OPinv, ncv=ncv, tol=1e-10)
    log(f"  {label}: {n} 個（分解と反復 {time.time() - t0:.1f} s）")
    o = np.argsort(lam)
    return lam[o], vec[:, o]


@dataclass
class ModalResult:
    mesh: Mesh
    freqs: np.ndarray                 # [Hz]
    modes: np.ndarray                 # (n_modes, N, 3) 質量正規化したモード形
    participation: np.ndarray         # (n_modes, 6) 刺激係数 X,Y,Z,RX,RY,RZ
    eff_mass: np.ndarray              # (n_modes, 6) 有効質量 [t]（回転は t·mm²）
    total_mass: float                 # [t]
    info: dict = field(default_factory=dict)
    kind: str = "modal"               # "modal" / "buckling"
    load_factors: np.ndarray | None = None   # 座屈荷重係数
    prestress: Result | None = None

    @property
    def n_modes(self):
        return len(self.modes)

    def mode_result(self, i: int, material: Material | None = None) -> Result:
        """i 番目のモードを、静解析と同じ Result（変形の表示用。応力は相対値）にする"""
        u = self.modes[i]
        z = np.zeros((self.mesh.n_nodes, 6))
        r = Result(self.mesh, material or Material("-", 1, 0.3, 1), u, z, z, {}, dict(self.info))
        if self.kind == "buckling":
            r.info["座屈荷重係数"] = float(self.load_factors[i])
        else:
            r.info["振動数 [Hz]"] = float(self.freqs[i])
        return r

    def table(self) -> list[dict]:
        rows = []
        m = max(self.total_mass, 1e-300)
        cum = np.zeros(3)
        for i in range(self.n_modes):
            if self.kind == "buckling":
                rows.append({"モード": i + 1, "座屈荷重係数": float(self.load_factors[i])})
                continue
            cum += self.eff_mass[i, :3]
            rows.append({"モード": i + 1, "振動数 [Hz]": float(self.freqs[i]),
                         **{f"有効質量比 {a}": float(self.eff_mass[i, k] / m) for k, a in enumerate("XYZ")},
                         **{f"累積比 {a}": float(cum[k] / m) for k, a in enumerate("XYZ")}})
        return rows


def _rigid_vectors(mesh):
    """剛体の並進 X,Y,Z と回転 RX,RY,RZ（重心まわり）の変位ベクトル (3N, 6)"""
    x = mesh.nodes - mesh.nodes.mean(axis=0)
    n = mesh.n_nodes
    R = np.zeros((3 * n, 6))
    for c in range(3):
        R[c::3, c] = 1.0
    # 回転 θ × r
    for k, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
        R[a::3, 3 + k] = -x[:, b]
        R[b::3, 3 + k] = x[:, a]
    return R


def solve_modal(mesh: Mesh, material: Material, bcs: list[BC], n_modes: int = 6, f_min: float = 0.0,
                elem_materials=None, log=print, progress=None, prestress: Result | None = None,
                K_sigma=None) -> ModalResult:
    """モーダル解析。f_min [Hz] より上の n_modes 個を求める（支持がなければ剛体モード 6 個も含む）。
    prestress / K_sigma を渡すとプリストレス付きモーダル（K + K_σ）"""
    from .dynamics import consistent_mass
    t0 = time.time()
    mats, elem_mat, D, rho_e = _materials(mesh, material, elem_materials)
    if progress:
        progress(0.05, "剛性・質量マトリクスを組み立て中…")
    K = assemble_stiffness(mesh, D, elem_mat=elem_mat)
    if K_sigma is not None:
        K = K + K_sigma
    M = consistent_mass(mesh, rho_e)
    nb = 3 * mesh.n_nodes
    _, fixed = build_load_and_constraints(mesh, mats[0], _supports_only(bcs))
    T = None
    pilots = []
    from .remote import REMOTE_KINDS, extra_terms, pad, reduce_system
    if any(b.kind in REMOTE_KINDS for b in bcs):
        ex = extra_terms(mesh, [b if b.kind != "remote_displacement" else
                                BC(b.kind, b.faces, [None if v is None else 0.0 for v in b.values], b.name, b.nodes,
                                   point=b.point, options=b.options) for b in bcs],
                         float(np.mean(np.abs(K.diagonal()))))
        K = pad(K, ex.n) + ex.K()
        M = pad(M, ex.n) + ex.M()
        fixed.update(ex.fixed)
        pilots = ex.pilots
        if ex.mpc:
            Mdummy = np.zeros(ex.n)
            K, _, fixed, T = reduce_system(K, Mdummy, fixed, ex, float(np.mean(np.abs(K.diagonal()[:nb]))))
            M = (T.T @ M @ T).tocsr()
    free = np.ones(K.shape[0], bool)
    if fixed:
        free[np.fromiter(fixed.keys(), dtype=np.int64)] = False
    Kff, Mff = K[free][:, free], M[free][:, free]
    t_asm = time.time() - t0
    nfree_body = not any(b.kind in SUPPORT_KINDS + ("remote_displacement", "elastic_support", "cylindrical_support",
                                                    "spring") for b in bcs)
    # シフト: 支持なし（剛体モード 0 Hz）でも分解できるよう負の値に
    lam_ref = float(Kff.diagonal().mean() / max(Mff.diagonal().mean(), 1e-300))
    w_min = 2 * np.pi * f_min
    sigma = w_min ** 2 if f_min > 0 else -1e-6 * lam_ref
    if progress:
        progress(0.3, "固有値を計算中…")
    log(f"モーダル解析: 自由度 {int(free.sum()):,}" + ("（支持なし: 剛体モードを含む）" if nfree_body else ""))
    t1 = time.time()
    # 重根（剛体モード・対称形状の 2 重根）を取りこぼさないよう、多めに求めてから小さい順に n_modes 個を使う
    extra = 8 if nfree_body else 4
    lam, vec = _eig(Kff, Mff, n_modes + extra, sigma, log)
    lam, vec = lam[:n_modes], vec[:, :n_modes]
    t_sol = time.time() - t1
    lam = np.where(np.abs(lam) < 1e-9 * lam_ref, 0.0, lam)
    freqs = np.sqrt(np.maximum(lam, 0.0)) / (2 * np.pi)
    # 質量正規化
    nrm = np.sqrt(np.einsum("ij,ij->j", vec, Mff @ vec))
    vec = vec / nrm
    full = np.zeros((K.shape[0], vec.shape[1]))
    full[free] = vec
    if T is not None:
        full = T @ full
        M = None
    Mfull = consistent_mass(mesh, rho_e)
    R = _rigid_vectors(mesh)
    if full.shape[0] > nb:
        # パイロット（点質量）の自由度: 並進は剛体の並進、回転は剛体の回転
        from .remote import pad as _pad
        ex_n = full.shape[0]
        Rp = np.zeros((ex_n, 6))
        Rp[:nb] = R
        c0 = mesh.nodes.mean(axis=0)
        for _, p, d0 in pilots:
            x = p - c0
            Rp[d0:d0 + 3, :3] = np.eye(3)
            for k, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
                Rp[d0 + a, 3 + k] = -x[b]
                Rp[d0 + b, 3 + k] = x[a]
            Rp[d0 + 3:d0 + 6, 3:] = np.eye(3)
        R = Rp
        Mfull = _pad(Mfull, ex_n) + (ex.M() if pilots else 0)
    MR = Mfull @ R
    gamma = full.T @ MR                         # (n_modes, 6)
    eff = gamma ** 2
    total_mass = float((rho_e * np.abs(mesh.element_volumes())).sum()) + float(sum(
        (Mfull.diagonal()[d0] for _, _, d0 in pilots), 0.0))
    modes = full[:nb].T.reshape(len(freqs), mesh.n_nodes, 3)
    info = {"解析": "モーダル", "自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "モード数": len(freqs), "組立時間 [s]": round(t_asm, 2), "求解時間 [s]": round(t_sol, 2),
            "合計時間 [s]": round(time.time() - t0, 2), "質量 [kg]": total_mass * 1000}
    for i, f in enumerate(freqs):
        log(f"  モード {i + 1}: {f:,.4g} Hz")
    if progress:
        progress(1.0, "完了")
    return ModalResult(mesh, freqs, modes, gamma, eff, total_mass, info, prestress=prestress)


# ---------------------------------------------------------------------------- 座屈（幾何剛性）

def stress_stiffness(mesh: Mesh, sig_gp_fn, chunk: int = 20000):
    """幾何剛性（応力剛性）K_σ。sig_gp_fn(a0, b0, q) → (E,6) その積分点の応力。

    K_σ = Σ ∫ G_ab σ dV ⊗ I3,  G_ab = ∂N_a/∂x · σ · ∂N_b/∂x
    """
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    rows, cols, vals = [], [], []
    ii, jj = np.meshgrid(np.arange(nn), np.arange(nn), indexing="ij")
    for a0 in range(0, mesh.n_elems, chunk):
        b0 = min(mesh.n_elems, a0 + chunk)
        el = mesh.elems[a0:b0]
        X = mesh.nodes[el]
        Ke = np.zeros((b0 - a0, nn, nn))
        for q, (dNq, w) in enumerate(zip(dN, wts)):
            dNdx, detJ = _dN_dx(X, dNq)                 # (E, nn, 3), (E,)
            s = sig_gp_fn(a0, b0, q)                    # (E, 6)
            S = np.empty((b0 - a0, 3, 3))
            S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
            S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
            S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
            S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
            Ke += np.einsum("eai,eij,ebj->eab", dNdx, S, dNdx) * (np.abs(detJ) * w)[:, None, None]
        r = el[:, ii.ravel()]
        c = el[:, jj.ravel()]
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(Ke.reshape(b0 - a0, -1).ravel())
    Kn = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(mesh.n_nodes, mesh.n_nodes))
    return sp.kron(Kn, sp.identity(3), format="csr")


def gauss_stress_fn(mesh: Mesh, u: np.ndarray, D, elem_mat, eps_th_m=None):
    """静解析の変位から積分点の応力を返す関数（stress_stiffness 用）"""
    nn = mesh.elems.shape[1]
    pts, _ = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    U = u.reshape(-1, 3)

    def fn(a0, b0, q):
        el = mesh.elems[a0:b0]
        dNdx, _ = _dN_dx(mesh.nodes[el], dN[q])
        eps = np.einsum("eij,ej->ei", _B_matrix(dNdx), U[el].reshape(len(el), -1))
        if eps_th_m is not None:
            eps = eps - eps_th_m[elem_mat[a0:b0]]
        De = _D_of(D, elem_mat, slice(a0, b0))
        return eps @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps)
    return fn


def solve_buckling(mesh: Mesh, material: Material, bcs: list[BC], n_modes: int = 3, elem_materials=None,
                   log=print, progress=None, static: Result | None = None) -> ModalResult:
    """線形座屈（固有値座屈）。静解析（与えた荷重）の応力から K_σ を作り (K + λ K_σ) φ = 0 を解く。
    λ = 座屈荷重係数（与えた荷重の何倍で座屈するか）。Ansys と同じく、正の係数を小さい順に返す"""
    from .solver import material_eps_th, solve
    t0 = time.time()
    mats, elem_mat, D, _ = _materials(mesh, material, elem_materials)
    if static is None:
        log("座屈解析: まず静解析（与えた荷重）を解きます")
        static = solve(mesh, material, bcs, log=log, elem_materials=elem_materials)
    if progress:
        progress(0.4, "幾何剛性を組み立て中…")
    eps_th_m = np.stack([material_eps_th(m) for m in mats])
    if not np.any(eps_th_m):
        eps_th_m = None
    Ks = stress_stiffness(mesh, gauss_stress_fn(mesh, static.u.ravel(), D, elem_mat, eps_th_m))
    K = assemble_stiffness(mesh, D, elem_mat=elem_mat)
    free = _free_dofs(mesh, mats[0], bcs, extra_fixed=None)
    # 強制変位で荷重をかけた場合も、座屈では拘束（0）として扱う
    Kff, Ksff = K[free][:, free], Ks[free][:, free]
    t1 = time.time()
    # K φ = μ (-K_σ) φ  で μ = λ。 -K_σ は不定値なので、K を「質量」にした逆問題 (-K_σ) φ = (1/λ) K φ を解く
    nd = Kff.shape[0]
    solve_K = factorize(Kff.tocsc(), log)
    k = max(1, min(nd - 2, 2 * n_modes + 4))
    OP = spla.LinearOperator((nd, nd), matvec=lambda x: solve_K(-(Ksff @ x)), dtype=float)
    mu, vec = spla.eigs(OP, k=k, which="LR", tol=1e-10, ncv=min(nd - 1, max(2 * k + 1, k + 20)))
    mu, vec = mu.real, vec.real
    pos = mu > 1e-12
    lam = 1.0 / mu[pos]
    vec = vec[:, pos]
    o = np.argsort(lam)[:n_modes]
    lam, vec = lam[o], vec[:, o]
    t_sol = time.time() - t1
    full = np.zeros((3 * mesh.n_nodes, len(lam)))
    full[free] = vec
    # 最大成分 1 に正規化（Ansys の座屈モードと同じ）
    for i in range(len(lam)):
        full[:, i] /= full[:, i][np.argmax(np.abs(full[:, i]))]
    modes = full.T.reshape(len(lam), mesh.n_nodes, 3)
    for i, l in enumerate(lam):
        log(f"  座屈モード {i + 1}: 荷重係数 {l:,.5g}")
    info = {"解析": "固有値座屈", "自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "静解析の時間 [s]": static.info.get("合計時間 [s]"), "求解時間 [s]": round(t_sol, 2),
            "合計時間 [s]": round(time.time() - t0, 2)}
    if progress:
        progress(1.0, "完了")
    return ModalResult(mesh, np.zeros(len(lam)), modes, np.zeros((len(lam), 6)), np.zeros((len(lam), 6)), 0.0,
                       info, kind="buckling", load_factors=lam, prestress=static)

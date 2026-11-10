"""静電場と圧電（Femtet の「静電場」「圧電」、Ansys の Electric / Piezoelectric に相当）。

静電場: ∇·(ε ∇φ) = 0。境界条件 voltage（電圧 V）・ground（0 V）・charge（面の電荷 C、等電位の導体としては扱わない）。
  結果: 電位 [V]、電界 [V/mm]、電束密度 [C/m²]、静電エネルギー → 静電容量 C = 2W / V²（電極が 2 つのとき）。
圧電（応力型）: σ = c ε − eᵀ E、 D = e ε + ε^S E（e: 圧電定数 [C/m²]、ε^S: ひずみ一定の誘電率）。
  静解析（電圧をかけたときの変形 / 力をかけたときの電圧）と、モーダル（電極を短絡 / 開放のときの共振・反共振）。
  分極の向きは材料座標の 3 軸（＝全体の Z。ボディの材料の向きの回転は未対応）。

内部の単位: mm, N, V。ε [F/m] はそのままの数値で V/mm と組んで MPa になる。電荷は ×1e-3 で C。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .solver import (_B_matrix, _dN_dx, _quadrature, assemble_stiffness, build_load_and_constraints, factorize,
                     material_D, node_points, shape_dN_dxi)

EPS0 = 8.8541878128e-12      # F/m
ELEC_KINDS = ("voltage", "ground", "charge")


def _eps_e(mats, elem_mat):
    return np.array([float(getattr(m, "permittivity", 1.0) or 1.0) for m in mats])[elem_mat] * EPS0


def _grad_matrix(mesh, coef_e):
    """∫ ∇Nᵀ coef ∇N dV（スカラー場）"""
    from .thermal import conductivity_matrix
    return conductivity_matrix(mesh, coef_e)


def _voltage_bcs(mesh, bcs):
    fixed, Q = {}, np.zeros(mesh.n_nodes)
    for b in bcs:
        if b.kind in ("voltage", "ground"):
            v = 0.0 if b.kind == "ground" else float(b.values[0])
            for nd in mesh.face_nodes(b.faces):
                fixed[int(nd)] = v
        elif b.kind == "charge":
            from .surface import surface_sets
            sets = surface_sets(mesh, b.faces)
            A = sum(s.area for s in sets)
            for ss in sets:
                idx, w = ss.lumped(float(b.values[0]) * 1e3 / max(A, 1e-300))
                np.add.at(Q, idx, w)
    return fixed, Q


@dataclass
class ElectricResult:
    mesh: object
    phi: np.ndarray
    E: np.ndarray                 # (N,3) V/mm
    D: np.ndarray                 # (N,3) C/m²
    info: dict = field(default_factory=dict)
    time: float = 1.0

    @property
    def u(self):
        return np.zeros((self.mesh.n_nodes, 3))

    def fields(self):
        return {"電位 [V]": self.phi, "電界の大きさ [V/mm]": np.linalg.norm(self.E, axis=1),
                "電界 X [V/mm]": self.E[:, 0], "電界 Y [V/mm]": self.E[:, 1], "電界 Z [V/mm]": self.E[:, 2],
                "電束密度の大きさ [C/m²]": np.linalg.norm(self.D, axis=1)}

    def summary(self):
        return {k: v for k, v in self.info.items() if isinstance(v, float)}


def _nodal_grad(mesh, phi):
    nn = mesh.elems.shape[1]
    Lp = node_points(nn)
    X = mesh.nodes[mesh.elems]
    pe = phi[mesh.elems]
    acc = np.zeros((mesh.n_nodes, 3))
    ge = np.zeros((mesh.n_elems, nn, 3))
    for a in range(nn):
        dNdx, _ = _dN_dx(X, shape_dN_dxi(Lp[a], nn))
        g = np.einsum("eai,ea->ei", dNdx, pe)
        ge[:, a] = g
        for c in range(3):
            np.add.at(acc[:, c], mesh.elems[:, a], g[:, c])
    cnt = np.maximum(np.bincount(mesh.elems.ravel(), minlength=mesh.n_nodes), 1)
    return acc / cnt[:, None], ge


def solve_electrostatic(mesh, material, bcs, elem_materials=None, log=print, contacts=None,
                        contact_tolerance: float = 0.0) -> ElectricResult:
    t0 = time.time()
    if elem_materials is not None:
        mats, em = elem_materials
        em = np.asarray(em, dtype=np.int64)
    else:
        mats, em = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    eps = _eps_e(mats, em)
    K = _grad_matrix(mesh, eps)
    if contacts:
        # 節点を共有していないボディの間は、接触面で電位をつなぐ
        from .contact import scalar_contact_matrix
        e_node = np.zeros(mesh.n_nodes)
        np.maximum.at(e_node, mesh.elems.ravel(), np.repeat(np.asarray(eps, float).reshape(len(em), -1).max(1),
                                                            mesh.elems.shape[1]))
        Kc = scalar_contact_matrix(mesh, contacts, mats, em, e_node, contact_tolerance)
        if Kc is not None:
            K = K + Kc
    fixed, Q = _voltage_bcs(mesh, [b for b in bcs if b.kind in ELEC_KINDS])
    if not fixed:
        raise ValueError("静電場: 電圧（または接地）の境界条件が必要です")
    n = mesh.n_nodes
    cd = np.fromiter(fixed.keys(), dtype=np.int64)
    cv = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(n, bool)
    free[cd] = False
    phi = np.zeros(n)
    phi[cd] = cv
    phi[free] = factorize(K[free][:, free].tocsc(), lambda *a: None)(Q[free] - K[free][:, cd] @ cv)
    g, _ = _nodal_grad(mesh, phi)
    E = -g
    eps_n = np.zeros(n)
    np.maximum.at(eps_n, mesh.elems.ravel(), np.repeat(eps, mesh.elems.shape[1]))
    D = eps_n[:, None] * E * 1e3          # ε[F/m]·E[V/mm]·1000 = C/m²
    W = 0.5 * phi @ (K @ phi) * 1e-3      # N·mm → J
    R = K @ phi - Q
    info = {"解析": "静電場", "節点数": n, "要素数": mesh.n_elems, "静電エネルギー [J]": float(W),
            "合計時間 [s]": round(time.time() - t0, 3)}
    vs = sorted(set(cv.tolist()))
    if len(vs) == 2:
        dV = vs[1] - vs[0]
        C = 2 * W / dV ** 2
        info["静電容量 [F]"] = float(C)
        info["静電容量 [pF]"] = float(C * 1e12)
        hi_nodes = cd[cv == vs[1]]
        info["電極の電荷 [C]"] = float(R[hi_nodes].sum() * 1e-3)
    log(f"静電場: エネルギー {W:.4g} J" + (f"、静電容量 {info['静電容量 [pF]']:.5g} pF" if "静電容量 [pF]" in info else ""))
    return ElectricResult(mesh, phi, E, D, info)


# ---------------------------------------------------------------------------- 圧電

def _piezo_mats(m):
    """圧電定数 e (6×3) と誘電率 (3×3) を全体座標で。材料座標系（ボディの向き）があれば回転する:
    e_g = Tᵀ e_m R、 ε_g = Rᵀ ε_m R（R の行 = 材料軸、T = ひずみの座標変換）"""
    pz = getattr(m, "piezo", None) or {}
    if not pz:
        return None, None
    e = np.asarray(pz["e"], float).reshape(6, 3) * 1e-3          # C/m² → N/(V·mm)
    epsr = np.asarray(pz.get("eps_s", [m.permittivity] * 3), float)
    ep = np.diag(epsr) * EPS0
    R = getattr(m, "R", None)
    if R is not None:
        from .materials import strain_rotation
        T = strain_rotation(R)
        e = T.T @ e @ R
        ep = R.T @ ep @ R
    return e, ep


def piezo_matrices(mesh, mats, elem_mat):
    """Kuu（3N×3N）、Kuφ（3N×N）、Kφφ（N×N）"""
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    X = mesh.nodes[mesh.elems]
    Kuf = np.zeros((mesh.n_elems, 3 * nn, nn))
    Kff = np.zeros((mesh.n_elems, nn, nn))
    eM = np.zeros((len(mats), 6, 3))
    epM = np.zeros((len(mats), 3, 3))
    for i, m in enumerate(mats):
        e, ep = _piezo_mats(m)
        if e is not None:
            eM[i], epM[i] = e, ep
        else:
            epM[i] = np.eye(3) * EPS0 * float(getattr(m, "permittivity", 1.0) or 1.0)
    ee, epe = eM[elem_mat], epM[elem_mat]
    for L, w in zip(pts, wts):
        dNdx, detJ = _dN_dx(X, shape_dN_dxi(L, nn))
        B = _B_matrix(dNdx)                                # (E,6,3nn)
        dv = np.abs(detJ) * w
        # σ = c ε − eᵀ E,  E = −∇φ  →  Kuφ = ∫ Bᵀ e ∇N dV（φ に対する符号は ε·(eᵀ∇φ)）
        Kuf += np.einsum("eki,ekj,eaj->eia", B, ee, dNdx) * dv[:, None, None]
        Kff += np.einsum("eai,eij,ebj->eab", dNdx, epe, dNdx) * dv[:, None, None]
    D = np.stack([material_D(m) for m in mats])
    Kuu = assemble_stiffness(mesh, D, elem_mat=elem_mat)
    dofs = (3 * mesh.elems[:, :, None] + np.arange(3)).reshape(mesh.n_elems, -1)
    r = np.repeat(dofs, nn, axis=1).ravel()
    c = np.tile(mesh.elems, (1, 3 * nn)).ravel()
    Kuf_g = sp.csr_matrix((Kuf.ravel(), (r, c)), shape=(3 * mesh.n_nodes, mesh.n_nodes))
    r2 = np.repeat(mesh.elems, nn, axis=1).ravel()
    c2 = np.tile(mesh.elems, (1, nn)).ravel()
    Kff_g = sp.csr_matrix((Kff.ravel(), (r2, c2)), shape=(mesh.n_nodes, mesh.n_nodes))
    return Kuu, Kuf_g, Kff_g


def solve_piezo_static(mesh, material, bcs, elem_materials=None, log=print):
    """圧電の静解析。構造の支持・荷重と、電極（電圧・接地・電荷）を同時に与える。戻り値: (構造 Result, ElectricResult)"""
    from .solver import Result, nodal_stress
    t0 = time.time()
    if elem_materials is not None:
        mats, em = elem_materials
        em = np.asarray(em, dtype=np.int64)
    else:
        mats, em = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    Kuu, Kuf, Kff = piezo_matrices(mesh, mats, em)
    nb, n = 3 * mesh.n_nodes, mesh.n_nodes
    F, fixed_u = build_load_and_constraints(mesh, mats[0], [b for b in bcs if b.kind not in ELEC_KINDS])
    fixed_p, Q = _voltage_bcs(mesh, [b for b in bcs if b.kind in ELEC_KINDS])
    # [Kuu  Kuφ ] [u]   [F]
    # [Kφu -Kφφ ] [φ] = [-Q]       （対称・不定値）
    A = sp.bmat([[Kuu, Kuf], [Kuf.T, -Kff]], format="csr")
    rhs = np.r_[F, -Q]
    fixed = dict(fixed_u)
    fixed.update({nb + k: v for k, v in fixed_p.items()})
    cd = np.fromiter(fixed.keys(), dtype=np.int64)
    cv = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(nb + n, bool)
    free[cd] = False
    x = np.zeros(nb + n)
    x[cd] = cv
    import scipy.sparse.linalg as spla
    Aff = A[free][:, free].tocsc()
    x[free] = spla.spsolve(Aff, rhs[free] - A[free][:, cd] @ cv)
    u, phi = x[:nb], x[nb:]
    D = np.stack([material_D(m) for m in mats])
    st, sn = nodal_stress(mesh, u, D, elem_mat=em)
    # 圧電の応力の補正 σ = c ε − eᵀ E（nodal_stress は c ε だけ）
    g, ge = _nodal_grad(mesh, phi)
    eM = np.zeros((len(mats), 6, 3))
    for i, m in enumerate(mats):
        e, _ = _piezo_mats(m)
        if e is not None:
            eM[i] = e
    corr = np.zeros((n, 6))
    cnt = np.zeros(n)
    for a in range(mesh.elems.shape[1]):
        s = np.einsum("ekj,ej->ek", eM[em], ge[:, a])        # eᵀ ∇φ → σ の補正（E = −∇φ なので −eᵀE = +e∇φ）
        np.add.at(corr, mesh.elems[:, a], s)
        np.add.at(cnt, mesh.elems[:, a], 1)
    st = st + corr / np.maximum(cnt, 1)[:, None]
    r = Result(mesh, mats[0], u.reshape(-1, 3), st, sn, {}, {"解析": "圧電（静解析）", "合計時間 [s]": round(time.time() - t0, 3)})
    er = ElectricResult(mesh, phi, -g, np.zeros((n, 3)), {"解析": "圧電（電位）"})
    r.extra = {"電位 [V]": phi, "電界の大きさ [V/mm]": np.linalg.norm(g, axis=1)}
    log(f"圧電: 最大変位 {np.linalg.norm(r.u, axis=1).max():.4g} mm、電位 {phi.min():.4g} – {phi.max():.4g} V")
    return r, er


def solve_piezo_modal(mesh, material, bcs, n_modes=6, elem_materials=None, open_circuit=False, log=print):
    """圧電のモーダル。電極（電圧・接地の面）を短絡（すべて 0 V）または開放（電極は等電位で電荷 0）"""
    from .dynamics import consistent_mass
    from .modal import _eig, _supports_only
    if elem_materials is not None:
        mats, em = elem_materials
        em = np.asarray(em, dtype=np.int64)
    else:
        mats, em = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    Kuu, Kuf, Kff = piezo_matrices(mesh, mats, em)
    rho_e = np.array([m.density_t_mm3 for m in mats])[em]
    M = consistent_mass(mesh, rho_e)
    _, fixed_u = build_load_and_constraints(mesh, mats[0], _supports_only(bcs))
    elec = [b for b in bcs if b.kind in ("voltage", "ground")]
    n = mesh.n_nodes
    ground = set()
    hot = []
    for b in elec:
        nodes = mesh.face_nodes(b.faces)
        if b.kind == "ground" or not open_circuit:
            ground |= set(int(x) for x in nodes)
        else:
            hot.append(np.asarray(nodes, int))
    # 電極の等電位: 開放の電極の節点は 1 つの自由度にまとめる（φ = T φ̃）
    keep = np.array(sorted(set(range(n)) - ground - set(int(x) for h in hot for x in h)), dtype=np.int64)
    cols = len(keep) + len(hot)
    rows, cc, vals = list(keep), list(range(len(keep))), [1.0] * len(keep)
    for j, h in enumerate(hot):
        rows += h.tolist()
        cc += [len(keep) + j] * len(h)
        vals += [1.0] * len(h)
    T = sp.csr_matrix((vals, (rows, cc)), shape=(n, cols))
    Kf = (T.T @ Kff @ T).tocsc()
    Kuf_r = (Kuf @ T).tocsc()
    # 静的縮約: K* = Kuu + Kuφ Kφφ⁻¹ Kφu
    fac = factorize(Kf, lambda *a: None)
    Z = fac(Kuf_r.T.toarray())                          # (cols, 3N) 密（小さなモデル向け）
    Kstar = Kuu + sp.csr_matrix(Kuf_r @ Z)
    free = np.ones(3 * n, bool)
    if fixed_u:
        free[np.fromiter(fixed_u.keys(), dtype=np.int64)] = False
    lam, vec = _eig(Kstar[free][:, free], M[free][:, free], n_modes + 2, 0.0, lambda *a: None)
    f = np.sqrt(np.maximum(lam[:n_modes], 0)) / (2 * np.pi)
    log(f"圧電モーダル（{'開放' if open_circuit else '短絡'}）: " + ", ".join(f"{x:.6g}" for x in f) + " Hz")
    return f

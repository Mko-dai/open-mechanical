"""ソリッドとシェルを 1 つの解析で解く（Ansys でソリッドボディとサーフェスボディを結合してつなぐのと同じ）

ソリッドのメッシュ（節点あたり 3 自由度）とシェルのメッシュ（6 自由度）を別々に作り、ソリッドの表面に
乗っているシェルの節点を「結合」する。 結合はシェルの節点を従属にした多点拘束で、消去法で厳密に扱う:

    並進  u_shell = ソリッドの面の上で内挿した変位
    回転  θ_shell = その点のまわりのソリッド表面の回転（面内の 2 方向に ±s ずらした点の変位の差から）

回転も結ぶので、シェルの曲げモーメントがソリッドに伝わる（突き合わせ・T 字・重ね、どの向きでもよい）。
線形の静解析。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .solver import (BC, Result, _linear_solve, _rel_res, assemble_stiffness, build_load_and_constraints,
                     material_D, nodal_stress)

SOLID_BC = ("fixed", "displacement", "symmetry", "frictionless", "force", "pressure", "moment", "gravity")
SHELL_BC = ("fixed", "displacement", "symmetry", "frictionless", "force", "pressure", "gravity")


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class _Surface:
    """ソリッドの表面への射影（最近点の節点と内挿の重み）"""

    def __init__(self, solid):
        from scipy.spatial import cKDTree
        self.X = solid.nodes
        self.tris = solid.boundary_tris_full
        self.tp = self.X[self.tris[:, :3]]
        n = np.cross(self.tp[:, 1] - self.tp[:, 0], self.tp[:, 2] - self.tp[:, 0])
        self.area2 = np.linalg.norm(n, axis=1)
        self.n = n / np.maximum(self.area2[:, None], 1e-300)
        self.tree = cKDTree(self.tp.mean(axis=1))

    def project(self, pts):
        """戻り値: (節点 (M,k), 重み (M,k), 射影点 (M,3), 距離 (M,), 面の法線 (M,3))"""
        from .contact import _project, _tri6
        pts = np.atleast_2d(pts)
        bt, bc, _d = _project(pts, self.tp, self.tree, min(12, len(self.tris)))
        if self.tris.shape[1] == 6:
            nodes, w = self.tris[bt], _tri6(bc[:, 1], bc[:, 2])[0]
        else:
            nodes, w = self.tris[bt][:, :3], bc
        q = np.einsum("mk,mkj->mj", w, self.X[nodes])
        return nodes, w, q, np.linalg.norm(q - pts, axis=1), self.n[bt]


def tie_shell_to_solid(solid, shell, tol=None, log=print):
    """ソリッドの表面に乗っているシェルの節点の多点拘束。
    戻り値: {シェルの自由度（6×節点+成分）: (ソリッドの自由度の配列, 係数の配列)}, 結合した節点の配列"""
    surf = _Surface(solid)
    diag = float(np.linalg.norm(solid.nodes.max(0) - solid.nodes.min(0)))
    tol = float(tol) if tol else 1e-4 * diag + 1e-9
    used = np.unique(shell.elems)
    nodes, w, q, d, nf = surf.project(shell.nodes[used])
    hit = d <= tol
    tied = used[hit]
    if not len(tied):
        return {}, tied
    # シェルの節点の法線・厚さ（まわりの要素の平均）
    en = np.cross(shell.nodes[shell.elems[:, 1]] - shell.nodes[shell.elems[:, 0]],
                  shell.nodes[shell.elems[:, 2]] - shell.nodes[shell.elems[:, 0]])
    en /= np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-300)
    nn = np.zeros((shell.n_nodes, 3))
    th = np.zeros(shell.n_nodes)
    cnt = np.zeros(shell.n_nodes)
    for a in range(3):
        np.add.at(nn, shell.elems[:, a], en)
        np.add.at(th, shell.elems[:, a], shell.thickness)
        np.add.at(cnt, shell.elems[:, a], 1.0)
    th /= np.maximum(cnt, 1)
    mpc = {}
    weak = 0
    for k, nd in enumerate(tied):
        j = np.nonzero(hit)[0][k]
        x, nface = q[j], nf[j]
        ns = nn[nd] / max(np.linalg.norm(nn[nd]), 1e-300)
        s = 0.5 * th[nd]
        # ソリッドの面内の 2 方向: d1 = シェルの法線を面に落とした向き（シェルが面に立っているとき = 厚さ方向）
        d1 = ns - (ns @ nface) * nface
        if np.linalg.norm(d1) < 0.2:                 # シェルが面に寝ている（重ね）: 面内の適当な向き
            d1 = np.cross(nface, [1.0, 0, 0]) if abs(nface[0]) < 0.9 else np.cross(nface, [0, 1.0, 0])
        d1 /= np.linalg.norm(d1)
        d2 = np.cross(nface, d1)
        m = np.cross(d1, d2)
        # 並進
        for c in range(3):
            mpc[6 * int(nd) + c] = (3 * nodes[j] + c, w[j].copy())
        # 回転: θ = (d2 d2ᵀ + ½ m mᵀ) [d1]× Δu1 / |Δx1|  +  (d1 d1ᵀ + ½ m mᵀ) [d2]× Δu2 / |Δx2|
        rows = {}
        ok = [False, False]
        for kk, (da, db) in enumerate(((d1, d2), (d2, d1))):
            np_, wp, qp, _dp, _ = surf.project(np.array([x + s * da, x - s * da]))
            dx = qp[0] - qp[1]
            L = np.linalg.norm(dx)
            if L < 0.4 * s:
                continue
            ok[kk] = True
            A = (np.outer(db, db) + 0.5 * np.outer(m, m)) @ _skew(dx / L) / L
            for sign, pn, pw in ((1.0, np_[0], wp[0]), (-1.0, np_[1], wp[1])):
                for nj, wj in zip(pn, pw):
                    for cc in range(3):
                        for dd in range(3):
                            v = sign * A[cc, dd] * wj
                            if v != 0.0:
                                rows.setdefault(cc, {}).setdefault(3 * int(nj) + dd, 0.0)
                                rows[cc][3 * int(nj) + dd] += v
        if not all(ok):
            weak += 1          # 面の縁などで 2 方向が取れない: この節点の回転は結ばない（並進だけ）
            continue
        for cc in range(3):
            r = rows.get(cc, {})
            if r:
                mpc[6 * int(nd) + 3 + cc] = (np.fromiter(r.keys(), np.int64), np.fromiter(r.values(), float))
    log(f"ソリッド－シェルの結合: シェルの節点 {len(tied):,} をソリッドの面に結合"
        + (f"（うち {weak} 点は面の縁で、回転を結べず並進だけ）" if weak else ""))
    return mpc, tied


def _shell_post(shell, u6, mats, elem_mat, reactions, info):
    """シェルの変位（6 自由度）→ Result（上面・下面の応力。 shell.solve_shell の線形の後処理と同じ）"""
    from .shell import (_BEN, _MEMR, _allman_B, _dkt_B, _frames, _layer_to_global, _mises, _nodal_avg, _shell_dofs,
                        _transform, plane_stress_D)
    Dp = np.stack([plane_stress_D(material_D(m)) for m in mats])[elem_mat]
    X = shell.nodes[shell.elems]
    R, xy, A = _frames(X)
    t = shell.thickness
    dofs = _shell_dofs(shell.elems)
    ul = np.einsum("eij,ej->ei", _transform(R), u6[dofs])
    em = np.einsum("eij,ej->ei", _allman_B(xy, A, (1 / 3, 1 / 3, 1 / 3))[0], ul[:, _MEMR])
    kap = np.einsum("eij,ej->ei", _dkt_B(xy, 1 / 3, 1 / 3), ul[:, _BEN])
    out = {}
    for name, z in (("上面", 0.5), ("下面", -0.5)):
        s = np.einsum("eij,ej->ei", Dp, em + (z * t)[:, None] * kap)
        out[name] = _nodal_avg(shell, _layer_to_global(R, s), A)
    vm = {k: _mises(v) for k, v in out.items()}
    stress = np.where((vm["上面"] >= vm["下面"])[:, None], out["上面"], out["下面"])
    U = u6.reshape(-1, 6)
    r = Result(shell, mats[0], U[:, :3].copy(), stress, np.zeros_like(stress), reactions, dict(info), mats, elem_mat)
    r.extra = {"相当応力（上面） [MPa]": vm["上面"], "相当応力（下面） [MPa]": vm["下面"],
               "回転 θx [rad]": U[:, 3], "回転 θy [rad]": U[:, 4], "回転 θz [rad]": U[:, 5]}
    r.rotations = U[:, 3:].copy()
    return r


def solve_mixed(solid, shell, material, bcs_solid: list[BC], bcs_shell: list[BC], solid_materials=None,
                shell_materials=None, tie_tol=None, log=print, method="auto"):
    """ソリッド＋シェルの線形静解析。 戻り値: (ソリッドの Result, シェルの Result)"""
    from .shell import _frames, _shell_dofs, plane_stress_D, shell_constraints, shell_element_K
    t0 = time.time()
    for b in bcs_solid:
        if b.kind not in SOLID_BC:
            raise ValueError(f"ソリッド＋シェルの解析では「{b.label()}」はまだ使えません")
    for b in bcs_shell:
        if b.kind not in SHELL_BC:
            raise ValueError(f"ソリッド＋シェルの解析では、シェル側の「{b.label()}」はまだ使えません")
    mats_s, em_s = solid_materials if solid_materials is not None else ([material], np.zeros(solid.n_elems, np.int64))
    mats_h, em_h = shell_materials if shell_materials is not None else ([material], np.zeros(shell.n_elems, np.int64))
    em_s, em_h = np.asarray(em_s, np.int64), np.asarray(em_h, np.int64)
    # ソリッド
    D = np.stack([material_D(m) for m in mats_s])
    Ks = assemble_stiffness(solid, D, elem_mat=em_s).tocsr()
    rho_e = np.array([m.density_t_mm3 for m in mats_s])[em_s]
    Fs, fx_s = build_load_and_constraints(solid, mats_s[0], bcs_solid, rho_e=rho_e)
    ns = Ks.shape[0]
    # シェル
    Dp = np.stack([plane_stress_D(material_D(m)) for m in mats_h])[em_h]
    Xh = shell.nodes[shell.elems]
    Ke, _R, _xy, Ah = shell_element_K(Xh, shell.thickness, Dp)
    dofs = _shell_dofs(shell.elems)
    nh = 6 * shell.n_nodes
    Kh = sp.coo_matrix((Ke.ravel(), (np.repeat(dofs, 18, axis=1).ravel(), np.tile(dofs, (1, 18)).ravel())),
                       shape=(nh, nh)).tocsr()
    Fh, fx_h = shell_constraints(shell, bcs_shell, mats_h, em_h, Ah, _frames(Xh)[0][:, 2])
    used = np.zeros(nh, bool)
    used[dofs.ravel()] = True
    # 結合（シェルの自由度を従属に）
    mpc, tied = tie_shell_to_solid(solid, shell, tie_tol, log)
    if not len(tied):
        raise ValueError("シェルの節点がソリッドの表面に乗っていません（シェルの辺・面をソリッドの面に接するように置いてください）")
    n = ns + nh
    K = sp.bmat([[Ks, None], [None, Kh]], format="csr")
    F = np.r_[Fs, Fh]
    is_slave = np.zeros(n, bool)
    for d in mpc:
        is_slave[ns + d] = True
    keep = np.nonzero(~is_slave)[0]
    newidx = -np.ones(n, np.int64)
    newidx[keep] = np.arange(len(keep))
    rows, cols, vals = [keep], [np.arange(len(keep))], [np.ones(len(keep))]
    for d, (md, cf) in mpc.items():
        rows.append(np.full(len(md), ns + d))
        cols.append(newidx[md])
        vals.append(cf)
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(keep)))
    Kr = (T.T @ K @ T).tocsr()
    Fr = T.T @ F
    fixed = {int(newidx[d]): v for d, v in fx_s.items()}
    for d, v in fx_h.items():
        if not is_slave[ns + d]:
            fixed[int(newidx[ns + d])] = v
    for d in np.nonzero(~used)[0]:
        if not is_slave[ns + d]:
            fixed[int(newidx[ns + d])] = 0.0
    if not fixed:
        raise ValueError("拘束（固定支持 または 変位）がありません")
    cd = np.array(sorted(fixed), np.int64)
    cv = np.array([fixed[int(d)] for d in cd])
    free = np.ones(len(keep), bool)
    free[cd] = False
    ur = np.zeros(len(keep))
    ur[cd] = cv
    Kff = Kr[free][:, free].tocsc()
    rhs = Fr[free] - Kr[free][:, cd] @ cv
    log(f"ソリッド＋シェル: ソリッド {solid.n_nodes:,} 節点・シェル {shell.n_nodes:,} 節点 / 自由度 {int(free.sum()):,}")
    uf = _linear_solve(Kff, rhs, lambda *a: None, method="direct" if method == "auto" else method)
    if not np.all(np.isfinite(uf)) or _rel_res(Kff, uf, rhs) > 1e-4:
        raise RuntimeError("解が求まりませんでした。拘束が不足していないか確認してください。")
    ur[free] = uf
    u = T @ ur
    Rall = K @ u - F
    us, uh = u[:ns], u[ns:]
    stress, strain = nodal_stress(solid, us, D, elem_mat=em_s)
    info = {"解析": "静的構造（ソリッド＋シェル）", "自由度数": int(free.sum()), "結合したシェルの節点": int(len(tied)),
            "合計時間 [s]": round(time.time() - t0, 2)}
    reac_s = {}
    for i, bc in enumerate(bcs_solid):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nd = solid.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
            reac_s[i] = Rall[:ns].reshape(-1, 3)[nd].sum(0)
    rs = Result(solid, mats_s[0], us.reshape(-1, 3).copy(), stress, strain, reac_s,
                dict(info, 節点数=solid.n_nodes, 要素数=solid.n_elems), mats_s, em_s)
    reac_h = {}
    for i, bc in enumerate(bcs_shell):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nd = shell.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
            reac_h[i] = Rall[ns:].reshape(-1, 6)[nd, :3].sum(0)
    rh = _shell_post(shell, uh, mats_h, em_h, reac_h, dict(info, 節点数=shell.n_nodes, 要素数=shell.n_elems))
    rs.tied_nodes = rh.tied_nodes = tied
    return rs, rh

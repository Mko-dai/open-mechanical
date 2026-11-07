"""2次元解析（Ansys の 2D 解析: 平面応力 / 平面ひずみ / 軸対称、PLANE183 相当の 6 節点三角形）

形状は XY 平面のサーフェス。軸対称は Y 軸が対称軸、X = 半径方向（Ansys と同じ）で、X ≥ 0 に置く。
境界条件は辺（エッジ）に与える。軸対称の力・反力は全周（360°）の値（Ansys と同じ）。
塑性材料（ミーゼス）にも対応（平面応力は σz = 0 を反復で満たすリターンマッピング）。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .shell import ShellMesh, _mises, _nodal_avg
from .solver import BC, Result, _linear_solve, _rel_res, material_D, material_eps_th

MODES = {"plane_stress": "平面応力", "plane_strain": "平面ひずみ", "axisymmetric": "軸対称"}

# 6 節点三角形: 頂点 0,1,2、中間 3:(0,1) 4:(1,2) 5:(2,0)
_GP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])   # (ξ, η)、重み 1/6
_GPW = np.full(3, 1 / 6)
_GP6 = np.array([[0.445948490915965, 0.445948490915965], [0.108103018168070, 0.445948490915965],
                 [0.445948490915965, 0.108103018168070], [0.091576213509771, 0.091576213509771],
                 [0.816847572980459, 0.091576213509771], [0.091576213509771, 0.816847572980459]])
_GP6W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2


def _N(xi, eta):
    L = 1 - xi - eta
    return np.array([L * (2 * L - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * L * xi, 4 * xi * eta, 4 * eta * L])


def _dN(xi, eta):
    L = 1 - xi - eta
    dxi = np.array([-(4 * L - 1), 4 * xi - 1, 0.0, 4 * (L - xi), 4 * eta, -4 * eta])
    deta = np.array([-(4 * L - 1), 0.0, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (L - eta)])
    return np.stack([dxi, deta], 1)    # (6,2)


class Mesh2D(ShellMesh):
    """2次元メッシュ（6 節点三角形）。辺の番号は面の番号の続き（シェルと同じ扱い）"""

    def __init__(self, nodes, tri6, surf_ids, edges3, thickness, elem_body=None, n_surf=None, mode="plane_stress"):
        e6 = np.asarray(tri6, np.int64)
        segs = [np.vstack([e[:, [0, 2]], e[:, [2, 1]]]) for e in edges3]   # 描画・選択用の線分
        super().__init__(nodes, e6, thickness, surf_ids, segs, elem_body=elem_body, n_surf=n_surf)
        self._btri = e6[:, :3]
        self._btri_full = e6
        self.edges3 = [np.asarray(e, np.int64) for e in edges3]
        self.is_shell = False
        self.is_2d = True
        self.mode = mode

    @property
    def order(self):
        return 2

    def face_tris(self, face_id, full=True):
        m = self.face_ids == int(face_id)
        return (self.elems if full else self.elems[:, :3])[m]

    def face_nodes(self, face_id):
        out = []
        for f in np.atleast_1d(face_id):
            f = int(f)
            out.append(self.edges3[f - self.n_surf].ravel() if f >= self.n_surf else self.elems[self.face_ids == f].ravel())
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def element_volumes(self):
        p = self.nodes[self.elems[:, :3]]
        a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                         - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        if self.mode == "axisymmetric":
            return a * 2 * np.pi * np.abs(p[:, :, 0].mean(1))
        return a * self.thickness


_NODE_XI = np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5]], float)


def _extrapolate_avg(mesh, sig_gp, gp):
    """sig_gp: (積分点, E, k) → 節点値 (N, k)"""
    P = np.column_stack([np.ones(len(gp)), gp])
    Q = np.column_stack([np.ones(6), _NODE_XI])
    Ex = Q @ np.linalg.pinv(P)                          # (6, 積分点)
    ne = np.einsum("ag,gek->eak", Ex, sig_gp)           # (E,6,k)
    A = np.abs(mesh.element_volumes())
    idx = mesh.elems.ravel()
    wt = np.repeat(A, 6)
    den = np.maximum(np.bincount(idx, wt, mesh.n_nodes), 1e-300)
    return np.column_stack([np.bincount(idx, ne[:, :, j].ravel() * wt, mesh.n_nodes)
                            for j in range(ne.shape[2])]) / den[:, None]


def _D4(m, mode):
    """[εx, εy, εz(θ), γxy] に対する剛性 (4,4)"""
    D = material_D(m)
    idx = [0, 1, 2, 3]
    D4 = D[np.ix_(idx, idx)]
    if mode == "plane_stress":
        # σz = 0 で縮約（εz は従属）→ 3×3 を 4×4 に埋め込む（εz 行列はゼロ）
        ip = [0, 1, 3]
        Dp = D[np.ix_(ip, ip)] - np.outer(D[ip, 2], D[2, ip]) / D[2, 2]
        out = np.zeros((4, 4))
        out[np.ix_([0, 1, 3], [0, 1, 3])] = Dp
        return out
    return D4


class Plane:
    """2次元要素の剛性・内力（線形・弾塑性）"""

    def __init__(self, mesh: Mesh2D, mats, elem_mat, mode, temps=None):
        self.mesh, self.mats, self.elem_mat, self.mode = mesh, mats, np.asarray(elem_mat), mode
        X = mesh.nodes[mesh.elems][:, :, :2]
        gp, gw = (_GP6, _GP6W) if mode == "axisymmetric" else (_GP, _GPW)
        self.B, self.w = [], []
        E = mesh.n_elems
        for (xi, eta), wg in zip(gp, gw):
            dN = _dN(xi, eta)
            J = np.einsum("eak,al->ekl", X, dN)            # (E,2,2) ∂x/∂ξ
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            if np.any(det <= 0):
                det = np.abs(det)
            Ji = np.linalg.inv(J)
            dNx = np.einsum("al,elk->eak", dN, Ji)          # (E,6,2)
            B = np.zeros((E, 4, 12))
            B[:, 0, 0::2] = dNx[..., 0]
            B[:, 1, 1::2] = dNx[..., 1]
            B[:, 3, 0::2] = dNx[..., 1]
            B[:, 3, 1::2] = dNx[..., 0]
            wt = wg * det
            if mode == "axisymmetric":
                N = _N(xi, eta)
                r = X[..., 0] @ N
                r = np.maximum(r, 1e-12)
                B[:, 2, 0::2] = N[None, :] / r[:, None]
                wt = wt * 2 * np.pi * r
            else:
                wt = wt * mesh.thickness if mode == "plane_stress" else wt * mesh.thickness
            self.B.append(B)
            self.w.append(wt)
        self.D = np.stack([_D4(m, mode) for m in mats])
        self.eth = np.stack([material_eps_th(m)[[0, 1, 2, 3]] for m in mats])
        if mode == "plane_stress":
            self.eth[:, 2] = 0.0
        from .plasticity import PlasticModel
        self.models = {i: PlasticModel(m) for i, m in enumerate(mats) if getattr(m, "plastic", False)}
        ng = len(self.w)
        self.ep = np.zeros((ng, E, 6))
        self.al = np.zeros((ng, E, 6))
        self.pb = np.zeros((ng, E))
        self.ez = np.zeros((ng, E))
        self.dofs = (2 * mesh.elems[:, :, None] + np.arange(2)).reshape(E, 12)
        self.rows = np.repeat(self.dofs, 12, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 12)).ravel()
        self.trial = None

    def internal(self, u, th_scale=1.0):
        from .newton import plane_stress_return
        from .plasticity import W_EPS, W_SIG
        E = self.mesh.n_elems
        ue = u[self.dofs]
        fe = np.zeros((E, 12))
        Ke = np.zeros((E, 12, 12))
        st = [self.ep.copy(), self.al.copy(), self.pb.copy(), self.ez.copy()]
        sig_all = np.zeros((len(self.w), E, 4))
        eth = self.eth[self.elem_mat] * th_scale
        for g, (B, w) in enumerate(zip(self.B, self.w)):
            eps = np.einsum("eij,ej->ei", B, ue) - eth
            C = self.D[self.elem_mat].copy()
            sig = np.einsum("eij,ej->ei", C, eps)
            for mi, model in self.models.items():
                sel = np.nonzero(self.elem_mat == mi)[0]
                if len(sel) == 0:
                    continue
                if self.mode == "plane_stress":
                    s3, C3, e1, a1, p1, z1 = plane_stress_return(model, eps[sel][:, [0, 1, 3]], self.ep[g, sel],
                                                                 self.al[g, sel], self.pb[g, sel], self.ez[g, sel])
                    s4 = np.zeros((len(sel), 4))
                    s4[:, [0, 1, 3]] = s3
                    C4 = np.zeros((len(sel), 4, 4))
                    C4[:, np.ix_([0, 1, 3], [0, 1, 3])[0], np.ix_([0, 1, 3], [0, 1, 3])[1]] = C3
                    st[3][g, sel] = z1
                else:
                    ev = np.zeros((len(sel), 6))
                    ev[:, :4] = eps[sel]
                    sM, e1, a1, p1, CM = model.update(ev * W_EPS, self.ep[g, sel], self.al[g, sel], self.pb[g, sel])
                    sv = sM / W_SIG
                    Cv = CM / W_SIG[None, :, None] * W_EPS[None, None, :]
                    s4 = sv[:, :4]
                    C4 = Cv[:, :4, :4]
                sig[sel], C[sel] = s4, C4
                st[0][g, sel], st[1][g, sel], st[2][g, sel] = e1, a1, p1
            sig_all[g] = sig
            fe += np.einsum("e,eki,ek->ei", w, B, sig)
            Ke += np.einsum("e,eki,ekl,elj->eij", w, B, C, B)
        self.trial = st + [sig_all]
        nd = 2 * self.mesh.n_nodes
        f = np.zeros(nd)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(nd, nd)).tocsr()
        return f, K

    def commit(self):
        self.ep, self.al, self.pb, self.ez = [x.copy() for x in self.trial[:4]]

    def thermal_force(self):
        """熱ひずみの等価節点力（線形用）"""
        E = self.mesh.n_elems
        eth = self.eth[self.elem_mat]
        if not np.any(eth):
            return None
        fe = np.zeros((E, 12))
        for B, w in zip(self.B, self.w):
            fe += np.einsum("e,eki,ek->ei", w, B, np.einsum("eij,ej->ei", self.D[self.elem_mat], eth))
        f = np.zeros(2 * self.mesh.n_nodes)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        return f


def _edge_axis(mesh, fid):
    seg = mesh.edges3[fid - mesh.n_surf]
    d = (mesh.nodes[seg[:, 1]] - mesh.nodes[seg[:, 0]]).sum(0)[:2]
    d /= max(np.linalg.norm(d), 1e-300)
    n = np.array([d[1], -d[0]])
    if np.max(np.abs(n)) < 0.999:
        raise ValueError("2次元の対称領域・摩擦なし支持は、X 軸または Y 軸に平行な辺に設定してください")
    return int(np.argmax(np.abs(n)))


def _edge_weights(mesh, fid, mode):
    """辺の Line3 に一様な表面力を配る重み（軸対称は 2πr を含む）と外向き法線"""
    seg = mesh.edges3[fid - mesh.n_surf]
    P = mesh.nodes[seg][:, :, :2]                 # (k,3,2): 端, 端, 中
    gs = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
    gw = np.array([5, 8, 5]) / 9.0
    nodes, w = [], []
    for s, ws in zip(gs, gw):
        N = np.array([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s * s])
        dN = np.array([s - 0.5, s + 0.5, -2 * s])
        x = np.einsum("a,kad->kd", N, P)
        t = np.einsum("a,kad->kd", dN, P)
        jac = np.linalg.norm(t, axis=1)
        fac = ws * jac * (2 * np.pi * np.abs(x[:, 0]) if mode == "axisymmetric" else 1.0)
        nodes.append(seg)
        w.append(fac[:, None] * N[None, :])
    n = np.concatenate(nodes).ravel()
    ww = np.concatenate(w).ravel()
    agg = np.bincount(n, ww, mesh.n_nodes)
    idx = np.nonzero(agg)[0]
    return idx, agg[idx]


def solve_2d(mesh: Mesh2D, material, bcs: list[BC], mode="plane_stress", elem_materials=None, log=print,
             settings=None, progress=None, n_cores=None, **_) -> Result:
    from .parallel import set_threads
    if n_cores is not None:
        set_threads(n_cores)
    t0 = time.time()
    if not any(b.kind in ("fixed", "displacement", "symmetry", "frictionless") for b in bcs):
        raise ValueError("拘束（固定支持・変位・対称領域）がありません。辺に設定してください。")
    if elem_materials is not None:
        mats, elem_mat = elem_materials
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, np.int64)
    elem_mat = np.asarray(elem_mat, np.int64)
    if mode == "axisymmetric" and mesh.nodes[:, 0].min() < -1e-9:
        raise ValueError("軸対称では、形状を X ≥ 0（Y 軸が対称軸）に置いてください")
    pl = Plane(mesh, mats, elem_mat, mode)
    nd = 2 * mesh.n_nodes
    F = np.zeros(nd)
    fixed = {}
    for bc in bcs:
        nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
        if bc.kind == "fixed":
            for c in range(2):
                for d in 2 * nodes + c:
                    fixed[int(d)] = 0.0
        elif bc.kind == "displacement":
            for c in range(2):
                if bc.values[c] is not None:
                    for d in 2 * nodes + c:
                        fixed[int(d)] = float(bc.values[c])
        elif bc.kind in ("symmetry", "frictionless"):
            for f in bc.faces:
                if not mesh.is_edge(f):
                    raise ValueError("2次元の対称領域は辺に設定してください")
                ax = _edge_axis(mesh, int(f))
                for d in 2 * mesh.face_nodes([f]) + ax:
                    fixed[int(d)] = 0.0
        elif bc.kind == "force":
            f = np.asarray(bc.values, float)[:2]
            ids, ws = [], []
            for fid in bc.faces:
                i, w = _edge_weights(mesh, int(fid), mode)   # 合計が指定の力になるよう配分（軸対称は全周の合計）
                ids.append(i)
                ws.append(w)
            i = np.concatenate(ids)
            w = np.concatenate(ws)
            for c in range(2):
                np.add.at(F, 2 * i + c, w / w.sum() * f[c])
        elif bc.kind == "pressure":
            p = float(bc.values[0])
            for fid in bc.faces:
                if not mesh.is_edge(fid):
                    continue
                seg = mesh.edges3[int(fid) - mesh.n_surf]
                # 外向き法線（辺に接する要素の重心から外へ）
                cen = mesh.nodes[mesh.elems[np.isin(mesh.elems, seg).any(1)][:, :3]].mean(axis=(0, 1))[:2]
                i, w = _edge_weights(mesh, int(fid), mode)
                a, b = mesh.nodes[seg[:, 0], :2], mesh.nodes[seg[:, 1], :2]
                t = b - a
                n = np.stack([t[:, 1], -t[:, 0]], 1)
                n /= np.linalg.norm(n, axis=1)[:, None]
                mid = 0.5 * (a + b)
                flip = np.einsum("ij,ij->i", n, mid - cen) < 0
                n[flip] *= -1
                # 節点ごとの法線（その節点を含む線分の平均）
                nn_ = np.zeros((mesh.n_nodes, 2))
                for k in range(3):
                    np.add.at(nn_, seg[:, k], n)
                nl = np.linalg.norm(nn_[i], axis=1)
                nv = nn_[i] / np.maximum(nl, 1e-300)[:, None]
                for c in range(2):
                    np.add.at(F, 2 * i + c, -p * w * nv[:, c])
        elif bc.kind == "gravity":
            acc = np.asarray(bc.values, float)[:2]
            rho = np.array([m.density_t_mm3 for m in mats])[elem_mat]
            fe = np.zeros((mesh.n_elems, 12))
            gp, gw = (_GP6, _GP6W) if mode == "axisymmetric" else (_GP, _GPW)
            for (xi, eta), wv in zip(gp, pl.w):
                N = _N(xi, eta)
                for c in range(2):
                    fe[:, c::2] += (rho * wv)[:, None] * N[None, :] * acc[c]
            np.add.at(F, pl.dofs.ravel(), fe.ravel())
        elif bc.kind == "thermal":
            pass
        elif bc.kind == "moment":
            raise ValueError("2次元ではモーメントに未対応です")
    plastic = bool(pl.models)
    used = np.zeros(nd, bool)
    used[pl.dofs.ravel()] = True
    log(f"2次元（{MODES[mode]}）: 節点 {mesh.n_nodes:,}、要素 {mesh.n_elems:,}" + ("（弾塑性）" if plastic else ""))
    if plastic:
        from .newton import solve_incremental
        from .nonlinear import AnalysisSettings
        th = pl.eth[pl.elem_mat]
        if np.any(th):
            raise ValueError("2次元の塑性と熱ひずみの組み合わせにはまだ対応していません")
        u = solve_incremental(pl.internal, pl.commit, F, fixed, nd, settings or AnalysisSettings(), log, free_mask=used)
        fint, K = pl.internal(u)
    else:
        fth = pl.thermal_force()
        if fth is not None:
            F = F + fth
        _, K = pl.internal(np.zeros(nd))
        cd = np.array(sorted(fixed), np.int64)
        cv = np.array([fixed[int(d)] for d in cd])
        free = np.ones(nd, bool)
        free[cd] = False
        free &= used
        u = np.zeros(nd)
        u[cd] = cv
        Kff = K[free][:, free].tocsc()
        rhs = F[free] - K[free][:, cd] @ cv
        uf = _linear_solve(Kff, rhs, log, method="direct")
        if not np.all(np.isfinite(uf)) or _rel_res(Kff, uf, rhs) > 1e-4:
            raise RuntimeError("解が求まりませんでした。拘束が不足していないか確認してください（剛体移動を止める支持が必要です）。")
        u[free] = uf
        fint, _ = pl.internal(u)
        if fth is not None:
            fint = fint + fth
    # 積分点の応力を要素内で 1 次式に当てはめて 6 節点へ外挿し、節点で面積平均（Ansys の節点平均と同じ考え方）
    gp = _GP6 if mode == "axisymmetric" else _GP
    S4 = _extrapolate_avg(mesh, pl.trial[4], gp)
    stress = np.zeros((mesh.n_nodes, 6))
    stress[:, 0], stress[:, 1], stress[:, 2], stress[:, 3] = S4[:, 0], S4[:, 1], S4[:, 2], S4[:, 3]
    if mode == "plane_stress":
        stress[:, 2] = 0.0
    U = np.zeros((mesh.n_nodes, 3))
    U[:, :2] = u.reshape(-1, 2)
    Rv = (fint - F).reshape(-1, 2)
    reactions = {}
    for i, bc in enumerate(bcs):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
            r3 = np.zeros(3)
            r3[:2] = Rv[nodes].sum(0)
            reactions[i] = r3
    info = {"自由度数": int(used.sum() - len(fixed)), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "要素タイプ": f"2次元 6節点三角形（{MODES[mode]}, PLANE183 相当）", "ソルバー": "直接法",
            "合計時間 [s]": round(time.time() - t0, 2)}
    r = Result(mesh, mats[0], U, stress, np.zeros_like(stress), reactions, info, mats, elem_mat)
    r.extra = {}
    if plastic:
        r.extra["相当塑性ひずみ"] = _nodal_avg(mesh, pl.pb.max(0)[:, None], np.abs(mesh.element_volumes()))[:, 0]
    if mode == "axisymmetric":
        r.extra["円周応力 σθ [MPa]"] = stress[:, 2]
    return r


def structured_rect(a, b, nx, ny, t=1.0, x0=0.0, y0=0.0, mode="plane_stress") -> Mesh2D:
    """長方形の 2 次元メッシュ（テスト用）。辺: 1:x=x0, 2:x=x0+a, 3:y=y0, 4:y=y0+b（面 0 の続き）"""
    nxp, nyp = 2 * nx + 1, 2 * ny + 1
    xs = np.linspace(x0, x0 + a, nxp)
    ys = np.linspace(y0, y0 + b, nyp)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    nid = lambda i, j: i * nyp + j  # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            I, J = 2 * i, 2 * j
            c00, c20, c22, c02 = nid(I, J), nid(I + 2, J), nid(I + 2, J + 2), nid(I, J + 2)
            m10, m21, m12, m01, m11 = nid(I + 1, J), nid(I + 2, J + 1), nid(I + 1, J + 2), nid(I, J + 1), nid(I + 1, J + 1)
            tris.append([c00, c20, c22, m10, m21, m11])
            tris.append([c00, c22, c02, m11, m12, m01])
    tris = np.array(tris)
    def edge(pts):
        return np.array([[pts[k], pts[k + 2], pts[k + 1]] for k in range(0, len(pts) - 2, 2)])
    edges = [edge([nid(0, j) for j in range(nyp)]), edge([nid(nxp - 1, j) for j in range(nyp)]),
             edge([nid(i, 0) for i in range(nxp)]), edge([nid(i, nyp - 1) for i in range(nxp)])]
    return Mesh2D(nodes, tris, np.zeros(len(tris), int), edges, np.full(len(tris), float(t)),
                  elem_body=np.zeros(len(tris), int), n_surf=1, mode=mode)

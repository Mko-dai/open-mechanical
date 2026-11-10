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

from .mesh import Mesh
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


def _mixed_system(solid, shell, material, bcs_solid, bcs_shell, solid_materials, shell_materials, tie_tol, log,
                  supports_only=False, mass=False, K_add=None):
    """ソリッドとシェルの剛性（と質量）を並べ、結合の多点拘束を消去した式を作る"""
    from .shell import _frames, _shell_dofs, plane_stress_D, shell_constraints, shell_element_K
    for b in bcs_solid:
        if b.kind not in SOLID_BC:
            raise ValueError(f"ソリッド＋シェルの解析では「{b.label()}」はまだ使えません")
    for b in bcs_shell:
        if b.kind not in SHELL_BC:
            raise ValueError(f"ソリッド＋シェルの解析では、シェル側の「{b.label()}」はまだ使えません")
    if supports_only:      # モーダル: 荷重は使わず、強制変位は 0 として扱う
        def sup(bcs):
            out = []
            for b in bcs:
                if b.kind == "displacement":
                    out.append(BC("displacement", b.faces, [None if v is None else 0.0 for v in b.values], b.name))
                elif b.kind in ("fixed", "symmetry", "frictionless"):
                    out.append(b)
            return out
        bcs_solid, bcs_shell = sup(bcs_solid), sup(bcs_shell)
    mats_s, em_s = solid_materials if solid_materials is not None else ([material], np.zeros(solid.n_elems, np.int64))
    mats_h, em_h = shell_materials if shell_materials is not None else ([material], np.zeros(shell.n_elems, np.int64))
    em_s, em_h = np.asarray(em_s, np.int64), np.asarray(em_h, np.int64)
    D = np.stack([material_D(m) for m in mats_s])
    Ks = assemble_stiffness(solid, D, elem_mat=em_s).tocsr()
    if K_add is not None:          # ソリッドどうしの接触（結合・分離なし）の剛性
        Ks = (Ks + K_add).tocsr()
    rho_e = np.array([m.density_t_mm3 for m in mats_s])[em_s]
    Fs, fx_s = build_load_and_constraints(solid, mats_s[0], bcs_solid, rho_e=rho_e)
    ns = Ks.shape[0]
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
    fixed = {int(newidx[d]): v for d, v in fx_s.items()}
    for d, v in fx_h.items():
        if not is_slave[ns + d]:
            fixed[int(newidx[ns + d])] = v
    for d in np.nonzero(~used)[0]:
        if not is_slave[ns + d]:
            fixed[int(newidx[ns + d])] = 0.0
    out = dict(K=K, F=F, T=T, fixed=fixed, ns=ns, nh=nh, D=D, mats_s=mats_s, em_s=em_s, mats_h=mats_h, em_h=em_h,
               tied=tied, bcs_solid=bcs_solid, bcs_shell=bcs_shell)
    if mass:
        from .dynamics import consistent_mass
        Ms = consistent_mass(solid, rho_e)
        # シェルの質量: 集中質量（要素の質量を 3 節点に等分）＋ 板厚による回転慣性 ρ t³ A / 12 の 1/3 ずつ
        rho_h = np.array([m.density_t_mm3 for m in mats_h])[em_h]
        me = rho_h * shell.thickness * Ah
        mt = np.bincount(shell.elems.ravel(), np.repeat(me / 3, 3), shell.n_nodes)
        mr = np.bincount(shell.elems.ravel(), np.repeat(me * shell.thickness ** 2 / 36, 3), shell.n_nodes)
        out["M"] = sp.bmat([[Ms, None], [None, sp.diags(np.column_stack([mt, mt, mt, mr, mr, mr]).ravel())]],
                           format="csr")
        out["mass"] = float((rho_e * np.abs(solid.element_volumes())).sum() + me.sum())
    return out


def solve_mixed(solid, shell, material, bcs_solid: list[BC], bcs_shell: list[BC], solid_materials=None,
                shell_materials=None, tie_tol=None, log=print, method="auto", K_add=None):
    """ソリッド＋シェルの線形静解析。 戻り値: (ソリッドの Result, シェルの Result)"""
    t0 = time.time()
    sy = _mixed_system(solid, shell, material, bcs_solid, bcs_shell, solid_materials, shell_materials, tie_tol, log,
                       K_add=K_add)
    K, F, T, fixed, ns = sy["K"], sy["F"], sy["T"], sy["fixed"], sy["ns"]
    mats_s, em_s, mats_h, em_h, D, tied = sy["mats_s"], sy["em_s"], sy["mats_h"], sy["em_h"], sy["D"], sy["tied"]
    Kr = (T.T @ K @ T).tocsr()
    Fr = T.T @ F
    if not fixed:
        raise ValueError("拘束（固定支持 または 変位）がありません")
    nk = Kr.shape[0]
    cd = np.array(sorted(fixed), np.int64)
    cv = np.array([fixed[int(d)] for d in cd])
    free = np.ones(nk, bool)
    free[cd] = False
    ur = np.zeros(nk)
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


def solve_mixed_modal(solid, shell, material, bcs_solid, bcs_shell, n_modes=6, solid_materials=None,
                      shell_materials=None, tie_tol=None, log=print, K_add=None):
    """ソリッド＋シェルのモーダル解析。 戻り値: (振動数 [Hz], [(ソリッドの Result, シェルの Result), …])
    モード形は最大変位 1 に正規化（変形・応力は相対値）"""
    import scipy.sparse.linalg as spla
    t0 = time.time()
    sy = _mixed_system(solid, shell, material, bcs_solid, bcs_shell, solid_materials, shell_materials, tie_tol, log,
                       supports_only=True, mass=True, K_add=K_add)
    T, ns = sy["T"], sy["ns"]
    Kr = (T.T @ sy["K"] @ T).tocsr()
    Mr = (T.T @ sy["M"] @ T).tocsr()
    free = np.ones(Kr.shape[0], bool)
    if sy["fixed"]:
        free[np.fromiter(sy["fixed"].keys(), np.int64)] = False
    Kff, Mff = Kr[free][:, free].tocsc(), Mr[free][:, free].tocsc()
    lam_ref = float(Kff.diagonal().mean() / max(Mff.diagonal().mean(), 1e-300))
    nm = min(int(n_modes), Kff.shape[0] - 2)
    w2, X = spla.eigsh(Kff, k=nm, M=Mff, sigma=-1e-6 * lam_ref, which="LM")
    o = np.argsort(w2)
    freqs = np.sqrt(np.maximum(w2[o], 0.0)) / (2 * np.pi)
    info = {"解析": "モーダル（ソリッド＋シェル）", "自由度数": int(free.sum()), "質量 [kg]": sy["mass"] * 1000,
            "合計時間 [s]": round(time.time() - t0, 2)}
    out = []
    for j in o:
        ur = np.zeros(Kr.shape[0])
        ur[free] = X[:, j]
        u = T @ ur
        us, uh = u[:ns], u[ns:]
        sc = 1.0 / max(np.abs(us).max(initial=0.0), np.abs(uh.reshape(-1, 6)[:, :3]).max(initial=0.0), 1e-300)
        us, uh = us * sc, uh * sc
        stress, strain = nodal_stress(solid, us, sy["D"], elem_mat=sy["em_s"])
        rs = Result(solid, sy["mats_s"][0], us.reshape(-1, 3).copy(), stress, strain, {}, dict(info), sy["mats_s"],
                    sy["em_s"])
        rh = _shell_post(shell, uh, sy["mats_h"], sy["em_h"], {}, dict(info))
        out.append((rs, rh))
    return freqs, out


# ---------------------------------------------------------------------------- 1 つのモデルとして扱う

class MixedMesh(Mesh):
    """ソリッドのメッシュとシェルのメッシュを、画面・選択・境界条件の上では 1 つのメッシュとして見せる。

    節点: ソリッドの節点 → シェルの節点 の順。 面番号: ソリッドの面 → シェルの面 → シェルの辺 の順。
    ボディ番号: ソリッドのボディ（フィーチャーの順）→ シェルのボディ（body_offset から）。
    elems はソリッドの要素だけ（断面表示など体積を使う処理用）。 解析は solid / shell を別々に組み立てて結合する。
    """

    def __init__(self, solid, shell, body_offset):
        self.solid, self.shell = solid, shell
        self.is_mixed = True
        ns = solid.n_nodes
        self.ns = ns
        if solid.face_ids is None:
            solid.group_faces()
        self.nfs = int(solid.n_faces)
        shell.elem_body = (np.zeros(shell.n_elems, np.int64) if shell.elem_body is None
                           else np.asarray(shell.elem_body, np.int64)) + int(body_offset)
        self.nodes = np.vstack([solid.nodes, shell.nodes])
        self.elems = solid.elems
        self.elem_body = (solid.elem_body if solid.elem_body is not None else np.zeros(solid.n_elems, np.int64))
        st = solid.boundary_tris
        sf = solid.boundary_tris_full
        ht = shell.elems + ns
        self._btri = np.vstack([st, ht])
        # 表示用: ソリッドが 2 次（6 節点三角形）なら、シェルの三角形も 6 列にそろえる（中間節点 = 頂点の繰り返し）
        self._btri_full = np.vstack([sf, np.hstack([ht, ht]) if sf.shape[1] == 6 else ht])
        self._btri_owner = np.r_[solid.boundary_owner, solid.n_elems + np.arange(shell.n_elems)]
        self._tri_body = np.r_[solid.tri_body(), shell.elem_body]
        self.face_ids = np.r_[solid.face_ids, self.nfs + shell.face_ids]
        self.n_surf = self.nfs + shell.n_surf                      # これ以降の面番号がシェルの辺
        self.edges = [e + ns for e in shell.edges]
        self.thickness = shell.thickness
        self.body_names = None
        self.node_origin = np.arange(len(self.nodes))
        self._bquad = self._bquad_of = None
        so = getattr(solid, "node_origin", None)
        self.solid_unshared = bool(so is not None and len(np.unique(so)) < solid.n_nodes)

    # --- 面・ボディ
    @property
    def n_faces(self):
        return self.nfs + self.shell.n_faces

    def is_edge(self, fid):
        return int(fid) >= self.n_surf

    def split_face(self, fid):
        """全体の面番号 → ("solid" | "shell", それぞれのメッシュでの面番号)"""
        fid = int(fid)
        return ("solid", fid) if fid < self.nfs else ("shell", fid - self.nfs)

    def face_info(self):
        out = list(self.solid.face_info())
        for f in self.shell.face_info():
            out.append(dict(f, id=f["id"] + self.nfs))
        return out

    def face_tris(self, face_id, full=True):
        which, f = self.split_face(face_id)
        if which == "solid":
            return self.solid.face_tris(f, full)
        return self.shell.face_tris(f) + self.ns

    def face_nodes(self, face_id):
        out = []
        for fid in np.atleast_1d(face_id):
            which, f = self.split_face(fid)
            out.append(self.solid.face_nodes(f) if which == "solid" else self.shell.face_nodes(f) + self.ns)
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def nearest_edge(self, point, tol=np.inf):
        k = self.shell.nearest_edge(point, tol)
        return k + self.nfs if k >= 0 else -1

    def body_ids(self):
        return np.unique(np.r_[self.solid.body_ids(), self.shell.elem_body])

    def tri_body(self):
        return self._tri_body

    def body_faces(self, body):
        return sorted(set(self.face_ids[self._tri_body == body].tolist()))

    def is_shell_body(self, body) -> bool:
        return bool(np.any(self.shell.elem_body == int(body)))

    @property
    def n_elems(self):
        return self.solid.n_elems + self.shell.n_elems

    def group_faces(self, angle_deg=35.0):
        return self.face_ids

    def _compute_boundary(self):
        pass

    def to_quadratic(self):
        return self

    def split_body_interfaces(self, share=None):
        return self


def run_mixed(project, mesh: MixedMesh, log=print) -> list:
    """プロジェクトの設定（材料・境界条件）で、ソリッド＋シェルのモデルを解く。 結果（Result のリスト）は MixedMesh の節点の並び"""
    p = project
    if p.analysis_type not in ("static", "modal"):
        raise ValueError("ソリッドとシェルを混ぜたモデルは、いまは静的構造（線形）とモーダルだけ解けます")
    s = p.settings
    if s.large_deflection or s.arc_length:
        raise ValueError("ソリッドとシェルを混ぜたモデルは、いまは線形の解析だけです（大変形・弧長法は不可）")
    bs, bh, origin = [], [], []
    import copy
    for i, b in enumerate(p.bcs):
        fs = [f for f in b.faces if int(f) < mesh.nfs]
        fh = [int(f) - mesh.nfs for f in b.faces if int(f) >= mesh.nfs]
        if b.kind == "gravity":
            bs.append(b)
            bh.append(b)
            origin += [("solid", i), ("shell", i)]
            continue
        if fs and fh and b.kind in ("force", "moment"):
            raise ValueError(f"{b.label()}: ソリッドの面とシェルの面・辺を 1 つの荷重にまとめられません（別々に設定してください）")
        if b.nodes:
            raise ValueError(f"{b.label()}: 節点で指定した条件は、ソリッド＋シェルのモデルではまだ使えません")
        if fs:
            nb = copy.copy(b)
            nb.faces = fs
            bs.append(nb)
            origin.append(("solid", i))
        if fh:
            nb = copy.copy(b)
            nb.faces = fh
            bh.append(nb)
            origin.append(("shell", i))
    em_s = p.elem_materials(mesh.solid)
    em_h = p.elem_materials(mesh.shell)
    ns = mesh.ns
    # ソリッドどうしの接触: 結合・分離なしを線形の剛性として足す（シェルとソリッドは自動で結合する）
    from .contact import LINEAR_TYPES, detect_contacts, linear_contact_stiffness
    cs = [c for c in p.contacts if not c.suppressed]
    if not cs and mesh.solid.elem_body is not None and len(np.unique(mesh.solid.elem_body)) > 1:
        p.contacts = detect_contacts(mesh.solid, p.contact_tolerance, p.contact_default_type, p.contact_default_mu)
        cs = list(p.contacts)
    Kc = None
    if cs:
        for c in cs:
            if any(int(f) >= mesh.nfs for f in list(c.contact_faces) + list(c.target_faces)):
                raise ValueError(f"接触「{c.name}」: シェルの面・辺は接触に使えません（シェルはソリッドに自動で結合されます）")
            if p.analysis_type == "static" and c.type not in LINEAR_TYPES:
                raise ValueError(f"接触「{c.name}」: ソリッドとシェルを混ぜたモデルの静解析では、結合 と 分離なし だけ使えます"
                                 "（摩擦なし・摩擦あり・粗いは不可）")
        Kc = linear_contact_stiffness(mesh.solid, cs, em_s[0], em_s[1], p.contact_tolerance)
        if Kc is not None:
            log(f"接触: ソリッドどうしの {len(cs)} 領域を線形の接触として含めます")
    if mesh.solid_unshared and Kc is None:
        raise ValueError("ソリッドのボディどうしがつながっていません（接触領域を設定するか、節点共有にしてください）")

    def combine(rs, rh, info):
        r = Result(mesh, em_s[0][0], np.vstack([rs.u, rh.u]), np.vstack([rs.stress, rh.stress]),
                   np.vstack([rs.strain, rh.strain]), {}, info, em_s[0], em_s[1])
        r.extra = {}
        for k, v in rh.extra.items():
            a = np.full(mesh.n_nodes, np.nan)
            a[ns:] = v
            r.extra[k + "（シェル）" if "シェル" not in k else k] = a
        r.parts = (rs, rh)
        return r
    if p.analysis_type == "modal":
        o = p.analysis_opts or {}
        if o.get("prestress"):
            raise ValueError("ソリッド＋シェルのモーダルでは、プリストレスはまだ使えません")
        freqs, modes = solve_mixed_modal(mesh.solid, mesh.shell, p.material, bs, bh, int(o.get("n_modes", 6)),
                                         solid_materials=em_s, shell_materials=em_h, log=log, K_add=Kc)
        out = []
        table = [{"モード": i + 1, "振動数 [Hz]": float(f)} for i, f in enumerate(freqs)]
        for i, (f, (rs, rh)) in enumerate(zip(freqs, modes)):
            r = combine(rs, rh, dict(rs.info, 節点数=mesh.n_nodes, 要素数=mesh.n_elems))
            r.time = float(i + 1)
            r.info["振動数 [Hz]"] = float(f)
            r.label = f"モード {i + 1}（{f:.5g} Hz）"
            r.modal_table = table
            r.mode_shape = True
            out.append(r)
        return out
    rs, rh = solve_mixed(mesh.solid, mesh.shell, p.material, bs, bh, solid_materials=em_s, shell_materials=em_h,
                         log=log, method=s.solver_type, K_add=Kc)
    reactions = {}
    js = jh = 0
    for which, i in origin:
        if which == "solid":
            if js in rs.reactions:
                reactions[i] = reactions.get(i, 0) + rs.reactions[js]
            js += 1
        else:
            if jh in rh.reactions:
                reactions[i] = reactions.get(i, 0) + rh.reactions[jh]
            jh += 1
    info = dict(rs.info, 節点数=mesh.n_nodes, 要素数=mesh.n_elems,
                要素タイプ="ソリッド（四面体）＋シェル（3 節点 DKT＋Allman 膜）")
    r = combine(rs, rh, info)
    r.reactions = reactions
    r.time = 1.0
    return [r]

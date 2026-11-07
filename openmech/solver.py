"""線形静的構造解析ソルバー（3次元、等方弾性、Tet4 / Tet10）。

単位系: mm, N, MPa, t（トン）, s
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import Material
from .mesh import Mesh

# ---------------------------------------------------------------------------- 形状関数

_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS4 = np.array([[_A, _B, _B, _B], [_B, _A, _B, _B], [_B, _B, _A, _B], [_B, _B, _B, _A]])
GAUSS4_W = np.full(4, 1.0 / 24.0)
GAUSS1 = np.array([[0.25, 0.25, 0.25, 0.25]])
GAUSS1_W = np.array([1.0 / 6.0])
# 体積座標 L → 自然座標 (ξ,η,ζ):  L1 = 1-ξ-η-ζ, L2 = ξ, L3 = η, L4 = ζ
_DL_DXI = np.array([[-1.0, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
_EDGES10 = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

# 各節点の体積座標（節点位置での応力評価用）
NODE_L4 = np.eye(4)
NODE_L10 = np.vstack([np.eye(4)] + [0.5 * (np.eye(4)[i] + np.eye(4)[j]) for i, j in _EDGES10])


def shape_N(L: np.ndarray, nn: int) -> np.ndarray:
    if nn == 4:
        return L.copy()
    N = np.empty(10)
    N[:4] = L * (2 * L - 1)
    for k, (i, j) in enumerate(_EDGES10):
        N[4 + k] = 4 * L[i] * L[j]
    return N


def shape_dN_dxi(L: np.ndarray, nn: int) -> np.ndarray:
    """(nn,3) 自然座標での形状関数の微分。"""
    if nn == 4:
        return _DL_DXI.copy()
    dNdL = np.zeros((10, 4))
    for i in range(4):
        dNdL[i, i] = 4 * L[i] - 1
    for k, (i, j) in enumerate(_EDGES10):
        dNdL[4 + k, i] = 4 * L[j]
        dNdL[4 + k, j] = 4 * L[i]
    return dNdL @ _DL_DXI


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _dN_dx(X: np.ndarray, dNdxi: np.ndarray):
    """X:(E,nn,3) → dN/dx:(E,nn,3), detJ:(E,)"""
    J = np.einsum("enk,nl->ekl", X, dNdxi)
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dNdx = np.einsum("nl,elk->enk", dNdxi, invJ)
    return dNdx, detJ


def _B_matrix(dNdx: np.ndarray) -> np.ndarray:
    """(E,nn,3) → (E,6,3nn)  ひずみ順: xx, yy, zz, xy, yz, zx（工学ひずみ）"""
    E, nn, _ = dNdx.shape
    B = np.zeros((E, 6, 3 * nn))
    dx, dy, dz = dNdx[..., 0], dNdx[..., 1], dNdx[..., 2]
    B[:, 0, 0::3] = dx
    B[:, 1, 1::3] = dy
    B[:, 2, 2::3] = dz
    B[:, 3, 0::3] = dy
    B[:, 3, 1::3] = dx
    B[:, 4, 1::3] = dz
    B[:, 4, 2::3] = dy
    B[:, 5, 0::3] = dz
    B[:, 5, 2::3] = dx
    return B


def _quadrature(nn):
    return (GAUSS4, GAUSS4_W) if nn == 10 else (GAUSS1, GAUSS1_W)


# ---------------------------------------------------------------------------- 組み立て

def _D_of(D, elem_mat, sl):
    """要素ごとの弾性マトリクス。D が (6,6) なら共通、(M,6,6) なら elem_mat で引く"""
    if D.ndim == 2:
        return D
    return D[elem_mat[sl]]


def assemble_stiffness(mesh: Mesh, D: np.ndarray, chunk: int = 20000,
                       progress=None, elem_mat=None) -> sp.csr_matrix:
    nn = mesh.elems.shape[1]
    ndof = 3 * mesh.n_nodes
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    K = sp.csr_matrix((ndof, ndof))
    ne = mesh.n_elems
    loc = np.arange(3 * nn)
    ii, jj = np.meshgrid(loc, loc, indexing="ij")
    for s in range(0, ne, chunk):
        el = mesh.elems[s:s + chunk]
        X = mesh.nodes[el]
        Ke = np.zeros((len(el), 3 * nn, 3 * nn))
        De = _D_of(D, elem_mat, slice(s, s + chunk))
        for g, w in enumerate(wts):
            dNdx, detJ = _dN_dx(X, dN[g])
            if np.any(detJ <= 0):
                raise ValueError("つぶれた（体積が0以下の）要素があります。メッシュを見直してください。")
            B = _B_matrix(dNdx)
            if De.ndim == 2:
                Ke += np.einsum("e,eki,kl,elj->eij", w * detJ, B, De, B, optimize=True)
            else:
                Ke += np.einsum("e,eki,ekl,elj->eij", w * detJ, B, De, B, optimize=True)
        dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(el), -1)
        rows = dofs[:, ii].ravel()
        cols = dofs[:, jj].ravel()
        K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof))
        if progress:
            progress(min(1.0, (s + chunk) / ne))
    return K


def body_force_vector(mesh: Mesh, b: np.ndarray) -> np.ndarray:
    """体積力 b [N/mm^3]（= 密度×加速度）の等価節点力。 b は (3,) または要素ごと (E,3)"""
    b = np.asarray(b, float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (mesh.n_elems, 3))
    nn = mesh.elems.shape[1]
    pts, wts = GAUSS4, GAUSS4_W  # 2次の形状関数まで厳密に積分できる
    X = mesh.nodes[mesh.elems]
    F = np.zeros((mesh.n_nodes, 3))
    for L, w in zip(pts, wts):
        _, detJ = _dN_dx(X, shape_dN_dxi(L, nn))
        N = shape_N(L, nn)
        contrib = (w * detJ)[:, None] * N[None, :]          # (E,nn)
        for c in range(3):
            np.add.at(F[:, c], mesh.elems.ravel(), (contrib * b[:, c:c + 1]).ravel())
    return F.ravel()


def surface_weights(mesh: Mesh, tris: np.ndarray):
    """一様な表面力を節点に配分する重み（= 各節点の受け持ち面積）。

    戻り値: (nodes, weights, tri_normals, tri_areas)
    Tri3: 各頂点に A/3。 Tri6(直線辺): 頂点 0、中間節点に A/3。
    """
    p = mesh.nodes[tris[:, :3]]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    A = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * A, 1e-300)[:, None]
    if tris.shape[1] == 6:
        w = np.zeros((len(tris), 6))
        w[:, 3:] = A[:, None] / 3.0
    else:
        w = np.repeat(A[:, None] / 3.0, 3, axis=1)
    return tris, w, n, A


# ---------------------------------------------------------------------------- 境界条件

@dataclass
class BC:
    """境界条件・荷重の定義。

    kind:
      "fixed"        固定支持（faces の全節点で ux=uy=uz=0）
      "displacement" 強制変位（values=[ux,uy,uz]、None の成分は自由）
      "force"        力 [N]（values=[Fx,Fy,Fz] を面全体に一様分布）
      "pressure"     圧力 [MPa]（values=[p]、正で面を押す向き）
      "moment"       モーメント [N·mm]（values=[Mx,My,Mz]、面の重心まわり）
      "gravity"      重力・加速度 [mm/s^2]（values=[ax,ay,az]、faces 不要）
    """
    kind: str
    faces: list[int] = field(default_factory=list)
    values: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    name: str = ""
    nodes: list[int] = field(default_factory=list)   # 面の代わりに節点を直接指定（線・点の拘束用）
    step_factors: list = field(default_factory=list)  # ステップごとの倍率（空ならすべて 1）

    def to_dict(self):
        d = {"kind": self.kind, "faces": list(map(int, self.faces)),
             "values": list(self.values), "name": self.name}
        if self.nodes:
            d["nodes"] = list(map(int, self.nodes))
        if self.step_factors:
            d["step_factors"] = [float(x) for x in self.step_factors]
        return d

    @staticmethod
    def from_dict(d):
        return BC(d["kind"], d.get("faces", []), d.get("values", [0, 0, 0]), d.get("name", ""),
                  d.get("nodes", []), d.get("step_factors", []))

    def label(self) -> str:
        jp = {"fixed": "固定支持", "displacement": "強制変位", "force": "力",
              "pressure": "圧力", "moment": "モーメント", "gravity": "重力加速度"}
        name = self.name or jp.get(self.kind, self.kind)
        unit = {"force": "N", "pressure": "MPa", "moment": "N·mm",
                "gravity": "mm/s²", "displacement": "mm"}.get(self.kind, "")
        v = ""
        if self.kind != "fixed":
            v = " [" + ", ".join("自由" if x is None else f"{x:g}" for x in self.values) + f"] {unit}"
        f = f" 面{','.join(str(i + 1) for i in self.faces)}" if self.faces else ""
        return f"{name}{f}{v}"


def build_load_and_constraints(mesh: Mesh, material: Material, bcs: list[BC], rho_e=None):
    ndof = 3 * mesh.n_nodes
    F = np.zeros(ndof)
    fixed = {}  # dof -> value
    for bc in bcs:
        if bc.kind in ("fixed", "displacement"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, dtype=np.int64)
            vals = [0.0, 0.0, 0.0] if bc.kind == "fixed" else bc.values
            for c in range(3):
                if vals[c] is None:
                    continue
                for d in (3 * nodes + c):
                    fixed[int(d)] = float(vals[c])
        elif bc.kind == "force":
            tris = np.concatenate([mesh.face_tris(f) for f in bc.faces])
            tri, w, _, A = surface_weights(mesh, tris)
            Atot = A.sum()
            t = np.asarray(bc.values, float) / Atot
            for c in range(3):
                np.add.at(F, 3 * tri.ravel() + c, (w * t[c]).ravel())
        elif bc.kind == "pressure":
            p = float(bc.values[0])
            tris = np.concatenate([mesh.face_tris(f) for f in bc.faces])
            tri, w, n, _ = surface_weights(mesh, tris)
            for c in range(3):
                np.add.at(F, 3 * tri.ravel() + c, (-p * w * n[:, c:c + 1]).ravel())
        elif bc.kind == "moment":
            tris = np.concatenate([mesh.face_tris(f) for f in bc.faces])
            tri, w, _, _ = surface_weights(mesh, tris)
            nodes = tri.ravel()
            wn = np.bincount(nodes, weights=w.ravel(), minlength=mesh.n_nodes)
            idx = np.nonzero(wn)[0]
            wi = wn[idx]
            r = mesh.nodes[idx] - (mesh.nodes[idx] * wi[:, None]).sum(0) / wi.sum()
            J = (wi[:, None, None] * (np.einsum("ij,ij->i", r, r)[:, None, None] * np.eye(3)
                                      - r[:, :, None] * r[:, None, :])).sum(0)
            omega = np.linalg.lstsq(J, np.asarray(bc.values, float), rcond=None)[0]
            f = wi[:, None] * np.cross(omega, r)
            for c in range(3):
                F[3 * idx + c] += f[:, c]
        elif bc.kind == "gravity":
            acc = np.asarray(bc.values, float)
            if rho_e is None:
                F += body_force_vector(mesh, material.density_t_mm3 * acc)
            else:
                F += body_force_vector(mesh, rho_e[:, None] * acc[None, :])
        else:
            raise ValueError(f"未対応の境界条件: {bc.kind}")
    return F, fixed


# ---------------------------------------------------------------------------- 結果

@dataclass
class Result:
    mesh: Mesh
    material: Material
    u: np.ndarray            # (N,3) 変位 [mm]
    stress: np.ndarray       # (N,6) 節点応力（平均化） [MPa]  xx,yy,zz,xy,yz,zx
    strain: np.ndarray       # (N,6) 節点ひずみ（工学ひずみ）
    reactions: dict          # BC index -> 反力ベクトル [N]
    info: dict
    mats: list | None = None       # ボディごと等の材料リスト
    elem_mat: np.ndarray | None = None  # 要素 → mats の番号

    def _node_yield(self):
        if self.mats is None or self.elem_mat is None:
            return None
        ys = np.array([m.yield_strength if m.yield_strength > 0 else np.inf for m in self.mats])
        ye = ys[self.elem_mat]
        ny = np.full(self.mesh.n_nodes, np.inf)
        np.minimum.at(ny, self.mesh.elems.ravel(), np.repeat(ye, self.mesh.elems.shape[1]))
        return ny

    @property
    def disp_mag(self):
        return np.linalg.norm(self.u, axis=1)

    @property
    def von_mises(self):
        s = self.stress
        return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2
                              + (s[:, 2] - s[:, 0]) ** 2)
                       + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))

    @property
    def principal(self):
        """(N,3) 主応力 σ1 ≥ σ2 ≥ σ3"""
        s = self.stress
        T = np.empty((len(s), 3, 3))
        T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
        T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
        T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
        return np.linalg.eigvalsh(T)[:, ::-1]

    @property
    def safety_factor(self):
        ny = self._node_yield()
        vm = self.von_mises
        if ny is not None:
            if not np.isfinite(ny).any():
                return None
            return np.minimum(15.0, np.where(np.isfinite(ny), ny, 15.0 * vm) / np.maximum(vm, 1e-12))
        sy = self.material.yield_strength
        if sy <= 0:
            return None
        return np.minimum(15.0, sy / np.maximum(vm, 1e-12))

    def fields(self) -> dict:
        """表示用の結果一覧 {名前: (N,) 配列}"""
        p = self.principal
        out = {
            "全変形 [mm]": self.disp_mag,
            "変位 X [mm]": self.u[:, 0],
            "変位 Y [mm]": self.u[:, 1],
            "変位 Z [mm]": self.u[:, 2],
            "相当応力（ミーゼス） [MPa]": self.von_mises,
            "最大主応力 [MPa]": p[:, 0],
            "最小主応力 [MPa]": p[:, 2],
            "応力 σx [MPa]": self.stress[:, 0],
            "応力 σy [MPa]": self.stress[:, 1],
            "応力 σz [MPa]": self.stress[:, 2],
            "せん断応力 τxy [MPa]": self.stress[:, 3],
            "せん断応力 τyz [MPa]": self.stress[:, 4],
            "せん断応力 τzx [MPa]": self.stress[:, 5],
        }
        sf = self.safety_factor
        if sf is not None:
            out["安全率（降伏基準）"] = sf
        return out

    def summary(self) -> dict:
        vm = self.von_mises
        i = int(np.argmax(vm))
        d = {
            "最大全変形 [mm]": float(self.disp_mag.max()),
            "最大相当応力 [MPa]": float(vm.max()),
            "最大相当応力の位置 [mm]": self.mesh.nodes[i].round(3).tolist(),
            "最大主応力 [MPa]": float(self.principal[:, 0].max()),
            "最小主応力 [MPa]": float(self.principal[:, 2].min()),
        }
        sf = self.safety_factor
        if sf is not None:
            d["最小安全率"] = float(sf.min())
        if self.mats is not None and self.elem_mat is not None:
            rho = np.array([m.density_t_mm3 for m in self.mats])[self.elem_mat]
            d["質量 [kg]"] = float((self.mesh.element_volumes() * rho).sum() * 1000)
        else:
            d["質量 [kg]"] = self.mesh.volume() * self.material.density_t_mm3 * 1000
        return d


def nodal_stress(mesh: Mesh, u: np.ndarray, D: np.ndarray, chunk: int = 50000, elem_mat=None):
    """各要素の節点位置で応力を評価し、節点で平均化する。"""
    nn = mesh.elems.shape[1]
    Lnodes = NODE_L10 if nn == 10 else NODE_L4
    acc_s = np.zeros((mesh.n_nodes, 6))
    acc_e = np.zeros((mesh.n_nodes, 6))
    cnt = np.zeros(mesh.n_nodes)
    U = u.reshape(-1, 3)
    for s in range(0, mesh.n_elems, chunk):
        el = mesh.elems[s:s + chunk]
        X = mesh.nodes[el]
        ue = U[el].reshape(len(el), -1)
        for a in range(nn):
            dNdx, _ = _dN_dx(X, shape_dN_dxi(Lnodes[a], nn))
            B = _B_matrix(dNdx)
            eps = np.einsum("eij,ej->ei", B, ue)
            De = _D_of(D, elem_mat, slice(s, s + chunk))
            sig = eps @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps)
            np.add.at(acc_e, el[:, a], eps)
            np.add.at(acc_s, el[:, a], sig)
            np.add.at(cnt, el[:, a], 1)
    cnt = np.maximum(cnt, 1)[:, None]
    return acc_s / cnt, acc_e / cnt


def solve(mesh: Mesh, material: Material, bcs: list[BC], progress=None, log=print,
          method: str = "auto", elem_materials=None) -> Result:
    """線形静解析を実行する。progress(0..1, メッセージ) を渡すと進捗を通知。"""
    t0 = time.time()

    def prog(x, msg):
        if progress:
            progress(x, msg)

    if not any(b.kind in ("fixed", "displacement") for b in bcs):
        raise ValueError("拘束（固定支持 または 強制変位）がありません。少なくとも1つ設定してください。")
    mats, elem_mat, rho_e = None, None, None
    if elem_materials is not None:
        # elem_materials = (材料リスト, 要素ごとの番号)  ボディごとに材料が違う場合
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
        D = np.stack([elasticity_matrix(m.E, m.nu) for m in mats])
        rho_e = np.array([m.density_t_mm3 for m in mats])[elem_mat]
        material = mats[0]
    else:
        D = elasticity_matrix(material.E, material.nu)
    prog(0.02, "剛性マトリクスを組み立て中…")
    K = assemble_stiffness(mesh, D, progress=lambda x: prog(0.02 + 0.4 * x, "剛性マトリクスを組み立て中…"),
                           elem_mat=elem_mat)
    t_asm = time.time() - t0
    F, fixed = build_load_and_constraints(mesh, material, bcs, rho_e=rho_e)
    ndof = K.shape[0]
    cdofs = np.fromiter(fixed.keys(), dtype=np.int64)
    cvals = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(ndof, bool)
    free[cdofs] = False
    u = np.zeros(ndof)
    u[cdofs] = cvals
    Kff = K[free][:, free].tocsc()
    rhs = F[free] - K[free][:, cdofs] @ cvals
    prog(0.45, f"連立方程式を解いています（自由度 {Kff.shape[0]:,}）…")
    t1 = time.time()
    dof_ids = np.nonzero(free)[0]
    coords = np.column_stack([mesh.nodes[dof_ids // 3], dof_ids % 3])
    with np.errstate(all="ignore"):
        uf = _linear_solve(Kff, rhs, log, coords=coords, method=method)
    if not np.all(np.isfinite(uf)):
        raise RuntimeError("解が求まりませんでした。拘束が不足して物体が自由に動ける状態"
                           "（剛体運動）になっていないか確認してください。")
    # 剛体運動のチェック（残差）
    res = _rel_res(Kff, uf, rhs)
    if res > 1e-4:
        raise RuntimeError(f"解の精度が不十分です（相対残差 {res:.1e}）。拘束不足の可能性があります。")
    u[free] = uf
    t_sol = time.time() - t1
    prog(0.85, "応力を計算中…")
    stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat)
    R = K @ u - F
    reactions = {}
    for i, bc in enumerate(bcs):
        if bc.kind in ("fixed", "displacement"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, dtype=np.int64)
            reactions[i] = R.reshape(-1, 3)[nodes].sum(0)
    prog(1.0, "完了")
    info = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "要素タイプ": "Tet10（2次）" if mesh.order == 2 else "Tet4（1次）",
            "組立時間 [s]": round(t_asm, 2), "求解時間 [s]": round(t_sol, 2),
            "合計時間 [s]": round(time.time() - t0, 2), "相対残差": float(res)}
    return Result(mesh, material, u.reshape(-1, 3), stress, strain, reactions, info, mats, elem_mat)


def _cg(A, b, M, tol=1e-10, maxiter=5000):
    try:
        return spla.cg(A, b, M=M, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:  # 古い SciPy
        return spla.cg(A, b, M=M, tol=tol, maxiter=maxiter)


def _rel_res(A, x, b):
    return np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)


def _linear_solve(A: sp.csc_matrix, b: np.ndarray, log=print, coords=None,
                  method: str = "auto") -> np.ndarray:
    """利用できる最速のソルバーを自動で選ぶ。

    優先順: PARDISO (pypardiso) → CHOLMOD (scikit-sparse) → AMG-CG (pyamg, 大規模時)
            → SciPy SuperLU（RCM 並べ替え＋対称モード）
    coords: 自由度ごとの (x,y,z, 成分) 情報（AMG の剛体モード用, (n,4)）
    """
    n = A.shape[0]
    if method in ("auto", "pardiso"):
        try:
            import pypardiso  # Intel MKL（Intel Mac / Windows / Linux）
            log("ソルバー: PARDISO")
            return pypardiso.spsolve(A.tocsr(), b)
        except ImportError:
            pass
    if method in ("auto", "cholmod"):
        try:
            from sksparse.cholmod import cholesky  # SuiteSparse（Apple Silicon でも高速）
            log("ソルバー: CHOLMOD")
            return cholesky(A)(b)
        except ImportError:
            pass
        except Exception as e:  # 正定値でない＝拘束不足
            log(f"CHOLMOD 失敗: {e}")
            return np.full(n, np.nan)
    if method in ("auto", "amg") and (n > 30_000 or method == "amg"):
        try:
            import pyamg
            log("ソルバー: AMG前処理付き共役勾配法 (pyamg)")
            B = None
            if coords is not None:
                # 剛体モード（並進3＋回転3）を与えると収束が大幅に速くなる
                x, comp = coords[:, :3], coords[:, 3].astype(int)
                B = np.zeros((n, 6))
                B[np.arange(n), comp] = 1.0
                rot = [(1, 2), (2, 0), (0, 1)]  # 回転軸まわりの (a,b) 成分
                for k, (a, c) in enumerate(rot):
                    B[comp == a, 3 + k] = -x[comp == a, c]
                    B[comp == c, 3 + k] = x[comp == c, a]
            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), B=B, symmetry="hermitian",
                                                   strength="symmetric", max_coarse=500)
            xsol, info = _cg(A, b, ml.aspreconditioner(cycle="V"), tol=1e-10, maxiter=2000)
            if info == 0 and _rel_res(A, xsol, b) < 1e-6:
                return xsol
            log("AMG が収束しなかったため直接法に切り替えます")
        except ImportError:
            pass
    # SciPy 直接法（RCM で帯幅を減らし、対称モードで分解）
    log("ソルバー: SciPy SuperLU")
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    import warnings
    p = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
    Ap = A[p][:, p].tocsc()
    with warnings.catch_warnings():
        warnings.simplefilter("error", spla.MatrixRankWarning)
        try:
            lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))
            xp = lu.solve(b[p])
        except (spla.MatrixRankWarning, RuntimeError):
            return np.full(n, np.nan)
    x = np.empty_like(xp)
    x[p] = xp
    return x

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

def material_D(m) -> np.ndarray:
    """材料の剛性マトリクス（温度・向き・異方性を反映した MaterialState なら Dm）"""
    Dm = getattr(m, "Dm", None)
    return np.asarray(Dm, float) if Dm is not None else elasticity_matrix(m.E, m.nu)


def material_eps_th(m) -> np.ndarray:
    e = getattr(m, "eps_th", None)
    return np.zeros(6) if e is None else np.asarray(e, float)


def thermal_load_vector(mesh: Mesh, D, elem_mat, eps_th_m):
    """熱ひずみ ε_th による等価節点力 Σ Bᵀ D ε_th dV（要素ごとの材料番号 elem_mat、材料ごとの eps_th_m (M,6)）"""
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    F = np.zeros(3 * mesh.n_nodes)
    em = elem_mat if elem_mat is not None else np.zeros(mesh.n_elems, dtype=np.int64)
    sel = np.nonzero(np.any(eps_th_m[em] != 0, axis=1))[0]
    if len(sel) == 0:
        return F
    el = mesh.elems[sel]
    X = mesh.nodes[el]
    De = D if D.ndim == 2 else D[em[sel]]
    sig = np.einsum("eij,ej->ei", np.broadcast_to(De, (len(sel), 6, 6)), eps_th_m[em[sel]])
    fe = np.zeros((len(sel), 3 * nn))
    for g, w in enumerate(wts):
        dNdx, detJ = _dN_dx(X, shape_dN_dxi(pts[g], nn))
        fe += np.einsum("e,eki,ek->ei", w * detJ, _B_matrix(dNdx), sig)
    dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(sel), -1)
    np.add.at(F, dofs.ravel(), fe.ravel())
    return F


def _D_of(D, elem_mat, sl):
    """要素ごとの弾性マトリクス。D が (6,6) なら共通、(M,6,6) なら elem_mat で引く"""
    if D.ndim == 2:
        return D
    return D[elem_mat[sl]]


def _element_K(X, De, dN, wts):
    """要素剛性 (E, 3nn, 3nn) = Σ w detJ Bᵀ D B"""
    Ke = None
    for g, w in enumerate(wts):
        dNdx, detJ = _dN_dx(X, dN[g])
        if np.any(detJ <= 0):
            raise ValueError("つぶれた（体積が0以下の）要素があります。メッシュを見直してください。")
        B = _B_matrix(dNdx)
        BtD = np.matmul(B.transpose(0, 2, 1), De)            # (E,3nn,6)
        k = np.matmul(BtD * (w * detJ)[:, None, None], B)    # (E,3nn,3nn)
        Ke = k if Ke is None else Ke + k
    return Ke


def assemble_stiffness(mesh: Mesh, D: np.ndarray, chunk: int = 4000,
                       progress=None, elem_mat=None) -> sp.csr_matrix:
    """全体剛性マトリクス。要素をブロックに分け、ブロックごとに並列（使用コア数）で組み立てて足し合わせる。

    ブロックごとに疎行列にしてから足すので、大きなモデルでもメモリを使いすぎない。
    """
    from .parallel import chunks, pmap
    nn = mesh.elems.shape[1]
    ndof = 3 * mesh.n_nodes
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    ne = mesh.n_elems
    nloc = 3 * nn
    blocks = chunks(ne, min_size=1000)
    blocks = [(a + i, min(b, a + i + chunk)) for a, b in blocks for i in range(0, b - a, chunk)]
    done = [0]
    loc = np.arange(3)

    def work(ab):
        a, b = ab
        el = mesh.elems[a:b]
        De = _D_of(D, elem_mat, slice(a, b))
        if De.ndim == 2:
            De = np.broadcast_to(De, (b - a, 6, 6))
        Ke = _element_K(mesh.nodes[el], De, dN, wts)
        dofs = (3 * el[:, :, None] + loc).reshape(b - a, nloc).astype(np.int32)
        rows = np.repeat(dofs, nloc, axis=1).ravel()
        cols = np.tile(dofs, (1, nloc)).ravel()
        Kb = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
        done[0] += b - a
        if progress:
            progress(min(1.0, done[0] / ne))
        return Kb

    parts = pmap(work, blocks)
    # 2つずつ足し合わせる（並列）
    while len(parts) > 1:
        pairs = [parts[i:i + 2] for i in range(0, len(parts), 2)]
        parts = pmap(lambda pr: pr[0] + pr[1] if len(pr) == 2 else pr[0], pairs)
    K = parts[0]
    K.sum_duplicates()
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
    bodies: list = field(default_factory=list)        # 熱条件などボディに与える条件（空 = すべてのボディ）

    def to_dict(self):
        d = {"kind": self.kind, "faces": list(map(int, self.faces)),
             "values": list(self.values), "name": self.name}
        if self.nodes:
            d["nodes"] = list(map(int, self.nodes))
        if self.step_factors:
            d["step_factors"] = [float(x) for x in self.step_factors]
        if self.bodies:
            d["bodies"] = [int(b) for b in self.bodies]
        return d

    @staticmethod
    def from_dict(d):
        return BC(d["kind"], d.get("faces", []), d.get("values", [0, 0, 0]), d.get("name", ""),
                  d.get("nodes", []), d.get("step_factors", []), d.get("bodies", []))

    def label(self) -> str:
        jp = {"symmetry": "対称領域", "frictionless": "摩擦なし支持", "fixed": "固定支持", "displacement": "強制変位", "force": "力",
              "pressure": "圧力", "moment": "モーメント", "gravity": "重力加速度", "thermal": "熱条件"}
        name = self.name or jp.get(self.kind, self.kind)
        unit = {"force": "N", "pressure": "MPa", "moment": "N·mm",
                "gravity": "mm/s²", "displacement": "mm", "thermal": "°C"}.get(self.kind, "")
        v = ""
        if self.kind != "fixed":
            v = " [" + ", ".join("自由" if x is None else f"{x:g}" for x in self.values) + f"] {unit}"
        f = f" 面{','.join(str(i + 1) for i in self.faces)}" if self.faces else ""
        return f"{name}{f}{v}"


def symmetry_axis(mesh: Mesh, faces) -> int:
    """対称面（座標軸に垂直な平面）の法線の軸 0/1/2。面ごとに求め、すべて同じ軸であること"""
    info = {f["id"]: f for f in mesh.face_info()}
    axes = set()
    for f in faces:
        n = np.asarray(info[int(f)]["normal"], float)
        if not info[int(f)].get("flat", False) or np.max(np.abs(n)) < 0.999:
            raise ValueError("対称領域・摩擦なし支持は、座標軸に垂直な平らな面に設定してください"
                             f"（面{int(f) + 1} は法線 ({n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f})）")
        axes.add(int(np.argmax(np.abs(n))))
    return axes


def build_load_and_constraints(mesh: Mesh, material: Material, bcs: list[BC], rho_e=None):
    ndof = 3 * mesh.n_nodes
    F = np.zeros(ndof)
    fixed = {}  # dof -> value
    for bc in bcs:
        if bc.kind in ("symmetry", "frictionless"):
            # 対称領域 / 摩擦なし支持: 面に垂直な方向の変位 = 0（面に沿っては自由）
            for f in bc.faces:
                ax = symmetry_axis(mesh, [f]).pop()
                for d in 3 * mesh.face_nodes(f) + ax:
                    fixed[int(d)] = 0.0
            continue
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
        elif bc.kind == "thermal":
            pass   # 熱条件は材料（温度での物性と熱ひずみ）として扱う
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
        out.update(getattr(self, "extra", None) or {})
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


def nodal_stress(mesh: Mesh, u: np.ndarray, D: np.ndarray, chunk: int = 50000, elem_mat=None, eps_th_m=None):
    """各要素の節点位置で応力を評価し、節点で平均化する（Ansys の節点平均と同じ）。"""
    from .parallel import chunks, pmap
    nn = mesh.elems.shape[1]
    Lnodes = NODE_L10 if nn == 10 else NODE_L4
    dNa = [shape_dN_dxi(Lnodes[a], nn) for a in range(nn)]
    U = u.reshape(-1, 3)
    ne = mesh.n_elems
    eps_all = np.empty((ne, nn, 6))
    sig_all = np.empty((ne, nn, 6))

    def work(ab):
        a0, b0 = ab
        el = mesh.elems[a0:b0]
        X = mesh.nodes[el]
        ue = U[el].reshape(len(el), -1)
        De = _D_of(D, elem_mat, slice(a0, b0))
        for a in range(nn):
            dNdx, _ = _dN_dx(X, dNa[a])
            eps = np.einsum("eij,ej->ei", _B_matrix(dNdx), ue)
            eps_all[a0:b0, a] = eps
            if eps_th_m is not None:
                eps = eps - eps_th_m[elem_mat[a0:b0] if elem_mat is not None else 0]
            sig_all[a0:b0, a] = eps @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps)

    pmap(work, chunks(ne, min_size=2000))
    idx = mesh.elems.ravel()
    cnt = np.maximum(np.bincount(idx, minlength=mesh.n_nodes), 1)[:, None]
    acc_s = np.column_stack([np.bincount(idx, sig_all[..., k].ravel(), mesh.n_nodes) for k in range(6)])
    acc_e = np.column_stack([np.bincount(idx, eps_all[..., k].ravel(), mesh.n_nodes) for k in range(6)])
    return acc_s / cnt, acc_e / cnt


def solve(mesh: Mesh, material: Material, bcs: list[BC], progress=None, log=print,
          method: str = "auto", elem_materials=None, n_cores: int | None = None) -> Result:
    """線形静解析を実行する。progress(0..1, メッセージ) を渡すと進捗を通知。

    n_cores: 使用コア数（None = 現在の設定, 0 = すべて）
    """
    from .parallel import get_threads, set_threads
    if n_cores is not None:
        set_threads(n_cores)
    t0 = time.time()

    def prog(x, msg):
        if progress:
            progress(x, msg)

    if not any(b.kind in ("fixed", "displacement", "symmetry", "frictionless") for b in bcs):
        raise ValueError("拘束（固定支持 または 強制変位）がありません。少なくとも1つ設定してください。")
    mats, elem_mat, rho_e = None, None, None
    if elem_materials is not None:
        # elem_materials = (材料リスト, 要素ごとの番号)  ボディごとに材料が違う場合
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
        D = np.stack([material_D(m) for m in mats])
        rho_e = np.array([m.density_t_mm3 for m in mats])[elem_mat]
        material = mats[0]
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
        D = material_D(material)
    eps_th_m = np.stack([material_eps_th(m) for m in mats])
    if not np.any(eps_th_m):
        eps_th_m = None
    prog(0.02, "剛性マトリクスを組み立て中…")
    K = assemble_stiffness(mesh, D, progress=lambda x: prog(0.02 + 0.4 * x, "剛性マトリクスを組み立て中…"),
                           elem_mat=elem_mat)
    t_asm = time.time() - t0
    F, fixed = build_load_and_constraints(mesh, material, bcs, rho_e=rho_e)
    if eps_th_m is not None:
        F = F + thermal_load_vector(mesh, D, elem_mat, eps_th_m)
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
    stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat, eps_th_m=eps_th_m)
    R = K @ u - F
    reactions = {}
    for i, bc in enumerate(bcs):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, dtype=np.int64)
            reactions[i] = R.reshape(-1, 3)[nodes].sum(0)
    prog(1.0, "完了")
    info = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "要素タイプ": "Tet10（2次）" if mesh.order == 2 else "Tet4（1次）",
            "使用コア数": get_threads(), "ソルバー": choose_method(int(free.sum()), method),
            "組立時間 [s]": round(t_asm, 2), "求解時間 [s]": round(t_sol, 2),
            "合計時間 [s]": round(time.time() - t0, 2), "相対残差": float(res)}
    return Result(mesh, material, u.reshape(-1, 3), stress, strain, reactions, info, mats, elem_mat)


def _cg(A, b, M, tol=1e-10, maxiter=5000):
    try:
        return spla.cg(A, b, M=M, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:  # 古い SciPy
        return spla.cg(A, b, M=M, tol=tol, maxiter=maxiter)


def _pcg(A, b, M, tol=1e-10, maxiter=3000):
    """前処理付き共役勾配法（行列×ベクトル積を使用コア数で並列化）"""
    from .parallel import ParallelMatvec
    mv = ParallelMatvec(A)
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.zeros_like(b), 0, 0
    x = np.zeros_like(b)
    r = b.copy()
    z = M(r)
    p = z.copy()
    rz = r @ z
    for it in range(1, maxiter + 1):
        Ap = mv(p)
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) < tol * nb:
            return x, 0, it
        z = M(r)
        rz_new = r @ z
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, 1, maxiter


def _rel_res(A, x, b):
    return np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)


def _rigid_modes(coords, n):
    """剛体モード（並進3＋回転3）。AMG の収束を大幅に速くする"""
    x, comp = coords[:, :3], coords[:, 3].astype(int)
    x = x - x.mean(axis=0)
    B = np.zeros((n, 6))
    B[np.arange(n), comp] = 1.0
    for k, (a, c) in enumerate([(1, 2), (2, 0), (0, 1)]):
        B[comp == a, 3 + k] = -x[comp == a, c]
        B[comp == c, 3 + k] = x[comp == c, a]
    return B


def _direct_backend():
    """利用できる高速な直接法（スパースコレスキー）"""
    try:
        import pypardiso  # noqa: F401
        return "pardiso"
    except ImportError:
        pass
    try:
        from sksparse.cholmod import cholesky  # noqa: F401
        return "cholmod"
    except ImportError:
        pass
    # cholespy（CHOLMOD の簡易版）は Apple Silicon で SuperLU より遅かったため自動選択しない
    return "superlu"


def _amg_available():
    try:
        import pyamg  # noqa: F401
        return True
    except ImportError:
        return False


# 自動選択のしきい値（自由度）: これより大きいと反復法（AMG-CG）。Mac での実測で決定
AUTO_ITERATIVE_DOF = {"pardiso": 400_000, "cholmod": 400_000, "cholespy": 15_000, "superlu": 15_000}


def choose_method(n: int, method: str = "auto") -> str:
    method = {"scipy": "direct", "direct": "direct", None: "auto"}.get(method, method)
    if method in ("pardiso", "cholmod", "cholespy", "superlu", "amg"):
        return method
    d = _direct_backend()
    if method == "direct":
        return d
    if n > AUTO_ITERATIVE_DOF[d] and _amg_available():
        return "amg"
    return d


def _linear_solve(A: sp.csc_matrix, b: np.ndarray, log=print, coords=None,
                  method: str = "auto") -> np.ndarray:
    """連立方程式 A x = b（A は対称正定値）を解く。

    method: "auto"（自由度と利用できるライブラリから自動選択）/ "direct"（直接法）/ "amg"（反復法）
            または "pardiso" / "cholmod" / "cholespy" / "superlu"
    直接法: PARDISO (Intel MKL) → CHOLMOD (scikit-sparse / cholespy) → SciPy SuperLU
    反復法: AMG 前処理付き共役勾配法（相対残差 1e-10 まで → 直接法と同じ精度）
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    m = choose_method(n, method)
    if m == "amg":
        try:
            import pyamg  # noqa: F401
            from .amg import build as build_amg
            log("ソルバー: 反復法（AMG 前処理付き共役勾配法・並列チェビシェフ平滑化）")
            B = _rigid_modes(coords, n) if coords is not None else None
            M = build_amg(A, B, comps=coords[:, 3].astype(int) if coords is not None else None)
            xsol, info, it = _pcg(A.tocsr(), b, M, tol=1e-10)
            if info == 0 and _rel_res(A, xsol, b) < 1e-8:
                log(f"  反復回数 {it}")
                return xsol
            log("反復法が収束しなかったため直接法に切り替えます")
        except ImportError:
            pass
        m = _direct_backend()
    if m == "pardiso":
        try:
            import pypardiso
            log("ソルバー: 直接法（PARDISO）")
            return pypardiso.spsolve(A.tocsr(), b)
        except ImportError:
            m = "cholmod"
    if m == "cholmod":
        try:
            from sksparse.cholmod import cholesky
            log("ソルバー: 直接法（CHOLMOD）")
            try:
                return cholesky(A.tocsc())(b)
            except Exception as e:  # 正定値でない＝拘束不足
                log(f"CHOLMOD 失敗: {e}")
                return np.full(n, np.nan)
        except ImportError:
            m = "cholespy"
    if m == "cholespy":
        try:
            from cholespy import CholeskySolverD, MatrixType
            log("ソルバー: 直接法（CHOLMOD / cholespy）")
            C = A.tocoo()
            try:
                solver = CholeskySolverD(n, C.row.astype(np.int32), C.col.astype(np.int32),
                                         C.data.astype(np.float64), MatrixType.COO)
                x = np.zeros(n)
                solver.solve(np.ascontiguousarray(b, dtype=np.float64), x)
                return x
            except Exception as e:
                log(f"cholespy 失敗: {e}")
                return np.full(n, np.nan)
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

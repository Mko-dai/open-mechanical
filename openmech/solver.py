"""線形静的構造解析ソルバー（3次元、等方弾性、Tet4 / Tet10 / 三角柱 Wedge6 / Wedge15）。

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
    if nn in (6, 15):
        from .elements import wedge_N
        return wedge_N(L, nn)
    if nn == 4:
        return L.copy()
    N = np.empty(10)
    N[:4] = L * (2 * L - 1)
    for k, (i, j) in enumerate(_EDGES10):
        N[4 + k] = 4 * L[i] * L[j]
    return N


def shape_dN_dxi(L: np.ndarray, nn: int) -> np.ndarray:
    """(nn,3) 自然座標での形状関数の微分。四面体は L が体積座標 (4,)、三角柱は (r,s,t)"""
    if nn in (6, 15):
        from .elements import wedge_dN
        return wedge_dN(L, nn)
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
    if nn in (6, 15):
        from .elements import wedge_quadrature
        return wedge_quadrature(nn)
    return (GAUSS4, GAUSS4_W) if nn == 10 else (GAUSS1, GAUSS1_W)


def _quadrature_full(nn):
    """質量・体積力用（2 次の形状関数の積まで十分な精度）"""
    if nn in (6, 15):
        from .elements import wedge_quadrature
        return wedge_quadrature(nn, full=True)
    return GAUSS4, GAUSS4_W


def node_points(nn):
    """要素の各節点の自然座標（節点での応力評価用）"""
    if nn in (6, 15):
        from .elements import wedge_node_points
        return wedge_node_points(nn)
    return NODE_L10 if nn == 10 else NODE_L4


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


def stiffness_pattern(elems: np.ndarray, n_nodes: int):
    """節点どうしの隣接（疎行列のパターン）。返り値 (indptr, indices)（各行の列は昇順）"""
    ne, nn = elems.shape
    C = sp.csr_matrix((np.ones(ne * nn, np.float32), elems.ravel().astype(np.int64),
                       np.arange(0, ne * nn + 1, nn)), shape=(ne, n_nodes))
    G = (C.T @ C).tocsr()                       # 要素を共有する節点の組
    G.sort_indices()
    return G.indptr.astype(np.int64), G.indices.astype(np.int64)


def assemble_stiffness(mesh: Mesh, D: np.ndarray, chunk: int = 4000,
                       progress=None, elem_mat=None) -> sp.csr_matrix:
    """全体剛性マトリクス（CSR）。

    先に節点の隣接から行列のパターン（非ゼロの位置）を作り、要素剛性をその位置に直接足し込む。
    要素剛性の計算は使用コア数で並列。途中で大きな COO 行列を作らないので、
    100 万節点級でもメモリは最終的な行列 + α で済む。
    """
    from . import parallel as _par
    from .parallel import pmap
    nn = mesh.elems.shape[1]
    N = mesh.n_nodes
    ndof = 3 * N
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    ne = mesh.n_elems
    ip, ix = stiffness_pattern(mesh.elems, N)
    nnzb = len(ix)
    rowlen = np.diff(ip)
    keys = np.repeat(np.arange(N, dtype=np.int64), rowlen) * N + ix      # 昇順
    # 自由度の CSR: 行 3r+i の並びは「列の節点 k 順 × 成分 j」
    indptr = np.zeros(ndof + 1, np.int64)
    indptr[1:] = np.cumsum(np.repeat(3 * rowlen, 3))
    big = 9 * nnzb >= 2**31 - 1
    itype = np.int64 if big else np.int32
    indices = np.empty(9 * nnzb, itype)
    data = np.zeros(9 * nnzb)
    prow = np.repeat(np.arange(N, dtype=np.int64), rowlen)
    base = 6 * ip[prow] + 3 * np.arange(nnzb, dtype=np.int64)          # = 9 ip[r] + 3 (p - ip[r])
    lr3 = 3 * rowlen[prow]
    for i in range(3):
        for j in range(3):
            indices[base + i * lr3 + j] = 3 * ix + j
    del prow, base, lr3
    blocks = [(a, min(ne, a + chunk)) for a in range(0, ne, chunk)]
    wave = max(1, int(_par._N))

    def work(ab):
        a, b = ab
        el = mesh.elems[a:b].astype(np.int64)
        De = _D_of(D, elem_mat, slice(a, b))
        if De.ndim == 2:
            De = np.broadcast_to(De, (b - a, 6, 6))
        Ke = _element_K(mesh.nodes[el], De, dN, wts).reshape(b - a, nn, 3, nn, 3)
        r = el[:, :, None]                                         # (B,nn,1)
        c = el[:, None, :]                                         # (B,1,nn)
        p = np.searchsorted(keys, (r * N + c).ravel()).reshape(b - a, nn, nn)
        pr = ip[r]                                                 # (B,nn,1)
        b0 = 6 * pr + 3 * p                                        # (B,nn,nn)
        l3 = 3 * (ip[r + 1] - pr)
        # 位置 (B, a, i, b, j) = b0[a,b] + i*l3[a] + j
        pos = (b0[:, :, None, :, None] + np.arange(3)[None, None, :, None, None] * l3[:, :, :, None, None]
               + np.arange(3)[None, None, None, None, :])
        return pos.ravel(), Ke.ravel()

    done = 0
    for w in range(0, len(blocks), wave):
        for pos, val in pmap(work, blocks[w:w + wave]):
            np.add.at(data, pos, val)
            done += len(val) // (9 * nn * nn)
        if progress:
            progress(min(1.0, done / ne))
    return sp.csr_matrix((data, indices, indptr.astype(itype)), shape=(ndof, ndof))


def body_force_vector(mesh: Mesh, b: np.ndarray) -> np.ndarray:
    """体積力 b [N/mm^3]（= 密度×加速度）の等価節点力。 b は (3,) または要素ごと (E,3)"""
    b = np.asarray(b, float)
    if b.ndim == 1:
        b = np.broadcast_to(b, (mesh.n_elems, 3))
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature_full(nn)  # 2次の形状関数まで厳密に積分できる
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


def face_load_weights(mesh: Mesh, faces):
    """面の上の一様な表面力を節点に配る重み。

    返り値: (節点番号 (K,), ∫N dA (K,), ∫N n dA (K,3), 面積の合計)
    四面体の面（Tri3/Tri6）に加え、三角柱の側面（Quad4/Quad8）も正しく積分する。
    """
    rows = np.concatenate([np.nonzero(mesh.face_ids == int(f))[0] for f in faces]) if len(faces) else np.zeros(0, int)
    qof = mesh.boundary_quad_of if getattr(mesh, "is_wedge", False) else None
    qof = qof[rows] if qof is not None else np.full(len(rows), -1)
    tri_rows = rows[qof < 0]
    tris = mesh.boundary_tris_full[tri_rows]
    tri, w, n, A = surface_weights(mesh, tris)
    idx = [tri.ravel()]
    ws = [w.ravel()]
    wn = [(w[:, :, None] * n[:, None, :]).reshape(-1, 3)]
    Atot = float(A.sum())
    if (qof >= 0).any():
        from .elements import quad_load_weights
        quads = mesh.boundary_quads_full[np.unique(qof[qof >= 0])]
        qi, qw, qwn, qa = quad_load_weights(mesh.nodes, quads)
        idx.append(qi.ravel())
        ws.append(qw.ravel())
        wn.append(qwn.reshape(-1, 3))
        Atot += float(qa.sum())
    return np.concatenate(idx), np.concatenate(ws), np.concatenate(wn), Atot


# ---------------------------------------------------------------------------- 境界条件

# 追加の境界条件（remote.py / thermal.py / electric.py で処理）
BC_EXTRA_JP = {"remote_force": "リモート荷重", "remote_displacement": "リモート変位", "point_mass": "点質量",
               "elastic_support": "弾性支持", "cylindrical_support": "円筒支持", "bearing_load": "ベアリング荷重",
               "bolt_pretension": "ボルト予張力", "spring": "ばね（ボディ－地面）", "compression_only": "圧縮のみ支持",
               "temperature": "温度", "heat_flux": "熱流束", "heat_flow": "熱流量", "convection": "対流",
               "radiation": "放射", "heat_generation": "内部発熱", "insulation": "断熱",
               "voltage": "電圧", "charge": "電荷", "ground": "接地"}
SUPPORT_KINDS = ("fixed", "displacement", "symmetry", "frictionless", "remote_displacement", "elastic_support",
                 "cylindrical_support", "spring")


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
    exprs: dict = field(default_factory=dict)         # 成分ごとの式 {成分番号: "F/2"}（values には評価値）
    point: list = field(default_factory=list)         # リモートポイント・ばねの位置 [x,y,z]（空 = 面の重心）
    options: dict = field(default_factory=dict)       # 挙動（rigid/deformable）・軸・ボディなど種類ごとの設定

    def to_dict(self):
        d = {"kind": self.kind, "faces": list(map(int, self.faces)),
             "values": list(self.values), "name": self.name}
        if self.nodes:
            d["nodes"] = list(map(int, self.nodes))
        if self.step_factors:
            d["step_factors"] = [float(x) for x in self.step_factors]
        if self.bodies:
            d["bodies"] = [int(b) for b in self.bodies]
        if self.exprs:
            d["exprs"] = {str(k): str(v) for k, v in self.exprs.items()}
        if self.point:
            d["point"] = [float(x) for x in self.point]
        if self.options:
            d["options"] = dict(self.options)
        return d

    @staticmethod
    def from_dict(d):
        return BC(d["kind"], d.get("faces", []), d.get("values", [0, 0, 0]), d.get("name", ""),
                  d.get("nodes", []), d.get("step_factors", []), d.get("bodies", []),
                  {int(k): v for k, v in d.get("exprs", {}).items()}, list(d.get("point", [])),
                  dict(d.get("options", {})))

    def label(self) -> str:
        jp = {"symmetry": "対称領域", "frictionless": "摩擦なし支持", "fixed": "固定支持", "displacement": "強制変位", "force": "力",
              "pressure": "圧力", "moment": "モーメント", "gravity": "重力加速度", "thermal": "熱条件",
              **{k: v for k, v in BC_EXTRA_JP.items()}}
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
            nodes, w, _, Atot = face_load_weights(mesh, bc.faces)
            t = np.asarray(bc.values, float) / Atot
            for c in range(3):
                np.add.at(F, 3 * nodes + c, w * t[c])
        elif bc.kind == "pressure":
            p = float(bc.values[0])
            nodes, _, wvec, _ = face_load_weights(mesh, bc.faces)
            for c in range(3):
                np.add.at(F, 3 * nodes + c, -p * wvec[:, c])
        elif bc.kind == "moment":
            nodes, w, _, _ = face_load_weights(mesh, bc.faces)
            wn = np.bincount(nodes, weights=w, minlength=mesh.n_nodes)
            idx = np.nonzero(wn)[0]
            wi = wn[idx]
            r = mesh.nodes[idx] - (mesh.nodes[idx] * wi[:, None]).sum(0) / wi.sum()
            J = (wi[:, None, None] * (np.einsum("ij,ij->i", r, r)[:, None, None] * np.eye(3)
                                      - r[:, :, None] * r[:, None, :])).sum(0)
            omega = np.linalg.lstsq(J, np.asarray(bc.values, float), rcond=None)[0]
            f = wi[:, None] * np.cross(omega, r)
            for c in range(3):
                F[3 * idx + c] += f[:, c]
        elif bc.kind == "thermal" or bc.kind in BC_EXTRA_JP:
            pass   # 熱条件は材料として、リモート・ばね・円筒支持などは remote.py で扱う
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
    Lnodes = node_points(nn)
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

    if not any(b.kind in SUPPORT_KINDS for b in bcs):
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
    from .remote import REMOTE_KINDS, extra_terms, pad
    ex = None
    if any(b.kind in REMOTE_KINDS for b in bcs):
        acc = next((np.asarray(b.values, float) for b in bcs if b.kind == "gravity"), None)
        ex = extra_terms(mesh, bcs, float(np.mean(np.abs(K.diagonal()))), acc=acc)
        for note in ex.notes:
            log(note)
        K = pad(K, ex.n) + ex.K()
        F = pad(F, ex.n) + ex.Fvec()
        fixed.update(ex.fixed)
        if (ex.direct or ex.mpc) and method == "auto":
            method = "direct"
    T_mpc = None
    if ex is not None and ex.mpc:
        from .remote import reduce_system
        K_full, F_full = K, F
        K, F, fixed, T_mpc = reduce_system(K, F, fixed, ex, float(np.mean(np.abs(K.diagonal()[:3 * mesh.n_nodes]))))
    ndof = K.shape[0]
    cdofs = np.fromiter(fixed.keys(), dtype=np.int64)
    cvals = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(ndof, bool)
    free[cdofs] = False
    u = np.zeros(ndof)
    u[cdofs] = cvals
    Kfr = K[free]                       # 行の取り出し（CSR のまま。コピーは 1 回だけ）
    rhs = F[free] - Kfr[:, cdofs] @ cvals
    Kff = Kfr[:, free]
    Kc = K[cdofs]                       # 反力用（拘束自由度の行だけ残す）
    del Kfr, K                          # 全体行列は手放す（大規模モデルのメモリ節約）
    prog(0.45, f"連立方程式を解いています（自由度 {Kff.shape[0]:,}）…")
    t1 = time.time()
    dof_ids = np.nonzero(free)[0]
    nb = 3 * mesh.n_nodes
    dn = np.minimum(dof_ids, nb - 1) if T_mpc is None else np.zeros(len(dof_ids), dtype=np.int64)
    coords = np.column_stack([mesh.nodes[dn // 3], dn % 3])
    sl = mesh.sweep_lines() if getattr(mesh, "is_wedge", False) else None
    lines = sl[dn // 3] if (sl is not None and ex is None) else None
    with np.errstate(all="ignore"):
        if ex is not None and ex.direct:
            # ペナルティ（剛体結合・円筒支持）で条件数が大きいので、一度分解して反復改良で精度を上げる
            fac = factorize(Kff.tocsc(), log)
            uf = fac(rhs)
            for _ in range(5):
                rr = rhs - Kff @ uf
                if np.linalg.norm(rr) <= 1e-12 * max(np.linalg.norm(rhs), 1e-300):
                    break
                uf = uf + fac(rr)
        else:
            uf = _linear_solve(Kff, rhs, log, coords=coords, method=method, lines=lines)
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
    R = np.zeros(ndof)                  # 拘束していない自由度は釣り合い（≈0）
    R[cdofs] = Kc @ u - F[cdofs]
    if T_mpc is not None:
        # 反力は縮約した式の残差（独立自由度）を元の番号に戻す
        R_r = R
        u = T_mpc @ u
        R = np.zeros(K_full.shape[0])
        R[T_mpc.keep] = R_r
    u_pilot = u[nb:]
    u = u[:nb]
    stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat, eps_th_m=eps_th_m)
    reactions = {}
    for i, bc in enumerate(bcs):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, dtype=np.int64)
            reactions[i] = R[:nb].reshape(-1, 3)[nodes].sum(0)
    reaction_moments = {}
    if ex is not None:
        for i, dofs in ex.reaction_dofs.items():
            reactions[i] = R[dofs][:3].copy()
            reaction_moments[i] = R[dofs][3:].copy()
    prog(1.0, "完了")
    info = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "要素タイプ": "Tet10（2次）" if mesh.order == 2 else "Tet4（1次）",
            "使用コア数": get_threads(), "ソルバー": choose_method(int(free.sum()), method),
            "組立時間 [s]": round(t_asm, 2), "求解時間 [s]": round(t_sol, 2),
            "合計時間 [s]": round(time.time() - t0, 2), "相対残差": float(res)}
    r = Result(mesh, material, u.reshape(-1, 3), stress, strain, reactions, info, mats, elem_mat)
    r.reaction_moments = reaction_moments
    if ex is not None and ex.pilots:
        r.pilots = {i: (p, u_pilot[d0 - nb:d0 - nb + 6]) for i, p, d0 in ex.pilots}
    return r


def _cg(A, b, M, tol=1e-10, maxiter=5000):
    try:
        return spla.cg(A, b, M=M, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:  # 古い SciPy
        return spla.cg(A, b, M=M, tol=tol, maxiter=maxiter)


def _pcg(A, b, M, tol=1e-10, maxiter=3000, x0=None, hist=None):
    """前処理付き共役勾配法（行列×ベクトル積を使用コア数で並列化）。hist に相対残差の履歴を追記"""
    from .parallel import ParallelMatvec
    mv = ParallelMatvec(A)
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.zeros_like(b), 0, 0
    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - mv(x)
    z = M(r)
    p = z.copy()
    rz = r @ z
    for it in range(1, maxiter + 1):
        Ap = mv(p)
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rn = np.linalg.norm(r)
        if hist is not None:
            hist.append(rn / nb)
        if rn < tol * nb:
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


def factorize(A, log=print):
    """対称正定値（または対称）行列 A を一度だけ分解し、x = A⁻¹ b を返す関数を作る（固有値解析のシフト逆反復用）"""
    n = A.shape[0]
    d = _direct_backend()
    if d == "pardiso":
        try:
            import pypardiso
            solver = pypardiso.PyPardisoSolver()
            Ac = A.tocsr()
            solver.factorize(Ac)
            log("  分解: PARDISO")
            return lambda b: solver.solve(Ac, b)
        except Exception:
            pass
    if d == "cholmod":
        try:
            from sksparse.cholmod import cholesky
            f = cholesky(A.tocsc())
            log("  分解: CHOLMOD")
            return f
        except Exception:
            pass
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    p = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=True)
    Ap = A[p][:, p].tocsc()
    lu = spla.splu(Ap, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    log("  分解: SciPy SuperLU")

    def sol(b):
        b = np.asarray(b)
        if b.ndim == 1:
            x = np.empty(n)
            x[p] = lu.solve(b[p])
        else:
            x = np.empty_like(b, dtype=float)
            x[p] = lu.solve(np.ascontiguousarray(b[p]))
        return x
    return sol


def _linear_solve(A: sp.csc_matrix, b: np.ndarray, log=print, coords=None,
                  method: str = "auto", lines=None) -> np.ndarray:
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
            log("ソルバー: 反復法（AMG 前処理付き共役勾配法・並列チェビシェフ平滑化"
                + ("・厚さ方向の線平滑化" if lines is not None else "") + "）")
            B = _rigid_modes(coords, n) if coords is not None else None
            t0 = time.time()
            import os as _os
            # 三角柱の積層メッシュ（線平滑化あり）は最初から平滑化集約: 積層板の実測で反復 1984 → 94 回、時間 1/7
            robust0 = _os.environ.get("OPENMECH_AMG", "") == "robust" or (
                lines is not None and _os.environ.get("OPENMECH_AMG", "") != "fast")
            M = build_amg(A, B, comps=coords[:, 3].astype(int) if coords is not None else None, lines=lines,
                          robust=robust0)
            t1 = time.time()
            log(f"  AMG 準備 {t1 - t0:.1f} s / 階層 " + " → ".join(f"{L.A.shape[0]:,}" for L in M.levels)
                + f" → {M.coarse_n:,}" + (f" / 線ブロック 非ゼロ {M.line_nnz / 1e6:.0f} 百万" if lines is not None else ""))
            hist = []
            xsol, info, it = _pcg(A.tocsr(), b, M, tol=1e-10, maxiter=400, hist=hist)
            if info != 0 and len(hist) > 150 and hist[-1] < hist[-150]:
                # 遅いが確実に下がっているなら、残りの回数を見積もって続ける（頑健 AMG の準備より速いことが多い）
                rate = (hist[-1] / hist[-150]) ** (1 / 150)
                need = int(np.log(1e-10 / hist[-1]) / np.log(rate)) + 20
                if need < 4000:
                    log(f"  収束が遅いため続けます（あと約 {need} 回の見込み）")
                    xsol, info, it2 = _pcg(A.tocsr(), b, M, tol=1e-10, maxiter=need + 400, x0=xsol, hist=hist)
                    it += it2
            t2 = time.time()
            rr = _rel_res(A, xsol, b)
            if rr < 1e-8 or (info == 0 and rr < 1e-6):   # 条件数が大きいと真の残差は 1e-8 まで下がらないことがある
                log(f"  反復回数 {it}（AMG 準備 {t1 - t0:.1f} s / 反復 {t2 - t1:.1f} s / 階層 "
                    + " → ".join(f"{L.A.shape[0]:,}" for L in M.levels) + f" → {M.coarse_n:,} / 相対残差 {rr:.1e}）")
                return xsol
            # 収束が遅い（剛性差の大きい材料・薄い層など）→ 頑健な AMG に作り直して続きから解く
            log(f"  {it} 回で収束しないため、頑健な AMG（平滑化集約）に切り替えます")
            del M
            M = build_amg(A, B, comps=coords[:, 3].astype(int) if coords is not None else None, robust=True,
                          lines=lines)
            t3 = time.time()
            xsol, info, it2 = _pcg(A.tocsr(), b, M, tol=1e-10, maxiter=3000, x0=xsol)
            rr = _rel_res(A, xsol, b)
            if rr < 1e-6:
                log(f"  反復回数 {it} + {it2}（頑健 AMG 準備 {t3 - t2:.1f} s / 反復 {time.time() - t3:.1f} s / "
                    f"相対残差 {rr:.1e}）")
                return xsol
            if n > 300_000:
                log(f"⚠ 反復法の相対残差が {rr:.1e} までしか下がりませんでした。自由度が大きく直接法はメモリ不足になるため、"
                    "この解を使います（拘束不足・浮いた部品がないか確認してください）")
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

"""シェル要素（Ansys の SHELL181 に相当する薄肉構造用）— 3節点の平板シェル

  膜: CST（定ひずみ三角形）＋ ドリリング回転の安定化
  曲げ: DKT（離散キルヒホッフ三角形, Batoz 1980）— 薄板の曲げに高精度
  1節点 6自由度（ux, uy, uz, θx, θy, θz）。厚さはボディ（面）ごと。
  応力は上面（法線側, z=+t/2）・下面（z=−t/2）で求める（Ansys と同じく上面/下面を表示）。

面の上の境界条件:
  面番号 0..S-1 = 面（サーフェス）、S.. = 辺（エッジ）。辺の固定支持・単純支持・力、面の圧力・力・重力に対応。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh
from .solver import BC, Result, _linear_solve, _rel_res, material_D


class ShellMesh(Mesh):
    """3節点シェルのメッシュ。boundary_tris = 要素そのもの（表示・選択は面と同じ扱い）"""

    def __init__(self, nodes, tris, thickness, surf_ids, edges, elem_body=None, n_surf=None):
        self.nodes = np.asarray(nodes, float)
        self.elems = np.asarray(tris, np.int64)
        self.thickness = np.asarray(thickness, float)       # 要素ごとの厚さ [mm]
        self._btri = self.elems
        self._btri_full = self.elems
        self._btri_owner = np.arange(len(self.elems))
        self.n_surf = int(n_surf if n_surf is not None else (int(np.max(surf_ids)) + 1 if len(surf_ids) else 0))
        self.face_ids = np.asarray(surf_ids, int)
        self.edges = [np.asarray(e, np.int64) for e in edges]   # 辺ごとの線分 (k,2)
        self.elem_body = None if elem_body is None else np.asarray(elem_body, np.int64)
        self.body_names = None
        self.node_origin = np.arange(len(self.nodes))
        self.is_shell = True

    @property
    def order(self):
        return 1

    def element_volumes(self):
        n, a = self.tri_normals_areas(self.elems)
        return a * self.thickness

    def _compute_boundary(self):
        pass

    @property
    def n_faces(self):
        return self.n_surf + len(self.edges)

    def is_edge(self, fid):
        return int(fid) >= self.n_surf

    def face_info(self):
        n, a = self.tri_normals_areas(self.elems)
        c = self.nodes[self.elems].mean(axis=1)
        info = []
        for f in range(self.n_surf):
            m = self.face_ids == f
            A = a[m].sum()
            cen = (c[m] * a[m, None]).sum(0) / max(A, 1e-300)
            nn = (n[m] * a[m, None]).sum(0)
            nl = np.linalg.norm(nn)
            info.append({"id": f, "area": float(A), "centroid": cen,
                         "normal": nn / nl if nl > 1e-12 else np.zeros(3),
                         "flat": bool(nl / A > 0.999) if A > 0 else False, "edge": False,
                         "thickness": float(self.thickness[m].mean()) if m.any() else 0.0})
        for k, seg in enumerate(self.edges):
            p = self.nodes[seg]
            L = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
            cen = ((p[:, 0] + p[:, 1]) / 2 * L[:, None]).sum(0) / max(L.sum(), 1e-300)
            info.append({"id": self.n_surf + k, "area": float(L.sum()), "centroid": cen,
                         "normal": np.zeros(3), "flat": False, "edge": True})
        return info

    def face_tris(self, face_id, full=True):
        return self.elems[self.face_ids == int(face_id)]

    def face_nodes(self, face_id):
        out = []
        for f in np.atleast_1d(face_id):
            f = int(f)
            if f >= self.n_surf:
                out.append(self.edges[f - self.n_surf].ravel())
            else:
                out.append(self.elems[self.face_ids == f].ravel())
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def nearest_edge(self, point, tol=np.inf):
        """座標に最も近い辺の番号（面番号の続き）"""
        best, bid = np.inf, -1
        q = np.asarray(point, float)
        for k, seg in enumerate(self.edges):
            a, b = self.nodes[seg[:, 0]], self.nodes[seg[:, 1]]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
            d = np.linalg.norm(a + t[:, None] * ab - q, axis=1).min()
            if d < best:
                best, bid = d, k
        return self.n_surf + bid if best <= tol else -1

    def to_quadratic(self):
        return self

    def split_body_interfaces(self, share=None):
        return self


# ---------------------------------------------------------------------------- 要素

def _frames(X):
    """要素ごとの局所座標系 R (E,3,3: 行 = 局所 x, y, z 軸) と局所2次元座標 (E,3,2)、面積"""
    e1 = X[:, 1] - X[:, 0]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    A2 = np.linalg.norm(n, axis=1)
    e3 = n / A2[:, None]
    e2 = np.cross(e3, e1)
    R = np.stack([e1, e2, e3], axis=1)
    xy = np.einsum("eij,ekj->eki", R[:, :2], X - X[:, :1])
    return R, xy, A2 / 2


def _dkt_B(xy, xi, eta):
    """DKT の曲率 κ = B U（U = [w1, θx1, θy1, w2, ...]）、(E,3,9)"""
    x, y = xy[..., 0], xy[..., 1]
    # 辺 4:(2,3) 5:(3,1) 6:(1,2) （節点 0,1,2）
    xij = np.stack([x[:, 1] - x[:, 2], x[:, 2] - x[:, 0], x[:, 0] - x[:, 1]], 1)
    yij = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    l2 = xij ** 2 + yij ** 2
    a = -xij / l2
    b = 0.75 * xij * yij / l2
    c = (0.25 * xij ** 2 - 0.5 * yij ** 2) / l2
    d = -yij / l2
    e = (0.25 * yij ** 2 - 0.5 * xij ** 2) / l2
    a4, a5, a6 = a[:, 0], a[:, 1], a[:, 2]
    b4, b5, b6 = b[:, 0], b[:, 1], b[:, 2]
    c4, c5, c6 = c[:, 0], c[:, 1], c[:, 2]
    d4, d5, d6 = d[:, 0], d[:, 1], d[:, 2]
    e4, e5, e6 = e[:, 0], e[:, 1], e[:, 2]
    # 2次の形状関数の ξ, η 微分（N1..N6）
    L = 1 - xi - eta
    dN_dxi = np.array([-(4 * L - 1), 4 * xi - 1, 0.0, 4 * eta, -4 * eta, 4 * (L - xi)])
    dN_deta = np.array([-(4 * L - 1), 0.0, 4 * eta - 1, 4 * xi, 4 * (L - eta), -4 * xi])

    def H(dN):
        N1, N2, N3, N4, N5, N6 = [np.full(len(xy), v) for v in dN]
        Hx = np.stack([1.5 * (a6 * N6 - a5 * N5), b5 * N5 + b6 * N6, N1 - c5 * N5 - c6 * N6,
                       1.5 * (a4 * N4 - a6 * N6), b6 * N6 + b4 * N4, N2 - c6 * N6 - c4 * N4,
                       1.5 * (a5 * N5 - a4 * N4), b4 * N4 + b5 * N5, N3 - c4 * N4 - c5 * N5], 1)
        Hy = np.stack([1.5 * (d6 * N6 - d5 * N5), -N1 + e5 * N5 + e6 * N6, -b5 * N5 - b6 * N6,
                       1.5 * (d4 * N4 - d6 * N6), -N2 + e6 * N6 + e4 * N4, -b6 * N6 - b4 * N4,
                       1.5 * (d5 * N5 - d4 * N4), -N3 + e4 * N4 + e5 * N5, -b4 * N4 - b5 * N5], 1)
        return Hx, Hy

    Hx_xi, Hy_xi = H(dN_dxi)
    Hx_eta, Hy_eta = H(dN_deta)
    x31, y31 = x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]
    x12, y12 = x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]
    twoA = x31 * y12 - x12 * y31          # = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
    # Batoz の式: ∂/∂x = (y31 ∂/∂ξ + y12 ∂/∂η)/2A, ∂/∂y = (−x31 ∂/∂ξ − x12 ∂/∂η)/2A
    dx = lambda Hxi, Heta: (y31[:, None] * Hxi + y12[:, None] * Heta) / twoA[:, None]  # noqa: E731
    dy = lambda Hxi, Heta: (-x31[:, None] * Hxi - x12[:, None] * Heta) / twoA[:, None]  # noqa: E731
    B = np.stack([dx(Hx_xi, Hx_eta), dy(Hy_xi, Hy_eta), dy(Hx_xi, Hx_eta) + dx(Hy_xi, Hy_eta)], 1)
    return B


GP3 = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]


def _cst_B(xy, A):
    """CST の膜ひずみ B (E,3,6)（U = [u1, v1, u2, v2, u3, v3]）"""
    x, y = xy[..., 0], xy[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
    B = np.zeros((len(xy), 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B / (2 * A)[:, None, None]


def _allman_T(xy):
    """LST(12) ← Allman(9) の変換と ∂L/∂x, ∂L/∂y"""
    E = len(xy)
    x, y = xy[..., 0], xy[..., 1]
    T = np.zeros((E, 12, 9))
    for i in range(3):
        T[:, 2 * i, 3 * i] = 1
        T[:, 2 * i + 1, 3 * i + 1] = 1
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        dx, dy = x[:, j] - x[:, i], y[:, j] - y[:, i]
        m = 3 + k
        for q in (i, j):
            T[:, 2 * m, 3 * q] = 0.5
            T[:, 2 * m + 1, 3 * q + 1] = 0.5
        # u_mid += L/8 (θj − θi) n,  L·n = (dy, −dx)（反時計回り）
        T[:, 2 * m, 3 * i + 2] = -dy / 8
        T[:, 2 * m, 3 * j + 2] = dy / 8
        T[:, 2 * m + 1, 3 * i + 2] = dx / 8
        T[:, 2 * m + 1, 3 * j + 2] = -dx / 8
    return T


def _allman_B(xy, A, Lp):
    """膜ひずみ B (E,3,9)（体積座標 Lp の点で）"""
    E = len(xy)
    x, y = xy[..., 0], xy[..., 1]
    twoA = 2 * A
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / twoA[:, None]
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / twoA[:, None]
    L1, L2, L3 = Lp
    dNdL = np.array([[4 * L1 - 1, 0, 0], [0, 4 * L2 - 1, 0], [0, 0, 4 * L3 - 1],
                     [4 * L2, 4 * L1, 0], [0, 4 * L3, 4 * L2], [4 * L3, 0, 4 * L1]])
    dNx = np.einsum("ak,ek->ea", dNdL, b)
    dNy = np.einsum("ak,ek->ea", dNdL, c)
    B = np.zeros((E, 3, 12))
    B[:, 0, 0::2] = dNx
    B[:, 1, 1::2] = dNy
    B[:, 2, 0::2] = dNy
    B[:, 2, 1::2] = dNx
    return np.einsum("eki,eij->ekj", B, _allman_T(xy)), b, c


def _allman_K(xy, A, t, Dp, alpha_s=0.05):
    """Allman 型の膜要素（回転自由度つき, 9×9: [u1,v1,θz1, u2,v2,θz2, u3,v3,θz3]）

    LST（6節点の2次三角形）の中間節点の変位を、両端の変位と回転で表す:
      u_mid = (u_i + u_j)/2 + L_ij/8 (θ_j − θ_i) n_ij   （n_ij: 辺の外向き法線）
    CST より面内曲げがずっと正確になる。回転がそろって動くゼロエネルギーモードは小さなばねで抑える。
    """
    E = len(xy)
    K = np.zeros((E, 9, 9))
    for Lp in ([2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]):
        B, b, c = _allman_B(xy, A, Lp)
        K += np.einsum("e,eki,ekl,elj->eij", A * t / 3, B, Dp, B)
    # 安定化: 平均回転 = 連続体の回転 ω = (∂v/∂x − ∂u/∂y)/2（CST の補間で評価）
    g = np.zeros((E, 9))
    for i in range(3):
        g[:, 3 * i] = -0.5 * c[:, i]
        g[:, 3 * i + 1] = 0.5 * b[:, i]
        g[:, 3 * i + 2] = -1.0 / 3.0
    ks = alpha_s * Dp[:, 2, 2] * t * A
    K += ks[:, None, None] * np.einsum("ei,ej->eij", g, g)
    return K


def plane_stress_D(Dm3):
    """3次元の剛性 (6,6) → 平面応力 (3,3)（σz=τyz=τzx=0 で縮約、成分 xx, yy, xy）"""
    D = np.asarray(Dm3, float)
    ip, op = [0, 1, 3], [2, 4, 5]
    Daa = D[np.ix_(ip, ip)]
    Dab = D[np.ix_(ip, op)]
    Dbb = D[np.ix_(op, op)]
    return Daa - Dab @ np.linalg.solve(Dbb, Dab.T)


# 局所 18 自由度 [u,v,w,θx,θy,θz]×3 のうちの膜・曲げの位置
_MEM = np.array([0, 1, 6, 7, 12, 13])
_BEN = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
_DRL = np.array([5, 11, 17])
_MEMR = np.array([0, 1, 5, 6, 7, 11, 12, 13, 17])   # 膜 + 回転（Allman）


def shell_element_K(X, t, Dp, drill=1e-3):
    """要素剛性（全体座標, (E,18,18)）と 局所座標系 R、局所座標 xy、面積 A"""
    E = len(X)
    R, xy, A = _frames(X)
    Dd = np.broadcast_to(Dp, (E, 3, 3)) if Dp.ndim == 2 else Dp
    Km = _allman_K(xy, A, t, Dd)
    Db = (t ** 3 / 12)[:, None, None] * (np.broadcast_to(Dp, (E, 3, 3)) if Dp.ndim == 2 else Dp)
    Kb = np.zeros((E, 9, 9))
    for xi, eta in GP3:
        B = _dkt_B(xy, xi, eta)
        Kb += np.einsum("e,eki,ekl,elj->eij", A / 3, B, Db, B)
    Kl = np.zeros((E, 18, 18))
    Kl[:, _MEMR[:, None], _MEMR[None, :]] = Km
    Kl[:, _BEN[:, None], _BEN[None, :]] = Kb
    T = np.zeros((E, 18, 18))
    for k in range(6):
        T[:, 3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
    return np.einsum("eki,ekl,elj->eij", T, Kl, T), R, xy, A


def _shell_dofs(el):
    return (6 * el[:, :, None] + np.arange(6)).reshape(len(el), -1)


def _allman_stab(xy, A, t, Dp, alpha_s=0.05):
    """Allman 膜のゼロエネルギーモード抑制ばね（弾性, 9×9）"""
    _, b, c = _allman_B(xy, A, (1 / 3, 1 / 3, 1 / 3))
    E = len(xy)
    g = np.zeros((E, 9))
    for i in range(3):
        g[:, 3 * i] = -0.5 * c[:, i]
        g[:, 3 * i + 1] = 0.5 * b[:, i]
        g[:, 3 * i + 2] = -1.0 / 3.0
    ks = alpha_s * Dp[:, 2, 2] * t * A
    return ks[:, None, None] * np.einsum("ei,ej->eij", g, g)


def _transform(R):
    T = np.zeros((len(R), 18, 18))
    for k in range(6):
        T[:, 3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
    return T


def shell_constraints(mesh: ShellMesh, bcs, mats, elem_mat, A, normals):
    """シェルの拘束 {自由度: 値} と外力ベクトル"""
    nd = 6 * mesh.n_nodes
    F = np.zeros(nd)
    fixed = {}
    for bc in bcs:
        nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
        if bc.kind == "fixed":
            for c in range(6):
                for d in 6 * nodes + c:
                    fixed[int(d)] = 0.0
        elif bc.kind == "displacement":
            for c in range(3):
                if bc.values[c] is None:
                    continue
                for d in 6 * nodes + c:
                    fixed[int(d)] = float(bc.values[c])
            # 面内の変位を与えた辺は、ドリリング回転（法線まわり）も止める（辺がまっすぐ動くように）
            if bc.faces:
                el = np.isin(mesh.elems, nodes).any(axis=1)
                n = normals[el].mean(0) if el.any() else np.zeros(3)
                if np.linalg.norm(n) > 0 and np.max(np.abs(n)) / np.linalg.norm(n) > 0.999:
                    ax = int(np.argmax(np.abs(n)))
                    if any(bc.values[c] is not None for c in range(3) if c != ax):
                        for d in 6 * nodes + 3 + ax:
                            fixed.setdefault(int(d), 0.0)
        elif bc.kind in ("symmetry", "frictionless"):
            for f in bc.faces:
                ax = _shell_sym_axis(mesh, int(f), normals)
                fn = mesh.face_nodes([f])
                comps = [ax] if bc.kind == "frictionless" else [ax, 3 + (ax + 1) % 3, 3 + (ax + 2) % 3]
                for c in comps:   # 対称: 面に垂直な変位と、面内の軸まわりの回転を止める
                    for d in 6 * fn + c:
                        fixed[int(d)] = 0.0
        elif bc.kind == "force":
            f = np.asarray(bc.values, float)
            w = _load_weights(mesh, bc.faces)
            for c in range(3):
                np.add.at(F, 6 * w[0] + c, w[1] / w[1].sum() * f[c])
        elif bc.kind == "pressure":
            p = float(bc.values[0])
            for fid in bc.faces:
                if mesh.is_edge(fid):
                    continue
                m = mesh.face_ids == int(fid)
                for c in range(3):
                    np.add.at(F, 6 * mesh.elems[m].ravel() + c, np.repeat(-p * A[m] / 3 * normals[m, c], 3))
        elif bc.kind == "gravity":
            acc = np.asarray(bc.values, float)
            rho = np.array([m.density_t_mm3 for m in mats])[elem_mat]
            mass = rho * mesh.thickness * A
            for c in range(3):
                np.add.at(F, 6 * mesh.elems.ravel() + c, np.repeat(mass / 3 * acc[c], 3))
        elif bc.kind in ("moment", "thermal"):
            raise ValueError("シェルではモーメント・熱条件にまだ対応していません")
    return F, fixed


def _shell_sym_axis(mesh, fid, normals):
    """シェルの辺（または面）の対称面の法線軸"""
    if mesh.is_edge(fid):
        seg = mesh.edges[fid - mesh.n_surf]
        d = mesh.nodes[seg[:, 1]] - mesh.nodes[seg[:, 0]]
        d = d.sum(0)
        d /= np.linalg.norm(d)
        # 辺に接する要素の法線
        nodes = set(seg.ravel().tolist())
        el = [i for i, e in enumerate(mesh.elems) if len(nodes.intersection(e.tolist())) >= 2]
        ns = normals[el].mean(0)
        n = np.cross(d, ns)
    else:
        n = normals[mesh.face_ids == fid].mean(0)
    n /= max(np.linalg.norm(n), 1e-300)
    if np.max(np.abs(n)) < 0.999:
        raise ValueError("シェルの対称領域・摩擦なし支持は、座標軸に垂直な平面上の辺（または面）に設定してください")
    return int(np.argmax(np.abs(n)))


NL_LAYERS = 5   # 板厚方向の積分点（Ansys の SHELL181 の塑性と同じ 5 点, シンプソン則）
_LZ = np.linspace(-0.5, 0.5, NL_LAYERS)
_LW = np.array([1, 4, 2, 4, 1], float) / 12.0      # シンプソン則（合計 1）
_GPL = ([2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3])
_GPX = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]   # 同じ点の (ξ, η)（DKT 用。L1 = 1−ξ−η）


class ShellPlastic:
    """塑性材料を含むシェルの内力・接線（板厚方向 5 層、面内 3 点、平面応力のリターンマッピング）"""

    def __init__(self, mesh, mats, elem_mat, Dp):
        from .plasticity import PlasticModel
        self.mesh, self.mats, self.elem_mat, self.Dp = mesh, mats, elem_mat, Dp
        X = mesh.nodes[mesh.elems]
        self.R, self.xy, self.A = _frames(X)
        self.T = _transform(self.R)
        self.t = mesh.thickness
        E = mesh.n_elems
        # 膜の B は体積座標 (L1,L2,L3)、DKT は (ξ,η)=(L2,L3)
        self.Bm = [_allman_B(self.xy, self.A, Lp)[0] for Lp in _GPL]
        self.Bb = [_dkt_B(self.xy, Lp[1], Lp[2]) for Lp in _GPL]
        self.Ks = _allman_stab(self.xy, self.A, self.t, Dp)
        self.models = {i: PlasticModel(m) for i, m in enumerate(mats) if getattr(m, "plastic", False)}
        sh = (3, NL_LAYERS, E)
        self.ep = np.zeros(sh + (6,))
        self.al = np.zeros(sh + (6,))
        self.pb = np.zeros(sh)
        self.ez = np.zeros(sh)
        self.trial = None
        self.dofs = _shell_dofs(mesh.elems)
        self.rows = np.repeat(self.dofs, 18, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 18)).ravel()

    def internal(self, u):
        from .newton import plane_stress_return
        E = self.mesh.n_elems
        ul = np.einsum("eij,ej->ei", self.T, u[self.dofs])
        um, ub = ul[:, _MEMR], ul[:, _BEN]
        fl = np.zeros((E, 18))
        Kl = np.zeros((E, 18, 18))
        Kl[:, _MEMR[:, None], _MEMR[None, :]] += self.Ks
        fl[:, _MEMR] += np.einsum("eij,ej->ei", self.Ks, um)
        st = [self.ep.copy(), self.al.copy(), self.pb.copy(), self.ez.copy()]
        sig_out = np.zeros((3, NL_LAYERS, E, 3))
        idx_m = np.ix_(range(E), _MEMR, _MEMR)
        for g in range(3):
            Bm, Bb = self.Bm[g], self.Bb[g]
            em = np.einsum("eij,ej->ei", Bm, um)
            kap = np.einsum("eij,ej->ei", Bb, ub)
            for l, (z, wl) in enumerate(zip(_LZ, _LW)):
                zz = z * self.t
                eps = em + zz[:, None] * kap
                C = self.Dp.copy()
                sig = np.einsum("eij,ej->ei", C, eps)
                for mi, model in self.models.items():
                    sel = np.nonzero(self.elem_mat == mi)[0]
                    if len(sel) == 0:
                        continue
                    sg, Cs, e1, a1, p1, z1 = plane_stress_return(model, eps[sel], self.ep[g, l, sel],
                                                                 self.al[g, l, sel], self.pb[g, l, sel],
                                                                 self.ez[g, l, sel])
                    sig[sel], C[sel] = sg, Cs
                    st[0][g, l, sel], st[1][g, l, sel], st[2][g, l, sel], st[3][g, l, sel] = e1, a1, p1, z1
                sig_out[g, l] = sig
                w = (self.A / 3) * self.t * wl            # 面積 × 層の厚さ
                fl[:, _MEMR] += np.einsum("e,eki,ek->ei", w, Bm, sig)
                fl[:, _BEN] += np.einsum("e,eki,ek->ei", w * zz, Bb, sig)
                Kl[:, _MEMR[:, None], _MEMR[None, :]] += np.einsum("e,eki,ekl,elj->eij", w, Bm, C, Bm)
                Kmb = np.einsum("e,eki,ekl,elj->eij", w * zz, Bm, C, Bb)
                Kl[:, _MEMR[:, None], _BEN[None, :]] += Kmb
                Kl[:, _BEN[:, None], _MEMR[None, :]] += Kmb.transpose(0, 2, 1)
                Kl[:, _BEN[:, None], _BEN[None, :]] += np.einsum("e,eki,ekl,elj->eij", w * zz * zz, Bb, C, Bb)
        del idx_m
        self.trial = st + [sig_out]
        fe = np.einsum("eki,ek->ei", self.T, fl)
        Ke = np.einsum("eki,ekl,elj->eij", self.T, Kl, self.T)
        nd = 6 * self.mesh.n_nodes
        f = np.zeros(nd)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(nd, nd)).tocsr()
        return f, K

    def commit(self):
        self.ep, self.al, self.pb, self.ez = [x.copy() for x in self.trial[:4]]


def _layer_to_global(R, s):
    S = np.zeros((len(s), 3, 3))
    S[:, 0, 0], S[:, 1, 1] = s[:, 0], s[:, 1]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 2]
    G = np.einsum("eki,ekl,elj->eij", R, S, R)
    return np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1], G[:, 1, 2], G[:, 0, 2]], 1)


def solve_shell(mesh: ShellMesh, material, bcs: list[BC], elem_materials=None, log=print, progress=None,
                method="auto", n_cores=None, settings=None) -> Result:
    """シェルの静解析（線形、または塑性材料があれば弾塑性）"""
    from .parallel import set_threads
    if n_cores is not None:
        set_threads(n_cores)
    t0 = time.time()
    if not any(b.kind in ("fixed", "displacement", "symmetry", "frictionless") for b in bcs):
        raise ValueError("拘束（固定支持 または 変位）がありません。辺または面に設定してください。")
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, np.int64)
    Dps = np.stack([plane_stress_D(material_D(m)) for m in mats])
    Dp = Dps[elem_mat]
    X = mesh.nodes[mesh.elems]
    t = mesh.thickness
    R, xy, A = _frames(X)
    normals = R[:, 2]
    nd = 6 * mesh.n_nodes
    dofs = _shell_dofs(mesh.elems)
    F, fixed = shell_constraints(mesh, bcs, mats, elem_mat, A, normals)
    used = np.zeros(nd, bool)
    used[dofs.ravel()] = True
    plastic = any(getattr(m, "plastic", False) for m in mats)
    log(f"シェル: 節点 {mesh.n_nodes:,}、要素 {mesh.n_elems:,}" + ("（弾塑性）" if plastic else ""))
    extra = {}
    if plastic:
        from .newton import solve_incremental
        from .nonlinear import AnalysisSettings
        ps = ShellPlastic(mesh, mats, elem_mat, Dp)
        u = solve_incremental(ps.internal, ps.commit, F, fixed, nd, settings or AnalysisSettings(), log,
                              free_mask=used)
        K = ps.internal(u)[1]
        sig = ps.trial[4]           # (3 点, 5 層, E, 3)
        out = {"上面": _layer_to_global(R, sig[:, -1].mean(0)), "下面": _layer_to_global(R, sig[:, 0].mean(0))}
        out = {k: _nodal_avg(mesh, v, A) for k, v in out.items()}
        peq = ps.pb.max(axis=(0, 1))
        extra["相当塑性ひずみ"] = _nodal_avg(mesh, peq[:, None], A)[:, 0]
        R_all = ps.internal(u)[0] - F
    else:
        Ke, R, xy, A = shell_element_K(X, t, Dp)
        rows = np.repeat(dofs, 18, axis=1).ravel()
        cols = np.tile(dofs, (1, 18)).ravel()
        K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nd, nd)).tocsr()
        cd = np.array(sorted(fixed), np.int64)
        cv = np.array([fixed[int(d)] for d in cd])
        free = np.ones(nd, bool)
        free[cd] = False
        free &= used
        u = np.zeros(nd)
        u[cd] = cv
        Kff = K[free][:, free].tocsc()
        rhs = F[free] - K[free][:, cd] @ cv
        uf = _linear_solve(Kff, rhs, log, method="direct" if method == "auto" else method)
        if not np.all(np.isfinite(uf)) or _rel_res(Kff, uf, rhs) > 1e-4:
            raise RuntimeError("解が求まりませんでした。拘束が不足していないか確認してください（シェルは辺に支持を設定します）。")
        u[free] = uf
        ul = np.einsum("eij,ej->ei", _transform(R), u[dofs])
        em = np.einsum("eij,ej->ei", _allman_B(xy, A, (1 / 3, 1 / 3, 1 / 3))[0], ul[:, _MEMR])
        kap = np.einsum("eij,ej->ei", _dkt_B(xy, 1 / 3, 1 / 3), ul[:, _BEN])
        out = {}
        for name, z in (("上面", 0.5), ("下面", -0.5)):
            s = np.einsum("eij,ej->ei", Dp, em + (z * t)[:, None] * kap)
            out[name] = _nodal_avg(mesh, _layer_to_global(R, s), A)
        R_all = K @ u - F
    vm = {k: _mises(v) for k, v in out.items()}
    top_bigger = vm["上面"] >= vm["下面"]
    stress = np.where(top_bigger[:, None], out["上面"], out["下面"])
    U = u.reshape(-1, 6)
    reactions = {}
    for i, bc in enumerate(bcs):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless"):
            nodes = mesh.face_nodes(bc.faces) if bc.faces else np.asarray(bc.nodes, np.int64)
            reactions[i] = R_all.reshape(-1, 6)[nodes, :3].sum(0)
    info = {"自由度数": int(used.sum() - len(fixed)), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "要素タイプ": "シェル（3節点 DKT＋Allman 膜, SHELL181 相当）" + ("・弾塑性 5 層" if plastic else ""),
            "ソルバー": "直接法", "合計時間 [s]": round(time.time() - t0, 2)}
    r = Result(mesh, mats[0], U[:, :3].copy(), stress, np.zeros_like(stress), reactions, info, mats, elem_mat)
    r.extra = {"相当応力（上面） [MPa]": vm["上面"], "相当応力（下面） [MPa]": vm["下面"],
               "回転 θx [rad]": U[:, 3], "回転 θy [rad]": U[:, 4], "回転 θz [rad]": U[:, 5], **extra}
    for k, lab in ((0, "σx"), (1, "σy"), (3, "τxy")):
        r.extra[f"応力 {lab}（上面） [MPa]"] = out["上面"][:, k]
        r.extra[f"応力 {lab}（下面） [MPa]"] = out["下面"][:, k]
    r.rotations = U[:, 3:].copy()
    return r


def _load_weights(mesh: ShellMesh, faces):
    """力を配る節点と重み（辺: 長さ、面: 面積）"""
    nodes, w = [], []
    for f in faces:
        f = int(f)
        if mesh.is_edge(f):
            seg = mesh.edges[f - mesh.n_surf]
            L = np.linalg.norm(mesh.nodes[seg[:, 1]] - mesh.nodes[seg[:, 0]], axis=1)
            nodes.append(seg.ravel())
            w.append(np.repeat(L / 2, 2))
        else:
            tri = mesh.elems[mesh.face_ids == f]
            _, A = mesh.tri_normals_areas(tri)
            nodes.append(tri.ravel())
            w.append(np.repeat(A / 3, 3))
    n = np.concatenate(nodes)
    ww = np.concatenate(w)
    agg = np.bincount(n, ww, mesh.n_nodes)
    idx = np.nonzero(agg)[0]
    return idx, agg[idx]


def _nodal_avg(mesh, ev, A):
    k = mesh.elems.shape[1]
    idx = mesh.elems.ravel()
    wt = np.repeat(A, k)
    den = np.maximum(np.bincount(idx, wt, mesh.n_nodes), 1e-300)
    return np.column_stack([np.bincount(idx, (np.repeat(ev[:, j], k) * wt), mesh.n_nodes) for j in range(ev.shape[1])]) / den[:, None]


def _mises(s):
    return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                   + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))


def structured_plate(a, b, nx, ny, t, origin=(0, 0, 0)) -> ShellMesh:
    """長方形の平板（XY 平面）のシェルメッシュ（テスト・gmsh なし用）。辺: 0:x=0, 1:x=a, 2:y=0, 3:y=b"""
    xs = np.linspace(0, a, nx + 1) + origin[0]
    ys = np.linspace(0, b, ny + 1) + origin[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, origin[2])])
    nid = lambda i, j: i * (ny + 1) + j  # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if (i + j) % 2 == 0:
                tris += [[n0, n1, n2], [n0, n2, n3]]
            else:
                tris += [[n0, n1, n3], [n1, n2, n3]]
    tris = np.array(tris)
    edges = [np.array([[nid(0, j), nid(0, j + 1)] for j in range(ny)]),
             np.array([[nid(nx, j), nid(nx, j + 1)] for j in range(ny)]),
             np.array([[nid(i, 0), nid(i + 1, 0)] for i in range(nx)]),
             np.array([[nid(i, ny), nid(i + 1, ny)] for i in range(nx)])]
    return ShellMesh(nodes, tris, np.full(len(tris), float(t)), np.zeros(len(tris), int), edges,
                     elem_body=np.zeros(len(tris), int), n_surf=1)

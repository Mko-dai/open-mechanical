"""四面体メッシュのデータ構造と、境界面の抽出・面グループ分け。

内部の節点順序は VTK に合わせています。
  Tet4 : 頂点 0,1,2,3
  Tet10: 頂点 0-3 + 中間節点 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(1,3) 9:(2,3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 各面（外向き）の頂点：ある頂点の対面
TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
# Tet10 の各面を Tri6 (a,b,c, ab,bc,ca) として
TET10_FACES = np.array([
    [0, 2, 1, 6, 5, 4],
    [0, 1, 3, 4, 8, 7],
    [1, 2, 3, 5, 9, 8],
    [0, 3, 2, 7, 9, 6],
])
TET10_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])


@dataclass
class Mesh:
    nodes: np.ndarray            # (N,3) float
    elems: np.ndarray            # (E,4) or (E,10) int
    # 境界三角形（頂点のみ, 外向き）と、その Tri6 版（Tet10 時）
    _btri: np.ndarray | None = field(default=None, repr=False)
    _btri_full: np.ndarray | None = field(default=None, repr=False)
    face_ids: np.ndarray | None = None   # 境界三角形ごとの面番号
    elem_body: np.ndarray | None = None  # 要素ごとのボディ番号（ボディキー）
    body_names: dict | None = None       # ボディ番号 → 表示名
    _btri_owner: np.ndarray | None = field(default=None, repr=False)
    node_origin: np.ndarray | None = None   # 分離前の節点番号（ボディ境界で複製した節点用）

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elems = np.asarray(self.elems, dtype=np.int64)
        if self.elems.shape[1] not in (4, 10):
            raise ValueError("Tet4 または Tet10 のみ対応しています")
        self._fix_orientation()

    # ------------------------------------------------------------------ 基本情報
    @property
    def order(self) -> int:
        return 2 if self.elems.shape[1] == 10 else 1

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elems(self) -> int:
        return len(self.elems)

    def bounds(self):
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def element_volumes(self) -> np.ndarray:
        p = self.nodes[self.elems[:, :4]]
        a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
        return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0

    def volume(self) -> float:
        return float(self.element_volumes().sum())

    # ------------------------------------------------------------------ 前処理
    def _fix_orientation(self):
        vol = self.element_volumes()
        bad = vol < 0
        if bad.any():
            e = self.elems[bad].copy()
            e[:, [1, 2]] = e[:, [2, 1]]
            if self.elems.shape[1] == 10:
                # 頂点1と2を入れ替えると (0,1)<->(0,2), (1,3)<->(2,3)
                e[:, [4, 6]] = e[:, [6, 4]]
                e[:, [8, 9]] = e[:, [9, 8]]
            self.elems[bad] = e

    def to_quadratic(self) -> "Mesh":
        """Tet4 → Tet10（辺の中点に節点を追加）。"""
        if self.order == 2:
            return self
        e = self.elems
        edges = np.concatenate([e[:, TET10_EDGES[k]] for k in range(6)])
        edges.sort(axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mids = 0.5 * (self.nodes[uniq[:, 0]] + self.nodes[uniq[:, 1]])
        new_nodes = np.vstack([self.nodes, mids])
        mid_idx = (inv + self.n_nodes).reshape(6, -1).T
        new = Mesh(new_nodes, np.hstack([e, mid_idx]))
        if self.face_ids is not None:
            # 頂点番号は変わらないので境界三角形の並びも同じ
            new.set_face_ids(self.face_ids.copy())
        new.elem_body = None if self.elem_body is None else self.elem_body.copy()
        new.body_names = self.body_names
        return new

    def split_body_interfaces(self) -> "Mesh":
        """ボディ同士で共有している節点を複製し、ボディごとに独立したメッシュにする。

        Ansys と同じく、ボディ間のつながりは「接触（結合を含む）」で与えるため。
        複製した節点は同じ座標なので、接触面は節点どうしがぴったり向かい合います。
        """
        if self.elem_body is None or len(np.unique(self.elem_body)) <= 1:
            out = self
            out.node_origin = np.arange(self.n_nodes)
            return out
        owner = np.full(self.n_nodes, -1, dtype=np.int64)
        nodes = [self.nodes]
        origin = [np.arange(self.n_nodes)]
        elems = self.elems.copy()
        nxt = self.n_nodes
        for b in np.unique(self.elem_body):
            em = self.elem_body == b
            nb = np.unique(self.elems[em])
            taken = owner[nb] >= 0
            owner[nb[~taken]] = b
            dup = nb[taken]
            if len(dup):
                new_idx = np.arange(nxt, nxt + len(dup))
                nxt += len(dup)
                lut = np.arange(self.n_nodes)
                lut[dup] = new_idx
                elems[em] = lut[self.elems[em]]
                nodes.append(self.nodes[dup])
                origin.append(dup)
        out = Mesh(np.vstack(nodes), elems)
        out.elem_body = self.elem_body.copy()
        out.body_names = self.body_names
        out.node_origin = np.concatenate(origin)
        return out

    # ------------------------------------------------------------------ 境界
    def _compute_boundary(self):
        e = self.elems
        faces = np.concatenate([e[:, TET_FACES[k]] for k in range(4)])
        key = np.sort(faces, axis=1)
        _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        once = idx[counts == 1]
        self._btri = faces[once]
        self._btri_owner = once % len(e)
        if self.order == 2:
            full = np.concatenate([e[:, TET10_FACES[k]] for k in range(4)])
            self._btri_full = full[once]
        else:
            self._btri_full = self._btri

    @property
    def boundary_tris(self) -> np.ndarray:
        """境界三角形（頂点のみ, (M,3)）。"""
        if self._btri is None:
            self._compute_boundary()
        return self._btri

    @property
    def boundary_tris_full(self) -> np.ndarray:
        """境界三角形（Tet10 なら Tri6 (M,6)）。"""
        if self._btri is None:
            self._compute_boundary()
        return self._btri_full

    def tri_normals_areas(self, tris=None):
        tris = self.boundary_tris if tris is None else tris
        p = self.nodes[tris[:, :3]]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        a = np.linalg.norm(n, axis=1)
        return n / np.maximum(a, 1e-300)[:, None], 0.5 * a

    def tri_centroids(self):
        return self.nodes[self.boundary_tris].mean(axis=1)

    def group_faces(self, angle_deg: float = 35.0) -> np.ndarray:
        """隣り合う三角形の法線の角度差で、境界を「面」に分ける。

        平面はもちろん、円筒面などの滑らかな曲面も1つの面になります。
        結果は self.face_ids に保存され、面積の大きい順に 0,1,2,... と番号付けされます。
        """
        tris = self.boundary_tris
        m = len(tris)
        normals, areas = self.tri_normals_areas()
        # 辺→三角形の隣接
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        owner = np.tile(np.arange(m), 3)
        edges.sort(axis=1)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        es, ow = edges[order], owner[order]
        same = np.all(es[1:] == es[:-1], axis=1)
        a_idx, b_idx = ow[:-1][same], ow[1:][same]
        cos_t = np.cos(np.radians(angle_deg))
        ok = np.einsum("ij,ij->i", normals[a_idx], normals[b_idx]) >= cos_t
        a_idx, b_idx = a_idx[ok], b_idx[ok]

        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        g = coo_matrix((np.ones(len(a_idx)), (a_idx, b_idx)), shape=(m, m))
        n_comp, lab = connected_components(g, directed=False)
        group_area = np.bincount(lab, weights=areas, minlength=n_comp)
        rank = np.empty(n_comp, dtype=int)
        rank[np.argsort(-group_area, kind="stable")] = np.arange(n_comp)
        self.face_ids = rank[lab]
        return self.face_ids

    def set_face_ids(self, face_ids):
        self.face_ids = np.asarray(face_ids, dtype=int)

    @property
    def n_faces(self) -> int:
        if self.face_ids is None:
            self.group_faces()
        return int(self.face_ids.max()) + 1 if len(self.face_ids) else 0

    def face_info(self):
        """各面の 面積・重心・平均法線 のリスト。"""
        if self.face_ids is None:
            self.group_faces()
        n, a = self.tri_normals_areas()
        c = self.tri_centroids()
        info = []
        for f in range(self.n_faces):
            m = self.face_ids == f
            A = a[m].sum()
            cen = (c[m] * a[m, None]).sum(0) / A
            nn = (n[m] * a[m, None]).sum(0)
            nl = np.linalg.norm(nn)
            info.append({"id": f, "area": float(A), "centroid": cen,
                         "normal": nn / nl if nl > 1e-12 else np.zeros(3),
                         "flat": bool(nl / A > 0.999) if A > 0 else False})
        return info

    @property
    def boundary_owner(self) -> np.ndarray:
        """境界三角形ごとの所属要素番号"""
        if self._btri is None:
            self._compute_boundary()
        return self._btri_owner

    @property
    def n_bodies(self) -> int:
        return 1 if self.elem_body is None else int(self.elem_body.max()) + 1

    def body_ids(self) -> np.ndarray:
        """ボディ番号の一覧（存在するもの）"""
        return np.array([0]) if self.elem_body is None else np.unique(self.elem_body)

    def tri_body(self) -> np.ndarray:
        """境界三角形ごとのボディ番号"""
        if self.elem_body is None:
            return np.zeros(len(self.boundary_tris), dtype=int)
        return self.elem_body[self.boundary_owner]

    def body_faces(self, body) -> list[int]:
        """ボディの外側の面番号"""
        if self.face_ids is None:
            self.group_faces()
        return sorted(set(self.face_ids[self.tri_body() == body].tolist()))

    def face_tris(self, face_id: int, full: bool = True) -> np.ndarray:
        if self.face_ids is None:
            self.group_faces()
        m = self.face_ids == face_id
        return (self.boundary_tris_full if full else self.boundary_tris)[m]

    def face_nodes(self, face_id) -> np.ndarray:
        ids = np.atleast_1d(face_id)
        tris = np.concatenate([self.face_tris(int(f)) for f in ids])
        return np.unique(tris)

    def nearest_face(self, point) -> int:
        """座標に最も近い境界三角形の面番号（3D画面でのクリック選択用）。"""
        if self.face_ids is None:
            self.group_faces()
        c = self.tri_centroids()
        d = np.linalg.norm(c - np.asarray(point)[None], axis=1)
        return int(self.face_ids[np.argmin(d)])


def structured_box(lx, ly, lz, nx, ny, nz, origin=(0.0, 0.0, 0.0)) -> Mesh:
    """直方体の構造格子四面体メッシュ（gmsh が無くても使える簡易版）。

    各六面体を6個の四面体に分割（Kuhn 分割）するので、隣接要素とも整合します。
    """
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    zs = np.linspace(0, lz, nz + 1) + origin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    v = [nid(i + a, j + b, k + c) for (a, b, c) in
         [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    v = np.stack(v, axis=1)
    # 対角 0-6 を共有する6分割
    kuhn = [[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]]
    elems = np.concatenate([v[:, t] for t in kuhn])
    mesh = Mesh(nodes, elems)
    # 面番号を gmsh の直方体と同じ順にする: x=0, x=lx, y=0, y=ly, z=0, z=lz
    n, _ = mesh.tri_normals_areas()
    axis = np.argmax(np.abs(n), axis=1)
    mesh.set_face_ids(2 * axis + (n[np.arange(len(n)), axis] > 0))
    return mesh

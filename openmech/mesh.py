"""四面体・三角柱メッシュのデータ構造と、境界面の抽出・面グループ分け。

内部の節点順序は VTK に合わせています。
  Tet4 : 頂点 0,1,2,3
  Tet10: 頂点 0-3 + 中間節点 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(1,3) 9:(2,3)
  Wedge6 / Wedge15（三角柱、多層スイープで作る）: elements.py を参照
三角柱の側面（四角形）は、境界の三角形の並びには 2 つの三角形として入れ（表示・面番号・選択用）、
荷重の積分用に四角形のまま boundary_quads_full にも持つ。
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
    _bquad_full: np.ndarray | None = field(default=None, repr=False)
    _bquad_of: np.ndarray | None = field(default=None, repr=False)
    node_origin: np.ndarray | None = None   # 分離前の節点番号（ボディ境界で複製した節点用）

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elems = np.asarray(self.elems, dtype=np.int64)
        if self.elems.shape[1] not in (4, 10, 6, 15, 8, 20):
            raise ValueError("Tet4 / Tet10 / Wedge6 / Wedge15 / Hex8 / Hex20 のみ対応しています")
        self._fix_orientation()

    # ------------------------------------------------------------------ 基本情報
    @property
    def order(self) -> int:
        return 2 if self.elems.shape[1] in (10, 15) else 1

    @property
    def is_wedge(self) -> bool:
        """三角柱（ソリッド）要素か。シェル・2D の 6 節点三角形は含まない"""
        return (self.elems.shape[1] in (6, 15) and not getattr(self, "is_shell", False)
                and not getattr(self, "is_2d", False))

    @property
    def is_hex(self) -> bool:
        """六面体（ソリッド）要素か"""
        return (self.elems.shape[1] in (8, 20) and not getattr(self, "is_shell", False)
                and not getattr(self, "is_2d", False))

    @property
    def has_quads(self) -> bool:
        """境界に四角形の面がある要素（三角柱・六面体）"""
        return self.is_wedge or self.is_hex

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elems(self) -> int:
        return len(self.elems)

    def bounds(self):
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def element_volumes(self) -> np.ndarray:
        if self.is_hex:
            from .elements import hex_volumes
            return hex_volumes(self.nodes, self.elems)
        if self.is_wedge:
            from .elements import wedge_volumes
            return wedge_volumes(self.nodes, self.elems)
        p = self.nodes[self.elems[:, :4]]
        a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
        return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0

    def volume(self) -> float:
        return float(self.element_volumes().sum())

    # ------------------------------------------------------------------ 前処理
    def _fix_orientation(self):
        if self.is_hex:
            # 体積が負（左手系）なら 1↔3・5↔7 を入れ替えて鏡映
            from .elements import hex_volumes
            bad = hex_volumes(self.nodes, self.elems[:, :8]) < 0
            if bad.any():
                perm = [0, 3, 2, 1, 4, 7, 6, 5]
                if self.elems.shape[1] == 20:
                    # 8:(0,1)↔11:(3,0) 9:(1,2)↔10:(2,3) 12:(4,5)↔15:(7,4) 13:(5,6)↔14:(6,7) 17:(1,5)↔19:(3,7)
                    perm += [11, 10, 9, 8, 15, 14, 13, 12, 16, 19, 18, 17]
                self.elems[bad] = self.elems[bad][:, perm]
            return
        if self.is_wedge:
            # 下面の三角形の法線が上面の方を向くように（逆なら 1↔2・4↔5 を入れ替え）
            p = self.nodes[self.elems[:, :6]]
            nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            up = p[:, 3:].mean(1) - p[:, :3].mean(1)
            bad = np.einsum("ij,ij->i", nrm, up) < 0
            if bad.any():
                perm = [0, 2, 1, 3, 5, 4]
                if self.elems.shape[1] == 15:
                    # 辺 6:(0,1)↔8:(2,0), 7:(1,2) はそのまま、9:(3,4)↔11:(5,3), 13:(1,4)↔14:(2,5)
                    perm += [8, 7, 6, 11, 10, 9, 12, 14, 13]
                self.elems[bad] = self.elems[bad][:, perm]
            return
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
        """Tet4 → Tet10 / Wedge6 → Wedge15（辺の中点に節点を追加）。"""
        if self.order == 2:
            return self
        e = self.elems
        if self.is_hex:
            from .elements import HEX20_EDGES
            elist = np.array(HEX20_EDGES)
        elif self.is_wedge:
            from .elements import WEDGE15_EDGES
            elist = np.array(WEDGE15_EDGES)
        else:
            elist = TET10_EDGES
        ne_ = len(elist)
        edges = np.concatenate([e[:, elist[k]] for k in range(ne_)])
        edges.sort(axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mids = 0.5 * (self.nodes[uniq[:, 0]] + self.nodes[uniq[:, 1]])
        new_nodes = np.vstack([self.nodes, mids])
        mid_idx = (inv + self.n_nodes).reshape(ne_, -1).T
        new = Mesh(new_nodes, np.hstack([e, mid_idx]))
        if self.face_ids is not None:
            # 頂点番号は変わらないので境界三角形の並びも同じ
            new.set_face_ids(self.face_ids.copy())
        new.elem_body = None if self.elem_body is None else self.elem_body.copy()
        new.body_names = self.body_names
        return new

    def split_body_interfaces(self, share: dict | None = None) -> "Mesh":
        """ボディ同士で共有している節点を複製し、ボディごとに独立したメッシュにする。

        Ansys と同じく、ボディ間のつながりは「接触（結合を含む）」で与えるため。
        複製した節点は同じ座標なので、接触面は節点どうしがぴったり向かい合います。

        share: {ボディ番号: パート番号}。同じパートのボディ同士は節点を共有したまま
               （Ansys の「トポロジ共有」、Femtet の節点共有）。接触は作られません。
        """
        share = share or {}
        if self.elem_body is not None:
            # パート単位で分割する（パートに属さないボディはそれぞれ単独）
            ub = np.unique(self.elem_body)
            gid = {}
            for b in ub.tolist():
                g = share.get("*", share.get(int(b)))
                gid[b] = ("p", g) if g is not None else ("b", b)
            keys = {k: i for i, k in enumerate(sorted(set(gid.values()), key=str))}
            group = np.array([keys[gid[b]] for b in self.elem_body.tolist()]) if len(self.elem_body) else None
        if self.elem_body is None or len(np.unique(group)) <= 1:
            out = self
            out.node_origin = np.arange(self.n_nodes)
            return out
        owner = np.full(self.n_nodes, -1, dtype=np.int64)
        nodes = [self.nodes]
        origin = [np.arange(self.n_nodes)]
        elems = self.elems.copy()
        nxt = self.n_nodes
        for b in np.unique(group):
            em = group == b
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
        if self.is_hex:
            return self._compute_boundary_hex()
        if self.is_wedge:
            return self._compute_boundary_wedge()
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

    def _compute_boundary_wedge(self):
        from .elements import WEDGE15_QUAD_LOOP, WEDGE15_TRI_FACES, WEDGE_QUAD_FACES, WEDGE_TRI_FACES
        e = self.elems
        ne = len(e)
        quad2 = self.elems.shape[1] == 15
        # 三角形の面（上下）
        tf = np.concatenate([e[:, WEDGE_TRI_FACES[k]] for k in range(2)])
        _, ti, tc = np.unique(np.sort(tf, axis=1), axis=0, return_index=True, return_counts=True)
        t_once = ti[tc == 1]
        # 四角形の面（側面）
        qf = np.concatenate([e[:, WEDGE_QUAD_FACES[k]] for k in range(3)])
        _, qi, qc = np.unique(np.sort(qf, axis=1), axis=0, return_index=True, return_counts=True)
        q_once = qi[qc == 1]
        quads = qf[q_once]
        if quad2:
            tfull = np.concatenate([e[:, WEDGE15_TRI_FACES[k]] for k in range(2)])[t_once]
            qloop = np.concatenate([e[:, WEDGE15_QUAD_LOOP[k]] for k in range(3)])[q_once]
        else:
            tfull = tf[t_once]
            qloop = quads
        a, b, c, d = quads.T
        # 四角形を 2 つの三角形に（向きはそのまま外向き）
        tri_q = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        nq = len(quads)
        if quad2:
            L = qloop        # [c0,m01,c1,m12,c2,m23,c3,m30]
            full_q = np.concatenate([np.column_stack([L[:, 0], L[:, 2], L[:, 4], L[:, 1], L[:, 3], L[:, 5]]),
                                     np.column_stack([L[:, 0], L[:, 4], L[:, 6], L[:, 5], L[:, 7], L[:, 1]])])
        else:
            full_q = tri_q
        self._btri = np.concatenate([tf[t_once], tri_q])
        self._btri_full = np.concatenate([tfull, full_q])
        self._btri_owner = np.concatenate([t_once % ne, np.tile(q_once % ne, 2)])
        self._bquad_full = qloop
        self._bquad_of = np.concatenate([np.full(len(t_once), -1), np.tile(np.arange(nq), 2)])

    def _compute_boundary_hex(self):
        from .elements import HEX20_QUAD_LOOP, HEX_QUAD_FACES
        e = self.elems
        ne = len(e)
        quad2 = e.shape[1] == 20
        qf = np.concatenate([e[:, HEX_QUAD_FACES[k]] for k in range(6)])
        # 縮退した六面体（ピラミッド・三角柱・四面体の形）: 面の鍵は「重複を除いた頂点」。点や線に潰れた面は捨てる
        key = np.sort(qf, axis=1)
        dup = np.zeros_like(key, dtype=bool)
        dup[:, 1:] = key[:, 1:] == key[:, :-1]
        nuniq = 4 - dup.sum(1)
        key = np.where(dup, -1, key)
        key = np.sort(key, axis=1)
        valid = nuniq >= 3
        _, qi, qc = np.unique(key, axis=0, return_index=True, return_counts=True)
        q_once = qi[(qc == 1) & valid[qi]]
        q_once.sort()
        quads = qf[q_once]
        qloop = np.concatenate([e[:, HEX20_QUAD_LOOP[k]] for k in range(6)])[q_once] if quad2 else quads
        a, b, c, d = quads.T
        tri_q = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        if quad2:
            L = qloop
            full_q = np.concatenate([np.column_stack([L[:, 0], L[:, 2], L[:, 4], L[:, 1], L[:, 3], L[:, 5]]),
                                     np.column_stack([L[:, 0], L[:, 4], L[:, 6], L[:, 5], L[:, 7], L[:, 1]])])
        else:
            full_q = tri_q
        nq = len(quads)
        owner = np.tile(q_once % ne, 2)
        qof = np.tile(np.arange(nq), 2)
        # 三角形に潰れた四角形の、面積 0 の三角形を除く
        okt = (tri_q[:, 0] != tri_q[:, 1]) & (tri_q[:, 1] != tri_q[:, 2]) & (tri_q[:, 0] != tri_q[:, 2])
        self._btri = tri_q[okt]
        self._btri_full = full_q[okt]
        self._btri_owner = owner[okt]
        self._bquad_full = qloop
        self._bquad_of = qof[okt]

    def sweep_lines(self):
        """三角柱メッシュの節点を、スイープ方向に並ぶ列（柱）ごとに番号付けする（反復法の線平滑化用）。

        返り値: 節点ごとの列番号 (N,)。三角柱でなければ None
        """
        if not self.is_wedge:
            return None
        e = self.elems
        d = self.nodes[e[:, 3]] - self.nodes[e[:, 0]]
        ax = int(np.argmax(np.abs(d).mean(axis=0)))
        other = [k for k in range(3) if k != ax]
        lo, hi = self.bounds()
        tol = 1e-7 * max(1.0, float(np.linalg.norm(hi - lo)))
        key = np.round(self.nodes[:, other] / tol).astype(np.int64)
        _, ids = np.unique(key, axis=0, return_inverse=True)
        return ids.reshape(-1)

    @property
    def boundary_quads_full(self) -> np.ndarray:
        """三角柱の側面（境界の四角形）。Quad4 か、周の順の Quad8"""
        if self._btri is None:
            self._compute_boundary()
        return self._bquad_full if self._bquad_full is not None else np.zeros((0, 4), dtype=np.int64)

    @property
    def boundary_quad_of(self):
        """境界の三角形ごとに、もとの四角形の番号（四角形から作った三角形でなければ -1）"""
        if self._btri is None:
            self._compute_boundary()
        return self._bquad_of

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


def structured_hex_box(lx, ly, lz, nx, ny, nz, origin=(0.0, 0.0, 0.0), quadratic=True) -> Mesh:
    """直方体の六面体メッシュ（Hex8、quadratic=True なら Hex20）。面番号は structured_box と同じ順"""
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    zs = np.linspace(0, lz, nz + 1) + origin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    elems = np.stack([nid(i + a, j + b, k + c) for (a, b, c) in
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]], 1)
    mesh = Mesh(nodes, elems)
    if quadratic:
        mesh = mesh.to_quadratic()
    n, _ = mesh.tri_normals_areas()
    axis = np.argmax(np.abs(n), axis=1)
    mesh.set_face_ids(2 * axis + (n[np.arange(len(n)), axis] > 0))
    return mesh

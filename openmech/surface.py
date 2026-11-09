"""面（境界）の上の積分: ∫N dA、∫N Nᵀ dA（対流・放射・熱流束・電荷など）。

四面体の面 Tri3/Tri6（曲がった Tri6 もアイソパラメトリックに積分）と、三角柱の側面 Quad4/Quad8 に対応。
"""
from __future__ import annotations

import numpy as np

# 三角形の 6 点ガウス積分（4 次まで厳密）。面積座標 (L1, L2) と重み（合計 1/2）
_a, _b = 0.445948490915965, 0.091576213509771
_wa, _wb = 0.223381589678011 / 2, 0.109951743655322 / 2
TRI6_PTS = np.array([[_a, _a], [1 - 2 * _a, _a], [_a, 1 - 2 * _a], [_b, _b], [1 - 2 * _b, _b], [_b, 1 - 2 * _b]])
TRI6_W = np.array([_wa, _wa, _wa, _wb, _wb, _wb])


def tri_N(r, s, k):
    t = 1 - r - s
    if k == 3:
        return np.array([t, r, s]), np.array([[-1.0, -1.0], [1, 0], [0, 1]])
    # Tri6: 頂点 0,1,2 と辺の中点 3:(0,1) 4:(1,2) 5:(2,0)
    N = np.array([t * (2 * t - 1), r * (2 * r - 1), s * (2 * s - 1), 4 * t * r, 4 * r * s, 4 * s * t])
    dN = np.array([[-(4 * t - 1), -(4 * t - 1)], [4 * r - 1, 0], [0, 4 * s - 1],
                   [4 * (t - r), -4 * r], [4 * s, 4 * r], [-4 * s, 4 * (t - s)]])
    return N, dN


class SurfaceSet:
    """面の集合の積分点データ。 conn: (S,k) 節点番号, N: (q,k), dA: (S,q) 面積要素×重み, n: (S,q,3) 外向き単位法線"""

    def __init__(self, conn, N, dA, normal, xq):
        self.conn, self.N, self.dA, self.normal, self.xq = conn, N, dA, normal, xq

    @property
    def area(self):
        return float(self.dA.sum())

    def lumped(self, val=1.0):
        """∫ val N dA を節点ごとに → (nodes, weights)"""
        v = np.broadcast_to(np.asarray(val, float), self.dA.shape)
        w = np.einsum("sq,qk->sk", self.dA * v, self.N)
        return self.conn.ravel(), w.ravel()

    def matrix(self, n_nodes, val=1.0):
        """∫ val N Nᵀ dA（節点数 × 節点数の疎行列）"""
        import scipy.sparse as sp
        v = np.broadcast_to(np.asarray(val, float), self.dA.shape)
        Me = np.einsum("sq,qa,qb->sab", self.dA * v, self.N, self.N)
        k = self.conn.shape[1]
        r = np.repeat(self.conn, k, axis=1).ravel()
        c = np.tile(self.conn, (1, k)).ravel()
        return sp.csr_matrix((Me.ravel(), (r, c)), shape=(n_nodes, n_nodes))


def _tri_set(nodes, tris):
    k = tris.shape[1]
    X = nodes[tris]
    Ns, dAs, ns, xs = [], [], [], []
    for (r, s), w in zip(TRI6_PTS, TRI6_W):
        N, dN = tri_N(r, s, k)
        t1 = np.einsum("snk,n->sk", X, dN[:, 0])
        t2 = np.einsum("snk,n->sk", X, dN[:, 1])
        nv = np.cross(t1, t2)
        j = np.linalg.norm(nv, axis=1)
        Ns.append(N)
        dAs.append(w * j)
        ns.append(nv / np.maximum(j, 1e-300)[:, None])
        xs.append(np.einsum("snk,n->sk", X, N))
    return SurfaceSet(tris, np.array(Ns), np.array(dAs).T, np.stack(ns, 1), np.stack(xs, 1))


def _quad_set(nodes, quads):
    from .elements import _G2, _G3, quad_dN, quad_N
    if quads.shape[1] == 8:
        idx = quads[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        n = 8
    else:
        idx, n = quads, 4
    X = nodes[idx]
    g, gw = _G3 if n == 8 else _G2
    Ns, dAs, ns, xs = [], [], [], []
    for xi, wa in zip(g, gw):
        for eta, wb in zip(g, gw):
            N = quad_N(xi, eta, n)
            dN = quad_dN(xi, eta, n)
            t1 = np.einsum("qnk,n->qk", X, dN[:, 0])
            t2 = np.einsum("qnk,n->qk", X, dN[:, 1])
            nv = np.cross(t1, t2)
            j = np.linalg.norm(nv, axis=1)
            Ns.append(N)
            dAs.append(wa * wb * j)
            ns.append(nv / np.maximum(j, 1e-300)[:, None])
            xs.append(np.einsum("qnk,n->qk", X, N))
    return SurfaceSet(idx, np.array(Ns), np.array(dAs).T, np.stack(ns, 1), np.stack(xs, 1))


def surface_sets(mesh, faces) -> list[SurfaceSet]:
    """面番号のリスト → 積分点データ（三角形の集合と四角形の集合）"""
    faces = [int(f) for f in faces]
    if not faces:
        return []
    rows = np.concatenate([np.nonzero(mesh.face_ids == f)[0] for f in faces])
    qof = mesh.boundary_quad_of[rows] if getattr(mesh, "has_quads", False) else np.full(len(rows), -1)
    out = []
    tri_rows = rows[qof < 0]
    if len(tri_rows):
        out.append(_tri_set(mesh.nodes, mesh.boundary_tris_full[tri_rows]))
    if (qof >= 0).any():
        out.append(_quad_set(mesh.nodes, mesh.boundary_quads_full[np.unique(qof[qof >= 0])]))
    return out

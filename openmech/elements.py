"""三角柱（ウェッジ）要素の形状関数・積分点・面（Ansys の SOLID186/187 のウェッジ形状と同じ 15 節点）

自然座標 (r, s, t): 三角形の面積座標 L1 = 1-r-s, L2 = r, L3 = s、厚さ方向 t ∈ [-1, 1]
節点の順序（VTK と同じ）:
  Wedge6 : 0,1,2 = 下面（t=-1）の三角形、3,4,5 = 上面（t=+1）の三角形
  Wedge15: 0-5 頂点 + 6:(0,1) 7:(1,2) 8:(2,0) 9:(3,4) 10:(4,5) 11:(5,3) 12:(0,3) 13:(1,4) 14:(2,5)
正の向き: 下面 0,1,2 を上（t の正の向き）から見て反時計回り
"""
from __future__ import annotations

import numpy as np

WEDGE15_EDGES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
WEDGE_TRI_FACES = np.array([[0, 2, 1], [3, 4, 5]])                         # 外向き
WEDGE15_TRI_FACES = np.array([[0, 2, 1, 8, 7, 6], [3, 4, 5, 9, 10, 11]])    # Tri6 (a,b,c, ab,bc,ca)
WEDGE_QUAD_FACES = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]])     # 外向き
# Quad8 を周に沿って（頂点, 中間, 頂点, 中間, …）
WEDGE15_QUAD_LOOP = np.array([[0, 6, 1, 13, 4, 9, 3, 12], [1, 7, 2, 14, 5, 10, 4, 13], [2, 8, 0, 12, 3, 11, 5, 14]])

_CORNER_RS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def is_wedge(nn: int) -> bool:
    return nn in (6, 15)


def _L(r, s):
    return np.array([1.0 - r - s, r, s])


def wedge_N(p, nn):
    r, s, t = p
    L = _L(r, s)
    if nn == 6:
        return np.concatenate([L * (1 - t) / 2, L * (1 + t) / 2])
    N = np.empty(15)
    for i in range(3):
        N[i] = 0.5 * L[i] * (2 * L[i] - 1) * (1 - t) - 0.5 * L[i] * (1 - t * t)
        N[3 + i] = 0.5 * L[i] * (2 * L[i] - 1) * (1 + t) - 0.5 * L[i] * (1 - t * t)
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
        N[6 + k] = 2 * L[i] * L[j] * (1 - t)
        N[9 + k] = 2 * L[i] * L[j] * (1 + t)
    for i in range(3):
        N[12 + i] = L[i] * (1 - t * t)
    return N


def wedge_dN(p, nn):
    """(nn,3) 自然座標 (r,s,t) での微分"""
    r, s, t = p
    L = _L(r, s)
    dLdrs = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])      # dL_i/dr, dL_i/ds
    if nn == 6:
        out = np.zeros((6, 3))
        for i in range(3):
            out[i, :2] = dLdrs[i] * (1 - t) / 2
            out[i, 2] = -L[i] / 2
            out[3 + i, :2] = dLdrs[i] * (1 + t) / 2
            out[3 + i, 2] = L[i] / 2
        return out
    dNdL = np.zeros((15, 3))
    dNdt = np.zeros(15)
    for i in range(3):
        a = 0.5 * (4 * L[i] - 1)
        dNdL[i, i] = a * (1 - t) - 0.5 * (1 - t * t)
        dNdL[3 + i, i] = a * (1 + t) - 0.5 * (1 - t * t)
        dNdt[i] = -0.5 * L[i] * (2 * L[i] - 1) + L[i] * t
        dNdt[3 + i] = 0.5 * L[i] * (2 * L[i] - 1) + L[i] * t
    for k, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
        dNdL[6 + k, i] = 2 * L[j] * (1 - t)
        dNdL[6 + k, j] = 2 * L[i] * (1 - t)
        dNdL[9 + k, i] = 2 * L[j] * (1 + t)
        dNdL[9 + k, j] = 2 * L[i] * (1 + t)
        dNdt[6 + k] = -2 * L[i] * L[j]
        dNdt[9 + k] = 2 * L[i] * L[j]
    for i in range(3):
        dNdL[12 + i, i] = 1 - t * t
        dNdt[12 + i] = -2 * L[i] * t
    out = np.zeros((15, 3))
    out[:, :2] = dNdL @ dLdrs
    out[:, 2] = dNdt
    return out


# 三角形の積分（重みの合計 = 1/2）
_TRI3 = (np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), np.full(3, 1 / 6))
_a1, _b1 = 0.445948490915965, 0.091576213509771
_w1, _w2 = 0.223381589678011 / 2, 0.109951743655322 / 2
_TRI6 = (np.array([[_a1, _a1], [1 - 2 * _a1, _a1], [_a1, 1 - 2 * _a1],
                   [_b1, _b1], [1 - 2 * _b1, _b1], [_b1, 1 - 2 * _b1]]),
         np.array([_w1, _w1, _w1, _w2, _w2, _w2]))
_G2 = (np.array([-1, 1]) / np.sqrt(3), np.ones(2))
_G3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9)


def _tensor(tri, line):
    (tp, tw), (lp, lw) = tri, line
    pts = np.array([[a, b, c] for (a, b) in tp for c in lp])
    wts = np.array([wa * wc for wa in tw for wc in lw])
    return pts, wts


def wedge_quadrature(nn, full=False):
    """剛性用（Wedge15: 3 点 × 3 点 = 9 点、Wedge6: 3 × 2）/ full=True は質量・体積力用（6 × 3 = 18 点）"""
    if full:
        return _tensor(_TRI6, _G3)
    return _tensor(_TRI3, _G3 if nn == 15 else _G2)


def wedge_node_points(nn):
    """各節点の自然座標 (nn,3)"""
    pts = [[*_CORNER_RS[i], -1.0] for i in range(3)] + [[*_CORNER_RS[i], 1.0] for i in range(3)]
    if nn == 15:
        for t in (-1.0, 1.0):
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                pts.append([*(0.5 * (_CORNER_RS[i] + _CORNER_RS[j])), t])
        for i in range(3):
            pts.append([*_CORNER_RS[i], 0.0])
    return np.array(pts)


def wedge_mass_ref(nn):
    """基準要素の整合質量 ∫ N_a N_b dV / V（形が一定の三角柱なら実要素でも同じ。合計 1）"""
    pts, wts = wedge_quadrature(nn, full=True)
    C = np.zeros((nn, nn))
    for p, w in zip(pts, wts):
        N = wedge_N(p, nn)
        C += w * np.outer(N, N)
    return C / wts.sum()


def wedge_volumes(nodes, elems):
    nn = elems.shape[1]
    pts, wts = wedge_quadrature(nn, full=True)
    X = nodes[elems]
    V = np.zeros(len(elems))
    for p, w in zip(pts, wts):
        J = np.einsum("enk,nl->ekl", X, wedge_dN(p, nn))
        V += w * np.linalg.det(J)
    return V


# ---------------------------------------------------------------------------- 四角形の面（荷重の配分）

def quad_N(xi, eta, n):
    if n == 4:
        return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    # Quad8: 頂点 0..3（反時計回り）+ 中間 4:(0,1) 5:(1,2) 6:(2,3) 7:(3,0)
    c = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    N = np.empty(8)
    for i, (a, b) in enumerate(c):
        N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
    N[4] = 0.5 * (1 - xi * xi) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta * eta)
    N[6] = 0.5 * (1 - xi * xi) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta * eta)
    return N


def quad_dN(xi, eta, n):
    h = 1e-6
    return np.column_stack([(quad_N(xi + h, eta, n) - quad_N(xi - h, eta, n)) / (2 * h),
                            (quad_N(xi, eta + h, n) - quad_N(xi, eta - h, n)) / (2 * h)])


def quad_load_weights(nodes, quads):
    """四角形の面の ∫N dA（節点ごとのスカラー）と ∫N n dA（ベクトル）。

    quads: (Q,4) 頂点（周の順）または (Q,8) 周の順の Quad8 [c0,m01,c1,m12,c2,m23,c3,m30]
    返り値: (node_idx (Q,k), w (Q,k), wn (Q,k,3), area (Q,))
    """
    if quads.shape[1] == 8:
        idx = quads[:, [0, 2, 4, 6, 1, 3, 5, 7]]      # → Quad8 の標準順
        n = 8
    else:
        idx = quads
        n = 4
    X = nodes[idx]
    g, gw = _G3 if n == 8 else _G2
    w = np.zeros((len(quads), n))
    wn = np.zeros((len(quads), n, 3))
    area = np.zeros(len(quads))
    for xi, wa in zip(g, gw):
        for eta, wb in zip(g, gw):
            N = quad_N(xi, eta, n)
            dN = quad_dN(xi, eta, n)
            t1 = np.einsum("qnk,n->qk", X, dN[:, 0])
            t2 = np.einsum("qnk,n->qk", X, dN[:, 1])
            nv = np.cross(t1, t2)                     # = n dA / (dξ dη)
            dA = np.linalg.norm(nv, axis=1)
            w += (wa * wb * dA)[:, None] * N[None, :]
            wn += wa * wb * N[None, :, None] * nv[:, None, :]
            area += wa * wb * dA
    return idx, w, wn, area

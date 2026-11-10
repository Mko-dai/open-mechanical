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


# ============================================================================ 六面体（Hex8 / Hex20）
# Ansys の SOLID185 / SOLID186（六面体）に相当。自然座標 (ξ, η, ζ) ∈ [-1, 1]³
# 節点の順序（VTK と同じ）: 0-3 下面（ζ=-1、反時計回り）、4-7 上面、
#   Hex20 の中間節点 8:(0,1) 9:(1,2) 10:(2,3) 11:(3,0) 12:(4,5) 13:(5,6) 14:(6,7) 15:(7,4) 16:(0,4) 17:(1,5) 18:(2,6) 19:(3,7)
HEX_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
HEX20_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
HEX_QUAD_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])  # 外向き
HEX20_QUAD_LOOP = np.array([[0, 11, 3, 10, 2, 9, 1, 8], [4, 12, 5, 13, 6, 14, 7, 15], [0, 8, 1, 17, 5, 12, 4, 16],
                            [1, 9, 2, 18, 6, 13, 5, 17], [2, 10, 3, 19, 7, 14, 6, 18], [3, 11, 0, 16, 4, 15, 7, 19]])
_HEX20_MID = np.array([0.5 * (HEX_CORNERS[i] + HEX_CORNERS[j]) for i, j in HEX20_EDGES])


def is_hex(nn: int) -> bool:
    return nn in (8, 20)


def hex_N(p, nn):
    x, y, z = p
    C = HEX_CORNERS
    if nn == 8:
        return 0.125 * (1 + C[:, 0] * x) * (1 + C[:, 1] * y) * (1 + C[:, 2] * z)
    N = np.empty(20)
    a, b, c = C[:, 0], C[:, 1], C[:, 2]
    N[:8] = 0.125 * (1 + a * x) * (1 + b * y) * (1 + c * z) * (a * x + b * y + c * z - 2)
    M = _HEX20_MID
    for k in range(12):
        mx, my, mz = M[k]
        if mx == 0:
            N[8 + k] = 0.25 * (1 - x * x) * (1 + my * y) * (1 + mz * z)
        elif my == 0:
            N[8 + k] = 0.25 * (1 + mx * x) * (1 - y * y) * (1 + mz * z)
        else:
            N[8 + k] = 0.25 * (1 + mx * x) * (1 + my * y) * (1 - z * z)
    return N


def hex_dN(p, nn):
    """(nn,3) 自然座標での微分"""
    x, y, z = p
    C = HEX_CORNERS
    a, b, c = C[:, 0], C[:, 1], C[:, 2]
    out = np.zeros((nn, 3))
    if nn == 8:
        out[:, 0] = 0.125 * a * (1 + b * y) * (1 + c * z)
        out[:, 1] = 0.125 * b * (1 + a * x) * (1 + c * z)
        out[:, 2] = 0.125 * c * (1 + a * x) * (1 + b * y)
        return out
    s = a * x + b * y + c * z - 2
    fx, fy, fz = (1 + a * x), (1 + b * y), (1 + c * z)
    out[:8, 0] = 0.125 * a * fy * fz * (s + fx)
    out[:8, 1] = 0.125 * b * fx * fz * (s + fy)
    out[:8, 2] = 0.125 * c * fx * fy * (s + fz)
    M = _HEX20_MID
    for k in range(12):
        mx, my, mz = M[k]
        i = 8 + k
        if mx == 0:
            out[i] = [-0.5 * x * (1 + my * y) * (1 + mz * z), 0.25 * (1 - x * x) * my * (1 + mz * z),
                      0.25 * (1 - x * x) * (1 + my * y) * mz]
        elif my == 0:
            out[i] = [0.25 * mx * (1 - y * y) * (1 + mz * z), -0.5 * y * (1 + mx * x) * (1 + mz * z),
                      0.25 * (1 + mx * x) * (1 - y * y) * mz]
        else:
            out[i] = [0.25 * mx * (1 + my * y) * (1 - z * z), 0.25 * (1 + mx * x) * my * (1 - z * z),
                      -0.5 * z * (1 + mx * x) * (1 + my * y)]
    return out


def _gauss_cube(n):
    g = {2: _G2, 3: _G3}[n]
    pts = np.array([[a, b, c] for a in g[0] for b in g[0] for c in g[0]])
    wts = np.array([wa * wb * wc for wa in g[1] for wb in g[1] for wc in g[1]])
    return pts, wts


def hex_quadrature(nn, full=False):
    """剛性用: Hex20 は 2×2×2 の低減積分（Ansys の SOLID186 の既定と同じ）、Hex8 は 2×2×2。
    full=True は質量・体積力用の 3×3×3"""
    return _gauss_cube(3 if full else 2)


def hex_node_points(nn):
    return HEX_CORNERS.copy() if nn == 8 else np.vstack([HEX_CORNERS, _HEX20_MID])


def hex_mass_ref(nn):
    pts, wts = hex_quadrature(nn, full=True)
    C = np.zeros((nn, nn))
    for p, w in zip(pts, wts):
        N = hex_N(p, nn)
        C += w * np.outer(N, N)
    return C / wts.sum()


def hex_volumes(nodes, elems):
    nn = elems.shape[1]
    pts, wts = hex_quadrature(nn, full=True)
    X = nodes[elems]
    V = np.zeros(len(elems))
    for p, w in zip(pts, wts):
        J = np.einsum("enk,nl->ekl", X, hex_dN(p, nn))
        V += w * np.linalg.det(J)
    return V


# ---------------------------------------------------------------------------- 縮退した六面体（Ansys の SOLID186 の
# ピラミッド・三角柱・四面体の形と同じ考え方）: 六面体優勢メッシュの要素をすべて Hex20 の配列で扱う
# gmsh の節点順（2 次・中間節点あり）から Hex20（VTK 順）の 20 個の節点番号へ
DEGEN_TET10 = [0, 1, 2, 2, 3, 3, 3, 3, 4, 5, 2, 6, 3, 3, 3, 3, 7, 9, 8, 8]          # gmsh tet10: 4(0,1) 5(1,2) 6(0,2) 7(0,3) 8(2,3) 9(1,3)
DEGEN_WEDGE15 = [0, 1, 2, 2, 3, 4, 5, 5, 6, 9, 2, 7, 12, 14, 5, 13, 8, 10, 11, 11]  # gmsh prism15
DEGEN_PYR13 = [0, 1, 2, 3, 4, 4, 4, 4, 5, 8, 10, 6, 4, 4, 4, 4, 7, 9, 11, 12]       # gmsh pyramid13
DEGEN_TET4 = [0, 1, 2, 2, 3, 3, 3, 3]
DEGEN_WEDGE6 = [0, 1, 2, 2, 3, 4, 5, 5]
DEGEN_PYR5 = [0, 1, 2, 3, 4, 4, 4, 4]

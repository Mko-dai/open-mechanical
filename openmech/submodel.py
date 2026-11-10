"""サブモデル（Ansys の「サブモデリング」と同じ考え方）

粗いモデル（全体）の結果の変位を、細かいモデル（一部分）の切断境界の節点に内挿して
強制変位として与える。 粗いモデルの結果は save_coarse() で .npz に保存しておく。

    1. 全体モデルを粗いメッシュで解く → save_coarse("coarse.npz", mesh, result.u)
    2. 注目部分だけのモデルを細かいメッシュで作り、切断面に
       BC("submodel", 切断面, options={"file": "coarse.npz"}) を置いて解く

切断境界は応力集中から十分離れた位置に置くこと（サン・ブナンの原理）。
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .solver import shape_N, shape_dN_dxi


def save_coarse(path, mesh, u, temperature=None):
    """粗いモデルの結果（節点・要素・変位、あれば温度）を保存"""
    d = {"nodes": np.asarray(mesh.nodes, float), "elems": np.asarray(mesh.elems, np.int64),
         "u": np.asarray(u, float).reshape(-1, 3)}
    if temperature is not None:
        d["temperature"] = np.asarray(temperature, float)
    np.savez_compressed(path, **d)


def _L_of(xi, nn):
    if nn in (4, 10):
        return np.array([1.0 - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]])
    return np.asarray(xi, float)


def _outside(xi, nn):
    """自然座標が要素の外にはみ出している量（0 = 内側）"""
    if nn in (4, 10):
        L = _L_of(xi, nn)
        return float(max(0.0, -L.min()))
    if nn in (6, 15):
        r, s, t = xi
        return float(max(0.0, -r, -s, r + s - 1.0, abs(t) - 1.0))
    return float(max(0.0, np.abs(xi).max() - 1.0))


def _clamp(xi, nn):
    if nn in (4, 10):
        L = np.maximum(_L_of(xi, nn), 0.0)
        L /= L.sum()
        return L[1:]
    if nn in (6, 15):
        r, s, t = max(xi[0], 0.0), max(xi[1], 0.0), float(np.clip(xi[2], -1, 1))
        if r + s > 1:
            r, s = r / (r + s), s / (r + s)
        return np.array([r, s, t])
    return np.clip(xi, -1.0, 1.0)


def _start(nn):
    if nn in (4, 10):
        return np.full(3, 0.25)
    if nn in (6, 15):
        return np.array([1 / 3, 1 / 3, 0.0])
    return np.zeros(3)


def _invert(Xe, x, nn, iters=12):
    """要素の節点座標 Xe (nn,3) で x になる自然座標（ニュートン法）"""
    xi = _start(nn)
    for _ in range(iters):
        L = _L_of(xi, nn)
        N = shape_N(L, nn)
        r = x - N @ Xe
        J = Xe.T @ shape_dN_dxi(L, nn)       # dx/dξ (3,3)
        try:
            dxi = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            break
        xi = xi + dxi
        if np.abs(dxi).max() < 1e-12:
            break
    return xi


class CoarseField:
    """粗いモデルの結果を、任意の点で内挿する"""

    def __init__(self, nodes, elems, u, temperature=None):
        self.nodes = np.asarray(nodes, float)
        self.elems = np.asarray(elems, np.int64)
        self.u = np.asarray(u, float).reshape(-1, 3)
        self.temperature = None if temperature is None else np.asarray(temperature, float)
        self.nn = self.elems.shape[1]
        nc = {4: 4, 10: 4, 6: 6, 15: 6, 8: 8, 20: 8}[self.nn]
        corners = self.nodes[self.elems[:, :nc]]
        self.cent = corners.mean(1)
        self.size = np.linalg.norm(corners - self.cent[:, None], axis=2).max(1)
        self.tree = cKDTree(self.cent)

    @staticmethod
    def load(path):
        z = np.load(path)
        return CoarseField(z["nodes"], z["elems"], z["u"], z["temperature"] if "temperature" in z.files else None)

    def locate(self, points, k=12):
        """各点の (要素番号, 自然座標, はみ出し量 [mm])"""
        P = np.atleast_2d(np.asarray(points, float))
        k = min(k, len(self.elems))
        _, cand = self.tree.query(P, k=k)
        cand = np.atleast_2d(cand).reshape(len(P), k)
        out_e = np.zeros(len(P), np.int64)
        out_xi = np.zeros((len(P), 3))
        out_d = np.zeros(len(P))
        for i, x in enumerate(P):
            best = (np.inf, 0, None)
            for e in cand[i]:
                if np.linalg.norm(x - self.cent[e]) > 2.0 * self.size[e] + 1e-12 and best[2] is not None:
                    continue
                Xe = self.nodes[self.elems[e]]
                xi = _invert(Xe, x, self.nn)
                o = _outside(xi, self.nn)
                if o < best[0]:
                    best = (o, e, xi)
                if o <= 1e-9:
                    break
            o, e, xi = best
            if o > 1e-9:
                xi = _clamp(xi, self.nn)
                Xe = self.nodes[self.elems[e]]
                o_mm = float(np.linalg.norm(shape_N(_L_of(xi, self.nn), self.nn) @ Xe - x))
            else:
                o_mm = 0.0
            out_e[i], out_xi[i], out_d[i] = e, xi, o_mm
        return out_e, out_xi, out_d

    def interpolate(self, points, field=None):
        """点での値（既定は変位 (n,3)）と、粗いモデルの外にはみ出した距離 (n,)"""
        f = self.u if field is None else field
        e, xi, d = self.locate(points)
        vals = np.array([shape_N(_L_of(xi[i], self.nn), self.nn) @ f[self.elems[e[i]]] for i in range(len(e))])
        return vals, d


_CACHE: dict = {}


def coarse_field(path) -> CoarseField:
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"サブモデル: 粗いモデルの結果ファイルがありません: {path}")
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = CoarseField.load(path)
    return _CACHE[key]


def cut_boundary_values(mesh, bc, nodes):
    """サブモデルの切断境界の節点 nodes に与える変位 (n,3)。 options: file, offset（座標のずれ [mm]）"""
    opts = bc.options or {}
    cf = coarse_field(opts.get("file", ""))
    off = np.asarray(opts.get("offset", [0.0, 0.0, 0.0]), float)
    X = mesh.nodes[nodes] + off
    vals, d = cf.interpolate(X)
    tol = 0.05 * float(np.median(cf.size))
    far = d > max(tol, 1e-6)
    if np.any(far):
        frac = far.mean()
        if frac > 0.2:
            raise ValueError(f"サブモデル: 切断境界の節点の {frac:.0%} が粗いモデルの外にあります"
                             f"（最大 {d.max():.3g} mm）。座標系・単位・切断面の位置を確認してください。")
    bc._submodel_far = float(d.max()) if len(d) else 0.0
    return vals

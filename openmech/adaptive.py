"""適応メッシュ（Ansys の「収束」ツールに相当）。

1. 解く → 2. 応力の誤差を要素ごとに見積もる（ZZ 法: 節点平均した応力と要素の応力の差のエネルギー）
3. 誤差の大きい要素の場所だけ要素サイズを半分にしてメッシュを切り直す → 4. 見たい結果（最大応力など）の
変化が許容値（既定 5 %）以下になるまで繰り返す。

使い方: converge(project, field="相当応力（ミーゼス） [MPa]", stat="max", allowable=5.0, max_loops=4)
"""
from __future__ import annotations

import time

import numpy as np

from .solver import _B_matrix, _D_of, _dN_dx, material_D, node_points, shape_dN_dxi


def element_errors(mesh, u, D, elem_mat=None, eps_th_m=None):
    """要素ごとの誤差 e_e（エネルギーノルム、√(N·mm)）と、全体のひずみエネルギー U。
    e_e² = ∫ (σ* − σ_h)ᵀ D⁻¹ (σ* − σ_h) dV（σ*: 節点平均の応力、σ_h: 要素の応力。要素の節点で評価）"""
    nn = mesh.elems.shape[1]
    Lp = node_points(nn)
    U = u.reshape(-1, 3)
    X = mesh.nodes[mesh.elems]
    ue = U[mesh.elems].reshape(mesh.n_elems, -1)
    D = np.asarray(D, float)
    if D.ndim == 3 and elem_mat is None:
        elem_mat = np.zeros(mesh.n_elems, int)
    De = _D_of(D, elem_mat, slice(0, mesh.n_elems))
    sig = np.empty((mesh.n_elems, nn, 6))
    for a in range(nn):
        dNdx, _ = _dN_dx(X, shape_dN_dxi(Lp[a], nn))
        eps = np.einsum("eij,ej->ei", _B_matrix(dNdx), ue)
        if eps_th_m is not None:
            eps = eps - eps_th_m[elem_mat if elem_mat is not None else 0]
        sig[:, a] = eps @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps)
    idx = mesh.elems.ravel()
    cnt = np.maximum(np.bincount(idx, minlength=mesh.n_nodes), 1)
    star = np.column_stack([np.bincount(idx, sig[..., k].ravel(), mesh.n_nodes) for k in range(6)]) / cnt[:, None]
    diff = star[mesh.elems] - sig                                  # (E, nn, 6)
    C = np.linalg.inv(De) if De.ndim == 2 else np.linalg.inv(De)
    if C.ndim == 2:
        e2 = np.einsum("eai,ij,eaj->e", diff, C, diff)
        w = np.einsum("eai,ij,eaj->e", sig, C, sig)
    else:
        e2 = np.einsum("eai,eij,eaj->e", diff, C, diff)
        w = np.einsum("eai,eij,eaj->e", sig, C, sig)
    vol = np.abs(mesh.element_volumes())
    err = np.sqrt(np.maximum(e2, 0) / nn * vol)
    energy = float((w / nn * vol).sum())
    return err, energy


def refine_points(mesh, err, frac=0.3, factor=0.5, h_min=None):
    """誤差の大きい要素（誤差の 2 乗の合計の frac を占める上位）の重心に、今の大きさ × factor のサイズ点"""
    vol = np.abs(mesh.element_volumes())
    h = np.cbrt(vol * 6 * np.sqrt(2))                              # 正四面体の辺の長さに相当
    o = np.argsort(-err)
    c = np.cumsum(err[o] ** 2)
    k = int(np.searchsorted(c, frac * c[-1])) + 1
    sel = o[:max(1, k)]
    cen = mesh.nodes[mesh.elems[sel, :4]].mean(axis=1)
    hs = h[sel] * factor
    if h_min:
        hs = np.maximum(hs, h_min)
    return np.column_stack([cen, hs])


def _merge(old, new, tol=None):
    """以前のサイズ点と合わせる（近い点は小さい方を残す）"""
    if old is None or not len(old):
        return new
    from scipy.spatial import cKDTree
    old = np.asarray(old, float).reshape(-1, 4)
    allp = np.vstack([old, new])
    tree = cKDTree(allp[:, :3])
    keep = np.ones(len(allp), bool)
    order = np.argsort(allp[:, 3])                                 # 小さいサイズを優先
    for i in order:
        if not keep[i]:
            continue
        for j in tree.query_ball_point(allp[i, :3], r=0.5 * allp[i, 3]):
            if j != i and allp[j, 3] >= allp[i, 3]:
                keep[j] = False
    return allp[keep]


def converge(project, field="相当応力（ミーゼス） [MPa]", stat="max", allowable=5.0, max_loops=4, frac=0.3,
             log=print, progress=None, cancel=None, mesh_fn=None):
    """適応メッシュで収束させる。mesh_fn() を渡すとそれでメッシュを作る。戻り値: (最後のメッシュ, 最後の結果のリスト, 履歴)"""
    from .parametric import evaluate_outputs, output_label
    p = project
    ms = dict(p.mesh_settings or {})
    pts = np.asarray(ms.get("size_points") or [], float).reshape(-1, 4)
    hist = []
    prev = None
    mesh = res = None
    out = {"field": field, "stat": stat}
    lab = output_label(out)
    for k in range(max_loops + 1):
        if cancel is not None and cancel():
            break
        t0 = time.time()
        p.mesh_settings = dict(ms, size_points=pts.tolist())
        mesh = mesh_fn() if mesh_fn else p.mesh(log=lambda *a: None)
        res = p.run_all(mesh, log=lambda *a: None)
        val = float(evaluate_outputs(p, mesh, res, [out])[lab])
        r = res[-1]
        em = p.elem_materials(mesh)
        mats, eidx = em if em is not None else ([p.material], np.zeros(mesh.n_elems, int))
        D = np.stack([material_D(m) for m in mats])
        err, U = element_errors(mesh, r.u.ravel(), D, np.asarray(eidx))
        rel = float(np.sqrt((err ** 2).sum() / max(U, 1e-300)) * 100)
        change = None if prev is None else abs(val - prev) / max(abs(val), 1e-300) * 100
        hist.append({"回": k + 1, "節点数": mesh.n_nodes, "要素数": mesh.n_elems, lab: val,
                     "変化 [%]": change, "推定誤差 [%]": rel, "時間 [s]": round(time.time() - t0, 1)})
        log(f"  収束 {k + 1} 回目: 要素 {mesh.n_elems:,} / {lab} = {val:.5g}"
            + (f"（変化 {change:.2f} %）" if change is not None else "") + f" / 推定誤差 {rel:.2f} %")
        if progress:
            progress(k + 1, max_loops + 1)
        if change is not None and change <= allowable:
            log(f"収束しました（変化 {change:.2f} % ≤ 許容 {allowable:g} %）")
            break
        if k == max_loops:
            log(f"最大 {max_loops} 回の細分化で収束しませんでした（最後の変化 {change:.2f} %）")
            break
        pts = _merge(pts, refine_points(mesh, err, frac=frac, h_min=float(ms.get("curvature_min_size") or 0) or None))
        prev = val
    p.mesh_settings = dict(ms, size_points=pts.tolist())
    return mesh, res, hist

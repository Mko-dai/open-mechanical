"""ボルト予張力（Ansys の「ボルト予張力」: ステップ 1 で予張力をかけ、ステップ 2 で長さを固定して外力をかける）。

ボルトのボディを、指定した点を通り軸に垂直な断面で考える。断面をまたぐ要素に軸方向の初期ひずみ ε₀（縮み）を与え、
予張力だけをかけたとき断面を通る力が指定値になるよう ε₀ を決める（線形なら 1 回、接触などの非線形なら割線法で数回）。
ステップ 2 では ε₀ を固定（Ansys の「ロック」と同じ: 調整量を保ったまま外力をかける）。
断面の力は、断面をまたぐ要素の内力のうち断面の＋側の節点に働く成分の和（要素の節点力による断面力）。
"""
from __future__ import annotations

import copy

import numpy as np

from .solver import _B_matrix, _D_of, _dN_dx, _quadrature, material_D, shape_dN_dxi


def strain_vec(a):
    a = np.asarray(a, float) / np.linalg.norm(a)
    return np.array([a[0] ** 2, a[1] ** 2, a[2] ** 2, 2 * a[0] * a[1], 2 * a[1] * a[2], 2 * a[0] * a[2]])


def bolt_geometry(mesh, body: int, point=None, axis=None):
    """ボルトの軸・断面の点（未指定なら主軸と重心）と、断面をまたぐ要素"""
    sel = np.nonzero(mesh.elem_body == int(body))[0] if mesh.elem_body is not None else np.arange(mesh.n_elems)
    if len(sel) == 0:
        raise ValueError(f"ボルト予張力: ボディ {body} がありません")
    X = mesh.nodes[np.unique(mesh.elems[sel])]
    c = X.mean(axis=0)
    if axis is None or not np.any(axis):
        w, v = np.linalg.eigh(np.cov((X - c).T))
        axis = v[:, -1]
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    p = np.asarray(point, float) if point is not None and len(point) == 3 else c
    s = (mesh.nodes[mesh.elems[sel]] - p) @ a          # (E, nn)
    straddle = sel[(s.min(axis=1) < 0) & (s.max(axis=1) > 0)]
    if len(straddle) == 0:
        raise ValueError("ボルト予張力: 断面をまたぐ要素がありません（点・軸を確認してください）")
    return p, a, straddle


def section_force(mesh, u, D, elem_mat, eps_th_m, elems, p, a):
    """要素 elems（断面をまたぐ）の内力のうち、断面の＋側の節点に働く力の軸方向成分の和 [N]（引張 正）"""
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    el = mesh.elems[elems]
    X = mesh.nodes[el]
    U = u.reshape(-1, 3)[el].reshape(len(el), -1)
    De = _D_of(D, elem_mat, elems) if D.ndim == 3 else D
    fe = np.zeros((len(el), 3 * nn))
    for L, w in zip(pts, wts):
        dNdx, detJ = _dN_dx(X, shape_dN_dxi(L, nn))
        B = _B_matrix(dNdx)
        eps = np.einsum("eij,ej->ei", B, U)
        if eps_th_m is not None:
            eps = eps - eps_th_m[elem_mat[elems]]
        sig = np.einsum("eij,ej->ei", De, eps) if De.ndim == 3 else eps @ De.T
        fe += np.einsum("e,eki,ek->ei", w * np.abs(detJ), B, sig)
    fe = fe.reshape(len(el), nn, 3)
    plus = ((X - p) @ a) > 0
    # 要素の内力 ∫Bᵀσ dV の＋側の節点の成分: 引張なら +a 向き（＝断面を通る引張力）
    return float(np.einsum("enk,k->", fe * plus[:, :, None], a))


def with_pretension(mats, elem_mat, straddle, a, eps0):
    """断面をまたぐ要素に初期ひずみ eps0（軸方向・縮みが負）を足した材料リスト"""
    mats = [m if hasattr(m, "Dm") else m.state() for m in mats]
    out = list(mats)
    em = np.array(elem_mat, dtype=np.int64, copy=True)
    cache = {}
    sv = strain_vec(a) * eps0
    for e in straddle:
        k = int(em[e])
        if k not in cache:
            st = copy.copy(mats[k])
            st.eps_th = np.asarray(getattr(mats[k], "eps_th", np.zeros(6)), float) + sv
            st.name = f"{mats[k].name}（予張力）"
            cache[k] = len(out)
            out.append(st)
        em[e] = cache[k]
    return out, em


def solve_pretension(mesh, material, bcs, solve_fn, elem_materials=None, log=print, max_iter=8, tol=1e-3):
    """bcs の中の bolt_pretension を扱って解く。 solve_fn(bcs, elem_materials) → Result（静解析）。
    戻り値: [ステップ 1（予張力のみ）, ステップ 2（ロック＋外力）]"""
    bolts = [b for b in bcs if b.kind == "bolt_pretension"]
    others = [b for b in bcs if b.kind != "bolt_pretension"]
    if elem_materials is not None:
        mats, elem_mat = elem_materials
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    geo = []
    for b in bolts:
        o = b.options or {}
        p, a, st = bolt_geometry(mesh, int(o.get("body", 0)), b.point or o.get("point"), o.get("axis"))
        geo.append((b, p, a, st, float(b.values[0])))
        log(f"ボルト予張力: ボディ {o.get('body', 0)}・断面の要素 {len(st):,}・予張力 {b.values[0]:g} N")
    supports_only = [b for b in others if b.kind in ("fixed", "displacement", "symmetry", "frictionless",
                                                     "remote_displacement", "elastic_support", "cylindrical_support",
                                                     "spring")]
    # 支持の強制変位はステップ 1 では 0 にする（Ansys のステップ 1 と同じく予張力だけ）
    step1_bcs = []
    for b in supports_only:
        if b.kind == "displacement":
            nb = copy.copy(b)
            nb.values = [None if v is None else 0.0 for v in b.values]
            step1_bcs.append(nb)
        else:
            step1_bcs.append(b)

    def build(eps):
        mm, em = mats, np.asarray(elem_mat)
        for (b, p, a, st, F), e in zip(geo, eps):
            mm, em = with_pretension(mm, em, st, a, e)
        return mm, em

    def forces(r, mm, em):
        D = np.stack([material_D(m) for m in mm])
        epsm = np.stack([np.asarray(getattr(m, "eps_th", np.zeros(6)), float) for m in mm])
        return np.array([section_force(mesh, r.u.ravel(), D, em, epsm, st, p, a) for (b, p, a, st, F) in geo])

    target = np.array([g[4] for g in geo])
    E0 = np.array([float(getattr(mats[int(elem_mat[g[3][0]])], "E", 2e5)) for g in geo])
    eps = -target / (E0 * 50.0)               # 初期値（断面積 ~50 mm² 相当の目安）
    r1 = None
    prev = None
    for it in range(max_iter):
        mm, em = build(eps)
        r1 = solve_fn(step1_bcs, (mm, em))
        Fs = forces(r1, mm, em)
        err = np.abs(Fs - target) / np.maximum(np.abs(target), 1e-9)
        log(f"  予張力の調整 {it + 1}: 断面力 " + ", ".join(f"{f:,.4g} N" for f in Fs))
        if np.all(err < tol):
            break
        if prev is None:
            slope = Fs / np.where(eps != 0, eps, 1e-30)
        else:
            de = eps - prev[0]
            slope = np.where(np.abs(de) > 0, (Fs - prev[1]) / np.where(de != 0, de, 1), Fs / eps)
        prev = (eps.copy(), Fs.copy())
        eps = eps + (target - Fs) / np.where(slope != 0, slope, 1e-30)
    r1.info["ステップ"] = 1
    r1.info["解析"] = "ボルト予張力（ステップ 1）"
    r1.time = 1.0
    mm, em = build(eps)
    r2 = solve_fn(others, (mm, em))
    F2 = forces(r2, mm, em)
    r2.info["ステップ"] = 2
    r2.info["解析"] = "ボルト予張力（ステップ 2: ロック＋外力）"
    r2.time = 2.0
    for k, (g, f1, f2) in enumerate(zip(geo, forces(r1, *build(eps)), F2)):
        r1.info[f"ボルト {k + 1} の力 [N]"] = round(float(f1), 3)
        r2.info[f"ボルト {k + 1} の力 [N]"] = round(float(f2), 3)
        r2.info[f"ボルト {k + 1} の調整ひずみ"] = float(eps[k])
    return [r1, r2]


def has_beam_pretension(bcs) -> bool:
    return any(b.kind == "beam" and len(b.values) > 3 and b.values[3] for b in bcs)


def solve_beam_pretension(bcs, solve_fn, log=print):
    """ビーム接続の予張力（Ansys の Bolt Pretension をビームにかけた場合と同じ 2 ステップ）。
    solve_fn(bcs) → Result（r.beams に断面力）。 戻り値: [ステップ 1（予張力のみ）, ステップ 2（ロック＋外力）]"""
    support = ("fixed", "displacement", "symmetry", "frictionless", "remote_displacement", "elastic_support",
               "cylindrical_support", "spring", "joint", "cyclic", "compression_only")

    def with_mode(b, **pre):
        nb = copy.copy(b)
        nb.options = dict(b.options or {}, _pre=pre)
        return nb
    step1, idx1 = [], {}
    for i, b in enumerate(bcs):
        if b.kind == "beam":
            idx1[i] = len(step1)
            step1.append(with_mode(b, mode="preload") if (len(b.values) > 3 and b.values[3]) else b)
        elif b.kind == "displacement":
            nb = copy.copy(b)
            nb.values = [None if v is None else 0.0 for v in b.values]
            step1.append(nb)
        elif b.kind in support:
            step1.append(b)
    r1 = solve_fn(step1)
    b1 = getattr(r1, "beams", {})
    step2 = []
    for i, b in enumerate(bcs):
        if b.kind == "beam" and len(b.values) > 3 and b.values[3]:
            du1 = float(b1[idx1[i]]["伸び [mm]"])
            log(f"ビームの予張力: {b.label()} … ステップ 1 の軸力 {b1[idx1[i]]['軸力 [N]']:,.5g} N（伸び {du1:.4g} mm）")
            step2.append(with_mode(b, mode="locked", du1=du1))
        else:
            step2.append(b)
    r2 = solve_fn(step2)
    r1.beams = {i: b1[j] for i, j in idx1.items()}          # 番号を元の境界条件の並びに合わせる
    for r, k, name in ((r1, 1, "ビームの予張力（ステップ 1）"), (r2, 2, "ビームの予張力（ステップ 2: ロック＋外力）")):
        r.info["ステップ"], r.info["解析"], r.time = k, name, float(k)
        for i, br in getattr(r, "beams", {}).items():
            r.info[f"ビーム {i + 1} の軸力 [N]"] = round(float(br["軸力 [N]"]), 3)
    return [r1, r2]

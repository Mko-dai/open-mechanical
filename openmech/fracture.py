"""破壊力学: J 積分（領域積分法）と応力拡大係数

Ansys の「破壊ツール」（Fracture Tool の J-Integral / SIFS）に相当。き裂先端（き裂前縁）を線分で与え、
前縁からの距離 r で q 関数（r ≤ r_in で 1、r_out で 0 に直線で下がる）を作り、

  J = −∫_V ( W δ_1i − σ_ij ∂u_j/∂x_1 ) ∂q/∂x_i dV / ∫_前縁 q ds

（x_1 = き裂の進む向き）を求める（Shih–Moran–Nakamura の領域積分。線形弾性で経路独立）。
K は J から K = √(J E')（E' = E/(1−ν²)：平面ひずみ・3D の前縁中央、平面応力なら E）、
モードの分離は前縁近くの開口変位から（変位外挿法、Ansys の KCALC と同じ考え方）。
対称モデル（き裂面が対称面の半分モデル）では J は 2 倍、K はそのまま（開口量を 2 倍）にする。
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .solver import _B_matrix, _D_of, _dN_dx, _quadrature, shape_dN_dxi


def _frame(front0, front1, direction):
    p0, p1 = np.asarray(front0, float), np.asarray(front1, float)
    t = p1 - p0
    L = float(np.linalg.norm(t))
    t = t / L if L > 0 else np.array([0.0, 0.0, 1.0])
    e1 = np.asarray(direction, float)
    e1 = e1 - (e1 @ t) * t
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)              # き裂面の法線（e1 × e2 = t の右手系）
    return p0, p1, L, t, e1, e2


def _dist_to_front(X, p0, p1, t, L):
    s = np.clip((X - p0) @ t, 0.0, L) if L > 0 else np.zeros(len(X))
    return np.linalg.norm(X - (p0 + s[:, None] * t), axis=1)


def j_integral(mesh: Mesh, u, D, front0, front1, direction, r_in, r_out, elem_mat=None, symmetric=False):
    """J [N/mm = kJ/m²·10⁻³…]。単位系 mm・N では J の単位は N/mm（= kJ/m²）"""
    p0, p1, L, t, e1, e2 = _frame(front0, front1, direction)
    X = mesh.nodes
    r = _dist_to_front(X, p0, p1, t, L)
    q = np.clip((r_out - r) / max(r_out - r_in, 1e-30), 0.0, 1.0)
    # 前縁の端（自由表面）は q を 0 にしない（前縁全長での平均の J）
    el = mesh.elems
    act = np.nonzero((q[el] > 0).any(1) & (q[el] < 1).any(1))[0]
    if len(act) == 0:
        raise ValueError("J 積分の領域に要素がありません（r_in・r_out を要素サイズより大きくしてください）")
    nn = el.shape[1]
    pts, wts = _quadrature(nn)
    U = np.asarray(u, float).reshape(-1, 3)
    E_ = el[act]
    Xe, Ue, qe = X[E_], U[E_], q[E_]
    De = _D_of(np.asarray(D, float), None if elem_mat is None else np.asarray(elem_mat)[act], slice(None)) \
        if np.ndim(D) == 3 else np.asarray(D, float)
    Jsum = 0.0
    for L_, w in zip(pts, wts):
        dNdx, detJ = _dN_dx(Xe, shape_dN_dxi(L_, nn))
        gu = np.einsum("eaj,eak->ejk", Ue, dNdx)            # ∂u_j/∂x_k
        gq = np.einsum("ea,eak->ek", qe, dNdx)
        eps = np.einsum("eij,ej->ei", _B_matrix(dNdx), Ue.reshape(len(act), -1))
        sig = eps @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps)
        W = 0.5 * np.einsum("ei,ei->e", sig, eps)
        S = np.stack([sig[:, [0, 3, 5]], sig[:, [3, 1, 4]], sig[:, [5, 4, 2]]], axis=1)
        du1 = gu @ e1                                         # ∂u_j/∂x_1
        integrand = np.einsum("eij,ej,ei->e", S, du1, gq) - W * (gq @ e1)
        Jsum += float(np.sum(integrand * w * detJ))
    J = Jsum / max(L, 1e-30) if L > 0 else Jsum
    return 2 * J if symmetric else J


def k_from_j(J, E, nu, plane_stress=False):
    Ep = E if plane_stress else E / (1 - nu ** 2)
    return float(np.sqrt(max(J, 0.0) * Ep))


def k_displacement(mesh: Mesh, u, E, nu, front0, front1, direction, r_max, symmetric=False, plane_stress=False,
                   quarter=False):
    """変位外挿法: 前縁の後ろ（き裂面上 θ = ±π）の開口・ずれ変位から K_I, K_II, K_III [MPa√mm]
    K_I = G √(2π/r) Δu_2 /(κ+1)（Δu: 上下面の開口量）を r → 0 に直線外挿"""
    p0, p1, L, t, e1, e2 = _frame(front0, front1, direction)
    X = mesh.nodes
    rel = X - p0
    x1, x2 = rel @ e1, rel @ e2
    s = rel @ t
    on_face = (x1 < -1e-9) & (x1 > -r_max) & (np.abs(x2) < 1e-6 * max(r_max, 1.0)) & (s >= -1e-9) & (s <= L + 1e-9)
    idx = np.nonzero(on_face)[0]
    if len(idx) < 2:
        raise ValueError("き裂面の上に節点が見つかりません（前縁・進展方向・r_max を確認してください）")
    U = np.asarray(u, float).reshape(-1, 3)
    G = E / (2 * (1 + nu))
    kappa = (3 - nu) / (1 + nu) if plane_stress else 3 - 4 * nu
    # 上下の面で節点の対を作る（対称モデルなら上面だけ、開口量は 2 倍）
    if symmetric:
        r = -x1[idx]
        d = 2 * U[idx]
    else:
        Xi = X[idx]
        key = np.round(np.c_[x1[idx], s[idx]] / (1e-6 * max(r_max, 1.0))).astype(np.int64)
        order = np.lexsort(key.T[::-1])
        pairs = []
        for a, b in zip(order[:-1], order[1:]):
            if np.all(key[a] == key[b]) and np.linalg.norm(Xi[a] - Xi[b]) < 1e-6 * max(r_max, 1.0):
                pairs.append((idx[a], idx[b]))
        if not pairs:
            raise ValueError("き裂面の上下の節点の対が見つかりません（き裂面は節点を共有しない必要があります）")
        pa = np.array(pairs)
        # どちらが上側か: 隣の要素の重心で判定する代わりに、開口が正になる向きをとる
        r = -x1[pa[:, 0]]
        d = U[pa[:, 0]] - U[pa[:, 1]]
        sgn = np.sign(np.median(d @ e2)) or 1.0
        d *= sgn
    f = np.sqrt(2 * np.pi / np.maximum(r, 1e-30)) * G / (1 + kappa)     # Δu = K (κ+1)/G √(r/2π)
    rmin = float(r.min())
    if quarter:
        # 1/4 点要素: 先端の要素の外側（1〜4 要素目）の値がほぼ一定になるので、それを r の 1 次式で外挿する
        # （1/4 点の節点そのものは要素のゆがみの影響が大きい。実測で 2 点式 KCALC より誤差が小さい）
        keep = (r >= 4 * rmin * (1 - 1e-6)) & (r <= 16 * rmin * (1 + 1e-6))
    else:
        # 先端の 2 要素以内は通常の 2 次要素では √r の変位を表せず低く出るので使わない
        keep = r >= 4 * rmin - 1e-12
    if keep.sum() >= 2:
        r, d, f = r[keep], d[keep], f[keep]
    out = []
    for comp in ((e2,) if symmetric else (e2, e1, t)):
        k = f * (d @ comp)
        if len(r) >= 2:
            A = np.c_[np.ones_like(r), r]
            c, *_ = np.linalg.lstsq(A, k, rcond=None)
            out.append(float(c[0]))
        else:
            out.append(float(k[0]))
    if symmetric:                              # 対称面のき裂はモード I のみ
        return out[0], 0.0, 0.0
    KI, KII, KIII = out
    KIII = KIII * (1 + kappa) / 4.0            # モード III は 2G 型の係数が異なる: K_III = G √(2π/r) Δu_3 / 2
    return KI, KII, KIII


def insert_crack(mesh: Mesh, point, normal, inside, tol=None):
    """平面き裂を入れる（き裂面の節点を複製して、法線の + 側の要素を新しい節点につなぎ替える）。

    point, normal: き裂面（平面）。inside(X) → bool: き裂の内側の節点（前縁の節点は False にする）。
    き裂面は要素の面と一致している必要がある（構造格子・押し出しメッシュ、または gmsh で面を埋め込んだメッシュ）。
    き裂面は新しい面番号（− 側・+ 側の 2 つ）になる。戻り値: (新しいメッシュ, (面番号 −, 面番号 +))
    """
    X = mesh.nodes
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    p = np.asarray(point, float)
    if tol is None:
        tol = 1e-6 * float(np.ptp(X, axis=0).max())
    h = (X - p) @ n
    on = np.abs(h) < tol
    dup = np.nonzero(on & np.asarray(inside(X), bool))[0]
    if len(dup) == 0:
        raise ValueError("き裂面の上に節点がありません")
    nc = mesh.elems.shape[1]
    corner = mesh.elems[:, :4] if nc in (4, 10) else mesh.elems[:, :8] if nc in (8, 20) else mesh.elems[:, :6]
    cen = X[corner].mean(axis=1)
    plus = ((cen - p) @ n) > 0
    lut = np.arange(mesh.n_nodes)
    lut[dup] = mesh.n_nodes + np.arange(len(dup))
    elems = mesh.elems.copy()
    elems[plus] = lut[mesh.elems[plus]]
    out = Mesh(np.vstack([X, X[dup]]), elems)
    out.elem_body = None if mesh.elem_body is None else mesh.elem_body.copy()
    out.body_names = mesh.body_names
    origin = np.r_[np.arange(mesh.n_nodes), dup]
    out.crack_origin = origin
    no = getattr(mesh, "node_origin", None)
    if no is not None:
        out.node_origin = np.asarray(no)[origin]
    for k in ("fix_log", "crack_faces"):
        if hasattr(mesh, k):
            setattr(out, k, getattr(mesh, k))
    # 面番号: 元の境界三角形と同じ節点（元の番号で）なら同じ面番号、新しくできた境界（き裂面）は新しい番号
    old_key = {tuple(sorted(t)): f for t, f in zip(mesh.boundary_tris.tolist(),
                                                   mesh.face_ids.tolist() if mesh.face_ids is not None else
                                                   mesh.group_faces().tolist())}
    nf = mesh.n_faces
    fid = []
    nrm, _ = out.tri_normals_areas()
    for k, t in enumerate(out.boundary_tris.tolist()):
        key = tuple(sorted(origin[t]))
        if key in old_key:
            fid.append(old_key[key])
        else:                                                # き裂で新しくできた面
            fid.append(nf if nrm[k] @ n > 0 else nf + 1)     # − 側の要素の面（外向き = +n）/ + 側
    out.set_face_ids(fid)
    return out, (nf, nf + 1)


def evaluate_cracks(mesh: Mesh, result, cracks, h_ref=None):
    """き裂ごとに J（5 つの積分路）と K を求める（Ansys の破壊ツールの結果に相当）。戻り値: 行の辞書のリスト"""
    from .geometry import crack_frame
    from .solver import material_D
    mats = getattr(result, "mats", None) or [result.material]
    em = getattr(result, "elem_mat", None)
    D = np.stack([material_D(m) for m in mats])
    em = np.zeros(mesh.n_elems, int) if em is None else np.asarray(em)
    u = np.asarray(result.u, float).ravel()
    rows = []
    vol = np.abs(mesh.element_volumes())
    for i, c in enumerate(cracks):
        if c.get("suppressed"):
            continue
        p0, p1, t, e1, n, L = crack_frame(c)
        # 前縁の近くの要素の大きさ
        X = mesh.nodes
        cen = X[mesh.elems[:, :4]].mean(1)
        r = _dist_to_front(cen, p0, p1, t, L)
        near = np.argsort(r)[:max(8, min(64, len(r)))]
        hf = float(c.get("front_size") or 0) or float(np.cbrt(np.median(vol[near]) * 6 * np.sqrt(2)))
        m0 = mats[int(np.bincount(em[near]).argmax())]
        Js = []
        for k in range(1, 6):
            Js.append(j_integral(mesh, u, D, p0, p1, e1, (k - 1) * hf, (k + 1) * hf, elem_mat=em,
                                 symmetric=bool(c.get("symmetric"))))
        J = float(np.mean(Js[1:]))
        row = {"き裂": c.get("name") or f"き裂 {i + 1}", "J [N/mm]": J,
               **{f"J 積分路 {k + 1}": float(v) for k, v in enumerate(Js)},
               "K（J から）[MPa√mm]": k_from_j(J, m0.E, m0.nu)}
        try:
            KI, KII, KIII = k_displacement(mesh, u, m0.E, m0.nu, p0, p1, e1, 10 * hf,
                                           symmetric=bool(c.get("symmetric")),
                                           quarter=bool(getattr(mesh, "quarter_point", False)))
            row.update({"K_I（変位）[MPa√mm]": KI, "K_II（変位）[MPa√mm]": KII, "K_III（変位）[MPa√mm]": KIII})
        except ValueError as e:
            row["メモ"] = str(e)
        if c.get("toughness"):
            row["破壊靭性 K_IC [MPa√mm]"] = float(c["toughness"])
            row["安全率（K_IC / K）"] = float(c["toughness"]) / max(row["K（J から）[MPa√mm]"], 1e-30)
        rows.append(row)
    return rows


def quarter_point(mesh: Mesh, front0, front1, tol=None) -> int:
    """前縁に頂点がある 2 次要素の辺の中間節点を、前縁から 1/4 の位置へ動かす（1/4 点要素、Barsoum 1976）。
    要素の中で変位が √r、応力が 1/√r になり、き裂先端の特異性を表せる（Ansys のき裂メッシュと同じ）。
    戻り値: 動かした中間節点の数"""
    from .solver import _EDGES10
    if mesh.elems.shape[1] != 10:
        return 0
    p0, p1 = np.asarray(front0, float), np.asarray(front1, float)
    t = p1 - p0
    L = float(np.linalg.norm(t))
    t = t / L
    X = mesh.nodes
    if tol is None:
        tol = 1e-6 * float(np.ptp(X, axis=0).max())
    onf = _dist_to_front(X, p0, p1, t, L) < tol
    moved = set()
    for k, (i, j) in enumerate(_EDGES10):
        a_, b_, m_ = mesh.elems[:, i], mesh.elems[:, j], mesh.elems[:, 4 + k]
        for f, o in ((a_, b_), (b_, a_)):
            sel = onf[f] & ~onf[o]
            for fi, oi, mi in zip(f[sel], o[sel], m_[sel]):
                if int(mi) not in moved:
                    X[mi] = 0.75 * X[fi] + 0.25 * X[oi]
                    moved.add(int(mi))
    return len(moved)

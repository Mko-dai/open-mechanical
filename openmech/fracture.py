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
        if c.get("shape") == "ellipse":
            rows += _evaluate_ellipse(mesh, u, D, em, mats, vol, c, i)
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


def quarter_point(mesh: Mesh, front0, front1=None, tol=None) -> int:
    """前縁に頂点がある 2 次要素の辺の中間節点を、前縁から 1/4 の位置へ動かす（1/4 点要素、Barsoum 1976）。
    要素の中で変位が √r、応力が 1/√r になり、き裂先端の特異性を表せる（Ansys のき裂メッシュと同じ）。
    戻り値: 動かした中間節点の数"""
    from .solver import _EDGES10
    if mesh.elems.shape[1] != 10:
        return 0
    X = mesh.nodes
    if tol is None:
        tol = 1e-6 * float(np.ptp(X, axis=0).max())
    if front1 is None:                       # 折れ線の前縁（曲がった前縁）
        P = np.asarray(front0, float)
        cand = np.nonzero(np.linalg.norm(X - P.mean(0), axis=1) <= np.linalg.norm(np.ptp(P, axis=0)) + 10 * tol)[0]
        onf = np.zeros(len(X), bool)
        d = _project_polyline(X[cand], P)[0]
        # 折れ線は楕円の弦なので、弦と弧の差（中央で最大 L²/8R）まで許す
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1).max()
        onf[cand] = d < max(tol, 0.02 * seg)
    else:
        p0, p1 = np.asarray(front0, float), np.asarray(front1, float)
        t = p1 - p0
        L = float(np.linalg.norm(t))
        t = t / L
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


# ============================================================================ 曲がった前縁（半楕円の表面き裂など）

class CrackGeom:
    """き裂の形。kind="straight"（直線の前縁・長方形のき裂面）/ "ellipse"（楕円・半楕円の前縁、き裂面は楕円の内側）

    楕円: {"shape": "ellipse", "center": 中心, "normal": き裂面の法線, "axis_a": 深さ方向（a の向き）,
           "a": 深さ方向の半径, "c": 長さ方向の半径}。中心を表面に置けば半楕円の表面き裂、内部なら楕円の埋没き裂
    """

    def __init__(self, c: dict):
        self.c = c
        self.kind = c.get("shape", "straight")
        if self.kind == "ellipse":
            self.center = np.asarray(c["center"], float)
            n = np.asarray(c.get("normal", [0, 1, 0]), float)
            self.n = n / np.linalg.norm(n)
            ua = np.asarray(c.get("axis_a", [0, 0, -1]), float)
            ua = ua - (ua @ self.n) * self.n
            self.ua = ua / np.linalg.norm(ua)
            self.uc = np.cross(self.n, self.ua)
            self.a, self.cc = float(c["a"]), float(c["c"])
            th = np.linspace(0.0, 2 * np.pi, 361)
            self.ring = self.center + np.outer(self.cc * np.cos(th), self.uc) + np.outer(self.a * np.sin(th), self.ua)
            self.p0 = self.center
        else:
            from .geometry import crack_frame
            p0, p1, t, e1, n, L = crack_frame(c)
            self.n, self.p0, self.p1, self.t, self.e1, self.L = n, p0, p1, t, e1, L
            self.ring = np.vstack([p0, p1])

    def front(self, inside_body=None):
        """前縁の折れ線（物体の中の部分だけ。inside_body(X)→bool で判定）"""
        if self.kind != "ellipse":
            return self.ring
        P = self.ring
        if inside_body is not None:
            keep = inside_body(P)
            if not keep.all() and keep.any():
                # 物体の中の連続した区間（半楕円なら 0〜π の側）を取り出す
                k0 = int(np.argmin(keep))                # 外の点から始めて一周
                rolled = np.roll(np.arange(len(P) - 1), -k0)
                inn = keep[rolled]
                idx = rolled[inn]
                P = P[idx]
        return P

    def inside(self, X, tol):
        """き裂面の内側（前縁を除く）の点"""
        h = (X - self.p0) @ self.n
        on = np.abs(h) < tol
        if self.kind == "ellipse":
            r = X - self.center
            xc, xa = r @ self.uc, r @ self.ua
            rho = (xc / self.cc) ** 2 + (xa / self.a) ** 2
            return on & (rho < 1 - 1e-5 - 4 * tol / min(self.a, self.cc))
        r = X - self.p0
        x1, s = r @ self.e1, r @ self.t
        a = float(self.c.get("depth", 0.0))
        ok = on & (x1 < -tol) & (s > -tol) & (s < self.L + tol)
        return ok & (bool(self.c.get("to_surface", True)) | (x1 > -a - tol))


def _project_polyline(X, P):
    """点 X から折れ線 P への最近点: 距離、弧長、接線、最近点"""
    A, B = P[:-1], P[1:]
    seg = B - A
    ls = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(ls)]
    best = np.full(len(X), np.inf)
    s_out = np.zeros(len(X))
    t_out = np.zeros((len(X), 3))
    f_out = np.zeros((len(X), 3))
    for k in range(len(A)):
        u = np.clip(((X - A[k]) @ seg[k]) / max(ls[k] ** 2, 1e-300), 0.0, 1.0)
        F = A[k] + np.outer(u, seg[k])
        d = np.linalg.norm(X - F, axis=1)
        b = d < best
        best[b], s_out[b], f_out[b] = d[b], cum[k] + u[b] * ls[k], F[b]
        t_out[b] = seg[k] / max(ls[k], 1e-300)
    return best, s_out, t_out, f_out, cum[-1]


def j_front(mesh: Mesh, u, D, geom: CrackGeom, P, stations, width, r_in, r_out, elem_mat=None, symmetric=False):
    """曲がった前縁に沿った J（局所の値）。仮想き裂進展の向きのベクトル場 Q = q_r(r)·q_s(s)·e1 を節点で作り、
    J(s0) = ∫ (σ_ij ∂u_j/∂x_k − W δ_ik) ∂Q_k/∂x_i dV / ∫ q_s ds（Shih–Moran–Nakamura の 3 次元の領域積分）"""
    X = mesh.nodes
    U = np.asarray(u, float).reshape(-1, 3)
    near = np.nonzero(np.linalg.norm(X - P.mean(0), axis=1) <= np.linalg.norm(np.ptp(P, axis=0)) + 2 * r_out)[0]
    r = np.full(len(X), np.inf)
    s = np.zeros(len(X))
    tt = np.zeros((len(X), 3))
    F = np.zeros((len(X), 3))
    r[near], s[near], tt[near], F[near], Ltot = _project_polyline(X[near], P)
    qr = np.clip((r_out - r) / max(r_out - r_in, 1e-30), 0.0, 1.0)
    e1 = np.cross(geom.n[None, :], tt)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-300)
    # e1 は き裂の外（進む向き）: 最近点から e1 の逆へ少し入った点が き裂面の内側になる向き
    eps = 0.25 * max(r_in, r_out * 0.1)
    tol = 1e-6 * float(np.ptp(X, axis=0).max())
    flip = ~geom.inside(F - eps * e1, max(tol, 1e-9)) & geom.inside(F + eps * e1, max(tol, 1e-9))
    e1[flip] *= -1
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    De_all = np.asarray(D, float)
    out = []
    for s0 in stations:
        qs = np.clip(1 - np.abs(s - s0) / width, 0.0, 1.0)
        q = qr * qs
        Q = q[:, None] * e1
        act = np.nonzero((q[mesh.elems] > 0).any(1))[0]
        if not len(act):
            out.append(np.nan)
            continue
        el = mesh.elems[act]
        Xe, Ue, Qe = X[el], U[el], Q[el]
        De = (De_all[np.asarray(elem_mat)[act]] if (De_all.ndim == 3 and elem_mat is not None) else
              (De_all[0] if De_all.ndim == 3 else De_all))
        tot = 0.0
        for L_, w in zip(pts, wts):
            dNdx, detJ = _dN_dx(Xe, shape_dN_dxi(L_, nn))
            gu = np.einsum("eaj,eak->ejk", Ue, dNdx)          # ∂u_j/∂x_k
            gQ = np.einsum("eak,eai->eki", Qe, dNdx)          # ∂Q_k/∂x_i
            eps_ = np.einsum("eij,ej->ei", _B_matrix(dNdx), Ue.reshape(len(act), -1))
            sig = eps_ @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, eps_)
            W = 0.5 * np.einsum("ei,ei->e", sig, eps_)
            S = np.stack([sig[:, [0, 3, 5]], sig[:, [3, 1, 4]], sig[:, [5, 4, 2]]], axis=1)
            val = np.einsum("eij,ejk,eki->e", S, gu, gQ) - W * np.trace(gQ, axis1=1, axis2=2)
            tot += float(np.sum(val * w * detJ))
        # 分母 ∫ q_s ds（前縁の端で切れる分も正しく）
        ss = np.linspace(0, Ltot, 2001)
        den = float(np.trapezoid(np.clip(1 - np.abs(ss - s0) / width, 0, 1), ss)) if hasattr(np, "trapezoid") else \
            float(np.trapz(np.clip(1 - np.abs(ss - s0) / width, 0, 1), ss))
        out.append((2.0 if symmetric else 1.0) * tot / max(den, 1e-300))
    return np.array(out), Ltot


def raju_newman(sig, a, c, t, b, phi):
    """半楕円の表面き裂（有限幅の板・引張）の K（Raju–Newman 1981、a/c ≤ 1、精度 ±5 %）。b = 板の幅の半分、φ: 前縁の角度"""
    ac, at = a / c, a / t
    Q = 1 + 1.464 * ac ** 1.65
    M1 = 1.13 - 0.09 * ac
    M2 = -0.54 + 0.89 / (0.2 + ac)
    M3 = 0.5 - 1 / (0.65 + ac) + 14 * (1 - ac) ** 24
    g = 1 + (0.1 + 0.35 * at ** 2) * (1 - np.sin(phi)) ** 2
    fphi = (ac ** 2 * np.cos(phi) ** 2 + np.sin(phi) ** 2) ** 0.25
    fw = np.sqrt(1 / np.cos(np.pi * c / (2 * b) * np.sqrt(at)))
    F = (M1 + M2 * at ** 2 + M3 * at ** 4) * g * fphi * fw
    return sig * np.sqrt(np.pi * a / Q) * F


def ellipse_front_nodes_range(mesh: Mesh, g: CrackGeom):
    """メッシュにある楕円の前縁の節点から、前縁の角度 θ の範囲（物体の中の部分）を求める"""
    X = mesh.nodes
    tol = 1e-6 * float(np.ptp(X, axis=0).max())
    r = X - g.center
    h = r @ g.n
    xc, xa = r @ g.uc, r @ g.ua
    rho = np.sqrt((xc / g.cc) ** 2 + (xa / g.a) ** 2)
    on = (np.abs(h) < 1e3 * tol) & (np.abs(rho - 1) < 0.02)
    if not on.any():
        raise ValueError("楕円の前縁の節点が見つかりません")
    th = np.arctan2(xa[on] / g.a, xc[on] / g.cc)
    return float(th.min()), float(th.max())


def _evaluate_ellipse(mesh, u, D, em, mats, vol, c, i):
    g = CrackGeom(c)
    t0, t1 = ellipse_front_nodes_range(mesh, g)
    th = np.linspace(t0, t1, 121)
    P = g.center + np.outer(g.cc * np.cos(th), g.uc) + np.outer(g.a * np.sin(th), g.ua)
    X = mesh.nodes
    cen = X[mesh.elems[:, :4]].mean(1)
    rr = _project_polyline(cen, P)[0]
    near = np.argsort(rr)[:max(8, min(64, len(rr)))]
    hf = float(c.get("front_size") or 0) or float(np.cbrt(np.median(vol[near]) * 6 * np.sqrt(2)))
    m0 = mats[int(np.bincount(em[near]).argmax())]
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    Ltot = seg[-1]
    nst = int(c.get("stations", 9))
    st = np.linspace(0.04 * Ltot, 0.96 * Ltot, nst)
    width = max(Ltot / (nst - 1), 3 * hf)
    J1, _ = j_front(mesh, u, D, g, P, st, width, hf, 3 * hf, elem_mat=em, symmetric=bool(c.get("symmetric")))
    J2, _ = j_front(mesh, u, D, g, P, st, width, 2 * hf, 5 * hf, elem_mat=em, symmetric=bool(c.get("symmetric")))
    rows = []
    for k, s0 in enumerate(st):
        thk = float(np.interp(s0, seg, th))
        J = 0.5 * (J1[k] + J2[k])
        row = {"き裂": c.get("name") or f"き裂 {i + 1}", "位置 φ [°]": round(np.degrees(thk), 2),
               "J [N/mm]": float(J), "J 積分路 1": float(J1[k]), "J 積分路 2": float(J2[k]),
               "K（J から）[MPa√mm]": k_from_j(J, m0.E, m0.nu)}
        if c.get("toughness"):
            row["破壊靭性 K_IC [MPa√mm]"] = float(c["toughness"])
            row["安全率（K_IC / K）"] = float(c["toughness"]) / max(row["K（J から）[MPa√mm]"], 1e-30)
        rows.append(row)
    return rows

"""複合材の積層（Ansys ACP / 積層シェル SHELL181 の layup に相当）— 古典積層理論（CLT）

各層: 材料（直交異方性の Ex, Ey, Gxy, νxy。等方性も可）、厚さ、繊維の角度 θ [度]（基準方向から法線まわり）。
  Q̄_k = T(θ)ᵀ Q_k T(θ)
  A = Σ Q̄_k (z_k − z_{k−1})、B = ½ Σ Q̄_k (z_k² − z_{k−1}²)、D = ⅓ Σ Q̄_k (z_k³ − z_{k−1}³)
z は板厚中央から（下面 = −t/2）。層の順番は下面から上面へ（Ansys と同じ）。

破損の評価（層ごと・上下面）: 最大応力、Tsai-Hill、Tsai-Wu（F12 = −½√(F11 F22)）。
強度は材料の composite_strength = {"Xt","Xc","Yt","Yc","S"} [MPa]（圧縮は正の値）。
"""
from __future__ import annotations

import numpy as np

FAILURE_JP = {"max_stress": "最大応力", "tsai_hill": "Tsai-Hill", "tsai_wu": "Tsai-Wu"}


def ply_Q(mat):
    """層の材料主軸での平面応力の剛性 (3,3)（成分 11, 22, 12・工学せん断）"""
    from .shell import plane_stress_D
    base = getattr(mat, "base", mat)
    return plane_stress_D(base.D_local())


def T_eps(theta):
    """ひずみ（工学せん断）の変換 ε_ply = T ε_ref（θ: 基準軸から繊維方向への角度, rad）"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, s * s, c * s], [s * s, c * c, -c * s], [-2 * c * s, 2 * c * s, c * c - s * s]])


def abd(plies, phi=0.0):
    """plies: [(Q (3,3), 厚さ, 角度 [度]), ...]（下から上）、phi: 要素の x 軸から基準方向への角度 [rad]
    戻り値: A, B, D (3,3)、z の境界 (n+1,)、各層の Q̄（要素座標）、各層の T"""
    t = np.array([p[1] for p in plies], float)
    z = np.r_[0.0, np.cumsum(t)] - t.sum() / 2
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    Qb, Ts = [], []
    for k, (Q, _, ang) in enumerate(plies):
        T = T_eps(np.radians(ang) + phi)
        Qbar = T.T @ Q @ T
        A += Qbar * (z[k + 1] - z[k])
        B += Qbar * (z[k + 1] ** 2 - z[k] ** 2) / 2
        D += Qbar * (z[k + 1] ** 3 - z[k] ** 3) / 3
        Qb.append(Qbar)
        Ts.append(T)
    return A, B, D, z, np.array(Qb), np.array(Ts)


def failure_index(sig, strength, criterion="tsai_wu"):
    """層の材料主軸の応力 sig (...,3) [σ1, σ2, τ12] → 破損指標（1 以上で破損。Ansys の IRF に相当）"""
    st = {k: float(v) for k, v in (strength or {}).items()}
    if not st:
        return np.zeros(sig.shape[:-1])
    Xt, Xc = st.get("Xt", np.inf), st.get("Xc", st.get("Xt", np.inf))
    Yt, Yc = st.get("Yt", np.inf), st.get("Yc", st.get("Yt", np.inf))
    S = st.get("S", np.inf)
    s1, s2, t12 = sig[..., 0], sig[..., 1], sig[..., 2]
    if criterion == "max_stress":
        return np.maximum.reduce([np.where(s1 >= 0, s1 / Xt, -s1 / Xc), np.where(s2 >= 0, s2 / Yt, -s2 / Yc),
                                  np.abs(t12) / S])
    if criterion == "tsai_hill":
        X = np.where(s1 >= 0, Xt, Xc)
        Y = np.where(s2 >= 0, Yt, Yc)
        return np.sqrt(np.maximum((s1 / X) ** 2 - s1 * s2 / X ** 2 + (s2 / Y) ** 2 + (t12 / S) ** 2, 0.0))
    # Tsai-Wu: F1 σ1 + F2 σ2 + F11 σ1² + F22 σ2² + F66 τ² + 2 F12 σ1 σ2 = 1 → 指標 = 1/(強度比)
    F1, F2 = 1 / Xt - 1 / Xc, 1 / Yt - 1 / Yc
    F11, F22, F66 = 1 / (Xt * Xc), 1 / (Yt * Yc), 1 / S ** 2
    F12 = -0.5 * np.sqrt(F11 * F22)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    # 強度比 R: a R² + b R = 1 の正の根。指標 = 1/R
    R = np.where(a > 1e-300, (-b + np.sqrt(b * b + 4 * a)) / (2 * np.maximum(a, 1e-300)),
                 np.where(b > 0, 1 / np.maximum(b, 1e-300), np.inf))
    return 1.0 / R


class Laminates:
    """要素ごとの積層（シェル）。lay_of_elem: 要素 → 積層の番号（−1 = 均質）、layups: [{"plies": [(材料, t, 角度)],
    "ref": 基準方向（全体座標）, "criterion": 破損則}]"""

    def __init__(self, lay_of_elem, layups):
        self.lay = np.asarray(lay_of_elem, np.int64)
        self.layups = layups

    @property
    def any(self):
        return bool(np.any(self.lay >= 0))

    def thickness(self):
        return {i: float(sum(p[1] for p in L["plies"])) for i, L in enumerate(self.layups)}

    def element_abd(self, R, t, Dp):
        """要素ごとの A, B, D (E,3,3)。積層でない要素は A = tD、B = 0、D = t³/12 D"""
        E = len(R)
        Am = t[:, None, None] * Dp
        Bm = np.zeros((E, 3, 3))
        Dm = (t ** 3 / 12)[:, None, None] * Dp
        self.info = {}
        for i, L in enumerate(self.layups):
            sel = np.nonzero(self.lay == i)[0]
            if not len(sel):
                continue
            ref = np.asarray(L.get("ref", [1, 0, 0]), float)
            # 基準方向を要素面に投影し、要素の x 軸からの角度 φ を求める（法線に平行なら要素の x 軸）
            rx = np.einsum("ej,j->e", R[sel, 0], ref)
            ry = np.einsum("ej,j->e", R[sel, 1], ref)
            phi = np.where(np.hypot(rx, ry) > 1e-9, np.arctan2(ry, rx), 0.0)
            Qs = [(ply_Q(m), float(th), float(an)) for m, th, an in L["plies"]]
            for e, ph in zip(sel, phi):
                A_, B_, D_, z, Qb, Ts = abd(Qs, ph)
                Am[e], Bm[e], Dm[e] = A_, B_, D_
                self.info[int(e)] = (z, Qb, Ts, i)
        return Am, Bm, Dm

    def ply_results(self, em, kap, n_elem):
        """層ごとの上下面の応力（材料主軸）と破損指標。戻り値: 要素ごとの最大破損指標、その層番号、詳細の表"""
        fi = np.zeros(n_elem)
        layer = np.full(n_elem, -1)
        rows = {}
        for e, (z, Qb, Ts, li) in self.info.items():
            L = self.layups[li]
            crit = L.get("criterion", "tsai_wu")
            worst, wl = 0.0, -1
            for k in range(len(Qb)):
                m = L["plies"][k][0]
                strength = getattr(getattr(m, "base", m), "composite_strength", None) or {}
                for zz in (z[k], z[k + 1]):
                    eps = em[e] + zz * kap[e]
                    s_ply = ply_Q(m) @ (Ts[k] @ eps)       # 材料主軸: ε_ply = T ε、σ_ply = Q ε_ply
                    f = float(failure_index(s_ply, strength, crit)) if strength else 0.0
                    if f > worst:
                        worst, wl = f, k
                    r = rows.setdefault((li, k), {"積層": li + 1, "層": k + 1, "角度 [°]": L["plies"][k][2],
                                                  "σ1 最大 [MPa]": -np.inf, "σ1 最小 [MPa]": np.inf,
                                                  "σ2 最大 [MPa]": -np.inf, "τ12 最大 [MPa]": 0.0,
                                                  "破損指標 最大": 0.0})
                    r["σ1 最大 [MPa]"] = max(r["σ1 最大 [MPa]"], float(s_ply[0]))
                    r["σ1 最小 [MPa]"] = min(r["σ1 最小 [MPa]"], float(s_ply[0]))
                    r["σ2 最大 [MPa]"] = max(r["σ2 最大 [MPa]"], float(s_ply[1]))
                    r["τ12 最大 [MPa]"] = max(r["τ12 最大 [MPa]"], abs(float(s_ply[2])))
                    r["破損指標 最大"] = max(r["破損指標 最大"], f)
            fi[e], layer[e] = worst, wl + 1
        return fi, layer, [rows[k] for k in sorted(rows)]

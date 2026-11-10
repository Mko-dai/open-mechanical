"""結果の評価ツール（Ansys の Fatigue Tool・ユーザー定義結果・パス結果に相当）。

疲労（応力寿命）: S-N 曲線（両対数で補間）、平均応力補正（なし / Goodman / Gerber / Soderberg / ASME 楕円）、
  荷重の種類（両振り R=-1 / 片振り R=0 / 応力比 R / 倍率）、応力の成分（符号付きミーゼス / 最大主応力 / 最大せん断）。
  出力: 寿命 [回]、損傷度 = 設計寿命 / 寿命、安全率（設計寿命で、応力振幅を何倍まで上げられるか、最大 15）。
ユーザー定義結果: 結果の名前を変数にした式（例 "sqrt(SX^2+SY^2)", "SEQV/250"）。
パス結果: 2 点を結ぶ線の上で結果を補間した値。
"""
from __future__ import annotations

import ast
import math

import numpy as np

MEAN_CORR = {"none": "なし", "goodman": "Goodman", "gerber": "Gerber", "soderberg": "Soderberg", "asme": "ASME 楕円"}
STRESS_COMP = {"signed_vm": "符号付き相当応力（ミーゼス）", "s1": "最大主応力", "tresca": "最大せん断応力（×2）",
               "vm": "相当応力（ミーゼス）"}


# ---------------------------------------------------------------------------- 疲労

def sn_life(Sa, table):
    """応力振幅 Sa [MPa] → 寿命 [回]（S-N 表 [[N, S], ...] を両対数で補間。表の上限を超える長寿命は表の最大回数で頭打ち）"""
    tab = sorted(([float(n), float(s)] for n, s in table), key=lambda r: r[0])
    N = np.log10([r[0] for r in tab])
    S = np.log10([r[1] for r in tab])
    Sa = np.maximum(np.asarray(Sa, float), 1e-30)
    ls = np.log10(Sa)
    # S は N とともに減る → S の昇順に並べ直して補間
    o = np.argsort(S)
    life = 10 ** np.interp(ls, S[o], N[o], left=N.max(), right=N.min())
    # 表の最大の応力を超える（＝最小寿命以下）は外挿（直線）
    hi = ls > S.max()
    if hi.any() and len(S) >= 2:
        k = (N[1] - N[0]) / (S[1] - S[0]) if S[1] != S[0] else 0
        life[hi] = 10 ** (N[0] + k * (ls[hi] - S[0]))
    return life


def sn_strength(N, table):
    """寿命 N 回の疲労強度（応力振幅）"""
    tab = sorted(([float(n), float(s)] for n, s in table), key=lambda r: r[0])
    return float(10 ** np.interp(np.log10(N), np.log10([r[0] for r in tab]), np.log10([r[1] for r in tab])))


def _equivalent(stress6, comp):
    s = stress6
    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                 + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    T = np.empty((len(s), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = s[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = s[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = s[:, 5]
    p = np.linalg.eigvalsh(T)
    if comp == "s1":
        return p[:, -1]
    if comp == "tresca":
        return p[:, -1] - p[:, 0]
    if comp == "vm":
        return vm
    # 符号付きミーゼス: 絶対値最大の主応力の符号（Ansys と同じ）
    sgn = np.where(np.abs(p[:, -1]) >= np.abs(p[:, 0]), np.sign(p[:, -1]), np.sign(p[:, 0]))
    return vm * np.where(sgn == 0, 1.0, sgn)


def fatigue(stress6, material, loading="reversed", ratio=-1.0, scale=1.0, mean_corr="none", comp="signed_vm",
            design_life=1e9, kf=1.0):
    """節点ごとの疲労寿命・損傷度・安全率。stress6: 荷重 1 回分の応力 (N,6)（静解析の結果）"""
    table = getattr(material, "sn_table", None) or []
    if len(table) < 2:
        raise ValueError(f"材料「{material.name}」に S-N 曲線（疲労）のデータがありません")
    se = _equivalent(np.asarray(stress6, float), comp) * scale * kf
    if loading == "reversed":
        R = -1.0
    elif loading == "zero":
        R = 0.0
    else:
        R = float(ratio)
    # 最大応力 se、応力比 R → 振幅 Sa = se (1-R)/2、平均 Sm = se (1+R)/2
    Sa = np.abs(se) * (1 - R) / 2
    Sm = se * (1 + R) / 2
    Su = float(getattr(material, "tensile_strength", 0) or 0)
    Sy = float(getattr(material, "yield_strength", 0) or 0)
    Sa_eq = _mean_correct(Sa, Sm, mean_corr, Su, Sy)
    life = sn_life(Sa_eq, table)
    Nmax = max(float(r[0]) for r in table)
    life = np.minimum(life, max(Nmax, design_life * 10))
    life = np.where(Sa_eq <= sn_strength(Nmax, table), np.maximum(life, Nmax), life)   # 疲労限度以下は表の最大回数以上
    damage = design_life / life
    # 安全率: 設計寿命の強度 S_D に対して、補正後の振幅が何倍まで許されるか（Sa ∝ 荷重 として）
    SD = sn_strength(min(design_life, Nmax), table)
    sf = np.minimum(15.0, SD / np.maximum(Sa_eq, 1e-30))
    if mean_corr != "none":
        # 平均応力も荷重に比例するので、補正式の上で倍率 x を二分法で求める
        lo, hi = np.zeros_like(Sa), np.full_like(Sa, 15.0)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            ok = _mean_correct(Sa * mid, Sm * mid, mean_corr, Su, Sy) <= SD
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        sf = lo
    return {"寿命 [回]": life, "損傷度": damage, "安全率（疲労）": sf, "応力振幅（補正後） [MPa]": Sa_eq}


def _mean_correct(Sa, Sm, method, Su, Sy):
    if method == "none":
        return Sa
    Smp = np.maximum(Sm, 0.0)       # 圧縮の平均応力は補正しない（Ansys の既定と同じ）
    if method in ("goodman", "gerber", "asme") and Su <= 0:
        raise ValueError("平均応力補正には引張強さが必要です")
    if method == "soderberg" and Sy <= 0:
        raise ValueError("Soderberg 補正には降伏応力が必要です")
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "goodman":
            d = 1 - Smp / Su
        elif method == "soderberg":
            d = 1 - Smp / Sy
        elif method == "gerber":
            d = 1 - (Smp / Su) ** 2
        else:   # ASME 楕円（Sy で）
            d = np.sqrt(np.maximum(1 - (Smp / (Sy or Su)) ** 2, 0))
        return np.where(d > 1e-9, Sa / np.maximum(d, 1e-9), 1e30)


# ---------------------------------------------------------------------------- ユーザー定義結果

ALIASES = {
    "UX": "変位 X [mm]", "UY": "変位 Y [mm]", "UZ": "変位 Z [mm]", "USUM": "全変形 [mm]",
    "SX": "応力 σx [MPa]", "SY": "応力 σy [MPa]", "SZ": "応力 σz [MPa]", "SXY": "せん断応力 τxy [MPa]",
    "SYZ": "せん断応力 τyz [MPa]", "SXZ": "せん断応力 τzx [MPa]", "SEQV": "相当応力（ミーゼス） [MPa]",
    "S1": "最大主応力 [MPa]", "S3": "最小主応力 [MPa]", "TEMP": "温度 [°C]",
}
_FN = {"sqrt": np.sqrt, "abs": np.abs, "exp": np.exp, "log": np.log, "log10": np.log10, "sin": np.sin,
       "cos": np.cos, "tan": np.tan, "max": np.maximum, "min": np.minimum, "atan2": np.arctan2,
       "hypot": np.hypot, "sign": np.sign}


def user_result(expr: str, fields: dict, mesh=None) -> np.ndarray:
    """結果の式を評価（節点ごと）。変数: ALIASES の略号、X/Y/Z（座標）、または結果の名前そのもの（全角の名前は使えない）"""
    env = {k: np.asarray(fields[v], float) for k, v in ALIASES.items() if v in fields}
    if mesh is not None:
        env.update(X=mesh.nodes[:, 0], Y=mesh.nodes[:, 1], Z=mesh.nodes[:, 2])
    src = str(expr).replace("^", "**")
    tree = ast.parse(src, mode="eval")
    ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id == "pi":
                return math.pi
            raise ValueError(f"結果「{n.id}」がありません（使える名前: {', '.join(sorted(env))}）")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN:
            return _FN[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"式に使えない要素があります: 「{expr}」")

    n_nodes = len(next(iter(env.values())))
    with np.errstate(all="ignore"):
        v = ev(tree)
    return np.broadcast_to(np.asarray(v, float), (n_nodes,)).copy()


# ---------------------------------------------------------------------------- パス結果

def path_values(mesh, values, p0, p1, n=101):
    """p0 → p1 の線上 n 点で値を補間（四面体の頂点の 1 次補間。要素の外の点は NaN）。戻り値: (距離, 値)"""
    from scipy.spatial import cKDTree
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)
    P = p0[None] + t[:, None] * (p1 - p0)[None]
    v = np.asarray(values, float)
    el = mesh.elems
    nn = el.shape[1]
    corners = el[:, :4] if nn in (4, 10) else el[:, :6]
    cen = mesh.nodes[corners].mean(axis=1)
    tree = cKDTree(cen)
    out = np.full(n, np.nan)
    k = min(24, len(el))
    _, cand = tree.query(P, k=k)
    cand = np.atleast_2d(cand)
    for i, p in enumerate(P):
        for e in cand[i]:
            if nn in (4, 10):
                X = mesh.nodes[el[e, :4]]
                A = np.vstack([X.T, np.ones(4)])
                try:
                    L = np.linalg.solve(A, np.r_[p, 1.0])
                except np.linalg.LinAlgError:
                    continue
                if L.min() >= -1e-6:
                    if nn == 10:
                        from .solver import shape_N
                        out[i] = shape_N(np.clip(L, 0, 1), 10) @ v[el[e]]
                    else:
                        out[i] = L @ v[el[e, :4]]
                    break
            else:   # 三角柱: 最寄りの節点の値
                X = mesh.nodes[el[e]]
                j = np.argmin(np.linalg.norm(X - p, axis=1))
                out[i] = v[el[e, j]]
                break
    return t * np.linalg.norm(p1 - p0), out


def _vm6(s):
    s = np.asarray(s, float)
    return float(np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2)
                         + 3 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)))


def linearize_stress(mesh, stress, p0, p1, n=49):
    """応力の線形化（Ansys の Linearized Stress。圧力容器の評価などに使う）。
    板厚を貫く線 p0 → p1 の上の応力を、膜（平均）・曲げ（直線分布）・ピーク（残り）に分ける。

        膜      σm = (1/t) ∫ σ ds
        曲げ    σb = ±(6/t²) ∫ σ (t/2 − s) ds     （始点で +、終点で −）
        ピーク  σp = σ − (σm + σb)

    成分（全体座標の 6 成分）ごとに分けてから、相当応力（ミーゼス）にする。
    戻り値: {"表": [行の辞書], "s": 距離, "全体": 相当応力の分布, "膜": …, "膜＋曲げ": …, "成分": {...}}"""
    from .submodel import CoarseField
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = float(np.linalg.norm(p1 - p0))
    if t <= 0:
        raise ValueError("応力の線形化: 始点と終点が同じです")
    n = int(n) | 1
    sv = np.linspace(0.0, t, n)
    P = p0[None] + (sv / t)[:, None] * (p1 - p0)[None]
    cf = CoarseField(mesh.nodes, mesh.elems, np.zeros((mesh.n_nodes, 3)))
    sig, dist = cf.interpolate(P, field=np.asarray(stress, float))
    if dist.max() > 0.02 * t + 1e-9:
        raise ValueError("応力の線形化: 線の一部がモデルの外にあります（始点と終点を板の両面の上に置いてください）")
    trap = getattr(np, "trapezoid", None) or np.trapz
    sm = trap(sig, sv, axis=0) / t
    sb = 6.0 / t ** 2 * trap(sig * (t / 2 - sv)[:, None], sv, axis=0)
    mb0, mb1 = sm + sb, sm - sb
    rows = [{"項目": "膜", "始点 [MPa]": _vm6(sm), "終点 [MPa]": _vm6(sm)},
            {"項目": "曲げ", "始点 [MPa]": _vm6(sb), "終点 [MPa]": _vm6(sb)},
            {"項目": "膜＋曲げ", "始点 [MPa]": _vm6(mb0), "終点 [MPa]": _vm6(mb1)},
            {"項目": "ピーク", "始点 [MPa]": _vm6(sig[0] - mb0), "終点 [MPa]": _vm6(sig[-1] - mb1)},
            {"項目": "全体", "始点 [MPa]": _vm6(sig[0]), "終点 [MPa]": _vm6(sig[-1])}]
    lin = sm[None] + (1 - 2 * sv / t)[:, None] * sb[None]
    return {"表": rows, "s": sv, "全体": np.array([_vm6(x) for x in sig]), "膜": np.full(n, _vm6(sm)),
            "膜＋曲げ": np.array([_vm6(x) for x in lin]), "成分": {"膜": sm, "曲げ": sb}, "厚さ": t}

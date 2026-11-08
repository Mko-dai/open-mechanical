"""LS-DYNA 検証マニュアル（Ansys LS-DYNA Verification Manual）の 5 例題を解いて、
理論値・LS-DYNA の公表値と比べる。

  python tools/lsdyna_compare.py [番号...]    結果: logs/lsdyna/summary.json, logs/lsdyna/*.png

  1 VM-LSDYNA-IMPACT-004     3D スプリット・ホプキンソン棒（圧縮弾性波）       陽解法
  2 VM-LSDYNA-SOLVE-061      柔らかい面に落ちるボール（重力・接触）            陽解法
  3 VM-LSDYNA-SOLVE-032      単純支持はりの固有振動数                          陰解法
  4 VM-LSDYNA-BENCHMARK-001  片持ちはりのたわみ（微小変形）                    陽解法＋減衰
  5 VM-LSDYNA-BENCHMARK-003  弾完全塑性はり・大変形（先端モーメント）          陽解法＋塑性＋大変形

LS-DYNA の値は検証マニュアル（Ansys 2026 R1 ヘルプ）に公表されている数値。
open-mechanical は四面体 2 次要素（LS-DYNA の例題は六面体・はり・シェル要素）。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from openmech.dynamics import DynamicSettings, solve_dynamic  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.mesh import structured_box  # noqa: E402
from openmech.solver import BC, _B_matrix, _dN_dx, material_D, shape_dN_dxi  # noqa: E402

OUT = os.path.join(ROOT, "logs", "lsdyna")
os.makedirs(OUT, exist_ok=True)
QUIET = lambda *a: None  # noqa: E731


def mat(name, E, nu, rho_consistent, **kw):
    """一貫単位系の密度（例: t/mm³, lbf·s²/in⁴）で材料を作る（Material の密度は kg/m³ 表記で ×1e-12 されるため）"""
    return Material(name, E, nu, rho_consistent * 1e12, **kw)


def faces_at(m, axis, value, tol=1e-6):
    return [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - value) < tol and f["flat"]
            and abs(abs(f["normal"][axis]) - 1) < 1e-6]


def rel(a, b):
    return (a - b) / b * 100.0


# ---------------------------------------------------------------------------- 1. SHPB
def case1():
    E, nu, rho, v = 210000.0, 0.3, 7.8e-9, 15000.0          # MPa, t/mm³, mm/s（15 mm/ms）
    Ls, Lb, D = 300.0, 1800.0, 20.0
    a = math.sqrt(math.pi * D * D / 4)                     # 断面積が同じ正方形断面
    m = structured_box(Ls + Lb, a, a, 700, 2, 2).to_quadratic()
    steel = mat("鋼", E, nu, rho)
    x = m.nodes[:, 0]
    v0 = np.zeros((m.n_nodes, 3))
    v0[x < Ls - 1e-6, 0] = v
    v0[np.abs(x - Ls) < 1e-6, 0] = v / 2                   # 打撃面（ストライカーと棒の境）
    # 測定点: 衝突面から 1000 mm の要素（断面内の要素の平均）
    cen = m.nodes[m.elems[:, :4]].mean(1)
    pe = np.nonzero(np.abs(cen[:, 0] - (Ls + 1000)) < 1.6)[0]
    X = m.nodes[m.elems[pe]]
    dNdx, _ = _dN_dx(X, shape_dN_dxi(np.array([0.25] * 4), 10))
    B = _B_matrix(dNdx)
    Dm = material_D(steel)

    def mon(t, u, vv):
        sig = np.einsum("ij,ej->ei", Dm, np.einsum("eij,ej->ei", B, u.reshape(-1, 3)[m.elems[pe]].reshape(len(pe), -1)))
        return {"sx": float(sig[:, 0].mean())}
    s = DynamicSettings(method="explicit", end_time=4.5e-4, n_frames=1)
    t0 = time.time()
    fr = solve_dynamic(m, steel, [], s, log=QUIET, v0=v0, monitor=mon)
    h = fr[-1].history
    t = np.array(h["時間 [s]"]) * 1e3
    sx = -np.array(h["sx"])                               # 圧縮を正に
    smax = sx.max()
    half = sx > 0.5 * smax
    i0, i1 = np.nonzero(half)[0][[0, -1]]

    def cross(i, j):
        return t[i] + (0.5 * smax - sx[i]) * (t[j] - t[i]) / (sx[j] - sx[i])
    dur = cross(i1, i1 + 1) - cross(i0 - 1, i0)
    plateau = sx[i0:i1 + 1]
    pk = [k for k in range(1, len(plateau) - 1) if plateau[k] >= plateau[k - 1] and plateau[k] >= plateau[k + 1]]
    mag = float(plateau[pk[0]:pk[-1] + 1].mean()) if len(pk) >= 2 else float(plateau.mean())
    c = math.sqrt(E / rho)
    tgt_mag, tgt_dur = rho * c * v / 2 / 1000, 2 * Ls / c * 1e3        # GPa, ms
    return {"id": "VM-LSDYNA-IMPACT-004", "名前": "3D スプリット・ホプキンソン棒（圧縮弾性波）", "解析": "陽解法",
            "要素": f"Tet10 {m.n_elems:,} 要素（LS-DYNA: 六面体）", "計算時間 [s]": round(time.time() - t0, 1),
            "項目": [
                {"名前": "第1圧縮波の大きさ [GPa]", "理論": tgt_mag, "LS-DYNA": 0.3036, "open-mechanical": mag / 1000},
                {"名前": "第1圧縮波の継続時間 [ms]", "理論": tgt_dur, "LS-DYNA": 0.1180, "open-mechanical": dur},
            ], "グラフ": {"x": t.tolist(), "y": (sx / 1000).tolist(), "xl": "時間 [ms]", "yl": "測定点の圧縮応力 [GPa]"}}


# ---------------------------------------------------------------------------- 2. ボールと柔らかい面
def case2():
    mball, k, h, g = 0.5, 1973.92, 1.0, 386.0             # lbf·s²/in, lbf/in, in, in/s²
    a, Lc, Hb = 4.0, 10.0, 4.0
    m = structured_box(a, a, Lc + Hb, 2, 2, int(Lc + Hb)).to_quadratic()
    zc = m.nodes[m.elems[:, :4]].mean(1)[:, 2]
    spring = mat("ばね（柱）", k * Lc / (a * a), 0.0, 0.01 * mball / (a * a * Lc))     # 質量はボールの 1%
    ball = mat("ボール（剛）", 3e7, 0.3, mball / (a * a * Hb))
    states = [spring.state(), ball.state()]
    em = (states, (zc > Lc).astype(np.int64))
    top = np.nonzero(m.nodes[:, 2] >= Lc - 1e-9)[0]
    bot = np.nonzero(m.nodes[:, 2] < 1e-9)[0]

    def mon(t, u, v):
        U, V = u.reshape(-1, 3), v.reshape(-1, 3)
        return {"zb": float(U[top, 2].mean()), "vb": float(V[top, 2].mean()), "zbot": float(U[bot, 2].mean())}
    s = DynamicSettings(method="explicit", end_time=0.11, n_frames=1, wall={"normal": [0, 0, 1], "point": [0, 0, -h]})
    t0 = time.time()
    fr = solve_dynamic(m, ball, [BC("gravity", [], [0, 0, -g])], s, elem_materials=em, log=QUIET, monitor=mon)
    hh = fr[-1].history
    t = np.array(hh["時間 [s]"])
    zb, vb, zbot = np.array(hh["zb"]), np.array(hh["vb"]), np.array(hh["zbot"])
    i = int(np.nonzero(zbot <= -h)[0][0])
    w = (-h - zbot[i - 1]) / (zbot[i] - zbot[i - 1])
    t_imp = t[i - 1] + w * (t[i] - t[i - 1])
    v_imp = vb[i - 1] + w * (vb[i] - vb[i - 1])
    z_imp = zb[i - 1] + w * (zb[i] - zb[i - 1])
    j = int(zb.argmin())
    A = np.polyfit(t[j - 2:j + 3] - t[j], zb[j - 2:j + 3], 2)        # 最下点を放物線で補間
    tm = -A[1] / (2 * A[0])
    zm = np.polyval(A, tm)
    x = (mball * g + math.sqrt((mball * g) ** 2 + 2 * k * mball * g * h)) / k
    return {"id": "VM-LSDYNA-SOLVE-061", "名前": "柔らかい面に落ちるボール（重力・接触）", "解析": "陽解法",
            "要素": f"Tet10 {m.n_elems:,} 要素（LS-DYNA: 質量要素＋シェル）", "計算時間 [s]": round(time.time() - t0, 1),
            "項目": [
                {"名前": "着地の時刻 [s]", "理論": math.sqrt(2 * h / g), "LS-DYNA": 0.071998, "open-mechanical": t_imp},
                {"名前": "着地時の変位 [in]", "理論": -1.0, "LS-DYNA": -1.0005, "open-mechanical": z_imp},
                {"名前": "着地時の速度 [in/s]", "理論": -math.sqrt(2 * g * h), "LS-DYNA": -27.791, "open-mechanical": v_imp},
                {"名前": "着地時の運動エネルギー [lbf·in]", "理論": mball * g * h, "LS-DYNA": 193.10,
                 "open-mechanical": 0.5 * mball * v_imp ** 2},
                {"名前": "最大変位の時刻 [s]", "理論": 0.10044, "LS-DYNA": 0.10000, "open-mechanical": float(t[j] + tm)},
                {"名前": "最大変位 [in]", "理論": -(h + x), "LS-DYNA": -1.5505, "open-mechanical": float(zm)},
            ], "グラフ": {"x": t.tolist(), "y": zb.tolist(), "xl": "時間 [s]", "yl": "ボールの変位 Z [in]"}}


# ---------------------------------------------------------------------------- 3. 単純支持はりの固有振動数
def case3():
    L, b, E, rho = 80.0, 2.0, 3e7, 7.33e-4
    m = structured_box(L, b, b, 40, 2, 2).to_quadratic()
    beam = mat("鋼", E, 0.0, rho)
    X = m.nodes
    end0 = np.nonzero((np.abs(X[:, 0]) < 1e-9) & (np.abs(X[:, 2] - b / 2) < 1e-9))[0]
    end1 = np.nonzero((np.abs(X[:, 0] - L) < 1e-9) & (np.abs(X[:, 2] - b / 2) < 1e-9))[0]
    bcs = [BC("displacement", [], [0.0, 0.0, 0.0], nodes=list(map(int, end0))),
           BC("displacement", [], [None, 0.0, 0.0], nodes=list(map(int, end1)))]
    v0 = np.zeros((m.n_nodes, 3))
    v0[:, 2] = np.sin(math.pi * X[:, 0] / L)              # 1 次モードの形の初速度
    mid = int(np.argmin(np.linalg.norm(X - [L / 2, b / 2, b / 2], axis=1)))
    I = b ** 4 / 12
    f_th = math.pi / 2 * math.sqrt(E * I / (rho * b * b * L ** 4))
    T = 1 / f_th
    s = DynamicSettings(method="implicit", end_time=4 * T, dt=T / 100, n_frames=1)
    t0 = time.time()
    fr = solve_dynamic(m, beam, bcs, s, log=QUIET, v0=v0,
                       monitor=lambda t, u, v: {"w": float(u.reshape(-1, 3)[mid, 2])})
    h = fr[-1].history
    t, w = np.array(h["時間 [s]"]), np.array(h["w"])
    z = [t[i] - w[i] * (t[i + 1] - t[i]) / (w[i + 1] - w[i]) for i in range(len(w) - 1) if w[i] > 0 >= w[i + 1]]
    f = 1 / float(np.mean(np.diff(z)))
    return {"id": "VM-LSDYNA-SOLVE-032", "名前": "単純支持はりの固有振動数", "解析": "陰解法（自由振動の周期から）",
            "要素": f"Tet10 {m.n_elems:,} 要素（LS-DYNA: はり要素・固有値解析）", "計算時間 [s]": round(time.time() - t0, 1),
            "項目": [{"名前": "1 次固有振動数 [Hz]", "理論": 28.766, "LS-DYNA": 28.67, "open-mechanical": f}],
            "グラフ": {"x": t.tolist(), "y": w.tolist(), "xl": "時間 [s]", "yl": "中央のたわみ [in]"}}


# ---------------------------------------------------------------------------- 4. 片持ちはり（微小変形）
def case4():
    L, b, E, nu, rho, P = 50.0, 2.0, 3e7, 0.3, 7.33e-4, 2000.0
    m = structured_box(L, b, b, 25, 2, 2).to_quadratic()
    beam = mat("鋼", E, nu, rho)
    tip = np.nonzero(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    bcs = [BC("fixed", faces_at(m, 0, 0.0)), BC("force", faces_at(m, 0, L), [0, 0, -P])]
    s = DynamicSettings(method="explicit", end_time=0.3, ramp_time=0.15, n_frames=1, alpha_damping=300.0)
    t0 = time.time()
    fr = solve_dynamic(m, beam, bcs, s, log=QUIET,
                       monitor=lambda t, u, v: {"w": float(u.reshape(-1, 3)[tip, 2].mean())})
    h = fr[-1].history
    t, w = np.array(h["時間 [s]"]), np.array(h["w"])
    I = b ** 4 / 12
    return {"id": "VM-LSDYNA-BENCHMARK-001", "名前": "片持ちはりのたわみ（微小変形）", "解析": "陽解法＋減衰（準静的）",
            "要素": f"Tet10 {m.n_elems:,} 要素（LS-DYNA: 六面体）", "計算時間 [s]": round(time.time() - t0, 1),
            "項目": [{"名前": "先端のたわみ [in]", "理論": -P * L ** 3 / (3 * E * I), "LS-DYNA": -2.0628,
                      "open-mechanical": float(w[-1])}],
            "グラフ": {"x": t.tolist(), "y": w.tolist(), "xl": "時間 [s]", "yl": "先端のたわみ [in]"}}


# ---------------------------------------------------------------------------- 5. 弾完全塑性はり・大変形
def case5():
    L, b, hgt, E, nu, sy, M = 50.0, 3.0, 5.0, 200000.0, 0.3, 1300.0, 22000.0
    m = structured_box(L, b, hgt, 50, 3, 5).to_quadratic()
    beam = Material("鋼（弾完全塑性）", E, nu, 7850, sy, plasticity="bilinear_iso", tangent_modulus=0.0)
    tip = np.nonzero(np.abs(m.nodes[:, 0] - L) < 1e-9)[0]
    bcs = [BC("fixed", faces_at(m, 0, 0.0)), BC("moment", faces_at(m, 0, L), [0, M, 0])]
    s = DynamicSettings(method="explicit", end_time=0.3, ramp_time=0.2, n_frames=1, large_deflection=True,
                        mass_scaling_dt=1e-5, alpha_damping=200.0)

    def mon(t, u, v):
        U = u.reshape(-1, 3)[tip]
        return {"ux": float(U[:, 0].mean()), "uz": float(U[:, 2].mean())}
    t0 = time.time()
    fr = solve_dynamic(m, beam, bcs, s, log=QUIET, monitor=mon)
    h = fr[-1].history
    t, ux, uz = np.array(h["時間 [s]"]), np.array(h["ux"]), np.array(h["uz"])
    # 理論（純曲げ・弾完全塑性の矩形断面）
    My = sy * b * hgt ** 2 / 6
    kap = (2 * sy / (E * hgt)) / math.sqrt(2 * (1.5 - M / My))
    th = L * kap
    R = 1 / kap
    sgn = -1 if uz[-1] < 0 else 1
    return {"id": "VM-LSDYNA-BENCHMARK-003", "名前": "弾完全塑性はりの大変形（先端モーメント）",
            "解析": "陽解法＋塑性＋大変形（準静的）",
            "要素": f"Tet10 {m.n_elems:,} 要素（LS-DYNA: 六面体）", "計算時間 [s]": round(time.time() - t0, 1),
            "項目": [
                {"名前": "先端のたわみ [mm]", "理論": sgn * R * (1 - math.cos(th)), "LS-DYNA": sgn * 5.967,
                 "open-mechanical": float(uz[-1])},
                {"名前": "軸方向の縮み（投影長さ − L）[mm]", "理論": R * math.sin(th) - L, "LS-DYNA": -0.4769,
                 "open-mechanical": float(ux[-1])},
            ], "グラフ": {"x": t.tolist(), "y": uz.tolist(), "xl": "時間 [s]", "yl": "先端のたわみ [mm]"}}


CASES = {1: case1, 2: case2, 3: case3, 4: case4, 5: case5}

if __name__ == "__main__":
    todo = [int(a) for a in sys.argv[1:]] or list(CASES)
    path = os.path.join(OUT, "summary.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    for n in todo:
        print(f"--- 例題 {n} ---", flush=True)
        r = CASES[n]()
        allres[str(n)] = r
        print(f"{r['id']} {r['名前']}（{r['計算時間 [s]']} s）")
        for it in r["項目"]:
            print(f"  {it['名前']}: 理論 {it['理論']:.5g} / LS-DYNA {it['LS-DYNA']:.5g} / open-mechanical "
                  f"{it['open-mechanical']:.5g}（理論との差 {rel(it['open-mechanical'], it['理論']):+.2f}%、"
                  f"LS-DYNA との差 {rel(it['open-mechanical'], it['LS-DYNA']):+.2f}%）", flush=True)
        with open(path, "w") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1)

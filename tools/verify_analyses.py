"""新しい解析機能の検証: Ansys の検証マニュアル（VM）と同じく、理論解のある問題を解いて誤差と解析時間を表にする。

  python tools/verify_analyses.py [出力の json]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from openmech.benchmarks import ellipse_ring, mapped_mesh  # noqa: E402
from openmech.materials import LIBRARY, Material  # noqa: E402
from openmech.mesh import structured_box  # noqa: E402
from openmech.solver import BC, solve  # noqa: E402

S = Material("鋼", 200000.0, 0.3, 7850.0, 250.0, 460.0)
q = dict(log=lambda *a: None)
rows = []


def face(m, axis, val, sign):
    for f in m.face_info():
        if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99:
            return f["id"]
    raise KeyError


def add(cat, name, desc, item, theory, value, t, dof, unit=""):
    err = (value - theory) / theory * 100 if theory else 0.0
    rows.append({"分類": cat, "問題": name, "内容": desc, "項目": item, "理論値": float(theory), "計算値": float(value),
                 "誤差 [%]": float(err), "時間 [s]": round(float(t), 3), "自由度": int(dof), "単位": unit})
    print(f"{cat:8} {name:28} {item:22} 理論 {theory:12.6g} 計算 {value:12.6g} 誤差 {err:+7.3f}%  {t:7.3f}s  {dof:,}")


def timed(fn):
    t0 = time.time()
    r = fn()
    return r, time.time() - t0


def bar(L, b, h, nx, ny, nz):
    m = structured_box(L, b, h, nx, ny, nz).to_quadratic()
    m.group_faces()
    return m


E, rho = 200000.0, 7850e-12
# ---------------------------------------------------------------- モーダル
from openmech.modal import solve_buckling, solve_modal  # noqa: E402

L, b, h = 400.0, 10.0, 20.0
m = bar(L, b, h, 80, 2, 4)
I = b * h ** 3 / 12
r, t = timed(lambda: solve_modal(m, S, [BC("fixed", [face(m, 0, 0, -1)])], n_modes=8, **q))
c = np.sqrt(E * I / (rho * b * h * L ** 4)) / (2 * np.pi)
fz = [r.freqs[i] for i in range(8) if abs(r.modes[i][:, 2]).max() > 3 * abs(r.modes[i][:, 1]).max()][:3]
for k, beta in enumerate([1.875104, 4.694091, 7.854757]):
    add("モーダル", "片持ちはりの曲げ振動", f"{L:g}×{b:g}×{h:g} mm、一端固定（Tet10）", f"{k + 1} 次（高さ方向）",
        beta ** 2 * c, fz[k], t, r.info["自由度数"], "Hz")
m2 = bar(L, b, h, 80, 2, 4)
r, t = timed(lambda: solve_modal(m2, S, [], n_modes=8, **q))
add("モーダル", "両端自由はりの曲げ振動", "同じはり・支持なし（剛体モード 6 個の次）", "1 次（高さ方向）",
    4.730041 ** 2 * c, [f for i, f in enumerate(r.freqs[6:], 6)
                        if abs(r.modes[i][:, 2]).max() > 3 * abs(r.modes[i][:, 1]).max()][0], t, r.info["自由度数"], "Hz")
mp = bar(200, 10, 10, 40, 2, 2)
r, t = timed(lambda: solve_modal(mp, S, [BC("fixed", [face(mp, 0, 0, -1)]),
                                         BC("point_mass", [face(mp, 0, 200, 1)], [1.0, 0, 0, 0], point=[200, 5, 5],
                                            options={"behavior": "rigid"})], n_modes=2, **q))
kk = 3 * E * (10 ** 4 / 12) / 200 ** 3
add("モーダル", "先端に点質量の片持ちはり", "200×10×10 mm、先端に 1 kg（剛体結合）", "1 次",
    np.sqrt(kk * 1e3 / (1.0 + 0.2357 * 0.157)) / (2 * np.pi), r.freqs[0], t, r.info["自由度数"], "Hz")
# ---------------------------------------------------------------- 座屈
mb = bar(300, 10, 10, 60, 2, 2)
r, t = timed(lambda: solve_buckling(mb, S, [BC("fixed", [face(mb, 0, 0, -1)]),
                                            BC("force", [face(mb, 0, 300, 1)], [-1.0, 0, 0])], n_modes=1, **q))
add("座屈", "オイラー座屈（一端固定・他端自由）", "300×10×10 mm、先端に圧縮 1 N", "座屈荷重係数",
    np.pi ** 2 * E * (10 ** 4 / 12) / (4 * 300 ** 2), r.load_factors[0], t, r.info["自由度数"], "N")
# ---------------------------------------------------------------- 熱
from openmech.thermal import ThermalSettings, solve_thermal, thermal_stress  # noqa: E402

St = LIBRARY["構造用鋼 (Structural Steel)"]
mt = bar(100, 10, 10, 20, 2, 2)
a, bb = face(mt, 0, 0, -1), face(mt, 0, 100, 1)
x = mt.nodes[:, 0]
r, t = timed(lambda: solve_thermal(mt, St, [BC("temperature", [a], [100.]), BC("convection", [bb], [1000., 0.])], **q)[-1])
Q = 100 / (0.1 / 60.5e-4 + 10)
add("定常熱", "伝導＋対流（1 次元の壁）", "長さ 100 mm、一端 100 °C、他端 h = 1000 W/(m²·K)・0 °C", "対流面の温度",
    100 - Q * 0.1 / 60.5e-4, r.T[np.isclose(x, 100)].mean(), t, mt.n_nodes, "°C")
add("定常熱", "伝導＋対流（1 次元の壁）", "同上", "熱流量", Q, r.reactions[0], t, mt.n_nodes, "W")
r, t = timed(lambda: solve_thermal(mt, St, [BC("temperature", [a], [0.]), BC("temperature", [bb], [0.]),
                                            BC("heat_generation", [], [1e7])], **q)[-1])
add("定常熱", "内部発熱のある壁", "両端 0 °C、発熱 10⁷ W/m³", "最高温度", 1e7 * 0.1 ** 2 / (8 * 60.5), r.T.max(), t,
    mt.n_nodes, "°C")
r, t = timed(lambda: solve_thermal(mt, St, [BC("temperature", [a], [500.]), BC("radiation", [bb], [0.8, 20.])], **q)[-1])
Tb = r.T[np.isclose(x, 100)].mean()
add("定常熱", "伝導＋放射（非線形）", "一端 500 °C、他端 放射率 0.8・周囲 20 °C", "放射面の熱流束の釣り合い",
    0.8 * 5.670374419e-8 * ((Tb + 273.15) ** 4 - 293.15 ** 4), 60.5 * (500 - Tb) / 0.1, t, mt.n_nodes, "W/m²")
rs, t = timed(lambda: solve_thermal(mt, St, [BC("temperature", [a], [100.])],
                                    ThermalSettings(transient=True, init_temp=0., end_time=100., n_steps=100), **q))
al = 60.5 / (7850 * 434)
th = 100 * (1 - sum(4 / ((2 * n + 1) * np.pi) * np.sin((2 * n + 1) * np.pi * 0.05 / 0.2)
                    * np.exp(-al * ((2 * n + 1) * np.pi / 0.2) ** 2 * 100) for n in range(200)))
add("過渡熱", "一端の温度を急に上げた棒", "0 °C の棒の一端を 100 °C（他端 断熱）、100 秒後", "中央の温度", th,
    rs[-1].T[np.isclose(x, 50)].mean(), t, mt.n_nodes, "°C")
m3 = bar(100, 10, 10, 40, 4, 4)
a3, b3 = face(m3, 0, 0, -1), face(m3, 0, 100, 1)
r, t = timed(lambda: thermal_stress(m3, St, [BC("frictionless", [a3]), BC("frictionless", [b3]),
                                             BC("frictionless", [face(m3, 1, 0, -1)]),
                                             BC("frictionless", [face(m3, 2, 0, -1)])], np.full(m3.n_nodes, 122.), **q))
add("熱応力", "両端を拘束した棒の温度上昇", "100 °C 上昇、軸方向のみ拘束", "軸方向応力", -200000 * 1.2e-5 * 100,
    r.stress[:, 0].mean(), t, r.info["自由度数"], "MPa")
# ---------------------------------------------------------------- リモート・支持・ボルト
mr = bar(200, 10, 10, 40, 3, 3)
Ir = 10 ** 4 / 12
for beh, jp in (("rigid", "剛体"), ("deformable", "変形可")):
    r, t = timed(lambda: solve(mr, S, [BC("fixed", [face(mr, 0, 0, -1)]),
                                       BC("remote_force", [face(mr, 0, 200, 1)], [0, 0, -100, 0, 0, 0], point=[250, 5, 5],
                                          options={"behavior": beh})], **q))
    add("リモート", f"離れた点の荷重（{jp}）", "片持ちはりの先端から 50 mm 先に 100 N", "先端のたわみ",
        100 * 200 ** 3 / (3 * E * Ir) + 100 * 50 * 200 ** 2 / (2 * E * Ir),
        -r.u[np.isclose(mr.nodes[:, 0], 200), 2].mean(), t, r.info["自由度数"], "mm")
r, t = timed(lambda: solve(mr, S, [BC("fixed", [face(mr, 0, 0, -1)]),
                                   BC("remote_displacement", [face(mr, 0, 200, 1)], [None, None, -1.0, None, None, None],
                                      options={"behavior": "rigid"})], **q))
add("リモート", "リモート変位（剛体）", "先端を 1 mm 押し下げる（回転は自由）", "反力", 3 * E * Ir / 200 ** 3,
    -r.reactions[1][2], t, r.info["自由度数"], "N")
aa, bo = 50., 80.
mc, P = mapped_mesh(ellipse_ring(aa, aa, bo, bo, 0, 5), 6, 16, 1)
r, t = timed(lambda: solve(mc, S, [BC("pressure", [0], [10.]), BC("cylindrical_support", [1], [0.0, None, None]),
                                   BC("frictionless", [2]), BC("frictionless", [3]),
                                   BC("displacement", [4], [None, None, 0.0])], **q))
nu = 0.3
B = -10 / (-(1 + nu) / ((1 - nu) * bo ** 2) - 1 / aa ** 2)
A_ = -(1 + nu) * B / ((1 - nu) * bo ** 2)
add("支持", "円筒支持（外周を半径方向に固定）", "内圧 10 MPa の厚肉リング（内径 100・外径 160 mm）", "内面の半径方向変位",
    (1 / E) * ((1 - nu) * A_ * aa + (1 + nu) * B / aa),
    np.linalg.norm(r.u[np.isclose(P[:, 0], 0)][:, :2], axis=1).mean(), t, r.info["自由度数"], "mm")
me = bar(200, 10, 10, 40, 3, 3)
r, t = timed(lambda: solve(me, S, [BC("elastic_support", [face(me, 2, 0, -1)], [10.0]),
                               BC("pressure", [face(me, 2, 10, 1)], [1.0])], **q))
add("支持", "弾性支持", "下面 基礎剛性 10 N/mm³、上面 圧力 1 MPa", "沈下量", 0.1, -r.u[:, 2].mean(), t,
    r.info["自由度数"], "mm")
from openmech.bolt import solve_pretension  # noqa: E402

mbo = bar(100, 10, 10, 20, 2, 2)
mbo.elem_body = np.zeros(mbo.n_elems, int)
fa, fb = face(mbo, 0, 0, -1), face(mbo, 0, 100, 1)
(r1, r2), t = timed(lambda: solve_pretension(mbo, S, [BC("fixed", [fa]), BC("fixed", [fb]),
                                                      BC("bolt_pretension", [], [1000.0],
                                                         options={"body": 0, "axis": [1, 0, 0]}, point=[50.5, 5, 5])],
                                             lambda bcs, em: solve(mbo, S, bcs, elem_materials=em, **q), **q))
add("ボルト", "ボルト予張力", "断面 10×10 mm の棒に予張力 1000 N", "軸方向応力", 10.0,
    r1.stress[mbo.nodes[:, 0] < 40, 0].mean(), t, r1.info["自由度数"], "MPa")
# ---------------------------------------------------------------- はり要素
from openmech.beam import Frame  # noqa: E402

mats = {"st": S}
Lb = 1000.
Ib = 20 * 40 ** 3 / 12
md = {"nodes": [[0, 0, 0], [Lb, 0, 0]], "members": [{"n": [0, 1], "section": "r", "material": "st", "div": 10}],
      "sections": {"r": {"type": "rect", "b": 20, "h": 40}}, "supports": [{"node": 0, "dof": [1] * 6}],
      "loads": [{"node": 1, "F": [0, 0, -1000, 0, 0, 0]}], "shear_deformation": False}
r, t = timed(lambda: Frame(md, mats).solve(**q))
add("はり", "片持ちはり（先端荷重）", "1000 mm・断面 20×40、先端 1000 N", "先端のたわみ", 1000 * Lb ** 3 / (3 * E * Ib),
    -r.u[1, 2], t, 66, "mm")
add("はり", "片持ちはり（先端荷重）", "同上", "根元の最大曲げ応力", 1000 * Lb * 20 / Ib, r.stress.max(), t, 66, "MPa")
md2 = dict(md, supports=[{"node": 0, "dof": [1, 1, 1, 1, 0, 0]}, {"node": 1, "dof": [0, 1, 1, 0, 0, 0]}],
           loads=[{"member": 0, "q": [0, 0, -1.0]}])
fr = Frame(md2, mats)
r, t = timed(lambda: fr.solve(**q))
add("はり", "単純支持はり（等分布荷重）", "1000 mm、1 N/mm", "中央のたわみ", 5 * Lb ** 4 / (384 * E * Ib),
    -r.u[np.argmin(np.abs(fr.nodes[:, 0] - Lb / 2)), 2], t, 66, "mm")
md3 = dict(md, loads=[])
r, t = timed(lambda: Frame(md3, mats).modal(3, **q))
add("はり", "片持ちはりの固有振動数", "1000 mm・断面 20×40", "1 次（高さ方向）",
    1.875104 ** 2 / (2 * np.pi) * np.sqrt(E * Ib / (rho * 800 * Lb ** 4)), r.freqs[1], t, 66, "Hz")
# ---------------------------------------------------------------- 振動応答
from openmech.vibration import solve_harmonic, solve_random  # noqa: E402

mv = bar(200, 20, 10, 40, 4, 2)
fix, tip = face(mv, 0, 0, -1), face(mv, 0, 200, 1)
bcs = [BC("fixed", [fix]), BC("force", [tip], [0, 0, -10.])]
mdv, t0 = timed(lambda: solve_modal(mv, S, bcs, n_modes=12, **q))
st = solve(mv, S, bcs, **q)
n = int(np.argmax(np.isclose(mv.nodes[:, 0], 200)))
h, t = timed(lambda: solve_harmonic(mdv, mv, S, bcs, 1, 1000, 200, zeta=0.02, **q))
amp = h.node_response(n, 2)
add("調和応答", "片持ちはりの周波数応答", "先端に 10 N の正弦波、減衰比 2 %", "低い周波数の振幅（＝静的たわみ）",
    abs(st.u[n, 2]), amp[0], t + t0, mdv.info["自由度数"], "mm")
k1 = int(np.argmin(np.abs(h.freqs - mdv.freqs[0])))
from openmech.solver import build_load_and_constraints  # noqa: E402
Fv = build_load_and_constraints(mv, S, bcs[1:])[0]
P1 = float(mdv.modes[0].ravel() @ Fv)
add("調和応答", "片持ちはりの周波数応答", "同上（1 次モードだけの理論: 共振倍率 1/(2ζ)）", "1 次共振の振幅",
    abs(mdv.modes[0][n, 2] * P1) / (2 * np.pi * mdv.freqs[0]) ** 2 / (2 * 0.02), amp[k1], t + t0,
    mdv.info["自由度数"], "mm")
r, t = timed(lambda: solve_random(mdv, mv, S, [[10, 0.01], [3000, 0.01]], direction=(0, 0, 1), zeta=0.02, **q))
i = int(np.argmax(np.abs(mdv.participation[:, 2])))
fn = mdv.freqs[i]
add("ランダム振動", "片持ちはりの PSD 加振（Miles の式）", "Z 方向 0.01 G²/Hz（10–3000 Hz）、減衰比 2 %", "先端の変位 1σ",
    abs(mdv.modes[i][n, 2] * mdv.participation[i, 2]) * np.sqrt(np.pi / 2 * fn * 25 * 0.01) * 9806.65 / (2 * np.pi * fn) ** 2,
    r.u[n, 2], t + t0, mdv.info["自由度数"], "mm")
# ---------------------------------------------------------------- 疲労
from openmech.postproc import fatigue, sn_life  # noqa: E402

mf = bar(100, 10, 10, 20, 2, 2)
r = solve(mf, St, [BC("fixed", [face(mf, 0, 0, -1)]), BC("force", [face(mf, 0, 100, 1)], [20000., 0, 0])], **q)
f, t = timed(lambda: fatigue(r.stress, St, loading="zero", mean_corr="goodman", design_life=1e6))
Sa = 100 / (1 - 100 / 460)
add("疲労", "片振り引張（Goodman 補正）", "公称応力 200 MPa の片振り、引張強さ 460 MPa", "等価応力振幅", Sa,
    np.median(f["応力振幅（補正後） [MPa]"]), t, 0, "MPa")
add("疲労", "片振り引張（Goodman 補正）", "同上（S-N 曲線を両対数で補間）", "寿命", float(sn_life([Sa], St.sn_table)[0]),
    np.median(f["寿命 [回]"]), t, 0, "回")
# ---------------------------------------------------------------- 電気
from openmech.electric import EPS0, solve_electrostatic, solve_piezo_static  # noqa: E402

mel = bar(20, 20, 1, 10, 10, 2)
bot, top = face(mel, 2, 0, -1), face(mel, 2, 1, 1)
r, t = timed(lambda: solve_electrostatic(mel, Material("誘電体", 1000, 0.3, 1000, permittivity=4.0),
                                         [BC("ground", [bot]), BC("voltage", [top], [100.])], **q))
add("静電場", "平行平板コンデンサ", "20×20 mm、間隔 1 mm、比誘電率 4", "静電容量", 4 * EPS0 * 4e-4 / 1e-3 * 1e12,
    r.info["静電容量 [pF]"], t, mel.n_nodes, "pF")
e = np.zeros((6, 3))
e[2, 2] = 15.1
Pz = Material("圧電体", 60000, 0.3, 7500, permittivity=1000, piezo={"e": e.tolist(), "eps_s": [1000] * 3})
allf = [ff["id"] for ff in mel.face_info()]
(s, _), t = timed(lambda: solve_piezo_static(mel, Pz, [BC("fixed", allf), BC("ground", [bot]),
                                                        BC("voltage", [top], [100.])], **q))
add("圧電", "拘束した圧電板に電圧", "厚さ 1 mm、100 V、e33 = 15.1 C/m²", "厚さ方向の応力", 15.1e-3 * 100,
    s.stress[:, 2].mean(), t, 4 * mel.n_nodes, "MPa")

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "logs", "verify_analyses.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fp:
    json.dump(rows, fp, ensure_ascii=False, indent=1)
print(f"{len(rows)} 項目 / 誤差の最大 {max(abs(r['誤差 [%]']) for r in rows):.3f}%")

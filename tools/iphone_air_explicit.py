"""iPhone Air 詳細モデルの落下（1 m）を陽解法（LS-DYNA 方式）で解く

  python tools/iphone_air_explicit.py [ケース名...]   （corner / back / face / bottom。省略時は corner と back）

剛体の床（剛体壁・摩擦 0.2）に、高さ H から落ちた速度 √(2gH) で当てる。等価静解析と違い、
衝撃の加速度は自動で決まり、応力の波・跳ね返り・回転も入る。結果は logs/iphone_air_explicit/
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from iphone_air_model import p, body_name, body_mat  # noqa: E402
from openmech.dynamics import DynamicSettings, element_dt, solve_explicit  # noqa: E402
from openmech.solver import _B_matrix, _dN_dx, _quadrature, material_D, shape_dN_dxi  # noqa: E402

OUT = os.path.join(ROOT, "logs", "iphone_air_explicit")
os.makedirs(OUT, exist_ok=True)
H = float(os.environ.get("HEIGHT", "1000"))
G0 = 9806.65
log = open(os.path.join(OUT, "log.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


CASES = {
    "corner": ("角から落下（左下の角）", (-1 / math.sqrt(2), -1 / math.sqrt(2), 0.0), 5e-4),
    "back": ("背面から落下（カメラ側から平らに）", (0.0, 0.0, -1.0), 8e-4),
    "face": ("画面から落下（平らに）", (0.0, 0.0, 1.0), 3e-4),
    "bottom": ("下端（USB-C 側）から落下", (0.0, -1.0, 0.0), 3e-4),
}
todo = [a for a in sys.argv[1:] if a in CASES] or ["corner", "back"]

t0 = time.time()
mesh = p.mesh(log=lambda *a: P("  ", *a))
P(f"メッシュ: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,}（{time.time() - t0:.0f} s）")
mats, emat = p.elem_materials(mesh)
D = np.stack([material_D(m) for m in mats])
rho_e = np.array([m.density_t_mm3 for m in mats])[emat]
vol = np.abs(mesh.element_volumes())
mass = float((rho_e * vol).sum())
P(f"質量 {mass * 1e6:.1f} g")

# 時間増分と質量スケーリング: 追加質量 2% 以内で一番大きな時間増分
t1 = time.time()
dte = element_dt(mesh, D, emat, rho_e) * 0.9
P(f"要素の安定時間増分: 最小 {dte.min():.2e} s / 中央値 {np.median(dte):.2e} s（{time.time() - t1:.0f} s）")
target = float(dte.min())
for cand in (2e-7, 1.5e-7, 1e-7, 7e-8, 5e-8, 3e-8, 2e-8):
    if cand <= dte.min():
        break
    add = float((rho_e * vol * (np.maximum(1, (cand / dte) ** 2) - 1)).sum())
    if add / mass < 0.02:
        target = cand
        P(f"質量スケーリング: 目標 {cand:.1e} s（追加質量 {add / mass * 100:.2f} %）")
        break

pts, _ = _quadrature(mesh.elems.shape[1])
dNs = [shape_dN_dxi(L, mesh.elems.shape[1]) for L in pts]


def element_stress_max(u):
    vm_e = np.zeros(mesh.n_elems)
    s1_e = np.full(mesh.n_elems, -np.inf)
    for a0 in range(0, mesh.n_elems, 20000):
        el = mesh.elems[a0:a0 + 20000]
        X = mesh.nodes[el]
        ue = u[el].reshape(len(el), -1)
        De = D[emat[a0:a0 + 20000]]
        for dN in dNs:
            dNdx, _ = _dN_dx(X, dN)
            sig = np.einsum("eij,ej->ei", De, np.einsum("eij,ej->ei", _B_matrix(dNdx), ue))
            sx, sy, sz, txy, tyz, tzx = sig.T
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
            T3 = np.stack([np.stack([sx, txy, tzx], -1), np.stack([txy, sy, tyz], -1),
                           np.stack([tzx, tyz, sz], -1)], -2)
            s1 = np.linalg.eigvalsh(T3)[:, -1]
            vm_e[a0:a0 + 20000] = np.maximum(vm_e[a0:a0 + 20000], vm)
            s1_e[a0:a0 + 20000] = np.maximum(s1_e[a0:a0 + 20000], s1)
    return vm_e, s1_e


summary = {"落下高さ [mm]": H, "質量 [g]": mass * 1e6, "時間増分 [s]": target,
           "要素の安定時間増分の最小 [s]": float(dte.min()), "ケース": {}}
v0 = math.sqrt(2 * G0 * H)
for key in todo:
    name, d, tend = CASES[key]
    d = np.array(d) / np.linalg.norm(d)
    far = mesh.nodes[int(np.argmax(mesh.nodes @ d))]
    s = DynamicSettings(method="explicit", end_time=tend, n_frames=25, init_velocity=list(v0 * d),
                        mass_scaling_dt=target if target > dte.min() else 0.0,
                        wall={"normal": list(-d), "point": list(far), "friction": 0.2})
    P(f"\n=== {name}（{H / 1000:g} m、{v0 / 1000:.2f} m/s）: 解析時間 {tend * 1e3:g} ms")
    t1 = time.time()
    frames = solve_explicit(mesh, p.material, [], s, elem_materials=(mats, emat), log=lambda *a: P("  ", *a))
    h = frames[-1].history
    fc = np.array(h["接触力 [N]"])
    ipk = int(fc.argmax())
    # 部品ごとの最大応力（全時刻の最大。要素の積分点で評価）
    vm_max = np.zeros(mesh.n_elems)
    s1_max = np.full(mesh.n_elems, -np.inf)
    worst_frame, worst_val = 0, -1.0
    for j, r in enumerate(frames):
        vm_e, s1_e = element_stress_max(r.u)
        vm_max = np.maximum(vm_max, vm_e)
        s1_max = np.maximum(s1_max, s1_e)
        if vm_e.max() > worst_val:
            worst_frame, worst_val = j, float(vm_e.max())
    per = {}
    for b in mesh.body_ids():
        em_b = mesh.elem_body == b
        nm = body_name.get(int(b), f"B{b}")
        mat = p.materials[body_mat[int(b)]]
        brittle = any(w in mat.name for w in ("ガラス", "サファイア", "IC", "磁石"))
        val = float(s1_max[em_b].max()) if brittle else float(vm_max[em_b].max())
        lim = (mat.tensile_strength or mat.yield_strength) if brittle else mat.yield_strength
        per[nm] = {"最大相当応力 [MPa]": float(vm_max[em_b].max()), "最大主応力 [MPa]": float(s1_max[em_b].max()),
                   "判定": "最大主応力（脆性）" if brittle else "相当応力（降伏）", "限界 [MPa]": lim,
                   "比（応力/限界）": val / lim if lim else 0.0}
    case = {"最大接触力 [N]": float(fc.max()), "最大接触力の時刻 [ms]": h["時間 [s]"][ipk] * 1e3,
            "ピーク加速度（接触力/質量）[G]": float(fc.max() / (mass * G0)),
            "接触時間 [ms]": float(np.ptp(np.array(h["時間 [s]"])[fc > 0.02 * fc.max()]) * 1e3) if fc.max() > 0 else 0,
            "時間増分 [s]": frames[-1].info["時間増分 [s]"], "ステップ数": frames[-1].info["ステップ数"],
            "追加質量 [%]": frames[-1].info["追加質量 [%]"], "計算時間 [s]": time.time() - t1,
            "部品": per, "時刻歴": {k: list(map(float, v)) for k, v in h.items()}}
    summary["ケース"][name] = case
    P(f"  最大接触力 {case['最大接触力 [N]']:,.0f} N（{case['最大接触力の時刻 [ms]']:.3f} ms）"
      f" → ピーク {case['ピーク加速度（接触力/質量）[G]']:,.0f} G、接触時間 {case['接触時間 [ms]']:.3f} ms"
      f"、計算 {case['計算時間 [s]']:.0f} s")
    for nm, v in sorted(per.items(), key=lambda kv: -kv[1]["比（応力/限界）"])[:12]:
        P(f"   {nm:<16} {v['判定']} 比 {v['比（応力/限界）']:.2f}（ミーゼス {v['最大相当応力 [MPa]']:.0f} / "
          f"主応力 {v['最大主応力 [MPa]']:.0f} / 限界 {v['限界 [MPa]']:g} MPa）")
    from openmech.contours import render  # noqa: E402
    r = frames[worst_frame]
    render(mesh, r.von_mises, f"{name} t={r.time * 1e3:.3f} ms", "MPa", os.path.join(OUT, f"{key}_vm.png"),
           view="below" if key == "back" else "iso")
    with open(os.path.join(OUT, "summary.json"), "w") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1, default=float)
    del frames

P(f"\n完了（合計 {time.time() - t0:.0f} s）")
log.close()

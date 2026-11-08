"""iPhone Air の詳細モデル（内部部品 約 40 点）を作り、落下（1 m）の等価静解析を行う

  python tools/iphone_air_drop.py [落下高さ mm]

公開情報（Apple 仕様・iFixit 分解）から寸法を決め、分からないところは推測。
落下はエネルギー等価の準静的解析（1 自由度ばね・剛な床・跳ね返りや減衰なし）:
  1 G の慣性力で床側を固定した線形解析 → 質量重みの平均変位 δ1 → 衝撃のピーク加速度 G = √(2h/δ1)
  → 応力・変形を G 倍。部品は接着（節点共有）としている。
結果: logs/iphone_air/ に 表（txt/json）・コンター図（png）・GUI で開けるプロジェクト（.omech）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from openmech.solver import BC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "iphone_air")
os.makedirs(OUT, exist_ok=True)
G0 = 9806.65                     # mm/s²
H = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0   # 落下高さ [mm]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iphone_air_model import *  # noqa: E402,F401,F403
from iphone_air_model import p, body_name, body_mat, R  # noqa: E402


log = open(os.path.join(OUT, "log.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


t0 = time.time()
mesh = p.mesh(log=lambda *a: P("  ", *a))
P(f"メッシュ: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / ボディ {len(mesh.body_ids())}（{time.time() - t0:.1f} s）")

# 浮いた部品（どこにもつながっていないボディ）がないか: 節点を共有するボディのつながり
import scipy.sparse as _sp  # noqa: E402
_nb = int(mesh.elem_body.max()) + 1
_nn = mesh.elems.shape[1]
_B = _sp.csr_matrix((np.ones(mesh.n_elems * _nn), (np.repeat(mesh.elem_body, _nn), mesh.elems.ravel())),
                    shape=(_nb, mesh.n_nodes))
_ncomp, _lab = _sp.csgraph.connected_components((_B @ _B.T) > 0)
if _ncomp > 1:
    _main = np.bincount(_lab[mesh.body_ids()]).argmax()
    for b in mesh.body_ids():
        if _lab[b] != _main:
            P(f"⚠ 浮いている部品: {body_name.get(int(b), b)}")
# 品質の悪い要素の場所（どの部品のどこか）
from openmech.geometry import mesh_quality  # noqa: E402
q = mesh_quality(mesh)["_quality"]
worst = np.argsort(q)[:15]
P("品質の低い要素（品質・部品・重心 [mm]）:")
for e in worst:
    c = mesh.nodes[mesh.elems[e, :4]].mean(0)
    P(f"   {q[e]:.4f}  {body_name.get(int(mesh.elem_body[e]), '?')}  ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
if os.environ.get("CHECK"):
    sys.exit(0)
vol = mesh.element_volumes()
rho = np.array([p.materials[body_mat[int(b)]].density for b in mesh.elem_body]) * 1e-12
mass_t = float((vol * rho).sum())
P(f"質量: {mass_t * 1e6:.1f} g（実機 165 g）")
mass_by_body = {}
for b in mesh.body_ids():
    msk = mesh.elem_body == b
    mass_by_body[body_name.get(int(b), f"B{b}")] = float((vol[msk] * rho[msk]).sum() * 1e6)
for n, g in mass_by_body.items():
    P(f"   {n}: {g:.1f} g")

# 節点質量（質量重みの平均変位用）
nn = mesh.elems.shape[1]
node_mass = np.bincount(mesh.elems.ravel(), np.repeat(vol * rho / nn, nn), mesh.n_nodes)

info = mesh.face_info()
lo, hi = mesh.bounds()


def faces_where(pred):
    return [f["id"] for f in info if pred(f)]


eps = 0.05
CASES = {
    "画面から落下（平面）": dict(
        faces=faces_where(lambda f: f["flat"] and f["normal"][2] > 0.99 and abs(f["centroid"][2] - hi[2]) < eps),
        dir=(0, 0, 1)),
    # 背面: カメラの出っ張りと、本体の下端（傾いて着地）の 2 点で床に当たる
    "背面から落下（カメラと下端が当たる）": dict(
        faces=faces_where(lambda f: f["flat"] and f["normal"][2] < -0.99 and abs(f["centroid"][2] - lo[2]) < eps),
        zfaces=faces_where(lambda f: f["flat"] and f["normal"][1] < -0.99 and abs(f["centroid"][1] - lo[1]) < eps),
        dir=(0, 0, -1)),
    "下端（USB-C 側）から落下": dict(
        faces=faces_where(lambda f: f["flat"] and f["normal"][1] < -0.99 and abs(f["centroid"][1] - lo[1]) < eps),
        dir=(0, -1, 0)),
    "角から落下（左下の角）": dict(
        faces=faces_where(lambda f: (not f["flat"]) and f["centroid"][0] < R and f["centroid"][1] < R
                          and np.hypot(*(f["centroid"][:2] - [R, R])) > 0.7 * R),
        dir=(-1 / math.sqrt(2), -1 / math.sqrt(2), 0)),
}

# 要素ごとの応力（積分点 4 点の最大）: 部品ごとの最大値は「その部品の要素だけ」で評価する
# （節点平均だと、柔らかい層と硬い部品の境目の節点で値が混ざるため）
from openmech.solver import _B_matrix, _dN_dx, _quadrature, material_D, shape_dN_dxi  # noqa: E402
_states, _emat = p.elem_materials(mesh)
_Ds = np.array([material_D(m) for m in _states])
_pts, _ = _quadrature(mesh.elems.shape[1])
_dN = [shape_dN_dxi(Lp, mesh.elems.shape[1]) for Lp in _pts]


def element_stress_max(u):
    """要素ごとの (最大相当応力, 最大主応力)"""
    vm_e = np.zeros(mesh.n_elems)
    s1_e = np.full(mesh.n_elems, -np.inf)
    for a0 in range(0, mesh.n_elems, 20000):
        el = mesh.elems[a0:a0 + 20000]
        X = mesh.nodes[el]
        ue = u[el].reshape(len(el), -1)
        D = _Ds[_emat[a0:a0 + 20000]]
        for dN in _dN:
            dNdx, _ = _dN_dx(X, dN)
            sig = np.einsum("eij,ej->ei", D, np.einsum("eij,ej->ei", _B_matrix(dNdx), ue))
            sx, sy, sz, txy, tyz, tzx = sig.T
            vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
            T3 = np.stack([np.stack([sx, txy, tzx], -1), np.stack([txy, sy, tyz], -1), np.stack([tzx, tyz, sz], -1)], -2)
            s1 = np.linalg.eigvalsh(T3)[:, -1]
            vm_e[a0:a0 + 20000] = np.maximum(vm_e[a0:a0 + 20000], vm)
            s1_e[a0:a0 + 20000] = np.maximum(s1_e[a0:a0 + 20000], s1)
    return vm_e, s1_e


body_of_node = {}
summary = {"落下高さ [mm]": H, "質量 [g]": mass_t * 1e6, "部品の質量 [g]": mass_by_body,
           "メッシュ": {"節点": mesh.n_nodes, "要素": mesh.n_elems}, "ケース": {}}
from openmech.contours import render  # noqa: E402

for name, c in CASES.items():
    if not c["faces"]:
        P(f"!! {name}: 床に当たる面が見つかりません")
        continue
    d = np.array(c["dir"], float)
    p.bcs = [BC("fixed", c["faces"]), BC("gravity", [], list(G0 * d))]
    if c.get("zfaces"):
        p.bcs.append(BC("displacement", c["zfaces"], [None, None, 0.0]))
    t1 = time.time()
    r = p.run(mesh, log=lambda *a: P("     ", *a) if a and ("反復" in str(a[0]) or "ソルバー" in str(a[0]) or "AMG" in str(a[0])) else None)
    ud = r.u @ d
    delta1 = float((node_mass * ud).sum() / node_mass.sum())       # 1 G での質量重み平均変位 [mm]
    Gpk = math.sqrt(2 * H / delta1)
    k = mass_t * G0 / delta1                                       # N/mm
    Tdur = math.pi * math.sqrt(mass_t / k)                         # 半正弦の衝撃時間 [s]
    vm = r.von_mises * Gpk
    s1 = r.principal[:, 0] * Gpk
    vm_e, s1_e = element_stress_max(r.u)
    vm_e *= Gpk
    s1_e *= Gpk
    per = {}
    for b in mesh.body_ids():
        em_b = mesh.elem_body == b
        nm = body_name.get(int(b), f"B{b}")
        mat = p.materials[body_mat[int(b)]]
        brittle = any(w in mat.name for w in ("ガラス", "サファイア", "IC", "NdFeB", "シリコン"))
        smax = float(s1_e[em_b].max()) if brittle else float(vm_e[em_b].max())
        lim = (mat.tensile_strength or mat.yield_strength) if brittle else mat.yield_strength
        per[nm] = {"最大相当応力 [MPa]": float(vm_e[em_b].max()), "最大主応力 [MPa]": float(s1_e[em_b].max()),
                   "強度 [MPa]": mat.tensile_strength or mat.yield_strength,
                   "降伏 [MPa]": mat.yield_strength,
                   "判定に使う応力": "最大主応力（脆性）" if brittle else "相当応力（降伏）",
                   "1000G あたり [MPa]": smax / Gpk * 1000,
                   "破損・降伏が始まる加速度 [G]": lim / (smax / Gpk) if smax > 0 else float("inf")}
    case = {"床に当たる面": c["faces"], "1G 平均変位 [mm]": delta1, "ピーク加速度 [G]": Gpk,
            "衝撃時間 [ms]": Tdur * 1e3, "最大変位 [mm]": float(r.disp_mag.max() * Gpk), "部品": per,
            "計算時間 [s]": time.time() - t1}
    summary["ケース"][name] = case
    P(f"\n=== {name}（{H / 1000:g} m）: ピーク {Gpk:,.0f} G / 衝撃時間 {Tdur * 1e3:.2f} ms / "
      f"最大変位 {case['最大変位 [mm]']:.3f} mm（{case['計算時間 [s]']:.1f} s）")
    for nm, v in per.items():
        P(f"   {nm:<14} ミーゼス {v['最大相当応力 [MPa]']:>8.0f}  主応力 {v['最大主応力 [MPa]']:>8.0f}  "
          f"1000G あたり {v['1000G あたり [MPa]']:>7.1f}  限界 {v['破損・降伏が始まる加速度 [G]']:>9,.0f} G")
    tag = ["face", "back", "bottom", "corner"][list(CASES).index(name)]
    for title, vals, unit, fn in (("相当応力（ミーゼス）", vm, "MPa", "vm"), ("最大主応力", s1, "MPa", "s1")):
        render(mesh, vals, f"{name} {title}", unit, os.path.join(OUT, f"{tag}_{fn}.png"),
               view="below" if tag == "back" else "iso")

with open(os.path.join(OUT, "summary.json"), "w") as fp:
    json.dump(summary, fp, ensure_ascii=False, indent=1, default=float)

# GUI で開けるプロジェクト（角から落下・ピーク加速度の慣性力）
cn = "角から落下（左下の角）"
if cn in summary["ケース"]:
    c = CASES[cn]
    g = summary["ケース"][cn]["ピーク加速度 [G]"]
    p.bcs = [BC("fixed", c["faces"]), BC("gravity", [], list(G0 * g * np.array(c["dir"])))]
p.save(os.path.join(ROOT, "examples", "iphone_air_drop.omech"))
P(f"\n完了（合計 {time.time() - t0:.0f} s）。プロジェクト: examples/iphone_air_drop.omech")
log.close()

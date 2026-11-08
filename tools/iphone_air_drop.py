"""iPhone Air の簡易モデル（内部部品込み）を作り、落下（1 m）の等価静解析を行う

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

from openmech.geometry import feature_body_keys, new_feature
from openmech.materials import Material
from openmech.project import Project
from openmech.solver import BC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "iphone_air")
os.makedirs(OUT, exist_ok=True)
G0 = 9806.65                     # mm/s²
H = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0   # 落下高さ [mm]

L, W, T = 156.2, 74.7, 5.64      # 本体 [mm]（Apple 公表）
R = 12.0                         # 角の丸み（推測）
WALL = 1.6                       # フレームの壁厚（推測）
LIP = 0.6                        # ガラスの外周とフレーム外形の差（推測）
Z_BG, Z_BAT, Z_MID, Z_DISP = 0.55, 3.25, 3.60, 4.94   # 積層の境界（推測）: 背面ガラス / 電池 / ミッドプレート / 表示部 / 前面ガラス

# 材料（入手できた値 + 推測）  E [MPa], ν, 密度 [kg/m³], 降伏 or 強度 [MPa]
MATS = {
    "チタン合金 Ti-6Al-4V（フレーム・プラトー）": Material("チタン合金 Ti-6Al-4V（フレーム・プラトー）", 113800, 0.342, 4430, 880, 950),
    "Ceramic Shield ガラスセラミック（前面・背面）": Material("Ceramic Shield ガラスセラミック（前面・背面）", 85000, 0.22, 2500, 600, 600),
    "OLED 表示部（積層・推測）": Material("OLED 表示部（積層・推測）", 6000, 0.30, 1800, 60, 80),
    "ミッドプレート（チタン・推測）": Material("ミッドプレート（チタン・推測）", 113800, 0.342, 4430, 880, 950),
    "電池（金属ケース入り・推測）": Material("電池（金属ケース入り・推測）", 12000, 0.30, 2200, 150, 200),
    "メイン基板 FR-4＋部品（推測）": Material("メイン基板 FR-4＋部品（推測）", 22000, 0.15, 2300, 250, 300),
    "シリコン（A19 Pro）": Material("シリコン（A19 Pro）", 130000, 0.28, 2330, 300, 300),
    "ステンレス SUS304（USB-C・Taptic）": Material("ステンレス SUS304（USB-C・Taptic）", 193000, 0.29, 7900, 215, 505),
    "Taptic Engine（鋼＋磁石・推測）": Material("Taptic Engine（鋼＋磁石・推測）", 150000, 0.30, 6000, 300, 400),
    "カメラ（アルミ＋サファイア・推測）": Material("カメラ（アルミ＋サファイア・推測）", 90000, 0.30, 3000, 275, 310),
}

feats, names, mat_of = [], [], []


def add(f, name, mat, fillet=None):
    feats.append(f)
    names.append(name)
    mat_of.append(mat)
    if fillet:
        k = feature_body_keys(feats)[len(feats) - 1]
        feats.append(new_feature("fillet", r=fillet, edges="Z 方向の辺", targets=[k]))
        names.append(None)
        mat_of.append(None)


def rbox(x0, y0, z0, x1, y1, z1, **kw):
    return new_feature("box", x=x0, y=y0, z=z0, dx=x1 - x0, dy=y1 - y0, dz=z1 - z0, **kw)


# 優先順位: 先に作ったボディが重なり部分を持つ（あとで作るフレームは残りの部分になる）
add(rbox(LIP, LIP, Z_DISP, W - LIP, L - LIP, T), "前面ガラス", "Ceramic Shield ガラスセラミック（前面・背面）", R - LIP)
add(rbox(6, 125.5, -2.2, W - 6, 150.5, 1.0), "メイン基板（プラトー内）", "メイン基板 FR-4＋部品（推測）")
add(rbox(38, 131, -2.6, 50, 143, -2.2), "A19 Pro", "シリコン（A19 Pro）")
add(rbox(LIP, LIP, 0, W - LIP, L - LIP, Z_BG), "背面ガラス", "Ceramic Shield ガラスセラミック（前面・背面）", R - LIP)
add(rbox(5, 14, Z_BG, W - 5, 121, Z_BAT), "電池 12.26 Wh", "電池（金属ケース入り・推測）")
add(rbox(8, 3, Z_BG, 28, 12, Z_BAT), "Taptic Engine", "Taptic Engine（鋼＋磁石・推測）")
add(rbox(30.85, 0, 0.8, 43.85, 7, Z_BAT), "USB-C ポート", "ステンレス SUS304（USB-C・Taptic）")
add(rbox(WALL, WALL, Z_BAT, W - WALL, L - WALL, Z_MID), "ミッドプレート", "ミッドプレート（チタン・推測）", R - WALL)
add(rbox(WALL, WALL, Z_MID, W - WALL, L - WALL, Z_DISP), "OLED 表示部", "OLED 表示部（積層・推測）", R - WALL)
add(new_feature("cylinder", x=16.0, y=138.0, z=-2.96 - 2.7, axis="Z", r=7.0, h=2.7), "カメラ（48MP）",
    "カメラ（アルミ＋サファイア・推測）")
add(rbox(3, 124, -2.96, W - 3, 152, 0), "カメラプラトー", "チタン合金 Ti-6Al-4V（フレーム・プラトー）", 6.0)
add(rbox(0, 0, 0, W, L, T), "チタンフレーム", "チタン合金 Ti-6Al-4V（フレーム・プラトー）", R)
k_frame = feature_body_keys(feats)[len(feats) - 2]
add(rbox(WALL, WALL, Z_BG, W - WALL, L - WALL, Z_DISP), None, None, R - WALL)   # くり抜き用
k_pocket = feature_body_keys(feats)[len(feats) - 2]
feats.append({"type": "bool", "op": "cut", "target": k_frame, "tools": [k_pocket]})
names.append(None)
mat_of.append(None)

keys = feature_body_keys(feats)
body_name, body_mat = {}, {}
for i, (n, m) in enumerate(zip(names, mat_of)):
    if n and i in keys:
        body_name[keys[i]] = n
        body_mat[keys[i]] = m

p = Project()
p.title = "iPhone Air 落下（等価静解析）"
p.geometry = {"kind": "model", "features": feats}
p.share_topology = "all"
p.mesh_size = float(os.environ.get("MESH", "2.0"))
p.mesh_order = 2
p.materials = {k: v for k, v in MATS.items()}
for b, m in body_mat.items():
    p.assign([b], attr=body_name[b], mat=m)
p.results = ["全変形 [mm]", "相当応力（ミーゼス） [MPa]", "最大主応力 [MPa]"]

log = open(os.path.join(OUT, "log.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


t0 = time.time()
mesh = p.mesh(log=lambda *a: P("  ", *a))
P(f"メッシュ: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / ボディ {len(mesh.body_ids())}（{time.time() - t0:.1f} s）")

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
    per = {}
    for b in mesh.body_ids():
        nodes = np.unique(mesh.elems[mesh.elem_body == b])
        nm = body_name.get(int(b), f"B{b}")
        mat = p.materials[body_mat[int(b)]]
        brittle = ("ガラス" in mat.name) or ("シリコン" in mat.name)
        smax = float(s1[nodes].max()) if brittle else float(vm[nodes].max())
        lim = (mat.tensile_strength or mat.yield_strength) if brittle else mat.yield_strength
        per[nm] = {"最大相当応力 [MPa]": float(vm[nodes].max()), "最大主応力 [MPa]": float(s1[nodes].max()),
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

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
# 積層（背面 z=0 → 前面 z=5.64）。電池厚 2.72 mm・プラトー 3.03 mm・カメラ込み 5.68 mm は公表／分解記事の値、他は推測
Z_BG = 0.55          # 背面ガラス 0 – 0.55
Z_MS = 1.00          # MagSafe（磁石リング・コイル）0.55 – 1.00
Z_BAT = Z_MS + 2.72  # 電池 1.00 – 3.72（鋼ケース入り 2.72 mm）
Z_ADH = 3.89         # 電池の接着層（電気剥離型）3.72 – 3.89
Z_MID = 4.19         # ミッドプレート 3.89 – 4.19
Z_FOAM = 4.39        # クッション材 4.19 – 4.39
Z_DISP = 4.94        # 表示パネル（OLED＋偏光板＋SUS 支持板）4.39 – 4.94、前面ガラス 4.94 – 5.64
PZ = -3.03           # プラトー（背面からの出っ張り 3.03 mm）
PC = PZ + 0.5        # プラトーのカバーガラス厚 0.5
CAM_Z = -5.68        # カメラの先端（背面から 5.68 mm）
PY0, PY1 = 127.0, 152.5          # プラトーの範囲（Y）

GLASS = "Ceramic Shield ガラスセラミック"
TI = "チタン合金 Ti-6Al-4V"
M = lambda *a, **k: Material(*a, **k)  # noqa: E731
MATS = {m.name: m for m in [
    M(TI, 113800, 0.342, 4430, 880, 950),
    M(GLASS, 85000, 0.22, 2500, 600, 600),
    M("サファイア（カメラカバー・Camera Control）", 345000, 0.29, 3980, 600, 600),
    M("表示パネル（OLED＋偏光板＋SUS 支持板・等価）", 40000, 0.30, 2860, 100, 120),
    M("クッション材（発泡ポリウレタン・推測）", 20, 0.30, 400, 1, 1.5),
    M("接着剤（電気剥離型・推測）", 100, 0.40, 1100, 8, 10),
    M("電池（鋼ケース・積層セル・直交異方性）", 15000, 0.30, 2500, 150, 200, elastic="orthotropic",
      ortho={"Ex": 15000, "Ey": 15000, "Ez": 1500, "nuxy": 0.3, "nuyz": 0.1, "nuxz": 0.1,
             "Gxy": 5800, "Gyz": 600, "Gxz": 600}),
    M("基板 FR-4（銅層込み）", 22000, 0.15, 2300, 250, 300),
    M("IC パッケージ（Si＋封止樹脂）", 30000, 0.25, 2300, 150, 150),
    M("ネオジム磁石 NdFeB", 160000, 0.24, 7500, 80, 80),
    M("ワイヤレス充電コイル（銅＋フェライト・等価）", 30000, 0.30, 4500, 70, 200),
    M("Taptic Engine（鋼＋磁石・等価）", 150000, 0.30, 6500, 300, 400),
    M("液晶ポリマー LCP（ガラス入り）", 10000, 0.35, 1600, 120, 150),
    M("カメラモジュール（アルミ筐体＋光学系・等価）", 40000, 0.30, 3200, 200, 250),
    M("アルミ合金 6061-T6", 69000, 0.33, 2700, 276, 310),
]}
SAPH = "サファイア（カメラカバー・Camera Control）"
IC = "IC パッケージ（Si＋封止樹脂）"
PCB = "基板 FR-4（銅層込み）"

feats, names, mat_of = [], [], []


def key_of_last():
    return feature_body_keys(feats)[len(feats) - 1]


def add(f, name, mat, fillet=None):
    feats.append(f)
    names.append(name)
    mat_of.append(mat)
    k = key_of_last()
    if fillet:
        feats.append(new_feature("fillet", r=fillet, edges="Z 方向の辺", targets=[k]))
        names.append(None)
        mat_of.append(None)
    return k


def cut(f, target):
    f = dict(f)
    f["op"] = "cut"
    f["targets"] = [target]
    feats.append(f)
    names.append(None)
    mat_of.append(None)


def bool_cut(target, tool):
    feats.append({"type": "bool", "op": "cut", "target": target, "tools": [tool]})
    names.append(None)
    mat_of.append(None)


def rbox(x0, y0, z0, x1, y1, z1, **kw):
    return new_feature("box", x=x0, y=y0, z=z0, dx=x1 - x0, dy=y1 - y0, dz=z1 - z0, **kw)


def cyl(x, y, z0, z1, r):
    return new_feature("cylinder", x=x, y=y, z=z0, axis="Z", r=r, h=z1 - z0)


# ---- 先に作ったボディが重なり部分を持つ（後のフレーム・プラトーは残りの部分になる）
# カメラ（48MP Fusion）: サファイアのカバー → 鏡筒 → チタンのリング → モジュール本体
CX, CY = 17.0, 139.5
add(cyl(CX, CY, CAM_Z, CAM_Z + 0.4, 6.5), "カメラ カバー（サファイア）", SAPH)
add(cyl(CX, CY, CAM_Z + 0.4, PC, 6.5), "カメラ 鏡筒（アルミ）", "アルミ合金 6061-T6")
add(cyl(CX, CY, CAM_Z, PC, 7.6), "カメラ リング（チタン）", TI)
add(rbox(CX - 8.5, CY - 8.5, PC, CX + 8.5, CY + 8.5, 3.3), "カメラ モジュール", "カメラモジュール（アルミ筐体＋光学系・等価）")
# 前面
add(rbox(LIP, LIP, Z_DISP, W - LIP, L - LIP, T), "前面ガラス（Ceramic Shield 2）", GLASS, R - LIP)
add(rbox(31, 147.0, 1.0, 43, 152.0, Z_MID - 0.3), "前面カメラ（TrueDepth）", "カメラモジュール（アルミ筐体＋光学系・等価）")
add(rbox(46, 150.0, 2.4, 60, 153.6, Z_ADH), "受話口スピーカー", "Taptic Engine（鋼＋磁石・等価）")
# メイン基板（プラトー内の 2 階建て: 下段基板 ＋ インターポーザー ＋ 上段基板）
add(rbox(40, 132, PC, 53, 145, -1.2), "A19 Pro（SoC・メモリ積層）", IC)
add(rbox(55, 132, PC + 0.2, 66, 145, -1.2), "NAND フラッシュ", IC)
add(rbox(28, 141, PC + 0.3, 37, 150, -1.2), "C1X モデム", IC)
add(rbox(28, 130, PC + 0.5, 34, 136, -1.2), "N1（無線）", IC)
add(rbox(5, 128.5, -1.2, 69, 151.0, -0.4), "メイン基板 下段", PCB)
k = add(rbox(27, 128.5, -0.4, 68, 151.0, 1.0), "インターポーザー（枠）", PCB)
cut(rbox(28.5, 130.0, -0.4, 66.5, 149.5, 1.0), k)
add(rbox(27, 128.5, 1.0, 68, 151.0, 1.8), "メイン基板 上段", PCB)
add(rbox(50, 134, 1.8, 58, 142, 2.6), "電源 IC（PMIC）", IC)
for i, (x, y) in enumerate([(29.5, 130.5), (65.5, 130.5), (29.5, 145.0), (65.5, 145.0)]):
    add(cyl(x, y, 1.8, Z_ADH, 1.2), f"基板の固定柱 {i + 1}", TI)
# 背面・プラトー
# プラトーは外形から 0.6 内側（上の角は外形と同心の丸み）。壁 0.6 の内側がカバーガラス
PW = 0.6
add(rbox(LIP + PW, PY0 + PW, PZ, W - LIP - PW, L - LIP - PW, PC), "プラトー カバーガラス", GLASS, R - LIP - PW)
kg = add(rbox(LIP, LIP, 0, W - LIP, L - LIP, Z_BG), "背面ガラス（Ceramic Shield）", GLASS, R - LIP)
cut(rbox(0, PY0 - 0.5, 0, W, L, Z_BG), kg)        # 上端はまっすぐ（その上はプラトー）
MSX, MSY = W / 2, 70.0
add(new_feature("pipe", x=MSX, y=MSY, z=Z_BG, axis="Z", ro=27.5, ri=23.5, h=Z_MS - Z_BG), "MagSafe 磁石リング",
    "ネオジム磁石 NdFeB")
add(new_feature("pipe", x=MSX, y=MSY, z=Z_BG, axis="Z", ro=20.0, ri=8.0, h=Z_MS - Z_BG), "充電コイル",
    "ワイヤレス充電コイル（銅＋フェライト・等価）")
# 電池（角の丸い本体＋下側の出っ張り。鋼ケース入り 2.72 mm）
kb = add(rbox(4.5, 20, Z_MS, W - 4.5, 124, Z_BAT), "電池 12.26 Wh", "電池（鋼ケース・積層セル・直交異方性）", 6.0)
feats.append(rbox(47, 13, Z_MS, 62, 21, Z_BAT, op="add", target=kb))
names.append(None)
mat_of.append(None)
add(rbox(6, 22, Z_BAT, W - 6, 122, Z_ADH), "電池の接着層", "接着剤（電気剥離型・推測）", 5.0)
add(rbox(8, 4, Z_MS, 28, 12.5, Z_BAT), "Taptic Engine", "Taptic Engine（鋼＋磁石・等価）")
# USB-C（3D プリントのチタン外殻＋樹脂の舌片）
ku = add(rbox(30.35, 0, 1.4, 44.35, 7, Z_BAT), "USB-C 外殻（チタン 3D プリント）", TI)
cut(rbox(33.0, 0, 1.95, 41.7, 6.0, 3.15), ku)
add(rbox(34.0, 0.8, 2.3, 40.7, 5.5, 2.8), "USB-C 舌片（LCP）", "液晶ポリマー LCP（ガラス入り）")
# ボタン（左: アクション・音量 / 右: サイド・Camera Control）
for nm, x0, x1, y0, y1, mat in [("アクションボタン", -0.5, 0.8, 112, 118, TI), ("音量＋", -0.5, 0.8, 99, 106, TI),
                                ("音量−", -0.5, 0.8, 89, 96, TI), ("サイドボタン", W - 0.8, W + 0.5, 100, 116, TI),
                                ("Camera Control", W - 0.8, W + 0.5, 46, 60, SAPH)]:
    add(rbox(x0, y0, 1.9, x1, y1, 3.7), nm, mat)
# アンテナの樹脂（フレームを 3 分割する切れ目。上下の左右）
for nm, y0 in [("アンテナ樹脂 下左", 16.0), ("アンテナ樹脂 上左", 138.0)]:
    add(rbox(0, y0, 0, WALL, y0 + 1.5, T), nm, "液晶ポリマー LCP（ガラス入り）")
for nm, y0 in [("アンテナ樹脂 下右", 16.0), ("アンテナ樹脂 上右", 138.0)]:
    add(rbox(W - WALL, y0, 0, W, y0 + 1.5, T), nm, "液晶ポリマー LCP（ガラス入り）")
# 中の板・表示部
add(rbox(WALL, WALL, Z_ADH, W - WALL, L - WALL, Z_MID), "ミッドプレート（チタン・推測）", TI, R - WALL)
add(rbox(WALL, WALL, Z_MID, W - WALL, L - WALL, Z_FOAM), "クッション材", "クッション材（発泡ポリウレタン・推測）", R - WALL)
add(rbox(WALL, WALL, Z_FOAM, W - WALL, L - WALL, Z_DISP), "表示パネル（OLED）", "表示パネル（OLED＋偏光板＋SUS 支持板・等価）",
    R - WALL)
# プラトー（チタンの枠。中は空洞）
kp = add(rbox(LIP, PY0, PZ, W - LIP, L - LIP, 0), "カメラプラトー（チタン）", TI, R - LIP)
kc = add(rbox(LIP + PW, PY0 + PW, PC, W - LIP - PW, L - LIP - PW, 0), None, None, R - LIP - PW)  # カバーガラスの層は残す
bool_cut(kp, kc)
# フレーム（チタン。内側をくり抜き、プラトーの下は開口）
kf = add(rbox(0, 0, 0, W, L, T), "チタンフレーム", TI, R)
# 空洞は電池の層だけくり抜く（ミッドプレート・クッション材・表示パネルは先に作ったので重なりはそちらが持つ。
# 同じ丸みの面を 2 つ重ねると細い隙間要素ができるため）
k1 = add(rbox(WALL, WALL, Z_BG, W - WALL, L - WALL, Z_ADH), None, None, R - WALL)
bool_cut(kf, k1)
# プラトーの下の開口: 空洞と同じ丸みの形を、プラトーより下（Y < PY0+0.6）で切った形（丸みの面が一致するように）
k2 = add(rbox(WALL, WALL, 0, W - WALL, L - WALL, Z_BG), None, None, R - WALL)
cut(rbox(0, 0, 0, W, PY0 + PW, Z_BG), k2)
bool_cut(kf, k2)

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
        brittle = any(w in mat.name for w in ("ガラス", "サファイア", "IC", "磁石", "シリコン"))
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

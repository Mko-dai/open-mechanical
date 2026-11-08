"""オリジナルのブリキ風ロボット（四角い頭・アンテナ 2 本・のぞき窓）を作り、足から落ちたときの等価静解析を行う

  python tools/tin_robot.py [落下高さ mm]

形はすべてフィーチャー（直方体・円柱・球・丸み付け・切り取り）で作るので、GUI で寸法を変えられる。
落下は iphone_air_drop.py と同じエネルギー等価の準静的解析（1 G → 平均変位 δ1 → ピーク G = √(2h/δ1)）。
結果: logs/tin_robot/ に コンター図（png）・表（txt）、examples/tin_robot.omech
"""
from __future__ import annotations

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
OUT = os.path.join(ROOT, "logs", "tin_robot")
os.makedirs(OUT, exist_ok=True)
G0 = 9806.65
H = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0

TIN = "ブリキ（鋼板・等価ソリッド）"
ALU = "アルミ合金 A5052"
MATS = {TIN: Material(TIN, 200000, 0.30, 7850, 250, 400),
        ALU: Material(ALU, 70600, 0.33, 2680, 195, 260)}

# ---- 寸法（Z が上、Y がマイナス側が正面）。変数にして GUI でも変えられるようにする
PARAMS = {"feet_h": 15.0, "leg_h": 50.0, "leg_r": 10.0, "body_w": 110.0, "body_d": 70.0, "body_h": 100.0,
          "head_w": 80.0, "head_d": 60.0, "head_h": 60.0, "arm_r": 9.0, "ant_r": 2.0, "ant_h": 25.0}
v = PARAMS

feats, names, mats = [], [], []


def add(f, name, mat, fillet=None):
    feats.append(f)
    names.append(name)
    mats.append(mat)
    k = feature_body_keys(feats)[len(feats) - 1]
    if fillet:
        feats.append(new_feature("fillet", r=fillet, edges="Z 方向の辺", targets=[k]))
        names.append(None)
        mats.append(None)
    return k


def cut(f, target):
    f = dict(f, op="cut", targets=[target])
    feats.append(f)
    names.append(None)
    mats.append(None)


def box(x0, y0, z0, x1, y1, z1):
    return new_feature("box", x=x0, y=y0, z=z0, dx=x1 - x0, dy=y1 - y0, dz=z1 - z0)


def cyl(x, y, z, axis, r, h):
    return new_feature("cylinder", x=x, y=y, z=z, axis=axis, r=r, h=h)


z_leg = v["feet_h"]
z_body = z_leg + v["leg_h"]
z_top = z_body + v["body_h"]
z_neck = z_top + 10.0
z_head_top = z_neck + v["head_h"]
bw, bd = v["body_w"] / 2, v["body_d"] / 2
hw, hd = v["head_w"] / 2, v["head_d"] / 2

# 小さい部品を先に（重なり部分は先に作ったボディに属する）
for s, side in ((-1, "左"), (1, "右")):
    add(box(s * 30 - 22, -32, 0, s * 30 + 22, 32, z_leg), f"{side}足", TIN, fillet=5)
    add(cyl(s * 30, 0, z_leg, "Z", v["leg_r"], v["leg_h"] + 2), f"{side}脚", TIN)
    # 腕: 肩から横へ → ひじから下へ → 四角い手
    xa = s * (bw - 2)
    add(cyl(xa, 0, z_top - 20, "X", v["arm_r"], s * 32), f"{side}上腕", TIN)
    xe = s * (bw + 30)
    add(cyl(xe, 0, z_top - 20 - 45, "Z", v["arm_r"], 45 + v["arm_r"]), f"{side}前腕", TIN)
    add(box(xe - 12, -12, z_top - 85, xe + 12, 12, z_top - 63), f"{side}手", TIN, fillet=3)
    # アンテナ（細い棒＋先端の玉）
    add(cyl(s * 22, 0, z_head_top - 2, "Z", v["ant_r"], v["ant_h"] + 2), f"{side}アンテナ", ALU)
    add(new_feature("sphere", x=s * 22, y=0, z=z_head_top + v["ant_h"] + 3, r=5.0), f"{side}アンテナの玉", ALU)

add(box(-30, -bd - 2, z_body + 35, 30, -bd + 1, z_body + 80), "胸のパネル", ALU, fillet=4)
add(cyl(0, 0, z_top - 2, "Z", 14, 14), "首", TIN)
head = add(box(-hw, -hd, z_neck, hw, hd, z_head_top), "頭", TIN, fillet=8)
cut(box(-28, -hd - 1, z_neck + 32, 28, -hd + 6, z_neck + 44), head)        # 目ののぞき窓
cut(box(-16, -hd - 1, z_neck + 12, 16, -hd + 3, z_neck + 18), head)        # 口のスリット
body = add(box(-bw, -bd, z_body, bw, bd, z_top), "胴体", TIN, fillet=10)

p = Project()
p.title = "ブリキのロボット 落下（等価静解析）"
p.geometry = {"kind": "model", "features": feats}
p.share_topology = "all"
p.mesh_size = float(os.environ.get("MESH", "5.0"))
p.mesh_order = 2
p.materials = dict(MATS)
keys = feature_body_keys(feats)
body_name, body_mat = {}, {}
for i, (n, m) in enumerate(zip(names, mats)):
    if n:
        body_name[keys[i]] = n
        body_mat[keys[i]] = m
for b, m in body_mat.items():
    p.assign([b], attr=body_name[b], mat=m)
p.results = ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"]

if __name__ == "__main__":
    log = open(os.path.join(OUT, "log.txt"), "w")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")

    t0 = time.time()
    mesh = p.mesh(log=lambda *a: None)
    P(f"メッシュ: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / ボディ {len(mesh.body_ids())}（{time.time() - t0:.1f} s）")
    from openmech.contours import render
    render(mesh, np.zeros(mesh.n_nodes), "ブリキのロボット（形状）", "-", os.path.join(OUT, "shape.png"))

    vol = mesh.element_volumes()
    rho = np.array([p.materials[body_mat[int(b)]].density for b in mesh.elem_body]) * 1e-12
    mass_t = float((vol * rho).sum())
    nn = mesh.elems.shape[1]
    node_mass = np.bincount(mesh.elems.ravel(), np.repeat(vol * rho / nn, nn), mesh.n_nodes)
    P(f"質量: {mass_t * 1e6:.0f} g（中身の詰まったソリッドとして）")

    lo, _ = mesh.bounds()
    feet = [f["id"] for f in mesh.face_info()
            if f["flat"] and f["normal"][2] < -0.99 and abs(f["centroid"][2] - lo[2]) < 0.05]
    d = np.array([0.0, 0.0, -1.0])
    p.bcs = [BC("fixed", feet), BC("gravity", [], list(G0 * d))]
    r = p.run(mesh, log=lambda *a: None)
    ud = r.u @ d
    delta1 = float((node_mass * ud).sum() / node_mass.sum())
    Gpk = math.sqrt(2 * H / delta1)
    P(f"足から {H / 1000:g} m 落下: ピーク {Gpk:,.0f} G / 最大変位 {r.disp_mag.max() * Gpk:.3f} mm / "
      f"最大ミーゼス {r.von_mises.max() * Gpk:.0f} MPa")
    for b in mesh.body_ids():
        nodes = np.unique(mesh.elems[mesh.elem_body == b])
        sv = r.von_mises[nodes].max() * Gpk
        ys = p.materials[body_mat[int(b)]].yield_strength
        P(f"   {body_name[int(b)]:<10} ミーゼス {sv:7.0f} MPa（降伏 {ys:.0f}）{'  ← 降伏' if sv > ys else ''}")
    render(mesh, r.von_mises * Gpk, f"足から {H / 1000:g} m 落下 相当応力", "MPa", os.path.join(OUT, "drop_vm.png"))
    render(mesh, r.disp_mag * Gpk, f"足から {H / 1000:g} m 落下 全変形", "mm", os.path.join(OUT, "drop_disp.png"))
    p.bcs = [BC("fixed", feet), BC("gravity", [], list(G0 * Gpk * d))]
    p.save(os.path.join(ROOT, "examples", "tin_robot.omech"))
    P(f"完了（{time.time() - t0:.0f} s）")

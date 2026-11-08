"""多層スイープの確認（Mac で実行。gmsh が必要）

  python tools/sweep_check.py      結果: logs/sweep_check.txt, logs/sweep_*.png

1. 3 層の積層板（層ごとに外形が違う＋貫通ピン）: 四面体（既定）とスイープでメッシュ品質・体積・面番号・
   曲げたわみを比べる
2. iPhone Air（42 部品）: スイープでメッシュ品質を測る（既定の四面体は 不良＋不可 8.4%）
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402

from openmech.geometry import build_mesh, new_feature  # noqa: E402
from openmech.materials import Material  # noqa: E402
from openmech.project import Project  # noqa: E402
from openmech.solver import BC  # noqa: E402

OUT = os.path.join(ROOT, "logs")
logf = open(os.path.join(OUT, "sweep_check.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def quality(m):
    import mesh_compare as mc
    st, skew, eq = mc.stats(m, 0.0, [])
    b = st["歪度の区分 [%]"]
    return (f"要素 {m.n_elems:,} / 品質 最小 {st['要素品質']['最小']:.3f} 平均 {st['要素品質']['平均']:.3f} / 歪度 最大 "
            f"{st['歪度']['最大']:.3f} / アスペクト比 最大 {st['アスペクト比']['最大']:.1f} / 不良＋不可 "
            f"{b['不良'] + b['不可']:.2f}%"), eq


def laminate():
    """B0 貫通ピン（先に作ったボディが重なり部分を持つ）/ B1 下の板（角 R5）/ B2 接着層 0.3 mm / B3 上の板"""
    return [new_feature("cylinder", x=50, y=20, z=0, axis="Z", r=3.0, h=3.3),
            new_feature("box", x=0, y=0, z=0, dx=100, dy=40, dz=2.0),
            new_feature("fillet", r=5.0, edges="Z 方向の辺", x0=-1, x1=101, y0=-1, y1=41, z0=-1, z1=2.5, targets=[1]),
            new_feature("box", x=10, y=5, z=2.0, dx=80, dy=30, dz=0.3),
            new_feature("box", x=0, y=0, z=2.3, dx=100, dy=40, dz=1.0)]


try:
    feats = laminate()
    spec = {"kind": "model", "features": feats}
    res = {}
    for name, ctrl in (("四面体（既定）", []), ("多層スイープ", [{"type": "sweep", "axis": "Z", "divs": 0, "layer_size": 0.7}])):
        t0 = time.time()
        logs = []
        st = {"capture_proximity": True} if ctrl is None else {}
        m = build_mesh(spec, 3.0, 2, log=lambda s: logs.append(s), local_sizes=ctrl or [], share={"*": 0},
                       settings=st)
        q, eq = quality(m)
        vol = {int(b): float(np.abs(m.element_volumes()[m.elem_body == b]).sum()) for b in np.unique(m.elem_body)}
        P(f"[積層板] {name}: {q} / {time.time() - t0:.1f} s")
        P("    体積 " + ", ".join(f"B{b} {v:.1f}" for b, v in vol.items()))
        for s in logs[-3:]:
            P("    " + str(s))
        res[name] = m
    # 面番号が四面体とスイープで同じ面を指すか
    a = {f["id"]: f for f in res["四面体（既定）"].face_info()}
    b = {f["id"]: f for f in res["多層スイープ"].face_info()}
    same = sum(1 for k in a if k in b and np.linalg.norm(np.subtract(a[k]["centroid"], b[k]["centroid"])) < 0.5)
    P(f"[積層板] 面の数 四面体 {len(a)} / スイープ {len(b)}、同じ番号で同じ場所の面 {same}")
    # 曲げ: x=0 の面を固定、x=100 の面に下向き 100 N
    for name, m in res.items():
        p = Project()
        p.geometry = spec
        p.share_topology = "all"
        p.material = Material("鋼", 200000, 0.3, 7850, 250)
        info = m.face_info()
        f0 = [f["id"] for f in info if abs(f["centroid"][0]) < 1e-6 and abs(f["normal"][0]) > 0.99]
        f1 = [f["id"] for f in info if abs(f["centroid"][0] - 100) < 1e-6 and abs(f["normal"][0]) > 0.99]
        p.bcs = [BC("fixed", f0), BC("force", f1, [0, 0, -100.0])]
        t0 = time.time()
        r = p.run(m, log=lambda *x: None)
        tip = np.abs(m.nodes[:, 0] - 100) < 1e-6
        P(f"[積層板] {name}: 先端たわみ {r.u[tip, 2].mean():.5f} mm / 最大ミーゼス {r.von_mises.max():.2f} MPa"
          f" / 解析 {time.time() - t0:.1f} s")
except Exception:
    import traceback
    P("NG 積層板\n" + traceback.format_exc())

try:
    os.environ.setdefault("MESH", "2.0")
    import iphone_air_model as ip
    spec = ip.p.geometry
    for name, ctrl in (("多層スイープ（面内 2 mm）", [{"type": "sweep", "axis": "Z", "divs": 0, "layer_size": 1.0}]),):
        t0 = time.time()
        logs = []
        m = build_mesh(spec, 2.0, 2, log=lambda s: logs.append(s), local_sizes=ctrl, share=ip.p.share_map())
        q, eq = quality(m)
        P(f"[iPhone] {name}: {q} / {time.time() - t0:.1f} s / ボディ {len(np.unique(m.elem_body))}")
        for s in logs[-4:]:
            P("    " + str(s))
        try:
            import mesh_compare as mc
            imgs = mc.render(m, eq, os.path.join(OUT, "sweep_iphone"))
            P("    画像 " + ", ".join(imgs))
        except Exception as e:  # noqa: BLE001
            P(f"    画像なし: {e}")
except Exception:
    import traceback
    P("NG iPhone\n" + traceback.format_exc())
P("完了")

"""歪な形状のメッシュ品質を、Ansys のメッシュメトリックと同じ指標・判定基準で調べる

  python tools/mesh_compare.py [番号...]      結果: logs/mesh_compare/summary.json, *.png, *.step

指標（Ansys Mechanical の「メッシュメトリック」と同じ定義、四面体の頂点 4 点で評価）
  要素品質   正四面体 = 1、つぶれた要素 = 0（Ansys の Element Quality）
  歪度       正四面体 = 0、つぶれた要素 = 1（等辺体積基準。Ansys の Skewness）
  アスペクト比 正四面体 = 1
  ヤコビアン比（2 次要素の曲がり）1 = 直線の要素と同じ、0 以下 = 裏返り
歪度は Ansys Meshing の判定区分（優 0–0.25 / 良好 0.25–0.5 / 良 0.5–0.8 / 可 0.8–0.95 /
不良 0.95–0.98 / 不可 0.98–1）で数える。

同じ形状を STEP で書き出すので、Ansys に読み込んで既定設定でメッシュし、
メッシュメトリックの値を summary.json と並べれば直接比較できる。
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from openmech.geometry import build_mesh, mesh_quality, new_feature, tet10_jacobian_ratio  # noqa: E402

OUT = os.path.join(ROOT, "logs", "mesh_compare")
os.makedirs(OUT, exist_ok=True)
BANDS = [("優", 0.0, 0.25), ("良好", 0.25, 0.5), ("良", 0.5, 0.8), ("可", 0.8, 0.95), ("不良", 0.95, 0.98),
         ("不可", 0.98, 1.0001)]


def F(t, op="new", **kw):
    return new_feature(t, op, **kw)


# ---------------------------------------------------------------------------- モデル
def m1():
    """L 字ブラケット: 内側の丸み R6、穴 2 つ、端の面取り"""
    f = [F("extrude", plane="XY", offset=0.0, points="0,0; 80,0; 80,8; 8,8; 8,60; 0,60", h=40.0),
         F("fillet", r=6.0, edges="Z 方向の辺", x0=7, x1=9, y0=7, y1=9, z0=-1, z1=41),
         F("cylinder", "cut", x=50.0, y=-1.0, z=20.0, axis="Y", r=5.0, h=10.0),
         F("cylinder", "cut", x=-1.0, y=42.0, z=20.0, axis="X", r=4.0, h=10.0),
         F("chamfer", d=2.0, edges="Z 方向の辺", x0=79, x1=81, y0=-1, y1=9, z0=-1, z1=41)]
    return "L 字ブラケット（丸み・穴・面取り）", f, 0.0


def m2():
    """薄肉ハウジング: 壁 1.5 mm、リブ 0.8 mm、ボス（パイプ）"""
    f = [F("box", dx=120.0, dy=80.0, dz=40.0),
         F("box", "cut", x=1.5, y=1.5, z=1.5, dx=117.0, dy=77.0, dz=40.0),
         F("box", "add", x=40.0, y=1.5, z=1.5, dx=0.8, dy=77.0, dz=28.0),
         F("box", "add", x=80.0, y=1.5, z=1.5, dx=0.8, dy=77.0, dz=28.0),
         F("pipe", "add", x=20.0, y=20.0, z=1.5, axis="Z", ro=4.0, ri=1.5, h=28.0),
         F("pipe", "add", x=100.0, y=60.0, z=1.5, axis="Z", ro=4.0, ri=1.5, h=28.0)]
    return "薄肉ハウジング（壁 1.5 mm・リブ 0.8 mm・ボス）", f, 0.0


def m3():
    """鋭角・接線ぎりぎりの形状: 1 mm だけ食い込む球、0.1 mm だけ重なる円柱、刃のような円錐の切り欠き"""
    f = [F("box", dx=60.0, dy=40.0, dz=20.0),
         F("sphere", "add", x=75.0, y=20.0, z=10.0, r=16.0),
         F("cylinder", "add", x=30.0, y=49.9, z=0.0, axis="Z", r=10.0, h=20.0),
         F("cone", "cut", x=10.0, y=20.0, z=-1.0, axis="Z", r1=14.0, r2=0.5, h=22.0)]
    return "鋭角・接線ぎりぎり（食い込み 1 mm の球・重なり 0.1 mm の円柱・円錐の切り欠き）", f, 0.0


def m4():
    """大小の寸法が混在: 200×100×5 の板に φ1 mm の小穴 5 個、φ20 の穴、角 R10、小さいボス"""
    f = [F("box", dx=200.0, dy=100.0, dz=5.0),
         F("fillet", r=10.0, edges="Z 方向の辺", x0=-1, x1=201, y0=-1, y1=101, z0=-1, z1=6),
         F("cylinder", "cut", x=60.0, y=50.0, z=-1.0, axis="Z", r=10.0, h=7.0)]
    for i in range(5):
        f.append(F("cylinder", "cut", x=110.0 + 15 * i, y=30.0, z=-1.0, axis="Z", r=0.5, h=7.0))
    f.append(F("cylinder", "add", x=150.0, y=75.0, z=5.0, axis="Z", r=1.5, h=3.0))
    return "寸法が混在した板（φ1 mm の小穴・R10・小さいボス）", f, 0.0


def m5():
    """iPhone Air の詳細モデル（42 部品・節点共有）"""
    os.environ.setdefault("MESH", "2.0")
    import importlib
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    mod = importlib.import_module("iphone_air_model")
    return "iPhone Air 詳細モデル（42 部品・節点共有）", mod.p.geometry["features"], float(mod.p.mesh_size), \
        mod.p.share_map()


MODELS = {1: m1, 2: m2, 3: m3, 4: m4, 5: m5}


# ---------------------------------------------------------------------------- 評価
def stats(mesh, secs, fix_log):
    q = mesh_quality(mesh)
    skew = q["_skew"]
    ar = q["_ar"]
    jr = tet10_jacobian_ratio(mesh)
    n = len(skew)
    bands = {name: round(100.0 * float(np.sum((skew >= lo) & (skew < hi))) / n, 3) for name, lo, hi in BANDS}
    eq = q["_quality"]
    return {
        "節点数": int(mesh.n_nodes), "要素数": int(mesh.n_elems), "ボディ数": int(len(mesh.body_ids()))
        if mesh.elem_body is not None else 1, "メッシュ時間 [s]": round(secs, 1),
        "要素品質": {**q["要素品質"], "1%点": float(np.percentile(eq, 1))},
        "歪度": {**q["歪度"], "99%点": float(np.percentile(skew, 99))},
        "アスペクト比": {**q["アスペクト比"], "99%点": float(np.percentile(ar, 99))},
        "ヤコビアン比": {"最小": float(jr.min()), "平均": float(jr.mean()), "0.5 未満の要素": int(np.sum(jr < 0.5))},
        "歪度の区分 [%]": bands,
        "要素品質 0.1 未満の要素": int(np.sum(eq < 0.1)),
        "要素の形": "三角柱" if getattr(mesh, "is_wedge", False) else "四面体",
        "自動修正": list(fix_log or []),
    }, skew, eq


def _ar(mesh):
    X = mesh.nodes[mesh.elems[:, :4]]
    e = [X[:, j] - X[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    l2 = np.stack([np.einsum("ij,ij->i", v, v) for v in e], 1)
    V = np.abs(np.einsum("ij,ij->i", np.cross(e[0], e[1]), e[2])) / 6
    faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    areas = np.stack([0.5 * np.linalg.norm(np.cross(X[:, f[1]] - X[:, f[0]], X[:, f[2]] - X[:, f[0]]), axis=1)
                      for f in faces], 1)
    return np.sqrt(l2.max(1)) / np.maximum(3 * V / np.maximum(areas.max(1), 1e-300), 1e-300) * np.sqrt(2 / 3)


def render(mesh, eq, path_base):
    """外観（メッシュ線あり）と断面（要素品質で色分け）の画像"""
    try:
        import pyvista as pv
    except ImportError:
        return []
    pv.OFF_SCREEN = True
    wedge = getattr(mesh, "is_wedge", False)
    k = 6 if wedge else 4
    cells = np.hstack([np.full((mesh.n_elems, 1), k), mesh.elems[:, :k]]).ravel()
    ctype = pv.CellType.WEDGE if wedge else pv.CellType.TETRA
    grid = pv.UnstructuredGrid(cells, np.full(mesh.n_elems, ctype, dtype=np.uint8), mesh.nodes)
    grid.cell_data["要素品質"] = eq
    out = []
    lo, hi = mesh.bounds()
    c = 0.5 * (lo + hi)
    for name, body in (("outer", grid.extract_surface()),
                       ("section", grid.clip(normal="y", origin=c, invert=True, crinkle=True))):
        pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
        pl.set_background("white")
        pl.add_mesh(body, scalars="要素品質", cmap="jet_r", clim=[0, 1], show_edges=True, edge_color="#333333",
                    line_width=0.5, n_colors=10,
                    scalar_bar_args={"title": "Element Quality", "vertical": True, "color": "black",
                                     "position_x": 0.88, "position_y": 0.15, "height": 0.7, "width": 0.06})
        pl.view_isometric()
        pl.camera.azimuth = -30
        pl.camera.elevation = 15 if name == "outer" else 5
        pl.reset_camera()
        pl.camera.zoom(1.25)
        p = f"{path_base}_{name}.png"
        pl.screenshot(p)
        pl.close()
        out.append(os.path.basename(p))
    return out


VARIANTS = {"既定": {}, "近接の捕捉": {"capture_proximity": True, "proximity_cells": 3}}
SKIP = {(5, "近接の捕捉")}      # 0.17 mm の層が多い iPhone は近接の捕捉だと数百万要素になるため省く


def run(n, variant="既定"):
    res = MODELS[n]()
    name, feats, size = res[:3]
    share = res[3] if len(res) > 3 else None
    spec = {"kind": "model", "features": feats}
    tag = f"model{n}" + ("" if variant == "既定" else "_prox")
    step = os.path.join(OUT, f"model{n}.step") if variant == "既定" else None
    logs = []
    t0 = time.time()
    try:
        mesh = build_mesh(spec, size, 2, log=lambda m: logs.append(str(m)), share=share, step_out=step,
                          settings=VARIANTS[variant])
    except Exception as e:  # noqa: BLE001
        return {"名前": name, "エラー": str(e), "ログ": logs[-5:]}
    s, skew, eq = stats(mesh, time.time() - t0, getattr(mesh, "fix_log", []))
    s["名前"] = name
    s["要素サイズ [mm]"] = size if size else "自動"
    s["画像"] = render(mesh, eq, os.path.join(OUT, tag))
    s["STEP"] = f"model{n}.step"
    s["ログ"] = logs[-5:]
    np.savez_compressed(os.path.join(OUT, f"{tag}_q.npz"), skew=skew.astype(np.float32), eq=eq.astype(np.float32))
    return s


if __name__ == "__main__":
    todo = [int(a) for a in sys.argv[1:]] or list(MODELS)
    path = os.path.join(OUT, "summary.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    for n in todo:
        for variant in VARIANTS:
            if (n, variant) in SKIP:
                continue
            print(f"--- モデル {n}（{variant}） ---", flush=True)
            r = run(n, variant)
            allres.setdefault(str(n), {})
            if "名前" in allres[str(n)]:
                allres[str(n)] = {}
            allres[str(n)][variant] = r
            if "エラー" in r:
                print(f"{r['名前']}: エラー {r['エラー']}", flush=True)
            else:
                print(f"{r['名前']}: 要素 {r['要素数']:,} / 節点 {r['節点数']:,} / {r['メッシュ時間 [s]']} s", flush=True)
                print(f"  要素品質 最小 {r['要素品質']['最小']:.3f} 平均 {r['要素品質']['平均']:.3f} / 歪度 最大 "
                      f"{r['歪度']['最大']:.3f} 平均 {r['歪度']['平均']:.3f} / アスペクト比 最大 {r['アスペクト比']['最大']:.1f}"
                      f" / ヤコビアン比 最小 {r['ヤコビアン比']['最小']:.3f}", flush=True)
                print("  歪度の区分 [%]: " + " ".join(f"{k} {v:.2f}" for k, v in r["歪度の区分 [%]"].items()), flush=True)
                for m in r["自動修正"]:
                    print("  自動修正: " + m, flush=True)
            with open(path, "w") as f:
                json.dump(allres, f, ensure_ascii=False, indent=1)
    print("完了", flush=True)

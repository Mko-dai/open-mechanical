"""形状の作成・読み込みとメッシュ生成（gmsh を使用）。

対応:
  - 簡単形状: 直方体 / 円柱 / パイプ / 穴あき板 / L字ブラケット / 球
  - STEP (.step/.stp)、IGES (.igs/.iges)、BREP (.brep) … CAD の面がそのまま「面」になります
  - STL (.stl) … 角度で面を自動分割して体積メッシュを作ります
  - Parasolid (.x_t/.x_b) … 無償のライブラリが存在しないため非対応（STEP で書き出してください）

単位は mm。
"""
from __future__ import annotations

import math
import os

import numpy as np

from .mesh import Mesh, structured_box

PRIMITIVES = {
    # kind: (表示名, [(パラメータ名, 表示名, 既定値)])
    "box": ("直方体", [("lx", "長さ X [mm]", 100.0), ("ly", "幅 Y [mm]", 20.0), ("lz", "高さ Z [mm]", 10.0)]),
    "cylinder": ("円柱", [("r", "半径 [mm]", 10.0), ("h", "長さ Z [mm]", 100.0)]),
    "pipe": ("パイプ", [("ro", "外半径 [mm]", 20.0), ("ri", "内半径 [mm]", 15.0), ("h", "長さ Z [mm]", 100.0)]),
    "plate_hole": ("穴あき板", [("lx", "長さ X [mm]", 200.0), ("ly", "幅 Y [mm]", 50.0),
                             ("t", "板厚 Z [mm]", 5.0), ("d", "穴径 [mm]", 20.0)]),
    "l_bracket": ("L字ブラケット", [("a", "水平長さ X [mm]", 80.0), ("b", "垂直高さ Z [mm]", 60.0),
                                ("w", "幅 Y [mm]", 30.0), ("t", "板厚 [mm]", 6.0),
                                ("d", "取付穴径 [mm]（0で無し）", 8.0)]),
    "sphere": ("球", [("r", "半径 [mm]", 25.0)]),
}

# ---------------------------------------------------------------------------- モデリング（フィーチャー）
# Ansys DesignModeler と同じく、形状を「フィーチャー」の積み重ねで作る。
# 各フィーチャーは 操作 op = add（追加）/ cut（切り取り）/ intersect（交差）を持つ。
AXES = ["X", "Y", "Z"]
PLANES = ["XY", "YZ", "ZX"]
FEATURES = {
    # type: (表示名, アイコン文字, [(キー, 表示名, 既定値)])  値が文字列の項目は選択肢
    "box": ("直方体", "箱", [("x", "原点 X [mm]", 0.0), ("y", "原点 Y [mm]", 0.0), ("z", "原点 Z [mm]", 0.0),
                          ("dx", "長さ X [mm]", 100.0), ("dy", "幅 Y [mm]", 50.0), ("dz", "高さ Z [mm]", 20.0)]),
    "cylinder": ("円柱", "円", [("x", "底面中心 X [mm]", 0.0), ("y", "底面中心 Y [mm]", 0.0),
                              ("z", "底面中心 Z [mm]", 0.0), ("axis", "軸方向", "Z"),
                              ("r", "半径 [mm]", 10.0), ("h", "長さ [mm]", 50.0)]),
    "pipe": ("パイプ", "管", [("x", "底面中心 X [mm]", 0.0), ("y", "底面中心 Y [mm]", 0.0),
                           ("z", "底面中心 Z [mm]", 0.0), ("axis", "軸方向", "Z"),
                           ("ro", "外半径 [mm]", 20.0), ("ri", "内半径 [mm]", 15.0), ("h", "長さ [mm]", 100.0)]),
    "cone": ("円錐", "錐", [("x", "底面中心 X [mm]", 0.0), ("y", "底面中心 Y [mm]", 0.0),
                          ("z", "底面中心 Z [mm]", 0.0), ("axis", "軸方向", "Z"),
                          ("r1", "底面半径 [mm]", 20.0), ("r2", "上面半径 [mm]", 5.0), ("h", "高さ [mm]", 40.0)]),
    "sphere": ("球", "球", [("x", "中心 X [mm]", 0.0), ("y", "中心 Y [mm]", 0.0), ("z", "中心 Z [mm]", 0.0),
                          ("r", "半径 [mm]", 25.0)]),
    "extrude": ("押し出し", "押", [("plane", "スケッチ平面", "XY"), ("offset", "平面の位置 [mm]", 0.0),
                               ("points", "頂点 (u,v; u,v; …) [mm]", "0,0; 80,0; 80,10; 10,10; 10,60; 0,60"),
                               ("h", "押し出し長さ [mm]", 30.0)]),
    # サーフェス（シェル要素で解く薄肉の面。Ansys のサーフェスボディ＋厚さ）
    "plate": ("平板（シェル）", "板", [("plane", "平面", "XY"), ("offset", "平面の位置 [mm]", 0.0),
                                  ("u0", "原点 u [mm]", 0.0), ("v0", "原点 v [mm]", 0.0),
                                  ("du", "幅 u [mm]", 100.0), ("dv", "幅 v [mm]", 50.0), ("t", "厚さ [mm]", 2.0)]),
    "sheet": ("多角形の面（シェル）", "面", [("plane", "平面", "XY"), ("offset", "平面の位置 [mm]", 0.0),
                                      ("points", "頂点 (u,v; u,v; …) [mm]", "0,0; 100,0; 100,40; 0,60"),
                                      ("t", "厚さ [mm]", 2.0)]),
    "cyl_shell": ("円筒面（シェル）", "筒", [("x", "底面中心 X [mm]", 0.0), ("y", "底面中心 Y [mm]", 0.0),
                                      ("z", "底面中心 Z [mm]", 0.0), ("axis", "軸方向", "Z"),
                                      ("r", "半径 [mm]", 50.0), ("h", "長さ [mm]", 100.0), ("t", "厚さ [mm]", 2.0)]),
    "disk": ("円板（シェル・2D）", "盤", [("plane", "平面", "XY"), ("offset", "平面の位置 [mm]", 0.0),
                                   ("u0", "中心 u [mm]", 0.0), ("v0", "中心 v [mm]", 0.0),
                                   ("r", "外半径 [mm]", 20.0), ("ri", "内半径 [mm]（0 = 穴なし）", 0.0),
                                   ("t", "厚さ [mm]", 2.0)]),
    # ボディの編集（Femtet の 移動・回転・コピー / 切断 / ストレッチ / フィレット / 面取り）
    "move": ("移動・回転", "移", [("copy", "操作", "移動"),
                             ("dx", "移動量 X [mm]", 0.0), ("dy", "移動量 Y [mm]", 0.0), ("dz", "移動量 Z [mm]", 0.0),
                             ("angle", "回転角 [°]", 0.0), ("axis", "回転軸", "Z"),
                             ("x", "回転中心 X [mm]", 0.0), ("y", "回転中心 Y [mm]", 0.0), ("z", "回転中心 Z [mm]", 0.0)]),
    "mirror": ("ミラー", "鏡", [("copy", "操作", "コピー"), ("axis", "鏡面の法線", "X"),
                            ("offset", "鏡面の位置 [mm]", 0.0)]),
    "split": ("切断", "切", [("axis", "切断面の法線", "X"), ("offset", "切断面の位置 [mm]", 0.0),
                           ("keep", "残す部分", "両方（2つのボディに分割）")]),
    "stretch": ("引き伸ばし", "伸", [("axis", "伸ばす方向", "X"), ("offset", "伸ばす断面の位置 [mm]", 0.0),
                                 ("length", "伸ばす長さ [mm]", 10.0)]),
    "fillet": ("丸み付け", "丸", [("r", "半径 [mm]", 2.0), ("edges", "対象の辺", "すべての辺"),
                               ("x0", "範囲 X 最小 [mm]", -1e6), ("x1", "範囲 X 最大 [mm]", 1e6),
                               ("y0", "範囲 Y 最小 [mm]", -1e6), ("y1", "範囲 Y 最大 [mm]", 1e6),
                               ("z0", "範囲 Z 最小 [mm]", -1e6), ("z1", "範囲 Z 最大 [mm]", 1e6)]),
    "chamfer": ("面取り", "角", [("d", "距離 [mm]", 2.0), ("edges", "対象の辺", "すべての辺"),
                              ("x0", "範囲 X 最小 [mm]", -1e6), ("x1", "範囲 X 最大 [mm]", 1e6),
                              ("y0", "範囲 Y 最小 [mm]", -1e6), ("y1", "範囲 Y 最大 [mm]", 1e6),
                              ("z0", "範囲 Z 最小 [mm]", -1e6), ("z1", "範囲 Z 最大 [mm]", 1e6)]),
    "import": ("インポート", "読", [("path", "ファイル", ""),
                                 ("dx", "移動 X [mm]", 0.0), ("dy", "移動 Y [mm]", 0.0), ("dz", "移動 Z [mm]", 0.0)]),
}
EDIT_FEATURES = ("move", "mirror", "split", "stretch", "fillet", "chamfer")
FEATURE_CHOICES = {"copy": ["移動", "コピー"],
                   "keep": ["両方（2つのボディに分割）", "＋側を残す", "−側を残す"],
                   "edges": ["すべての辺", "X 方向の辺", "Y 方向の辺", "Z 方向の辺"]}
OPS = {"new": "新規ボディ", "add": "結合", "cut": "除去（切り取り）", "intersect": "交差"}


def new_feature(ftype, op="new", **over):
    f = {"type": ftype, "op": "edit" if ftype in EDIT_FEATURES else op}
    for k, _, d in FEATURES[ftype][2]:
        f[k] = d
    f.update(over)
    return f


SHELL_FEATURES = ("plate", "sheet", "cyl_shell", "disk")
PLANE_MODES = {"plane_stress": "平面応力", "plane_strain": "平面ひずみ", "axisymmetric": "軸対称"}


def analysis_2d_mode(spec) -> str | None:
    """2D 解析のモード（Ansys のジオメトリ「2D の挙動」）。3D なら None"""
    m = spec.get("analysis_2d")
    return m if m in PLANE_MODES else None


def is_shell_model(spec) -> bool:
    feats = [f for f in spec.get("features", []) if not f.get("suppressed")
             and f.get("type") != "bool" and f.get("type") not in EDIT_FEATURES]
    return spec.get("kind") == "model" and bool(feats) and all(f["type"] in SHELL_FEATURES for f in feats)


def is_mixed_model(spec) -> bool:
    """ソリッドのフィーチャーとシェル（サーフェス）のフィーチャーが両方あるモデルか"""
    if spec.get("kind") != "model" or analysis_2d_mode(spec):
        return False
    feats = [f for f in spec.get("features", []) if not f.get("suppressed")
             and f.get("type") != "bool" and f.get("type") not in EDIT_FEATURES]
    sh = [f["type"] in SHELL_FEATURES for f in feats]
    return any(sh) and not all(sh)


def split_mixed(spec):
    """ソリッド＋シェルのモデルを、ソリッドだけ・シェルだけの 2 つの形状指定に分ける。
    シェルのフィーチャーは、ソリッドの作業（作成・ブール演算・編集）が終わったあとに並べる決まり。
    戻り値: (ソリッドの spec, シェルの spec, シェルのボディ番号の始まり)"""
    feats = list(spec.get("features", []))
    first = next(i for i, f in enumerate(feats) if not f.get("suppressed") and f.get("type") in SHELL_FEATURES)
    tail = [f for f in feats[first:] if not f.get("suppressed")]
    bad = [f for f in tail if f.get("type") not in SHELL_FEATURES or f.get("op", "new") != "new"]
    if bad:
        raise GeometryError("ソリッドとシェルを混ぜるときは、シェル（平板・多角形の面・円筒シェル・円板）を"
                            "ソリッドの作業がすべて終わったあとに「新規ボディ」で追加してください"
                            "（シェルのあとにソリッドのフィーチャー・ブール演算・編集は置けません）")
    keys = feature_body_keys(feats[:first], all_keys=True)
    offset = 1 + max([k for lst in keys.values() for k in lst], default=-1)
    return dict(spec, features=feats[:first]), dict(spec, features=tail), offset


def _plane_map(plane, off):
    return {"XY": lambda u, v: (u, v, off), "YZ": lambda u, v: (off, u, v),
            "ZX": lambda u, v: (v, off, u)}[plane]


def _make_surface(gmsh, f):
    occ = gmsh.model.occ
    t = f["type"]
    g = lambda k: float(f.get(k, 0.0))  # noqa: E731
    if g("t") <= 0 and f.get("op", "new") == "new":
        raise GeometryError("シェルの厚さは正の値にしてください")
    if t == "disk":
        plane = f.get("plane", "XY")
        to3 = _plane_map(plane, g("offset"))
        normal = {"XY": (0, 0, 1), "YZ": (1, 0, 0), "ZX": (0, 1, 0)}[plane]
        if g("r") <= 0 or g("ri") >= g("r") or g("ri") < 0:
            raise GeometryError("円板の半径は 0 ≤ 内半径 < 外半径 にしてください")
        cx = to3(g("u0"), g("v0"))
        outer = occ.addCurveLoop([occ.addCircle(*cx, g("r"), zAxis=normal)])
        loops = [outer]
        if g("ri") > 0:
            loops.append(occ.addCurveLoop([occ.addCircle(*cx, g("ri"), zAxis=normal)]))
        return [(2, occ.addPlaneSurface(loops))]
    if t == "plate":
        to3 = _plane_map(f.get("plane", "XY"), g("offset"))
        u0, v0, du, dv = g("u0"), g("v0"), g("du"), g("dv")
        pts = [(u0, v0), (u0 + du, v0), (u0 + du, v0 + dv), (u0, v0 + dv)]
    elif t == "sheet":
        to3 = _plane_map(f.get("plane", "XY"), g("offset"))
        pts = parse_points(f.get("points", ""))
    elif t == "cyl_shell":
        c = occ.addCircle(g("x"), g("y"), g("z"), g("r"), zAxis=_axis_vec(f.get("axis", "Z"), 1.0))
        out = occ.extrude([(1, c)], *_axis_vec(f.get("axis", "Z"), g("h")))
        return [e for e in out if e[0] == 2]
    else:
        raise GeometryError(f"未知のサーフェス: {t}")
    ptags = [occ.addPoint(*to3(u, v)) for u, v in pts]
    lines = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
    return [(2, occ.addPlaneSurface([occ.addCurveLoop(lines)]))]


def _build_shell_features(gmsh, features):
    """サーフェス（シェル・2D）のボディを作る。ソリッドと同じく op（new/add/cut）と
    ブール演算（結合・除去）が使える。接する辺・交わる線は節点を共有（フラグメント）"""
    occ = gmsh.model.occ
    bodies: dict[int, list] = {}
    thick = {}
    nkey, last = 0, None
    for i, f in enumerate(features):
        if f.get("suppressed"):
            continue
        t, op = f["type"], f.get("op", "new")
        if t == "bool":
            tgt = int(f.get("target", -1))
            tools = [int(k) for k in f.get("tools", []) if int(k) in bodies and int(k) != tgt]
            if tgt not in bodies or not tools:
                continue
            tt = [dt for k in tools for dt in bodies[k]]
            bodies[tgt] = (occ.fuse if op == "union" else occ.cut)(bodies[tgt], tt)[0]
            for k in tools:
                bodies.pop(k, None)
            continue
        if t in ("move", "mirror"):
            tg, k0 = _edit_targets(f, bodies), nkey
            nkey = _edit_bodies(gmsh, f, bodies, nkey)
            for j in range(nkey - k0):
                thick[k0 + j] = thick.get(tg[j], 1.0)
            continue
        if t == "split":                      # 切断: 面を平面で 2 つに分ける（板厚は元のボディと同じ）
            made = {}
            nkey = _edit_bodies(gmsh, f, bodies, nkey, dim=2, made=made)
            for kn, ko in made.items():
                thick[kn] = thick.get(ko, 1.0)
            for k in [k for k, v in bodies.items() if not v]:      # 片側だけ残して何も残らなかったボディ
                bodies.pop(k)
            continue
        if t in EDIT_FEATURES:
            raise GeometryError(f"サーフェス（シェル・2D）では「{FEATURES[t][0]}」にまだ対応していません"
                                "（移動・回転・ミラー・切断は可）")
        if t not in SHELL_FEATURES:
            raise GeometryError("ソリッドとシェル（サーフェス）を同じモデルで混ぜることには、まだ対応していません")
        tool = _make_surface(gmsh, f)
        if op == "new" or not bodies:
            bodies[nkey] = tool
            thick[nkey] = float(f.get("t", 1.0))
            last = nkey
            nkey += 1
        elif op == "add":
            tgt = int(f.get("target", last if last is not None else -1))
            tgt = tgt if tgt in bodies else max(bodies)
            bodies[tgt] = occ.fuse(bodies[tgt], tool)[0]
        elif op == "cut":
            for k in [int(k) for k in (f.get("targets") or list(bodies)) if int(k) in bodies]:
                bodies[k] = occ.cut(bodies[k], tool, removeTool=False)[0]
            occ.remove(tool, recursive=True)
    bodies = {k: [dt for dt in v if dt[0] == 2] for k, v in bodies.items()}
    bodies = {k: v for k, v in bodies.items() if v}
    if not bodies:
        raise GeometryError("サーフェスがありません")
    keys = sorted(bodies)
    objs = [(dt, k) for k in keys for dt in bodies[k]]
    groups = {}
    if len(objs) > 1:
        _, omap = occ.fragment([objs[0][0]], [o[0] for o in objs[1:]])
        for idx, children in enumerate(omap):
            for dt in children:
                if dt[0] == 2 and not any(dt[1] in g for g in groups.values()):
                    groups.setdefault(objs[idx][1], set()).add(dt[1])
    else:
        groups[objs[0][1]] = {objs[0][0][1]}
    occ.synchronize()
    for k, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, sorted(tags), tag=k + 1)
    return thick


def _extract_shell_mesh(gmsh, thick):
    from .shell import ShellMesh
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    tag2idx = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    tag2idx[tags.astype(np.int64)] = np.arange(len(tags))
    surfs = sorted(gmsh.model.getEntities(2), key=lambda e: e[1])
    tris, sid, body, th = [], [], [], []
    for i, (_, st) in enumerate(surfs):
        phys = gmsh.model.getPhysicalGroupsForEntity(2, st)
        b = int(phys[0]) - 1 if len(phys) else 0
        ets, _, ens = gmsh.model.mesh.getElements(2, st)
        for t, en in zip(ets, ens):
            if int(t) != 2:
                continue
            tri = tag2idx[en.astype(np.int64)].reshape(-1, 3)
            tris.append(tri)
            sid.append(np.full(len(tri), i))
            body.append(np.full(len(tri), b))
            th.append(np.full(len(tri), thick.get(b, 1.0)))
    if not tris:
        raise GeometryError("面を三角形でメッシュできませんでした")
    tris = np.vstack(tris)
    edges = []
    for _, ct in sorted(gmsh.model.getEntities(1), key=lambda e: e[1]):
        ets, _, ens = gmsh.model.mesh.getElements(1, ct)
        seg = [tag2idx[en.astype(np.int64)].reshape(-1, 2) for t, en in zip(ets, ens) if int(t) == 1]
        if seg:
            edges.append(np.vstack(seg))
    used = np.unique(np.concatenate([tris.ravel()] + [e.ravel() for e in edges]))
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return ShellMesh(coords[used], remap[tris], np.concatenate(th), np.concatenate(sid),
                     [remap[e] for e in edges], elem_body=np.concatenate(body), n_surf=len(surfs))


def _extract_2d_mesh(gmsh, thick, mode):
    """XY 平面の 6 節点三角形メッシュ → Mesh2D（辺は 3 節点の線要素）"""
    from .plane import Mesh2D
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    if np.abs(coords[:, 2]).max() > 1e-6 * max(1.0, np.abs(coords).max()):
        raise GeometryError("2D 解析では、面を XY 平面（Z = 0）に作成してください")
    tag2idx = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    tag2idx[tags.astype(np.int64)] = np.arange(len(tags))
    surfs = sorted(gmsh.model.getEntities(2), key=lambda e: e[1])
    tris, sid, body, th = [], [], [], []
    for i, (_, st) in enumerate(surfs):
        phys = gmsh.model.getPhysicalGroupsForEntity(2, st)
        b = int(phys[0]) - 1 if len(phys) else 0
        ets, _, ens = gmsh.model.mesh.getElements(2, st)
        for t, en in zip(ets, ens):
            if int(t) != 9:          # 9 = 6 節点三角形
                continue
            tri = tag2idx[en.astype(np.int64)].reshape(-1, 6)
            tris.append(tri)
            sid.append(np.full(len(tri), i))
            body.append(np.full(len(tri), b))
            th.append(np.full(len(tri), thick.get(b, 1.0) if mode == "plane_stress" else 1.0))
    if not tris:
        raise GeometryError("面を三角形でメッシュできませんでした")
    tris = np.vstack(tris)
    p = coords[tris[:, :3], :2]
    neg = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
           - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) < 0
    tris[neg] = tris[neg][:, [0, 2, 1, 5, 4, 3]]          # 反時計回りにそろえる
    edges = []
    for _, ct in sorted(gmsh.model.getEntities(1), key=lambda e: e[1]):
        ets, _, ens = gmsh.model.mesh.getElements(1, ct)
        seg = [tag2idx[en.astype(np.int64)].reshape(-1, 3) for t, en in zip(ets, ens) if int(t) == 8]
        if seg:
            edges.append(np.vstack(seg))
    used = np.unique(np.concatenate([tris.ravel()] + [e.ravel() for e in edges]))
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh2D(coords[used], remap[tris], np.concatenate(sid), [remap[e] for e in edges],
                  np.concatenate(th), elem_body=np.concatenate(body), n_surf=len(surfs), mode=mode)


def _axis_vec(axis, h):
    return {"X": (h, 0, 0), "Y": (0, h, 0), "Z": (0, 0, h)}[str(axis).upper()]


def parse_points(text):
    pts = []
    for chunk in str(text).replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a = [float(v) for v in chunk.replace(" ", ",").split(",") if v.strip()]
        if len(a) != 2:
            raise GeometryError(f"頂点の書式が正しくありません: 「{chunk}」（例: 0,0; 50,0; 50,20）")
        pts.append(a)
    if len(pts) < 3:
        raise GeometryError("押し出しには 3 点以上の頂点が必要です")
    return pts


def _make_feature(gmsh, f):
    occ = gmsh.model.occ
    t = f["type"]
    if t in SHELL_FEATURES:
        raise GeometryError("ソリッドとシェル（サーフェス）を同じモデルで混ぜることには、まだ対応していません")
    g = lambda k: float(f.get(k, 0.0))  # noqa: E731
    if t == "box":
        if min(g("dx"), g("dy"), g("dz")) <= 0:
            raise GeometryError("直方体の寸法は正の値にしてください")
        return [(3, occ.addBox(g("x"), g("y"), g("z"), g("dx"), g("dy"), g("dz")))]
    if t == "cylinder":
        return [(3, occ.addCylinder(g("x"), g("y"), g("z"), *_axis_vec(f.get("axis", "Z"), g("h")), g("r")))]
    if t == "pipe":
        if g("ri") >= g("ro"):
            raise GeometryError("パイプの内半径は外半径より小さくしてください")
        a = occ.addCylinder(g("x"), g("y"), g("z"), *_axis_vec(f.get("axis", "Z"), g("h")), g("ro"))
        if g("ri") <= 0:
            return [(3, a)]
        b = occ.addCylinder(g("x"), g("y"), g("z"), *_axis_vec(f.get("axis", "Z"), g("h")), g("ri"))
        out, _ = occ.cut([(3, a)], [(3, b)])
        return out
    if t == "cone":
        return [(3, occ.addCone(g("x"), g("y"), g("z"), *_axis_vec(f.get("axis", "Z"), g("h")), g("r1"), g("r2")))]
    if t == "sphere":
        return [(3, occ.addSphere(g("x"), g("y"), g("z"), g("r")))]
    if t == "extrude":
        pts = parse_points(f.get("points", ""))
        off, h, pl = g("offset"), g("h"), f.get("plane", "XY")
        to3 = {"XY": lambda u, v: (u, v, off), "YZ": lambda u, v: (off, u, v),
               "ZX": lambda u, v: (v, off, u)}[pl]
        vec = {"XY": (0, 0, h), "YZ": (h, 0, 0), "ZX": (0, h, 0)}[pl]
        ptags = [occ.addPoint(*to3(u, v)) for u, v in pts]
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        ext = occ.extrude([(2, surf)], *vec)
        return [e for e in ext if e[0] == 3]
    if t == "import":
        path = f.get("path", "")
        ext = os.path.splitext(path)[1].lower()
        if ext in PARASOLID_EXT:
            raise GeometryError(PARASOLID_MESSAGE)
        if ext not in CAD_EXT:
            raise GeometryError("モデリングに取り込めるのは STEP / IGES / BREP です（STL は「ジオメトリ → インポート」で）")
        if not os.path.exists(path):
            raise GeometryError(f"ファイルが見つかりません: {path}")
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        out = [e for e in occ.importShapes(path) if e[0] == 3]
        if g("dx") or g("dy") or g("dz"):
            occ.translate(out, g("dx"), g("dy"), g("dz"))
        return out
    raise GeometryError(f"未知のフィーチャー: {t}")


def _build_features(gmsh, features):
    """フィーチャー履歴からボディを作る（Femtet と同じく、作成した形状はそれぞれ別のボディ）。

    op:
      new        新しいボディを作る（既定）
      add        対象ボディ target に結合（旧形式。target 省略時は直前のボディ）
      cut        対象ボディ targets から切り取る（省略時は全ボディ）
      intersect  全ボディとの共通部分を残す
    type="bool": ボディ同士のブール演算 {"op": "union"|"cut", "target": k, "tools": [k, ...]}

    重なったボディは最後に分割（fragment）し、重なり部分はボディ番号の小さい方に属させます
    （Femtet の自動干渉除去と同じルール）。接している面は節点を共有するのでつながって計算されます。
    """
    occ = gmsh.model.occ
    bodies: dict[int, list] = {}
    nkey = 0
    last = None
    for i, f in enumerate(features):
        if f.get("suppressed"):
            continue
        t = f["type"]
        op = f.get("op", "new")
        label = f"フィーチャー {i + 1}（{FEATURES.get(t, ('ブール演算',))[0]}）"
        try:
            if t == "bool":
                tgt = int(f.get("target", -1))
                tools = [int(k) for k in f.get("tools", []) if int(k) in bodies and int(k) != tgt]
                if tgt not in bodies or not tools:
                    continue
                tool_tags = [dt for k in tools for dt in bodies[k]]
                if op == "union":
                    bodies[tgt] = occ.fuse(bodies[tgt], tool_tags)[0]
                else:
                    bodies[tgt] = occ.cut(bodies[tgt], tool_tags)[0]
                for k in tools:
                    bodies.pop(k, None)
                continue
            if t in EDIT_FEATURES:
                nkey = _edit_bodies(gmsh, f, bodies, nkey)
                continue
            tool = _make_feature(gmsh, f)
            if not tool:
                continue
            if op == "new" or (op == "add" and not bodies):
                bodies[nkey] = tool
                last = nkey
                nkey += 1
            elif op == "add":
                tgt = int(f.get("target", last if last is not None else -1))
                tgt = tgt if tgt in bodies else max(bodies)
                bodies[tgt] = occ.fuse(bodies[tgt], tool)[0]
            elif op == "cut":
                targets = [int(k) for k in (f.get("targets") or list(bodies)) if int(k) in bodies]
                for k in targets:
                    bodies[k] = occ.cut(bodies[k], tool, removeTool=False)[0]
                occ.remove(tool, recursive=True)
            elif op == "intersect":
                for k in list(bodies):
                    cp = occ.copy(tool)
                    bodies[k] = occ.intersect(bodies[k], cp)[0]
                occ.remove(tool, recursive=True)
        except GeometryError:
            raise
        except Exception as e:
            what = "操作" if t in EDIT_FEATURES else OPS.get(op, op)
            raise GeometryError(f"{label}の{what}に失敗しました: {e}")
    bodies = {k: [dt for dt in v if dt[0] == 3] for k, v in bodies.items()}
    bodies = {k: v for k, v in bodies.items() if v}
    if not bodies:
        occ.synchronize()
        raise GeometryError("ソリッドがありません。形状を作成してください。")
    objs, owner = [], []
    for k in sorted(bodies):
        for dt in bodies[k]:
            objs.append(dt)
            owner.append(k)
    groups: dict[int, list] = {}
    if len(objs) > 1:
        _, omap = occ.fragment(objs[:1], objs[1:])
        assign = {}
        for idx, children in enumerate(omap):
            for dt in children:
                if dt[0] == 3 and (dt not in assign or owner[idx] < assign[dt]):
                    assign[dt] = owner[idx]
        for dt, k in assign.items():
            groups.setdefault(k, []).append(dt[1])
    else:
        groups[owner[0]] = [objs[0][1]]
    occ.synchronize()
    for k, tags in groups.items():
        gmsh.model.addPhysicalGroup(3, sorted(set(tags)), tag=k + 1)


def _edit_targets(f, bodies):
    return [int(k) for k in (f.get("targets") or sorted(bodies)) if int(k) in bodies]


def _creates_bodies(f) -> bool:
    """編集フィーチャーのうち、新しいボディを作るもの（コピー・2つに切断）"""
    t = f.get("type")
    if t in ("move", "mirror"):
        return f.get("copy", "移動" if t == "move" else "コピー") == "コピー"
    if t == "split":
        return str(f.get("keep", "両方")).startswith("両方")
    return False


def _halfspace(occ, bodies_tags, axis, c, positive=True):
    """座標 axis ≥ c（positive）の半空間を表す大きな箱"""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for dt in bodies_tags:
        b = occ.getBoundingBox(*dt)
        lo, hi = np.minimum(lo, b[:3]), np.maximum(hi, b[3:])
    L = float(np.max(hi - lo)) + 1.0
    lo, hi = lo - L, hi + L
    ax = "XYZ".index(str(axis).upper())
    if positive:
        lo[ax] = c
    else:
        hi[ax] = c
    if hi[ax] <= lo[ax]:
        return None
    return [(3, occ.addBox(*lo, *(hi - lo)))]


def _edge_filter(gmsh, f, vol_tags):
    """丸み付け・面取りの対象の辺（方向と範囲で絞り込み）"""
    gmsh.model.occ.synchronize()
    surfs = gmsh.model.getBoundary(vol_tags, combined=False, oriented=False)
    curves = sorted({abs(c[1]) for c in gmsh.model.getBoundary(surfs, combined=False, oriented=False) if c[0] == 1})
    kind = str(f.get("edges", "すべての辺"))
    box = [float(f.get(k, d)) for k, d in (("x0", -1e6), ("y0", -1e6), ("z0", -1e6),
                                           ("x1", 1e6), ("y1", 1e6), ("z1", 1e6))]
    out = []
    for c in curves:
        b = np.array(gmsh.model.occ.getBoundingBox(1, c))
        if np.any(b[:3] < np.array(box[:3]) - 1e-9) or np.any(b[3:] > np.array(box[3:]) + 1e-9):
            continue
        ext = b[3:] - b[:3]
        if kind != "すべての辺":
            ax = "XYZ".index(kind[0])
            if ext[ax] <= 1e-9 or np.delete(ext, ax).max() > 1e-6 * max(1.0, ext[ax]):
                continue
        out.append(c)
    return out


def _edit_bodies(gmsh, f, bodies, nkey, dim=3, made=None):
    """ボディの編集フィーチャーを実行。新しく作ったボディの分だけ nkey を進めて返す。
    dim: ボディの次元（ソリッド 3、サーフェス 2）。 made: 切断で新しくできたボディ {新しいキー: 元のキー} を入れる辞書"""
    occ = gmsh.model.occ
    t = f["type"]
    g = lambda k, d=0.0: float(f.get(k, d))  # noqa: E731
    targets = _edit_targets(f, bodies)
    if not targets:
        raise GeometryError("対象のボディがありません")
    axis = str(f.get("axis", "X")).upper()
    av = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]
    if t in ("move", "mirror"):
        copy = _creates_bodies(f)
        for k in targets:
            tags = occ.copy(bodies[k]) if copy else bodies[k]
            if t == "move":
                if g("angle"):
                    occ.rotate(tags, g("x"), g("y"), g("z"), *av, math.radians(g("angle")))
                if g("dx") or g("dy") or g("dz"):
                    occ.translate(tags, g("dx"), g("dy"), g("dz"))
            else:
                occ.mirror(tags, *av, -g("offset"))
            if copy:
                bodies[nkey] = tags
                nkey += 1
        return nkey
    if t == "split":
        keep = str(f.get("keep", "両方"))
        for k in targets:
            plus_box = _halfspace(occ, bodies[k], axis, g("offset"), True)
            if plus_box is None:
                continue
            plus = occ.intersect(occ.copy(bodies[k]), plus_box)[0]
            minus_box = _halfspace(occ, bodies[k], axis, g("offset"), True)
            minus = occ.cut(bodies[k], minus_box)[0]
            plus = [dt for dt in plus if dt[0] == dim]
            minus = [dt for dt in minus if dt[0] == dim]
            if keep.startswith("＋"):
                occ.remove(minus, recursive=True)
                bodies[k] = plus
            elif keep.startswith("−"):
                occ.remove(plus, recursive=True)
                bodies[k] = minus
            elif not plus or not minus:          # 切断面がボディを通っていない: そのまま
                bodies[k] = minus or plus
            else:
                bodies[k] = minus
                bodies[nkey] = plus
                if made is not None:
                    made[nkey] = k
                nkey += 1
        return nkey
    if t == "stretch":
        L = g("length")
        if L <= 0:
            raise GeometryError("伸ばす長さは正の値にしてください")
        c = g("offset")
        d = [L * v for v in av]
        for k in targets:
            box = _halfspace(occ, bodies[k], axis, c, True)
            if box is None:
                continue
            # 断面（ボディと平面の共通部分）を押し出して間を埋める
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for dt in bodies[k]:
                b = occ.getBoundingBox(*dt)
                lo, hi = np.minimum(lo, b[:3]), np.maximum(hi, b[3:])
            ax = "XYZ".index(axis)
            o = lo - 1.0
            o[ax] = c
            size = hi - lo + 2.0
            uvec = [0, 0, 0]
            vvec = [0, 0, 0]
            others = [i for i in range(3) if i != ax]
            uvec[others[0]] = size[others[0]]
            vvec[others[1]] = size[others[1]]
            p = [occ.addPoint(*o), occ.addPoint(*(o + uvec)), occ.addPoint(*(o + np.add(uvec, vvec))),
                 occ.addPoint(*(o + vvec))]
            ln = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
            plane = [(2, occ.addPlaneSurface([occ.addCurveLoop(ln)]))]
            sec = [dt for dt in occ.intersect(occ.copy(bodies[k]), plane)[0] if dt[0] == 2]
            if not sec:
                occ.remove(box, recursive=True)
                raise GeometryError(f"伸ばす断面（{axis} = {c:g}）がボディ B{k} を横切っていません")
            mid = [dt for dt in occ.extrude(sec, *d) if dt[0] == 3]
            plus = [dt for dt in occ.intersect(occ.copy(bodies[k]), box)[0] if dt[0] == 3]
            box2 = _halfspace(occ, bodies[k], axis, c, True)
            minus = [dt for dt in occ.cut(bodies[k], box2)[0] if dt[0] == 3]
            occ.translate(plus, *d)
            parts = minus + mid + plus
            bodies[k] = occ.fuse(parts[:1], parts[1:])[0] if len(parts) > 1 else parts
        return nkey
    if t in ("fillet", "chamfer"):
        size = g("r", 2.0) if t == "fillet" else g("d", 2.0)
        if size <= 0:
            raise GeometryError("半径・距離は正の値にしてください")
        for k in targets:
            vols = [dt[1] for dt in bodies[k] if dt[0] == 3]
            curves = _edge_filter(gmsh, f, [(3, v) for v in vols])
            if not curves:
                raise GeometryError(f"ボディ B{k} に条件に合う辺がありません")
            if t == "fillet":
                out = occ.fillet(vols, curves, [size], removeVolume=True)
            else:
                surfs = []
                for cv in curves:
                    up, _ = gmsh.model.getAdjacencies(1, cv)
                    surfs.append(int(up[0]))
                out = occ.chamfer(vols, curves, surfs, [size], removeVolume=True)
            out = [dt for dt in out if dt[0] == 3]
            if not out:
                raise GeometryError(("丸み付け" if t == "fillet" else "面取り")
                                    + "に失敗しました（半径・距離が大きすぎないか確認してください）")
            bodies[k] = out
        return nkey
    raise GeometryError(f"未知の編集: {t}")


def feature_body_keys(features, all_keys=False) -> dict:
    """フィーチャー番号 → それが作ったボディ番号（_build_features と同じ規則）。
    all_keys=True なら 番号 → 作ったボディ番号のリスト（コピー・切断は複数）"""
    out, nkey, live = {}, 0, set()
    created = {}
    for i, f in enumerate(features):
        if f.get("suppressed"):
            continue
        t = f.get("type")
        if t == "bool":
            tgt = int(f.get("target", -1))
            tools = [int(k) for k in f.get("tools", []) if int(k) in live and int(k) != tgt]
            if tgt in live and tools:
                live -= set(tools)
            continue
        if t in EDIT_FEATURES:
            if _creates_bodies(f):
                n = len([k for k in (f.get("targets") or sorted(live)) if int(k) in live])
                if n:
                    out[i] = nkey
                    created[i] = list(range(nkey, nkey + n))
                for _ in range(n):
                    live.add(nkey)
                    nkey += 1
            continue
        op = f.get("op", "new")
        if op == "new" or (op == "add" and not live):
            out[i] = nkey
            created[i] = [nkey]
            live.add(nkey)
            nkey += 1
    return created if all_keys else out


def primitive_to_features(spec):
    """「簡単形状」をフィーチャーに変換（モデリングで編集できるようにする）"""
    k = spec.get("kind")
    g = lambda key: float(spec.get(key, dict((a, d) for a, _, d in PRIMITIVES[k][1])[key]))  # noqa: E731
    if k == "box":
        return [new_feature("box", dx=g("lx"), dy=g("ly"), dz=g("lz"))]
    if k == "cylinder":
        return [new_feature("cylinder", r=g("r"), h=g("h"))]
    if k == "pipe":
        return [new_feature("pipe", ro=g("ro"), ri=g("ri"), h=g("h"))]
    if k == "sphere":
        return [new_feature("sphere", r=g("r"))]
    if k == "plate_hole":
        lx, ly, t, d = g("lx"), g("ly"), g("t"), g("d")
        return [new_feature("box", dx=lx, dy=ly, dz=t),
                new_feature("cylinder", "cut", x=lx / 2, y=ly / 2, z=-1.0, r=d / 2, h=t + 2)]
    if k == "l_bracket":
        a, b, w, t, d = g("a"), g("b"), g("w"), g("t"), g("d")
        out = [new_feature("box", dx=a, dy=w, dz=t), new_feature("box", dx=t, dy=w, dz=b)]
        if d > 0:
            out.append(new_feature("cylinder", "cut", x=t + (a - t) * 0.6, y=w / 2, z=-1.0, r=d / 2, h=t + 2))
        return out
    if k == "file" and os.path.splitext(spec.get("path", ""))[1].lower() in CAD_EXT:
        return [new_feature("import", path=spec["path"])]
    return None


CAD_EXT = (".step", ".stp", ".igs", ".iges", ".brep", ".brp")
MESH_EXT = (".stl",)
PARASOLID_EXT = (".x_t", ".x_b", ".xmt_txt", ".xmt_bin")

PARASOLID_MESSAGE = (
    "Parasolid 形式（.x_t / .x_b）は Siemens 社の独自形式で、無償で読み込めるライブラリが"
    "存在しないため直接は開けません。\n\n"
    "お使いの CAD（SolidWorks、NX、Solid Edge、Fusion など）から\n"
    "  「名前を付けて保存」→ STEP (*.step / *.stp)\n"
    "で書き出して読み込んでください。STEP なら CAD の面がそのまま選択できます。"
)


class GeometryError(Exception):
    pass


def gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except Exception:
        return False


def _gmsh_init():
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    try:  # GUI の別スレッドから呼んでも落ちないように
        gmsh.initialize(interruptible=False)
    except TypeError:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("openmech")
    return gmsh


def _build_primitive(gmsh, kind: str, p: dict):
    occ = gmsh.model.occ
    g = lambda k: float(p.get(k, dict((a, d) for a, _, d in PRIMITIVES[kind][1])[k]))  # noqa: E731
    if kind == "box":
        occ.addBox(0, 0, 0, g("lx"), g("ly"), g("lz"))
    elif kind == "cylinder":
        occ.addCylinder(0, 0, 0, 0, 0, g("h"), g("r"))
    elif kind == "pipe":
        if g("ri") >= g("ro"):
            raise GeometryError("内半径は外半径より小さくしてください")
        a = occ.addCylinder(0, 0, 0, 0, 0, g("h"), g("ro"))
        b = occ.addCylinder(0, 0, 0, 0, 0, g("h"), g("ri"))
        occ.cut([(3, a)], [(3, b)])
    elif kind == "plate_hole":
        lx, ly, t, d = g("lx"), g("ly"), g("t"), g("d")
        if d >= ly:
            raise GeometryError("穴径は板の幅より小さくしてください")
        a = occ.addBox(0, 0, 0, lx, ly, t)
        b = occ.addCylinder(lx / 2, ly / 2, -1, 0, 0, t + 2, d / 2)
        occ.cut([(3, a)], [(3, b)])
    elif kind == "l_bracket":
        a, b, w, t, d = g("a"), g("b"), g("w"), g("t"), g("d")
        h = occ.addBox(0, 0, 0, a, w, t)
        v = occ.addBox(0, 0, 0, t, w, b)
        out, _ = occ.fuse([(3, h)], [(3, v)])
        if d > 0:
            hole = occ.addCylinder(t + (a - t) * 0.6, w / 2, -1, 0, 0, t + 2, d / 2)
            out, _ = occ.cut(out, [(3, hole)])
        # 外側に余分な面が残らないよう統合
        try:
            occ.removeAllDuplicates()
        except Exception:
            pass
    elif kind == "sphere":
        occ.addSphere(0, 0, 0, g("r"))
    else:
        raise GeometryError(f"未知の形状: {kind}")
    occ.synchronize()


def _import_cad(gmsh, path: str):
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    try:
        gmsh.model.occ.importShapes(path)
    except Exception as e:
        raise GeometryError(f"CAD ファイルを読み込めませんでした: {e}")
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if not vols:
        raise GeometryError("ソリッド（体積）が見つかりません。サーフェスのみのモデルは解析できません。")
    if len(vols) > 1:
        # 複数部品は結合して1つの連続体として扱う（接触は未対応）
        gmsh.model.occ.fragment(vols[:1], vols[1:])
        gmsh.model.occ.synchronize()


def _import_stl(gmsh, path: str, angle_deg: float = 40.0):
    gmsh.merge(path)
    gmsh.model.mesh.classifySurfaces(angle_deg * math.pi / 180.0, True, True, math.pi)
    gmsh.model.mesh.createGeometry()
    surfs = gmsh.model.getEntities(2)
    if not surfs:
        raise GeometryError("STL に面がありません")
    loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()


def _bbox_diag(gmsh) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    return float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))


def _reorder_tet10(nodes: np.ndarray, el: np.ndarray) -> np.ndarray:
    """中間節点を幾何学的に辺へ対応付け、VTK の順序に並べ替える（形式に依存しない）。"""
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    corners = nodes[el[:, :4]]
    mids = nodes[el[:, 4:]]                                   # (E,6,3)
    target = np.stack([0.5 * (corners[:, i] + corners[:, j]) for i, j in edges], axis=1)
    d = np.linalg.norm(target[:, :, None, :] - mids[:, None, :, :], axis=3)   # (E,6,6)
    pick = np.argmin(d, axis=2)
    return np.hstack([el[:, :4], np.take_along_axis(el[:, 4:], pick, axis=1)])


# gmsh の 20 節点六面体（中間節点 (0,1),(0,3),(0,4),(1,2),(1,5),(2,3),(2,6),(3,7),(4,5),(4,7),(5,6),(6,7)）→ VTK の順
GMSH_HEX20_TO_VTK = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15])


def _extract_mesh(gmsh, share: dict | None = None) -> Mesh:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    tag2idx = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    tag2idx[tags.astype(np.int64)] = np.arange(len(tags))

    blocks, bodies = [], []
    for _, vtag in gmsh.model.getEntities(3):
        phys = gmsh.model.getPhysicalGroupsForEntity(3, vtag)
        body = int(phys[0]) - 1 if len(phys) else 0
        etypes, _, enodes = gmsh.model.mesh.getElements(3, vtag)
        mixed = len({int(t) for t in etypes} & {4, 11, 6, 18, 7, 19}) > 0 and \
            len({int(t) for t in etypes} & {5, 17}) > 0
        for t, en in zip(etypes, enodes):
            t = int(t)
            npe = {4: 4, 11: 10, 5: 8, 17: 20, 6: 6, 18: 15, 7: 5, 19: 13}.get(t)
            if npe is None:
                continue
            b = tag2idx[en.astype(np.int64)].reshape(-1, npe)
            if npe == 20:
                b = b[:, GMSH_HEX20_TO_VTK]
            elif mixed or t in (6, 18, 7, 19):
                # 六面体優勢メッシュ: 三角柱・ピラミッド・四面体を、縮退した六面体として同じ配列に入れる
                from .elements import DEGEN_PYR5, DEGEN_PYR13, DEGEN_TET4, DEGEN_TET10, DEGEN_WEDGE6, DEGEN_WEDGE15
                b = b[:, {4: DEGEN_TET4, 11: DEGEN_TET10, 6: DEGEN_WEDGE6, 18: DEGEN_WEDGE15, 7: DEGEN_PYR5,
                          19: DEGEN_PYR13}[t]]
            blocks.append(b)
            bodies.append(np.full(len(b), body, dtype=np.int64))
    if not blocks:
        raise GeometryError("四面体メッシュを作成できませんでした。メッシュサイズを変えて試してください。")
    if len({b.shape[1] for b in blocks}) > 1:
        raise GeometryError("要素の形が混ざったメッシュには対応していません（四面体か六面体のどちらかにしてください）")
    elems = np.vstack(blocks)
    elem_body = np.concatenate(bodies)
    # 使われている節点だけ残す
    used = np.unique(elems)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes = coords[used]
    elems = remap[elems]
    if elems.shape[1] == 10:
        elems = _reorder_tet10(nodes, elems)
    mesh = Mesh(nodes, elems)
    mesh.elem_body = elem_body
    # ボディ同士の境界の節点を分ける（つながりは接触で与える）
    mesh = mesh.split_body_interfaces(share)

    # CAD のサーフェスを「面」として割り当てる
    surfs = sorted(gmsh.model.getEntities(2), key=lambda e: e[1])
    n_surf = len(surfs)
    key_of = {}
    for fid, (_, stag) in enumerate(surfs):
        ets, _, ens = gmsh.model.mesh.getElements(2, stag)
        for t, en in zip(ets, ens):
            npe, nc = {2: (3, 3), 9: (6, 3), 3: (4, 4), 16: (8, 4)}.get(int(t), (None, None))
            if npe is None:
                continue
            tri = remap[tag2idx[en.astype(np.int64)].reshape(-1, npe)[:, :nc]]
            for k in map(tuple, np.sort(tri, axis=1)):
                key_of[k] = fid
    if mesh.is_hex:
        # 六面体: 境界の四角形の 4 頂点で対応を取り、そこから作った三角形へ
        bq = mesh.boundary_quads_full
        corners = bq[:, ::2] if bq.shape[1] == 8 else bq
        qk = np.sort(mesh.node_origin[corners], axis=1)
        qf = np.array([key_of.get(tuple(k), -1) for k in qk])
        fids = qf[mesh.boundary_quad_of]
    else:
        orig = mesh.node_origin[mesh.boundary_tris]
        bt = np.sort(orig, axis=1)
        fids = np.array([key_of.get(tuple(k), -1) for k in bt])
    if (fids < 0).any():
        # 対応が取れない場合は角度で自動分割
        mesh.group_faces()
        return mesh
    _surface_face_ids(mesh, fids, n_surf)
    return mesh


def _surface_face_ids(mesh, fids, n_surf):
    """CAD サーフェスの番号（ソート順）→ 面番号。ボディ境界の両側を分け、詰めて番号を振る"""
    # ボディ境界のサーフェスは両側に現れる: 番号の小さいボディ側はそのまま、もう一方は n_surf + 番号
    tb = mesh.tri_body()
    first_body = {}
    for f, b in zip(fids.tolist(), tb.tolist()):
        first_body[f] = min(b, first_body.get(f, b))
    second = np.array([b != first_body[f] for f, b in zip(fids.tolist(), tb.tolist())])
    fids = np.where(second, fids + n_surf, fids)
    _, fids = np.unique(fids, return_inverse=True)
    mesh.set_face_ids(fids)


# ---------------------------------------------------------------------------- き裂（破壊力学）

def crack_frame(c: dict):
    """き裂の定義 {"front0": [x,y,z], "front1": [...], "direction": [進展方向], "normal": [き裂面の法線],
    "depth": 前縁から後ろの長さ, "to_surface": 後ろを表面まで延ばすか} → (p0, p1, t, e1, n, L)"""
    p0, p1 = np.asarray(c["front0"], float), np.asarray(c["front1"], float)
    t = p1 - p0
    L = float(np.linalg.norm(t))
    if L <= 0:
        raise GeometryError("き裂前縁の 2 点が同じです")
    t /= L
    n = np.asarray(c.get("normal", [0, 1, 0]), float)
    n = n - (n @ t) * t
    if np.linalg.norm(n) < 1e-9:
        raise GeometryError("き裂面の法線が前縁と平行です")
    n /= np.linalg.norm(n)
    e1 = np.cross(n, t)
    d = np.asarray(c.get("direction", e1), float)
    if d @ e1 < 0:
        e1 = -e1
    return p0, p1, t, e1, n, L


def _surface_reference(gmsh):
    """き裂を入れる前の境界の面（三角形の重心・法線・サーフェス番号）を、2D メッシュから記録する"""
    gmsh.model.mesh.generate(2)
    surfs = sorted(gmsh.model.getEntities(2), key=lambda e: e[1])
    tags, coords, _ = gmsh.model.mesh.getNodes()
    P = np.zeros((int(tags.max()) + 1, 3))
    P[tags.astype(np.int64)] = coords.reshape(-1, 3)
    cen, nrm, fid = [], [], []
    for k, (_, stag) in enumerate(surfs):
        ets, _, ens = gmsh.model.mesh.getElements(2, stag)
        for t_, en in zip(ets, ens):
            npe = {2: 3, 9: 6}.get(int(t_))
            if npe is None:
                continue
            tri = P[en.astype(np.int64).reshape(-1, npe)[:, :3]]
            cen.append(tri.mean(1))
            v = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300))
            fid.append(np.full(len(tri), k))
    gmsh.model.mesh.clear()
    return np.vstack(cen), np.vstack(nrm), np.concatenate(fid), len(surfs)


def _embed_cracks(gmsh, cracks, diag):
    """き裂面（長方形）を形状に埋め込む（OCC の fragment）。体積の外に出た部分は消す"""
    occ = gmsh.model.occ
    occ.synchronize()
    tools = []
    m = 0.05 * diag
    for c in cracks:
        if c.get("shape") == "ellipse":
            from .fracture import CrackGeom
            g = CrackGeom(c)
            ax1, ax2, r1, r2 = (g.uc, g.ua, g.cc, g.a) if g.cc >= g.a else (g.ua, -g.uc, g.a, g.cc)
            d = occ.addDisk(0, 0, 0, r1, r2)
            n3 = np.cross(ax1, ax2)
            Rm = np.column_stack([ax1, ax2, n3])               # 局所 (X, Y, Z) → 全体
            occ.affineTransform([(2, d)], [*Rm[0], g.center[0], *Rm[1], g.center[1], *Rm[2], g.center[2]])
            tools.append((2, d))
            continue
        p0, p1, t, e1, n, L = crack_frame(c)
        back = (float(c.get("depth", 0.0)) + (m if c.get("to_surface", True) else 0.0))
        if back <= 0:
            raise GeometryError("き裂の深さ（前縁から後ろの長さ）を入れてください")
        a0, a1 = p0 - m * t, p1 + m * t
        corners = [a0, a1, a1 - back * e1, a0 - back * e1]
        pts = [occ.addPoint(*q) for q in corners]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        tools.append((2, occ.addPlaneSurface([occ.addCurveLoop(lines)])))
    vols = gmsh.model.getEntities(3)
    occ.fragment(vols, tools)
    occ.synchronize()
    # 体積に付いていない面（外にはみ出した部分）を消す
    for dim, tag in gmsh.model.getEntities(2):
        up, _ = gmsh.model.getAdjacencies(2, tag)
        if len(up) == 0:
            occ.remove([(2, tag)], recursive=True)
    occ.synchronize()


def _apply_cracks(mesh, ref, cracks, tol):
    """面番号を元の形状のサーフェスに合わせ直し、き裂面の節点を複製する"""
    from scipy.spatial import cKDTree
    from .fracture import insert_crack
    cen, nrm, fid0, n_surf = ref
    bt = mesh.boundary_tris
    X = mesh.nodes
    c = X[bt].mean(1)
    v = np.cross(X[bt[:, 1]] - X[bt[:, 0]], X[bt[:, 2]] - X[bt[:, 0]])
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)
    tree = cKDTree(cen)
    k = min(8, len(cen))
    d, idx = tree.query(c, k=k)
    idx = np.asarray(idx).reshape(len(c), -1)
    dots = np.einsum("ekj,ej->ek", nrm[idx], v)
    best = np.argmax(dots > 0.5, axis=1)
    fids = fid0[idx[np.arange(len(c)), best]]
    _surface_face_ids(mesh, fids, n_surf)
    crack_faces = []
    for cr in cracks:
        if cr.get("shape") == "ellipse":
            from .fracture import CrackGeom
            g = CrackGeom(cr)
            mesh, fp = insert_crack(mesh, g.center, g.n, lambda P, g=g: g.inside(P, tol), tol=tol)
            crack_faces.append(fp)
            continue
        p0, p1, t, e1, n, L = crack_frame(cr)
        a = float(cr.get("depth", 0.0))
        to_s = cr.get("to_surface", True)

        def inside(P, p0=p0, t=t, e1=e1, L=L, a=a, to_s=to_s):
            r = P - p0
            x1, s = r @ e1, r @ t
            ok = (x1 < -tol) & (s > -tol) & (s < L + tol)
            return ok & (to_s | (x1 > -a - tol))
        mesh, fp = insert_crack(mesh, p0, n, inside, tol=tol)
        crack_faces.append(fp)
    mesh.crack_faces = crack_faces
    return mesh


MESH_DEFAULTS = {
    "resolution": 2,              # 解像度 0（粗い）〜 7（細かい）。要素サイズ未指定のとき使う
    "growth_rate": 1.5,           # 成長率（局所サイズから周りへ大きくなる割合）
    "capture_curvature": True,    # 曲率を捕捉
    "curvature_normal_angle": 30.0,   # 曲率法線角度 [度]（1要素が張る最大角度）
    "curvature_min_size": 0.0,    # 曲率の最小サイズ [mm]（0 = 自動）
    "smoothing": "medium",        # スムージング low / medium / high
    "optimize": True,             # 要素形状の最適化
    "auto_fix": True,             # 品質の悪い要素を自動修正
    "min_quality": 0.05,          # 自動修正の目標: 要素品質の下限（Ansys の要素品質と同じ定義）
    # 近接の捕捉（Ansys の Capture Proximity）: 薄肉・狭いすき間に指定の要素数が入るよう要素を小さくする
    "capture_proximity": False,
    "size_points": [],
    "cracks": [],                 # き裂（破壊力学）: geometry.crack_frame の辞書のリスト
    "element_shape": "tet",       # 要素の形: "tet"（四面体）/ "hex"（すべて六面体: 四面体を 4 つの六面体に分割）/ "hexdom"（六面体優勢）             # 局所の要素サイズ [[x, y, z, h], ...]（適応メッシュが設定）
    "proximity_cells": 3,         # すき間（板厚）を横切る要素数（Ansys の Num Cells Across Gap、既定 3）
    "proximity_min_size": 0.0,    # 近接の最小サイズ [mm]（0 = 曲率の最小サイズと同じ）
    "proximity_max_elements": 3e6,  # 近接の捕捉で見積もった要素数がこれを超えたら中止
}

CONTROL_TYPES = {"sizing": "サイズ指定", "refinement": "リファインメント", "sweep": "手法：スイープ",
                 "sphere": "影響球サイズ"}


def resolved_size(diag, size, ms):
    """全体の要素サイズ（Ansys の「要素サイズ」既定 → 解像度から決める）"""
    if size and size > 0:
        return float(size)
    r = int(ms.get("resolution", 2))
    return diag / (17.0 + 4.0 * r)


def _apply_local_sizes(gmsh, h, local_sizes, growth=1.5):
    """局所メッシュ制御（Ansys の サイズ指定 / 影響球 / リファインメント 相当）。

    local_sizes の各項目:
      {"type":"sizing", "scope":"faces", "faces":[面番号], "size":mm, "behavior":"soft"|"hard"}
      {"type":"sizing", "scope":"bodies", "bodies":[ボディ番号], "size":mm}
      {"type":"sphere", "center":[x,y,z], "radius":r, "size":mm}
      {"type":"refinement", "faces":[面番号], "level":1〜3}   周りの要素を 1/2^level に細分
    （type 省略 = 面サイズ指定）。指定部分から離れると成長率で全体サイズへ大きくなります。
    """
    surfs = [t for _, t in sorted(gmsh.model.getEntities(2), key=lambda e: e[1])]
    vols = {}
    for _, vtag in gmsh.model.getEntities(3):
        phys = gmsh.model.getPhysicalGroupsForEntity(3, vtag)
        vols.setdefault(int(phys[0]) - 1 if len(phys) else 0, []).append(vtag)
    g = max(float(growth), 1.01)
    fields, mins = [], []
    for ls in local_sizes:
        t = ls.get("type", "sizing")
        if ls.get("suppressed"):
            continue
        if t == "sphere":
            r, hs = float(ls.get("radius", 0)), float(ls.get("size", 0))
            c = ls.get("center", [0, 0, 0])
            if r <= 0 or hs <= 0:
                continue
            fb = gmsh.model.mesh.field.add("Ball")
            for k, v in zip(("XCenter", "YCenter", "ZCenter"), c):
                gmsh.model.mesh.field.setNumber(fb, k, float(v))
            gmsh.model.mesh.field.setNumber(fb, "Radius", r)
            gmsh.model.mesh.field.setNumber(fb, "Thickness", max(h - hs, 0.0) / (g - 1))
            gmsh.model.mesh.field.setNumber(fb, "VIn", hs)
            gmsh.model.mesh.field.setNumber(fb, "VOut", h)
            fields.append(fb)
            mins.append(hs)
            continue
        if t == "sizing" and ls.get("scope") == "bodies":
            hs = float(ls.get("size", 0))
            tags = sorted({v for b in ls.get("bodies", []) for v in vols.get(int(b), [])})
            if not tags or hs <= 0:
                continue
            fc = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumbers(fc, "VolumesList", tags)
            gmsh.model.mesh.field.setNumber(fc, "VIn", hs)
            gmsh.model.mesh.field.setNumber(fc, "VOut", 1e22)
            try:
                gmsh.model.mesh.field.setNumber(fc, "IncludeBoundary", 1)
            except Exception:
                pass
            fields.append(fc)
            mins.append(hs)
            continue
        # 面: サイズ指定・リファインメント（ボディ境界の裏側の面は表側と同じサーフェスなので省略）
        tags = sorted({surfs[f] for f in ls.get("faces", []) if 0 <= f < len(surfs)})
        if t == "refinement":
            hs = h / 2 ** max(1, min(3, int(ls.get("level", 1))))
        else:
            hs = float(ls.get("size", 0))
        if not tags or hs <= 0:
            continue
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", tags)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 40)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", hs)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
        hard = ls.get("behavior") == "hard"
        # ハード: 面上は指定サイズちょうど・すぐ全体サイズへ。ソフト: 成長率でなめらかに
        gmsh.model.mesh.field.setNumber(ft, "DistMin", hs * 0.5 if hard else 0.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", max(h - hs, 0.0) / (g - 1))
        fields.append(ft)
        mins.append(hs)
    if fields:
        fm = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fm)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    return mins


def _sweep_mesh(gmsh, ls, order, h, log):
    """スイープメッシュ（Ansys の 手法: スイープ）。

    断面（ソース面）の2次元メッシュを、掃引方向に Sweep Num Divs 層だけ押し出してプリズムにし、
    各プリズムを3つの四面体に分割する（隣り合う面で分割の向きがそろう規則で）。
    2次要素の中間節点は CAD の面に投影するので、円柱などの曲面も正しく表されます。
    """
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        raise GeometryError("スイープは、ボディが1つのモデルで使えます（複数ボディのスイープは今後対応予定）")
    vtag = vols[0][1]
    surfs = [t for _, t in sorted(gmsh.model.getEntities(2), key=lambda e: e[1])]
    # ソース面: 平らな面のうち、法線方向に同じ形の向かい合う面（ターゲット）を持つもの
    info = []
    for st in surfs:
        lo = gmsh.model.getBoundingBox(2, st)
        mass = gmsh.model.occ.getMass(2, st)
        typ = gmsh.model.getType(2, st)
        info.append((st, typ, mass, np.array(lo[:3]), np.array(lo[3:])))
    blo, bhi = np.array(gmsh.model.getBoundingBox(3, vtag)[:3]), np.array(gmsh.model.getBoundingBox(3, vtag)[3:])
    V = gmsh.model.occ.getMass(3, vtag)
    want = ls.get("axis", "auto")
    cands = []
    for ax in range(3):
        if want in "XYZ" and want != "auto" and "XYZ".index(want) != ax:
            continue
        L = bhi[ax] - blo[ax]
        lo_f = [x for x in info if x[1] == "Plane" and abs(x[3][ax] - blo[ax]) < 1e-6 * max(1, L) and abs(x[4][ax] - blo[ax]) < 1e-6 * max(1, L)]
        hi_f = [x for x in info if x[1] == "Plane" and abs(x[3][ax] - bhi[ax]) < 1e-6 * max(1, L) and abs(x[4][ax] - bhi[ax]) < 1e-6 * max(1, L)]
        if len(lo_f) == 1 and len(hi_f) == 1 and abs(lo_f[0][2] * L - V) < 1e-4 * V and abs(lo_f[0][2] - hi_f[0][2]) < 1e-6 * lo_f[0][2]:
            cands.append((L, ax, lo_f[0][0]))
    if not cands:
        raise GeometryError("このボディはスイープできません（座標軸方向に同じ断面が続く形状が必要です）")
    L, ax, src = min(cands) if want == "auto" else cands[0]    # 自動: 薄い方向（Ansys の Automatic Thin と同じ考え）
    n = int(ls.get("divs", 0)) or max(1, int(round(L / h)))
    hexa = ls.get("element") == "hex"
    log(f"スイープ: {'XYZ'[ax]} 方向に {n} 分割（長さ {L:.4g} mm）" + ("・六面体" if hexa else ""))
    if hexa:
        # ソース面を四角形だけで切る（Ansys の六面体スイープ: Blossom で全四角形化）
        gmsh.model.mesh.setRecombine(2, src)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
    else:
        gmsh.model.mesh.generate(2)
    ets, _, ens = gmsh.model.mesh.getElements(2, src)
    tri = quad = None
    for t, en in zip(ets, ens):
        if int(t) == 2:
            tri = en.astype(np.int64).reshape(-1, 3)
        elif int(t) == 3:
            quad = en.astype(np.int64).reshape(-1, 4)
    if hexa:
        if quad is None or tri is not None:
            raise GeometryError("ソース面を四角形だけで切れませんでした（六面体スイープ）。要素の形を三角柱にしてください")
        return _hex_sweep(gmsh, quad, ax, blo, L, n, order, h, info, surfs, vtag)
    if tri is None:
        raise GeometryError("ソース面を三角形でメッシュできませんでした")
    used = np.unique(tri)
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    cmap = dict(zip(ntag.astype(np.int64).tolist(), ncoord.reshape(-1, 3)))
    P = np.array([cmap[int(t)] for t in used])
    tri = np.searchsorted(used, tri)
    P[:, ax] = blo[ax]
    nb = len(P)
    layers = [P + np.eye(3)[ax] * (L * k / n) for k in range(n + 1)]
    nodes = np.vstack(layers)
    tets = []
    for k in range(n):
        o0, o1 = k * nb, (k + 1) * nb
        t = np.sort(tri, axis=1)                  # v0 < v1 < v2（分割の向きをそろえる規則）
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        tets += [np.column_stack([a + o0, b + o0, c + o0, c + o1]),
                 np.column_stack([a + o0, b + o0, b + o1, c + o1]),
                 np.column_stack([a + o0, a + o1, b + o1, c + o1])]
    el = np.vstack(tets)
    X = nodes[el]
    vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    flip = vol < 0
    el[flip] = el[flip][:, [0, 2, 1, 3]]
    m = Mesh(nodes, el)
    if order == 2:
        m = m.to_quadratic()
        # 曲面上の中間節点を CAD の面へ投影
        bt = m.boundary_tris_full
        mids = np.unique(bt[:, 3:])
        curved = [st for st, typ, *_ in info if typ != "Plane"]
        if curved and len(mids):
            Xm = m.nodes[mids]
            best = np.full(len(mids), np.inf)
            proj = Xm.copy()
            for st in curved:
                cp = np.array(gmsh.model.getClosestPoint(2, st, Xm.ravel().tolist())[0]).reshape(-1, 3)
                d = np.linalg.norm(cp - Xm, axis=1)
                better = d < best
                best[better], proj[better] = d[better], cp[better]
            ok = best < 0.05 * h
            m.nodes[mids[ok]] = proj[ok]
    phys = gmsh.model.getPhysicalGroupsForEntity(3, vtag)
    m.elem_body = np.full(m.n_elems, int(phys[0]) - 1 if len(phys) else 0, dtype=np.int64)
    m.node_origin = np.arange(m.n_nodes)
    # 面番号 = CAD のサーフェス番号（四面体メッシュと同じ番号になるので境界条件がそのまま使える）
    cen = m.nodes[m.boundary_tris].mean(axis=1)
    best = np.full(len(cen), np.inf)
    fid = np.zeros(len(cen), dtype=np.int64)
    for i, st in enumerate(surfs):
        cp = np.array(gmsh.model.getClosestPoint(2, st, cen.ravel().tolist())[0]).reshape(-1, 3)
        d = np.linalg.norm(cp - cen, axis=1)
        better = d < best
        best[better], fid[better] = d[better], i
    m.set_face_ids(fid)
    return m


def _hex_sweep(gmsh, quad, ax, blo, L, n, order, h, info, surfs, vtag):
    """四角形のソース面を押し出して六面体（Hex8 → Hex20）にする"""
    used = np.unique(quad)
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    cmap = dict(zip(ntag.astype(np.int64).tolist(), ncoord.reshape(-1, 3)))
    P = np.array([cmap[int(t)] for t in used])
    q = np.searchsorted(used, quad)
    P[:, ax] = blo[ax]
    nb = len(P)
    nodes = np.vstack([P + np.eye(3)[ax] * (L * k / n) for k in range(n + 1)])
    el = np.vstack([np.column_stack([q + k * nb, q + (k + 1) * nb]) for k in range(n)])
    m = Mesh(nodes, el)              # 向きは Mesh が直す（体積が負なら鏡映）
    if order == 2:
        m = m.to_quadratic()
        bq = m.boundary_quads_full
        mids = np.unique(bq[:, 1::2]) if len(bq) else np.zeros(0, int)
        curved = [st for st, typ, *_ in info if typ != "Plane"]
        if curved and len(mids):
            Xm = m.nodes[mids]
            best = np.full(len(mids), np.inf)
            proj = Xm.copy()
            for st in curved:
                cp = np.array(gmsh.model.getClosestPoint(2, st, Xm.ravel().tolist())[0]).reshape(-1, 3)
                d = np.linalg.norm(cp - Xm, axis=1)
                better = d < best
                best[better], proj[better] = d[better], cp[better]
            ok = best < 0.05 * h
            m.nodes[mids[ok]] = proj[ok]
    phys = gmsh.model.getPhysicalGroupsForEntity(3, vtag)
    m.elem_body = np.full(m.n_elems, int(phys[0]) - 1 if len(phys) else 0, dtype=np.int64)
    m.node_origin = np.arange(m.n_nodes)
    m._btri = None
    cen = m.nodes[m.boundary_tris].mean(axis=1)
    best = np.full(len(cen), np.inf)
    fid = np.zeros(len(cen), dtype=np.int64)
    for i, st in enumerate(surfs):
        cp = np.array(gmsh.model.getClosestPoint(2, st, cen.ravel().tolist())[0]).reshape(-1, 3)
        d = np.linalg.norm(cp - cen, axis=1)
        better = d < best
        best[better], fid[better] = d[better], i
    m.set_face_ids(fid)
    return m


def _prism_tets(tri, o0, o1):
    """三角柱（下の三角形 tri + o0、上 + o1）を四面体 3 つに分ける。

    頂点番号の小さい順に分割の向きを決める規則なので、隣り合う三角柱の共有面で分割がそろう。"""
    t = np.sort(tri, axis=1)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    return [np.column_stack([a + o0, b + o0, c + o0, c + o1]),
            np.column_stack([a + o0, b + o0, b + o1, c + o1]),
            np.column_stack([a + o0, a + o1, b + o1, c + o1])]


def _layered_sweep_mesh(gmsh, ls, order, h, log, share=None, wedge=False):
    """多層スイープ（複数ボディ・積層構造）。Ansys の「手法: スイープ」を積層品向けに広げたもの。

    すべてのボディの断面（スイープ方向に垂直な平らな面）を 1 つの平面に重ねて分割（fragment）し、
    その平面を三角形で切る。断面が変わる高さ（レベル）で区切った各層について、三角形ごとに
    どのボディの中かを調べ、三角柱を積み上げる。層の境目の節点は共有されるので、
    ボディ同士の境界はぴったり一致する（節点共有しない場合は後で分ける）。

    条件: すべての面が「スイープ方向に垂直な平面」か「スイープ方向に平行な面（側面）」であること。
    wedge=True なら三角柱要素（Wedge6/15）のまま返す。False なら三角柱を四面体 3 つに分ける。
    """
    occ = gmsh.model.occ
    vols = [t for _, t in gmsh.model.getEntities(3)]
    if not vols:
        raise GeometryError("スイープするボディがありません")
    surfs = [t for _, t in sorted(gmsh.model.getEntities(2), key=lambda e: e[1])]
    blo = np.array(gmsh.model.getBoundingBox(-1, -1)[:3])
    bhi = np.array(gmsh.model.getBoundingBox(-1, -1)[3:])
    diag = float(np.linalg.norm(bhi - blo))
    tol = 1e-6 * max(1.0, diag)
    want = ls.get("axis", "auto")
    ax = int(np.argmin(bhi - blo)) if want not in ("X", "Y", "Z") else "XYZ".index(want)
    e_ax = np.eye(3)[ax]
    # 面の分類: 垂直な平面（断面）/ 平行な面（側面）
    perp, levels, bad = [], [], []
    for st in surfs:
        bb = gmsh.model.getBoundingBox(2, st)
        typ = gmsh.model.getType(2, st)
        if typ == "Plane" and bb[ax + 3] - bb[ax] < tol * 10:
            perp.append((st, 0.5 * (bb[ax] + bb[ax + 3])))
            levels.append(0.5 * (bb[ax] + bb[ax + 3]))
            continue
        # 側面: いくつかの点で法線がスイープ方向と直交していること
        (u0, v0), (u1, v1) = [np.asarray(x) for x in gmsh.model.getParametrizationBounds(2, st)]
        uv = []
        for a in (0.25, 0.5, 0.75):
            for b in (0.25, 0.5, 0.75):
                uv += [u0 + a * (u1 - u0), v0 + b * (v1 - v0)]
        nrm = np.asarray(gmsh.model.getNormal(st, uv)).reshape(-1, 3)
        if np.max(np.abs(nrm @ e_ax)) > 1e-3:
            up, _ = gmsh.model.getAdjacencies(2, st)
            bad.append(st)
    if bad:
        bodies = set()
        for st in bad:
            up, _ = gmsh.model.getAdjacencies(2, st)
            for v in up:
                ph = gmsh.model.getPhysicalGroupsForEntity(3, int(v))
                bodies.add(int(ph[0]) - 1 if len(ph) else 0)
        raise GeometryError(
            f"スイープできない面があります（{'XYZ'[ax]} 方向に断面が一定でない形状）: ボディ "
            + ", ".join(f"B{b}" for b in sorted(bodies)) + "\n丸み付け（Z 方向以外の辺）・球・斜めの面などはスイープできません。"
            "四面体メッシュ（近接の捕捉）を使ってください。")
    lv = np.sort(np.asarray(levels))
    keep = np.concatenate([[True], np.diff(lv) > tol * 10])
    lv = lv[keep]
    if len(lv) < 2:
        raise GeometryError("スイープ方向に垂直な面が見つかりません")
    # 断面を 1 つの平面（最下のレベル）に重ねて分割する
    copies = []
    for st, z in perp:
        c = occ.copy([(2, st)])
        d = [0.0, 0.0, 0.0]
        d[ax] = float(lv[0] - z)
        occ.translate(c, *d)
        copies += c
    if len(copies) > 1:
        arr, amap = occ.fragment(copies[:1], copies[1:])
    else:
        arr = copies
    occ.synchronize()
    arr_tags = sorted({t for d_, t in arr if d_ == 2})
    hin = float(ls.get("size") or 0) or h
    for t in arr_tags:
        for (d_, e) in gmsh.model.getBoundary([(2, t)], oriented=False, recursive=True):
            if d_ == 0:
                gmsh.model.mesh.setSize([(0, e)], hin)
    log(f"多層スイープ: {'XYZ'[ax]} 方向、断面が変わる高さ {len(lv)} か所、平面の区画 {len(arr_tags)} 個")
    gmsh.model.mesh.generate(2)
    ntags, ncoord, _ = gmsh.model.mesh.getNodes()
    ncoord = np.asarray(ncoord).reshape(-1, 3)
    idx = np.full(int(np.max(ntags)) + 1, -1, dtype=np.int64)
    idx[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    tris, piece = [], []
    for t in arr_tags:
        ets, _, ens = gmsh.model.mesh.getElements(2, t)
        for et, en in zip(ets, ens):
            if int(et) == 2:
                tt = idx[np.asarray(en, dtype=np.int64)].reshape(-1, 3)
                tris.append(tt)
                piece.append(np.full(len(tt), t))
    if not tris:
        raise GeometryError("断面を三角形で切れませんでした")
    tri = np.concatenate(tris)
    piece = np.concatenate(piece)
    used = np.unique(tri)
    P = ncoord[used]
    tri = np.searchsorted(used, tri)
    n2 = len(P)
    # 区画ごとの代表点（区画の中の一番大きい三角形の重心）
    area = 0.5 * np.linalg.norm(np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]), axis=1)
    rep = {}
    for t in arr_tags:
        ii = np.nonzero(piece == t)[0]
        if len(ii):
            k = ii[np.argmax(area[ii])]
            rep[t] = P[tri[k]].mean(0)
    # ボディの範囲（候補を絞る）
    vbox = {v: np.asarray(gmsh.model.getBoundingBox(3, v)) for v in vols}
    vbody = {}
    for v in vols:
        ph = gmsh.model.getPhysicalGroupsForEntity(3, v)
        vbody[v] = int(ph[0]) - 1 if len(ph) else 0
    # 層ごとの分割数と高さの列
    hz = float(ls.get("layer_size") or 0) or h
    divs = int(ls.get("divs", 0) or 0)
    zs = [float(lv[0])]
    slab_rng = []
    for a, b in zip(lv[:-1], lv[1:]):
        n = divs if divs > 0 else max(1, int(round((b - a) / hz)))
        k0 = len(zs) - 1
        zs += [float(a + (b - a) * (j + 1) / n) for j in range(n)]
        slab_rng.append((k0, len(zs) - 1, 0.5 * (a + b)))
    zs = np.asarray(zs)
    # 区画 × 層 → ボディ
    elems, ebody = [], []
    for k0, k1, zm in slab_rng:
        owner = {}
        for t, c in rep.items():
            q = c.copy()
            q[ax] = zm
            for v in vols:
                bb = vbox[v]
                if np.all(q >= bb[:3] - tol) and np.all(q <= bb[3:] + tol):
                    if gmsh.model.isInside(3, v, q.tolist()):
                        owner[t] = v
                        break
        if not owner:
            continue
        sel = np.array([t in owner for t in piece])
        if not sel.any():
            continue
        tsel = tri[sel]
        bsel = np.array([vbody[owner[t]] for t in piece[sel]], dtype=np.int64)
        for k in range(k0, k1):
            o0, o1 = k * n2, (k + 1) * n2
            if wedge:
                elems.append(np.column_stack([tsel + o0, tsel + o1]))
                ebody.append(bsel)
            else:
                for tt in _prism_tets(tsel, o0, o1):
                    elems.append(tt)
                    ebody.append(bsel)
    if not elems:
        raise GeometryError("スイープでボディの中に要素ができませんでした")
    el = np.vstack(elems)
    eb = np.concatenate(ebody)
    nodes_all = np.repeat(P[None], len(zs), axis=0)
    nodes_all[:, :, ax] = zs[:, None]
    nodes_all = nodes_all.reshape(-1, 3)
    usedn = np.unique(el)
    remap = np.full(len(nodes_all), -1, dtype=np.int64)
    remap[usedn] = np.arange(len(usedn))
    nodes = nodes_all[usedn]
    el = remap[el]
    if not wedge:
        X = nodes[el]
        vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
        flip = vol < 0
        el[flip] = el[flip][:, [0, 2, 1, 3]]
    else:
        X = nodes[el]
        vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
        flip = vol < 0
        el[flip] = el[flip][:, [0, 2, 1, 3, 5, 4]]
    m = Mesh(nodes, el)
    m.elem_body = eb
    # ボディごとの体積の確認（CAD の体積と比べる）
    ev = np.abs(m.element_volumes())
    worst = 0.0
    vcad = {}
    for v in vols:
        vcad[vbody[v]] = vcad.get(vbody[v], 0.0) + occ.getMass(3, v)
    for b, V in vcad.items():
        mv = float(ev[eb == b].sum())
        if V > 0:
            worst = max(worst, abs(mv - V) / V)
    if worst > 0.05:
        log(f"注意: スイープのメッシュの体積が CAD と最大 {worst * 100:.1f}% 違います（曲面の近似）")
    if order == 2:
        m = m.to_quadratic()
        m.elem_body = eb
        curved = [st for st in surfs if gmsh.model.getType(2, st) != "Plane"]
        if curved:
            _project_midnodes(gmsh, m, curved, h)
    m = m.split_body_interfaces(share)
    _assign_face_ids(gmsh, m, surfs, tol=max(1e-3 * h, 10 * tol))
    log(f"多層スイープ完了: 層 {len(zs) - 1} / {'三角柱' if wedge else '四面体（三角柱を 3 分割）'} {m.n_elems:,} 要素"
        f"（体積の差 最大 {worst * 100:.2f}%）")
    return m


def _project_midnodes(gmsh, m, curved, h):
    """2 次要素の中間節点のうち、曲面のすぐ近くにあるものを CAD の面へ移す"""
    corner = m.elems.shape[1] // 2 if m.elems.shape[1] in (10, 20) else (6 if m.elems.shape[1] == 15 else 4)
    mids = np.unique(m.elems[:, corner:])
    if not len(mids):
        return
    Xm = m.nodes[mids]
    best = np.full(len(mids), np.inf)
    proj = Xm.copy()
    for st in curved:
        bb = np.asarray(gmsh.model.getBoundingBox(2, st))
        pad = 0.1 * h
        near = np.nonzero(np.all(Xm >= bb[:3] - pad, axis=1) & np.all(Xm <= bb[3:] + pad, axis=1))[0]
        if not len(near):
            continue
        cp = np.array(gmsh.model.getClosestPoint(2, st, Xm[near].ravel().tolist())[0]).reshape(-1, 3)
        d = np.linalg.norm(cp - Xm[near], axis=1)
        better = d < best[near]
        best[near[better]], proj[near[better]] = d[better], cp[better]
    ok = best < 0.05 * h
    m.nodes[mids[ok]] = proj[ok]


def _assign_face_ids(gmsh, m, surfs, tol):
    """境界の三角形に CAD のサーフェス番号を付ける（四面体メッシュの _extract_mesh と同じ番号の付け方）"""
    bt = m.boundary_tris
    cen = m.nodes[bt].mean(axis=1)
    best = np.full(len(cen), np.inf)
    fid = np.full(len(cen), -1, dtype=np.int64)
    for i, st in enumerate(surfs):
        bb = np.asarray(gmsh.model.getBoundingBox(2, st))
        pad = 10 * tol + 1e-9
        near = np.nonzero(np.all(cen >= bb[:3] - pad, axis=1) & np.all(cen <= bb[3:] + pad, axis=1))[0]
        if not len(near):
            continue
        cp = np.array(gmsh.model.getClosestPoint(2, st, cen[near].ravel().tolist())[0]).reshape(-1, 3)
        d = np.linalg.norm(cp - cen[near], axis=1)
        better = d < best[near]
        best[near[better]], fid[near[better]] = d[better], i
    if (fid < 0).any():
        m.group_faces()
        return
    n_surf = len(surfs)
    tb = m.tri_body() if m.elem_body is not None else np.zeros(len(fid), dtype=np.int64)
    first_body = {}
    for f, b in zip(fid.tolist(), tb.tolist()):
        first_body[f] = min(b, first_body.get(f, b))
    second = np.array([b != first_body[f] for f, b in zip(fid.tolist(), tb.tolist())])
    fid = np.where(second, fid + n_surf, fid)
    _, fid = np.unique(fid, return_inverse=True)
    m.set_face_ids(fid)


def mesh_quality(mesh) -> dict:
    """メッシュの品質（Ansys の メッシュメトリック 相当、頂点4点で評価）

    要素品質: 正四面体 = 1（Ansys と同じ定義: C·V / √(Σ辺²)³）
    歪度: 体積基準（正四面体 = 0、つぶれた要素 = 1）
    アスペクト比: 最長辺 / 最短の高さ × √(2/3)（正四面体 = 1）
    三角柱は _wedge_quality（要素品質は Ansys と同じ定義、歪度は角度基準）
    """
    if getattr(mesh, "is_hex", False):
        return _hex_quality(mesh)
    if getattr(mesh, "is_wedge", False):
        return _wedge_quality(mesh)
    X = mesh.nodes[mesh.elems[:, :4]]
    e = [X[:, j] - X[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    l2 = np.stack([np.einsum("ij,ij->i", v, v) for v in e], 1)
    V = np.abs(np.einsum("ij,ij->i", np.cross(e[0], e[1]), e[2])) / 6
    q = 124.70765802 * V / np.sqrt(l2.sum(1)) ** 3
    q = np.clip(q, 0, 1)
    # 外接球半径から理想体積
    a, b, c = e[0], e[1], e[2]
    num = np.linalg.norm(np.einsum("i,ij->ij", (a * a).sum(1), np.cross(b, c)) + np.einsum("i,ij->ij", (b * b).sum(1), np.cross(c, a))
                         + np.einsum("i,ij->ij", (c * c).sum(1), np.cross(a, b)), axis=1)
    R = num / np.maximum(12 * V, 1e-300)
    Vopt = 8 * np.sqrt(3) / 27 * R ** 3
    skew = np.clip((Vopt - V) / np.maximum(Vopt, 1e-300), 0, 1)
    faces = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    areas = np.stack([0.5 * np.linalg.norm(np.cross(X[:, f[1]] - X[:, f[0]], X[:, f[2]] - X[:, f[0]]), axis=1) for f in faces], 1)
    hmin = 3 * V / np.maximum(areas.max(1), 1e-300)
    ar = np.sqrt(l2.max(1)) / np.maximum(hmin, 1e-300) * np.sqrt(2.0 / 3.0)
    st = lambda v: {"最小": float(v.min()), "最大": float(v.max()), "平均": float(v.mean())}  # noqa: E731
    return {"要素品質": st(q), "歪度": st(skew), "アスペクト比": st(ar), "_quality": q, "_skew": skew, "_ar": ar}


def _wedge_quality(mesh) -> dict:
    """三角柱の品質（頂点 6 点で評価）

    要素品質: C·V / √(Σ辺²)³（Ansys の Element Quality、三角柱の C = 62.35382。正三角形 × 高さ = 辺 で 1）
    歪度: 面の内角の理想（三角形 60°、四角形 90°）からのずれの最大（等角歪度。0 = 理想、1 = つぶれ）
    アスペクト比: 最長辺 / 最短辺
    """
    from .elements import WEDGE_QUAD_FACES, wedge_volumes
    X = mesh.nodes[mesh.elems[:, :6]]
    E9 = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    l2 = np.stack([np.einsum("ij,ij->i", X[:, j] - X[:, i], X[:, j] - X[:, i]) for i, j in E9], 1)
    V = np.abs(wedge_volumes(mesh.nodes, mesh.elems[:, :6]))
    q = np.clip(62.35382 * V / np.sqrt(l2.sum(1)) ** 3, 0, 1)

    def angles(P, cyc):
        out = []
        n = len(cyc)
        for k in range(n):
            a, b, c = P[:, cyc[k - 1]], P[:, cyc[k]], P[:, cyc[(k + 1) % n]]
            u, v = a - b, c - b
            cosv = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
            out.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        return np.stack(out, 1)
    sk = []
    for cyc, th in (([0, 1, 2], 60.0), ([3, 4, 5], 60.0)):
        a = angles(X, cyc)
        sk.append(np.maximum((a.max(1) - th) / (180 - th), (th - a.min(1)) / th))
    for f in WEDGE_QUAD_FACES:
        a = angles(X, list(f))
        sk.append(np.maximum((a.max(1) - 90) / 90, (90 - a.min(1)) / 90))
    skew = np.clip(np.max(np.stack(sk, 1), axis=1), 0, 1)
    ar = np.sqrt(l2.max(1) / np.maximum(l2.min(1), 1e-300))
    st = lambda v: {"最小": float(v.min()), "最大": float(v.max()), "平均": float(v.mean())}  # noqa: E731
    return {"要素品質": st(q), "歪度": st(skew), "アスペクト比": st(ar), "_quality": q, "_skew": skew, "_ar": ar}


def _hex_quality(mesh) -> dict:
    """六面体の品質（頂点 8 点）: 要素品質 = C·V / √(Σ辺²)³（立方体で 1、C = 41.569219）、歪度 = 面の内角の 90° からの
    ずれ（等角歪度）、アスペクト比 = 最長辺 / 最短辺"""
    from .elements import HEX20_EDGES, HEX_QUAD_FACES, hex_volumes
    X = mesh.nodes[mesh.elems[:, :8]]
    l2 = np.stack([np.einsum("ij,ij->i", X[:, j] - X[:, i], X[:, j] - X[:, i]) for i, j in HEX20_EDGES], 1)
    V = np.abs(hex_volumes(mesh.nodes, mesh.elems[:, :8]))
    q = np.clip(41.569219 * V / np.sqrt(l2.sum(1)) ** 3, 0, 1)
    sk = []
    for f in HEX_QUAD_FACES:
        for k in range(4):
            a, b, c = X[:, f[k - 1]], X[:, f[k]], X[:, f[(k + 1) % 4]]
            u, v = a - b, c - b
            cosv = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
            sk.append(np.abs(np.degrees(np.arccos(np.clip(cosv, -1, 1))) - 90) / 90)
    skew = np.clip(np.max(np.stack(sk, 1), axis=1), 0, 1)
    ar = np.sqrt(l2.max(1) / np.maximum(l2.min(1), 1e-300))
    st = lambda v: {"最小": float(v.min()), "最大": float(v.max()), "平均": float(v.mean())}  # noqa: E731
    return {"要素品質": st(q), "歪度": st(skew), "アスペクト比": st(ar), "_quality": q, "_skew": skew, "_ar": ar}


def tet10_jacobian_ratio(mesh) -> np.ndarray:
    """2次四面体の曲がり具合: 角と積分点でのヤコビアン ÷ 直線要素のヤコビアン の最小値（要素ごと）。
    0 以下は要素が裏返っている（解析できない）、1 は直線の要素と同じ。"""
    from .solver import shape_dN_dxi
    if mesh.elems.shape[1] == 15 and getattr(mesh, "is_wedge", False):
        return _wedge15_jacobian_ratio(mesh)
    if mesh.elems.shape[1] != 10:
        return np.ones(mesh.n_elems)
    X = mesh.nodes[mesh.elems]
    lin = shape_dN_dxi(None, 4)
    J0 = np.linalg.det(np.einsum("eai,ak->eik", X[:, :4], lin))
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = [np.array(v) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                 [a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a], [.25] * 4)]
    r = np.full(mesh.n_elems, np.inf)
    for L in pts:
        J = np.linalg.det(np.einsum("eai,ak->eik", X, shape_dN_dxi(L, 10)))
        r = np.minimum(r, J / np.where(np.abs(J0) > 1e-300, J0, 1e-300))
    return r


def _wedge15_jacobian_ratio(mesh) -> np.ndarray:
    """2 次三角柱の曲がり具合: 節点・積分点でのヤコビアン ÷ 頂点だけ（直線の辺）の要素のヤコビアン の最小値"""
    from .elements import wedge_dN, wedge_node_points, wedge_quadrature
    X = mesh.nodes[mesh.elems]
    pts = np.vstack([wedge_node_points(15), wedge_quadrature(15)[0]])
    r = np.full(mesh.n_elems, np.inf)
    X6 = X[:, :6]
    for p in pts:
        J = np.linalg.det(np.einsum("eai,ak->eik", X, wedge_dN(p, 15)))
        J0 = np.linalg.det(np.einsum("eai,ak->eik", X6, wedge_dN(p, 6)))
        r = np.minimum(r, J / np.where(np.abs(J0) > 1e-300, J0, 1e-300))
    return r


def straighten_bad_tet10(mesh, limit=0.2, log=print, rounds=4) -> int:
    """曲がりすぎた 2 次要素（ヤコビアン比 < limit）の中間節点を辺の中点に戻す。直した要素数を返す（三角柱も）"""
    from .solver import _EDGES10
    nn = mesh.elems.shape[1]
    if nn == 15 and getattr(mesh, "is_wedge", False):
        from .elements import WEDGE15_EDGES
        edges, off = WEDGE15_EDGES, 6
    elif nn == 10:
        edges, off = _EDGES10, 4
    else:
        return 0
    fixed = 0
    for _ in range(rounds):
        bad = np.nonzero(tet10_jacobian_ratio(mesh) < limit)[0]
        if len(bad) == 0:
            break
        el = mesh.elems[bad]
        for k, (i, j) in enumerate(edges):
            mesh.nodes[el[:, off + k]] = 0.5 * (mesh.nodes[el[:, i]] + mesh.nodes[el[:, j]])
        fixed += len(bad)
    return fixed


def _proximity_points(gmsh, h, cells, pmin):
    """表面メッシュから板厚・すき間を測り、(点, 要素サイズ) を返す（Ansys の近接の捕捉と同じ考え方）。

    各表面三角形の重心から法線の両向きに光線を飛ばし、最初に当たる別の三角形までの距離を板厚（すき間）とする。
    サイズ = 板厚 / cells（pmin 以上 h 未満のところだけ返す）。返り値 (重心, サイズ, 三角形の外接半径)
    """
    from scipy.spatial import cKDTree
    ntags, xyz, _ = gmsh.model.mesh.getNodes()
    if len(ntags) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    P = np.asarray(xyz).reshape(-1, 3)
    idx = np.zeros(int(np.max(ntags)) + 1, dtype=np.int64)
    idx[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    tris = []
    for _, tag in gmsh.model.getEntities(2):
        ets, _, enodes = gmsh.model.mesh.getElements(2, tag)
        for t, en in zip(ets, enodes):
            if int(t) == 2:
                tris.append(idx[np.asarray(en, dtype=np.int64).reshape(-1, 3)])
    if not tris:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    T = np.concatenate(tris)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    cen = (A + B + C) / 3
    nrm = np.cross(B - A, C - A)
    area2 = np.linalg.norm(nrm, axis=1)
    ok = area2 > 0
    nrm[ok] /= area2[ok, None]
    tri_r = np.max(np.linalg.norm(np.stack([A, B, C], 1) - cen[:, None], axis=2), axis=1)
    R = cells * h                        # これより広いすき間はサイズが h 以上になるので見なくてよい
    tree = cKDTree(cen)
    rmax = 2 * float(tri_r.max())
    thick = np.full(len(T), np.inf)
    # 近い距離から順に探し、まだ相手の面が見つからない三角形だけ探す半径を広げる（薄い所はすぐ見つかる）
    stages = [r for r in (0.5 * h, 1.5 * h, R) if r < R] + [R]
    todo = np.arange(len(T))
    for r_s in stages:
        if not len(todo):
            break
        found = np.zeros(len(T), bool)
        chunk = 4000
        for c0 in range(0, len(todo), chunk):
            q = todo[c0:c0 + chunk]
            # 近い順に最大 K 個（半径内）。リストではなく配列で返るので速い
            K = min(len(T), 160)
            dd, jj = tree.query(cen[q], k=K, distance_upper_bound=r_s + rmax)
            ok = np.isfinite(dd)
            if not ok.any():
                continue
            i = np.broadcast_to(q[:, None], jj.shape)[ok]
            j = jj[ok]
            # 頂点を共有する三角形（隣どうし・自分自身）は除く
            share = np.zeros(len(i), bool)
            for a in range(3):
                for b in range(3):
                    share |= T[i, a] == T[j, b]
            i, j = i[~share], j[~share]
            if not len(i):
                continue
            # 向かい合う面だけ（法線がほぼ平行）: 斜めに当たる面（角の近く）は板厚ではない
            par = np.abs(np.einsum("ij,ij->i", nrm[i], nrm[j])) > 0.7
            i, j = i[par], j[par]
            # 光線 o + t·d（d = 三角形 i の法線、両向き）と三角形 j の交差（Möller–Trumbore）
            o, d = cen[i], nrm[i]
            e1, e2 = B[j] - A[j], C[j] - A[j]
            pv = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            good = np.abs(det) > 1e-14
            inv = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
            tv = o - A[j]
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            v = np.einsum("ij,ij->i", d, qv) * inv
            t = np.abs(np.einsum("ij,ij->i", e2, qv) * inv)
            hit = good & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6 * h) & (t < r_s)
            if hit.any():
                np.minimum.at(thick, i[hit], t[hit])
                found[i[hit]] = True
        todo = todo[~found[todo]]
    sel = np.isfinite(thick)
    size = np.clip(thick[sel] / cells, pmin, None)
    keep = size < 0.99 * h
    # 薄い部分にできる要素数の見積もり（両面で数えるので半分。正四面体の体積 ≈ サイズ³ / 8.5）
    a_t = 0.5 * area2[sel][keep]
    est = float(0.5 * np.sum(a_t * thick[sel][keep] / (size[keep] ** 3 / 8.5)))
    _proximity_points.estimate = est
    return cen[sel][keep], size[keep], tri_r[sel][keep]


def _apply_proximity(gmsh, h, ms, log):
    """近接の捕捉: 表面を一度粗く切って板厚を測り、要素サイズの関数（コールバック）を付けて切り直す"""
    from scipy.spatial import cKDTree
    cells = max(1, int(ms.get("proximity_cells", 3)))
    pmin = float(ms.get("proximity_min_size") or 0) or float(ms.get("curvature_min_size") or 0) or h * 0.05
    sp = np.asarray(ms.get("size_points") or [], float).reshape(-1, 4)
    if ms.get("capture_proximity"):
        gmsh.model.mesh.generate(2)
        pts, size, rad = _proximity_points(gmsh, h, cells, pmin)
        gmsh.model.mesh.clear()
        if not len(pts):
            log("近接の捕捉: 薄い部分・狭いすき間は見つかりませんでした")
    else:
        pts, size, rad = np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    est = getattr(_proximity_points, "estimate", 0.0) if ms.get("capture_proximity") else 0.0
    if len(sp):
        # 適応メッシュ（adaptive.py）が決めた局所の要素サイズ: 点 (x,y,z) の周り（半径 = そのサイズ）を細かく
        pts = np.vstack([pts, sp[:, :3]])
        size = np.r_[size, sp[:, 3]]
        rad = np.r_[rad, sp[:, 3]]
        pmin = min(pmin, float(sp[:, 3].min()))
        log(f"局所の細分化: {len(sp):,} 点（最小 {float(sp[:, 3].min()):.3g} mm）")
    if not len(pts):
        return None
    limit = float(ms.get("proximity_max_elements", 3e6))
    if est > limit:
        raise GeometryError(
            f"近接の捕捉で要素が約 {est / 1e6:.1f} 百万個になる見込みのため中止しました（上限 {limit / 1e6:.0f} 百万）。\n"
            f"最も薄い所は {float(size.min()) * cells:.3g} mm です。次のどれかを試してください:\n"
            "・「近接の最小サイズ」を大きくする\n・「すき間を横切る要素数」を 2 に減らす\n"
            "・積層品なら メッシュ制御 →「手法：スイープ」（多層スイープ）を使う")
    g = max(1.05, float(ms.get("growth_rate", 1.5)))
    tree = cKDTree(pts)
    # 要素サイズを格子に前計算しておき、コールバックでは格子を引くだけにする（gmsh は点ごとに呼ぶので速さが大事）
    lo = np.asarray(gmsh.model.getBoundingBox(-1, -1)[:3]) - h
    hi = np.asarray(gmsh.model.getBoundingBox(-1, -1)[3:]) + h
    ext = hi - lo
    dx = max(float(size.min()), float(np.prod(ext) / 4e6) ** (1 / 3))
    n = np.maximum(2, np.ceil(ext / dx).astype(int) + 1)
    ax = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    kk = min(8, len(pts))
    grid = np.full(len(G), h)
    for c0 in range(0, len(G), 500000):
        dist, ii = tree.query(G[c0:c0 + 500000], k=kk)
        dist, ii = dist.reshape(len(dist), -1), ii.reshape(len(ii), -1)
        # 各点は自分の三角形の範囲（外接半径）を覆う。その外側では成長率で大きくなる
        grid[c0:c0 + 500000] = np.min(size[ii] + (g - 1.0) * np.maximum(dist - rad[ii], 0.0), axis=1)
    grid = np.clip(grid, pmin, h).reshape(n)
    step = ext / (n - 1)

    def cb(dim, tag, x, y, z, lc):
        i = min(max(int(round((x - lo[0]) / step[0])), 0), n[0] - 1)
        j = min(max(int(round((y - lo[1]) / step[1])), 0), n[1] - 1)
        k = min(max(int(round((z - lo[2]) / step[2])), 0), n[2] - 1)
        return min(lc, float(grid[i, j, k]))
    gmsh.model.mesh.setSizeCallback(cb)
    if ms.get("capture_proximity"):
        log(f"近接の捕捉: 薄い部分・すき間 {len(pts) - len(sp):,} か所（最小 {float(size.min()):.3g} mm、すき間に {cells} 要素）")
    return float(size.min())


def _gmsh_quality_fix(gmsh, qmin, log):
    """gmsh の要素品質（γ）を見て、悪ければ最適化を重ねがけ。返り値 (修正前の不良数, 修正後の不良数)"""
    def count_bad():
        n, worst = 0, 1.0
        for _, tag in gmsh.model.getEntities(3):
            ets, etags, _ = gmsh.model.mesh.getElements(3, tag)
            for t, tg in zip(ets, etags):
                if int(t) != 4 or len(tg) == 0:
                    continue
                q = np.asarray(gmsh.model.mesh.getElementQualities(tg, "gamma"))
                n += int((q < qmin).sum())
                worst = min(worst, float(q.min()))
        return n, worst
    n0, w0 = count_bad()
    if n0 == 0:
        return 0, 0, w0, w0
    n1, w1 = n0, w0
    for method in ("", "Netgen", "Relocate3D", ""):
        try:
            gmsh.model.mesh.optimize(method, force=True, niter=2, threshold=max(qmin * 2, 0.3))
        except Exception:
            continue
        n1, w1 = count_bad()
        if n1 == 0:
            break
    return n0, n1, w0, w1


def build_mesh(spec: dict, size: float = 0.0, order: int = 2, curvature: int = 12,
               log=print, local_sizes: list | None = None, share: dict | None = None,
               settings: dict | None = None, step_out: str | None = None) -> Mesh:
    """形状 spec からメッシュを作る。

    spec 例: {"kind": "box", "lx": 100, "ly": 20, "lz": 10}
             {"kind": "file", "path": "/path/part.step"}
    size: 要素サイズ [mm]（0 なら自動）  order: 1=Tet4, 2=Tet10
    """
    kind = spec.get("kind")
    path = spec.get("path", "")
    ext = os.path.splitext(path)[1].lower() if kind == "file" else ""
    if kind == "model" and is_mixed_model(spec) and gmsh_available():
        # ソリッド＋シェル: 別々にメッシュして、1 つのモデル（MixedMesh）として返す
        s_spec, h_spec, off = split_mixed(spec)
        solid = build_mesh(s_spec, size, order, curvature, log, local_sizes, share, settings)
        shell = build_mesh(h_spec, size, order, curvature, log, None, None, settings)
        from .mixed import MixedMesh
        m = MixedMesh(solid, shell, off)
        log(f"ソリッド＋シェル: ソリッド {solid.n_nodes:,} 節点 / シェル {shell.n_nodes:,} 節点・{len(shell.edges)} 辺")
        return m
    if ext in PARASOLID_EXT:
        raise GeometryError(PARASOLID_MESSAGE)

    if not gmsh_available():
        if kind == "box":
            lx, ly, lz = (float(spec.get(k, d)) for k, d in (("lx", 100), ("ly", 20), ("lz", 10)))
            h = size if size > 0 else max(lx, ly, lz) / 20
            n = [max(1, int(round(v / h))) for v in (lx, ly, lz)]
            log("gmsh が無いため簡易メッシャーで直方体を分割します")
            m = structured_box(lx, ly, lz, *n)
            return m.to_quadratic() if order == 2 else m
        raise GeometryError("gmsh がインストールされていません。ターミナルで\n  pip install gmsh\nを実行してください。")

    gmsh = _gmsh_init()
    try:
        shell = kind == "model" and is_shell_model(spec)
        if shell:
            thick = _build_shell_features(gmsh, spec.get("features", []))
        elif kind == "model":
            _build_features(gmsh, spec.get("features", []))
        elif kind in PRIMITIVES:
            _build_primitive(gmsh, kind, spec)
        elif kind == "file":
            if not os.path.exists(path):
                raise GeometryError(f"ファイルが見つかりません: {path}")
            if ext in CAD_EXT:
                _import_cad(gmsh, path)
            elif ext in MESH_EXT:
                _import_stl(gmsh, path, float(spec.get("angle", 40.0)))
            else:
                raise GeometryError(f"未対応の形式です: {ext}")
        else:
            raise GeometryError(f"未知の形状指定: {spec}")

        if step_out:      # 作った形状を STEP で書き出す（Ansys などで同じ形状をメッシュして比べる用）
            gmsh.model.occ.synchronize()
            gmsh.write(step_out)
        ms = dict(MESH_DEFAULTS)
        ms.update(settings or {})
        diag = _bbox_diag(gmsh)
        h = resolved_size(diag, size, ms)
        log(f"メッシュ生成中（要素サイズ {h:.3g} mm, {'2次' if order == 2 else '1次'}要素）…")
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        hmin = float(ms.get("curvature_min_size") or 0) or h * 0.05
        gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
        if ms.get("capture_curvature", True):
            ang = max(1.0, float(ms.get("curvature_normal_angle", 30.0)))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(1, int(round(360.0 / ang))))
        else:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT（並列・高速）
        gmsh.option.setNumber("Mesh.Optimize", 1 if ms.get("optimize", True) else 0)
        gmsh.option.setNumber("Mesh.Smoothing", {"low": 1, "medium": 3, "high": 6}.get(ms.get("smoothing"), 3))
        if ms.get("smoothing") == "high":
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        ctrl = [x for x in (local_sizes or []) if not x.get("suppressed")]
        sweeps = [x for x in ctrl if x.get("type") == "sweep"]
        if ctrl:
            loc = [x for x in ctrl if x.get("type") != "sweep"]
            if ms.get("element_shape") == "hex" and not sweeps:
                # すべて六面体は四面体を 4 分割するので、局所サイズも 2 倍で切って同じ細かさにする
                loc = [dict(x, size=2 * float(x["size"])) if x.get("size") else x for x in loc]
            mins = _apply_local_sizes(gmsh, h, loc, ms.get("growth_rate", 1.5))
            if mins:
                gmsh.option.setNumber("Mesh.MeshSizeMin", min([hmin] + [v * 0.5 for v in mins]))
        mode2d = analysis_2d_mode(spec)
        if mode2d is not None:
            if not shell:
                raise GeometryError("2D 解析では、形状を XY 平面上のサーフェス（平板・多角形の面・円板）で作成してください")
            gmsh.model.mesh.generate(2)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.model.mesh.setOrder(2)
            mesh = _extract_2d_mesh(gmsh, thick, mode2d)
            log(f"2D メッシュ完了（{PLANE_MODES[mode2d]}）: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / 辺 {len(mesh.edges3)}")
            return mesh
        if shell:
            gmsh.model.mesh.generate(2)
            mesh = _extract_shell_mesh(gmsh, thick)
            log(f"シェルメッシュ完了: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / 面 {mesh.n_surf} / 辺 {len(mesh.edges)}")
            return mesh
        fix_msgs = []
        if sweeps:
            sw = sweeps[-1]
            if sw.get("element") == "hex":
                if len(gmsh.model.getEntities(3)) > 1:
                    raise GeometryError("六面体スイープは、ボディが 1 つのモデルで使えます")
                mesh = _sweep_mesh(gmsh, sw, order, h, log)
            elif len(gmsh.model.getEntities(3)) > 1 or sw.get("layered") or sw.get("element", "wedge") == "wedge":
                mesh = _layered_sweep_mesh(gmsh, sw, order, h, log, share=share,
                                           wedge=sw.get("element", "wedge") == "wedge")
            else:
                mesh = _sweep_mesh(gmsh, sw, order, h, log)
        else:
            cracks = [c for c in (ms.get("cracks") or []) if not c.get("suppressed")]
            ref = None
            if cracks:
                ref = _surface_reference(gmsh)
                _embed_cracks(gmsh, cracks, diag)
                log(f"き裂 {len(cracks)} 個を形状に埋め込みました")
                # 前縁の周りを細かく（Ansys のき裂の「前縁の分割数」に相当: 既定は全体サイズの 1/8）
                sp = list(ms.get("size_points") or [])
                for c in cracks:
                    hf = float(c.get("front_size") or 0) or h / 8
                    if c.get("shape") == "ellipse":
                        from .fracture import CrackGeom
                        ring = CrackGeom(c).ring
                        per = float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
                        idx = np.linspace(0, len(ring) - 1, max(8, int(np.ceil(per / hf)) + 1)).astype(int)
                        sp += [[*ring[k], hf] for k in idx]
                        continue
                    p0, p1, t, e1, n, L = crack_frame(c)
                    for u_ in np.linspace(0, 1, max(2, int(np.ceil(L / hf)) + 1)):
                        sp.append([*(p0 + u_ * L * t), hf])
                ms["size_points"] = sp
            if ms.get("capture_proximity") or ms.get("size_points"):
                pm = _apply_proximity(gmsh, h, ms, log)
                if pm is not None:
                    gmsh.option.setNumber("Mesh.MeshSizeMin", min(gmsh.option.getNumber("Mesh.MeshSizeMin"), pm))
            hexall = ms.get("element_shape") == "hex"
            if hexall:
                # すべて六面体: 四面体メッシュの各要素を 4 つの六面体に分ける（gmsh の細分化）。
                # 要素数は 4 倍になるので、サイズを 2 倍にして同じくらいの節点数にする
                gmsh.option.setNumber("Mesh.MeshSizeMax", 2 * h)
                gmsh.option.setNumber("Mesh.MeshSizeMin", 2 * gmsh.option.getNumber("Mesh.MeshSizeMin"))
                gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
                gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
            hexdom_mesh = None
            if ms.get("element_shape") == "hexdom":
                # 六面体優勢: 内側に六面体の格子、表面とのすき間を四面体で埋める（hexdom.py）
                if len(gmsh.model.getEntities(3)) != 1:
                    raise GeometryError("六面体優勢メッシュは、ボディが 1 つのモデルで使えます")
                if ref is not None:
                    raise GeometryError("六面体優勢メッシュは、き裂と一緒には使えません")
                from .hexdom import gmsh_hexdom
                try:
                    hexdom_mesh = gmsh_hexdom(gmsh, h, order, log)
                except GeometryError:
                    raise
                except Exception as e:      # すき間を切れなかったときは四面体に戻す
                    log(f"六面体優勢メッシュを作れませんでした（{type(e).__name__}: {e}）。四面体で切ります")
                    gmsh.model.mesh.clear()
                    gmsh.model.mesh.setOrder(1)
            if hexdom_mesh is None:
                gmsh.model.mesh.generate(3)
            if hexdom_mesh is not None:
                pass
            elif ms.get("auto_fix", True) and not hexall:
                n0, n1, w0, w1 = _gmsh_quality_fix(gmsh, float(ms.get("min_quality", 0.05)), log)
                if n0:
                    fix_msgs.append(f"品質 {ms.get('min_quality', 0.05):g} 未満の要素 {n0:,} 個 → 最適化で {n1:,} 個"
                                    f"（最小品質 {w0:.3f} → {w1:.3f}）")
            if order == 2 and hexdom_mesh is None:
                gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # 曲面に沿わせる
                gmsh.model.mesh.setOrder(2)
            mesh = hexdom_mesh if hexdom_mesh is not None else _extract_mesh(gmsh, share)
            if ref is not None:
                mesh = _apply_cracks(mesh, ref, cracks, 1e-6 * diag)
        if order == 2 and ms.get("auto_fix", True):
            nfix = straighten_bad_tet10(mesh, log=log)
            if nfix:
                fix_msgs.append(f"曲がりすぎた 2 次要素 {nfix:,} 個の中間節点を直線化")
        crk = [c for c in (ms.get("cracks") or []) if not c.get("suppressed")] if not shell else []
        if crk and order == 2 and hasattr(mesh, "crack_faces"):
            # き裂前縁の 1/4 点要素（中間節点の直線化のあとで。1/4 点要素はヤコビアン比が低くて当然なので）
            from .fracture import quarter_point
            nq = 0
            for c in crk:
                if not c.get("quarter_point", True):
                    continue
                if c.get("shape") == "ellipse":
                    from .fracture import CrackGeom
                    nq += quarter_point(mesh, CrackGeom(c).ring, tol=1e-6 * diag)
                else:
                    nq += quarter_point(mesh, c["front0"], c["front1"], tol=1e-6 * diag)
            if nq:
                mesh.quarter_point = True
                log(f"き裂前縁の 1/4 点要素: 中間節点 {nq:,} 個")
        if fix_msgs:
            log("メッシュ自動修正: " + " / ".join(fix_msgs))
        mesh.fix_log = fix_msgs
        vols_ = mesh.element_volumes()
        if np.any(vols_ <= 0):
            bad = np.nonzero(vols_ <= 0)[0]
            where = ""
            if mesh.elem_body is not None:
                bb = np.unique(mesh.elem_body[bad])
                where = "（ボディ " + ", ".join(f"B{b}" for b in bb[:10]) + (" …" if len(bb) > 10 else "") + "）"
            c = mesh.nodes[mesh.elems[bad[0]]].mean(0)
            raise GeometryError(f"不正な要素（体積 0 以下）が {len(bad):,} 個できました{where}。"
                                f"例: 座標 ({c[0]:.3g}, {c[1]:.3g}, {c[2]:.3g}) 付近。\n"
                                "要素サイズを小さくするか、別の手法（四面体／スイープ）で試してください。")
        log(f"メッシュ完了: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / 面 {mesh.n_faces}")
        return mesh
    finally:
        gmsh.finalize()

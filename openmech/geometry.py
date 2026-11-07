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
        if t in EDIT_FEATURES:
            raise GeometryError(f"サーフェス（シェル・2D）では「{FEATURES[t][0]}」にまだ対応していません（移動・回転・ミラーは可）")
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
    curves = sorted({abs(c[1]) for c in gmsh.model.getBoundary(vol_tags, combined=False, oriented=False,
                                                                 recursive=True) if c[0] == 1})
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


def _edit_bodies(gmsh, f, bodies, nkey):
    """ボディの編集フィーチャーを実行。新しく作ったボディの分だけ nkey を進めて返す"""
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
            plus = [dt for dt in plus if dt[0] == 3]
            minus = [dt for dt in minus if dt[0] == 3]
            if keep.startswith("＋"):
                occ.remove(minus, recursive=True)
                bodies[k] = plus
            elif keep.startswith("−"):
                occ.remove(plus, recursive=True)
                bodies[k] = minus
            else:
                bodies[k] = minus
                bodies[nkey] = plus
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
        for t, en in zip(etypes, enodes):
            npe = {4: 4, 11: 10}.get(int(t))
            if npe is None:
                continue
            b = tag2idx[en.astype(np.int64)].reshape(-1, npe)
            blocks.append(b)
            bodies.append(np.full(len(b), body, dtype=np.int64))
    if not blocks:
        raise GeometryError("四面体メッシュを作成できませんでした。メッシュサイズを変えて試してください。")
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
            npe = {2: 3, 9: 6}.get(int(t))
            if npe is None:
                continue
            tri = remap[tag2idx[en.astype(np.int64)].reshape(-1, npe)[:, :3]]
            for k in map(tuple, np.sort(tri, axis=1)):
                key_of[k] = fid
    orig = mesh.node_origin[mesh.boundary_tris]
    bt = np.sort(orig, axis=1)
    fids = np.array([key_of.get(tuple(k), -1) for k in bt])
    if (fids < 0).any():
        # 対応が取れない場合は角度で自動分割
        mesh.group_faces()
        return mesh
    # ボディ境界のサーフェスは両側に現れる: 番号の小さいボディ側はそのまま、もう一方は n_surf + 番号
    tb = mesh.tri_body()
    first_body = {}
    for f, b in zip(fids.tolist(), tb.tolist()):
        first_body[f] = min(b, first_body.get(f, b))
    second = np.array([b != first_body[f] for f, b in zip(fids.tolist(), tb.tolist())])
    fids = np.where(second, fids + n_surf, fids)
    _, fids = np.unique(fids, return_inverse=True)
    mesh.set_face_ids(fids)
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
    log(f"スイープ: {'XYZ'[ax]} 方向に {n} 分割（長さ {L:.4g} mm）")
    gmsh.model.mesh.generate(2)
    ets, _, ens = gmsh.model.mesh.getElements(2, src)
    tri = None
    for t, en in zip(ets, ens):
        if int(t) == 2:
            tri = en.astype(np.int64).reshape(-1, 3)
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


def mesh_quality(mesh) -> dict:
    """メッシュの品質（Ansys の メッシュメトリック 相当、頂点4点で評価）

    要素品質: 正四面体 = 1（Ansys と同じ定義: C·V / √(Σ辺²)³）
    歪度: 体積基準（正四面体 = 0、つぶれた要素 = 1）
    アスペクト比: 最長辺 / 最短の高さ × √(2/3)（正四面体 = 1）
    """
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
    return {"要素品質": st(q), "歪度": st(skew), "アスペクト比": st(ar), "_quality": q}


def tet10_jacobian_ratio(mesh) -> np.ndarray:
    """2次四面体の曲がり具合: 角と積分点でのヤコビアン ÷ 直線要素のヤコビアン の最小値（要素ごと）。
    0 以下は要素が裏返っている（解析できない）、1 は直線の要素と同じ。"""
    from .solver import shape_dN_dxi
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


def straighten_bad_tet10(mesh, limit=0.2, log=print, rounds=4) -> int:
    """曲がりすぎた 2 次要素（ヤコビアン比 < limit）の中間節点を辺の中点に戻す。直した要素数を返す"""
    from .solver import _EDGES10
    if mesh.elems.shape[1] != 10:
        return 0
    fixed = 0
    for _ in range(rounds):
        bad = np.nonzero(tet10_jacobian_ratio(mesh) < limit)[0]
        if len(bad) == 0:
            break
        el = mesh.elems[bad]
        for k, (i, j) in enumerate(_EDGES10):
            mesh.nodes[el[:, 4 + k]] = 0.5 * (mesh.nodes[el[:, i]] + mesh.nodes[el[:, j]])
        fixed += len(bad)
    return fixed


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
               settings: dict | None = None) -> Mesh:
    """形状 spec からメッシュを作る。

    spec 例: {"kind": "box", "lx": 100, "ly": 20, "lz": 10}
             {"kind": "file", "path": "/path/part.step"}
    size: 要素サイズ [mm]（0 なら自動）  order: 1=Tet4, 2=Tet10
    """
    kind = spec.get("kind")
    path = spec.get("path", "")
    ext = os.path.splitext(path)[1].lower() if kind == "file" else ""
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
            mins = _apply_local_sizes(gmsh, h, [x for x in ctrl if x.get("type") != "sweep"],
                                      ms.get("growth_rate", 1.5))
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
            mesh = _sweep_mesh(gmsh, sweeps[-1], order, h, log)
        else:
            gmsh.model.mesh.generate(3)
            if ms.get("auto_fix", True):
                n0, n1, w0, w1 = _gmsh_quality_fix(gmsh, float(ms.get("min_quality", 0.05)), log)
                if n0:
                    fix_msgs.append(f"品質 {ms.get('min_quality', 0.05):g} 未満の要素 {n0:,} 個 → 最適化で {n1:,} 個"
                                    f"（最小品質 {w0:.3f} → {w1:.3f}）")
            if order == 2:
                gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # 曲面に沿わせる
                gmsh.model.mesh.setOrder(2)
            mesh = _extract_mesh(gmsh, share)
        if order == 2 and ms.get("auto_fix", True):
            nfix = straighten_bad_tet10(mesh, log=log)
            if nfix:
                fix_msgs.append(f"曲がりすぎた 2 次要素 {nfix:,} 個の中間節点を直線化")
        if fix_msgs:
            log("メッシュ自動修正: " + " / ".join(fix_msgs))
        mesh.fix_log = fix_msgs
        if np.any(mesh.element_volumes() <= 0):
            raise GeometryError("不正な要素ができました。要素サイズを小さくして再試行してください。")
        log(f"メッシュ完了: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / 面 {mesh.n_faces}")
        return mesh
    finally:
        gmsh.finalize()

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
    "import": ("インポート", "読", [("path", "ファイル", ""),
                                 ("dx", "移動 X [mm]", 0.0), ("dy", "移動 Y [mm]", 0.0), ("dz", "移動 Z [mm]", 0.0)]),
}
OPS = {"new": "新規ボディ", "add": "結合", "cut": "除去（切り取り）", "intersect": "交差"}


def new_feature(ftype, op="new", **over):
    f = {"type": ftype, "op": op}
    for k, _, d in FEATURES[ftype][2]:
        f[k] = d
    f.update(over)
    return f


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
            raise GeometryError(f"{label}の{OPS.get(op, op)}に失敗しました: {e}")
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


def feature_body_keys(features) -> dict:
    """フィーチャー番号 → それが作ったボディ番号（_build_features と同じ規則）"""
    out, nkey, have = {}, 0, False
    for i, f in enumerate(features):
        if f.get("suppressed") or f.get("type") == "bool":
            continue
        op = f.get("op", "new")
        if op == "new" or (op == "add" and not have):
            out[i] = nkey
            nkey += 1
            have = True
    return out


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


def _extract_mesh(gmsh) -> Mesh:
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
    mesh = mesh.split_body_interfaces()

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


def _apply_local_sizes(gmsh, h, local_sizes):
    """面ごとのメッシュサイズ（Ansys の「面サイズ指定」相当）。

    local_sizes: [{"faces": [面番号...], "size": mm}, ...]  面番号は本ソフトの面番号（0始まり）
    指定面から離れるにつれて全体サイズ h へなめらかに大きくなります。
    """
    surfs = [t for _, t in sorted(gmsh.model.getEntities(2), key=lambda e: e[1])]
    fields = []
    for ls in local_sizes:
        # ボディ境界の裏側の面（番号 >= サーフェス数）は表側と同じサーフェスなので省略
        tags = sorted({surfs[f] for f in ls.get("faces", []) if 0 <= f < len(surfs)})
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
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", max(h - hs, 0.0) * 4.0)  # 成長率 約1.25
        fields.append(ft)
    if fields:
        fm = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(fm)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def build_mesh(spec: dict, size: float = 0.0, order: int = 2, curvature: int = 12,
               log=print, local_sizes: list | None = None) -> Mesh:
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
        if kind == "model":
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

        diag = _bbox_diag(gmsh)
        h = size if size > 0 else diag / 25.0
        log(f"メッシュ生成中（要素サイズ {h:.3g} mm, {'2次' if order == 2 else '1次'}要素）…")
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.05)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT（並列・高速）
        gmsh.option.setNumber("Mesh.Optimize", 1)
        if local_sizes:
            _apply_local_sizes(gmsh, h, local_sizes)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min([h * 0.05] + [float(x["size"]) * 0.5
                                                                         for x in local_sizes]))
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # 曲面に沿わせる
            gmsh.model.mesh.setOrder(2)
        mesh = _extract_mesh(gmsh)
        if np.any(mesh.element_volumes() <= 0):
            raise GeometryError("不正な要素ができました。要素サイズを小さくして再試行してください。")
        log(f"メッシュ完了: 節点 {mesh.n_nodes:,} / 要素 {mesh.n_elems:,} / 面 {mesh.n_faces}")
        return mesh
    finally:
        gmsh.finalize()

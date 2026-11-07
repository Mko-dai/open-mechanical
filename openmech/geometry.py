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

    etypes, _, enodes = gmsh.model.mesh.getElements(3)
    elems = None
    for t, en in zip(etypes, enodes):
        if t == 4:
            elems = tag2idx[en.astype(np.int64)].reshape(-1, 4)
        elif t == 11:
            elems = tag2idx[en.astype(np.int64)].reshape(-1, 10)
    if elems is None or len(elems) == 0:
        raise GeometryError("四面体メッシュを作成できませんでした。メッシュサイズを変えて試してください。")
    # 使われている節点だけ残す
    used = np.unique(elems)
    remap = np.full(len(coords), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes = coords[used]
    elems = remap[elems]
    if elems.shape[1] == 10:
        elems = _reorder_tet10(nodes, elems)
    mesh = Mesh(nodes, elems)

    # CAD のサーフェスを「面」として割り当てる
    key_of = {}
    for fid, (_, stag) in enumerate(sorted(gmsh.model.getEntities(2), key=lambda e: e[1])):
        ets, _, ens = gmsh.model.mesh.getElements(2, stag)
        for t, en in zip(ets, ens):
            npe = {2: 3, 9: 6}.get(int(t))
            if npe is None:
                continue
            tri = remap[tag2idx[en.astype(np.int64)].reshape(-1, npe)[:, :3]]
            for k in map(tuple, np.sort(tri, axis=1)):
                key_of[k] = fid
    bt = np.sort(mesh.boundary_tris, axis=1)
    fids = np.array([key_of.get(tuple(k), -1) for k in bt])
    if (fids < 0).any():
        # 対応が取れない場合は角度で自動分割
        mesh.group_faces()
    else:
        # 境界に現れない（内部の）サーフェスを除いて詰め直す
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
        tags = [surfs[f] for f in ls.get("faces", []) if 0 <= f < len(surfs)]
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
        if kind in PRIMITIVES:
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

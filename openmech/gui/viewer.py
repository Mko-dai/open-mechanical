"""3D 表示（PyVista）。VTK の文字は日本語が化けるため、3D画面内の文字は英語にしています。"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..mesh import Mesh

FACE_COLORS = ["#8fb8de", "#a7d3a6", "#f2c14e", "#c9a0dc", "#f4a259", "#7fc8c8",
               "#e5989b", "#b5c99a", "#9fa0ff", "#d4a373", "#90be6d", "#bdb2ff"]
SELECT_COLOR = "#ff6b00"

FIELD_EN = {
    "全変形 [mm]": "Total Deformation [mm]",
    "変位 X [mm]": "Directional Deformation X [mm]",
    "変位 Y [mm]": "Directional Deformation Y [mm]",
    "変位 Z [mm]": "Directional Deformation Z [mm]",
    "相当応力（ミーゼス） [MPa]": "Equivalent (von-Mises) Stress [MPa]",
    "最大主応力 [MPa]": "Maximum Principal Stress [MPa]",
    "最小主応力 [MPa]": "Minimum Principal Stress [MPa]",
    "応力 σx [MPa]": "Normal Stress X [MPa]",
    "応力 σy [MPa]": "Normal Stress Y [MPa]",
    "応力 σz [MPa]": "Normal Stress Z [MPa]",
    "せん断応力 τxy [MPa]": "Shear Stress XY [MPa]",
    "せん断応力 τyz [MPa]": "Shear Stress YZ [MPa]",
    "せん断応力 τzx [MPa]": "Shear Stress ZX [MPa]",
    "安全率（降伏基準）": "Safety Factor",
}


def surface_polydata(mesh: Mesh, points: np.ndarray | None = None) -> tuple[pv.PolyData, np.ndarray]:
    """境界面の PolyData。Tri6 は4分割して2次要素の値も滑らかに表示。

    戻り値: (polydata, 各セルの元の境界三角形番号)
    """
    pts = mesh.nodes if points is None else points
    t = mesh.boundary_tris_full
    if getattr(mesh, "has_quads", False):
        # 三角柱・六面体: 上下の三角形はそのまま（Tri6 は 4 分割）、側面の四角形は周の節点をつないだ多角形で描く
        qof = mesh.boundary_quad_of
        trow = np.nonzero(qof < 0)[0]
        cells, src = [], []
        tt = t[trow]
        if tt.shape[1] == 6:
            a, b, c, ab, bc, ca = tt.T
            for x in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]:
                cells.append(np.hstack([np.full((len(tt), 1), 3), np.stack(x, 1)]).ravel())
                src.append(trow)
        else:
            cells.append(np.hstack([np.full((len(tt), 1), 3), tt]).ravel())
            src.append(trow)
        q = mesh.boundary_quads_full
        first = np.full(len(q), -1, dtype=np.int64)
        qr = np.nonzero(qof >= 0)[0]
        first[qof[qr][::-1]] = qr[::-1]                      # 四角形ごとに最初の三角形の行
        cells.append(np.hstack([np.full((len(q), 1), q.shape[1]), q]).ravel())
        src.append(first)
        return pv.PolyData(pts.copy(), np.concatenate(cells)), np.concatenate(src)
    if t.shape[1] == 6:
        a, b, c, ab, bc, ca = t.T
        tris = np.concatenate([np.stack(x, 1) for x in
                               [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]])
        src = np.tile(np.arange(len(t)), 4)
    else:
        tris = t
        src = np.arange(len(t))
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    return pv.PolyData(pts.copy(), faces), src


def feature_edges(mesh: Mesh, points=None) -> pv.PolyData | None:
    """面と面の境目（形状の輪郭線）"""
    if mesh.face_ids is None:
        return None
    pts = mesh.nodes if points is None else points
    if getattr(mesh, "is_shell", False):
        if not mesh.edges:
            return None
        lines = np.vstack(mesh.edges)
        return pv.PolyData(pts.copy(), lines=np.hstack([np.full((len(lines), 1), 2), lines]).ravel())
    tris = mesh.boundary_tris
    fid = mesh.face_ids
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    f = np.tile(fid, 3)
    es = np.sort(e, axis=1)
    order = np.lexsort((es[:, 1], es[:, 0]))
    es, f = es[order], f[order]
    same = np.all(es[1:] == es[:-1], axis=1)
    border = same & (f[1:] != f[:-1])
    lines = es[:-1][border]
    if len(lines) == 0:
        return None
    cells = np.hstack([np.full((len(lines), 1), 2), lines]).ravel()
    return pv.PolyData(pts.copy(), lines=cells)


def volume_grid(mesh: Mesh, points=None) -> pv.UnstructuredGrid:
    """要素（体積）の格子。断面表示（切断）に使う。2 次要素は VTK の 2 次セルのまま"""
    pts = mesh.nodes if points is None else points
    nn = mesh.elems.shape[1]
    ctype = {4: pv.CellType.TETRA, 10: pv.CellType.QUADRATIC_TETRA, 6: pv.CellType.WEDGE,
             15: pv.CellType.QUADRATIC_WEDGE, 8: pv.CellType.HEXAHEDRON, 20: pv.CellType.QUADRATIC_HEXAHEDRON}[nn]
    cells = np.hstack([np.full((mesh.n_elems, 1), nn), mesh.elems]).ravel()
    return pv.UnstructuredGrid(cells, np.full(mesh.n_elems, ctype, dtype=np.uint8), np.asarray(pts, float).copy())

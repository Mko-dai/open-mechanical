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

"""検証5モデルのコンター図（Ansys 風・9段階）を logs/contours/*.png に保存

  python -m openmech.contours
"""
from __future__ import annotations

import os

import numpy as np

from . import benchmarks as bm
from .verification import MODELS

# モデルごとに表示する結果（Ansys の結果名、値の取り出し方、単位）
FIELDS = {
    "VMC1": [("方向応力 X（SX）", lambda r: r.stress[:, 0], "psi"),
             ("方向変形 Z（UZ）", lambda r: r.u[:, 2], "in")],
    "LE1": [("方向応力 Y（σyy）", lambda r: r.stress[:, 1], "MPa")],
    "LE10": [("方向応力 Y（σyy）", lambda r: r.stress[:, 1], "MPa")],
    "LAME": [("相当応力（ミーゼス）", lambda r: r.von_mises, "MPa"),
             ("全変形", lambda r: r.disp_mag, "mm")],
    "HOLE": [("方向応力 X（σx）", lambda r: r.stress[:, 0], "MPa"),
             ("相当応力（ミーゼス）", lambda r: r.von_mises, "MPa")],
}
VIEWS = {"VMC1": "below", "LE1": "xy", "LE10": "iso", "LAME": "xy", "HOLE": "hole"}


def _font():
    for p in ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
              "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if os.path.exists(p):
            return p
    return None


def render(mesh, values, title, unit, path, view="iso"):
    import pyvista as pv
    from .gui.viewer import feature_edges, surface_polydata
    poly, src = surface_polydata(mesh)
    poly.point_data["v"] = np.asarray(values, float)
    used = np.unique(mesh.boundary_tris_full)
    vals = np.asarray(values, float)
    vmin, vmax = float(vals[used].min()), float(vals[used].max())
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("#eef3fa", top="#c9d6ea")
    font = _font()
    sb = {"title": unit + "\n", "vertical": True, "fmt": "%.4g", "n_labels": 10, "position_x": 0.03,
          "position_y": 0.2, "height": 0.6, "width": 0.05, "label_font_size": 22, "title_font_size": 22,
          "color": "black"}
    pl.add_mesh(poly, scalars="v", cmap="jet", n_colors=9, clim=[vmin, vmax], scalar_bar_args=sb,
                smooth_shading=False)
    fe = feature_edges(mesh)
    if fe is not None:
        pl.add_mesh(fe, color="black", line_width=2)
    imax, imin = used[np.argmax(vals[used])], used[np.argmin(vals[used])]
    # 最大・最小の位置は印だけ付け、値は右上にまとめて表示（ラベルの重なりを防ぐ）
    pl.add_points(mesh.nodes[[imax]], color="red", point_size=18, render_points_as_spheres=True)
    pl.add_points(mesh.nodes[[imin]], color="blue", point_size=18, render_points_as_spheres=True)
    pl.add_text(f"Max {vals[imax]:.5g} {unit}  (red)\nMin {vals[imin]:.5g} {unit}  (blue)",
                position="upper_right", font_size=14, color="black")
    if view == "xy":
        pl.view_xy()
        pl.reset_camera()
        pl.camera.zoom(1.15)
    elif view == "below":
        lo, hi = mesh.bounds()
        c = 0.5 * (lo + hi)
        L = float(np.linalg.norm(hi - lo))
        pl.camera.focal_point = c
        pl.camera.position = c + np.array([1.0, 1.0, -0.8]) * L
        pl.camera.up = (0, 0, -1)
        pl.reset_camera()
        pl.camera.zoom(1.1)
    elif view == "hole":
        lo, hi = mesh.bounds()
        c = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2]])
        pl.camera.focal_point = c
        pl.camera.position = c + np.array([-1.0, -1.2, 1.4]) * 40
        pl.camera.up = (0, 0, 1)
    else:
        pl.view_isometric()
        pl.reset_camera()
        pl.camera.zoom(1.15)
    pl.screenshot(path)
    pl.close()


def main():
    out = os.path.join("logs", "contours")
    os.makedirs(out, exist_ok=True)
    holder = {}
    orig = bm.solve

    def wrapped(*a, **k):
        r = orig(*a, **k)
        holder["r"] = r
        return r

    bm.solve = wrapped
    try:
        from .geometry import gmsh_available
        for M in MODELS:
            if M["id"] == "HOLE" and not gmsh_available():
                continue
            mesh, _ = M["run"]()
            r = holder["r"]
            for k, (name, fn, unit) in enumerate(FIELDS[M["id"]]):
                path = os.path.join(out, f"{M['id']}_{k + 1}.png")
                render(mesh, fn(r), f"{M['title']}：{name}", unit, path, VIEWS[M["id"]])
                print("保存:", path, flush=True)
    finally:
        bm.solve = orig


if __name__ == "__main__":
    main()

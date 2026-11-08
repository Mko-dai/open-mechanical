"""三角柱要素（Wedge6 / Wedge15）: はり理論・薄板・荷重の合計・質量・陽解法"""
import math

import numpy as np

from openmech.materials import Material
from openmech.mesh import Mesh
from openmech.solver import BC, solve

E, NU = 200000.0, 0.3


def prism_mesh(lx, ly, lz, nx, ny, nz, quadratic=True):
    """xy 平面の長方形を三角形に分け、z 方向に nz 層押し出した三角柱メッシュ"""
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: i * (ny + 1) + j  # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    tris = np.array(tris)
    zs = np.linspace(0, lz, nz + 1)
    n2 = len(P)
    nodes = np.vstack([np.column_stack([P, np.full(n2, z)]) for z in zs])
    el = np.vstack([np.column_stack([tris + k * n2, tris + (k + 1) * n2]) for k in range(nz)])
    m = Mesh(nodes, el)
    return m.to_quadratic() if quadratic else m


def faces_where(m, axis, value, tol=1e-6):
    out = []
    for f in m.face_info():
        if abs(f["centroid"][axis] - value) < tol and abs(abs(f["normal"][axis]) - 1) < 1e-6:
            out.append(f["id"])
    return out


def cantilever(m, L, b, h, P=-100.0):
    mat = Material("鋼", E, NU, 7850)
    bcs = [BC("fixed", faces_where(m, 0, 0.0)), BC("force", faces_where(m, 0, L), [0, 0, P])]
    r = solve(m, mat, bcs, log=lambda *a: None)
    tip = np.abs(m.nodes[:, 0] - L) < 1e-9
    return r, float(r.u[tip, 2].mean())


def test_shape_and_volume():
    m = prism_mesh(10, 4, 2, 5, 2, 2)
    assert m.elems.shape[1] == 15 and m.is_wedge
    assert math.isclose(m.volume(), 80.0, rel_tol=1e-12)
    # 境界: 上下の三角形 + 側面の四角形
    assert len(m.boundary_quads_full) == 2 * (5 + 2) * 2 and m.boundary_quads_full.shape[1] == 8
    assert m.n_faces == 6


def test_cantilever_beam_theory():
    L, b, h = 100.0, 10.0, 10.0
    m = prism_mesh(L, b, h, 20, 2, 4)
    _, d = cantilever(m, L, b, h)
    th = -100 * L ** 3 / (3 * E * b * h ** 3 / 12)
    assert abs(d / th - 1) < 0.03, (d, th)


def test_thin_plate_one_layer():
    """板厚 0.5 mm を 1 層、面内 5 mm（縦横比 10）の三角柱でも曲げが正しい（四面体では難しい条件）"""
    L, b, h = 100.0, 20.0, 0.5
    m = prism_mesh(L, b, h, 20, 4, 1)
    _, d = cantilever(m, L, b, h, P=-0.01)
    I = b * h ** 3 / 12
    th_beam = -0.01 * L ** 3 / (3 * E * I)
    th_plate = th_beam * (1 - NU ** 2)          # 幅の広い板（平面ひずみ）
    assert th_plate * 1.1 < d < th_beam * 0.9 or abs(d / th_beam - 1) < 0.1, (d, th_beam, th_plate)


def test_pressure_total():
    from openmech.solver import build_load_and_constraints
    m = prism_mesh(10, 4, 2, 5, 2, 2)
    mat = Material("鋼", E, NU, 7850)
    for axis, val, area in ((2, 2.0, 40.0), (0, 10.0, 8.0), (1, 0.0, 20.0)):    # 上面（三角形）/ 側面（四角形）
        f = faces_where(m, axis, val)
        F, _ = build_load_and_constraints(m, mat, [BC("pressure", f, [3.0])])
        tot = F.reshape(-1, 3).sum(0)
        assert math.isclose(abs(tot[axis]), 3.0 * area, rel_tol=1e-9), (axis, tot)
        F, _ = build_load_and_constraints(m, mat, [BC("force", f, [1.0, 2.0, 3.0])])
        assert np.allclose(F.reshape(-1, 3).sum(0), [1, 2, 3])


def test_lumped_mass_and_gravity():
    from openmech.dynamics import lumped_mass
    from openmech.solver import body_force_vector
    m = prism_mesh(10, 4, 2, 5, 2, 2)
    rho = 7.85e-9
    M = lumped_mass(m, np.full(m.n_elems, rho))
    assert math.isclose(M.sum(), rho * 80, rel_tol=1e-12) and M.min() > 0
    F = body_force_vector(m, np.array([0, 0, -rho * 9810]))
    assert math.isclose(F.reshape(-1, 3)[:, 2].sum(), -rho * 9810 * 80, rel_tol=1e-12)


def test_explicit_quasi_static_matches_static():
    from openmech.dynamics import DynamicSettings, solve_dynamic
    L, b, h = 100.0, 10.0, 10.0
    m = prism_mesh(L, b, h, 10, 2, 2)
    _, d_static = cantilever(m, L, b, h)
    mat = Material("鋼", E, NU, 7850)
    bcs = [BC("fixed", faces_where(m, 0, 0.0)), BC("force", faces_where(m, 0, L), [0, 0, -100.0])]
    # 1 次の固有振動数 ≈ 820 Hz → 周期 1.2 ms。ゆっくり荷重を上げて減衰させる
    s = DynamicSettings(method="explicit", end_time=0.02, ramp_time=0.01, n_frames=1, alpha_damping=3000.0)
    tip = np.abs(m.nodes[:, 0] - L) < 1e-9
    fr = solve_dynamic(m, mat, bcs, s, log=lambda *a: None)
    d = float(fr[-1].u[tip, 2].mean())
    assert abs(d / d_static - 1) < 0.02, (d, d_static)


def test_linear_wedge6():
    L, b, h = 100.0, 10.0, 10.0
    m = prism_mesh(L, b, h, 40, 4, 8, quadratic=False)
    _, d = cantilever(m, L, b, h)
    th = -100 * L ** 3 / (3 * E * b * h ** 3 / 12)
    assert 0.6 < d / th < 1.02

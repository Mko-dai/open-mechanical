"""2次元解析（平面応力・平面ひずみ・軸対称）"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.plane import solve_2d, structured_rect
from openmech.solver import BC

MAT = Material("鋼", 200000, 0.3, 7850, 250)


def test_plane_stress_cantilever():
    m = structured_rect(100, 10, 40, 4, t=5.0)
    r = solve_2d(m, MAT, [BC("fixed", [1]), BC("force", [2], [0, -100.0, 0])], "plane_stress", log=lambda *a: None)
    I = 5 * 10 ** 3 / 12
    d = 100 * 100 ** 3 / (3 * 200000 * I) + 100 * 100 / (200000 / 2.6 * 50 * 5 / 6)
    assert abs(-r.u[:, 1].min() / d - 1) < 0.02


def _lame(mode):
    a, b, p = 100.0, 200.0, 100.0
    m = structured_rect(b - a, 20, 20, 4, x0=a, mode=mode)
    bcs = [BC("symmetry", [3]), BC("symmetry", [4]), BC("pressure", [1], [p])]
    if mode == "plane_strain":
        raise RuntimeError
    r = solve_2d(m, MAT, bcs, mode, log=lambda *a: None)
    inner = np.isclose(m.nodes[:, 0], a)
    return r, inner, a, b, p


def test_axisymmetric_lame():
    """厚肉円筒（軸方向は拘束 = 平面ひずみ）の内圧: ラメの解"""
    r, inner, a, b, p = _lame("axisymmetric")
    st = p * (b ** 2 + a ** 2) / (b ** 2 - a ** 2)
    ur = (1 + 0.3) * p * a ** 2 / (200000 * (b ** 2 - a ** 2)) * ((1 - 0.6) * a + b ** 2 / a)
    assert abs(r.u[inner, 0].mean() / ur - 1) < 0.005
    assert abs(r.stress[inner, 2].mean() / st - 1) < 0.03       # 円周応力（節点平均のため内面で少し小さい）


def test_plane_strain_uniaxial_and_plastic():
    m = structured_rect(10, 10, 4, 4, mode="plane_strain")
    bcs = [BC("symmetry", [1]), BC("symmetry", [3]), BC("displacement", [2], [0.01, None, None])]
    r = solve_2d(m, MAT, bcs, "plane_strain", log=lambda *a: None)
    E, nu, e = 200000, 0.3, 0.001
    sx = E / (1 - nu ** 2) * e                 # 平面ひずみの単軸（σy = 0, εz = 0）
    assert np.allclose(r.stress[:, 0], sx, rtol=1e-6)
    assert np.allclose(r.stress[:, 2], nu * sx, rtol=1e-6)
    pl = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    for mode in ("plane_stress", "plane_strain", "axisymmetric"):
        mm = structured_rect(10, 10, 4, 4, x0=0 if mode != "axisymmetric" else 10, mode=mode)
        b2 = [BC("symmetry", [3]), BC("displacement", [1], [0.0, None, None]) if mode != "axisymmetric" else BC("symmetry", [3]),
              BC("displacement", [4], [None, 0.05, None])]
        rr = solve_2d(mm, pl, b2, mode, elem_materials=([pl.state()], np.zeros(mm.n_elems, int)), log=lambda *a: None)
        assert rr.fields()["相当塑性ひずみ"].max() > 0.001


def test_geometry_2d_plate_with_hole():
    """gmsh で穴あき平板（平面応力）→ 応力集中係数 ≈ 3（無限板の理論値）"""
    pytest.importorskip("gmsh")
    from openmech.geometry import build_mesh, new_feature
    from openmech.project import Project
    from openmech.solver import BC
    spec = {"kind": "model", "analysis_2d": "plane_stress",
            "features": [new_feature("plate", du=100.0, dv=50.0, t=1.0),
                         new_feature("disk", "cut", u0=0.0, v0=0.0, r=5.0)]}
    m = build_mesh(spec, size=1.0, log=lambda *a: None, local_sizes=[])
    assert m.is_2d and m.mode == "plane_stress"
    # 1/4 モデル: x=0 と y=0 が対称面、x=100 に引張 100 MPa
    info = {}
    for e, seg in enumerate(m.edges3):
        p = m.nodes[seg.ravel()]
        info[m.n_surf + e] = (p.min(0), p.max(0))
    xsym = [k for k, (lo, hi) in info.items() if hi[0] < 1e-6]
    ysym = [k for k, (lo, hi) in info.items() if hi[1] < 1e-6]
    right = [k for k, (lo, hi) in info.items() if lo[0] > 100 - 1e-6]
    pr = Project(geometry=spec)
    pr.bcs = [BC("symmetry", xsym), BC("symmetry", ysym), BC("pressure", right, [-100.0])]
    r = pr.run(m, log=lambda *a: None)
    top = np.argmin(np.linalg.norm(m.nodes[:, :2] - [0, 5], axis=1))
    k = r.stress[top, 0] / 100.0
    assert 2.6 < k < 3.3

"""理論解との比較による検証テスト（gmsh 不要）。  実行: python -m pytest -q"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import structured_box
from openmech.solver import BC, solve

STEEL = Material("test", 200000.0, 0.3, 7850.0, 250.0)


def face_at(mesh, axis, value, tol=1e-6):
    """座標 axis=value の平面上にある面番号"""
    for f in mesh.face_info():
        if abs(f["centroid"][axis] - value) < tol and abs(abs(f["normal"][axis]) - 1) < 1e-6:
            return f["id"]
    raise AssertionError("face not found")


def test_box_has_six_faces():
    m = structured_box(10, 5, 3, 4, 2, 2)
    assert m.n_faces == 6
    assert m.volume() == pytest.approx(150.0)
    m2 = m.to_quadratic()
    assert m2.n_faces == 6
    assert m2.volume() == pytest.approx(150.0)


@pytest.mark.parametrize("quad", [False, True])
def test_uniaxial_tension(quad):
    """引張棒: σ = F/A,  δ = F L / (E A)"""
    L, a, F = 100.0, 10.0, 10000.0
    m = structured_box(L, a, a, 10, 2, 2)
    if quad:
        m = m.to_quadratic()
    x0, x1 = face_at(m, 0, 0.0), face_at(m, 0, L)
    y0, z0 = face_at(m, 1, 0.0), face_at(m, 2, 0.0)
    bcs = [BC("displacement", [x0], [0.0, None, None]),
           BC("displacement", [y0], [None, 0.0, None]),
           BC("displacement", [z0], [None, None, 0.0]),
           BC("force", [x1], [F, 0, 0])]
    r = solve(m, STEEL, bcs, log=lambda *a: None)
    sigma = F / a**2
    assert r.stress[:, 0].mean() == pytest.approx(sigma, rel=1e-6)
    assert r.u[:, 0].max() == pytest.approx(F * L / (STEEL.E * a * a), rel=1e-6)
    # 反力の釣り合い
    assert r.reactions[0][0] == pytest.approx(-F, rel=1e-6)


def test_cantilever_tet10():
    """片持ち梁の先端たわみ: δ = P L^3 / (3 E I)（＋せん断変形）"""
    L, b, h, P = 200.0, 20.0, 20.0, 1000.0
    m = structured_box(L, b, h, 20, 3, 3).to_quadratic()
    x0, x1 = face_at(m, 0, 0.0), face_at(m, 0, L)
    r = solve(m, STEEL, [BC("fixed", [x0]), BC("force", [x1], [0, 0, -P])], log=lambda *a: None)
    I = b * h**3 / 12
    G = STEEL.E / (2 * (1 + STEEL.nu))
    delta = P * L**3 / (3 * STEEL.E * I) + 1.2 * P * L / (G * b * h)  # ティモシェンコ梁
    tip = -r.u[np.isclose(m.nodes[:, 0], L), 2].mean()
    assert tip == pytest.approx(delta, rel=0.03)
    # 根元の曲げ応力 σ = M c / I （固定端の拘束効果を避けて L/4 の位置で比較）
    x = L / 4
    sel = np.isclose(m.nodes[:, 0], x) & np.isclose(m.nodes[:, 2], h)
    sigma = P * (L - x) * (h / 2) / I
    assert r.stress[sel, 0].mean() == pytest.approx(sigma, rel=0.05)


def test_pressure_and_gravity_balance():
    m = structured_box(50, 20, 10, 5, 2, 2).to_quadratic()
    z0, z1 = face_at(m, 2, 0.0), face_at(m, 2, 10.0)
    r = solve(m, STEEL, [BC("fixed", [z0]), BC("pressure", [z1], [1.0]),
                         BC("gravity", [], [0, 0, -9810.0])], log=lambda *a: None)
    # 反力 Z = 圧力×面積 + 重量
    weight = 50 * 20 * 10 * STEEL.density_t_mm3 * 9810.0
    assert r.reactions[0][2] == pytest.approx(1.0 * 50 * 20 + weight, rel=1e-6)


def test_moment_torsion_balance():
    m = structured_box(100, 10, 10, 10, 3, 3).to_quadratic()
    x0, x1 = face_at(m, 0, 0.0), face_at(m, 0, 100.0)
    T = 5000.0
    r = solve(m, STEEL, [BC("fixed", [x0]), BC("moment", [x1], [T, 0, 0])], log=lambda *a: None)
    # 合力はゼロ
    assert np.abs(r.reactions[0]).max() < 1e-6 * T
    assert r.von_mises.max() > 0


def test_unconstrained_raises():
    m = structured_box(10, 10, 10, 2, 2, 2)
    with pytest.raises(ValueError):
        solve(m, STEEL, [BC("force", [0], [1, 0, 0])], log=lambda *a: None)


def test_insufficient_constraint_detected():
    """X方向だけ拘束してY方向に引っ張る → 剛体運動でエラーになること"""
    m = structured_box(10, 10, 10, 2, 2, 2).to_quadratic()
    x0, x1 = face_at(m, 0, 0.0), face_at(m, 0, 10.0)
    with pytest.raises(RuntimeError):
        solve(m, STEEL, [BC("displacement", [x0], [0.0, None, None]),
                         BC("force", [x1], [0, 100.0, 0])], log=lambda *a: None)


def test_tet10_reorder_gmsh_style():
    """gmsh 形式の中間節点順（8,9 が入れ替わる）を正しく並べ直せること"""
    from openmech.geometry import _reorder_tet10
    m = structured_box(10, 10, 10, 2, 2, 2).to_quadratic()
    g = m.elems.copy()
    g[:, [8, 9]] = g[:, [9, 8]]
    assert np.array_equal(_reorder_tet10(m.nodes, g), m.elems)
    rng = np.random.default_rng(0)
    g2 = m.elems.copy()
    g2[:, 4:] = g2[:, 4 + rng.permutation(6)]
    assert np.array_equal(_reorder_tet10(m.nodes, g2), m.elems)


def test_export(tmp_path):
    from openmech.export import write_csv, write_report, write_vtu
    import xml.etree.ElementTree as ET
    m = structured_box(20, 5, 5, 4, 1, 1).to_quadratic()
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -10])]
    r = solve(m, STEEL, bcs, log=lambda *a: None)
    write_vtu(tmp_path / "r.vtu", r)
    ET.parse(tmp_path / "r.vtu")
    write_csv(tmp_path / "r.csv", r)
    write_report(tmp_path / "r.html", r, bcs)
    assert (tmp_path / "r.html").stat().st_size > 1000


def test_two_materials_in_series():
    """鋼とアルミを直列につないだ棒: 伸び = F L1/(E1 A) + F L2/(E2 A)"""
    L, a, F = 100.0, 10.0, 1000.0
    m = structured_box(L, a, a, 20, 2, 2).to_quadratic()
    cx = m.nodes[m.elems[:, :4]].mean(axis=1)[:, 0]
    elem_mat = (cx > L / 2).astype(int)
    al = Material("al", 70000.0, 0.3, 2700.0, 100.0)
    x0, x1 = face_at(m, 0, 0.0), face_at(m, 0, L)
    y0, z0 = face_at(m, 1, 0.0), face_at(m, 2, 0.0)
    bcs = [BC("displacement", [x0], [0.0, None, None]), BC("displacement", [y0], [None, 0.0, None]),
           BC("displacement", [z0], [None, None, 0.0]), BC("force", [x1], [F, 0, 0])]
    r = solve(m, STEEL, bcs, log=lambda *a: None, elem_materials=([STEEL, al], elem_mat))
    d = F * (L / 2) / (STEEL.E * a * a) + F * (L / 2) / (al.E * a * a)
    assert r.u[:, 0].max() == pytest.approx(d, rel=0.01)  # 境界面の横方向拘束でわずかに剛くなる
    assert r.summary()["質量 [kg]"] == pytest.approx(50 * a * a * (7850 + 2700) * 1e-9, rel=1e-6)
    sf = r.safety_factor
    far = m.nodes[:, 0] > 0.8 * L   # 境界面から離れたアルミ側: σ=10MPa, 降伏100 → 安全率10
    assert np.median(sf[far]) == pytest.approx(10.0, rel=1e-2)
    near_steel = m.nodes[:, 0] < 0.2 * L   # 鋼側: 降伏250 → 25 だが上限15
    assert np.median(sf[near_steel]) == pytest.approx(15.0)

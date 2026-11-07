"""材料: 温度依存・直交異方性・熱ひずみ・塑性"""
import numpy as np

from openmech.materials import Material, rotation_xyz
from openmech.mesh import structured_box
from openmech.nonlinear import AnalysisSettings, solve_static
from openmech.solver import BC, solve


def _bar():
    m = structured_box(10, 2, 2, 5, 1, 1).to_quadratic()
    X = m.nodes
    on = lambda ax, v: np.nonzero(np.isclose(X[:, ax], v))[0].tolist()  # noqa: E731
    sym = [BC("displacement", [], [0.0, None, None], nodes=on(0, 0)),
           BC("displacement", [], [None, 0.0, None], nodes=on(1, 0)),
           BC("displacement", [], [None, None, 0.0], nodes=on(2, 0))]
    return m, on, sym


def _em(m, st):
    return [st], np.zeros(m.n_elems, dtype=int)


def test_temperature_table_interpolation():
    mat = Material("t", 200000, 0.3, 7850, temp_table=[[20, 200000, 0.3], [220, 180000, 0.32]])
    E, nu = mat.E_nu_at(120)
    assert abs(E - 190000) < 1e-6 and abs(nu - 0.31) < 1e-9
    assert mat.E_nu_at(500)[0] == 180000   # 範囲外は端の値


def test_thermal_free_and_constrained():
    m, on, sym = _bar()
    mat = Material("t", 200000, 0.3, 7850, alpha=1.2e-5, ref_temp=22)
    st = mat.state(T=122)
    r = solve(m, mat, sym, elem_materials=_em(m, st), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 10 * 1.2e-5 * 100, rtol=1e-8)
    assert np.abs(r.stress).max() < 1e-6
    bcs = sym + [BC("displacement", [], [0.0, None, None], nodes=on(0, 10))]
    r = solve(m, mat, bcs, elem_materials=_em(m, st), log=lambda *a: None)
    assert np.allclose(r.stress[:, 0], -200000 * 1.2e-5 * 100, rtol=1e-6)


def test_orthotropic_rotation():
    m, on, sym = _bar()
    o = dict(Ex=100000, Ey=10000, Ez=10000, nuxy=0.3, nuyz=0.4, nuxz=0.3, Gxy=5000, Gyz=3000, Gxz=5000)
    mat = Material("o", 1, 0.3, 1000, elastic="orthotropic", ortho=o)
    bcs = sym + [BC("force", [1], [400.0, 0, 0])]   # 断面 4 mm² に 100 MPa
    r = solve(m, mat, bcs, elem_materials=_em(m, mat.state()), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 100 / 100000 * 10, rtol=1e-6)
    r = solve(m, mat, bcs, elem_materials=_em(m, mat.state(R=rotation_xyz(0, 0, 90))), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 100 / 10000 * 10, rtol=1e-6)


def _uniaxial(mat, d_list):
    m, on, sym = _bar()
    bcs = sym + [BC("displacement", [], [1.0, None, None], nodes=on(0, 10), step_factors=d_list)]
    s = AnalysisSettings(n_steps=len(d_list), step_end_times=list(range(1, len(d_list) + 1)), initial_substeps=4)
    rs = solve_static(m, mat, bcs, settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)
    return [float(r.stress[:, 0].mean()) for r in rs]


def test_bilinear_isotropic():
    mat = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    H = 200000 * 2000 / 198000
    ep = (0.005 - 250 / 200000) / (1 + H / 200000)
    assert abs(_uniaxial(mat, [0.05])[0] - (250 + H * ep)) < 1e-6


def test_kinematic_bauschinger():
    """移動硬化: +0.5% まで引張 → 除荷 → 圧縮で、逆降伏は σmax − 2σy で起きる"""
    H = 200000 * 2000 / 198000
    smax = 250 + H * (0.005 - 250 / 200000) / (1 + H / 200000)
    kin = Material("k", 200000, 0.3, 7850, 250, plasticity="bilinear_kin", tangent_modulus=2000)
    iso = Material("i", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    # 引張後、弾性除荷で σ が smax - 2*250 まで下がるひずみ
    ep = (0.005 - smax / 200000)
    e_rev = ep + (smax - 2 * 250) / 200000 + 0.0002   # 逆降伏の少し手前
    sk = _uniaxial(kin, [0.05, e_rev * 10])
    si = _uniaxial(iso, [0.05, e_rev * 10])
    assert abs(sk[1] - si[1]) < 1e-6          # まだ弾性（両者同じ）
    e2 = -0.003
    sk = _uniaxial(kin, [0.05, e2 * 10])
    si = _uniaxial(iso, [0.05, e2 * 10])
    assert sk[1] > si[1] + 10                  # 移動硬化は早く逆降伏する（圧縮応力の絶対値が小さい）

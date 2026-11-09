"""六面体要素（Hex8 / Hex20、SOLID185/186 相当）"""
import numpy as np

from openmech.materials import Material
from openmech.mesh import structured_hex_box
from openmech.solver import BC, solve

MAT = Material("鋼", 200000, 0.3, 7850)
Q = dict(log=lambda *a: None)


def test_patch_uniform_tension():
    """ゆがんだメッシュ（内部の頂点をずらす。辺は直線）でも一様応力を厳密に表す（パッチテスト）"""
    from openmech.mesh import Mesh
    m8 = structured_hex_box(10, 4, 3, 3, 2, 2, quadratic=False)
    X = m8.nodes
    inner = np.nonzero((X[:, 0] > 1e-9) & (X[:, 0] < 10 - 1e-9) & (X[:, 1] > 1e-9) & (X[:, 1] < 4 - 1e-9)
                       & (X[:, 2] > 1e-9) & (X[:, 2] < 3 - 1e-9))[0]
    m8.nodes[inner] += np.random.default_rng(0).uniform(-0.3, 0.3, (len(inner), 3))
    for m in (m8, Mesh(m8.nodes.copy(), m8.elems.copy()).to_quadratic()):
        Xn = m.nodes
        e = 1e-3
        bnd = [n for n, (x, y, z) in enumerate(Xn)
               if min(x, y, z) < 1e-9 or x > 10 - 1e-9 or y > 4 - 1e-9 or z > 3 - 1e-9]
        bcs = [BC("displacement", [], [e * Xn[n, 0], -0.3 * e * Xn[n, 1], -0.3 * e * Xn[n, 2]], nodes=[n]) for n in bnd]
        r = solve(m, MAT, bcs, **Q)
        assert np.allclose(r.stress[:, 0], 200.0, rtol=1e-8)
        assert np.allclose(r.stress[:, 1:], 0.0, atol=1e-6)


def test_hex20_cantilever_bending():
    L, b, h, P = 100.0, 10.0, 10.0, 100.0
    m = structured_hex_box(L, b, h, 20, 2, 2)
    assert m.is_hex and m.has_quads and m.elems.shape[1] == 20
    r = solve(m, MAT, [BC("fixed", [0]), BC("force", [1], [0, 0, -P])], **Q)
    I = b * h ** 3 / 12
    G = 200000 / 2.6
    exact = P * L ** 3 / (3 * 200000 * I) + P * L / (5 / 6 * G * b * h)
    assert abs(-r.u[:, 2].min() / exact - 1) < 0.01
    assert abs(m.volume() - L * b * h) < 1e-6


def test_hex_modal():
    from openmech.modal import solve_modal
    L, b, h = 200.0, 10.0, 10.0
    m = structured_hex_box(L, b, h, 30, 2, 2)
    r = solve_modal(m, MAT, [BC("fixed", [0])], n_modes=2, **Q)
    I = b * h ** 3 / 12
    f1 = 1.875104 ** 2 / (2 * np.pi * L ** 2) * np.sqrt(200000 * I / (7850e-12 * b * h))
    assert abs(r.freqs[0] / f1 - 1) < 0.01


def test_hex_thermal_and_plastic():
    from openmech.materials import LIBRARY
    from openmech.nonlinear import AnalysisSettings, solve_static
    from openmech.thermal import solve_thermal
    St = LIBRARY["構造用鋼 (Structural Steel)"]
    m = structured_hex_box(100, 10, 10, 10, 1, 1)
    x = m.nodes[:, 0]
    r = solve_thermal(m, St, [BC("temperature", [0], [100.]), BC("convection", [1], [1000., 0.])], **Q)[-1]
    q_ = 100 / (0.1 / 60.5e-4 + 10)
    assert abs(r.T[np.isclose(x, 100)].mean() / (100 - q_ * 0.1 / 60.5e-4) - 1) < 1e-6
    pl = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=0.0)
    m2 = structured_hex_box(20, 4, 4, 4, 1, 1)
    rr = solve_static(m2, pl, [BC("symmetry", [0]), BC("symmetry", [2]), BC("symmetry", [4]),
                               BC("displacement", [1], [0.05, None, None])],
                      settings=AnalysisSettings(initial_substeps=5), elem_materials=([pl.state()], np.zeros(m2.n_elems, int)),
                      **Q)[-1]
    assert abs(np.median(rr.stress[:, 0]) / 250 - 1) < 0.01

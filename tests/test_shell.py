"""シェル要素（DKT＋Allman 膜）"""
import numpy as np

from openmech.materials import Material
from openmech.shell import solve_shell, structured_plate
from openmech.solver import BC

MAT = Material("鋼", 200000, 0.3, 7850)


def _D(t):
    return 200000 * t ** 3 / (12 * (1 - 0.3 ** 2))


def test_simply_supported_plate():
    a, t, q, n = 100.0, 2.0, 0.01, 16
    m = structured_plate(a, a, n, n, t)
    bcs = [BC("displacement", [1, 2, 3, 4], [None, None, 0.0]),
           BC("displacement", [], [0.0, 0.0, None], nodes=[0]),
           BC("displacement", [], [None, 0.0, None], nodes=[n * (n + 1)]),
           BC("pressure", [0], [q])]
    r = solve_shell(m, MAT, bcs, log=lambda *a: None)
    w = np.abs(r.u[:, 2]).max()
    assert abs(w / (0.00406 * q * a ** 4 / _D(t)) - 1) < 0.01     # Timoshenko
    c = int(np.argmin(np.linalg.norm(m.nodes - [50, 50, 0], axis=1)))
    sx = abs(r.extra["応力 σx（上面） [MPa]"][c])
    assert abs(sx / (6 * 0.0479 * q * a ** 2 / t ** 2) - 1) < 0.03


def test_clamped_plate():
    a, t, q = 100.0, 2.0, 0.01
    m = structured_plate(a, a, 24, 24, t)
    r = solve_shell(m, MAT, [BC("fixed", [1, 2, 3, 4]), BC("pressure", [0], [q])], log=lambda *a: None)
    w = np.abs(r.u[:, 2]).max()
    assert abs(w / (0.00126 * q * a ** 4 / _D(t)) - 1) < 0.015


def test_inplane_cantilever():
    """面内曲げ（Allman 膜）: 粗いメッシュでも梁理論に近い"""
    m = structured_plate(100, 20, 40, 8, 2.0)
    r = solve_shell(m, MAT, [BC("fixed", [1]), BC("force", [2], [0, 100.0, 0])], log=lambda *a: None)
    I = 2 * 20 ** 3 / 12
    d = 100 * 100 ** 3 / (3 * 200000 * I) + 100 * 100 / (200000 / 2.6 * 40 * 5 / 6)
    assert abs(r.u[:, 1].max() / d - 1) < 0.02


def test_uniform_tension():
    m = structured_plate(100, 20, 20, 4, 2.0)
    bcs = [BC("displacement", [1], [0.0, None, 0.0]), BC("displacement", [], [None, 0.0, None], nodes=[0]),
           BC("force", [2], [4000.0, 0, 0])]
    r = solve_shell(m, MAT, bcs, log=lambda *a: None)
    assert np.allclose(r.u[m.nodes[:, 0] == 100, 0], 0.05, rtol=0.02)

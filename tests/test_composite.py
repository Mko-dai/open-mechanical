"""複合材の積層シェル（古典積層理論）の検証"""
import numpy as np

from openmech.composite import Laminates, T_eps, abd, failure_index, ply_Q
from openmech.materials import Material
from openmech.shell import solve_shell, structured_plate
from openmech.solver import BC

CFRP = Material("CFRP", 135000, 0.3, 1600, elastic="orthotropic",
                ortho={"Ex": 135000, "Ey": 10000, "Ez": 10000, "nuxy": 0.3, "nuyz": 0.4, "nuxz": 0.3,
                       "Gxy": 5000, "Gyz": 3600, "Gxz": 5000},
                composite_strength={"Xt": 1500, "Xc": 1200, "Yt": 50, "Yc": 250, "S": 70})
STEEL = Material("鋼", 200000, 0.3, 7850)
Q = dict(log=lambda *a: None)


def lam(m, angles, tp=0.25, ref=(1, 0, 0), crit="max_stress"):
    return Laminates(np.zeros(m.n_elems, int),
                     [{"plies": [(CFRP.state(), tp, a) for a in angles], "ref": list(ref), "criterion": crit}])


def test_single_isotropic_ply_equals_homogeneous():
    m = structured_plate(100, 20, 20, 4, 2.0)
    bcs = [BC("fixed", [1]), BC("force", [2], [0, 50.0, -10.0])]
    r0 = solve_shell(m, STEEL, bcs, **Q)
    L = Laminates(np.zeros(m.n_elems, int), [{"plies": [(STEEL.state(), 2.0, 30.0)]}])
    r1 = solve_shell(m, STEEL, bcs, laminate=L, **Q)
    assert np.allclose(r0.u, r1.u, rtol=1e-9, atol=1e-12)


def test_crossply_tension_matches_clt():
    """[0/90]s の一様引張: 伸び = (A⁻¹)11 N L"""
    m = structured_plate(100, 20, 20, 4, 1.0)
    N = 100.0                                       # N/mm
    bcs = [BC("displacement", [1], [0.0, None, 0.0]), BC("displacement", [], [None, 0.0, None], nodes=[0]),
           BC("force", [2], [N * 20, 0, 0])]
    r = solve_shell(m, CFRP, bcs, laminate=lam(m, [0, 90, 90, 0]), **Q)
    A, B, D, *_ = abd([(ply_Q(CFRP.state()), 0.25, a) for a in (0, 90, 90, 0)])
    assert np.allclose(B, 0, atol=1e-6)
    exact = np.linalg.inv(A)[0, 0] * N * 100
    assert abs(np.median(r.u[m.nodes[:, 0] == 100, 0]) / exact - 1) < 0.005      # 角は力の配分で少し大きい


def test_crossply_cantilever_bending():
    """[0/90]s の細長い片持ち板の先端荷重: w = P L³ (D⁻¹)11 / (3 b)（細い梁の仮定 My = 0）"""
    Lx, b = 200.0, 10.0
    m = structured_plate(Lx, b, 40, 2, 1.0)
    P = 1.0
    r = solve_shell(m, CFRP, [BC("fixed", [1]), BC("force", [2], [0, 0, -P])], laminate=lam(m, [0, 90, 90, 0]), **Q)
    _, _, D, *_ = abd([(ply_Q(CFRP.state()), 0.25, a) for a in (0, 90, 90, 0)])
    exact = P * Lx ** 3 * np.linalg.inv(D)[0, 0] / (3 * b)
    assert abs(-r.u[:, 2].min() / exact - 1) < 0.02


def test_unsymmetric_laminate_couples_tension_and_bending():
    """[0/90] は B ≠ 0: 引張だけで曲がる（対称積層では曲がらない）"""
    m = structured_plate(100, 20, 20, 4, 0.5)
    bcs = [BC("fixed", [1]), BC("force", [2], [500.0, 0, 0])]
    w_sym = np.abs(solve_shell(m, CFRP, bcs, laminate=lam(m, [0, 90, 90, 0], 0.125), **Q).u[:, 2]).max()
    w_un = np.abs(solve_shell(m, CFRP, bcs, laminate=lam(m, [0, 90]), **Q).u[:, 2]).max()
    assert w_sym < 1e-9 and w_un > 1e-3


def test_angle_ply_rotation_and_failure():
    """θ の回転: Q̄(90°) は Q の 11 と 22 を入れ替えたもの。一様引張の 0° 層の破損指標 = σ1/Xt"""
    Qm = ply_Q(CFRP.state())
    T = T_eps(np.pi / 2)
    Q90 = T.T @ Qm @ T
    assert abs(Q90[0, 0] - Qm[1, 1]) < 1e-6 and abs(Q90[1, 1] - Qm[0, 0]) < 1e-6
    s = np.array([750.0, 0.0, 0.0])
    st = CFRP.composite_strength
    assert abs(failure_index(s, st, "max_stress") - 0.5) < 1e-12
    assert abs(failure_index(s, st, "tsai_hill") - 0.5) < 1e-12
    assert abs(failure_index(np.array([0.0, 0.0, 35.0]), st, "tsai_wu") - 0.5) < 1e-9
    # シェルで: [0]×4 の一様引張 σ1 = N/t
    m = structured_plate(100, 20, 10, 2, 1.0)
    N = 300.0
    bcs = [BC("displacement", [1], [0.0, None, 0.0]), BC("displacement", [], [None, 0.0, None], nodes=[0]),
           BC("force", [2], [N * 20, 0, 0])]
    r = solve_shell(m, CFRP, bcs, laminate=lam(m, [0, 0, 0, 0]), **Q)
    fi = r.extra["複合材 破損指標"]
    assert abs(np.median(fi) - (N / 1.0) / 1500) < 0.01 * (N / 1500)
    assert len(r.plies) == 4

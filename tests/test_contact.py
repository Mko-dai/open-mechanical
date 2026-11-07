"""接触・大変形の検証（Ansys の接触タイプと同じ挙動になること）"""
import numpy as np
import pytest

from openmech.contact import ST_SLIDING, ST_STICKING, detect_contacts
from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.nonlinear import AnalysisSettings, solve_static
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def stack(b1, b2):
    """2つのボディ（節点は共有しない）"""
    m1, m2 = b1.to_quadratic(), b2.to_quadratic()
    m = Mesh(np.vstack([m1.nodes, m2.nodes]), np.vstack([m1.elems, m2.elems + m1.n_nodes]))
    m.elem_body = np.r_[np.zeros(m1.n_elems, int), np.ones(m2.n_elems, int)]
    m.group_faces()
    return m


def face(m, axis, val, sign):
    for f in m.face_info():
        if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99:
            return f["id"]
    raise KeyError


def blocks():
    m = stack(structured_box(40, 40, 10, 4, 4, 2), structured_box(40, 40, 10, 4, 4, 2, origin=(0, 0, 10)))
    return m, detect_contacts(m)


def test_split_body_interfaces():
    m = structured_box(20, 10, 10, 4, 2, 2).to_quadratic()
    cx = m.nodes[m.elems[:, :4]].mean(axis=1)[:, 0]
    m.elem_body = (cx > 10).astype(int)
    s = m.split_body_interfaces()
    assert s.n_nodes > m.n_nodes
    assert len(np.intersect1d(np.unique(s.elems[s.elem_body == 0]), np.unique(s.elems[s.elem_body == 1]))) == 0
    s.group_faces()
    assert len(detect_contacts(s)) == 1


def test_bonded_equals_single_body():
    m = stack(structured_box(100, 20, 20, 10, 3, 3), structured_box(100, 20, 20, 10, 3, 3, origin=(100, 0, 0)))
    cs = detect_contacts(m)
    assert len(cs) == 1 and cs[0].type == "bonded"
    r = solve_static(m, S, [BC("fixed", [face(m, 0, 0, -1)]), BC("force", [face(m, 0, 200, 1)], [0, 0, -1000])],
                     cs, **quiet)[-1]
    m1 = structured_box(200, 20, 20, 20, 3, 3).to_quadratic()
    r1 = solve(m1, S, [BC("fixed", [0]), BC("force", [1], [0, 0, -1000])], **quiet)
    tip = -r.u[np.isclose(m.nodes[:, 0], 200), 2].mean()
    tip1 = -r1.u[np.isclose(m1.nodes[:, 0], 200), 2].mean()
    assert tip == pytest.approx(tip1, rel=1e-4)


def _press_bcs(m, p=10.0):
    ys = [f["id"] for f in m.face_info() if abs(f["centroid"][1]) < 1e-6]
    xs = [f["id"] for f in m.face_info() if abs(f["centroid"][0]) < 1e-6]
    return [BC("fixed", [face(m, 2, 0, -1)]), BC("displacement", xs, [0.0, None, None]),
            BC("displacement", ys, [None, 0.0, None]), BC("pressure", [face(m, 2, 20, 1)], [p])]


@pytest.mark.parametrize("form", ["augmented_lagrange", "penalty"])
def test_frictionless_contact_pressure(form):
    m, cs = blocks()
    for c in cs:
        c.type, c.formulation = "frictionless", form
    r = solve_static(m, S, _press_bcs(m), cs, **quiet)[-1]
    p = r.contact["接触圧 [MPa]"]
    assert np.nanmedian(p) == pytest.approx(10.0, rel=0.01)
    assert np.nanmax(r.contact["貫通量 [mm]"]) < 0.01


def test_frictionless_separation():
    m, cs = blocks()
    for c in cs:
        c.type = "frictionless"
    bcs = _press_bcs(m)[:3] + [BC("displacement", [face(m, 2, 20, 1)], [None, None, 0.01])]
    r = solve_static(m, S, bcs, cs, **quiet)[-1]
    assert np.nanmax(r.contact["接触圧 [MPa]"]) == 0.0
    assert np.nanmax(r.contact["隙間 [mm]"]) == pytest.approx(0.01, rel=1e-3)


def test_frictional_stick_and_slide():
    m, cs = blocks()
    for c in cs:
        c.type, c.mu = "frictional", 0.3
    ys = [f["id"] for f in m.face_info() if abs(f["centroid"][1]) < 1e-6]
    side = [f["id"] for f in m.face_info() if abs(f["centroid"][0] - 40) < 1e-6 and f["centroid"][2] > 10]
    base = [BC("fixed", [face(m, 2, 0, -1)]), BC("displacement", ys, [None, 0.0, None]),
            BC("pressure", [face(m, 2, 20, 1)], [10.0])]
    # 小さな横力 → 固着
    r = solve_static(m, S, base + [BC("force", side, [-1000.0, 0, 0])], cs,
                     settings=AnalysisSettings(initial_substeps=4, weak_springs=True), **quiet)[-1]
    st = r.contact["接触状態"]
    assert np.all(st[~np.isnan(st)] == ST_STICKING)
    # 強制的にずらす → 全面すべり、横方向の反力 = μN
    r = solve_static(m, S, base + [BC("displacement", side, [-0.2, None, None])], cs,
                     settings=AnalysisSettings(initial_substeps=5), **quiet)[-1]
    st = r.contact["接触状態"]
    assert np.all(st[~np.isnan(st)] == ST_SLIDING)
    assert -r.reactions[3][0] == pytest.approx(0.3 * 16000, rel=0.01)


def test_large_deflection_cantilever():
    """大変形の片持ち梁（エラスティカ）: PL²/EI = 1 で先端たわみ 0.3017 L（線形理論は 0.333 L）"""
    L, b, h = 200.0, 10.0, 10.0
    P = S.E * b * h**3 / 12 / L**2
    m = structured_box(L, b, h, 30, 2, 2).to_quadratic()
    r = solve_static(m, S, [BC("fixed", [0]), BC("force", [1], [0, 0, -P])],
                     settings=AnalysisSettings(large_deflection=True, initial_substeps=5), **quiet)[-1]
    tip = m.nodes[:, 0] > L - 1e-6
    assert -r.u[tip, 2].mean() / L == pytest.approx(0.3017, rel=0.01)
    assert -r.u[tip, 0].mean() / L == pytest.approx(0.0566, rel=0.03)


def test_multi_step_loads():
    """2ステップ: ステップ1で荷重 1倍、ステップ2で 2倍 → 線形なら変位も 2倍"""
    m = structured_box(100, 10, 10, 10, 2, 2).to_quadratic()
    bc = BC("force", [1], [0, 0, -100.0], step_factors=[1.0, 2.0])
    rs = solve_static(m, S, [BC("fixed", [0]), bc], settings=AnalysisSettings(n_steps=2, step_end_times=[1, 2]),
                      **quiet)
    assert len(rs) == 2
    assert rs[1].u[:, 2].min() == pytest.approx(2 * rs[0].u[:, 2].min(), rel=1e-9)

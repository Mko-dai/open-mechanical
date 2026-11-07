"""節点共有（トポロジ共有）"""
import numpy as np

from openmech.materials import Material
from openmech.mesh import structured_box
from openmech.solver import BC, solve
from openmech.parallel import set_threads


def _two_body_mesh():
    m = structured_box(100, 20, 10, 20, 4, 2).to_quadratic()
    c = m.nodes[m.elems[:, :4]].mean(1)
    m.elem_body = (c[:, 0] > 50).astype(int)
    return m


def test_shared_mesh_equals_single_body():
    mat = Material("鋼", 200000, 0.3, 7850, 250, 400)
    m = _two_body_mesh()
    ref = solve(m, mat, [BC("fixed", [0]), BC("force", [1], [0, 0, -100.0])], log=lambda *a: None)
    sh = m.split_body_interfaces({"*": 0})
    assert sh.n_nodes == m.n_nodes
    sep = m.split_body_interfaces({})
    assert sep.n_nodes > m.n_nodes
    parts = m.split_body_interfaces({0: 0, 1: 0})
    assert parts.n_nodes == m.n_nodes
    r = solve(sh, mat, [BC("fixed", [0]), BC("force", [1], [0, 0, -100.0])], log=lambda *a: None)
    assert np.allclose(r.u, ref.u)


def test_cores_same_result():
    mat = Material("鋼", 200000, 0.3, 7850, 250, 400)
    m = structured_box(100, 20, 10, 20, 4, 2).to_quadratic()
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -100.0])]
    set_threads(1)
    a = solve(m, mat, bcs, log=lambda *a: None)
    set_threads(0)
    b = solve(m, mat, bcs, log=lambda *a: None)
    assert np.allclose(a.u, b.u, rtol=1e-10, atol=1e-14)
    assert np.allclose(a.stress, b.stress, rtol=1e-9, atol=1e-9)


def test_straighten_bad_tet10():
    """曲がりすぎて裏返った 2 次要素は、中間節点を直線に戻して自動修正される"""
    from openmech.mesh import structured_box
    from openmech.geometry import straighten_bad_tet10, tet10_jacobian_ratio
    m = structured_box(10, 10, 10, 3, 3, 3).to_quadratic()
    e = m.elems[5]
    m.nodes[e[4]] += 0.45 * (m.nodes[e[1]] - m.nodes[e[0]]) + np.array([0, 0, 3.0])
    assert tet10_jacobian_ratio(m).min() < 0
    assert straighten_bad_tet10(m, log=lambda *a: None) > 0
    assert tet10_jacobian_ratio(m).min() > 0.9


def test_auto_fix_gmsh():
    """gmsh のメッシュで自動修正が動き、裏返った 2 次要素が残らない"""
    import pytest
    pytest.importorskip("gmsh")
    from openmech.geometry import build_mesh, mesh_quality, new_feature, tet10_jacobian_ratio
    spec = {"kind": "model", "features": [new_feature("cylinder", r=10.0, h=3.0),
                                          new_feature("cylinder", "cut", r=9.6, h=3.0)]}
    logs = []
    m = build_mesh(spec, size=2.5, log=logs.append, settings={"min_quality": 0.2})
    assert tet10_jacobian_ratio(m).min() > 0.0
    assert mesh_quality(m)["要素品質"]["最小"] > 0.0
    print("\n".join(logs))

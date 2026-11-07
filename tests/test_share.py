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

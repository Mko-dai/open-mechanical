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


def test_body_edit_keys():
    """コピー・2分割は新しいボディ番号を作る（_build_features と同じ規則）"""
    from openmech.geometry import feature_body_keys, new_feature
    f = [new_feature("box"), new_feature("cylinder"),
         new_feature("move", copy="コピー", dx=10.0, targets=[0]),
         new_feature("split", axis="X", offset=5.0),
         new_feature("box")]
    k = feature_body_keys(f)
    assert k == {0: 0, 1: 1, 2: 2, 3: 3, 4: 6}


def test_body_edits_gmsh():
    """移動・コピー・切断・引き伸ばし・丸み付け・面取りが gmsh で動く"""
    import pytest
    pytest.importorskip("gmsh")
    from openmech.geometry import build_mesh, new_feature
    base = new_feature("box", dx=100.0, dy=20.0, dz=10.0)
    cases = {
        "move": ([base, new_feature("move", dx=5.0, angle=90.0, axis="Z")], 1, 100 * 20 * 10),
        "copy": ([base, new_feature("move", copy="コピー", dz=10.0)], 2, 2 * 100 * 20 * 10),
        "mirror": ([base, new_feature("mirror", axis="X", offset=0.0)], 2, 2 * 100 * 20 * 10),
        "split": ([base, new_feature("split", axis="X", offset=40.0)], 2, 100 * 20 * 10),
        "split_keep": ([base, new_feature("split", axis="X", offset=40.0, keep="＋側を残す")], 1, 60 * 20 * 10),
        "stretch": ([base, new_feature("stretch", axis="X", offset=50.0, length=30.0)], 1, 130 * 20 * 10),
        "fillet": ([base, new_feature("fillet", r=2.0, edges="X 方向の辺")], 1, 100 * (20 * 10 - 4 * (4 - np.pi))),
        "chamfer": ([base, new_feature("chamfer", d=2.0, edges="X 方向の辺")], 1, 100 * (20 * 10 - 4 * 2.0)),
    }
    for name, (feats, nb, vol) in cases.items():
        m = build_mesh({"kind": "model", "features": feats}, size=4.0, order=1, log=lambda *a: None)
        assert len(m.body_ids()) == nb, name
        assert abs(m.volume() / vol - 1) < 0.01, (name, m.volume(), vol)

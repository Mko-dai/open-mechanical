"""六面体優勢メッシュ: 内側の六面体の格子＋つなぎのピラミッド＋すき間の四面体"""
import numpy as np
import pytest

from openmech.hexdom import build_core, clean_cells, core_cells, points_inside, to_quadratic
from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)


def _surface(b):
    u = np.unique(b.boundary_tris)
    mp = -np.ones(b.n_nodes, int)
    mp[u] = np.arange(len(u))
    return b.nodes[u], mp[b.boundary_tris]


def test_points_inside_and_clean():
    X, tr = _surface(structured_box(40, 30, 20, 10, 8, 5))
    P = np.array([[5, 5, 5], [50, 5, 5], [39.9, 29.9, 19.9], [-1, 1, 1], [20, 15, 25]], float)
    assert points_inside(P, X, tr).tolist() == [True, False, True, False, False]
    k = np.zeros((3, 3, 3), bool)
    k[0, 0, 0] = k[1, 1, 0] = k[2, 2, 2] = True          # 辺だけで触れ合う 2 つ → 片方を外す
    assert clean_cells(k).sum() == 2
    k = np.ones((3, 1, 1), bool)
    k[1] = False                                         # 空きセルの向かい合う 2 面だけが六面体 → 片方を外す
    assert clean_cells(k).sum() == 1


def test_core_is_conforming_and_passes_patch_test():
    """六面体の格子＋ピラミッド（へこんだ角では頂点を共有）: 体積・表面積が合い、一様な引張で応力が一様になる"""
    k = np.ones((4, 4, 2), bool)
    k[2:, 2:, :] = False                                 # L 字（へこんだ角がある）
    k = clean_cells(k)
    assert k.sum() == 24
    g = np.arange(5.0) * 2
    cn, ce, ct, nh = build_core(k, g, g, np.arange(3.0) * 2)
    n20, e20 = to_quadratic(cn, ce)
    m = Mesh(n20, e20)
    m.elem_body = np.zeros(len(e20), int)
    npyr = len(ce) - nh
    assert nh == 24 and m.volume() == pytest.approx(24 * 8 + npyr * 8 / 6)
    _, ar = m.tri_normals_areas()
    assert ar.sum() == pytest.approx(len(ct) * np.sqrt(2.0))         # 外に出ているのはピラミッドの側面だけ
    assert m.element_volumes().min() > 0
    # 箱の内側に置いた格子（六面体＋ピラミッド）だけで一様引張のパッチテスト
    X, tr = _surface(structured_box(40, 30, 20, 10, 8, 5))
    keep, xs, ys, zs = core_cells(X, tr, 4.0)
    assert keep.shape == (8, 5, 3) and keep.all()
    cn, ce, ct, nh = build_core(keep, xs, ys, zs)
    n20, e20 = to_quadratic(cn, ce)
    u = 1e-3 * n20 * np.array([1.0, -0.3, -0.3])                     # 一様な 1 軸引張の変位
    m = Mesh(n20, e20)
    m.elem_body = np.zeros(len(e20), int)
    from openmech.solver import material_D, nodal_stress
    st, _ = nodal_stress(m, u.ravel(), np.stack([material_D(S)]), elem_mat=np.zeros(m.n_elems, int))
    assert np.allclose(st[:, 0], 200.0, rtol=1e-6) and np.abs(st[:, 1:]).max() < 1e-6


def test_hexdom_mesh_matches_tet():
    """gmsh で切った六面体優勢メッシュ: 体積・表面積・面の数が四面体メッシュと同じで、片持ちはりのたわみも合う"""
    from openmech.geometry import build_mesh, gmsh_available, new_feature
    if not gmsh_available():
        pytest.skip("gmsh がありません")
    spec = {"kind": "model", "features": [new_feature("box", dx=100.0, dy=20.0, dz=20.0)]}
    out = {}
    for shape in ("tet", "hexdom"):
        m = build_mesh(spec, size=4.0, log=lambda *a: None, settings={"element_shape": shape})
        inf = m.face_info()
        lo = min(inf, key=lambda f: f["centroid"][0])["id"]
        hi = max(inf, key=lambda f: f["centroid"][0])["id"]
        r = solve(m, S, [BC("fixed", [lo]), BC("force", [hi], [0, 0, -1000.0])], log=lambda *a: None)
        _, ar = m.tri_normals_areas()
        out[shape] = (m, m.volume(), ar.sum(), m.n_faces, float(np.abs(r.u).max()))
    m = out["hexdom"][0]
    assert m.hexdom_counts["六面体"] > 150 and m.element_volumes().min() > 0
    assert out["hexdom"][1] == pytest.approx(40000.0, rel=1e-6) and out["hexdom"][2] == pytest.approx(8800.0, rel=1e-6)
    assert out["hexdom"][3] == out["tet"][3] == 6
    assert out["hexdom"][4] == pytest.approx(out["tet"][4], rel=0.01)

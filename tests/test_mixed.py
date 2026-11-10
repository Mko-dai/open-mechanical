"""ソリッド＋シェル: シェルの節点をソリッドの面に結合（並進と回転）"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import structured_box
from openmech.mixed import solve_mixed
from openmech.shell import structured_plate
from openmech.solver import BC

S = Material("steel", 200000.0, 0.0, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def face(m, axis, val, sign):
    return [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - val) < 1e-6
            and f["normal"][axis] * sign > 0.99]


def test_butt_joint_cantilever():
    """片持ちはりの根元半分をソリッド、先半分をシェルにして突き合わせでつなぐ。 先端のたわみ = PL³/(3EI)。
    回転を結ばないと（ピン結合だと）先半分が折れ曲がってしまうので、モーメントが伝わっていることの確認になる"""
    t, b, P = 2.0, 20.0, 10.0
    solid = structured_box(50, b, t, 25, 10, 2).to_quadratic()
    shell = structured_plate(50, b, 10, 4, t, origin=(50, 0, t / 2))
    rs, rh = solve_mixed(solid, shell, S, [BC("fixed", face(solid, 0, 0, -1))],
                         [BC("force", [shell.n_surf + 1], [0, 0, -P])], **quiet)
    I = b * t ** 3 / 12
    tip = -rh.u[np.isclose(shell.nodes[:, 0], 100.0), 2].mean()
    assert tip == pytest.approx(P * 100.0 ** 3 / (3 * 200000.0 * I), rel=0.02)
    # つなぎ目: たわみと傾きが、はりの式（x = 50）と合う
    xj = 50.0
    w_j = P * xj ** 2 * (3 * 100.0 - xj) / (6 * 200000.0 * I)
    th_j = P * xj * (2 * 100.0 - xj) / (2 * 200000.0 * I)
    edge = np.isclose(shell.nodes[:, 0], 50.0)
    assert -rh.u[edge, 2].mean() == pytest.approx(w_j, rel=0.02)
    assert rh.rotations[edge, 1].mean() == pytest.approx(th_j, rel=0.05)      # θy（+x に進むと下がる → 正）
    # 反力: ソリッドの固定面が荷重を全部受ける
    assert rs.reactions[0][2] == pytest.approx(P, rel=1e-6)
    # ソリッドの根元の曲げ応力 σ = 6 P L / (b t²)
    top = np.isclose(solid.nodes[:, 2], t) & np.isclose(solid.nodes[:, 0], 10.0)
    assert rs.stress[top, 0].mean() == pytest.approx(6 * P * 90.0 / (b * t ** 2), rel=0.05)


def test_tee_joint_plate_on_block():
    """ブロックの上面に立てた板（T 字）。 板の先端を横に押すと、根元が固定の片持ち板としてたわむ"""
    block = structured_box(40, 20, 20, 8, 4, 4).to_quadratic()
    stiff = Material("stiff", 2e7, 0.0, 7850.0, 1e9)
    plate = structured_plate(30, 20, 8, 4, 2.0)           # XY 平面（x: 0〜30, y: 0〜20）を立てる
    X = plate.nodes.copy()
    plate.nodes = np.c_[np.full(len(X), 20.0), X[:, 1], 20.0 + X[:, 0]]       # x → z（高さ）、面は x = 20
    P = 5.0
    rs, rh = solve_mixed(block, plate, S, [BC("fixed", face(block, 2, 0, -1))],
                         [BC("force", [plate.n_surf + 1], [P, 0, 0])],
                         solid_materials=([stiff], np.zeros(block.n_elems, int)), **quiet)
    I = 20.0 * 2.0 ** 3 / 12
    tip = rh.u[np.isclose(plate.nodes[:, 2], 50.0), 0].mean()
    assert tip == pytest.approx(P * 30.0 ** 3 / (3 * 200000.0 * I), rel=0.03)
    assert rs.reactions[0][0] == pytest.approx(-P, rel=1e-6)


def test_not_touching_error():
    solid = structured_box(10, 10, 10, 2, 2, 2).to_quadratic()
    shell = structured_plate(10, 10, 2, 2, 1.0, origin=(30, 0, 5))
    with pytest.raises(ValueError, match="乗っていません"):
        solve_mixed(solid, shell, S, [BC("fixed", [0])], [BC("force", [shell.n_surf + 1], [0, 0, -1])], **quiet)


def test_mixed_mesh_and_project_run():
    """MixedMesh: 面番号（ソリッドの面 → シェルの面 → シェルの辺）と、プロジェクトからの実行"""
    from openmech.geometry import is_mixed_model, new_feature, split_mixed
    from openmech.mixed import MixedMesh
    from openmech.project import Project
    t, b, P = 2.0, 20.0, 10.0
    solid = structured_box(50, b, t, 25, 10, 2).to_quadratic()
    solid.elem_body = np.zeros(solid.n_elems, int)
    solid.group_faces()
    shell = structured_plate(50, b, 10, 4, t, origin=(50, 0, t / 2))
    m = MixedMesh(solid, shell, 1)
    assert m.n_nodes == solid.n_nodes + shell.n_nodes and m.n_faces == solid.n_faces + 1 + 4
    assert list(m.body_ids()) == [0, 1] and m.is_shell_body(1) and not m.is_shell_body(0)
    info = m.face_info()
    assert [f["id"] for f in info] == list(range(m.n_faces))
    tip_edge = next(f["id"] for f in info if f.get("edge") and abs(f["centroid"][0] - 100) < 1e-6)
    root = next(f["id"] for f in info if not f.get("edge") and abs(f["centroid"][0]) < 1e-6)
    assert m.is_edge(tip_edge) and not m.is_edge(root)
    assert np.allclose(m.nodes[m.face_nodes(tip_edge)][:, 0], 100.0)
    assert len(m.boundary_tris) == len(m.face_ids) == len(m.tri_body()) == len(m.boundary_tris_full)
    p = Project()
    p.material = S
    p.bcs = [BC("fixed", [root]), BC("force", [tip_edge], [0, 0, -P])]
    r = p.run_all(m, log=lambda *a: None)[-1]
    I = b * t ** 3 / 12
    assert -r.u[:, 2].min() == pytest.approx(P * 100.0 ** 3 / (3 * 200000.0 * I), rel=0.02)
    assert r.u.shape == (m.n_nodes, 3) and r.stress.shape == (m.n_nodes, 6)
    assert r.reactions[0][2] == pytest.approx(P, rel=1e-6)
    assert np.isnan(r.extra["相当応力（上面） [MPa]（シェル）"][:solid.n_nodes]).all()
    # 形状の指定: シェルはソリッドのあと。 ボディ番号はソリッドの続き
    spec = {"kind": "model", "features": [new_feature("box"), new_feature("plate")]}
    assert is_mixed_model(spec)
    s_spec, h_spec, off = split_mixed(spec)
    assert len(s_spec["features"]) == 1 and len(h_spec["features"]) == 1 and off == 1
    from openmech.geometry import GeometryError
    with pytest.raises(GeometryError):
        split_mixed({"kind": "model", "features": [new_feature("plate"), new_feature("box")]})

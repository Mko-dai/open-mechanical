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


def test_mixed_modal_cantilever():
    """ソリッド＋シェルのモーダル: 片持ちはり（根元半分ソリッド・先半分シェル）の 1 次の曲げ振動数"""
    from openmech.mixed import MixedMesh
    from openmech.project import Project
    t, b = 2.0, 20.0
    solid = structured_box(50, b, t, 25, 10, 2).to_quadratic()
    solid.elem_body = np.zeros(solid.n_elems, int)
    solid.group_faces()
    shell = structured_plate(50, b, 10, 4, t, origin=(50, 0, t / 2))
    m = MixedMesh(solid, shell, 1)
    root = next(f["id"] for f in m.face_info() if not f.get("edge") and abs(f["centroid"][0]) < 1e-6)
    p = Project()
    p.material = S
    p.analysis_type = "modal"
    p.analysis_opts = {"n_modes": 3}
    p.bcs = [BC("fixed", [root])]
    rs = p.run_all(m, log=lambda *a: None)
    f1 = 0.5596 * np.sqrt(200000.0 * (b * t ** 3 / 12) / (7.85e-9 * b * t * 100.0 ** 4))
    assert len(rs) == 3 and rs[0].info["振動数 [Hz]"] == pytest.approx(f1, rel=0.02)
    assert np.abs(rs[0].u).max() == pytest.approx(1.0, rel=1e-6)


def test_mixed_with_two_solid_bodies_bonded():
    """ソリッドが 2 つのボディ（節点は共有しない）＋シェル: ソリッドどうしは接触（結合）でつなぐ。
    片持ちはりの先端のたわみと 1 次の振動数が、ソリッド 1 ボディのときと合う"""
    from openmech.mesh import Mesh
    from openmech.mixed import MixedMesh
    from openmech.project import Project
    t, b, P = 2.0, 20.0, 10.0

    def model(split):
        if split:
            a = structured_box(25, b, t, 12, 10, 2).to_quadratic()
            c = structured_box(25, b, t, 12, 10, 2, origin=(25, 0, 0)).to_quadratic()
            solid = Mesh(np.vstack([a.nodes, c.nodes]), np.vstack([a.elems, c.elems + a.n_nodes]))
            solid.elem_body = np.r_[np.zeros(a.n_elems, int), np.ones(c.n_elems, int)]
            solid.node_origin = np.arange(solid.n_nodes)
        else:
            solid = structured_box(50, b, t, 24, 10, 2).to_quadratic()
            solid.elem_body = np.zeros(solid.n_elems, int)
        solid.group_faces()
        shell = structured_plate(50, b, 10, 4, t, origin=(50, 0, t / 2))
        return MixedMesh(solid, shell, 2 if split else 1)

    out = {}
    for split in (True, False):
        m = model(split)
        info = m.face_info()
        tip = next(f["id"] for f in info if f.get("edge") and abs(f["centroid"][0] - 100) < 1e-6)
        root = next(f["id"] for f in info if not f.get("edge") and abs(f["centroid"][0]) < 1e-6
                    and abs(f["normal"][0]) > 0.99)
        p = Project()
        p.material = S
        p.bcs = [BC("fixed", [root]), BC("force", [tip], [0, 0, -P])]
        r = p.run_all(m, log=lambda *a: None)[-1]
        if split:
            assert len(p.contacts) == 1 and p.contacts[0].type == "bonded"
        p.analysis_type = "modal"
        p.analysis_opts = {"n_modes": 2}
        f1 = p.run_all(m, log=lambda *a: None)[0].info["振動数 [Hz]"]
        out[split] = (-r.u[:, 2].min(), f1)
    assert out[True][0] == pytest.approx(out[False][0], rel=0.01)
    assert out[True][0] == pytest.approx(P * 100.0 ** 3 / (3 * 200000.0 * b * t ** 3 / 12), rel=0.03)
    assert out[True][1] == pytest.approx(out[False][1], rel=0.01)
    # 分離のある接触は、静解析ではエラーで知らせる
    m = model(True)
    p = Project()
    p.material = S
    p.contact_default_type = "frictionless"
    p.bcs = [BC("fixed", [0])]
    with pytest.raises(ValueError, match="結合 と 分離なし"):
        p.run_all(m, log=lambda *a: None)


@pytest.mark.parametrize("root", ["solid", "shell"])
def test_mixed_plastic_cantilever(root):
    """ソリッド＋シェルの弾塑性: 片持ちはりの根元（ソリッド側 または シェル側）が降伏する。
    降伏より下はほぼ線形と同じ（角・つなぎ目の局所的な降伏の分だけ違う）、降伏荷重の 1.3 倍では塑性ひずみが出てたわみが線形より大きく、反力は荷重とつり合う"""
    t, b, L = 2.0, 20.0, 100.0
    pl = Material("p", 200000.0, 0.0, 7850.0, 250.0, plasticity="bilinear_iso", tangent_modulus=2000.0)
    solid = structured_box(50, b, t, 25, 10, 2).to_quadratic()
    shell = structured_plate(50, b, 10, 4, t, origin=(50, 0, t / 2))
    Py = 250.0 * b * t ** 2 / (6 * L)

    def em(mat):
        return dict(solid_materials=([mat.state()], np.zeros(solid.n_elems, int)),
                    shell_materials=([mat.state()], np.zeros(shell.n_elems, int)))

    def run(P, mat):
        if root == "solid":
            rs, rh = solve_mixed(solid, shell, mat, [BC("fixed", face(solid, 0, 0, -1))],
                                 [BC("force", [shell.n_surf + 1], [0, 0, -P])], **em(mat), **quiet)
            return -rh.u[np.isclose(shell.nodes[:, 0], 100.0), 2].mean(), rs, rh, rs.reactions[0][2]
        rs, rh = solve_mixed(solid, shell, mat, [BC("force", face(solid, 0, 0, -1), [0, 0, -P])],
                             [BC("fixed", [shell.n_surf + 1])], **em(mat), **quiet)
        return -rs.u[np.isclose(solid.nodes[:, 0], 0.0), 2].mean(), rs, rh, rh.reactions[0][2]
    w_el = run(0.6 * Py, S)[0]
    w, rs, rh, Rz = run(0.6 * Py, pl)
    assert w == pytest.approx(w_el, rel=3e-3)
    w13, rs, rh, Rz = run(1.3 * Py, pl)
    assert Rz == pytest.approx(1.3 * Py, rel=1e-3)
    assert w13 > 1.04 * w_el * 1.3 / 0.6
    peq = (rs if root == "solid" else rh).extra["相当塑性ひずみ"]
    assert np.nanmax(peq) > 1e-4                 # 根元が降伏している（つなぎ目のソリッドも局所的に降伏しうる）

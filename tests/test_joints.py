"""ジョイント（Ansys の Joint: 固定・回転・円筒・並進・球面など）の検証"""
import numpy as np

from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0)
Q = dict(log=lambda *a: None)


def face(m, axis, val, sign):
    for f in m.face_info():
        if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99:
            return f["id"]
    raise KeyError


def stack(b1, b2):
    m1, m2 = b1.to_quadratic(), b2.to_quadratic()
    m = Mesh(np.vstack([m1.nodes, m2.nodes]), np.vstack([m1.elems, m2.elems + m1.n_nodes]))
    m.elem_body = np.r_[np.zeros(m1.n_elems, int), np.ones(m2.n_elems, int)]
    m.group_faces()
    return m


def test_ground_fixed_joint_equals_fixed_support():
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    load = BC("force", [face(m, 0, 100, 1)], [0, 0, -100.0])
    r0 = solve(m, S, [BC("fixed", [face(m, 0, 0, -1)]), load], **Q)
    r1 = solve(m, S, [BC("joint", [face(m, 0, 0, -1)], [], options={"type": "fixed", "ground": True}), load], **Q)
    # 固定ジョイントは面を剛体で固定する（固定支持はポアソン収縮も止める）ので、たわみはほぼ同じ
    assert abs(r1.u[:, 2].min() / r0.u[:, 2].min() - 1) < 0.01
    jf = r1.joints[0]["力（全体）"]
    assert abs(jf[2] - 100.0) < 1e-6 * 100              # 可動側（はり）が受ける力 = 荷重と釣り合う +100
    assert abs(abs(r1.joints[0]["モーメント（全体）"][1]) - 100.0 * 100) < 1e-6 * 1e4


def test_ground_revolute_with_torsional_spring():
    """地面との回転ジョイント（z 軸）＋ねじりばね k: 先端の横荷重 P で回転 θ = P L / k"""
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    k, P, L = 1e7, 50.0, 100.0
    j = BC("joint", [face(m, 0, 0, -1)], [], point=[0, 5, 5],
           options={"type": "revolute", "ground": True, "axis": [0, 0, 1], "stiffness": [0, 0, 0, 0, 0, k]})
    r = solve(m, S, [j, BC("force", [face(m, 0, L, 1)], [0, P, 0])], **Q)
    th = r.joints[0]["相対回転（局所） [rad]"][2]
    assert abs(th / (P * L / k) - 1) < 1e-6
    assert abs(abs(r.joints[0]["力（全体）"][1]) - P) < 1e-6 * P
    assert abs(abs(r.joints[0]["モーメント（全体）"][2]) - P * L) < 1e-6 * P * L   # ねじりばねが受けるモーメント


def test_body_body_revolute_prescribed_rotation():
    """ボディ間の回転ジョイントに回転 θ を与えると、可動側のボディは軸まわりに剛体回転する（応力なし）"""
    m = stack(structured_box(10, 10, 10, 2, 2, 2, origin=(-10, 0, 0)), structured_box(40, 10, 10, 8, 2, 2))
    th = 1e-3
    p = np.array([0.0, 5.0, 5.0])
    j = BC("joint", [face(m, 0, 0, 1)], [], point=list(p),
           options={"type": "revolute", "faces_b": [face(m, 0, 0, -1)], "axis": [0, 0, 1],
                    "disp": [None, None, None, None, None, th]})
    r = solve(m, S, [BC("fixed", [face(m, 0, -10, -1)]), j], **Q)
    B = m.nodes[:, 0] > 1e-9
    exp = np.cross([0, 0, th], m.nodes[B] - p)
    assert np.allclose(r.u[B], exp, atol=1e-9)
    assert np.abs(r.u[m.nodes[:, 0] < -1e-9]).max() < 1e-12
    assert r.von_mises.max() < 1e-6


def test_translational_joint_slides():
    """並進ジョイント（x 方向）: 可動側の押し付け方向の力は伝わるが、x 方向には自由（ばね k で受ける）"""
    m = stack(structured_box(10, 10, 10, 2, 2, 2, origin=(0, 0, -10)), structured_box(10, 10, 10, 2, 2, 2))
    k, F = 1e5, 100.0
    j = BC("joint", [face(m, 2, 0, 1)], [], point=[5, 5, 0],
           options={"type": "translational", "faces_b": [face(m, 2, 0, -1)], "axis": [0, 0, 1], "axis_x": [1, 0, 0],
                    "stiffness": [k, 0, 0, 0, 0, 0]})
    r = solve(m, S, [BC("fixed", [face(m, 2, -10, -1)]), j, BC("force", [face(m, 2, 10, 1)], [F, 0, -F])], **Q)
    assert abs(r.joints[1]["相対変位（局所）"][0] / (F / k) - 1) < 1e-6
    assert abs(r.joints[1]["力（全体）"][2] - F) < 1e-6 * F

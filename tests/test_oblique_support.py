"""摩擦なし支持・対称領域を、斜めの平面や円筒面にも設定できる（法線方向だけ拘束）"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def test_frictionless_on_oblique_plane():
    """z 軸まわりに 30° 回した直方体。 斜めの端面を摩擦なし支持、反対の端面に圧力 → 一様な単軸圧縮"""
    L, p = 40.0, 50.0
    b = structured_box(L, 10, 10, 8, 2, 2).to_quadratic()
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = Mesh(b.nodes @ Rz.T, b.elems)
    m.group_faces()
    e1, e2 = Rz[:, 0], Rz[:, 1]
    info = m.face_info()
    end0 = [f["id"] for f in info if f["normal"] @ e1 < -0.99]
    end1 = [f["id"] for f in info if f["normal"] @ e1 > 0.99]
    side = [f["id"] for f in info if f["normal"] @ e2 < -0.99]
    bot = [f["id"] for f in info if f["normal"][2] < -0.99]
    bcs = [BC("frictionless", end0), BC("frictionless", side), BC("frictionless", bot), BC("pressure", end1, [p])]
    r = solve(m, S, bcs, **quiet)
    u1 = r.u @ e1
    x1 = m.nodes @ e1
    assert np.allclose(u1, -p / 200000.0 * x1, atol=2e-5 * p / 200000.0 * L + 1e-7)
    assert np.allclose(r.reactions[0], p * 100.0 * e1, rtol=1e-3, atol=1e-3 * p * 100)     # 支持が押し返す力
    s11 = np.einsum("i,nij,j->n", e1, r.stress[:, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(-1, 3, 3), e1)
    assert np.allclose(s11, -p, rtol=1e-3)


def test_frictionless_on_cylinder_lame():
    """厚肉円筒（1/4）の外周を摩擦なし支持（半径方向を拘束）、内面に圧力。 平面応力のラメの式と比べる"""
    a, bo, t, p, E, nu = 20.0, 40.0, 2.0, 100.0, 200000.0, 0.3
    bx = structured_box(bo - a, 1.0, t, 10, 16, 1).to_quadratic()
    r, th = a + bx.nodes[:, 0], np.radians(90.0 * bx.nodes[:, 1])
    m = Mesh(np.c_[r * np.cos(th), r * np.sin(th), bx.nodes[:, 2]], bx.elems)
    m.group_faces()
    R = np.hypot(m.nodes[:, 0], m.nodes[:, 1])

    def faces(test):
        return [f["id"] for f in m.face_info() if test(m.face_nodes(f["id"]))]
    inner = faces(lambda nd: np.allclose(R[nd], a, atol=1e-6))
    outer = faces(lambda nd: np.allclose(R[nd], bo, atol=1e-6))
    x0 = faces(lambda nd: np.allclose(m.nodes[nd, 0], 0, atol=1e-6))
    y0 = faces(lambda nd: np.allclose(m.nodes[nd, 1], 0, atol=1e-6))
    z0 = faces(lambda nd: np.allclose(m.nodes[nd, 2], 0, atol=1e-6))
    res = solve(m, S, [BC("frictionless", outer), BC("symmetry", x0 + y0), BC("frictionless", z0),
                       BC("pressure", inner, [p])], **quiet)
    A = -p * (1 - nu ** 2) / E / ((1 + nu) + (1 - nu) * bo ** 2 / a ** 2)
    ur = (res.u[:, 0] * m.nodes[:, 0] + res.u[:, 1] * m.nodes[:, 1]) / R
    assert np.abs(ur[np.isclose(R, bo)]).max() < 1e-4 * abs(A * (a - bo ** 2 / a))      # ペナルティなのでごく小さく残る
    assert ur[np.isclose(R, a)].mean() == pytest.approx(A * (a - bo ** 2 / a), rel=0.02)


def test_modal_with_oblique_symmetry():
    """斜めの対称面があってもモーダルが解ける（剛体モードが止まる）"""
    from openmech.modal import solve_modal
    b = structured_box(40, 10, 10, 6, 2, 2).to_quadratic()
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = Mesh(b.nodes @ Rz.T, b.elems)
    m.group_faces()
    info = m.face_info()
    end0 = [f["id"] for f in info if f["normal"] @ Rz[:, 0] < -0.99]
    side = [f["id"] for f in info if f["normal"] @ Rz[:, 1] < -0.99]
    bot = [f["id"] for f in info if f["normal"][2] < -0.99]
    f = solve_modal(m, S, [BC("frictionless", end0), BC("frictionless", side), BC("frictionless", bot)], 2,
                    **quiet).freqs
    # 3 面を垂直方向に止めた棒の 1 次（軸方向の伸び縮み、片側固定・片側自由）: f = c / (4 L)
    assert f[0] == pytest.approx(np.sqrt(200000.0 / 7.85e-9) / (4 * 40.0), rel=0.03)


def test_rotational_velocity_thin_ring():
    """回転する薄いリング: 周方向の応力 σ = ρ ω² r²、半径の増加 = ρ ω² r³ / E"""
    a, t, w = 100.0, 2.0, 500.0
    bx = structured_box(t, 1.0, t, 1, 24, 1).to_quadratic()
    r, th = a - t / 2 + bx.nodes[:, 0], np.radians(90.0 * bx.nodes[:, 1])
    m = Mesh(np.c_[r * np.cos(th), r * np.sin(th), bx.nodes[:, 2]], bx.elems)
    m.group_faces()

    def faces(test):
        return [f["id"] for f in m.face_info() if test(m.nodes[m.face_nodes(f["id"])])]
    x0 = faces(lambda X: np.allclose(X[:, 0], 0, atol=1e-6))
    y0 = faces(lambda X: np.allclose(X[:, 1], 0, atol=1e-6))
    z0 = faces(lambda X: np.allclose(X[:, 2], 0, atol=1e-6))
    res = solve(m, S, [BC("symmetry", x0 + y0), BC("frictionless", z0),
                       BC("rotation", [], [w], point=[0, 0, 0], options={"axis": [0, 0, 1]})], **quiet)
    R = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
    ur = (res.u[:, 0] * m.nodes[:, 0] + res.u[:, 1] * m.nodes[:, 1]) / R
    rho = 7.85e-9
    assert ur.mean() == pytest.approx(rho * w ** 2 * a ** 3 / 200000.0, rel=0.01)
    c, s_ = m.nodes[:, 0] / R, m.nodes[:, 1] / R
    sig_t = res.stress[:, 0] * s_ ** 2 + res.stress[:, 1] * c ** 2 - 2 * res.stress[:, 3] * s_ * c
    assert np.median(sig_t) == pytest.approx(rho * w ** 2 * a ** 2, rel=0.01)


def test_cylindrical_result_components():
    """円筒座標系の結果: 内圧を受ける厚肉円筒（外周を拘束）の σr・σθ がラメの式と合い、周方向の変位は 0"""
    from openmech.postproc import cylindrical_fields
    a, bo, t, p, E, nu = 20.0, 40.0, 2.0, 100.0, 200000.0, 0.3
    bx = structured_box(bo - a, 1.0, t, 10, 16, 1).to_quadratic()
    r, th = a + bx.nodes[:, 0], np.radians(90.0 * bx.nodes[:, 1])
    m = Mesh(np.c_[r * np.cos(th), r * np.sin(th), bx.nodes[:, 2]], bx.elems)
    m.group_faces()
    R = np.hypot(m.nodes[:, 0], m.nodes[:, 1])

    def faces(test):
        return [f["id"] for f in m.face_info() if test(m.face_nodes(f["id"]))]
    res = solve(m, S, [BC("frictionless", faces(lambda nd: np.allclose(R[nd], bo, atol=1e-6))),
                       BC("symmetry", faces(lambda nd: np.allclose(m.nodes[nd, 0], 0, atol=1e-6))
                          + faces(lambda nd: np.allclose(m.nodes[nd, 1], 0, atol=1e-6))),
                       BC("frictionless", faces(lambda nd: np.allclose(m.nodes[nd, 2], 0, atol=1e-6))),
                       BC("pressure", faces(lambda nd: np.allclose(R[nd], a, atol=1e-6)), [p])], **quiet)
    f = cylindrical_fields(m, res.u, res.stress, 2)
    A = -p * (1 - nu ** 2) / E / ((1 + nu) + (1 - nu) * bo ** 2 / a ** 2)
    B = -A * bo ** 2
    mid = np.isclose(R, 30.0)
    sr = E / (1 - nu ** 2) * ((1 + nu) * A - (1 - nu) * B / 30.0 ** 2)
    st = E / (1 - nu ** 2) * ((1 + nu) * A + (1 - nu) * B / 30.0 ** 2)
    assert f["応力 σr（半径） [MPa]"][mid].mean() == pytest.approx(sr, rel=0.03)
    assert f["応力 σθ（周） [MPa]"][mid].mean() == pytest.approx(st, rel=0.05, abs=1.0)
    assert np.abs(f["変位 周方向 [mm]"]).max() < 1e-3 * np.abs(f["変位 半径方向 [mm]"]).max()


def test_hydrostatic_pressure_resultant():
    """静水圧: 水槽の壁（幅 b、水深 h）が受ける合力 = ρ g b h² / 2、作用点は底から h/3"""
    b, H, t, h = 40.0, 100.0, 10.0, 80.0
    m = structured_box(t, b, H, 2, 4, 10).to_quadratic()        # 壁: x 方向が厚さ、z が高さ
    wet = [f["id"] for f in m.face_info() if f["normal"][0] > 0.99]
    base = [f["id"] for f in m.face_info() if f["normal"][2] < -0.99]
    r = solve(m, S, [BC("fixed", base), BC("hydrostatic", wet, [1000.0, 9806.65, h], options={"axis": [0, 0, 1]})],
              **quiet)
    Fx = 1000e-12 * 9806.65 * b * h ** 2 / 2
    assert r.reactions[0][0] == pytest.approx(Fx, rel=0.01)          # 支持が押し返す力（壁は −x に押される）
    # 曲げ応力 σ = M c / I（z = 10 の断面。それより上の水圧のモーメント）
    low = np.isclose(m.nodes[:, 2], 10.0) & np.isclose(m.nodes[:, 0], t)
    Mz10 = 1000e-12 * 9806.65 * b * (h - 10.0) ** 3 / 6              # z = 10 より上の水圧のモーメント
    assert r.stress[low, 2].mean() == pytest.approx(Mz10 * (t / 2) / (b * t ** 3 / 12), rel=0.05)

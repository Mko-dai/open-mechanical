"""モーダル・座屈・熱・リモート・ボルト・はり・疲労・振動応答・最適化・静電場/圧電の検証（理論解と比較）"""
import numpy as np
import pytest

from openmech.benchmarks import ellipse_ring, mapped_mesh
from openmech.materials import LIBRARY, Material
from openmech.mesh import structured_box
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
q = dict(log=lambda *a: None)


def face(m, axis, val, sign):
    for f in m.face_info():
        if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99:
            return f["id"]
    raise KeyError


def bar(L=200.0, b=10.0, h=10.0, nx=40, ny=2, nz=2):
    m = structured_box(L, b, h, nx, ny, nz).to_quadratic()
    m.group_faces()
    return m


def test_modal_cantilever_and_free():
    from openmech.modal import solve_modal
    m = bar()
    r = solve_modal(m, S, [BC("fixed", [face(m, 0, 0, -1)])], n_modes=4, **q)
    I = 10 * 10 ** 3 / 12
    f1 = 1.875104 ** 2 / (2 * np.pi) * np.sqrt(200000 * I / (7850e-12 * 100 * 200 ** 4))
    assert r.freqs[0] == pytest.approx(f1, rel=0.005)
    r2 = solve_modal(m, S, [], n_modes=8, **q)
    assert np.all(r2.freqs[:6] < 1e-3 * r2.freqs[6])         # 剛体モード 6 個
    assert r.eff_mass[:, 1:3].sum() / r.total_mass > 0.5


def test_buckling_euler():
    from openmech.modal import solve_buckling
    m = bar()
    r = solve_buckling(m, S, [BC("fixed", [face(m, 0, 0, -1)]), BC("force", [face(m, 0, 200, 1)], [-1.0, 0, 0])],
                       n_modes=1, **q)
    Pcr = np.pi ** 2 * 200000 * (10 ** 4 / 12) / (4 * 200 ** 2)
    assert r.load_factors[0] == pytest.approx(Pcr, rel=0.005)


def test_thermal_steady_and_transient():
    from openmech.thermal import ThermalSettings, solve_thermal, thermal_stress
    St = LIBRARY["構造用鋼 (Structural Steel)"]
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    m.group_faces()
    a, b = face(m, 0, 0, -1), face(m, 0, 100, 1)
    x = m.nodes[:, 0]
    r = solve_thermal(m, St, [BC("temperature", [a], [100.]), BC("convection", [b], [1000., 0.])], **q)[-1]
    Q = 100 / (0.1 / 60.5e-4 + 10)
    assert r.T[np.isclose(x, 100)].mean() == pytest.approx(100 - Q * 0.1 / 60.5e-4, rel=1e-6)
    r = solve_thermal(m, St, [BC("temperature", [a], [0.]), BC("temperature", [b], [0.]),
                              BC("heat_generation", [], [1e7])], **q)[-1]
    assert r.T.max() == pytest.approx(1e7 * 0.1 ** 2 / (8 * 60.5), rel=1e-4)
    r = solve_thermal(m, St, [BC("temperature", [a], [500.]), BC("radiation", [b], [0.8, 20.])], **q)[-1]
    Tb = r.T[np.isclose(x, 100)].mean()
    assert 60.5 * (500 - Tb) / 0.1 == pytest.approx(0.8 * 5.670374419e-8 * ((Tb + 273.15) ** 4 - 293.15 ** 4), rel=1e-6)
    rs = solve_thermal(m, St, [BC("temperature", [a], [100.])],
                       ThermalSettings(transient=True, init_temp=0., end_time=100., n_steps=100), **q)
    al = 60.5 / (7850 * 434)
    th = 100 * (1 - sum(4 / ((2 * n + 1) * np.pi) * np.sin((2 * n + 1) * np.pi * 0.05 / 0.2)
                        * np.exp(-al * ((2 * n + 1) * np.pi / 0.2) ** 2 * 100) for n in range(200)))
    assert rs[-1].T[np.isclose(x, 50)].mean() == pytest.approx(th, rel=0.01)
    y0, z0 = face(m, 1, 0, -1), face(m, 2, 0, -1)
    rr = thermal_stress(m, St, [BC("frictionless", [a]), BC("frictionless", [b]), BC("frictionless", [y0]),
                                BC("frictionless", [z0])], np.full(m.n_nodes, 122.), **q)
    assert rr.stress[:, 0].mean() == pytest.approx(-200000 * 1.2e-5 * 100, rel=1e-6)


@pytest.mark.parametrize("beh", ["rigid", "deformable"])
def test_remote_force(beh):
    m = bar(nx=40, ny=3, nz=3)
    r = solve(m, S, [BC("fixed", [face(m, 0, 0, -1)]),
                     BC("remote_force", [face(m, 0, 200, 1)], [0, 0, -100, 0, 0, 0], point=[250, 5, 5],
                        options={"behavior": beh})], **q)
    E, I = 200000., 10 * 10 ** 3 / 12
    exp = -100 * 200 ** 3 / (3 * E * I) - 100 * 50 * 200 ** 2 / (2 * E * I)
    assert r.u[np.isclose(m.nodes[:, 0], 200), 2].mean() == pytest.approx(exp, rel=0.01)


def test_remote_displacement_and_supports():
    m = bar(nx=40, ny=3, nz=3)
    r = solve(m, S, [BC("fixed", [face(m, 0, 0, -1)]),
                     BC("remote_displacement", [face(m, 0, 200, 1)], [None, None, -1.0, None, None, None],
                        options={"behavior": "rigid"})], **q)
    assert r.u[np.isclose(m.nodes[:, 0], 200), 2].mean() == pytest.approx(-1.0, abs=1e-9)
    assert r.reactions[1][2] == pytest.approx(-3 * 200000 * (10 ** 4 / 12) / 200 ** 3, rel=0.01)
    r = solve(m, S, [BC("elastic_support", [face(m, 2, 0, -1)], [10.0]), BC("pressure", [face(m, 2, 10, 1)], [1.0])], **q)
    assert r.u[:, 2].mean() == pytest.approx(-0.1, rel=1e-3)


def test_cylindrical_support_and_bearing():
    a, b = 50., 80.
    m, P = mapped_mesh(ellipse_ring(a, a, b, b, 0, 5), 6, 16, 1)
    r = solve(m, S, [BC("pressure", [0], [10.]), BC("cylindrical_support", [1], [0.0, None, None]),
                     BC("frictionless", [2]), BC("frictionless", [3]), BC("displacement", [4], [None, None, 0.0])], **q)
    nu, E = 0.3, 200000.
    B = -10 / (-(1 + nu) / ((1 - nu) * b ** 2) - 1 / a ** 2)
    A = -(1 + nu) * B / ((1 - nu) * b ** 2)
    ua = (1 / E) * ((1 - nu) * A * a + (1 + nu) * B / a)
    assert np.linalg.norm(r.u[np.isclose(P[:, 0], 0)][:, :2], axis=1).mean() == pytest.approx(ua, rel=0.005)
    r = solve(m, S, [BC("bearing_load", [0], [100.0, 100.0, 0]), BC("fixed", [1])], **q)
    assert r.reactions[1][:2] == pytest.approx([-100, -100], rel=1e-6)


def test_point_mass_modal():
    from openmech.modal import solve_modal
    m = bar()
    r = solve_modal(m, S, [BC("fixed", [face(m, 0, 0, -1)]),
                           BC("point_mass", [face(m, 0, 200, 1)], [1.0, 0, 0, 0], point=[200, 5, 5],
                              options={"behavior": "rigid"})], n_modes=2, **q)
    k = 3 * 200000 * (10 ** 4 / 12) / 200 ** 3
    f = np.sqrt(k * 1e3 / (1.0 + 0.2357 * 0.157)) / (2 * np.pi)
    assert r.freqs[0] == pytest.approx(f, rel=0.01)


def test_bolt_pretension():
    from openmech.bolt import solve_pretension
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    m.group_faces()
    m.elem_body = np.zeros(m.n_elems, int)
    a, b = face(m, 0, 0, -1), face(m, 0, 100, 1)
    bcs = [BC("fixed", [a]), BC("fixed", [b]),
           BC("bolt_pretension", [], [1000.0], options={"body": 0, "axis": [1, 0, 0]}, point=[50.5, 5, 5])]
    r1, r2 = solve_pretension(m, S, bcs, lambda bb, em: solve(m, S, bb, elem_materials=em, **q), **q)
    assert r1.info["ボルト 1 の力 [N]"] == pytest.approx(1000, rel=1e-3)
    assert r1.stress[m.nodes[:, 0] < 40, 0].mean() == pytest.approx(10.0, rel=0.01)


def test_beam_frame():
    from openmech.beam import Frame
    mats = {"st": Material("st", 200000, 0.3, 7850)}
    L = 1000.
    md = {"nodes": [[0, 0, 0], [L, 0, 0]], "members": [{"n": [0, 1], "section": "r", "material": "st", "div": 10}],
          "sections": {"r": {"type": "rect", "b": 20, "h": 40}}, "supports": [{"node": 0, "dof": [1] * 6}],
          "loads": [{"node": 1, "F": [0, 0, -1000, 0, 0, 0]}], "shear_deformation": False}
    r = Frame(md, mats).solve(**q)
    I = 20 * 40 ** 3 / 12
    assert r.u[1, 2] == pytest.approx(-1000 * L ** 3 / (3 * 200000 * I), rel=1e-9)
    assert r.stress.max() == pytest.approx(1000 * L * 20 / I, rel=1e-9)
    md["loads"] = []
    m = Frame(md, mats).modal(2, **q)
    f1 = 1.875104 ** 2 / (2 * np.pi) * np.sqrt(200000 * I / (7850e-12 * 800 * L ** 4))
    assert m.freqs[1] == pytest.approx(f1, rel=1e-4)


def test_fatigue_and_user_result():
    from openmech.postproc import fatigue, path_values, user_result
    St = LIBRARY["構造用鋼 (Structural Steel)"]
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    m.group_faces()
    r = solve(m, St, [BC("fixed", [face(m, 0, 0, -1)]), BC("force", [face(m, 0, 100, 1)], [20000., 0, 0])], **q)
    f = fatigue(r.stress, St, loading="zero", mean_corr="goodman", design_life=1e6)
    assert np.median(f["応力振幅（補正後） [MPa]"]) == pytest.approx(100 / (1 - 100 / 460), rel=1e-6)
    assert np.median(user_result("SX/2", r.fields(), m)) == pytest.approx(100, rel=1e-9)
    _, v = path_values(m, r.fields()["応力 σx [MPa]"], [1, 5, 5], [99, 5, 5], 11)
    assert v[5] == pytest.approx(200, rel=1e-3)


def test_harmonic_and_random():
    from openmech.modal import solve_modal
    from openmech.vibration import solve_harmonic, solve_random
    m = structured_box(200, 20, 10, 40, 4, 2).to_quadratic()
    m.group_faces()
    fix, tip = face(m, 0, 0, -1), face(m, 0, 200, 1)
    bcs = [BC("fixed", [fix]), BC("force", [tip], [0, 0, -10.])]
    md = solve_modal(m, S, bcs, n_modes=10, **q)
    st = solve(m, S, bcs, **q)
    n = int(np.argmax(np.isclose(m.nodes[:, 0], 200)))
    h = solve_harmonic(md, m, S, bcs, 1, 1000, 50, zeta=0.02, **q)
    amp = h.node_response(n, 2)
    assert amp[0] == pytest.approx(abs(st.u[n, 2]), rel=0.01)
    r = solve_random(md, m, S, [[10, 0.01], [3000, 0.01]], direction=(0, 0, 1), zeta=0.02, **q)
    i = int(np.argmax(np.abs(md.participation[:, 2])))
    fn = md.freqs[i]
    miles = abs(md.modes[i][n, 2] * md.participation[i, 2]) * np.sqrt(np.pi / 2 * fn * 25 * 0.01) * 9806.65 / (2 * np.pi * fn) ** 2
    assert r.u[n, 2] == pytest.approx(miles, rel=0.02)


def test_optimizer():
    from openmech.optimize import optimize
    f = lambda v: {"obj": (v["x"] - 2) ** 2 + (v["y"] + 1) ** 2, "c": v["x"] + v["y"]}  # noqa: E731
    r = optimize(f, [{"name": "x", "min": -5, "max": 5}, {"name": "y", "min": -5, "max": 5}],
                 {"output": "obj", "goal": "min"}, [{"output": "c", "min": 1.5}], n_iter=25, **q)
    assert r["best"]["目的"] < 0.2


def test_electrostatic_and_piezo():
    from openmech.electric import EPS0, solve_electrostatic, solve_piezo_modal, solve_piezo_static
    m = structured_box(20, 20, 1, 10, 10, 2).to_quadratic()
    m.group_faces()
    bot, top = face(m, 2, 0, -1), face(m, 2, 1, 1)
    r = solve_electrostatic(m, Material("d", 1000, 0.3, 1000, permittivity=4.0),
                            [BC("ground", [bot]), BC("voltage", [top], [100.])], **q)
    assert r.info["静電容量 [F]"] == pytest.approx(4 * EPS0 * 4e-4 / 1e-3, rel=1e-9)
    e = np.zeros((6, 3))
    e[2, 2] = 15.1
    P = Material("pz", 60000, 0.3, 7500, permittivity=1000, piezo={"e": e.tolist(), "eps_s": [1000] * 3})
    allf = [f["id"] for f in m.face_info()]
    s, _ = solve_piezo_static(m, P, [BC("fixed", allf), BC("ground", [bot]), BC("voltage", [top], [100.])], **q)
    assert s.stress[:, 2].mean() == pytest.approx(15.1e-3 * 100, rel=1e-6)
    b = structured_box(2, 2, 20, 2, 2, 20).to_quadratic()
    b.group_faces()
    bb, bt = face(b, 2, 0, -1), face(b, 2, 20, 1)
    bc = [BC("fixed", [bb]), BC("ground", [bb]), BC("voltage", [bt], [0.])]
    fs = solve_piezo_modal(b, P, bc, n_modes=10, **q)
    fo = solve_piezo_modal(b, P, bc, n_modes=10, open_circuit=True, **q)
    assert np.any(fo > fs * 1.05)                # 縦振動のモード: 開放（反共振）は短絡より高い


def test_nonlinear_with_supports_and_remote():
    from openmech.nonlinear import AnalysisSettings, solve_static
    m = bar(nx=40, ny=3, nz=3)
    r = solve_static(m, S, [BC("elastic_support", [face(m, 2, 0, -1)], [10.0]),
                            BC("pressure", [face(m, 2, 10, 1)], [1.0])], None,
                     AnalysisSettings(large_deflection=True), **q)[-1]
    assert r.u[:, 2].mean() == pytest.approx(-0.1, rel=1e-3)
    fix, tip = face(m, 0, 0, -1), face(m, 0, 200, 1)
    tipn = np.isclose(m.nodes[:, 0], 200)
    cases = [[BC("fixed", [fix]), BC("remote_force", [tip], [0, 0, -100, 0, 0, 0], point=[250, 5, 5],
                                     options={"behavior": "rigid"})],
             [BC("fixed", [fix]), BC("remote_displacement", [tip], [None, None, -1.0, None, None, None],
                                     options={"behavior": "deformable"})],
             [BC("fixed", [fix]), BC("spring", [tip], [50.0, 0], options={"behavior": "rigid", "ground": [200, 5, -300]}),
              BC("force", [tip], [0, 0, -100.])],
             [BC("fixed", [fix]), BC("point_mass", [tip], [1.0, 0, 0, 0], point=[250, 5, 5],
                                     options={"behavior": "rigid"}), BC("gravity", [], [0, 0, -9806.65])]]
    for bcs in cases:
        rl = solve(m, S, bcs, **q)
        rn = solve_static(m, S, bcs, None, AnalysisSettings(), **q)[-1]
        assert rn.u[tipn, 2].mean() == pytest.approx(rl.u[tipn, 2].mean(), rel=1e-6)
    rL = solve_static(m, S, cases[1], None, AnalysisSettings(large_deflection=True), **q)[-1]
    assert rL.u[tipn, 2].mean() == pytest.approx(-1.0, abs=1e-6)


def test_piezo_rotated_polarization():
    """分極を Z から X に回したボディ: X 方向に電圧 → X 方向の拘束応力 = e33 × E"""
    from openmech.electric import solve_piezo_static
    from openmech.materials import rotation_xyz
    e = np.zeros((6, 3))
    e[2, 2] = 15.1
    P = Material("pz", 60000, 0.3, 7500, permittivity=1000, piezo={"e": e.tolist(), "eps_s": [1000] * 3})
    m = structured_box(1, 20, 20, 2, 10, 10).to_quadratic()
    m.group_faces()
    st = P.state(None, rotation_xyz(0, 90, 0))      # 材料の 3 軸 → 全体の X
    allf = [f["id"] for f in m.face_info()]
    s, _ = solve_piezo_static(m, P, [BC("fixed", allf), BC("ground", [face(m, 0, 0, -1)]),
                                     BC("voltage", [face(m, 0, 1, 1)], [100.])],
                              elem_materials=([st], np.zeros(m.n_elems, int)), **q)
    assert abs(s.stress[:, 0].mean()) == pytest.approx(15.1e-3 * 100, rel=1e-6)


def test_adaptive_error_estimator():
    """ZZ 誤差推定: 片持ちはりで要素を細かくすると推定誤差が下がり、細分化点は固定端側に集まる"""
    from openmech.adaptive import _merge, element_errors, refine_points
    from openmech.solver import material_D
    D = material_D(S)[None]
    rels = []
    for nx in (10, 20, 40):
        m = bar(nx=nx)
        r = solve(m, S, [BC("fixed", [face(m, 0, 0, -1)]), BC("force", [face(m, 0, 200, 1)], (0, 0, -100.0))])
        err, U = element_errors(m, r.u.ravel(), D)
        rels.append(np.sqrt((err ** 2).sum() / U))
        assert err.shape == (m.n_elems,) and U > 0
    assert rels[0] > rels[1] > rels[2]
    pts = refine_points(m, err, frac=0.2)
    assert pts.shape[1] == 4 and len(pts) > 0
    assert np.mean(pts[:, 0]) < 100.0                       # 曲げモーメントが大きい固定端側
    merged = _merge(pts, pts * [1, 1, 1, 0.5])
    assert len(merged) >= len(pts) and merged[:, 3].min() <= pts[:, 3].min()


def test_lobpcg_block_matches_shift_invert():
    """LOBPCG（ブロック）の固有値がシフト逆反復と一致（前処理は分解した K）"""
    from openmech.dynamics import consistent_mass
    from openmech.modal import _eig, lobpcg_block
    from openmech.solver import assemble_stiffness, factorize, material_D
    m = bar(nx=20)
    K = assemble_stiffness(m, material_D(S))
    M = consistent_mass(m, np.full(m.n_elems, S.density_t_mm3))
    free = np.ones(K.shape[0], bool)
    fx = np.nonzero(m.nodes[:, 0] < 1e-9)[0]
    free[(3 * fx[:, None] + np.arange(3)).ravel()] = False
    Kf, Mf = K[free][:, free].tocsr(), M[free][:, free].tocsr()
    lam0, _ = _eig(Kf, Mf, 6, 0.0, lambda *a: None)
    solve = factorize(Kf.tocsc(), lambda *a: None)

    def prec(R):
        return np.column_stack([solve(R[:, j]) for j in range(R.shape[1])])
    lam, vec, it, hist = lobpcg_block(Kf, Mf, 6, prec, tol=1e-6)
    assert np.allclose(lam, lam0, rtol=1e-9) and it < 30
    assert np.allclose(vec.T @ (Mf @ vec), np.eye(6), atol=1e-8)


def test_response_spectrum_single_mode_and_cqc():
    """応答スペクトル: 一定の加速度スペクトル Sa のとき、モード i の応答 = Γ_i Sa / ω_i² φ_i。
    SRSS は各モードの 2 乗和の平方根、振動数の離れたモードでは CQC ≈ SRSS"""
    from openmech.modal import solve_modal
    from openmech.vibration import cqc_coeff, solve_spectrum
    m = bar(nx=20, b=10.0, h=16.0, nz=3)              # 断面を長方形にして y・z の曲げの振動数を離す
    md = solve_modal(m, S, [BC("fixed", [face(m, 0, 0, -1)])], n_modes=4, log=lambda *a: None)
    Sa = 2.0                                          # [G] 一定
    r = solve_spectrum(md, m, S, [[1, Sa], [1e5, Sa]], direction=[0, 0, 1], unit="G", method="srss",
                       log=lambda *a: None)
    w = 2 * np.pi * md.freqs
    amp = md.participation[:, 2] * Sa * 9806.65 / w ** 2
    exp = np.sqrt(np.sum((amp[:, None, None] * md.modes) ** 2, axis=0))
    assert np.allclose(r.u, exp, rtol=1e-10, atol=1e-14)
    rc = solve_spectrum(md, m, S, [[1, Sa], [1e5, Sa]], direction=[0, 0, 1], unit="G", method="cqc", zeta=0.02,
                        log=lambda *a: None)
    rho = cqc_coeff(w, 0.02)
    assert np.allclose(np.diag(rho), 1.0)
    # z 方向の曲げモードどうしは振動数が離れている → CQC と SRSS はほぼ同じ（正方形断面だと y・z が重なり CQC が大きくなる）
    assert abs(rc.u[:, 2].max() / r.u[:, 2].max() - 1) < 0.02
    assert r.extra["相当応力（応答スペクトル） [MPa]"].max() > 0

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


# ---- 高度な設定（Ansys の詳細と同じ項目）
def test_detect_default_type_and_mu():
    m = stack(structured_box(40, 40, 10, 4, 4, 2), structured_box(40, 40, 10, 4, 4, 2, origin=(0, 0, 10)))
    cs = detect_contacts(m, 0.0, "frictional", 0.35)
    assert cs[0].type == "frictional" and cs[0].mu == 0.35


@pytest.mark.parametrize("kw", [dict(stiffness_mode="absolute", stiffness_value=2e5),
                                dict(stiffness_mode="factor", stiffness_factor=5.0),
                                dict(pen_tol_mode="value", pen_tol=1e-4),
                                dict(update_stiffness="each_iteration", formulation="penalty"),
                                dict(detection="nodal_contact"),
                                dict(pinball_mode="radius", pinball=2.0),
                                dict(stabilization=0.01)])
def test_frictionless_advanced_settings(kw):
    m, cs = blocks()
    for c in cs:
        c.type = "frictionless"
        for k, v in kw.items():
            setattr(c, k, v)
    r = solve_static(m, S, _press_bcs(m), cs, **quiet)[-1]
    assert np.nanmedian(r.contact["接触圧 [MPa]"]) == pytest.approx(10.0, rel=0.02)
    if "pen_tol" in kw:
        assert np.nanmax(r.contact["貫通量 [mm]"]) <= 1e-4 * 1.01


def test_frictional_elastic_slip_tolerance():
    m, cs = blocks()
    for c in cs:
        c.type, c.mu, c.slip_tol_mode, c.slip_tol = "frictional", 0.3, "value", 1e-3
    ys = [f["id"] for f in m.face_info() if abs(f["centroid"][1]) < 1e-6]
    side = [f["id"] for f in m.face_info() if abs(f["centroid"][0] - 40) < 1e-6 and f["centroid"][2] > 10]
    base = [BC("fixed", [face(m, 2, 0, -1)]), BC("displacement", ys, [None, 0.0, None]),
            BC("pressure", [face(m, 2, 20, 1)], [10.0])]
    r = solve_static(m, S, base + [BC("displacement", side, [-0.2, None, None])], cs,
                     settings=AnalysisSettings(initial_substeps=5), **quiet)[-1]
    assert -r.reactions[3][0] == pytest.approx(0.3 * 16000, rel=0.01)


def test_contact_roundtrip_old_file():
    from openmech.contact import ContactRegion
    c = ContactRegion.from_dict({"type": "frictional", "stiffness_factor": 3.0, "pinball": 1.5})
    assert c.stiffness_mode == "factor" and c.pinball_mode == "radius"
    c2 = ContactRegion.from_dict(c.to_dict())
    assert c2 == c


def test_bolt_thread_flank_wedging():
    """ボルトねじの幾何補正: 軸方向に引くとフランクが軸力を伝え、半径方向に tan α 倍の力（くさび作用）が出る"""
    from openmech.contact import ContactRegion
    m = stack(structured_box(10, 10, 20, 2, 2, 4, origin=(-10, 0, 0)), structured_box(10, 10, 20, 2, 2, 4))
    ca, tb = face(m, 0, 0, 1), face(m, 0, 0, -1)
    reg = ContactRegion("ねじ", "frictionless", contact_faces=[ca], target_faces=[tb], contact_body=0, target_body=1,
                        thread={"point": [-1000, 5, 0], "axis": [0, 0, 1], "half_angle": 30.0})
    F = 1000.0
    bcs = [BC("fixed", [face(m, 0, 10, 1)]), BC("frictionless", [face(m, 0, -10, -1)]),
           BC("symmetry", [f["id"] for f in m.face_info() if abs(f["centroid"][1]) < 1e-9]),
           BC("force", [f["id"] for f in m.face_info() if abs(f["centroid"][2] - 20) < 1e-9 and f["centroid"][0] < 0],
              [0, 0, F])]
    r = solve_static(m, S, bcs, [reg], AnalysisSettings(initial_substeps=2), **quiet)[-1]
    Rx = r.reactions[1][0]
    assert abs(abs(Rx) / (F * np.tan(np.radians(30))) - 1) < 0.02
    # 軸方向は ねじで止まる（ねじがなければ摩擦なし接触では止まらない）
    assert np.abs(r.u[:, 2]).max() < 0.01


def test_bolt_thread_lead_angle_gives_torque():
    """リード角 λ のねじを軸方向に引くと、周方向に F·tan λ の力（ゆるめトルク）が出る"""
    from openmech.contact import ContactRegion
    m = stack(structured_box(10, 10, 20, 2, 2, 4, origin=(-10, 0, 0)), structured_box(10, 10, 20, 2, 2, 4))
    ca, tb = face(m, 0, 0, 1), face(m, 0, 0, -1)
    lam = np.radians(10.0)
    R = 1000.0
    pitch = np.pi * 2 * (R - 0.0) * np.tan(lam)
    reg = ContactRegion("ねじ", "frictionless", contact_faces=[ca], target_faces=[tb], contact_body=0, target_body=1,
                        thread={"point": [-R, 5, 0], "axis": [0, 0, 1], "half_angle": 30.0, "pitch": pitch})
    F = 1000.0
    y0 = [f["id"] for f in m.face_info() if abs(f["centroid"][1]) < 1e-9 and f["centroid"][0] < 0]
    bcs = [BC("fixed", [face(m, 0, 10, 1)]), BC("frictionless", [face(m, 0, -10, -1)]), BC("symmetry", y0),
           BC("force", [f["id"] for f in m.face_info() if abs(f["centroid"][2] - 20) < 1e-9 and f["centroid"][0] < 0],
              [0, 0, F])]
    r = solve_static(m, S, bcs, [reg], AnalysisSettings(initial_substeps=2), **quiet)[-1]
    Ry = abs(r.reactions[2][1])
    assert abs(Ry / (F * np.tan(lam)) - 1) < 0.05


def test_gasket_pressure_closure():
    """ガスケット: 面圧が表どおり（閉じ量から読んだ値 = 加えた圧力）で、閉じ量も表から逆算した値になる"""
    from openmech.contact import gasket_pressure
    gk = {"closure": [0.0, 0.05, 0.1, 0.2], "pressure": [0.0, 2.0, 8.0, 30.0]}
    m, cs = blocks()
    for c in cs:
        c.type, c.gasket = "gasket", dict(gk)
    for P in (5.0, 20.0):
        r = solve_static(m, S, _press_bcs(m, P), cs, **quiet)[-1]
        p = r.contact["接触圧 [MPa]"]
        clo = r.contact["貫通量 [mm]"]
        assert np.nanmedian(p) == pytest.approx(P, rel=0.01)
        c_exp = np.interp(P, gk["pressure"], gk["closure"])
        assert np.nanmedian(clo) == pytest.approx(c_exp, rel=0.02)
    # 除荷: 傾き unload の直線で戻り、0 で止まる（永久変形）
    gk["unload"] = 400.0
    p, k = gasket_pressure(gk, np.array([0.15, 0.12, 0.0]), np.array([0.2, 0.2, 0.2]))
    assert p[0] == pytest.approx(30 - 400 * 0.05) and k[0] == 400
    assert p[1] == 0.0 and p[2] == 0.0


def test_gasket_load_unload_set():
    """ガスケットの除荷: 押してから戻すと、除荷の傾きで戻り永久変形が残る"""
    gk = {"closure": [0.0, 0.1, 0.2], "pressure": [0.0, 5.0, 30.0], "unload": 1000.0}
    m, cs = blocks()
    for c in cs:
        c.type, c.gasket = "gasket", dict(gk)
    bcs = _press_bcs(m, 1.0)
    bcs[-1].step_factors = [20.0, 10.0]
    st = AnalysisSettings(n_steps=2, step_end_times=[1.0, 2.0])
    rs = solve_static(m, S, bcs, cs, settings=st, **quiet)
    c1 = np.nanmedian(rs[0].contact["貫通量 [mm]"])
    c2 = np.nanmedian(rs[-1].contact["貫通量 [mm]"])
    assert c1 == pytest.approx(0.1 + 0.1 * 15 / 25, rel=0.02)
    assert c2 == pytest.approx(c1 - 10.0 / 1000.0, rel=0.02)


def _slider(sliding, mu, d=30.0, nsub=6):
    """長い台（60×10×10）の上を、小さなブロック（10×10×10）が押し付けられたまま d だけすべる"""
    from openmech.contact import ContactRegion
    m = stack(structured_box(60, 10, 10, 12, 2, 2), structured_box(10, 10, 10, 2, 2, 2, origin=(5, 0, 10)))
    top = face(m, 2, 20, 1)
    c = ContactRegion(type="frictional" if mu else "frictionless", mu=mu, contact_faces=[face(m, 2, 10, 1)],
                      target_faces=[face(m, 2, 10, -1)], sliding=sliding, auto=False)
    bcs = [BC("fixed", [face(m, 2, 0, -1)]), BC("pressure", [top], [10.0]), BC("displacement", [top], [d, 0.0, None])]
    bcs[1].step_factors, bcs[2].step_factors = [1.0, 1.0], [0.0, 1.0]
    st = AnalysisSettings(n_steps=2, step_end_times=[1.0, 2.0], initial_substeps=nsub, max_substeps=200)
    r = solve_static(m, S, bcs, [c], st, **quiet)[-1]
    p = np.nan_to_num(r.contact["接触圧 [MPa]"])
    xc = float((m.nodes[:, 0] * p).sum() / p.sum())
    return r, xc


@pytest.mark.parametrize("mu", [0.0, 0.2])
def test_finite_sliding_contact_patch_follows(mu):
    """有限すべり: 接触する範囲がブロックについて動く。面圧の重心 = ブロックの中心 + μH（摩擦の偶力の分）、
    摩擦力 = μN。 微小すべりでは接触する範囲が最初の位置に残る"""
    r, xc = _slider("finite", mu)
    assert xc == pytest.approx(10.0 + 30.0 + mu * 10.0, abs=0.3)
    assert r.reactions[0][2] == pytest.approx(1000.0, rel=1e-3)
    assert r.reactions[0][0] == pytest.approx(-mu * 1000.0, abs=1.0)
    _r0, xc0 = _slider("small", mu)
    assert xc0 < 15.0


def test_contact_face_wider_than_target():
    """接触面がターゲットより広いとき、ターゲットの外の節点は接触しない（面圧 = 荷重 / ターゲットの面積）"""
    from openmech.contact import ContactRegion
    m = stack(structured_box(60, 10, 10, 12, 2, 2), structured_box(10, 10, 10, 2, 2, 2, origin=(25, 0, 10)))
    c = ContactRegion(type="frictionless", contact_faces=[face(m, 2, 10, 1)], target_faces=[face(m, 2, 10, -1)],
                      auto=False)
    xs = [f["id"] for f in m.face_info() if abs(f["centroid"][0] - 25) < 1e-6 or abs(f["centroid"][0] - 35) < 1e-6]
    ys = [f["id"] for f in m.face_info() if f["centroid"][2] > 10 and abs(f["normal"][1]) > 0.99]
    r = solve_static(m, S, [BC("fixed", [face(m, 2, 0, -1)]), BC("pressure", [face(m, 2, 20, 1)], [10.0]),
                            BC("displacement", xs, [0.0, None, None]), BC("displacement", ys, [None, 0.0, None])],
                     [c], **quiet)[-1]
    p = r.contact["接触圧 [MPa]"]
    x = m.nodes[:, 0]
    assert np.nanmax(np.nan_to_num(p)[(x < 22) | (x > 38)]) == 0.0
    inside = (x > 26) & (x < 34) & np.isfinite(p)
    assert np.median(p[inside]) == pytest.approx(10.0, rel=0.15)

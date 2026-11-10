"""過渡構造（陰解法）の非線形対応: 静解析と同じ解き方に慣性力を加える（接触・塑性・六面体）"""
import numpy as np
import pytest

from openmech.contact import detect_contacts
from openmech.dynamics import DynamicSettings, solve_implicit
from openmech.materials import Material
from openmech.mesh import Mesh, structured_box, structured_hex_box
from openmech.nonlinear import AnalysisSettings, solve_static
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def face(m, axis, val, sign):
    return [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - val) < 1e-6
            and f["normal"][axis] * sign > 0.99]


def test_matches_linear_implicit():
    """線形のときは、これまでの陰解法（Newmark 平均加速度法）と同じ答えになる"""
    m = structured_box(100, 10, 10, 8, 2, 2).to_quadratic()
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -100.0])]
    ds = DynamicSettings(end_time=2e-3, dt=2e-5, n_frames=10)
    a = solve_implicit(m, S, bcs, ds, **quiet)
    b = solve_static(m, S, bcs, dynamic=ds, **quiet)
    assert len(a) == len(b) == 10
    for ra, rb in zip(a, b):
        assert rb.time == pytest.approx(ra.time)
        assert np.abs(ra.u - rb.u).max() < 1e-9
    assert "速度 [mm/s]" in b[-1].extra and "運動エネルギー [mJ]" in b[-1].history


def test_hex_step_load_doubles_static():
    """六面体のはりに一定の荷重を急にかけると、最大たわみは静解析の 2 倍（減衰なし）"""
    m = structured_hex_box(100, 10, 10, 8, 2, 2)
    bcs = [BC("fixed", face(m, 0, 0, -1)), BC("force", face(m, 0, 100, 1), [0, 0, -100.0])]
    st = -solve(m, S, bcs, **quiet).u[:, 2].min()
    f1 = 0.5596 * np.sqrt(200000.0 * (10 ** 4 / 12) / (7.85e-9 * 100 * 100 ** 4))     # 1 次の固有振動数 [Hz]
    ds = DynamicSettings(end_time=1.2 / f1, dt=1 / f1 / 80, n_frames=48)
    rs = solve_static(m, S, bcs, dynamic=ds, **quiet)
    peak = max(-r.u[:, 2].min() for r in rs)
    assert peak == pytest.approx(2 * st, rel=0.05)


def test_bar_impact_on_stiff_wall():
    """弾性棒が硬い壁に衝突: 接触時間 2L/c、接触力 ρ c v A、力積 2 m v、はね返りの速さ v（1 次元の波動理論）"""
    bar = Material("bar", 200000.0, 0.0, 7850.0, 250.0)
    wall = Material("wall", 2e8, 0.0, 7850.0, 250.0)
    m1 = structured_box(10, 10, 5, 2, 2, 1).to_quadratic()
    m2 = structured_box(10, 10, 50, 2, 2, 10, origin=(0, 0, 5.002)).to_quadratic()
    m = Mesh(np.vstack([m1.nodes, m2.nodes]), np.vstack([m1.elems, m2.elems + m1.n_nodes]))
    m.elem_body = np.r_[np.zeros(m1.n_elems, int), np.ones(m2.n_elems, int)]
    m.group_faces()
    cs = detect_contacts(m, tolerance=0.01)
    for c in cs:
        c.type, c.interface = "frictionless", "none"
    v, L, A = 1000.0, 50.0, 100.0
    c0 = np.sqrt(200000.0 / 7.85e-9)
    ds = DynamicSettings(end_time=4e-5, dt=2e-7, n_frames=8, body_velocity={1: [0, 0, -v]})
    rs = solve_static(m, bar, [BC("fixed", face(m, 2, 0, -1))], cs, AnalysisSettings(),
                      elem_materials=([wall, bar], m.elem_body), dynamic=ds, **quiet)
    h = rs[-1].history
    T, Fc = np.array(h["時間 [s]"]), np.array(h["接触力 [N]"])
    on = Fc > 0.05 * Fc.max()
    assert T[on][-1] - T[on][0] == pytest.approx(2 * L / c0, rel=0.12)
    assert np.median(Fc[on]) == pytest.approx(7.85e-9 * c0 * v * A, rel=0.05)
    assert np.trapezoid(Fc, T) == pytest.approx(2 * 7.85e-9 * A * L * v, rel=0.03)
    nb = np.unique(m.elems[m.elem_body == 1])
    assert rs[-1].extra["速度 Z [mm/s]"][nb].mean() == pytest.approx(v, rel=0.04)


def test_plastic_transient_leaves_permanent_set():
    """降伏を超える衝撃荷重: 塑性ひずみが出て、荷重がなくなっても変形が残る"""
    mat = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
    mat.plasticity, mat.tangent_modulus = "bilinear_iso", 2000.0
    m = structured_box(100, 10, 10, 8, 2, 2).to_quadratic()
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -700.0])]
    ds = DynamicSettings(end_time=3e-3, dt=3e-5, n_frames=10, beta_damping=2e-5)
    rs = solve_static(m, mat, bcs, dynamic=ds, elem_materials=([mat.state()], np.zeros(m.n_elems, int)), **quiet)
    assert np.nanmax(rs[-1].extra["相当塑性ひずみ"]) > 1e-4
    el = solve_static(m, S, bcs, dynamic=ds, **quiet)
    assert max(-r.u[:, 2].min() for r in rs) > 1.02 * max(-r.u[:, 2].min() for r in el)

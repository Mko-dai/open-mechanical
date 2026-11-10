"""材料: 温度依存・直交異方性・熱ひずみ・塑性"""
import numpy as np
import pytest

from openmech.materials import Material, rotation_xyz
from openmech.mesh import structured_box
from openmech.nonlinear import AnalysisSettings, solve_static
from openmech.solver import BC, solve


def _bar():
    m = structured_box(10, 2, 2, 5, 1, 1).to_quadratic()
    X = m.nodes
    on = lambda ax, v: np.nonzero(np.isclose(X[:, ax], v))[0].tolist()  # noqa: E731
    sym = [BC("displacement", [], [0.0, None, None], nodes=on(0, 0)),
           BC("displacement", [], [None, 0.0, None], nodes=on(1, 0)),
           BC("displacement", [], [None, None, 0.0], nodes=on(2, 0))]
    return m, on, sym


def _em(m, st):
    return [st], np.zeros(m.n_elems, dtype=int)


def test_temperature_table_interpolation():
    mat = Material("t", 200000, 0.3, 7850, temp_table=[[20, 200000, 0.3], [220, 180000, 0.32]])
    E, nu = mat.E_nu_at(120)
    assert abs(E - 190000) < 1e-6 and abs(nu - 0.31) < 1e-9
    assert mat.E_nu_at(500)[0] == 180000   # 範囲外は端の値


def test_thermal_free_and_constrained():
    m, on, sym = _bar()
    mat = Material("t", 200000, 0.3, 7850, alpha=1.2e-5, ref_temp=22)
    st = mat.state(T=122)
    r = solve(m, mat, sym, elem_materials=_em(m, st), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 10 * 1.2e-5 * 100, rtol=1e-8)
    assert np.abs(r.stress).max() < 1e-6
    bcs = sym + [BC("displacement", [], [0.0, None, None], nodes=on(0, 10))]
    r = solve(m, mat, bcs, elem_materials=_em(m, st), log=lambda *a: None)
    assert np.allclose(r.stress[:, 0], -200000 * 1.2e-5 * 100, rtol=1e-6)


def test_orthotropic_rotation():
    m, on, sym = _bar()
    o = dict(Ex=100000, Ey=10000, Ez=10000, nuxy=0.3, nuyz=0.4, nuxz=0.3, Gxy=5000, Gyz=3000, Gxz=5000)
    mat = Material("o", 1, 0.3, 1000, elastic="orthotropic", ortho=o)
    bcs = sym + [BC("force", [1], [400.0, 0, 0])]   # 断面 4 mm² に 100 MPa
    r = solve(m, mat, bcs, elem_materials=_em(m, mat.state()), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 100 / 100000 * 10, rtol=1e-6)
    r = solve(m, mat, bcs, elem_materials=_em(m, mat.state(R=rotation_xyz(0, 0, 90))), log=lambda *a: None)
    assert np.allclose(r.u[on(0, 10), 0], 100 / 10000 * 10, rtol=1e-6)


def _uniaxial(mat, d_list):
    m, on, sym = _bar()
    bcs = sym + [BC("displacement", [], [1.0, None, None], nodes=on(0, 10), step_factors=d_list)]
    s = AnalysisSettings(n_steps=len(d_list), step_end_times=list(range(1, len(d_list) + 1)), initial_substeps=4)
    rs = solve_static(m, mat, bcs, settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)
    return [float(r.stress[:, 0].mean()) for r in rs]


def test_bilinear_isotropic():
    mat = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    H = 200000 * 2000 / 198000
    ep = (0.005 - 250 / 200000) / (1 + H / 200000)
    assert abs(_uniaxial(mat, [0.05])[0] - (250 + H * ep)) < 1e-6


def test_kinematic_bauschinger():
    """移動硬化: +0.5% まで引張 → 除荷 → 圧縮で、逆降伏は σmax − 2σy で起きる"""
    H = 200000 * 2000 / 198000
    smax = 250 + H * (0.005 - 250 / 200000) / (1 + H / 200000)
    kin = Material("k", 200000, 0.3, 7850, 250, plasticity="bilinear_kin", tangent_modulus=2000)
    iso = Material("i", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    # 引張後、弾性除荷で σ が smax - 2*250 まで下がるひずみ
    ep = (0.005 - smax / 200000)
    e_rev = ep + (smax - 2 * 250) / 200000 + 0.0002   # 逆降伏の少し手前
    sk = _uniaxial(kin, [0.05, e_rev * 10])
    si = _uniaxial(iso, [0.05, e_rev * 10])
    assert abs(sk[1] - si[1]) < 1e-6          # まだ弾性（両者同じ）
    e2 = -0.003
    sk = _uniaxial(kin, [0.05, e2 * 10])
    si = _uniaxial(iso, [0.05, e2 * 10])
    assert sk[1] > si[1] + 10                  # 移動硬化は早く逆降伏する（圧縮応力の絶対値が小さい）


def test_plastic_large_deflection_objective():
    """塑性＋大変形: 伸ばしたあと 90° 回転させても応力・塑性ひずみは同じ（客観性）"""
    m, on, sym = _bar()
    mat = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    X = m.nodes
    out = []
    for ang in (0, 90):
        th = np.radians(ang)
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        U = (X @ np.diag([1.01, 0.997, 0.997]).T) @ Rz.T - X
        bcs = [BC("displacement", [], list(U[i]), nodes=[i]) for i in range(len(X))]
        s = AnalysisSettings(initial_substeps=1, auto_time_stepping="off", large_deflection=True)
        r = solve_static(m, mat, bcs, settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
        out.append((r.von_mises.mean(), r.fields()["相当塑性ひずみ"].mean()))
    assert np.allclose(out[0], out[1], rtol=1e-8)
    assert out[0][1] > 0


def test_plastic_large_deflection_cantilever():
    """塑性＋大変形の片持ち梁: 収束し、幾何学的な剛性で微小変形の解よりたわみが小さい"""
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    mat = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -1500.0])]
    tips = []
    for LD in (False, True):
        s = AnalysisSettings(initial_substeps=10, max_substeps=100, large_deflection=LD)
        r = solve_static(m, mat, bcs, settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
        tips.append(-r.u[:, 2].min())
    assert tips[1] < 0.7 * tips[0]


@pytest.mark.parametrize("h", [{"model": "neo_hookean", "mu": 1.0},
                               {"model": "mooney_rivlin", "C10": 0.3, "C01": 0.1},
                               {"model": "yeoh", "C10": 0.4, "C20": -0.02, "C30": 0.004}])
def test_hyperelastic_uniaxial(h):
    """超弾性の単軸引張（ストレッチ 2 倍）: 公称応力が非圧縮の理論式と一致（d=0 → K=1000μ のほぼ非圧縮）"""
    from openmech.hyperelastic import uniaxial_nominal
    m, on, sym = _bar()
    mat = Material("ゴム", 1.0, 0.49, 1100, hyper=dict(h))
    lam = 2.0
    pull = BC("displacement", [], [10.0 * (lam - 1), None, None], nodes=on(0, 10))
    s = AnalysisSettings(initial_substeps=5, max_substeps=50)
    r = solve_static(m, mat, sym + [pull], settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
    P = r.reactions[3][0] / (2 * 2)
    assert abs(P / uniaxial_nominal(h, lam) - 1) < 0.01
    # 断面は λ^-1/2 に縮む（非圧縮）
    assert abs((2 + r.u[on(1, 2)[0], 1]) / 2 - lam ** -0.5) < 0.005


def test_norton_creep_constant_stress():
    """Norton クリープ: 一定応力の棒の伸び = L(σ/E + C1 σ^C2 t)"""
    m, on, sym = _bar()
    A, n, sig, T = 1e-12, 3.0, 100.0, 1000.0
    mat = Material("鋼", 200000, 0.3, 7850, creep={"C1": A, "C2": n})
    info = m.face_info()
    fx = [f["id"] for f in info if abs(f["centroid"][0] - 10) < 1e-9]
    F = BC("force", fx, [sig * 4.0, 0, 0])
    # ステップ 1（ごく短時間）で荷重をかけ、ステップ 2 で保持（Ansys のクリープ解析と同じ手順）
    s = AnalysisSettings(n_steps=2, step_end_times=[1e-6, T], initial_substeps=10, auto_time_stepping="on")
    r = solve_static(m, mat, sym + [F], settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
    ux = r.u[on(0, 10), 0].mean()
    exact = 10 * (sig / 200000 + A * sig ** n * T)
    assert abs(ux / exact - 1) < 1e-3
    assert abs(r.fields()["相当クリープひずみ"].mean() / (A * sig ** n * T) - 1) < 1e-3


def test_norton_creep_relaxation():
    """Norton クリープの応力緩和（伸びを保持）: σ(t) = [σ0^(1−n) + (n−1) E A t]^(1/(1−n))"""
    m, on, sym = _bar()
    A, n, E, T = 1e-12, 3.0, 200000.0, 1000.0
    mat = Material("鋼", E, 0.3, 7850, creep={"C1": A, "C2": n})
    eps0 = 1e-3
    pull = BC("displacement", [], [10 * eps0, None, None], nodes=on(0, 10))
    s = AnalysisSettings(n_steps=2, step_end_times=[1e-6, T], initial_substeps=200, max_substeps=400,
                         auto_time_stepping="on")
    r = solve_static(m, mat, sym + [pull], settings=s, elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
    s0 = E * eps0
    exact = (s0 ** (1 - n) + (n - 1) * E * A * T) ** (1 / (1 - n))
    assert abs(r.stress[:, 0].mean() / exact - 1) < 0.01
    assert r.stress[:, 0].mean() < 0.8 * s0


def test_hyperelastic_nearly_incompressible_bending_no_locking():
    """ほぼ非圧縮（K = 1000 μ）のゴムの片持ちはり: 体積成分の選択的低減積分で体積ロッキングしない
    （微小荷重なので、はり理論（せん断変形込み、E = 9Kμ/(3K+μ)）と比べる）"""
    m = structured_box(100, 10, 10, 20, 2, 2).to_quadratic()
    mat = Material("ゴム", 1.0, 0.49, 1100, hyper={"model": "neo_hookean", "mu": 1.0})
    P = 0.01
    r = solve_static(m, mat, [BC("fixed", [0]), BC("force", [1], [0, 0, -P])], settings=AnalysisSettings(),
                     elem_materials=_em(m, mat.state()), log=lambda *a: None)[-1]
    K, mu = 1000.0, 1.0
    E = 9 * K * mu / (3 * K + mu)
    exact = P * 100 ** 3 / (3 * E * 10 * 10 ** 3 / 12) + P * 100 / (5 / 6 * mu * 100)
    assert abs(-r.u[:, 2].min() / exact - 1) < 0.03


def test_chaboche_uniaxial_and_cyclic():
    """Chaboche（背応力 1 項、降伏応力一定）: 単軸の単調負荷 σ = k + (C/γ)(1 − e^{−γ εp})、
    ひずみ振幅一定の繰り返しで安定したループの頂点 σ = k + (C/γ) tanh(γ Δεp / 2)"""
    from openmech.mesh import structured_box
    from openmech.nonlinear import AnalysisSettings, solve_static
    from openmech.solver import BC
    E, k, C, g = 200000.0, 200.0, 50000.0, 500.0
    mat = Material("m", E, 0.3, 7850, k, plasticity="chaboche", chaboche=[[C, g]])
    m = structured_box(20, 4, 4, 2, 1, 1).to_quadratic()

    def run(factors):
        bc = BC("displacement", [1], [0.2, None, None])
        bc.step_factors = factors
        st = AnalysisSettings(n_steps=len(factors), step_end_times=list(range(1, len(factors) + 1)),
                              initial_substeps=40, max_substeps=400)
        return solve_static(m, mat, [BC("symmetry", [0]), BC("symmetry", [2]), BC("symmetry", [4]), bc], settings=st,
                            elem_materials=([mat.state()], np.zeros(m.n_elems, int)), log=lambda *a: None)
    eps = 0.2 / 20
    s = float(np.median(run([1.0])[-1].stress[:, 0]))
    assert s == pytest.approx(k + C / g * (1 - np.exp(-g * (eps - s / E))), rel=0.005)
    rs = run([1, -1, 1, -1, 1])
    s = float(np.median(rs[-1].stress[:, 0]))
    assert s == pytest.approx(k + C / g * np.tanh(g * (eps - s / E)), rel=0.005)
    assert float(np.median(rs[-2].stress[:, 0])) == pytest.approx(-s, rel=0.005)      # 引張と圧縮で対称なループ

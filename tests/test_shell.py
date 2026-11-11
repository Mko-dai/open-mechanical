"""シェル要素（DKT＋Allman 膜）"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.shell import solve_shell, structured_plate
from openmech.solver import BC

MAT = Material("鋼", 200000, 0.3, 7850)


def _D(t):
    return 200000 * t ** 3 / (12 * (1 - 0.3 ** 2))


def test_simply_supported_plate():
    a, t, q, n = 100.0, 2.0, 0.01, 16
    m = structured_plate(a, a, n, n, t)
    bcs = [BC("displacement", [1, 2, 3, 4], [None, None, 0.0]),
           BC("displacement", [], [0.0, 0.0, None], nodes=[0]),
           BC("displacement", [], [None, 0.0, None], nodes=[n * (n + 1)]),
           BC("pressure", [0], [q])]
    r = solve_shell(m, MAT, bcs, log=lambda *a: None)
    w = np.abs(r.u[:, 2]).max()
    assert abs(w / (0.00406 * q * a ** 4 / _D(t)) - 1) < 0.01     # Timoshenko
    c = int(np.argmin(np.linalg.norm(m.nodes - [50, 50, 0], axis=1)))
    sx = abs(r.extra["応力 σx（上面） [MPa]"][c])
    assert abs(sx / (6 * 0.0479 * q * a ** 2 / t ** 2) - 1) < 0.03


def test_clamped_plate():
    a, t, q = 100.0, 2.0, 0.01
    m = structured_plate(a, a, 24, 24, t)
    r = solve_shell(m, MAT, [BC("fixed", [1, 2, 3, 4]), BC("pressure", [0], [q])], log=lambda *a: None)
    w = np.abs(r.u[:, 2]).max()
    assert abs(w / (0.00126 * q * a ** 4 / _D(t)) - 1) < 0.015


def test_inplane_cantilever():
    """面内曲げ（Allman 膜）: 粗いメッシュでも梁理論に近い"""
    m = structured_plate(100, 20, 40, 8, 2.0)
    r = solve_shell(m, MAT, [BC("fixed", [1]), BC("force", [2], [0, 100.0, 0])], log=lambda *a: None)
    I = 2 * 20 ** 3 / 12
    d = 100 * 100 ** 3 / (3 * 200000 * I) + 100 * 100 / (200000 / 2.6 * 40 * 5 / 6)
    assert abs(r.u[:, 1].max() / d - 1) < 0.02


def test_uniform_tension():
    m = structured_plate(100, 20, 20, 4, 2.0)
    bcs = [BC("displacement", [1], [0.0, None, 0.0]), BC("displacement", [], [None, 0.0, None], nodes=[0]),
           BC("force", [2], [4000.0, 0, 0])]
    r = solve_shell(m, MAT, bcs, log=lambda *a: None)
    assert abs(r.u[m.nodes[:, 0] == 100, 0].mean() / 0.05 - 1) < 0.02


def test_shell_plastic_tension_and_elastic_equivalence():
    from openmech.materials import Material
    pl = Material("p", 200000, 0.3, 7850, 250, plasticity="bilinear_iso", tangent_modulus=2000)
    m = structured_plate(100, 20, 20, 4, 2.0)
    em = ([pl.state()], np.zeros(m.n_elems, int))
    small = [BC("fixed", [1]), BC("force", [2], [0, 0, -1.0])]
    a = solve_shell(m, MAT, small, log=lambda *x: None)
    b = solve_shell(m, pl, small, elem_materials=em, log=lambda *x: None)
    assert np.abs(a.u - b.u).max() < 1e-8 * np.abs(a.u).max()
    bcs = [BC("displacement", [1], [0.0, None, 0.0]), BC("displacement", [], [None, 0.0, None], nodes=[0]),
           BC("displacement", [2], [0.5, None, None])]
    r = solve_shell(m, pl, bcs, elem_materials=em, log=lambda *x: None)
    H = 200000 * 2000 / 198000
    ep = (0.005 - 250 / 200000) / (1 + H / 200000)
    assert abs(r.stress[:, 0].mean() / (250 + H * ep) - 1) < 0.002
    assert abs(r.fields()["相当塑性ひずみ"].mean() / ep - 1) < 0.01


def test_shell_symmetry_half_model():
    """単純支持板の 1/4 モデル（対称領域2辺）が全体モデルと同じ"""
    a, t, q = 100.0, 2.0, 0.01
    full = structured_plate(a, a, 16, 16, t)
    rf = solve_shell(full, MAT, [BC("displacement", [1, 2, 3, 4], [None, None, 0.0]),
                                 BC("displacement", [], [0.0, 0.0, None], nodes=[0]),
                                 BC("displacement", [], [None, 0.0, None], nodes=[16 * 17]),
                                 BC("pressure", [0], [q])], log=lambda *x: None)
    quarter = structured_plate(a / 2, a / 2, 8, 8, t)
    rq = solve_shell(quarter, MAT, [BC("displacement", [1, 3], [None, None, 0.0]),
                                    BC("symmetry", [2]), BC("symmetry", [4]), BC("pressure", [0], [q])],
                     log=lambda *x: None)
    assert abs(rq.u[:, 2].min() / rf.u[:, 2].min() - 1) < 1e-6


def test_shell_large_deflection_cantilever():
    """シェルの大変形（共回転の定式化）: 片持ちはりの先端に横荷重 P L² / (E I) = 4。
    先端のたわみ w / L = 0.670、縮み u / L = −0.329（Bisshopp・Drucker の解）。 微小変形なら w / L = 1.333"""
    from openmech.nonlinear import AnalysisSettings
    S0 = Material("s", 200000.0, 0.0, 7850.0, 250.0)
    m = structured_plate(100, 10, 40, 2, 1.0)
    inf = m.face_info()
    root = next(f["id"] for f in inf if f.get("edge") and abs(f["centroid"][0]) < 1e-6)
    tip = next(f["id"] for f in inf if f.get("edge") and abs(f["centroid"][0] - 100) < 1e-6)
    EI = 200000.0 * 10 / 12
    bcs = [BC("fixed", [root]), BC("force", [tip], [0, 0, 4 * EI / 100 ** 2])]
    tn = np.isclose(m.nodes[:, 0], 100)
    r0 = solve_shell(m, S0, bcs, log=lambda *a: None)
    assert r0.u[tn, 2].mean() / 100 == pytest.approx(4 / 3, rel=0.01)
    st = AnalysisSettings(large_deflection=True, initial_substeps=10, max_substeps=100)
    r = solve_shell(m, S0, bcs, settings=st, log=lambda *a: None)
    assert r.u[tn, 2].mean() / 100 == pytest.approx(0.670, abs=0.003)
    assert r.u[tn, 0].mean() / 100 == pytest.approx(-0.329, abs=0.003)
    assert r.reactions[0] == pytest.approx([0, 0, -4 * EI / 100 ** 2], abs=0.5)
    assert "大変形" in r.info["要素タイプ"]


def test_shell_large_deflection_membrane_stiffening():
    """両端をピン止めした帯板に横の圧力: 大変形では膜力が効いて、たわみが微小変形よりずっと小さくなる。
    荷重を 2 倍にしてもたわみは 2 倍にならない（3 乗根に近い伸び方）"""
    from openmech.nonlinear import AnalysisSettings
    S0 = Material("s", 200000.0, 0.0, 7850.0, 250.0)
    m = structured_plate(100, 10, 40, 2, 1.0)
    inf = m.face_info()
    ends = [f["id"] for f in inf if f.get("edge") and (abs(f["centroid"][0]) < 1e-6 or abs(f["centroid"][0] - 100) < 1e-6)]
    surf = next(f["id"] for f in inf if not f.get("edge"))
    st = AnalysisSettings(large_deflection=True, initial_substeps=10, max_substeps=100)
    w = {}
    for p in (0.5, 1.0):
        bcs = [BC("displacement", ends, [0, 0, 0]), BC("pressure", [surf], [p])]
        w[p] = np.abs(solve_shell(m, S0, bcs, settings=st, log=lambda *a: None).u[:, 2]).max()
    lin = np.abs(solve_shell(m, S0, [BC("displacement", ends, [0, 0, 0]), BC("pressure", [surf], [1.0])],
                             log=lambda *a: None).u[:, 2]).max()
    assert lin == pytest.approx(5 * 10.0 * 100 ** 4 / (384 * 200000.0 * 10 / 12), rel=0.02)     # 単純支持はり
    assert w[1.0] < 0.4 * lin
    assert 1.2 < w[1.0] / w[0.5] < 1.5


def test_shell_large_deflection_follower_pressure():
    """大変形の圧力は変形後の面に垂直なまま（追従荷重）。 片持ちの帯板に一様な圧力 q L³ / (E I) = 1 と 10。
    小さい荷重では線形の q L⁴ / (8 E I)、大きい荷重では板が巻いて、反力に長さ方向の成分が出る"""
    from openmech.nonlinear import AnalysisSettings
    S0 = Material("s", 200000.0, 0.0, 7850.0, 250.0)
    m = structured_plate(100, 10, 40, 2, 1.0)
    inf = m.face_info()
    root = next(f["id"] for f in inf if f.get("edge") and abs(f["centroid"][0]) < 1e-6)
    surf = next(f["id"] for f in inf if not f.get("edge"))
    EI = 200000.0 * 10 / 12
    st = AnalysisSettings(large_deflection=True, initial_substeps=10, max_substeps=100)
    tn = np.isclose(m.nodes[:, 0], 100)
    out = {}
    for k in (1, 10):
        p = k * EI / 1e6 / 10
        r = solve_shell(m, S0, [BC("fixed", [root]), BC("pressure", [surf], [p])], settings=st, log=lambda *a: None)
        out[k] = (r.u[tn, 2].mean() / 100, r.u[tn, 0].mean() / 100, r.reactions[0], p * 1000.0)
    assert out[1][0] == pytest.approx(-0.125, rel=0.02)
    assert abs(out[1][2][0]) < 0.2 * out[1][3]                      # まだほとんど真横の力
    assert -0.86 < out[10][0] < -0.80 and -0.68 < out[10][1] < -0.61
    assert abs(out[10][2][0]) > 0.7 * out[10][3]                    # 巻いた板にかかる圧力は長さ方向の成分が大きい
    assert np.hypot(out[10][2][0], out[10][2][2]) < out[10][3]      # 向きがばらけるので、合力は p A より小さい


def test_shell_large_deflection_with_laminate_and_plasticity():
    """大変形＋積層・大変形＋塑性。 等方材料 1 層の積層は、積層なしの大変形と同じ答え。
    塑性は、降伏より下では弾性の大変形と同じで、降伏を超えるとたわみが増えて塑性ひずみが出る"""
    from openmech.composite import Laminates
    from openmech.nonlinear import AnalysisSettings
    S0 = Material("s", 200000.0, 0.0, 7850.0, 250.0)
    m = structured_plate(100, 10, 30, 2, 1.0)
    inf = m.face_info()
    root = next(f["id"] for f in inf if f.get("edge") and abs(f["centroid"][0]) < 1e-6)
    tip = next(f["id"] for f in inf if f.get("edge") and abs(f["centroid"][0] - 100) < 1e-6)
    EI = 200000.0 * 10 / 12
    st = AnalysisSettings(large_deflection=True, initial_substeps=10, max_substeps=100)
    tn = np.isclose(m.nodes[:, 0], 100)
    q = dict(settings=st, log=lambda *a: None)
    bcs = [BC("fixed", [root]), BC("force", [tip], [0, 0, 2 * EI / 100 ** 2])]
    w0 = solve_shell(m, S0, bcs, **q).u[tn, 2].mean()
    L = Laminates(np.zeros(m.n_elems, int), [{"plies": [(S0.state(), 1.0, 0.0)]}])
    rl = solve_shell(m, S0, bcs, laminate=L, **q)
    assert rl.u[tn, 2].mean() == pytest.approx(w0, rel=2e-3)
    assert "複合材 破損指標" in rl.extra
    # 塑性: 降伏モーメント My = σy b t² / 6 = 416.7 N mm → 先端荷重 4.17 N で根元が降伏
    pl = Material("p", 200000.0, 0.0, 7850.0, 250.0, plasticity="bilinear_iso", tangent_modulus=2000.0)
    em = ([pl.state()], np.zeros(m.n_elems, int))
    r1 = solve_shell(m, pl, [BC("fixed", [root]), BC("force", [tip], [0, 0, 2.5])], elem_materials=em, **q)
    e1 = solve_shell(m, S0, [BC("fixed", [root]), BC("force", [tip], [0, 0, 2.5])], **q)
    assert r1.u[tn, 2].mean() == pytest.approx(e1.u[tn, 2].mean(), rel=5e-3)
    assert r1.extra["相当塑性ひずみ"].max() < 1e-6
    r2 = solve_shell(m, pl, [BC("fixed", [root]), BC("force", [tip], [0, 0, 5.8])], elem_materials=em, **q)
    e2 = solve_shell(m, S0, [BC("fixed", [root]), BC("force", [tip], [0, 0, 5.8])], **q)
    assert r2.u[tn, 2].mean() > 1.05 * e2.u[tn, 2].mean()
    assert r2.extra["相当塑性ひずみ"].max() > 1e-4
    assert r2.reactions[0][2] == pytest.approx(-5.8, abs=0.05)


def test_surface_body_split():
    """サーフェス（平板）の切断: Y = 40 で 2 つのボディに分かれ、切った線の節点は共有される。
    片側だけ残すこともでき、板厚は元のボディと同じ。 解析の答えは切る前と同じ"""
    from openmech.geometry import build_mesh, gmsh_available, new_feature
    if not gmsh_available():
        pytest.skip("gmsh がありません")
    plate = new_feature("plate", du=100.0, dv=50.0, t=2.0)

    def mesh(*feats):
        return build_mesh({"kind": "model", "features": [plate, *feats]}, size=5.0, log=lambda *a: None)

    def area(m, body=None):
        X = m.nodes[m.elems[:, :3]]
        a = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
        return a.sum() if body is None else a[m.elem_body == body].sum()

    m0 = mesh()
    m2 = mesh(new_feature("split", axis="Y", offset=40.0))
    assert m2.is_shell and sorted(np.unique(m2.elem_body).tolist()) == [0, 1]
    assert sorted([round(area(m2, 0)), round(area(m2, 1))]) == [1000, 4000]
    assert np.allclose(m2.thickness, 2.0)
    on_cut = np.isclose(m2.nodes[:, 1], 40.0)
    nb0 = np.unique(m2.elems[m2.elem_body == 0])
    nb1 = np.unique(m2.elems[m2.elem_body == 1])
    assert on_cut.sum() > 10 and set(np.nonzero(on_cut)[0]) <= (set(nb0) & set(nb1))      # 切った線の節点は共有
    mp = mesh(new_feature("split", axis="Y", offset=40.0, keep="＋側を残す"))
    assert area(mp) == pytest.approx(1000.0) and mp.nodes[:, 1].min() == pytest.approx(40.0)
    mm = mesh(new_feature("split", axis="Y", offset=40.0, keep="−側を残す"))
    assert area(mm) == pytest.approx(4000.0) and mm.nodes[:, 1].max() == pytest.approx(40.0)

    def tip(m):
        inf = m.face_info()
        root = [f["id"] for f in inf if m.is_edge(f["id"]) and abs(f["centroid"][0]) < 1e-6]
        end = [f["id"] for f in inf if m.is_edge(f["id"]) and abs(f["centroid"][0] - 100) < 1e-6]
        r = solve_shell(m, S, [BC("fixed", root), BC("force", end, [0, 0, -10.0])], log=lambda *a: None)
        return r.u[:, 2].min()
    assert tip(m2) == pytest.approx(tip(m0), rel=0.01)

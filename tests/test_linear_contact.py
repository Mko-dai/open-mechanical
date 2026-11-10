"""線形の解析（モーダル・座屈・熱）での接触: 節点を共有していないボディを、接触（結合）でつなぐ"""
import numpy as np
import pytest

from openmech.contact import detect_contacts
from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.project import Project
from openmech.solver import BC

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0, conductivity=50.0, specific_heat=460.0)
quiet = lambda *a: None      # noqa: E731


def two_bodies(nx=10):
    """長さ 200 のはりを、真ん中で 2 つのボディに分けたもの（節点は共有しない）と、1 つのボディのもの"""
    a = structured_box(100, 20, 20, nx, 3, 3).to_quadratic()
    b = structured_box(100, 20, 20, nx, 3, 3, origin=(100, 0, 0)).to_quadratic()
    m = Mesh(np.vstack([a.nodes, b.nodes]), np.vstack([a.elems, b.elems + a.n_nodes]))
    m.elem_body = np.r_[np.zeros(a.n_elems, int), np.ones(b.n_elems, int)]
    m.node_origin = np.arange(m.n_nodes)
    m.group_faces()
    one = structured_box(200, 20, 20, 2 * nx, 3, 3).to_quadratic()
    one.elem_body = np.zeros(one.n_elems, int)
    one.group_faces()
    return m, one


def face(m, axis, val, sign):
    return [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - val) < 1e-6
            and f["normal"][axis] * sign > 0.99]


def project(m, atype, bcs, **opts):
    p = Project()
    p.material = S
    p.analysis_type = atype
    p.analysis_opts = dict(opts)
    p.bcs = bcs
    return p


def test_modal_with_bonded_contact_matches_single_body():
    m, one = two_bodies()
    f2 = project(m, "modal", [BC("fixed", face(m, 0, 0, -1))], n_modes=4).run_all(m, log=quiet)
    f1 = project(one, "modal", [BC("fixed", face(one, 0, 0, -1))], n_modes=4).run_all(one, log=quiet)
    a = np.array([r.info["振動数 [Hz]"] for r in f2])
    b = np.array([r.info["振動数 [Hz]"] for r in f1])
    assert np.allclose(a, b, rtol=0.01)
    assert a[0] > 100       # つながっていないと、先のボディが浮いて 0 Hz の剛体モードになる


def test_buckling_with_bonded_contact_matches_single_body():
    m, one = two_bodies()
    P = 1000.0
    r2 = project(m, "buckling", [BC("fixed", face(m, 0, 0, -1)), BC("force", face(m, 0, 200, 1), [-P, 0, 0])],
                 n_buckling=1).run_all(m, log=quiet)
    r1 = project(one, "buckling", [BC("fixed", face(one, 0, 0, -1)), BC("force", face(one, 0, 200, 1), [-P, 0, 0])],
                 n_buckling=1).run_all(one, log=quiet)
    assert r2[1].info["座屈荷重係数"] == pytest.approx(r1[1].info["座屈荷重係数"], rel=0.01)
    euler = np.pi ** 2 * 200000.0 * (20 ** 4 / 12) / (4 * 200.0 ** 2) / P
    assert r2[1].info["座屈荷重係数"] == pytest.approx(euler, rel=0.03)


def test_thermal_conduction_across_contact():
    """2 つのボディをまたぐ 1 次元の熱伝導: 両端の温度を決めると、温度は直線（接触面で温度がつながる）"""
    m, _one = two_bodies(6)
    p = project(m, "thermal", [BC("temperature", face(m, 0, 0, -1), [100.0]),
                               BC("temperature", face(m, 0, 200, 1), [0.0])])
    r = p.run_all(m, log=quiet)[-1]
    T = r.extra["温度 [°C]"]
    x = m.nodes[:, 0]
    assert np.abs(T - (100.0 - 0.5 * x)).max() < 0.5
    # 接触を抑制すると熱が流れない（先のボディは 0 °C のまま）
    for c in p.contacts:
        c.suppressed = True
    T2 = p.run_all(m, log=quiet)[-1].extra["温度 [°C]"]
    assert np.allclose(T2[x > 100 + 1e-6], 0.0, atol=1e-6)


def test_electrostatic_across_contact():
    """2 つのボディをまたぐ平行平板: 電位は直線"""
    m, _one = two_bodies(6)
    p = project(m, "electrostatic", [BC("voltage", face(m, 0, 0, -1), [100.0]), BC("ground", face(m, 0, 200, 1))])
    r = p.run_all(m, log=quiet)[-1]
    phi = r.extra["電位 [V]"]
    assert np.abs(phi - (100.0 - 0.5 * m.nodes[:, 0])).max() < 0.5


def test_thermal_contact_conductance():
    """熱接触コンダクタンス h を入れると、接触面で温度が ΔT = q / h だけ飛ぶ（1 次元: q = ΔT_total / (L/k + 1/h)）"""
    m, _one = two_bodies(6)
    p = project(m, "thermal", [BC("temperature", face(m, 0, 0, -1), [100.0]),
                               BC("temperature", face(m, 0, 200, 1), [0.0])])
    p.ensure_contacts(m)
    h = 200.0                                  # W/(m²·K)
    for c in p.contacts:
        c.conductance = h
    T = p.run_all(m, log=quiet)[-1].extra["温度 [°C]"]
    k = 50.0                                   # W/(m·K)
    q = 100.0 / (0.2 / k + 1.0 / h)            # W/m²（長さ 200 mm = 0.2 m）
    x = m.nodes[:, 0]
    left = T[np.isclose(x, 100.0) & (np.arange(m.n_nodes) < m.n_nodes // 2)].mean()
    right = T[np.isclose(x, 100.0) & (np.arange(m.n_nodes) >= m.n_nodes // 2)].mean()
    assert left - right == pytest.approx(q / h, rel=0.02)


def test_unsupported_analysis_refuses_disconnected_bodies():
    """接触を扱えない解析（陽解法など）は、節点を共有していないボディがあるとエラーで知らせる"""
    m, _one = two_bodies(4)
    m.node_origin = np.r_[np.arange(m.n_nodes - 5), np.arange(5)]      # 界面で節点を分けたメッシュの印
    p = project(m, "transient", [BC("fixed", face(m, 0, 0, -1))])
    p.dynamic = {"method": "explicit", "end_time": 1e-5}
    with pytest.raises(ValueError, match="節点共有"):
        p.run_all(m, log=quiet)

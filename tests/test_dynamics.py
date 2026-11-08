import math

import numpy as np

from openmech.dynamics import (DynamicSettings, _tet_mass_ref, consistent_mass, element_dt, lumped_mass,
                               solve_dynamic)
from openmech.materials import Material
from openmech.mesh import structured_box
from openmech.solver import BC, material_D

STEEL = Material("鋼", 200000, 0.3, 7850, 250)


def test_mass_matrices():
    for nn in (4, 10):
        C = _tet_mass_ref(nn)
        assert abs(C.sum() - 1) < 1e-12
        assert np.all(np.diag(C) > 0)
    m = structured_box(10, 4, 2, 3, 2, 1).to_quadratic()
    rho = np.full(m.n_elems, 7.85e-9)
    total = 7.85e-9 * 80
    assert abs(lumped_mass(m, rho).sum() - total) / total < 1e-12
    M = consistent_mass(m, rho)
    assert abs(M.sum() / 3 - total) / total < 1e-12


def _beam():
    m = structured_box(100, 10, 10, 10, 2, 2).to_quadratic()
    info = m.face_info()
    f0 = [f["id"] for f in info if f["centroid"][0] < 1e-6]
    f1 = [f["id"] for f in info if f["centroid"][0] > 100 - 1e-6]
    tip = int(np.argmin(np.linalg.norm(m.nodes - [100, 5, 5], axis=1)))
    return m, f0, f1, tip


def _period(times, y):
    y = y - y.mean()
    z = [times[i] - y[i] * (times[i + 1] - times[i]) / (y[i + 1] - y[i])
         for i in range(len(y) - 1) if y[i] < 0 <= y[i + 1]]
    return float(np.mean(np.diff(z)))


def test_cantilever_period_implicit_and_explicit():
    """先端に急に力を加えた片持ち梁の振動周期: 陰解法と陽解法が一致し、梁理論とも近い"""
    m, f0, f1, tip = _beam()
    bcs = [BC("fixed", f0), BC("force", f1, [0, 0, -100.0])]
    I, A, L = 10 ** 4 / 12, 100.0, 100.0
    w1 = 1.875104 ** 2 * math.sqrt(200000 * I / (7.85e-9 * A * L ** 4))
    T = 2 * math.pi / w1
    out = {}
    for meth, dt in (("implicit", T / 60), ("explicit", 0.0)):
        s = DynamicSettings(method=meth, end_time=3 * T, dt=dt, n_frames=150)
        fr = solve_dynamic(m, STEEL, bcs, s, log=lambda *a: None)
        t = np.array([r.time for r in fr])
        y = np.array([r.u[tip, 2] for r in fr])
        out[meth] = _period(t, y)
    assert abs(out["implicit"] / out["explicit"] - 1) < 0.03
    assert abs(out["explicit"] / T - 1) < 0.10      # せん断変形・粗いメッシュの分だけ長め


def test_explicit_stable_dt_and_energy():
    """自由振動（減衰なし）で全エネルギーが保存され、時間増分は要素の安定限界以下"""
    m, f0, f1, tip = _beam()
    s = DynamicSettings(method="explicit", end_time=1e-3, n_frames=5, init_velocity=[0, 0, -1000.0])
    fr = solve_dynamic(m, STEEL, [BC("fixed", f0)], s, log=lambda *a: None)
    h = fr[-1].history
    E = np.array(h["運動エネルギー [mJ]"]) + np.array(h["ひずみエネルギー [mJ]"])
    assert np.ptp(E[5:]) / E[5:].mean() < 0.02
    dte = element_dt(m, material_D(STEEL)[None], np.zeros(m.n_elems, int), np.full(m.n_elems, 7.85e-9))
    assert fr[-1].info["時間増分 [s]"] <= 0.9 * dte.min() * 1.0001


def test_drop_on_rigid_wall_rebounds():
    """剛体壁に落ちたブロック: 接触中に止まって跳ね返り、壁にめり込まない（陽解法・陰解法）"""
    m = structured_box(20, 20, 10, 2, 2, 2).to_quadratic()
    v0 = 1000.0
    for meth in ("explicit", "implicit"):
        s = DynamicSettings(method=meth, end_time=3e-5, n_frames=30, init_velocity=[0, 0, -v0],
                            dt=(0 if meth == "explicit" else 2.5e-7),
                            wall={"normal": [0, 0, 1], "point": [0, 0, 0]})
        fr = solve_dynamic(m, STEEL, [], s, log=lambda *a: None)
        zmin = min(float((m.nodes[:, 2] + r.u[:, 2]).min()) for r in fr)
        vz = np.mean([r.extra["速度 Z [mm/s]"].mean() for r in fr[-3:]])
        assert zmin > -0.002 * 10, (meth, zmin)
        assert vz > 0.3 * v0, (meth, vz)          # 上向きに跳ね返っている
        assert fr[-1].info["最大接触力 [N]"] > 0

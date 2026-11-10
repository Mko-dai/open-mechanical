"""ビーム接続: 面と地面・面と面を、丸棒のはり 1 本でつなぐ"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.solver import BC, solve

STIFF = Material("stiff", 2e8, 0.3, 7850.0, 1e9)      # ブロックはほぼ剛体（ビームの変形だけを見る）
quiet = dict(log=lambda *a: None)
E, RAD, L = 200000.0, 2.0, 30.0
A, I = np.pi * RAD ** 2, np.pi * RAD ** 4 / 4
G = E / 2.6
PHI = 12 * E * I / (0.9 * G * A * L ** 2)


def face(m, axis, val, sign, body=None):
    out = [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99]
    assert out
    return out


def test_beam_to_ground_axial_and_bending():
    """ブロックの下面を、長さ 30 の丸棒で地面につなぐ。 軸方向: δ = FL/EA。
    横方向（ブロック上面に力）: 先端の力 F とモーメント F h を受ける片持ちはり（せん断変形込み）"""
    m = structured_box(10, 10, 10, 2, 2, 2).to_quadratic()
    bot, top = face(m, 2, 0, -1), face(m, 2, 10, 1)
    beam = BC("beam", bot, [RAD, E, 0.3], options={"ground": True, "ground_point": [5.0, 5.0, -L], "behavior": "rigid"})
    F = 1000.0
    r = solve(m, STIFF, [beam, BC("force", top, [0, 0, F])], **quiet)
    uz = r.u[m.face_nodes(bot), 2].mean()
    assert uz == pytest.approx(F * L / (E * A), rel=1e-3)
    assert r.beams[0]["軸力 [N]"] == pytest.approx(F, rel=1e-6)
    r = solve(m, STIFF, [beam, BC("force", top, [F, 0, 0])], **quiet)
    h = 10.0                                              # 力の作用点（上面）からビームの先端（下面）までの高さ
    d_exp = F * L ** 3 / (3 * E * I) * (1 + PHI / 4) + F * h * L ** 2 / (2 * E * I)
    ux = r.u[m.face_nodes(bot), 0].mean()
    assert ux == pytest.approx(d_exp, rel=2e-3)
    b = r.beams[0]
    assert b["せん断力 [N]"] == pytest.approx(F, rel=1e-6)
    assert b["曲げモーメント（端 A） [N·mm]"] == pytest.approx(F * h, rel=1e-3)          # ブロック側
    assert b["曲げモーメント（端 B） [N·mm]"] == pytest.approx(F * (h + L), rel=1e-3)    # 地面側
    assert b["最大応力（軸＋曲げ） [MPa]"] == pytest.approx(F * (h + L) * RAD / I, rel=1e-3)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_beam_between_two_bodies(nonlinear):
    """離れた 2 つのブロックをビームでつなぐ（接触なし）。 下を固定して上を引くと、間隔の伸び = FL/EA"""
    m1 = structured_box(10, 10, 10, 2, 2, 2).to_quadratic()
    m2 = structured_box(10, 10, 10, 2, 2, 2, origin=(0, 0, 10 + L)).to_quadratic()
    m = Mesh(np.vstack([m1.nodes, m2.nodes]), np.vstack([m1.elems, m2.elems + m1.n_nodes]))
    m.elem_body = np.r_[np.zeros(m1.n_elems, int), np.ones(m2.n_elems, int)]
    m.group_faces()
    lo_top, up_bot = face(m, 2, 10, 1), face(m, 2, 10 + L, -1)
    F = 2000.0
    bcs = [BC("fixed", face(m, 2, 0, -1)), BC("force", face(m, 2, 20 + L, 1), [0, 0, F]),
           BC("beam", lo_top, [RAD, E, 0.3], options={"faces_b": up_bot, "behavior": "rigid"})]
    if nonlinear:
        from openmech.nonlinear import AnalysisSettings, solve_static
        r = solve_static(m, STIFF, bcs, [], AnalysisSettings(n_steps=2, step_end_times=[1.0, 2.0]), **quiet)[-1]
    else:
        r = solve(m, STIFF, bcs, **quiet)
    gap = r.u[m.face_nodes(up_bot), 2].mean() - r.u[m.face_nodes(lo_top), 2].mean()
    assert gap == pytest.approx(F * L / (E * A), rel=1e-3)
    assert r.beams[2]["軸力 [N]"] == pytest.approx(F, rel=1e-4)
    assert r.beams[2]["伸び [mm]"] == pytest.approx(F * L / (E * A), rel=1e-4)

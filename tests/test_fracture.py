"""破壊力学（J 積分・応力拡大係数）の検証: 片側き裂の帯板（SENT）の引張、Tada の式と比較"""
import numpy as np

from openmech.fracture import evaluate_cracks, insert_crack, j_integral, k_from_j
from openmech.materials import Material
from openmech.mesh import structured_box
from openmech.solver import BC, material_D, solve

W, H, t, a, SIG = 50.0, 50.0, 2.0, 15.0, 100.0
MAT = Material("s", 200000, 0.3, 7850)
q = a / W
K_TADA = SIG * np.sqrt(np.pi * a) * (1.12 - 0.231 * q + 10.55 * q ** 2 - 21.72 * q ** 3 + 30.39 * q ** 4)
LOG = dict(log=lambda *x: None)


def test_sent_half_model_path_independent():
    nx = 40
    m = structured_box(W, H, t, nx, nx, 1).to_quadratic()
    X = m.nodes
    lig = np.nonzero(np.isclose(X[:, 1], 0) & (X[:, 0] >= a - 1e-9))[0]
    zz = np.nonzero(np.isclose(X[:, 2], 0) | np.isclose(X[:, 2], t))[0]
    top = [f["id"] for f in m.face_info() if abs(f["centroid"][1] - H) < 1e-9]
    c = np.nonzero(np.isclose(X[:, 0], W) & np.isclose(X[:, 1], 0))[0][:1]
    r = solve(m, MAT, [BC("displacement", [], [None, 0.0, None], nodes=lig.tolist()),
                       BC("displacement", [], [None, None, 0.0], nodes=zz.tolist()),
                       BC("displacement", [], [0.0, None, None], nodes=c.tolist()), BC("pressure", top, [-SIG])], **LOG)
    D = material_D(MAT.state())
    h = W / nx
    Ks = [k_from_j(j_integral(m, r.u, D, [a, 0, 0], [a, 0, t], [1, 0, 0], ri * h, ro * h, symmetric=True), 2e5, 0.3)
          for ri, ro in [(0, 3), (1, 5), (3, 8)]]
    assert np.ptp(Ks) / K_TADA < 0.003                  # 積分路によらない
    assert abs(np.mean(Ks) / K_TADA - 1) < 0.015


def test_sent_full_model_inserted_crack():
    nx = 30          # き裂先端が節点に乗るように（a/h が整数）
    m = structured_box(W, 2 * H, t, nx, 2 * nx, 1).to_quadratic()
    m.face_info()
    m2, (fm, fp) = insert_crack(m, [0, H, 0], [0, 1, 0], lambda X: X[:, 0] < a - 1e-6)
    assert m2.n_faces == m.n_faces + 2 and m2.n_nodes > m.n_nodes
    X = m2.nodes
    info = m2.face_info()
    top = [f["id"] for f in info if abs(f["centroid"][1] - 2 * H) < 1e-9]
    bot = [f["id"] for f in info if abs(f["centroid"][1]) < 1e-9]
    zz = np.nonzero(np.isclose(X[:, 2], 0) | np.isclose(X[:, 2], t))[0]
    c = np.nonzero(np.isclose(X[:, 0], W) & np.isclose(X[:, 1], H))[0][:1]
    r = solve(m2, MAT, [BC("displacement", [], [None, None, 0.0], nodes=zz.tolist()),
                        BC("displacement", [], [0.0, 0.0, None], nodes=c.tolist()),
                        BC("pressure", top, [-SIG]), BC("pressure", bot, [-SIG])], **LOG)
    # き裂面が開いている（上下の面の節点が離れる）
    rows = evaluate_cracks(m2, r, [{"front0": [a, H, 0], "front1": [a, H, t], "normal": [0, 1, 0],
                                    "direction": [1, 0, 0], "front_size": W / nx, "toughness": 2000.0}])
    row = rows[0]
    assert abs(row["K（J から）[MPa√mm]"] / K_TADA - 1) < 0.02
    assert abs(row["K_I（変位）[MPa√mm]"] / K_TADA - 1) < 0.06
    assert abs(row["K_II（変位）[MPa√mm]"]) < 0.02 * K_TADA
    assert row["安全率（K_IC / K）"] > 1

"""サブモデル: 粗い全体モデルの変位を切断境界に与えた細かい部分モデルが、細かい全体モデルと一致する"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import structured_box, structured_hex_box
from openmech.solver import BC, solve
from openmech.submodel import CoarseField, save_coarse

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def face(m, axis, val, sign):
    ids = [f["id"] for f in m.face_info() if abs(f["centroid"][axis] - val) < 1e-6 and f["normal"][axis] * sign > 0.99]
    assert ids
    return ids


def test_interpolation_exact_for_quadratic_field():
    """Tet10 / Hex20 は 2 次の変位場をそのまま再現する"""
    rng = np.random.default_rng(0)
    for m in (structured_box(10, 8, 6, 3, 2, 2).to_quadratic(), structured_hex_box(10, 8, 6, 3, 2, 2)):
        X = m.nodes
        u = np.c_[X[:, 0] ** 2, X[:, 0] * X[:, 1], 0.5 * X[:, 2] ** 2 - X[:, 1]]
        cf = CoarseField(m.nodes, m.elems, u)
        P = rng.uniform([0, 0, 0], [10, 8, 6], (40, 3))
        v, d = cf.interpolate(P)
        ex = np.c_[P[:, 0] ** 2, P[:, 0] * P[:, 1], 0.5 * P[:, 2] ** 2 - P[:, 1]]
        assert np.abs(v - ex).max() < 1e-9 and d.max() == 0


def test_submodel_matches_full_model(tmp_path):
    """片持ち梁（200×20×20、先端に横荷重）の根元側の一部 x∈[0,60] を細かく切り出す"""
    def bcs_full(m):
        return [BC("fixed", face(m, 0, 0, -1)), BC("force", face(m, 0, 200, 1), [0, 0, -2000.0])]
    coarse = structured_box(200, 20, 20, 10, 2, 2).to_quadratic()
    rc = solve(coarse, S, bcs_full(coarse), **quiet)
    f = tmp_path / "coarse.npz"
    save_coarse(f, coarse, rc.u)
    sub = structured_box(60, 20, 20, 12, 4, 4).to_quadratic()
    rs = solve(sub, S, [BC("fixed", face(sub, 0, 0, -1)),
                        BC("submodel", face(sub, 0, 60, 1), options={"file": str(f)})], **quiet)
    fine = structured_box(200, 20, 20, 40, 4, 4).to_quadratic()
    rf = solve(fine, S, bcs_full(fine), **quiet)
    # 切断面から離れた位置（x = 30 の上面中央）の曲げ応力を比較
    def sx_at(r, m, x):
        k = np.argmin(np.linalg.norm(m.nodes - [x, 10, 20], axis=1))
        return r.stress[k, 0]
    s_fine, s_sub, s_coarse = sx_at(rf, fine, 30), sx_at(rs, sub, 30), sx_at(rc, coarse, 30)
    assert s_sub == pytest.approx(s_fine, rel=0.02)
    # 切断境界の反力 ≈ 断面のせん断力（先端荷重 2000 N）。 変位で与えるので粗いモデルの誤差分ずれる（Ansys も同じ）
    assert rs.reactions[1][2] == pytest.approx(-2000.0, rel=0.2)
    assert s_coarse == pytest.approx(s_fine, rel=0.05)


def test_submodel_outside_error(tmp_path):
    coarse = structured_box(10, 10, 10, 2, 2, 2)
    f = tmp_path / "c.npz"
    save_coarse(f, coarse, np.zeros((coarse.n_nodes, 3)))
    sub = structured_box(10, 10, 10, 2, 2, 2, origin=(100, 0, 0))
    with pytest.raises(ValueError, match="粗いモデルの外"):
        solve(sub, S, [BC("submodel", face(sub, 0, 100, -1), options={"file": str(f)})], **quiet)

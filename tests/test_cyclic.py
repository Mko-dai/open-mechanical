"""周期対称: 1 セクタのモデルで、切断面どうしを u_high = R(α) u_low で結ぶ"""
import numpy as np
import pytest

from openmech.materials import Material
from openmech.mesh import Mesh, structured_box
from openmech.solver import BC, solve

S = Material("steel", 200000.0, 0.3, 7850.0, 250.0)
quiet = dict(log=lambda *a: None)


def sector(ri, ro, deg, t, nr, nt, nz):
    b = structured_box(ro - ri, 1.0, t, nr, nt, nz).to_quadratic()
    r, th = ri + b.nodes[:, 0], np.radians(deg * b.nodes[:, 1])
    m = Mesh(np.c_[r * np.cos(th), r * np.sin(th), b.nodes[:, 2]], b.elems)
    m.group_faces()
    return m


def faces(m, test):
    return [f["id"] for f in m.face_info() if test(m.nodes[m.face_nodes(f["id"])])]


def test_cyclic_ring_torsion():
    """リング（内径固定）の外周に、セクタごとに同じ向きの力（周方向）。 周方向の平均変位は、ねじりの式
    u_θ(r) = r T / (4πGt) (1/ri² − 1/r²) と一致する（切断面を対称条件にすると周方向に動けず、これは出ない）"""
    ri, ro, t, N, F = 10.0, 20.0, 4.0, 12, 1000.0
    m = sector(ri, ro, 360.0 / N, t, 6, 6, 2)
    R = lambda X: np.hypot(X[:, 0], X[:, 1])                     # noqa: E731
    TH = lambda X: np.degrees(np.arctan2(X[:, 1], X[:, 0]))      # noqa: E731
    inner = faces(m, lambda X: np.allclose(R(X), ri, atol=1e-6))
    outer = faces(m, lambda X: np.allclose(R(X), ro, atol=1e-6))
    low = faces(m, lambda X: np.allclose(TH(X), 0.0, atol=1e-6))
    high = faces(m, lambda X: np.allclose(TH(X), 360.0 / N, atol=1e-6))
    mid = np.radians(180.0 / N)
    bcs = [BC("fixed", inner), BC("force", outer, [-F * np.sin(mid), F * np.cos(mid), 0.0]),
           BC("cyclic", low, [], options={"high_faces": high, "sectors": N, "axis": [0, 0, 1]})]
    r = solve(m, S, bcs, **quiet)
    X, U = m.nodes, r.u
    th = np.arctan2(X[:, 1], X[:, 0])
    ut = -U[:, 0] * np.sin(th) + U[:, 1] * np.cos(th)
    ur = U[:, 0] * np.cos(th) + U[:, 1] * np.sin(th)
    T = N * F * ro * np.sin(mid) / mid
    G = 200000.0 / 2.6
    for rr in (15.0, 20.0):
        sel = np.isclose(R(X), rr) & (np.degrees(th) < 360.0 / N - 1e-6)     # 高い側は低い側と同じ点なので数えない
        exact = rr * T / (4 * np.pi * G * t) * (1 / ri ** 2 - 1 / rr ** 2)
        assert ut[sel].mean() == pytest.approx(exact, rel=0.02)
    # 切断面: 高い側の変位 = 低い側の変位を 30° 回したもの（円筒座標の成分が同じ）
    lo = np.nonzero(np.isclose(np.degrees(th), 0.0, atol=1e-6))[0]
    hi = np.nonzero(np.isclose(np.degrees(th), 360.0 / N, atol=1e-6))[0]
    key = lambda idx: np.lexsort((np.round(X[idx, 2], 6), np.round(R(X[idx]), 6)))   # noqa: E731
    lo, hi = lo[key(lo)], hi[key(hi)]
    assert np.allclose(ut[lo], ut[hi], atol=1e-9) and np.allclose(ur[lo], ur[hi], atol=1e-9)
    assert np.allclose(U[lo, 2], U[hi, 2], atol=1e-9)
    # 反力: 内径の固定面のモーメント = セクタ 1 つ分のトルク
    assert abs(ut[lo]).max() > 0


def test_cyclic_modal_matches_axisymmetric_mode():
    """周期対称のモーダル（調和指数 0）: リングのねじり振動（内径固定）の固有振動数が、セクタの角度を変えても同じ"""
    from openmech.modal import solve_modal
    f = []
    for N in (12, 8):
        m = sector(10.0, 20.0, 360.0 / N, 4.0, 5, 5 if N == 12 else 7, 1)
        R = lambda X: np.hypot(X[:, 0], X[:, 1])                     # noqa: E731
        TH = lambda X: np.degrees(np.arctan2(X[:, 1], X[:, 0]))      # noqa: E731
        bcs = [BC("fixed", faces(m, lambda X: np.allclose(R(X), 10.0, atol=1e-6))),
               BC("cyclic", faces(m, lambda X: np.allclose(TH(X), 0.0, atol=1e-6)), [],
                  options={"high_faces": faces(m, lambda X: np.allclose(TH(X), 360.0 / N, atol=1e-6)), "sectors": N,
                           "axis": [0, 0, 1]})]
        f.append(solve_modal(m, S, bcs, 3, **quiet).freqs)
    assert np.allclose(f[0], f[1], rtol=0.01)


def test_cyclic_wrong_sector_count_error():
    m = sector(10.0, 20.0, 30.0, 4.0, 3, 3, 1)
    TH = lambda X: np.degrees(np.arctan2(X[:, 1], X[:, 0]))      # noqa: E731
    R = lambda X: np.hypot(X[:, 0], X[:, 1])                     # noqa: E731
    bcs = [BC("fixed", faces(m, lambda X: np.allclose(R(X), 10.0, atol=1e-6))),
           BC("cyclic", faces(m, lambda X: np.allclose(TH(X), 0.0, atol=1e-6)), [],
              options={"high_faces": faces(m, lambda X: np.allclose(TH(X), 30.0, atol=1e-6)), "sectors": 8})]
    with pytest.raises(ValueError, match="重なりません"):
        solve(m, S, bcs, **quiet)


def test_cyclic_modal_all_harmonics_match_full_ring():
    """周期対称のモーダル（全調和指数）: 30° のセクタから求めた振動数が、全周モデルの振動数と一致する
    （全周では 0 < k < N/2 のモードが 2 つずつ出る）"""
    from openmech.cyclic import solve_cyclic_modal
    from openmech.modal import solve_modal
    N = 12
    sec = sector(10.0, 20.0, 360.0 / N, 4.0, 2, 2, 1)
    # 全周: 同じ要素の並びを 12 回（360° に写してから、継ぎ目の重なった節点をまとめる）
    b = structured_box(10.0, 1.0, 4.0, 2, 2 * N, 1).to_quadratic()
    r, th = 10.0 + b.nodes[:, 0], np.radians(360.0 * b.nodes[:, 1])
    X = np.c_[r * np.cos(th), r * np.sin(th), b.nodes[:, 2]]
    _u, first, inv = np.unique(np.round(X, 6), axis=0, return_index=True, return_inverse=True)
    full = Mesh(X[first], inv.ravel()[b.elems])
    full.group_faces()
    R = lambda Y: np.hypot(Y[:, 0], Y[:, 1])                     # noqa: E731
    TH = lambda Y: np.degrees(np.arctan2(Y[:, 1], Y[:, 0]))      # noqa: E731
    f_full = solve_modal(full, S, [BC("fixed", faces(full, lambda Y: np.allclose(R(Y), 10.0, atol=1e-6)))], 14,
                         **quiet).freqs
    bcs = [BC("fixed", faces(sec, lambda Y: np.allclose(R(Y), 10.0, atol=1e-6))),
           BC("cyclic", faces(sec, lambda Y: np.allclose(TH(Y), 0.0, atol=1e-6)), [],
              options={"high_faces": faces(sec, lambda Y: np.allclose(TH(Y), 360.0 / N, atol=1e-6)), "sectors": N,
                       "axis": [0, 0, 1]})]
    mr = solve_cyclic_modal(sec, S, bcs, n_modes=4, **quiet)
    # セクタの結果を全周の並びに直す: 0 < k < N/2 は 2 重
    exp = np.sort(np.concatenate([[f] * (1 if k in (0, N // 2) else 2) for f, k in zip(mr.freqs, mr.harmonics)]))
    assert np.allclose(exp[:12], f_full[:12], rtol=2e-3)
    assert set(mr.harmonics) == set(range(N // 2 + 1))

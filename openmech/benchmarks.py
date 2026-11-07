"""精度検証ベンチマーク（Ansys Verification Manual / NAFEMS 標準ベンチマーク）。

  python -m openmech.benchmarks            # 全問題を実行して表を表示
  python -m openmech.benchmarks --converge # メッシュ収束の様子も表示

形状は「写像メッシュ」（パラメータ空間の格子を実形状へ写像）で作るので gmsh 不要です。
2次要素の中間節点も写像で曲面上に置くため、曲がった境界も正確に表現されます。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from .materials import Material
from .mesh import Mesh, structured_box
from .solver import BC, solve


# ---------------------------------------------------------------------------- 写像メッシュ

def mapped_mesh(F, nu, nv, nw, order=2):
    """単位立方体 [0,1]^3 の構造格子を写像 F(P)->(N,3) で実形状にする。

    戻り値: (mesh, P)  P はパラメータ座標 (N,3)。面番号は u0,u1,v0,v1,w0,w1 = 0..5
    """
    pm = structured_box(1.0, 1.0, 1.0, nu, nv, nw)
    if order == 2:
        pm = pm.to_quadratic()
    P = pm.nodes.copy()
    m = Mesh(F(P), pm.elems.copy())
    m.set_face_ids(pm.face_ids.copy())
    return m, P


def sel(P, **cond):
    """パラメータ座標で節点を選ぶ。 sel(P, u=1, w=0.5)"""
    ok = np.ones(len(P), bool)
    for k, v in cond.items():
        ok &= np.isclose(P[:, "uvw".index(k)], v, atol=1e-9)
    return np.nonzero(ok)[0]


def ellipse_ring(a1, b1, a2, b2, z0, z1):
    """楕円 (a1,b1) と (a2,b2) に挟まれた 1/4 リング。u=半径方向, v=角度(0→90°), w=厚さ"""
    def F(P):
        u, v, w = P.T
        t = v * np.pi / 2
        x = (1 - u) * a1 * np.cos(t) + u * a2 * np.cos(t)
        y = (1 - u) * b1 * np.sin(t) + u * b2 * np.sin(t)
        z = z0 + w * (z1 - z0)
        return np.column_stack([x, y, z])
    return F


# ---------------------------------------------------------------------------- 問題定義

@dataclass
class Check:
    name: str
    quantity: str
    target: float
    value: float
    unit: str
    source: str
    ansys: float | None = None  # Ansys 公開値（目標値に対する比）

    @property
    def ratio(self):
        return self.value / self.target


def le1(n=12):
    """NAFEMS LE1 楕円膜（平面応力）  目標: 点D の σyy = 92.7 MPa"""
    mat = Material("LE1", 210000.0, 0.3, 7850.0)
    m, P = mapped_mesh(ellipse_ring(2000, 1000, 3250, 2750, 0, 100), n, 2 * n, 1)
    bcs = [
        BC("displacement", [], [None, 0.0, None], nodes=sel(P, v=0).tolist()),   # y=0 対称
        BC("displacement", [], [0.0, None, None], nodes=sel(P, v=1).tolist()),   # x=0 対称
        BC("displacement", [], [None, None, 0.0], nodes=sel(P, w=0).tolist()),   # 面外
        BC("pressure", [1], [-10.0]),                                           # 外周に外向き 10 MPa
    ]
    r = solve(m, mat, bcs, log=lambda *a: None)
    d = sel(P, u=0, v=0)
    return m, Check("NAFEMS LE1 楕円膜", "点D σyy", 92.7, float(r.stress[d, 1].mean()), "MPa",
                    "NAFEMS LE1")


def le10(n=6):
    """NAFEMS LE10 厚板の圧力  目標: 点D (2000,0,300) の σyy = -5.38 MPa"""
    mat = Material("LE10", 210000.0, 0.3, 7850.0)
    nw = 2 * max(1, n // 3)  # 中立面 z=0 に節点が来るよう偶数
    m, P = mapped_mesh(ellipse_ring(2000, 1000, 3250, 2750, -300, 300), n, int(1.5 * n), nw)
    bcs = [
        BC("displacement", [], [None, 0.0, None], nodes=sel(P, v=0).tolist()),
        BC("displacement", [], [0.0, None, None], nodes=sel(P, v=1).tolist()),
        BC("displacement", [], [0.0, 0.0, None], nodes=sel(P, u=1).tolist()),   # 外周 ux=uy=0
        BC("displacement", [], [None, None, 0.0], nodes=sel(P, u=1, w=0.5).tolist()),  # 外周中立線 uz=0
        BC("pressure", [5], [1.0]),                                             # 上面に 1 MPa
    ]
    r = solve(m, mat, bcs, log=lambda *a: None)
    d = sel(P, u=0, v=0, w=1)
    return m, Check("NAFEMS LE10 厚板の圧力", "点D σyy", -5.38, float(r.stress[d, 1].mean()), "MPa",
                    "NAFEMS LE10")


def lame(n=6):
    """厚肉円筒の内圧（ラメの解・平面ひずみ）  a=100, b=200, p=100 MPa"""
    a, b, p = 100.0, 200.0, 100.0
    mat = Material("steel", 200000.0, 0.3, 7850.0)
    m, P = mapped_mesh(ellipse_ring(a, a, b, b, 0, 20), n, 2 * n, 1)
    bcs = [
        BC("displacement", [], [None, 0.0, None], nodes=sel(P, v=0).tolist()),
        BC("displacement", [], [0.0, None, None], nodes=sel(P, v=1).tolist()),
        BC("displacement", [], [None, None, 0.0], nodes=np.concatenate([sel(P, w=0), sel(P, w=1)]).tolist()),
        BC("pressure", [0], [p]),
    ]
    r = solve(m, mat, bcs, log=lambda *a: None)
    inner = sel(P, u=0)
    x = m.nodes[inner]
    rr = np.hypot(x[:, 0], x[:, 1])
    s = r.stress[inner]
    c, sn = x[:, 0] / rr, x[:, 1] / rr
    s_theta = s[:, 0] * sn**2 + s[:, 1] * c**2 - 2 * s[:, 3] * sn * c
    ur = r.u[inner, 0] * c + r.u[inner, 1] * sn
    st_exact = p * (b**2 + a**2) / (b**2 - a**2)
    ur_exact = (1 + mat.nu) * p * a**2 / (mat.E * (b**2 - a**2)) * ((1 - 2 * mat.nu) * a + b**2 / a)
    return m, [Check("厚肉円筒の内圧（ラメ）", "内面 円周応力 σθ", st_exact, float(s_theta.mean()), "MPa", "理論解"),
               Check("厚肉円筒の内圧（ラメ）", "内面 半径方向変位", ur_exact, float(ur.mean()), "mm", "理論解")]


def vmc1(n1=10, n2=2):
    """Ansys VMC1 周辺固定板の等分布荷重（1/8 対称モデル）

    目標値（Ansys が SOLID186 の細かいメッシュで求めたもの）:
      UZ(1) = -0.0172 in, SX(2) = -32.124 ksi, SX(3) = 14.465 ksi
    Ansys SOLID187（N1=10, N2=2）の結果は目標値に対し 0.984 / 0.966 / 1.000
    """
    mat = Material("VMC1", 1.0e7, 0.3, 0.0)
    m = structured_box(5.0, 5.0, 0.5, n1, n1, n2, origin=(0.0, 0.0, -0.5)).to_quadratic()
    X = m.nodes
    on = lambda ax, v: np.nonzero(np.isclose(X[:, ax], v))[0].tolist()  # noqa: E731
    line_x = np.nonzero(np.isclose(X[:, 0], 5) & np.isclose(X[:, 2], 0))[0].tolist()
    line_y = np.nonzero(np.isclose(X[:, 1], 5) & np.isclose(X[:, 2], 0))[0].tolist()
    bcs = [
        BC("displacement", [], [0.0, None, None], nodes=on(0, 0.0)),
        BC("displacement", [], [None, 0.0, None], nodes=on(1, 0.0)),
        BC("displacement", [], [0.0, 0.0, None], nodes=on(2, 0.0)),   # 中立面（逆対称）
        BC("displacement", [], [0.0, None, None], nodes=on(0, 5.0)),
        BC("displacement", [], [None, 0.0, None], nodes=on(1, 5.0)),
        BC("displacement", [], [None, None, 0.0], nodes=line_x + line_y),
        BC("pressure", [4], [-500.0]),                                  # 下面 z=-t/2 に P=-500
    ]
    r = solve(m, mat, bcs, log=lambda *a: None)
    pt = lambda x, y, z: int(np.argmin(np.linalg.norm(X - [x, y, z], axis=1)))  # noqa: E731
    p1, p2, p3 = pt(0, 0, -0.5), pt(5, 0, -0.5), pt(0, 0, -0.5)
    return m, [Check("Ansys VMC1 周辺固定板", "中央たわみ UZ(1)", -0.0172, float(r.u[p1, 2]), "in", "Ansys VMC1", 0.984),
               Check("Ansys VMC1 周辺固定板", "固定辺 SX(2)", -32124.0, float(r.stress[p2, 0]), "psi", "Ansys VMC1", 0.966),
               Check("Ansys VMC1 周辺固定板", "中央 SX(3)", 14465.0, float(r.stress[p3, 0]), "psi", "Ansys VMC1", 1.000)]


def cantilever(n=10):
    """片持ち梁の先端荷重（ティモシェンコ梁理論）"""
    L, b, h, Pz = 200.0, 20.0, 20.0, 1000.0
    mat = Material("steel", 200000.0, 0.3, 7850.0)
    m = structured_box(L, b, h, n, max(2, n // 5), max(2, n // 5)).to_quadratic()
    r = solve(m, mat, [BC("fixed", [0]), BC("force", [1], [0, 0, -Pz])], log=lambda *a: None)
    I = b * h**3 / 12
    G = mat.E / (2 * (1 + mat.nu))
    d = Pz * L**3 / (3 * mat.E * I) + 1.2 * Pz * L / (G * b * h)
    tip = -r.u[np.isclose(m.nodes[:, 0], L), 2].mean()
    return m, Check("片持ち梁", "先端たわみ", d, float(tip), "mm", "梁理論")


def face_where(mesh, axis, value, sign, tol=1e-6):
    """座標 axis=value の平面で、法線が sign 方向の面番号"""
    for f in mesh.face_info():
        if f["flat"] and abs(f["centroid"][axis] - value) < tol * max(1.0, abs(value)) \
                and f["normal"][axis] * sign > 0.99:
            return f["id"]
    raise ValueError("face not found")


def hole_plate(size=1.5):
    """穴あき板の引張（gmsh の自由四面体メッシュ＝Ansys Mechanical と同じ作り方）

    W=50, d=10 (d/W=0.2), t=5, 荷重 5000 N。応力集中係数（正味断面基準）は
    Peterson の式 Kt = 3 - 3.14(d/W) + 3.667(d/W)^2 - 1.527(d/W)^3
    """
    from .geometry import build_mesh
    L, W, t, d, F = 200.0, 50.0, 5.0, 10.0, 5000.0
    mat = Material("steel", 200000.0, 0.3, 7850.0)
    m = build_mesh({"kind": "plate_hole", "lx": L, "ly": W, "t": t, "d": d}, size=size, order=2,
                   log=lambda *a: None)
    x0, x1 = face_where(m, 0, 0.0, -1), face_where(m, 0, L, +1)
    # 左端は x 方向のみ拘束（ポアソン収縮を妨げない）、剛体運動は最小限の拘束で止める
    X = m.nodes
    corner = int(np.argmin(np.linalg.norm(X - [0, 0, 0], axis=1)))
    corner2 = int(np.argmin(np.linalg.norm(X - [0, W, 0], axis=1)))
    bcs = [BC("displacement", [x0], [0.0, None, None]),
           BC("displacement", [], [None, 0.0, 0.0], nodes=[corner]),
           BC("displacement", [], [None, None, 0.0], nodes=[corner2]),
           BC("force", [x1], [F, 0.0, 0.0])]
    r = solve(m, mat, bcs, log=lambda *a: None)
    q = d / W
    kt = 3 - 3.14 * q + 3.667 * q**2 - 1.527 * q**3
    target = kt * F / ((W - d) * t)
    near = np.linalg.norm(X[:, :2] - [L / 2, W / 2], axis=1) < d / 2 + 1e-6 * L
    smax = float(r.stress[near, 0].max())
    return m, Check("穴あき板の引張（gmsh自由メッシュ）", "穴縁 最大σx", target, smax, "MPa", "Peterson 式")


# ---------------------------------------------------------------------------- 実行

def run_all(level="normal"):
    """level: 'coarse' | 'normal' | 'fine'"""
    k = {"coarse": 0.5, "normal": 1.0, "fine": 2.0}[level]
    out = []
    cases = [
        (lambda: le1(int(12 * k))),
        (lambda: le10(int(6 * k))),
        (lambda: lame(int(6 * k))),
        (lambda: vmc1(*{"coarse": (5, 1), "normal": (10, 2), "fine": (20, 4)}[level])),
        (lambda: cantilever(int(10 * k))),
    ]
    from .geometry import gmsh_available
    if gmsh_available():
        cases.append(lambda: hole_plate({"coarse": 2.5, "normal": 1.25, "fine": 0.6}[level]))
    for f in cases:
        t0 = time.time()
        m, chk = f()
        for c in (chk if isinstance(chk, list) else [chk]):
            c.dof = 3 * m.n_nodes
            c.time = time.time() - t0
            out.append(c)
    return out


def table(checks) -> str:
    lines = [f"{'問題':<22} {'評価量':<18} {'目標値':>10} {'本ソフト':>10} {'比':>7} {'Ansys比':>8}  自由度",
             "-" * 92]
    for c in checks:
        a = f"{c.ansys:.3f}" if c.ansys is not None else "   -"
        lines.append(f"{c.name:<22} {c.quantity:<18} {c.target:>10.4g} {c.value:>10.4g} "
                     f"{c.ratio:>7.3f} {a:>8}  {c.dof:,}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="normal", choices=["coarse", "normal", "fine"])
    ap.add_argument("--converge", action="store_true")
    a = ap.parse_args(argv)
    levels = ["coarse", "normal", "fine"] if a.converge else [a.level]
    for lv in levels:
        print(f"\n=== メッシュ: {lv} ===")
        print(table(run_all(lv)))


if __name__ == "__main__":
    main()

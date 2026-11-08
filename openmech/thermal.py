"""熱解析（Ansys Mechanical の「定常熱」「過渡熱」に相当）と、その温度を使った熱応力。

境界条件（BC.kind、値は SI 単位で入力）:
  temperature      温度 [°C]                       values=[T]
  heat_flux        熱流束 [W/m²]（面に入る向き 正） values=[q]
  heat_flow        熱流量 [W]（面全体に一様）       values=[Q]
  convection       対流  values=[h W/(m²·K), 周囲温度 °C]
  radiation        放射（周囲へ） values=[放射率 ε, 周囲温度 °C]
  heat_generation  内部発熱 [W/m³]（bodies で対象ボディ、空なら全体） values=[q''']
  insulation       断熱（何もしない＝既定）
過渡: 初期温度・終了時間・ステップ数（後退オイラー）。温度は Ansys と同じく節点の自由度 1 つ。

内部の単位は mm, t, s, mJ（W/(m·K) は数値そのまま mJ/(s·mm·K)）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh
from .solver import BC, _dN_dx, _quadrature, factorize, node_points, shape_dN_dxi
from .surface import surface_sets

THERMAL_KINDS = ("temperature", "heat_flux", "heat_flow", "convection", "radiation", "heat_generation", "insulation")
SIGMA_SB = 5.670374419e-11      # mJ/(s·mm²·K⁴)
K0 = 273.15


@dataclass
class ThermalSettings:
    transient: bool = False
    init_temp: float = 22.0       # 初期温度 [°C]（過渡）
    end_time: float = 1.0         # [s]
    n_steps: int = 10
    max_iter: int = 30            # 放射の非線形の反復
    tol: float = 1e-6

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d):
        s = ThermalSettings()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


@dataclass
class ThermalResult:
    mesh: Mesh
    T: np.ndarray              # (N,) 温度 [°C]
    flux: np.ndarray           # (N,3) 熱流束 [W/m²]
    time: float = 0.0
    info: dict = field(default_factory=dict)
    reactions: dict = field(default_factory=dict)   # BC 番号 → 熱流量 [W]（温度指定の面から入る量）

    # 構造の Result と同じように表示できるよう、同じ名前の属性を持たせる
    @property
    def u(self):
        return np.zeros((self.mesh.n_nodes, 3))

    def fields(self):
        q = self.flux
        return {"温度 [°C]": self.T, "全熱流束 [W/m²]": np.linalg.norm(q, axis=1),
                "方向熱流束 X [W/m²]": q[:, 0], "方向熱流束 Y [W/m²]": q[:, 1], "方向熱流束 Z [W/m²]": q[:, 2]}

    def summary(self):
        return {"最高温度 [°C]": float(self.T.max()), "最低温度 [°C]": float(self.T.min()),
                "最大熱流束 [W/m²]": float(np.linalg.norm(self.flux, axis=1).max())}


def _mat_lists(mesh, material, elem_materials):
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    k = np.array([float(getattr(m, "conductivity", 0.0) or 0.0) for m in mats])
    if np.any(k[np.unique(elem_mat)] <= 0):
        bad = [m.name for i, m in enumerate(mats) if k[i] <= 0 and i in set(np.unique(elem_mat).tolist())]
        raise ValueError("熱伝導率が設定されていない材料があります: " + ", ".join(bad))
    rc = np.array([float(m.density) * float(getattr(m, "specific_heat", 0.0) or 0.0) * 1e-6 for m in mats])
    return mats, elem_mat, k, rc


def conductivity_matrix(mesh, k_e):
    """K_T = ∫ ∇Nᵀ k ∇N dV（k は要素ごとの等方熱伝導率）"""
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature(nn)
    X = mesh.nodes[mesh.elems]
    Ke = np.zeros((mesh.n_elems, nn, nn))
    for L, w in zip(pts, wts):
        dNdx, detJ = _dN_dx(X, shape_dN_dxi(L, nn))
        Ke += np.einsum("eai,ebi->eab", dNdx, dNdx) * (np.abs(detJ) * w * k_e)[:, None, None]
    r = np.repeat(mesh.elems, nn, axis=1).ravel()
    c = np.tile(mesh.elems, (1, nn)).ravel()
    return sp.csr_matrix((Ke.ravel(), (r, c)), shape=(mesh.n_nodes, mesh.n_nodes))


def capacity_matrix(mesh, rc_e):
    """C = ∫ ρc N Nᵀ dV（整合）"""
    from .dynamics import _tet_mass_ref
    nn = mesh.elems.shape[1]
    C = _tet_mass_ref(nn)
    me = rc_e * np.abs(mesh.element_volumes())
    r = np.repeat(mesh.elems, nn, axis=1).ravel()
    c = np.tile(mesh.elems, (1, nn)).ravel()
    return sp.csr_matrix(((me[:, None, None] * C[None]).ravel(), (r, c)), shape=(mesh.n_nodes, mesh.n_nodes))


def _gen_vector(mesh, q_e):
    """∫ q N dV（要素ごとの発熱 q [mJ/(s·mm³)]）"""
    from .solver import _quadrature_full, shape_N
    nn = mesh.elems.shape[1]
    pts, wts = _quadrature_full(nn)
    X = mesh.nodes[mesh.elems]
    F = np.zeros(mesh.n_nodes)
    for L, w in zip(pts, wts):
        _, detJ = _dN_dx(X, shape_dN_dxi(L, nn))
        contrib = (w * np.abs(detJ) * q_e)[:, None] * shape_N(L, nn)[None, :]
        np.add.at(F, mesh.elems.ravel(), contrib.ravel())
    return F


def nodal_flux(mesh, T, k_e):
    """節点の熱流束 q = -k ∇T（節点平均）→ [W/m²]"""
    nn = mesh.elems.shape[1]
    Lp = node_points(nn)
    X = mesh.nodes[mesh.elems]
    Te = T[mesh.elems]
    acc = np.zeros((mesh.n_nodes, 3))
    for a in range(nn):
        dNdx, _ = _dN_dx(X, shape_dN_dxi(Lp[a], nn))
        g = np.einsum("eai,ea->ei", dNdx, Te)
        q = -k_e[:, None] * g
        for c in range(3):
            np.add.at(acc[:, c], mesh.elems[:, a], q[:, c])
    cnt = np.maximum(np.bincount(mesh.elems.ravel(), minlength=mesh.n_nodes), 1)
    return acc / cnt[:, None] * 1000.0          # mJ/(s·mm²) → W/m²


def _assemble_bcs(mesh, bcs, elem_mat_body):
    """熱の境界条件 → 温度指定 {node: T}、定数の熱入力 F、対流の行列 H と ベクトル、放射の面"""
    n = mesh.n_nodes
    fixed = {}
    F = np.zeros(n)
    H = sp.csr_matrix((n, n))
    rad = []
    for i, b in enumerate(bcs):
        v = [0.0 if x is None else float(x) for x in (list(b.values) + [0.0, 0.0])]
        if b.kind == "temperature":
            for nd in mesh.face_nodes(b.faces) if b.faces else np.asarray(b.nodes, int):
                fixed[int(nd)] = v[0]
        elif b.kind == "heat_flux":
            for ss in surface_sets(mesh, b.faces):
                idx, w = ss.lumped(v[0] * 1e-3)
                np.add.at(F, idx, w)
        elif b.kind == "heat_flow":
            sets = surface_sets(mesh, b.faces)
            A = sum(ss.area for ss in sets)
            for ss in sets:
                idx, w = ss.lumped(v[0] * 1000.0 / max(A, 1e-300))
                np.add.at(F, idx, w)
        elif b.kind == "convection":
            h = v[0] * 1e-3
            for ss in surface_sets(mesh, b.faces):
                H = H + ss.matrix(n, h)
                idx, w = ss.lumped(h * v[1])
                np.add.at(F, idx, w)
        elif b.kind == "radiation":
            for ss in surface_sets(mesh, b.faces):
                rad.append((ss, v[0], v[1]))
        elif b.kind == "heat_generation":
            q = v[0] * 1e-6
            bodies = set(int(x) for x in (b.bodies or []))
            if elem_mat_body is not None and bodies:
                q_e = np.where(np.isin(elem_mat_body, list(bodies)), q, 0.0)
            else:
                q_e = np.full(mesh.n_elems, q)
            F += _gen_vector(mesh, q_e)
    return fixed, F, H.tocsr(), rad


def _radiation(rad, T, n):
    """放射の熱入力と接線行列（温度 T [°C] で線形化）。 q = εσ (T∞⁴ - T⁴)"""
    F = np.zeros(n)
    Kr = sp.csr_matrix((n, n))
    for ss, eps, Tinf in rad:
        Tq = np.einsum("qk,sk->sq", ss.N, T[ss.conn]) + K0       # 積分点の温度 [K]
        qv = eps * SIGMA_SB * ((Tinf + K0) ** 4 - Tq ** 4)
        dq = 4 * eps * SIGMA_SB * Tq ** 3                         # -∂q/∂T
        w = np.einsum("sq,qk->sk", ss.dA * qv, ss.N)
        np.add.at(F, ss.conn.ravel(), w.ravel())
        Kr = Kr + ss.matrix(n, dq)
    return F, Kr.tocsr()


def solve_thermal(mesh: Mesh, material, bcs: list[BC], settings: ThermalSettings | None = None,
                  elem_materials=None, log=print, progress=None) -> list[ThermalResult]:
    """定常または過渡の熱解析。結果（過渡なら各ステップ）のリストを返す"""
    t0 = time.time()
    s = settings or ThermalSettings()
    bcs = [b for b in bcs if b.kind in THERMAL_KINDS]
    mats, elem_mat, k_m, rc_m = _mat_lists(mesh, material, elem_materials)
    k_e, rc_e = k_m[elem_mat], rc_m[elem_mat]
    K = conductivity_matrix(mesh, k_e)
    fixed, F0, H, rad = _assemble_bcs(mesh, bcs, getattr(mesh, "elem_body", None))
    n = mesh.n_nodes
    if not fixed and H.nnz == 0 and not rad and not s.transient:
        raise ValueError("定常熱解析には、温度・対流・放射のどれかが少なくとも 1 つ必要です（温度が決まりません）")
    cd = np.fromiter(fixed.keys(), dtype=np.int64)
    cv = np.fromiter(fixed.values(), dtype=float)
    free = np.ones(n, bool)
    free[cd] = False
    A0 = (K + H).tocsr()
    C = capacity_matrix(mesh, rc_e) if s.transient else None
    if s.transient and np.any(rc_m[np.unique(elem_mat)] <= 0):
        raise ValueError("過渡熱解析には材料の密度と比熱が必要です")
    t_asm = time.time() - t0
    log(f"熱解析（{'過渡' if s.transient else '定常'}）: 節点 {n:,} / 温度指定 {len(cd):,} 節点"
        + (" / 放射あり（非線形）" if rad else ""))

    def step(T_prev, dt):
        """1 ステップ（定常は dt=None）。放射があればニュートン法"""
        T = T_prev.copy()
        T[cd] = cv
        A_lin = A0 + (C / dt if dt else 0)
        b_lin = F0 + (C @ T_prev / dt if dt else 0)
        fac = None
        for it in range(s.max_iter if rad else 1):
            if rad:
                Fr, Kr = _radiation(rad, T, n)
                A = (A_lin + Kr).tocsr()
                r = b_lin + Fr - A_lin @ T            # 残差 → ΔT
                Aff = A[free][:, free]
                dT = factorize(Aff.tocsc(), lambda *a: None)(r[free])
                T[free] += dT
                if np.abs(dT).max() <= s.tol * max(1.0, np.abs(T).max()):
                    break
            else:
                if fac is None:
                    Aff = A_lin[free][:, free]
                    fac = factorize(Aff.tocsc(), lambda *a: None)
                rhs = b_lin[free] - A_lin[free][:, cd] @ cv
                T[free] = fac(rhs)
        return T

    out = []
    reac_bcs = [(i, mesh.face_nodes(b.faces)) for i, b in enumerate(bcs) if b.kind == "temperature" and b.faces]

    def make(T, t, dt=None, T_prev=None):
        # 温度指定の面から入る熱流量 = (K T - F) の拘束節点の和（Ansys の反力の熱流量と同じ向き: 入る方が正）
        R = A0 @ T - F0
        if rad:
            R = R - _radiation(rad, T, n)[0]
        if dt:
            R = R + C @ (T - T_prev) / dt
        reac = {i: float(R[nd].sum()) / 1000.0 for i, nd in reac_bcs}
        r = ThermalResult(mesh, T.copy(), nodal_flux(mesh, T, k_e), t, {}, reac)
        return r

    if not s.transient:
        T = np.full(n, float(s.init_temp))
        T = step(T, None)
        out.append(make(T, 1.0))
    else:
        T = np.full(n, float(s.init_temp))
        T[cd] = cv
        dt = s.end_time / max(1, int(s.n_steps))
        for k in range(int(s.n_steps)):
            Tn = step(T, dt)
            out.append(make(Tn, (k + 1) * dt, dt, T))
            T = Tn
            if progress:
                progress((k + 1) / s.n_steps, f"時間 {(k + 1) * dt:g} s")
    t_all = time.time() - t0
    for r in out:
        r.info = {"解析": "過渡熱" if s.transient else "定常熱", "自由度数": int(free.sum()), "節点数": n,
                  "要素数": mesh.n_elems, "組立時間 [s]": round(t_asm, 2), "合計時間 [s]": round(t_all, 2),
                  "時間 [s]": r.time, **r.summary()}
    log(f"  完了: 最高 {out[-1].T.max():.4g} °C / 最低 {out[-1].T.min():.4g} °C（{t_all:.1f} s）")
    return out


def thermal_stress(mesh, material, bcs, T_nodes, elem_materials=None, log=print, ref_temp=None):
    """熱解析の温度（節点）から熱応力の静解析。要素の温度は節点平均（Ansys の「熱条件（インポート）」と同じ）"""
    from .materials import MaterialState  # noqa: F401
    from .solver import solve
    Te = T_nodes[mesh.elems].mean(axis=1)
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    # 要素ごとに温度が違う → 温度を 0.01 °C に丸めて材料状態を作る（同じ温度はまとめる）
    key = np.round(Te, 2)
    states, idx = [], np.empty(mesh.n_elems, dtype=np.int64)
    cache = {}
    for e in range(mesh.n_elems):
        kk = (int(elem_mat[e]), float(key[e]))
        if kk not in cache:
            m = mats[kk[0]]
            st = m.state(T=kk[1]) if hasattr(m, "state") else m
            cache[kk] = len(states)
            states.append(st)
        idx[e] = cache[kk]
    log(f"熱応力: 温度の種類 {len(states):,}（{Te.min():.4g} – {Te.max():.4g} °C）")
    sbcs = [b for b in bcs if b.kind not in THERMAL_KINDS and b.kind != "thermal"]
    r = solve(mesh, material, sbcs, log=log, elem_materials=(states, idx))
    r.info["解析"] = "熱応力（熱解析の温度）"
    return r

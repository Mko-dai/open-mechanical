"""静的構造解析（Ansys Mechanical の「解析設定」に対応）。

- ステップ制御: ステップ数・各ステップ終了時間・サブステップ（初期/最小/最大）・自動時間ステップ
- ソルバー制御: ソルバータイプ・弱いバネ・大変形（オン/オフ）
- 非線形制御: ニュートン・ラフソン、力の収束（許容値）、変位の収束、ラインサーチ、最大反復回数
- 出力制御: 結果を保存する時点（全ステップ/最終のみ）
- 接触（結合・分離なし・摩擦なし・粗い・摩擦あり）

大変形は Total Lagrangian（St.Venant-Kirchhoff 材料）。応力はコーシー応力で出力します。
圧力は変形前の向きで作用します（非追従荷重）。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp

from .contact import (ContactRegion, build_pairs, commit_states, penetration_ok, contact_results, contact_system,
                      repair_in_step, update_states)
from .materials import Material
from .mesh import Mesh
from .solver import (BC, SUPPORT_KINDS, GAUSS4, GAUSS4_W, NODE_L10, NODE_L4, Result, _B_matrix, _dN_dx, _D_of,
                     _linear_solve, _quadrature, _rel_res, assemble_stiffness, build_load_and_constraints,
                     elasticity_matrix, nodal_stress, shape_dN_dxi)


@dataclass
class AnalysisSettings:
    # ステップ制御
    n_steps: int = 1
    step_end_times: list = field(default_factory=lambda: [1.0])
    auto_time_stepping: str = "auto"      # auto（プログラム制御）/ on / off
    initial_substeps: int = 1
    min_substeps: int = 1
    max_substeps: int = 10
    # ソルバー制御
    solver_type: str = "auto"             # auto / direct（直接法）/ amg（反復法）
    n_cores: int = 0                      # 使用コア数（0 = すべて）
    weak_springs: bool = False
    large_deflection: bool = False
    # 非線形制御
    newton: str = "auto"                  # auto / full
    force_convergence: float = 0.005      # 0.5%（Ansys 既定）
    displacement_convergence: float = 0.0  # 0 = 使わない
    line_search: str = "auto"             # auto / on / off
    max_iterations: int = 50
    penetration_tolerance: float = 0.1    # 要素サイズに対する比（拡張ラグランジュ）
    # 弧長法（Ansys の ARCLEN）: 荷重係数 λ も未知数にして、座屈・飛び移りの極限点を越えて経路を追う
    arc_length: bool = False
    arc_initial: float = 0.1              # 最初の荷重係数の増分（与えた荷重に対する比）
    arc_max_steps: int = 60               # 追う点の数の上限
    arc_growth: float = 50.0              # 弧長を最初の何倍まで伸ばしてよいか（大きいと点が粗くなる）
    arc_max_load: float = 1.0             # 荷重係数がこれに達したら終了
    arc_max_disp: float = 0.0             # 最大変位 [mm] がこれに達したら終了（0 = 使わない）
    # 出力制御
    store_results: str = "all"            # all（全ステップ）/ last（最終のみ）
    calc_stress: bool = True
    calc_contact: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        s = AnalysisSettings()
        for k, v in (d or {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def end_times(self):
        t = list(self.step_end_times)[: self.n_steps]
        while len(t) < self.n_steps:
            t.append((t[-1] if t else 0.0) + 1.0)
        return [float(x) for x in t]


class ConvergenceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- 大変形（Total Lagrangian）

def _tl_BL(F, dNdX):
    """非線形ひずみ-変位マトリクス B_L (E,6,3nn)"""
    E, nn, _ = dNdX.shape
    B = np.zeros((E, 6, nn, 3))
    for k in range(3):
        Fk0, Fk1, Fk2 = F[:, k, 0][:, None], F[:, k, 1][:, None], F[:, k, 2][:, None]
        d0, d1, d2 = dNdX[..., 0], dNdX[..., 1], dNdX[..., 2]
        B[:, 0, :, k] = Fk0 * d0
        B[:, 1, :, k] = Fk1 * d1
        B[:, 2, :, k] = Fk2 * d2
        B[:, 3, :, k] = Fk0 * d1 + Fk1 * d0
        B[:, 4, :, k] = Fk1 * d2 + Fk2 * d1
        B[:, 5, :, k] = Fk0 * d2 + Fk2 * d0
    return B.reshape(E, 6, nn * 3)


def _green(F):
    C = np.einsum("eki,ekj->eij", F, F)
    Eg = 0.5 * (C - np.eye(3))
    return np.stack([Eg[:, 0, 0], Eg[:, 1, 1], Eg[:, 2, 2],
                     2 * Eg[:, 0, 1], 2 * Eg[:, 1, 2], 2 * Eg[:, 0, 2]], axis=1)


def tl_internal(mesh: Mesh, D, elem_mat, u, chunk=20000):
    """内力ベクトルと接線剛性（Total Lagrangian）"""
    nn = mesh.elems.shape[1]
    ndof = 3 * mesh.n_nodes
    pts, wts = _quadrature(nn)
    dN = [shape_dN_dxi(L, nn) for L in pts]
    U = u.reshape(-1, 3)
    fint = np.zeros(ndof)
    K = sp.csr_matrix((ndof, ndof))
    loc = np.arange(3 * nn)
    ii, jj = np.meshgrid(loc, loc, indexing="ij")
    for s in range(0, mesh.n_elems, chunk):
        el = mesh.elems[s:s + chunk]
        X = mesh.nodes[el]
        Ue = U[el]
        De = _D_of(D, elem_mat, slice(s, s + chunk))
        Ke = np.zeros((len(el), 3 * nn, 3 * nn))
        fe = np.zeros((len(el), 3 * nn))
        for g, w in enumerate(wts):
            dNdX, detJ = _dN_dx(X, dN[g])
            H = np.einsum("eak,eal->ekl", Ue, dNdX)
            F = np.eye(3) + H
            Ev = _green(F)
            S = Ev @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, Ev)
            B = _tl_BL(F, dNdX)
            wd = w * detJ
            fe += np.einsum("e,eki,ek->ei", wd, B, S)
            if De.ndim == 2:
                Ke += np.einsum("e,eki,kl,elj->eij", wd, B, De, B, optimize=True)
            else:
                Ke += np.einsum("e,eki,ekl,elj->eij", wd, B, De, B, optimize=True)
            Sm = np.stack([S[:, [0, 3, 5]], S[:, [3, 1, 4]], S[:, [5, 4, 2]]], axis=1)
            G = np.einsum("eai,eij,ebj->eab", dNdX, Sm, dNdX) * wd[:, None, None]
            for k in range(3):
                Ke[:, k::3, k::3] += G
        dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(el), -1)
        np.add.at(fint, dofs.ravel(), fe.ravel())
        K = K + sp.csr_matrix((Ke.ravel(), (dofs[:, ii].ravel(), dofs[:, jj].ravel())), shape=(ndof, ndof))
    return fint, K


def tl_cauchy_nodal(mesh: Mesh, u, D, elem_mat, chunk=50000):
    """節点のコーシー応力とグリーン・ラグランジュひずみ"""
    nn = mesh.elems.shape[1]
    from .solver import node_points
    Lnodes = node_points(nn)
    U = u.reshape(-1, 3)
    acc_s = np.zeros((mesh.n_nodes, 6))
    acc_e = np.zeros((mesh.n_nodes, 6))
    cnt = np.zeros(mesh.n_nodes)
    for s in range(0, mesh.n_elems, chunk):
        el = mesh.elems[s:s + chunk]
        X = mesh.nodes[el]
        Ue = U[el]
        De = _D_of(D, elem_mat, slice(s, s + chunk))
        for a in range(nn):
            dNdX, _ = _dN_dx(X, shape_dN_dxi(Lnodes[a], nn))
            F = np.eye(3) + np.einsum("eak,eal->ekl", Ue, dNdX)
            Ev = _green(F)
            S = Ev @ De.T if De.ndim == 2 else np.einsum("eij,ej->ei", De, Ev)
            Sm = np.stack([S[:, [0, 3, 5]], S[:, [3, 1, 4]], S[:, [5, 4, 2]]], axis=1)
            J = np.linalg.det(F)
            sig = np.einsum("eik,ekl,ejl->eij", F, Sm, F) / J[:, None, None]
            sv = np.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2], sig[:, 0, 1], sig[:, 1, 2], sig[:, 0, 2]], 1)
            np.add.at(acc_s, el[:, a], sv)
            np.add.at(acc_e, el[:, a], Ev)
            np.add.at(cnt, el[:, a], 1)
    cnt = np.maximum(cnt, 1)[:, None]
    return acc_s / cnt, acc_e / cnt


# ---------------------------------------------------------------------------- ドライバー

def _step_factor(bc: BC, t: float, ends: list[float]) -> float:
    """時刻 t での荷重の倍率（0 から各ステップの値へ直線で増える＝ランプ）"""
    fac = list(getattr(bc, "step_factors", None) or [1.0] * len(ends))
    while len(fac) < len(ends):
        fac.append(fac[-1] if fac else 1.0)
    times = [0.0] + list(ends)
    vals = [0.0] + [float(f) for f in fac[: len(ends)]]
    return float(np.interp(t, times, vals))


def solve_static(mesh: Mesh, material: Material, bcs: list[BC], contacts: list[ContactRegion] | None = None,
                 settings: AnalysisSettings | None = None, elem_materials=None, log=print,
                 progress=None, contact_tolerance: float = 0.0, dynamic=None) -> list[Result]:
    """静的構造解析（接触・大変形・複数ステップ対応）。各ステップ終了時の Result のリストを返す。

    dynamic（DynamicSettings）を渡すと、同じ非線形の解き方に慣性力と減衰力を加えた過渡構造解析（陰解法・
    Newmark 法）になる。接触・塑性・大変形・リモート境界条件がそのまま使え、保存した時刻ごとの Result を返す。
    """
    t_start = time.time()
    settings = settings or AnalysisSettings()
    dyn = dynamic
    if dyn is not None:
        import copy
        settings = copy.copy(settings)
        settings.n_steps, settings.step_end_times = 1, [float(dyn.end_time)]
        settings.large_deflection = bool(settings.large_deflection or dyn.large_deflection)
    contacts = [c for c in (contacts or []) if not c.suppressed]
    if dyn is None and not any(b.kind in SUPPORT_KINDS for b in bcs):
        raise ValueError("拘束（固定支持 または 変位）がありません。少なくとも1つ設定してください。")

    # 材料
    if elem_materials is not None:
        mats, elem_mat = elem_materials
        elem_mat = np.asarray(elem_mat, dtype=np.int64)
    else:
        mats, elem_mat = [material], np.zeros(mesh.n_elems, dtype=np.int64)
    from .solver import material_D, material_eps_th, thermal_load_vector
    D = np.stack([material_D(m) for m in mats])
    eps_th_m = np.stack([material_eps_th(m) for m in mats])
    has_th = bool(np.any(eps_th_m))
    hyper = any(getattr(m, "hyperelastic", False) for m in mats)
    plastic = any(getattr(m, "plastic", False) for m in mats) or hyper
    rho_e = np.array([m.density_t_mm3 for m in mats])[elem_mat]
    ndof = 3 * mesh.n_nodes

    # 荷重・拘束（境界条件ごと）
    from .remote import REMOTE_KINDS
    per_bc = [build_load_and_constraints(mesh, mats[0], [bc], rho_e=rho_e, allow_oblique=True) for bc in bcs]
    nb = ndof
    ex = None
    from .solver import oblique_support_faces
    if any(b.kind in REMOTE_KINDS for b in bcs) or oblique_support_faces(mesh, bcs):
        # リモートポイント・ばね・弾性支持・円筒支持など: パイロットの 6 自由度を後ろに足し、
        # 多点拘束（剛体・RBE3）は各反復の連立方程式で消去する（remote.reduce_system）
        from .remote import extra_terms
        vol0 = np.abs(mesh.element_volumes())
        h0 = float(np.cbrt(np.median(vol0) * 6 * np.sqrt(2))) if len(vol0) else 1.0
        kref = float(np.mean([m.E for m in mats])) * h0
        acc = next((np.asarray(b.values, float) for b in bcs if b.kind == "gravity"), None)
        ex = extra_terms(mesh, bcs, kref, acc=acc)
        ndof = ex.n
        for note in ex.notes:
            log(note)
        for i, b in enumerate(bcs):
            if b.kind in REMOTE_KINDS:
                per_bc[i] = (ex.Fvec_of(i), dict(ex.fixed_by.get(i, {})))
    per_bc = [(np.r_[Fi, np.zeros(ndof - len(Fi))] if len(Fi) < ndof else Fi, fx) for Fi, fx in per_bc]
    ends = settings.end_times()

    th_bcs = [bc for bc in bcs if bc.kind == "thermal"]
    F_th = thermal_load_vector(mesh, D, elem_mat, eps_th_m) if (has_th and not plastic) else None

    def sfac(bc, t):
        if dyn is not None:      # 過渡: ステップ荷重、または ramp_time までに直線で増やす
            return 1.0 if dyn.ramp_time <= 0 else min(1.0, t / dyn.ramp_time)
        return _step_factor(bc, t, ends)

    def th_factor(t):
        """熱ひずみの倍率（熱条件のステップ倍率。熱条件が無く材料温度だけなら 1 で一定）"""
        if not th_bcs:
            return 1.0
        return sfac(th_bcs[0], t)

    def loads_at(t):
        F = np.zeros(ndof)
        fixed = {}
        if F_th is not None:
            F[:nb] += th_factor(t) * F_th
        for bc, (Fi, fx) in zip(bcs, per_bc):
            f = sfac(bc, t)
            if bc.kind in ("fixed", "symmetry", "frictionless"):
                for d in fx:
                    fixed[d] = 0.0
            elif bc.kind in ("displacement", "remote_displacement", "submodel"):
                for d, v in fx.items():
                    fixed[d] = f * v
                if bc.kind == "remote_displacement":
                    F += f * Fi
            else:
                F += f * Fi
        return F, fixed

    _, fixed0 = loads_at(ends[0])
    cdofs = np.array(sorted(fixed0), dtype=np.int64)
    free = np.ones(ndof, bool)
    free[cdofs] = False

    # 接触ペア
    E_node = np.zeros(mesh.n_nodes)
    Ee = np.array([m.E for m in mats])[elem_mat]
    np.maximum.at(E_node, mesh.elems.ravel(), np.repeat(Ee, mesh.elems.shape[1]))
    pairs = build_pairs(mesh, contacts, E_node, contact_tolerance,
                        large_deflection=bool(settings.large_deflection or hyper)) if contacts else []
    if any(p.finite for p in pairs):
        log("接触: 有限すべり（収束したサブステップごとに、変形後の位置で接触相手を決め直します）")
    if contacts:
        log(f"接触: {len(contacts)} 領域 / 接触節点 {sum(len(p.slave) for p in pairs):,}")
    nonlinear_contact = any(c.nonlinear for c in contacts)
    friction = any(c.type == "frictional" for c in contacts)
    al = any(c.resolved_formulation() == "augmented_lagrange" for c in contacts)
    LD = settings.large_deflection or hyper
    if hyper and not settings.large_deflection:
        log("超弾性材料あり: 大変形をオンにして解きます")
    vol = np.abs(mesh.element_volumes())
    h_char = float(np.cbrt(np.median(vol) * 6 * np.sqrt(2))) if len(vol) else 1.0
    pen_tol = settings.penetration_tolerance * h_char

    if LD and has_th and not plastic:
        raise ValueError("熱ひずみと大変形の組み合わせ（弾性材料のみ）にはまだ対応していません。")
    ps = None
    if plastic:
        from .plasticity import PlasticSolid
        ps = PlasticSolid(mesh, mats, elem_mat, D, eps_th_m)
        kinds = (["弾塑性（ミーゼス）"] if any(m.plasticity != "none" for m in mats) else []) \
            + (["クリープ（Norton）"] if any(getattr(m, "creeping", False) for m in mats) else []) \
            + (["超弾性"] if hyper else [])
        log("非線形材料あり: " + "・".join(kinds) + ("＋大変形" if LD else "") + "で解きます")
    K_lin = None if (LD or plastic) else assemble_stiffness(mesh, D, elem_mat=elem_mat)
    k_weak = 0.0
    if settings.weak_springs:
        diagK = (K_lin.diagonal() if K_lin is not None else assemble_stiffness(mesh, D, elem_mat=elem_mat).diagonal())
        k_weak = 1e-8 * float(np.mean(np.abs(diagK)))
    W = sp.identity(ndof, format="csr") * k_weak if k_weak > 0 else None
    if ex is not None:
        exk = ex.K()
        if exk.nnz:
            W = exk if W is None else (W + exk)
    from .parallel import set_threads
    set_threads(settings.n_cores)
    # 接触の剛性で条件が悪くなるため、非線形では既定で直接法
    method = "direct" if settings.solver_type == "auto" else settings.solver_type
    if settings.solver_type == "auto" and ex is None and not hyper:
        # 多点拘束が無ければ、大きなモデルは反復法（AMG）に任せる（接触のペナルティ剛性があっても対称正定値）。
        # 反復法が収束せず直接法に切り替わったら、この解析の残りは最初から直接法にする（amg_failed）
        method = "auto"
    amg_failed = [False]
    import os as _os
    _slog = _os.environ.get("OPENMECH_SOLVER_LOG", "") == "1"

    def lin_log(msg):
        if "直接法に切り替えます" in str(msg):
            amg_failed[0] = True
        if _slog:
            log(msg)
    _fi = np.nonzero(free)[0]
    amg_coords = np.column_stack([mesh.nodes[np.minimum(_fi // 3, mesh.n_nodes - 1)], _fi % 3])

    cur_t = [0.0]

    def internal_base(u):
        if ps is not None:
            return ps.internal(u, th_factor(cur_t[0]), large=LD)
        if LD:
            return tl_internal(mesh, D, elem_mat, u)
        return K_lin @ u, K_lin

    if ex is None:
        internal = internal_base
    else:
        from .remote import pad as _pad

        def internal(u):
            fi, Kb = internal_base(u[:nb])
            return np.r_[fi, np.zeros(ndof - nb)], _pad(Kb, ndof)

    nonlinear = LD or nonlinear_contact or al or plastic
    # 過渡（Newmark 法）: 慣性力 M a と減衰力 C v を残差に、M/(β dt²) + γ C/(β dt) を接線に加える
    dst = None
    if dyn is not None:
        from .dynamics import consistent_mass
        from .remote import pad as _padm
        Mm = consistent_mass(mesh, rho_e)
        if ndof > nb:
            Mm = _padm(Mm, ndof)
        if ex is not None:
            Mm = Mm + ex.M()
        Mm = Mm.tocsr()
        dlt = -float(dyn.hht_alpha)            # 数値減衰（0 = 平均加速度法）
        if dlt == 0.0 and nonlinear_contact:
            dlt = 0.1                          # 接触の衝突は数値減衰なしだとエネルギーが増える（Ansys の既定も 0.1）
            log("接触あり: 数値減衰 0.1 を入れます（HHT-α = -0.1 相当。解析設定で変更できます）")
        if not (0.0 <= dlt <= 1 / 3 + 1e-12):
            raise ValueError("HHT-α（数値減衰）は -1/3 〜 0 の範囲で指定してください")
        Cm = dyn.alpha_damping * Mm if dyn.alpha_damping else None
        if dyn.beta_damping:
            K0 = K_lin if K_lin is not None else assemble_stiffness(mesh, D, elem_mat=elem_mat)
            if ndof > nb:
                K0 = _padm(K0, ndof)
            Cm = dyn.beta_damping * K0 if Cm is None else Cm + dyn.beta_damping * K0
        dst = {"beta": (1 + dlt) ** 2 / 4, "gamma": 0.5 + dlt, "dt": 1.0, "un": np.zeros(ndof),
               "v": np.zeros(ndof), "a": np.zeros(ndof)}

    def kinem(uu):
        """Newmark 法の加速度・速度（前の時刻の状態 dst と今の変位 uu から）"""
        b_, g_, h = dst["beta"], dst["gamma"], dst["dt"]
        a_ = (uu - dst["un"]) / (b_ * h * h) - dst["v"] / (b_ * h) - (0.5 / b_ - 1.0) * dst["a"]
        v_ = dst["v"] + h * ((1 - g_) * dst["a"] + g_ * a_)
        return a_, v_

    def inertia(uu):
        a_, v_ = kinem(uu)
        fd = Mm @ a_
        if Cm is not None:
            fd = fd + Cm @ v_
        return fd

    def inertia_committed():
        fd = Mm @ dst["a"]
        if Cm is not None:
            fd = fd + Cm @ dst["v"]
        return fd

    def inertia_K():
        b_, g_, h = dst["beta"], dst["gamma"], dst["dt"]
        Kd = Mm / (b_ * h * h)
        if Cm is not None:
            Kd = Kd + Cm * (g_ / (b_ * h))
        return Kd

    history = []
    results: list[Result] = []
    u = np.zeros(ndof)

    _PAIR_STATE = ("lam_n", "lam_t", "active", "stick", "tau_c", "dt_c", "kn", "kt", "Ktan", "ratio", "tau_now",
                   "pressure", "gap", "slide", "tau", "masters", "w", "n", "g0", "c_max")

    def save_pair_state():
        return [{k: (None if getattr(p, k) is None else getattr(p, k).copy()) for k in _PAIR_STATE} for p in pairs]

    def load_pair_state(st):
        for p, d in zip(pairs, st):
            for k, v in d.items():
                setattr(p, k, None if v is None else v.copy())

    def _contact(uu):
        Kc, fc = contact_system(pairs, contacts, nb, uu[:nb])
        if ndof > nb:
            from .remote import pad as _pad
            Kc, fc = _pad(Kc, ndof), np.r_[fc, np.zeros(ndof - nb)]
        return Kc, fc

    # 多点拘束の変換 u = T ũ（独立自由度 ũ）。収束の判定もこの独立自由度の残差 Tᵀ R で行う
    Tm0 = None
    if ex is not None and ex.mpc:
        from .remote import reduce_system as _rs
        Tm0 = _rs(sp.identity(ndof, format="csr"), np.zeros(ndof), {}, ex, 1.0)[3]
        Tm0T = Tm0.T.tocsr()
        _newidx = -np.ones(ndof, dtype=np.int64)
        _newidx[Tm0.keep] = np.arange(len(Tm0.keep))
        free_r = np.ones(len(Tm0.keep), bool)
        _cr = _newidx[cdofs]
        free_r[_cr[_cr >= 0]] = False

    pen_state = {}
    lu_cache = {}

    def res_norms(Rv, uu=None):
        """収束判定に使う残差（自由な自由度）と反力（拘束した自由度）の大きさ。
        変形可のリモート変位（従属の自由度の拘束）はペナルティのばねの力も釣り合いに含める"""
        if Tm0 is None:
            return np.linalg.norm(Rv[free]), np.linalg.norm(Rv[~free])
        Rr = Tm0T @ Rv
        react = np.linalg.norm(Rr[~free_r])
        if uu is not None and pen_state.get("fix"):
            from .remote import PENALTY
            kp = PENALTY * pen_state["kref"]
            ur = uu[Tm0.keep]
            for d, v in pen_state["fix"].items():
                trow = Tm0[d]
                fpen = -kp * (float((trow @ ur)[0]) - v) * np.asarray(trow.todense()).ravel()
                Rr = Rr + fpen
                react = max(react, np.linalg.norm(fpen))
        return np.linalg.norm(Rr[free_r]), react

    def equilibrium(t, u0, u_init=None):
        """時刻 t の釣り合いを解く。収束しなければ ConvergenceError。
        u_init: 同じ時刻を解き直すときの出発点（有限すべりで接触相手を決め直したあと）"""
        u = (u0 if u_init is None else u_init).copy()
        cur_t[0] = t
        F, fixed = loads_at(t)
        vals = np.array([fixed.get(int(d), 0.0) for d in cdofs])
        u[cdofs] = vals
        slave_fix = {}
        if Tm0 is not None:
            # 拘束した独立自由度（パイロットの強制変位）を従属の節点にも反映して、拘束式を満たす位置から始める
            u = Tm0 @ u[Tm0.keep]
            ind = _newidx[cdofs] >= 0
            u[cdofs[ind]] = vals[ind]
            # 従属の自由度の強制変位（変形可のリモート変位のパイロット）はペナルティで目標へ
            slave_fix = {int(d): float(v) for d, v, k in zip(cdofs, vals, ind) if not k}
            pen_state["fix"] = slave_fix
        max_it = settings.max_iterations if nonlinear else 1
        for it in range(1, max_it + 1):
            # 塑性: 1回目は前の釣り合い位置の接線で、強制変位の増分を全体に広げる（弾性予測子）。
            # 境界の節点だけを動かして評価すると局所的に大きな塑性ひずみが出てニュートン法が発散するため
            predictor = ps is not None and it == 1 and u_init is None
            ue = u0 if predictor else u
            fint, KT = internal(ue)
            if pairs:
                Kc, fc = _contact(ue)
            else:
                Kc, fc = None, np.zeros(ndof)
            Kt = KT if Kc is None else (KT + Kc)
            if W is not None:
                Kt = Kt + W
            R = F + fc - fint - (Kc @ ue if Kc is not None else 0.0)
            if W is not None:
                R = R - W @ ue
            if dst is not None:
                Kt = Kt + inertia_K()
                R = R - inertia(ue)
            if predictor and len(cdofs):
                R = R - Kt[:, cdofs] @ (vals - u0[cdofs])
            Rf = R[free]
            if ex is not None and ex.mpc:
                # 多点拘束を消去した式で増分を解く（拘束した自由度の増分は 0）
                from .remote import reduce_system
                fx_inc = {int(d): 0.0 for d in cdofs}
                if slave_fix:
                    ucur = Tm0 @ u[Tm0.keep]
                    for d, v in slave_fix.items():
                        fx_inc[d] = v - ucur[d]
                pen_state["kref"] = float(np.mean(np.abs(Kt.diagonal()[:nb])))
                Kr, Rr, fr, Tm = reduce_system(Kt.tocsr(), R, fx_inc, ex, pen_state["kref"])
                fre = np.ones(Kr.shape[0], bool)
                if fr:
                    fre[np.fromiter(fr.keys(), dtype=np.int64)] = False
                Kff = Kr[fre][:, fre].tocsc()
                Rf_r = Rr[fre]
                with np.errstate(all="ignore"):
                    dr = np.zeros(Kr.shape[0])
                    dr[fre] = _linear_solve(Kff, Rf_r, lambda *a: None,
                                            method="superlu" if any(p.nonsym for p in pairs) else "direct")
                du_f = (Tm @ dr)[free]
                Rf_lin = Rf_r
                du_lin = dr[fre]
            elif dst is not None and not nonlinear and int(free.sum()) <= 300_000:
                # 線形の過渡: 係数行列は時間増分が同じなら変わらないので、分解を使い回す
                key = float(f"{dst['dt']:.9e}")
                if key not in lu_cache:
                    from scipy.sparse.linalg import splu
                    lu_cache.clear()
                    lu_cache[key] = (splu(Kt[free][:, free].tocsc()), Kt[free][:, free].tocsr())
                lu, Kff = lu_cache[key]
                du_f = lu.solve(Rf)
                Rf_lin, du_lin = Rf, du_f
            else:
                Kff = Kt[free][:, free].tocsc()
                with np.errstate(all="ignore"):
                    # 摩擦ですべっている節点があると接線が非対称になるので、LU 分解で解く
                    nonsym = any(p.nonsym for p in pairs)
                    meth = "superlu" if nonsym else ("direct" if (method == "auto" and amg_failed[0]) else method)
                    du_f = _linear_solve(Kff, Rf, lin_log, method=meth,
                                         coords=amg_coords if meth == "auto" else None)
                Rf_lin, du_lin = Rf, du_f
            if not np.all(np.isfinite(du_f)):
                raise RuntimeError("剛性マトリクスが特異です。拘束が不足しているか、接触だけで支えられている"
                                   "ボディが浮いています。解析設定で「弱いバネ」をオンにするか、拘束を見直してください。")
            # ラインサーチ
            alpha = 1.0
            if LD and settings.line_search == "on":
                r0 = np.linalg.norm(Rf)
                for _ in range(5):
                    ut = u.copy()
                    ut[free] += alpha * du_f
                    fi, _k = internal(ut)
                    rv = F + fc - fi - (Kc @ ut if Kc is not None else 0.0)
                    if dst is not None:
                        rv = rv - inertia(ut)
                    rt = np.linalg.norm(rv[free])
                    if rt < r0:
                        break
                    alpha *= 0.5
            u[free] += alpha * du_f
            changed, pen = (update_states(pairs, contacts, u, pen_tol=pen_tol) if pairs else (0, 0.0))
            # 収束判定（Ansys と同じく力の残差 / 参照力）。接触は更新後の状態での実際の接触力で評価
            if not nonlinear:
                crit = _rel_res(Kff, du_lin, Rf_lin)
            else:
                fint2 = internal(u)[0]
                R2 = F - fint2
                if pairs:
                    Kc2, fc2 = _contact(u)
                    R2 = R2 + fc2 - Kc2 @ u
                if W is not None:
                    R2 = R2 - W @ u
                fd2 = 0.0
                if dst is not None:
                    fdv = inertia(u)
                    R2 = R2 - fdv
                    fd2 = res_norms(fdv)[0]
                # 参照力 = 外力と反力の大きさ（Ansys と同じ考え方）。過渡では慣性力も含める
                rf2, rc2 = res_norms(R2, u)
                ref2 = max(res_norms(F)[0], rc2, fd2, 0.01)  # 最小参照値 0.01 N（Ansys と同じ）
                crit = rf2 / ref2
            dcrit = np.linalg.norm(alpha * du_f) / max(np.linalg.norm(u[free]), 1e-30)
            history.append({"時間": round(t, 6), "反復": it, "力の残差比": float(crit),
                            "変位の変化比": float(dcrit), "接触状態の変化": int(changed),
                            "最大貫通 [mm]": float(pen)})
            ok_force = crit <= settings.force_convergence
            ok_disp = settings.displacement_convergence <= 0 or dcrit <= settings.displacement_convergence
            # 接触状態は数節点が行き来しても、力が釣り合っていれば収束とみなす（Ansys と同様）
            n_cn = sum(len(p.slave) for p in pairs)
            ok_contact = changed <= max(0, int(0.01 * n_cn)) and (not al or penetration_ok(pairs, pen_tol))
            if not nonlinear:
                return u, F
            if it > 1 or not (LD or ps is not None):
                if ok_force and ok_disp and ok_contact:
                    log(f"  時間 {t:g}: {it} 回で収束（力の残差比 {crit:.2e}、貫通 {pen:.2e} mm）")
                    return u, F
        raise ConvergenceError(f"時間 {t:g} で収束しませんでした（最大反復 {max_it} 回）")

    if settings.arc_length and dyn is None:
        return _arc_length(locals())
    t_prev = 0.0
    n_done = 0
    fin_pairs = [p for p in pairs if p.finite]
    frame_t, hist = [], {}
    if dst is not None:
        import math
        from .dynamics import _frame_times
        frame_t = [float(x) for x in _frame_times(dyn)]
        hist = {"時間 [s]": [], "運動エネルギー [mJ]": [], "ひずみエネルギー [mJ]": [], "接触力 [N]": [],
                "最大変位 [mm]": []}
        v0 = np.zeros(ndof)
        v0[:nb] = np.tile(np.asarray(dyn.init_velocity, float), mesh.n_nodes)
        bv = getattr(dyn, "body_velocity", None) or {}
        if bv and mesh.elem_body is not None:
            for b, vb in bv.items():      # ボディごとの初速度（衝突の解析用）
                nd = np.unique(mesh.elems[mesh.elem_body == int(b)])
                v0[3 * nd[:, None] + np.arange(3)] = np.asarray(vb, float)
        v0[cdofs] = 0.0
        if Tm0 is not None:
            v0 = Tm0 @ v0[Tm0.keep]
        dst["v"] = v0
        # 最初の加速度: 集中質量で a = (F - C v) / m（拘束した自由度は 0）
        ml = np.asarray(Mm.sum(axis=1)).ravel()
        f_init = loads_at(0.0)[0] - (Cm @ v0 if Cm is not None else 0.0)
        a0 = np.where(ml > 0, f_init / np.where(ml > 0, ml, 1.0), 0.0)
        a0[cdofs] = 0.0
        dst["a"] = (Tm0 @ a0[Tm0.keep]) if Tm0 is not None else a0
        dt_dyn = float(dyn.dt) if dyn.dt else dyn.end_time / max(20, int(dyn.n_frames) * 5)
        nsteps_dyn = int(math.ceil(dyn.end_time / dt_dyn - 1e-9))
        log(f"過渡（陰解法・Newmark）: 時間増分 {dyn.end_time / nsteps_dyn:.3e} s × {nsteps_dyn:,} ステップ"
            + ("、非線形" if nonlinear else "、線形"))
    for si, t_end in enumerate(ends):
        nsub = max(1, settings.initial_substeps if (nonlinear or settings.auto_time_stepping == "on") else 1)
        if (LD or any(p.finite for p in pairs)) and settings.auto_time_stepping == "auto" \
                and settings.initial_substeps <= 1:
            nsub = max(nsub, 5)
        if dst is not None:
            nsub = nsteps_dyn
        dt = (t_end - t_prev) / nsub
        t = t_prev
        min_dt = (t_end - t_prev) / max(settings.max_substeps, 1)
        if dst is not None:
            min_dt = dt / 64.0
        dt_user = dt
        while t < t_end - 1e-12 * max(t_end, 1.0):
            t_next = min(t + dt, t_end)
            if frame_t and t_next > frame_t[0] - 1e-9 * t_end:
                t_next = frame_t[0]
            state = save_pair_state()
            if ps is not None:
                ps.dt = t_next - t
            if dst is not None:
                dst["dt"], dst["un"] = t_next - t, u.copy()
            try:
                u_new, F_now = equilibrium(t_next, u)
                for _pass in range(4 if fin_pairs else 0):
                    # 有限すべり: 収束した位置で接触相手を決め直し、Target の内・外が変わった節点があれば
                    # 同じ時刻を解き直す（接触する範囲が、すべった先の位置と合うまで）
                    nchg = sum(repair_in_step(p, contacts[p.region], u_new) for p in fin_pairs)
                    if nchg == 0:
                        break
                    update_states(fin_pairs, contacts, u_new, al_update=False, pen_tol=pen_tol)
                    u_new, F_now = equilibrium(t_next, u, u_init=u_new)
            except ConvergenceError as e:
                if settings.auto_time_stepping != "off" and dt / 2 >= min_dt * 0.999:
                    load_pair_state(state)
                    dt /= 2
                    log(f"  {e} → 時間増分を半分にして再試行（{dt:g}）")
                    continue
                err = RuntimeError(f"{e}。解析設定でサブステップ数を増やすか、荷重・接触設定を見直してください。")
                err.history = history
                raise err
            if dst is not None:
                a_n, v_n = kinem(u_new)
                dst["a"], dst["v"] = a_n, v_n
                dt = min(dt_user, dt * 2.0)       # 収束したら時間増分を戻していく
            u, t = u_new, t_next
            if pairs:
                commit_states(pairs, contacts, u)
            if ps is not None:
                ps.internal(u, th_factor(t), large=LD)
                ps.commit()
            n_done += 1
            if progress:
                progress(min(0.95, t / ends[-1]), f"時間 {t:g} / {ends[-1]:g}")
            if dst is not None:
                vv = dst["v"]
                hist["時間 [s]"].append(t)
                hist["運動エネルギー [mJ]"].append(0.5 * float(vv @ (Mm @ vv)))
                hist["ひずみエネルギー [mJ]"].append(0.5 * float(u @ internal(u)[0]) if not (ps is not None or LD)
                                                 else float("nan"))
                hist["接触力 [N]"].append(float(sum((p.pressure * p.area).sum() for p in pairs
                                                   if p.pressure is not None)))
                hist["最大変位 [mm]"].append(float(np.abs(u[:nb]).max()))
                if frame_t and t >= frame_t[0] - 1e-9 * t_end:
                    frame_t.pop(0)
                    cur_t[0] = t
                    fdv = inertia_committed()

                    def internal_dyn(uu, fdv=fdv):
                        fi, Kb = internal_base(uu)
                        return fi + fdv[:nb], Kb
                    rr = _make_result(mesh, mats, elem_mat, D, u[:nb], bcs,
                                      [(Fi[:nb], {d: v for d, v in fx.items() if d < nb}) for Fi, fx in per_bc],
                                      loads_at(t)[0][:nb], pairs, contacts, LD, internal_dyn, t, len(results) + 1,
                                      settings, ps, eps_th_m * th_factor(t) if has_th else None)
                    v3 = vv[:nb].reshape(-1, 3)
                    rr.extra.update({"速度 [mm/s]": np.linalg.norm(v3, axis=1), "速度 X [mm/s]": v3[:, 0].copy(),
                                     "速度 Y [mm/s]": v3[:, 1].copy(), "速度 Z [mm/s]": v3[:, 2].copy()})
                    results.append(rr)
        t_prev = t_end
        if dst is None and (settings.store_results == "all" or si == len(ends) - 1):
            cur_t[0] = t_end
            results.append(_make_result(mesh, mats, elem_mat, D, u[:nb], bcs,
                                        [(Fi[:nb], {d: v for d, v in fx.items() if d < nb}) for Fi, fx in per_bc],
                                        loads_at(t_end)[0][:nb], pairs,
                                        contacts, LD, internal_base, t_end, si + 1, settings, ps,
                                        eps_th_m * th_factor(t_end) if has_th else None))
            if ex is not None and ex.beams:
                from .remote import beam_results
                results[-1].beams = beam_results(ex, u)
            if ex is not None and ex.normal_sup:
                from .remote import normal_support_reactions
                for i_, r_ in normal_support_reactions(ex, u).items():
                    results[-1].reactions[i_] = results[-1].reactions.get(i_, 0) + r_
    info_common = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
                   "要素タイプ": "Tet10（2次）" if mesh.order == 2 else "Tet4（1次）",
                   "大変形": "オン" if LD else "オフ", "接触領域数": len(contacts),
                   "非線形": "はい" if nonlinear else "いいえ", "サブステップ数（合計）": n_done,
                   "合計時間 [s]": round(time.time() - t_start, 2)}
    if dst is not None:
        info_common.update({"解析": "陰解法（Newmark、非線形対応）", "ステップ数": n_done,
                            "計算時間 [s]": round(time.time() - t_start, 1),
                            "最大接触力 [N]": max(hist["接触力 [N]"] or [0.0])})
    for r in results:
        r.info.update(info_common)
        r.history = hist if dst is not None else history
        if dst is not None:
            r.iterations = history
    return results


def _arc_length(c):
    """弧長法（Crisfield の円筒型）。 solve_static の準備（内力・荷重・拘束）をそのまま使う。

    釣り合い  λ F_ref − f_int(u) = 0  を、増分の長さ |Δu| = Δl の条件と一緒に解く。 予測子は接線の向き
    （前の増分と同じ側）、修正子は 2 次方程式の根のうち前の向きに近い方。 収束の回数で Δl を伸び縮みさせる。
    戻り値: 収束した点ごとの Result（time = 荷重係数 λ）。
    """
    from scipy.sparse.linalg import splu
    s, log, mesh = c["settings"], c["log"], c["mesh"]
    ends, free, nb, ndof = c["ends"], c["free"], c["nb"], c["ndof"]
    pairs, contacts, ps, LD, W = c["pairs"], c["contacts"], c["ps"], c["LD"], c["W"]
    if c["ex"] is not None or any(k.nonlinear for k in contacts):
        raise ValueError("弧長法は、リモート境界条件・ジョイント・非線形の接触（摩擦なし・摩擦ありなど）とは"
                         "まだ組み合わせられません（結合・分離なしの接触は使えます）")
    c["cur_t"][0] = ends[-1]
    Fref, fixed = c["loads_at"](ends[-1])
    if any(abs(v) > 0 for v in fixed.values()):
        raise ValueError("弧長法では 0 以外の強制変位は使えません（力・圧力・重力で荷重を与えてください）")
    if not np.any(Fref[free]):
        raise ValueError("弧長法には荷重（力・圧力など）が必要です")
    Kc, fc = (c["_contact"](np.zeros(ndof)) if pairs else (None, 0.0))
    internal = c["internal"]

    def resid(u, lam):
        fint, Kt = internal(u)
        R = lam * Fref - fint
        if Kc is not None:
            R, Kt = R + fc - Kc @ u, Kt + Kc
        if W is not None:
            R, Kt = R - W @ u, Kt + W
        return R, Kt

    def solver(Kt):
        lu = splu(Kt[free][:, free].tocsc())

        def sol(b):
            x = np.zeros(ndof)
            x[free] = lu.solve(b[free])
            return x
        return sol

    fn = float(np.linalg.norm(Fref[free]))
    u, lam, dl, dl0, du_prev = np.zeros(ndof), 0.0, None, None, None
    lam_peak, results, hist = 0.0, [], []
    t0 = time.time()
    per_bc = [(Fi[:nb], {d: v for d, v in fx.items() if d < nb}) for Fi, fx in c["per_bc"]]
    for step in range(1, int(s.arc_max_steps) + 1):
        _R, Kt = resid(u, lam)
        dut = solver(Kt)(Fref)
        nt = float(np.linalg.norm(dut))
        if not np.isfinite(nt) or nt == 0:
            raise RuntimeError("弧長法: 接線の解が求まりません（拘束が不足していないか確認してください）")
        if dl is None:
            dl = dl0 = abs(float(s.arc_initial)) * nt
        sgn = 1.0 if du_prev is None else (1.0 if float(du_prev @ dut) >= 0 else -1.0)
        ok, it = False, 0
        for _attempt in range(10):
            Du, Dl = sgn * dl / nt * dut, sgn * dl / nt
            for it in range(1, s.max_iterations + 1):
                Rk, Ktk = resid(u + Du, lam + Dl)
                ref = max(max(abs(lam + Dl), lam_peak) * fn, 1e-2)
                if it > 1 and np.linalg.norm(Rk[free]) <= s.force_convergence * ref:
                    ok = True
                    break
                sol = solver(Ktk)
                dub, dutk = sol(Rk), sol(Fref)
                if not (np.all(np.isfinite(dub)) and np.all(np.isfinite(dutk))):
                    break
                v = Du + dub
                a, b, cc = float(dutk @ dutk), 2.0 * float(v @ dutk), float(v @ v) - dl * dl
                disc = b * b - 4 * a * cc
                if disc < 0:
                    break
                r1, r2 = (-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)
                r = r1 if float(Du @ (v + r1 * dutk)) >= float(Du @ (v + r2 * dutk)) else r2
                Du, Dl = v + r * dutk, Dl + r
            if ok:
                break
            dl *= 0.5
            if ps is not None:
                ps.trial = None
        if not ok:
            err = RuntimeError(f"弧長法: {step} 点目で収束しませんでした（弧長を {dl:.3g} まで縮めました）。"
                               "最初の荷重係数の増分を小さくしてみてください。")
            err.history = hist
            raise err
        u, lam, du_prev = u + Du, lam + Dl, Du
        lam_peak = max(lam_peak, abs(lam))
        if ps is not None:
            ps.internal(u[:nb], c["th_factor"](ends[-1]), large=LD)
            ps.commit()
        umax = float(np.abs(u[:nb]).max())
        hist.append({"時間": round(lam, 6), "反復": it, "荷重係数": float(lam), "最大変位 [mm]": umax,
                     "力の残差比": float(np.linalg.norm(Rk[free]) / ref), "変位の変化比": 0.0,
                     "接触状態の変化": 0, "最大貫通 [mm]": 0.0})
        log(f"  弧長法 {step}: 荷重係数 {lam:.5g} / 最大変位 {umax:.5g} mm / 反復 {it}")
        rr = _make_result(mesh, c["mats"], c["elem_mat"], c["D"], u[:nb], c["bcs"], per_bc, lam * Fref[:nb], pairs,
                          contacts, LD, c["internal_base"], lam, step, s, ps,
                          c["eps_th_m"] * c["th_factor"](ends[-1]) if c["has_th"] else None)
        rr.label = f"λ = {lam:.4g}"
        rr.info["荷重係数"] = float(lam)
        results.append(rr)
        if c["progress"]:
            c["progress"](min(0.95, step / s.arc_max_steps), f"弧長法 {step} 点 / 荷重係数 {lam:.4g}")
        if lam >= s.arc_max_load or (s.arc_max_disp > 0 and umax >= s.arc_max_disp):
            break
        dl = float(np.clip(dl * min(np.sqrt(4.0 / max(it - 1, 1)), 1.6), 0.02 * dl0, float(s.arc_growth) * dl0))
    lams = [r.time for r in results]
    info = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems, "解析": "弧長法",
            "大変形": "オン" if LD else "オフ", "点の数": len(results), "最大の荷重係数": float(max(lams)),
            "合計時間 [s]": round(time.time() - t0, 2)}
    for r in results:
        r.info.update(info)
        r.history = hist
    return results


def _make_result(mesh, mats, elem_mat, D, u, bcs, per_bc, F, pairs, contacts, LD, internal, t, step, settings,
                 ps=None, eps_th=None):
    extra = {}
    if LD and ps is None:
        stress, strain = tl_cauchy_nodal(mesh, u, D, elem_mat)
    elif ps is None:
        stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat, eps_th_m=eps_th)
    else:
        strain = (tl_cauchy_nodal(mesh, u, D, elem_mat)[1] if LD
                  else nodal_stress(mesh, u, D, elem_mat=elem_mat)[1])
        internal(u)
        stress, peeq = ps.nodal()
        if any(m.plasticity != "none" for m in mats):
            extra["相当塑性ひずみ"] = peeq
        if any(getattr(m, "creeping", False) for m in mats):
            extra["相当クリープひずみ"] = ps.nodal_scalar(ps.pcr)
    fint, _ = internal(u)
    Rv = fint - F
    if pairs:
        Kc, fc = contact_system(pairs, contacts, 3 * mesh.n_nodes, u)
        Rv = Rv + Kc @ u - fc
    reactions = {}
    for i, (bc, (_, fx)) in enumerate(zip(bcs, per_bc)):
        if bc.kind in ("fixed", "displacement", "symmetry", "frictionless", "submodel"):
            d = np.fromiter(fx.keys(), dtype=np.int64)
            r = np.zeros(3)
            for c in range(3):
                r[c] = Rv[d[d % 3 == c]].sum()
            reactions[i] = r
    r = Result(mesh, mats[0], u.reshape(-1, 3).copy(), stress, strain, reactions,
               {"時間 [s]": t, "ステップ": step}, mats, elem_mat)
    r.time = t
    r.extra = extra
    r.contact = contact_results(pairs, contacts, mesh.n_nodes) if (pairs and settings.calc_contact) else {}
    return r

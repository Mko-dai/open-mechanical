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

from .contact import (ContactRegion, build_pairs, commit_states, contact_results, contact_system,
                      update_states)
from .materials import Material
from .mesh import Mesh
from .solver import (BC, GAUSS4, GAUSS4_W, NODE_L10, NODE_L4, Result, _B_matrix, _dN_dx, _D_of,
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
    Lnodes = NODE_L10 if nn == 10 else NODE_L4
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
                 progress=None) -> list[Result]:
    """静的構造解析（接触・大変形・複数ステップ対応）。各ステップ終了時の Result のリストを返す。"""
    t_start = time.time()
    settings = settings or AnalysisSettings()
    contacts = [c for c in (contacts or []) if not c.suppressed]
    if not any(b.kind in ("fixed", "displacement") for b in bcs):
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
    plastic = any(getattr(m, "plastic", False) for m in mats)
    rho_e = np.array([m.density_t_mm3 for m in mats])[elem_mat]
    ndof = 3 * mesh.n_nodes

    # 荷重・拘束（境界条件ごと）
    per_bc = [build_load_and_constraints(mesh, mats[0], [bc], rho_e=rho_e) for bc in bcs]
    ends = settings.end_times()

    th_bcs = [bc for bc in bcs if bc.kind == "thermal"]
    F_th = thermal_load_vector(mesh, D, elem_mat, eps_th_m) if (has_th and not plastic) else None

    def th_factor(t):
        """熱ひずみの倍率（熱条件のステップ倍率。熱条件が無く材料温度だけなら 1 で一定）"""
        if not th_bcs:
            return 1.0
        return _step_factor(th_bcs[0], t, ends)

    def loads_at(t):
        F = np.zeros(ndof)
        fixed = {}
        if F_th is not None:
            F += th_factor(t) * F_th
        for bc, (Fi, fx) in zip(bcs, per_bc):
            f = _step_factor(bc, t, ends)
            if bc.kind in ("fixed",):
                for d in fx:
                    fixed[d] = 0.0
            elif bc.kind == "displacement":
                for d, v in fx.items():
                    fixed[d] = f * v
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
    pairs = build_pairs(mesh, contacts, E_node) if contacts else []
    if contacts:
        log(f"接触: {len(contacts)} 領域 / 接触節点 {sum(len(p.slave) for p in pairs):,}")
    nonlinear_contact = any(c.nonlinear for c in contacts)
    friction = any(c.type == "frictional" for c in contacts)
    al = any(c.resolved_formulation() == "augmented_lagrange" for c in contacts)
    LD = settings.large_deflection
    vol = np.abs(mesh.element_volumes())
    h_char = float(np.cbrt(np.median(vol) * 6 * np.sqrt(2))) if len(vol) else 1.0
    pen_tol = settings.penetration_tolerance * h_char

    if plastic and LD:
        raise ValueError("塑性と大変形の組み合わせにはまだ対応していません。解析設定の「大変形」をオフにしてください。")
    if LD and has_th:
        raise ValueError("熱ひずみと大変形の組み合わせにはまだ対応していません。")
    ps = None
    if plastic:
        from .plasticity import PlasticSolid
        ps = PlasticSolid(mesh, mats, elem_mat, D, eps_th_m)
        log("塑性材料あり: 弾塑性（ミーゼス）で解きます")
    K_lin = None if (LD or plastic) else assemble_stiffness(mesh, D, elem_mat=elem_mat)
    k_weak = 0.0
    if settings.weak_springs:
        diagK = (K_lin.diagonal() if K_lin is not None else assemble_stiffness(mesh, D, elem_mat=elem_mat).diagonal())
        k_weak = 1e-8 * float(np.mean(np.abs(diagK)))
    W = sp.identity(ndof, format="csr") * k_weak if k_weak > 0 else None
    from .parallel import set_threads
    set_threads(settings.n_cores)
    # 接触の剛性で条件が悪くなるため、非線形では既定で直接法
    method = "direct" if settings.solver_type == "auto" else settings.solver_type

    cur_t = [0.0]

    def internal(u):
        if LD:
            return tl_internal(mesh, D, elem_mat, u)
        if ps is not None:
            return ps.internal(u, th_factor(cur_t[0]))
        return K_lin @ u, K_lin

    nonlinear = LD or nonlinear_contact or al or plastic
    history = []
    results: list[Result] = []
    u = np.zeros(ndof)

    def save_pair_state():
        return [(p.lam_n.copy(), p.lam_t.copy(), p.active.copy(), p.stick.copy(), p.tau_c.copy(),
                 p.dt_c.copy()) for p in pairs]

    def load_pair_state(st):
        for p, (a, b, c, d, e, f) in zip(pairs, st):
            p.lam_n, p.lam_t, p.active, p.stick = a.copy(), b.copy(), c.copy(), d.copy()
            p.tau_c, p.dt_c = e.copy(), f.copy()

    def equilibrium(t, u0):
        """時刻 t の釣り合いを解く。収束しなければ ConvergenceError"""
        u = u0.copy()
        cur_t[0] = t
        F, fixed = loads_at(t)
        vals = np.array([fixed.get(int(d), 0.0) for d in cdofs])
        u[cdofs] = vals
        max_it = settings.max_iterations if nonlinear else 1
        for it in range(1, max_it + 1):
            # 塑性: 1回目は前の釣り合い位置の接線で、強制変位の増分を全体に広げる（弾性予測子）。
            # 境界の節点だけを動かして評価すると局所的に大きな塑性ひずみが出てニュートン法が発散するため
            predictor = ps is not None and it == 1
            ue = u0 if predictor else u
            fint, KT = internal(ue)
            if pairs:
                Kc, fc = contact_system(pairs, contacts, ndof, ue)
            else:
                Kc, fc = None, np.zeros(ndof)
            Kt = KT if Kc is None else (KT + Kc)
            if W is not None:
                Kt = Kt + W
            R = F + fc - fint - (Kc @ ue if Kc is not None else 0.0)
            if W is not None:
                R = R - W @ ue
            if predictor and len(cdofs):
                R = R - Kt[:, cdofs] @ (vals - u0[cdofs])
            Rf = R[free]
            ref = max(np.linalg.norm(F[free]), np.linalg.norm(R[~free]), 1e-12)
            Kff = Kt[free][:, free].tocsc()
            with np.errstate(all="ignore"):
                du_f = _linear_solve(Kff, Rf, lambda *a: None, method=method)
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
                    rt = np.linalg.norm((F + fc - fi - (Kc @ ut if Kc is not None else 0.0))[free])
                    if rt < r0:
                        break
                    alpha *= 0.5
            u[free] += alpha * du_f
            changed, pen = (update_states(pairs, contacts, u, pen_tol=pen_tol) if pairs else (0, 0.0))
            # 収束判定（Ansys と同じく力の残差 / 参照力）。接触は更新後の状態での実際の接触力で評価
            if not nonlinear:
                crit = _rel_res(Kff, du_f, Rf)
            else:
                fint2 = internal(u)[0] if (LD or ps is not None) else K_lin @ u
                R2 = F - fint2
                if pairs:
                    Kc2, fc2 = contact_system(pairs, contacts, ndof, u)
                    R2 = R2 + fc2 - Kc2 @ u
                if W is not None:
                    R2 = R2 - W @ u
                # 参照力 = 外力と反力の大きさ（Ansys と同じ考え方）
                ref2 = max(np.linalg.norm(F[free]), np.linalg.norm(R2[~free]), 0.01)  # 最小参照値 0.01 N（Ansys と同じ）
                crit = np.linalg.norm(R2[free]) / ref2
            dcrit = np.linalg.norm(alpha * du_f) / max(np.linalg.norm(u[free]), 1e-30)
            history.append({"時間": round(t, 6), "反復": it, "力の残差比": float(crit),
                            "変位の変化比": float(dcrit), "接触状態の変化": int(changed),
                            "最大貫通 [mm]": float(pen)})
            ok_force = crit <= settings.force_convergence
            ok_disp = settings.displacement_convergence <= 0 or dcrit <= settings.displacement_convergence
            # 接触状態は数節点が行き来しても、力が釣り合っていれば収束とみなす（Ansys と同様）
            n_cn = sum(len(p.slave) for p in pairs)
            ok_contact = changed <= max(0, int(0.01 * n_cn)) and (not al or pen <= pen_tol)
            if not nonlinear:
                return u, F
            if it > 1 or not (LD or ps is not None):
                if ok_force and ok_disp and ok_contact:
                    log(f"  時間 {t:g}: {it} 回で収束（力の残差比 {crit:.2e}、貫通 {pen:.2e} mm）")
                    return u, F
        raise ConvergenceError(f"時間 {t:g} で収束しませんでした（最大反復 {max_it} 回）")

    t_prev = 0.0
    n_done = 0
    for si, t_end in enumerate(ends):
        nsub = max(1, settings.initial_substeps if (nonlinear or settings.auto_time_stepping == "on") else 1)
        if LD and settings.auto_time_stepping == "auto" and settings.initial_substeps <= 1:
            nsub = max(nsub, 5)
        dt = (t_end - t_prev) / nsub
        t = t_prev
        min_dt = (t_end - t_prev) / max(settings.max_substeps, 1)
        while t < t_end - 1e-12:
            t_next = min(t + dt, t_end)
            state = save_pair_state()
            try:
                u_new, F_now = equilibrium(t_next, u)
            except ConvergenceError as e:
                if settings.auto_time_stepping != "off" and dt / 2 >= min_dt * 0.999:
                    load_pair_state(state)
                    dt /= 2
                    log(f"  {e} → 時間増分を半分にして再試行（{dt:g}）")
                    continue
                err = RuntimeError(f"{e}。解析設定でサブステップ数を増やすか、荷重・接触設定を見直してください。")
                err.history = history
                raise err
            u, t = u_new, t_next
            if pairs:
                commit_states(pairs, contacts, u)
            if ps is not None:
                ps.internal(u, th_factor(t))
                ps.commit()
            n_done += 1
            if progress:
                progress(min(0.95, t / ends[-1]), f"時間 {t:g} / {ends[-1]:g}")
        t_prev = t_end
        if settings.store_results == "all" or si == len(ends) - 1:
            cur_t[0] = t_end
            results.append(_make_result(mesh, mats, elem_mat, D, u, bcs, per_bc, loads_at(t_end)[0], pairs,
                                        contacts, LD, internal, t_end, si + 1, settings, ps,
                                        eps_th_m * th_factor(t_end) if has_th else None))
    info_common = {"自由度数": int(free.sum()), "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
                   "要素タイプ": "Tet10（2次）" if mesh.order == 2 else "Tet4（1次）",
                   "大変形": "オン" if LD else "オフ", "接触領域数": len(contacts),
                   "非線形": "はい" if nonlinear else "いいえ", "サブステップ数（合計）": n_done,
                   "合計時間 [s]": round(time.time() - t_start, 2)}
    for r in results:
        r.info.update(info_common)
        r.history = history
    return results


def _make_result(mesh, mats, elem_mat, D, u, bcs, per_bc, F, pairs, contacts, LD, internal, t, step, settings,
                 ps=None, eps_th=None):
    extra = {}
    if LD:
        stress, strain = tl_cauchy_nodal(mesh, u, D, elem_mat)
    else:
        stress, strain = nodal_stress(mesh, u, D, elem_mat=elem_mat, eps_th_m=eps_th)
    if ps is not None:
        internal(u)
        stress, peeq = ps.nodal()
        extra["相当塑性ひずみ"] = peeq
    fint, _ = internal(u)
    Rv = fint - F
    if pairs:
        Kc, fc = contact_system(pairs, contacts, 3 * mesh.n_nodes, u)
        Rv = Rv + Kc @ u - fc
    reactions = {}
    for i, (bc, (_, fx)) in enumerate(zip(bcs, per_bc)):
        if bc.kind in ("fixed", "displacement"):
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

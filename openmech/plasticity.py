"""金属の塑性（ミーゼス降伏・等方硬化／移動硬化、2直線／多直線）— 微小ひずみ

Ansys の Bilinear / Multilinear Isotropic Hardening, Bilinear / Multilinear Kinematic Hardening に相当。
積分点ごとにリターンマッピング（半径方向への戻し）で応力を求め、整合接線剛性を使ってニュートン法で解く。

  等方硬化: 降伏曲面が大きくなる     σy(ε̄p) = 表の補間
  移動硬化: 降伏曲面が移動する（バウシンガー効果） 背応力 α、σy は初期値のまま、硬化係数 H(ε̄p) は表の傾き
            ※ 多直線移動硬化は「累積塑性ひずみに応じた傾きの線形移動硬化」で近似（Ansys の KINH の多層モデルとは異なる）
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh
from .solver import _B_matrix, _dN_dx, _quadrature, shape_dN_dxi

S2 = np.sqrt(2.0)
# Voigt（工学せん断）⇔ Mandel 変換
W_SIG = np.array([1, 1, 1, S2, S2, S2])         # σ_M = W_SIG σ_V
W_EPS = np.array([1, 1, 1, 1 / S2, 1 / S2, 1 / S2])  # ε_M = W_EPS ε_V（ε_V は工学せん断）
I_VOL = np.array([1, 1, 1, 0, 0, 0], float)
P_DEV = np.eye(6) - np.outer(I_VOL, I_VOL) / 3.0  # Mandel での偏差射影


def _table_eval(tab, p):
    """σy(ε̄p) と その傾き H（表の外は最後の傾きで外挿）"""
    ep = np.array([r[0] for r in tab])
    sg = np.array([r[1] for r in tab])
    if len(ep) == 1:
        return np.full_like(p, sg[0]), np.zeros_like(p)
    k = np.clip(np.searchsorted(ep, p, side="right") - 1, 0, len(ep) - 2)
    H = (sg[k + 1] - sg[k]) / np.maximum(ep[k + 1] - ep[k], 1e-30)
    return sg[k] + H * (p - ep[k]), H


class PlasticModel:
    def __init__(self, mstate):
        self.E, self.nu = mstate.E, mstate.nu
        self.G = self.E / (2 * (1 + self.nu))
        self.K = self.E / (3 * (1 - 2 * self.nu))
        self.sy0, self.tab, self.kin = mstate.hardening()
        self.D_M = self.K * np.outer(I_VOL, I_VOL) + 2 * self.G * P_DEV    # Mandel 弾性
        self.yields = mstate.plasticity != "none"
        # Chaboche: 背応力を項ごとに持つ（状態 alpha の幅 = 6 × 項数）
        self.chab = None
        if mstate.plasticity == "chaboche":
            self.chab = np.array([[float(r[0]), float(r[1])] for r in mstate.base.chaboche
                                  if len(r) >= 2 and float(r[0]) > 0])
        self.n_alpha = 6 * (len(self.chab) if self.chab is not None else 1)
        self.creep = None
        if getattr(mstate, "creeping", False):
            c = mstate.base.creep
            A = float(c.get("C1", 0.0))
            c3 = float(c.get("C3", 0.0) or 0.0)
            if c3:
                T = mstate.T if mstate.T is not None else float(getattr(mstate.base, "ref_temp", 22.0))
                A *= np.exp(-c3 / (T + 273.15))
            self.creep = (A, float(c.get("C2", 1.0)))

    def creep_update(self, eps_M, ep_M, pcr, dt):
        """Norton クリープ ε̇ = A σ^n の後退オイラー（半径方向への戻しと同じ形）。戻り値: σ_M, ep_M, pcr, C_M"""
        n_e = len(eps_M)
        sig_tr = (eps_M - ep_M) @ self.D_M.T
        xi = sig_tr @ P_DEV.T
        nxi = np.linalg.norm(xi, axis=1)
        qt = np.sqrt(1.5) * nxi
        C = np.broadcast_to(self.D_M, (n_e, 6, 6)).copy()
        A, nexp = self.creep
        if dt <= 0 or A <= 0:
            return sig_tr, ep_M, pcr, C
        idx = np.nonzero(qt > 1e-12 * max(float(qt.max(initial=0.0)), 1e-30))[0]
        if len(idx) == 0:
            return sig_tr, ep_M, pcr, C
        G = self.G
        q0 = qt[idx]
        # r(Δp) = Δp − dt A (q_tr − 3GΔp)^n = 0、0 ≤ Δp < q_tr/3G（単調なので二分法で囲みつつニュートン）
        lo = np.zeros(len(idx))
        hi = q0 / (3 * G)
        dp = np.minimum(dt * A * q0 ** nexp, 0.5 * hi)
        for _ in range(100):
            q = np.maximum(q0 - 3 * G * dp, 1e-300)
            r = dp - dt * A * q ** nexp
            lo = np.where(r < 0, dp, lo)
            hi = np.where(r > 0, dp, hi)
            dr = 1 + dt * A * nexp * q ** (nexp - 1) * 3 * G
            new = dp - r / dr
            bad = (new <= lo) | (new >= hi)
            new = np.where(bad, 0.5 * (lo + hi), new)
            if np.all(np.abs(new - dp) <= 1e-13 * np.maximum(hi, 1e-30)):
                dp = new
                break
            dp = new
        q = np.maximum(q0 - 3 * G * dp, 1e-300)
        h = dt * A * nexp * q ** (nexp - 1)               # dΔp/dq（この増分での粘性の傾き）
        nvec = xi[idx] / nxi[idx, None]
        dep = np.sqrt(1.5) * dp[:, None] * nvec
        sig = sig_tr.copy()
        sig[idx] = sig_tr[idx] - 2 * G * dep
        ep_M = ep_M.copy()
        ep_M[idx] += dep
        pcr = pcr.copy()
        pcr[idx] += dp
        theta = 1 - 3 * G * dp / q0
        theta_b = 3 * G * h / (1 + 3 * G * h) - (1 - theta)   # 塑性の式 1/(1+H/3G) − (1−θ) で H = 1/h
        C[idx] = (self.K * np.outer(I_VOL, I_VOL))[None] + 2 * G * theta[:, None, None] * P_DEV[None] \
            - 2 * G * theta_b[:, None, None] * np.einsum("ei,ej->eij", nvec, nvec)
        return sig, ep_M, pcr, C

    def update_all(self, eps_M, ep_M, alpha_M, pbar, pcr, dt):
        """クリープ（あれば）→ 塑性 の順に更新。戻り値: σ_M, ep_M, alpha_M, pbar, pcr, C_M"""
        Cc = None
        if self.creep is not None:
            s_c, ep_M, pcr, Cc = self.creep_update(eps_M, ep_M, pcr, dt)
            if not self.yields:
                return s_c, ep_M, alpha_M, pbar, pcr, Cc
        s, ep2, al2, pb2, C = self.update(eps_M, ep_M, alpha_M, pbar)
        if Cc is not None:
            yl = pb2 > pbar
            C = np.where(yl[:, None, None], C, Cc)
        return s, ep2, al2, pb2, pcr, C

    def radius(self, p):
        """降伏応力（等方: 表の値、移動: 初期値）と硬化係数 (Hiso, Hkin)"""
        s, H = _table_eval(self.tab, p)
        if self.kin:
            return np.full_like(p, self.sy0), np.zeros_like(p), H
        return s, H, np.zeros_like(p)

    def update(self, eps_M, ep_M, alpha_M, pbar):
        """リターンマッピング。全て Mandel。戻り値: σ_M, ep_M, alpha_M, pbar, C_M（整合接線）"""
        if self.chab is not None:
            return self._update_chaboche(eps_M, ep_M, alpha_M, pbar)
        if alpha_M.shape[1] == 6:
            return self._update6(eps_M, ep_M, alpha_M, pbar)
        s, e, a6, pb, C = self._update6(eps_M, ep_M, alpha_M[:, :6], pbar)
        a = alpha_M.copy()
        a[:, :6] = a6
        return s, e, a, pb, C

    def _chab_stress(self, eps_M, ep_M, alpha_M, pbar):
        """Chaboche の応力更新（後退オイラー）。 背応力 α_i = θ_i (α_i,n + √(2/3) C_i Δp n)、θ_i = 1/(1+γ_i Δp)。
        Δp は 1 変数の式 √(3/2)|s_tr − Σθ_i α_i,n| − (3G + Σθ_i C_i) Δp − σy(p+Δp) = 0 から求める"""
        nb = len(self.chab)
        Cc, gm = self.chab[:, 0], self.chab[:, 1]
        sig_tr = (eps_M - ep_M) @ self.D_M.T
        sdev = sig_tr @ P_DEV.T
        a_n = alpha_M[:, :6 * nb].reshape(len(eps_M), nb, 6)
        q_tr = np.sqrt(1.5) * np.linalg.norm(sdev - a_n.sum(1), axis=1)
        sy = _table_eval(self.tab, pbar)[0]
        yl = q_tr - sy > 1e-10 * max(self.sy0, 1.0)
        sig, ep, al, pb = sig_tr.copy(), ep_M, alpha_M, pbar
        if not np.any(yl):
            return sig, ep, al, pb, yl
        idx = np.nonzero(yl)[0]
        G, sd, an, p0 = self.G, sdev[idx], a_n[idx], pbar[idx]

        def gfun(dp):
            th = 1.0 / (1.0 + gm[None, :] * dp[:, None])
            xt = sd - np.einsum("mi,mij->mj", th, an)
            nx = np.linalg.norm(xt, axis=1)
            return np.sqrt(1.5) * nx - (3 * G + th @ Cc) * dp - _table_eval(self.tab, p0 + dp)[0], xt, nx, th
        lo = np.zeros(len(idx))
        hi = q_tr[idx] / (3 * G) + 1e-12
        for _ in range(60):
            bad = gfun(hi)[0] > 0
            if not bad.any():
                break
            hi = np.where(bad, 2 * hi, hi)
        dp = 0.5 * hi * 0 + np.minimum((q_tr[idx] - sy[idx]) / (3 * G + Cc.sum()), hi)
        for _ in range(80):
            g0 = gfun(dp)[0]
            lo = np.where(g0 > 0, dp, lo)
            hi = np.where(g0 < 0, dp, hi)
            h = 1e-7 * dp + 1e-14
            dg = (gfun(dp + h)[0] - g0) / h
            new = dp - g0 / np.where(dg < 0, dg, -1.0)
            new = np.where((new <= lo) | (new >= hi), 0.5 * (lo + hi), new)
            done = np.abs(new - dp) <= 1e-13 * np.maximum(hi, 1e-30)
            dp = new
            if done.all():
                break
        _g, xt, nx, th = gfun(dp)
        nvec = xt / np.maximum(nx, 1e-300)[:, None]
        dep = np.sqrt(1.5) * dp[:, None] * nvec
        sig[idx] = sig_tr[idx] - 2 * G * dep
        ep = ep_M.copy()
        ep[idx] += dep
        al = alpha_M.copy()
        a_new = th[:, :, None] * (an + np.sqrt(2.0 / 3.0) * (Cc[None, :] * dp[:, None])[:, :, None] * nvec[:, None, :])
        al[idx, :6 * nb] = a_new.reshape(len(idx), -1)
        pb = pbar.copy()
        pb[idx] = p0 + dp
        return sig, ep, al, pb, yl

    def _update_chaboche(self, eps_M, ep_M, alpha_M, pbar):
        sig, ep, al, pb, yl = self._chab_stress(eps_M, ep_M, alpha_M, pbar)
        C = np.broadcast_to(self.D_M, (len(eps_M), 6, 6)).copy()
        if np.any(yl):
            # 整合接線は応力更新の数値微分（降伏している点だけ）。 γ があると非対称になるので対称部分を使う
            idx = np.nonzero(yl)[0]
            h = 1e-5 * max(self.sy0, 1.0) / self.E
            Cy = np.empty((len(idx), 6, 6))
            for j in range(6):
                e2 = eps_M[idx].copy()
                e2[:, j] += h
                Cy[:, :, j] = (self._chab_stress(e2, ep_M[idx], alpha_M[idx], pbar[idx])[0] - sig[idx]) / h
            C[idx] = 0.5 * (Cy + Cy.transpose(0, 2, 1))
        return sig, ep, al, pb, C

    def _update6(self, eps_M, ep_M, alpha_M, pbar):
        n = len(eps_M)
        sig_tr = (eps_M - ep_M) @ self.D_M.T
        xi = sig_tr @ P_DEV.T - alpha_M
        nxi = np.linalg.norm(xi, axis=1)
        q_tr = np.sqrt(1.5) * nxi
        sy, Hi, Hk = self.radius(pbar)
        f = q_tr - sy
        C = np.broadcast_to(self.D_M, (n, 6, 6)).copy()
        sig = sig_tr.copy()
        yl = f > 1e-10 * max(self.sy0, 1.0)
        if not np.any(yl):
            return sig, ep_M, alpha_M, pbar, C
        idx = np.nonzero(yl)[0]
        G = self.G
        p0 = pbar[idx]
        dp = np.zeros(len(idx))
        qt = q_tr[idx]
        for _ in range(50):   # 区分線形の硬化に対するニュートン法
            syk, Hik, Hkk = self.radius(p0 + dp)
            r = qt - 3 * G * dp - (Hkk * dp if self.kin else 0.0) - syk
            dr = -3 * G - Hkk - Hik
            step = -r / dr
            dp = np.maximum(dp + step, 0.0)
            if np.all(np.abs(step) <= 1e-12 * np.maximum(dp, 1e-12) + 1e-16):
                break
        syk, Hik, Hkk = self.radius(p0 + dp)
        nvec = xi[idx] / nxi[idx, None]
        dep = np.sqrt(1.5) * dp[:, None] * nvec
        sig[idx] = sig_tr[idx] - 2 * G * dep
        ep_M = ep_M.copy()
        ep_M[idx] += dep
        alpha_M = alpha_M.copy()
        if self.kin:
            alpha_M[idx] += np.sqrt(2.0 / 3.0) * (Hkk * dp)[:, None] * nvec   # Prager: Δα = (2/3) H Δεp
        pbar = pbar.copy()
        pbar[idx] = p0 + dp
        # 整合接線: C = K 1⊗1 + 2Gθ P − 2G θ̄ n⊗n
        Htot = Hik + Hkk
        theta = 1 - 3 * G * dp / qt
        theta_b = 1 / (1 + Htot / (3 * G)) - (1 - theta)
        C[idx] = (self.K * np.outer(I_VOL, I_VOL))[None] + 2 * G * theta[:, None, None] * P_DEV[None] \
            - 2 * G * theta_b[:, None, None] * np.einsum("ei,ej->eij", nvec, nvec)
        return sig, ep_M, alpha_M, pbar, C


class PlasticSolid:
    """塑性材料を含むモデルの内力・接線剛性（積分点の状態を持つ）"""

    def __init__(self, mesh: Mesh, mats, elem_mat, D, eps_th_m=None):
        self.mesh, self.mats, self.elem_mat, self.D = mesh, mats, np.asarray(elem_mat), D
        nn = mesh.elems.shape[1]
        self.nn = nn
        self.pts, self.wts = _quadrature(nn)
        self.dN = [shape_dN_dxi(L, nn) for L in self.pts]
        ng = len(self.wts)
        E = mesh.n_elems
        self.models = {i: PlasticModel(m) for i, m in enumerate(mats) if getattr(m, "plastic", False)}
        from .hyperelastic import HyperModel
        self.hyper = {i: HyperModel(m.base.hyper) for i, m in enumerate(mats) if getattr(m, "hyperelastic", False)}
        self.dt = 0.0                     # クリープの時間増分（ドライバーが各サブステップで設定）
        self.eps_th_m = eps_th_m if eps_th_m is not None else np.zeros((len(mats), 6))
        # 積分点ごとの B・重み（毎回作り直さないよう保存）
        X = mesh.nodes[mesh.elems]
        self.B, self.wd, self.dNdX = [], [], []
        for g in range(ng):
            dNdx, detJ = _dN_dx(X, self.dN[g])
            self.B.append(_B_matrix(dNdx))
            self.wd.append(self.wts[g] * detJ)
            self.dNdX.append(dNdx)
        # 超弾性の選択的低減積分（体積成分は要素中心 1 点）。Tet10 で、ほぼ非圧縮（K/μ > 100）のときだけ。
        # 圧縮性のある材料では通常の積分の方が正確（低減積分は少し柔らかくなる）
        self.hyper_sri = nn == 10 and any(2.0 / hm.d / (2 * (hm.c10 + hm.c01)) > 100 for hm in self.hyper.values())
        if self.hyper_sri:
            Lc = np.full(4, 0.25)
            self.dNdX_c, detc = _dN_dx(X, shape_dN_dxi(Lc, nn))
            self.wd_c = np.sum(self.wts) * detc
        self.ep = np.zeros((ng, E, 6))
        self.alpha = np.zeros((ng, E, max([6] + [m.n_alpha for m in self.models.values()])))
        self.pbar = np.zeros((ng, E))
        self.pcr = np.zeros((ng, E))      # 相当クリープひずみ
        self.trial = None
        dofs = (3 * mesh.elems[:, :, None] + np.arange(3)).reshape(E, -1)
        self.dofs = dofs
        loc = np.arange(3 * nn)
        ii, jj = np.meshgrid(loc, loc, indexing="ij")
        self.rows = dofs[:, ii].ravel()
        self.cols = dofs[:, jj].ravel()

    def internal(self, u, th_scale=1.0, large=False):
        """内力と接線剛性。th_scale = 熱ひずみの倍率（荷重ステップのランプ）

        large=True: 大変形（Total Lagrangian）。グリーン・ラグランジュひずみ E = Ee + Ep の加法分解で
        第2ピオラ・キルヒホッフ応力 S にリターンマッピングする（大回転・中程度のひずみ向け）。
        """
        if large or self.hyper:
            return self._internal_large(u, th_scale)
        mesh = self.mesh
        E = mesh.n_elems
        ue = u[self.dofs]
        fe = np.zeros((E, 3 * self.nn))
        Ke = np.zeros((E, 3 * self.nn, 3 * self.nn))
        ep_n, al_n, pb_n, pc_n = self.ep.copy(), self.alpha.copy(), self.pbar.copy(), self.pcr.copy()
        sig_all = np.zeros((len(self.wts), E, 6))
        eth = self.eps_th_m[self.elem_mat] * th_scale
        for g in range(len(self.wts)):
            B, wd = self.B[g], self.wd[g]
            eps = np.einsum("eij,ej->ei", B, ue) - eth
            Ct = self.D[self.elem_mat].copy() if self.D.ndim == 3 else np.broadcast_to(self.D, (E, 6, 6)).copy()
            sig = np.einsum("eij,ej->ei", Ct, eps)
            for mi, model in self.models.items():
                sel = np.nonzero(self.elem_mat == mi)[0]
                if len(sel) == 0:
                    continue
                s_M, e_M, a_M, pb, pc, C_M = model.update_all(eps[sel] * W_EPS, self.ep[g, sel], self.alpha[g, sel],
                                                               self.pbar[g, sel], self.pcr[g, sel], self.dt)
                sig[sel] = s_M / W_SIG
                Ct[sel] = C_M / W_SIG[None, :, None] * W_EPS[None, None, :]
                ep_n[g, sel], al_n[g, sel], pb_n[g, sel], pc_n[g, sel] = e_M, a_M, pb, pc
            sig_all[g] = sig
            fe += np.einsum("e,eki,ek->ei", wd, B, sig)
            Ke += np.einsum("e,eki,ekl,elj->eij", wd, B, Ct, B, optimize=True)
        self.trial = (ep_n, al_n, pb_n, sig_all, pc_n)
        f = np.zeros(3 * mesh.n_nodes)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(f.size, f.size)).tocsr()
        return f, K

    def _internal_large(self, u, th_scale):
        from .nonlinear import _green, _tl_BL
        mesh = self.mesh
        E = mesh.n_elems
        nn = self.nn
        Ue = u.reshape(-1, 3)[mesh.elems]
        fe = np.zeros((E, 3 * nn))
        Ke = np.zeros((E, 3 * nn, 3 * nn))
        ep_n, al_n, pb_n, pc_n = self.ep.copy(), self.alpha.copy(), self.pbar.copy(), self.pcr.copy()
        sig_all = np.zeros((len(self.wts), E, 6))
        eth = self.eps_th_m[self.elem_mat] * th_scale
        for g in range(len(self.wts)):
            dNdX, wd = self.dNdX[g], self.wd[g]
            F = np.eye(3) + np.einsum("eak,eal->ekl", Ue, dNdX)
            Eg = _green(F) - eth
            Ct = self.D[self.elem_mat].copy() if self.D.ndim == 3 else np.broadcast_to(self.D, (E, 6, 6)).copy()
            S = np.einsum("eij,ej->ei", Ct, Eg)
            for mi, model in self.models.items():
                sel = np.nonzero(self.elem_mat == mi)[0]
                if len(sel) == 0:
                    continue
                s_M, e_M, a_M, pb, pc, C_M = model.update_all(Eg[sel] * W_EPS, self.ep[g, sel], self.alpha[g, sel],
                                                               self.pbar[g, sel], self.pcr[g, sel], self.dt)
                S[sel] = s_M / W_SIG
                Ct[sel] = C_M / W_SIG[None, :, None] * W_EPS[None, None, :]
                ep_n[g, sel], al_n[g, sel], pb_n[g, sel], pc_n[g, sel] = e_M, a_M, pb, pc
            for mi, hm in self.hyper.items():
                sel = np.nonzero(self.elem_mat == mi)[0]
                if len(sel):
                    # 偏差成分だけ積分点で。体積成分は要素の中心 1 点で（選択的低減積分: ほぼ非圧縮の体積ロッキング対策）
                    S[sel], Ct[sel] = hm.S_C(Eg[sel], "iso" if self.hyper_sri else "all")
            B = _tl_BL(F, dNdX)
            fe += np.einsum("e,eki,ek->ei", wd, B, S)
            Ke += np.einsum("e,eki,ekl,elj->eij", wd, B, Ct, B, optimize=True)
            Sm = np.stack([S[:, [0, 3, 5]], S[:, [3, 1, 4]], S[:, [5, 4, 2]]], axis=1)
            Gm = np.einsum("eai,eij,ebj->eab", dNdX, Sm, dNdX) * wd[:, None, None]
            for k in range(3):
                Ke[:, k::3, k::3] += Gm
            # 表示用のコーシー応力 σ = F S Fᵀ / J
            J = np.linalg.det(F)
            sg = np.einsum("eik,ekl,ejl->eij", F, Sm, F) / J[:, None, None]
            sig_all[g] = np.stack([sg[:, 0, 0], sg[:, 1, 1], sg[:, 2, 2], sg[:, 0, 1], sg[:, 1, 2], sg[:, 0, 2]], 1)
        if self.hyper and self.hyper_sri:
            # 体積成分: 要素の中心で J を評価し、要素の体積で積分（W_vol は 1 点、偏差は 4 点）
            dNc, wc = self.dNdX_c, self.wd_c
            Fc = np.eye(3) + np.einsum("eak,eal->ekl", Ue, dNc)
            Egc = _green(Fc)
            Bc = _tl_BL(Fc, dNc)
            for mi, hm in self.hyper.items():
                sel = np.nonzero(self.elem_mat == mi)[0]
                if not len(sel):
                    continue
                Sv, Cv = hm.S_C(Egc[sel], "vol")
                ws = wc[sel]
                Bs = Bc[sel]
                fe[sel] += np.einsum("e,eki,ek->ei", ws, Bs, Sv)
                Ke[sel] += np.einsum("e,eki,ekl,elj->eij", ws, Bs, Cv, Bs, optimize=True)
                Sm = np.stack([Sv[:, [0, 3, 5]], Sv[:, [3, 1, 4]], Sv[:, [5, 4, 2]]], axis=1)
                Gm = np.einsum("eai,eij,ebj->eab", dNc[sel], Sm, dNc[sel]) * ws[:, None, None]
                for k in range(3):
                    Ke[sel, k::3, k::3] += Gm
                p = hm.pressure(np.linalg.det(Fc[sel]))      # 表示用: コーシー応力に体積成分を足す
                sig_all[:, sel, :3] += p[None, :, None]
        self.trial = (ep_n, al_n, pb_n, sig_all, pc_n)
        f = np.zeros(3 * mesh.n_nodes)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(f.size, f.size)).tocsr()
        return f, K

    def commit(self):
        if self.trial is not None:
            self.ep, self.alpha, self.pbar = self.trial[0].copy(), self.trial[1].copy(), self.trial[2].copy()
            self.pcr = self.trial[4].copy()

    def nodal_scalar(self, arr):
        """積分点のスカラー (ng,E) → 節点（Tet10 は頂点へ外挿、中間節点は平均）"""
        mesh, nn = self.mesh, self.nn
        if nn == 10 and len(self.wts) == 4:
            Ainv = np.linalg.inv(np.asarray(self.pts, float)[:, :4])
            vp = np.einsum("ag,ge->ea", Ainv, arr)
            edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
            vp = np.concatenate([vp, np.stack([(vp[:, i] + vp[:, j]) / 2 for i, j in edges], 1)], 1)
        else:
            vp = np.repeat(arr.mean(axis=0)[:, None], nn, axis=1)
        idx = mesh.elems.ravel()
        cnt = np.maximum(np.bincount(idx, minlength=mesh.n_nodes), 1)
        return np.bincount(idx, np.maximum(vp, 0).ravel(), mesh.n_nodes) / cnt

    def nodal(self):
        """節点の応力（積分点の値を要素内で外挿せず平均）と相当塑性ひずみ"""
        mesh = self.mesh
        sig = self.trial[3] if self.trial is not None else np.zeros((len(self.wts), mesh.n_elems, 6))
        pb = self.pbar
        se = sig.mean(axis=0)              # 要素平均
        pe = pb.mean(axis=0)
        # 積分点の値を節点へ（形状関数で外挿：Tet10 の4点積分 → 頂点）
        nn = self.nn
        if nn == 10 and len(self.wts) == 4:
            A = np.asarray(self.pts, float)[:, :4]   # 積分点の体積座標 (4,4): 値_g = Σ_a A[g,a] 値_a
            Ainv = np.linalg.inv(A)       # 頂点値 = Ainv @ 積分点値
            vs = np.einsum("ag,gek->eak", Ainv, sig)    # (E,4,6)
            vp = np.einsum("ag,ge->ea", Ainv, pb)
            # 中間節点 = 両端頂点の平均
            edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
            vs = np.concatenate([vs, np.stack([(vs[:, i] + vs[:, j]) / 2 for i, j in edges], 1)], 1)
            vp = np.concatenate([vp, np.stack([(vp[:, i] + vp[:, j]) / 2 for i, j in edges], 1)], 1)
        else:
            vs = np.repeat(se[:, None], nn, axis=1)
            vp = np.repeat(pe[:, None], nn, axis=1)
        idx = mesh.elems.ravel()
        cnt = np.maximum(np.bincount(idx, minlength=mesh.n_nodes), 1)
        S = np.column_stack([np.bincount(idx, vs[..., k].ravel(), mesh.n_nodes) for k in range(6)]) / cnt[:, None]
        P = np.bincount(idx, np.maximum(vp, 0).ravel(), mesh.n_nodes) / cnt
        return S, P

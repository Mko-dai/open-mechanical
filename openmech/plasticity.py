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

    def radius(self, p):
        """降伏応力（等方: 表の値、移動: 初期値）と硬化係数 (Hiso, Hkin)"""
        s, H = _table_eval(self.tab, p)
        if self.kin:
            return np.full_like(p, self.sy0), np.zeros_like(p), H
        return s, H, np.zeros_like(p)

    def update(self, eps_M, ep_M, alpha_M, pbar):
        """リターンマッピング。全て Mandel。戻り値: σ_M, ep_M, alpha_M, pbar, C_M（整合接線）"""
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
        self.eps_th_m = eps_th_m if eps_th_m is not None else np.zeros((len(mats), 6))
        # 積分点ごとの B・重み（毎回作り直さないよう保存）
        X = mesh.nodes[mesh.elems]
        self.B, self.wd = [], []
        for g in range(ng):
            dNdx, detJ = _dN_dx(X, self.dN[g])
            self.B.append(_B_matrix(dNdx))
            self.wd.append(self.wts[g] * detJ)
        self.ep = np.zeros((ng, E, 6))
        self.alpha = np.zeros((ng, E, 6))
        self.pbar = np.zeros((ng, E))
        self.trial = None
        dofs = (3 * mesh.elems[:, :, None] + np.arange(3)).reshape(E, -1)
        self.dofs = dofs
        loc = np.arange(3 * nn)
        ii, jj = np.meshgrid(loc, loc, indexing="ij")
        self.rows = dofs[:, ii].ravel()
        self.cols = dofs[:, jj].ravel()

    def internal(self, u, th_scale=1.0):
        """内力と接線剛性。th_scale = 熱ひずみの倍率（荷重ステップのランプ）"""
        mesh = self.mesh
        E = mesh.n_elems
        ue = u[self.dofs]
        fe = np.zeros((E, 3 * self.nn))
        Ke = np.zeros((E, 3 * self.nn, 3 * self.nn))
        ep_n, al_n, pb_n = self.ep.copy(), self.alpha.copy(), self.pbar.copy()
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
                s_M, e_M, a_M, pb, C_M = model.update(eps[sel] * W_EPS, self.ep[g, sel], self.alpha[g, sel],
                                                      self.pbar[g, sel])
                sig[sel] = s_M / W_SIG
                Ct[sel] = C_M / W_SIG[None, :, None] * W_EPS[None, None, :]
                ep_n[g, sel], al_n[g, sel], pb_n[g, sel] = e_M, a_M, pb
            sig_all[g] = sig
            fe += np.einsum("e,eki,ek->ei", wd, B, sig)
            Ke += np.einsum("e,eki,ekl,elj->eij", wd, B, Ct, B, optimize=True)
        self.trial = (ep_n, al_n, pb_n, sig_all)
        f = np.zeros(3 * mesh.n_nodes)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        K = sp.coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(f.size, f.size)).tocsr()
        return f, K

    def commit(self):
        if self.trial is not None:
            self.ep, self.alpha, self.pbar = self.trial[0].copy(), self.trial[1].copy(), self.trial[2].copy()

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

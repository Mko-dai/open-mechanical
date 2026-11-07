"""並列向け AMG 前処理（pyamg で階層を作り、平滑化はチェビシェフ多項式）。

pyamg 標準の Gauss-Seidel 平滑化は1コアでしか動かないため、
行列×ベクトル積だけでできるチェビシェフ平滑化に置き換え、使用コア数で並列に計算する。
共役勾配法の前処理として対称（SPD）なので、収束判定（相対残差 1e-10）まで解けば直接法と同じ解になる。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .parallel import ParallelMatvec


def _lambda_max(mv, n, dinv, iters=12, seed=0):
    """D⁻¹A の最大固有値の推定（べき乗法）"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lam = 1.0
    for _ in range(iters):
        y = dinv * mv(x)
        lam = np.linalg.norm(y) / max(np.linalg.norm(x), 1e-300)
        x = y / max(np.linalg.norm(y), 1e-300)
    return lam


class _Level:
    def __init__(self, A, P=None, R=None, degree=3):
        self.A = sp.csr_matrix(A)
        self.mv = ParallelMatvec(self.A)
        d = self.A.diagonal()
        self.dinv = 1.0 / np.where(d != 0, d, 1.0)
        lm = _lambda_max(self.mv, self.A.shape[0], self.dinv) * 1.15
        self.a, self.b = lm / 30.0, lm
        self.P = P
        self.R = R
        self.degree = degree
        if P is not None:
            self.Pmv = ParallelMatvec(P)
            self.Rmv = ParallelMatvec(R)

    def smooth(self, x, rhs):
        """チェビシェフ平滑化（Jacobi 前処理付き）"""
        a, b = self.a, self.b
        theta, delta = 0.5 * (b + a), 0.5 * (b - a)
        sigma = theta / delta
        rho = 1.0 / sigma
        r = self.dinv * (rhs - self.mv(x)) if x is not None else self.dinv * rhs
        if x is None:
            x = np.zeros_like(rhs)
        d = r / theta
        for k in range(self.degree):
            x += d
            if k == self.degree - 1:
                break
            r -= self.dinv * self.mv(d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * r
            rho = rho_new
        return x


class ChebyshevAMG:
    """M(r) ≈ A⁻¹ r（V サイクル1回）"""

    def __init__(self, ml, degree=3):
        lv = ml.levels
        self.levels = []
        for i in range(len(lv) - 1):
            self.levels.append(_Level(lv[i].A, sp.csr_matrix(lv[i].P), sp.csr_matrix(lv[i].R), degree))
        Ac = sp.csc_matrix(lv[-1].A)
        self.coarse = spla.splu(Ac)
        self.shape = lv[0].A.shape

    def _vcycle(self, k, rhs):
        if k == len(self.levels):
            return self.coarse.solve(rhs)
        L = self.levels[k]
        x = L.smooth(None, rhs)
        res = rhs - L.mv(x)
        x += L.Pmv(self._vcycle(k + 1, L.Rmv(res)))
        return L.smooth(x, rhs)

    def __call__(self, r):
        return self._vcycle(0, np.asarray(r, float))


def build(A, B=None, max_coarse=3000, degree=3):
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A), B=B, symmetry="hermitian", strength="symmetric",
                                           max_coarse=max_coarse, improve_candidates=None)
    return ChebyshevAMG(ml, degree)

"""並列計算の設定（解析に使うコア数）。

Ansys の「使用コア数（Distributed / Shared memory parallel）」に相当します。
  - 剛性マトリクスの組み立て・応力計算: 要素をコア数に分けてスレッドで並列計算
  - 反復法ソルバー（AMG-CG）の行列×ベクトル積: 行ブロックごとに並列
  - 直接法ソルバー・NumPy の行列演算（BLAS）: スレッド数を設定
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np


def available_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


_N = available_cores()
_POOL: ThreadPoolExecutor | None = None
_POOL_N = 0


def get_threads() -> int:
    return _N


def set_threads(n: int | None):
    """使用コア数を設定（0 または None = すべてのコア）"""
    global _N
    n = int(n or 0)
    _N = available_cores() if n <= 0 else max(1, min(n, available_cores()))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = str(_N)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(_N)
    except Exception:
        pass
    return _N


@contextmanager
def threads(n: int | None):
    """with threads(4): ...  の間だけコア数を変える"""
    old = _N
    set_threads(n)
    try:
        yield _N
    finally:
        set_threads(old)


def pool() -> ThreadPoolExecutor:
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != _N:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
        _POOL = ThreadPoolExecutor(max_workers=_N)
        _POOL_N = _N
    return _POOL


def pmap(fn, items):
    """items を並列に処理（コア数1なら逐次）"""
    items = list(items)
    if _N <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    return list(pool().map(fn, items))


def chunks(n: int, min_size: int = 2000, per_core: int = 2):
    """0..n をコア数に応じたブロックに分ける"""
    k = max(1, min(_N * per_core, n // max(min_size, 1)))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class ParallelMatvec:
    """CSR 行列 × ベクトルを行ブロックに分けて並列計算（SciPy の疎行列積は GIL を解放する）"""

    def __init__(self, A):
        import scipy.sparse as sp
        A = sp.csr_matrix(A)
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
        n = A.shape[0]
        if _N > 1 and n > 20000:
            # 非ゼロ数が均等になるように分割
            nnz = A.indptr
            targets = np.linspace(0, nnz[-1], _N + 1)
            cuts = np.searchsorted(nnz, targets)
            cuts[0], cuts[-1] = 0, n
            cuts = np.unique(cuts)
            # 行ブロックはデータを複製せず元の配列を参照する（メモリ節約）
            self.blocks = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                s0, s1 = A.indptr[a], A.indptr[b]
                B = sp.csr_matrix((A.data[s0:s1], A.indices[s0:s1], A.indptr[a:b + 1] - s0),
                                  shape=(b - a, A.shape[1]), copy=False)
                self.blocks.append((a, b, B))
        else:
            self.blocks = None

    def __call__(self, x):
        if self.blocks is None:
            return self.A @ x
        out = np.empty(self.shape[0])

        def run(blk):
            a, b, B = blk
            out[a:b] = B @ x
        list(pool().map(run, self.blocks))
        return out

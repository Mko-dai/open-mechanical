"""ソルバーの速度比較（同じ問題をソルバー・コア数を変えて解き、時間と結果の一致を確認）

  python -m openmech.speedtest
"""
from __future__ import annotations

import platform
import time

import numpy as np

from .materials import Material
from .mesh import structured_box
from .parallel import available_cores, set_threads
from .solver import BC, _direct_backend, _linear_solve, assemble_stiffness, elasticity_matrix, solve


def cantilever(n):
    """100×20×20 の片持ち梁、Tet10。n = 長さ方向の分割数"""
    m = structured_box(100, 20, 20, n, max(2, n // 5), max(2, n // 5)).to_quadratic()
    bcs = [BC("fixed", [0]), BC("force", [1], [0, 0, -1000.0])]
    return m, bcs


def block(n):
    """立方体ブロック（3次元的に詰まった、直接法に厳しい形）"""
    m = structured_box(50, 50, 50, n, n, n).to_quadratic()
    bcs = [BC("fixed", [4]), BC("pressure", [5], [10.0])]
    return m, bcs


def main():
    mat = Material("鋼", 200000, 0.3, 7850, 250, 400)
    ncore = available_cores()
    print(f"CPU: {platform.machine()} / {platform.processor()}  コア数 {ncore}")
    print(f"直接法バックエンド: {_direct_backend()}")
    try:
        import pyamg
        print(f"pyamg {pyamg.__version__}")
    except ImportError:
        print("pyamg なし")
    print()
    cases = [("片持ち梁", cantilever, [40, 80]), ("立方体", block, [18, 26])]
    print(f"{'モデル':10s}{'自由度':>10s}  {'ソルバー':10s}{'コア':>4s}{'組立':>8s}{'求解':>8s}{'合計':>8s}  最大変位 [mm]")
    for name, fn, sizes in cases:
        for n in sizes:
            m, bcs = fn(n)
            ref = None
            runs = [("auto", ncore), ("auto", 1)]
            if 3 * m.n_nodes <= 80_000:
                runs.append(("superlu", 1))
            for meth, cores in runs:
                try:
                    set_threads(cores)
                    r = solve(m, mat, bcs, log=lambda *a: None, method=meth)
                except Exception as e:
                    print(f"{name:10s}{3 * m.n_nodes:>10,d}  {meth:10s}{cores:>4d}  失敗: {e}")
                    continue
                d = float(r.disp_mag.max())
                ref = d if ref is None else ref
                i = r.info
                print(f"{name:10s}{i['自由度数']:>10,d}  {i['ソルバー']:10s}{cores:>4d}{i['組立時間 [s]']:>8.2f}"
                      f"{i['求解時間 [s]']:>8.2f}{i['合計時間 [s]']:>8.2f}  {d:.8g}  (差 {abs(d - ref) / ref:.1e})",
                      flush=True)
    set_threads(0)


if __name__ == "__main__":
    main()

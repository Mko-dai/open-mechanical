"""NAFEMS / Ansys VM ベンチマークの回帰テスト（標準メッシュで目標値の ±3% 以内）"""
import pytest

from openmech import benchmarks as B


@pytest.mark.parametrize("fn", [B.le1, B.le10, B.lame, B.vmc1, B.cantilever])
def test_benchmark_within_3pct(fn):
    _, checks = fn()
    for c in (checks if isinstance(checks, list) else [checks]):
        assert abs(c.ratio - 1) < 0.03, f"{c.name} {c.quantity}: ratio {c.ratio:.3f}"

"""変数・式・パラメトリック解析"""
import math

import pytest

from openmech.params import (ParamError, eval_points, evaluate, resolve, set_field, split_points, sweep_cases,
                             sweep_values)
from openmech.parametric import run_parametric, table
from openmech.project import Project
from openmech.solver import BC


def test_evaluate():
    assert evaluate("2^3 + 1") == 9
    assert evaluate("L/2 + 5", {"L": 10}) == 10
    assert evaluate("（１０＋２）×３") == 36
    assert evaluate("=sqrt(16)") == 4
    assert math.isclose(evaluate("sind(30)"), 0.5)
    assert evaluate("max(a, b, 3)", {"a": 1, "b": 7}) == 7
    assert evaluate("iif(L > 5, 1, 2)", {"L": 6}) == 1
    for bad in ["L+", "foo(1)", "x", "1/0", "__import__('os')", "a.b"]:
        with pytest.raises(ParamError):
            evaluate(bad)


def test_resolve_order_and_cycle():
    env = resolve([{"name": "W", "expr": "L/5"}, {"name": "L", "expr": "100"}, {"name": "H", "expr": "W*2"}])
    assert env == {"L": 100, "W": 20, "H": 40}
    assert resolve([{"name": "L", "expr": "100"}, {"name": "W", "expr": "L/5"}], {"L": 50})["W"] == 10
    with pytest.raises(ParamError):
        resolve([{"name": "a", "expr": "b"}, {"name": "b", "expr": "a"}])
    with pytest.raises(ParamError):
        resolve([{"name": "sin", "expr": "1"}])
    with pytest.raises(ParamError):
        resolve([{"name": "a", "expr": "1"}, {"name": "a", "expr": "2"}])


def test_points_and_fields():
    assert split_points("0,0; L,max(H,2); 0,H") == [["0", "0"], ["L", "max(H,2)"], ["0", "H"]]
    assert eval_points("0,0; L,0; L,H", {"L": 10, "H": 3}) == "0,0; 10,0; 10,3"
    f = {"type": "box", "dx": 1.0}
    set_field(f, "dx", "L*2", {"L": 4})
    assert f["dx"] == 8 and f["expr"] == {"dx": "L*2"}
    set_field(f, "dx", 5.0, {})
    assert f["dx"] == 5 and "expr" not in f


def test_sweep():
    assert sweep_values(10, 20, n=4) == [10, 12.5, 15, 17.5, 20]
    assert sweep_values(10, 20, step=3) == [10, 13, 16, 19]
    assert sweep_values(0, 0, values_text="1, 2.5; 4") == [1, 2.5, 4]
    assert len(sweep_cases({"a": [1, 2], "b": [5, 6, 7]})) == 6
    assert sweep_cases({"a": [1, 2], "b": [5]}, "each") == [{"a": 1}, {"a": 2}, {"b": 5}]
    assert sweep_cases({"a": [1, 2], "b": [5, 6]}, "zip") == [{"a": 1, "b": 5}, {"a": 2, "b": 6}]


def _cantilever():
    p = Project()
    p.params = [{"name": "L", "expr": "100"}, {"name": "H", "expr": "10"}, {"name": "F", "expr": "-100"}]
    p.geometry = {"kind": "box", "lx": 100.0, "ly": 10.0, "lz": 10.0}
    env = p.env()
    set_field(p.geometry, "lx", "L", env)
    set_field(p.geometry, "lz", "H", env)
    p.mesh_size_expr = "H/2"
    p.apply_params()
    m = p.mesh(log=lambda *a: None)
    info = m.face_info()
    f0 = [f["id"] for f in info if abs(f["centroid"][0]) < 1e-6][0]
    f1 = [f["id"] for f in info if abs(f["centroid"][0] - 100) < 1e-6][0]
    p.bcs = [BC("fixed", [f0]), BC("force", [f1], [0, 0, -100.0], exprs={2: "F"})]
    return p, m


def test_project_roundtrip():
    p, _ = _cantilever()
    q = p.copy()
    assert q.params == p.params and q.bcs[1].exprs == {2: "F"} and q.mesh_size_expr == "H/2"
    q.params[0]["expr"] = "50"
    q.apply_params()
    assert q.geometry["lx"] == 50 and p.geometry["lx"] == 100


def test_parametric_cantilever():
    p, m = _cantilever()
    rows = run_parametric(p, sweep_cases({"L": [80, 120], "F": [-100, -200]}),
                          [{"field": "変位 Z [mm]", "stat": "absmax"}, {"field": "質量 [kg]"}],
                          base_mesh=m, log=lambda *a: None)
    assert all(r["状態"] == "完了" for r in rows)
    E, I = p.material.E, 10 * 10 ** 3 / 12
    for r in rows:
        L, F = r["変数"]["L"], r["変数"]["F"]
        d = r["出力"]["絶対値最大 変位 Z [mm]"]
        assert math.isclose(d, F * L ** 3 / (3 * E * I), rel_tol=0.02)
        assert math.isclose(r["出力"]["質量 [kg]"], L * 10 * 10 * p.material.density * 1e-9, rel_tol=1e-6)
    head, body = table(rows, ["L", "F"])
    assert head[:3] == ["番号", "L", "F"] and len(body) == 4
    assert p.geometry["lx"] == 100            # 元のモデルは変わらない

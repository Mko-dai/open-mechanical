import numpy as np
import pytest

from openmech.gui.sketch import Sketch, nice_pitch


def ray_down(x, y):
    return np.array([x, y, 100.0]), np.array([0, 0, -1.0])


def test_box_by_clicks():
    s = Sketch("box", "XY", 0.0, pitch=1.0)
    assert not s.click(*ray_down(10.2, 5.4))
    assert not s.click(*ray_down(-20.1, 30.0))
    # 高さ: 横から見た光線（x 方向に進む）で z=15 の高さ
    assert s.click(np.array([-100.0, 30.0, 15.2]), np.array([1.0, 0, 0]))
    v = s.feature_values()
    assert v == {"x": -20.0, "y": 5.0, "z": 0.0, "dx": 30.0, "dy": 25.0, "dz": 15.0}


def test_cylinder_negative_height_on_yz():
    s = Sketch("cylinder", "YZ", 5.0, pitch=0.5)
    s.click(value=s.to3(1.0, 2.0))
    s.click(value=s.to3(4.0, 6.0))
    assert s.click(value=-10.0)
    v = s.feature_values()
    assert v["axis"] == "X" and v["r"] == pytest.approx(5.0) and v["h"] == 10.0 and v["x"] == -5.0


def test_extrude_polygon_close_by_first_point():
    s = Sketch("extrude", "XY", 0.0, pitch=1.0)
    for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        assert not s.click(*ray_down(x, y))
    assert not s.click(*ray_down(0.1, 0.2))      # 最初の点 → 閉じる
    assert s.kind == "h"
    assert s.click(value=5.0)
    v = s.feature_values()
    assert v["points"] == "0,0; 10,0; 10,10; 0,10" and v["h"] == 5.0


def test_undo_and_pitch():
    s = Sketch("plate", "ZX", 0.0, pitch=1.0)
    s.click(value=s.to3(0, 0))
    s.undo_point()
    assert s.step == 0 and not s.points
    assert nice_pitch(600) == 10 and nice_pitch(150) == 5

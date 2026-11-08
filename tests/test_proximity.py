"""近接の捕捉（板厚の検出）: gmsh の代わりに表面三角形を渡して確かめる"""
import numpy as np

from openmech.geometry import _proximity_points
from openmech.mesh import structured_box


def _fake(m):
    tris = m.boundary_tris

    class G:
        class model:
            class mesh:
                @staticmethod
                def getNodes():
                    return np.arange(1, m.n_nodes + 1), m.nodes.ravel(), None

                @staticmethod
                def getElements(dim, tag):
                    return [2], [None], [(tris + 1).ravel()]

            @staticmethod
            def getEntities(d):
                return [(2, 1)]
    return G


def test_thin_plate_detected():
    pts, size, _ = _proximity_points(_fake(structured_box(100, 50, 0.8, 17, 8, 1)), 6.0, 3, 0.05)
    assert len(pts) > 0 and np.allclose(size, 0.8 / 3)


def test_thick_block_untouched():
    pts, _, _ = _proximity_points(_fake(structured_box(100, 50, 40, 10, 5, 4)), 6.0, 3, 0.05)
    assert len(pts) == 0

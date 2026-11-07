"""メッシュ制御・スイープ・シェルの動作確認（開発用）"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from openmech.geometry import build_mesh, new_feature, mesh_quality
from openmech.project import Project
from openmech.solver import BC
from openmech.materials import Material

def run(name, fn):
    try:
        print("==", name, "->", fn(), flush=True)
    except Exception:
        print("==", name, "FAILED", flush=True)
        traceback.print_exc()

q = lambda m: {k: round(v["最小"], 3) for k, v in mesh_quality(m).items() if not k.startswith("_")}
box = {"kind": "model", "features": [new_feature("box", dx=100, dy=20, dz=5)]}
cyl = {"kind": "model", "features": [new_feature("cylinder", r=10, h=60)]}
two = {"kind": "model", "features": [new_feature("box"), new_feature("box", x=100, dx=50)]}
lg = lambda *a: None
run("tet default", lambda: (build_mesh(box, 0, 2, log=lg).n_elems, q(build_mesh(box, 0, 2, log=lg))))
run("resolution 6", lambda: build_mesh(box, 0, 2, log=lg, settings={"resolution": 6}).n_elems)
run("sweep box divs3", lambda: (lambda m: (m.n_elems, m.n_faces, sorted(set(np.round(m.nodes[:, 2], 3)))[:6], q(m)))(
    build_mesh(box, 4, 2, log=print, local_sizes=[{"type": "sweep", "divs": 3}])))
run("sweep cyl", lambda: (lambda m: (m.n_elems, m.n_faces, m.volume(), np.pi * 100 * 60))(
    build_mesh(cyl, 3, 2, log=print, local_sizes=[{"type": "sweep", "divs": 10, "axis": "Z"}])))
run("body sizing", lambda: [build_mesh(two, 10, 2, log=lg).n_elems,
                            build_mesh(two, 10, 2, log=lg, local_sizes=[{"type": "sizing", "scope": "bodies", "bodies": [1], "size": 3}]).n_elems])
run("sphere", lambda: build_mesh(box, 5, 2, log=lg, local_sizes=[{"type": "sphere", "center": [50, 10, 5], "radius": 10, "size": 0.8}]).n_elems)
run("refinement", lambda: build_mesh(box, 5, 2, log=lg, local_sizes=[{"type": "refinement", "faces": [0], "level": 2}]).n_elems)
run("hard face", lambda: build_mesh(box, 5, 2, log=lg, local_sizes=[{"type": "sizing", "faces": [0], "size": 1, "behavior": "hard"}]).n_elems)

def shell_plate():
    p = Project()
    p.geometry = {"kind": "model", "features": [new_feature("plate", du=100, dv=100, t=2.0)]}
    p.mesh_size = 4
    m = p.mesh(log=print)
    info = m.face_info()
    edges = [f for f in info if f.get("edge")]
    p.bcs = [BC("fixed", [f["id"] for f in edges]), BC("pressure", [0], [0.01])]
    r = p.run(m, log=lg)
    D = 200000 * 8 / (12 * 0.91)
    return m.n_elems, len(edges), r.u[:, 2].min() / (-0.00126 * 0.01 * 1e8 / D)
run("shell plate clamped", shell_plate)

def shell_T():
    p = Project()
    p.geometry = {"kind": "model", "features": [new_feature("plate", du=100, dv=50, t=2.0),
                                                 new_feature("plate", plane="ZX", offset=25, u0=0, v0=0, du=30, dv=100, t=2.0)]}
    p.mesh_size = 5
    m = p.mesh(log=print)
    return m.n_elems, m.n_surf, len(m.edges), len(m.body_ids())
run("shell T", shell_T)

def shell_cyl():
    p = Project()
    p.geometry = {"kind": "model", "features": [new_feature("cyl_shell", r=50, h=100, t=2)]}
    p.mesh_size = 5
    m = p.mesh(log=print)
    return m.n_elems, m.n_surf, len(m.edges)
run("shell cylinder", shell_cyl)

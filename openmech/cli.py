"""コマンドライン実行（GUI なしでバッチ解析）。

  python -m openmech run examples/cantilever.omech --out results/
  python -m openmech faces examples/cantilever.omech      # 面番号の一覧
"""
from __future__ import annotations

import argparse
import os
import sys

from .export import write_csv, write_report, write_vtu
from .project import Project


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openmech", description="open-mechanical 構造解析")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="解析を実行")
    r.add_argument("project")
    r.add_argument("--out", default="results")
    f = sub.add_parser("faces", help="メッシュを作って面の一覧を表示")
    f.add_argument("project")
    sub.add_parser("gui", help="GUI を起動")
    c = sub.add_parser("selfcheck", help="必要なライブラリがそろっていて、メッシュと解析が動くかを確かめる")
    c.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "selfcheck":
        return selfcheck(a.out)

    if a.cmd in (None, "gui"):
        from .gui.app import main as gui_main
        return gui_main()

    p = Project.load(a.project)
    mesh = p.mesh()
    if a.cmd == "faces":
        print(f"{'面':>4} {'面積[mm²]':>12}  重心[mm]                      法線")
        for fi in mesh.face_info():
            c, n = fi["centroid"], fi["normal"]
            print(f"{fi['id'] + 1:>4} {fi['area']:12.4g}  ({c[0]:8.2f},{c[1]:8.2f},{c[2]:8.2f})"
                  f"  ({n[0]:5.2f},{n[1]:5.2f},{n[2]:5.2f})")
        return 0
    res = p.run(mesh)
    os.makedirs(a.out, exist_ok=True)
    write_vtu(os.path.join(a.out, "result.vtu"), res)
    write_csv(os.path.join(a.out, "result.csv"), res)
    write_report(os.path.join(a.out, "report.html"), res, p.bcs, p.title, geometry_desc=p.geometry_desc())
    for k, v in res.summary().items():
        print(f"{k}: {v:.5g}" if isinstance(v, float) else f"{k}: {v}")
    print(f"結果を {a.out}/ に保存しました（result.vtu, result.csv, report.html）")
    return 0


def selfcheck(out=""):
    """配布版の動作確認: ライブラリの読み込み、メッシュ作成、片持ちはりの解析（式と比べる）。 結果を out にも書く"""
    import traceback
    lines, code = [], 0
    try:
        import gmsh
        import numpy as np
        import pyamg
        import pyvista
        import pyvistaqt  # noqa: F401
        import scipy
        from PySide6 import QtCore, QtWidgets  # noqa: F401
        lines.append(f"Python {sys.version.split()[0]} / numpy {np.__version__} / scipy {scipy.__version__} / "
                     f"gmsh {gmsh.__version__} / Qt {QtCore.qVersion()} / pyvista {pyvista.__version__} / "
                     f"pyamg {pyamg.__version__}")
        from .geometry import build_mesh, new_feature
        from .materials import Material
        from .solver import BC, solve
        m = build_mesh({"kind": "model", "features": [new_feature("box", dx=100.0, dy=20.0, dz=20.0)]}, size=5.0,
                       log=lambda *x: None)
        inf = m.face_info()
        lo = min(inf, key=lambda f: f["centroid"][0])["id"]
        hi = max(inf, key=lambda f: f["centroid"][0])["id"]
        r = solve(m, Material("steel", 200000.0, 0.3, 7850.0, 250.0),
                  [BC("fixed", [lo]), BC("force", [hi], [0.0, 0.0, -1000.0])], log=lambda *x: None)
        w, ref = float(-r.u[:, 2].min()), 1000.0 * 100.0 ** 3 / (3 * 200000.0 * 20.0 ** 4 / 12)
        lines.append(f"メッシュ 節点 {m.n_nodes:,} / 片持ちはりのたわみ {w:.5f} mm（はりの式 {ref:.5f}）")
        if abs(w / ref - 1) > 0.05:
            raise RuntimeError("解析の答えが式と合いません")
        from .gui import app as _gui  # noqa: F401    画面の部品が読み込めること
        lines.append("OK")
    except Exception:
        lines.append("NG\n" + traceback.format_exc())
        code = 1
    text = "\n".join(lines)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())

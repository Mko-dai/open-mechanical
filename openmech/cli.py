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
    a = ap.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())

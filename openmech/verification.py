"""Ansys Mechanical との比較検証（5モデル）

  python -m openmech.verification          # 結果を表示し logs/verification.json に保存

各モデルは Ansys Verification Manual / NAFEMS と同じ形状・材料・拘束・荷重・要素次数（SOLID187 相当の
2次四面体）で解き、目標値（理論解・NAFEMS 参照解）と Ansys の公表結果と比べる。
"""
from __future__ import annotations

import json
import os
import platform
import time

from . import benchmarks as bm
from .parallel import available_cores, get_threads, set_threads

MODELS = [
    {
        "id": "VMC1",
        "title": "周辺固定板の等分布荷重（Ansys VMC1）",
        "source": "Ansys Mechanical APDL Verification Manual VMC1（Timoshenko）",
        "ansys_setting": [
            ("解析", "静的構造（線形）"),
            ("形状", "10×10×1 in の板の 1/8 対称モデル（5×5×0.5 in）"),
            ("材料", "E = 1×10⁷ psi, ν = 0.3"),
            ("要素", "SOLID187（2次四面体）, 辺 N1 = 10 分割, 板厚 N2 = 2 分割"),
            ("拘束", "対称面: 変位 0（法線方向）、固定辺: 中立面上の UZ = 0、中立面: UX = UY = 0"),
            ("荷重", "下面に圧力 -500 psi（上下面で合計 1000 psi）"),
            ("評価", "UZ(1) 中央たわみ、SX(2) 固定辺中央の曲げ応力、SX(3) 中央の曲げ応力"),
        ],
        "run": lambda: bm.vmc1(10, 2),
    },
    {
        "id": "LE1",
        "title": "楕円膜の外圧（NAFEMS LE1）",
        "source": "NAFEMS LE1 / Ansys VMFEBSTA-LE1",
        "ansys_setting": [
            ("解析", "静的構造（線形）"),
            ("形状", "内側楕円 (x/2000)²+(y/1000)²=1、外側楕円 (x/3250)²+(y/2750)²=1 の 1/4、厚さ 100 mm"),
            ("材料", "E = 210 GPa, ν = 0.3"),
            ("要素", "2次四面体（SOLID187 相当）、周方向 48 × 半径方向 24 × 厚さ 1"),
            ("拘束", "x=0 面 UX=0、y=0 面 UY=0、面外 UZ=0（平面応力）"),
            ("荷重", "外周面に外向き圧力 10 MPa"),
            ("評価", "点D (2000, 0) の σyy"),
        ],
        "run": lambda: bm.le1(24),
    },
    {
        "id": "LE10",
        "title": "厚板の圧力（NAFEMS LE10）",
        "source": "NAFEMS LE10",
        "ansys_setting": [
            ("解析", "静的構造（線形）"),
            ("形状", "LE1 と同じ楕円形状の 1/4、厚さ 600 mm"),
            ("材料", "E = 210 GPa, ν = 0.3"),
            ("要素", "2次四面体（SOLID187 相当）"),
            ("拘束", "x=0 面 UX=0、y=0 面 UY=0、外周面 UX=UY=0、外周の中立線 UZ=0"),
            ("荷重", "上面に圧力 1 MPa"),
            ("評価", "点D (2000, 0, 300) の σyy"),
        ],
        "run": lambda: bm.le10(12),
    },
    {
        "id": "LAME",
        "title": "厚肉円筒の内圧（ラメの理論解）",
        "source": "弾性理論（ラメの式・平面ひずみ）",
        "ansys_setting": [
            ("解析", "静的構造（線形）"),
            ("形状", "内径 100 mm、外径 200 mm の円筒の 1/4、長さ 20 mm"),
            ("材料", "E = 200 GPa, ν = 0.3"),
            ("要素", "2次四面体（SOLID187 相当）"),
            ("拘束", "対称面: 変位 0（法線方向）、両端面: UZ = 0（平面ひずみ）"),
            ("荷重", "内面に圧力 100 MPa"),
            ("評価", "内面の円周応力 σθ、内面の半径方向変位"),
        ],
        "run": lambda: bm.lame(12),
    },
    {
        "id": "HOLE",
        "title": "穴あき板の引張（応力集中・自由メッシュ）",
        "source": "Peterson の応力集中係数式",
        "ansys_setting": [
            ("解析", "静的構造（線形）"),
            ("形状", "200 × 50 × 5 mm の板、中央に φ10 mm の穴（d/W = 0.2）"),
            ("材料", "E = 200 GPa, ν = 0.3"),
            ("要素", "2次四面体（SOLID187 相当）、自由メッシュ。全体 5 mm、穴の面に面サイズ 0.5 mm"),
            ("拘束", "左端面: 変位 X=0（Y, Z 自由）、剛体運動を止める最小限の点拘束"),
            ("荷重", "右端面に力 X = 5000 N"),
            ("評価", "穴縁の最大 σx（板厚方向の平均）"),
        ],
        "run": lambda: bm.hole_plate(5.0, 0.5),
    },
]


def run(n_cores=0, log=print):
    set_threads(n_cores)
    infos = []
    orig = bm.solve

    def wrapped(*a, **k):
        r = orig(*a, **k)
        infos.append(r.info)
        return r

    bm.solve = wrapped
    out = {"machine": f"{platform.system()} {platform.machine()}", "cores": get_threads(),
           "cores_available": available_cores(), "date": time.strftime("%Y-%m-%d %H:%M"), "models": []}
    try:
        for M in MODELS:
            from .geometry import gmsh_available
            if M["id"] == "HOLE" and not gmsh_available():
                continue
            infos.clear()
            t0 = time.time()
            mesh, chk = M["run"]()
            wall = time.time() - t0
            info = infos[-1] if infos else {}
            chks = chk if isinstance(chk, list) else [chk]
            rec = {"id": M["id"], "title": M["title"], "source": M["source"], "setting": M["ansys_setting"],
                   "dof": info.get("自由度数"), "nodes": mesh.n_nodes, "elems": mesh.n_elems,
                   "solver": info.get("ソルバー"), "t_asm": info.get("組立時間 [s]"), "t_sol": info.get("求解時間 [s]"),
                   "t_total": info.get("合計時間 [s]"), "t_wall": round(wall, 2),
                   "checks": [{"quantity": c.quantity, "unit": c.unit, "target": c.target, "value": c.value,
                               "ratio": c.ratio, "ansys_ratio": c.ansys} for c in chks]}
            out["models"].append(rec)
            log(f"{M['id']:5s} 自由度 {rec['dof']:>8,}  解析 {rec['t_total']:.2f} s（ソルバー {rec['solver']}）")
            for c in rec["checks"]:
                a = f"  Ansys比 {c['ansys_ratio']:.3f}" if c["ansys_ratio"] is not None else ""
                log(f"      {c['quantity']:<22s} 目標 {c['target']:>10.5g}  本ソフト {c['value']:>10.5g}  比 {c['ratio']:.4f}{a}")
    finally:
        bm.solve = orig
    return out


def main():
    os.makedirs("logs", exist_ok=True)
    allres = {}
    cores = sorted({1, available_cores()})
    for nc in cores:
        print(f"\n=== 使用コア数 {nc} ===", flush=True)
        allres[str(nc)] = run(nc)
    with open(os.path.join("logs", "verification.json"), "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=1)
    print("\nlogs/verification.json に保存しました")


if __name__ == "__main__":
    main()

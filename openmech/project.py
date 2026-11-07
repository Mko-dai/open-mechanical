"""解析プロジェクト（形状・メッシュ設定・材料・境界条件）の保存と実行。

保存形式は JSON (.omech)。面番号は CAD の面（STEP）または角度による自動分割に基づくので、
メッシュサイズを変えても同じ面を指します。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .geometry import PRIMITIVES, build_mesh
from .materials import DEFAULT, LIBRARY, Material
from .solver import BC, solve


@dataclass
class Project:
    geometry: dict = field(default_factory=lambda: {"kind": "model", "features": []})
    mesh_size: float = 0.0      # 0 = 自動
    mesh_order: int = 2         # 2 = Tet10（推奨）
    material: Material = field(default_factory=lambda: Material(**LIBRARY[DEFAULT].to_dict()))
    bcs: list[BC] = field(default_factory=list)
    title: str = "構造解析"
    local_sizes: list = field(default_factory=list)  # [{"faces": [...], "size": mm}]
    results: list = field(default_factory=lambda: ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"])
    solver: str = "auto"

    def geometry_desc(self) -> str:
        g = self.geometry
        if g.get("kind") == "model":
            n = len(g.get("features", []))
            return f"モデル（フィーチャー {n} 個）" if n else "（形状なし）"
        if g.get("kind") == "file":
            return g.get("path", "")
        name, params = PRIMITIVES.get(g.get("kind"), (g.get("kind"), []))
        return name + "（" + ", ".join(f"{lab.split(' [')[0]} {g.get(k, d):g}" for k, lab, d in params) + "）"

    def to_dict(self):
        return {"version": 1, "title": self.title, "geometry": self.geometry,
                "mesh": {"size": self.mesh_size, "order": self.mesh_order,
                         "local_sizes": self.local_sizes},
                "material": self.material.to_dict(),
                "bcs": [b.to_dict() for b in self.bcs],
                "results": list(self.results), "solver": self.solver}

    @staticmethod
    def from_dict(d) -> "Project":
        return Project(geometry=d["geometry"], mesh_size=d.get("mesh", {}).get("size", 0.0),
                       mesh_order=d.get("mesh", {}).get("order", 2),
                       material=Material.from_dict(d["material"]),
                       bcs=[BC.from_dict(b) for b in d.get("bcs", [])],
                       title=d.get("title", "構造解析"),
                       local_sizes=d.get("mesh", {}).get("local_sizes", []),
                       results=d.get("results", ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"]),
                       solver=d.get("solver", "auto"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path) -> "Project":
        with open(path, encoding="utf-8") as f:
            return Project.from_dict(json.load(f))

    def mesh(self, log=print):
        return build_mesh(self.geometry, self.mesh_size, self.mesh_order, log=log,
                          local_sizes=self.local_sizes)

    def run(self, mesh=None, log=print, progress=None):
        mesh = mesh if mesh is not None else self.mesh(log=log)
        return solve(mesh, self.material, self.bcs, progress=progress, log=log)

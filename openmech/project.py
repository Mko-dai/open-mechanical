"""解析プロジェクト（形状・メッシュ設定・材料・境界条件）の保存と実行。

保存形式は JSON (.omech)。面番号は CAD の面（STEP）または角度による自動分割に基づくので、
メッシュサイズを変えても同じ面を指します。
"""
from __future__ import annotations

import json

import numpy as np
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
    # Femtet と同じ「ボディ属性名」「材料定数名」の割り当て
    materials: dict = field(default_factory=dict)       # 材料名 → Material（材料定数）
    body_attrs: dict = field(default_factory=dict)      # ボディ属性名 → {"mesh_size": mm}
    body_assign: dict = field(default_factory=dict)     # "ボディ番号" → {"attr": 名, "mat": 名}

    # ------------------------------------------------------------- ボディ属性・材料
    def assignment(self, body: int) -> dict:
        return self.body_assign.get(str(int(body)), {})

    def assign(self, bodies, attr=None, mat=None):
        for b in bodies:
            d = dict(self.body_assign.get(str(int(b)), {}))
            if attr is not None:
                d["attr"] = attr
                if attr and attr not in self.body_attrs:
                    self.body_attrs[attr] = {"mesh_size": 0.0}
            if mat is not None:
                d["mat"] = mat
            self.body_assign[str(int(b))] = d

    def body_local_sizes(self, mesh):
        """ボディ属性のメッシュサイズ → 面サイズ指定に変換"""
        out = []
        if mesh is None:
            return out
        for b in mesh.body_ids():
            attr = self.assignment(b).get("attr")
            h = float(self.body_attrs.get(attr, {}).get("mesh_size", 0) or 0) if attr else 0.0
            if h > 0:
                out.append({"faces": mesh.body_faces(b), "size": h})
        return out

    def elem_materials(self, mesh):
        """要素ごとの材料 (材料リスト, 要素→番号)。割り当てが無ければ従来どおり共通の材料。"""
        if not self.body_assign and not self.materials:
            return None
        bodies = mesh.body_ids()
        missing = [int(b) for b in bodies if self.assignment(b).get("mat") not in self.materials]
        if missing:
            names = ", ".join(f"B{b}" for b in missing)
            raise ValueError(f"材料定数が設定されていないボディがあります: {names}\n"
                             "ボディを選択して「材料名」を入力し、「設定」を押してください。")
        names = sorted({self.assignment(b)["mat"] for b in bodies})
        mats = [self.materials[n] for n in names]
        lut = {int(b): names.index(self.assignment(b)["mat"]) for b in bodies}
        eb = mesh.elem_body if mesh.elem_body is not None else np.zeros(mesh.n_elems, dtype=int)
        table = np.zeros(int(eb.max()) + 1, dtype=np.int64)
        for b, i in lut.items():
            table[b] = i
        elem_mat = table[eb]
        return mats, elem_mat

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
                "results": list(self.results), "solver": self.solver,
                "materials": {k: m.to_dict() for k, m in self.materials.items()},
                "body_attrs": self.body_attrs, "body_assign": self.body_assign}

    @staticmethod
    def from_dict(d) -> "Project":
        return Project(geometry=d["geometry"], mesh_size=d.get("mesh", {}).get("size", 0.0),
                       mesh_order=d.get("mesh", {}).get("order", 2),
                       material=Material.from_dict(d["material"]),
                       bcs=[BC.from_dict(b) for b in d.get("bcs", [])],
                       title=d.get("title", "構造解析"),
                       local_sizes=d.get("mesh", {}).get("local_sizes", []),
                       results=d.get("results", ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"]),
                       solver=d.get("solver", "auto"),
                       materials={k: Material.from_dict(v) for k, v in d.get("materials", {}).items()},
                       body_attrs=(d.get("body_attrs") if isinstance(d.get("body_attrs"), dict)
                                   else {n: {"mesh_size": 0.0} for n in d.get("body_attrs", [])}),
                       body_assign=d.get("body_assign", {}))

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
        return solve(mesh, self.material, self.bcs, progress=progress, log=log,
                     elem_materials=self.elem_materials(mesh))

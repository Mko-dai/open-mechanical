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
from .contact import ContactRegion
from .nonlinear import AnalysisSettings, solve_static
from .solver import BC, solve


@dataclass
class Project:
    geometry: dict = field(default_factory=lambda: {"kind": "model", "features": []})
    mesh_size: float = 0.0      # 0 = 自動
    mesh_order: int = 2         # 2 = Tet10（推奨）
    material: Material = field(default_factory=lambda: Material(**LIBRARY[DEFAULT].to_dict()))
    bcs: list[BC] = field(default_factory=list)
    title: str = "構造解析"
    local_sizes: list = field(default_factory=list)  # メッシュ制御 [{"type":"sizing","faces":[...],"size":mm}, ...]
    mesh_settings: dict = field(default_factory=dict)  # 全体のメッシュ設定（geometry.MESH_DEFAULTS を上書き）
    results: list = field(default_factory=lambda: ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"])
    solver: str = "auto"
    # Femtet と同じ「ボディ属性名」「材料定数名」の割り当て
    materials: dict = field(default_factory=dict)       # 材料名 → Material（材料定数）
    body_attrs: dict = field(default_factory=dict)      # ボディ属性名 → {"mesh_size": mm}
    body_assign: dict = field(default_factory=dict)     # "ボディ番号" → {"attr": 名, "mat": 名}
    contacts: list = field(default_factory=list)        # ContactRegion（接続 → 接触）
    contact_tolerance: float = 0.0                      # 自動検出の許容距離（0 = 自動）
    contact_default_type: str = "bonded"                # 自動検出で作る接触のタイプ
    contact_default_mu: float = 0.2                     # 自動検出で「摩擦あり」のときの摩擦係数
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    # 節点共有（Ansys のトポロジ共有）: "none" = しない / "all" = 接するボディすべて / "parts" = パートごと
    env_temp: float = 22.0      # 環境温度 [°C]（Ansys の「静的構造」の環境温度）
    share_topology: str = "none"
    parts: list = field(default_factory=list)           # [{"name": 名, "bodies": [ボディ番号...]}]
    # 解析の種類: "static"（静的構造）/ "transient"（過渡構造。陰解法・陽解法は dynamic.method）
    analysis_type: str = "static"
    dynamic: dict = field(default_factory=dict)         # dynamics.DynamicSettings の内容
    # 変数（Femtet の「変数」）: [{"name": "L", "expr": "100", "comment": ""}]。寸法・荷重の欄に式で使える
    params: list = field(default_factory=list)
    mesh_size_expr: str = ""                            # メッシュサイズの式（空 = 数値のみ）
    parametric: dict = field(default_factory=dict)      # パラメトリック解析の設定と最後の結果
    # 解析の種類ごとの設定（モーダルのモード数・熱の過渡設定・調和応答の周波数範囲など。analyses.py）
    analysis_opts: dict = field(default_factory=dict)

    def share_map(self) -> dict:
        """{ボディ番号: パート番号}（節点を共有するボディのグループ）"""
        if self.share_topology == "all":
            return {"*": 0}
        if self.share_topology == "parts":
            return {int(b): i for i, p in enumerate(self.parts) for b in p.get("bodies", [])
                    if len(p.get("bodies", [])) > 1}
        return {}

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
                out.append({"type": "sizing", "scope": "bodies", "bodies": [int(b)], "size": h})
        return out

    def body_temperature(self, b) -> float:
        """ボディの温度: 熱条件があればその温度、無ければ環境温度"""
        T = float(self.env_temp)
        for bc in self.bcs:
            if bc.kind == "thermal" and (not bc.bodies or int(b) in [int(x) for x in bc.bodies]):
                T = float(bc.values[0])
        return T

    def body_rotation(self, b):
        attr = self.assignment(b).get("attr")
        cs = (self.body_attrs.get(attr, {}) if attr else {}).get("cs") or self.assignment(b).get("cs")
        if not cs or not any(float(x) for x in cs):
            return None
        from .materials import rotation_xyz
        return rotation_xyz(*[float(x) for x in cs])

    def elem_materials(self, mesh):
        """要素ごとの材料 (MaterialState のリスト, 要素→番号)。

        ボディごとの材料定数・温度（環境温度／熱条件）・材料座標系を反映する。
        """
        bodies = mesh.body_ids() if mesh.elem_body is not None else np.array([0])
        use_assign = bool(self.body_assign or self.materials)
        if use_assign:
            missing = [int(b) for b in bodies if self.assignment(b).get("mat") not in self.materials]
            if missing:
                names = ", ".join(f"B{b}" for b in missing)
                raise ValueError(f"材料定数が設定されていないボディがあります: {names}\n"
                                 "ボディを選択して「材料名」を入力し、「設定」を押してください。")
        keys, states, lut = [], [], {}
        for b in bodies:
            mat = self.materials[self.assignment(b)["mat"]] if use_assign else self.material
            T = self.body_temperature(b)
            attr = self.assignment(b).get("attr")
            cs = tuple(float(x) for x in ((self.body_attrs.get(attr, {}) if attr else {}).get("cs")
                                          or self.assignment(b).get("cs") or [0, 0, 0]))
            key = (mat.name, T, cs)
            if key not in keys:
                keys.append(key)
                states.append(mat.state(T, self.body_rotation(b)))
            lut[int(b)] = keys.index(key)
        eb = mesh.elem_body if mesh.elem_body is not None else np.zeros(mesh.n_elems, dtype=int)
        table = np.zeros(int(eb.max()) + 1, dtype=np.int64)
        for b, i in lut.items():
            table[b] = i
        return states, table[eb]

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
                         "local_sizes": self.local_sizes, "settings": self.mesh_settings},
                "material": self.material.to_dict(),
                "bcs": [b.to_dict() for b in self.bcs],
                "results": list(self.results), "solver": self.solver,
                "materials": {k: m.to_dict() for k, m in self.materials.items()},
                "body_attrs": self.body_attrs, "body_assign": self.body_assign,
                "contacts": [c.to_dict() for c in self.contacts], "contact_tolerance": self.contact_tolerance,
                "contact_default_type": self.contact_default_type, "contact_default_mu": self.contact_default_mu,
                "settings": self.settings.to_dict(),
                "share_topology": self.share_topology, "parts": self.parts, "env_temp": self.env_temp,
                "analysis_type": self.analysis_type, "dynamic": self.dynamic,
                "params": self.params, "mesh_size_expr": self.mesh_size_expr, "parametric": self.parametric,
                "analysis_opts": self.analysis_opts}

    @staticmethod
    def from_dict(d) -> "Project":
        return Project(geometry=d["geometry"], mesh_size=d.get("mesh", {}).get("size", 0.0),
                       mesh_order=d.get("mesh", {}).get("order", 2),
                       material=Material.from_dict(d["material"]),
                       bcs=[BC.from_dict(b) for b in d.get("bcs", [])],
                       title=d.get("title", "構造解析"),
                       local_sizes=d.get("mesh", {}).get("local_sizes", []),
                       mesh_settings=d.get("mesh", {}).get("settings", {}),
                       results=d.get("results", ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"]),
                       solver=d.get("solver", "auto"),
                       materials={k: Material.from_dict(v) for k, v in d.get("materials", {}).items()},
                       body_attrs=(d.get("body_attrs") if isinstance(d.get("body_attrs"), dict)
                                   else {n: {"mesh_size": 0.0} for n in d.get("body_attrs", [])}),
                       body_assign=d.get("body_assign", {}),
                       contacts=[ContactRegion.from_dict(c) for c in d.get("contacts", [])],
                       contact_tolerance=d.get("contact_tolerance", 0.0),
                       contact_default_type=d.get("contact_default_type", "bonded"),
                       contact_default_mu=d.get("contact_default_mu", 0.2),
                       settings=AnalysisSettings.from_dict(d.get("settings", {})),
                       share_topology=d.get("share_topology", "none"), parts=d.get("parts", []),
                       env_temp=d.get("env_temp", 22.0),
                       analysis_type=d.get("analysis_type", "static"), dynamic=d.get("dynamic", {}),
                       params=d.get("params", []), mesh_size_expr=d.get("mesh_size_expr", ""),
                       parametric=d.get("parametric", {}), analysis_opts=d.get("analysis_opts", {}))

    # ------------------------------------------------------------- 変数・式
    def env(self, overrides=None) -> dict:
        """変数の値 {変数名: 値}"""
        from .params import resolve
        return resolve(self.params, overrides)

    def has_expressions(self) -> bool:
        g = self.geometry
        return bool(self.params or self.mesh_size_expr or g.get("expr")
                    or any(f.get("expr") for f in g.get("features", [])) or any(b.exprs for b in self.bcs))

    def apply_params(self, overrides=None) -> dict:
        """変数を評価し、式の入った欄（寸法・荷重・メッシュサイズ）の数値を計算し直す。変数の値を返す"""
        from .params import ParamError, apply_to_dict, evaluate
        env = self.env(overrides)
        g = self.geometry
        apply_to_dict(g, env, "形状")
        for i, f in enumerate(g.get("features", [])):
            apply_to_dict(f, env, f"フィーチャー {i + 1}")
        for b in self.bcs:
            for k, ex in b.exprs.items():
                try:
                    b.values[int(k)] = evaluate(ex, env)
                except ParamError as e:
                    raise ParamError(f"境界条件「{b.name or b.kind}」: {e}") from None
        if self.mesh_size_expr:
            try:
                self.mesh_size = max(0.0, evaluate(self.mesh_size_expr, env))
            except ParamError as e:
                raise ParamError(f"メッシュサイズ: {e}") from None
        return env

    def copy(self) -> "Project":
        return Project.from_dict(json.loads(json.dumps(self.to_dict())))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path) -> "Project":
        with open(path, encoding="utf-8") as f:
            return Project.from_dict(json.load(f))

    def mesh(self, log=print):
        return build_mesh(self.geometry, self.mesh_size, self.mesh_order, log=log,
                          local_sizes=self.local_sizes, share=self.share_map(), settings=self.mesh_settings)

    def check_shell_supported(self):
        s = self.settings
        if s.large_deflection or any(c for c in self.contacts if not c.suppressed):
            raise ValueError("シェル要素・2D 要素は現在、微小変形の静解析（接触なし）に対応しています（塑性は可）")
        if any(b.kind == "thermal" for b in self.bcs) and not self.geometry.get("analysis_2d"):
            raise ValueError("シェル要素ではまだ熱条件に対応していません")

    def needs_nonlinear_solver(self, mesh) -> bool:
        s = self.settings
        # ボディ同士が節点を共有していない（＝接触でつなぐ必要がある）かどうか
        multi = (mesh is not None and getattr(mesh, "node_origin", None) is not None
                 and len(np.unique(mesh.node_origin)) < mesh.n_nodes)
        plastic = any(m.plasticity != "none" for m in (list(self.materials.values()) or [self.material]))
        return bool(self.contacts or multi or plastic or s.large_deflection or s.n_steps > 1 or s.weak_springs
                    or any(b.step_factors for b in self.bcs))

    def run(self, mesh=None, log=print, progress=None):
        """解析を実行して最終時点の結果を返す（全ステップは run_all）"""
        return self.run_all(mesh, log, progress)[-1]

    def run_all(self, mesh=None, log=print, progress=None):
        mesh = mesh if mesh is not None else self.mesh(log=log)
        em = self.elem_materials(mesh)
        from .analyses import EXTRA_TYPES, run_extra
        if self.analysis_type in EXTRA_TYPES or any(b.kind == "bolt_pretension" for b in self.bcs) \
                or (self.analysis_opts or {}).get("thermal_import"):
            return run_extra(self, mesh, em, log=log, progress=progress)
        if self.analysis_type == "transient":
            from .dynamics import DynamicSettings, solve_dynamic
            return solve_dynamic(mesh, self.material, self.bcs, DynamicSettings.from_dict(self.dynamic),
                                 elem_materials=em, log=log, progress=progress)
        if getattr(mesh, "is_2d", False):
            from .plane import solve_2d
            self.check_shell_supported()
            return [solve_2d(mesh, self.material, self.bcs, mode=mesh.mode, elem_materials=em, log=log,
                             settings=self.settings)]
        if getattr(mesh, "is_shell", False):
            from .shell import solve_shell
            self.check_shell_supported()
            return [solve_shell(mesh, self.material, self.bcs, elem_materials=em, log=log, progress=progress,
                                n_cores=self.settings.n_cores, settings=self.settings)]
        if not self.needs_nonlinear_solver(mesh):
            return [solve(mesh, self.material, self.bcs, progress=progress, log=log, elem_materials=em,
                          method=self.settings.solver_type, n_cores=self.settings.n_cores)]
        if not self.contacts and mesh.elem_body is not None and len(mesh.body_ids()) > 1:
            from .contact import detect_contacts
            self.contacts = detect_contacts(mesh, self.contact_tolerance, self.contact_default_type,
                                            self.contact_default_mu)
        return solve_static(mesh, self.material, self.bcs, self.contacts, self.settings, elem_materials=em,
                            log=log, progress=progress, contact_tolerance=self.contact_tolerance)

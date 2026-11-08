"""はり要素（骨組み）解析のダイアログ: 節点・部材・断面・支持・荷重を表で入力し、静解析とモーダルを解いて表示する"""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ..beam import Frame
from ..materials import LIBRARY, Material

EXAMPLE = {
    "nodes": [[0, 0, 0], [0, 0, 3000], [4000, 0, 3000], [4000, 0, 0]],
    "members": [{"n": [0, 1], "section": "柱", "material": "SS400", "div": 10},
                {"n": [1, 2], "section": "梁", "material": "SS400", "div": 16},
                {"n": [2, 3], "section": "柱", "material": "SS400", "div": 10}],
    "sections": {"柱": {"type": "box", "b": 150, "h": 150, "t": 6},
                 "梁": {"type": "I", "h": 300, "b": 150, "tw": 6.5, "tf": 9}},
    "supports": [{"node": 0, "dof": [1, 1, 1, 1, 1, 1]}, {"node": 3, "dof": [1, 1, 1, 1, 1, 1]}],
    "loads": [{"member": 1, "q": [0, 0, -10.0]}, {"node": 1, "F": [5000, 0, 0, 0, 0, 0]}],
    "gravity": [0, 0, -9806.65],
}

SEC_TYPES = {"rect": ("長方形", ["b", "h"]), "circle": ("円", ["d"]), "pipe": ("パイプ", ["d", "t"]),
             "box": ("角パイプ", ["b", "h", "t"]), "I": ("I 形", ["h", "b", "tw", "tf"]),
             "custom": ("任意（A, Iy, Iz, J）", ["A", "Iy", "Iz", "J"])}


def _table(headers, rows):
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))
    return t


def _rows(t):
    out = []
    for i in range(t.rowCount()):
        r = []
        for j in range(t.columnCount()):
            it = t.item(i, j)
            r.append(it.text().strip() if it is not None else "")
        if any(r):
            out.append(r)
    return out


def _buttons(t):
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    a = QPushButton("行を追加")
    d = QPushButton("行を削除")
    a.clicked.connect(lambda: t.insertRow(t.rowCount()))
    d.clicked.connect(lambda: t.removeRow(t.currentRow()) if t.currentRow() >= 0 else None)
    h.addWidget(a)
    h.addWidget(d)
    h.addStretch()
    return w


class FrameDialog(QDialog):
    """［はり（骨組み）解析］"""

    def __init__(self, parent=None, model=None, materials=None):
        super().__init__(parent)
        self.setWindowTitle("はり（骨組み）解析")
        self.resize(1300, 820)
        self.materials = dict(LIBRARY)
        self.materials.update(materials or {})
        self.result = None
        lay = QVBoxLayout(self)
        sp = QSplitter()
        lay.addWidget(sp)
        left = QWidget()
        lv = QVBoxLayout(left)
        self.tabs = QTabWidget()
        lv.addWidget(self.tabs)
        bt = QHBoxLayout()
        for text, fn in [("例を読み込む", lambda: self.load(EXAMPLE)), ("開く…", self.open_file),
                         ("保存…", self.save_file)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            bt.addWidget(b)
        bt.addStretch()
        lv.addLayout(bt)
        run = QHBoxLayout()
        b = QPushButton("静解析")
        b.clicked.connect(self.run_static)
        run.addWidget(b)
        b = QPushButton("モーダル")
        b.clicked.connect(self.run_modal)
        run.addWidget(b)
        self.nmodes = QSpinBox()
        self.nmodes.setRange(1, 50)
        self.nmodes.setValue(6)
        run.addWidget(QLabel("モード数"))
        run.addWidget(self.nmodes)
        self.show_combo = QComboBox()
        self.show_combo.currentIndexChanged.connect(lambda _: self.draw())
        run.addWidget(self.show_combo)
        run.addStretch()
        lv.addLayout(run)
        self.msg = QPlainTextEdit()
        self.msg.setReadOnly(True)
        self.msg.setMaximumHeight(170)
        lv.addWidget(self.msg)
        sp.addWidget(left)
        try:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#ffffff", top="#aecbea")
            sp.addWidget(self.plotter.interactor)
        except Exception:  # noqa: BLE001
            self.plotter = None
            sp.addWidget(QLabel("（3D 表示には pyvistaqt が必要です）"))
        sp.setSizes([620, 680])
        self.load(model or EXAMPLE)

    # ------------------------------------------------------------- 入力
    def load(self, md):
        self.tabs.clear()
        self.t_nodes = _table(["X [mm]", "Y [mm]", "Z [mm]"], md.get("nodes", []))
        self.t_mem = _table(["節点 i", "節点 j", "断面", "材料", "分割数", "向き y（x,y,z 空欄可）"],
                            [[m["n"][0], m["n"][1], m["section"], m.get("material", "SS400"), m.get("div", 8),
                              ",".join(str(x) for x in m.get("orient", []) or [])] for m in md.get("members", [])])
        secs = []
        for name, s in md.get("sections", {}).items():
            keys = SEC_TYPES[s.get("type", "rect")][1]
            secs.append([name, s.get("type", "rect")] + [s.get(k, "") for k in keys] + [""] * (4 - len(keys)))
        self.t_sec = _table(["名前", "種類（rect/circle/pipe/box/I/custom）", "寸法1", "寸法2", "寸法3", "寸法4"], secs)
        self.t_sup = _table(["節点", "UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"],
                            [[s["node"]] + list(s.get("dof", [1] * 6)) for s in md.get("supports", [])])
        lds = []
        for ld in md.get("loads", []):
            if "node" in ld:
                lds.append(["節点", ld["node"]] + list((list(ld["F"]) + [0] * 6)[:6]))
            else:
                lds.append(["部材（分布 N/mm）", ld["member"]] + list(ld["q"]) + [0, 0, 0])
        self.t_load = _table(["種類", "番号", "Fx / qx", "Fy / qy", "Fz / qz", "Mx", "My", "Mz"], lds)
        g = md.get("gravity", [0, 0, 0])
        self.t_grav = _table(["ax [mm/s²]", "ay", "az"], [g])
        for t, name, note in [(self.t_nodes, "節点", "節点番号は 0 から（行の順）"),
                              (self.t_mem, "部材", "材料は材料ライブラリの名前。向き y は断面の y 軸の向き（空欄 = 自動）"),
                              (self.t_sec, "断面", "rect: b h / circle: d / pipe: d t / box: b h t / I: h b tw tf / "
                                                   "custom: A Iy Iz J（mm 系）"),
                              (self.t_sup, "支持", "1 = 拘束、0 = 自由"),
                              (self.t_load, "荷重", "種類「節点」は Fx..Mz [N, N·mm]、「部材（分布 N/mm）」は全体座標の qx qy qz"),
                              (self.t_grav, "重力", "自重（材料の密度）")]:
            w = QWidget()
            v = QVBoxLayout(w)
            v.addWidget(QLabel(note))
            v.addWidget(t)
            v.addWidget(_buttons(t))
            self.tabs.addTab(w, name)
        self.result = None
        self.draw()

    def model(self) -> dict:
        f = float
        nodes = [[f(x) for x in r[:3]] for r in _rows(self.t_nodes)]
        sections = {}
        for r in _rows(self.t_sec):
            typ = r[1] or "rect"
            if typ not in SEC_TYPES:
                raise ValueError(f"断面「{r[0]}」の種類「{typ}」は使えません（rect/circle/pipe/box/I/custom）")
            keys = SEC_TYPES[typ][1]
            sections[r[0]] = {"type": typ, **{k: f(r[2 + i]) for i, k in enumerate(keys)}}
        members = []
        for r in _rows(self.t_mem):
            m = {"n": [int(r[0]), int(r[1])], "section": r[2], "material": r[3] or "SS400", "div": int(r[4] or 8)}
            if r[5]:
                m["orient"] = [f(x) for x in r[5].split(",")]
            if m["section"] not in sections:
                raise ValueError(f"断面「{m['section']}」がありません")
            if m["material"] not in self.materials:
                raise ValueError(f"材料「{m['material']}」がありません")
            members.append(m)
        supports = [{"node": int(r[0]), "dof": [int(float(x or 0)) for x in r[1:7]]} for r in _rows(self.t_sup)]
        loads = []
        for r in _rows(self.t_load):
            vals = [f(x or 0) for x in r[2:8]]
            if r[0].startswith("部材"):
                loads.append({"member": int(r[1]), "q": vals[:3]})
            else:
                loads.append({"node": int(r[1]), "F": vals})
        g = _rows(self.t_grav)
        grav = [f(x or 0) for x in g[0][:3]] if g else [0, 0, 0]
        return {"nodes": nodes, "members": members, "sections": sections, "supports": supports, "loads": loads,
                "gravity": grav}

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "骨組みモデルを開く", "", "JSON (*.json)")
        if path:
            with open(path, encoding="utf-8") as fh:
                self.load(json.load(fh))

    def save_file(self):
        try:
            md = self.model()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "保存", str(e))
            return
        path, _ = QFileDialog.getSaveFileName(self, "骨組みモデルを保存", "frame.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(md, fh, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------- 解析
    def _frame(self):
        md = self.model()
        return Frame(md, {k: m for k, m in self.materials.items() if isinstance(m, Material)})

    def run_static(self):
        try:
            fr = self._frame()
            self.result = fr.solve(log=lambda *a: None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "はり解析", str(e))
            return
        r = self.result
        lines = [f"{k}: {v:.5g}" for k, v in r.summary().items()]
        lines.append("反力 [N, N·mm]:")
        for n, R in r.reactions.items():
            lines.append(f"  節点 {n}: " + ", ".join(f"{x:.4g}" for x in R))
        self.msg.setPlainText("\n".join(lines))
        self.show_combo.blockSignals(True)
        self.show_combo.clear()
        for lab in ("組み合わせ応力 [MPa]", "軸力 [N]", "曲げモーメント [N·mm]", "変位 [mm]"):
            self.show_combo.addItem(lab)
        self.show_combo.blockSignals(False)
        self.draw()

    def run_modal(self):
        try:
            fr = self._frame()
            self.result = fr.modal(self.nmodes.value(), log=lambda *a: None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "はり解析", str(e))
            return
        self.msg.setPlainText("\n".join(f"モード {i + 1}: {f:.5g} Hz" for i, f in enumerate(self.result.freqs)))
        self.show_combo.blockSignals(True)
        self.show_combo.clear()
        for i, f in enumerate(self.result.freqs):
            self.show_combo.addItem(f"モード {i + 1}（{f:.4g} Hz）")
        self.show_combo.blockSignals(False)
        self.draw()

    # ------------------------------------------------------------- 表示
    def draw(self):
        if self.plotter is None:
            return
        import pyvista as pv
        pl = self.plotter
        pl.clear()
        pl.add_axes()
        try:
            fr = self._frame()
        except Exception:  # noqa: BLE001
            return
        X = fr.nodes
        E = fr.elems
        r = self.result
        lines = np.c_[np.full(len(E), 2), E].ravel()
        if r is None:
            poly = pv.PolyData(X, lines=lines)
            pl.add_mesh(poly, color="#4a78c2", line_width=4)
        else:
            k = self.show_combo.currentIndex()
            if r.modes is not None:
                u = r.modes[max(k, 0)][:, :3]
                sc = 0.1 * np.ptp(X, axis=0).max() / max(np.abs(u).max(), 1e-30)
                vals = np.linalg.norm(u, axis=1)
                poly = pv.PolyData(X + sc * u, lines=lines)
                poly.point_data["v"] = vals
                pl.add_mesh(pv.PolyData(X, lines=lines), color="#999999", line_width=1)
                pl.add_mesh(poly, scalars="v", cmap="jet", line_width=5, show_scalar_bar=False)
            else:
                u = r.u[:, :3]
                sc = 0.05 * np.ptp(X, axis=0).max() / max(np.abs(u).max(), 1e-30)
                if k == 3:
                    cell = None
                    pv_vals = np.linalg.norm(u, axis=1)
                else:
                    cell = {0: r.stress, 1: np.abs(r.forces[:, :, 0]).max(axis=1),
                            2: np.hypot(r.forces[:, :, 4], r.forces[:, :, 5]).max(axis=1)}[max(k, 0)]
                poly = pv.PolyData(X + sc * u, lines=lines)
                if cell is not None:
                    poly.cell_data["v"] = cell
                else:
                    poly.point_data["v"] = pv_vals
                pl.add_mesh(pv.PolyData(X, lines=lines), color="#999999", line_width=1)
                pl.add_mesh(poly, scalars="v", cmap="jet", line_width=6, n_colors=9,
                            scalar_bar_args={"title": self.show_combo.currentText(), "vertical": True,
                                             "fmt": "%.4g"})
        for i, x in enumerate(np.asarray(self.model()["nodes"])):
            pl.add_point_labels([x], [str(i)], font_size=12, point_size=6, shape_opacity=0.4)
        pl.reset_camera()
        pl.render()

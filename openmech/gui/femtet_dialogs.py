"""Femtet と同じ操作感のダイアログ。

- 形状の作成（直方体・円柱 …）: 基準点と寸法を入力して［OK］
- 材料定数の編集: 材料名ごとに 弾性定数 / 密度 / 強度 のタブ。材料DBから読み込み可
- Body 属性の編集: ボディ属性名ごとの設定（メッシュサイズ）
- 境界条件の編集: 境界条件名ごとに 機械（変位・荷重・圧力・モーメント）
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QRadioButton, QTabWidget, QVBoxLayout, QWidget)

from ..geometry import AXES, FEATURES, PLANES
from ..materials import LIBRARY, Material


def _spin(v, lo=-1e9, hi=1e9, dec=4, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(float(v))
    s.setMinimumWidth(130)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


def _buttons(dlg, lay):
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.button(QDialogButtonBox.Ok).setText("OK")
    bb.button(QDialogButtonBox.Cancel).setText("キャンセル")
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)


# ---------------------------------------------------------------------------- 形状の作成

class FeatureDialog(QDialog):
    """［直方体の作成］など。Femtet と同じく 基準点 → 寸法 の順に入力"""

    GROUPS = {"x": "基準点", "y": "基準点", "z": "基準点", "axis": "方向", "plane": "スケッチ",
              "offset": "スケッチ", "points": "スケッチ", "path": "ファイル"}

    def __init__(self, ftype, values: dict, title_suffix="の作成", parent=None, targets_text=""):
        super().__init__(parent)
        self.ftype = ftype
        name, _, params = FEATURES[ftype]
        self.setWindowTitle(f"{name}{title_suffix}")
        lay = QVBoxLayout(self)
        groups: dict[str, QFormLayout] = {}
        self.widgets = {}
        for key, label, default in params:
            gname = self.GROUPS.get(key, "寸法")
            if gname not in groups:
                g = QGroupBox(gname)
                groups[gname] = QFormLayout(g)
                lay.addWidget(g)
            v = values.get(key, default)
            if key == "axis":
                w = QComboBox()
                w.addItems(AXES)
                w.setCurrentText(str(v))
            elif key == "plane":
                w = QComboBox()
                w.addItems(PLANES)
                w.setCurrentText(str(v))
            elif key in ("points", "path"):
                w = QLineEdit(str(v))
                w.setMinimumWidth(320)
            else:
                w = _spin(v, suffix="")
            self.widgets[key] = w
            groups[gname].addRow(label.replace(" [mm]", "") + (" [mm]" if "[mm]" in label else ""), w)
        if ftype == "extrude":
            hint = QLabel("頂点はスケッチ平面上の (u, v) 座標を「;」で区切って入力します。\n"
                          "XY 平面: u=X, v=Y ／ YZ 平面: u=Y, v=Z ／ ZX 平面: u=Z, v=X")
            hint.setStyleSheet("color:#555")
            lay.addWidget(hint)
        if targets_text:
            t = QLabel(targets_text)
            t.setStyleSheet("color:#555")
            lay.addWidget(t)
        _buttons(self, lay)

    def values(self) -> dict:
        out = {}
        for k, w in self.widgets.items():
            if isinstance(w, QComboBox):
                out[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                out[k] = w.text()
            else:
                out[k] = w.value()
        return out


# ---------------------------------------------------------------------------- 材料定数

class MaterialDialog(QDialog):
    """［材料定数の編集］"""

    def __init__(self, name: str, mat: Material | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("材料定数の編集")
        self.resize(460, 380)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(name)
        form.addRow("材料名", self.name)
        self.db = QComboBox()
        self.db.addItem("（材料DBから読み込む）")
        self.db.addItems(list(LIBRARY.keys()))
        self.db.activated.connect(self._load_db)
        form.addRow("材料DB", self.db)
        lay.addLayout(form)
        m = mat or LIBRARY.get(name) or Material(name, 200000.0, 0.3, 7850.0, 0.0, 0.0)
        tabs = QTabWidget()
        t1 = QWidget()
        f1 = QFormLayout(t1)
        self.E = _spin(m.E, 1e-3, 1e8, 1, "MPa")
        self.nu = _spin(m.nu, 0.0, 0.4999, 4)
        f1.addRow("ヤング率", self.E)
        f1.addRow("ポアソン比", self.nu)
        f1.addRow(QLabel("等方性・線形弾性"))
        tabs.addTab(t1, "弾性定数")
        t2 = QWidget()
        f2 = QFormLayout(t2)
        self.rho = _spin(m.density, 0.0, 1e6, 2, "kg/m³")
        f2.addRow("密度", self.rho)
        tabs.addTab(t2, "密度")
        t3 = QWidget()
        f3 = QFormLayout(t3)
        self.sy = _spin(m.yield_strength, 0.0, 1e6, 2, "MPa")
        self.su = _spin(m.tensile_strength, 0.0, 1e6, 2, "MPa")
        f3.addRow("引張降伏応力", self.sy)
        f3.addRow("引張強さ", self.su)
        f3.addRow(QLabel("降伏応力を入れると安全率を表示できます"))
        tabs.addTab(t3, "強度")
        lay.addWidget(tabs)
        _buttons(self, lay)

    def _load_db(self, idx):
        if idx <= 0:
            return
        m = LIBRARY[self.db.currentText()]
        self.E.setValue(m.E)
        self.nu.setValue(m.nu)
        self.rho.setValue(m.density)
        self.sy.setValue(m.yield_strength)
        self.su.setValue(m.tensile_strength)
        if not self.name.text().strip():
            self.name.setText(m.name)

    def material(self) -> Material:
        return Material(self.name.text().strip(), self.E.value(), self.nu.value(), self.rho.value(),
                        self.sy.value(), self.su.value())


# ---------------------------------------------------------------------------- Body 属性

class BodyAttrDialog(QDialog):
    """［Body 属性の編集］"""

    def __init__(self, name: str, props: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Body 属性の編集")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(name)
        form.addRow("ボディ属性名", self.name)
        lay.addLayout(form)
        tabs = QTabWidget()
        t = QWidget()
        f = QFormLayout(t)
        self.mesh = _spin(props.get("mesh_size", 0.0), 0.0, 1e6, 3, "mm")
        self.mesh.setSpecialValueText("自動（全体の設定に従う）")
        f.addRow("メッシュサイズ", self.mesh)
        tabs.addTab(t, "メッシュサイズ")
        t2 = QWidget()
        f2 = QFormLayout(t2)
        f2.addRow(QLabel("解析領域: このボディを解析に含める（静的構造解析）"))
        tabs.addTab(t2, "解析領域")
        lay.addWidget(tabs)
        _buttons(self, lay)

    def result(self):
        return self.name.text().strip(), {"mesh_size": self.mesh.value()}


# ---------------------------------------------------------------------------- 境界条件

class BCDialog(QDialog):
    """［境界条件の編集］ 機械タブ: 変位 / 荷重 / 圧力 / モーメント"""

    KINDS = [("displacement", "変位（拘束）"), ("force", "荷重（力）"), ("pressure", "圧力"),
             ("moment", "モーメント")]

    def __init__(self, name: str, kind: str = "displacement", values=None, parent=None, n_faces=0):
        super().__init__(parent)
        self.setWindowTitle("境界条件の編集")
        self.resize(440, 360)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(name)
        form.addRow("境界条件名", self.name)
        if n_faces:
            form.addRow("トポロジ", QLabel(f"面 {n_faces} 個"))
        lay.addLayout(form)
        tabs = QTabWidget()
        mech = QWidget()
        v = QVBoxLayout(mech)
        row = QHBoxLayout()
        self.radios = {}
        if kind == "fixed":
            kind, values = "displacement", [0.0, 0.0, 0.0]
        for k, lab in self.KINDS:
            r = QRadioButton(lab)
            r.toggled.connect(self._switch)
            self.radios[k] = r
            row.addWidget(r)
        v.addLayout(row)
        values = list(values) if values is not None else ([0.0, 0.0, 0.0] if kind == "displacement"
                                                           else [0.0, 0.0, 0.0])
        # 変位
        self.g_disp = QGroupBox("変位（チェックした方向を拘束）")
        fd = QFormLayout(self.g_disp)
        self.dchk, self.dval = [], []
        for i, ax in enumerate("XYZ"):
            val = values[i] if (kind == "displacement" and i < len(values)) else 0.0
            c = QCheckBox("拘束")
            c.setChecked(val is not None)
            sp = _spin(0.0 if val is None else val, dec=6, suffix="mm")
            c.toggled.connect(sp.setEnabled)
            sp.setEnabled(val is not None)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(c)
            h.addWidget(sp)
            fd.addRow(f"{ax} 方向", w)
            self.dchk.append(c)
            self.dval.append(sp)
        v.addWidget(self.g_disp)
        # 荷重 / モーメント
        self.g_vec = QGroupBox("荷重")
        fv = QFormLayout(self.g_vec)
        self.vec = []
        for i, ax in enumerate("XYZ"):
            val = values[i] if (kind in ("force", "moment") and i < len(values) and values[i] is not None) else 0.0
            sp = _spin(val, dec=4)
            fv.addRow(f"{ax} 成分", sp)
            self.vec.append(sp)
        self.vec_note = QLabel("")
        self.vec_note.setStyleSheet("color:#555")
        fv.addRow(self.vec_note)
        v.addWidget(self.g_vec)
        # 圧力
        self.g_p = QGroupBox("圧力")
        fp = QFormLayout(self.g_p)
        self.p = _spin(values[0] if (kind == "pressure" and values and values[0] is not None) else 0.0,
                       dec=6, suffix="MPa")
        fp.addRow("圧力（正: 面を押す向き）", self.p)
        v.addWidget(self.g_p)
        v.addStretch()
        tabs.addTab(mech, "機械")
        lay.addWidget(tabs)
        _buttons(self, lay)
        self.radios.get(kind, self.radios["displacement"]).setChecked(True)
        self._switch()

    def _switch(self, *_):
        k = self.kind()
        self.g_disp.setVisible(k == "displacement")
        self.g_vec.setVisible(k in ("force", "moment"))
        self.g_p.setVisible(k == "pressure")
        self.g_vec.setTitle("荷重（面全体の合計） [N]" if k == "force" else "モーメント（面の重心まわり） [N·mm]")

    def kind(self):
        for k, r in self.radios.items():
            if r.isChecked():
                return k
        return "displacement"

    def result(self):
        k = self.kind()
        if k == "displacement":
            vals = [sp.value() if c.isChecked() else None for c, sp in zip(self.dchk, self.dval)]
            if all(v == 0.0 for v in vals):
                k = "fixed"
        elif k == "pressure":
            vals = [self.p.value()]
        else:
            vals = [sp.value() for sp in self.vec]
        return self.name.text().strip(), k, vals

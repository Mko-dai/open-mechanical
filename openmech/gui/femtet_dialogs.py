"""Femtet と同じ操作感のダイアログ。

- 形状の作成（直方体・円柱 …）: 基準点と寸法を入力して［OK］
- 材料定数の編集: 材料名ごとに 弾性定数 / 密度 / 強度 のタブ。材料DBから読み込み可
- Body 属性の編集: ボディ属性名ごとの設定（メッシュサイズ）
- 境界条件の編集: 境界条件名ごとに 機械（変位・荷重・圧力・モーメント）
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QRadioButton, QTabWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..geometry import AXES, FEATURE_CHOICES, FEATURES, PLANES
from ..materials import LIBRARY, Material
from ..params import ParamError, eval_points, field_text, points_have_expr, set_field
from .expr import ExprEdit, env_of


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
        self._env = env_of(parent)
        name, _, params = FEATURES[ftype]
        self.setWindowTitle(f"{name}{title_suffix}")
        lay = QVBoxLayout(self)
        groups: dict[str, QFormLayout] = {}
        self.widgets = {}
        special = {"move": {"copy": "操作", "dx": "移動", "dy": "移動", "dz": "移動", "angle": "回転",
                            "axis": "回転", "x": "回転", "y": "回転", "z": "回転"},
                   "mirror": {"copy": "操作", "axis": "鏡面", "offset": "鏡面"},
                   "split": {"axis": "切断面", "offset": "切断面", "keep": "切断面"},
                   "stretch": {"axis": "断面", "offset": "断面", "length": "断面"}}.get(ftype, {})
        for key, label, default in params:
            if ftype in ("fillet", "chamfer") and key[0] in "xyz" and key[1:] in ("0", "1"):
                gname = "範囲（この箱に収まる辺だけ）"
            else:
                gname = special.get(key) or self.GROUPS.get(key, "寸法")
            if gname not in groups:
                g = QGroupBox(gname)
                groups[gname] = QFormLayout(g)
                lay.addWidget(g)
            v = values.get(key, default)
            if key in FEATURE_CHOICES:
                w = QComboBox()
                w.addItems(FEATURE_CHOICES[key])
                w.setCurrentText(str(v))
            elif key == "axis":
                w = QComboBox()
                w.addItems(AXES)
                w.setCurrentText(str(v))
            elif key == "plane":
                w = QComboBox()
                w.addItems(PLANES)
                w.setCurrentText(str(v))
            elif key in ("points", "path"):
                w = QLineEdit(field_text(values, key, default) if key == "points" else str(v))
                w.setMinimumWidth(320)
                if key == "points":
                    w.setToolTip("頂点の座標にも式・変数が使えます（例: 0,0; L,0; L,H; 0,H）")
            else:
                w = ExprEdit(field_text(values, key, default), env=lambda: self._env)
            self.widgets[key] = w
            groups[gname].addRow(label.replace(" [mm]", "") + (" [mm]" if "[mm]" in label else ""), w)
        if ftype == "extrude":
            hint = QLabel("頂点はスケッチ平面上の (u, v) 座標を「;」で区切って入力します。\n"
                          "XY 平面: u=X, v=Y ／ YZ 平面: u=Y, v=Z ／ ZX 平面: u=Z, v=X")
            hint.setStyleSheet("color:#555")
            lay.addWidget(hint)
        hints = {"move": "回転してから移動します。「コピー」は元のボディを残して新しいボディを作ります。",
                 "split": "切断面で 2 つのボディに分けます（片側だけ残すこともできます）。",
                 "stretch": "指定位置の断面を、指定方向へ長さぶん引き伸ばします（Femtet のストレッチ）。",
                 "fillet": "条件に合う辺を丸めます。範囲の既定は選択中のボディ全体です。",
                 "chamfer": "条件に合う辺を面取りします。範囲の既定は選択中のボディ全体です。"}
        if ftype in hints:
            h = QLabel(hints[ftype])
            h.setWordWrap(True)
            h.setStyleSheet("color:#555")
            lay.addWidget(h)
        if targets_text:
            t = QLabel(targets_text)
            t.setStyleSheet("color:#555")
            lay.addWidget(t)
        if self._env:
            t = QLabel("変数: " + ", ".join(f"{k} = {v:.6g}" for k, v in list(self._env.items())[:12])
                       + (" …" if len(self._env) > 12 else "") + "（寸法の欄に変数や式を書けます）")
            t.setWordWrap(True)
            t.setStyleSheet("color:#2a6fdb")
            lay.addWidget(t)
        _buttons(self, lay)

    def accept(self):
        for k, w in self.widgets.items():
            try:
                if isinstance(w, ExprEdit):
                    w.value()
                elif k == "points" and points_have_expr(w.text()):
                    eval_points(w.text(), self._env)
            except ParamError as e:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "式の入力", f"{e}")
                return
        super().accept()

    def values(self) -> dict:
        """入力値（式の欄は式の文字列のまま）"""
        out = {}
        for k, w in self.widgets.items():
            if isinstance(w, QComboBox):
                out[k] = w.currentText()
            elif isinstance(w, QLineEdit):
                out[k] = w.text()
            else:
                out[k] = w.raw()
        return out

    def apply_to(self, f: dict):
        """フィーチャー f に入力値を入れる（式は f["expr"] に保存し、数値の欄には計算した値）"""
        for k, v in self.values().items():
            if isinstance(self.widgets[k], ExprEdit) or k == "points":
                set_field(f, k, v, self._env)
            else:
                f[k] = v


# ---------------------------------------------------------------------------- 材料定数

def _table(headers, rows, n_min=3):
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
    t = QTableWidget(max(len(rows), n_min), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            t.setItem(i, j, QTableWidgetItem(f"{float(v):g}"))
    t.horizontalHeader().setStretchLastSection(True)
    t.setMinimumHeight(140)
    return t


def _table_rows(t, ncol):
    rows = []
    for i in range(t.rowCount()):
        vals = []
        for j in range(ncol):
            it = t.item(i, j)
            txt = it.text().strip() if it is not None else ""
            if not txt:
                break
            try:
                vals.append(float(txt))
            except ValueError:
                break
        if len(vals) == ncol:
            rows.append(vals)
    return rows


def _table_buttons(t):
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    a = QPushButton("行を追加")
    a.clicked.connect(lambda: t.insertRow(t.rowCount()))
    d = QPushButton("選択行を削除")
    d.clicked.connect(lambda: [t.removeRow(r) for r in sorted({i.row() for i in t.selectedIndexes()}, reverse=True)])
    h.addWidget(a)
    h.addWidget(d)
    h.addStretch()
    return w


class MaterialDialog(QDialog):
    """［材料定数の編集］（Ansys Engineering Data 相当: 弾性・温度依存・異方性・熱膨張・塑性）"""

    def __init__(self, name: str, mat: Material | None, parent=None):
        from ..materials import ELASTIC_MODELS, PLASTIC_MODELS
        super().__init__(parent)
        self.setWindowTitle("材料定数の編集")
        self.resize(560, 560)
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
        self._m = m
        tabs = QTabWidget()
        # --- 弾性
        t1 = QWidget()
        v1 = QVBoxLayout(t1)
        f1 = QFormLayout()
        self.model = QComboBox()
        for k, lab in ELASTIC_MODELS.items():
            self.model.addItem(lab, k)
        self.model.setCurrentIndex(max(0, self.model.findData(m.elastic)))
        f1.addRow("弾性モデル", self.model)
        v1.addLayout(f1)
        from PySide6.QtWidgets import QStackedWidget
        self.stack = QStackedWidget()
        # 等方
        w_iso = QWidget()
        fi = QVBoxLayout(w_iso)
        ff = QFormLayout()
        self.E = _spin(m.E, 1e-3, 1e8, 1, "MPa")
        self.nu = _spin(m.nu, 0.0, 0.4999, 4)
        ff.addRow("ヤング率", self.E)
        ff.addRow("ポアソン比", self.nu)
        fi.addLayout(ff)
        fi.addWidget(QLabel("温度依存（行があれば温度で補間。範囲外は端の値）"))
        self.ttab = _table(["温度 [°C]", "ヤング率 [MPa]", "ポアソン比"], m.temp_table, 3)
        fi.addWidget(self.ttab)
        fi.addWidget(_table_buttons(self.ttab))
        self.stack.addWidget(w_iso)
        # 直交異方性
        w_or = QWidget()
        fo = QFormLayout(w_or)
        o = dict(Ex=m.E, Ey=m.E, Ez=m.E, nuxy=m.nu, nuyz=m.nu, nuxz=m.nu,
                 Gxy=m.E / (2 * (1 + m.nu)), Gyz=m.E / (2 * (1 + m.nu)), Gxz=m.E / (2 * (1 + m.nu)))
        o.update(m.ortho or {})
        self.ortho = {}
        for k, lab, unit in (("Ex", "ヤング率 X", "MPa"), ("Ey", "ヤング率 Y", "MPa"), ("Ez", "ヤング率 Z", "MPa"),
                             ("nuxy", "ポアソン比 XY", ""), ("nuyz", "ポアソン比 YZ", ""), ("nuxz", "ポアソン比 XZ", ""),
                             ("Gxy", "せん断弾性係数 XY", "MPa"), ("Gyz", "せん断弾性係数 YZ", "MPa"),
                             ("Gxz", "せん断弾性係数 XZ", "MPa")):
            sp = _spin(o[k], 0.0 if k.startswith("nu") else 1e-3, 0.99 if k.startswith("nu") else 1e8,
                       4 if k.startswith("nu") else 1, unit)
            self.ortho[k] = sp
            fo.addRow(lab, sp)
        fo.addRow(QLabel("X, Y, Z は材料座標系（ボディ属性の「材料の向き」で回転できます）"))
        self.stack.addWidget(w_or)
        # 異方性
        w_an = QWidget()
        fa = QVBoxLayout(w_an)
        fa.addWidget(QLabel("剛性マトリクス D [MPa]（成分順 xx, yy, zz, xy, yz, zx・工学せん断ひずみ、対称）"))
        D0 = m.aniso if len(m.aniso) == 6 else m.D_local().tolist() if m.elastic != "anisotropic" else m.aniso
        self.atab = _table(["xx", "yy", "zz", "xy", "yz", "zx"], D0, 6)
        self.atab.setRowCount(6)
        self.atab.setVerticalHeaderLabels(["xx", "yy", "zz", "xy", "yz", "zx"])
        fa.addWidget(self.atab)
        self.stack.addWidget(w_an)
        v1.addWidget(self.stack)
        self.model.currentIndexChanged.connect(lambda i: self.stack.setCurrentIndex(i))
        self.stack.setCurrentIndex(self.model.currentIndex())
        tabs.addTab(t1, "弾性")
        # --- 密度
        t2 = QWidget()
        f2 = QFormLayout(t2)
        self.rho = _spin(m.density, 0.0, 1e6, 2, "kg/m³")
        f2.addRow("密度", self.rho)
        tabs.addTab(t2, "密度")
        # --- 熱膨張
        t4 = QWidget()
        f4 = QFormLayout(t4)
        self.alpha = _spin(m.alpha * 1e6, 0.0, 1e4, 4, "×10⁻⁶ /°C")
        f4.addRow("線膨張係数（等方）", self.alpha)
        ao = list(m.alpha_ortho or [0, 0, 0]) + [0, 0, 0]
        self.ao = [_spin(ao[i] * 1e6, 0.0, 1e4, 4, "×10⁻⁶ /°C") for i in range(3)]
        for i, ax in enumerate("XYZ"):
            f4.addRow(f"線膨張係数 {ax}（直交異方性）", self.ao[i])
        self.tref = _spin(m.ref_temp, -273, 1e4, 2, "°C")
        f4.addRow("ゼロ熱ひずみの参照温度", self.tref)
        f4.addRow(QLabel("熱ひずみ = α ×（温度 − 参照温度）。温度は［環境］タブの「熱条件」か環境温度"))
        tabs.addTab(t4, "熱膨張")
        # --- 強度・塑性
        t3 = QWidget()
        v3 = QVBoxLayout(t3)
        f3 = QFormLayout()
        self.sy = _spin(m.yield_strength, 0.0, 1e6, 2, "MPa")
        self.su = _spin(m.tensile_strength, 0.0, 1e6, 2, "MPa")
        f3.addRow("引張降伏応力", self.sy)
        f3.addRow("引張強さ", self.su)
        self.plast = QComboBox()
        for k, lab in PLASTIC_MODELS.items():
            self.plast.addItem(lab, k)
        self.plast.setCurrentIndex(max(0, self.plast.findData(m.plasticity)))
        f3.addRow("塑性（ミーゼス）", self.plast)
        self.et = _spin(m.tangent_modulus, 0.0, 1e8, 1, "MPa")
        f3.addRow("接線係数（2直線）", self.et)
        v3.addLayout(f3)
        v3.addWidget(QLabel("多直線: 塑性ひずみ と 応力（1行目 = 塑性ひずみ 0 の降伏応力）"))
        self.mtab = _table(["塑性ひずみ [-]", "応力 [MPa]"], m.ml_table, 4)
        v3.addWidget(self.mtab)
        v3.addWidget(_table_buttons(self.mtab))
        v3.addWidget(QLabel("塑性を設定すると非線形解析（弾塑性）になります。安全率は降伏応力で評価します。"))
        tabs.addTab(t3, "強度・塑性")
        # --- 熱伝導（熱解析）
        t5 = QWidget()
        f5 = QFormLayout(t5)
        self.kc = _spin(getattr(m, "conductivity", 0.0), 0.0, 1e5, 4, "W/(m·K)")
        self.cp = _spin(getattr(m, "specific_heat", 0.0), 0.0, 1e6, 2, "J/(kg·K)")
        f5.addRow("熱伝導率", self.kc)
        f5.addRow("比熱", self.cp)
        f5.addRow(QLabel("熱解析（定常熱は熱伝導率、過渡熱は密度と比熱も）に使います"))
        tabs.addTab(t5, "熱伝導")
        # --- 電気（静電場・圧電）
        t6 = QWidget()
        f6 = QFormLayout(t6)
        self.epsr = _spin(getattr(m, "permittivity", 1.0), 1e-3, 1e7, 3)
        f6.addRow("比誘電率 εr", self.epsr)
        pz = getattr(m, "piezo", None) or {}
        e = np.zeros((6, 3)) if not pz else np.asarray(pz.get("e"), float).reshape(6, 3)
        es = list(pz.get("eps_s", [0, 0, 0])) if pz else [0, 0, 0]
        self.e31 = _spin(e[0, 2], -1e3, 1e3, 4, "C/m²")
        self.e33 = _spin(e[2, 2], -1e3, 1e3, 4, "C/m²")
        self.e15 = _spin(e[4, 1], -1e3, 1e3, 4, "C/m²")
        self.es11 = _spin(es[0], 0.0, 1e7, 2)
        self.es33 = _spin(es[2], 0.0, 1e7, 2)
        f6.addRow("圧電定数 e31", self.e31)
        f6.addRow("圧電定数 e33", self.e33)
        f6.addRow("圧電定数 e15", self.e15)
        f6.addRow("比誘電率 ε11ˢ（ひずみ一定）", self.es11)
        f6.addRow("比誘電率 ε33ˢ（ひずみ一定）", self.es33)
        f6.addRow(QLabel("圧電定数がすべて 0 なら圧電材料ではありません。分極は Z 方向（横等方性: e32 = e31、e24 = e15）"))
        tabs.addTab(t6, "電気・圧電")
        # --- 疲労（S-N 曲線）
        t7 = QWidget()
        v7 = QVBoxLayout(t7)
        v7.addWidget(QLabel("S-N 曲線: 繰返し数 と 応力振幅（両振り）。両対数で補間します"))
        self.sntab = _table(["繰返し数 [回]", "応力振幅 [MPa]"], getattr(m, "sn_table", []) or [], 4)
        v7.addWidget(self.sntab)
        v7.addWidget(_table_buttons(self.sntab))
        tabs.addTab(t7, "疲労")
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
        self.alpha.setValue(m.alpha * 1e6)
        self.kc.setValue(getattr(m, "conductivity", 0.0))
        self.cp.setValue(getattr(m, "specific_heat", 0.0))
        if getattr(m, "sn_table", None):
            self.sntab.setRowCount(0)
            for row in m.sn_table:
                r = self.sntab.rowCount()
                self.sntab.insertRow(r)
                for c, v in enumerate(row):
                    self.sntab.setItem(r, c, QTableWidgetItem(f"{v:g}"))
        if not self.name.text().strip():
            self.name.setText(m.name)

    def material(self) -> Material:
        model = self.model.currentData()
        aniso = _table_rows(self.atab, 6) if model == "anisotropic" else []
        if model == "anisotropic" and len(aniso) != 6:
            raise ValueError("異方性の剛性マトリクスは 6 行すべて入力してください")
        mt = _table_rows(self.mtab, 2)
        if self.plast.currentData().startswith("multilinear") and not mt:
            raise ValueError("多直線硬化の表（塑性ひずみ・応力）を入力してください")
        if self.plast.currentData().startswith("bilinear") and self.sy.value() <= 0:
            raise ValueError("2直線硬化には降伏応力が必要です")
        return Material(self.name.text().strip(), self.E.value(), self.nu.value(), self.rho.value(),
                        self.sy.value(), self.su.value(), elastic=model,
                        temp_table=_table_rows(self.ttab, 3),
                        ortho={k: sp.value() for k, sp in self.ortho.items()} if model == "orthotropic" else {},
                        aniso=aniso, alpha=self.alpha.value() * 1e-6,
                        alpha_ortho=[sp.value() * 1e-6 for sp in self.ao] if any(sp.value() for sp in self.ao) else [],
                        ref_temp=self.tref.value(), plasticity=self.plast.currentData(),
                        tangent_modulus=self.et.value(), ml_table=mt,
                        conductivity=self.kc.value(), specific_heat=self.cp.value(),
                        permittivity=self.epsr.value(), piezo=self._piezo(),
                        sn_table=_table_rows(self.sntab, 2))

    def _piezo(self):
        e31, e33, e15 = self.e31.value(), self.e33.value(), self.e15.value()
        if not (e31 or e33 or e15):
            return {}
        e = np.zeros((6, 3))
        e[0, 2] = e[1, 2] = e31          # σx, σy ← Ez
        e[2, 2] = e33                    # σz ← Ez
        e[4, 1] = e15                    # τyz ← Ey
        e[5, 0] = e15                    # τzx ← Ex
        es = [self.es11.value() or self.epsr.value(), self.es11.value() or self.epsr.value(),
              self.es33.value() or self.epsr.value()]
        return {"e": e.tolist(), "eps_s": es}


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
        t3 = QWidget()
        f3 = QFormLayout(t3)
        cs = list(props.get("cs") or [0, 0, 0]) + [0, 0, 0]
        self.cs = [_spin(cs[i], -360, 360, 3, "°") for i in range(3)]
        for i, ax in enumerate("XYZ"):
            f3.addRow(f"{ax} 軸まわりの回転", self.cs[i])
        f3.addRow(QLabel("直交異方性・異方性の材料の向き（材料座標系）。X→Y→Z の順に回転"))
        tabs.addTab(t3, "材料の向き")
        t2 = QWidget()
        f2 = QFormLayout(t2)
        f2.addRow(QLabel("解析領域: このボディを解析に含める（静的構造解析）"))
        tabs.addTab(t2, "解析領域")
        lay.addWidget(tabs)
        _buttons(self, lay)

    def result(self):
        return self.name.text().strip(), {"mesh_size": self.mesh.value(), "cs": [sp.value() for sp in self.cs]}


# ---------------------------------------------------------------------------- 境界条件

class BCDialog(QDialog):
    """［境界条件の編集］ 機械タブ: 変位 / 荷重 / 圧力 / モーメント"""

    KINDS = [("displacement", "変位（拘束）"), ("force", "荷重（力）"), ("pressure", "圧力"),
             ("moment", "モーメント")]

    def __init__(self, name: str, kind: str = "displacement", values=None, parent=None, n_faces=0, exprs=None):
        super().__init__(parent)
        self._env = env_of(parent)
        exprs = {int(k): v for k, v in (exprs or {}).items()}

        def txt(i, v):
            return exprs.get(i, v)
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
            sp = ExprEdit(txt(i, 0.0 if val is None else val), env=lambda: self._env, suffix="mm")
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
            sp = ExprEdit(txt(i, val) if kind in ("force", "moment") else val, env=lambda: self._env)
            fv.addRow(f"{ax} 成分", sp)
            self.vec.append(sp)
        self.vec_note = QLabel("")
        self.vec_note.setStyleSheet("color:#555")
        fv.addRow(self.vec_note)
        v.addWidget(self.g_vec)
        # 圧力
        self.g_p = QGroupBox("圧力")
        fp = QFormLayout(self.g_p)
        self.p = ExprEdit(txt(0, values[0]) if (kind == "pressure" and values and values[0] is not None) else 0.0,
                          env=lambda: self._env, suffix="MPa")
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

    def _edits(self):
        k = self.kind()
        if k == "displacement":
            return [sp if c.isChecked() else None for c, sp in zip(self.dchk, self.dval)]
        if k == "pressure":
            return [self.p]
        return list(self.vec)

    def accept(self):
        for sp in self._edits():
            if sp is not None and not sp.ok():
                from PySide6.QtWidgets import QMessageBox
                try:
                    sp.value()
                except ParamError as e:
                    QMessageBox.warning(self, "式の入力", str(e))
                return
        super().accept()

    def result(self):
        """(名前, 種類, 値, 式 {成分: 式})"""
        k = self.kind()
        eds = self._edits()
        vals = [None if sp is None else sp.value() for sp in eds]
        exprs = {i: sp.raw() for i, sp in enumerate(eds) if sp is not None and isinstance(sp.raw(), str)}
        if k == "displacement" and all(v == 0.0 for v in vals) and not exprs:
            k = "fixed"
        return self.name.text().strip(), k, vals, exprs

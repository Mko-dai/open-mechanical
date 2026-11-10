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
            t.setItem(i, j, QTableWidgetItem(v if isinstance(v, str) else f"{float(v):g}"))
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
        v3.addWidget(QLabel("Chaboche（非線形移動硬化）: 背応力の項ごとに C と γ（繰り返し荷重のバウシンガー効果・"
                            "ラチェット）。降伏応力は上の引張降伏応力。上の多直線の表を入れると等方硬化と組み合わせます"))
        self.chtab = _table(["C [MPa]", "γ [-]"], getattr(m, "chaboche", []), 2)
        self.chtab.setMinimumHeight(90)
        v3.addWidget(self.chtab)
        v3.addWidget(_table_buttons(self.chtab))
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
        # --- 超弾性（ゴム）
        from ..hyperelastic import HYPER_MODELS
        t8 = QWidget()
        f8 = QFormLayout(t8)
        hy = dict(getattr(m, "hyper", None) or {})
        self.hmodel = QComboBox()
        self.hmodel.addItem("なし", "")
        for k, v in HYPER_MODELS.items():
            self.hmodel.addItem(v, k)
        self.hmodel.setCurrentIndex(max(0, self.hmodel.findData(hy.get("model", ""))))
        f8.addRow("モデル", self.hmodel)
        self.hp = {}
        for k, lab in [("mu", "初期せん断弾性率 μ"), ("C10", "C10"), ("C01", "C01"), ("C20", "C20"), ("C30", "C30")]:
            self.hp[k] = _spin(float(hy.get(k, 0.0) or 0.0), -1e6, 1e6, 6, "MPa")
            f8.addRow(lab, self.hp[k])
        self.hd = _spin(float(hy.get("d", 0.0) or 0.0), 0.0, 1e6, 6, "1/MPa")
        f8.addRow("非圧縮性パラメータ d（0 = 非圧縮）", self.hd)
        f8.addRow(QLabel("試験データ（単軸引張: ストレッチ λ・公称応力 MPa）から定数を求められます（カーブフィット）"))
        self.htab = _table(["ストレッチ λ", "公称応力 [MPa]"], hy.get("test", []) or [], 6)
        f8.addRow(self.htab)
        f8.addRow(_table_buttons(self.htab))
        bfit = QPushButton("カーブフィット")
        self.hfit_msg = QLabel("")
        bfit.clicked.connect(self._hyper_fit)
        f8.addRow(bfit, self.hfit_msg)
        f8.addRow(QLabel("超弾性を選ぶと、大変形の非線形静解析で解きます（弾性のヤング率・ポアソン比は使いません）"))
        self.hmodel.currentIndexChanged.connect(self._hyper_enable)
        self._hyper_enable()
        tabs.addTab(t8, "超弾性")
        # --- クリープ（Norton）
        t9 = QWidget()
        f9 = QFormLayout(t9)
        cr = dict(getattr(m, "creep", None) or {})
        self.c1 = QLineEdit(f"{float(cr.get('C1', 0.0) or 0.0):g}")
        self.c1.setPlaceholderText("例: 1e-12")
        self.c2 = _spin(float(cr.get("C2", 1.0) or 1.0), 0.0, 50.0, 4)
        self.c3 = _spin(float(cr.get("C3", 0.0) or 0.0), 0.0, 1e7, 2, "K")
        f9.addRow("C1（係数）", self.c1)
        f9.addRow("C2（応力の指数 n）", self.c2)
        f9.addRow("C3（活性化温度 Q/R）", self.c3)
        f9.addRow(QLabel("Norton 則 ε̇cr = C1 σ^C2 exp(−C3/T)（Ansys の TBOPT=10 と同じ。σ [MPa]、時間 [s]、T [K]）。"
                         "C1 = 0 ならクリープなし。解析設定のステップ終了時間が実時間になります"))
        tabs.addTab(t9, "クリープ")
        # --- 複合材の層の強度
        t10 = QWidget()
        f10 = QFormLayout(t10)
        cs_ = dict(getattr(m, "composite_strength", None) or {})
        self.cst = {}
        for k, lab in [("Xt", "繊維方向 引張 Xt"), ("Xc", "繊維方向 圧縮 Xc"), ("Yt", "直角方向 引張 Yt"),
                       ("Yc", "直角方向 圧縮 Yc"), ("S", "面内せん断 S")]:
            self.cst[k] = _spin(float(cs_.get(k, 0.0) or 0.0), 0.0, 1e6, 2, "MPa")
            f10.addRow(lab, self.cst[k])
        f10.addRow(QLabel("積層シェルの層の破損評価（最大応力・Tsai-Hill・Tsai-Wu）に使います。0 なら評価しません。\n"
                          "繊維方向 = 材料座標の X（直交異方性の Ex の向き）"))
        tabs.addTab(t10, "複合材強度")
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
        ch = _table_rows(self.chtab, 2)
        if self.plast.currentData() == "chaboche" and (not ch or self.sy.value() <= 0):
            raise ValueError("Chaboche には降伏応力と、C・γ の表（1 行以上）が必要です")
        return Material(self.name.text().strip(), self.E.value(), self.nu.value(), self.rho.value(),
                        self.sy.value(), self.su.value(), elastic=model,
                        temp_table=_table_rows(self.ttab, 3),
                        ortho={k: sp.value() for k, sp in self.ortho.items()} if model == "orthotropic" else {},
                        aniso=aniso, alpha=self.alpha.value() * 1e-6,
                        alpha_ortho=[sp.value() * 1e-6 for sp in self.ao] if any(sp.value() for sp in self.ao) else [],
                        ref_temp=self.tref.value(), plasticity=self.plast.currentData(),
                        tangent_modulus=self.et.value(), ml_table=mt, chaboche=ch,
                        conductivity=self.kc.value(), specific_heat=self.cp.value(),
                        permittivity=self.epsr.value(), piezo=self._piezo(),
                        sn_table=_table_rows(self.sntab, 2), hyper=self._hyper(), creep=self._creep(),
                        composite_strength={k: w.value() for k, w in self.cst.items()}
                        if all(w.value() > 0 for w in self.cst.values()) else {})

    def _hyper_enable(self):
        from ..hyperelastic import HYPER_PARAMS
        used = {k for k, _ in HYPER_PARAMS.get(self.hmodel.currentData() or "", [])}
        for k, w in self.hp.items():
            w.setEnabled(k in used)
        self.hd.setEnabled(bool(used))

    def _hyper_fit(self):
        from ..hyperelastic import fit_uniaxial
        model = self.hmodel.currentData()
        if not model:
            self.hfit_msg.setText("モデルを選んでください")
            return
        try:
            c, rms = fit_uniaxial(model, _table_rows(self.htab, 2))
        except ValueError as e:
            self.hfit_msg.setText(str(e))
            return
        for k, v in c.items():
            self.hp[k].setValue(v)
        self.hfit_msg.setText("誤差（相対 RMS） " + f"{rms * 100:.2f} %")

    def _hyper(self):
        from ..hyperelastic import HYPER_PARAMS, coefficients
        model = self.hmodel.currentData()
        if not model:
            return {}
        h = {"model": model, "d": self.hd.value()}
        h.update({k: self.hp[k].value() for k, _ in HYPER_PARAMS[model]})
        tab = _table_rows(self.htab, 2)
        if tab:
            h["test"] = tab
        coefficients(h)            # 定数の確認（初期せん断弾性率 > 0）
        return h

    def _creep(self):
        try:
            c1 = float(self.c1.text().strip() or 0)
        except ValueError:
            raise ValueError("クリープの C1 は数値で入力してください（例: 1e-12）")
        if c1 <= 0:
            return {}
        return {"C1": c1, "C2": self.c2.value(), "C3": self.c3.value()}

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
        # --- 積層（複合材・シェル）
        from ..composite import FAILURE_JP
        t4 = QWidget()
        v4 = QVBoxLayout(t4)
        ly = dict(props.get("layup") or {})
        v4.addWidget(QLabel("積層シェル（複合材）: 層を下面から上面の順に。材料は「材料定数」で直交異方性と強度を設定した名前"))
        self.plies = _table(["材料名", "厚さ [mm]", "角度 [°]"], ly.get("plies", []) or [], 4)
        v4.addWidget(self.plies)
        v4.addWidget(_table_buttons(self.plies))
        f4 = QFormLayout()
        ref = list(ly.get("ref") or [1, 0, 0])
        w4 = QWidget()
        h4 = QHBoxLayout(w4)
        h4.setContentsMargins(0, 0, 0, 0)
        self.ref = [_spin(ref[i], -1e3, 1e3, 3) for i in range(3)]
        for sp_ in self.ref:
            h4.addWidget(sp_)
        f4.addRow("角度 0° の基準方向（全体座標）", w4)
        self.crit = QComboBox()
        for k, v in FAILURE_JP.items():
            self.crit.addItem(v, k)
        self.crit.setCurrentIndex(max(0, self.crit.findData(ly.get("criterion", "tsai_wu"))))
        f4.addRow("破損則", self.crit)
        v4.addLayout(f4)
        v4.addWidget(QLabel("層が無ければ均質なシェル（板厚はサーフェスの厚さ）。積層の板厚は層の合計になります"))
        tabs.addTab(t4, "積層（複合材）")
        self._props = dict(props)
        lay.addWidget(tabs)
        _buttons(self, lay)

    def _layup(self):
        rows = []
        for i in range(self.plies.rowCount()):
            it = [self.plies.item(i, c) for c in range(3)]
            txt = [x.text().strip() if x is not None else "" for x in it]
            if not txt[0]:
                continue
            try:
                rows.append([txt[0], float(txt[1]), float(txt[2] or 0)])
            except ValueError:
                raise ValueError(f"積層の {i + 1} 行目の厚さ・角度は数値で入力してください")
        if not rows:
            return {}
        return {"plies": rows, "ref": [sp_.value() for sp_ in self.ref], "criterion": self.crit.currentData()}

    def result(self):
        out = dict(self._props)
        out.update({"mesh_size": self.mesh.value(), "cs": [sp.value() for sp in self.cs], "layup": self._layup()})
        return self.name.text().strip(), out


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

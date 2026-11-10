"""open-mechanical メインウィンドウ — Ansys Mechanical と同じ画面構成。

  ┌ リボン（ホーム / 環境 / 解析結果 / 表示）──────────────────────┐
  │ アウトライン（ツリー） │                                          │
  │  プロジェクト          │        グラフィックス（3D）              │
  │   └ モデル             │    左クリック: 面を選択 / Ctrl+クリック: 追加 │
  │      ├ ジオメトリ      │    左ドラッグ: 回転  ホイール: 拡大縮小   │
  │      ├ 材料            │                                          │
  │      ├ メッシュ        │                                          │
  │      └ 静的構造        │                                          │
  │         ├ 解析設定     │                                          │
  │         ├ 固定支持 …   │                                          │
  │         └ 解析結果     │──────────────────────────────────────────│
  │ 詳細（プロパティ）     │ メッセージ                               │
  └────────────────────────┴──────────────────────────────────────────┘
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QProgressDialog, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QSpinBox,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSplitter, QTabWidget, QToolButton, QTreeWidget,
                               QListWidget, QListWidgetItem, QSlider, QTreeWidgetItem, QVBoxLayout, QWidget)

import pyvista as pv
from pyvistaqt import QtInteractor

from .. import __version__
from ..contact import (BEHAVIORS, CONTACT_TYPES, DETECTIONS, FORMULATIONS, INTERFACE, PINBALL_MODES, STATUS_JP,
                       STIFFNESS_MODES, TOL_MODES, UPDATE_STIFFNESS, ContactRegion, detect_contacts)
from ..export import write_csv, write_report, write_vtu
from ..params import ParamError, evaluate, field_text, is_number, set_field
from .expr import ExprEdit
from ..geometry import (AXES, CAD_EXT, CONTROL_TYPES, FEATURES, MESH_DEFAULTS, OPS, PLANES, PRIMITIVES,
                        GeometryError, build_mesh, mesh_quality,
                        EDIT_FEATURES, FEATURE_CHOICES, PLANE_MODES, feature_body_keys, gmsh_available, is_shell_model, new_feature,
                        primitive_to_features)
from ..materials import LIBRARY, Material
from ..history import History
from ..project import Project
from ..nonlinear import solve_static
from ..analyses import ANALYSIS_JP, EXTRA_TYPES
from ..solver import BC, solve
from .femtet_dialogs import BCDialog, BodyAttrDialog, FeatureDialog, MaterialDialog
from .viewer import FIELD_EN, feature_edges, surface_polydata, volume_grid

# ---------------------------------------------------------------------------- 定義

BC_JP = {"fixed": "固定支持", "displacement": "変位", "force": "力", "pressure": "圧力",
         "moment": "モーメント", "gravity": "標準地球重力", "thermal": "熱条件",
         "symmetry": "対称領域", "frictionless": "摩擦なし支持"}
from ..solver import BC_EXTRA_JP as _BCX  # noqa: E402
BC_JP.update(_BCX)
BC_COLORS = {"fixed": "#5a4fcf", "displacement": "#e8b100", "force": "#e03131",
             "pressure": "#d6336c", "moment": "#ae3ec9", "gravity": "#f08c00", "thermal": "#c92a2a",
             "symmetry": "#1c7ed6", "frictionless": "#7048e8"}
BC_COLORS.update({"remote_force": "#c2255c", "remote_displacement": "#b8860b", "point_mass": "#495057",
                  "elastic_support": "#2b8a3e", "cylindrical_support": "#364fc7", "bearing_load": "#e8590c",
                  "spring": "#5f3dc4", "joint": "#0b7285", "bolt_pretension": "#862e9c", "temperature": "#e03131",
                  "heat_flux": "#f76707", "heat_flow": "#fd7e14", "convection": "#1098ad", "radiation": "#f59f00",
                  "heat_generation": "#c92a2a", "insulation": "#868e96", "voltage": "#1971c2", "ground": "#343a40",
                  "charge": "#9c36b5", "compression_only": "#2f9e44", "submodel": "#5c940d"})
BC_UNIT = {"displacement": "mm", "force": "N", "pressure": "MPa", "moment": "N·mm", "gravity": "mm/s²",
           "thermal": "°C"}

# 解析結果の種類（Ansys の名称に合わせる）  key = Result.fields() のキー
RESULT_GROUPS = {
    "変形": [("全変形 [mm]", "全変形"), ("変位 X [mm]", "方向変形 X"),
           ("変位 Y [mm]", "方向変形 Y"), ("変位 Z [mm]", "方向変形 Z")],
    "応力": [("相当応力（ミーゼス） [MPa]", "相当応力"), ("最大主応力 [MPa]", "最大主応力"),
           ("最小主応力 [MPa]", "最小主応力"), ("応力 σx [MPa]", "法線応力 X"),
           ("応力 σy [MPa]", "法線応力 Y"), ("応力 σz [MPa]", "法線応力 Z"),
           ("せん断応力 τxy [MPa]", "せん断応力 XY"), ("せん断応力 τyz [MPa]", "せん断応力 YZ"),
           ("せん断応力 τzx [MPa]", "せん断応力 ZX")],
    "ツール": [("安全率（降伏基準）", "安全率（応力ツール）")],
    "速度（過渡）": [("速度 [mm/s]", "速度"), ("速度 X [mm/s]", "方向速度 X"), ("速度 Y [mm/s]", "方向速度 Y"),
                ("速度 Z [mm/s]", "方向速度 Z")],
    "ひずみ・シェル": [("相当塑性ひずみ", "相当塑性ひずみ"), ("相当応力（上面） [MPa]", "相当応力（シェル上面）"),
                 ("相当応力（下面） [MPa]", "相当応力（シェル下面）"), ("相当クリープひずみ", "相当クリープひずみ"),
                 ("複合材 破損指標", "複合材 破損指標（積層シェル）")],
    "接触ツール": [("接触:接触状態", "接触状態"), ("接触:接触圧 [MPa]", "接触圧"), ("接触:貫通量 [mm]", "貫通量"),
               ("接触:すべり距離 [mm]", "すべり距離"), ("接触:摩擦応力 [MPa]", "摩擦応力"), ("接触:隙間 [mm]", "隙間")],
    "熱": [("温度 [°C]", "温度"), ("全熱流束 [W/m²]", "全熱流束"), ("方向熱流束 X [W/m²]", "方向熱流束 X"),
          ("方向熱流束 Y [W/m²]", "方向熱流束 Y"), ("方向熱流束 Z [W/m²]", "方向熱流束 Z")],
    "電気": [("電位 [V]", "電位"), ("電界の大きさ [V/mm]", "電界の大きさ"), ("電界 X [V/mm]", "電界 X"),
           ("電界 Y [V/mm]", "電界 Y"), ("電界 Z [V/mm]", "電界 Z"), ("電束密度の大きさ [C/m²]", "電束密度")],
    "振動": [("変位振幅 [mm]", "変位振幅（調和応答）"), ("変位 1σ（相対） [mm]", "変位 1σ（ランダム振動）"),
           ("相当応力 1σ（Segalman） [MPa]", "相当応力 1σ（ランダム振動）"),
           ("変位（応答スペクトル） [mm]", "変位（応答スペクトル）"),
           ("相当応力（応答スペクトル） [MPa]", "相当応力（応答スペクトル）")],
    "疲労ツール": [("疲労:寿命 [回]", "寿命"), ("疲労:損傷度", "損傷度"), ("疲労:安全率（疲労）", "安全率（疲労）"),
               ("疲労:応力振幅（補正後） [MPa]", "等価応力振幅")],
}
# 解析の種類ごとに最初に表示する結果
DEFAULT_RESULTS = {"thermal": ["温度 [°C]", "全熱流束 [W/m²]"], "electrostatic": ["電位 [V]", "電界の大きさ [V/mm]"],
                   "harmonic": ["変位振幅 [mm]", "相当応力（ミーゼス） [MPa]"],
                   "random": ["変位 1σ（相対） [mm]", "相当応力 1σ（Segalman） [MPa]"],
                   "spectrum": ["変位（応答スペクトル） [mm]", "相当応力（応答スペクトル） [MPa]"],
                   "modal": ["全変形 [mm]"], "buckling": ["全変形 [mm]"],
                   "piezo": ["全変形 [mm]", "電位 [V]", "相当応力（ミーゼス） [MPa]"]}
# 追加の境界条件: 種類 → (値の欄 [(ラベル, 単位, 既定値, 自由にできるか)], オプション)
#   オプション: point（位置）, behavior（剛体/変形可）, bodies（ボディに与える）, ground（地面の点）
BC_SPEC = {
    "remote_force": ([("力 X", "N", 0.0, False), ("力 Y", "N", 0.0, False), ("力 Z", "N", -1000.0, False),
                      ("モーメント X", "N·mm", 0.0, False), ("モーメント Y", "N·mm", 0.0, False),
                      ("モーメント Z", "N·mm", 0.0, False)], {"point", "behavior"}),
    "remote_displacement": ([("変位 X", "mm", 0.0, True), ("変位 Y", "mm", 0.0, True), ("変位 Z", "mm", 0.0, True),
                             ("回転 X", "rad", 0.0, True), ("回転 Y", "rad", 0.0, True), ("回転 Z", "rad", 0.0, True)],
                            {"point", "behavior"}),
    "point_mass": ([("質量", "kg", 1.0, False), ("慣性モーメント Ixx", "kg·mm²", 0.0, False),
                    ("慣性モーメント Iyy", "kg·mm²", 0.0, False), ("慣性モーメント Izz", "kg·mm²", 0.0, False)],
                   {"point", "behavior"}),
    "elastic_support": ([("基礎剛性（面積あたり）", "N/mm³", 10.0, False)], set()),
    "cylindrical_support": ([("半径方向", "mm", 0.0, True), ("軸方向", "mm", None, True), ("接線方向", "mm", None, True)],
                            set()),
    "bearing_load": ([("力 X", "N", 0.0, False), ("力 Y", "N", -1000.0, False), ("力 Z", "N", 0.0, False)], set()),
    "spring": ([("ばね定数", "N/mm", 1000.0, False), ("予荷重（引張 正）", "N", 0.0, False)], {"behavior", "ground"}),
    "bolt_pretension": ([("予張力", "N", 10000.0, False)], {"point", "bodies"}),
    "temperature": ([("温度", "°C", 100.0, False)], set()),
    "heat_flux": ([("熱流束（入る向き 正）", "W/m²", 1000.0, False)], set()),
    "heat_flow": ([("熱流量", "W", 10.0, False)], set()),
    "convection": ([("熱伝達係数", "W/(m²·K)", 10.0, False), ("周囲温度", "°C", 22.0, False)], set()),
    "radiation": ([("放射率", "", 0.8, False), ("周囲温度", "°C", 22.0, False)], set()),
    "heat_generation": ([("内部発熱", "W/m³", 1e6, False)], {"bodies"}),
    "voltage": ([("電圧", "V", 100.0, False)], set()),
    "ground": ([], set()),
    "charge": ([("電荷", "C", 1e-9, False)], set()),
    "joint": ([], {"point"}),
    "submodel": ([], set()),
}
NO_FACE_KINDS = ("gravity", "thermal", "heat_generation", "bolt_pretension")
SUPPORT_LIKE = ("fixed", "displacement", "symmetry", "frictionless", "remote_displacement", "elastic_support",
                "cylindrical_support", "spring", "joint", "submodel")
RESULT_NAME = {k: v for g in RESULT_GROUPS.values() for k, v in g}
RESULT_TYPE_JP = {
    "全変形 [mm]": "全変形", "相当応力（ミーゼス） [MPa]": "相当 (von-Mises) 応力",
    "安全率（降伏基準）": "安全率（最大相当応力基準）",
}
OP_COLORS = {"new": "#4a78c2", "add": "#2c7fb8", "cut": "#c0392b", "intersect": "#8e44ad", "union": "#2c7fb8",
             "edit": "#e67700"}
MAT_PALETTE = ["#8fb8de", "#e5b25d", "#a7d3a6", "#d9a0c8", "#9fd6d2", "#e8a08f", "#b9b3e6", "#c9d68a"]
GEOM_COLOR = "#c9cdd3"
CONTACT_COLOR, TARGET_COLOR = "#e03131", "#1c7ed6"
SELECT_COLOR = "#33d633"
STATUS_OK, STATUS_PENDING, STATUS_ERROR, STATUS_NONE = "ok", "pending", "error", ""


# ---------------------------------------------------------------------------- アイコン

_ICON_CACHE: dict = {}


def make_icon(color: str, text: str = "", status: str = "", size: int = 32) -> QIcon:
    key = (color, text, status, size)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    m = size // 8
    p.drawRoundedRect(m, m, size - 2 * m, size - 2 * m, size // 6, size // 6)
    if text:
        f = QFont()
        f.setBold(True)
        f.setPixelSize(int(size * (0.55 if len(text) == 1 else 0.38)))
        p.setFont(f)
        p.setPen(QColor("white"))
        p.drawText(pm.rect(), Qt.AlignCenter, text)
    if status:
        r = size // 2 - 2
        x0, y0 = size - r - 1, size - r - 1
        bg = {"ok": "#2e9d3a", "pending": "#f2b705", "error": "#2f6fdc"}[status]
        p.setBrush(QColor(bg))
        p.setPen(QColor("white"))
        p.drawEllipse(x0, y0, r, r)
        f = QFont()
        f.setBold(True)
        f.setPixelSize(int(r * 0.8))
        p.setFont(f)
        p.drawText(x0, y0, r, r, Qt.AlignCenter, {"ok": "✓", "pending": "⚡", "error": "?"}[status])
    p.end()
    ic = QIcon(pm)
    _ICON_CACHE[key] = ic
    return ic


# ---------------------------------------------------------------------------- 補助

class Worker(QThread):
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            self.done.emit(self.fn(*self.args, log=lambda m: self.log.emit(str(m)), **self.kw))
        except (GeometryError, ValueError, RuntimeError) as e:
            self.history = getattr(e, "history", None)
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc())


def dspin(val, lo=-1e12, hi=1e12, dec=4, step=1.0, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _labels(pl, pts, texts, size, color, point_color=None, bg=None):
    kw = dict(font_size=size, text_color=color, shape_opacity=0.85 if bg else 0.0,
              show_points=point_color is not None, point_size=10, render_points_as_spheres=True,
              margin=2)
    if point_color:
        kw["point_color"] = point_color
    if bg:
        kw["shape_color"] = bg
    try:
        pl.add_point_labels(np.asarray(pts, float), list(texts), always_visible=True, **kw)
    except TypeError:
        kw.pop("margin", None)
        pl.add_point_labels(np.asarray(pts, float), list(texts), **kw)


def menu_action(menu, text, slot, icon=None):
    a = QAction(text, menu)
    if icon is not None:
        a.setIcon(icon)
    a.triggered.connect(lambda _=False: slot())
    menu.addAction(a)
    return a


def letters(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


# ---------------------------------------------------------------------------- リボン

class Ribbon(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setMaximumHeight(118)
        self.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #c8c8c8; }
            QToolButton { border: 1px solid transparent; border-radius: 3px; padding: 2px 4px; }
            QToolButton:hover { background: #dde8f6; border-color: #9bbbe6; }
            QToolButton:checked { background: #c8daf2; border-color: #6e9bd8; }
        """)

    def add_tab(self, title):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)
        lay.addStretch()
        self.addTab(w, title)
        return w

    def add_group(self, tab, title):
        g = QFrame()
        v = QVBoxLayout(g)
        v.setContentsMargins(4, 0, 4, 0)
        v.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(1)
        v.addLayout(row)
        lab = QLabel(title)
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("color:#666; font-size:11px;")
        v.addWidget(lab)
        lay = tab.layout()
        lay.insertWidget(lay.count() - 1, g)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color:#d0d0d0")
        lay.insertWidget(lay.count() - 1, sep)
        return row

    @staticmethod
    def button(row, text, icon, slot=None, menu=None, checkable=False, big=True):
        b = QToolButton()
        b.setText(text)
        b.setIcon(icon)
        b.setIconSize(QSize(28, 28) if big else QSize(18, 18))
        b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon if big else Qt.ToolButtonTextBesideIcon)
        b.setCheckable(checkable)
        b.setMinimumWidth(56 if big else 0)
        if menu is not None:
            b.setMenu(menu)
            b.setPopupMode(QToolButton.InstantPopup)
        elif slot is not None:
            b.clicked.connect(slot)
        row.addWidget(b)
        return b


# ---------------------------------------------------------------------------- メインウィンドウ

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1500, 950)
        self.project = Project()
        self.project_path: str | None = None
        self.mesh = None
        self.mesh_dirty = True
        self.result = None
        self.results: list = []
        self.result_index = -1
        self.solve_error = ""
        self.solver_log: list[str] = []
        self.selected_faces: list[int] = []
        self.selected_bodies: list[int] = []
        self.worker: Worker | None = None
        self.current = ("geometry",)
        self._tri_src = None
        self._main_actor = None
        self._press = None
        self.probe_text = None
        self._sk = None          # マウスでの作図中の状態（sketch.Sketch）
        # 操作履歴（Femtet と同じく、操作ごとに記録して どの時点にも戻れる）
        self.history = History()
        self.history.reset(self.project.to_dict())
        self._hist_label = None
        self._restoring = False
        self._hist_dialog = None
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.timeout.connect(self._hist_commit)

        self._build_ui()
        self._update_title()
        self.log(f"open-mechanical {__version__} を起動しました。単位系: メートル法 (mm, t, N, s, MPa)")
        if not gmsh_available():
            self.log("⚠ gmsh が見つかりません。直方体以外の形状と CAD 読込には `pip install gmsh` が必要です。")
        self.remesh(reset_camera=True)
        if self.mesh is None:
            self.ribbon.setCurrentIndex(self.tab_model)

    # ======================================================================= UI 構築
    def _build_ui(self):
        # ---- リボン
        self.ribbon = Ribbon()
        self._build_ribbon()
        # ---- アウトライン
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(20, 20))
        self.tree.itemSelectionChanged.connect(self.on_tree_select)
        self.tree.itemDoubleClicked.connect(self.on_tree_double)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_menu)
        self.tree.setStyleSheet("QTreeWidget::item { padding: 2px; }")
        outline = self._titled("アウトライン", self.tree)
        # ---- 詳細
        self.details = QTreeWidget()
        self.details.setColumnCount(2)
        self.details.setHeaderHidden(True)
        self.details.setIndentation(10)
        self.details.setRootIsDecorated(False)
        self.details.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.details.setColumnWidth(0, 150)
        self.details.setStyleSheet("QTreeWidget::item { padding: 1px; }")
        self.details_box = self._titled("詳細", self.details)

        left = QSplitter(Qt.Vertical)
        left.addWidget(outline)
        left.addWidget(self.details_box)
        left.setSizes([420, 420])

        # ---- グラフィックス
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#ffffff", top="#aecbea")
        self.plotter.add_axes()
        gfx = QWidget()
        g = QGridLayout(gfx)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(self._graphics_toolbar(), 0, 0)
        g.addWidget(self._attr_bar(), 1, 0)
        g.addWidget(self._sketch_bar(), 2, 0)
        g.addWidget(self.plotter.interactor, 3, 0)
        self.overlay = QLabel(self.plotter.interactor)
        self.overlay.setStyleSheet("background: rgba(255,255,255,215); color:#111; padding:6px;"
                                   "border:1px solid #bbb; font-size:12px;")
        self.overlay.move(10, 10)
        self.overlay.hide()
        self._install_picking()

        # ---- メッセージ
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(3000)
        bottom = QTabWidget()
        bottom.setDocumentMode(True)
        bottom.addTab(self.logbox, "メッセージ")

        right = QSplitter(Qt.Vertical)
        right.addWidget(gfx)
        right.addWidget(bottom)
        right.setSizes([760, 150])
        main = QSplitter(Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(right)
        main.setSizes([360, 1140])

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self.ribbon)
        v.addWidget(main)
        self.setCentralWidget(central)

        # ---- ステータスバー
        self.sel_status = QLabel("選択なし")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        sb = self.statusBar()
        sb.addWidget(self.sel_status)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(QLabel("メートル法 (mm, t, N, s, mV, mA)　度　rad/s　摂氏"))
        self._build_menu()
        self.rebuild_tree()

    def _titled(self, title, w):
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self_title = QLabel(" " + title)
        self_title.setStyleSheet("background:#e4e8ee; border-bottom:1px solid #c5ccd6; padding:3px;"
                                 "font-weight:bold;")
        v.addWidget(self_title)
        v.addWidget(w)
        if title == "詳細":
            self.details_title = self_title
        return box

    def _build_ribbon(self):
        R = self.ribbon
        # --- ホーム
        t = R.add_tab("ホーム")
        g = R.add_group(t, "ファイル")
        R.button(g, "新規", make_icon("#6c7a89", "新"), self.new_project)
        R.button(g, "開く", make_icon("#6c7a89", "開"), self.open_project)
        R.button(g, "保存", make_icon("#6c7a89", "保"), self.save_project)
        g = R.add_group(t, "解析")
        R.button(g, "解析", make_icon("#2e9d3a", "⚡"), self.run_solve)
        R.button(g, "パラメトリック", make_icon("#1971c2", "パ"), self.run_parametric)
        g = R.add_group(t, "ジオメトリ")
        R.button(g, "インポート", make_icon("#4a78c2", "読"), self.import_file)
        R.button(g, "簡単形状", make_icon("#4a78c2", "形"), lambda: self.select_node(("geometry",)))
        g = R.add_group(t, "メッシュ")
        R.button(g, "生成", make_icon("#3a8fa8", "網"), lambda: self.remesh())
        mm = QMenu(self)
        for key, lab in self._control_menu_items():
            menu_action(mm, lab, lambda key=key: self.add_control(key))
        R.button(g, "メッシュ制御", make_icon("#3a8fa8", "細"), menu=mm)
        g = R.add_group(t, "挿入")
        R.button(g, "支持", make_icon(BC_COLORS["fixed"], "支"), menu=self._bc_menu(["fixed", "displacement", "frictionless", "symmetry"]))
        R.button(g, "荷重", make_icon(BC_COLORS["force"], "荷"),
                 menu=self._bc_menu(["force", "pressure", "moment", "gravity", "thermal"]))
        g = R.add_group(t, "出力")
        R.button(g, "レポート", make_icon("#7a5cc2", "報"), self.export_report)
        R.button(g, "VTU", make_icon("#7a5cc2", "V"), self.export_vtu)
        R.button(g, "CSV", make_icon("#7a5cc2", "表"), self.export_csv)
        # --- モデル（Femtet と同じ: 作成した形状はそれぞれ別のボディ）
        t = R.add_tab("モデル")
        g = R.add_group(t, "作成")
        for ft in ["box", "cylinder", "pipe", "cone", "sphere", "extrude"]:
            R.button(g, FEATURES[ft][0], make_icon(OP_COLORS["new"], FEATURES[ft][1]),
                     lambda _=False, ft=ft: self.create_body(ft))
        g = R.add_group(t, "サーフェス（シェル）")
        for ft in ["plate", "sheet", "disk", "cyl_shell"]:
            R.button(g, FEATURES[ft][0].replace("（シェル）", ""), make_icon("#0ca678", FEATURES[ft][1]),
                     lambda _=False, ft=ft: self.create_body(ft))
        g = R.add_group(t, "ボディ編集")
        for ft in EDIT_FEATURES:
            R.button(g, FEATURES[ft][0], make_icon(OP_COLORS["edit"], FEATURES[ft][1]),
                     lambda _=False, ft=ft: self.edit_body_op(ft))
        g = R.add_group(t, "ブール演算")
        R.button(g, "結合", make_icon(OP_COLORS["union"], "∪"), lambda: self.boolean("union"))
        R.button(g, "除去", make_icon(OP_COLORS["cut"], "−"), lambda: self.boolean("cut"))
        R.button(g, "穴", make_icon(OP_COLORS["cut"], "穴"), self.create_hole)
        g = R.add_group(t, "インポート")
        R.button(g, "STEP", make_icon(OP_COLORS["new"], "読"), self.add_import_feature)
        g = R.add_group(t, "変数")
        R.button(g, "変数", make_icon("#1971c2", "変"), self.edit_variables)
        R.button(g, "パラメトリック\n解析", make_icon("#1971c2", "パ"), self.run_parametric)
        R.button(g, "結果", make_icon("#1971c2", "表"), self.show_parametric_results)
        g = R.add_group(t, "編集")
        R.button(g, "元に戻す", make_icon("#5d6d7e", "↶"), self.undo)
        R.button(g, "やり直す", make_icon("#5d6d7e", "↷"), self.redo)
        R.button(g, "操作履歴", make_icon("#5d6d7e", "履"), self.show_history)
        R.button(g, "形状編集", make_icon("#5d6d7e", "編"), self.edit_selected_body)
        R.button(g, "削除", make_icon("#888888", "×"), self.delete_current)
        R.button(g, "再生成", make_icon("#3a8fa8", "⟳"), lambda: self.remesh())
        g = R.add_group(t, "節点共有")
        R.button(g, "パート作成\n（節点共有）", make_icon("#2f9e44", "共"), self.make_part)
        m_sh = QMenu(self)
        menu_action(m_sh, "節点共有しない（接触でつなぐ）", lambda: self.set_share("none"))
        menu_action(m_sh, "接しているボディすべてで節点共有", lambda: self.set_share("all"))
        menu_action(m_sh, "パートごとに節点共有", lambda: self.set_share("parts"))
        R.button(g, "共有方法", make_icon("#2f9e44", "⇔"), menu=m_sh)
        g = R.add_group(t, "属性")
        R.button(g, "ボディ属性\n材料定数", make_icon("#b07b2c", "材"), lambda: self.set_attr_mode(0))
        R.button(g, "境界条件", make_icon(BC_COLORS["fixed"], "境"), lambda: self.set_attr_mode(1))
        self.tab_model = R.count() - 1
        # --- 接続（Ansys の Connections）
        t = R.add_tab("接続")
        g = R.add_group(t, "接触")
        R.button(g, "自動検出", make_icon("#e8590c", "自"), self.auto_contacts)
        R.button(g, "接触領域", make_icon("#e8590c", "接"), self.add_contact)
        m = QMenu(self)
        for k, v in CONTACT_TYPES.items():
            menu_action(m, v, lambda k=k: self.set_contact_type(k))
        R.button(g, "タイプ", make_icon("#e8590c", "型"), menu=m)
        R.button(g, "反転", make_icon("#e8590c", "⇄"), self.flip_contact)
        self.tab_conn = R.count() - 1
        # --- 環境（静的構造）
        t = R.add_tab("環境")
        g = R.add_group(t, "静的構造")
        R.button(g, "荷重", make_icon(BC_COLORS["force"], "荷"),
                 menu=self._bc_menu(["force", "pressure", "moment", "gravity", "thermal"]))
        R.button(g, "支持", make_icon(BC_COLORS["fixed"], "支"), menu=self._bc_menu(["fixed", "displacement", "frictionless", "symmetry"]))
        for k in ["fixed", "displacement", "frictionless", "symmetry", "force", "pressure", "moment", "gravity", "thermal"]:
            R.button(g, BC_JP[k], make_icon(BC_COLORS[k], BC_JP[k][0]), lambda _=False, k=k: self.add_bc(k))
        g = R.add_group(t, "リモート・接続")
        R.button(g, "リモート", make_icon(BC_COLORS["remote_force"], "遠"),
                 menu=self._bc_menu(["remote_force", "remote_displacement", "point_mass"]))
        R.button(g, "支持（追加）", make_icon(BC_COLORS["cylindrical_support"], "円"),
                 menu=self._bc_menu(["cylindrical_support", "elastic_support", "spring"]))
        R.button(g, "ベアリング", make_icon(BC_COLORS["bearing_load"], "軸"), lambda: self.add_bc("bearing_load"))
        R.button(g, "ボルト予張力", make_icon(BC_COLORS["bolt_pretension"], "ボ"), lambda: self.add_bc("bolt_pretension"))
        R.button(g, "ジョイント", make_icon(BC_COLORS["joint"], "関"), lambda: self.add_bc("joint"))
        g = R.add_group(t, "サブモデル")
        R.button(g, "切断境界", make_icon(BC_COLORS["submodel"], "切"), lambda: self.add_bc("submodel"))
        R.button(g, "粗い結果を保存", make_icon(BC_COLORS["submodel"], "保"), self.save_coarse_result)
        g = R.add_group(t, "熱")
        R.button(g, "熱荷重", make_icon(BC_COLORS["temperature"], "熱"),
                 menu=self._bc_menu(["temperature", "heat_flux", "heat_flow", "convection", "radiation",
                                     "heat_generation"]))
        g = R.add_group(t, "電気")
        R.button(g, "電極", make_icon(BC_COLORS["voltage"], "電"), menu=self._bc_menu(["voltage", "ground", "charge"]))
        g = R.add_group(t, "破壊")
        R.button(g, "き裂", make_icon("#a61e4d", "裂"), self.edit_cracks)
        g = R.add_group(t, "解析")
        R.button(g, "解析", make_icon("#2e9d3a", "⚡"), self.run_solve)
        R.button(g, "はり（骨組み）", make_icon("#4a78c2", "梁"), self.open_frame)
        self.tab_env = R.count() - 1
        # --- 解析結果
        t = R.add_tab("解析結果")
        g = R.add_group(t, "挿入")
        for grp, items in RESULT_GROUPS.items():
            m = QMenu(self)
            for key, name in items:
                menu_action(m, name, lambda k=key: self.add_result(k))
            if grp == "ツール":
                menu_action(m, "ユーザー定義結果（式）", lambda: self.add_result("式:SEQV"))
            R.button(g, grp, make_icon("#c0392b" if grp == "応力" else "#2c7fb8" if grp == "変形" else "#8e6c2e",
                                       grp[0]), menu=m)
        g2 = R.add_group(t, "破壊")
        R.button(g2, "J 積分・K", make_icon("#a61e4d", "J"), self.show_fracture)
        R.button(g2, "積層の層", make_icon("#5f3dc4", "層"), self.show_plies)
        g = R.add_group(t, "表示")
        self.scale_combo = QComboBox()
        for label, v in [("変形なし", 0.0), ("実寸", 1.0), ("自動（強調）", -1.0), ("×10", 10.0),
                         ("×100", 100.0), ("×1000", 1000.0)]:
            self.scale_combo.addItem(label, v)
        self.scale_combo.setCurrentIndex(2)
        self.scale_combo.currentIndexChanged.connect(lambda _: self.refresh_view())
        g.addWidget(QLabel(" 変形倍率 "))
        g.addWidget(self.scale_combo)
        self.bands = QComboBox()
        for label, v in [("9段階", 9), ("連続", 256), ("16段階", 16)]:
            self.bands.addItem(label, v)
        self.bands.currentIndexChanged.connect(lambda _: self.refresh_view())
        g.addWidget(QLabel(" コンター "))
        g.addWidget(self.bands)
        self.edges_combo = QComboBox()
        for label in ["変形前の輪郭を表示", "輪郭なし", "要素を表示"]:
            self.edges_combo.addItem(label)
        self.edges_combo.currentIndexChanged.connect(lambda _: self.refresh_view())
        g.addWidget(self.edges_combo)
        self.minmax_chk = QCheckBox("最大・最小を表示")
        self.minmax_chk.setChecked(True)
        self.minmax_chk.setToolTip("コンター上の最大値（赤）・最小値（青）の札を表示します")
        self.minmax_chk.toggled.connect(lambda _: self.refresh_view())
        g.addWidget(self.minmax_chk)
        self.anim_btn = R.button(g, "アニメーション", make_icon("#c0392b", "▶"), self.toggle_animation, checkable=True,
                                 big=False)
        self.section_combo = QComboBox()
        for label, v in [("断面なし", None), ("断面 X", 0), ("断面 Y", 1), ("断面 Z", 2)]:
            self.section_combo.addItem(label, v)
        self.section_combo.setToolTip("断面（切断面）で内部のコンターを表示（Ansys の断面）")
        self.section_combo.currentIndexChanged.connect(lambda _: self.refresh_view())
        g.addWidget(self.section_combo)
        self.section_pos = QSlider(Qt.Horizontal)
        self.section_pos.setRange(0, 100)
        self.section_pos.setValue(50)
        self.section_pos.setFixedWidth(90)
        self.section_pos.setToolTip("断面の位置（形状の範囲の %）")
        self.section_pos.valueChanged.connect(lambda _: self.refresh_view() if self.section_combo.currentData() is not None else None)
        g.addWidget(self.section_pos)
        self.tab_res = R.count() - 1
        # --- 表示
        t = R.add_tab("表示")
        g = R.add_group(t, "視点")
        for text, fn in [("等角", lambda: self.plotter.view_isometric()),
                         ("全体", lambda: self.plotter.reset_camera()),
                         ("+X", lambda: self.plotter.view_yz()), ("-X", lambda: self.plotter.view_yz(negative=True)),
                         ("+Y", lambda: self.plotter.view_xz(negative=True)), ("-Y", lambda: self.plotter.view_xz()),
                         ("+Z", lambda: self.plotter.view_xy()), ("-Z", lambda: self.plotter.view_xy(negative=True))]:
            R.button(g, text, make_icon("#5d6d7e", text[:2]), fn)
        g = R.add_group(t, "表示")
        self.show_mesh_btn = R.button(g, "メッシュ表示", make_icon("#3a8fa8", "網"),
                                      lambda: self.refresh_view(), checkable=True)
        R.button(g, "画像保存", make_icon("#5d6d7e", "画"), self.save_screenshot)

    def _bc_menu(self, kinds):
        m = QMenu(self)
        for k in kinds:
            menu_action(m, BC_JP[k], lambda k=k: self.add_bc(k), make_icon(BC_COLORS[k], BC_JP[k][0]))
        return m

    def _graphics_toolbar(self):
        bar = QWidget()
        bar.setStyleSheet("QToolButton{border:1px solid transparent;padding:2px;}"
                          "QToolButton:checked{background:#c8daf2;border-color:#6e9bd8;}"
                          "QToolButton:hover{background:#e6eef9;}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(2)
        self.sel_face_btn = Ribbon.button(h, "面", make_icon(SELECT_COLOR, "面"), checkable=True, big=False)
        self.sel_body_btn = Ribbon.button(h, "ボディ", make_icon("#888888", "体"), checkable=True, big=False)
        self.sel_edge_btn = Ribbon.button(h, "辺", make_icon("#0ca678", "辺"), checkable=True, big=False)
        self.sel_edge_btn.setToolTip("シェル（サーフェス）の辺を選びます（支持・力を辺に与えるとき）")
        self.sel_edge_btn.clicked.connect(lambda: self.set_select_mode("edge"))
        self.sel_face_btn.setChecked(True)
        self.sel_face_btn.clicked.connect(lambda: self.set_select_mode("face"))
        self.sel_body_btn.clicked.connect(lambda: self.set_select_mode("body"))
        Ribbon.button(h, "全選択", make_icon("#888888", "全"), self.select_body, big=False)
        Ribbon.button(h, "選択解除", make_icon("#888888", "×"),
                      lambda: (self.set_selection([]), self.set_body_selection([])), big=False)
        self.face_list_btn = Ribbon.button(h, "面の一覧", make_icon("#888888", "≡"), self.face_list_menu, big=False)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        h.addWidget(sep)
        for text, fn in [("等角", lambda: self.plotter.view_isometric()),
                         ("全体", lambda: self.plotter.reset_camera())]:
            Ribbon.button(h, text, make_icon("#5d6d7e", text[0]), fn, big=False)
        hint = QLabel("　クリック: 選択　Ctrl(⌘)+クリック: 追加　ドラッグ: 回転　"
                      "Shift+ドラッグ: 移動　ホイール: 拡大縮小")
        hint.setStyleSheet("color:#666")
        h.addWidget(hint)
        h.addStretch()
        return bar

    # ---------------------------------------------------------------- Femtet 風 属性入力バー
    def _attr_bar(self):
        """Femtet の［モデル］タブと同じ属性入力欄。

        ボディ属性/材料定数: ボディを選択 → ボディ属性名・材料名を入力 → ［設定］
        境界条件          : 面を選択 → 境界条件名を入力 → ［設定］（新しい名前なら編集画面が開く）
        """
        bar = QFrame()
        bar.setStyleSheet("QFrame{background:#f4f6f9;border-top:1px solid #d5dbe3;border-bottom:1px solid #d5dbe3}"
                          "QLabel{border:none;background:transparent}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(4)
        self.attr_mode = QComboBox()
        self.attr_mode.addItems(["ボディ属性/材料定数", "境界条件"])
        self.attr_mode.currentIndexChanged.connect(self._attr_mode_changed)
        h.addWidget(self.attr_mode)
        # ボディ属性・材料
        self.body_attr_w = QWidget()
        hb = QHBoxLayout(self.body_attr_w)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(QLabel("ボディ属性名"))
        self.attr_name = QComboBox()
        self.attr_name.setEditable(True)
        self.attr_name.setMinimumWidth(130)
        hb.addWidget(self.attr_name)
        b = QPushButton("編集")
        b.clicked.connect(self.edit_attr)
        hb.addWidget(b)
        hb.addSpacing(10)
        hb.addWidget(QLabel("材料名"))
        self.mat_name = QComboBox()
        self.mat_name.setEditable(True)
        self.mat_name.setMinimumWidth(170)
        hb.addWidget(self.mat_name)
        b = QPushButton("編集")
        b.clicked.connect(self.edit_material)
        hb.addWidget(b)
        h.addWidget(self.body_attr_w)
        # 境界条件
        self.bc_attr_w = QWidget()
        hc = QHBoxLayout(self.bc_attr_w)
        hc.setContentsMargins(0, 0, 0, 0)
        hc.addWidget(QLabel("境界条件名"))
        self.bc_name = QComboBox()
        self.bc_name.setEditable(True)
        self.bc_name.setMinimumWidth(170)
        hc.addWidget(self.bc_name)
        b = QPushButton("編集")
        b.clicked.connect(self.edit_bc_by_name)
        hc.addWidget(b)
        h.addWidget(self.bc_attr_w)
        self.bc_attr_w.hide()
        self.attr_set_btn = QPushButton("設定")
        self.attr_set_btn.setStyleSheet("font-weight:bold;padding:2px 14px")
        self.attr_set_btn.clicked.connect(self.apply_attr)
        h.addWidget(self.attr_set_btn)
        self.attr_hint = QLabel("")
        self.attr_hint.setStyleSheet("color:#555")
        h.addWidget(self.attr_hint)
        h.addStretch()
        self._fill_attr_combos()
        return bar

    def set_attr_mode(self, idx):
        self.attr_mode.setCurrentIndex(idx)
        self._attr_mode_changed(idx)

    def _attr_mode_changed(self, idx):
        self.body_attr_w.setVisible(idx == 0)
        self.bc_attr_w.setVisible(idx == 1)
        self.set_select_mode("body" if idx == 0 else "face")

    def _surface_mesh(self):
        """シェルまたは 2D のメッシュ（辺を選べる）"""
        return bool(getattr(self.mesh, "is_shell", False) or getattr(self.mesh, "is_2d", False))

    def set_select_mode(self, mode):
        self.sel_body_btn.setChecked(mode == "body")
        self.sel_face_btn.setChecked(mode == "face")
        self.sel_edge_btn.setChecked(mode == "edge")
        if mode == "edge" and not self._surface_mesh():
            self.log("辺の選択はシェル（サーフェス）・2D のモデルで使えます")
        if mode == "body":
            self.selected_faces = []
            self.attr_hint.setText("ボディをクリックして選び、名前を入力して［設定］")
        else:
            self.selected_bodies = []
            self.attr_hint.setText("面をクリックして選び、名前を入力して［設定］")
        if getattr(self, "attr_mode", None) is not None:
            want = 0 if mode == "body" else 1
            if self.attr_mode.currentIndex() != want:
                self.attr_mode.blockSignals(True)
                self.attr_mode.setCurrentIndex(want)
                self.attr_mode.blockSignals(False)
                self.body_attr_w.setVisible(want == 0)
                self.bc_attr_w.setVisible(want == 1)
        self._update_sel_status()
        self.refresh_view()

    def _fill_attr_combos(self):
        p = self.project
        for combo, items in [(self.attr_name, list(p.body_attrs.keys())),
                             (self.mat_name, list(p.materials.keys()) + [n for n in LIBRARY if n not in p.materials]),
                             (self.bc_name, [b.name for b in p.bcs if b.name and b.kind != "gravity"])]:
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setEditText(cur)
            combo.blockSignals(False)

    def _sync_attr_from_selection(self):
        """選択したボディの現在の名前を表示（Femtet と同じく、複数で違う場合は空欄）"""
        if not self.selected_bodies:
            return
        attrs = {self.project.assignment(b).get("attr", "") for b in self.selected_bodies}
        mats = {self.project.assignment(b).get("mat", "") for b in self.selected_bodies}
        self.attr_name.setEditText(attrs.pop() if len(attrs) == 1 else "")
        self.mat_name.setEditText(mats.pop() if len(mats) == 1 else "")

    def _ensure_material(self, name) -> bool:
        if name in self.project.materials:
            return True
        if name in LIBRARY:
            self.project.materials[name] = Material(**LIBRARY[name].to_dict())
            self.log(f"材料DBから「{name}」を読み込みました")
            return True
        dlg = MaterialDialog(name, None, self)
        if not dlg.exec():
            return False
        try:
            m = dlg.material()
        except ValueError as e:
            QMessageBox.warning(self, "材料定数", str(e))
            return False
        m.name = m.name or name
        self.project.materials[m.name] = m
        if m.name != name:
            self.mat_name.setEditText(m.name)
        return True

    def apply_attr(self):
        if self.attr_mode.currentIndex() == 0:
            if self.mesh is None or not self.selected_bodies:
                QMessageBox.information(self, "ボディを選択", "グラフィックスでボディをクリックして選んでから［設定］を押してください。")
                return
            attr = self.attr_name.currentText().strip()
            mat = self.mat_name.currentText().strip()
            if not attr and not mat:
                QMessageBox.information(self, "名前を入力", "ボディ属性名と材料名を入力してください。")
                return
            if mat and not self._ensure_material(mat):
                return
            mat = self.mat_name.currentText().strip()
            self.project.assign(self.selected_bodies, attr=attr or None, mat=mat or None)
            self.log(f"ボディ {', '.join(f'B{b}' for b in self.selected_bodies)} に "
                     f"ボディ属性「{attr or '（変更なし）'}」/ 材料「{mat or '（変更なし）'}」を設定しました")
        else:
            if self.mesh is None or not self.selected_faces:
                QMessageBox.information(self, "面を選択", "グラフィックスで面をクリックして選んでから［設定］を押してください。")
                return
            name = self.bc_name.currentText().strip() or f"境界条件{len(self.project.bcs) + 1}"
            idx = next((i for i, b in enumerate(self.project.bcs) if b.name == name), None)
            if idx is not None:
                bc = self.project.bcs[idx]
                bc.faces = sorted(set(bc.faces) | set(self.selected_faces))
                self.log(f"境界条件「{name}」に面を追加しました（計 {len(bc.faces)} 面）")
            else:
                dlg = BCDialog(name, "displacement", None, self, n_faces=len(self.selected_faces))
                if not dlg.exec():
                    return
                name, kind, vals, exprs = dlg.result()
                self.project.bcs.append(BC(kind, list(self.selected_faces), vals, name, exprs=exprs))
                self.log(f"境界条件「{name}」を作成しました")
            self.bc_name.setEditText(name)
        self._fill_attr_combos()
        self.invalidate()
        self.build_details()
        self.refresh_view()

    def edit_attr(self):
        name = self.attr_name.currentText().strip() or f"Body_{len(self.project.body_attrs) + 1}"
        props = self.project.body_attrs.get(name, {"mesh_size": 0.0})
        dlg = BodyAttrDialog(name, props, self)
        if not dlg.exec():
            return
        new, props2 = dlg.result()
        new = new or name
        if new != name and name in self.project.body_attrs:
            del self.project.body_attrs[name]
            for d in self.project.body_assign.values():
                if d.get("attr") == name:
                    d["attr"] = new
        changed = self.project.body_attrs.get(new, {}).get("mesh_size") != props2["mesh_size"]
        self.project.body_attrs[new] = props2
        self.attr_name.setEditText(new)
        self._fill_attr_combos()
        self.invalidate(mesh=changed)
        self.refresh_view()

    def edit_material(self, name=None):
        name = name or self.mat_name.currentText().strip() or f"材料{len(self.project.materials) + 1}"
        mat = self.project.materials.get(name) or (LIBRARY.get(name) and Material(**LIBRARY[name].to_dict()))
        dlg = MaterialDialog(name, mat, self)
        if not dlg.exec():
            return
        try:
            m = dlg.material()
        except ValueError as e:
            QMessageBox.warning(self, "材料定数", str(e))
            return False
        m.name = m.name or name
        if m.name != name and name in self.project.materials:
            del self.project.materials[name]
            for d in self.project.body_assign.values():
                if d.get("mat") == name:
                    d["mat"] = m.name
        self.project.materials[m.name] = m
        self.mat_name.setEditText(m.name)
        self._fill_attr_combos()
        self.invalidate()
        if self.current[0] in ("mat", "mats"):
            self.select_node(("mat", m.name))

    def edit_bc_by_name(self):
        name = self.bc_name.currentText().strip()
        idx = next((i for i, b in enumerate(self.project.bcs) if b.name == name), None)
        if idx is None:
            QMessageBox.information(self, "境界条件", "その名前の境界条件はまだありません。\n"
                                    "面を選んで名前を入力し［設定］を押すと作成されます。")
            return
        bc = self.project.bcs[idx]
        dlg = BCDialog(bc.name, bc.kind, bc.values, self, n_faces=len(bc.faces), exprs=bc.exprs)
        if dlg.exec():
            bc.name, bc.kind, bc.values, bc.exprs = dlg.result()
            self.bc_name.setEditText(bc.name)
            self._fill_attr_combos()
            self.invalidate()
            self.refresh_view()

    # ======================================================================= 変数・パラメトリック解析
    def _variable_usage(self) -> dict:
        """{変数名: [使っている場所…]}"""
        from ..params import names_in, split_points
        used = {}

        def note(expr, where):
            for nm in names_in(expr):
                used.setdefault(nm, []).append(where)
        for i, f in enumerate(self.project.geometry.get("features", [])):
            for k, ex in (f.get("expr") or {}).items():
                note(ex if k != "points" else " + ".join(c for r in split_points(ex) for c in r),
                     f"{FEATURES.get(f['type'], (f['type'],))[0]}{i + 1} の {k}")
        for b in self.project.bcs:
            for c, ex in b.exprs.items():
                note(ex, f"境界条件「{b.name or BC_JP.get(b.kind, b.kind)}」")
        if self.project.mesh_size_expr:
            note(self.project.mesh_size_expr, "メッシュサイズ")
        for p in self.project.params:
            note(p.get("expr", ""), f"変数 {p['name']} の式")
        return used

    def edit_variables(self):
        from .parametric_dialogs import VariablesDialog
        dlg = VariablesDialog(self.project.params, self, used=self._variable_usage())
        if not dlg.exec():
            return
        old = [dict(p) for p in self.project.params]
        self.project.params = dlg.params()
        if not self._apply_variables(old):
            return
        self.log("変数を更新しました: " + ", ".join(f"{k} = {v:g}" for k, v in self.project.env().items()))

    def _apply_variables(self, old_params=None) -> bool:
        """変数の変更を寸法・荷重に反映して形状を作り直す（エラーなら元の変数に戻す）"""
        before = json.dumps(self.project.geometry, sort_keys=True), self.project.mesh_size
        try:
            self.project.apply_params()
        except ParamError as e:
            QMessageBox.warning(self, "変数", f"{e}\n\n変数の変更を取り消します。")
            if old_params is not None:
                self.project.params = old_params
                self.project.apply_params()
            return False
        if (json.dumps(self.project.geometry, sort_keys=True), self.project.mesh_size) != before:
            self.remesh()
        else:
            self.invalidate()
            self.rebuild_tree()
        return True

    def run_parametric(self):
        from .parametric_dialogs import ParametricDialog
        if not self.project.params:
            r = QMessageBox.question(self, "パラメトリック解析",
                                     "変数がありません。［変数］で変数を作り、寸法や荷重の欄で使ってから実行してください。\n"
                                     "いま変数を作りますか？")
            if r == QMessageBox.Yes:
                self.edit_variables()
            return
        try:
            env = self.project.apply_params()
        except ParamError as e:
            QMessageBox.warning(self, "パラメトリック解析", str(e))
            return
        if self.mesh is None or self.mesh_dirty:
            if not self.remesh():
                return
        if not self.project.bcs and self.project.analysis_type == "static":
            QMessageBox.information(self, "パラメトリック解析", "先に支持・荷重を設定してください。")
            return
        dlg = ParametricDialog(self.project.params, env, self.project.parametric.get("settings"), self,
                               transient=self.project.analysis_type == "transient")
        if not dlg.exec():
            return
        st = dlg.settings()
        cases, outputs = dlg.cases(), dlg.outputs()
        self.project.parametric = {"settings": st}
        from ..parametric import run_parametric
        prog = QProgressDialog("パラメトリック解析を実行しています…", "中断", 0, len(cases), self)
        prog.setWindowTitle("パラメトリック解析")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        self._busy(True)

        def progress(i, n):
            prog.setValue(i)
            prog.setLabelText(f"ケース {min(i + 1, n)} / {n} を解析しています…")
            QApplication.processEvents()

        def log(msg):
            self.log(str(msg))
            QApplication.processEvents()
        t0 = datetime.datetime.now()
        try:
            rows = run_parametric(self.project, cases, outputs, log=log, progress=progress,
                                  cancel=prog.wasCanceled, base_mesh=self.mesh)
        finally:
            prog.close()
            self._busy(False)
        n_ok = sum(r["状態"] == "完了" for r in rows)
        self.project.parametric["results"] = rows
        self.log(f"パラメトリック解析: {n_ok}/{len(rows)} ケース完了（{(datetime.datetime.now() - t0).total_seconds():.0f} 秒）")
        self.rebuild_tree()
        self.show_parametric_results()

    def show_parametric_results(self):
        from .parametric_dialogs import ParametricResultDialog
        rows = self.project.parametric.get("results")
        if not rows:
            QMessageBox.information(self, "パラメトリック解析", "まだ結果がありません。［パラメトリック解析］で実行してください。")
            return
        swept = list(self.project.parametric.get("settings", {}).get("sweeps", {}))
        ParametricResultDialog(rows, swept, self, apply_case=self._apply_case).exec()

    def _apply_case(self, values: dict):
        """パラメトリック解析のケースの値を変数に入れる（その変数の式は値で置き換わる）"""
        old = [dict(p) for p in self.project.params]
        for p in self.project.params:
            if p["name"] in values:
                p["expr"] = f"{values[p['name']]:.10g}"
        if self._apply_variables(old):
            self.log("変数に反映: " + ", ".join(f"{k} = {v:g}" for k, v in values.items()))

    def _details_variables(self):
        s = self._section("変数")
        try:
            env = self.project.env()
        except ParamError as e:
            env = {}
            self._row(s, "エラー", str(e))
        for p in self.project.params:
            v = env.get(p["name"])
            txt = (f"{v:.8g}" if v is not None else "—") + ("" if is_number(p.get("expr", 0)) else f"　（= {p['expr']}）")
            self._row(s, p["name"], txt + (f"　{p['comment']}" if p.get("comment") else ""))
        b = QPushButton("変数を編集…")
        b.clicked.connect(lambda: self._later(self.edit_variables))
        self._row(s, "", b)
        used = self._variable_usage()
        if used:
            s2 = self._section("使用箇所")
            for nm, wh in used.items():
                self._row(s2, nm, "、".join(wh[:6]) + (" …" if len(wh) > 6 else ""))

    def _details_parametric(self):
        st = self.project.parametric.get("settings", {})
        rows = self.project.parametric.get("results", [])
        s = self._section("設定")
        for nm, sw in st.get("sweeps", {}).items():
            self._row(s, nm, sw.get("values") or f"{sw.get('start')} → {sw.get('stop')}（{sw.get('n')} 分割）")
        self._row(s, "ケース数", f"{len(rows)}（完了 {sum(r['状態'] == '完了' for r in rows)}）")
        b = QPushButton("結果の表・グラフ…")
        b.clicked.connect(lambda: self._later(self.show_parametric_results))
        self._row(s, "", b)
        b = QPushButton("もう一度実行…")
        b.clicked.connect(lambda: self._later(self.run_parametric))
        self._row(s, "", b)

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("ファイル")
        for text, sc, fn in [("新規", QKeySequence.New, self.new_project),
                             ("開く…", QKeySequence.Open, self.open_project),
                             ("保存", QKeySequence.Save, self.save_project),
                             ("名前を付けて保存…", QKeySequence.SaveAs, self.save_project_as)]:
            a = QAction(text, self)
            a.setShortcut(sc)
            a.triggered.connect(fn)
            fm.addAction(a)
        fm.addSeparator()
        for text, fn in [("ジオメトリをインポート…", self.import_file),
                         ("レポートを作成（HTML）…", self.export_report),
                         ("結果を VTU で書き出す（ParaView）…", self.export_vtu),
                         ("結果を CSV で書き出す…", self.export_csv),
                         ("サブモデル用に結果を保存（粗いモデル）…", self.save_coarse_result),
                         ("画像を保存…", self.save_screenshot)]:
            a = QAction(text, self)
            a.triggered.connect(fn)
            fm.addAction(a)
        em = mb.addMenu("編集")
        self.undo_action = QAction("元に戻す", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo)
        em.addAction(self.undo_action)
        self.redo_action = QAction("やり直す", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self.redo)
        em.addAction(self.redo_action)
        a = QAction("操作履歴…", self)
        a.setShortcut(QKeySequence("Ctrl+H"))
        a.triggered.connect(self.show_history)
        em.addAction(a)
        em.addSeparator()
        a = QAction("削除", self)
        a.setShortcut(QKeySequence.Delete)
        a.triggered.connect(self.delete_current)
        em.addAction(a)
        a = QAction("すべての面を選択", self)
        a.setShortcut(QKeySequence.SelectAll)
        a.triggered.connect(self.select_body)
        em.addAction(a)
        vm = mb.addMenu("変数")
        for text, fn in [("変数…", self.edit_variables), ("パラメトリック解析…", self.run_parametric),
                         ("パラメトリック解析の結果…", self.show_parametric_results)]:
            a = QAction(text, self)
            a.triggered.connect(fn)
            vm.addAction(a)
        hm = mb.addMenu("ヘルプ")
        a = QAction("使い方", self)
        a.triggered.connect(self.show_help)
        hm.addAction(a)

    # ======================================================================= 状態
    def log(self, msg: str):
        self.logbox.appendPlainText(msg)
        self.statusBar().showMessage(msg.splitlines()[0] if msg else "", 6000)

    def _update_title(self):
        name = os.path.basename(self.project_path) if self.project_path else "無題"
        self.setWindowTitle(f"{name} — {self._analysis_label()} — open-mechanical {__version__}")

    def invalidate(self, mesh=False):
        """設定が変わったら結果を無効にする（Ansys の黄色い ⚡ 状態）"""
        if mesh:
            self.mesh_dirty = True
        self.result = None
        self.results = []
        self.solve_error = ""
        self.rebuild_tree(keep=True, details=False)

    def _later(self, fn):
        """詳細パネルのウィジェット自身のシグナル内で、そのウィジェットを作り直さないよう後回しにする"""
        QTimer.singleShot(0, fn)

    def status_of(self, node) -> str:
        k = node[0]
        if k == "geometry":
            return STATUS_OK if self.mesh is not None else STATUS_ERROR
        if k == "material":
            return STATUS_OK if self.project.material.E > 0 else STATUS_ERROR
        if k == "mats":
            if self.mesh is None:
                return STATUS_NONE
            try:
                self.project.elem_materials(self.mesh)
                return STATUS_OK
            except ValueError:
                return STATUS_ERROR
        if k == "contact":
            c = self.project.contacts[node[1]]
            if c.suppressed:
                return STATUS_NONE
            return STATUS_OK if (c.contact_faces and c.target_faces) else STATUS_ERROR
        if k in ("connections", "contacts"):
            bad = any(self.status_of(("contact", i)) == STATUS_ERROR for i in range(len(self.project.contacts)))
            return STATUS_ERROR if bad else STATUS_OK
        if k == "body":
            return STATUS_OK if self.project.assignment(node[1]).get("mat") in self.project.materials else STATUS_ERROR
        if k in ("mesh", "sizing"):
            ls_ = self.project.local_sizes[node[1]] if k == "sizing" else {}
            if k == "sizing" and ls_.get("type", "sizing") in ("sizing", "refinement") and \
                    not (ls_.get("faces") if ls_.get("scope", "faces") == "faces" else ls_.get("bodies")):
                return STATUS_ERROR
            return STATUS_PENDING if self.mesh_dirty else STATUS_OK
        if k == "bc":
            bc = self.project.bcs[node[1]]
            return STATUS_ERROR if (bc.kind not in NO_FACE_KINDS and not bc.faces) else STATUS_OK
        if k == "analysis":
            bad = any(self.status_of(("bc", i)) == STATUS_ERROR for i in range(len(self.project.bcs)))
            return STATUS_ERROR if bad or not self._has_support() else STATUS_OK
        if k in ("solution", "result", "info"):
            if self.solve_error:
                return STATUS_ERROR
            return STATUS_OK if self.result is not None else STATUS_PENDING
        return STATUS_NONE

    def _has_support(self):
        return any(b.kind in SUPPORT_LIKE for b in self.project.bcs)

    # ======================================================================= ツリー
    # ======================================================================= 操作履歴
    def _hist_touch(self):
        if not self._restoring:
            self._hist_timer.start(400)

    def _hist_commit(self):
        if self._restoring:
            return
        try:
            state = self.project.to_dict()
        except Exception:
            return
        h = self.history
        if h.pos > 0 and self._hist_label is None:
            prev = h.state(h.pos)
            diff = [k for k in set(prev) | set(state) if prev.get(k) != state.get(k)]
            if diff == ["contacts"] and all(c.get("auto") for c in state.get("contacts", [])):
                h.replace_current(state)        # 形状変更に伴う接触の自動作成は、直前の操作に含める
                return
        if h.record(state, self._hist_label):
            self._hist_refresh()
        self._hist_label = None

    def _hist_reset(self):
        self._hist_timer.stop()
        self.history.reset(self.project.to_dict())
        self._hist_label = None
        self._hist_refresh()

    def undo(self):
        self._hist_commit()
        if self.history.can_undo():
            self.restore_history(self.history.pos - 1)

    def redo(self):
        self._hist_commit()
        if self.history.can_redo():
            self.restore_history(self.history.pos + 1)

    def restore_history(self, i):
        """履歴の i 番目の状態に戻す（その後に操作すると、先の履歴は消えます）"""
        self._hist_commit()
        h = self.history
        if not (0 <= i < len(h.items)) or i == h.pos:
            return
        old = self.project.to_dict()
        new = h.state(i)
        self._restoring = True
        self._hist_timer.stop()
        try:
            self.project = Project.from_dict(new)
            h.pos = i
            self.selected_faces, self.selected_bodies = [], []
            self.result, self.results, self.solve_error = None, [], ""
            if not self._node_exists(self.current):
                self.current = ("geometry",)
            same_mesh = (old.get("geometry") == new.get("geometry") and old.get("mesh") == new.get("mesh")
                         and old.get("share_topology") == new.get("share_topology") and old.get("parts") == new.get("parts")
                         and old.get("body_attrs") == new.get("body_attrs") and old.get("body_assign") == new.get("body_assign"))
            if same_mesh and self.mesh is not None:
                self._fill_attr_combos()
                self.rebuild_tree()
                self.refresh_view()
            else:
                self.mesh_dirty = True
                self.current = ("geometry",)
                self.remesh()
            h.replace_current(self.project.to_dict())   # メッシュ作成時の自動調整はこの時点の状態とする
            self.log(f"操作履歴: 「{h.items[i]['label']}」（{h.items[i]['time']}）の時点に戻しました")
        finally:
            self._restoring = False
        self._hist_refresh()
        self.build_details()

    def _node_exists(self, node):
        k = node[0]
        if k not in ("bc", "feature", "contact", "sizing", "part", "attr", "mat"):
            return True
        try:
            return {"bc": lambda: node[1] < len(self.project.bcs),
                    "feature": lambda: node[1] < len(self._features()),
                    "contact": lambda: node[1] < len(self.project.contacts),
                    "sizing": lambda: node[1] < len(self.project.local_sizes),
                    "part": lambda: node[1] < len(self.project.parts),
                    "attr": lambda: node[1] in self.project.body_attrs,
                    "mat": lambda: node[1] in self.project.materials}[k]()
        except Exception:
            return False

    def show_history(self):
        """操作履歴ウィンドウ（ダブルクリックでその時点に戻る）"""
        if self._hist_dialog is None:
            d = QDialog(self)
            d.setWindowTitle("操作履歴")
            d.resize(420, 520)
            lay = QVBoxLayout(d)
            info = QLabel("操作ごとに自動で記録しています。行をダブルクリックすると、その時点の状態に戻ります。\n"
                          "戻った後に別の操作をすると、それより先の履歴は消えます。")
            info.setWordWrap(True)
            info.setStyleSheet("color:#555")
            lay.addWidget(info)
            lst = QListWidget()
            lst.itemDoubleClicked.connect(lambda it: self.restore_history(lst.row(it)))
            lay.addWidget(lst)
            row = QHBoxLayout()
            for text, fn in [("◀ 元に戻す", self.undo), ("やり直す ▶", self.redo),
                             ("選んだ時点に戻す", lambda: self.restore_history(lst.currentRow()))]:
                b = QPushButton(text)
                b.clicked.connect(fn)
                row.addWidget(b)
            lay.addLayout(row)
            d.list = lst
            self._hist_dialog = d
        self._hist_refresh()
        self._hist_dialog.show()
        self._hist_dialog.raise_()

    def _hist_refresh(self):
        d = self._hist_dialog
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(self.history.can_undo())
            self.redo_action.setEnabled(self.history.can_redo())
        if d is None:
            return
        lst = d.list
        lst.clear()
        for i, it in enumerate(self.history.items):
            mark = "▶ " if i == self.history.pos else "　 "
            w = QListWidgetItem(f"{mark}{i + 1:>3}  {it['time']}  {it['label']}")
            if i > self.history.pos:
                w.setForeground(QBrush(QColor("#999999")))
            if i == self.history.pos:
                f = w.font()
                f.setBold(True)
                w.setFont(f)
            lst.addItem(w)
        lst.setCurrentRow(self.history.pos)
        lst.scrollToItem(lst.currentItem())

    def rebuild_tree(self, keep=True, details=True):
        self._hist_touch()
        cur = self.current if keep else ("geometry",)
        self.tree.blockSignals(True)
        self.tree.clear()

        def add(parent, text, node, icon_color, icon_text, bold=False):
            it = QTreeWidgetItem([text])
            it.setData(0, Qt.UserRole, node)
            st = self.status_of(node) if node else ""
            it.setIcon(0, make_icon(icon_color, icon_text, st))
            if bold:
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
            (parent.addChild(it) if isinstance(parent, QTreeWidgetItem) else self.tree.addTopLevelItem(it))
            return it

        proj = add(None, "プロジェクト", ("project",), "#6c7a89", "P", bold=True)
        model = add(proj, "モデル (A4)", ("model",), "#6c7a89", "M")
        if self.project.params:
            add(model, f"変数（{len(self.project.params)}）", ("variables",), "#1971c2", "変")
        if self.project.parametric.get("results"):
            add(model, f"パラメトリック解析（{len(self.project.parametric['results'])} ケース）", ("parametric",),
                "#1971c2", "パ")
        geo = add(model, "ジオメトリ", ("geometry",), "#4a78c2", "G")
        gk = self.project.geometry.get("kind")
        live = set(self.mesh.body_ids().tolist()) if self.mesh is not None else set()
        if gk == "model":
            feats = self.project.geometry.get("features", [])
            keys = feature_body_keys(feats)
            for i, f in enumerate(feats):
                op = f.get("op", "new")
                if f["type"] == "bool":
                    tools = ", ".join(f"B{k}" for k in f.get("tools", []))
                    label = f"{'結合' if op == 'union' else '除去'}: B{f.get('target')} ← {tools}"
                    add(geo, label, ("feature", i), OP_COLORS.get(op, "#888"), "∪" if op == "union" else "−")
                    continue
                name, ch, _ = FEATURES[f["type"]]
                if f["type"] in EDIT_FEATURES:
                    it = add(geo, self._feature_label(f), ("feature", i), OP_COLORS["edit"], ch)
                    continue
                if i in keys:
                    k = keys[i]
                    a = self.project.assignment(k)
                    label = (f"B{k}: {name}" + ("" if k in live or self.mesh is None else "（結合済み）")
                             + f"　[{a.get('attr') or '未定義'} / {a.get('mat') or '未定義'}]")
                    it = add(geo, label, ("feature", i), OP_COLORS["new"], ch)
                    if k in live:
                        it.setIcon(0, make_icon(OP_COLORS["new"], ch, self.status_of(("body", k))))
                else:
                    add(geo, f"{OPS.get(op, op)}: {name}{i + 1}", ("feature", i), OP_COLORS.get(op, "#888"), ch)
        else:
            add(geo, self.project.geometry_desc() if gk != "file"
                else os.path.basename(self.project.geometry.get("path", "")), ("geometry",), "#7f9cc9", "体")
        if self.project.share_topology == "all":
            add(geo, "節点共有: 接しているボディすべて", ("share",), "#2f9e44", "共")
        elif self.project.share_topology == "parts":
            for i, pt in enumerate(self.project.parts):
                add(geo, f"{pt.get('name', f'パート{i + 1}')}: " + ", ".join(f"B{b}" for b in pt.get("bodies", []))
                    + "（節点共有）", ("part", i), "#2f9e44", "共")
        at = add(model, "ボディ属性", ("attrs",), "#7a8a99", "属")
        for name in self.project.body_attrs:
            n = sum(1 for b in live if self.project.assignment(b).get("attr") == name)
            add(at, f"{name}（{n} ボディ）", ("attr", name), "#7a8a99", "属")
        mt = add(model, "材料定数", ("mats",), "#b07b2c", "材")
        for name in self.project.materials:
            n = sum(1 for b in live if self.project.assignment(b).get("mat") == name)
            add(mt, f"{name}（{n} ボディ）", ("mat", name), "#b07b2c", "材")
        if len(live) > 1 or self.project.contacts:
            cn = add(model, "接続", ("connections",), "#e8590c", "接")
            ct = add(cn, "接触", ("contacts",), "#e8590c", "接")
            for i, c in enumerate(self.project.contacts):
                label = self._contact_label(c) + ("（抑制）" if c.suppressed else "")
                add(ct, label, ("contact", i), "#e8590c" if not c.suppressed else "#aaaaaa",
                    CONTACT_TYPES[c.type][0])
        add(model, "座標系", ("coords",), "#6c7a89", "XY")
        mesh = add(model, "メッシュ", ("mesh",), "#3a8fa8", "網")
        for i, ls in enumerate(self.project.local_sizes):
            add(mesh, self._control_label(i, ls), ("sizing", i), "#3a8fa8" if not ls.get("suppressed") else "#aaaaaa",
                {"sweep": "掃", "refinement": "再", "sphere": "球"}.get(ls.get("type", "sizing"), "細"))
        an = add(model, self._analysis_label() + " (A5)", ("analysis",), "#2e6db4",
                 "T" if self.project.analysis_type == "transient" else "S")
        add(an, "解析設定", ("settings",), "#6c7a89", "設")
        for i, bc in enumerate(self.project.bcs):
            add(an, bc.name or BC_JP[bc.kind], ("bc", i), BC_COLORS[bc.kind], BC_JP[bc.kind][0])
        sol = add(an, "解析結果 (A6)", ("solution",), "#2e9d3a", "解")
        add(sol, "解析情報", ("info",), "#6c7a89", "i")
        for i, key in enumerate(self.project.results):
            add(sol, RESULT_NAME.get(key, key), ("result", i), "#c0392b" if "応力" in key else "#2c7fb8"
                if "変" in key else "#8e6c2e", RESULT_NAME.get(key, key)[0])
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self.select_node(cur, refresh=False, details=details)

    def _find_item(self, node):
        it = self.tree.invisibleRootItem()
        stack = [it]
        while stack:
            x = stack.pop()
            for i in range(x.childCount()):
                c = x.child(i)
                par = c.parent()
                if c.data(0, Qt.UserRole) == node and (par is None or par.data(0, Qt.UserRole) != node):
                    return c
                stack.append(c)
        return None

    def select_node(self, node, refresh=True, details=True):
        it = self._find_item(tuple(node))
        if it is None:
            it = self._find_item(("geometry",))
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(it)
        self.tree.blockSignals(False)
        self.current = tuple(it.data(0, Qt.UserRole))
        if details:
            self.build_details()
        if refresh:
            self.refresh_view()

    def on_tree_select(self):
        it = self.tree.currentItem()
        if it is None:
            return
        self.current = tuple(it.data(0, Qt.UserRole))
        self.probe_text = None
        self.build_details()
        k = self.current[0]
        if k in ("feature", "attrs", "attr", "mats", "mat") or (k == "geometry" and self.project.geometry.get("kind") == "model"):
            self.ribbon.setCurrentIndex(self.tab_model)
            if k in ("attr", "mat", "attrs", "mats"):
                self.set_attr_mode(0)
                if k == "attr":
                    self.attr_name.setEditText(self.current[1])
                if k == "mat":
                    self.mat_name.setEditText(self.current[1])
        elif k in ("connections", "contacts", "contact"):
            self.ribbon.setCurrentIndex(self.tab_conn)
            if k == "contact":
                self.set_select_mode("face")
        elif k in ("analysis", "bc", "settings"):
            self.ribbon.setCurrentIndex(self.tab_env)
        elif k in ("solution", "result", "info"):
            self.ribbon.setCurrentIndex(self.tab_res)
        self.refresh_view()

    def on_tree_double(self, item, _col=0):
        node = tuple(item.data(0, Qt.UserRole))
        if node[0] == "variables":
            self._later(self.edit_variables)
        elif node[0] == "parametric":
            self._later(self.show_parametric_results)
        elif node[0] == "feature":
            self._later(lambda: self.edit_feature(node[1]))
        elif node[0] == "mat":
            self._later(lambda: self.edit_material(node[1]))
        elif node[0] == "attr":
            self._later(lambda: (self.attr_name.setEditText(node[1]), self.edit_attr()))
        elif node[0] == "bc":
            bc = self.project.bcs[node[1]]
            if bc.name and bc.kind != "gravity":
                self.bc_name.setEditText(bc.name)
                self._later(self.edit_bc_by_name)

    def on_tree_menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        node = tuple(it.data(0, Qt.UserRole))
        self.tree.setCurrentItem(it)
        m = QMenu(self)
        k = node[0]
        if k in ("analysis", "bc", "settings"):
            ins = m.addMenu("挿入")
            for kind in ["force", "pressure", "moment", "gravity", "fixed", "displacement"]:
                menu_action(ins, BC_JP[kind], lambda kind=kind: self.add_bc(kind),
                            make_icon(BC_COLORS[kind], BC_JP[kind][0]))
            menu_action(m, "解析", self.run_solve, make_icon("#2e9d3a", "⚡"))
        if k in ("solution", "result", "info"):
            ins = m.addMenu("挿入")
            for grp, items in RESULT_GROUPS.items():
                sm = ins.addMenu(grp)
                for key, name in items:
                    menu_action(sm, name, lambda key=key: self.add_result(key))
            menu_action(m, "解析", self.run_solve, make_icon("#2e9d3a", "⚡"))
        if k in ("mesh", "sizing"):
            menu_action(m, "メッシュを生成", lambda: self.remesh())
            ins = m.addMenu("挿入")
            for key, lab in self._control_menu_items():
                menu_action(ins, lab, lambda key=key: self.add_control(key))
        if k in ("geometry", "feature"):
            ins = m.addMenu("作成")
            for ft in ["box", "cylinder", "pipe", "cone", "sphere", "extrude"]:
                menu_action(ins, FEATURES[ft][0], lambda ft=ft: self.create_body(ft))
            menu_action(m, "穴をあける…", self.create_hole)
            menu_action(m, "結合（選択ボディ）", lambda: self.boolean("union"))
            menu_action(m, "除去（選択ボディ）", lambda: self.boolean("cut"))
            if k == "feature":
                menu_action(m, "形状を編集…", lambda: self.edit_feature(node[1]))
            menu_action(m, "STEP を取り込む…", self.add_import_feature)
            menu_action(m, "ジオメトリを置き換え（STEP / STL）…", self.import_file)
        if k in ("mats", "mat"):
            menu_action(m, "新しい材料定数…", lambda: self.edit_material(f"材料{len(self.project.materials) + 1}"))
        if k in ("attrs", "attr"):
            menu_action(m, "新しいボディ属性…", lambda: (self.attr_name.setEditText(""), self.edit_attr()))
        if k in ("connections", "contacts", "contact"):
            menu_action(m, "接続を作成（自動検出）", self.auto_contacts)
            menu_action(m, "接触領域を挿入", self.add_contact)
            tm = m.addMenu("タイプを変更")
            for kk, v in CONTACT_TYPES.items():
                menu_action(tm, v, lambda kk=kk: self.set_contact_type(kk))
            if k == "contact":
                menu_action(m, "接触とターゲットを反転", self.flip_contact)
        if k in ("bc", "result", "sizing", "feature", "attr", "mat", "contact"):
            m.addSeparator()
            menu_action(m, "削除", self.delete_current)
        if not m.isEmpty():
            m.exec(self.tree.viewport().mapToGlobal(pos))

    # ======================================================================= 詳細パネル
    def build_details(self):
        d = self.details
        d.clear()
        k = self.current[0]
        title = {"geometry": "ジオメトリ", "material": "材料", "mesh": "メッシュ", "sizing": "メッシュ制御",
                 "analysis": "静的構造 (A5)", "settings": "解析設定", "solution": "解析結果 (A6)",
                 "info": "解析情報", "project": "プロジェクト", "model": "モデル (A4)", "coords": "座標系",
                 "variables": "変数", "parametric": "パラメトリック解析"}
        if k == "feature":
            f = self.project.geometry["features"][self.current[1]]
            name = (f"{FEATURES[f['type']][0]}{self.current[1] + 1}" if f["type"] in FEATURES
                    else "ブール演算")
        elif k in ("attr", "mat"):
            name = self.current[1]
        elif k in ("attrs", "mats"):
            name = "ボディ属性" if k == "attrs" else "材料定数"
        elif k == "contact":
            name = self.project.contacts[self.current[1]].name
        elif k == "part":
            name = self.project.parts[self.current[1]].get("name", "パート")
        elif k == "share":
            name = "節点共有"
        elif k in ("connections", "contacts"):
            name = "接続" if k == "connections" else "接触"
        elif k == "bc":
            bc = self.project.bcs[self.current[1]]
            name = bc.name or BC_JP[bc.kind]
        elif k == "result":
            name = RESULT_NAME.get(self.project.results[self.current[1]], "")
        else:
            name = title.get(k, "")
        self.details_title.setText(f" 「{name}」の詳細")
        getattr(self, f"_details_{k}", self._details_default)()
        d.expandAll()

    def _section(self, title):
        it = QTreeWidgetItem([title])
        f = it.font(0)
        f.setBold(True)
        it.setFont(0, f)
        it.setFirstColumnSpanned(True)
        for c in range(2):
            it.setBackground(c, QBrush(QColor("#e9edf2")))
        self.details.addTopLevelItem(it)
        it.setFirstColumnSpanned(True)
        return it

    def _row(self, sec, label, value):
        it = QTreeWidgetItem([label, "" if isinstance(value, QWidget) else str(value)])
        sec.addChild(it)
        if isinstance(value, QWidget):
            self.details.setItemWidget(it, 1, value)
        return it

    def _details_default(self):
        s = self._section("情報")
        self._row(s, "単位系", "メートル法 (mm, t, N, s, MPa)")
        if self.mesh is not None:
            self._row(s, "節点数", f"{self.mesh.n_nodes:,}")
            self._row(s, "要素数", f"{self.mesh.n_elems:,}")

    def _details_coords(self):
        s = self._section("全体座標系")
        self._row(s, "タイプ", "直交座標")
        self._row(s, "原点", "(0, 0, 0) mm")

    def _share_section(self):
        s = self._section("節点共有（トポロジ共有）")
        self._row(s, "共有方法", self._combo(
            [("しない（ボディ間は接触でつなぐ）", "none"), ("接しているボディすべて", "all"), ("パートごと", "parts")],
            self.project.share_topology, lambda v: self._later(lambda: self.set_share(v))))
        if self.project.share_topology == "parts":
            self._row(s, "パート数", len(self.project.parts))
            b = QPushButton("選択中のボディでパートを作成")
            b.clicked.connect(lambda: self._later(self.make_part))
            self._row(s, "", b)
        self._row(s, "", "節点共有したボディ同士は、界面の節点を共有して一体で解かれます（接触は作られません）")

    def _details_share(self):
        self._share_section()

    def _details_part(self):
        i = self.current[1]
        pt = self.project.parts[i]
        s = self._section("パート")
        name = QLineEdit(pt.get("name", f"パート{i + 1}"))

        def setname():
            pt["name"] = name.text().strip() or pt.get("name")
            self._later(self.rebuild_tree)
        name.editingFinished.connect(setname)
        self._row(s, "名前", name)
        self._row(s, "ボディ", ", ".join(f"B{b}" for b in pt.get("bodies", [])))
        self._row(s, "節点共有", "はい（Ansys のトポロジ共有と同じ）")
        b = QPushButton("選択中のボディに置き換え")
        b.clicked.connect(lambda: self._later(lambda: self._part_set(i, list(self.selected_bodies))))
        self._row(s, "", b)
        d = QPushButton("パートを解除")
        d.clicked.connect(lambda: self._later(self.delete_current))
        self._row(s, "", d)

    def _part_set(self, i, bodies):
        if len(bodies) < 2:
            QMessageBox.information(self, "パート", "ボディ選択モードで2つ以上のボディを選んでください。")
            return
        self.project.parts[i]["bodies"] = sorted(int(b) for b in bodies)
        self.remesh()

    def set_share(self, mode):
        if self.project.share_topology == mode:
            return
        self.project.share_topology = mode
        self.log({"none": "節点共有をやめ、ボディ同士を接触でつなぎます",
                  "all": "接しているボディすべてで節点を共有します",
                  "parts": "パートごとに節点を共有します"}[mode])
        self.invalidate(mesh=True)
        self.remesh()

    def make_part(self):
        """選択中のボディを1つのパートにまとめて節点共有（Ansys の「新規パートを作成」）"""
        bodies = sorted(set(int(b) for b in self.selected_bodies))
        if len(bodies) < 2:
            self.set_select_mode("body")
            QMessageBox.information(self, "パート作成",
                                    "グラフィックスで2つ以上のボディを Ctrl(⌘)+クリックで選んでから押してください。")
            return
        # 既存パートから外してから追加
        for pt in self.project.parts:
            pt["bodies"] = [b for b in pt.get("bodies", []) if b not in bodies]
        self.project.parts = [pt for pt in self.project.parts if len(pt.get("bodies", [])) > 1]
        self.project.parts.append({"name": f"パート{len(self.project.parts) + 1}", "bodies": bodies})
        self.project.share_topology = "parts"
        self.log(f"パートを作成しました（{', '.join(f'B{b}' for b in bodies)} で節点共有）")
        self.invalidate(mesh=True)
        self.remesh()

    def _set_analysis_2d(self, mode):
        g = self.project.geometry
        if mode in PLANE_MODES:
            g["analysis_2d"] = mode
        else:
            g.pop("analysis_2d", None)
        self.invalidate(mesh=True)
        self.remesh()
        self.build_details()

    def _details_geometry(self):
        g = self.project.geometry
        s = self._section("定義")
        kind = QComboBox()
        kind.addItem("モデリング（フィーチャー）", "model")
        for k, (name, _) in PRIMITIVES.items():
            kind.addItem("簡単形状: " + name, k)
        kind.addItem("CAD ファイル（STEP / STL）", "file")
        kind.setCurrentIndex(max(0, kind.findData(g.get("kind"))))
        kind.currentIndexChanged.connect(lambda _: self._later(lambda d=kind.currentData(): self._geometry_kind_changed(d)))
        self._row(s, "形状", kind)
        if g.get("kind") == "model":
            mode = QComboBox()
            mode.addItem("3D", "3d")
            for key, label in PLANE_MODES.items():
                mode.addItem("2D: " + label, key)
            mode.setCurrentIndex(max(0, mode.findData(g.get("analysis_2d") or "3d")))
            mode.currentIndexChanged.connect(
                lambda _: self._later(lambda d=mode.currentData(): self._set_analysis_2d(d)))
            self._row(s, "解析タイプ（2D の挙動）", mode)
            if g.get("analysis_2d") in PLANE_MODES:
                self._row(s, "", "XY 平面（Z = 0）上に「平板・多角形の面・円板」で断面を作ります。"
                          + ("X = 半径方向、Y = 対称軸です。" if g.get("analysis_2d") == "axisymmetric"
                             else "平面応力では面の厚さが板厚になります。"))
            n = len(g.get("features", []))
            self._row(s, "フィーチャー数", n)
            self._row(s, "", "リボンの［モデル］タブの「作成」で形状を作ります" if n == 0
                      else "ツリーのフィーチャーを選ぶと寸法を編集できます")
        elif g.get("kind") == "file":
            self._row(s, "ファイル", os.path.basename(g.get("path", "")))
            b = QPushButton("別のファイルを読み込む…")
            b.clicked.connect(self.import_file)
            self._row(s, "", b)
        else:
            self._geo_spins = {}
            for key, label, default in PRIMITIVES[g["kind"]][1]:
                sp = dspin(float(g.get(key, default)), 0.0, 1e7, 3, 1.0)
                sp.valueChanged.connect(lambda v, key=key: self._later(lambda: self._geometry_param(key, v)))
                self._row(s, label, sp)
            b = QPushButton("CAD ファイルを読み込む（STEP / STL）…")
            b.clicked.connect(self.import_file)
            self._row(s, "", b)
        if self.mesh is not None and len(self.mesh.body_ids()) > 1 or self.project.share_topology != "none":
            self._share_section()
        if self.mesh is not None:
            s2 = self._section("プロパティ")
            lo, hi = self.mesh.bounds()
            sz = hi - lo
            self._row(s2, "長さ X", f"{sz[0]:.6g} mm")
            self._row(s2, "長さ Y", f"{sz[1]:.6g} mm")
            self._row(s2, "長さ Z", f"{sz[2]:.6g} mm")
            V = self.mesh.volume()
            self._row(s2, "体積", f"{V:.6g} mm³")
            self._row(s2, "質量", f"{V * self.project.material.density_t_mm3 * 1000:.6g} kg")
            self._row(s2, "面の数", self.mesh.n_faces)
            self._row(s2, "材料", self.project.material.name)

    def _geometry_kind_changed(self, kind):
        if kind == "file":
            self.import_file()
            return
        if kind == "model":
            feats = primitive_to_features(self.project.geometry) or []
            self._set_geometry({"kind": "model", "features": feats}, keep_bcs=True)
            return
        spec = {"kind": kind, **{k: d for k, _, d in PRIMITIVES[kind][1]}}
        self._set_geometry(spec)

    def _geometry_param(self, key, v):
        spec = dict(self.project.geometry)
        spec[key] = v
        self._set_geometry(spec, keep_bcs=True)

    # ---------------------------------------------------------------- モデリング
    def _features(self):
        return self.project.geometry.setdefault("features", [])

    def _ensure_model(self):
        """簡単形状・STEP をフィーチャーに変換してモデリングできるようにする"""
        g = self.project.geometry
        if g.get("kind") == "model":
            return True
        feats = primitive_to_features(g)
        if feats is None:
            r = QMessageBox.question(self, "モデリング", "現在の形状（STL など）はモデリングで編集できません。\n"
                                     "形状を消して新しくモデリングを始めますか？")
            if r != QMessageBox.Yes:
                return False
            feats = []
        self.project.geometry = {"kind": "model", "features": feats}
        return True

    def _selected_bounds(self):
        """選択中のボディ（無ければ全体）の範囲"""
        if self.mesh is None:
            return None
        if self.selected_bodies and self.mesh.elem_body is not None:
            m = np.isin(self.mesh.elem_body, self.selected_bodies)
            pts = self.mesh.nodes[np.unique(self.mesh.elems[m])]
            return pts.min(axis=0), pts.max(axis=0)
        return self.mesh.bounds()

    # ======================================================================= マウスで作図
    def _sketch_bar(self):
        bar = QFrame()
        bar.setStyleSheet("QFrame#skbar{background:#fff4d6;border-bottom:1px solid #e0b850;}"
                          "QFrame#skbar QLabel{color:#222;}"
                          "QFrame#skbar QPushButton{color:#222;background:#fff;border:1px solid #c9a227;"
                          "border-radius:3px;padding:2px 8px;}"
                          "QFrame#skbar QPushButton:hover{background:#ffe9a8;}")
        bar.setObjectName("skbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 3, 6, 3)
        self.sk_title = QLabel()
        self.sk_title.setStyleSheet("font-weight:bold;")
        h.addWidget(self.sk_title)
        h.addWidget(QLabel("作業平面"))
        self.sk_plane = QComboBox()
        self.sk_plane.addItems(["XY", "YZ", "ZX"])
        self.sk_plane.currentTextChanged.connect(lambda _: self._sketch_plane_changed())
        h.addWidget(self.sk_plane)
        self.sk_offset = dspin(0.0, dec=3, suffix="mm")
        self.sk_offset.setToolTip("作業平面の位置（XY 平面なら Z 座標）。既存ボディの平らな面をクリックすると、その面に合わせます")
        self.sk_offset.valueChanged.connect(lambda _: self._sketch_plane_changed())
        h.addWidget(self.sk_offset)
        h.addWidget(QLabel("スナップ"))
        self.sk_pitch = dspin(1.0, 0.0, 1e6, 3, 1.0, "mm")
        self.sk_pitch.setToolTip("クリック位置をこの間隔の格子に合わせます（0 = 合わせない）")
        self.sk_pitch.valueChanged.connect(lambda v: self._sketch_pitch(v))
        h.addWidget(self.sk_pitch)
        self.sk_value = QLineEdit()
        self.sk_value.setMinimumWidth(170)
        self.sk_value.returnPressed.connect(self._sketch_typed)
        h.addWidget(self.sk_value)
        self.sk_close_btn = QPushButton("閉じる")
        self.sk_close_btn.clicked.connect(self._sketch_enter)
        h.addWidget(self.sk_close_btn)
        b = QPushButton("1つ戻る")
        b.setToolTip("直前のクリックを取り消します（Backspace）")
        b.clicked.connect(self._sketch_back)
        h.addWidget(b)
        b = QPushButton("数値で入力…")
        b.clicked.connect(lambda: self._sketch_finish(dialog_only=True))
        h.addWidget(b)
        b = QPushButton("中止")
        b.setToolTip("Esc / 右クリックでも中止できます")
        b.clicked.connect(self.sketch_cancel)
        h.addWidget(b)
        h.addStretch(1)
        bar.hide()
        self.sk_bar = bar
        self._sk_shortcuts = []
        for key, fn in ((Qt.Key_Escape, self.sketch_cancel), (Qt.Key_Backspace, self._sketch_back),
                        (Qt.Key_Return, self._sketch_enter), (Qt.Key_Enter, self._sketch_enter)):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda fn=fn: fn() if self._sk is not None and not self.sk_value.hasFocus() else None)
            sc.setEnabled(False)
            self._sk_shortcuts.append(sc)
        return bar

    def sketch_start(self, ftype):
        from .sketch import Sketch, nice_pitch
        if self._sk is not None:
            self.sketch_cancel()
        span = 200.0
        if self.mesh is not None:
            lo, hi = self.mesh.bounds()
            span = max(float(np.max(hi - lo)) * 2, 10.0)
        pitch = nice_pitch(span)
        plane = "XY"
        offset = 0.0
        if ftype in ("plate", "sheet", "disk") and self.project.geometry.get("analysis_2d"):
            plane, offset = "XY", 0.0
        self._sk = Sketch(ftype, plane, offset, pitch)
        self._sk_span = span
        for w, v in ((self.sk_plane, plane),):
            w.blockSignals(True)
            w.setCurrentText(v)
            w.blockSignals(False)
        self.sk_offset.blockSignals(True)
        self.sk_offset.setValue(offset)
        self.sk_offset.blockSignals(False)
        self.sk_pitch.blockSignals(True)
        self.sk_pitch.setValue(pitch)
        self.sk_pitch.blockSignals(False)
        self.sk_bar.show()
        for sc in self._sk_shortcuts:
            sc.setEnabled(True)
        self.plotter.interactor.setFocus()
        self.sk_close_btn.setVisible(ftype in ("sheet", "extrude"))
        self._sketch_update()
        self.log(f"{FEATURES[ftype][0]}: 画面上をクリックして指定します（数値はバーに入力して Enter。Esc / 右クリックで中止）")

    def sketch_cancel(self):
        if self._sk is None:
            return
        self._sk = None
        self.sk_bar.hide()
        for sc in self._sk_shortcuts:
            sc.setEnabled(False)
        for name in ("sk_grid", "sk_lines", "sk_cursor", "sk_pts", "sk_label"):
            try:
                self.plotter.remove_actor(name, render=False)
            except Exception:
                pass
        self.plotter.render()

    def _sketch_ray(self, x, y):
        ren = self.plotter.renderer

        def d2w(z):
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            w = ren.GetWorldPoint()
            return np.array(w[:3]) / (w[3] if w[3] else 1.0)
        o, f = d2w(0.0), d2w(1.0)
        return o, f - o

    def _sketch_face_plane(self, x, y):
        """1点目の前: 既存ボディの軸に垂直な平らな面の上なら、その面を作業平面にする"""
        sk = self._sk
        if sk.step != 0 or sk.points or sk.poly or self._main_actor is None or self.mesh is None:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(self._main_actor)
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return
        n = np.array(picker.GetPickNormal())
        k = int(np.argmax(np.abs(n)))
        if abs(n[k]) < 0.999:
            return
        plane = {0: "YZ", 1: "ZX", 2: "XY"}[k]
        off = float(picker.GetPickPosition()[k])
        if plane != sk.plane or abs(off - sk.offset) > 1e-9:
            sk.set_plane(plane, off)
            self.sk_plane.blockSignals(True)
            self.sk_plane.setCurrentText(plane)
            self.sk_plane.blockSignals(False)
            self.sk_offset.blockSignals(True)
            self.sk_offset.setValue(off)
            self.sk_offset.blockSignals(False)

    def _sketch_plane_changed(self):
        if self._sk is None:
            return
        self._sk.set_plane(self.sk_plane.currentText(), self.sk_offset.value())
        self._sketch_update()

    def _sketch_pitch(self, v):
        if self._sk is not None:
            self._sk.pitch = float(v)

    def _sketch_move(self, x, y):
        sk = self._sk
        self._sketch_face_plane(x, y)
        o, d = self._sketch_ray(x, y)
        before = (None if sk.cursor is None else tuple(sk.cursor), sk.cursor_h)
        sk.move(o, d)
        if (None if sk.cursor is None else tuple(sk.cursor), sk.cursor_h) != before:
            self._sketch_update()

    def _sketch_click(self, x, y):
        sk = self._sk
        if sk is None:
            return
        self._sketch_face_plane(x, y)
        o, d = self._sketch_ray(x, y)
        if sk.click(o, d):
            self._sketch_finish()
        else:
            self._sketch_update()

    def _sketch_enter(self):
        if self._sk is not None and self._sk.close_poly():
            self._sketch_finish()
        else:
            self._sketch_update()

    def _sketch_back(self):
        if self._sk is not None:
            self._sk.undo_point()
            self._sketch_update()

    def _sketch_typed(self):
        """バーに打った数値で今の段階を確定（X,Y,Z / 平面上の u,v / @ で前の点からの相対 / 長さ）"""
        sk = self._sk
        if sk is None:
            return
        txt = self.sk_value.text().strip().replace("、", ",").replace(" ", ",")
        rel = txt.startswith("@")
        try:
            nums = [float(t) for t in txt.lstrip("@").split(",") if t != ""]
        except ValueError:
            self.log("数値を「X, Y, Z」または「u, v」（@ を付けると前の点からの相対）で入力してください")
            return
        if sk.kind == "h":
            if len(nums) != 1:
                return
            done = sk.click(value=nums[0])
        else:
            if len(nums) == 2:
                base = (sk.poly[-1] if sk.poly else sk.points[-1]) if rel and (sk.poly or sk.points) else None
                if base is not None:
                    u0, v0 = sk.uv(base)
                    p = sk.to3(u0 + nums[0], v0 + nums[1], base[sk.axes[2]])
                else:
                    p = sk.to3(nums[0], nums[1])
            elif len(nums) == 3:
                p = np.array(nums)
                if rel and (sk.poly or sk.points):
                    p = p + (sk.poly[-1] if sk.poly else sk.points[-1])
                p[sk.axes[2]] = sk.offset if not (sk.points or sk.poly) else (sk.poly or sk.points)[0][sk.axes[2]]
            else:
                return
            done = sk.click(value=p)
        self.sk_value.clear()
        self.plotter.interactor.setFocus()
        if done:
            self._sketch_finish()
        else:
            self._sketch_update()

    def _sketch_update(self):
        sk = self._sk
        if sk is None:
            return
        name = FEATURES[sk.ftype][0]
        total = len(sk.steps)
        self.sk_title.setText(f"{name}（{min(sk.step + 1, total)}/{total}）: {sk.prompt} をクリック")
        if sk.kind == "h":
            self.sk_value.setPlaceholderText("長さ [mm] を入力して Enter")
            if not self.sk_value.hasFocus():
                self.sk_value.setText("" if sk.cursor_h is None else f"{sk.cursor_h:g}")
        else:
            self.sk_value.setPlaceholderText("X, Y, Z（または u, v / @ で相対）を入力して Enter")
            if not self.sk_value.hasFocus():
                c = sk.cursor
                self.sk_value.setText("" if c is None else f"{c[0]:g}, {c[1]:g}, {c[2]:g}")
        pl = self.plotter
        # 作業平面の格子
        a = sk.axes
        center = sk.points[0] if sk.points else (sk.poly[0] if sk.poly else (sk.cursor if sk.cursor is not None
                                                                             else sk.to3(0, 0)))
        span = self._sk_span
        step = max(sk.pitch * max(1, round(span / 20 / max(sk.pitch, 1e-9))), 1e-9)
        cu, cv = sk.uv(center)
        cu, cv = round(cu / step) * step, round(cv / step) * step
        n = 10
        segs = []
        for i in range(-n, n + 1):
            segs.append((sk.to3(cu + i * step, cv - n * step), sk.to3(cu + i * step, cv + n * step)))
            segs.append((sk.to3(cu - n * step, cv + i * step), sk.to3(cu + n * step, cv + i * step)))
        grid = pv.PolyData(np.array([p for sgm in segs for p in sgm]))
        grid.lines = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(segs))])
        pl.add_mesh(grid, color="#e0a000", opacity=0.35, line_width=1, name="sk_grid", pickable=False,
                    reset_camera=False, render=False)
        lines, info = sk.preview()
        if lines:
            pts, cells, off = [], [], 0
            for L in lines:
                L = np.asarray(L, float)
                pts.append(L)
                cells += [len(L)] + list(range(off, off + len(L)))
                off += len(L)
            pd = pv.PolyData(np.vstack(pts))
            pd.lines = np.array(cells)
            pl.add_mesh(pd, color="#d9480f", line_width=3, name="sk_lines", pickable=False,
                        reset_camera=False, render=False)
        else:
            pl.remove_actor("sk_lines", render=False)
        marks = list(sk.points) + list(sk.poly)
        if sk.cursor is not None and sk.kind in ("pt", "poly"):
            marks.append(sk.cursor)
        if marks:
            pl.add_points(np.array(marks, float), color="#d9480f", point_size=10, render_points_as_spheres=True,
                          name="sk_pts", pickable=False, reset_camera=False, render=False)
        # 3D 画面の文字は日本語が出ないので英数字で（説明は上のバーに日本語で出している）
        c = sk.cursor
        lines_txt = []
        if c is not None and sk.kind in ("pt", "poly"):
            lines_txt.append(f"X {c[0]:g}  Y {c[1]:g}  Z {c[2]:g}")
        if sk.kind == "h" and sk.cursor_h is not None:
            lines_txt.append(f"H {sk.cursor_h:g} mm")
        if info:
            for jp, en in (("半径", "R"), ("長さ", "L"), ("頂点", "Point"), (" / ", "   "), ("×", "x")):
                info = info.replace(jp, en)
            lines_txt.append(info)
        txt = "\n".join(lines_txt)
        pl.add_text(txt, position="upper_right", font_size=10, color="#d9480f", name="sk_label", render=False)
        pl.render()

    def _sketch_finish(self, dialog_only=False):
        sk = self._sk
        if sk is None:
            return
        ftype = sk.ftype
        f = new_feature(ftype, "new")
        if not dialog_only:
            try:
                f.update(sk.feature_values())
            except Exception as e:
                self.log(f"作図の値を作れませんでした: {e}")
        self.sketch_cancel()
        dlg = FeatureDialog(ftype, f, "の作成（クリックした寸法。必要なら修正して OK）", self)
        if not dlg.exec():
            return
        dlg.apply_to(f)
        self._features().append(f)
        self.selected_faces, self.selected_bodies = [], []
        self.current = ("feature", len(self._features()) - 1)
        self.remesh(reset_camera=len(self._features()) == 1)

    def create_body(self, ftype, dialog=False):
        """［作成］→ 画面上をクリックして基準点・寸法を指定（Femtet と同じ）→ 確認ダイアログ → 新しいボディ"""
        if not self._ensure_model():
            return
        from .sketch import supported
        if not dialog and supported(ftype):
            self.sketch_start(ftype)
            return
        f = new_feature(ftype, "new")
        if self.mesh is not None:
            lo, hi = self.mesh.bounds()
            if "x" in f:
                f["x"] = float(hi[0])      # 既存形状の +X 側に並べる（重ねると接合されます）
            if ftype == "extrude":
                f["offset"] = float(hi[2])
        dlg = FeatureDialog(ftype, f, "の作成", self)
        if not dlg.exec():
            return
        dlg.apply_to(f)
        self._features().append(f)
        self.selected_faces, self.selected_bodies = [], []
        self.current = ("feature", len(self._features()) - 1)
        self.remesh(reset_camera=len(self._features()) == 1)

    def _feature_label(self, f):
        t = f.get("type")
        if t == "bool":
            return ("結合" if f.get("op") == "union" else "除去") + f": B{f.get('target')}"
        name = FEATURES.get(t, (t,))[0]
        if t not in EDIT_FEATURES:
            return name
        tg = ", ".join(f"B{k}" for k in f.get("targets") or []) or "全ボディ"
        g = lambda k, d=0.0: float(f.get(k, d))  # noqa: E731
        if t == "move":
            parts = []
            if g("dx") or g("dy") or g("dz"):
                parts.append(f"({g('dx'):g}, {g('dy'):g}, {g('dz'):g})")
            if g("angle"):
                parts.append(f"{f.get('axis', 'Z')} 軸 {g('angle'):g}°")
            d = f"{f.get('copy', '移動')} " + " ".join(parts)
        elif t == "mirror":
            d = f"{f.get('copy', 'コピー')} {f.get('axis', 'X')} = {g('offset'):g}"
        elif t == "split":
            d = f"{f.get('axis', 'X')} = {g('offset'):g}（{str(f.get('keep', '両方'))[:2]}）"
        elif t == "stretch":
            d = f"{f.get('axis', 'X')} = {g('offset'):g} から {g('length'):g} mm"
        elif t == "fillet":
            d = f"R{g('r', 2):g} {f.get('edges', 'すべての辺')}"
        else:
            d = f"C{g('d', 2):g} {f.get('edges', 'すべての辺')}"
        return f"{name}: {tg} {d}"

    def edit_body_op(self, ftype):
        """ボディ編集（移動・回転・コピー / ミラー / 切断 / 引き伸ばし / 丸み付け / 面取り）"""
        if self.mesh is None or not self._ensure_model() or not self._features():
            QMessageBox.information(self, FEATURES[ftype][0], "先に形状を作成してください。")
            return
        lo, hi = self._selected_bounds()
        c = 0.5 * (lo + hi)
        f = new_feature(ftype)
        if ftype == "move":
            f.update(x=float(c[0]), y=float(c[1]), z=float(c[2]))
        elif ftype in ("split", "stretch"):
            f["axis"] = "XYZ"[int(np.argmax(hi - lo))]
            f["offset"] = float(c["XYZ".index(f["axis"])])
        elif ftype == "mirror":
            f["offset"] = float(hi[0])
        elif ftype in ("fillet", "chamfer"):
            pad = 1e-3 * float(np.linalg.norm(hi - lo)) + 1e-6
            for j, a in enumerate("xyz"):
                f[a + "0"], f[a + "1"] = float(lo[j] - pad), float(hi[j] + pad)
            small = float(np.min(hi - lo))
            f["r" if ftype == "fillet" else "d"] = round(max(small / 10, 0.1), 3)
        targets = list(self.selected_bodies)
        txt = ("対象ボディ: " + ", ".join(f"B{b}" for b in targets)) if targets else "対象ボディ: すべて（ボディを選んでから押すと対象を絞れます）"
        dlg = FeatureDialog(ftype, f, "", self, targets_text=txt)
        if not dlg.exec():
            return
        dlg.apply_to(f)
        if targets:
            f["targets"] = targets
        self._hist_label = self._feature_label(f)
        self._features().append(f)
        self.current = ("feature", len(self._features()) - 1)
        if not self.remesh():
            self._features().pop()
            self.current = ("geometry",)
            self.remesh()

    def create_hole(self):
        """選択したボディ（未選択なら全ボディ）に円柱の穴をあける"""
        if self.mesh is None or not self._ensure_model():
            QMessageBox.information(self, "穴", "先に形状を作成してください。")
            return
        lo, hi = self._selected_bounds()
        c, sz = 0.5 * (lo + hi), hi - lo
        if is_shell_model(self.project.geometry) and sz[2] < 1e-6 * max(1.0, float(sz.max())):
            # 平らなサーフェス（2D・シェル）の穴 = 円板で切り取る
            f = new_feature("disk", "cut", u0=float(c[0]), v0=float(c[1]), offset=float(lo[2]),
                            r=float(max(min(sz[0], sz[1]) / 6, 0.5)))
            dlg = FeatureDialog("disk", f, "（穴）", self)
            if not dlg.exec():
                return
            dlg.apply_to(f)
            if self.selected_bodies:
                f["targets"] = list(self.selected_bodies)
            self._features().append(f)
            self.current = ("feature", len(self._features()) - 1)
            self.remesh()
            return
        f = new_feature("cylinder", "cut", x=float(c[0]), y=float(c[1]), z=float(lo[2] - 1), axis="Z",
                        h=float(sz[2] + 2), r=float(max(min(sz[0], sz[1]) / 6, 0.5)))
        targets = list(self.selected_bodies)
        txt = ("対象ボディ: " + ", ".join(f"B{b}" for b in targets)) if targets else "対象ボディ: すべて"
        dlg = FeatureDialog("cylinder", f, "（穴）", self, targets_text=txt)
        if not dlg.exec():
            return
        dlg.apply_to(f)
        if targets:
            f["targets"] = targets
        self._features().append(f)
        self.current = ("feature", len(self._features()) - 1)
        self.remesh()

    def boolean(self, op):
        """結合 / 除去: 最初に選んだボディが対象、残りがツール（Femtet と同じ）"""
        if len(self.selected_bodies) < 2:
            QMessageBox.information(self, "ブール演算",
                                    "ボディ選択モードで、まず対象ボディ、続けて Ctrl(⌘)+クリックで\n"
                                    "ツールボディを選んでから押してください。")
            return
        tgt, tools = self.selected_bodies[0], self.selected_bodies[1:]
        self._features().append({"type": "bool", "op": op, "target": int(tgt), "tools": [int(t) for t in tools]})
        self.log(f"{'結合' if op == 'union' else '除去'}: B{tgt} ← {', '.join(f'B{t}' for t in tools)}")
        self.selected_bodies = [tgt]
        self.current = ("feature", len(self._features()) - 1)
        self.remesh()

    def edit_feature(self, i):
        f = self._features()[i]
        if f["type"] not in FEATURES:
            return
        dlg = FeatureDialog(f["type"], f, "の編集", self)
        if dlg.exec():
            dlg.apply_to(f)
            self.remesh()

    def edit_selected_body(self):
        if self.current[0] == "feature":
            return self.edit_feature(self.current[1])
        if self.selected_bodies and self.project.geometry.get("kind") == "model":
            keys = {v: k for k, v in feature_body_keys(self._features()).items()}
            if self.selected_bodies[0] in keys:
                return self.edit_feature(keys[self.selected_bodies[0]])
        QMessageBox.information(self, "形状編集", "ツリーで形状を選ぶか、ボディを1つ選んでから押してください。")

    def add_import_feature(self):
        path, _ = QFileDialog.getOpenFileName(self, "STEP を取り込む", "", "CAD (*.step *.stp *.STEP *.STP *.igs *.iges *.brep)")
        if not path:
            return
        if not self._ensure_model():
            return
        self._features().append(new_feature("import", path=path))
        self.current = ("feature", len(self._features()) - 1)
        self.remesh(reset_camera=True)

    def _details_feature(self):
        i = self.current[1]
        f = self._features()[i]
        if f["type"] == "bool":
            s = self._section("ブール演算")
            self._row(s, "種類", "結合" if f.get("op") == "union" else "除去")
            self._row(s, "対象ボディ", f"B{f.get('target')}")
            self._row(s, "ツールボディ", ", ".join(f"B{k}" for k in f.get("tools", [])))
            b = QPushButton("この操作を取り消す（削除）")
            b.clicked.connect(lambda: self._later(self.delete_current))
            self._row(s, "", b)
            return
        name, _, params = FEATURES[f["type"]]
        keys = feature_body_keys(self._features())
        if i in keys:
            k = keys[i]
            a = self.project.assignment(k)
            s0 = self._section("ボディ")
            self._row(s0, "ボディキー", f"B{k}")
            self._row(s0, "ボディ属性名", a.get("attr") or "未定義")
            self._row(s0, "材料名", a.get("mat") or "未定義")
        s = self._section("フィーチャー")
        self._row(s, "タイプ", name)
        eb = QPushButton("形状を編集（ダイアログ）…")
        eb.clicked.connect(lambda: self._later(lambda: self.edit_feature(i)))
        self._row(s, "", eb)
        if f["type"] in EDIT_FEATURES:
            self._row(s, "対象ボディ", ", ".join(f"B{k}" for k in f.get("targets") or []) or "すべて")
            tb = QPushButton("選択中のボディを対象にする")
            tb.clicked.connect(lambda: self._feature_set(i, "targets", list(self.selected_bodies), rebuild=True))
            self._row(s, "", tb)
        else:
            op = QComboBox()
            for k, v in OPS.items():
                op.addItem(v, k)
            op.setCurrentIndex(max(0, op.findData(f.get("op", "add"))))
            op.currentIndexChanged.connect(lambda _: self._feature_set(i, "op", op.currentData(), rebuild=True))
            self._row(s, "操作", op)
        s2 = self._section("寸法" if f["type"] not in EDIT_FEATURES else "設定")
        for key, label, default in params:
            v = f.get(key, default)
            if key in FEATURE_CHOICES:
                w = QComboBox()
                w.addItems(FEATURE_CHOICES[key])
                w.setCurrentText(str(v))
                w.currentTextChanged.connect(lambda t, key=key: self._feature_set(i, key, t, rebuild=True))
            elif key == "axis":
                w = QComboBox()
                w.addItems(AXES)
                w.setCurrentText(str(v))
                w.currentTextChanged.connect(lambda t, key=key: self._feature_set(i, key, t))
            elif key == "plane":
                w = QComboBox()
                w.addItems(PLANES)
                w.setCurrentText(str(v))
                w.currentTextChanged.connect(lambda t, key=key: self._feature_set(i, key, t))
            elif key == "points":
                w = QLineEdit(field_text(f, key, default))
                w.setToolTip("スケッチ平面上の (u, v) 座標を「;」区切りで。XY 平面なら u=X, v=Y、"
                             "YZ 平面なら u=Y, v=Z、ZX 平面なら u=Z, v=X")
                w.editingFinished.connect(lambda w=w, key=key: self._feature_set(i, key, w.text()))
            elif key == "path":
                w = QWidget()
                h = QHBoxLayout(w)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(QLabel(os.path.basename(str(v)) or "（未選択）"))
                b = QPushButton("選択…")

                def pick(_=False):
                    pth, _f = QFileDialog.getOpenFileName(self, "STEP を選択", "", "CAD (*.step *.stp *.STEP *.STP *.igs *.iges *.brep)")
                    if pth:
                        self._feature_set(i, "path", pth, rebuild=True)
                b.clicked.connect(pick)
                h.addWidget(b)
            else:
                w = ExprEdit(field_text(f, key, default))
                w.rawChanged.connect(lambda val, key=key: self._feature_set(i, key, val))
            self._row(s2, label, w)
        s3 = self._section("操作")
        sup = QCheckBox("抑制（一時的に無効）")
        sup.setChecked(bool(f.get("suppressed")))
        sup.toggled.connect(lambda on: self._feature_set(i, "suppressed", on))
        self._row(s3, "", sup)
        b = QPushButton("このフィーチャーを削除")
        b.clicked.connect(lambda: self._later(self.delete_current))
        self._row(s3, "", b)

    def _feature_set(self, i, key, value, rebuild=False):
        feats = self._features()
        if i >= len(feats):
            return
        if key == "points" or (isinstance(value, (int, float, str)) and not isinstance(value, bool)
                               and key not in FEATURE_CHOICES and key not in ("axis", "plane", "path", "op")):
            try:
                set_field(feats[i], key, value, self.project.env())
            except (ParamError, ValueError) as e:
                QMessageBox.warning(self, "式の入力", str(e))
                return
        else:
            feats[i][key] = value
        self.mesh_dirty = True
        self.result = None
        # 連続で値を変えても1回だけ作り直す
        if not hasattr(self, "_regen_timer"):
            self._regen_timer = QTimer(self)
            self._regen_timer.setSingleShot(True)
            self._regen_timer.timeout.connect(lambda: self.remesh())
        self._regen_timer.start(350)

    def _details_mats(self):
        s = self._section("材料定数")
        self._row(s, "登録数", len(self.project.materials))
        if self.mesh is not None:
            s2 = self._section("ボディへの割り当て")
            for b in self.mesh.body_ids():
                a = self.project.assignment(b)
                self._row(s2, f"B{b}", f"{a.get('attr') or '未定義'} / {a.get('mat') or '未定義'}")
        b = QPushButton("新しい材料定数…")
        b.clicked.connect(lambda: self._later(lambda: self.edit_material(f"材料{len(self.project.materials) + 1}")))
        self._row(self._section("操作"), "", b)

    def _details_attrs(self):
        s = self._section("ボディ属性")
        self._row(s, "登録数", len(self.project.body_attrs))
        self._row(s, "", "ボディを選び、上の欄に名前を入力して［設定］")

    def _details_mat(self):
        name = self.current[1]
        m = self.project.materials.get(name)
        if m is None:
            return
        s = self._section("材料定数")
        self._row(s, "材料名", name)
        for attr, label, lo, hi, dec, step in [("E", "ヤング率 [MPa]", 1e-3, 1e8, 1, 1000),
                                               ("nu", "ポアソン比", 0, 0.4999, 4, 0.01),
                                               ("density", "密度 [kg/m³]", 0, 1e6, 1, 10),
                                               ("yield_strength", "引張降伏応力 [MPa]", 0, 1e6, 1, 5),
                                               ("tensile_strength", "引張強さ [MPa]", 0, 1e6, 1, 5)]:
            sp = dspin(getattr(m, attr), lo, hi, dec, step)
            sp.valueChanged.connect(lambda v, attr=attr: (setattr(m, attr, v), self.invalidate()))
            self._row(s, label, sp)
        from ..materials import ELASTIC_MODELS, PLASTIC_MODELS
        s2 = self._section("モデル（詳しくは［編集］で）")
        self._row(s2, "弾性", ELASTIC_MODELS.get(m.elastic, m.elastic)
                  + ("（温度依存 %d 点）" % len(m.temp_table) if m.elastic == "isotropic" and m.temp_table else ""))
        if m.elastic == "orthotropic" and m.ortho:
            o = m.ortho
            self._row(s2, "Ex / Ey / Ez", f"{o['Ex']:g} / {o['Ey']:g} / {o['Ez']:g} MPa")
        self._row(s2, "線膨張係数", f"{m.alpha * 1e6:g} ×10⁻⁶/°C（参照 {m.ref_temp:g}°C）")
        self._row(s2, "塑性", PLASTIC_MODELS.get(m.plasticity, m.plasticity)
                  + (f"（接線係数 {m.tangent_modulus:g} MPa）" if m.plasticity.startswith("bilinear") else "")
                  + (f"（{len(m.ml_table)} 点）" if m.plasticity.startswith("multilinear") else ""))
        if self.mesh is not None:
            used = [f"B{b}" for b in self.mesh.body_ids() if self.project.assignment(b).get("mat") == name]
            self._row(self._section("使用しているボディ"), "", ", ".join(used) or "なし")
        b = QPushButton("編集（弾性・温度依存・異方性・熱膨張・塑性）…")
        b.clicked.connect(lambda: self._later(lambda: self.edit_material(name)))
        self._row(self._section("操作"), "", b)

    def _details_attr(self):
        name = self.current[1]
        props = self.project.body_attrs.get(name, {})
        s = self._section("Body 属性")
        self._row(s, "ボディ属性名", name)
        sp = dspin(float(props.get("mesh_size", 0) or 0), 0, 1e6, 3, 0.5, "mm")
        sp.setSpecialValueText("自動")

        def setsize(v):
            self.project.body_attrs.setdefault(name, {})["mesh_size"] = v
            self.invalidate(mesh=True)
        sp.valueChanged.connect(setsize)
        self._row(s, "メッシュサイズ", sp)
        s3 = self._section("材料の向き（材料座標系, X→Y→Z の順に回転）")
        cs = list(props.get("cs") or [0, 0, 0])
        for k, ax in enumerate("XYZ"):
            spr = dspin(float(cs[k]), -360, 360, 2, 5, "°")

            def setcs(v, k=k):
                c = list(self.project.body_attrs.setdefault(name, {}).get("cs") or [0, 0, 0])
                c[k] = v
                self.project.body_attrs[name]["cs"] = c
                self.invalidate()
            spr.valueChanged.connect(setcs)
            self._row(s3, f"{ax} 軸まわり", spr)
        if self.mesh is not None:
            used = [f"B{b}" for b in self.mesh.body_ids() if self.project.assignment(b).get("attr") == name]
            self._row(self._section("使用しているボディ"), "", ", ".join(used) or "なし")

    # ---------------------------------------------------------------- 接続（接触）
    def _body_label(self, b):
        if b is None or int(b) < 0:
            return "-"
        return self.project.assignment(int(b)).get("attr") or f"B{int(b)}"

    def _contact_bodies(self, faces):
        if self.mesh is None or not faces:
            return []
        tb = self.mesh.tri_body()
        return sorted({int(tb[self.mesh.face_ids == f][0]) for f in faces if np.any(self.mesh.face_ids == f)})

    def _contact_label(self, c):
        """Ansys と同じく「タイプ - 接触ボディ と ターゲットボディ」（名前を変えたときはその名前）"""
        if c.name and not c.name.startswith("接触領域"):
            return f"{c.name}　[{CONTACT_TYPES[c.type]}]"
        cb = self._contact_bodies(c.contact_faces) or ([c.contact_body] if c.contact_body >= 0 else [])
        tb = self._contact_bodies(c.target_faces) or ([c.target_body] if c.target_body >= 0 else [])
        names = lambda bs: "・".join(self._body_label(b) for b in bs) or "未選択"  # noqa: E731
        mu = f"（μ={c.mu:g}）" if c.type == "frictional" else ""
        return f"{CONTACT_TYPES[c.type]}{mu} - {names(cb)} と {names(tb)}"

    def _details_connections(self):
        p = self.project
        s = self._section("自動検出")
        tol = dspin(p.contact_tolerance, 0.0, 1e6, 4, 0.1, "mm")
        tol.setSpecialValueText("プログラム制御")
        tol.valueChanged.connect(lambda v: setattr(p, "contact_tolerance", v))
        self._row(s, "許容値", tol)
        self._row(s, "面/面", "はい")
        self._row(s, "面/辺・辺/辺", "いいえ（未対応）")
        self._row(s, "グループ化", "ボディ（ボディの組ごとに 1 つの接触領域）")
        self._row(s, "作成する接触のタイプ", self._combo(list((v, k) for k, v in CONTACT_TYPES.items()),
                                                 p.contact_default_type, lambda v: (
                                                     setattr(p, "contact_default_type", v),
                                                     self._later(self.build_details))))
        if p.contact_default_type == "frictional":
            mu = dspin(p.contact_default_mu, 0.0, 10.0, 4, 0.05)
            mu.valueChanged.connect(lambda v: setattr(p, "contact_default_mu", v))
            self._row(s, "摩擦係数", mu)
        b = QPushButton("接続を作成（自動検出）")
        b.clicked.connect(lambda: self._later(self.auto_contacts))
        self._row(s, "", b)
        s = self._section("すべての接触領域を一括変更")
        self._row(s, "タイプ", self._combo([("（選んでください）", "")] + list((v, k) for k, v in CONTACT_TYPES.items()),
                                         "", lambda v: v and self.set_contact_type(v)))
        s2 = self._section("情報")
        self._row(s2, "接触領域の数", len(p.contacts))
        if self.mesh is not None:
            self._row(s2, "ボディ数", len(self.mesh.body_ids()))
        if p.share_topology != "none":
            self._row(s2, "", "節点共有がオンなので、共有したボディ同士には接触は作られません。"
                              "接触でつなぐときは「モデル → 節点共有しない」にしてください")
        else:
            self._row(s2, "", "ボディ同士は接触でつながります。接触領域ごとにタイプ・摩擦係数を変えられます")

    _details_contacts = _details_connections

    GASKET_DEFAULT = {"closure": [0.0, 0.05, 0.1, 0.15], "pressure": [0.0, 5.0, 20.0, 60.0],
                      "unload": 0.0, "shear": 0.0}

    def _gasket_rows(self, s, i, c):
        """ガスケット: 閉じ量-面圧の表、除荷の傾き、面内のせん断剛性"""
        if not c.gasket:
            c.gasket = dict(self.GASKET_DEFAULT)
        gk = c.gasket

        def fmt(v):
            return ", ".join(f"{x:g}" for x in v)

        def set_list(key, ed):
            def f():
                try:
                    vals = [float(x) for x in ed.text().replace("、", ",").split(",") if x.strip()]
                except ValueError:
                    QMessageBox.warning(self, "ガスケット", "数値をカンマ区切りで入力してください")
                    ed.setText(fmt(c.gasket.get(key, [])))
                    return
                self._contact_set(i, "gasket", dict(c.gasket, **{key: vals}))
            return f
        for key, lab in (("closure", "閉じ量 [mm]（カンマ区切り）"), ("pressure", "面圧 [MPa]（カンマ区切り）")):
            ed = QLineEdit(fmt(gk.get(key, [])))
            ed.editingFinished.connect(set_list(key, ed))
            self._row(s, lab, ed)

        def set_num(key):
            return lambda v: self._contact_set(i, "gasket", dict(c.gasket, **{key: float(v)}))
        un = QDoubleSpinBox()
        un.setRange(0.0, 1e9)
        un.setDecimals(2)
        un.setSuffix(" MPa/mm")
        un.setSpecialValueText("0（負荷曲線をたどる）")
        un.setValue(float(gk.get("unload", 0.0) or 0.0))
        un.valueChanged.connect(set_num("unload"))
        self._row(s, "除荷の傾き", un)
        sh = QDoubleSpinBox()
        sh.setRange(0.0, 1e9)
        sh.setDecimals(2)
        sh.setSuffix(" MPa/mm")
        sh.setSpecialValueText("0（面内は拘束しない）")
        sh.setValue(float(gk.get("shear", 0.0) or 0.0))
        sh.valueChanged.connect(set_num("shear"))
        self._row(s, "面内のせん断剛性", sh)
        self._row(s, "", "ガスケットの圧縮特性（閉じ量 → 面圧）を、2 つの面の間の層として扱います。"
                         "除荷の傾きを入れると、除荷時はその傾きで戻り、面圧 0 で永久変形が残ります。")

    def _details_contact(self):
        i = self.current[1]
        c = self.project.contacts[i]
        st = lambda key, rebuild=False: (lambda v: self._contact_set(i, key, v, rebuild=rebuild))  # noqa: E731
        combo = lambda d, cur, key, rebuild=True: self._combo(list((v, k) for k, v in d.items()), cur,  # noqa: E731
                                                              st(key, rebuild))
        s = self._section("スコープ")
        self._row(s, "スコープ方法", "ジオメトリの選択")
        self._scope_row_named(s, "接触", c.contact_faces, lambda f: self._contact_set(i, "contact_faces", f))
        self._scope_row_named(s, "ターゲット", c.target_faces, lambda f: self._contact_set(i, "target_faces", f))
        self._row(s, "接触ボディ", "・".join(self._body_label(b) for b in self._contact_bodies(c.contact_faces)) or "-")
        self._row(s, "ターゲットボディ", "・".join(self._body_label(b) for b in self._contact_bodies(c.target_faces)) or "-")
        s = self._section("定義")
        self._row(s, "タイプ", combo(CONTACT_TYPES, c.type, "type"))
        if c.type == "frictional":
            mu = dspin(c.mu, 0.0, 10.0, 4, 0.05)
            mu.valueChanged.connect(st("mu", True))
            self._row(s, "摩擦係数", mu)
        if c.type == "gasket":
            self._gasket_rows(s, i, c)
        self._row(s, "スコープモード", "自動" if c.auto else "手動")
        self._row(s, "動作", combo(BEHAVIORS, c.behavior, "behavior", False))
        self._row(s, "トリム接触", "プログラム制御")
        sup = QCheckBox("抑制")
        sup.setChecked(c.suppressed)
        sup.toggled.connect(st("suppressed", True))
        self._row(s, "抑制", sup)
        name = QLineEdit(c.name)
        name.editingFinished.connect(lambda: self._contact_set(i, "name", name.text().strip() or c.name))
        self._row(s, "名前", name)

        s = self._section("高度な設定")
        self._row(s, "定式化", combo(FORMULATIONS, c.formulation, "formulation"))
        if c.nonlinear and c.type != "gasket":
            from ..contact import SLIDING
            self._row(s, "すべり", combo(SLIDING, c.sliding, "sliding", False))
        else:
            self._row(s, "微小すべり", "オン（接触相手は変形前の位置で決める）")
        self._row(s, "検出方法", combo(DETECTIONS, c.detection, "detection", False))
        if c.resolved_formulation() == "augmented_lagrange" or c.formulation == "penalty":
            self._row(s, "貫通許容値", combo(TOL_MODES, c.pen_tol_mode, "pen_tol_mode"))
            if c.pen_tol_mode == "factor":
                w = dspin(c.pen_tol or 0.1, 0.0, 1e3, 4, 0.05)
                w.valueChanged.connect(st("pen_tol"))
                self._row(s, "貫通許容係数（× 要素サイズ）", w)
            elif c.pen_tol_mode == "value":
                w = dspin(c.pen_tol, 0.0, 1e6, 5, 0.001, "mm")
                w.valueChanged.connect(st("pen_tol"))
                self._row(s, "貫通許容値", w)
        if c.type == "frictional":
            self._row(s, "弾性すべり許容値", combo(TOL_MODES, c.slip_tol_mode, "slip_tol_mode"))
            if c.slip_tol_mode == "factor":
                w = dspin(c.slip_tol or 0.01, 0.0, 1e3, 4, 0.01)
                w.valueChanged.connect(st("slip_tol"))
                self._row(s, "弾性すべり許容係数（× 要素サイズ）", w)
            elif c.slip_tol_mode == "value":
                w = dspin(c.slip_tol, 0.0, 1e6, 5, 0.001, "mm")
                w.valueChanged.connect(st("slip_tol"))
                self._row(s, "弾性すべり許容値", w)
        self._row(s, "法線剛性", combo(STIFFNESS_MODES, c.stiffness_mode, "stiffness_mode"))
        if c.stiffness_mode == "factor":
            kf = dspin(c.stiffness_factor or 1.0, 0.0, 1e6, 4, 0.1)
            kf.valueChanged.connect(st("stiffness_factor"))
            self._row(s, "法線剛性係数", kf)
        elif c.stiffness_mode == "absolute":
            kv = dspin(c.stiffness_value, 0.0, 1e12, 3, 100.0, "N/mm³")
            kv.valueChanged.connect(st("stiffness_value"))
            self._row(s, "法線剛性", kv)
        self._row(s, "剛性の更新", combo(UPDATE_STIFFNESS, c.update_stiffness, "update_stiffness", False))
        if c.nonlinear:
            sd = dspin(c.stabilization, 0.0, 1.0, 4, 0.01)
            sd.setSpecialValueText("0（なし）")
            sd.valueChanged.connect(st("stabilization"))
            self._row(s, "安定化減衰係数", sd)
        self._row(s, "ピンボール領域", combo(PINBALL_MODES, c.pinball_mode, "pinball_mode"))
        if c.pinball_mode == "radius":
            pb = dspin(c.pinball, 0.0, 1e6, 3, 0.5, "mm")
            pb.valueChanged.connect(st("pinball"))
            self._row(s, "ピンボール半径", pb)

        s = self._section("ジオメトリ修正")
        self._row(s, "インターフェース処理", combo(INTERFACE, c.interface, "interface"))
        if c.interface == "offset":
            off = dspin(c.offset, -1e6, 1e6, 4, 0.01, "mm")
            off.valueChanged.connect(st("offset"))
            self._row(s, "オフセット", off)
        th = dict(c.thread or {})
        on = bool(th.get("axis"))
        cb = QComboBox()
        cb.addItem("なし", False)
        cb.addItem("ボルトねじ", True)
        cb.setCurrentIndex(1 if on else 0)

        def set_thread(_=None, cb=cb):
            if cb.currentData():
                faces = c.contact_faces or []
                ax, pt = [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]
                if self.mesh is not None and faces:
                    # 円筒面の軸を推定: 面の節点の主成分（最も長い方向）
                    X = self.mesh.nodes[self.mesh.face_nodes(faces)]
                    cen = X.mean(0)
                    w_, V = np.linalg.eigh(np.cov((X - cen).T))
                    ax, pt = V[:, int(np.argmax(w_))].tolist(), cen.tolist()
                self._contact_set(i, "thread", dict(th or {}, point=pt, axis=ax, half_angle=th.get("half_angle", 30.0),
                                                     pitch=th.get("pitch", 1.5)), True)
            else:
                self._contact_set(i, "thread", {}, True)
        cb.currentIndexChanged.connect(set_thread)
        self._row(s, "接触ジオメトリ修正", cb)
        if on:
            def tset(k, j=None):
                def f(v):
                    t2 = dict(c.thread)
                    if j is None:
                        t2[k] = float(v)
                    else:
                        lst = list(t2.get(k, [0, 0, 0]))
                        lst[j] = float(v)
                        t2[k] = lst
                    self._contact_set(i, "thread", t2)
                return f
            for k, lab in (("point", "軸上の点"), ("axis", "軸の向き")):
                for j, xyz in enumerate("XYZ"):
                    sp = dspin(float(th[k][j]), -1e6, 1e6, 4, 1.0)
                    sp.valueChanged.connect(tset(k, j))
                    self._row(s, f"{lab} {xyz}", sp)
            ha = dspin(float(th.get("half_angle", 30.0)), 1.0, 89.0, 2, 1.0, "°")
            ha.valueChanged.connect(tset("half_angle"))
            self._row(s, "ねじ山の半角（メートルねじ 30°）", ha)
            pc = dspin(float(th.get("pitch", 1.5)), 0.01, 100.0, 3, 0.25, "mm")
            pc.valueChanged.connect(tset("pitch"))
            self._row(s, "ピッチ", pc)
            hc = QComboBox()
            hc.addItem("右ねじ", "right")
            hc.addItem("左ねじ", "left")
            hc.setCurrentIndex(1 if th.get("hand") == "left" else 0)
            hc.currentIndexChanged.connect(lambda _=None, hc=hc: self._contact_set(i, "thread", dict(c.thread, hand=hc.currentData())))
            self._row(s, "ねじの向き", hc)
            self._row(s, "", "円筒の接触面を、ねじ山の 2 つの斜面として扱います（軸方向の力がねじ山を通り、"
                             "tan(半角) 倍の半径方向の力と、リード角によるゆるめトルクが出ます）。軸は面の形から推定しています。")
        self._row(s, "ターゲットジオメトリ修正", "なし")

    def _scope_row_named(self, sec, label, faces, apply_fn):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(f"{len(faces)} 面" if faces else "未選択")
        if not faces:
            lab.setStyleSheet("background:#fff3b0; padding:1px 4px;")
        h.addWidget(lab)
        b = QPushButton("適用")

        def _apply():
            if not self.selected_faces:
                QMessageBox.information(self, "面を選択", "グラフィックスで面をクリックして選んでから「適用」を押してください。")
                return
            apply_fn(list(self.selected_faces))
            self._later(lambda: (self.build_details(), self.refresh_view()))
        b.clicked.connect(_apply)
        h.addWidget(b)
        h.addStretch()
        self._row(sec, label, w)

    def _contact_set(self, i, key, value, rebuild=False):
        c = self.project.contacts[i]
        setattr(c, key, value)
        if key in ("contact_faces", "target_faces"):
            c.auto = False
        self.invalidate()
        if rebuild or key in ("type", "name", "mu"):
            self._later(lambda: (self.rebuild_tree(), self.build_details(), self.refresh_view()))
        else:
            self.refresh_view()

    def auto_contacts(self, quiet=False):
        """接続の自動作成（手動で作った接触領域は残す）"""
        if self.mesh is None:
            return
        old_auto = {(c.contact_body, c.target_body): c for c in self.project.contacts if c.auto}
        manual = [c for c in self.project.contacts if not c.auto]
        found = detect_contacts(self.mesh, self.project.contact_tolerance, self.project.contact_default_type,
                                self.project.contact_default_mu)
        keep = [k for k in ContactRegion.__dataclass_fields__
                if k not in ("contact_faces", "target_faces", "contact_body", "target_body", "auto")]
        for c in found:
            o = old_auto.get((c.contact_body, c.target_body))
            if o is not None:   # 以前の設定（タイプ・摩擦係数・高度な設定など）を引き継ぐ
                for k in keep:
                    setattr(c, k, getattr(o, k))
        self.project.contacts = found + manual
        if not quiet:
            self.log(f"接触を自動検出しました: {len(found)} 領域")
        self.invalidate()
        self.rebuild_tree()
        self.refresh_view()

    def add_contact(self):
        """手動の接触領域（選択中の面を接触側に）"""
        c = ContactRegion(name=f"接触領域 {len(self.project.contacts) + 1}", contact_faces=list(self.selected_faces),
                          auto=False)
        self.project.contacts.append(c)
        self.invalidate()
        self.select_node(("contact", len(self.project.contacts) - 1))
        self.log("接触領域を追加しました。接触側・ターゲット側の面を選び、詳細の「適用」で割り当ててください。")

    def set_contact_type(self, kind):
        if self.current[0] == "contact":
            idx = [self.current[1]]
        else:
            idx = list(range(len(self.project.contacts)))
            if not idx:
                return
            if QMessageBox.question(self, "接触タイプ", f"すべての接触領域を「{CONTACT_TYPES[kind]}」にしますか？") != QMessageBox.Yes:
                return
        for i in idx:
            self.project.contacts[i].type = kind
        self.invalidate()
        self.rebuild_tree()
        self._later(self.build_details)

    def flip_contact(self):
        if self.current[0] != "contact":
            QMessageBox.information(self, "反転", "ツリーで接触領域を選んでから押してください。")
            return
        c = self.project.contacts[self.current[1]]
        c.contact_faces, c.target_faces = c.target_faces, c.contact_faces
        c.contact_body, c.target_body = c.target_body, c.contact_body
        self.invalidate()
        self.build_details()
        self.refresh_view()

    def _fields(self, r):
        out = dict(r.fields())
        for k, v in (getattr(r, "contact", None) or {}).items():
            out["接触:" + k] = v
        want = set(self.project.results)
        if any(k.startswith("疲労:") for k in want) and not getattr(r, "only_extra", False):
            try:
                from ..postproc import fatigue
                fo = (self.project.analysis_opts or {}).get("fatigue", {})
                mat = self.project.material
                if self.mesh is not None and self.mesh.elem_body is not None and self.project.materials:
                    names = {self.project.assignment(b).get("mat") for b in self.mesh.body_ids()}
                    names.discard(None)
                    if len(names) == 1:
                        mat = self.project.materials.get(names.pop(), mat)
                f = fatigue(r.stress, mat, loading=fo.get("loading", "reversed"), ratio=fo.get("ratio", -1.0),
                            scale=fo.get("scale", 1.0), mean_corr=fo.get("mean_corr", "none"),
                            comp=fo.get("comp", "signed_vm"), design_life=fo.get("design_life", 1e9))
                for k, v in f.items():
                    out["疲労:" + k] = v
            except ValueError as e:
                self._fatigue_error = str(e)
        for k in want:
            if k.startswith("式:"):
                try:
                    from ..postproc import user_result
                    out[k] = user_result(k[2:], out, self.mesh)
                except Exception as e:  # noqa: BLE001
                    self._expr_error = str(e)
        return out

    def _details_material(self):
        m = self.project.material
        s = self._section("割り当て")
        combo = QComboBox()
        combo.addItems(list(LIBRARY.keys()) + ["カスタム"])
        combo.setCurrentText(m.name if m.name in LIBRARY else "カスタム")
        combo.currentTextChanged.connect(self._material_pick)
        self._row(s, "材料", combo)
        s2 = self._section("特性")
        for attr, label, lo, hi, dec, step in [("E", "ヤング率 [MPa]", 1, 1e7, 1, 1000),
                                               ("nu", "ポアソン比", 0, 0.4999, 4, 0.01),
                                               ("density", "密度 [kg/m³]", 0, 1e5, 1, 10),
                                               ("yield_strength", "引張降伏応力 [MPa]", 0, 1e5, 1, 5),
                                               ("tensile_strength", "引張強さ [MPa]", 0, 1e5, 1, 5)]:
            sp = dspin(getattr(m, attr), lo, hi, dec, step)
            sp.valueChanged.connect(lambda v, attr=attr: self._material_edit(attr, v))
            self._row(s2, label, sp)
        note = self._section("メモ")
        self._row(note, "", "降伏応力を入れると安全率を表示できます")

    def _material_pick(self, name):
        if name in LIBRARY:
            self.project.material = Material(**LIBRARY[name].to_dict())
            self.invalidate()
            self._later(self.build_details)

    def _material_edit(self, attr, v):
        m = self.project.material
        setattr(m, attr, v)
        if m.name in LIBRARY and not m.name.endswith("（変更）"):
            m.name += "（変更）"
        self.invalidate()

    def _ms(self, key):
        return self.project.mesh_settings.get(key, MESH_DEFAULTS[key])

    def _ms_set(self, key, v, rebuild=False):
        self.project.mesh_settings[key] = v
        self.invalidate(mesh=True)
        if rebuild:
            self._later(self.build_details)

    def _details_mesh(self):
        """Ansys の「メッシュ」の詳細と同じ並び"""
        s = self._section("既定")
        self._row(s, "物理プリファレンス", "メカニカル")
        order = QComboBox()
        order.addItem("プログラム制御（2次）", 2)
        order.addItem("2次（Tet10）", 2)
        order.addItem("1次（Tet4）", 1)
        order.setCurrentIndex(0 if self.project.mesh_order == 2 else 2)
        order.currentIndexChanged.connect(lambda _: self._mesh_set("order", order.currentData()))
        self._row(s, "要素次数", order)
        size = ExprEdit(self.project.mesh_size_expr or self.project.mesh_size, suffix="mm（0 = 既定）")
        size.rawChanged.connect(lambda v: self._mesh_set("size", v))
        self._row(s, "要素サイズ", size)
        s = self._section("サイズ調整")
        res = QDoubleSpinBox()
        res.setDecimals(0)
        res.setRange(0, 7)
        res.setValue(self._ms("resolution"))
        res.setKeyboardTracking(False)
        res.valueChanged.connect(lambda v: self._ms_set("resolution", int(v)))
        self._row(s, "解像度（0 粗い 〜 7 細かい）", res)
        self._row(s, "要素の形", self._combo(
            [("四面体（Tet10）", "tet"), ("すべて六面体（Hex20、四面体を 4 分割）", "hex")],
            self._ms("element_shape") or "tet", lambda v: self._ms_set("element_shape", v)))
        gr = dspin(self._ms("growth_rate"), 1.01, 5.0, 2, 0.05)
        gr.valueChanged.connect(lambda v: self._ms_set("growth_rate", v))
        self._row(s, "成長率", gr)
        self._row(s, "曲率の捕捉", self._combo([("はい", True), ("いいえ", False)], self._ms("capture_curvature"),
                                           lambda v: self._ms_set("capture_curvature", v, True)))
        if self._ms("capture_curvature"):
            ang = dspin(self._ms("curvature_normal_angle"), 1.0, 90.0, 1, 5.0, "°")
            ang.valueChanged.connect(lambda v: self._ms_set("curvature_normal_angle", v))
            self._row(s, "曲率法線角度", ang)
            cmin = dspin(self._ms("curvature_min_size"), 0.0, 1e6, 3, 0.1, "mm")
            cmin.setSpecialValueText("既定（要素サイズの 5%）")
            cmin.valueChanged.connect(lambda v: self._ms_set("curvature_min_size", v))
            self._row(s, "曲率の最小サイズ", cmin)
        self._row(s, "近接の捕捉", self._combo([("はい", True), ("いいえ", False)], self._ms("capture_proximity"),
                                           lambda v: self._ms_set("capture_proximity", v, True)))
        if self._ms("capture_proximity"):
            nc = dspin(float(self._ms("proximity_cells")), 1, 10, 0, 1)
            nc.valueChanged.connect(lambda v: self._ms_set("proximity_cells", int(v)))
            self._row(s, "すき間を横切る要素数", nc)
            pm = dspin(self._ms("proximity_min_size"), 0.0, 1e6, 3, 0.1, "mm")
            pm.setSpecialValueText("既定（曲率の最小サイズ）")
            pm.valueChanged.connect(lambda v: self._ms_set("proximity_min_size", v))
            self._row(s, "近接の最小サイズ", pm)
            self._row(s, "", "薄肉や狭いすき間を見つけて、板厚方向にこの数の要素が入るよう細かくします（Ansys の近接の捕捉）")
        if self.mesh is not None:
            lo, hi = self.mesh.bounds()
            self._row(s, "バウンディングボックスの対角", f"{np.linalg.norm(hi - lo):.4g} mm")
        s = self._section("品質")
        self._row(s, "スムージング", self._combo([("低", "low"), ("中", "medium"), ("高", "high")],
                                            self._ms("smoothing"), lambda v: self._ms_set("smoothing", v)))
        self._row(s, "形状の最適化", self._combo([("はい", True), ("いいえ", False)], self._ms("optimize"),
                                             lambda v: self._ms_set("optimize", v)))
        self._row(s, "品質の自動修正", self._combo([("はい", True), ("いいえ", False)], self._ms("auto_fix"),
                                              lambda v: self._ms_set("auto_fix", v)))
        if self._ms("auto_fix"):
            mq = dspin(float(self._ms("min_quality")), 0.0, 0.9, 3, 0.01)
            mq.valueChanged.connect(lambda v: self._ms_set("min_quality", v))
            self._row(s, "目標の要素品質（下限）", mq)
            self._row(s, "", "下限より悪い要素を最適化で直し、裏返りそうな 2 次要素は中間節点を直線に戻します")
        b = QPushButton("メッシュを生成")
        b.clicked.connect(lambda: self.remesh())
        self._row(self._section("操作"), "", b)
        if self.mesh is not None:
            s2 = self._section("統計")
            self._row(s2, "節点数", f"{self.mesh.n_nodes:,}")
            self._row(s2, "要素数", f"{self.mesh.n_elems:,}")
            if getattr(self.mesh, "is_2d", False):
                self._row(s2, "要素", "2次三角形（6節点, PLANE183 相当）")
            elif getattr(self.mesh, "is_shell", False):
                self._row(s2, "要素", "シェル（3節点, SHELL181 相当）")
            else:
                self._row(s2, "要素", "2次四面体（Tet10, SOLID187 相当）" if self.mesh.order == 2 else "1次四面体（Tet4）")
                q = getattr(self, "_mesh_q", None)
                if q is None or q[0] is not self.mesh:
                    q = (self.mesh, mesh_quality(self.mesh))
                    self._mesh_q = q
                for key in ("要素品質", "歪度", "アスペクト比"):
                    v = q[1][key]
                    self._row(s2, f"{key}（最小/平均/最大）", f"{v['最小']:.3f} / {v['平均']:.3f} / {v['最大']:.3f}")
            for msg in getattr(self.mesh, "fix_log", []) or []:
                self._row(s2, "自動修正", msg)
            self._row(s2, "状態", "最新" if not self.mesh_dirty else "再生成が必要（⚡）")

    def _mesh_set(self, what, v):
        if what == "order":
            self.project.mesh_order = v
        elif isinstance(v, str):
            try:
                self.project.mesh_size = max(0.0, evaluate(v, self.project.env()))
            except ParamError as e:
                QMessageBox.warning(self, "メッシュサイズ", str(e))
                return
            self.project.mesh_size_expr = v
        else:
            self.project.mesh_size = max(0.0, float(v))
            self.project.mesh_size_expr = ""
        self.invalidate(mesh=True)

    # ------------------------------------------------------------ メッシュ制御（Ansys のメッシュ制御）
    @staticmethod
    def _control_menu_items():
        return [("face_sizing", "サイズ指定（面）"), ("body_sizing", "サイズ指定（ボディ）"),
                ("sphere", "影響球サイズ"), ("refinement", "リファインメント（面）"), ("sweep", "手法：スイープ")]

    def _control_label(self, i, ls):
        t = ls.get("type", "sizing")
        if t == "sizing":
            sc = "ボディ" if ls.get("scope") == "bodies" else "面"
            return f"サイズ指定（{sc}） {i + 1}: {float(ls.get('size', 0)):g} mm"
        if t == "sphere":
            return f"影響球サイズ {i + 1}: {float(ls.get('size', 0)):g} mm"
        if t == "refinement":
            return f"リファインメント {i + 1}: レベル {int(ls.get('level', 1))}"
        if t == "sweep":
            return f"手法：スイープ {i + 1}" + (f"（{int(ls['divs'])} 分割）" if ls.get("divs") else "")
        return f"メッシュ制御 {i + 1}"

    def add_control(self, key):
        if key == "face_sizing":
            ls = {"type": "sizing", "scope": "faces", "faces": list(self.selected_faces), "size": 1.0,
                  "behavior": "soft"}
        elif key == "body_sizing":
            ls = {"type": "sizing", "scope": "bodies", "bodies": list(self.selected_bodies), "size": 2.0}
        elif key == "sphere":
            c = [0.0, 0.0, 0.0]
            if self.mesh is not None and self.selected_faces:
                info = {f["id"]: f for f in self.mesh.face_info()}
                c = np.mean([info[f]["centroid"] for f in self.selected_faces], axis=0).tolist()
            ls = {"type": "sphere", "center": c, "radius": 10.0, "size": 1.0}
        elif key == "refinement":
            ls = {"type": "refinement", "faces": list(self.selected_faces), "level": 1}
        else:
            ls = {"type": "sweep", "divs": 0, "axis": "auto"}
        self.project.local_sizes.append(ls)
        self.invalidate(mesh=True)
        self.select_node(("sizing", len(self.project.local_sizes) - 1))

    def add_sizing(self):
        self.add_control("face_sizing")

    def _ls_set(self, i, key, v, rebuild=False):
        self.project.local_sizes[i][key] = v
        self.invalidate(mesh=True)
        if rebuild:
            self._later(lambda: (self.rebuild_tree(), self.build_details()))

    def _details_sizing(self):
        i = self.current[1]
        ls = self.project.local_sizes[i]
        t = ls.get("type", "sizing")
        s = self._section("スコープ")
        if t == "sizing" and ls.get("scope") == "bodies":
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            bl = ls.get("bodies", [])
            lab = QLabel(", ".join(f"B{b}" for b in bl) if bl else "未選択")
            if not bl:
                lab.setStyleSheet("background:#fff3b0; padding:1px 4px;")
            h.addWidget(lab)
            ap = QPushButton("適用")

            def _apply_b():
                if not self.selected_bodies:
                    QMessageBox.information(self, "ボディを選択", "ボディ選択モードでボディを選んでから「適用」を押してください。")
                    return
                self._ls_set(i, "bodies", list(self.selected_bodies), True)
            ap.clicked.connect(_apply_b)
            h.addWidget(ap)
            h.addStretch()
            self._row(s, "スコープ方法", "ジオメトリの選択")
            self._row(s, "ジオメトリ", w)
        elif t in ("sizing", "refinement"):
            self._scope_row(s, ls.get("faces", []), lambda faces: self._ls_set(i, "faces", list(faces), True))
        elif t == "sweep":
            self._row(s, "ジオメトリ", "すべてのボディ（積層品は多層スイープ）")
        else:
            self._row(s, "ジオメトリ", "影響球の中の領域")
        s2 = self._section("定義")
        sup = QCheckBox("抑制")
        sup.setChecked(bool(ls.get("suppressed")))
        sup.toggled.connect(lambda on: self._ls_set(i, "suppressed", on, True))
        self._row(s2, "抑制", sup)
        if t in ("sizing", "sphere"):
            sp = dspin(float(ls.get("size", 1.0)), 0.001, 1e6, 3, 0.1, "mm")
            sp.valueChanged.connect(lambda v: self._ls_set(i, "size", v))
            self._row(s2, "要素サイズ", sp)
        if t == "sizing" and ls.get("scope") != "bodies":
            self._row(s2, "挙動", self._combo([("ソフト", "soft"), ("ハード", "hard")], ls.get("behavior", "soft"),
                                             lambda v: self._ls_set(i, "behavior", v)))
        if t == "sphere":
            for k, ax in enumerate("XYZ"):
                c = dspin(float(ls["center"][k]), -1e7, 1e7, 3, 1.0, "mm")

                def setc(v, k=k):
                    ls["center"][k] = v
                    self.invalidate(mesh=True)
                c.valueChanged.connect(setc)
                self._row(s2, f"中心 {ax}", c)
            r = dspin(float(ls.get("radius", 10)), 0.001, 1e7, 3, 1.0, "mm")
            r.valueChanged.connect(lambda v: self._ls_set(i, "radius", v))
            self._row(s2, "球の半径", r)
        if t == "refinement":
            lv = QDoubleSpinBox()
            lv.setDecimals(0)
            lv.setRange(1, 3)
            lv.setValue(int(ls.get("level", 1)))
            lv.setKeyboardTracking(False)
            lv.valueChanged.connect(lambda v: self._ls_set(i, "level", int(v), True))
            self._row(s2, "リファインメント（1〜3）", lv)
        if t == "sweep":
            self._row(s2, "手法", "スイープ")
            self._row(s2, "要素次数", "メッシュの設定に従う")
            self._row(s2, "ソース/ターゲットの選択", self._combo(
                [("自動（薄い方向）", "auto"), ("X 方向", "X"), ("Y 方向", "Y"), ("Z 方向", "Z")],
                ls.get("axis", "auto"), lambda v: self._ls_set(i, "axis", v)))
            dv = QDoubleSpinBox()
            dv.setDecimals(0)
            dv.setRange(0, 1000)
            dv.setSpecialValueText("自動（要素サイズから）")
            dv.setValue(int(ls.get("divs", 0)))
            dv.setKeyboardTracking(False)
            dv.valueChanged.connect(lambda v: self._ls_set(i, "divs", int(v), True))
            self._row(s2, "スイープ分割数（層ごと）", dv)
            self._row(s2, "要素の形", self._combo(
                [("三角柱（Wedge15、薄い層向き）", "wedge"), ("六面体（Hex20、断面を四角形で切る）", "hex"),
                 ("四面体（三角柱を 3 分割）", "tet")],
                ls.get("element", "wedge"), lambda v: self._ls_set(i, "element", v, True)))
            ly = dspin(float(ls.get("layer_size", 0) or 0), 0.0, 1e6, 3, 0.1, "mm")
            ly.setSpecialValueText("既定（要素サイズ）")
            ly.valueChanged.connect(lambda v: self._ls_set(i, "layer_size", v))
            self._row(s2, "厚さ方向の要素サイズ", ly)
            self._row(s2, "", "断面（スイープ方向に垂直な面）を 1 枚の平面に重ねて三角形で切り、断面が変わる高さごとの"
                      "層に押し出します。ボディ同士の境目は節点がそろいます。丸み付け・球など断面が一定でない形は"
                      "スイープできません（四面体＋近接の捕捉を使ってください）。")

    def _scope_row(self, sec, faces, apply_fn):
        """Ansys と同じ「ジオメトリ: n 面 [適用] [キャンセル]」"""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(f"{len(faces)} 面" if faces else "未選択")
        if not faces:
            lab.setStyleSheet("background:#fff3b0; padding:1px 4px;")
        h.addWidget(lab)
        b = QPushButton("適用")
        b.setToolTip("グラフィックスで選択中の面をこの項目に割り当てます")

        def _apply():
            if not self.selected_faces:
                QMessageBox.information(self, "面を選択", "グラフィックスで面を左クリックして選択してから「適用」を押してください。")
                return
            apply_fn(list(self.selected_faces))
            self._later(lambda: (self.build_details(), self.refresh_view()))
        b.clicked.connect(_apply)
        h.addWidget(b)
        h.addStretch()
        self._row(sec, "スコープ方法", "ジオメトリの選択")
        self._row(sec, "ジオメトリ", w)

    def _analysis_label(self):
        t = self.project.analysis_type
        if t == "thermal":
            return "過渡熱" if (self.project.analysis_opts or {}).get("thermal", {}).get("transient") else "定常熱"
        if t in ANALYSIS_JP and t != "transient":
            return ANALYSIS_JP[t]
        return "過渡構造（陽解法）" if (self.project.dynamic or {}).get("method") == "explicit" else "過渡構造（陰解法）"

    def _set_analysis_type(self, key):
        p = self.project
        if key in ("static",) + EXTRA_TYPES or key == "thermal_t":
            p.analysis_type = "thermal" if key == "thermal_t" else key
            if key in ("thermal", "thermal_t"):
                o = dict(p.analysis_opts or {})
                o["thermal"] = dict(o.get("thermal", {}), transient=(key == "thermal_t"))
                p.analysis_opts = o
            if p.analysis_type in DEFAULT_RESULTS:
                p.results = list(DEFAULT_RESULTS[p.analysis_type])
            elif not any(r in ("全変形 [mm]", "相当応力（ミーゼス） [MPa]") for r in p.results):
                p.results = ["全変形 [mm]", "相当応力（ミーゼス） [MPa]"]
            self.invalidate()
            self.rebuild_tree()
            self._later(self.build_details)
            return
        else:
            p.analysis_type = "transient"
            d = dict(p.dynamic or {})
            d["method"] = key
            if key == "explicit" and not d.get("end_time"):
                d["end_time"] = 1e-3
            p.dynamic = d
        self.invalidate()
        self.rebuild_tree()
        self._later(self.build_details)

    def _details_analysis(self):
        s = self._section("定義")
        p = self.project
        t = p.analysis_type
        cur = ((p.dynamic or {}).get("method", "implicit") if t == "transient" else
               "thermal_t" if (t == "thermal" and (p.analysis_opts or {}).get("thermal", {}).get("transient")) else t)
        self._row(s, "解析タイプ", self._combo(
            [("静的構造", "static"), ("過渡構造（陰解法・Newmark / HHT）", "implicit"),
             ("過渡構造（陽解法・LS-DYNA 方式）", "explicit"), ("モーダル（固有振動数）", "modal"),
             ("固有値座屈", "buckling"), ("定常熱", "thermal"), ("過渡熱", "thermal_t"),
             ("調和応答（モード重ね合わせ）", "harmonic"), ("ランダム振動（PSD）", "random"),
             ("応答スペクトル", "spectrum"),
             ("静電場", "electrostatic"), ("圧電（静解析）", "piezo")], cur, self._set_analysis_type))
        if t in EXTRA_TYPES or t == "static":
            self._details_analysis_opts(s)
        if self.project.analysis_type == "transient":
            self._row(s, "", "時間・初速度・剛体壁などは「解析設定」で設定します。荷重（力・圧力・重力）は t=0 から"
                      "（またはランプ時間で）かかります。")
        self._row(s, "物理タイプ", "構造")
        self._row(s, "支持", "あり" if self._has_support() else "なし（固定支持を追加してください）")
        self._row(s, "荷重・支持の数", len(self.project.bcs))
        s2 = self._section("オプション")
        et = dspin(self.project.env_temp, -273, 1e5, 2, 1.0, "°C")

        def set_env(v):
            self.project.env_temp = v
            self.invalidate()
        et.valueChanged.connect(set_env)
        self._row(s2, "環境温度", et)

    def _opt(self, key, value, sub=None, rebuild=False):
        o = dict(self.project.analysis_opts or {})
        if sub:
            o[sub] = dict(o.get(sub, {}), **{key: value})
        else:
            o[key] = value
        self.project.analysis_opts = o
        self.invalidate()
        if rebuild:
            self._later(self.build_details)

    def _details_analysis_opts(self, s):
        """解析の種類ごとの設定（Ansys の解析設定に相当）"""
        p = self.project
        o = p.analysis_opts or {}
        t = p.analysis_type

        def ispin(key, val, lo, hi, label, sub=None):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(int(val))
            w.valueChanged.connect(lambda v: self._opt(key, int(v), sub))
            self._row(s, label, w)

        def fspin(key, val, lo, hi, dec, step, unit, label, sub=None):
            w = dspin(float(val), lo, hi, dec, step, unit)
            w.valueChanged.connect(lambda v: self._opt(key, float(v), sub))
            self._row(s, label, w)

        if t == "static":
            chk = QCheckBox("熱条件（温度・対流など）で定常熱解析を解き、その温度で熱応力を計算する")
            chk.setChecked(bool(o.get("thermal_import")))
            chk.toggled.connect(lambda v: self._opt("thermal_import", bool(v)))
            self._row(s, "熱応力の連成", chk)
        if t in ("modal", "harmonic", "random", "spectrum"):
            ispin("n_modes", o.get("n_modes", 6 if t == "modal" else 20), 1, 500, "モード数")
        if t == "modal":
            fspin("f_min", o.get("f_min", 0.0), 0, 1e9, 3, 10, "Hz", "探す範囲の下限")
            chk = QCheckBox("静解析の応力を含める（プリストレス）")
            chk.setChecked(bool(o.get("prestress")))
            chk.toggled.connect(lambda v: self._opt("prestress", bool(v)))
            self._row(s, "プリストレス", chk)
            self._row(s, "", "結果の「表示時間」でモードを切り替えます。変形・応力の大きさは相対値（質量正規化）")
        if t == "buckling":
            ispin("n_buckling", o.get("n_buckling", 3), 1, 50, "モード数")
            self._row(s, "", "与えた荷重で静解析 → 座屈荷重係数（その何倍で座屈するか）")
        if t == "thermal":
            th = o.get("thermal", {})
            fspin("init_temp", th.get("init_temp", 22.0), -273, 1e5, 2, 1, "°C", "初期温度", "thermal")
            if th.get("transient"):
                fspin("end_time", th.get("end_time", 1.0), 1e-9, 1e9, 4, 1, "s", "終了時間", "thermal")
                ispin("n_steps", th.get("n_steps", 10), 1, 100000, "ステップ数", "thermal")
            self._row(s, "", "材料の熱伝導率（過渡は密度・比熱も）が必要です。荷重は［環境］→［熱］から")
        if t in ("harmonic", "random"):
            fspin("zeta", o.get("zeta", 0.02), 0, 1, 4, 0.005, "", "減衰比")
        if t == "spectrum":
            spec = o.get("spectrum") or [[1, 0.5], [10, 1.0], [100, 1.0], [1000, 0.5]]
            ed2 = QLineEdit("; ".join(f"{a:g}, {b:g}" for a, b in spec))
            ed2.setToolTip("周波数 Hz, 加速度 を ; で区切る（例: 1, 0.5; 10, 1; 100, 1; 1000, 0.5）。範囲外は端の値")

            def set_spec():
                try:
                    rows = [[float(x) for x in r.split(",")] for r in ed2.text().split(";") if r.strip()]
                    assert len(rows) >= 2 and all(len(r) == 2 for r in rows)
                    self._opt("spectrum", rows)
                except Exception:  # noqa: BLE001
                    QMessageBox.warning(self, "応答スペクトル", "「周波数, 値; 周波数, 値; …」の形で入力してください")
            ed2.editingFinished.connect(set_spec)
            self._row(s, "加速度スペクトル [Hz, 加速度]", ed2)
            self._row(s, "加速度の単位", self._combo([("G", "G"), ("mm/s²", "mm/s2")], o.get("spec_unit", "G"),
                                               lambda v: self._opt("spec_unit", v)))
            d = list(o.get("spec_dir", [0, 0, 1]))
            self._row(s, "入力の方向", self._combo([("X", "X"), ("Y", "Y"), ("Z", "Z")], "XYZ"[int(np.argmax(np.abs(d)))],
                                               lambda v: self._opt("spec_dir", [float(v == a) for a in "XYZ"])))
            self._row(s, "モードの組み合わせ", self._combo([("SRSS", "srss"), ("CQC", "cqc")], o.get("spec_method", "srss"),
                                                    lambda v: self._opt("spec_method", v)))
            fspin("zeta", o.get("zeta", 0.05), 0, 1, 4, 0.005, "", "減衰比（CQC）")
            self._row(s, "", "モードの応答を組み合わせた最大値（符号なし）です。入力方向の有効質量の比が 90 % 以上に"
                             "なるようにモード数を増やしてください（結果の情報に出ます）")
        if t == "harmonic":
            fspin("f_min_h", o.get("f_min_h", 1.0), 0, 1e9, 3, 10, "Hz", "周波数の下限")
            fspin("f_max_h", o.get("f_max_h", 1000.0), 0, 1e9, 3, 10, "Hz", "周波数の上限")
            ispin("n_freq", o.get("n_freq", 100), 2, 100000, "周波数の点数")
            ba = list(o.get("base_acc") or [0.0, 0.0, 0.0])
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            sps = []
            for k in range(3):
                sp = dspin(float(ba[k]), -1e6, 1e6, 4, 0.1)
                sps.append(sp)
                h.addWidget(sp)
                sp.valueChanged.connect(lambda _=0: self._opt("base_acc", [x.value() for x in sps]))
            self._row(s, "基礎加振 X, Y, Z [G]", w)
            self._row(s, "", "基礎加振が 0 なら、荷重（力・圧力・重力）を振幅として使います。支持は固定")
        if t == "random":
            psd = o.get("psd") or [[20, 0.01], [2000, 0.01]]
            ed = QLineEdit("; ".join(f"{a:g}, {b:g}" for a, b in psd))
            ed.setToolTip("周波数 Hz, PSD G²/Hz を ; で区切る（例: 20, 0.01; 80, 0.04; 350, 0.04; 2000, 0.007）")

            def set_psd():
                try:
                    rows = [[float(x) for x in r.split(",")] for r in ed.text().split(";") if r.strip()]
                    assert len(rows) >= 2 and all(len(r) == 2 for r in rows)
                    self._opt("psd", rows)
                except Exception:  # noqa: BLE001
                    QMessageBox.warning(self, "PSD", "「周波数, 値; 周波数, 値; …」の形で入力してください")
            ed.editingFinished.connect(set_psd)
            self._row(s, "加速度 PSD [Hz, G²/Hz]", ed)
            d = list(o.get("psd_dir", [0, 0, 1]))
            self._row(s, "入力の方向", self._combo([("X", "X"), ("Y", "Y"), ("Z", "Z")], "XYZ"[int(np.argmax(np.abs(d)))],
                                               lambda v: self._opt("psd_dir", [float(v == a) for a in "XYZ"])))
        if t == "electrostatic":
            self._row(s, "", "材料の比誘電率を使います。電極は［環境］→［電気］の電圧・接地")
        if t == "piezo":
            self._row(s, "", "圧電材料（材料の piezo: 圧電定数 e・誘電率）と電極（電圧・接地）、支持が必要です")

    def _combo(self, items, current, on_change):
        c = QComboBox()
        for label, key in items:
            c.addItem(label, key)
        c.setCurrentIndex(max(0, c.findData(current)))
        c.currentIndexChanged.connect(lambda _: on_change(c.currentData()))
        return c

    def _set_setting(self, key, value, rebuild=False):
        setattr(self.project.settings, key, value)
        self.invalidate()
        if rebuild:
            self._later(self.build_details)

    def _dyn(self):
        from ..dynamics import DynamicSettings
        return DynamicSettings.from_dict(self.project.dynamic)

    def _dyn_set(self, key, value, rebuild=False):
        d = self._dyn().to_dict()
        d[key] = value
        self.project.dynamic = d
        self.invalidate()
        if rebuild:
            self._later(self.build_details)

    def _details_dynamic(self):
        """過渡構造（陰解法 / 陽解法）の解析設定"""
        ds = self._dyn()
        ex = ds.method == "explicit"
        s = self._section("時間")
        et = dspin(ds.end_time, 1e-12, 1e6, 9, 1e-4, "s")
        et.valueChanged.connect(lambda v: self._dyn_set("end_time", v))
        self._row(s, "終了時間", et)
        dt = dspin(ds.dt, 0, 1e3, 12, 1e-7, "s")
        dt.setToolTip("0 = 自動（陽解法は安定限界×安全係数、陰解法は終了時間÷(出力数×5)）")
        dt.valueChanged.connect(lambda v: self._dyn_set("dt", v))
        self._row(s, "時間増分（0 = 自動）", dt)
        nf = dspin(ds.n_frames, 1, 1000, 0, 1)
        nf.valueChanged.connect(lambda v: self._dyn_set("n_frames", int(v)))
        self._row(s, "結果を保存する時刻の数", nf)
        if ex:
            sf = dspin(ds.dt_safety, 0.1, 1.0, 2, 0.05)
            sf.valueChanged.connect(lambda v: self._dyn_set("dt_safety", v))
            self._row(s, "時間増分の安全係数（TSSFAC）", sf)
            ms = dspin(ds.mass_scaling_dt, 0, 1, 12, 1e-8, "s")
            ms.setToolTip("この時間増分より小さい要素だけ質量を増やして計算を速くします（LS-DYNA の DT2MS）。0 = なし")
            ms.valueChanged.connect(lambda v: self._dyn_set("mass_scaling_dt", v))
            self._row(s, "質量スケーリングの目標時間増分", ms)
            self._row(s, "大変形", self._combo([("オフ", False), ("オン（Total Lagrangian）", True)],
                                            ds.large_deflection, lambda v: self._dyn_set("large_deflection", v)))
            self._row(s, "", "材料に塑性（2直線・多直線）を設定すると、陽解法では自動で塑性を考慮します")
        else:
            ha = dspin(ds.hht_alpha, -1 / 3, 0, 3, 0.05)
            ha.setToolTip("0 = Newmark 平均加速度法（エネルギー保存）。負にすると高い周波数の振動を減衰")
            ha.valueChanged.connect(lambda v: self._dyn_set("hht_alpha", v))
            self._row(s, "HHT-α（数値減衰）", ha)
            self._row(s, "大変形", self._combo([("オフ", False), ("オン", True)],
                                            ds.large_deflection, lambda v: self._dyn_set("large_deflection", v)))
            self._row(s, "", "接触・塑性・超弾性・大変形・リモート境界条件・ジョイントがあると、静解析と同じ非線形の"
                             "解き方に慣性力を加えて解きます（Ansys の過渡構造と同じ）。接触があるときは数値減衰 0.1 が既定です")
        s2 = self._section("初期条件・荷重")
        for i, ax in enumerate("XYZ"):
            w = dspin(float(ds.init_velocity[i]), -1e9, 1e9, 3, 100.0, "mm/s")
            w.valueChanged.connect(lambda v, i=i: self._dyn_set("init_velocity",
                                                                [v if j == i else float(self._dyn().init_velocity[j])
                                                                 for j in range(3)]))
            self._row(s2, f"初速度 {ax}", w)
        if not ex and self.mesh is not None and len(self.mesh.body_ids()) > 1:
            # ボディごとの初速度（衝突の解析用）。空欄 = 上の初速度を使う
            for bid in self.mesh.body_ids():
                cur = (ds.body_velocity or {}).get(str(int(bid)), (ds.body_velocity or {}).get(int(bid)))
                ed = QLineEdit("" if cur is None else ", ".join(f"{float(x):g}" for x in cur))
                ed.setPlaceholderText("vx, vy, vz [mm/s]（空欄 = 上の初速度）")

                def set_bv(bid=bid, ed=ed):
                    bv = {str(k): v for k, v in (self._dyn().body_velocity or {}).items()}
                    txt = ed.text().replace("、", ",").strip()
                    if not txt:
                        bv.pop(str(int(bid)), None)
                    else:
                        try:
                            vals = [float(x) for x in txt.split(",")]
                            assert len(vals) == 3
                        except Exception:
                            QMessageBox.warning(self, "初速度", "vx, vy, vz の 3 つの数値をカンマ区切りで入力してください")
                            return
                        bv[str(int(bid))] = vals
                    self._dyn_set("body_velocity", bv)
                ed.editingFinished.connect(set_bv)
                self._row(s2, f"初速度 {self._body_label(bid)}", ed)
        rt = dspin(ds.ramp_time, 0, 1e6, 9, 1e-4, "s")
        rt.setToolTip("荷重を 0 からこの時間で直線的に増やします。0 = t=0 から一定（ステップ荷重）")
        rt.valueChanged.connect(lambda v: self._dyn_set("ramp_time", v))
        self._row(s2, "荷重のランプ時間", rt)
        s3 = self._section("減衰（レイリー）")
        a = dspin(ds.alpha_damping, 0, 1e9, 4, 10.0, "1/s")
        a.valueChanged.connect(lambda v: self._dyn_set("alpha_damping", v))
        self._row(s3, "質量比例 α", a)
        b = dspin(ds.beta_damping, 0, 1, 10, 1e-7, "s")
        b.valueChanged.connect(lambda v: self._dyn_set("beta_damping", v))
        self._row(s3, "剛性比例 β", b)
        s4 = self._section("剛体壁（床）")
        w = dict(ds.wall or {})
        on = bool(w.get("normal"))
        dirs = [("なし", ""), ("床が −Z 側（物体は +Z 側）", "+Z"), ("床が +Z 側", "-Z"), ("床が −Y 側", "+Y"),
                ("床が +Y 側", "-Y"), ("床が −X 側", "+X"), ("床が +X 側", "-X")]
        cur = ""
        if on:
            n = np.asarray(w["normal"], float)
            k = int(np.argmax(np.abs(n)))
            cur = ("+" if n[k] > 0 else "-") + "XYZ"[k] if abs(abs(n[k]) - 1) < 1e-9 else "custom"
            if cur == "custom":
                dirs.append((f"斜め ({n[0]:.3g}, {n[1]:.3g}, {n[2]:.3g})", "custom"))

        def set_dir(v):
            if not v:
                self._dyn_set("wall", {}, rebuild=True)
                return
            if v == "custom":
                return
            n = [0.0, 0.0, 0.0]
            n["XYZ".index(v[1])] = 1.0 if v[0] == "+" else -1.0
            ww = dict(self._dyn().wall or {})
            ww["normal"] = n
            ww.setdefault("point", [0.0, 0.0, 0.0])
            if self.mesh is not None:
                k = "XYZ".index(v[1])
                lo, hi = self.mesh.bounds()
                p = [0.0, 0.0, 0.0]
                p[k] = float(lo[k] if v[0] == "+" else hi[k])
                ww["point"] = p
            self._dyn_set("wall", ww, rebuild=True)
        self._row(s4, "剛体壁", self._combo(dirs, cur, set_dir))
        if on:
            for i, ax in enumerate("XYZ"):
                pw = dspin(float(w.get("point", [0, 0, 0])[i]), -1e9, 1e9, 4, 1.0, "mm")

                def set_p(v, i=i):
                    ww = dict(self._dyn().wall)
                    pp = list(ww.get("point", [0, 0, 0]))
                    pp[i] = v
                    ww["point"] = pp
                    self._dyn_set("wall", ww)
                pw.valueChanged.connect(set_p)
                self._row(s4, f"壁の通る点 {ax}", pw)
            fr = dspin(float(w.get("friction", 0.0)), 0, 10, 3, 0.05)

            def set_mu(v):
                ww = dict(self._dyn().wall)
                ww["friction"] = v
                self._dyn_set("wall", ww)
            fr.valueChanged.connect(set_mu)
            self._row(s4, "摩擦係数", fr)
        s5 = self._section("落下試験（かんたん設定）")
        hb = QPushButton("落下の高さと向きから設定…")
        hb.clicked.connect(lambda: self._later(self.setup_drop))
        self._row(s5, "", hb)
        self._row(s5, "", "高さから初速度 √(2gh) を求め、物体がちょうど床に触れる位置に剛体壁を置きます。"
                  "落ちる向き（斜めも可）を指定できます。")

    def setup_drop(self):
        """落下試験の設定: 高さ・落ちる向き → 初速度と剛体壁"""
        if self.mesh is None:
            QMessageBox.information(self, "落下試験", "先にメッシュを作成してください。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("落下試験の設定")
        lay = QGridLayout(dlg)
        h = dspin(1000.0, 0.001, 1e6, 1, 100.0, "mm")
        dx, dy, dz = dspin(0.0, -1, 1, 4, 0.1), dspin(0.0, -1, 1, 4, 0.1), dspin(-1.0, -1, 1, 4, 0.1)
        mu = dspin(0.2, 0, 10, 2, 0.05)
        dur = dspin(5e-4, 1e-7, 1, 7, 1e-4, "s")
        rows = [("落下の高さ", h), ("落ちる向き X", dx), ("落ちる向き Y", dy), ("落ちる向き Z", dz),
                ("床の摩擦係数", mu), ("解析時間（衝突後）", dur)]
        for i, (lbl, w) in enumerate(rows):
            lay.addWidget(QLabel(lbl), i, 0)
            lay.addWidget(w, i, 1)
        note = QLabel("例: 角から落とすなら向きを (−1, −1, 0) のように斜めに。床は向きに垂直な面で、"
                      "物体の一番先の点に接するよう置きます。陽解法（LS-DYNA 方式）をおすすめします。")
        note.setWordWrap(True)
        lay.addWidget(note, len(rows), 0, 1, 2)
        bb = QPushButton("設定する")
        bb.clicked.connect(dlg.accept)
        lay.addWidget(bb, len(rows) + 1, 1)
        if not dlg.exec():
            return
        d = np.array([dx.value(), dy.value(), dz.value()], float)
        if np.linalg.norm(d) < 1e-9:
            return
        d /= np.linalg.norm(d)
        v0 = math.sqrt(2 * 9806.65 * h.value())
        far = self.mesh.nodes[int(np.argmax(self.mesh.nodes @ d))]
        p = self.project
        if p.analysis_type != "transient":
            p.analysis_type = "transient"
        dd = self._dyn().to_dict()
        if not p.dynamic.get("method"):
            dd["method"] = "explicit"
        dd.update(init_velocity=list(map(float, v0 * d)), end_time=float(dur.value()),
                  wall={"normal": list(map(float, -d)), "point": list(map(float, far)), "friction": float(mu.value())})
        p.dynamic = dd
        self.log(f"落下試験: 高さ {h.value():g} mm → 衝突速度 {v0:,.0f} mm/s（{v0 / 1000:.2f} m/s）、"
                 f"向き ({d[0]:.3g}, {d[1]:.3g}, {d[2]:.3g})")
        self.invalidate()
        self.rebuild_tree()
        self._later(self.build_details)

    def _details_settings(self):
        """Ansys の「解析設定」と同じ項目"""
        if self.project.analysis_type == "transient":
            self._details_dynamic()
            return
        st = self.project.settings
        # ステップ制御
        s = self._section("ステップ制御")
        ns = QDoubleSpinBox()
        ns.setDecimals(0)
        ns.setRange(1, 100)
        ns.setValue(st.n_steps)
        ns.setKeyboardTracking(False)

        def set_ns(v):
            st.n_steps = int(v)
            st.step_end_times = st.end_times()
            self.invalidate()
            self._later(self.build_details)
        ns.valueChanged.connect(set_ns)
        self._row(s, "ステップ数", ns)
        cur = getattr(self, "_cur_step", 1)
        cur = min(max(1, cur), st.n_steps)
        cs_ = QDoubleSpinBox()
        cs_.setDecimals(0)
        cs_.setRange(1, st.n_steps)
        cs_.setValue(cur)
        cs_.setKeyboardTracking(False)
        cs_.valueChanged.connect(lambda v: (setattr(self, "_cur_step", int(v)), self._later(self.build_details)))
        self._row(s, "現在のステップ番号", cs_)
        ends = st.end_times()
        et = dspin(ends[cur - 1], 1e-9, 1e9, 4, 1.0, "s")

        def set_end(v, k=cur - 1):
            e = st.end_times()
            e[k] = v
            st.step_end_times = e
            self.invalidate()
        et.valueChanged.connect(set_end)
        self._row(s, "ステップ終了時間", et)
        self._row(s, "自動時間ステップ", self._combo([("プログラム制御", "auto"), ("オン", "on"), ("オフ", "off")],
                                              st.auto_time_stepping, lambda v: self._set_setting("auto_time_stepping", v)))
        self._row(s, "定義方法", "サブステップ")
        for key, label in [("initial_substeps", "初期サブステップ"), ("min_substeps", "最小サブステップ"),
                           ("max_substeps", "最大サブステップ")]:
            sp = QDoubleSpinBox()
            sp.setDecimals(0)
            sp.setRange(1, 10000)
            sp.setValue(getattr(st, key))
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v, key=key: self._set_setting(key, int(v)))
            self._row(s, label, sp)
        # ソルバー制御
        s = self._section("ソルバー制御")
        from ..parallel import available_cores
        nc = QDoubleSpinBox()
        nc.setDecimals(0)
        nc.setRange(0, available_cores())
        nc.setSpecialValueText(f"すべて（{available_cores()} コア）")
        nc.setValue(st.n_cores)
        nc.setKeyboardTracking(False)
        nc.valueChanged.connect(lambda v: self._set_setting("n_cores", int(v)))
        self._row(s, "使用コア数", nc)
        self._row(s, "ソルバータイプ", self._combo([("プログラム制御", "auto"), ("直接法", "direct"), ("反復法", "amg")],
                                             st.solver_type, lambda v: self._set_setting("solver_type", v)))
        self._row(s, "弱いバネ", self._combo([("オフ", False), ("オン", True)], st.weak_springs,
                                         lambda v: self._set_setting("weak_springs", v)))
        self._row(s, "大変形", self._combo([("オフ", False), ("オン", True)], st.large_deflection,
                                        lambda v: self._set_setting("large_deflection", v)))
        self._row(s, "慣性リリーフ", "オフ")
        # 非線形制御
        s = self._section("非線形制御")
        self._row(s, "ニュートン・ラフソン オプション", self._combo([("プログラム制御", "auto"), ("フル", "full")],
                                                           st.newton, lambda v: self._set_setting("newton", v)))
        fc = dspin(st.force_convergence * 100, 0.001, 100, 3, 0.1, "%")
        fc.valueChanged.connect(lambda v: self._set_setting("force_convergence", v / 100))
        self._row(s, "力の収束（許容値）", fc)
        dc = dspin(st.displacement_convergence * 100, 0.0, 100, 3, 0.1, "%")
        dc.setSpecialValueText("除外")
        dc.valueChanged.connect(lambda v: self._set_setting("displacement_convergence", v / 100))
        self._row(s, "変位の収束（許容値）", dc)
        self._row(s, "ラインサーチ", self._combo([("プログラム制御", "auto"), ("オン", "on"), ("オフ", "off")],
                                           st.line_search, lambda v: self._set_setting("line_search", v)))
        mi = QDoubleSpinBox()
        mi.setDecimals(0)
        mi.setRange(1, 1000)
        mi.setValue(st.max_iterations)
        mi.setKeyboardTracking(False)
        mi.valueChanged.connect(lambda v: self._set_setting("max_iterations", int(v)))
        self._row(s, "最大平衡反復回数", mi)
        pt = dspin(st.penetration_tolerance, 0.001, 10, 3, 0.05)
        pt.valueChanged.connect(lambda v: self._set_setting("penetration_tolerance", v))
        self._row(s, "貫通許容係数（要素サイズ比）", pt)
        self._row(s, "安定化", "オフ")
        # 出力制御
        s = self._section("出力制御")
        self._row(s, "応力", "はい")
        self._row(s, "ひずみ", "はい")
        self._row(s, "接触の結果", self._combo([("はい", True), ("いいえ", False)], st.calc_contact,
                                           lambda v: self._set_setting("calc_contact", v)))
        self._row(s, "結果の保存", self._combo([("すべてのステップ", "all"), ("最後のステップのみ", "last")],
                                          st.store_results, lambda v: self._set_setting("store_results", v)))
        # 解析データ管理
        s = self._section("解析データ管理")
        nl = self.project.needs_nonlinear_solver(self.mesh) if self.mesh is not None else False
        self._row(s, "非線形解析", "はい" if (nl and (st.large_deflection or any(
            c.type not in ("bonded", "no_separation") for c in self.project.contacts))) else "いいえ")
        self._row(s, "ソルバー単位", "アクティブシステム（mm, t, N, s, MPa）")

    def _details_bc(self):
        i = self.current[1]
        bc = self.project.bcs[i]
        if bc.kind == "thermal":
            s = self._section("スコープ")
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel(", ".join(f"B{b}" for b in bc.bodies) if bc.bodies else "すべてのボディ"))
            ap = QPushButton("選択中のボディに適用")

            def _apb():
                bc.bodies = list(self.selected_bodies)
                self.invalidate()
                self._later(lambda: (self.build_details(), self.refresh_view()))
            ap.clicked.connect(_apb)
            h.addWidget(ap)
            h.addStretch()
            self._row(s, "ジオメトリ", w)
        elif bc.kind not in NO_FACE_KINDS:
            s = self._section("スコープ")
            self._scope_row(s, bc.faces, lambda faces: self._bc_apply(i, faces))
        s2 = self._section("定義")
        self._row(s2, "タイプ", BC_JP[bc.kind])
        name = QLineEdit(bc.name)
        name.setPlaceholderText(BC_JP[bc.kind])
        name.editingFinished.connect(lambda: self._bc_name(i, name.text()))
        self._row(s2, "名前", name)
        unit = BC_UNIT.get(bc.kind, "")
        if bc.kind == "fixed":
            self._row(s2, "拘束", "X, Y, Z 方向すべて固定" + ("（シェルは回転も固定）" if getattr(self.mesh, "is_shell", False) else ""))
        elif bc.kind == "symmetry":
            self._row(s2, "拘束", "面（2D は辺）の法線方向の変位を 0（シェルは面内軸まわりの回転も 0）")
            self._row(s2, "", "対称面は座標軸に垂直な平面にしてください（Ansys の「対称領域」と同じ。結果は対称モデル側の値）")
        elif bc.kind == "frictionless":
            self._row(s2, "拘束", "面（2D は辺）の法線方向の変位を 0、接線方向は自由")
            self._row(s2, "", "現在は座標軸に垂直な平らな面に対応しています")
        elif bc.kind == "thermal":
            sp = dspin(float(bc.values[0] or 0), -273, 1e5, 3, 10.0, "°C")
            sp.valueChanged.connect(lambda v: self._bc_value(i, 0, v))
            self._row(s2, "大きさ（温度）", sp)
            self._row(s2, "", f"材料の物性をこの温度で評価し、熱ひずみ α(T − 参照温度) を与えます（環境温度 {self.project.env_temp:g}°C）")
        elif bc.kind in BC_SPEC:
            self._details_bc_spec(i, bc, s2)
        elif bc.kind == "pressure":
            self._row(s2, "定義方法", "垂直方向")
            sp = ExprEdit(bc.exprs.get(0, bc.values[0] or 0), suffix=unit)
            sp.rawChanged.connect(lambda v: self._bc_value(i, 0, v))
            self._row(s2, "大きさ", sp)
        else:
            self._row(s2, "定義方法", "成分")
            for c, ax in enumerate("XYZ"):
                v = bc.values[c] if c < len(bc.values) else 0.0
                if bc.kind == "displacement":
                    w = QWidget()
                    h = QHBoxLayout(w)
                    h.setContentsMargins(0, 0, 0, 0)
                    cb = QComboBox()
                    cb.addItems(["自由", "固定値"])
                    cb.setCurrentIndex(0 if v is None else 1)
                    sp = ExprEdit(bc.exprs.get(c, 0.0 if v is None else float(v)), suffix=unit)
                    sp.setEnabled(v is not None)
                    cb.currentIndexChanged.connect(lambda idx, c=c, sp=sp: (sp.setEnabled(idx == 1),
                                                   self._bc_value(i, c, sp.raw() if idx == 1 else None)))
                    sp.rawChanged.connect(lambda val, c=c, cb=cb: cb.currentIndex() == 1 and self._bc_value(i, c, val))
                    h.addWidget(cb)
                    h.addWidget(sp)
                    self._row(s2, f"{ax} 成分", w)
                else:
                    sp = ExprEdit(bc.exprs.get(c, float(v or 0)), suffix=unit)
                    sp.rawChanged.connect(lambda val, c=c: self._bc_value(i, c, val))
                    self._row(s2, f"{ax} 成分", sp)
        ns = self.project.settings.n_steps
        if ns > 1 and bc.kind != "fixed":
            s4 = self._section("ステップごとの大きさ（倍率）")
            fac = list(bc.step_factors) + [1.0] * max(0, ns - len(bc.step_factors))
            for k in range(ns):
                sp = dspin(float(fac[k]), -1e6, 1e6, 4, 0.1)

                def setf(v, k=k):
                    f = list(bc.step_factors) + [1.0] * max(0, ns - len(bc.step_factors))
                    f[k] = v
                    bc.step_factors = f[:ns]
                    self.invalidate()
                sp.valueChanged.connect(setf)
                self._row(s4, f"ステップ {k + 1}（終了 {self.project.settings.end_times()[k]:g} s）", sp)
        if self.result is not None and i in self.result.reactions:
            r = self.result.reactions[i]
            s3 = self._section("反力（解析結果）")
            for c, ax in enumerate("XYZ"):
                self._row(s3, f"{ax} 成分", f"{r[c]:.6g} N")
            self._row(s3, "合計", f"{np.linalg.norm(r):.6g} N")

    def _details_bc_spec(self, i, bc, sec):
        """追加の境界条件（リモート・ばね・熱・電気など）の詳細: 値の欄と位置・挙動"""
        fields, opts = BC_SPEC[bc.kind]
        vals = list(bc.values) + [None] * max(0, len(fields) - len(bc.values))
        bc.values = vals[:max(len(fields), len(bc.values))]
        for c, (lab, unit, dflt, can_free) in enumerate(fields):
            v = vals[c]
            if can_free:
                w = QWidget()
                h = QHBoxLayout(w)
                h.setContentsMargins(0, 0, 0, 0)
                cb = QComboBox()
                cb.addItems(["自由", "固定値"])
                cb.setCurrentIndex(0 if v is None else 1)
                sp = dspin(0.0 if v is None else float(v), -1e12, 1e12, 6, 0.1, unit)
                sp.setEnabled(v is not None)
                cb.currentIndexChanged.connect(lambda idx, c=c, sp=sp: (sp.setEnabled(idx == 1),
                                               self._bc_value(i, c, sp.value() if idx == 1 else None)))
                sp.valueChanged.connect(lambda val, c=c, cb=cb: cb.currentIndex() == 1 and self._bc_value(i, c, val))
                h.addWidget(cb)
                h.addWidget(sp)
                self._row(sec, lab, w)
            else:
                sp = ExprEdit(bc.exprs.get(c, float(v or 0.0)), suffix=unit)
                sp.rawChanged.connect(lambda val, c=c: self._bc_value(i, c, val))
                self._row(sec, lab, sp)
        if "behavior" in opts:
            self._row(sec, "挙動", self._combo([("変形可（RBE3）", "deformable"), ("剛体", "rigid")],
                                             (bc.options or {}).get("behavior", "deformable"),
                                             lambda v: self._bc_opt(i, "behavior", v)))
        if "point" in opts or "ground" in opts:
            key = "point" if "point" in opts else "ground"
            cur = list(bc.point) if key == "point" else list((bc.options or {}).get("ground", []))
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            sps = []
            for k in range(3):
                sp = dspin(float(cur[k]) if len(cur) == 3 else 0.0, -1e9, 1e9, 4, 1.0)
                sps.append(sp)
                h.addWidget(sp)
            auto = QCheckBox("面の重心" if key == "point" else "")
            auto.setChecked(len(cur) != 3)
            if key == "point":
                h.addWidget(auto)

            def apply_pt(*_):
                pt = [] if (key == "point" and auto.isChecked()) else [sp.value() for sp in sps]
                if key == "point":
                    bc.point = pt
                else:
                    bc.options = dict(bc.options or {}, ground=pt)
                self.invalidate()
            for sp in sps:
                sp.valueChanged.connect(apply_pt)
            auto.toggled.connect(apply_pt)
            self._row(sec, "位置 X, Y, Z [mm]" if key == "point" else "地面の点 X, Y, Z [mm]", w)
        if bc.kind == "joint":
            self._details_joint(i, bc, sec)
        if bc.kind == "submodel":
            self._details_submodel(i, bc, sec)
        if bc.kind == "bolt_pretension":
            self._row(sec, "ボルトのボディ", self._combo(
                [(self._body_label(b), int(b)) for b in (self.mesh.body_ids() if self.mesh is not None else [0])],
                int((bc.options or {}).get("body", 0)), lambda v: self._bc_opt(i, "body", int(v))))
            self._row(sec, "", "断面は「位置」を通りボルトの軸（ボディの長手方向）に垂直な面。"
                               "ステップ 1 で予張力、ステップ 2 で長さを固定して外力をかけます（Ansys と同じ）")
        if bc.kind == "heat_generation":
            self._row(sec, "", "「ボディ」選択で対象のボディを選び、もう一度この欄を開いて適用（空 = すべて）")
            ap = QPushButton("選択中のボディに適用")
            ap.clicked.connect(lambda: (setattr(bc, "bodies", list(self.selected_bodies)), self.invalidate()))
            self._row(sec, "ボディ", ap)
        notes = {"cylindrical_support": "円筒面に設定。各方向を「固定値」（0 = 固定）か「自由」に",
                 "elastic_support": "面の法線方向のばね（Ansys の弾性支持）",
                 "spring": "面（剛体結合）と地面の点を結ぶ、線の方向のばね",
                 "bearing_load": "穴の内面に設定。力の向きに押される半分に分布します",
                 "convection": "q = h (T∞ − T)", "radiation": "周囲への放射 q = εσ (T∞⁴ − T⁴)",
                 "remote_force": "変形可は面を硬くしない（RBE3）。剛体は面を剛体として結ぶ"}
        if bc.kind in notes:
            self._row(sec, "", notes[bc.kind])

    def _details_submodel(self, i, bc, sec):
        """サブモデルの切断境界: 粗いモデルの結果ファイルと座標のずれ"""
        o = bc.options or {}
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        ed = QLineEdit(str(o.get("file", "")))
        ed.setPlaceholderText("粗いモデルの結果（.npz）")
        ed.editingFinished.connect(lambda: self._bc_opt(i, "file", ed.text().strip()))
        btn = QPushButton("参照…")

        def browse():
            path, _ = QFileDialog.getOpenFileName(self, "粗いモデルの結果", ed.text() or "", "サブモデル用の結果 (*.npz)")
            if path:
                ed.setText(path)
                self._bc_opt(i, "file", path)
        btn.clicked.connect(browse)
        h.addWidget(ed)
        h.addWidget(btn)
        self._row(sec, "粗いモデルの結果", w)
        off = list(o.get("offset", [0.0, 0.0, 0.0]))
        w2 = QWidget()
        h2 = QHBoxLayout(w2)
        h2.setContentsMargins(0, 0, 0, 0)
        sps = []
        for k in range(3):
            sp = dspin(float(off[k]), -1e9, 1e9, 4, 1.0)
            sps.append(sp)
            h2.addWidget(sp)
        for sp in sps:
            sp.valueChanged.connect(lambda *_: self._bc_opt(i, "offset", [x.value() for x in sps]))
        self._row(sec, "座標のずれ X, Y, Z [mm]", w2)
        far = getattr(bc, "_submodel_far", None)
        if far is not None:
            self._row(sec, "粗いモデルの外への最大距離", f"{far:.3g} mm")
        self._row(sec, "", "全体モデルを粗く解いて「粗い結果を保存」で .npz に保存し、注目部分だけの細かいモデルの"
                           "切断面にこの条件を置きます。切断面の節点に粗いモデルの変位を内挿して与えます"
                           "（Ansys のサブモデリングと同じ）。切断面は応力集中から離れた位置に置いてください。")

    def save_coarse_result(self):
        """サブモデル用に、今の結果（節点・要素・変位）を保存"""
        if not self._need_result():
            return
        r = self.result
        if r.mesh.elems.shape[1] not in (4, 10, 6, 15, 8, 20) or getattr(r.mesh, "is_shell", False) \
                or getattr(r.mesh, "is_2d", False):
            QMessageBox.information(self, "サブモデル", "ソリッドの結果だけ保存できます。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "サブモデル用に結果を保存", "coarse.npz", "サブモデル用の結果 (*.npz)")
        if path:
            from ..submodel import save_coarse
            save_coarse(path, r.mesh, r.u)
            self.log(f"サブモデル用の結果を保存しました: {path}")

    def _details_joint(self, i, bc, sec):
        """ジョイントの詳細（Ansys の Joint: 種類・地面かボディか・座標系・自由な方向のばね/荷重/強制変位）"""
        from ..remote import JOINT_FREE, JOINT_JP
        o = bc.options or {}
        self._row(sec, "種類", self._combo([(v, k) for k, v in JOINT_JP.items()], o.get("type", "revolute"),
                                         lambda v: (self._bc_opt(i, "type", v), self._later(self.build_details))))
        self._row(sec, "接続", self._combo([("ボディ－地面", True), ("ボディ－ボディ", False)], bool(o.get("ground", True)),
                                         lambda v: (self._bc_opt(i, "ground", bool(v)), self._later(self.build_details))))
        if not o.get("ground", True):
            fb = list(o.get("faces_b") or [])
            b2 = QPushButton(f"選択中の面を可動側にする（いま {len(fb)} 面）")
            b2.clicked.connect(lambda: (self._bc_opt(i, "faces_b", list(self.selected_faces)),
                                        self._later(self.build_details)))
            self._row(sec, "可動側の面", b2)
            self._row(sec, "", "上の「面」が参照側、ここで決めた面が可動側です（Ansys の Reference / Mobile）")
        for key, lab, dflt in (("axis", "軸（局所 z）", [0, 0, 1]), ("axis_x", "局所 x の向き", [1, 0, 0])):
            cur = list(o.get(key) or dflt)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            sps = []
            for k in range(3):
                sp = dspin(float(cur[k]), -1e6, 1e6, 4, 0.1)
                sps.append(sp)
                h.addWidget(sp)
            for sp in sps:
                sp.valueChanged.connect(lambda _=None, key=key, sps=sps: self._bc_opt(i, key, [x.value() for x in sps]))
            self._row(sec, lab, w)
        jt = o.get("type", "revolute")
        names = ["並進 x", "並進 y", "並進 z", "回転 x", "回転 y", "回転 z"]
        if jt == "general":
            free = list(o.get("free") or [])
            for k in range(6):
                cb = QCheckBox("自由")
                cb.setChecked(k in free)
                cb.toggled.connect(lambda on, k=k: self._bc_opt(
                    i, "free", sorted(set((bc.options or {}).get("free") or []) ^ {k})))
                self._row(sec, names[k], cb)
            free = list(o.get("free") or [])
        else:
            free = JOINT_FREE[jt]
        self._row(sec, "自由な方向", "、".join(names[k] for k in free) or "なし（固定）")

        def setlist(key, k, v, n=6, fill=0.0):
            lst = (list((bc.options or {}).get(key) or []) + [fill] * n)[:n]
            lst[k] = v
            self._bc_opt(i, key, lst)
        for k in free:
            unit = "mm" if k < 3 else "rad"
            st = dspin(float(((o.get("stiffness") or [0] * 6) + [0] * 6)[k]), 0.0, 1e15, 4, 10.0,
                       "N/mm" if k < 3 else "N·mm/rad")
            st.valueChanged.connect(lambda v, k=k: setlist("stiffness", k, float(v)))
            self._row(sec, f"{names[k]} のばね", st)
            ld = dspin(float(((o.get("load") or [0] * 6) + [0] * 6)[k]), -1e12, 1e12, 4, 10.0, "N" if k < 3 else "N·mm")
            ld.valueChanged.connect(lambda v, k=k: setlist("load", k, float(v)))
            self._row(sec, f"{names[k]} の荷重", ld)
            dv = ((o.get("disp") or [None] * 6) + [None] * 6)[k]
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            cb = QComboBox()
            cb.addItems(["なし", "与える"])
            cb.setCurrentIndex(0 if dv is None else 1)
            sp = dspin(0.0 if dv is None else float(dv), -1e6, 1e6, 6, 0.01, unit)
            sp.setEnabled(dv is not None)
            cb.currentIndexChanged.connect(lambda idx, k=k, sp=sp: (sp.setEnabled(idx == 1), setlist(
                "disp", k, sp.value() if idx == 1 else None, fill=None)))
            sp.valueChanged.connect(lambda v, k=k, cb=cb: cb.currentIndex() == 1 and setlist("disp", k, float(v), fill=None))
            h.addWidget(cb)
            h.addWidget(sp)
            self._row(sec, f"{names[k]} の強制変位", w)
        self._row(sec, "", "面は剛体としてジョイントの点に結びます。自由な方向は、ばね・荷重・強制変位で動きを決めます"
                         "（どれもないと動きが止まらず解けないことがあります）")
        r = self.result
        jr = getattr(r, "joints", {}).get(i) if r is not None else None
        if jr:
            fl, ml = jr["力（局所）"], jr["モーメント（局所）"]
            self._row(sec, "ジョイントの力（局所）", f"{fl[0]:.4g}, {fl[1]:.4g}, {fl[2]:.4g} N")
            self._row(sec, "ジョイントのモーメント（局所）", f"{ml[0]:.4g}, {ml[1]:.4g}, {ml[2]:.4g} N·mm")
            rel = list(jr["相対変位（局所）"]) + list(jr["相対回転（局所） [rad]"])
            self._row(sec, "相対運動（自由な方向）", "、".join(f"{names[k]} {rel[k]:.4g}" for k in free) or "—")

    def _bc_opt(self, i, key, value):
        bc = self.project.bcs[i]
        bc.options = dict(bc.options or {}, **{key: value})
        self.invalidate()

    def _bc_apply(self, i, faces):
        self.project.bcs[i].faces = list(faces)
        self.invalidate()

    def _bc_name(self, i, text):
        if self.project.bcs[i].name != text.strip():
            self.project.bcs[i].name = text.strip()
            self.rebuild_tree(details=False)

    def _bc_value(self, i, c, v):
        bc = self.project.bcs[i]
        vals = list(bc.values)
        while len(vals) <= c:
            vals.append(0.0)
        if isinstance(v, str):
            try:
                vals[c] = evaluate(v, self.project.env())
            except ParamError as e:
                QMessageBox.warning(self, "式の入力", str(e))
                return
            bc.exprs[c] = v
        else:
            vals[c] = v
            bc.exprs.pop(c, None)
        self.project.bcs[i].values = vals
        self.invalidate()
        self.refresh_view()

    def _details_solution(self):
        s = self._section("情報")
        st = "解析済み" if self.result is not None else ("エラー" if self.solve_error else "未解析（⚡ 解析を押してください）")
        self._row(s, "状態", st)
        if self.result is not None:
            for k, v in self.result.summary().items():
                self._row(s, k, f"{v:.6g}" if isinstance(v, float) else str(v))

    def _details_info(self):
        hist = getattr(self.result, "history", None) or getattr(self, "_fail_history", None)
        if hist:
            s0 = self._section("収束の履歴（力の残差比 / 許容値 " +
                               f"{self.project.settings.force_convergence * 100:g}%）")
            for h in hist[-40:]:
                self._row(s0, f"時間 {h['時間']:g} 反復 {h['反復']}",
                          f"残差 {h['力の残差比']:.2e}　状態変化 {h['接触状態の変化']}　貫通 {h['最大貫通 [mm]']:.1e}")
        s = self._section("ソルバー出力")
        if self.result is not None:
            for k, v in self.result.info.items():
                self._row(s, k, f"{v:.3g}" if isinstance(v, float) else str(v))
        if self.solve_error:
            self._row(s, "エラー", self.solve_error)
        if self.solver_log:
            s2 = self._section("ログ")
            for line in self.solver_log[-12:]:
                self._row(s2, "", line)

    def show_history_plot(self):
        """過渡解析の時刻歴（エネルギー・接触力・最大変位）のグラフ"""
        h = getattr(self.results[-1], "history", None) if self.results else None
        if not h:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("時刻歴")
        dlg.resize(820, 560)
        lay = QVBoxLayout(dlg)
        t = np.asarray(h["時間 [s]"]) * 1e3
        keys = [k for k in h if k != "時間 [s]"]
        try:
            from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
            tabs = QTabWidget()
            for k in keys:
                ch = QChart()
                ser = QLineSeries()
                ys = np.asarray(h[k], float)
                step = max(1, len(t) // 1500)
                for x, y in zip(t[::step], ys[::step]):
                    ser.append(float(x), float(y))
                ch.addSeries(ser)
                ax, ay = QValueAxis(), QValueAxis()
                ax.setTitleText("時間 [ms]")
                ay.setTitleText(k)
                ax.setRange(float(t.min()), float(t.max()) if t.max() > t.min() else float(t.min()) + 1)
                lo, hi = float(ys.min()), float(ys.max())
                ay.setRange(lo, hi if hi > lo else lo + 1)
                ch.addAxis(ax, Qt.AlignBottom)
                ch.addAxis(ay, Qt.AlignLeft)
                ser.attachAxis(ax)
                ser.attachAxis(ay)
                ch.legend().hide()
                ch.setTitle(f"{k}（最大 {hi:.4g}）")
                v = QChartView(ch)
                v.setRenderHint(QPainter.Antialiasing)
                tabs.addTab(v, k.split(" [")[0])
            lay.addWidget(tabs)
        except Exception:
            box = QPlainTextEdit()
            box.setReadOnly(True)
            lines = ["時間 [ms]\t" + "\t".join(keys)]
            step = max(1, len(t) // 200)
            for i in range(0, len(t), step):
                lines.append(f"{t[i]:.5g}\t" + "\t".join(f"{h[k][i]:.5g}" for k in keys))
            box.setPlainText("\n".join(lines))
            lay.addWidget(box)
        dlg.show()

    def _details_result(self):
        key = self.project.results[self.current[1]]
        s = self._section("スコープ")
        self._row(s, "スコープ方法", "ジオメトリの選択")
        self._row(s, "ジオメトリ", "接触面" if key.startswith("接触:") else "すべてのボディ")
        if len(self.results) > 1:
            st = self._section("時間")
            c = QComboBox()
            for j, r in enumerate(self.results):
                if getattr(r, "label", None):
                    c.addItem(r.label, j)
                elif getattr(r, "history", None) is not None:
                    c.addItem(f"時刻 {getattr(r, 'time', 0):.4g} s", j)
                else:
                    c.addItem(f"ステップ {r.info.get('ステップ', j + 1)}（{getattr(r, 'time', j + 1):g} s）", j)
            c.setCurrentIndex(self.result_index if self.result_index >= 0 else len(self.results) - 1)

            def pick(_):
                self.result_index = c.currentData()
                self.result = self.results[self.result_index]
                self._later(lambda: (self.build_details(), self.refresh_view()))
            c.currentIndexChanged.connect(pick)
            self._row(st, "モード" if getattr(self.results[-1], "modal_table", None) else
                      "周波数" if getattr(self.results[-1], "frf", None) is not None else "表示時間", c)
            if getattr(self.results[-1], "history", None) is not None:
                hb = QPushButton("時刻歴グラフ（エネルギー・接触力）…")
                hb.clicked.connect(lambda: self._later(self.show_history_plot))
                self._row(st, "", hb)
        s2 = self._section("定義")
        self._row(s2, "タイプ", RESULT_TYPE_JP.get(key, RESULT_NAME.get(key, key[2:] if key.startswith("式:") else key)))
        self._details_result_tools(key)
        self._row(s2, "単位", key.split("[")[-1].rstrip("]") if "[" in key else "なし")
        if self.result is not None:
            vals = np.asarray(self._fields(self.result).get(key, []), float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                s3 = self._section("結果")
                self._row(s3, "最小", f"{vals.min():.6g}")
                self._row(s3, "最大", f"{vals.max():.6g}")
                self._row(s3, "平均", f"{vals.mean():.6g}")
                if key == "接触:接触状態":
                    for code, lab in STATUS_JP.items():
                        self._row(s3, f"{code}: {lab}", f"{int(np.sum(vals == code))} 節点")
            elif key.startswith("接触:"):
                self._row(self._section("結果"), "", "接触の結果がありません（接触領域がないか、出力がオフ）")
            if self.probe_text:
                s4 = self._section("プローブ（クリックした点）")
                for line in self.probe_text.splitlines():
                    a, _, b = line.partition(": ")
                    self._row(s4, a, b)
        else:
            s3 = self._section("結果")
            self._row(s3, "状態", "未解析（⚡）")

    # ======================================================================= 追加・削除
    def add_bc(self, kind):
        faces = [] if kind in NO_FACE_KINDS else list(self.selected_faces)
        if kind in BC_SPEC:
            vals = [d for _, _, d, _ in BC_SPEC[kind][0]]
        else:
            vals = {"fixed": [0.0, 0.0, 0.0], "displacement": [0.0, None, None], "force": [0.0, 0.0, -1000.0],
                    "pressure": [1.0], "moment": [0.0, 0.0, 1000.0], "gravity": [0.0, 0.0, -9806.65],
                    "thermal": [100.0], "symmetry": [], "frictionless": []}[kind]
        bc = BC(kind, faces, vals, "")
        if kind in BC_SPEC and "behavior" in BC_SPEC[kind][1]:
            bc.options = {"behavior": "rigid" if kind == "spring" else "deformable"}
        if kind == "bolt_pretension":
            bc.options = {"body": int(self.selected_bodies[0]) if self.selected_bodies else 0}
        if kind == "submodel":
            bc.options = {"file": "", "offset": [0.0, 0.0, 0.0]}
        if kind == "joint":
            # 最初に選んでいる面を参照側にする（可動側は詳細の「選択中の面を可動側に」で。なければ地面とのジョイント）
            bc.options = {"type": "revolute", "ground": True, "axis": [0.0, 0.0, 1.0]}
        if kind in ("thermal", "heat_generation"):
            bc.bodies = list(self.selected_bodies)
        self.project.bcs.append(bc)
        self.invalidate()
        self.select_node(("bc", len(self.project.bcs) - 1))
        if kind not in NO_FACE_KINDS and not faces:
            self.log(f"{BC_JP[kind]} を追加しました。面を選択して詳細の「適用」を押してください。")

    def add_result(self, key):
        self.project.results.append(key)
        self.rebuild_tree()
        self.select_node(("result", len(self.project.results) - 1))

    def delete_current(self):
        k = self.current[0]
        if k == "feature":
            i = self.current[1]
            feats = self._features()
            gone = set(feature_body_keys(feats, all_keys=True).get(i, []))
            self._hist_label = f"削除: {self._feature_label(feats[i])}"
            del feats[i]
            if gone:
                # 後ろのボディ番号を詰める（属性の割り当てと参照を付け替え）
                fix = lambda x: x - sum(1 for g in gone if g < x)  # noqa: E731
                ba = {}
                for kk, v in self.project.body_assign.items():
                    kk = int(kk)
                    if kk in gone:
                        continue
                    ba[str(fix(kk))] = v
                self.project.body_assign = ba
                for f in feats:
                    if f.get("type") == "bool":
                        f["target"] = fix(int(f["target"]))
                        f["tools"] = [fix(int(t)) for t in f.get("tools", []) if int(t) not in gone]
                    if f.get("targets"):
                        f["targets"] = [fix(int(t)) for t in f["targets"] if int(t) not in gone]
            self.current = ("geometry",)
            self.selected_bodies = []
            self.remesh()
            return
        if k == "part":
            del self.project.parts[self.current[1]]
            self.invalidate(mesh=True)
            self.current = ("geometry",)
            self.remesh()
            return
        if k == "contact":
            del self.project.contacts[self.current[1]]
            self.invalidate()
            self.select_node(("contacts",))
            return
        if k == "attr":
            name = self.current[1]
            self.project.body_attrs.pop(name, None)
            for d in self.project.body_assign.values():
                if d.get("attr") == name:
                    d.pop("attr", None)
            self._fill_attr_combos()
            self.invalidate(mesh=True)
            self.select_node(("attrs",))
            return
        if k == "mat":
            name = self.current[1]
            self.project.materials.pop(name, None)
            for d in self.project.body_assign.values():
                if d.get("mat") == name:
                    d.pop("mat", None)
            self._fill_attr_combos()
            self.invalidate()
            self.select_node(("mats",))
            return
        if k == "bc":
            del self.project.bcs[self.current[1]]
            self.invalidate()
            self.select_node(("analysis",))
        elif k == "result":
            del self.project.results[self.current[1]]
            self.rebuild_tree()
            self.select_node(("solution",))
        elif k == "sizing":
            del self.project.local_sizes[self.current[1]]
            self.invalidate(mesh=True)
            self.select_node(("mesh",))

    # ======================================================================= 形状・メッシュ
    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ジオメトリをインポート", "",
            "形状ファイル (*.step *.stp *.STEP *.STP *.igs *.iges *.brep *.stl *.STL *.x_t *.x_b);;すべて (*)")
        if path:
            self._set_geometry({"kind": "file", "path": path})

    def _set_geometry(self, spec, keep_bcs=False):
        if spec == self.project.geometry:
            return
        old_kind = self.project.geometry.get("kind")
        self.project.geometry = spec
        if not keep_bcs or spec.get("kind") != old_kind:
            if self.project.bcs or self.project.local_sizes:
                r = QMessageBox.question(self, "形状の変更", "形状が変わります。設定済みの荷重・支持と面サイズ指定を削除しますか？\n"
                                         "（面の番号が変わるため、通常は削除をおすすめします）")
                if r == QMessageBox.Yes:
                    self.project.bcs = []
                    self.project.local_sizes = []
        self.selected_faces = []
        self.remesh(reset_camera=True)

    def remesh(self, reset_camera=False):
        """gmsh はスレッドセーフでなく macOS で落ちるため、メインスレッドで実行"""
        g = self.project.geometry
        if g.get("kind") == "model" and not [f for f in g.get("features", []) if not f.get("suppressed")]:
            self.mesh = None
            self.mesh_dirty = False
            self.result = None
            self.selected_faces = []
            self.selected_bodies = []
            self._fill_attr_combos()
            self.rebuild_tree()
            self.refresh_view()
            return False
        try:
            mesh = build_mesh(self.project.geometry, self.project.mesh_size, self.project.mesh_order,
                              log=lambda m: (self.log(str(m)), QApplication.processEvents()),
                              local_sizes=self.project.local_sizes + self.project.body_local_sizes(self.mesh),
                              share=self.project.share_map(), settings=self.project.mesh_settings)
        except Exception as e:
            self._busy(False)
            msg = str(e) if isinstance(e, (GeometryError, ValueError, RuntimeError)) else traceback.format_exc()
            self.log("エラー: " + msg)
            self.rebuild_tree()
            QMessageBox.warning(self, "形状・メッシュ", msg[-1500:])
            return False
        self._busy(False)
        self.mesh = mesh
        self.mesh_dirty = False
        mesh.face_info()
        p = self.project
        if not p.body_assign and not p.materials and p.geometry.get("kind") != "model":
            # 旧形式（材料が1つ）のプロジェクト: すべてのボディに同じ材料を割り当て
            p.materials[p.material.name] = p.material
            p.assign(mesh.body_ids().tolist(), mat=p.material.name)
        self.selected_bodies = [b for b in self.selected_bodies if b in set(mesh.body_ids().tolist())]
        self._fill_attr_combos()
        if len(mesh.body_ids()) > 1 and not getattr(mesh, "is_shell", False) and not getattr(mesh, "is_2d", False):
            # Ansys と同じく、ボディ同士の接触を自動で作成（手動の接触は残す）
            self.auto_contacts(quiet=True)
        else:
            self.project.contacts = [c for c in self.project.contacts if not c.auto]
        self.selected_faces = [f for f in self.selected_faces if f < mesh.n_faces]
        self.result = None
        self.solve_error = ""
        self.log(f"メッシュ: 節点 {mesh.n_nodes:,}  要素 {mesh.n_elems:,}  面 {mesh.n_faces}")
        self.rebuild_tree()
        self.refresh_view()
        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        return True

    # ======================================================================= 解析
    def _busy(self, on, msg=""):
        self.progress.setVisible(on)
        self.progress.setRange(0, 0 if on else 100)
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.log(msg)
            QApplication.processEvents()
        else:
            QApplication.restoreOverrideCursor()

    def _details_result_tools(self, key):
        """モードの表・周波数応答・パス結果・疲労の設定・結果の式"""
        from .charts import chart_dialog, table_dialog
        r = self.result
        if key.startswith("式:"):
            s = self._section("ユーザー定義結果")
            ed = QLineEdit(key[2:])
            ed.setToolTip("使える名前: UX UY UZ USUM SX SY SZ SXY SYZ SXZ SEQV S1 S3 TEMP X Y Z、関数 sqrt abs max min …")

            def set_expr():
                i = self.current[1]
                self.project.results[i] = "式:" + ed.text().strip()
                self._later(lambda: (self.rebuild_tree(), self.select_node(("result", i))))
            ed.editingFinished.connect(set_expr)
            self._row(s, "式", ed)
            if getattr(self, "_expr_error", ""):
                self._row(s, "", "⚠ " + self._expr_error)
                self._expr_error = ""
        if key.startswith("疲労:"):
            s = self._section("疲労ツール（応力寿命）")
            fo = (self.project.analysis_opts or {}).get("fatigue", {})
            from ..postproc import MEAN_CORR, STRESS_COMP
            self._row(s, "荷重の種類", self._combo([("両振り（R = −1）", "reversed"), ("片振り（R = 0）", "zero"),
                                                  ("応力比を指定", "ratio")], fo.get("loading", "reversed"),
                                                 lambda v: self._opt("loading", v, "fatigue", rebuild=True)))
            if fo.get("loading") == "ratio":
                w = dspin(float(fo.get("ratio", -1.0)), -1e3, 1, 3, 0.1)
                w.valueChanged.connect(lambda v: (self._opt("ratio", v, "fatigue"), self.refresh_view()))
                self._row(s, "応力比 R", w)
            w = dspin(float(fo.get("scale", 1.0)), 0, 1e6, 4, 0.1)
            w.valueChanged.connect(lambda v: (self._opt("scale", v, "fatigue"), self.refresh_view()))
            self._row(s, "倍率", w)
            self._row(s, "平均応力補正", self._combo([(v, k) for k, v in MEAN_CORR.items()], fo.get("mean_corr", "none"),
                                                   lambda v: (self._opt("mean_corr", v, "fatigue"), self.refresh_view())))
            self._row(s, "応力の成分", self._combo([(v, k) for k, v in STRESS_COMP.items()], fo.get("comp", "signed_vm"),
                                                lambda v: (self._opt("comp", v, "fatigue"), self.refresh_view())))
            w = dspin(float(fo.get("design_life", 1e9)), 1, 1e15, 0, 1e6, "回")
            w.valueChanged.connect(lambda v: (self._opt("design_life", v, "fatigue"), self.refresh_view()))
            self._row(s, "設計寿命", w)
            self._row(s, "", "材料の S-N 曲線（材料の sn_table）を使います。構造用鋼には代表的な曲線が入っています")
            if getattr(self, "_fatigue_error", ""):
                self._row(s, "", "⚠ " + self._fatigue_error)
                self._fatigue_error = ""
        if r is None:
            return
        tools = self._section("グラフ・表")
        if getattr(r, "modal_table", None):
            b = QPushButton("モードの一覧（振動数・有効質量）…")
            b.clicked.connect(lambda: table_dialog(self, "モードの一覧", r.modal_table))
            self._row(tools, "", b)
        if getattr(r, "frf", None) is not None:
            b = QPushButton("周波数応答のグラフ…")

            def frf():
                f, a = r.frf
                chart_dialog(self, "周波数応答（全節点の最大変位振幅）", "周波数 [Hz]", "変位振幅 [mm]",
                             [("最大変位振幅", f, a)], logy=True)
            b.clicked.connect(frf)
            self._row(tools, "", b)
        if getattr(r, "thermal", None) is not None and r.thermal.reactions:
            for bi, qv in r.thermal.reactions.items():
                self._row(tools, f"熱流量（{self.project.bcs[bi].label() if bi < len(self.project.bcs) else bi}）",
                          f"{qv:.6g} W")
        b = QPushButton("パスに沿ったグラフ…")
        b.clicked.connect(lambda: self._path_plot(key))
        self._row(tools, "", b)
        if not key.startswith(("疲労:", "式:", "接触:")) and self.project.analysis_type in ("static",):
            s3 = self._section("収束（適応メッシュ）")
            co = (self.project.analysis_opts or {}).get("convergence", {})
            w = dspin(float(co.get("allowable", 5.0)), 0.1, 100, 2, 1.0, "%")
            w.valueChanged.connect(lambda v: self._conv_opt("allowable", float(v)))
            self._row(s3, "許容変化", w)
            sp = QSpinBox()
            sp.setRange(1, 10)
            sp.setValue(int(co.get("max_loops", 4)))
            sp.valueChanged.connect(lambda v: self._conv_opt("max_loops", int(v)))
            self._row(s3, "最大の細分化回数", sp)
            b = QPushButton("この結果の最大値が収束するまでメッシュを細かくする")
            b.clicked.connect(lambda: self.run_convergence(key))
            self._row(s3, "", b)
            self._row(s3, "", "応力の誤差（ZZ 法）が大きい所だけ要素を半分の大きさにして解き直し、"
                              "最大値の変化が許容値以下になるまで繰り返します（Ansys の収束ツールと同じ考え方）")
            hist = (self.project.analysis_opts or {}).get("convergence_history")
            if hist:
                hb = QPushButton("収束の履歴…")
                from .charts import table_dialog as _td
                hb.clicked.connect(lambda: _td(self, "収束の履歴", hist))
                self._row(s3, "", hb)

    def _conv_opt(self, k, v):
        """収束ツールの設定（結果は無効にしない）"""
        o = dict(self.project.analysis_opts or {})
        o["convergence"] = dict(o.get("convergence", {}), **{k: v})
        self.project.analysis_opts = o

    def run_convergence(self, key):
        """適応メッシュで、結果 key の最大値が収束するまで解き直す。
        gmsh はスレッドセーフでなく macOS で落ちるため、メッシュ作成と同じくメインスレッドで実行する"""
        from ..adaptive import converge
        co = (self.project.analysis_opts or {}).get("convergence", {})
        p = self.project
        body_ls = p.body_local_sizes(self.mesh) if self.mesh is not None else []

        def mesh_fn():
            return build_mesh(p.geometry, p.mesh_size, p.mesh_order, log=lambda *a: None,
                              local_sizes=p.local_sizes + body_ls, share=p.share_map(), settings=p.mesh_settings)

        def log(m):
            self.solver_log.append(str(m))
            self.log(str(m))
            QApplication.processEvents()
        self._busy(True, "適応メッシュで収束させています…")
        QApplication.processEvents()
        try:
            mesh, res, hist = converge(p, field=key, stat="max", allowable=float(co.get("allowable", 5.0)),
                                       max_loops=int(co.get("max_loops", 4)), log=log, mesh_fn=mesh_fn)
        except Exception as e:
            self._on_solve_failed(str(e) if isinstance(e, (GeometryError, ValueError, RuntimeError))
                                  else traceback.format_exc())
            return
        self.mesh = mesh
        self.mesh_dirty = False
        mesh.face_info()
        self.project.analysis_opts = dict(self.project.analysis_opts or {}, convergence_history=[
            {k: (round(v, 6) if isinstance(v, float) else ("-" if v is None else v)) for k, v in h.items()}
            for h in hist])
        self._on_solved(res)
        self.refresh_view()
        from .charts import table_dialog
        table_dialog(self, "収束の履歴", self.project.analysis_opts["convergence_history"])

    def open_frame(self):
        """はり要素（骨組み）解析のダイアログ。材料は材料ライブラリとプロジェクトの材料定数"""
        from .frame_dialog import FrameDialog
        md = (self.project.analysis_opts or {}).get("frame")
        dlg = FrameDialog(self, model=md, materials=dict(self.project.materials))
        dlg.exec()
        try:
            self.project.analysis_opts = dict(self.project.analysis_opts or {}, frame=dlg.model())
        except Exception:  # noqa: BLE001
            pass

    def _path_plot(self, key):
        from .charts import chart_dialog
        from ..postproc import path_values
        if self.result is None or self.mesh is None:
            return
        vals = self._fields(self.result).get(key)
        if vals is None:
            return
        lo, hi = self.mesh.bounds()
        dlg = QDialog(self)
        dlg.setWindowTitle("パス（始点と終点）")
        lay = QVBoxLayout(dlg)
        sp = []
        for lab, pt in (("始点", lo), ("終点", hi)):
            h = QHBoxLayout()
            h.addWidget(QLabel(lab + " X, Y, Z [mm]"))
            row = []
            for k in range(3):
                w = dspin(float(pt[k]), -1e9, 1e9, 4, 1.0)
                row.append(w)
                h.addWidget(w)
            sp.append(row)
            lay.addLayout(h)
        if len(self.selected_faces) >= 1:
            lay.addWidget(QLabel("（面を選んでいれば、その重心が初期値になります）"))
        ok = QPushButton("グラフを表示")
        ok.clicked.connect(dlg.accept)
        lay.addWidget(ok)
        info = {f["id"]: f for f in self.mesh.face_info()}
        if len(self.selected_faces) >= 2:
            for row, fid in zip(sp, self.selected_faces[:2]):
                for k in range(3):
                    row[k].setValue(float(info[int(fid)]["centroid"][k]))
        if dlg.exec() != QDialog.Accepted:
            return
        p0 = [w.value() for w in sp[0]]
        p1 = [w.value() for w in sp[1]]
        d, v = path_values(self.mesh, vals, p0, p1, 101)
        chart_dialog(self, f"パス結果: {RESULT_NAME.get(key, key)}", "始点からの距離 [mm]", key, [(key, d, v)])

    def run_solve(self):
        if self.worker and self.worker.isRunning():
            return
        bad = [b for b in self.project.bcs if b.kind not in NO_FACE_KINDS and not b.faces]
        if bad:
            QMessageBox.warning(self, "解析", "面が割り当てられていない荷重・支持があります（? マーク）。\n"
                                "面を選択して詳細の「適用」を押してください。")
            return
        transient = self.project.analysis_type == "transient"
        free_ok = self.project.analysis_type in ("modal", "thermal", "electrostatic")
        if not self._has_support() and not transient and not free_ok:
            QMessageBox.warning(self, "解析", "支持がありません。「固定支持」などを追加してください。")
            return
        if self.mesh is None or self.mesh_dirty:
            if not self.remesh():
                return
        try:
            em = self.project.elem_materials(self.mesh)
        except ValueError as e:
            QMessageBox.warning(self, "材料定数", str(e))
            self.set_attr_mode(0)
            self.ribbon.setCurrentIndex(self.tab_model)
            return
        self.solver_log = []
        self._fail_history = None
        p = self.project
        if not self._surface_mesh() and p.needs_nonlinear_solver(self.mesh) and not p.contacts \
                and len(self.mesh.body_ids()) > 1:
            self.auto_contacts(quiet=True)
        bad_c = [c.name for c in p.contacts if not c.suppressed and not (c.contact_faces and c.target_faces)]
        if bad_c:
            QMessageBox.warning(self, "接触", "面が割り当てられていない接触領域があります: " + ", ".join(bad_c))
            return
        if (p.analysis_type in EXTRA_TYPES or any(b.kind == "bolt_pretension" for b in p.bcs)
                or (p.analysis_type == "static" and (p.analysis_opts or {}).get("thermal_import"))):
            self._busy(True, f"{self._analysis_label()}を実行しています…")
            self.worker = Worker(p.run_all, self.mesh)
            self.worker.log.connect(lambda m: (self.solver_log.append(m), self.log(m)))
            self.worker.done.connect(self._on_solved)
            self.worker.failed.connect(self._on_solve_failed)
            self.worker.start()
            return
        if transient:
            from ..dynamics import DynamicSettings
            ds = DynamicSettings.from_dict(p.dynamic)
            if not self._has_support() and not (ds.wall or {}).get("normal") and not any(ds.init_velocity) \
                    and not ds.body_velocity:
                QMessageBox.warning(self, "過渡構造", "支持・剛体壁・初速度のどれもありません。")
                return
            if p.transient_nonlinear(self.mesh) and not p.contacts and len(self.mesh.body_ids()) > 1:
                self.auto_contacts(quiet=True)
            self._busy(True, "過渡解析を実行しています…")
            self.worker = Worker(p.run_transient, self.mesh, em)
            self.worker.log.connect(lambda m: (self.solver_log.append(m), self.log(m)))
            self.worker.done.connect(self._on_solved)
            self.worker.failed.connect(self._on_solve_failed)
            self.worker.start()
            return
        if self._surface_mesh():
            try:
                p.check_shell_supported()
            except ValueError as e:
                QMessageBox.warning(self, "シェル・2D", str(e))
                return
            self._busy(True, "解析を実行しています…")
            if getattr(self.mesh, "is_2d", False):
                from ..plane import solve_2d
                self.worker = Worker(solve_2d, self.mesh, p.material, list(p.bcs), mode=self.mesh.mode,
                                     elem_materials=em, settings=p.settings, n_cores=p.settings.n_cores)
            else:
                from ..shell import solve_shell
                try:
                    lam = p.laminate(self.mesh)
                except ValueError as e:
                    self._busy(False)
                    QMessageBox.warning(self, "積層", str(e))
                    return
                self.worker = Worker(solve_shell, self.mesh, p.material, list(p.bcs), elem_materials=em, laminate=lam,
                                     n_cores=p.settings.n_cores, settings=p.settings)
        elif p.needs_nonlinear_solver(self.mesh):
            self._busy(True, "解析を実行しています…")
            self.worker = Worker(solve_static, self.mesh, p.material, list(p.bcs), list(p.contacts), p.settings,
                                 elem_materials=em)
        else:
            self._busy(True, "解析を実行しています…")
            self.worker = Worker(solve, self.mesh, p.material, list(p.bcs), method=p.settings.solver_type,
                                 elem_materials=em, n_cores=p.settings.n_cores)
        self.worker.log.connect(lambda m: (self.solver_log.append(m), self.log(m)))
        self.worker.done.connect(self._on_solved)
        self.worker.failed.connect(self._on_solve_failed)
        self.worker.start()

    def _on_solved(self, result):
        self._busy(False)
        self.results = list(result) if isinstance(result, list) else [result]
        self.result_index = len(self.results) - 1
        result = self.results[-1]
        self.result = result
        self.solve_error = ""
        if self.project.cracks and hasattr(self.mesh, "crack_faces"):
            for row in self._fracture_rows():
                self.log(f"{row['き裂']}: J = {row['J [N/mm]']:.4g} N/mm、K = {row['K（J から）[MPa√mm]']:.4g} MPa√mm")
        self.log("解析が完了しました（" + ", ".join(f"{k} {v}" for k, v in result.info.items()
                                               if k in ("自由度数", "合計時間 [s]", "解析", "ステップ数", "計算時間 [s]")) + "）")
        self.rebuild_tree()
        if self.project.results:
            self.select_node(("result", min(1, len(self.project.results) - 1)))

    def _fracture_rows(self):
        r = self.results[-1] if self.results else None
        if r is None or self.mesh is None:
            return []
        rows = getattr(r, "fracture", None)
        if rows is None:
            try:
                rows = self.project.fracture(self.mesh, r)
            except Exception as e:      # 破壊の評価が失敗しても解析結果は使えるようにする
                self.log("破壊パラメータ: " + str(e))
                rows = []
            r.fracture = rows
        return rows

    def show_fracture(self):
        from .charts import table_dialog
        if not self.project.cracks:
            QMessageBox.information(self, "破壊", "き裂がありません。環境タブの「き裂」で定義してください。")
            return
        rows = self._fracture_rows()
        if not rows:
            QMessageBox.information(self, "破壊", "解析結果がありません（き裂を入れてメッシュを作り直し、解析してください）。")
            return
        table_dialog(self, "破壊パラメータ（J 積分・応力拡大係数）", [
            {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in rows])

    def show_plies(self):
        from .charts import table_dialog
        r = self.results[-1] if self.results else None
        rows = getattr(r, "plies", None) if r is not None else None
        if not rows:
            QMessageBox.information(self, "積層", "積層シェルの結果がありません（ボディ属性の「積層（複合材）」で層を定義して解析してください）。")
            return
        table_dialog(self, "積層の層ごとの応力（材料主軸）と破損指標", [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows])

    def edit_cracks(self):
        from .crack_dialog import CrackDialog
        lo, hi = (self.mesh.bounds() if self.mesh is not None else (np.zeros(3), np.ones(3) * 100))
        d = CrackDialog(self, list((self.project.mesh_settings or {}).get("cracks", [])), lo, hi)
        if d.exec():
            ms = dict(self.project.mesh_settings or {})
            ms["cracks"] = d.cracks()
            self.project.mesh_settings = ms
            self.invalidate(mesh=True)
            self.log(f"き裂 {len(ms['cracks'])} 個（メッシュを作り直すと形状に埋め込みます）")

    def _on_solve_failed(self, msg):
        self._busy(False)
        self._fail_history = getattr(self.worker, "history", None)
        self.solve_error = msg.strip().splitlines()[-1]
        self.log("エラー: " + msg)
        self.rebuild_tree()
        QMessageBox.warning(self, "解析エラー", msg[-1500:])

    # ======================================================================= 選択
    def _install_picking(self):
        """クリック（ドラッグでない）を Qt のイベントで検出する。

        VTK の LeftButtonRelease は回転操作に吸収されて届かないことがあるため、Qt 側で拾う。
        """
        from PySide6.QtCore import QEvent, QObject

        win = self

        class ClickFilter(QObject):
            def eventFilter(self, obj, ev):
                t = ev.type()
                if win._sk is not None:
                    dpr = obj.devicePixelRatioF() if hasattr(obj, "devicePixelRatioF") else 1.0
                    if t == QEvent.MouseMove and not ev.buttons():
                        p = ev.position().toPoint()
                        win._sketch_move(int(p.x() * dpr), int((obj.height() - p.y() - 1) * dpr))
                    elif t == QEvent.MouseButtonPress and ev.button() == Qt.RightButton:
                        win._press_r = ev.position().toPoint()
                    elif t == QEvent.MouseButtonRelease and ev.button() == Qt.RightButton:
                        p0 = getattr(win, "_press_r", None)
                        p = ev.position().toPoint()
                        if p0 is not None and abs(p.x() - p0.x()) + abs(p.y() - p0.y()) <= 4:
                            QTimer.singleShot(0, win.sketch_cancel)
                    elif t == QEvent.MouseButtonDblClick and ev.button() == Qt.LeftButton:
                        QTimer.singleShot(0, win._sketch_enter)
                        return True
                if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                    win._press = ev.position().toPoint()
                elif t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton and win._press is not None:
                    p = ev.position().toPoint()
                    moved = abs(p.x() - win._press.x()) + abs(p.y() - win._press.y())
                    win._press = None
                    if moved <= 4:
                        dpr = obj.devicePixelRatioF() if hasattr(obj, "devicePixelRatioF") else 1.0
                        x = int(p.x() * dpr)
                        y = int((obj.height() - p.y() - 1) * dpr)
                        if win._triad_hit(x, y) is not None:
                            QTimer.singleShot(0, lambda: win._triad_click(x, y))
                        elif win._sk is not None:
                            QTimer.singleShot(0, lambda: win._sketch_click(x, y))
                        else:
                            QTimer.singleShot(0, lambda: win._pick((x, y)))
                return False

        self._click_filter = ClickFilter(self)
        self.plotter.interactor.installEventFilter(self._click_filter)
        self.plotter.interactor.setMouseTracking(True)

    # ---------------------------------------------------------------- 座標軸（トライアド）のクリック
    TRIAD_VIEWPORT = 0.2        # pyvista add_axes の既定（画面左下 20% の正方形）

    def _triad_geometry(self):
        """トライアドの中心・半径（画面ピクセル、VTK の座標＝左下原点）と、各軸の先端の位置"""
        ren = self.plotter.renderer
        w, h = ren.GetSize()
        vp = self.TRIAD_VIEWPORT
        cx, cy = 0.5 * vp * w, 0.5 * vp * h
        rad = 0.5 * vp * min(w, h)
        cam = ren.GetActiveCamera()
        d = np.array(cam.GetFocalPoint()) - np.array(cam.GetPosition())
        d /= max(np.linalg.norm(d), 1e-300)
        up = np.array(cam.GetViewUp(), float)
        up -= d * (up @ d)
        up /= max(np.linalg.norm(up), 1e-300)
        right = np.cross(d, up)
        tips = {}
        for i, ax in enumerate("XYZ"):
            for s in (1, -1):
                a = s * np.eye(3)[i]
                tips[("+" if s > 0 else "-") + ax] = (cx + 0.75 * rad * (a @ right), cy + 0.75 * rad * (a @ up),
                                                    float(a @ -d))      # 手前ほど大きい
        return cx, cy, rad, tips

    def _triad_hit(self, x, y):
        """クリック位置がトライアドの上なら "iso" / "+X" などを返す（それ以外は None）"""
        if self.mesh is None:
            return None
        try:
            cx, cy, rad, tips = self._triad_geometry()
        except Exception:
            return None
        if np.hypot(x - cx, y - cy) > 1.05 * rad:
            return None
        if np.hypot(x - cx, y - cy) < 0.18 * rad:
            return "iso"
        best, bd = None, 0.4 * rad
        for k, (tx, ty, depth) in sorted(tips.items(), key=lambda kv: -kv[1][2]):
            dd = np.hypot(x - tx, y - ty)
            if dd < bd - 1e-9:
                best, bd = k, dd
        return best or "none"

    def _triad_click(self, x, y):
        """Ansys と同じく、トライアドの軸をクリックするとその軸の方向から見る（中心は等角表示）。
        すでにその向きから見ているときは反対側から見る"""
        k = self._triad_hit(x, y)
        if k in (None, "none"):
            return
        pl = self.plotter
        if k == "iso":
            pl.view_isometric()
            self.log("視点: 等角")
            pl.render()
            return
        i = "XYZ".index(k[1])
        v = (1.0 if k[0] == "+" else -1.0) * np.eye(3)[i]
        cam = pl.renderer.GetActiveCamera()
        cur = np.array(cam.GetPosition()) - np.array(cam.GetFocalPoint())
        cur /= max(np.linalg.norm(cur), 1e-300)
        if cur @ v > 0.999:
            v = -v
        up = (0.0, 1.0, 0.0) if i == 2 else (0.0, 0.0, 1.0)
        pl.view_vector(v, viewup=up)
        pl.reset_camera()
        pl.render()
        self.log(f"視点: {'+' if v[i] > 0 else '-'}{k[1]} 方向から")

    def _pick(self, pos):
        if self.mesh is None or self._main_actor is None:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        picker.AddPickList(self._main_actor)
        hit = picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cell = picker.GetCellId() if hit else -1
        add = bool(QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier))
        if self.current[0] == "result" and self.result is not None:
            if cell >= 0:
                if self._tri_src is None:          # 断面表示: 切り口の点の値
                    self._probe_section(np.array(picker.GetPickPosition()))
                else:
                    self._probe(np.array(picker.GetPickPosition()), self._tri_src[cell])
            return
        if cell < 0:
            if not add:
                self.set_selection([])
                self.set_body_selection([])
            return
        tri = self._tri_src[cell]
        if self.sel_edge_btn.isChecked() and self._surface_mesh():
            lo, hi = self.mesh.bounds()
            e = self.mesh.nearest_edge(np.array(picker.GetPickPosition()), tol=0.05 * float(np.linalg.norm(hi - lo)))
            if e < 0:
                if not add:
                    self.set_selection([])
                return
            sel = list(self.selected_faces)
            if add:
                sel.remove(e) if e in sel else sel.append(e)
            else:
                sel = [e]
            self.set_selection(sel)
            return
        if self.sel_body_btn.isChecked():
            b = int(self.mesh.tri_body()[tri])
            sel = list(self.selected_bodies)
            if add:
                sel.remove(b) if b in sel else sel.append(b)
            else:
                sel = [b]
            self.set_body_selection(sel)
            return
        f = int(self.mesh.face_ids[tri])
        sel = list(self.selected_faces)
        if add:
            sel.remove(f) if f in sel else sel.append(f)
        else:
            sel = [f]
        self.set_selection(sel)

    def face_list_menu(self):
        """面を一覧から選ぶ（クリックしにくい面や、隠れた面を選ぶとき用）"""
        if self.mesh is None:
            return
        m = QMenu(self)
        for f in self.mesh.face_info():
            c, n = f["centroid"], f["normal"]
            if f.get("edge"):
                txt = (f"辺{f['id'] + 1}  長さ {f['area']:.4g} mm  中心 ({c[0]:.3g}, {c[1]:.3g}, {c[2]:.3g})")
                a = QAction(txt, m)
                a.setCheckable(True)
                a.setChecked(f["id"] in self.selected_faces)
                a.toggled.connect(lambda on, fid=f["id"]: self.set_selection(
                    (set(self.selected_faces) | {fid}) if on else (set(self.selected_faces) - {fid})))
                m.addAction(a)
                continue
            txt = (f"面{f['id'] + 1}  {'平面' if f['flat'] else '曲面'}  面積 {f['area']:.4g} mm²  "
                   f"中心 ({c[0]:.3g}, {c[1]:.3g}, {c[2]:.3g})"
                   + (f"  法線 ({n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f})" if f["flat"] else ""))
            a = QAction(txt, m)
            a.setCheckable(True)
            a.setChecked(f["id"] in self.selected_faces)
            a.toggled.connect(lambda on, fid=f["id"]: self.set_selection(
                (set(self.selected_faces) | {fid}) if on else (set(self.selected_faces) - {fid})))
            m.addAction(a)
        m.exec(self.face_list_btn.mapToGlobal(self.face_list_btn.rect().bottomLeft()))

    def select_body(self):
        """全選択（ボディモードなら全ボディ、面モードなら全ての面）"""
        if self.mesh is None:
            return
        if self.sel_body_btn.isChecked():
            self.set_body_selection(self.mesh.body_ids().tolist())
        else:
            self.set_selection(list(range(self.mesh.n_faces)))

    def set_body_selection(self, bodies):
        seen = []
        for b in bodies:   # 選んだ順を保つ（ブール演算の「対象」は最初のボディ）
            if int(b) not in seen:
                seen.append(int(b))
        self.selected_bodies = seen
        self._sync_attr_from_selection()
        self._update_sel_status()
        self.refresh_view()

    def _update_sel_status(self):
        if self.sel_body_btn.isChecked():
            if self.selected_bodies:
                names = []
                for b in self.selected_bodies:
                    a = self.project.assignment(b)
                    names.append(f"B{b}[{a.get('attr') or '未定義'}/{a.get('mat') or '未定義'}]")
                self.sel_status.setText(f"{len(self.selected_bodies)} ボディを選択中: " + "  ".join(names))
            else:
                self.sel_status.setText("選択なし（ボディ選択モード）")
            return
        self.set_selection(self.selected_faces, refresh=False)

    def set_selection(self, faces, refresh=True):
        self.selected_faces = sorted(set(int(f) for f in faces))
        if self.mesh is not None and self.selected_faces:
            info = {f["id"]: f for f in self.mesh.face_info()}
            ne = sum(1 for f in self.selected_faces if info.get(f, {}).get("edge"))
            A = sum(info[f]["area"] for f in self.selected_faces if f in info and not info[f].get("edge"))
            Lsum = sum(info[f]["area"] for f in self.selected_faces if f in info and info[f].get("edge"))
            nf = len(self.selected_faces) - ne
            txt = (f"{nf} 面" + (f"・{ne} 辺" if ne else "") + "を選択中　"
                   + (f"面積 = {A:.6g} mm²" if nf else "") + (f"　長さ = {Lsum:.6g} mm" if ne else ""))
            if len(self.selected_faces) == 1:
                c = info[self.selected_faces[0]]["centroid"]
                txt += f"　重心 = ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g}) mm"
        else:
            txt = "選択なし"
        self.sel_status.setText(txt)
        if refresh:
            self.refresh_view()

    # ======================================================================= 描画
    def toggle_animation(self, *_):
        """変形のアニメーション（静解析: 0 → 1 → 0、モード: −1 → 1）"""
        on = self.anim_btn.isChecked()
        if not on or self.result is None or self.current[0] != "result":
            self._animating = False
            if hasattr(self, "_anim_timer"):
                self._anim_timer.stop()
            self.anim_btn.setChecked(False)
            self.refresh_view()
            return
        self._animating = True
        self._anim_t = 0
        self.refresh_view()            # 輪郭・札なしで描き直す
        if not hasattr(self, "_anim_timer"):
            self._anim_timer = QTimer(self)
            self._anim_timer.timeout.connect(self._anim_step)
        self._anim_timer.start(50)

    def _anim_step(self):
        a = getattr(self, "_anim", None)
        if a is None or not getattr(self, "_animating", False):
            return
        poly, X0, u, scale, modal_ = a
        if scale <= 0 or poly.n_points != len(X0):
            return
        self._anim_t = (self._anim_t + 1) % 30
        ph = 2 * np.pi * self._anim_t / 30
        f = np.sin(ph) if modal_ else 0.5 * (1 - np.cos(ph))
        poly.points = X0 + (f * scale) * u
        self.plotter.render()

    def _deform_scale(self):
        v = self.scale_combo.currentData()
        if v is None or self.result is None:
            return 0.0
        if v < 0:
            lo, hi = self.mesh.bounds()
            umax = self.result.disp_mag.max()
            if umax <= 0:
                return 1.0
            s = 0.05 * np.linalg.norm(hi - lo) / umax
            p = 10 ** np.floor(np.log10(s))
            return float(max(1.0, np.floor(s / p) * p))
        return float(v)

    def refresh_view(self, *_):
        if self.mesh is None:
            self._draw_empty()
            return
        pl = self.plotter
        pl.clear()
        pl.add_axes()
        self._main_actor = None
        k = self.current[0]
        if k == "result" and self.result is not None:
            self._draw_result()
        else:
            self._draw_model(show_mesh=(k in ("mesh", "sizing") or self.show_mesh_btn.isChecked()))
        pl.render()

    def _draw_empty(self):
        """形状がないときの「まっさらな」作業画面（グリッドと原点の座標軸）"""
        pl = self.plotter
        pl.clear()
        pl.add_axes()
        self._main_actor = None
        grid = pv.Plane(center=(50, 50, 0), direction=(0, 0, 1), i_size=200, j_size=200,
                        i_resolution=20, j_resolution=20)
        pl.add_mesh(grid, style="wireframe", color="#9fb3c8", line_width=0.5, pickable=False)
        for d, col in [((60, 0, 0), "red"), ((0, 60, 0), "green"), ((0, 0, 60), "blue")]:
            pl.add_mesh(pv.Arrow(start=(0, 0, 0), direction=d, scale=60, tip_length=0.15, shaft_radius=0.01,
                                 tip_radius=0.03), color=col, pickable=False)
        _labels(pl, [(66, 0, 0), (0, 66, 0), (0, 0, 66)], ["X", "Y", "Z"], 14, "black")
        self._set_overlay("<b>新しいモデル</b><br>リボンの［モデル］タブの「作成」から<br>"
                          "直方体・円柱などを作ります（それぞれ別のボディになります）。<br>"
                          "ボディを選んで ボディ属性名・材料名 を入力し［設定］。<br>"
                          "面を選んで 境界条件名 を入力し［設定］。")
        pl.view_isometric()
        pl.reset_camera()
        pl.render()

    def _draw_model(self, show_mesh=False):
        pl, mesh = self.plotter, self.mesh
        poly, src = surface_polydata(mesh)
        self._tri_src = src
        fid = mesh.face_ids[src]
        rgb = np.tile(_hex(GEOM_COLOR), (len(src), 1))
        k = self.current[0]
        overlay = []
        # 荷重・支持の表示（静的構造または個別の項目を選んでいるとき）
        if k in ("analysis", "bc", "settings", "solution", "info"):
            show = range(len(self.project.bcs)) if k != "bc" else [self.current[1]]
            pts, labs = [], []
            for n, i in enumerate(show):
                bc = self.project.bcs[i]
                if bc.kind == "thermal" and k == "bc":
                    tb = mesh.tri_body()[src]
                    for b in (bc.bodies or mesh.body_ids().tolist()):
                        rgb[tb == b] = 0.5 * _hex(BC_COLORS["thermal"]) + 0.5
                is_sh = getattr(mesh, "is_shell", False)
                for f in bc.faces:
                    if is_sh and mesh.is_edge(f):
                        seg = mesh.edges[f - mesh.n_surf]
                        lines = np.hstack([np.full((len(seg), 1), 2), seg]).ravel()
                        pl.add_mesh(pv.PolyData(mesh.nodes.copy(), lines=lines), color=BC_COLORS[bc.kind],
                                    line_width=6, pickable=False)
                    else:
                        rgb[fid == f] = _hex(BC_COLORS[bc.kind])
                L = letters(i)
                overlay.append(f"<span style='color:{BC_COLORS[bc.kind]}'>■</span> {L}: "
                               f"{bc.name or BC_JP[bc.kind]}{self._bc_value_text(bc)}")
                p = self._bc_anchor(bc)
                if p is not None:
                    pts.append(p)
                    labs.append(L)
            self._draw_bc_glyphs(show)
        elif k == "sizing":
            ls = self.project.local_sizes[self.current[1]]
            for f in ls.get("faces", []):
                rgb[fid == f] = _hex("#3a8fa8")
            if ls.get("scope") == "bodies" and mesh.elem_body is not None:
                tb = mesh.tri_body()[src]
                for b in ls.get("bodies", []):
                    rgb[tb == b] = _hex("#3a8fa8")
            if ls.get("type") == "sphere":
                pl.add_mesh(pv.Sphere(radius=float(ls.get("radius", 1)), center=ls.get("center", [0, 0, 0])),
                            color="#3a8fa8", opacity=0.25, pickable=False)
        elif k in ("connections", "contacts", "contact"):
            # Ansys と同じく 接触側＝赤、ターゲット側＝青（他は半透明風の淡色）
            rgb[:] = _hex("#e6e8eb")
            show = range(len(self.project.contacts)) if k != "contact" else [self.current[1]]
            for i in show:
                c = self.project.contacts[i]
                if c.suppressed:
                    continue
                rgb[np.isin(fid, c.contact_faces)] = _hex(CONTACT_COLOR)
                rgb[np.isin(fid, c.target_faces)] = _hex(TARGET_COLOR)
            if k == "contact":
                c = self.project.contacts[self.current[1]]
                overlay = [f"<b>{c.name}</b>", f"タイプ: {CONTACT_TYPES[c.type]}"
                           + (f"（μ = {c.mu:g}）" if c.type == "frictional" else ""),
                           f"<span style='color:{CONTACT_COLOR}'>■</span> 接触 {len(c.contact_faces)} 面",
                           f"<span style='color:{TARGET_COLOR}'>■</span> ターゲット {len(c.target_faces)} 面"]
            else:
                overlay = ["<b>接続</b>", f"接触領域 {len(self.project.contacts)}",
                           f"<span style='color:{CONTACT_COLOR}'>■</span> 接触　"
                           f"<span style='color:{TARGET_COLOR}'>■</span> ターゲット"]
        elif k in ("geometry", "feature", "attrs", "attr", "mats", "mat", "model", "project", "coords", "part",
                   "share") \
                and mesh.elem_body is not None:
            # ボディごとに材料の色で塗り分け（未定義は灰色）
            tb = mesh.tri_body()[src]
            mats = list(self.project.materials.keys())
            for b in mesh.body_ids():
                a = self.project.assignment(b)
                col = (MAT_PALETTE[mats.index(a["mat"]) % len(MAT_PALETTE)]
                       if a.get("mat") in self.project.materials else GEOM_COLOR)
                if k == "attr" and a.get("attr") == self.current[1]:
                    col = "#3a8fa8"
                if k == "mat" and a.get("mat") != self.current[1]:
                    col = "#e0e0e0"
                if k == "part":
                    col = "#2f9e44" if b in self.project.parts[self.current[1]].get("bodies", []) else "#e0e0e0"
                rgb[tb == b] = _hex(col)
                a_txt = f"{a.get('attr') or '未定義'} / {a.get('mat') or '未定義'}"
                sw = col if a.get("mat") in self.project.materials else "#999999"
                overlay.append(f"<span style='color:{sw}'>■</span> B{b}: {a_txt}")
            if self.selected_bodies:
                for b in self.selected_bodies:
                    rgb[tb == b] = _hex(SELECT_COLOR)
        for f in self.selected_faces:
            if getattr(mesh, "is_shell", False) and mesh.is_edge(f):
                seg = mesh.edges[f - mesh.n_surf]
                pl.add_mesh(pv.PolyData(mesh.nodes.copy(), lines=np.hstack([np.full((len(seg), 1), 2), seg]).ravel()),
                            color=SELECT_COLOR, line_width=7, pickable=False)
                continue
            rgb[fid == f] = _hex(SELECT_COLOR)
        poly.cell_data["rgb"] = (rgb * 255).astype(np.uint8)
        self._main_actor = pl.add_mesh(poly, scalars="rgb", rgb=True, show_edges=show_mesh,
                                       edge_color="#303030", line_width=0.6, show_scalar_bar=False,
                                       smooth_shading=False, ambient=0.25, diffuse=0.8,
                                       opacity=0.3 if k in ("connections", "contacts", "contact") else 1.0)
        if k in ("connections", "contacts", "contact"):
            # 接触面は部品の内側にあることが多いので、他を半透明にして接触面だけ不透明で描く
            hl = np.zeros(len(src), bool)
            for c in self.project.contacts:
                if not c.suppressed:
                    hl |= np.isin(fid, c.contact_faces) | np.isin(fid, c.target_faces)
            hl |= np.isin(fid, self.selected_faces)
            if hl.any():
                # 接触面とターゲット面は同じ位置にあるので、それぞれ自分のボディ側へ少しずらして重なりのちらつきを防ぐ
                fc = np.asarray(poly.faces).reshape(-1, 4)[:, 1:][hl]
                P = np.asarray(poly.points)[fc]                       # (n, 3, 3)
                cen = P.mean(axis=1)
                tb = mesh.tri_body()[src][hl]
                bc_ = {int(b): mesh.nodes[np.unique(mesh.elems[mesh.elem_body == b])].mean(axis=0)
                       for b in np.unique(tb)}
                toward = np.array([bc_[int(b)] for b in tb]) - cen
                nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
                nrm *= np.sign(np.sum(nrm * toward, axis=1))[:, None]
                lo_, hi_ = mesh.bounds()
                P = P + 0.002 * float(np.linalg.norm(hi_ - lo_)) * nrm[:, None, :]
                n = len(P)
                sub = pv.PolyData(P.reshape(-1, 3), faces=np.c_[np.full(n, 3), np.arange(3 * n).reshape(-1, 3)].ravel())
                sub.cell_data["rgb"] = (rgb[hl] * 255).astype(np.uint8)
                pl.add_mesh(sub, scalars="rgb", rgb=True, pickable=False, show_scalar_bar=False)
        fe = feature_edges(mesh)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=1.5, pickable=False)
        if k in ("analysis", "bc", "settings", "solution", "info") and self.project.bcs:
            if pts:
                _labels(pl, pts, labs, 14, "white", bg="#333333")
            head = f"<b>{self._analysis_label()}</b><br>" + datetime.datetime.now().strftime("%Y/%m/%d %H:%M") + "<br>"
            self._set_overlay(head + "<br>".join(overlay))
        elif k in ("connections", "contacts", "contact"):
            self._set_overlay("<br>".join(overlay))
        elif overlay:
            self._set_overlay("<b>ボディ</b>（ボディ属性 / 材料）<br>" + "<br>".join(overlay))
        elif k in ("mesh", "sizing"):
            self._set_overlay(f"<b>メッシュ</b><br>節点 {mesh.n_nodes:,}<br>要素 {mesh.n_elems:,}"
                              + ("<br><span style='color:#b8860b'>⚡ 再生成が必要</span>" if self.mesh_dirty else ""))
        else:
            self._set_overlay(None)

    @staticmethod
    def _bc_value_text(bc):
        if bc.kind in ("fixed", "symmetry", "frictionless"):
            return ""
        if bc.kind == "pressure":
            return f": {bc.values[0]:g} MPa"
        if bc.kind == "thermal":
            return f": {bc.values[0]:g} °C"
        if bc.kind in BC_SPEC:
            fl = BC_SPEC[bc.kind][0]
            if not fl:
                return ""
            if len(fl) == 1 or bc.kind in ("convection", "radiation", "spring"):
                return ": " + ", ".join(f"{v:g} {u}".strip() for v, (_, u, _, _) in zip(bc.values, fl) if v is not None)
            mag = float(np.linalg.norm([v or 0 for v in bc.values[:3]]))
            return f": {mag:g} {fl[0][1]}"
        vals = [v for v in bc.values[:3]]
        if bc.kind in ("force", "moment", "gravity"):
            mag = float(np.linalg.norm([v or 0 for v in vals]))
            return f": {mag:g} {BC_UNIT[bc.kind]}"
        return ": " + ", ".join("自由" if v is None else f"{v:g}" for v in vals) + " mm"

    def _bc_anchor(self, bc):
        if not bc.faces:
            if bc.kind in NO_FACE_KINDS:
                lo, hi = self.mesh.bounds()
                return 0.5 * (lo + hi)
            return None
        info = {f["id"]: f for f in self.mesh.face_info()}
        c = self.mesh.tri_centroids()
        cs = [info[f]["centroid"] for f in bc.faces if f in info]
        if not cs:
            return None
        p = np.mean(cs, axis=0)
        m = np.isin(self.mesh.face_ids, bc.faces)
        cc = c[m]
        if len(cc) == 0:
            return p
        return cc[np.argmin(np.linalg.norm(cc - p, axis=1))]

    def _draw_bc_glyphs(self, which):
        pl, mesh = self.plotter, self.mesh
        lo, hi = mesh.bounds()
        L = float(np.linalg.norm(hi - lo))
        for i in which:
            bc = self.project.bcs[i]
            c = self._bc_anchor(bc)
            if c is None:
                continue
            v = None
            if bc.kind in ("force", "gravity"):
                v = np.asarray([x or 0 for x in bc.values[:3]], float)
            elif bc.kind == "pressure" and bc.faces:
                info = {f["id"]: f for f in mesh.face_info()}
                n = sum(info[f]["normal"] * info[f]["area"] for f in bc.faces if f in info)
                v = -np.sign(bc.values[0] or 1) * n
            elif bc.kind == "moment":
                v = np.asarray([x or 0 for x in bc.values[:3]], float)
            if v is None or np.linalg.norm(v) == 0:
                continue
            d = v / np.linalg.norm(v)
            col = BC_COLORS[bc.kind]
            start = c - d * 0.2 * L if bc.kind != "moment" else c
            pl.add_mesh(pv.Arrow(start=start, direction=d, scale=0.2 * L), color=col, pickable=False)
            if bc.kind == "moment":
                pl.add_mesh(pv.Arrow(start=c + d * 0.04 * L, direction=d, scale=0.2 * L, tip_length=0.3),
                            color=col, pickable=False)

    def _draw_result(self):
        pl, mesh, r = self.plotter, self.mesh, self.result
        key = self.project.results[self.current[1]]
        fields = self._fields(r)
        if key not in fields:
            why = ("接触の結果がありません" if key.startswith("接触:") else
                   "過渡構造の結果だけにあります" if key.startswith("速度") else "降伏応力が未設定のため表示できません")
            self._set_overlay(f"<b>{RESULT_NAME.get(key, key)}</b><br>（{why}）")
            self._draw_model()
            return
        vals = np.asarray(fields[key], float)
        scale = self._deform_scale()
        pts = mesh.nodes + scale * r.u
        poly, src = surface_polydata(mesh, pts)
        self._tri_src = src
        en = FIELD_EN.get(key, key)
        poly.point_data[en] = vals
        used = np.unique(mesh.boundary_tris_full)
        fin = used[np.isfinite(vals[used])]
        if len(fin) == 0:
            self._set_overlay(f"<b>{RESULT_NAME.get(key, key)}</b><br>（値がありません）")
            self._draw_model()
            return
        used = fin
        vmin, vmax = float(vals[used].min()), float(vals[used].max())
        if vmax - vmin < 1e-30:
            vmax = vmin + 1e-30
        n_colors = self.bands.currentData() or 9
        cmap = "jet_r" if "Safety" in en else "jet"
        real = (vmin, vmax)
        if key == "接触:接触状態":
            n_colors, vmin, vmax = 4, -0.5, 3.5
        show_el = self.edges_combo.currentIndex() == 2
        if key.startswith("接触:"):
            # 接触面だけをコンター表示し、他は半透明（接触面は部品の内側にあるため）
            fc = np.asarray(poly.faces).reshape(-1, 4)[:, 1:]
            mask = np.all(np.isfinite(vals[fc]), axis=1)
            rest = pv.PolyData(poly.points, faces=np.c_[np.full((~mask).sum(), 3), fc[~mask]].ravel())
            if rest.n_cells:
                pl.add_mesh(rest, color="#bfc4ca", opacity=0.12, pickable=False, show_scalar_bar=False)
            sub = pv.PolyData(poly.points, faces=np.c_[np.full(mask.sum(), 3), fc[mask]].ravel())
            sub.point_data[en] = vals
            poly = sub
            self._tri_src = src[mask]
        sec_ax = self.section_combo.currentData() if hasattr(self, "section_combo") else None
        if sec_ax is not None and not key.startswith("接触:") and not getattr(mesh, "is_shell", False) \
                and not getattr(mesh, "is_2d", False):
            # 断面: 要素（体積）を平面で切り、切り口と手前側の外面にコンター
            grid = volume_grid(mesh, pts)
            grid.point_data[en] = vals
            lo_, hi_ = mesh.bounds()
            org = 0.5 * (lo_ + hi_)
            org[sec_ax] = lo_[sec_ax] + (hi_[sec_ax] - lo_[sec_ax]) * self.section_pos.value() / 100.0
            nrm = np.zeros(3)
            nrm[sec_ax] = 1.0
            clipped = grid.clip(normal=nrm, origin=org + scale * 0.0, invert=True)
            try:
                poly = clipped.extract_surface(nonlinear_subdivision=2, algorithm="dataset_surface")
            except TypeError:                     # 古い pyvista
                poly = clipped.extract_surface(nonlinear_subdivision=2)
            self._tri_src = None
        self._main_actor = pl.add_mesh(
            poly, scalars=en, cmap=cmap, n_colors=n_colors, clim=[vmin, vmax], show_edges=show_el,
            edge_color="#303030", line_width=0.4, smooth_shading=False, nan_color="#d9d9d9",
            scalar_bar_args={"title": "", "vertical": True, "fmt": "%.4g", "n_labels": min(n_colors + 1, 10),
                             "position_x": 0.015, "position_y": 0.25, "height": 0.5, "width": 0.045,
                             "label_font_size": 13, "title_font_size": 1})
        self._anim = (poly, mesh.nodes, np.asarray(r.u, float), scale, "モード" in str(getattr(r, "label", "") or ""))
        if getattr(self, "_animating", False):
            return
        fe = feature_edges(mesh, pts)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=1, pickable=False)
        if self.edges_combo.currentIndex() == 0 and scale > 0:
            fe0 = feature_edges(mesh)
            if fe0 is not None:
                pl.add_mesh(fe0, color="#555555", line_width=1, opacity=0.5, pickable=False)
        imax = used[np.argmax(vals[used])]
        imin = used[np.argmin(vals[used])]
        xmax, xmin = mesh.nodes[imax], mesh.nodes[imin]          # 変形前の座標
        if self.minmax_chk.isChecked():
            _labels(pl, pts[[imax]], [f"Max {vals[imax]:.4g}"], 12, "black", point_color="red", bg="#ffffff")
            _labels(pl, pts[[imin]], [f"Min {vals[imin]:.4g}"], 12, "black", point_color="blue", bg="#ffffff")
        if self.probe_text and getattr(self, "_probe_pt", None) is not None:
            _labels(pl, [pts[self._probe_pt]], [self._probe_val], 12, "black", point_color="magenta", bg="#ffffcc")
        unit = key.split("[")[-1].rstrip("]") if "[" in key else ""
        self._set_overlay(
            f"<b>B: {self._analysis_label()}</b><br>{RESULT_NAME.get(key, key)}<br>"
            f"タイプ: {RESULT_TYPE_JP.get(key, RESULT_NAME.get(key, key))}<br>"
            + (f"単位: {unit}<br>" if unit else "")
            + (f"{r.label}<br>" if getattr(r, "label", "") else f"時間: {getattr(r, 'time', 1.0):g} s<br>")
            + f"{datetime.datetime.now():%Y/%m/%d %H:%M}<br>"
            f"変形倍率: {'変形なし' if scale == 0 else f'{scale:g} 倍'}<br><br>"
            f"<span style='color:#c00'>最大 {real[1]:.5g}</span>"
            f"<br><span style='color:#c00;font-size:11px'>　位置 ({xmax[0]:.4g}, {xmax[1]:.4g}, {xmax[2]:.4g}) mm</span>"
            f"<br><span style='color:#00c'>最小 {real[0]:.5g}</span>"
            f"<br><span style='color:#00c;font-size:11px'>　位置 ({xmin[0]:.4g}, {xmin[1]:.4g}, {xmin[2]:.4g}) mm</span>"
            + ("<br><br>0: 遠い　1: 近い<br>2: すべり　3: 固着" if key == "接触:接触状態" else ""))

    def _probe(self, pos, tri):
        r = self.result
        key = self.project.results[self.current[1]]
        scale = self._deform_scale()
        pts = self.mesh.nodes + scale * r.u
        cand = self.mesh.boundary_tris_full[tri]
        i = int(cand[np.argmin(np.linalg.norm(pts[cand] - pos, axis=1))])
        v = self._fields(r)[key][i]
        x = self.mesh.nodes[i]
        self._probe_pt = i
        self._probe_val = f"{v:.5g}"
        self.probe_text = (f"値: {v:.6g}\n節点: {i + 1}\nX: {x[0]:.5g} mm\nY: {x[1]:.5g} mm\nZ: {x[2]:.5g} mm")
        self.log(f"プローブ 節点{i + 1} ({x[0]:.4g}, {x[1]:.4g}, {x[2]:.4g}) mm: {RESULT_NAME.get(key, key)} = {v:.6g}")
        self.build_details()
        self.refresh_view()

    def _probe_section(self, pos):
        a = getattr(self, "_anim", None)
        if a is None:
            return
        poly = a[0]
        key = self.project.results[self.current[1]]
        en = FIELD_EN.get(key, key)
        j = int(np.argmin(np.linalg.norm(poly.points - pos, axis=1)))
        v = float(poly.point_data[en][j])
        x = poly.points[j]
        self.probe_text = f"値: {v:.6g}（断面）\nX: {x[0]:.5g} mm\nY: {x[1]:.5g} mm\nZ: {x[2]:.5g} mm"
        self.log(f"プローブ（断面） ({x[0]:.4g}, {x[1]:.4g}, {x[2]:.4g}) mm: {RESULT_NAME.get(key, key)} = {v:.6g}")
        self.build_details()

    def _set_overlay(self, html):
        if not html:
            self.overlay.hide()
            return
        self.overlay.setText(html)
        self.overlay.adjustSize()
        self.overlay.move(80 if self.current[0] == "result" else 10, 10)
        self.overlay.show()
        self.overlay.raise_()

    # ======================================================================= ファイル
    def new_project(self):
        if self.project.bcs or self.project.geometry.get("features") or self.project.geometry.get("kind") != "model":
            if QMessageBox.question(self, "新規", "現在のモデルを閉じて、新しいモデルを作りますか？\n"
                                    "（保存していない変更は失われます）") != QMessageBox.Yes:
                return
        self.project = Project()
        self.project_path = None
        self.selected_faces = []
        self.result = None
        self.results = []
        self.current = ("geometry",)
        self._update_title()
        self.remesh(reset_camera=True)
        self._hist_reset()
        self.ribbon.setCurrentIndex(self.tab_model)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "プロジェクトを開く", "", "open-mechanical (*.omech)")
        if not path:
            return
        try:
            self.project = Project.load(path)
        except Exception as e:
            QMessageBox.warning(self, "読込エラー", str(e))
            return
        self.project_path = path
        self.selected_faces = []
        self.current = ("analysis",)
        self._update_title()
        self.remesh(reset_camera=True)
        self._hist_reset()

    def save_project(self):
        if not self.project_path:
            return self.save_project_as()
        self.project.save(self.project_path)
        self.log(f"保存しました: {self.project_path}")

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "名前を付けて保存", "analysis.omech", "open-mechanical (*.omech)")
        if path:
            if not path.endswith(".omech"):
                path += ".omech"
            self.project_path = path
            self._update_title()
            self.save_project()

    def _need_result(self):
        if self.result is None:
            QMessageBox.information(self, "結果なし", "先に「解析」を実行してください。")
            return False
        return True

    def export_vtu(self):
        if self._need_result():
            path, _ = QFileDialog.getSaveFileName(self, "VTU で保存", "result.vtu", "VTK (*.vtu)")
            if path:
                write_vtu(path, self.result)
                self.log(f"書き出しました: {path}")

    def export_csv(self):
        if self._need_result():
            path, _ = QFileDialog.getSaveFileName(self, "CSV で保存", "result.csv", "CSV (*.csv)")
            if path:
                write_csv(path, self.result)
                self.log(f"書き出しました: {path}")

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "画像を保存", "view.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)
            self.log(f"保存しました: {path}")

    def export_report(self):
        if not self._need_result():
            return
        path, _ = QFileDialog.getSaveFileName(self, "レポートを保存", "report.html", "HTML (*.html)")
        if not path:
            return
        tmp = tempfile.mkdtemp()
        images = {}
        prev = self.current
        try:
            self.current = ("analysis",)
            self.refresh_view()
            images["荷重・支持"] = os.path.join(tmp, "bc.png")
            self.plotter.screenshot(images["荷重・支持"])
            for i, key in enumerate(self.project.results):
                if key in self._fields(self.result):
                    self.current = ("result", i)
                    self.refresh_view()
                    p = os.path.join(tmp, f"r{i}.png")
                    self.plotter.screenshot(p)
                    images[RESULT_NAME.get(key, key)] = p
        finally:
            self.current = prev
            self.refresh_view()
        write_report(path, self.result, self.project.bcs, self.project.title, images,
                     self.project.geometry_desc())
        self.log(f"レポートを作成しました: {path}")
        try:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            pass

    def show_help(self):
        QMessageBox.information(self, "使い方", (
            "Ansys Mechanical と同じ流れで操作します。\n\n"
            "1. アウトラインの「ジオメトリ」: 詳細で形状を選ぶか、リボン「インポート」で STEP/STL を読込\n"
            "2. 「材料」: 詳細で材料を選択\n"
            "3. 「メッシュ」: 要素サイズを指定（穴や角は「面サイズ」で細かく）\n"
            "4. グラフィックスで面を左クリック → リボン「支持」「荷重」から追加\n"
            "   （後から面を変えるときは、面を選んで詳細の「適用」）\n"
            "5. リボン「⚡ 解析」\n"
            "6. 「解析結果」の下の項目を選ぶとコンター表示。クリックでその点の値（プローブ）\n\n"
            "マウス: 左クリック=選択 / Ctrl(⌘)+クリック=追加 / 左ドラッグ=回転 / Shift+ドラッグ=移動 / ホイール=拡大縮小\n"
            "ツリーの印: ✓=完了  ⚡=更新が必要  ?=設定が足りない"))

    def closeEvent(self, e):
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(e)


# ---------------------------------------------------------------------------- 起動

def _setup_crash_log():
    """異常終了したときの原因を logs/crash.log に残す"""
    import faulthandler
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, "crash.log"), "a", buffering=1)
    f.write(f"\n=== 起動 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  Python {sys.version.split()[0]} ===\n")
    faulthandler.enable(file=f, all_threads=True)
    return f


def _install_excepthook(crash_file):
    """GUI 内で起きた例外を logs/crash.log とメッセージ欄に残す（アプリは落とさない）"""
    def hook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        try:
            crash_file.write(f"[{datetime.datetime.now():%H:%M:%S}] 例外\n{text}\n")
        except Exception:
            pass
        sys.__stderr__.write(text)
        w = QApplication.activeWindow()
        if isinstance(w, MainWindow):
            w.log("内部エラー: " + text.strip().splitlines()[-1])
    sys.excepthook = hook


def _light_palette(app):
    """Ansys と同じ明るい配色に固定（macOS のダークモードでも）"""
    from PySide6.QtGui import QPalette
    p = QPalette()
    c = {QPalette.Window: "#f0f0f0", QPalette.WindowText: "#1a1a1a", QPalette.Base: "#ffffff",
         QPalette.AlternateBase: "#f5f7fa", QPalette.Text: "#1a1a1a", QPalette.Button: "#f3f3f3",
         QPalette.ButtonText: "#1a1a1a", QPalette.Highlight: "#3d7bd9", QPalette.HighlightedText: "#ffffff",
         QPalette.ToolTipBase: "#ffffe1", QPalette.ToolTipText: "#1a1a1a", QPalette.PlaceholderText: "#888888"}
    for role, col in c.items():
        p.setColor(role, QColor(col))
    app.setPalette(p)


def main():
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    _crash = _setup_crash_log()
    _install_excepthook(_crash)
    try:
        pv.global_theme.allow_empty_mesh = True
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("open-mechanical")
    app.setStyle("Fusion")
    _light_palette(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

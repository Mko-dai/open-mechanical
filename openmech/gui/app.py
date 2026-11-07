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
import os
import sys
import tempfile
import traceback

import numpy as np
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSplitter, QTabWidget, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

import pyvista as pv
from pyvistaqt import QtInteractor

from .. import __version__
from ..export import write_csv, write_report, write_vtu
from ..geometry import (AXES, CAD_EXT, FEATURES, OPS, PLANES, PRIMITIVES, GeometryError, build_mesh,
                        feature_body_keys, gmsh_available, new_feature, primitive_to_features)
from ..materials import LIBRARY, Material
from ..project import Project
from ..solver import BC, solve
from .femtet_dialogs import BCDialog, BodyAttrDialog, FeatureDialog, MaterialDialog
from .viewer import FIELD_EN, feature_edges, surface_polydata

# ---------------------------------------------------------------------------- 定義

BC_JP = {"fixed": "固定支持", "displacement": "変位", "force": "力", "pressure": "圧力",
         "moment": "モーメント", "gravity": "標準地球重力"}
BC_COLORS = {"fixed": "#5a4fcf", "displacement": "#e8b100", "force": "#e03131",
             "pressure": "#d6336c", "moment": "#ae3ec9", "gravity": "#f08c00"}
BC_UNIT = {"displacement": "mm", "force": "N", "pressure": "MPa", "moment": "N·mm", "gravity": "mm/s²"}

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
}
RESULT_NAME = {k: v for g in RESULT_GROUPS.values() for k, v in g}
RESULT_TYPE_JP = {
    "全変形 [mm]": "全変形", "相当応力（ミーゼス） [MPa]": "相当 (von-Mises) 応力",
    "安全率（降伏基準）": "安全率（最大相当応力基準）",
}
OP_COLORS = {"new": "#4a78c2", "add": "#2c7fb8", "cut": "#c0392b", "intersect": "#8e44ad", "union": "#2c7fb8"}
MAT_PALETTE = ["#8fb8de", "#e5b25d", "#a7d3a6", "#d9a0c8", "#9fd6d2", "#e8a08f", "#b9b3e6", "#c9d68a"]
GEOM_COLOR = "#c9cdd3"
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
        g.addWidget(self.plotter.interactor, 2, 0)
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
        g = R.add_group(t, "ジオメトリ")
        R.button(g, "インポート", make_icon("#4a78c2", "読"), self.import_file)
        R.button(g, "簡単形状", make_icon("#4a78c2", "形"), lambda: self.select_node(("geometry",)))
        g = R.add_group(t, "メッシュ")
        R.button(g, "生成", make_icon("#3a8fa8", "網"), lambda: self.remesh())
        R.button(g, "面サイズ", make_icon("#3a8fa8", "細"), self.add_sizing)
        g = R.add_group(t, "挿入")
        R.button(g, "支持", make_icon(BC_COLORS["fixed"], "支"), menu=self._bc_menu(["fixed", "displacement"]))
        R.button(g, "荷重", make_icon(BC_COLORS["force"], "荷"),
                 menu=self._bc_menu(["force", "pressure", "moment", "gravity"]))
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
        g = R.add_group(t, "ブール演算")
        R.button(g, "結合", make_icon(OP_COLORS["union"], "∪"), lambda: self.boolean("union"))
        R.button(g, "除去", make_icon(OP_COLORS["cut"], "−"), lambda: self.boolean("cut"))
        R.button(g, "穴", make_icon(OP_COLORS["cut"], "穴"), self.create_hole)
        g = R.add_group(t, "インポート")
        R.button(g, "STEP", make_icon(OP_COLORS["new"], "読"), self.add_import_feature)
        g = R.add_group(t, "編集")
        R.button(g, "形状編集", make_icon("#5d6d7e", "編"), self.edit_selected_body)
        R.button(g, "削除", make_icon("#888888", "×"), self.delete_current)
        R.button(g, "再生成", make_icon("#3a8fa8", "⟳"), lambda: self.remesh())
        g = R.add_group(t, "属性")
        R.button(g, "ボディ属性\n材料定数", make_icon("#b07b2c", "材"), lambda: self.set_attr_mode(0))
        R.button(g, "境界条件", make_icon(BC_COLORS["fixed"], "境"), lambda: self.set_attr_mode(1))
        self.tab_model = R.count() - 1
        # --- 環境（静的構造）
        t = R.add_tab("環境")
        g = R.add_group(t, "静的構造")
        R.button(g, "荷重", make_icon(BC_COLORS["force"], "荷"),
                 menu=self._bc_menu(["force", "pressure", "moment", "gravity"]))
        R.button(g, "支持", make_icon(BC_COLORS["fixed"], "支"), menu=self._bc_menu(["fixed", "displacement"]))
        for k in ["fixed", "displacement", "force", "pressure", "moment", "gravity"]:
            R.button(g, BC_JP[k], make_icon(BC_COLORS[k], BC_JP[k][0]), lambda _=False, k=k: self.add_bc(k))
        g = R.add_group(t, "解析")
        R.button(g, "解析", make_icon("#2e9d3a", "⚡"), self.run_solve)
        self.tab_env = R.count() - 1
        # --- 解析結果
        t = R.add_tab("解析結果")
        g = R.add_group(t, "挿入")
        for grp, items in RESULT_GROUPS.items():
            m = QMenu(self)
            for key, name in items:
                menu_action(m, name, lambda k=key: self.add_result(k))
            R.button(g, grp, make_icon("#c0392b" if grp == "応力" else "#2c7fb8" if grp == "変形" else "#8e6c2e",
                                       grp[0]), menu=m)
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

    def set_select_mode(self, mode):
        self.sel_body_btn.setChecked(mode == "body")
        self.sel_face_btn.setChecked(mode == "face")
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
        m = dlg.material()
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
                name, kind, vals = dlg.result()
                self.project.bcs.append(BC(kind, list(self.selected_faces), vals, name))
                self.log(f"境界条件「{name}」を作成しました")
            self.bc_name.setEditText(name)
        self._fill_attr_combos()
        self.invalidate()
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
        m = dlg.material()
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
        dlg = BCDialog(bc.name, bc.kind, bc.values, self, n_faces=len(bc.faces))
        if dlg.exec():
            bc.name, bc.kind, bc.values = dlg.result()
            self.bc_name.setEditText(bc.name)
            self._fill_attr_combos()
            self.invalidate()
            self.refresh_view()

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
                         ("画像を保存…", self.save_screenshot)]:
            a = QAction(text, self)
            a.triggered.connect(fn)
            fm.addAction(a)
        em = mb.addMenu("編集")
        a = QAction("削除", self)
        a.setShortcut(QKeySequence.Delete)
        a.triggered.connect(self.delete_current)
        em.addAction(a)
        a = QAction("すべての面を選択", self)
        a.setShortcut(QKeySequence.SelectAll)
        a.triggered.connect(self.select_body)
        em.addAction(a)
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
        self.setWindowTitle(f"{name} — 静的構造 — open-mechanical {__version__}")

    def invalidate(self, mesh=False):
        """設定が変わったら結果を無効にする（Ansys の黄色い ⚡ 状態）"""
        if mesh:
            self.mesh_dirty = True
        self.result = None
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
        if k == "body":
            return STATUS_OK if self.project.assignment(node[1]).get("mat") in self.project.materials else STATUS_ERROR
        if k in ("mesh", "sizing"):
            if k == "sizing" and not self.project.local_sizes[node[1]]["faces"]:
                return STATUS_ERROR
            return STATUS_PENDING if self.mesh_dirty else STATUS_OK
        if k == "bc":
            bc = self.project.bcs[node[1]]
            return STATUS_ERROR if (bc.kind != "gravity" and not bc.faces) else STATUS_OK
        if k == "analysis":
            bad = any(self.status_of(("bc", i)) == STATUS_ERROR for i in range(len(self.project.bcs)))
            return STATUS_ERROR if bad or not self._has_support() else STATUS_OK
        if k in ("solution", "result", "info"):
            if self.solve_error:
                return STATUS_ERROR
            return STATUS_OK if self.result is not None else STATUS_PENDING
        return STATUS_NONE

    def _has_support(self):
        return any(b.kind in ("fixed", "displacement") for b in self.project.bcs)

    # ======================================================================= ツリー
    def rebuild_tree(self, keep=True, details=True):
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
        at = add(model, "ボディ属性", ("attrs",), "#7a8a99", "属")
        for name in self.project.body_attrs:
            n = sum(1 for b in live if self.project.assignment(b).get("attr") == name)
            add(at, f"{name}（{n} ボディ）", ("attr", name), "#7a8a99", "属")
        mt = add(model, "材料定数", ("mats",), "#b07b2c", "材")
        for name in self.project.materials:
            n = sum(1 for b in live if self.project.assignment(b).get("mat") == name)
            add(mt, f"{name}（{n} ボディ）", ("mat", name), "#b07b2c", "材")
        add(model, "座標系", ("coords",), "#6c7a89", "XY")
        mesh = add(model, "メッシュ", ("mesh",), "#3a8fa8", "網")
        for i, ls in enumerate(self.project.local_sizes):
            add(mesh, f"面サイズ指定 {i + 1}", ("sizing", i), "#3a8fa8", "細")
        an = add(model, "静的構造 (A5)", ("analysis",), "#2e6db4", "S")
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
        elif k in ("analysis", "bc", "settings"):
            self.ribbon.setCurrentIndex(self.tab_env)
        elif k in ("solution", "result", "info"):
            self.ribbon.setCurrentIndex(self.tab_res)
        self.refresh_view()

    def on_tree_double(self, item, _col=0):
        node = tuple(item.data(0, Qt.UserRole))
        if node[0] == "feature":
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
            menu_action(m, "挿入 → 面サイズ指定", self.add_sizing)
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
        if k in ("bc", "result", "sizing", "feature", "attr", "mat"):
            m.addSeparator()
            menu_action(m, "削除", self.delete_current)
        if not m.isEmpty():
            m.exec(self.tree.viewport().mapToGlobal(pos))

    # ======================================================================= 詳細パネル
    def build_details(self):
        d = self.details
        d.clear()
        k = self.current[0]
        title = {"geometry": "ジオメトリ", "material": "材料", "mesh": "メッシュ", "sizing": "面サイズ指定",
                 "analysis": "静的構造 (A5)", "settings": "解析設定", "solution": "解析結果 (A6)",
                 "info": "解析情報", "project": "プロジェクト", "model": "モデル (A4)", "coords": "座標系"}
        if k == "feature":
            f = self.project.geometry["features"][self.current[1]]
            name = (f"{FEATURES[f['type']][0]}{self.current[1] + 1}" if f["type"] in FEATURES
                    else "ブール演算")
        elif k in ("attr", "mat"):
            name = self.current[1]
        elif k in ("attrs", "mats"):
            name = "ボディ属性" if k == "attrs" else "材料定数"
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
            n = len(g.get("features", []))
            self._row(s, "フィーチャー数", n)
            self._row(s, "", "リボンの「モデリング」タブで形状を追加します" if n == 0
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

    def create_body(self, ftype):
        """［作成］→ ダイアログで基準点と寸法を入力 → 新しいボディ"""
        if not self._ensure_model():
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
        f.update(dlg.values())
        self._features().append(f)
        self.selected_faces, self.selected_bodies = [], []
        self.current = ("feature", len(self._features()) - 1)
        self.remesh(reset_camera=len(self._features()) == 1)

    def create_hole(self):
        """選択したボディ（未選択なら全ボディ）に円柱の穴をあける"""
        if self.mesh is None or not self._ensure_model():
            QMessageBox.information(self, "穴", "先に形状を作成してください。")
            return
        lo, hi = self._selected_bounds()
        c, sz = 0.5 * (lo + hi), hi - lo
        f = new_feature("cylinder", "cut", x=float(c[0]), y=float(c[1]), z=float(lo[2] - 1), axis="Z",
                        h=float(sz[2] + 2), r=float(max(min(sz[0], sz[1]) / 6, 0.5)))
        targets = list(self.selected_bodies)
        txt = ("対象ボディ: " + ", ".join(f"B{b}" for b in targets)) if targets else "対象ボディ: すべて"
        dlg = FeatureDialog("cylinder", f, "（穴）", self, targets_text=txt)
        if not dlg.exec():
            return
        f.update(dlg.values())
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
            f.update(dlg.values())
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
        op = QComboBox()
        for k, v in OPS.items():
            op.addItem(v, k)
        op.setCurrentIndex(max(0, op.findData(f.get("op", "add"))))
        op.currentIndexChanged.connect(lambda _: self._feature_set(i, "op", op.currentData(), rebuild=True))
        self._row(s, "操作", op)
        s2 = self._section("寸法")
        for key, label, default in params:
            v = f.get(key, default)
            if key == "axis":
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
                w = QLineEdit(str(v))
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
                w = dspin(float(v), -1e7, 1e7, 3, 1.0)
                w.valueChanged.connect(lambda val, key=key: self._feature_set(i, key, val))
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
        if self.mesh is not None:
            used = [f"B{b}" for b in self.mesh.body_ids() if self.project.assignment(b).get("mat") == name]
            self._row(self._section("使用しているボディ"), "", ", ".join(used) or "なし")
        b = QPushButton("編集（名前変更・材料DB）…")
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
        if self.mesh is not None:
            used = [f"B{b}" for b in self.mesh.body_ids() if self.project.assignment(b).get("attr") == name]
            self._row(self._section("使用しているボディ"), "", ", ".join(used) or "なし")

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

    def _details_mesh(self):
        s = self._section("既定")
        order = QComboBox()
        order.addItem("2次（Tet10・推奨）", 2)
        order.addItem("1次（Tet4）", 1)
        order.setCurrentIndex(0 if self.project.mesh_order == 2 else 1)
        order.currentIndexChanged.connect(lambda _: self._mesh_set("order", order.currentData()))
        self._row(s, "要素次数", order)
        size = dspin(self.project.mesh_size, 0.0, 1e6, 3, 0.5, "mm")
        size.setSpecialValueText("自動")
        size.valueChanged.connect(lambda v: self._mesh_set("size", v))
        self._row(s, "要素サイズ", size)
        b = QPushButton("メッシュを生成")
        b.clicked.connect(lambda: self.remesh())
        self._row(s, "", b)
        if self.mesh is not None:
            s2 = self._section("統計")
            self._row(s2, "節点数", f"{self.mesh.n_nodes:,}")
            self._row(s2, "要素数", f"{self.mesh.n_elems:,}")
            self._row(s2, "状態", "最新" if not self.mesh_dirty else "再生成が必要（⚡）")

    def _mesh_set(self, what, v):
        if what == "order":
            self.project.mesh_order = v
        else:
            self.project.mesh_size = v
        self.invalidate(mesh=True)

    def _details_sizing(self):
        i = self.current[1]
        ls = self.project.local_sizes[i]
        s = self._section("スコープ")
        self._scope_row(s, ls["faces"], lambda faces: self._sizing_apply(i, faces))
        s2 = self._section("定義")
        sp = dspin(float(ls["size"]), 0.001, 1e6, 3, 0.1, "mm")
        sp.valueChanged.connect(lambda v: self._sizing_size(i, v))
        self._row(s2, "要素サイズ", sp)

    def _sizing_apply(self, i, faces):
        self.project.local_sizes[i]["faces"] = list(faces)
        self.invalidate(mesh=True)

    def _sizing_size(self, i, v):
        self.project.local_sizes[i]["size"] = v
        self.invalidate(mesh=True)

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

    def _details_analysis(self):
        s = self._section("定義")
        self._row(s, "解析タイプ", "静的構造（線形）")
        self._row(s, "物理タイプ", "構造")
        self._row(s, "支持", "あり" if self._has_support() else "なし（固定支持を追加してください）")
        self._row(s, "荷重・支持の数", len(self.project.bcs))

    def _details_settings(self):
        s = self._section("ソルバー制御")
        combo = QComboBox()
        for label, key in [("プログラム制御（推奨）", "auto"), ("直接法", "scipy"), ("反復法（AMG）", "amg")]:
            combo.addItem(label, key)
        combo.setCurrentIndex(max(0, combo.findData(self.project.solver)))
        combo.currentIndexChanged.connect(lambda _: setattr(self.project, "solver", combo.currentData()))
        self._row(s, "ソルバータイプ", combo)
        self._row(s, "大変形", "オフ")
        self._row(s, "ステップ数", "1")

    def _details_bc(self):
        i = self.current[1]
        bc = self.project.bcs[i]
        if bc.kind != "gravity":
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
            self._row(s2, "拘束", "X, Y, Z 方向すべて固定")
        elif bc.kind == "pressure":
            self._row(s2, "定義方法", "垂直方向")
            sp = dspin(float(bc.values[0] or 0), dec=6, suffix=unit)
            sp.valueChanged.connect(lambda v: self._bc_value(i, 0, v))
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
                    sp = dspin(0.0 if v is None else float(v), dec=6, suffix=unit)
                    sp.setEnabled(v is not None)
                    cb.currentIndexChanged.connect(lambda idx, c=c, sp=sp: (sp.setEnabled(idx == 1),
                                                   self._bc_value(i, c, sp.value() if idx == 1 else None)))
                    sp.valueChanged.connect(lambda val, c=c, cb=cb: cb.currentIndex() == 1 and self._bc_value(i, c, val))
                    h.addWidget(cb)
                    h.addWidget(sp)
                    self._row(s2, f"{ax} 成分", w)
                else:
                    sp = dspin(float(v or 0), dec=6 if bc.kind == "gravity" else 4, suffix=unit)
                    sp.valueChanged.connect(lambda val, c=c: self._bc_value(i, c, val))
                    self._row(s2, f"{ax} 成分", sp)
        if self.result is not None and i in self.result.reactions:
            r = self.result.reactions[i]
            s3 = self._section("反力（解析結果）")
            for c, ax in enumerate("XYZ"):
                self._row(s3, f"{ax} 成分", f"{r[c]:.6g} N")
            self._row(s3, "合計", f"{np.linalg.norm(r):.6g} N")

    def _bc_apply(self, i, faces):
        self.project.bcs[i].faces = list(faces)
        self.invalidate()

    def _bc_name(self, i, text):
        if self.project.bcs[i].name != text.strip():
            self.project.bcs[i].name = text.strip()
            self.rebuild_tree(details=False)

    def _bc_value(self, i, c, v):
        vals = list(self.project.bcs[i].values)
        while len(vals) <= c:
            vals.append(0.0)
        vals[c] = v
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

    def _details_result(self):
        key = self.project.results[self.current[1]]
        s = self._section("スコープ")
        self._row(s, "スコープ方法", "ジオメトリの選択")
        self._row(s, "ジオメトリ", "すべてのボディ")
        s2 = self._section("定義")
        self._row(s2, "タイプ", RESULT_TYPE_JP.get(key, RESULT_NAME.get(key, key)))
        self._row(s2, "単位", key.split("[")[-1].rstrip("]") if "[" in key else "なし")
        if self.result is not None:
            vals = np.asarray(self.result.fields().get(key, []), float)
            if len(vals):
                s3 = self._section("結果")
                self._row(s3, "最小", f"{vals.min():.6g}")
                self._row(s3, "最大", f"{vals.max():.6g}")
                self._row(s3, "平均", f"{vals.mean():.6g}")
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
        faces = [] if kind == "gravity" else list(self.selected_faces)
        vals = {"fixed": [0.0, 0.0, 0.0], "displacement": [0.0, None, None], "force": [0.0, 0.0, -1000.0],
                "pressure": [1.0], "moment": [0.0, 0.0, 1000.0], "gravity": [0.0, 0.0, -9806.65]}[kind]
        self.project.bcs.append(BC(kind, faces, vals, ""))
        self.invalidate()
        self.select_node(("bc", len(self.project.bcs) - 1))
        if kind != "gravity" and not faces:
            self.log(f"{BC_JP[kind]} を追加しました。面を選択して詳細の「適用」を押してください。")

    def add_result(self, key):
        self.project.results.append(key)
        self.rebuild_tree()
        self.select_node(("result", len(self.project.results) - 1))

    def add_sizing(self):
        self.project.local_sizes.append({"faces": list(self.selected_faces), "size": 1.0})
        self.invalidate(mesh=True)
        self.select_node(("sizing", len(self.project.local_sizes) - 1))

    def delete_current(self):
        k = self.current[0]
        if k == "feature":
            i = self.current[1]
            feats = self._features()
            keys = feature_body_keys(feats)
            gone = keys.get(i)
            del feats[i]
            if gone is not None:
                # 後ろのボディ番号を詰める（属性の割り当てと参照を付け替え）
                ba = {}
                for kk, v in self.project.body_assign.items():
                    kk = int(kk)
                    if kk == gone:
                        continue
                    ba[str(kk - 1 if kk > gone else kk)] = v
                self.project.body_assign = ba
                fix = lambda x: x - 1 if x > gone else x  # noqa: E731
                for f in feats:
                    if f.get("type") == "bool":
                        f["target"] = fix(int(f["target"]))
                        f["tools"] = [fix(int(t)) for t in f.get("tools", []) if int(t) != gone]
                    if f.get("targets"):
                        f["targets"] = [fix(int(t)) for t in f["targets"] if int(t) != gone]
            self.current = ("geometry",)
            self.selected_bodies = []
            self.remesh()
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
                              local_sizes=self.project.local_sizes + self.project.body_local_sizes(self.mesh))
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

    def run_solve(self):
        if self.worker and self.worker.isRunning():
            return
        bad = [b for b in self.project.bcs if b.kind != "gravity" and not b.faces]
        if bad:
            QMessageBox.warning(self, "解析", "面が割り当てられていない荷重・支持があります（? マーク）。\n"
                                "面を選択して詳細の「適用」を押してください。")
            return
        if not self._has_support():
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
        self._busy(True, "解析を実行しています…")
        self.worker = Worker(solve, self.mesh, self.project.material, list(self.project.bcs),
                             method=self.project.solver, elem_materials=em)
        self.worker.log.connect(lambda m: (self.solver_log.append(m), self.log(m)))
        self.worker.done.connect(self._on_solved)
        self.worker.failed.connect(self._on_solve_failed)
        self.worker.start()

    def _on_solved(self, result):
        self._busy(False)
        self.result = result
        self.solve_error = ""
        self.log("解析が完了しました（" + ", ".join(f"{k} {v}" for k, v in result.info.items()
                                               if k in ("自由度数", "合計時間 [s]")) + "）")
        self.rebuild_tree()
        if self.project.results:
            self.select_node(("result", min(1, len(self.project.results) - 1)))

    def _on_solve_failed(self, msg):
        self._busy(False)
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
                        QTimer.singleShot(0, lambda: win._pick((x, y)))
                return False

        self._click_filter = ClickFilter(self)
        self.plotter.interactor.installEventFilter(self._click_filter)

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
                self._probe(np.array(picker.GetPickPosition()), self._tri_src[cell])
            return
        if cell < 0:
            if not add:
                self.set_selection([])
                self.set_body_selection([])
            return
        tri = self._tri_src[cell]
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
            A = sum(info[f]["area"] for f in self.selected_faces)
            txt = f"{len(self.selected_faces)} 面を選択中　面積 = {A:.6g} mm²"
            if len(self.selected_faces) == 1:
                c = info[self.selected_faces[0]]["centroid"]
                txt += f"　重心 = ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g}) mm"
        else:
            txt = "選択なし"
        self.sel_status.setText(txt)
        if refresh:
            self.refresh_view()

    # ======================================================================= 描画
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
        self._set_overlay("<b>新しいモデル</b><br>リボンの「モデリング」タブから<br>"
                          "直方体・円柱などを追加して形を作ります。<br>"
                          "穴は「切り取り」→「穴」。CAD は「STEP 取込」")
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
                for f in bc.faces:
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
            for f in self.project.local_sizes[self.current[1]]["faces"]:
                rgb[fid == f] = _hex("#3a8fa8")
        elif k in ("geometry", "feature", "attrs", "attr", "mats", "mat", "model", "project", "coords") \
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
                rgb[tb == b] = _hex(col)
                a_txt = f"{a.get('attr') or '未定義'} / {a.get('mat') or '未定義'}"
                sw = col if a.get("mat") in self.project.materials else "#999999"
                overlay.append(f"<span style='color:{sw}'>■</span> B{b}: {a_txt}")
            if self.selected_bodies:
                for b in self.selected_bodies:
                    rgb[tb == b] = _hex(SELECT_COLOR)
        for f in self.selected_faces:
            rgb[fid == f] = _hex(SELECT_COLOR)
        poly.cell_data["rgb"] = (rgb * 255).astype(np.uint8)
        self._main_actor = pl.add_mesh(poly, scalars="rgb", rgb=True, show_edges=show_mesh,
                                       edge_color="#303030", line_width=0.6, show_scalar_bar=False,
                                       smooth_shading=False, ambient=0.25, diffuse=0.8)
        fe = feature_edges(mesh)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=1.5, pickable=False)
        if k in ("analysis", "bc", "settings", "solution", "info") and self.project.bcs:
            if pts:
                _labels(pl, pts, labs, 14, "white", bg="#333333")
            head = "<b>静的構造</b><br>" + datetime.datetime.now().strftime("%Y/%m/%d %H:%M") + "<br>"
            self._set_overlay(head + "<br>".join(overlay))
        elif overlay:
            self._set_overlay("<b>ボディ</b>（ボディ属性 / 材料）<br>" + "<br>".join(overlay))
        elif k in ("mesh", "sizing"):
            self._set_overlay(f"<b>メッシュ</b><br>節点 {mesh.n_nodes:,}<br>要素 {mesh.n_elems:,}"
                              + ("<br><span style='color:#b8860b'>⚡ 再生成が必要</span>" if self.mesh_dirty else ""))
        else:
            self._set_overlay(None)

    @staticmethod
    def _bc_value_text(bc):
        if bc.kind == "fixed":
            return ""
        if bc.kind == "pressure":
            return f": {bc.values[0]:g} MPa"
        vals = [v for v in bc.values[:3]]
        if bc.kind in ("force", "moment", "gravity"):
            mag = float(np.linalg.norm([v or 0 for v in vals]))
            return f": {mag:g} {BC_UNIT[bc.kind]}"
        return ": " + ", ".join("自由" if v is None else f"{v:g}" for v in vals) + " mm"

    def _bc_anchor(self, bc):
        if not bc.faces:
            if bc.kind == "gravity":
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
        fields = r.fields()
        if key not in fields:
            self._set_overlay(f"<b>{RESULT_NAME.get(key, key)}</b><br>（降伏応力が未設定のため表示できません）")
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
        vmin, vmax = float(vals[used].min()), float(vals[used].max())
        if vmax - vmin < 1e-30:
            vmax = vmin + 1e-30
        n_colors = self.bands.currentData() or 9
        cmap = "jet_r" if "Safety" in en else "jet"
        show_el = self.edges_combo.currentIndex() == 2
        self._main_actor = pl.add_mesh(
            poly, scalars=en, cmap=cmap, n_colors=n_colors, clim=[vmin, vmax], show_edges=show_el,
            edge_color="#303030", line_width=0.4, smooth_shading=False,
            scalar_bar_args={"title": "", "vertical": True, "fmt": "%.4g", "n_labels": min(n_colors + 1, 10),
                             "position_x": 0.015, "position_y": 0.25, "height": 0.5, "width": 0.045,
                             "label_font_size": 13, "title_font_size": 1})
        fe = feature_edges(mesh, pts)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=1, pickable=False)
        if self.edges_combo.currentIndex() == 0 and scale > 0:
            fe0 = feature_edges(mesh)
            if fe0 is not None:
                pl.add_mesh(fe0, color="#555555", line_width=1, opacity=0.5, pickable=False)
        imax = used[np.argmax(vals[used])]
        imin = used[np.argmin(vals[used])]
        _labels(pl, pts[[imax]], [f"Max {vals[imax]:.4g}"], 12, "black", point_color="red", bg="#ffffff")
        _labels(pl, pts[[imin]], [f"Min {vals[imin]:.4g}"], 12, "black", point_color="blue", bg="#ffffff")
        if self.probe_text and getattr(self, "_probe_pt", None) is not None:
            _labels(pl, [pts[self._probe_pt]], [self._probe_val], 12, "black", point_color="magenta", bg="#ffffcc")
        unit = key.split("[")[-1].rstrip("]") if "[" in key else ""
        self._set_overlay(
            f"<b>B: 静的構造</b><br>{RESULT_NAME.get(key, key)}<br>"
            f"タイプ: {RESULT_TYPE_JP.get(key, RESULT_NAME.get(key, key))}<br>"
            + (f"単位: {unit}<br>" if unit else "")
            + f"時間: 1 s<br>{datetime.datetime.now():%Y/%m/%d %H:%M}<br>"
            f"変形倍率: {'変形なし' if scale == 0 else f'{scale:g} 倍'}<br><br>"
            f"<span style='color:#c00'>最大 {vmax:.5g}</span><br><span style='color:#00c'>最小 {vmin:.5g}</span>")

    def _probe(self, pos, tri):
        r = self.result
        key = self.project.results[self.current[1]]
        scale = self._deform_scale()
        pts = self.mesh.nodes + scale * r.u
        cand = self.mesh.boundary_tris_full[tri]
        i = int(cand[np.argmin(np.linalg.norm(pts[cand] - pos, axis=1))])
        v = r.fields()[key][i]
        x = self.mesh.nodes[i]
        self._probe_pt = i
        self._probe_val = f"{v:.5g}"
        self.probe_text = (f"値: {v:.6g}\n節点: {i + 1}\nX: {x[0]:.5g} mm\nY: {x[1]:.5g} mm\nZ: {x[2]:.5g} mm")
        self.log(f"プローブ 節点{i + 1} ({x[0]:.4g}, {x[1]:.4g}, {x[2]:.4g}) mm: {RESULT_NAME.get(key, key)} = {v:.6g}")
        self.build_details()
        self.refresh_view()

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
        self.current = ("geometry",)
        self._update_title()
        self.remesh(reset_camera=True)
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
                if key in self.result.fields():
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

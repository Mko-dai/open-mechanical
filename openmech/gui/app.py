"""open-mechanical メインウィンドウ（PySide6 + PyVista）。

操作の流れは Ansys Mechanical と同じく
  形状 → メッシュ → 材料 → 境界条件 → 解析 → 結果
の順に左のタブを進めます。
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

import pyvista as pv
from pyvistaqt import QtInteractor

from .. import __version__
from ..export import write_csv, write_report, write_vtu
from ..geometry import PRIMITIVES, GeometryError, build_mesh, gmsh_available
from ..materials import DEFAULT, LIBRARY, Material
from ..project import Project
from ..solver import BC, solve
from .dialogs import KIND_JP, BCDialog
from .viewer import FACE_COLORS, FIELD_EN, SELECT_COLOR, feature_edges, surface_polydata

BC_COLORS = {"fixed": "#3b6fd8", "displacement": "#3bb0d8", "force": "#e03131",
             "pressure": "#d6336c", "moment": "#ae3ec9", "gravity": "#f08c00"}
CAD_FILTER = ("形状ファイル (*.step *.stp *.STEP *.STP *.igs *.iges *.brep *.stl *.STL *.x_t *.x_b);;"
              "STEP (*.step *.stp *.STEP *.STP);;STL (*.stl *.STL);;すべてのファイル (*)")


class Worker(QThread):
    """重い処理（メッシュ・求解）を別スレッドで実行"""
    progress = Signal(float, str)
    log = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw

    def run(self):
        try:
            out = self.fn(*self.args, log=lambda m: self.log.emit(str(m)), **self.kw)
            self.done.emit(out)
        except (GeometryError, ValueError, RuntimeError) as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc())


def dspin(val, lo=0.0, hi=1e9, dec=3, step=1.0):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    return s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"open-mechanical {__version__} — 構造解析")
        self.resize(1400, 900)
        self.project = Project()
        self.project_path: str | None = None
        self.mesh = None
        self.result = None
        self.selected_faces: list[int] = []
        self.worker: Worker | None = None
        self.mode = "geometry"   # geometry | results
        self._tri_src = None

        # ---------------- 3D ビュー
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#e9eef5", top="#ffffff")
        self.plotter.add_axes()
        try:
            self.plotter.track_click_position(self.on_pick, side="right")
        except TypeError:
            self.plotter.track_click_position(self.on_pick)

        # ---------------- 左パネル
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(380)
        self.tabs.addTab(self._tab_geometry(), "1 形状")
        self.tabs.addTab(self._tab_mesh(), "2 メッシュ")
        self.tabs.addTab(self._tab_material(), "3 材料")
        self.tabs.addTab(self._tab_bc(), "4 境界条件")
        self.tabs.addTab(self._tab_solve(), "5 解析")
        self.tabs.addTab(self._tab_results(), "6 結果")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # ---------------- ログ
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(2000)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.plotter.interactor)
        right.addWidget(self.logbox)
        right.setSizes([720, 140])
        main = QSplitter(Qt.Horizontal)
        main.addWidget(self.tabs)
        main.addWidget(right)
        main.setSizes([400, 1000])
        self.setCentralWidget(main)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self._build_menu()

        self.log(f"open-mechanical {__version__} を起動しました。単位系: mm・N・MPa")
        if not gmsh_available():
            self.log("⚠ gmsh が見つかりません。直方体以外の形状・CAD 読込には `pip install gmsh` が必要です。")
        self.load_project_to_ui()
        self.create_geometry()

    # ======================================================================= UI 構築
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
        a = QAction("形状ファイルを読み込む…", self)
        a.triggered.connect(self.import_file)
        fm.addAction(a)
        fm.addSeparator()
        for text, fn in [("結果を VTU で書き出す（ParaView）…", self.export_vtu),
                         ("結果を CSV で書き出す…", self.export_csv),
                         ("レポートを作成（HTML）…", self.export_report),
                         ("画像を保存…", self.save_screenshot)]:
            a = QAction(text, self)
            a.triggered.connect(fn)
            fm.addAction(a)
        vm = mb.addMenu("表示")
        for text, fn in [("等角", lambda: self.plotter.view_isometric()),
                         ("+X から", lambda: self.plotter.view_yz()),
                         ("-X から", lambda: self.plotter.view_yz(negative=True)),
                         ("+Y から", lambda: self.plotter.view_xz(negative=True)),
                         ("-Y から", lambda: self.plotter.view_xz()),
                         ("+Z（上）から", lambda: self.plotter.view_xy()),
                         ("-Z（下）から", lambda: self.plotter.view_xy(negative=True)),
                         ("全体表示", lambda: self.plotter.reset_camera())]:
            a = QAction(text, self)
            a.triggered.connect(fn)
            vm.addAction(a)
        hm = mb.addMenu("ヘルプ")
        a = QAction("使い方", self)
        a.triggered.connect(self.show_help)
        hm.addAction(a)

    def _tab_geometry(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("簡単な形状を作成")
        gl = QVBoxLayout(g)
        self.prim_combo = QComboBox()
        for k, (name, _) in PRIMITIVES.items():
            self.prim_combo.addItem(name, k)
        self.prim_combo.currentIndexChanged.connect(self._rebuild_prim_form)
        gl.addWidget(self.prim_combo)
        self.prim_form_host = QWidget()
        self.prim_form = QFormLayout(self.prim_form_host)
        gl.addWidget(self.prim_form_host)
        b = QPushButton("この形状を作成")
        b.clicked.connect(self.create_geometry)
        gl.addWidget(b)
        lay.addWidget(g)

        g2 = QGroupBox("CAD ファイルを読み込む")
        gl2 = QVBoxLayout(g2)
        lab = QLabel("STEP (.step/.stp) 推奨。STL、IGES にも対応。\n"
                     ".x_t（Parasolid）は CAD から STEP で書き出してください。")
        lab.setWordWrap(True)
        gl2.addWidget(lab)
        b2 = QPushButton("ファイルを開く…")
        b2.clicked.connect(self.import_file)
        gl2.addWidget(b2)
        lay.addWidget(g2)
        self.geo_label = QLabel("")
        self.geo_label.setWordWrap(True)
        lay.addWidget(self.geo_label)
        lay.addStretch()
        self.prim_spins = {}
        self._rebuild_prim_form()
        return w

    def _rebuild_prim_form(self):
        while self.prim_form.rowCount():
            self.prim_form.removeRow(0)
        kind = self.prim_combo.currentData()
        self.prim_spins = {}
        for key, label, default in PRIMITIVES[kind][1]:
            val = self.project.geometry.get(key, default) if self.project.geometry.get("kind") == kind else default
            s = dspin(float(val), 0.0, 1e6, 3)
            self.prim_spins[key] = s
            self.prim_form.addRow(label, s)

    def _tab_mesh(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.mesh_size = dspin(0.0, 0.0, 1e5, 3, 0.5)
        self.mesh_size.setSpecialValueText("自動")
        form.addRow("要素サイズ [mm]", self.mesh_size)
        self.mesh_order = QComboBox()
        self.mesh_order.addItem("2次要素 Tet10（推奨・高精度）", 2)
        self.mesh_order.addItem("1次要素 Tet4（高速・剛すぎる傾向）", 1)
        form.addRow("要素次数", self.mesh_order)
        lay.addLayout(form)
        b = QPushButton("メッシュを作成")
        b.clicked.connect(self.make_mesh)
        lay.addWidget(b)
        g = QGroupBox("面のサイズ指定（穴・角など応力が集中する面を細かく）")
        gl = QVBoxLayout(g)
        hb = QHBoxLayout()
        self.local_size = dspin(0.5, 0.001, 1e5, 3, 0.1)
        hb.addWidget(QLabel("サイズ [mm]"))
        hb.addWidget(self.local_size)
        b2 = QPushButton("選択中の面に設定")
        b2.clicked.connect(self.add_local_size)
        hb.addWidget(b2)
        gl.addLayout(hb)
        self.local_list = QListWidget()
        self.local_list.setMaximumHeight(90)
        gl.addWidget(self.local_list)
        b3 = QPushButton("削除")
        b3.clicked.connect(self.delete_local_size)
        gl.addWidget(b3)
        tip2 = QLabel("面は「4 境界条件」タブか3D画面の右クリックで選択してから設定します。")
        tip2.setWordWrap(True)
        tip2.setStyleSheet("color:#666")
        gl.addWidget(tip2)
        lay.addWidget(g)
        self.show_edges = QCheckBox("メッシュの線を表示")
        self.show_edges.setChecked(True)
        self.show_edges.toggled.connect(self.refresh_view)
        lay.addWidget(self.show_edges)
        self.mesh_label = QLabel("")
        self.mesh_label.setWordWrap(True)
        lay.addWidget(self.mesh_label)
        tip = QLabel("ヒント: 応力集中部（穴・角・溝）の応力を正しく出すには、"
                     "要素サイズを小さくして結果が変わらなくなるまで確認してください（収束確認）。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#666")
        lay.addWidget(tip)
        lay.addStretch()
        return w

    def _tab_material(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.mat_combo = QComboBox()
        for name in LIBRARY:
            self.mat_combo.addItem(name)
        self.mat_combo.addItem("カスタム")
        self.mat_combo.currentTextChanged.connect(self.on_material_pick)
        lay.addWidget(self.mat_combo)
        form = QFormLayout()
        self.mat_name = QLineEdit()
        self.mat_E = dspin(200000, 1, 1e7, 1, 1000)
        self.mat_nu = dspin(0.3, 0.0, 0.499, 3, 0.01)
        self.mat_rho = dspin(7850, 0, 1e5, 1, 10)
        self.mat_sy = dspin(250, 0, 1e5, 1, 5)
        self.mat_su = dspin(400, 0, 1e5, 1, 5)
        form.addRow("名称", self.mat_name)
        form.addRow("ヤング率 [MPa]", self.mat_E)
        form.addRow("ポアソン比", self.mat_nu)
        form.addRow("密度 [kg/m³]", self.mat_rho)
        form.addRow("引張降伏応力 [MPa]", self.mat_sy)
        form.addRow("引張強さ [MPa]", self.mat_su)
        lay.addLayout(form)
        for s in (self.mat_E, self.mat_nu, self.mat_rho, self.mat_sy, self.mat_su):
            s.valueChanged.connect(self.on_material_edit)
        self.mat_name.textEdited.connect(self.on_material_edit)
        note = QLabel("降伏応力を入れると「安全率」が表示されます。\n※ 値は代表値です。設計には規格値・ミルシートの値を使ってください。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        lay.addWidget(note)
        lay.addStretch()
        return w

    def _tab_bc(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        info = QLabel("① 3D画面で面を<b>右クリック</b>して選択（Shift/⌘＋右クリックで追加選択）、"
                      "または下の一覧から選択<br>② 追加したい条件のボタンを押す")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.face_list = QListWidget()
        self.face_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.face_list.itemSelectionChanged.connect(self.on_face_list_select)
        self.face_list.setMaximumHeight(180)
        lay.addWidget(QLabel("面の一覧"))
        lay.addWidget(self.face_list)
        self.sel_label = QLabel("選択中: なし")
        lay.addWidget(self.sel_label)
        grid = QGridLayout()
        for i, kind in enumerate(["fixed", "displacement", "force", "pressure", "moment", "gravity"]):
            b = QPushButton("＋ " + KIND_JP[kind])
            b.setStyleSheet(f"QPushButton{{border-left:6px solid {BC_COLORS[kind]};padding:6px}}")
            b.clicked.connect(lambda _=False, k=kind: self.add_bc(k))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        lay.addWidget(QLabel("設定済みの条件（ダブルクリックで編集）"))
        self.bc_list = QListWidget()
        self.bc_list.itemDoubleClicked.connect(self.edit_bc)
        self.bc_list.currentRowChanged.connect(lambda _: self.refresh_view())
        lay.addWidget(self.bc_list)
        hb = QHBoxLayout()
        b = QPushButton("編集")
        b.clicked.connect(lambda: self.edit_bc(self.bc_list.currentItem()))
        hb.addWidget(b)
        b = QPushButton("削除")
        b.clicked.connect(self.delete_bc)
        hb.addWidget(b)
        lay.addLayout(hb)
        return w

    def _tab_solve(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.solver_combo = QComboBox()
        for label, key in [("自動（推奨）", "auto"), ("直接法 SciPy", "scipy"),
                           ("AMG 反復法（大規模向け, pyamg）", "amg")]:
            self.solver_combo.addItem(label, key)
        form.addRow("ソルバー", self.solver_combo)
        lay.addLayout(form)
        self.solve_btn = QPushButton("▶ 解析を実行")
        self.solve_btn.setStyleSheet("font-size:16px;padding:10px")
        self.solve_btn.clicked.connect(self.run_solve)
        lay.addWidget(self.solve_btn)
        self.solve_label = QLabel("解析タイプ: 線形静的構造解析（微小変形・線形弾性）")
        self.solve_label.setWordWrap(True)
        lay.addWidget(self.solve_label)
        lay.addStretch()
        return w

    def _tab_results(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.field_combo = QComboBox()
        self.field_combo.currentIndexChanged.connect(self.refresh_view)
        form.addRow("表示する結果", self.field_combo)
        self.scale_combo = QComboBox()
        for label, v in [("実寸（×1）", 1.0), ("自動（強調）", -1.0), ("×10", 10.0), ("×100", 100.0),
                         ("×1000", 1000.0), ("変形なし", 0.0)]:
            self.scale_combo.addItem(label, v)
        self.scale_combo.setCurrentIndex(1)
        self.scale_combo.currentIndexChanged.connect(self.refresh_view)
        form.addRow("変形の倍率", self.scale_combo)
        self.bands = QComboBox()
        for label, v in [("9段階（Ansys風）", 9), ("連続", 256), ("16段階", 16)]:
            self.bands.addItem(label, v)
        self.bands.currentIndexChanged.connect(self.refresh_view)
        form.addRow("コンター", self.bands)
        lay.addLayout(form)
        self.show_undeformed = QCheckBox("変形前の形状を重ねて表示")
        self.show_undeformed.setChecked(True)
        self.show_undeformed.toggled.connect(self.refresh_view)
        lay.addWidget(self.show_undeformed)
        self.show_maxmin = QCheckBox("最大・最小の位置を表示")
        self.show_maxmin.setChecked(True)
        self.show_maxmin.toggled.connect(self.refresh_view)
        lay.addWidget(self.show_maxmin)
        lay.addWidget(QLabel("右クリックでその点の値を表示します"))
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["項目", "値"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        lay.addWidget(self.summary_table)
        hb = QHBoxLayout()
        for text, fn in [("レポート", self.export_report), ("VTU", self.export_vtu), ("CSV", self.export_csv)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        lay.addLayout(hb)
        return w

    # ======================================================================= 共通
    def log(self, msg: str):
        self.logbox.appendPlainText(msg)
        self.statusBar().showMessage(msg.splitlines()[0] if msg else "", 8000)

    def busy(self, on: bool, msg: str = ""):
        self.progress.setVisible(on)
        self.progress.setRange(0, 0 if on else 100)
        for b in (self.solve_btn,):
            b.setEnabled(not on)
        if on:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.log(msg)
        else:
            QApplication.restoreOverrideCursor()

    def start(self, msg, fn, on_done, *args, **kw):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "処理中", "前の処理が終わるまでお待ちください。")
            return
        self.busy(True, msg)
        self.worker = Worker(fn, *args, **kw)
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda out: (self.busy(False), on_done(out)))
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_failed(self, msg):
        self.busy(False)
        self.log("エラー: " + msg)
        QMessageBox.warning(self, "エラー", msg if len(msg) < 1500 else msg[-1500:])

    # ======================================================================= 形状・メッシュ
    def create_geometry(self):
        kind = self.prim_combo.currentData()
        spec = {"kind": kind, **{k: s.value() for k, s in self.prim_spins.items()}}
        self._set_geometry(spec)

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "形状ファイルを開く", "", CAD_FILTER)
        if path:
            self._set_geometry({"kind": "file", "path": path})

    def _set_geometry(self, spec):
        old = self.project.geometry
        self.project.geometry = spec
        if old != spec and not self.project.bcs:
            self.project.local_sizes = []
            self.refresh_local_list()
        if old != spec and self.project.bcs:
            r = QMessageBox.question(self, "境界条件", "形状が変わります。設定済みの境界条件を削除しますか？")
            if r == QMessageBox.Yes:
                self.project.bcs = []
                self.project.local_sizes = []
                self.refresh_bc_list()
                self.refresh_local_list()
        self.geo_label.setText("現在の形状: " + self.project.geometry_desc())
        self.make_mesh(reset_camera=True)

    def make_mesh(self, reset_camera=False):
        """gmsh はスレッドセーフでなく、macOS では別スレッドで動かすと落ちるため、メインスレッドで実行"""
        self.project.mesh_size = self.mesh_size.value()
        self.project.mesh_order = self.mesh_order.currentData()
        self.busy(True, "メッシュを作成しています…")
        QApplication.processEvents()
        try:
            mesh = build_mesh(self.project.geometry, self.project.mesh_size, self.project.mesh_order,
                              log=lambda m: (self.log(str(m)), QApplication.processEvents()),
                              local_sizes=self.project.local_sizes)
        except (GeometryError, ValueError, RuntimeError) as e:
            self.on_failed(str(e))
            return
        except Exception:
            self.on_failed(traceback.format_exc())
            return
        self.busy(False)
        self.on_mesh(mesh, reset_camera)

    def on_mesh(self, mesh, reset_camera=False):
        self.mesh = mesh
        self.result = None
        self.selected_faces = []
        mesh.face_info()
        self.mode = "geometry"
        n_faces = mesh.n_faces
        # 存在しない面を参照する境界条件を警告
        bad = [b for b in self.project.bcs if any(f >= n_faces for f in b.faces)]
        if bad:
            self.log("⚠ 一部の境界条件が存在しない面を参照しています。設定し直してください。")
        lo, hi = mesh.bounds()
        size = hi - lo
        self.mesh_label.setText(
            f"節点 {mesh.n_nodes:,} ／ 要素 {mesh.n_elems:,}（{'Tet10' if mesh.order == 2 else 'Tet4'}）\n"
            f"面の数 {n_faces}\n寸法 X {size[0]:.3g} × Y {size[1]:.3g} × Z {size[2]:.3g} mm\n"
            f"体積 {mesh.volume():.6g} mm³ ／ 質量 {mesh.volume() * self.current_material().density_t_mm3 * 1000:.4g} kg")
        self.log(f"メッシュ: 節点 {mesh.n_nodes:,} 要素 {mesh.n_elems:,} 面 {n_faces}")
        self.refresh_face_list()
        self.refresh_view()
        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()

    def add_local_size(self):
        if not self.selected_faces:
            QMessageBox.information(self, "面を選択してください",
                                    "3D画面で面を右クリックするか、「4 境界条件」の面の一覧から選択してください。")
            return
        self.project.local_sizes.append({"faces": list(self.selected_faces), "size": self.local_size.value()})
        self.refresh_local_list()
        if QMessageBox.question(self, "メッシュ", "今すぐメッシュを作り直しますか？") == QMessageBox.Yes:
            self.make_mesh()

    def delete_local_size(self):
        i = self.local_list.currentRow()
        if 0 <= i < len(self.project.local_sizes):
            del self.project.local_sizes[i]
            self.refresh_local_list()

    def refresh_local_list(self):
        self.local_list.clear()
        for ls in self.project.local_sizes:
            self.local_list.addItem(", ".join(f"面{f + 1}" for f in ls["faces"]) + f"  →  {ls['size']:g} mm")

    # ======================================================================= 材料
    def on_material_pick(self, name):
        if name in LIBRARY:
            m = LIBRARY[name]
            self._set_mat_fields(m)

    def _set_mat_fields(self, m: Material):
        for s in (self.mat_E, self.mat_nu, self.mat_rho, self.mat_sy, self.mat_su):
            s.blockSignals(True)
        self.mat_name.setText(m.name)
        self.mat_E.setValue(m.E)
        self.mat_nu.setValue(m.nu)
        self.mat_rho.setValue(m.density)
        self.mat_sy.setValue(m.yield_strength)
        self.mat_su.setValue(m.tensile_strength)
        for s in (self.mat_E, self.mat_nu, self.mat_rho, self.mat_sy, self.mat_su):
            s.blockSignals(False)
        self.project.material = m

    def on_material_edit(self, *_):
        m = self.current_material()
        if m.name in LIBRARY and LIBRARY[m.name] != m:
            m.name = m.name + "（変更）"
            self.mat_name.setText(m.name)
        self.mat_combo.blockSignals(True)
        self.mat_combo.setCurrentText(m.name if m.name in LIBRARY else "カスタム")
        self.mat_combo.blockSignals(False)
        self.project.material = m

    def current_material(self) -> Material:
        return Material(self.mat_name.text() or "カスタム", self.mat_E.value(), self.mat_nu.value(),
                        self.mat_rho.value(), self.mat_sy.value(), self.mat_su.value())

    # ======================================================================= 境界条件
    def refresh_face_list(self):
        self.face_list.blockSignals(True)
        self.face_list.clear()
        if self.mesh is not None:
            for f in self.mesh.face_info():
                c, n = f["centroid"], f["normal"]
                kind = "平面" if f["flat"] else "曲面"
                txt = (f"面{f['id'] + 1}  {kind}  面積 {f['area']:.4g} mm²  "
                       f"中心({c[0]:.3g}, {c[1]:.3g}, {c[2]:.3g})"
                       + (f"  法線({n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f})" if f["flat"] else ""))
                it = QListWidgetItem(txt)
                it.setData(Qt.UserRole, f["id"])
                self.face_list.addItem(it)
        self.face_list.blockSignals(False)
        self._sync_selection_label()

    def on_face_list_select(self):
        self.selected_faces = sorted(it.data(Qt.UserRole) for it in self.face_list.selectedItems())
        self._sync_selection_label()
        self.refresh_view()

    def _sync_selection_label(self):
        self.sel_label.setText("選択中: " + (", ".join(f"面{f + 1}" for f in self.selected_faces) or "なし"))
        self.face_list.blockSignals(True)
        for i in range(self.face_list.count()):
            it = self.face_list.item(i)
            it.setSelected(it.data(Qt.UserRole) in self.selected_faces)
        self.face_list.blockSignals(False)

    def on_pick(self, pos, *args):
        if self.mesh is None or pos is None:
            return
        pos = np.asarray(pos, float)
        if self.mode == "results" and self.result is not None:
            self._probe(pos)
            return
        # 背景をクリックした場合は無視
        lo, hi = self.mesh.bounds()
        if np.linalg.norm(self.mesh.tri_centroids() - pos, axis=1).min() > 0.1 * np.linalg.norm(hi - lo):
            return
        f = self.mesh.nearest_face(pos)
        mods = QApplication.keyboardModifiers()
        add = bool(mods & (Qt.ShiftModifier | Qt.ControlModifier | Qt.MetaModifier))
        if add:
            if f in self.selected_faces:
                self.selected_faces.remove(f)
            else:
                self.selected_faces.append(f)
        else:
            self.selected_faces = [f]
        self.selected_faces.sort()
        self._sync_selection_label()
        if self.tabs.currentIndex() != 3:
            self.tabs.setCurrentIndex(3)
        self.refresh_view()

    def add_bc(self, kind):
        if self.mesh is None:
            QMessageBox.information(self, "メッシュ未作成", "先にメッシュを作成してください。")
            return
        if kind != "gravity" and not self.selected_faces:
            QMessageBox.information(self, "面を選択してください",
                                    "3D画面で面を右クリックするか、面の一覧から選択してください。")
            return
        dlg = BCDialog(kind, self.selected_faces, parent=self)
        if dlg.exec():
            self.project.bcs.append(dlg.result_bc())
            self.refresh_bc_list()
            self.bc_list.setCurrentRow(len(self.project.bcs) - 1)

    def edit_bc(self, item):
        if item is None:
            return
        i = self.bc_list.row(item)
        bc = self.project.bcs[i]
        faces = self.selected_faces if (self.selected_faces and bc.kind != "gravity" and
                                        QMessageBox.question(self, "対象の面",
                                                             "対象の面を現在の選択に置き換えますか？")
                                        == QMessageBox.Yes) else bc.faces
        dlg = BCDialog(bc.kind, faces, bc, parent=self)
        if dlg.exec():
            self.project.bcs[i] = dlg.result_bc()
            self.refresh_bc_list()

    def delete_bc(self):
        i = self.bc_list.currentRow()
        if 0 <= i < len(self.project.bcs):
            del self.project.bcs[i]
            self.refresh_bc_list()

    def refresh_bc_list(self):
        self.bc_list.blockSignals(True)
        self.bc_list.clear()
        for bc in self.project.bcs:
            it = QListWidgetItem(bc.label())
            it.setToolTip(bc.label())
            it.setBackground(QColor(BC_COLORS[bc.kind]).lighter(170))
            self.bc_list.addItem(it)
        self.bc_list.blockSignals(False)
        self.refresh_view()

    # ======================================================================= 解析
    def run_solve(self):
        if self.mesh is None:
            QMessageBox.information(self, "メッシュ未作成", "先にメッシュを作成してください。")
            return
        self.project.material = self.current_material()
        if self.project.material.E <= 0:
            QMessageBox.warning(self, "材料", "ヤング率を入力してください。")
            return
        self.start("解析を開始します…", solve, self.on_solved, self.mesh, self.project.material,
                   list(self.project.bcs), method=self.solver_combo.currentData())

    def on_solved(self, result):
        self.result = result
        self.log("解析が完了しました: " + ", ".join(f"{k} {v}" for k, v in result.info.items()))
        self.field_combo.blockSignals(True)
        cur = self.field_combo.currentText()
        self.field_combo.clear()
        self.field_combo.addItems(list(result.fields().keys()))
        if cur:
            self.field_combo.setCurrentText(cur)
        else:
            self.field_combo.setCurrentText("相当応力（ミーゼス） [MPa]")
        self.field_combo.blockSignals(False)
        self._fill_summary()
        self.mode = "results"
        self.tabs.setCurrentIndex(5)
        self.refresh_view()

    def _fill_summary(self):
        r = self.result
        rows = []
        for k, v in r.summary().items():
            rows.append((k, f"{v:.5g}" if isinstance(v, float) else str(v)))
        for i, f in r.reactions.items():
            rows.append((f"反力 {self.project.bcs[i].label()} [N]",
                         f"({f[0]:.4g}, {f[1]:.4g}, {f[2]:.4g})  |F|={np.linalg.norm(f):.4g}"))
        for k, v in r.info.items():
            rows.append((k, str(v)))
        self.summary_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.summary_table.setItem(i, 0, QTableWidgetItem(k))
            self.summary_table.setItem(i, 1, QTableWidgetItem(v))

    def on_tab_changed(self, idx):
        new_mode = "results" if (idx == 5 and self.result is not None) else "geometry"
        if new_mode != self.mode:
            self.mode = new_mode
            self.refresh_view()

    # ======================================================================= 描画
    def _deform_scale(self):
        v = self.scale_combo.currentData()
        if v is None:
            return 1.0
        if v < 0:  # 自動: 最大変位がモデル寸法の 5% に見える倍率
            lo, hi = self.mesh.bounds()
            L = np.linalg.norm(hi - lo)
            umax = self.result.disp_mag.max()
            if umax <= 0:
                return 1.0
            s = 0.05 * L / umax
            # きりの良い数字に
            p = 10 ** np.floor(np.log10(s))
            return float(max(1.0, np.floor(s / p) * p))
        return float(v)

    def refresh_view(self, *_):
        if self.mesh is None:
            return
        pl = self.plotter
        pl.clear()
        pl.add_axes()
        if self.mode == "results" and self.result is not None:
            self._draw_results()
        else:
            self._draw_geometry()
        pl.render()

    def _draw_geometry(self):
        pl, mesh = self.plotter, self.mesh
        poly, src = surface_polydata(mesh)
        fid = mesh.face_ids[src]
        colors = np.array([_hex(FACE_COLORS[i % len(FACE_COLORS)]) for i in range(mesh.n_faces)])
        rgb = colors[fid]
        # 選択中の BC の面
        cur = self.bc_list.currentRow() if self.tabs.currentIndex() == 3 else -1
        if 0 <= cur < len(self.project.bcs):
            bc = self.project.bcs[cur]
            for f in bc.faces:
                if f < mesh.n_faces:
                    rgb[fid == f] = _hex(BC_COLORS[bc.kind])
        for f in self.selected_faces:
            rgb[fid == f] = _hex(SELECT_COLOR)
        poly.cell_data["rgb"] = (rgb * 255).astype(np.uint8)
        pl.add_mesh(poly, scalars="rgb", rgb=True, show_edges=self.show_edges.isChecked(),
                    edge_color="#55606e", line_width=0.5, smooth_shading=False, show_scalar_bar=False)
        fe = feature_edges(mesh)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=2)
        # 面番号ラベル
        info = mesh.face_info()
        if len(info) <= 60:
            pts = np.array([f["centroid"] for f in info])
            # 重心が面上にない曲面は最も近い三角形の重心へ
            c = mesh.tri_centroids()
            for i, f in enumerate(info):
                m = mesh.face_ids == f["id"]
                cc = c[m]
                pts[i] = cc[np.argmin(np.linalg.norm(cc - pts[i], axis=1))]
            _labels(pl, pts, [str(f["id"] + 1) for f in info], 11, "#333333")
        # 境界条件の矢印・マーカー
        self._draw_bc_glyphs()

    def _draw_bc_glyphs(self):
        pl, mesh = self.plotter, self.mesh
        lo, hi = mesh.bounds()
        L = float(np.linalg.norm(hi - lo))
        info = {f["id"]: f for f in mesh.face_info()}
        for bc in self.project.bcs:
            faces = [f for f in bc.faces if f in info]
            if not faces:
                if bc.kind == "gravity":
                    c = 0.5 * (lo + hi)
                    v = np.asarray(bc.values, float)
                    if np.linalg.norm(v) > 0:
                        pl.add_mesh(pv.Arrow(start=c, direction=v / np.linalg.norm(v), scale=0.25 * L),
                                    color=BC_COLORS["gravity"])
                continue
            A = sum(info[f]["area"] for f in faces)
            c = sum(info[f]["centroid"] * info[f]["area"] for f in faces) / A
            n = sum(info[f]["normal"] * info[f]["area"] for f in faces)
            n = n / (np.linalg.norm(n) or 1)
            if bc.kind == "force":
                v = np.asarray(bc.values, float)
                if np.linalg.norm(v) > 0:
                    d = v / np.linalg.norm(v)
                    pl.add_mesh(pv.Arrow(start=c - d * 0.2 * L, direction=d, scale=0.2 * L), color=BC_COLORS["force"])
            elif bc.kind == "pressure":
                s = 1 if (bc.values[0] or 0) >= 0 else -1
                pl.add_mesh(pv.Arrow(start=c + s * n * 0.15 * L, direction=-s * n, scale=0.15 * L),
                            color=BC_COLORS["pressure"])
            elif bc.kind == "moment":
                v = np.asarray(bc.values, float)
                if np.linalg.norm(v) > 0:
                    d = v / np.linalg.norm(v)
                    pl.add_mesh(pv.Arrow(start=c, direction=d, scale=0.15 * L, tip_length=0.4), color=BC_COLORS["moment"])
                    pl.add_mesh(pv.Arrow(start=c + d * 0.05 * L, direction=d, scale=0.17 * L, tip_length=0.35),
                                color=BC_COLORS["moment"])

    def _draw_results(self):
        pl, mesh, r = self.plotter, self.mesh, self.result
        name = self.field_combo.currentText()
        fields = r.fields()
        if name not in fields:
            name = "相当応力（ミーゼス） [MPa]"
        vals = np.asarray(fields[name], float)
        scale = self._deform_scale()
        pts = mesh.nodes + scale * r.u
        poly, _ = surface_polydata(mesh, pts)
        title = FIELD_EN.get(name, name)
        poly.point_data[title] = vals
        n_colors = self.bands.currentData() or 9
        cmap = "jet"
        if "Safety" in title:
            cmap = "jet_r"
        used = np.unique(mesh.boundary_tris_full)
        vmin, vmax = float(vals[used].min()), float(vals[used].max())
        if vmax - vmin < 1e-30:
            vmax = vmin + 1e-30
        pl.add_mesh(poly, scalars=title, cmap=cmap, n_colors=n_colors, clim=[vmin, vmax],
                    show_edges=False, smooth_shading=False,
                    scalar_bar_args={"title": title + f"\nScale x{scale:g}", "vertical": True,
                                     "fmt": "%.4g", "n_labels": min(n_colors + 1, 10),
                                     "position_x": 0.85, "position_y": 0.1, "height": 0.8,
                                     "width": 0.06, "title_font_size": 14, "label_font_size": 12})
        fe = feature_edges(mesh, pts)
        if fe is not None:
            pl.add_mesh(fe, color="black", line_width=1)
        if self.show_undeformed.isChecked() and scale > 0:
            fe0 = feature_edges(mesh)
            if fe0 is not None:
                pl.add_mesh(fe0, color="#777777", line_width=1, opacity=0.6)
        if self.show_maxmin.isChecked():
            imax = used[np.argmax(vals[used])]
            imin = used[np.argmin(vals[used])]
            _labels(pl, pts[[imax, imin]], [f"Max {vals[imax]:.4g}", f"Min {vals[imin]:.4g}"],
                    13, "black", point_colors=["red", "blue"])

    def _probe(self, pos):
        r = self.result
        scale = self._deform_scale()
        pts = self.mesh.nodes + scale * r.u
        used = np.unique(self.mesh.boundary_tris_full)
        i = used[np.argmin(np.linalg.norm(pts[used] - pos, axis=1))]
        name = self.field_combo.currentText()
        v = r.fields()[name][i]
        x = self.mesh.nodes[i]
        self.log(f"プローブ 節点{i + 1} ({x[0]:.4g}, {x[1]:.4g}, {x[2]:.4g}) mm: {name} = {v:.5g}")
        self.refresh_view()
        _labels(self.plotter, pts[[i]], [f"{v:.4g}"], 13, "black", point_colors=["#ff00ff"])
        self.plotter.render()

    # ======================================================================= ファイル
    def load_project_to_ui(self):
        p = self.project
        g = p.geometry
        if g.get("kind") in PRIMITIVES:
            idx = self.prim_combo.findData(g["kind"])
            self.prim_combo.blockSignals(True)
            self.prim_combo.setCurrentIndex(idx)
            self.prim_combo.blockSignals(False)
            self._rebuild_prim_form()
        self.mesh_size.setValue(p.mesh_size)
        self.mesh_order.setCurrentIndex(0 if p.mesh_order == 2 else 1)
        self.mat_combo.blockSignals(True)
        self.mat_combo.setCurrentText(p.material.name if p.material.name in LIBRARY else "カスタム")
        self.mat_combo.blockSignals(False)
        self._set_mat_fields(p.material)
        self.geo_label.setText("現在の形状: " + p.geometry_desc())
        self.refresh_bc_list()
        self.refresh_local_list()

    def new_project(self):
        self.project = Project()
        self.project_path = None
        self.result = None
        self.load_project_to_ui()
        self.make_mesh(reset_camera=True)

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
        self.result = None
        self.setWindowTitle(f"open-mechanical — {os.path.basename(path)}")
        self.load_project_to_ui()
        self.make_mesh(reset_camera=True)

    def save_project(self):
        if not self.project_path:
            return self.save_project_as()
        self.project.material = self.current_material()
        self.project.mesh_size = self.mesh_size.value()
        self.project.mesh_order = self.mesh_order.currentData()
        self.project.save(self.project_path)
        self.log(f"保存しました: {self.project_path}")

    def save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "名前を付けて保存", "analysis.omech",
                                              "open-mechanical (*.omech)")
        if path:
            if not path.endswith(".omech"):
                path += ".omech"
            self.project_path = path
            self.save_project()
            self.setWindowTitle(f"open-mechanical — {os.path.basename(path)}")

    def _need_result(self):
        if self.result is None:
            QMessageBox.information(self, "結果なし", "先に解析を実行してください。")
            return False
        return True

    def export_vtu(self):
        if not self._need_result():
            return
        path, _ = QFileDialog.getSaveFileName(self, "VTU で保存", "result.vtu", "VTK (*.vtu)")
        if path:
            write_vtu(path, self.result)
            self.log(f"書き出しました: {path}")

    def export_csv(self):
        if not self._need_result():
            return
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
        cur_field = self.field_combo.currentText()
        prev_mode, prev_tab = self.mode, self.tabs.currentIndex()
        try:
            # 境界条件図
            self.mode = "geometry"
            self.tabs.blockSignals(True)
            self.tabs.setCurrentIndex(3)
            self.refresh_view()
            p = os.path.join(tmp, "bc.png")
            self.plotter.screenshot(p)
            images["境界条件"] = p
            self.mode = "results"
            for name in ["全変形 [mm]", "相当応力（ミーゼス） [MPa]", "安全率（降伏基準）"]:
                if name in self.result.fields():
                    self.field_combo.setCurrentText(name)
                    self.refresh_view()
                    p = os.path.join(tmp, f"{len(images)}.png")
                    self.plotter.screenshot(p)
                    images[name] = p
        finally:
            self.field_combo.setCurrentText(cur_field)
            self.tabs.setCurrentIndex(prev_tab)
            self.tabs.blockSignals(False)
            self.mode = prev_mode
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
            "【基本の流れ】\n"
            "1 形状: 簡単な形状を作るか、STEP/STL を読み込む\n"
            "2 メッシュ: 要素サイズを決めて作成（2次要素推奨）\n"
            "3 材料: 一覧から選ぶか値を入力\n"
            "4 境界条件: 3D画面で面を右クリック → 「固定支持」「力」などを追加\n"
            "5 解析: 「解析を実行」\n"
            "6 結果: 変形・応力・安全率をコンター表示、右クリックで値を確認\n\n"
            "【マウス】左ドラッグ: 回転 / 中ドラッグ or Shift+左: 移動 / ホイール: 拡大縮小\n"
            "右クリック: 面の選択（結果表示中は値の確認）\n\n"
            "【単位】長さ mm、力 N、応力 MPa、モーメント N·mm"))

    def closeEvent(self, e):
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(e)


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _labels(pl, pts, texts, size, color, point_colors=None):
    kw = dict(font_size=size, text_color=color, shape_opacity=0.7, point_size=12 if point_colors else 0,
              show_points=bool(point_colors), render_points_as_spheres=True)
    if point_colors:
        kw["point_color"] = point_colors[0]
    try:
        if point_colors and len(pts) > 1:
            for p, t, c in zip(pts, texts, point_colors):
                kw["point_color"] = c
                pl.add_point_labels([p], [t], always_visible=True, **kw)
        else:
            pl.add_point_labels(pts, texts, always_visible=True, **kw)
    except TypeError:
        pl.add_point_labels(pts, texts, **kw)


def _setup_crash_log():
    """異常終了したときの原因を logs/crash.log に残す"""
    import faulthandler
    import datetime
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, "crash.log"), "a", buffering=1)
    f.write(f"\n=== 起動 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  Python {sys.version.split()[0]} ===\n")
    faulthandler.enable(file=f, all_threads=True)
    return f


def main():
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    _crash = _setup_crash_log()  # noqa: F841
    try:
        pv.global_theme.allow_empty_mesh = True
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("open-mechanical")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

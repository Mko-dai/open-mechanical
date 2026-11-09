"""き裂（破壊力学）の定義ダイアログ — Ansys の Fracture → Crack に相当（直線の前縁・平面き裂）"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget)


def _vec_edit(v):
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    sp = []
    for x in v:
        s = QDoubleSpinBox()
        s.setRange(-1e7, 1e7)
        s.setDecimals(4)
        s.setValue(float(x))
        s.setMinimumWidth(90)
        h.addWidget(s)
        sp.append(s)
    return w, sp


class CrackDialog(QDialog):
    def __init__(self, parent, cracks, lo, hi):
        super().__init__(parent)
        self.setWindowTitle("き裂（破壊力学）")
        self.resize(720, 520)
        self._cracks = [dict(c) for c in cracks]
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.lst = QListWidget()
        left.addWidget(self.lst)
        ba = QPushButton("追加")
        ba.clicked.connect(self._add)
        bd = QPushButton("削除")
        bd.clicked.connect(self._delete)
        left.addWidget(ba)
        left.addWidget(bd)
        lay.addLayout(left, 1)
        right = QVBoxLayout()
        f = QFormLayout()
        self.name = QLineEdit()
        f.addRow("名前", self.name)
        w, self.f0 = _vec_edit([0, 0, 0])
        f.addRow("前縁の始点 [mm]", w)
        w, self.f1 = _vec_edit([0, 0, 0])
        f.addRow("前縁の終点 [mm]", w)
        w, self.nrm = _vec_edit([0, 1, 0])
        f.addRow("き裂面の法線", w)
        w, self.dir = _vec_edit([1, 0, 0])
        f.addRow("進展方向（前縁から先）", w)
        self.depth = QDoubleSpinBox()
        self.depth.setRange(0, 1e7)
        self.depth.setDecimals(4)
        self.depth.setSuffix(" mm")
        f.addRow("深さ（前縁から後ろ）", self.depth)
        self.to_surf = QCheckBox("後ろ側は表面まで（表面き裂・縁き裂）")
        self.to_surf.setChecked(True)
        f.addRow("", self.to_surf)
        self.fsize = QDoubleSpinBox()
        self.fsize.setRange(0, 1e6)
        self.fsize.setDecimals(4)
        self.fsize.setSuffix(" mm")
        f.addRow("前縁の要素サイズ（0 = 全体の 1/8）", self.fsize)
        self.kic = QDoubleSpinBox()
        self.kic.setRange(0, 1e7)
        self.kic.setDecimals(2)
        self.kic.setSuffix(" MPa√mm")
        f.addRow("破壊靭性 K_IC（0 = 評価しない）", self.kic)
        self.sym = QCheckBox("き裂面が対称面（半分モデル。J を 2 倍）")
        f.addRow("", self.sym)
        right.addLayout(f)
        right.addWidget(QLabel(
            "前縁（直線）の両端は物体の表面に置いてください（板厚を貫く前縁）。き裂面は「深さ」だけ前縁の後ろに\n"
            "広がる長方形で、メッシュを作るときに形状へ埋め込み、き裂面の節点を分けます。\n"
            "結果は 解析結果タブ →「J 積分・K」。J は 5 つの積分路で求め、2〜5 番目の平均を使います。"))
        right.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        lay.addLayout(right, 3)
        self._cur = -1
        self.lst.currentRowChanged.connect(self._select)
        self._refresh()
        if self._cracks:
            self.lst.setCurrentRow(0)

    # ------------------------------------------------------------------
    def _refresh(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for i, c in enumerate(self._cracks):
            self.lst.addItem(c.get("name") or f"き裂 {i + 1}")
        self.lst.blockSignals(False)

    def _store(self):
        if 0 <= self._cur < len(self._cracks):
            self._cracks[self._cur] = {
                "name": self.name.text().strip() or f"き裂 {self._cur + 1}",
                "front0": [s.value() for s in self.f0], "front1": [s.value() for s in self.f1],
                "normal": [s.value() for s in self.nrm], "direction": [s.value() for s in self.dir],
                "depth": self.depth.value(), "to_surface": self.to_surf.isChecked(),
                "front_size": self.fsize.value(), "toughness": self.kic.value(), "symmetric": self.sym.isChecked()}

    def _select(self, i):
        self._store()
        self._cur = i
        if not (0 <= i < len(self._cracks)):
            return
        c = self._cracks[i]
        self.name.setText(c.get("name", ""))
        for sp, key, d in [(self.f0, "front0", [0, 0, 0]), (self.f1, "front1", [0, 0, 0]),
                           (self.nrm, "normal", [0, 1, 0]), (self.dir, "direction", [1, 0, 0])]:
            for s, v in zip(sp, c.get(key, d)):
                s.setValue(float(v))
        self.depth.setValue(float(c.get("depth", 0.0)))
        self.to_surf.setChecked(bool(c.get("to_surface", True)))
        self.fsize.setValue(float(c.get("front_size", 0.0) or 0.0))
        self.kic.setValue(float(c.get("toughness", 0.0) or 0.0))
        self.sym.setChecked(bool(c.get("symmetric", False)))

    def _add(self):
        self._store()
        lo, hi = self.lo, self.hi
        mid = (lo + hi) / 2
        # 既定: X 方向の縁き裂（深さ = 幅の 1/4）、前縁は Z 方向に板厚を貫く
        a = float(hi[0] - lo[0]) / 4
        self._cracks.append({"name": f"き裂 {len(self._cracks) + 1}",
                             "front0": [lo[0] + a, mid[1], lo[2]], "front1": [lo[0] + a, mid[1], hi[2]],
                             "normal": [0, 1, 0], "direction": [1, 0, 0], "depth": a, "to_surface": True})
        self._cur = -1
        self._refresh()
        self.lst.setCurrentRow(len(self._cracks) - 1)

    def _delete(self):
        i = self.lst.currentRow()
        if 0 <= i < len(self._cracks):
            self._cur = -1
            del self._cracks[i]
            self._refresh()
            if self._cracks:
                self.lst.setCurrentRow(min(i, len(self._cracks) - 1))

    def _ok(self):
        self._store()
        self.accept()

    def cracks(self):
        return self._cracks

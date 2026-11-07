"""境界条件の入力ダイアログ。"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout,
                               QWidget)

from ..solver import BC

KIND_JP = {"fixed": "固定支持", "displacement": "強制変位", "force": "力",
           "pressure": "圧力", "moment": "モーメント", "gravity": "重力加速度"}
UNITS = {"displacement": "mm", "force": "N", "pressure": "MPa", "moment": "N·mm", "gravity": "mm/s²"}
HELP = {
    "fixed": "選択した面のすべての節点を X・Y・Z 方向に固定します。",
    "displacement": "チェックした成分だけ指定値で拘束します（0 にすれば「その方向だけ固定」）。\n"
                    "チェックを外した成分は自由に動けます（ローラー支持など）。",
    "force": "選択した面全体に、合計がこの値になるよう一様に力を分布させます。",
    "pressure": "面に垂直な圧力。正の値で面を押し込む向きです。",
    "moment": "選択した面の重心まわりのモーメントです（ねじり・曲げ）。",
    "gravity": "モデル全体に加速度をかけます。地球の重力 = 9806.65 mm/s²（向きに注意）。",
}


def _spin(val=0.0, lo=-1e12, hi=1e12, dec=4):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setMinimumWidth(140)
    return s


class BCDialog(QDialog):
    def __init__(self, kind: str, faces: list[int], bc: BC | None = None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.faces = list(faces)
        self.setWindowTitle(f"{KIND_JP[kind]} の設定")
        lay = QVBoxLayout(self)
        h = QLabel(HELP[kind])
        h.setWordWrap(True)
        lay.addWidget(h)
        if kind != "gravity":
            lay.addWidget(QLabel("対象の面: " + (", ".join(f"面{f + 1}" for f in self.faces) or "（未選択）")))
        form = QFormLayout()
        self.name = QLineEdit(bc.name if bc else "")
        self.name.setPlaceholderText(KIND_JP[kind])
        form.addRow("名前", self.name)
        vals = list(bc.values) if bc else ([0.0, 0.0, -9806.65] if kind == "gravity" else [0.0, 0.0, 0.0])
        self.spins, self.checks = [], []
        unit = UNITS.get(kind, "")
        if kind == "pressure":
            s = _spin(float(vals[0] or 0.0), dec=6)
            self.spins.append(s)
            form.addRow(f"圧力 [{unit}]", s)
        elif kind in ("force", "moment", "gravity", "displacement"):
            for i, ax in enumerate("XYZ"):
                v = vals[i] if i < len(vals) else 0.0
                s = _spin(0.0 if v is None else float(v), dec=6 if kind == "displacement" else 4)
                self.spins.append(s)
                if kind == "displacement":
                    c = QCheckBox("拘束")
                    c.setChecked(v is not None)
                    c.toggled.connect(s.setEnabled)
                    s.setEnabled(v is not None)
                    self.checks.append(c)
                    w = QWidget()
                    hl = QHBoxLayout(w)
                    hl.setContentsMargins(0, 0, 0, 0)
                    hl.addWidget(c)
                    hl.addWidget(s)
                    form.addRow(f"{ax} [{unit}]", w)
                else:
                    form.addRow(f"{ax} 成分 [{unit}]", s)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def result_bc(self) -> BC:
        if self.kind == "fixed":
            vals = [0.0, 0.0, 0.0]
        elif self.kind == "pressure":
            vals = [self.spins[0].value()]
        elif self.kind == "displacement":
            vals = [s.value() if c.isChecked() else None for s, c in zip(self.spins, self.checks)]
        else:
            vals = [s.value() for s in self.spins]
        faces = [] if self.kind == "gravity" else self.faces
        return BC(self.kind, faces, vals, self.name.text().strip())

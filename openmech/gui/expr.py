"""式を入力できる数値欄（Femtet と同じく、数値の代わりに "L/2+5" のような式や変数名を書ける）"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from ..params import ParamError, evaluate, is_number


def env_of(widget) -> dict:
    """ウィンドウのプロジェクトの変数の値（評価できなければ空）"""
    w = widget
    while w is not None:
        p = getattr(w, "project", None)
        if p is not None and hasattr(p, "env"):
            try:
                return p.env()
            except ParamError:
                return {}
        w = w.parentWidget() if hasattr(w, "parentWidget") else None
    return {}


def fmt(v) -> str:
    return f"{float(v):.10g}"


class ExprEdit(QWidget):
    """数値または式の入力欄。式のときは右に計算結果「= 12.5」を出し、エラーなら赤くする。

    raw(): 数値（float）または式（str）  value(): 計算した値  rawChanged(raw): 入力が確定したとき
    """
    rawChanged = Signal(object)

    def __init__(self, text="0", env=None, suffix="", parent=None, width=130):
        super().__init__(parent)
        self._env = env          # dict または dict を返す関数（None なら親ウィンドウの変数）
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self.edit = QLineEdit(text if isinstance(text, str) else fmt(text))
        self.edit.setMinimumWidth(width)
        self.edit.setToolTip("数値のほか、式（例: L/2+5、sqrt(A)*2、10^3）や変数名を入力できます。\n"
                             "使える関数: sin cos tan（ラジアン）、sind cosd tand（度）、sqrt exp log abs min max など")
        h.addWidget(self.edit, 1)
        self.res = QLabel("")
        self.res.setStyleSheet("color:#2a6fdb")
        h.addWidget(self.res)
        if suffix:
            h.addWidget(QLabel(suffix))
        self.edit.textChanged.connect(self._update)
        self.edit.editingFinished.connect(self._finished)
        self._last = None
        self._update()
        self._last = self.raw() if self.ok() else None

    # -------------------------------------------------------------------
    def env(self) -> dict:
        e = self._env
        if callable(e):
            e = e()
        return e if e is not None else env_of(self.parentWidget() or self)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t if isinstance(t, str) else fmt(t))

    def ok(self) -> bool:
        try:
            self.value()
            return True
        except ParamError:
            return False

    def raw(self):
        t = self.text()
        return float(t) if is_number(t) else t

    def value(self) -> float:
        t = self.text()
        return float(t) if is_number(t) else evaluate(t, self.env())

    def setValue(self, v):
        self.setText(fmt(v))

    def _update(self, *_):
        t = self.text()
        if is_number(t):
            self.res.setText("")
            self.edit.setStyleSheet("")
            return
        try:
            v = evaluate(t, self.env())
            self.res.setText(f"= {v:.6g}")
            self.edit.setStyleSheet("")
            self.edit.setToolTip(f"{t} = {v:.10g}")
        except ParamError as e:
            self.res.setText("⚠")
            self.edit.setStyleSheet("background:#ffe3e3")
            self.edit.setToolTip(str(e))

    def showEvent(self, e):
        self._update()          # 親ウィンドウに入ってから変数を読み直す
        super().showEvent(e)

    def _finished(self):
        if not self.ok():
            return
        r = self.raw()
        if r != self._last:
            self._last = r
            self.rawChanged.emit(r)

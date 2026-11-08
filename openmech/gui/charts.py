"""簡単な折れ線グラフと表のダイアログ（周波数応答・パス結果・モードの表）"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)


def chart_dialog(parent, title, xlabel, ylabel, series, logy=False, logx=False):
    """series: [(名前, x 配列, y 配列)]"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(820, 560)
    lay = QVBoxLayout(dlg)
    try:
        from PySide6.QtCharts import QChart, QChartView, QLineSeries, QLogValueAxis, QValueAxis
        from PySide6.QtGui import QPainter
        ch = QChart()
        ch.setTitle(title)
        ax = QLogValueAxis() if logx else QValueAxis()
        ay = QLogValueAxis() if logy else QValueAxis()
        ax.setTitleText(xlabel)
        ay.setTitleText(ylabel)
        for a in (ax, ay):
            a.setLabelFormat("%.4g")
        ch.addAxis(ax, 0x20)      # Qt.AlignBottom
        ch.addAxis(ay, 0x1)       # Qt.AlignLeft
        for name, x, y in series:
            ln = QLineSeries()
            ln.setName(name)
            for a, b in zip(np.asarray(x, float), np.asarray(y, float)):
                if np.isfinite(b) and (not logy or b > 0) and (not logx or a > 0):
                    ln.append(float(a), float(b))
            ch.addSeries(ln)
            ln.attachAxis(ax)
            ln.attachAxis(ay)
        v = QChartView(ch)
        v.setRenderHint(QPainter.Antialiasing)
        lay.addWidget(v)
    except Exception:  # noqa: BLE001  QtCharts が無い環境では表だけ
        pass
    rows = max(len(s[1]) for s in series)
    tab = QTableWidget(rows, 1 + len(series))
    tab.setHorizontalHeaderLabels([xlabel] + [f"{s[0]}（{ylabel}）" for s in series])
    for i in range(rows):
        tab.setItem(i, 0, QTableWidgetItem(f"{series[0][1][i]:.6g}"))
        for j, s in enumerate(series):
            if i < len(s[2]):
                tab.setItem(i, j + 1, QTableWidgetItem(f"{s[2][i]:.6g}"))
    tab.setMaximumHeight(170)
    lay.addWidget(tab)
    h = QHBoxLayout()
    b = QPushButton("CSV に保存…")

    def save():
        path, _ = QFileDialog.getSaveFileName(dlg, "CSV に保存", "data.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(",".join([xlabel] + [s[0] for s in series]) + "\n")
                for i in range(rows):
                    f.write(",".join([f"{series[0][1][i]:.8g}"] + [f"{s[2][i]:.8g}" if i < len(s[2]) else ""
                                                                   for s in series]) + "\n")
    b.clicked.connect(save)
    h.addStretch()
    h.addWidget(b)
    lay.addLayout(h)
    dlg.exec()


def table_dialog(parent, title, rows: list[dict]):
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(820, 420)
    lay = QVBoxLayout(dlg)
    cols = list(rows[0].keys()) if rows else []
    tab = QTableWidget(len(rows), len(cols))
    tab.setHorizontalHeaderLabels(cols)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r[c]
            tab.setItem(i, j, QTableWidgetItem(f"{v:.6g}" if isinstance(v, float) else str(v)))
    lay.addWidget(tab)
    dlg.exec()

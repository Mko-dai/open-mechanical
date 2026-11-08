"""変数（Femtet の［変数］）とパラメトリック解析のダイアログ

- VariablesDialog: 変数名 / 式（値） / 値 / コメント の表。式の中で他の変数を使える
- ParametricDialog: スイープする変数（開始・終了・分割数 または 値の列挙）、組み合わせ方、結果の出力項目
- ParametricResultDialog: 結果の表とグラフ（X 軸 = 変数、Y 軸 = 出力、系列 = もう 1 つの変数）、CSV 保存
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..params import ParamError, check_name, evaluate, is_number, resolve, sweep_cases, sweep_values
from ..parametric import SPECIAL, STATS, output_label, table, write_csv

FIELDS = ["全変形 [mm]", "変位 X [mm]", "変位 Y [mm]", "変位 Z [mm]", "相当応力（ミーゼス） [MPa]",
          "最大主応力 [MPa]", "最小主応力 [MPa]", "応力 σx [MPa]", "応力 σy [MPa]", "応力 σz [MPa]",
          "安全率（降伏基準）", "相当塑性ひずみ"]
DEFAULT_OUT = {"全変形 [mm]": "max", "相当応力（ミーゼス） [MPa]": "max", "質量 [kg]": None}


def _ok_cancel(dlg, lay, ok="OK"):
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.button(QDialogButtonBox.Ok).setText(ok)
    bb.button(QDialogButtonBox.Cancel).setText("キャンセル")
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    return bb


def _item(text, editable=True):
    it = QTableWidgetItem(str(text))
    if not editable:
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setForeground(QColor("#2a6fdb"))
    return it


# ---------------------------------------------------------------------------- 変数

class VariablesDialog(QDialog):
    """［変数］ 変数名・式・コメント。値の列は式を計算した結果（読み取り専用）"""

    def __init__(self, params: list, parent=None, used: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("変数")
        self.resize(640, 420)
        self.used = used or {}
        lay = QVBoxLayout(self)
        note = QLabel("寸法・荷重・メッシュサイズの入力欄に、数値の代わりに変数名や式（例: L/2+5）を書けます。\n"
                      "変数の式には他の変数も使えます（例: W = L/5）。べき乗は ^、関数は sqrt, sin(ラジアン), sind(度), "
                      "min, max, abs など")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555")
        lay.addWidget(note)
        self.t = QTableWidget(0, 4)
        self.t.setHorizontalHeaderLabels(["変数名", "式（値）", "値", "コメント"])
        self.t.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.t.setColumnWidth(0, 110)
        self.t.setColumnWidth(1, 170)
        self.t.setColumnWidth(2, 110)
        lay.addWidget(self.t)
        for p in params:
            self._add(p.get("name", ""), p.get("expr", ""), p.get("comment", ""))
        row = QHBoxLayout()
        for text, fn in [("追加", lambda: self._add(self._new_name(), "0", "")), ("削除", self._del),
                         ("上へ", lambda: self._move(-1)), ("下へ", lambda: self._move(1))]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        _ok_cancel(self, lay)
        self.t.itemChanged.connect(lambda *_: self._recalc())
        self._recalc()

    def _new_name(self):
        names = {p["name"] for p in self.params(validate=False)}
        for c in ["L", "W", "H", "T", "R", "D", "F", "P"] + [f"x{i}" for i in range(1, 100)]:
            if c not in names:
                return c
        return "x"

    def _add(self, name, expr, comment):
        self.t.blockSignals(True)
        r = self.t.rowCount()
        self.t.insertRow(r)
        self.t.setItem(r, 0, _item(name))
        self.t.setItem(r, 1, _item(expr if isinstance(expr, str) else f"{expr:g}"))
        self.t.setItem(r, 2, _item("", editable=False))
        self.t.setItem(r, 3, _item(comment))
        self.t.blockSignals(False)
        self._recalc()

    def _del(self):
        r = self.t.currentRow()
        if r < 0:
            return
        nm = self.t.item(r, 0).text().strip()
        if nm in self.used:
            if QMessageBox.question(self, "変数の削除", f"変数「{nm}」は次の場所で使われています:\n  "
                                    + "\n  ".join(self.used[nm][:8]) + "\n削除しますか？（使っている欄はエラーになります）"
                                    ) != QMessageBox.Yes:
                return
        self.t.removeRow(r)
        self._recalc()

    def _move(self, d):
        r = self.t.currentRow()
        r2 = r + d
        if r < 0 or not (0 <= r2 < self.t.rowCount()):
            return
        self.t.blockSignals(True)
        for c in range(4):
            a, b = self.t.takeItem(r, c), self.t.takeItem(r2, c)
            self.t.setItem(r, c, b)
            self.t.setItem(r2, c, a)
        self.t.blockSignals(False)
        self.t.setCurrentCell(r2, 1)

    def params(self, validate=True) -> list:
        out = []
        for r in range(self.t.rowCount()):
            nm = (self.t.item(r, 0).text() if self.t.item(r, 0) else "").strip()
            ex = (self.t.item(r, 1).text() if self.t.item(r, 1) else "").strip() or "0"
            cm = (self.t.item(r, 3).text() if self.t.item(r, 3) else "").strip()
            if not nm and not validate:
                continue
            out.append({"name": nm, "expr": ex, "comment": cm})
        return out

    def _recalc(self):
        self.t.blockSignals(True)
        try:
            ps = self.params(validate=False)
            try:
                for p in ps:
                    check_name(p["name"])
                env = resolve(ps)
                self.msg.setText(f"変数 {len(ps)} 個")
                self.msg.setStyleSheet("color:#2f9e44")
            except ParamError as e:
                env = {}
                self.msg.setText(f"⚠ {e}")
                self.msg.setStyleSheet("color:#c92a2a")
            for r in range(self.t.rowCount()):
                nm = self.t.item(r, 0).text().strip() if self.t.item(r, 0) else ""
                it = self.t.item(r, 2)
                if it is not None:
                    it.setText(f"{env[nm]:.8g}" if nm in env else "—")
        finally:
            self.t.blockSignals(False)

    def accept(self):
        try:
            ps = self.params()
            for p in ps:
                check_name(p["name"])
            resolve(ps)
        except ParamError as e:
            QMessageBox.warning(self, "変数", str(e))
            return
        super().accept()


# ---------------------------------------------------------------------------- パラメトリック解析の設定

class ParametricDialog(QDialog):
    MODES = [("all", "全組み合わせ（総当たり）"), ("each", "1 変数ずつ（ほかは現在の値）"),
             ("zip", "同時に変える（同じ番号の値どうし）")]

    def __init__(self, params: list, env: dict, settings: dict | None = None, parent=None, transient=False):
        super().__init__(parent)
        self.setWindowTitle("パラメトリック解析の設定")
        self.resize(820, 620)
        self.params_ = params
        self.env = env
        st = settings or {}
        sw = st.get("sweeps", {})
        lay = QVBoxLayout(self)
        if not params:
            lay.addWidget(QLabel("変数がありません。先に［変数］で変数を作り、寸法などの欄で使ってください。"))
        lay.addWidget(QLabel("<b>スイープする変数</b>（チェックした変数を変えながら解析します）"))
        self.vt = QTableWidget(0, 6)
        self.vt.setHorizontalHeaderLabels(["変数", "現在の値", "開始値", "終了値", "分割数", "値を列挙（任意: 10, 20, 35）"])
        self.vt.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.checks = []
        for p in params:
            nm = p["name"]
            r = self.vt.rowCount()
            self.vt.insertRow(r)
            cb = QCheckBox(nm)
            cb.setChecked(nm in sw)
            cb.toggled.connect(self._count)
            self.checks.append(cb)
            self.vt.setCellWidget(r, 0, cb)
            cur = env.get(nm, 0.0)
            self.vt.setItem(r, 1, _item(f"{cur:.6g}" + ("" if is_number(p.get("expr", 0)) else f"（{p['expr']}）"),
                                        editable=False))
            s = sw.get(nm, {})
            self.vt.setItem(r, 2, _item(s.get("start", f"{cur * 0.8:.6g}" if cur else "0")))
            self.vt.setItem(r, 3, _item(s.get("stop", f"{cur * 1.2:.6g}" if cur else "1")))
            self.vt.setItem(r, 4, _item(s.get("n", "4")))
            self.vt.setItem(r, 5, _item(s.get("values", "")))
        self.vt.itemChanged.connect(lambda *_: self._count())
        lay.addWidget(self.vt, 2)
        row = QHBoxLayout()
        row.addWidget(QLabel("組み合わせ方"))
        self.mode = QComboBox()
        for k, v in self.MODES:
            self.mode.addItem(v, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(st.get("mode", "all"))))
        self.mode.currentIndexChanged.connect(self._count)
        row.addWidget(self.mode)
        row.addStretch()
        self.count = QLabel("")
        row.addWidget(self.count)
        lay.addLayout(row)
        lay.addWidget(QLabel("<b>結果の出力</b>（ケースごとに表とグラフにまとめる値。過渡解析は全時刻での最大・最小）"))
        self.ot = QTableWidget(0, 2)
        self.ot.setHorizontalHeaderLabels(["結果", "値"])
        self.ot.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        prev = {o["field"]: o.get("stat") for o in st.get("outputs", [])} or DEFAULT_OUT
        self.out_rows = []
        for f in FIELDS + SPECIAL:
            r = self.ot.rowCount()
            self.ot.insertRow(r)
            cb = QCheckBox(f)
            cb.setChecked(f in prev)
            self.ot.setCellWidget(r, 0, cb)
            if f in SPECIAL:
                combo = None
                self.ot.setItem(r, 1, _item("—", editable=False))
            else:
                combo = QComboBox()
                for k, v in STATS.items():
                    combo.addItem(v, k)
                combo.setCurrentIndex(max(0, combo.findData(prev.get(f) or ("min" if f.startswith("安全率") or
                                                                             f.startswith("最小主") else "max"))))
                self.ot.setCellWidget(r, 1, combo)
            self.out_rows.append((f, cb, combo))
        lay.addWidget(self.ot, 2)
        note = QLabel("ケースごとに 形状の作成 → メッシュ → 解析 を行います。境界条件の面は番号で対応させるため、"
                      "形状の変更で面の向きが変わったケースはエラーとして飛ばします。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555")
        lay.addWidget(note)
        _ok_cancel(self, lay, "実行")
        self._count()

    def sweeps(self) -> dict:
        out = {}
        for r, cb in enumerate(self.checks):
            if not cb.isChecked():
                continue
            g = lambda c: (self.vt.item(r, c).text().strip() if self.vt.item(r, c) else "")  # noqa: E731
            out[cb.text()] = {"start": g(2), "stop": g(3), "n": g(4), "values": g(5)}
        return out

    def sweep_lists(self) -> dict:
        out = {}
        for nm, s in self.sweeps().items():
            try:
                if s["values"]:
                    vals = sweep_values(0, 0, values_text=s["values"])
                else:
                    vals = sweep_values(evaluate(s["start"], self.env), evaluate(s["stop"], self.env),
                                        n=int(float(s["n"] or 1)))
            except (ParamError, ValueError) as e:
                raise ParamError(f"変数「{nm}」のスイープ: {e}") from None
            if not vals:
                raise ParamError(f"変数「{nm}」の値がありません")
            out[nm] = vals
        return out

    def cases(self) -> list:
        return sweep_cases(self.sweep_lists(), self.mode.currentData())

    def outputs(self) -> list:
        return [{"field": f, **({"stat": c.currentData()} if c is not None else {})}
                for f, cb, c in self.out_rows if cb.isChecked()]

    def settings(self) -> dict:
        return {"sweeps": self.sweeps(), "mode": self.mode.currentData(), "outputs": self.outputs()}

    def _count(self, *_):
        try:
            n = len(self.cases())
            self.count.setText(f"ケース数: {n}")
            self.count.setStyleSheet("color:#2f9e44" if n else "color:#c92a2a")
        except ParamError as e:
            self.count.setText(f"⚠ {e}")
            self.count.setStyleSheet("color:#c92a2a")

    def accept(self):
        try:
            cs = self.cases()
        except ParamError as e:
            QMessageBox.warning(self, "パラメトリック解析", str(e))
            return
        if not cs:
            QMessageBox.warning(self, "パラメトリック解析", "スイープする変数をチェックしてください。")
            return
        if not self.outputs():
            QMessageBox.warning(self, "パラメトリック解析", "出力する結果を 1 つ以上選んでください。")
            return
        if len(cs) > 200 and QMessageBox.question(self, "パラメトリック解析",
                                                  f"ケースが {len(cs)} 個あります。実行しますか？") != QMessageBox.Yes:
            return
        super().accept()


# ---------------------------------------------------------------------------- 結果

class ParametricResultDialog(QDialog):
    """結果の表とグラフ。［このケースを反映］で選んだ行の変数の値をモデルに入れる"""

    def __init__(self, rows: list, swept: list, parent=None, apply_case=None):
        super().__init__(parent)
        self.setWindowTitle("パラメトリック解析の結果")
        self.resize(1000, 680)
        self.rows = rows
        self.swept = list(swept)
        self.apply_case = apply_case
        all_vars = list(rows[0]["変数"]) if rows else []
        self.var_names = self.swept + [v for v in all_vars if v not in self.swept]
        lay = QVBoxLayout(self)
        sp = QSplitter(Qt.Vertical)
        lay.addWidget(sp, 1)
        # 表
        head, body = table(rows, self.var_names)
        self.head = head
        self.t = QTableWidget(len(body), len(head))
        self.t.setHorizontalHeaderLabels(head)
        self.t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t.setSelectionBehavior(QAbstractItemView.SelectRows)
        for r, row in enumerate(body):
            for c, v in enumerate(row):
                it = QTableWidgetItem(f"{v:.6g}" if isinstance(v, float) else str(v))
                if isinstance(v, (int, float)):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == len(row) - 1 and str(v).startswith("エラー"):
                    it.setForeground(QColor("#c92a2a"))
                self.t.setItem(r, c, it)
        self.t.resizeColumnsToContents()
        sp.addWidget(self.t)
        # グラフ
        gw = QWidget()
        gl = QVBoxLayout(gw)
        gl.setContentsMargins(0, 0, 0, 0)
        ctl = QHBoxLayout()
        self.outs = [h for h in head if h not in ("番号", "状態") and h not in self.var_names]
        self.cx, self.cy, self.cs = QComboBox(), QComboBox(), QComboBox()
        self.cx.addItems(self.var_names or ["番号"])
        self.cy.addItems(self.outs)
        self.cs.addItem("なし")
        self.cs.addItems([v for v in self.swept])
        if len(self.swept) >= 2:
            self.cs.setCurrentIndex(2)
        for lab, w in (("横軸", self.cx), ("縦軸", self.cy), ("系列", self.cs)):
            ctl.addWidget(QLabel(lab))
            ctl.addWidget(w)
            w.currentIndexChanged.connect(self._plot)
        ctl.addStretch()
        gl.addLayout(ctl)
        self.chart_host = QVBoxLayout()
        gl.addLayout(self.chart_host, 1)
        sp.addWidget(gw)
        sp.setSizes([300, 360])
        # ボタン
        row = QHBoxLayout()
        b = QPushButton("CSV に保存…")
        b.clicked.connect(self._csv)
        row.addWidget(b)
        if apply_case is not None:
            b = QPushButton("選んだケースの値をモデルに反映")
            b.clicked.connect(self._apply)
            row.addWidget(b)
        row.addStretch()
        b = QPushButton("閉じる")
        b.clicked.connect(self.accept)
        row.addWidget(b)
        lay.addLayout(row)
        self.view = None
        self._plot()

    def _plot(self, *_):
        x_name, y_name, s_name = self.cx.currentText(), self.cy.currentText(), self.cs.currentText()
        ok = [r for r in self.rows if r["状態"] == "完了" and y_name in r["出力"]]
        groups = {}
        for r in ok:
            key = r["変数"].get(s_name) if s_name != "なし" else None
            x = r["変数"].get(x_name, r["番号"])
            groups.setdefault(key, []).append((float(x), float(r["出力"][y_name])))
        try:
            from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
        except Exception:      # QtCharts が無い環境では表だけ
            return
        ch = QChart()
        xs = [p[0] for g in groups.values() for p in g]
        ys = [p[1] for g in groups.values() for p in g]
        ax, ay = QValueAxis(), QValueAxis()
        ax.setTitleText(x_name)
        ay.setTitleText(y_name)
        if xs:
            lo, hi = min(xs), max(xs)
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.setRange(lo - pad, hi + pad)
            lo, hi = min(ys), max(ys)
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi) * 0.1, 1e-9)
            ay.setRange(lo - pad, hi + pad)
        ch.addAxis(ax, Qt.AlignBottom)
        ch.addAxis(ay, Qt.AlignLeft)
        for key, pts in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            pts.sort()
            name = y_name if key is None else f"{s_name} = {key:g}"
            ln = QLineSeries()
            ln.setName(name)
            sc = QScatterSeries()
            sc.setMarkerSize(8)
            for x, y in pts:
                ln.append(x, y)
                sc.append(x, y)
            for s in (ln, sc):
                ch.addSeries(s)
                s.attachAxis(ax)
                s.attachAxis(ay)
            sc.setColor(ln.color())
            ch.legend().markers(sc)[0].setVisible(False)
        ch.legend().setVisible(len(groups) > 1)
        ch.setTitle(f"{y_name} と {x_name}")
        if self.view is not None:
            self.view.setParent(None)
        self.view = QChartView(ch)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.chart_host.addWidget(self.view)

    def _csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "パラメトリック解析の結果", "parametric.csv", "CSV (*.csv)")
        if path:
            write_csv(path, self.rows, self.var_names)

    def _apply(self):
        r = self.t.currentRow()
        if r < 0:
            QMessageBox.information(self, "反映", "表で行を選んでください。")
            return
        row = self.rows[r]
        self.apply_case({k: row["変数"][k] for k in self.swept if k in row["変数"]})

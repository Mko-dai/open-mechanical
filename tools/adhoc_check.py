"""ウィンドウの大きさの確認（Mac で実行）: 画面より大きくならないか、フル画面・狭い幅で端が切れないか"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading  # noqa: E402
threading.Timer(90.0, lambda: os._exit(3)).start()      # 止まったままにならないよう、90 秒で必ず終わる
import subprocess  # noqa: E402
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_shell.py", "-k", "surface_body_split"],
                    cwd=_root, capture_output=True, text=True)
print("=== サーフェスの切断のテスト\n" + _r.stdout[-1500:] + _r.stderr[-500:], flush=True)
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
import pyvista as pv  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from openmech.gui import app as A  # noqa: E402
from openmech.gui.femtet_dialogs import FeatureDialog  # noqa: E402

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
FeatureDialog.exec = lambda self: 1
QMessageBox.warning = staticmethod(lambda *a, **k: 0)
QMessageBox.information = staticmethod(lambda *a, **k: 0)
QDialog.exec = lambda self: 0
pv.global_theme.allow_empty_mesh = True
app = QApplication.instance() or QApplication(sys.argv)
scr = app.primaryScreen()
print("画面", scr.geometry().width(), scr.geometry().height(), "/ 使える範囲", scr.availableGeometry().width(),
      scr.availableGeometry().height(), flush=True)
w = A.MainWindow()
w.show()
w.create_body("box", dialog=True)


def pump(ms):
    t = QTimer()
    t.setSingleShot(True)
    t.timeout.connect(app.quit)
    t.start(ms)
    app.exec()


def report(tag):
    pump(1500)
    cw = w.centralWidget()
    print(f"{tag}: ウィンドウ {w.width()}×{w.height()} / 最小 {w.minimumWidth()}×{w.minimumHeight()} / "
          f"中身が欲しい最小 {cw.minimumSizeHint().width()}×{cw.minimumSizeHint().height()} / "
          f"リボン {w.ribbon.width()}×{w.ribbon.height()} / 3D 表示 {w.plotter.interactor.width()}×{w.plotter.interactor.height()}",
          flush=True)
    for i in range(w.ribbon.count()):
        pg = w.ribbon.widget(i)
        print(f"   タブ「{w.ribbon.tabText(i)}」: 中身の幅 {pg.inner.minimumSizeHint().width()} / 見えている幅 {pg.viewport().width()}",
              flush=True)
    w.grab().save(os.path.join(out, f"window_{tag}.png"))


report("normal")
# フル画面と同じ大きさにして確かめる（本当のフル画面にはしない: 作業中の画面を奪わないため）
w.resize(scr.geometry().width(), scr.geometry().height())
report("fullsize")
for i in range(w.ribbon.count()):
    w.ribbon.setCurrentIndex(i)
    pump(300)
    w.grab().save(os.path.join(out, f"window_fullsize_tab{i}.png"))
gb, ab = [x for x in w.findChildren(A.HScroll) if x.fixed][:2]
print(f"ツールバーの高さ {gb.height()}・属性バーの高さ {ab.height()} / 3D 表示 {w.plotter.interactor.height()}", flush=True)
ok = gb.height() < 60 and ab.height() < 80 and w.plotter.interactor.height() > 400 and w.width() <= scr.geometry().width()
print(("OK  " if ok else "NG  ") + "フル画面: 画面の幅に収まり、ツールバーは 1 行ぶんの高さ")
print("完了", flush=True)
os._exit(0)

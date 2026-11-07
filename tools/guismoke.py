"""GUI の動作確認（開発用）: ダイアログを自動で OK にして、ボディ編集・操作履歴・2D・対称を一通り実行"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
from PySide6.QtWidgets import QApplication, QMessageBox
import pyvista as pv
from openmech.gui import app as A
from openmech.gui.femtet_dialogs import FeatureDialog
from openmech.solver import BC

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(out, exist_ok=True)
logf = open(os.path.join(out, "guismoke.txt"), "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


FeatureDialog.exec = lambda self: 1
msgs = []
QMessageBox.warning = staticmethod(lambda *a, **k: msgs.append(a[2] if len(a) > 2 else a) or 0)
QMessageBox.information = staticmethod(lambda *a, **k: msgs.append(a[2] if len(a) > 2 else a) or 0)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
pv.global_theme.allow_empty_mesh = True
app = QApplication.instance() or QApplication(sys.argv)
w = A.MainWindow()
w.show()


def step(name, fn):
    try:
        fn()
        app.processEvents()
        w._hist_commit()
        nb = len(w.mesh.body_ids()) if w.mesh is not None else 0
        vol = w.mesh.volume() if w.mesh is not None and hasattr(w.mesh, "volume") else float("nan")
        P(f"OK  {name}: ボディ {nb} / 体積 {vol:.6g} / 履歴 {len(w.history.items)} 件 (pos {w.history.pos})"
          + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
        msgs.clear()
    except Exception:
        P(f"NG  {name}\n" + traceback.format_exc())


def sel(bodies):
    w.selected_bodies = list(bodies)


step("直方体", lambda: w.create_body("box"))
step("移動（コピー）", lambda: (sel([0]), w.edit_body_op("move")))
f = w._features()
step("コピーを Z に 30 移動", lambda: (f[-1].update(dz=30.0, copy="コピー"), w.remesh()))
step("切断", lambda: (sel([]), w.edit_body_op("split")))
step("引き伸ばし", lambda: (sel([0]), w.edit_body_op("stretch")))
step("丸み付け", lambda: (sel([0]), w.edit_body_op("fillet")))
step("面取り", lambda: (sel([1]), w.edit_body_op("chamfer")))
step("ミラー", lambda: (sel([2]), w.edit_body_op("mirror")))
for it in w.history.items:
    P("   履歴:", it["time"], it["label"])
step("元に戻す", w.undo)
step("元に戻す", w.undo)
step("やり直す", w.redo)
step("3番目の時点に戻す", lambda: w.restore_history(2))
w.show_history()
app.processEvents()
w._hist_dialog.grab().save(os.path.join(out, "gui_history.png"))
w.grab().save(os.path.join(out, "gui_edit.png"))
P("フィーチャー:", [w._feature_label(x) for x in w._features()])

# 2D（平面応力）: 穴あき板 1/4 モデル
w.project.geometry = {"kind": "model", "analysis_2d": "plane_stress", "features": []}
w.project.bcs = []
w.remesh()
step("2D 平板", lambda: w.create_body("plate"))
step("2D 穴", lambda: (sel([]), w.create_hole()))
m = w.mesh
P("2D メッシュ:", type(m).__name__, getattr(m, "mode", None), m.n_nodes, "節点")
info = {}
for e, seg in enumerate(getattr(m, "edges3", [])):
    pts = m.nodes[seg.ravel()]
    info[m.n_surf + e] = (pts.min(0), pts.max(0))
lo, hi = m.bounds()
try:
    w.project.bcs = [BC("fixed", [k for k, (a, b) in info.items() if b[0] < lo[0] + 1e-6]),
                     BC("force", [k for k, (a, b) in info.items() if a[0] > hi[0] - 1e-6], [1000.0, 0, 0])]
    r = w.project.run(m, log=lambda *a: None)
    P("2D 解析 OK: 最大ミーゼス", round(float(r.von_mises.max()), 2), "MPa")
    w.result = r
    w.results = [r]
    w.rebuild_tree()
    w.refresh_view()
    app.processEvents()
    w.grab().save(os.path.join(out, "gui_2d.png"))
except Exception:
    P("NG 2D 解析\n" + traceback.format_exc())
P("完了")
logf.close()
w.close()
app.quit()

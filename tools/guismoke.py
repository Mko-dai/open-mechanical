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


def w2d(p):
    ren = w.plotter.renderer
    ren.SetWorldPoint(*p, 1.0)
    ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return int(round(x)), int(round(y))


def sketch_box():
    w.create_body("box")
    assert w._sk is not None, "作図モードに入らない"
    w._sk.pitch = 1.0
    for p in [(0, 0, 0), (50, 30, 0), (50, 30, 20)]:
        x, y = w2d(p)
        w._sketch_move(x, y)
        app.processEvents()
        w._sketch_click(x, y)
        app.processEvents()
    P("   作図の結果:", {k: w._features()[-1][k] for k in ("x", "y", "z", "dx", "dy", "dz")})


step("直方体（マウスで作図）", sketch_box)
w.grab().save(os.path.join(out, "gui_sketch_done.png"))
w.create_body("cylinder")
w._sk.pitch = 1.0
x, y = w2d((80, 15, 20))
w._sketch_move(x, y)
x2, y2 = w2d((90, 15, 20))
w._sketch_click(x, y)
w._sketch_move(x2, y2)
app.processEvents()
w.grab().save(os.path.join(out, "gui_sketch_preview.png"))
P("   円柱の作図中: 作業平面", w._sk.plane, w._sk.offset, "点", [list(map(float, q)) for q in w._sk.points])
w.sketch_cancel()
P("   作図の取り消し: OK" if w._sk is None else "NG 作図の取り消し")
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
step("2D 平板", lambda: w.create_body("plate", dialog=True))
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
# 過渡構造（陽解法）: 直方体を床に落とす
try:
    w.project.geometry = {"kind": "model", "features": []}
    w.project.bcs = []
    w.remesh()
    w.create_body("box", dialog=True)
    w._set_analysis_type("explicit")
    app.processEvents()
    lo, hi = w.mesh.bounds()
    w.project.dynamic.update(end_time=2e-5, n_frames=10, init_velocity=[0, 0, -4429.0],
                             wall={"normal": [0, 0, 1], "point": [0, 0, float(lo[2])], "friction": 0.2})
    w.select_node(("settings",))
    app.processEvents()
    fr = w.project.run_all(w.mesh, log=lambda *a: None)
    w._on_solved(fr)
    app.processEvents()
    w.show_history_plot()
    app.processEvents()
    P(f"過渡（陽解法）OK: {len(fr)} 時刻 / 最大接触力 {fr[-1].info['最大接触力 [N]']:.1f} N / "
      f"時間増分 {fr[-1].info['時間増分 [s]']:.2e} s / 木の表示 {w._analysis_label()}")
    w.grab().save(os.path.join(out, "gui_transient.png"))
    w._set_analysis_type("implicit")
    w.project.dynamic.update(dt=1e-6)
    fr = w.project.run_all(w.mesh, log=lambda *a: None)
    w._on_solved(fr)
    app.processEvents()
    P(f"過渡（陰解法）OK: {len(fr)} 時刻 / 最大接触力 {fr[-1].info['最大接触力 [N]']:.1f} N")
except Exception:
    P("NG 過渡\n" + traceback.format_exc())
P("完了")
logf.close()
w.close()
app.quit()

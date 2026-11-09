"""GUI の動作確認（新しい解析の種類と境界条件）: 直方体の片持ちはりで、各解析を GUI から実行して画面を保存する"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
import pyvista as pv  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from openmech.gui import app as A  # noqa: E402
from openmech.gui import charts as C  # noqa: E402
from openmech.gui.femtet_dialogs import FeatureDialog  # noqa: E402
from openmech.solver import BC  # noqa: E402

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "gui_analyses")
os.makedirs(out, exist_ok=True)
logf = open(os.path.join(out, "log.txt"), "w")


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
shown = []
C.chart_dialog = lambda parent, title, *a, **k: shown.append(title)
C.table_dialog = lambda parent, title, rows: shown.append((title, len(rows)))
QDialog.exec = lambda self: 0
pv.global_theme.allow_empty_mesh = True
app = QApplication.instance() or QApplication(sys.argv)
w = A.MainWindow()
w.resize(1500, 900)
w.show()

# 直方体 100×50×20（作成の既定）
w.create_body("box", dialog=True)
app.processEvents()
info = w.mesh.face_info()
lo, hi = w.mesh.bounds()


def fx(axis, val, sign):
    return [f["id"] for f in info if f["flat"] and abs(f["centroid"][axis] - val) < 1e-6
            and f["normal"][axis] * sign > 0.99]


left, right, top, bot = fx(0, lo[0], -1), fx(0, hi[0], 1), fx(2, hi[2], 1), fx(2, lo[2], -1)
mat = w.project.material
mat.conductivity, mat.specific_heat = 60.5, 434.0


def run(name, atype, bcs, opts=None, results=None, shot=None):
    try:
        w._set_analysis_type(atype)
        if opts:
            w.project.analysis_opts.update(opts)
        w.project.bcs = list(bcs)
        if results:
            w.project.results = list(results)
        w.invalidate()
        w.rebuild_tree()
        w.select_node(("analysis",))
        app.processEvents()
        fr = w.project.run_all(w.mesh, log=lambda *a: None)
        w._on_solved(fr)
        app.processEvents()
        # 各結果を選んで詳細とグラフのボタンを試す
        for i in range(len(w.project.results)):
            w.select_node(("result", i))
            app.processEvents()
        w.select_node(("result", 0))
        app.processEvents()
        w.grab().save(os.path.join(out, (shot or atype) + ".png"))
        r = w.results[-1]
        P(f"OK  {name}: 結果 {len(fr)} 個 / 表示 {[getattr(x, 'label', '') for x in fr][:4]} / "
          + ", ".join(f"{k} {v:.4g}" for k, v in [kv for kv in r.summary().items()
                                                   if isinstance(kv[1], (int, float))][:3])
          + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
        msgs.clear()
        return fr
    except Exception:
        P(f"NG  {name}\n" + traceback.format_exc())


fix = BC("fixed", left)
run("モーダル", "modal", [fix], {"n_modes": 6})
w._details_result_tools("全変形 [mm]")
P("   グラフ・表:", shown[-1:] if shown else "なし")
run("固有値座屈", "buckling", [fix, BC("force", right, [-1000.0, 0, 0])], {"n_buckling": 2})
run("定常熱", "thermal", [BC("temperature", left, [100.0]), BC("convection", right, [50.0, 22.0])])
run("過渡熱", "thermal_t", [BC("temperature", left, [100.0])],
    {"thermal": {"transient": True, "end_time": 60.0, "n_steps": 6, "init_temp": 22.0}}, shot="thermal_transient")
run("熱応力（連成）", "static", [fix, BC("temperature", left, [100.0]), BC("convection", right, [50.0, 22.0])],
    {"thermal_import": True}, shot="thermal_stress")
w.project.analysis_opts["thermal_import"] = False
run("調和応答", "harmonic", [fix, BC("force", right, [0, 0, -100.0])], {"n_modes": 10, "f_max_h": 3000.0,
                                                                       "n_freq": 60},
    results=["変位振幅 [mm]", "相当応力（ミーゼス） [MPa]"])
w.select_node(("result", 0))
w._details_result_tools("変位振幅 [mm]")
P("   グラフ:", shown[-1:])
run("ランダム振動", "random", [fix], {"n_modes": 10, "psd": [[20, 0.01], [2000, 0.01]], "psd_dir": [0, 0, 1]})
run("リモート荷重＋ばね", "static", [fix, BC("remote_force", right, [0, 0, -1000, 0, 0, 0], point=[hi[0] + 50, 25, 10],
                                            options={"behavior": "rigid"})], shot="remote")
run("ベアリング・弾性支持", "static", [BC("elastic_support", bot, [10.0]), BC("pressure", top, [1.0])], shot="elastic")
run("疲労ツール・式", "static", [fix, BC("force", right, [0, 0, -5000.0])],
    results=["疲労:寿命 [回]", "疲労:安全率（疲労）", "式:SEQV/250"], shot="fatigue")
w.select_node(("result", 0))
w._path_plot("疲労:寿命 [回]")
mat.permittivity = 4.0
run("静電場", "electrostatic", [BC("ground", bot), BC("voltage", top, [100.0])])
# 環境タブのメニュー（追加の境界条件）
for k in A.BC_SPEC:
    try:
        w.add_bc(k)
        w.select_node(("bc", len(w.project.bcs) - 1))
        app.processEvents()
    except Exception:
        P(f"NG  境界条件の追加 {k}\n" + traceback.format_exc())
P(f"境界条件の追加 OK: {len(A.BC_SPEC)} 種類")
w.grab().save(os.path.join(out, "bcs.png"))
# はり（骨組み）
# 収束ツール（適応メッシュ）: 結果の詳細から実行
try:
    run("収束前の静解析", "static", [fix, BC("force", right, [0, 0, -1000.0])], shot="conv_before")
    w.project.analysis_opts["convergence"] = {"allowable": 2.0, "max_loops": 2}
    w.select_node(("result", 1))
    app.processEvents()
    w.run_convergence("相当応力（ミーゼス） [MPa]")
    w.worker.wait()
    for _ in range(20):
        app.processEvents()
    hist = w.project.analysis_opts.get("convergence_history") or []
    w.select_node(("result", 1))
    app.processEvents()
    w.grab().save(os.path.join(out, "convergence.png"))
    P(f"OK  収束ツール: {len(hist)} 回 / 要素 {[h['要素数'] for h in hist]} / "
      f"変化 {[h['変化 [%]'] for h in hist]} / 表 {shown[-1] if shown else '-'}"
      + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
    msgs.clear()
except Exception:
    P("NG  収束ツール\n" + traceback.format_exc())
try:
    from openmech.gui.frame_dialog import EXAMPLE, FrameDialog
    d = FrameDialog(w, model=EXAMPLE)
    d.show()
    app.processEvents()
    d.run_static()
    app.processEvents()
    d.grab().save(os.path.join(out, "frame_static.png"))
    d.run_modal()
    app.processEvents()
    d.grab().save(os.path.join(out, "frame_modal.png"))
    P("OK  はり（骨組み）: " + d.msg.toPlainText().splitlines()[0])
    d.close()
except Exception:
    P("NG  はり（骨組み）\n" + traceback.format_exc())
# 材料のダイアログ（熱・電気・疲労の欄）
try:
    from openmech.gui.femtet_dialogs import MaterialDialog
    md = MaterialDialog("鋼", w.project.material, w)
    md.show()
    app.processEvents()
    mm = md.material()
    P(f"OK  材料ダイアログ: 熱伝導率 {mm.conductivity} / S-N {len(mm.sn_table)} 点 / 圧電 {bool(mm.piezo)}")
    md.close()
except Exception:
    P("NG  材料ダイアログ\n" + traceback.format_exc())
P("完了")
logf.close()
w.close()
app.quit()

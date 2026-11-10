"""GUI の動作確認（新しい解析の種類と境界条件）: 直方体の片持ちはりで、各解析を GUI から実行して画面を保存する"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from openmech.gui import app as A  # noqa: E402
from openmech.gui import charts as C  # noqa: E402
from openmech.gui.femtet_dialogs import FeatureDialog  # noqa: E402
from openmech.materials import Material  # noqa: E402
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
# 弧長法（大変形）: 先端を押す片持ちはりで、荷重係数 1 まで追う
try:
    w._set_analysis_type("static")
    st0 = w.project.settings
    st0.arc_length, st0.large_deflection, st0.arc_initial, st0.arc_max_steps = True, True, 0.25, 6
    fr = run("弧長法（大変形）", "static", [fix, BC("force", right, [0, 0, -20000.0])], shot="arclength")
    w.select_node(("analysis",))
    app.processEvents()
    P(f"OK  弧長法: 点 {len(fr)} / 荷重係数 {[round(float(r.time), 3) for r in fr][:6]}")
    st0.arc_length, st0.large_deflection = False, False
except Exception:
    P("NG  弧長法\n" + traceback.format_exc())
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
# 応答スペクトル
run("応答スペクトル（CQC）", "spectrum", [fix], opts={"n_modes": 10, "spec_method": "cqc", "spec_dir": [0, 0, 1],
                                               "spectrum": [[1, 0.5], [10, 1.0], [100, 1.0], [1000, 0.5]]},
    results=["変位（応答スペクトル） [mm]", "相当応力（応答スペクトル） [MPa]"], shot="spectrum")
# 過渡構造（陰解法）を非線形の解き方で（大変形オン → 静解析と同じ解き方に慣性力）
try:
    w._set_analysis_type("transient")
    w.project.dynamic = dict(w.project.dynamic or {}, method="implicit", end_time=1.5e-4, dt=5e-5, n_frames=3,
                             large_deflection=True, wall={})
    w.project.bcs = [fix, BC("force", right, [0, 0, -1000.0])]
    w.invalidate()
    w.select_node(("analysis",))
    app.processEvents()
    assert w.project.transient_nonlinear(w.mesh)
    fr = w.project.run_all(w.mesh, log=lambda *a: None)
    w._on_solved(fr)
    app.processEvents()
    w.grab().save(os.path.join(out, "transient_nl.png"))
    P(f"OK  過渡（陰解法・非線形）: 結果 {len(fr)} 個 / 最大たわみ {max(-r.u[:, 2].min() for r in fr):.4g} mm / "
      f"{fr[-1].info.get('解析')}")
    w.project.dynamic = dict(w.project.dynamic, large_deflection=False)
    w._set_analysis_type("static")
except Exception:
    P("NG  過渡（陰解法・非線形）\n" + traceback.format_exc())
# ビーム接続（面－地面）: 右の面を丸棒で地面につないで、上面を押す
try:
    w._set_analysis_type("static")
    bb = BC("beam", list(right), [3.0, 200000.0, 0.3],
            options={"ground": True, "ground_point": [float(hi[0]) + 40.0, 25.0, 10.0], "behavior": "rigid"})
    run("ビーム接続（面－地面）", "static", [fix, bb, BC("pressure", top, [1.0])], shot="beam")
    w.select_node(("bc", 1))
    app.processEvents()
    br = w.results[-1].beams[1]
    P(f"OK  ビーム接続: 軸力 {br['軸力 [N]']:.4g} N / せん断力 {br['せん断力 [N]']:.4g} N / "
      f"最大応力 {br['最大応力（軸＋曲げ） [MPa]']:.4g} MPa")
    bb2 = BC("beam", list(right), [3.0, 200000.0, 0.3, 1000.0], options=dict(bb.options))
    fr = run("ビーム接続（予張力 2 ステップ）", "static", [fix, bb2, BC("pressure", top, [1.0])], shot="beam_pre")
    P(f"OK  ビームの予張力: ステップ 1 の軸力 {fr[0].beams[1]['軸力 [N]']:.5g} N / ステップ 2 {fr[1].beams[1]['軸力 [N]']:.5g} N")
except Exception:
    P("NG  ビーム接続\n" + traceback.format_exc())
# サブモデル: 粗い結果を保存して、切断境界に与える
try:
    import tempfile
    from PySide6.QtWidgets import QFileDialog
    cpath = os.path.join(tempfile.mkdtemp(), "coarse.npz")
    run("サブモデル（粗い全体）", "static", [fix, BC("force", right, [0, 0, -1000.0])], shot="sub_coarse")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (cpath, ""))
    w.save_coarse_result()
    tip0 = float(w.results[-1].u[:, 2].min())
    sbc = BC("submodel", list(right), [], options={"file": cpath, "offset": [0.0, 0.0, 0.0]})
    run("サブモデル（切断境界）", "static", [fix, sbc], shot="submodel")
    w.select_node(("bc", 1))
    app.processEvents()
    tip1 = float(w.results[-1].u[:, 2].min())
    P(f"OK  サブモデル: 先端たわみ 全体 {tip0:.5g} / サブモデル {tip1:.5g} mm")
except Exception:
    P("NG  サブモデル\n" + traceback.format_exc())
# ガスケット接触の詳細
try:
    from openmech.contact import ContactRegion
    w.project.contacts.append(ContactRegion(name="ガスケット", type="gasket", contact_faces=list(top),
                                            target_faces=list(bot), auto=False))
    w.rebuild_tree()
    w.select_node(("contact", len(w.project.contacts) - 1))
    app.processEvents()
    P(f"OK  ガスケットの詳細: 表 {w.project.contacts[-1].gasket}")
    w.project.contacts[-1].type, w.project.contacts[-1].sliding = "frictional", "finite"
    w.select_node(("contact", len(w.project.contacts) - 1))
    app.processEvents()
    P("OK  摩擦接触の詳細（有限すべり）")
    w.project.contacts.pop()
    w.rebuild_tree()
except Exception:
    P("NG  ガスケットの詳細\n" + traceback.format_exc())
# ジョイント（地面との回転ジョイント＋ねじりばね）
try:
    jbc = BC("joint", list(fix.faces), [], options={"type": "revolute", "ground": True, "axis": [0, 0, 1],
                                                    "stiffness": [0, 0, 0, 0, 0, 1e9]})
    fr = run("ジョイント（回転＋ねじりばね）", "static", [jbc, BC("force", right, [0, -1000.0, 0])], shot="joint")
    w.select_node(("bc", 0))
    app.processEvents()
    jr = w.results[-1].joints[0]
    P(f"OK  ジョイント: 回転 {jr['相対回転（局所） [rad]'][2]:.4g} rad / 力 {np.round(jr['力（全体）'], 3)} / "
      f"モーメント z {jr['モーメント（全体）'][2]:.4g}")
except Exception:
    P("NG  ジョイント\n" + traceback.format_exc())
# 断面表示とアニメーション
try:
    w.select_node(("result", 1))
    app.processEvents()
    w.section_combo.setCurrentIndex(3)
    w.section_pos.setValue(40)
    app.processEvents()
    w.grab().save(os.path.join(out, "section.png"))
    npts = w._anim[0].n_points
    w.section_combo.setCurrentIndex(0)
    app.processEvents()
    w.anim_btn.setChecked(True)
    w.toggle_animation()
    p0 = np.array(w._anim[0].points)
    for _ in range(7):
        w._anim_step()
    moved = float(np.abs(np.array(w._anim[0].points) - p0).max())
    w.anim_btn.setChecked(False)
    w.toggle_animation()
    P(f"OK  断面・アニメーション: 断面の点 {npts:,} / アニメーションで動いた量 {moved:.4g} mm")
except Exception:
    P("NG  断面・アニメーション\n" + traceback.format_exc())
# すべて六面体のメッシュ
try:
    w.project.mesh_settings = dict(w.project.mesh_settings or {}, element_shape="hex")
    w.remesh()
    app.processEvents()
    info4 = w.mesh.face_info()
    xl = [f["id"] for f in info4 if f["flat"] and abs(f["centroid"][0] - lo[0]) < 1e-6]
    xh = [f["id"] for f in info4 if f["flat"] and abs(f["centroid"][0] - hi[0]) < 1e-6]
    fr = run("六面体メッシュの静解析", "static", [BC("fixed", xl), BC("force", xh, [0, 0, -1000.0])], shot="hexmesh")
    P(f"OK  六面体: 要素 {w.mesh.n_elems:,}（{w.mesh.elems.shape[1]} 節点）/ 面 {w.mesh.n_faces}")
    w.project.mesh_settings = dict(w.project.mesh_settings, element_shape="tet")
    w.remesh()
    app.processEvents()
except Exception:
    P("NG  六面体\n" + traceback.format_exc())
# 破壊（き裂を埋め込んで J 積分・K）
try:
    w.project.mesh_settings = dict(w.project.mesh_settings or {}, cracks=[
        {"name": "縁き裂", "front0": [25.0, 25.0, 0.0], "front1": [25.0, 25.0, 20.0], "normal": [0, 1, 0],
         "direction": [1, 0, 0], "depth": 25.0, "to_surface": True, "toughness": 3000.0}])
    w.remesh()
    app.processEvents()
    info2 = w.mesh.face_info()
    ylo = [f["id"] for f in info2 if f["flat"] and abs(f["centroid"][1]) < 1e-6]
    yhi = [f["id"] for f in info2 if f["flat"] and abs(f["centroid"][1] - 50) < 1e-6]
    fr = run("き裂（縁き裂の引張）", "static", [BC("fixed", ylo), BC("pressure", yhi, [-10.0])], shot="crack")
    n0 = len(shown)
    w.show_fracture()
    row = (w.results[-1].fracture or [{}])[0]
    P(f"OK  破壊: き裂面 {getattr(w.mesh, 'crack_faces', None)} / 節点 {w.mesh.n_nodes:,} / "
      f"J {row.get('J [N/mm]', float('nan')):.4g} / K {row.get('K（J から）[MPa√mm]', float('nan')):.4g} / "
      f"K_I(変位) {row.get('K_I（変位）[MPa√mm]', float('nan')):.4g} / 表 {shown[n0:] }"
      + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
    msgs.clear()
    w.project.mesh_settings = dict(w.project.mesh_settings, cracks=[])
except Exception:
    P("NG  破壊\n" + traceback.format_exc())
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
    # 超弾性のカーブフィット・クリープ
    from openmech.hyperelastic import uniaxial_nominal
    md = MaterialDialog("ゴム", Material("ゴム", 1.0, 0.49, 1100), w)
    md.hmodel.setCurrentIndex(md.hmodel.findData("mooney_rivlin"))
    from PySide6.QtWidgets import QTableWidgetItem
    lams = [1.2, 1.5, 2.0, 2.5, 3.0]
    md.htab.setRowCount(len(lams))
    for i, l in enumerate(lams):
        md.htab.setItem(i, 0, QTableWidgetItem(str(l)))
        md.htab.setItem(i, 1, QTableWidgetItem(str(float(uniaxial_nominal({"model": "mooney_rivlin", "C10": 0.3,
                                                                              "C01": 0.1}, l)))))
    md._hyper_fit()
    md.c1.setText("1e-12")
    mm = md.material()
    P(f"OK  超弾性・クリープ: {mm.hyper.get('model')} C10 {mm.hyper.get('C10'):.4f} C01 {mm.hyper.get('C01'):.4f} "
      f"（{md.hfit_msg.text()}） / クリープ {mm.creep}")
    md.close()
except Exception:
    P("NG  材料ダイアログ\n" + traceback.format_exc())
# 積層シェル（複合材）
try:
    from openmech.gui.femtet_dialogs import BodyAttrDialog
    w.new_project()
    app.processEvents()
    w.create_body("plate", dialog=True)
    app.processEvents()
    pj = w.project
    pj.materials["CFRP"] = Material("CFRP", 135000, 0.3, 1600, elastic="orthotropic",
                                    ortho={"Ex": 135000, "Ey": 10000, "Ez": 10000, "nuxy": 0.3, "nuyz": 0.4,
                                           "nuxz": 0.3, "Gxy": 5000, "Gyz": 3600, "Gxz": 5000},
                                    composite_strength={"Xt": 1500, "Xc": 1200, "Yt": 50, "Yc": 250, "S": 70})
    bd = BodyAttrDialog("積層板", {"mesh_size": 0.0, "layup": {"plies": [["CFRP", 0.25, a] for a in (0, 45, -45, 90, 90, -45, 45, 0)]}}, w)
    nm, props = bd.result()
    pj.body_attrs[nm] = props
    pj.assign(w.mesh.body_ids().tolist(), attr=nm, mat="CFRP")
    info3 = w.mesh.face_info()
    edges = [f for f in info3 if w.mesh.is_edge(f["id"])]
    e_lo = min(edges, key=lambda f: f["centroid"][0])["id"]
    e_hi = max(edges, key=lambda f: f["centroid"][0])["id"]
    fr = run("積層シェル（[0/45/-45/90]s）", "static", [BC("fixed", [e_lo]), BC("force", [e_hi], [0, 0, -50.0])],
             results=["全変形 [mm]", "複合材 破損指標"], shot="laminate")
    n0 = len(shown)
    w.show_plies()
    P(f"OK  積層: {w.results[-1].info.get('積層')} / 破損指標 最大 {w.results[-1].extra['複合材 破損指標'].max():.4g} / "
      f"層の表 {shown[n0:]}" + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
    msgs.clear()
except Exception:
    P("NG  積層シェル\n" + traceback.format_exc())
# ソリッド＋シェル: 直方体（50×20×2）の端面に平板（シェル）を突き合わせる。 面・辺の選択、解析、結果の表示
try:
    from openmech.geometry import new_feature
    w.new_project()
    app.processEvents()
    w.project.geometry = {"kind": "model", "features": [
        new_feature("box", dx=50.0, dy=20.0, dz=2.0),
        new_feature("plate", plane="XY", offset=1.0, u0=50.0, v0=0.0, du=50.0, dv=20.0, t=2.0)]}
    w.remesh(reset_camera=True)
    app.processEvents()
    mm = w.mesh
    assert getattr(mm, "is_mixed", False), "MixedMesh になっていません"
    inf = mm.face_info()
    root = min((f for f in inf if not f.get("edge") and f["id"] < mm.nfs), key=lambda f: f["centroid"][0])["id"]
    tip = max((f for f in inf if f.get("edge")), key=lambda f: f["centroid"][0])["id"]
    w.set_select_mode("edge")
    w.set_selection([tip])
    app.processEvents()
    w.set_select_mode("body")
    w.set_body_selection([int(mm.body_ids()[-1])])
    app.processEvents()
    w.set_select_mode("face")
    fr = run("ソリッド＋シェル（突き合わせ）", "static", [BC("fixed", [root]), BC("force", [tip], [0, 0, -10.0])],
             results=["全変形 [mm]", "相当応力（ミーゼス） [MPa]"], shot="mixed")
    for i in range(2):
        w.select_node(("bc", i))
        app.processEvents()
    w.select_node(("mesh",))
    app.processEvents()
    tipw = -float(fr[-1].u[:, 2].min())
    exact = 10.0 * 100.0 ** 3 / (3 * w.project.material.E * 20.0 * 2.0 ** 3 / 12)
    P(f"OK  ソリッド＋シェル: 節点 {mm.n_nodes:,}（ソリッド {mm.ns:,}）/ 面 {mm.n_faces} / ボディ {mm.body_ids().tolist()} / "
      f"先端たわみ {tipw:.4g} mm（はりの式 {exact:.4g}、ポアソン比の分だけ小さく出ます）"
      + (f" / メッセージ: {msgs[-1]}" if msgs else ""))
    msgs.clear()
    fr = run("ソリッド＋シェルのモーダル", "modal", [BC("fixed", [root])], {"n_modes": 3},
             results=["全変形 [mm]"], shot="mixed_modal")
    P(f"OK  ソリッド＋シェルのモーダル: {[round(float(r.info['振動数 [Hz]']), 1) for r in fr]} Hz")
except Exception:
    P("NG  ソリッド＋シェル\n" + traceback.format_exc())
# 組立品（節点を共有しない 2 つの直方体）: モーダルと熱で、接触（結合）が使われること
try:
    from openmech.geometry import new_feature
    w.new_project()
    app.processEvents()
    w.project.geometry = {"kind": "model", "features": [
        new_feature("box", dx=100.0, dy=20.0, dz=20.0),
        new_feature("box", x=100.0, dx=100.0, dy=20.0, dz=20.0)]}
    w.remesh(reset_camera=True)
    app.processEvents()
    am = w.mesh
    inf = am.face_info()
    x_lo = min(inf, key=lambda f: f["centroid"][0])["id"]
    x_hi = max(inf, key=lambda f: f["centroid"][0])["id"]
    mt = w.project.material
    mt.conductivity, mt.specific_heat = 60.5, 434.0
    fr = run("組立品のモーダル（接触で結合）", "modal", [BC("fixed", [x_lo])], {"n_modes": 3}, results=["全変形 [mm]"],
             shot="assembly_modal")
    f1 = float(fr[0].info["振動数 [Hz]"])
    fb = 0.5596 * np.sqrt(mt.E * (20.0 ** 4 / 12) / (mt.density * 1e-12 * 400.0 * 200.0 ** 4))
    w.select_node(("contact", 0))
    app.processEvents()
    fr = run("組立品の熱（接触で熱が流れる）", "thermal", [BC("temperature", [x_lo], [100.0]), BC("temperature", [x_hi], [0.0])],
             shot="assembly_thermal")
    T = fr[-1].extra["温度 [°C]"]
    Tm = float(T[np.abs(am.nodes[:, 0] - 150.0) < 1e-6].mean()) if np.any(np.abs(am.nodes[:, 0] - 150.0) < 1e-6) \
        else float(np.interp(150.0, np.sort(am.nodes[:, 0]), T[np.argsort(am.nodes[:, 0])]))
    P(f"OK  組立品: ボディ {am.body_ids().tolist()} / 接触 {len(w.project.contacts)} 領域 / 1 次 {f1:.5g} Hz"
      f"（1 本のはりの式 {fb:.5g}）/ x = 150 の温度 {Tm:.4g} °C（直線なら 25）")
    assert abs(f1 / fb - 1) < 0.05 and abs(Tm - 25.0) < 2.0, "組立品の結果が 1 ボディの値と合いません"
except Exception:
    P("NG  組立品\n" + traceback.format_exc())
# パイプ: 外周（円筒面）の摩擦なし支持＋内圧、回転速度（遠心力）
try:
    w.new_project()
    app.processEvents()
    w.project.geometry = {"kind": "model", "features": [new_feature("pipe", ro=40.0, ri=20.0, h=10.0)]}
    w.remesh(reset_camera=True)
    app.processEvents()
    pm = w.mesh
    inf = pm.face_info()
    curved = sorted((f for f in inf if not f["flat"]), key=lambda f: f["area"])
    inner, outer = curved[0]["id"], curved[-1]["id"]
    bottom = min((f for f in inf if f["flat"]), key=lambda f: f["centroid"][2])["id"]
    w.project.settings.weak_springs = True          # 軸まわりの回転は止めていないので、弱いバネで支える
    mt = w.project.material
    fr = run("円筒面の摩擦なし支持＋内圧", "static", [BC("frictionless", [outer]), BC("frictionless", [bottom]),
                                           BC("pressure", [inner], [100.0])], shot="cyl_frictionless")
    for i in range(3):
        w.select_node(("bc", i))
        app.processEvents()
    Rr = np.hypot(pm.nodes[:, 0], pm.nodes[:, 1])
    ur = (fr[-1].u[:, 0] * pm.nodes[:, 0] + fr[-1].u[:, 1] * pm.nodes[:, 1]) / Rr
    A_ = -100.0 * (1 - mt.nu ** 2) / mt.E / ((1 + mt.nu) + (1 - mt.nu) * 40.0 ** 2 / 20.0 ** 2)
    ua, ub = float(ur[np.abs(Rr - 20.0) < 1e-3].mean()), float(np.abs(ur[np.abs(Rr - 40.0) < 1e-3]).max())
    w.add_bc("rotation")
    app.processEvents()
    w.select_node(("bc", len(w.project.bcs) - 1))
    app.processEvents()
    fr2 = run("回転速度（遠心力）", "static", [BC("frictionless", [bottom]),
                                        BC("rotation", [], [1000.0], point=[0, 0, 0], options={"axis": [0, 0, 1]})],
              shot="rotation")
    ur2 = (fr2[-1].u[:, 0] * pm.nodes[:, 0] + fr2[-1].u[:, 1] * pm.nodes[:, 1]) / Rr
    # 回転する円板（穴あき、平面応力）の外周の半径の増加
    a_, b_, rho, om, nu, E_ = 20.0, 40.0, mt.density * 1e-12, 1000.0, mt.nu, mt.E
    ub_th = rho * om ** 2 * b_ / (4 * E_) * ((1 - nu) * b_ ** 2 + (3 + nu) * a_ ** 2)
    ub2 = float(ur2[np.abs(Rr - 40.0) < 1e-3].mean())
    P(f"OK  円筒面の摩擦なし支持: 内面の半径方向の変位 {ua:.4g} mm（ラメの式 {A_ * (20 - 1600 / 20):.4g}）/ 外周の変位 {ub:.2g} mm"
      f" ｜ 回転速度: 外周の半径の増加 {ub2:.4g} mm（回転円板の式 {ub_th:.4g}）")
    assert abs(ua / (A_ * (20 - 1600 / 20)) - 1) < 0.05 and abs(ub2 / ub_th - 1) < 0.05, "理論値と合いません"
    w.project.settings.weak_springs = False
except Exception:
    P("NG  パイプ（摩擦なし支持・回転速度）\n" + traceback.format_exc())
P("完了")
logf.close()
w.close()
app.quit()

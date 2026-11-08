"""パラメトリック解析（Femtet の「パラメトリック解析」と同じ考え方）

変数をいくつかの値に変えながら、形状の作成 → メッシュ → 解析 を繰り返し、
指定した結果（最大変位・最大応力・質量 …）を表にまとめる。

  rows = run_parametric(project, cases, outputs)
    cases:   [{"L": 80}, {"L": 100}, ...]       （params.sweep_cases で作る）
    outputs: [{"field": "全変形 [mm]", "stat": "max"}, {"field": "質量 [kg]"}, ...]

面の番号: 形状を変えても同じ作り方なら面の番号は変わらない。念のため、境界条件の面の向きが
基準のモデルと変わっていないかを確かめ、変わっていたらそのケースはエラーにする。
"""
from __future__ import annotations

import csv
import time

import numpy as np

STATS = {"max": "最大", "min": "最小", "absmax": "絶対値最大"}
SPECIAL = ["質量 [kg]", "体積 [mm³]", "要素数", "節点数"]


def output_label(o: dict) -> str:
    f = o["field"]
    if f in SPECIAL:
        return f
    name, _, unit = f.partition(" [")
    return f"{STATS.get(o.get('stat', 'max'), '')} {name}" + (f" [{unit}" if unit else "")


def _stat(v, stat):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    if stat == "min":
        return float(v.min())
    if stat == "absmax":
        return float(v[np.argmax(np.abs(v))])
    return float(v.max())


def evaluate_outputs(project, mesh, results, outputs) -> dict:
    """結果（静解析は最終ステップ、過渡解析は全時刻）から出力項目の値を求める"""
    frames = results if isinstance(results, list) else [results]
    out = {}
    for o in outputs:
        f = o["field"]
        lab = output_label(o)
        if f == "要素数":
            out[lab] = float(mesh.n_elems)
        elif f == "節点数":
            out[lab] = float(mesh.n_nodes)
        elif f in ("質量 [kg]", "体積 [mm³]"):
            vol = np.abs(mesh.element_volumes())
            if f == "体積 [mm³]":
                out[lab] = float(vol.sum())
            else:
                mats, em = project.elem_materials(mesh)
                rho = np.array([m.density_t_mm3 for m in mats])[em]      # t/mm³
                out[lab] = float((vol * rho).sum() * 1000.0)
        else:
            use = frames if project.analysis_type == "transient" else frames[-1:]
            vals = []
            for r in use:
                fl = r.fields()
                if f not in fl:
                    raise ValueError(f"結果「{f}」がありません")
                vals.append(_stat(fl[f], o.get("stat", "max")))
            st = o.get("stat", "max")
            out[lab] = (min(vals) if st == "min" else
                        max(vals, key=abs) if st == "absmax" else max(vals))
    return out


def _face_signature(mesh, faces):
    info = {f["id"]: f for f in mesh.face_info()}
    return {int(f): np.asarray(info[int(f)]["normal"], float) for f in faces if int(f) in info}


def check_faces(base_sig: dict, mesh) -> str | None:
    """境界条件の面が基準モデルと同じ向きの面のままかを確かめる（変わっていればメッセージ）"""
    info = {f["id"]: f for f in mesh.face_info()}
    for fid, n0 in base_sig.items():
        if fid not in info:
            return f"面 {fid + 1} がありません（形状の変更で面の数が変わりました）"
        n1 = np.asarray(info[fid]["normal"], float)
        if np.linalg.norm(n0) > 0.5 and np.linalg.norm(n1) > 0.5 and float(n0 @ n1) < 0.95:
            return f"面 {fid + 1} の向きが変わりました（形状の変更で面の番号が入れ替わった可能性）"
    return None


def run_parametric(project, cases, outputs, log=print, progress=None, cancel=None, base_mesh=None):
    """各ケースを解いて行のリストを返す。

    行: {"番号": i, "変数": {全変数の値}, "出力": {ラベル: 値}, "状態": "完了"|"エラー: …", "時間 [s]": t}
    progress(i, n): ケースごとに呼ぶ  cancel(): True を返すと中断
    """
    bc_faces = sorted({int(f) for b in project.bcs for f in b.faces})
    sig = None
    if base_mesh is not None:
        sig = _face_signature(base_mesh, bc_faces)
    rows = []
    n = len(cases)
    for i, ov in enumerate(cases):
        if cancel is not None and cancel():
            log("パラメトリック解析を中断しました")
            break
        if progress is not None:
            progress(i, n)
        t0 = time.time()
        row = {"番号": i + 1, "変数": {}, "出力": {}, "状態": "", "時間 [s]": 0.0}
        desc = ", ".join(f"{k} = {v:g}" for k, v in ov.items())
        log(f"--- ケース {i + 1}/{n}: {desc}")
        try:
            p = project.copy()
            row["変数"] = p.apply_params(ov)
            mesh = p.mesh(log=lambda *a: None)
            if sig is None:
                sig = _face_signature(mesh, bc_faces)
            else:
                msg = check_faces(sig, mesh)
                if msg:
                    raise ValueError(msg)
            res = p.run_all(mesh, log=lambda *a: None)
            row["出力"] = evaluate_outputs(p, mesh, res, outputs)
            row["状態"] = "完了"
            log("    " + ", ".join(f"{k} = {v:.5g}" for k, v in row["出力"].items()))
        except Exception as e:  # noqa: BLE001 — 1 ケースの失敗で全体を止めない
            row["状態"] = f"エラー: {e}"
            log(f"    エラー: {e}")
        row["時間 [s]"] = round(time.time() - t0, 2)
        rows.append(row)
    if progress is not None:
        progress(n, n)
    return rows


def table(rows, var_names=None) -> tuple[list[str], list[list]]:
    """表の見出しと行（変数 → 出力 → 状態）"""
    if not rows:
        return [], []
    var_names = var_names or list(rows[0]["変数"])
    outs = []
    for r in rows:
        for k in r["出力"]:
            if k not in outs:
                outs.append(k)
    head = ["番号"] + list(var_names) + outs + ["状態"]
    body = [[r["番号"]] + [r["変数"].get(v, "") for v in var_names] + [r["出力"].get(k, "") for k in outs]
            + [r["状態"]] for r in rows]
    return head, body


def write_csv(path, rows, var_names=None):
    head, body = table(rows, var_names)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # Excel で文字化けしないよう BOM 付き
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(body)

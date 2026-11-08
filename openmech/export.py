"""結果の書き出し: VTU（ParaView 用）、CSV、HTML レポート。"""
from __future__ import annotations

import base64
import datetime as _dt
import html
import os

import numpy as np

from .solver import Result


def _b64(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr).tobytes()
    header = np.array([len(raw)], dtype=np.uint64).tobytes()
    return base64.b64encode(header + raw).decode()


def write_vtu(path: str, result: Result):
    """VTK XML 非構造格子（バイナリ base64）。ParaView でそのまま開けます。"""
    m = result.mesh
    nn = m.elems.shape[1]
    vtk_type = {10: 24, 4: 10, 6: 13, 15: 26}[nn]
    offsets = np.arange(1, m.n_elems + 1, dtype=np.int64) * nn
    point_data = {"Displacement": result.u.astype(np.float64)}
    for name, arr in result.fields().items():
        key = name.split(" [")[0].replace(" ", "_")
        point_data[key] = np.asarray(arr, dtype=np.float64)
    point_data["Stress"] = result.stress.astype(np.float64)

    def da(name, arr, ncomp=1, typ="Float64"):
        return (f'<DataArray type="{typ}" Name="{html.escape(name)}" NumberOfComponents="{ncomp}" '
                f'format="binary">{_b64(arr)}</DataArray>\n')

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" '
                'byte_order="LittleEndian" header_type="UInt64">\n<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{m.n_nodes}" NumberOfCells="{m.n_elems}">\n')
        f.write("<PointData>\n")
        for k, v in point_data.items():
            f.write(da(k, v, 1 if v.ndim == 1 else v.shape[1]))
        f.write("</PointData>\n<Points>\n")
        f.write(da("Points", m.nodes.astype(np.float64), 3))
        f.write("</Points>\n<Cells>\n")
        f.write(da("connectivity", m.elems.astype(np.int64), 1, "Int64"))
        f.write(da("offsets", offsets, 1, "Int64"))
        f.write(da("types", np.full(m.n_elems, vtk_type, np.uint8), 1, "UInt8"))
        f.write("</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")


def write_csv(path: str, result: Result):
    """節点ごとの座標・変位・応力（Excel で開けるよう UTF-8 BOM 付き）"""
    fields = result.fields()
    cols = ["節点", "X [mm]", "Y [mm]", "Z [mm]"] + list(fields.keys())
    data = np.column_stack([np.arange(1, result.mesh.n_nodes + 1), result.mesh.nodes]
                           + [np.asarray(v) for v in fields.values()])
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(cols) + "\n")
        np.savetxt(f, data, delimiter=",", fmt=["%d"] + ["%.6g"] * (data.shape[1] - 1))


def write_report(path: str, result: Result, bcs, title: str = "構造解析レポート",
                 images: dict | None = None, geometry_desc: str = ""):
    """HTML レポート（ブラウザで開いて印刷→PDF 保存できます）"""
    images = images or {}
    s = result.summary()
    mat = result.material

    def row(k, v):
        if isinstance(v, float):
            v = f"{v:.4g}"
        return f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"

    reac = []
    for i, r in result.reactions.items():
        reac.append(f"<tr><td>{html.escape(bcs[i].label())}</td>"
                    + "".join(f"<td>{x:.4g}</td>" for x in r)
                    + f"<td>{np.linalg.norm(r):.4g}</td></tr>")
    imgs = []
    for cap, p in images.items():
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                b = base64.b64encode(fh.read()).decode()
            imgs.append(f'<figure><img src="data:image/png;base64,{b}"><figcaption>'
                        f"{html.escape(cap)}</figcaption></figure>")
    doc = f"""<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,"Hiragino Sans","Yu Gothic","Meiryo",sans-serif;max-width:900px;margin:2em auto;color:#222}}
h1{{border-bottom:3px solid #1f6feb;padding-bottom:.3em}} h2{{margin-top:1.6em;color:#1f6feb}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
th{{background:#f3f6fa;width:40%}} figure{{margin:1em 0}} img{{max-width:100%;border:1px solid #ddd}}
figcaption{{color:#555;font-size:.9em}} .note{{color:#666;font-size:.85em}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>作成日時: {_dt.datetime.now():%Y-%m-%d %H:%M}　ソフト: open-mechanical</p>
<h2>1. モデル</h2><table>
{row("形状", geometry_desc)}
{row("体積 [mm³]", result.mesh.volume())}
{"".join(row(k, v) for k, v in result.info.items())}
</table>
<h2>2. 材料</h2><table>
{row("名称", mat.name)}{row("ヤング率 [MPa]", mat.E)}{row("ポアソン比", mat.nu)}
{row("密度 [kg/m³]", mat.density)}{row("引張降伏応力 [MPa]", mat.yield_strength or "未設定")}
</table>
<h2>3. 境界条件・荷重</h2><ul>{"".join(f"<li>{html.escape(b.label())}</li>" for b in bcs)}</ul>
<h2>4. 結果</h2><table>{"".join(row(k, v) for k, v in s.items())}</table>
<h3>反力 [N]</h3><table><tr><th>拘束</th><th>Fx</th><th>Fy</th><th>Fz</th><th>合計</th></tr>
{"".join(reac)}</table>
{"".join(imgs)}
<p class="note">※ 線形弾性・微小変形の仮定による結果です。応力集中部（角・穴の縁など）はメッシュを細かくして収束を確認してください。
固定端の角に出る応力の特異点は実際の値ではありません。</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)

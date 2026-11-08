"""LS-DYNA 比較のコンター図と時刻歴グラフを作る（matplotlib のみ。GUI・pyvista 不要）

  python tools/lsdyna_figs.py      入力: logs/lsdyna/summary.json, case*.npz  出力: logs/lsdyna/fig_*.png
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "lsdyna")

for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
          "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# 表示の設定: 変形の倍率（None で実寸）、軸の拡大（細長いモデルの断面方向を広げる）、視点
VIEW = {
    "case1": dict(scale=1.0, stretch=(1, 1, 12), elev=0, azim=0, note="側面図（断面方向を 12 倍に拡大）"),
    "case2": dict(scale=1.0, stretch=(1, 1, 1), elev=20, azim=-35, note="最大圧縮の時刻・実寸"),
    "case3": dict(scale=400.0, stretch=(1, 1, 1), elev=20, azim=-25, note="変形 400 倍"),
    "case4": dict(scale=1.0, stretch=(1, 1, 1), elev=20, azim=-25, note="実寸の変形"),
    "case5": dict(scale=1.0, stretch=(1, 1, 1), elev=20, azim=-25, note="実寸の変形（先端 2 mm は荷重ブロック）"),
}


def _rot(elev, azim):
    a, e = np.radians(azim), np.radians(elev)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    # 視線は -y'（Rz で回してから仰角）。画面の横 = x'、縦 = z'、奥行き = y'
    return Rx @ Rz


def contour(case, k, ax_img):
    d = np.load(os.path.join(OUT, case + ".npz"))
    cfg = VIEW[case]
    X = d["nodes"] + cfg["scale"] * d["u"]
    X = X * np.asarray(cfg["stretch"], float)
    R = _rot(cfg["elev"], cfg["azim"])
    P = X @ R.T
    tris = d["tris"]
    if tris.shape[1] == 6:          # Tri6 → 線形三角形 4 枚（中間節点は 01, 12, 20 の順）
        tris = np.concatenate([tris[:, [0, 3, 5]], tris[:, [3, 1, 4]], tris[:, [5, 4, 2]], tris[:, [3, 4, 5]]])
    v = d["vals"][k]
    used = np.unique(tris)
    vmin, vmax = float(v[used].min()), float(v[used].max())
    if vmax - vmin < 1e-12 * max(1.0, abs(vmax)):
        vmax = vmin + 1e-12 * max(1.0, abs(vmin)) + 1e-30
    # 裏向きの三角形を除き、奥から描く
    # 三角形を外向きにそろえる（変形前の形で判定。検証モデルはどれも直方体なので中心から外向き）、
    # 裏向きを捨てて奥から塗る
    X0 = d["nodes"]
    q = X0[tris]
    n0 = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    lo, hi = X0.min(0), X0.max(0)
    out = (q.mean(1) - 0.5 * (lo + hi)) / np.maximum(hi - lo, 1e-12)
    ax_ = np.argmax(np.abs(out), axis=1)
    flip = np.sign(n0[np.arange(len(tris)), ax_]) != np.sign(out[np.arange(len(tris)), ax_])
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    p2 = P[tris]
    n = np.cross(p2[:, 1] - p2[:, 0], p2[:, 2] - p2[:, 0])
    tf = tris[n[:, 1] < 0]
    depth = P[tf][:, :, 1].mean(1)
    order = np.argsort(-depth)
    tf = tf[order]
    levels = np.linspace(vmin, vmax, 10)
    cmap = plt.get_cmap("jet", 9)
    norm = mcolors.BoundaryNorm(levels, 9)
    fv = v[tf].mean(1)
    pc = PolyCollection(P[tf][:, :, [0, 2]], array=fv, cmap=cmap, norm=norm, edgecolors="face", linewidths=0.3)
    ax_img.add_collection(pc)
    ax_img.autoscale()
    ax_img.set_aspect("equal")
    ax_img.axis("off")
    cb = plt.colorbar(pc, ax=ax_img, ticks=levels, fraction=0.035, pad=0.02, format="%.4g")
    cb.ax.tick_params(labelsize=8)
    cb.set_label(str(d["units"][k]), fontsize=9)
    imax, imin = used[np.argmax(v[used])], used[np.argmin(v[used])]
    for i, c in ((imax, "red"), (imin, "blue")):
        ax_img.plot(P[i, 0], P[i, 2], "o", ms=7, mfc=c, mec="white", mew=1.2, zorder=5)
    ax_img.set_title(f"{d['names'][k]}   t = {float(d['t']) * 1e3:.4g} ms\n"
                     f"最大 {v[imax]:.4g}（赤）／最小 {v[imin]:.4g}（青） {d['units'][k]}　{cfg['note']}", fontsize=9)
    return len(d["names"])


def main():
    summ = json.load(open(os.path.join(OUT, "summary.json")))
    made = []
    for n in sorted(summ, key=int):
        case = f"case{n}"
        if not os.path.exists(os.path.join(OUT, case + ".npz")):
            continue
        nf = len(np.load(os.path.join(OUT, case + ".npz"))["names"])
        wide = n == "1"
        for k in range(nf):
            fig, ax = plt.subplots(figsize=(10, 3.2) if wide else ((5.2, 5.6) if n == "2" else (7.5, 4.6)), dpi=150)
            contour(case, k, ax)
            fig.tight_layout()
            p = os.path.join(OUT, f"fig_{case}_contour{k + 1}.png")
            fig.savefig(p, facecolor="white")
            plt.close(fig)
            made.append(p)
        g = summ[n]["グラフ"]
        it = summ[n]["項目"]
        x, y = np.array(g["x"]), np.array(g["y"])
        fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
        ax.plot(x, y, color="#2a6fdb", lw=1.8, label="open-mechanical")
        LS = dict(color="#d9534f", lw=1.6, ls="--")
        MK = dict(marker="X", ms=10, mfc="#d9534f", mec="white", mew=1.0, ls="none", zorder=6)
        if n == "1":      # LS-DYNA の第1圧縮波（大きさ×継続時間）を矩形で重ねる（立ち上がりは open-mechanical の半値に合わせる）
            mag, dur = it[0]["LS-DYNA"], it[1]["LS-DYNA"]
            i0 = int(np.nonzero(y > 0.5 * y.max())[0][0])
            t0 = x[i0 - 1] + (0.5 * y.max() - y[i0 - 1]) * (x[i0] - x[i0 - 1]) / (y[i0] - y[i0 - 1])
            ax.plot([t0, t0, t0 + dur, t0 + dur], [0, mag, mag, 0], label=f"LS-DYNA（{mag} GPa × {dur} ms）", **LS)
        elif n == "2":    # LS-DYNA の着地点と最下点
            ax.plot([it[0]["LS-DYNA"], it[4]["LS-DYNA"]], [it[1]["LS-DYNA"], it[5]["LS-DYNA"]],
                    label="LS-DYNA（着地・最下点）", **MK)
            ax.plot([it[0]["open-mechanical"], it[4]["open-mechanical"]],
                    [it[1]["open-mechanical"], it[5]["open-mechanical"]], marker="o", ms=6, mfc="#2a6fdb",
                    mec="white", ls="none", zorder=7, label="open-mechanical（着地・最下点）")
        elif n == "3":    # LS-DYNA の振動数で同じ振幅の正弦波
            f = it[0]["LS-DYNA"]
            ax.plot(x, np.abs(y).max() * np.sin(2 * np.pi * f * x), label=f"LS-DYNA の振動数 {f} Hz", **LS)
        else:             # 最終値
            ax.axhline(it[0]["LS-DYNA"], label=f"LS-DYNA 最終値 {it[0]['LS-DYNA']}", **LS)
            ax.plot([x[-1]], [y[-1]], marker="o", ms=6, mfc="#2a6fdb", mec="white", ls="none", zorder=7)
            ax.annotate(f"{y[-1]:.4g}（差 {(y[-1] - it[0]['LS-DYNA']) / it[0]['LS-DYNA'] * 100:+.2f}%）",
                        (x[-1], y[-1]), textcoords="offset points", xytext=(-8, 10), ha="right", fontsize=8)
        ax.set_xlabel(g["xl"])
        ax.set_ylabel(g["yl"])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{summ[n]['id']}  open-mechanical と LS-DYNA の比較", fontsize=10)
        fig.tight_layout()
        p = os.path.join(OUT, f"fig_{case}_history.png")
        fig.savefig(p, facecolor="white")
        plt.close(fig)
        made.append(p)
    print("\n".join(made))


if __name__ == "__main__":
    sys.exit(main())

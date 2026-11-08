"""マウスで基準点・寸法を指定してボディを作る（Femtet のように作業平面上をクリック）

流れ（例: 直方体）
  1. 基準点をクリック（作業平面上。既存ボディの平らな面をクリックすると、その面が作業平面になる）
  2. 対角の点をクリック（長方形のプレビュー）
  3. 高さの位置をクリック（作業平面の法線方向）
  → 寸法入りのダイアログが出るので、確認・微調整して OK
どの段階でも、上のバーに数値を打って Enter で確定できる。Esc / 右クリックで中止。
"""
from __future__ import annotations

import math

import numpy as np

# 作業平面: 名前 → (u 軸, v 軸, 法線) の座標番号（geometry._plane_map と同じ並び）
PLANES = {"XY": (0, 1, 2), "YZ": (1, 2, 0), "ZX": (2, 0, 1)}
AXIS_OF_NORMAL = {"XY": "Z", "YZ": "X", "ZX": "Y"}

# 種類ごとの手順: "pt" = 作業平面上の点、"h" = 法線方向の長さ、"poly" = 多角形（複数の点）
STEPS = {
    "box": [("pt", "基準点（角）"), ("pt", "対角の点"), ("h", "高さ")],
    "cylinder": [("pt", "底面の中心"), ("pt", "半径の位置"), ("h", "長さ")],
    "pipe": [("pt", "底面の中心"), ("pt", "外半径の位置"), ("h", "長さ")],
    "cone": [("pt", "底面の中心"), ("pt", "底面半径の位置"), ("h", "高さ")],
    "cyl_shell": [("pt", "底面の中心"), ("pt", "半径の位置"), ("h", "長さ")],
    "sphere": [("pt", "中心"), ("pt", "半径の位置")],
    "plate": [("pt", "基準点（角）"), ("pt", "対角の点")],
    "disk": [("pt", "中心"), ("pt", "半径の位置")],
    "sheet": [("poly", "頂点（最初の点をクリック / Enter で閉じる）")],
    "extrude": [("poly", "断面の頂点（最初の点をクリック / Enter で閉じる）"), ("h", "押し出し長さ")],
}


def supported(ftype) -> bool:
    return ftype in STEPS


def nice_pitch(span: float) -> float:
    """画面の大きさに合わせたスナップ間隔（1, 2, 5 × 10^n）"""
    if span <= 0:
        return 1.0
    raw = span / 60.0
    e = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * e:
            return m * e
    return 10 * e


def snap(v: float, pitch: float) -> float:
    return round(v / pitch) * pitch if pitch > 0 else v


class Sketch:
    """作図の状態（GUI に依存しない計算部分）"""

    def __init__(self, ftype: str, plane="XY", offset=0.0, pitch=1.0):
        self.ftype = ftype
        self.steps = STEPS[ftype]
        self.plane = plane
        self.offset = float(offset)
        self.pitch = pitch
        self.step = 0
        self.points: list[np.ndarray] = []     # 確定した点（3D）
        self.poly: list[np.ndarray] = []       # 多角形の頂点（3D）
        self.height: float | None = None
        self.cursor: np.ndarray | None = None  # いまのマウス位置（スナップ後, 3D）
        self.cursor_h: float | None = None

    # ---- 平面
    @property
    def axes(self):
        return PLANES[self.plane]

    @property
    def normal(self):
        n = np.zeros(3)
        n[self.axes[2]] = 1.0
        return n

    def set_plane(self, plane, offset):
        if not self.points and not self.poly:
            self.plane, self.offset = plane, float(offset)

    def to3(self, u, v, w=None):
        p = np.zeros(3)
        a = self.axes
        p[a[0]], p[a[1]], p[a[2]] = u, v, self.offset if w is None else w
        return p

    def uv(self, p):
        a = self.axes
        return float(p[a[0]]), float(p[a[1]])

    @property
    def kind(self):
        return self.steps[self.step][0] if self.step < len(self.steps) else None

    @property
    def prompt(self):
        return self.steps[self.step][1] if self.step < len(self.steps) else ""

    # ---- マウスの光線 → 点・長さ
    def ray_point(self, origin, direction):
        """光線と作業平面の交点（スナップ済み）。平面と平行なら None"""
        k = self.axes[2]
        if abs(direction[k]) < 1e-9:
            return None
        t = (self.offset - origin[k]) / direction[k]
        p = origin + t * direction
        u, v = self.uv(p)
        return self.to3(snap(u, self.pitch), snap(v, self.pitch))

    def ray_height(self, origin, direction):
        """光線と、最後の点を通る法線の直線との最接近点 → 法線方向の長さ（スナップ済み）"""
        base = self._height_base()
        n = self.normal
        d = direction / np.linalg.norm(direction)
        w = origin - base
        dn = float(d @ n)
        den = 1.0 - dn * dn
        if den < 1e-9:
            return None
        t = (float(w @ n) - float(w @ d) * dn) / den
        return snap(t, self.pitch)

    def _height_base(self):
        if self.poly:
            return self.poly[-1]
        return self.points[-1] if self.points else self.to3(0, 0)

    # ---- 入力
    def move(self, origin, direction):
        if self.kind in ("pt", "poly"):
            self.cursor = self.ray_point(origin, direction)
        elif self.kind == "h":
            self.cursor_h = self.ray_height(origin, direction)

    def click(self, origin=None, direction=None, value=None) -> bool:
        """クリック（または数値入力）で今の段階を確定。全部そろったら True"""
        kind = self.kind
        if kind == "h":
            h = value if value is not None else (self.ray_height(origin, direction) if origin is not None
                                                 else self.cursor_h)
            if h is None or abs(h) < 1e-12:
                return False
            self.height = float(h)
        else:
            p = value if value is not None else (self.ray_point(origin, direction) if origin is not None
                                                 else self.cursor)
            if p is None:
                return False
            p = np.asarray(p, float)
            if kind == "poly":
                if len(self.poly) >= 3 and np.linalg.norm(p - self.poly[0]) < max(self.pitch, 1e-9) * 0.5:
                    return self.close_poly()
                if self.poly and np.linalg.norm(p - self.poly[-1]) < 1e-12:
                    return False
                self.poly.append(p)
                return False
            if self.points and np.linalg.norm(p - self.points[-1]) < 1e-12:
                return False
            self.points.append(p)
        self.step += 1
        return self.step >= len(self.steps)

    def close_poly(self) -> bool:
        if self.kind != "poly" or len(self.poly) < 3:
            return False
        self.step += 1
        return self.step >= len(self.steps)

    def undo_point(self):
        """1つ前の入力を取り消す（Backspace）"""
        if self.kind == "poly" and self.poly:
            self.poly.pop()
            return
        if self.step == 0:
            return
        self.step -= 1
        k = self.steps[self.step][0]
        if k == "h":
            self.height = None
        elif k == "pt" and self.points:
            self.points.pop()
        elif k == "poly":
            self.poly.pop() if self.poly else None

    # ---- プレビュー（線分の列, 3D）
    def preview(self):
        """(線のリスト[(N,3)], 情報の文字列)"""
        pts = list(self.points)
        h = self.height
        if self.kind in ("pt", "poly") and self.cursor is not None:
            cur = self.cursor
        else:
            cur = None
        if self.kind == "h":
            h = self.cursor_h
        lines, info = [], ""
        t = self.ftype
        n = self.normal
        if t in ("sheet", "extrude"):
            poly = list(self.poly) + ([cur] if cur is not None else [])
            if len(poly) >= 2:
                closed = self.kind != "poly"
                lines.append(np.array(poly + ([poly[0]] if closed else [])))
            if h is not None and len(self.poly) >= 3:
                top = [p + h * n for p in self.poly]
                lines.append(np.array(top + [top[0]]))
                lines += [np.array([a, b]) for a, b in zip(self.poly, top)]
                info = f"長さ {h:g} mm"
            elif cur is not None:
                info = f"頂点 {len(self.poly) + 1}"
            return lines, info
        p1 = pts[0] if pts else cur
        p2 = pts[1] if len(pts) > 1 else (cur if pts else None)
        if p1 is None:
            return lines, info
        u1, v1 = self.uv(p1)
        if t in ("box", "plate"):
            if p2 is not None:
                u2, v2 = self.uv(p2)
                rect = [self.to3(u1, v1, p1[self.axes[2]]), self.to3(u2, v1, p1[self.axes[2]]),
                        self.to3(u2, v2, p1[self.axes[2]]), self.to3(u1, v2, p1[self.axes[2]])]
                lines.append(np.array(rect + [rect[0]]))
                info = f"{abs(u2 - u1):g} × {abs(v2 - v1):g} mm"
                if t == "box" and h is not None:
                    top = [q + h * n for q in rect]
                    lines.append(np.array(top + [top[0]]))
                    lines += [np.array([a, b]) for a, b in zip(rect, top)]
                    info += f" × {abs(h):g} mm"
            return lines, info
        # 円・円柱・球
        if p2 is not None:
            r = float(np.linalg.norm(p2 - p1))
            ang = np.linspace(0, 2 * np.pi, 49)
            a = self.axes
            circ = np.zeros((49, 3))
            circ[:, a[0]] = p1[a[0]] + r * np.cos(ang)
            circ[:, a[1]] = p1[a[1]] + r * np.sin(ang)
            circ[:, a[2]] = p1[a[2]]
            lines.append(circ)
            lines.append(np.array([p1, p2]))
            info = f"半径 {r:g} mm"
            if h is not None and t != "sphere":
                r2 = r * 0.25 if t == "cone" else r
                top = circ.copy()
                top[:, a[0]] = p1[a[0]] + r2 * np.cos(ang)
                top[:, a[1]] = p1[a[1]] + r2 * np.sin(ang)
                top += h * n
                lines.append(top)
                for k in (0, 12, 24, 36):
                    lines.append(np.array([circ[k], top[k]]))
                info += f" / 長さ {abs(h):g} mm"
            if t == "sphere":
                for (i, j) in ((a[0], a[2]), (a[1], a[2])):
                    c2 = np.tile(p1, (49, 1))
                    c2[:, i] += r * np.cos(ang)
                    c2[:, j] += r * np.sin(ang)
                    lines.append(c2)
        return lines, info

    # ---- 結果 → フィーチャーの値
    def feature_values(self) -> dict:
        t = self.ftype
        n_ax = AXIS_OF_NORMAL[self.plane]
        k = self.axes[2]
        if t in ("sheet", "extrude"):
            pts = "; ".join(f"{u:g},{v:g}" for u, v in (self.uv(p) for p in self.poly))
            out = {"plane": self.plane, "offset": float(self.poly[0][k]), "points": pts}
            if t == "extrude":
                h = float(self.height)
                if h < 0:          # 裏向きは平面を下げて正の長さに
                    out["offset"] += h
                    h = -h
                out["h"] = h
            return out
        p1 = self.points[0]
        p2 = self.points[1]
        if t == "box":
            q = [p1, p2, p2 + self.height * self.normal]
            lo, hi = np.min(q, axis=0), np.max(q, axis=0)
            lo = np.minimum(lo, p1)
            return {"x": float(lo[0]), "y": float(lo[1]), "z": float(lo[2]),
                    "dx": float(hi[0] - lo[0]), "dy": float(hi[1] - lo[1]), "dz": float(hi[2] - lo[2])}
        if t == "plate":
            (u1, v1), (u2, v2) = self.uv(p1), self.uv(p2)
            return {"plane": self.plane, "offset": float(p1[k]), "u0": min(u1, u2), "v0": min(v1, v2),
                    "du": abs(u2 - u1), "dv": abs(v2 - v1)}
        r = float(np.linalg.norm(p2 - p1))
        if t == "disk":
            u0, v0 = self.uv(p1)
            return {"plane": self.plane, "offset": float(p1[k]), "u0": u0, "v0": v0, "r": r}
        if t == "sphere":
            return {"x": float(p1[0]), "y": float(p1[1]), "z": float(p1[2]), "r": r}
        h = float(self.height)
        c = p1.copy()
        if h < 0:
            c = c + h * self.normal
            h = -h
        out = {"x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "axis": n_ax, "h": h}
        if t == "pipe":
            out.update(ro=r, ri=round(0.75 * r, 6))
        elif t == "cone":
            if self.height < 0:        # 下向きに伸ばしたときは、上下の半径を入れ替える
                out.update(r1=round(0.25 * r, 6), r2=r)
            else:
                out.update(r1=r, r2=round(0.25 * r, 6))
        else:
            out["r"] = r
        return out

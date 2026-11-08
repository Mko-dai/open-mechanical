"""はり要素（3 次元骨組み。Ansys の BEAM188 を線形・ベルヌーイ＝オイラー（せん断変形あり/なし）で近似）。

節点 6 自由度（ux,uy,uz,θx,θy,θz）、2 節点要素。断面: 長方形・円・パイプ・I 形・角パイプ・任意（A, Iy, Iz, J を直接）。
静解析（節点荷重・分布荷重・重力）とモーダル（整合質量）。結果: 変位、軸力・せん断力・曲げモーメント・ねじり、
断面の最大応力（軸＋曲げの組み合わせ σ = N/A ± My·z/Iy ± Mz·y/Iz の最大）。

モデル（dict）:
  nodes:   [[x,y,z], ...]  [mm]
  members: [{"n": [i, j], "section": 名前, "material": 名前, "orient": [x,y,z]（断面の y 軸の向きの目安）, "div": 分割数}]
  sections:{名前: {"type": "rect", "b":, "h":} | {"type": "circle", "d":} | {"type": "pipe", "d":, "t":}
            | {"type": "I", "h":, "b":, "tw":, "tf":} | {"type": "box", "b":, "h":, "t":} | {"type": "custom", "A","Iy","Iz","J"}}
  supports:[{"node": i, "dof": [1,1,1,1,1,1]}]    1 = 拘束
  loads:   [{"node": i, "F": [Fx,Fy,Fz,Mx,My,Mz]}] / [{"member": k, "q": [qx,qy,qz]}]（全体座標の分布荷重 N/mm）
  gravity: [ax, ay, az]（mm/s²）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .materials import Material


def section_props(s: dict) -> dict:
    t = s.get("type", "rect")
    if t == "rect":
        b, h = float(s["b"]), float(s["h"])
        a, c = max(b, h), min(b, h)
        J = a * c ** 3 * (1 / 3 - 0.21 * c / a * (1 - c ** 4 / (12 * a ** 4)))
        return dict(A=b * h, Iy=b * h ** 3 / 12, Iz=h * b ** 3 / 12, J=J, cy=b / 2, cz=h / 2, ky=5 / 6, kz=5 / 6)
    if t == "circle":
        d = float(s["d"])
        A = np.pi * d ** 2 / 4
        I = np.pi * d ** 4 / 64
        return dict(A=A, Iy=I, Iz=I, J=2 * I, cy=d / 2, cz=d / 2, ky=0.9, kz=0.9)
    if t == "pipe":
        d, th = float(s["d"]), float(s["t"])
        di = d - 2 * th
        A = np.pi * (d ** 2 - di ** 2) / 4
        I = np.pi * (d ** 4 - di ** 4) / 64
        return dict(A=A, Iy=I, Iz=I, J=2 * I, cy=d / 2, cz=d / 2, ky=0.5, kz=0.5)
    if t == "box":
        b, h, th = float(s["b"]), float(s["h"]), float(s["t"])
        bi, hi = b - 2 * th, h - 2 * th
        A = b * h - bi * hi
        Iy = (b * h ** 3 - bi * hi ** 3) / 12
        Iz = (h * b ** 3 - hi * bi ** 3) / 12
        Am = (b - th) * (h - th)
        J = 4 * Am ** 2 * th / (2 * (b - th) + 2 * (h - th))
        return dict(A=A, Iy=Iy, Iz=Iz, J=J, cy=b / 2, cz=h / 2, ky=0.44, kz=0.44)
    if t == "I":
        h, b, tw, tf = float(s["h"]), float(s["b"]), float(s["tw"]), float(s["tf"])
        hw = h - 2 * tf
        A = 2 * b * tf + hw * tw
        Iy = (b * h ** 3 - (b - tw) * hw ** 3) / 12            # 強軸（ウェブ方向 = 断面の z）
        Iz = 2 * tf * b ** 3 / 12 + hw * tw ** 3 / 12
        J = (2 * b * tf ** 3 + hw * tw ** 3) / 3
        return dict(A=A, Iy=Iy, Iz=Iz, J=J, cy=b / 2, cz=h / 2, ky=hw * tw / A, kz=5 / 6 * 2 * b * tf / A)
    if t == "custom":
        d = {k: float(s[k]) for k in ("A", "Iy", "Iz", "J")}
        d.update(cy=float(s.get("cy", 0)), cz=float(s.get("cz", 0)), ky=0.0, kz=0.0)
        return d
    raise ValueError(f"未知の断面: {t}")


def _frame(x1, x2, orient):
    ex = x2 - x1
    L = np.linalg.norm(ex)
    ex = ex / L
    o = np.asarray(orient, float) if orient is not None else None
    if o is None or np.linalg.norm(np.cross(ex, o)) < 1e-6:
        o = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        # 既定: 断面の z 軸を全体の Z（鉛直）に近い向き → y = z × x
        ez = o - ex * (o @ ex)
        ez /= np.linalg.norm(ez)
        ey = np.cross(ez, ex)
    else:
        ey = o - ex * (o @ ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
    return L, np.vstack([ex, ey, ez])


def _k_local(E, G, A, Iy, Iz, J, L, phi_y=0.0, phi_z=0.0):
    """12×12 の局所剛性（せん断変形の係数 φ = 12EI/(κGA L²)、0 でベルヌーイ）"""
    k = np.zeros((12, 12))
    a = E * A / L
    k[0, 0] = k[6, 6] = a
    k[0, 6] = k[6, 0] = -a
    t = G * J / L
    k[3, 3] = k[9, 9] = t
    k[3, 9] = k[9, 3] = -t
    # xy 面の曲げ（v, θz）: Iz
    c = E * Iz / (L ** 3 * (1 + phi_y))
    idx = [1, 5, 7, 11]
    m = c * np.array([[12, 6 * L, -12, 6 * L], [6 * L, (4 + phi_y) * L ** 2, -6 * L, (2 - phi_y) * L ** 2],
                      [-12, -6 * L, 12, -6 * L], [6 * L, (2 - phi_y) * L ** 2, -6 * L, (4 + phi_y) * L ** 2]])
    k[np.ix_(idx, idx)] += m
    # xz 面の曲げ（w, θy）: Iy（符号が逆）
    c = E * Iy / (L ** 3 * (1 + phi_z))
    idx = [2, 4, 8, 10]
    m = c * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, (4 + phi_z) * L ** 2, 6 * L, (2 - phi_z) * L ** 2],
                      [-12, 6 * L, 12, 6 * L], [-6 * L, (2 - phi_z) * L ** 2, 6 * L, (4 + phi_z) * L ** 2]])
    k[np.ix_(idx, idx)] += m
    return k


def _m_local(rho, A, Iy, Iz, L):
    """12×12 の整合質量（並進＋ねじりの回転慣性）"""
    m = np.zeros((12, 12))
    c = rho * A * L
    for i, j in ((0, 6),):
        m[i, i] = m[j, j] = c / 3
        m[i, j] = m[j, i] = c / 6
    Ip = (Iy + Iz) / A
    m[3, 3] = m[9, 9] = c * Ip / 3
    m[3, 9] = m[9, 3] = c * Ip / 6
    for idx, s in (([1, 5, 7, 11], 1), ([2, 4, 8, 10], -1)):
        mm = c / 420 * np.array([[156, s * 22 * L, 54, -s * 13 * L], [s * 22 * L, 4 * L ** 2, s * 13 * L, -3 * L ** 2],
                                 [54, s * 13 * L, 156, -s * 22 * L], [-s * 13 * L, -3 * L ** 2, -s * 22 * L, 4 * L ** 2]])
        m[np.ix_(idx, idx)] += mm
    return m


@dataclass
class BeamResult:
    nodes: np.ndarray
    elems: np.ndarray            # (E,2)
    u: np.ndarray                # (N,6)
    forces: np.ndarray           # (E,2,6) 要素端の断面力（局所: N, Vy, Vz, T, My, Mz）
    stress: np.ndarray           # (E,) 断面の最大の組み合わせ応力 [MPa]
    reactions: dict
    info: dict = field(default_factory=dict)
    freqs: np.ndarray | None = None
    modes: np.ndarray | None = None

    def summary(self):
        d = {"最大変位 [mm]": float(np.linalg.norm(self.u[:, :3], axis=1).max()),
             "最大軸力 [N]": float(np.abs(self.forces[:, :, 0]).max()),
             "最大曲げモーメント [N·mm]": float(np.hypot(self.forces[:, :, 4], self.forces[:, :, 5]).max()),
             "最大組み合わせ応力 [MPa]": float(self.stress.max())}
        return d


class Frame:
    def __init__(self, model: dict, materials: dict[str, Material] | None = None):
        self.model = model
        self.materials = materials or {}
        nodes = [np.asarray(x, float) for x in model["nodes"]]
        elems, props, member_of = [], [], []
        for k, mb in enumerate(model["members"]):
            i, j = mb["n"]
            div = max(1, int(mb.get("div", 1)))
            prev = i
            for d in range(1, div + 1):
                if d == div:
                    nxt = j
                else:
                    nodes.append(nodes[i] + (nodes[j] - nodes[i]) * d / div)
                    nxt = len(nodes) - 1
                elems.append((prev, nxt))
                member_of.append(k)
                prev = nxt
            sec = section_props(model["sections"][mb["section"]])
            mat = self.materials.get(mb.get("material"), None) or Material("steel", 200000, 0.3, 7850)
            props.append((sec, mat, mb.get("orient")))
        self.nodes = np.array(nodes)
        self.elems = np.array(elems, dtype=np.int64)
        self.member_of = np.array(member_of)
        self.props = props
        self.shear = bool(model.get("shear_deformation", True))

    @property
    def n(self):
        return 6 * len(self.nodes)

    def _elem(self, e):
        i, j = self.elems[e]
        sec, mat, orient = self.props[self.member_of[e]]
        L, R = _frame(self.nodes[i], self.nodes[j], orient)
        T = np.zeros((12, 12))
        for b in range(4):
            T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
        return L, T, sec, mat

    def assemble(self, mass=False):
        rows, cols, kv, mv = [], [], [], []
        for e in range(len(self.elems)):
            L, T, s, mat = self._elem(e)
            E, nu = mat.E, mat.nu
            G = E / (2 * (1 + nu))
            phy = 12 * E * s["Iz"] / (s["ky"] * G * s["A"] * L ** 2) if (self.shear and s["ky"] > 0) else 0.0
            phz = 12 * E * s["Iy"] / (s["kz"] * G * s["A"] * L ** 2) if (self.shear and s["kz"] > 0) else 0.0
            ke = T.T @ _k_local(E, G, s["A"], s["Iy"], s["Iz"], s["J"], L, phy, phz) @ T
            dofs = np.r_[6 * self.elems[e, 0] + np.arange(6), 6 * self.elems[e, 1] + np.arange(6)]
            rows.append(np.repeat(dofs, 12))
            cols.append(np.tile(dofs, 12))
            kv.append(ke.ravel())
            if mass:
                mv.append((T.T @ _m_local(mat.density_t_mm3, s["A"], s["Iy"], s["Iz"], L) @ T).ravel())
        r, c = np.concatenate(rows), np.concatenate(cols)
        K = sp.csr_matrix((np.concatenate(kv), (r, c)), shape=(self.n, self.n))
        M = sp.csr_matrix((np.concatenate(mv), (r, c)), shape=(self.n, self.n)) if mass else None
        return K, M

    def load_vector(self):
        F = np.zeros(self.n)
        md = self.model
        g = np.asarray(md.get("gravity", [0, 0, 0]), float)
        for ld in md.get("loads", []):
            if "node" in ld:
                F[6 * int(ld["node"]):6 * int(ld["node"]) + 6] += np.asarray((list(ld["F"]) + [0] * 6)[:6], float)
        dist = {}
        for ld in md.get("loads", []):
            if "member" in ld:
                dist[int(ld["member"])] = dist.get(int(ld["member"]), np.zeros(3)) + np.asarray(ld["q"], float)
        for e in range(len(self.elems)):
            L, T, s, mat = self._elem(e)
            q = dist.get(int(self.member_of[e]), np.zeros(3)) + mat.density_t_mm3 * s["A"] * g
            if not np.any(q):
                continue
            # 全体座標の分布荷重 → 局所 → 等価節点力（両端固定の端力）→ 全体
            R = T[:3, :3]
            ql = R @ q
            fl = np.zeros(12)
            fl[0] = fl[6] = ql[0] * L / 2
            fl[1] = fl[7] = ql[1] * L / 2
            fl[5], fl[11] = ql[1] * L ** 2 / 12, -ql[1] * L ** 2 / 12
            fl[2] = fl[8] = ql[2] * L / 2
            fl[4], fl[10] = -ql[2] * L ** 2 / 12, ql[2] * L ** 2 / 12
            dofs = np.r_[6 * self.elems[e, 0] + np.arange(6), 6 * self.elems[e, 1] + np.arange(6)]
            F[dofs] += T.T @ fl
        return F

    def fixed(self):
        fx = {}
        for s in self.model.get("supports", []):
            for k, f in enumerate((list(s.get("dof", [1] * 6)) + [0] * 6)[:6]):
                if f:
                    fx[6 * int(s["node"]) + k] = float((s.get("value") or [0] * 6)[k]) if s.get("value") else 0.0
        return fx

    def solve(self, log=print) -> BeamResult:
        from .solver import factorize
        t0 = time.time()
        K, _ = self.assemble()
        F = self.load_vector()
        fx = self.fixed()
        cd = np.fromiter(fx.keys(), dtype=np.int64)
        cv = np.fromiter(fx.values(), dtype=float)
        free = np.ones(self.n, bool)
        free[cd] = False
        u = np.zeros(self.n)
        u[cd] = cv
        Kff = K[free][:, free]
        u[free] = factorize(Kff.tocsc(), lambda *a: None)(F[free] - K[free][:, cd] @ cv)
        R = K @ u - F
        res = self._post(u)
        reac = {int(s["node"]): R[6 * int(s["node"]):6 * int(s["node"]) + 6].copy() for s in self.model.get("supports", [])}
        res.reactions = reac
        res.info = {"解析": "はり（静解析）", "節点数": len(self.nodes), "要素数": len(self.elems),
                    "自由度数": int(free.sum()), "合計時間 [s]": round(time.time() - t0, 4)}
        log(f"はり: 節点 {len(self.nodes)} / 要素 {len(self.elems)} / 最大変位 {res.summary()['最大変位 [mm]']:.4g} mm")
        return res

    def _post(self, u):
        forces = np.zeros((len(self.elems), 2, 6))
        stress = np.zeros(len(self.elems))
        dist = {}
        g = np.asarray(self.model.get("gravity", [0, 0, 0]), float)
        for ld in self.model.get("loads", []):
            if "member" in ld:
                dist[int(ld["member"])] = dist.get(int(ld["member"]), np.zeros(3)) + np.asarray(ld["q"], float)
        for e in range(len(self.elems)):
            L, T, s, mat = self._elem(e)
            E, nu = mat.E, mat.nu
            G = E / (2 * (1 + nu))
            phy = 12 * E * s["Iz"] / (s["ky"] * G * s["A"] * L ** 2) if (self.shear and s["ky"] > 0) else 0.0
            phz = 12 * E * s["Iy"] / (s["kz"] * G * s["A"] * L ** 2) if (self.shear and s["kz"] > 0) else 0.0
            dofs = np.r_[6 * self.elems[e, 0] + np.arange(6), 6 * self.elems[e, 1] + np.arange(6)]
            ul = T @ u[dofs]
            f = _k_local(E, G, s["A"], s["Iy"], s["Iz"], s["J"], L, phy, phz) @ ul
            q = dist.get(int(self.member_of[e]), np.zeros(3)) + mat.density_t_mm3 * s["A"] * g
            if np.any(q):
                ql = T[:3, :3] @ q
                fl = np.zeros(12)
                fl[0] = fl[6] = ql[0] * L / 2
                fl[1] = fl[7] = ql[1] * L / 2
                fl[5], fl[11] = ql[1] * L ** 2 / 12, -ql[1] * L ** 2 / 12
                fl[2] = fl[8] = ql[2] * L / 2
                fl[4], fl[10] = -ql[2] * L ** 2 / 12, ql[2] * L ** 2 / 12
                f = f - fl
            # 断面力（内力の符号: 端 i は -f_i、端 j は f_j）
            fi, fj = -f[:6], f[6:]
            forces[e, 0], forces[e, 1] = fi, fj
            for ff in (fi, fj):
                N, My, Mz = ff[0], ff[4], ff[5]
                cy, cz = s.get("cy", 0) or np.sqrt(s["Iz"] / s["A"]), s.get("cz", 0) or np.sqrt(s["Iy"] / s["A"])
                sig = abs(N) / s["A"] + abs(My) * cz / s["Iy"] + abs(Mz) * cy / s["Iz"]
                stress[e] = max(stress[e], sig)
        return BeamResult(self.nodes, self.elems, u.reshape(-1, 6), forces, stress, {})

    def modal(self, n_modes=6, log=print) -> BeamResult:
        import scipy.sparse.linalg as spla
        t0 = time.time()
        K, M = self.assemble(mass=True)
        fx = self.fixed()
        free = np.ones(self.n, bool)
        free[np.fromiter(fx.keys(), dtype=np.int64)] = False
        Kff, Mff = K[free][:, free], M[free][:, free]
        # 回転自由度の質量が 0 の成分があっても扱えるよう、シフト逆反復
        k = min(n_modes + 4, Kff.shape[0] - 2)
        sig = -1e-6 * float(Kff.diagonal().mean() / max(Mff.diagonal().mean(), 1e-300))
        lam, vec = spla.eigsh(Kff.tocsc(), k=k, M=Mff.tocsc(), sigma=sig, which="LM")
        o = np.argsort(lam)[:n_modes]
        lam, vec = lam[o], vec[:, o]
        full = np.zeros((self.n, len(lam)))
        full[free] = vec
        res = self._post(np.zeros(self.n))
        res.freqs = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
        res.modes = full.T.reshape(len(lam), -1, 6)
        res.info = {"解析": "はり（モーダル）", "節点数": len(self.nodes), "要素数": len(self.elems),
                    "合計時間 [s]": round(time.time() - t0, 4)}
        return res

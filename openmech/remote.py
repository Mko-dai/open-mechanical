"""リモートポイント・ばね・弾性支持・円筒支持・ベアリング荷重・点質量（Ansys Mechanical の同名の機能）。

リモートポイント（パイロット節点）は 6 自由度（ux,uy,uz,θx,θy,θz）を全体の自由度の後ろに追加する。
  rigid（剛体）      面の各節点を  u_i = u_p + θ × r_i  で結ぶ（ペナルティ法の MPC）
  deformable（変形可）RBE3 と同じ: パイロットの動き＝面の重み付き平均。荷重は面に分配するだけで面を硬くしない
使い方: extra = extra_terms(mesh, bcs, kref) → 剛性・質量・荷重・拘束を全体の式に足す（solver.solve / modal から呼ぶ）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .surface import surface_sets

REMOTE_KINDS = ("remote_force", "remote_displacement", "point_mass", "elastic_support", "cylindrical_support",
                "bearing_load", "spring", "joint")
# ジョイント（Ansys の Joint と同じ種類）: 局所座標の自由な自由度（0-2 並進 x,y,z、3-5 回転 x,y,z。z が軸）
JOINT_FREE = {"fixed": [], "revolute": [5], "cylindrical": [2, 5], "translational": [0], "slot": [0, 3, 4, 5],
              "universal": [3, 5], "spherical": [3, 4, 5], "planar": [0, 1, 5], "general": None}
JOINT_JP = {"fixed": "固定", "revolute": "回転（レボリュート）", "cylindrical": "円筒", "translational": "並進",
            "slot": "スロット", "universal": "ユニバーサル", "spherical": "球面", "planar": "平面", "general": "一般"}
PENALTY = 1e5           # 剛体結合のペナルティ（剛性の平均の何倍か）


@dataclass
class Extra:
    n_base: int
    n_extra: int = 0
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    vals: list = field(default_factory=list)
    mrows: list = field(default_factory=list)
    mcols: list = field(default_factory=list)
    mvals: list = field(default_factory=list)
    F: dict = field(default_factory=dict)          # dof → 力
    fixed: dict = field(default_factory=dict)      # dof → 値
    pilots: list = field(default_factory=list)     # [(bc番号, 位置, 先頭の自由度)]
    reaction_dofs: dict = field(default_factory=dict)   # bc番号 → 反力を集計する自由度（拘束したパイロット）
    direct: bool = False                           # 直接法を使うべきか（ペナルティで条件数が悪い）
    notes: list = field(default_factory=list)
    mpc: dict = field(default_factory=dict)        # 従属自由度 → (独立自由度の配列, 係数の配列)  u_s = Σ c u_m
    F_by: dict = field(default_factory=dict)       # 境界条件の番号 → {dof: 力}（非線形のステップ倍率用）
    fixed_by: dict = field(default_factory=dict)   # 境界条件の番号 → {dof: 値}
    joints: dict = field(default_factory=dict)     # 境界条件の番号 → ジョイントの情報（結果の集計用）

    def Fvec_of(self, i):
        F = np.zeros(self.n)
        for d, v in self.F_by.get(i, {}).items():
            F[d] += v
        return F

    @property
    def n(self):
        return self.n_base + self.n_extra

    def add_block(self, dofs_r, dofs_c, mat):
        dofs_r = np.asarray(dofs_r)
        dofs_c = np.asarray(dofs_c)
        self.rows.append(np.repeat(dofs_r, len(dofs_c)))
        self.cols.append(np.tile(dofs_c, len(dofs_r)))
        self.vals.append(np.asarray(mat, float).ravel())

    def K(self):
        if not self.rows:
            return sp.csr_matrix((self.n, self.n))
        return sp.csr_matrix((np.concatenate(self.vals), (np.concatenate(self.rows), np.concatenate(self.cols))),
                             shape=(self.n, self.n))

    def M(self):
        if not self.mrows:
            return sp.csr_matrix((self.n, self.n))
        return sp.csr_matrix((np.concatenate(self.mvals), (np.concatenate(self.mrows), np.concatenate(self.mcols))),
                             shape=(self.n, self.n))

    def Fvec(self):
        F = np.zeros(self.n)
        for d, v in self.F.items():
            F[d] += v
        return F

    @property
    def empty(self):
        return not (self.n_extra or self.rows or self.F or self.fixed)


def _skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def face_weights(mesh, faces):
    """面の節点と受け持ち面積（∫N dA）・面の重心・面積"""
    idx, w = [], []
    for ss in surface_sets(mesh, faces):
        i, ww = ss.lumped()
        idx.append(i)
        w.append(ww)
    if not idx:
        raise ValueError("面が選ばれていません")
    i = np.concatenate(idx)
    wv = np.concatenate(w)
    wn = np.bincount(i, weights=wv, minlength=mesh.n_nodes)
    nodes = np.nonzero(wn)[0]
    # Tri6 の頂点は ∫N dA = 0 になる → 剛体結合・重心には面の全節点を使う（重みは小さな正の値）
    allnodes = np.unique(i)
    wts = wn[allnodes]
    wts = np.where(wts > 0, wts, 1e-6 * max(wts.max(), 1e-300))
    c = (mesh.nodes[allnodes] * wts[:, None]).sum(0) / wts.sum()
    return allnodes, wts, c, float(wv.sum()), nodes


def _distribute_load(mesh, faces, F, M, p, F_out):
    """変形可の荷重: 点 p に働く力 F・モーメント M → 面の節点力（RBE3 と同じ配分: 力は面積比、モーメントは r に比例）"""
    nodes, w, c, _, _ = face_weights(mesh, faces)
    Mc = np.asarray(M, float) + np.cross(np.asarray(p, float) - c, np.asarray(F, float))
    f = w[:, None] * np.asarray(F, float)[None, :] / w.sum()
    r = mesh.nodes[nodes] - c
    J = (w[:, None, None] * (np.einsum("ij,ij->i", r, r)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(0)
    if np.any(Mc):
        om = np.linalg.lstsq(J, Mc, rcond=None)[0]
        f += w[:, None] * np.cross(om, r)
    for k in range(3):
        np.add.at(F_out, 3 * nodes + k, f[:, k])


def _cyl_axis(mesh, faces):
    """円筒面の軸（点・方向）。法線は軸に垂直 → 法線の共分散の最小固有ベクトルが軸方向。中心は最小二乗"""
    sets = surface_sets(mesh, faces)
    n = np.concatenate([s.normal.reshape(-1, 3) for s in sets])
    x = np.concatenate([s.xq.reshape(-1, 3) for s in sets])
    w, v = np.linalg.eigh(n.T @ n)
    a = v[:, 0]
    # 軸に垂直な平面に射影した点の円の中心: x - r n （r は半径）を最小二乗で
    # 点 x_i と法線 n_i（外向き）に対し 中心 c + t a = x_i - R n_i → R と c を解く
    P = np.eye(3) - np.outer(a, a)
    xp = x @ P.T
    npj = n @ P.T
    A = np.zeros((3 * len(x), 4))
    for k in range(3):
        A[k::3, k] = 1.0
        A[k::3, 3] = npj[:, k]
    b = xp.ravel()
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = P @ sol[:3]
    R = abs(sol[3])
    return c, a, R


def extra_terms(mesh, bcs, kref: float, acc=None) -> Extra:
    from .solver import BC
    """リモート・ばね・弾性支持・円筒支持・ベアリング荷重・点質量の剛性・質量・荷重・拘束。
    kref: 剛性の大きさの目安（全体剛性の対角の平均）、acc: 重力加速度（点質量の重量）"""
    ex = Extra(3 * mesh.n_nodes)
    kp = PENALTY * kref
    cur = [None]

    def addF(d, v):
        d = int(d)
        ex.F[d] = ex.F.get(d, 0.0) + float(v)
        fb = ex.F_by.setdefault(cur[0], {})
        fb[d] = fb.get(d, 0.0) + float(v)

    def addFb(arr):
        for d in np.nonzero(arr)[0]:
            addF(d, arr[d])

    def new_pilot(i, p):
        d0 = ex.n
        ex.n_extra += 6
        ex.pilots.append((i, np.asarray(p, float), d0))
        return d0

    def couple(i, b, p, d0):
        """パイロット d0..d0+5 を面に結ぶ（多点拘束 MPC を消去法で厳密に扱う）
        剛体:   面の節点が従属  u_i = u_p + θ × r_i
        変形可: パイロットが従属（RBE3）  u_p = Σ ŵ u_i、 θ_p = J⁻¹ Σ ŵ r_i × u_i"""
        nodes, w, c, _, _ = face_weights(mesh, b.faces)
        beh = (b.options or {}).get("behavior", "deformable")
        pd = np.arange(d0, d0 + 6)
        if beh == "rigid":
            for nd in nodes:
                r = mesh.nodes[nd] - p
                # (θ × r)_x = θy rz - θz ry,  _y = θz rx - θx rz,  _z = θx ry - θy rx
                rot = [(np.array([4, 5]), np.array([r[2], -r[1]])), (np.array([5, 3]), np.array([r[0], -r[2]])),
                       (np.array([3, 4]), np.array([r[1], -r[0]]))]
                for k in range(3):
                    ex.mpc[3 * int(nd) + k] = (np.r_[d0 + k, d0 + rot[k][0]], np.r_[1.0, rot[k][1]])
        else:
            r = mesh.nodes[nodes] - p
            wh = w / w.sum()
            J = (wh[:, None, None] * (np.einsum("ij,ij->i", r, r)[:, None, None] * np.eye(3)
                                     - r[:, :, None] * r[:, None, :])).sum(0)
            fdofs = (3 * nodes[:, None] + np.arange(3)).ravel()
            for k in range(3):
                ex.mpc[d0 + k] = (3 * nodes + k, wh.copy())
            # θ = J⁻¹ Σ ŵ (r × u) ;  r × u = [r]× u
            Jinv = np.linalg.pinv(J)
            Sk = np.stack([_skew(rr) * ww for rr, ww in zip(r, wh)])      # (m,3,3)
            A = np.einsum("ij,mjk->imk", Jinv, Sk).reshape(3, -1)         # (3, 3m)
            for k in range(3):
                ex.mpc[d0 + 3 + k] = (fdofs, A[k].copy())

    def joint(i, b):
        """ジョイント: 参照側（A）と可動側（B）の面を剛体でパイロットに結び、局所座標の自由な方向だけ相対運動を許す。
        B の面の節点:  u = u_A + θ_A × r + Σ_並進 e_k α_k + (Σ_回転 e_k β_k) × r   （r = x − p）
        α・β は新しい自由度（ジョイントの相対変位・回転）。地面とのジョイントは u_A = θ_A = 0"""
        o = b.options or {}
        jt = o.get("type", "revolute")
        free = o.get("free") if jt == "general" else JOINT_FREE[jt]
        free = sorted(int(k) for k in (free or []))
        ground = bool(o.get("ground", False))
        faces_b = list(b.faces) if ground else list(o.get("faces_b") or [])
        if not faces_b or (not ground and not b.faces):
            raise ValueError(f"{b.label()}: ジョイントの面（参照側と可動側）を選んでください")
        p = np.asarray(b.point, float) if len(b.point) == 3 else face_weights(mesh, faces_b)[2]
        ez = np.asarray(o.get("axis", [0, 0, 1]), float)
        ez /= np.linalg.norm(ez)
        ex_ = np.asarray(o.get("axis_x", [1, 0, 0]), float)
        ex_ = ex_ - (ex_ @ ez) * ez
        if np.linalg.norm(ex_) < 1e-9:
            ex_ = np.cross(ez, [0, 1, 0]) if abs(ez[1]) < 0.9 else np.cross(ez, [1, 0, 0])
        ex_ /= np.linalg.norm(ex_)
        E = np.vstack([ex_, np.cross(ez, ex_), ez])           # 行 = 局所 x, y, z（全体座標）
        dA = None
        if not ground:
            dA = new_pilot(i, p)
            bb = BC(b.kind, list(b.faces), [], options={"behavior": "rigid"})
            couple(i, bb, p, dA)
        dj = ex.n
        ex.n_extra += len(free)
        nodes, _, _, _, _ = face_weights(mesh, faces_b)
        for nd in nodes:
            if any(3 * int(nd) + k in ex.mpc for k in range(3)):
                raise ValueError(f"{b.label()}: 可動側の面がほかの剛体結合と重なっています")
            r = mesh.nodes[nd] - p
            rot = [(np.array([4, 5]), np.array([r[2], -r[1]])), (np.array([5, 3]), np.array([r[0], -r[2]])),
                   (np.array([3, 4]), np.array([r[1], -r[0]]))]
            for c in range(3):
                m, cf = [], []
                if dA is not None:
                    m += [dA + c, dA + rot[c][0][0], dA + rot[c][0][1]]
                    cf += [1.0, rot[c][1][0], rot[c][1][1]]
                for j, k in enumerate(free):
                    e = E[k % 3]
                    m.append(dj + j)
                    cf.append(e[c] if k < 3 else np.cross(e, r)[c])
                ex.mpc[3 * int(nd) + c] = (np.asarray(m, np.int64), np.asarray(cf, float))
        # 自由な方向の荷重・強制変位・ばね（局所座標の 6 成分）
        load = (list(o.get("load") or []) + [0.0] * 6)[:6]
        disp = (list(o.get("disp") or []) + [None] * 6)[:6]
        kk = (list(o.get("stiffness") or []) + [0.0] * 6)[:6]
        for j, k in enumerate(free):
            if load[k]:
                addF(dj + j, float(load[k]))
            if disp[k] is not None:
                ex.fixed[dj + j] = float(disp[k])
                ex.fixed_by.setdefault(i, {})[dj + j] = float(disp[k])
            if kk[k]:
                ex.add_block([dj + j], [dj + j], [[float(kk[k])]])
        if not free and dA is None:
            ex.notes.append(f"{b.label()}: 地面との固定ジョイント（固定支持と同じ）")
        ex.joints[i] = {"type": jt, "free": free, "E": E, "point": p, "nodes_b": nodes, "dof": dj, "ground": ground}

    for i, b in enumerate(bcs):
        if b.kind not in REMOTE_KINDS:
            continue
        cur[0] = i
        if b.kind == "joint":
            joint(i, b)
            continue
        Fb = np.zeros(3 * mesh.n_nodes)
        v = [None if x is None else float(x) for x in (list(b.values) + [None] * 6)[:6]]
        vz = [0.0 if x is None else x for x in v]
        if b.kind == "elastic_support":
            k = vz[0]
            for ss in surface_sets(mesh, b.faces):
                # K += k ∫ (N n)(N n)ᵀ dA （面の法線方向のばね・面積あたり k [N/mm³]）
                kk = ss.conn.shape[1]
                Ne = ss.N[None, :, :, None] * ss.normal[:, :, None, :]           # (S,q,k,3)
                Ne = Ne.reshape(len(ss.conn), len(ss.N), 3 * kk)
                Ke = np.einsum("sq,sqa,sqb->sab", ss.dA * k, Ne, Ne)
                dofs = (3 * ss.conn[:, :, None] + np.arange(3)).reshape(len(ss.conn), -1)
                for e in range(len(ss.conn)):
                    ex.add_block(dofs[e], dofs[e], Ke[e])
            continue
        if b.kind == "cylindrical_support":
            c, a, R = _cyl_axis(mesh, b.faces)
            nodes, w, _, _, _ = face_weights(mesh, b.faces)
            ex.notes.append(f"円筒支持: 半径 {R:.4g} mm、軸方向 ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})")
            for nd in nodes:
                x = mesh.nodes[nd]
                rv = (x - c) - a * ((x - c) @ a)
                er = rv / max(np.linalg.norm(rv), 1e-300)
                et = np.cross(a, er)
                for val, e in zip(v[:3], (er, a, et)):
                    if val is None:
                        continue
                    ex.add_block(3 * nd + np.arange(3), 3 * nd + np.arange(3), kp * np.outer(e, e))
                    if val:
                        f = kp * val * e
                        for k in range(3):
                            addF(3 * nd + k, f[k])
            ex.direct = True
            continue
        if b.kind == "bearing_load":
            Fv = np.array(vz[:3])
            Fn = np.linalg.norm(Fv)
            if Fn == 0:
                continue
            d = Fv / Fn
            acc_w = np.zeros(mesh.n_nodes)
            for ss in surface_sets(mesh, b.faces):
                # 荷重の向きに押される側: 穴の内面の外向き法線は軸を向くので、押される面は n·d < 0
                proj = np.maximum(0.0, -(ss.normal @ d))
                idx, ww = ss.lumped(proj)
                np.add.at(acc_w, idx, ww)
            if acc_w.sum() <= 0:
                raise ValueError("ベアリング荷重: 荷重の向きに押される面がありません（円筒の内面を選んでください）")
            f = acc_w / acc_w.sum()
            for k in range(3):
                Fb[k::3] += f * Fv[k]
            addFb(Fb)
            continue
        if not b.faces:
            raise ValueError(f"{b.label()}: 面を選んでください")
        p = np.asarray(b.point, float) if len(b.point) == 3 else face_weights(mesh, b.faces)[2]
        beh = (b.options or {}).get("behavior", "deformable")
        if b.kind == "remote_force" and beh != "rigid":
            _distribute_load(mesh, b.faces, vz[:3], vz[3:6], p, Fb)
            addFb(Fb)
            continue
        d0 = new_pilot(i, p)
        couple(i, b, p, d0)
        if b.kind == "remote_force":
            for k in range(6):
                addF(d0 + k, vz[k])
        elif b.kind == "remote_displacement":
            for k in range(6):
                if v[k] is not None:
                    ex.fixed[d0 + k] = v[k]
                    ex.fixed_by.setdefault(i, {})[d0 + k] = v[k]
            ex.reaction_dofs[i] = np.arange(d0, d0 + 6)
        elif b.kind == "point_mass":
            m_t = vz[0] * 1e-3                      # kg → t
            I = np.array(vz[1:4]) * 1e-3            # kg·mm² → t·mm²
            for k in range(3):
                ex.mrows.append([d0 + k]); ex.mcols.append([d0 + k]); ex.mvals.append([m_t])
                ex.mrows.append([d0 + 3 + k]); ex.mcols.append([d0 + 3 + k]); ex.mvals.append([I[k]])
            if acc is not None:
                for k in range(3):
                    addF(d0 + k, m_t * float(acc[k]))
            # 回転自由度に質量がないと特異になるので、ごく小さな回転慣性を足す
            if not np.any(I):
                for k in range(3):
                    ex.mrows.append([d0 + 3 + k]); ex.mcols.append([d0 + 3 + k]); ex.mvals.append([1e-12 * m_t])
        elif b.kind == "spring":
            # ボディ－地面のばね: パイロット p と 地面の点 g を結ぶ、線の方向の剛性 k [N/mm]
            g = np.asarray((b.options or {}).get("ground", list(p + np.array([0, 0, -1.0]))), float)
            e = g - p
            L = np.linalg.norm(e)
            e = e / max(L, 1e-300)
            k = vz[0]
            ex.add_block(d0 + np.arange(3), d0 + np.arange(3), k * np.outer(e, e))
            pre = vz[1]
            if pre:
                f = pre * e                           # 予荷重（引張 正: 地面の方へ引く）
                for kk in range(3):
                    addF(d0 + kk, f[kk])
    return ex


def reduce_system(K, F, fixed, ex: Extra, kref: float):
    """MPC を消去法で取り込む:  u = T ũ（従属自由度を独立自由度で表す）。
    戻り値: (K̃, F̃, 拘束 {縮約後の自由度: 値}, T)。従属自由度に拘束がある場合（変形可のリモート変位）は
    その拘束式 tᵀ ũ = d をペナルティで与える"""
    n = K.shape[0]
    if not ex.mpc:
        return K, F, fixed, None
    slaves = np.array(sorted(ex.mpc), dtype=np.int64)
    is_slave = np.zeros(n, bool)
    is_slave[slaves] = True
    # 従属の従属（剛体結合の面がほかのリモートの面と重なる）は 1 段だけ解決
    keep = np.nonzero(~is_slave)[0]
    newidx = -np.ones(n, dtype=np.int64)
    newidx[keep] = np.arange(len(keep))
    rows, cols, vals = [keep], [np.arange(len(keep))], [np.ones(len(keep))]
    for sd in slaves:
        m, c = ex.mpc[int(sd)]
        ok = ~is_slave[m]
        if not ok.all():
            raise ValueError("リモートポイントの面が重なっています（同じ面を 2 つの剛体結合に使えません）")
        rows.append(np.full(len(m), sd))
        cols.append(newidx[m])
        vals.append(c)
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(keep)))
    Kr = (T.T @ K @ T).tocsr()
    Fr = T.T @ F
    fr = {}
    pen_rows = []
    for d, v in fixed.items():
        if is_slave[d]:
            pen_rows.append((d, v))
        else:
            fr[int(newidx[d])] = v
    if pen_rows:
        kp = PENALTY * kref
        for d, v in pen_rows:
            t = T[d]
            tt = t.T @ t
            Kr = Kr + kp * tt
            Fr = Fr + kp * v * np.asarray(t.todense()).ravel()
    T.keep = keep
    return Kr.tocsr(), Fr, fr, T


def pad(A, n):
    """疎行列・ベクトルを n 自由度に広げる"""
    if sp.issparse(A):
        A = A.tocoo()
        return sp.csr_matrix((A.data, (A.row, A.col)), shape=(n, n))
    out = np.zeros(n)
    out[:len(A)] = A
    return out

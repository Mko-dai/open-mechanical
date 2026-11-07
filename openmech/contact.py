"""接触（Ansys Mechanical の「接続 → 接触」相当）。

接触の種類（Ansys と同じ）:
  bonded         結合         … すべりも離れもしない（線形）
  no_separation  分離なし     … 法線方向は離れない・すべりは自由（線形）
  frictionless   摩擦なし     … 離れることができ、すべりは自由（非線形）
  rough          粗い         … 離れることができ、すべらない（非線形）
  frictional     摩擦あり     … クーロン摩擦（摩擦係数 μ）（非線形）

定式化:
  augmented_lagrange 拡張ラグランジュ法（既定・貫通が小さい）
  penalty            ペナルティ法
  mpc                MPC（結合・分離なし用。非常に大きな剛性の拘束で近似）

ノード-サーフェス方式・微小すべり（接触相手の位置は変形前の形状で決める）。
接触面上の各節点（Contact 側）を相手面（Target 側）に投影してペアを作ります。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp

CONTACT_TYPES = {"bonded": "結合", "no_separation": "分離なし", "frictionless": "摩擦なし",
                 "rough": "粗い", "frictional": "摩擦あり"}
FORMULATIONS = {"auto": "プログラム制御", "augmented_lagrange": "拡張ラグランジュ法",
                "penalty": "ペナルティ法", "mpc": "MPC"}
BEHAVIORS = {"auto": "プログラム制御", "asymmetric": "非対称", "symmetric": "対称"}
INTERFACE = {"adjust": "接触するよう調整", "offset": "オフセットを追加（ランプなし）", "none": "なし（実際の隙間）"}
LINEAR_TYPES = ("bonded", "no_separation")
CORNER_WEIGHT = 0.005

# 接触状態（Ansys の接触ツール「状態」と同じ番号）
ST_FAR, ST_NEAR, ST_SLIDING, ST_STICKING = 0, 1, 2, 3
STATUS_JP = {ST_FAR: "遠い（開放）", ST_NEAR: "近い（開放）", ST_SLIDING: "すべり", ST_STICKING: "固着"}


@dataclass
class ContactRegion:
    name: str = "接触領域"
    type: str = "bonded"
    mu: float = 0.2
    contact_faces: list = field(default_factory=list)   # Contact 側の面番号
    target_faces: list = field(default_factory=list)    # Target 側の面番号
    contact_body: int = -1
    target_body: int = -1
    behavior: str = "auto"
    formulation: str = "auto"
    stiffness_factor: float = 0.0       # 0 = プログラム制御
    pinball: float = 0.0                # 0 = プログラム制御
    interface: str = "adjust"
    offset: float = 0.0
    auto: bool = True                   # 自動検出で作られたか
    suppressed: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        c = ContactRegion()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c

    @property
    def nonlinear(self):
        return self.type not in LINEAR_TYPES

    def resolved_formulation(self):
        if self.formulation != "auto":
            return self.formulation
        return "mpc" if self.type == "bonded" else "augmented_lagrange"

    def resolved_behavior(self):
        if self.behavior != "auto":
            return self.behavior
        return "asymmetric"


# ---------------------------------------------------------------------------- 自動検出

def detect_contacts(mesh, tolerance: float = 0.0) -> list[ContactRegion]:
    """ボディ同士で向かい合っている面を探して接触領域を作る（Ansys の「接続の自動作成」）。

    tolerance: この距離以内の面を接触とみなす [mm]（0 = モデル寸法から自動）
    """
    from scipy.spatial import cKDTree

    if mesh.elem_body is None or len(mesh.body_ids()) < 2:
        return []
    lo, hi = mesh.bounds()
    diag = float(np.linalg.norm(hi - lo))
    tol = tolerance if tolerance > 0 else 0.005 * diag + 1e-9
    tris = mesh.boundary_tris
    cen = mesh.tri_centroids()
    nrm, area = mesh.tri_normals_areas()
    tb = mesh.tri_body()
    fid = mesh.face_ids
    bodies = mesh.body_ids().tolist()
    trees = {b: cKDTree(cen[tb == b]) for b in bodies}
    idx_of = {b: np.nonzero(tb == b)[0] for b in bodies}
    del tris
    out = []
    for ia, a in enumerate(bodies):
        for b in bodies[ia + 1:]:
            ta = idx_of[a]
            d, j = trees[b].query(cen[ta], k=1, distance_upper_bound=tol * 3 + 1e-12)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            ta, jb = ta[ok], idx_of[b][j[ok]]
            facing = np.einsum("ij,ij->i", nrm[ta], nrm[jb]) < -0.5
            # 面中心から相手面までの距離（法線方向）で判定
            gap = np.abs(np.einsum("ij,ij->i", cen[ta] - cen[jb], nrm[jb]))
            good = facing & (gap <= tol)
            if not good.any():
                continue
            fa = sorted(set(fid[ta[good]].tolist()))
            fb = sorted(set(fid[jb[good]].tolist()))
            out.append(ContactRegion(name=f"接触領域（B{a} - B{b}）", contact_faces=fa, target_faces=fb,
                                     contact_body=int(a), target_body=int(b), auto=True))
    return out


# ---------------------------------------------------------------------------- ペアの作成

@dataclass
class Pairs:
    region: int
    slave: np.ndarray       # (M,) Contact 側の節点
    masters: np.ndarray     # (M,3) Target 側の節点（node-to-node は [m, m, m]）
    w: np.ndarray           # (M,3) 内挿の重み
    n: np.ndarray           # (M,3) Target 面の法線（Contact 側を向く）
    g0: np.ndarray          # (M,) 初期隙間（正: 離れている）
    area: np.ndarray        # (M,) 節点の受け持ち面積
    kn: np.ndarray          # (M,) 法線剛性 [N/mm^3]（面積あたり）
    kt: np.ndarray          # (M,) 接線剛性
    # 状態
    lam_n: np.ndarray = None    # 法線の接触圧（拡張ラグランジュ乗数, 圧縮 正）
    lam_t: np.ndarray = None    # (M,3) 接線の摩擦応力
    active: np.ndarray = None
    stick: np.ndarray = None
    # 結果
    pressure: np.ndarray = None
    gap: np.ndarray = None
    slide: np.ndarray = None
    tau: np.ndarray = None
    ctris: np.ndarray = None    # Contact 側の三角形（結果の平滑化用）
    tau_c: np.ndarray = None    # (M,3) 前の収束状態の摩擦応力
    dt_c: np.ndarray = None     # (M,3) 前の収束状態の接線相対変位
    tau_now: np.ndarray = None  # (M,3) 現在の摩擦応力
    Ktan: np.ndarray = None     # (M,3,3) 接線方向の剛性（摩擦: 固着 kt·P、すべり 整合接線）


def _closest_on_tri(p, a, b, c):
    """点 p から三角形 abc 上の最近点の重心座標（ベクトル化）。 p,a,b,c: (M,3)"""
    ab, ac, ap = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    bc = np.stack([u, v, w], axis=1)
    # 三角形の外なら内側へクリップ（簡易）
    bc = np.clip(bc, 0, None)
    bc /= np.maximum(bc.sum(axis=1, keepdims=True), 1e-300)
    return bc


def _region_pairs(mesh, reg: ContactRegion, contact_faces, target_faces, E_of_node, h_char,
                  ridx, diag):
    from scipy.spatial import cKDTree

    if not contact_faces or not target_faces:
        return None
    ctris = np.concatenate([mesh.face_tris(f) for f in contact_faces])
    ttris = np.concatenate([mesh.face_tris(f, full=False) for f in target_faces])
    # Contact 側の節点と受け持ち面積（全節点に均等配分）
    p = mesh.nodes[ctris[:, :3]]
    A = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    k = ctris.shape[1]
    slave_all = ctris.ravel()
    if k == 6:
        # 2次要素の面: 一様な圧力の等価節点力と同じ配分（頂点 ほぼ0、中間節点 A/3）にすると
        # 接触圧が均一に出る。頂点にもごく小さな面積を持たせて食い込みを防ぐ
        wc = CORNER_WEIGHT
        w_all = (A[:, None] * np.array([wc, wc, wc, (1 - 3 * wc) / 3, (1 - 3 * wc) / 3, (1 - 3 * wc) / 3])).ravel()
    else:
        w_all = np.repeat(A / k, k)
    slave, inv = np.unique(slave_all, return_inverse=True)
    area = np.bincount(inv, weights=w_all)
    # Target 側
    tp = mesh.nodes[ttris]
    tn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-300)
    tcen = tp.mean(axis=1)
    tree = cKDTree(tcen)
    xs = mesh.nodes[slave]
    kq = min(8, len(ttris))
    _, cand = tree.query(xs, k=kq)
    cand = cand.reshape(len(xs), -1)
    best_d = np.full(len(xs), np.inf)
    best_bc = np.zeros((len(xs), 3))
    best_t = np.zeros(len(xs), dtype=int)
    for j in range(cand.shape[1]):
        t = cand[:, j]
        bc = _closest_on_tri(xs, tp[t, 0], tp[t, 1], tp[t, 2])
        q = np.einsum("ij,ijk->ik", bc, tp[t])
        d = np.linalg.norm(xs - q, axis=1)
        better = d < best_d
        best_d[better], best_bc[better], best_t[better] = d[better], bc[better], t[better]
    masters = ttris[best_t]
    n = tn[best_t]
    # 2次要素の中間節点と一致する場合は節点どうしで結ぶ（ぴったり合ったメッシュで正確）
    tnodes = np.unique(np.concatenate([mesh.face_tris(f) for f in target_faces]).ravel())
    ttree = cKDTree(mesh.nodes[tnodes])
    dn, jn = ttree.query(xs, k=1)
    coincide = dn < 1e-7 * diag
    if coincide.any():
        m = tnodes[jn[coincide]]
        masters[coincide] = m[:, None]
        best_bc[coincide] = np.array([1.0, 0.0, 0.0])
        # 法線は節点まわりの Target 三角形の平均
    q = np.einsum("ij,ijk->ik", best_bc, mesh.nodes[masters])
    g0 = np.einsum("ij,ij->i", xs - q, n)
    # ピンボール: これより離れた節点は接触の対象外
    pin = reg.pinball if reg.pinball > 0 else (max(4.0 * h_char, 1e-9) if reg.nonlinear else 1e30)
    keep = best_d <= pin
    if reg.type in LINEAR_TYPES and reg.pinball <= 0:
        keep = best_d <= max(4.0 * h_char, 0.02 * diag)
    if not keep.any():
        return None
    slave, masters, best_bc, n, g0, area = slave[keep], masters[keep], best_bc[keep], n[keep], g0[keep], area[keep]
    if reg.interface == "adjust":
        g0 = np.zeros_like(g0)          # 接触するよう調整（初期の隙間・食い込みを無視）
    elif reg.interface == "offset":
        g0 = g0 + reg.offset
    # 剛性: Ansys と同じく 係数 × E / 要素サイズ（面積あたり）
    form = reg.resolved_formulation()
    fkn = reg.stiffness_factor if reg.stiffness_factor > 0 else (10.0 if reg.type in LINEAR_TYPES else 1.0)
    if form == "mpc":
        fkn *= 1e3
    Eloc = E_of_node[slave]
    kn = fkn * Eloc / h_char
    kt = kn.copy() if reg.type in ("bonded", "rough", "frictional") else np.zeros_like(kn)
    if reg.type == "frictional":
        kt = 0.5 * kn
    M = len(slave)
    pr = Pairs(ridx, slave, masters, best_bc, n, g0, area, kn, kt)
    pr.ctris = ctris
    pr.lam_n = np.zeros(M)
    pr.lam_t = np.zeros((M, 3))
    pr.active = np.ones(M, bool) if reg.type in LINEAR_TYPES else (g0 <= 0)
    pr.stick = np.ones(M, bool)
    pr.tau_c = np.zeros((M, 3))
    pr.dt_c = np.zeros((M, 3))
    pr.tau_now = np.zeros((M, 3))
    return pr


def build_pairs(mesh, regions: list[ContactRegion], E_of_node: np.ndarray) -> list[Pairs]:
    lo, hi = mesh.bounds()
    diag = float(np.linalg.norm(hi - lo))
    vol = np.abs(mesh.element_volumes())
    h_char = float(np.cbrt(np.median(vol) * 6 * np.sqrt(2))) if len(vol) else diag / 20
    out = []
    for i, reg in enumerate(regions):
        if reg.suppressed:
            continue
        sides = [(reg.contact_faces, reg.target_faces)]
        if reg.resolved_behavior() == "symmetric":
            sides.append((reg.target_faces, reg.contact_faces))
        for cf, tf in sides:
            pr = _region_pairs(mesh, reg, cf, tf, E_of_node, h_char, i, diag)
            if pr is not None:
                if len(sides) == 2:
                    pr.area = pr.area * 0.5
                out.append(pr)
    return out


# ---------------------------------------------------------------------------- 剛性と力

def _B_rows(pr: Pairs, ndof: int):
    """相対変位 d = u_s - Σ w u_m を作る疎行列（成分ごと）。 戻り値 [Bx, By, Bz] (M, ndof)"""
    M = len(pr.slave)
    rows = np.repeat(np.arange(M), 4)
    out = []
    for c in range(3):
        cols = np.column_stack([3 * pr.slave + c, 3 * pr.masters + c]).ravel()
        vals = np.column_stack([np.ones(M), -pr.w]).ravel()
        out.append(sp.csr_matrix((vals, (rows, cols)), shape=(M, ndof)))
    return out


def contact_system(pairs: list[Pairs], regions: list[ContactRegion], ndof: int, u: np.ndarray):
    """現在の状態（有効/固着）での接触の剛性 Kc と力 fc を返す。

    平衡式:  (K + Kc) u = F + fc
    """
    Kc = sp.csr_matrix((ndof, ndof))
    fc = np.zeros(ndof)
    for pr in pairs:
        reg = regions[pr.region]
        B = _B_rows(pr, ndof)
        n = pr.n
        Bn = sp.diags(n[:, 0]) @ B[0] + sp.diags(n[:, 1]) @ B[1] + sp.diags(n[:, 2]) @ B[2]
        act = pr.active.astype(float)
        a_kn = pr.area * pr.kn * act
        Kc = Kc + Bn.T @ sp.diags(a_kn) @ Bn
        # 拡張ラグランジュの乗数と初期隙間による力:  A (λ - kn g0) n
        fc += Bn.T @ (pr.area * act * (pr.lam_n - pr.kn * pr.g0))
        if reg.type in ("bonded", "rough", "frictional"):
            st = (pr.active & pr.stick).astype(float)
            sl = (pr.active & ~pr.stick).astype(float)
            if reg.type == "frictional" and pr.Ktan is not None:
                Kt3 = pr.Ktan * (pr.area * (st + sl))[:, None, None]
            else:
                P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
                Kt3 = P * (pr.area * pr.kt * st)[:, None, None]
            for c in range(3):
                for d in range(3):
                    coef = Kt3[:, c, d]
                    if np.any(coef):
                        Kc = Kc + B[c].T @ sp.diags(coef) @ B[d]
            for c in range(3):
                fc += B[c].T @ (pr.area * (st + sl) * pr.lam_t[:, c])
    return Kc.tocsr(), fc


def update_states(pairs: list[Pairs], regions: list[ContactRegion], u: np.ndarray, al_update=True,
                  pen_tol: float = 0.0):
    """解いた変位から接触状態を更新。 戻り値: (状態が変わった数, 最大貫通量)"""
    U = u.reshape(-1, 3)
    changed = 0
    max_pen = 0.0
    for pr in pairs:
        reg = regions[pr.region]
        d = U[pr.slave] - np.einsum("ij,ijk->ik", pr.w, U[pr.masters])
        dn = np.einsum("ij,ij->i", d, pr.n)
        g = pr.g0 + dn
        dt = d - dn[:, None] * pr.n
        form = reg.resolved_formulation()
        use_al = form == "augmented_lagrange" and al_update
        # 法線
        trial = pr.lam_n - pr.kn * g                 # 接触圧の試行値（圧縮 正）
        if reg.type in LINEAR_TYPES:
            new_active = np.ones(len(g), bool)
            p = trial
        else:
            new_active = trial > 0
            p = np.maximum(trial, 0.0)
        changed += int(np.sum(new_active != pr.active))
        pr.active = new_active
        if use_al:
            # Ansys と同じく、貫通が許容値を超えた節点だけ乗数を更新する
            over = (np.abs(g) > pen_tol) if reg.type in LINEAR_TYPES else (-g > pen_tol)
            pr.lam_n = np.where(new_active, np.where(over, p, pr.lam_n), 0.0)
        pen = np.where(new_active, np.maximum(-g, 0.0), 0.0)
        if reg.type in LINEAR_TYPES:
            pen = np.abs(g)
        max_pen = max(max_pen, float(pen.max()) if len(pen) else 0.0)
        # 接線
        if reg.type in ("bonded", "rough"):
            tt = pr.lam_t - pr.kt[:, None] * dt
            if use_al:
                over_t = np.linalg.norm(dt, axis=1) > pen_tol
                pr.lam_t = np.where(new_active[:, None], np.where(over_t[:, None], tt, pr.lam_t), 0.0)
            pr.stick = np.ones(len(g), bool)
            tau = np.linalg.norm(np.where(new_active[:, None], tt, 0.0), axis=1)
        elif reg.type == "frictional":
            # 増分形のリターンマッピング（前の収束状態からの弾性試行）
            tt = pr.tau_c - pr.kt[:, None] * (dt - pr.dt_c)
            tn = np.linalg.norm(tt, axis=1)
            limit = reg.mu * p
            stick = tn <= limit * (1 + 1e-9) + 1e-12
            changed += int(np.sum((stick != pr.stick) & new_active))
            pr.stick = stick
            scale = np.where(stick, 1.0, limit / np.maximum(tn, 1e-300))
            tt = tt * scale[:, None]
            # 固着: 定数部分 τc + kt·dtc を力、kt を剛性として与える
            # すべり: 摩擦応力 μp（向きは試行値の向き）を力として与える
            # 整合接線: 固着 kt·P、すべり kt·(μp/|t|)·(P - e e^T)（すべり方向の剛性は 0）
            P = np.eye(3)[None] - pr.n[:, :, None] * pr.n[:, None, :]
            e = tt / np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-300)
            ratio = np.where(stick, 1.0, limit / np.maximum(tn, 1e-300))
            Kt3 = pr.kt[:, None, None] * ratio[:, None, None] * (P - np.where(stick[:, None, None], 0.0,
                                                                            e[:, :, None] * e[:, None, :]))
            pr.Ktan = Kt3
            const = tt + np.einsum("mij,mj->mi", Kt3, dt)
            pr.lam_t = np.where(new_active[:, None], const, 0.0)
            pr.tau_now = np.where(new_active[:, None], tt, 0.0)
            tau = np.linalg.norm(pr.tau_now, axis=1)
        else:
            tau = np.zeros(len(g))
        pr.pressure = np.where(new_active, p, 0.0)
        pr.gap = g
        pr.slide = np.linalg.norm(dt, axis=1)
        pr.tau = tau
    return changed, max_pen


def commit_states(pairs: list[Pairs], regions, u: np.ndarray):
    """時間増分が収束したら、摩擦の履歴（摩擦応力と接線変位）を確定する"""
    U = u.reshape(-1, 3)
    for pr in pairs:
        if regions[pr.region].type != "frictional":
            continue
        d = U[pr.slave] - np.einsum("ij,ijk->ik", pr.w, U[pr.masters])
        dn = np.einsum("ij,ij->i", d, pr.n)
        dt = d - dn[:, None] * pr.n
        pr.tau_c = pr.tau_now.copy()
        pr.dt_c = np.where(pr.active[:, None], dt, dt)


def contact_results(pairs: list[Pairs], regions, n_nodes: int) -> dict:
    """接触ツールの結果（節点値、接触面以外は NaN）"""
    out = {k: np.full(n_nodes, np.nan) for k in
           ("接触状態", "接触圧 [MPa]", "貫通量 [mm]", "すべり距離 [mm]", "摩擦応力 [MPa]", "隙間 [mm]")}
    for pr in pairs:
        if pr.pressure is None:
            continue
        reg = regions[pr.region]
        g = pr.gap
        if reg.type in LINEAR_TYPES:
            status = np.full(len(g), ST_STICKING if reg.type == "bonded" else ST_SLIDING)
        else:
            near = (~pr.active) & (g < 3 * np.maximum(np.abs(pr.g0), 1e-3) + 1e-3)
            status = np.where(pr.active, np.where(pr.stick & (reg.type != "frictionless"), ST_STICKING, ST_SLIDING),
                              np.where(near, ST_NEAR, ST_FAR))
        s = pr.slave
        pres = pr.pressure.copy()
        tau = pr.tau.copy()
        if pr.ctris is not None and pr.ctris.shape[1] == 6:
            # 2次要素: 頂点の値は周りの中間節点の平均で表示（頂点は受け持ち面積がほぼ0のため）
            pos = {int(v): i for i, v in enumerate(s)}
            acc = np.zeros(len(s))
            acc_t = np.zeros(len(s))
            cnt = np.zeros(len(s))
            for tri in pr.ctris:
                mids = [pos[int(v)] for v in tri[3:] if int(v) in pos]
                if not mids:
                    continue
                pm, tm = pr.pressure[mids].mean(), pr.tau[mids].mean()
                for v in tri[:3]:
                    j = pos.get(int(v))
                    if j is not None:
                        acc[j] += pm
                        acc_t[j] += tm
                        cnt[j] += 1
            cm = cnt > 0
            pres[cm] = np.where(pr.active[cm], acc[cm] / cnt[cm], 0.0)
            tau[cm] = acc_t[cm] / cnt[cm]
        out["接触状態"][s] = status
        out["接触圧 [MPa]"][s] = pres
        out["貫通量 [mm]"][s] = np.where(pr.active, np.maximum(-g, 0), 0)
        out["すべり距離 [mm]"][s] = np.where(status == ST_SLIDING, pr.slide, 0.0)
        out["摩擦応力 [MPa]"][s] = tau
        out["隙間 [mm]"][s] = np.where(pr.active, 0.0, np.maximum(g, 0))
    return out

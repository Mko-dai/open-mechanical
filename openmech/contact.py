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
                 "rough": "粗い", "frictional": "摩擦あり", "gasket": "ガスケット"}
FORMULATIONS = {"auto": "プログラム制御", "augmented_lagrange": "拡張ラグランジュ法",
                "penalty": "ペナルティ法", "mpc": "MPC"}
BEHAVIORS = {"auto": "プログラム制御", "asymmetric": "非対称", "symmetric": "対称"}
INTERFACE = {"adjust": "接触するよう調整", "offset": "オフセットを追加（ランプなし）", "none": "なし（実際の隙間）"}
DETECTIONS = {"auto": "プログラム制御", "nodal_target": "節点（ターゲット面の法線方向）",
              "nodal_contact": "節点（接触面の法線方向）"}
STIFFNESS_MODES = {"auto": "プログラム制御", "factor": "係数", "absolute": "絶対値"}
SLIDING = {"auto": "プログラム制御", "small": "微小すべり", "finite": "有限すべり（接触相手を更新）"}
UPDATE_STIFFNESS = {"auto": "プログラム制御", "never": "しない", "each_iteration": "各反復"}
TOL_MODES = {"auto": "プログラム制御", "factor": "係数", "value": "値"}
PINBALL_MODES = {"auto": "プログラム制御", "radius": "半径", "detection": "自動検出値"}
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
    # 高度な設定（Ansys の「詳細 → 高度な設定」と同じ項目）
    detection: str = "auto"             # 検出方法
    stiffness_mode: str = "auto"        # 法線剛性: プログラム制御 / 係数（stiffness_factor）/ 絶対値（stiffness_value）
    stiffness_value: float = 0.0        # 法線剛性の絶対値 [N/mm³]
    update_stiffness: str = "auto"      # 剛性の更新
    pen_tol_mode: str = "auto"          # 貫通許容値: プログラム制御 / 係数（× 要素サイズ）/ 値 [mm]
    pen_tol: float = 0.0
    slip_tol_mode: str = "auto"         # 弾性すべり許容値（摩擦あり）: プログラム制御 / 係数 / 値 [mm]
    slip_tol: float = 0.0
    pinball_mode: str = "auto"          # ピンボール領域: プログラム制御 / 半径（pinball）/ 自動検出値
    # すべり: small = 接触相手は変形前の位置で決めて固定 / finite = サブステップごとに変形後の位置で決め直す
    # auto = 大変形オンのとき finite（Ansys の「微小すべり: プログラム制御」と同じ）
    sliding: str = "auto"
    stabilization: float = 0.0          # 安定化減衰係数（0 = なし）: 開いている接触にも弱いばねを入れて浮きを防ぐ
    # ボルトのねじ（Ansys の「幾何補正: ボルトねじ」）: 円筒の接触面を、ねじ山の斜面として扱う
    # {"point": 軸上の点, "axis": 軸の向き, "half_angle": ねじ山の半角 [度]（メートルねじ 30）, "pitch": ピッチ [mm]}
    thread: dict = field(default_factory=dict)
    # ガスケット（Ansys のガスケット要素と同じ考え方）: 圧縮量（閉じ量）と面圧の表
    # {"closure": [mm], "pressure": [MPa], "unload": 除荷の傾き [MPa/mm]（0 = 負荷曲線をたどる）,
    #  "shear": 面内のせん断剛性 [MPa/mm]（0 = 面内は拘束しない）}
    gasket: dict = field(default_factory=dict)
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
        # 古いファイル: 法線剛性係数・ピンボール半径が入っていれば、そのモードにする
        if "stiffness_mode" not in d and c.stiffness_factor > 0:
            c.stiffness_mode = "factor"
        if "pinball_mode" not in d and c.pinball > 0:
            c.pinball_mode = "radius"
        return c


    @property
    def nonlinear(self):
        return self.type not in LINEAR_TYPES

    def finite_sliding(self, large_deflection: bool) -> bool:
        if self.type in LINEAR_TYPES or self.type == "gasket" or (self.thread and self.thread.get("axis")):
            return False
        return self.sliding == "finite" or (self.sliding == "auto" and bool(large_deflection))

    def resolved_formulation(self):
        if self.type == "gasket":
            return "penalty"            # ガスケットは面圧-閉じ量の表そのもの（拡張ラグランジュは使わない）
        if self.formulation != "auto":
            return self.formulation
        return "mpc" if self.type == "bonded" else "augmented_lagrange"

    def resolved_behavior(self):
        if self.behavior != "auto":
            return self.behavior
        return "asymmetric"


# ---------------------------------------------------------------------------- 自動検出

def detect_contacts(mesh, tolerance: float = 0.0, default_type: str = "bonded",
                    default_mu: float = 0.2) -> list[ContactRegion]:
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
                                     contact_body=int(a), target_body=int(b), auto=True,
                                     type=default_type if default_type in CONTACT_TYPES else "bonded",
                                     mu=float(default_mu)))
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
    pen_tol: float = 0.0        # 貫通許容値 [mm]
    slip_tol: float = 0.0       # 弾性すべり許容値 [mm]（0 = 接線剛性は固定）
    kn0: np.ndarray = None      # 最初の法線剛性（剛性の更新の上限・下限に使う）
    stab: float = 0.0           # 安定化（開いている節点にも入れる弱いばねの係数）
    max_pen: float = 0.0        # 最後の更新での最大貫通量
    Ktan: np.ndarray = None     # (M,3,3) 接線方向の剛性（摩擦: 固着 kt·P、すべり 整合接線）
    c_max: np.ndarray = None    # ガスケット: 確定した最大閉じ量
    # 有限すべり: 接触相手を決め直すための情報
    finite: bool = False
    X: np.ndarray = None        # 節点座標（変形前）
    ttris: np.ndarray = None    # Target 側の三角形
    ttris6: np.ndarray = None   # Target 側の 6 節点三角形（2 次要素の面。ttris と同じ並び）
    adj: np.ndarray = None      # 「接触するよう調整」「オフセット」で動かした隙間の量
    h: float = 1.0
    pin: float = 1e30
    nonsym: bool = False        # 接線が非対称（摩擦ですべっている節点がある）
    ratio: np.ndarray = None    # 摩擦: すべり中の 摩擦限界 / 試行値（整合接線の係数）
    n_fixed: bool = False       # 法線を接触面側の法線で固定（検出方法: 接触面の法線方向）


def gasket_curve(gk: dict):
    """ガスケットの表（閉じ量, 面圧）を、原点から始まる単調な折れ線にして返す"""
    c = np.asarray(gk.get("closure", []), float)
    p = np.asarray(gk.get("pressure", []), float)
    if len(c) == 0 or len(c) != len(p):
        raise ValueError("ガスケットの閉じ量と面圧の表を入力してください（同じ個数）")
    o = np.argsort(c)
    c, p = c[o], p[o]
    if c[0] > 0:
        c, p = np.r_[0.0, c], np.r_[0.0, p]
    if np.any(np.diff(c) <= 0):
        raise ValueError("ガスケットの閉じ量が重複しています")
    return c, p


def gasket_pressure(gk: dict, c, c_max=None):
    """閉じ量 c [mm] での面圧 p と接線剛性 dp/dc（面積あたり）

    表の外は最後の区間の傾きで延長。 除荷（c < これまでの最大閉じ量）は傾き unload の直線で、
    0 を下回らない（永久変形）。 c ≤ 0 は開いている（p = 0）。
    """
    cc, pp = gasket_curve(gk)
    c = np.asarray(c, float)
    slopes = np.diff(pp) / np.diff(cc)
    k = slopes[np.clip(np.searchsorted(cc, c, side="right") - 1, 0, len(slopes) - 1)]
    p = np.interp(c, cc, pp)
    hi = c > cc[-1]
    p = np.where(hi, pp[-1] + slopes[-1] * (c - cc[-1]), p)
    ku = float(gk.get("unload", 0.0) or 0.0)
    if ku > 0 and c_max is not None:
        cm = np.maximum(np.asarray(c_max, float), 0.0)
        pm = np.where(cm > cc[-1], pp[-1] + slopes[-1] * (cm - cc[-1]), np.interp(cm, cc, pp))
        unl = c < cm - 1e-15
        pu = pm - ku * (cm - c)
        p = np.where(unl, pu, p)
        k = np.where(unl, ku, k)
    open_ = p <= 0
    p = np.where(open_, 0.0, p)
    k = np.where(open_, 0.0, np.maximum(k, 0.0))
    return p, k


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


def _project(xs, tp, tree, kq):
    """点 xs を三角形群 tp (T,3,3) に射影: (三角形番号, 重心座標, 距離)"""
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
    return best_t, best_bc, best_d


def _tri6(xi, et):
    """6 節点三角形（a,b,c, ab,bc,ca）の形状関数と ξ・η での微分。 L = (1−ξ−η, ξ, η)"""
    L0 = 1.0 - xi - et
    N = np.stack([L0 * (2 * L0 - 1), xi * (2 * xi - 1), et * (2 * et - 1), 4 * L0 * xi, 4 * xi * et, 4 * et * L0], 1)
    z = np.zeros_like(xi)
    dNx = np.stack([1 - 4 * L0, 4 * xi - 1, z, 4 * (L0 - xi), 4 * et, -4 * et], 1)
    dNe = np.stack([1 - 4 * L0, z, 4 * et - 1, -4 * xi, 4 * xi, 4 * (L0 - et)], 1)
    return N, dNx, dNe


def repair_pairs(pr: Pairs, u: np.ndarray):
    """有限すべり: 変形後の位置で接触相手（Target の三角形・内挿の重み・法線）を決め直す。

    隙間は g = g0 + (u_s − Σ w u_m)·n のまま使えるよう、g0 = (X_s − Σ w X_m)·n（＋最初の調整量）に置き直す。
    Target の外にはみ出した節点（射影が三角形の縁に張り付いた節点）は接触の対象外にする。
    """
    from scipy.spatial import cKDTree
    x = pr.X + u.reshape(-1, 3)[:len(pr.X)]
    tp = x[pr.ttris]
    tn = np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0])
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-300)
    tree = cKDTree(tp.mean(axis=1))
    xs = x[pr.slave]
    bt, bc, _d = _project(xs, tp, tree, min(8, len(pr.ttris)))
    masters = pr.ttris[bt]
    n = pr.n if pr.n_fixed else tn[bt]
    if pr.ttris6 is not None:
        # 2 次要素の面: 中間節点まで使った曲面（6 節点三角形）に射影する。平らな三角形で近似すると、
        # 曲面をすべるときに「面の粗さ」（弦と弧の差）がそのまま隙間・食い込みの誤差になる
        masters = pr.ttris6[bt]
        xm = x[masters]                                    # (M,6,3)
        xi, et = bc[:, 1].copy(), bc[:, 2].copy()
        for _ in range(6):
            N, dNx, dNe = _tri6(xi, et)
            r = np.einsum("ma,mak->mk", N, xm) - xs
            tx = np.einsum("ma,mak->mk", dNx, xm)
            te = np.einsum("ma,mak->mk", dNe, xm)
            a11, a12, a22 = (tx * tx).sum(1), (tx * te).sum(1), (te * te).sum(1)
            b1, b2 = (r * tx).sum(1), (r * te).sum(1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-300)
            xi = xi - (a22 * b1 - a12 * b2) / det
            et = et - (a11 * b2 - a12 * b1) / det
            # 三角形の中に収める
            xi, et = np.maximum(xi, 0.0), np.maximum(et, 0.0)
            over = np.maximum(xi + et, 1.0)
            xi, et = xi / over, et / over
        N, dNx, dNe = _tri6(xi, et)
        bc = N
        if not pr.n_fixed:
            n = np.cross(np.einsum("ma,mak->mk", dNx, xm), np.einsum("ma,mak->mk", dNe, xm))
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    q = np.einsum("ij,ijk->ik", bc, x[masters])
    off = xs - q
    gn = np.einsum("ij,ij->i", off, n)
    lateral = np.linalg.norm(off - gn[:, None] * n, axis=1)
    # Target の外、または裏側に深く回り込んだ節点（薄い部品の反対側など）は対象外
    outside = (lateral > 0.25 * pr.h) | (gn < -pr.pin)
    Xq = np.einsum("ij,ijk->ik", bc, pr.X[masters])
    g0 = np.einsum("ij,ij->i", pr.X[pr.slave] - Xq, n) + pr.adj
    pr.masters, pr.w, pr.n = masters, bc, n
    pr.g0 = np.where(outside, 1e30, g0)


def repair_in_step(pr: Pairs, reg: ContactRegion, u: np.ndarray) -> int:
    """サブステップの途中で接触相手を決め直す（収束した位置 u で）。 戻り値: Target の内・外が変わった節点の数。

    摩擦のすべりは「前の収束状態からの増分」で測っているので、相手を替えてもその増分（dt − dt_c）が
    変わらないよう、基準 dt_c を新しい相手での値に置き直す。
    """
    U = u.reshape(-1, 3)

    def tangential():
        d = U[pr.slave] - np.einsum("ij,ijk->ik", pr.w, U[pr.masters])
        return d - np.einsum("ij,ij->i", d, pr.n)[:, None] * pr.n
    fric = reg.type == "frictional"
    inc = tangential() - pr.dt_c if fric else None
    was_in = pr.g0 < 1e29
    repair_pairs(pr, u)
    if fric:
        inc = inc - np.einsum("ij,ij->i", inc, pr.n)[:, None] * pr.n
        pr.dt_c = tangential() - inc
    now_in = pr.g0 < 1e29
    return int(np.sum(was_in != now_in))


def _region_pairs(mesh, reg: ContactRegion, contact_faces, target_faces, E_of_node, h_char,
                  ridx, diag, det_tol=0.0, finite=False):
    from scipy.spatial import cKDTree

    if not contact_faces or not target_faces:
        return None
    ctris = np.concatenate([mesh.face_tris(f) for f in contact_faces])
    ttris = np.concatenate([mesh.face_tris(f, full=False) for f in target_faces])
    # Contact 側の節点と受け持ち面積（全節点に均等配分）
    p = mesh.nodes[ctris[:, :3]]
    cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    A = 0.5 * np.linalg.norm(cr, axis=1)
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
    # 接触面の節点法線（面積重みの平均、ボディの外向き）
    cn_node = np.zeros((len(slave), 3))
    for c in range(3):
        cn_node[:, c] = np.bincount(inv, weights=np.repeat(cr[:, c], k), minlength=len(slave))
    cn_node /= np.maximum(np.linalg.norm(cn_node, axis=1, keepdims=True), 1e-300)
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
    if reg.detection == "nodal_contact":
        # 検出方法「節点（接触面の法線方向）」: 法線は接触面の外向き法線の逆（ターゲットから接触側へ）
        n = -cn_node
    # 2次要素の中間節点と一致する場合は節点どうしで結ぶ（ぴったり合ったメッシュで正確）
    tnodes = np.unique(np.concatenate([mesh.face_tris(f) for f in target_faces]).ravel())
    ttree = cKDTree(mesh.nodes[tnodes])
    dn, jn = ttree.query(xs, k=1)
    coincide = dn < 1e-7 * diag
    if coincide.any():
        m = tnodes[jn[coincide]]
        masters[coincide] = m[:, None]
        best_bc[coincide] = np.array([1.0, 0.0, 0.0])
    q = np.einsum("ij,ijk->ik", best_bc, mesh.nodes[masters])
    g0 = np.einsum("ij,ij->i", xs - q, n)
    # ピンボール: これより離れた節点は接触の対象外
    pin = pinball_radius(reg, h_char, diag, det_tol)
    keep = best_d <= pin
    # Target の外にはみ出した節点（最近点が三角形の縁に張り付く）は相手がいないので対象外。
    # 接触面がターゲットより広いとき、外側の節点までターゲットの縁を支えてしまうのを防ぐ
    lat = np.linalg.norm((xs - q) - g0[:, None] * n, axis=1)
    keep &= lat <= 0.25 * h_char
    if not keep.any() and not finite:
        return None
    g_geom = g0.copy()
    if finite:
        # 有限すべり: 今は離れている節点も、あとで近づけば接触する。すべての節点を残し、調整は近い節点だけ
        if reg.interface == "adjust":
            g0 = np.where(keep, 0.0, g0)
        elif reg.interface == "offset":
            g0 = g0 + reg.offset
        far0 = ~keep & ((lat > 0.25 * h_char) | (g_geom < 0))      # Target の外、または裏側
        adj = g0 - g_geom
        g0 = np.where(far0, 1e30, g0)
    else:
        slave, masters, best_bc, n, g0, area = (slave[keep], masters[keep], best_bc[keep], n[keep], g0[keep],
                                                area[keep])
        if reg.interface == "adjust":
            g0 = np.zeros_like(g0)          # 接触するよう調整（初期の隙間・食い込みを無視）
        elif reg.interface == "offset":
            g0 = g0 + reg.offset
    if reg.thread and reg.thread.get("axis"):
        slave, masters, best_bc, n, g0, area = _thread_flanks(mesh, reg.thread, slave, masters, best_bc, n, g0, area)
    # 法線剛性: Ansys と同じく 係数 × E / 要素サイズ（面積あたり）、または絶対値
    form = reg.resolved_formulation()
    Eloc = E_of_node[slave]
    if reg.stiffness_mode == "absolute" and reg.stiffness_value > 0:
        kn = np.full(len(slave), float(reg.stiffness_value))
    else:
        use_f = reg.stiffness_mode == "factor" or (reg.stiffness_mode == "auto" and reg.stiffness_factor > 0)
        fkn = reg.stiffness_factor if (use_f and reg.stiffness_factor > 0) else \
            (10.0 if reg.type in LINEAR_TYPES else 1.0)
        if form == "mpc":
            fkn *= 1e3
        kn = fkn * Eloc / h_char
    kt = kn.copy() if reg.type in ("bonded", "rough", "frictional") else np.zeros_like(kn)
    if reg.type == "frictional":
        kt = 0.5 * kn
    if reg.type == "gasket":
        _p, k0 = gasket_pressure(reg.gasket, np.array([1e-12]))
        kn = np.full(len(slave), max(float(k0[0]), 1e-12))
        kt = np.full(len(slave), max(float(reg.gasket.get("shear", 0.0) or 0.0), 0.0))
    M = len(slave)
    pr = Pairs(ridx, slave, masters, best_bc, n, g0, area, kn, kt)
    pr.ctris = None if (reg.thread and reg.thread.get("axis")) else ctris
    if finite:
        pr.finite, pr.X, pr.ttris, pr.adj, pr.h = True, mesh.nodes, ttris, adj, float(h_char)
        pr.pin = float(pin)
        t6 = np.concatenate([mesh.face_tris(f) for f in target_faces])
        if t6.shape[1] == 6 and len(t6) == len(ttris) and np.array_equal(t6[:, :3], ttris):
            pr.ttris6 = t6
        pr.n_fixed = reg.detection == "nodal_contact"
    pr.lam_n = np.zeros(M)
    pr.lam_t = np.zeros((M, 3))
    pr.active = np.ones(M, bool) if reg.type in LINEAR_TYPES else (g0 <= 0)
    pr.stick = np.ones(M, bool)
    pr.tau_c = np.zeros((M, 3))
    pr.dt_c = np.zeros((M, 3))
    pr.tau_now = np.zeros((M, 3))
    pr.kn0 = kn.copy()
    if reg.type == "gasket":
        pr.c_max = np.zeros(M)          # これまでの最大閉じ量（除荷の履歴）
        pr.active = g0 <= 0             # 最初は表の初期勾配で閉じている（面圧 0 から）
        p0, _k = gasket_pressure(reg.gasket, np.maximum(-g0, 0.0))
        pr.lam_n = np.where(pr.active, p0 + kn * g0, 0.0)       # λ - kn g0 = p(初期閉じ量)
    # 貫通許容値（0 = プログラム制御: 解析設定の値 × 要素サイズ）
    if reg.pen_tol_mode == "value" and reg.pen_tol > 0:
        pr.pen_tol = float(reg.pen_tol)
    elif reg.pen_tol_mode == "factor" and reg.pen_tol > 0:
        pr.pen_tol = float(reg.pen_tol) * h_char
    # 弾性すべり許容値（摩擦あり）: 設定すると接線剛性を μ·p / 許容値 で毎回更新（Ansys と同じ）
    if reg.type == "frictional" and reg.slip_tol_mode != "auto" and reg.slip_tol > 0:
        pr.slip_tol = reg.slip_tol if reg.slip_tol_mode == "value" else reg.slip_tol * h_char
    pr.stab = max(float(reg.stabilization), 0.0) if reg.nonlinear else 0.0
    return pr


def _thread_flanks(mesh, th, slave, masters, bc, n, g0, area):
    """ねじの幾何補正: 円筒の接触面の各節点に、ねじ山の 2 つの斜面（フランク）の接触を作る。

    フランクの法線 = ±cos α · 軸方向 + sin α · 半径方向（α = ねじ山の半角。メートルねじは 30°）。
    軸方向に引くと片方のフランクが閉じて軸力を伝え、その tan α 倍の半径方向の力（くさび作用）が生じる。
    もう片方は開く（非線形の接触で自動的に判定）。面積はフランクの実面積（投影 ÷ sin α の半分ずつ）。
    ピッチを与えると、らせんのリード角でフランクの向きを周方向に傾ける（軸力から ゆるめトルク F·r·tan λ が出る）。
    ※ 山 1 つ 1 つの位置（位相）までは表さない（ねじ山 1 ピッチより粗い要素向けの平均化モデル）
    """
    a = np.radians(float(th.get("half_angle", 30.0)))
    e = np.asarray(th["axis"], float)
    e = e / np.linalg.norm(e)
    p0 = np.asarray(th.get("point", [0, 0, 0]), float)
    x = mesh.nodes[slave] - p0
    rv = x - np.outer(x @ e, e)
    r = rv / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-300)
    sgn = np.sign(np.einsum("ij,ij->i", n, r))
    sgn[sgn == 0] = 1.0
    # らせんのリード角 λ = atan(リード / (π d_m))（d_m = 接触面の平均直径）。ねじ山の面の軸方向の向きは
    # らせんに沿って傾く: cos λ e − (右ねじ +1 / 左ねじ −1) sin λ θ̂（θ̂ = e × r̂、周方向）
    pitch = float(th.get("pitch", 0.0) or 0.0)
    lead = pitch * max(1, int(th.get("starts", 1) or 1))
    dm = 2.0 * float(np.mean(np.linalg.norm(rv, axis=1)))
    lam = np.arctan(lead / (np.pi * dm)) if (lead > 0 and dm > 0) else 0.0
    hand = -1.0 if th.get("hand", "right") == "left" else 1.0
    th_dir = np.cross(e, r)
    ax_dir = np.cos(lam) * e[None, :] - hand * np.sin(lam) * th_dir
    out = []
    for s_ax in (1.0, -1.0):
        nf = sgn[:, None] * np.sin(a) * r + s_ax * np.cos(a) * ax_dir
        out.append(nf / np.linalg.norm(nf, axis=1, keepdims=True))
    f = 0.5 / max(np.sin(a), 1e-6)
    cat = lambda v: np.concatenate([v, v])  # noqa: E731
    return cat(slave), cat(masters), cat(bc), np.vstack(out), cat(g0), cat(area) * f


def pinball_radius(reg: ContactRegion, h_char: float, diag: float, det_tol: float = 0.0) -> float:
    """ピンボール領域の半径 [mm]"""
    if reg.pinball_mode == "radius" and reg.pinball > 0:
        return float(reg.pinball)
    if reg.pinball_mode == "auto" and reg.pinball > 0:      # 古いファイル
        return float(reg.pinball)
    if reg.pinball_mode == "detection":
        return float(det_tol if det_tol > 0 else 0.005 * diag + 1e-9)
    if reg.type in LINEAR_TYPES or reg.type == "gasket":
        return max(4.0 * h_char, 0.02 * diag)
    return max(4.0 * h_char, 1e-9)


def build_pairs(mesh, regions: list[ContactRegion], E_of_node: np.ndarray, det_tol: float = 0.0,
                large_deflection: bool = False) -> list[Pairs]:
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
            pr = _region_pairs(mesh, reg, cf, tf, E_of_node, h_char, i, diag, det_tol,
                               finite=reg.finite_sliding(large_deflection))
            if pr is not None:
                if len(sides) == 2:
                    pr.area = pr.area * 0.5
                out.append(pr)
    return out


# ---------------------------------------------------------------------------- 剛性と力

def _B_rows(pr: Pairs, ndof: int):
    """相対変位 d = u_s - Σ w u_m を作る疎行列（成分ごと）。 戻り値 [Bx, By, Bz] (M, ndof)"""
    M = len(pr.slave)
    rows = np.repeat(np.arange(M), 1 + pr.masters.shape[1])
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
        if pr.stab > 0:
            # 安定化: 開いている節点にも 係数 × 法線剛性 の弱いばね（力は入れない）。浮いた部品の剛体移動を防ぐ
            a_kn = a_kn + pr.area * pr.kn * pr.stab * (1.0 - act)
        Kc = Kc + Bn.T @ sp.diags(a_kn) @ Bn
        # 拡張ラグランジュの乗数と初期隙間による力:  A (λ - kn g0) n
        fc += Bn.T @ (pr.area * act * (pr.lam_n - pr.kn * pr.g0))
        if reg.type == "gasket":
            if np.any(pr.kt > 0):
                P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
                Kt3 = P * (pr.area * pr.kt)[:, None, None]
                for c in range(3):
                    for d in range(3):
                        coef = Kt3[:, c, d]
                        if np.any(coef):
                            Kc = Kc + B[c].T @ sp.diags(coef) @ B[d]
            continue
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


def _friction_tangent(pr: Pairs, mu: float, tt: np.ndarray, d: np.ndarray):
    """摩擦の整合接線と定数項。 摩擦応力を τ ≈ lam_t − Ktan·d（d = 相対変位）の形にする。

    固着: kt·P。 すべり: kt·(μp/|試行値|)·(P − e eᵀ) ＋ μ·kn·e nᵀ。
    最後の項は「摩擦限界 μp が法線方向の食い込みで変わる」分で、非対称になる。これを入れないと
    面圧と摩擦が強く影響し合う問題（すべりで部品が傾く等）でニュートン法が収束しない。
    """
    P = np.eye(3)[None] - pr.n[:, :, None] * pr.n[:, None, :]
    e = tt / np.maximum(np.linalg.norm(tt, axis=1, keepdims=True), 1e-300)
    Kt3 = pr.kt[:, None, None] * pr.ratio[:, None, None] * (
        P - np.where(pr.stick[:, None, None], 0.0, e[:, :, None] * e[:, None, :]))
    slip = pr.active & ~pr.stick
    Kt3 = Kt3 + (slip * mu * pr.kn)[:, None, None] * e[:, :, None] * pr.n[:, None, :]
    pr.Ktan = Kt3
    pr.lam_t = np.where(pr.active[:, None], tt + np.einsum("mij,mj->mi", Kt3, d), 0.0)
    pr.nonsym = bool(slip.any())


def update_states(pairs: list[Pairs], regions: list[ContactRegion], u: np.ndarray, al_update=True,
                  pen_tol: float = 0.0):
    """解いた変位から接触状態を更新。 戻り値: (状態が変わった数, 最大貫通量)"""
    U = u.reshape(-1, 3)
    changed = 0
    max_pen = 0.0
    for pr in pairs:
        reg = regions[pr.region]
        tol = pr.pen_tol if pr.pen_tol > 0 else pen_tol
        d = U[pr.slave] - np.einsum("ij,ijk->ik", pr.w, U[pr.masters])
        dn = np.einsum("ij,ij->i", d, pr.n)
        g = pr.g0 + dn
        dt = d - dn[:, None] * pr.n
        form = reg.resolved_formulation()
        use_al = form == "augmented_lagrange" and al_update
        if reg.type == "gasket":
            # 面圧 = 表(閉じ量)。 次の反復は今の点での接線（ニュートン法）: p ≈ p0 + k (c - c0)
            c = -g
            p, k = gasket_pressure(reg.gasket, c, pr.c_max)
            new_active = p > 0
            changed += int(np.sum(new_active != pr.active))
            pr.active = new_active
            kk = np.where(new_active, np.maximum(k, 1e-9 * pr.kn0), pr.kn0)
            pr.kn = kk
            pr.lam_n = np.where(new_active, p + kk * g, 0.0)    # λ - kn g = p
            pr.max_pen = 0.0
            pr.pressure = p
            pr.gap = g
            pr.slide = np.linalg.norm(dt, axis=1)
            pr.tau = pr.kt * pr.slide
            continue
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
            over = (np.abs(g) > tol) if reg.type in LINEAR_TYPES else (-g > tol)
            pr.lam_n = np.where(new_active, np.where(over, p, pr.lam_n), 0.0)
        pen = np.where(new_active, np.maximum(-g, 0.0), 0.0)
        if reg.type in LINEAR_TYPES:
            pen = np.abs(g)
        pr.max_pen = float(pen.max()) if len(pen) else 0.0
        max_pen = max(max_pen, pr.max_pen)
        if reg.update_stiffness == "each_iteration" and pr.kn0 is not None:
            # 剛性の更新（各反復）: 貫通が許容値を超えた節点は剛性を 2 倍（最初の 100 倍まで）、
            # 接触圧が出ていない節点は最初の値に戻す
            big = new_active & (pen > tol)
            pr.kn = np.where(big, np.minimum(pr.kn * 2.0, 100.0 * pr.kn0), np.where(new_active, pr.kn, pr.kn0))
        # 接線
        if reg.type in ("bonded", "rough"):
            tt = pr.lam_t - pr.kt[:, None] * dt
            if use_al:
                over_t = np.linalg.norm(dt, axis=1) > tol
                pr.lam_t = np.where(new_active[:, None], np.where(over_t[:, None], tt, pr.lam_t), 0.0)
            pr.stick = np.ones(len(g), bool)
            tau = np.linalg.norm(np.where(new_active[:, None], tt, 0.0), axis=1)
        elif reg.type == "frictional":
            if pr.slip_tol > 0:
                # 弾性すべり許容値: 固着中のずれが許容値で摩擦限界に達する接線剛性 kt = μ·p / 許容値
                kt_new = reg.mu * p / pr.slip_tol
                pr.kt = np.where(p > 0, np.clip(kt_new, 1e-3 * pr.kn0, pr.kn0), pr.kt)
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
            pr.ratio = np.where(stick, 1.0, limit / np.maximum(tn, 1e-300))
            _friction_tangent(pr, reg.mu, tt, d)
            pr.tau_now = np.where(new_active[:, None], tt, 0.0)
            tau = np.linalg.norm(pr.tau_now, axis=1)
        else:
            tau = np.zeros(len(g))
        pr.pressure = np.where(new_active, p, 0.0)
        pr.gap = g
        pr.slide = np.linalg.norm(dt, axis=1)
        pr.tau = tau
    return changed, max_pen


def penetration_ok(pairs: list[Pairs], pen_tol: float) -> bool:
    """すべての接触ペアの貫通が、それぞれの許容値以内か"""
    return all(pr.max_pen <= (pr.pen_tol if pr.pen_tol > 0 else pen_tol) for pr in pairs)


def commit_states(pairs: list[Pairs], regions, u: np.ndarray):
    """時間増分が収束したら、摩擦の履歴（摩擦応力と接線変位）を確定する"""
    U = u.reshape(-1, 3)
    for pr in pairs:
        if regions[pr.region].type == "gasket" and pr.gap is not None:
            pr.c_max = np.maximum(pr.c_max, -pr.gap)
            continue
        fric = regions[pr.region].type == "frictional"
        if fric:
            pr.tau_c = pr.tau_now.copy()
        if pr.finite:
            # 有限すべり: 収束した位置で接触相手を決め直す。摩擦応力は新しい接平面に向きを合わせ、
            # すべりの基準（dt_c）は決め直した相手との今の相対変位にする（ここからの増分ですべりを測る）
            repair_pairs(pr, u)
            if fric:
                mag = np.linalg.norm(pr.tau_c, axis=1, keepdims=True)
                tp = pr.tau_c - np.einsum("ij,ij->i", pr.tau_c, pr.n)[:, None] * pr.n
                pr.tau_c = tp * (mag / np.maximum(np.linalg.norm(tp, axis=1, keepdims=True), 1e-300))
        if fric:
            d = U[pr.slave] - np.einsum("ij,ijk->ik", pr.w, U[pr.masters])
            dn = np.einsum("ij,ij->i", d, pr.n)
            pr.dt_c = d - dn[:, None] * pr.n
        if pr.finite:
            # Target の外に出た節点は外す。固着・すべりの状態はそのままにして、決め直した相手で
            # 同じ摩擦応力になるよう、接線と定数項を作り直す（次のサブステップの最初の反復で使う）
            pr.active = pr.active & (pr.g0 < 1e29)
            if fric and pr.ratio is not None:
                _friction_tangent(pr, regions[pr.region].mu, pr.tau_c, d)
                pr.tau_now = np.where(pr.active[:, None], pr.tau_c, 0.0)


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
            near = (~pr.active) & (g < 3 * np.maximum(np.abs(pr.g0), 1e-3) + 1e-3) & (pr.g0 < 1e29)
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
        if len(np.unique(s)) < len(s):
            # ねじ（同じ節点に 2 つのフランク）: 接触圧の大きい方を表示（後に書いた値が残るので昇順に並べる）
            o = np.argsort(pres, kind="stable")
            s, status, pres, tau, g = s[o], status[o], pres[o], tau[o], g[o]
            act, slide = pr.active[o], pr.slide[o]
        else:
            act, slide = pr.active, pr.slide
        out["接触状態"][s] = status
        out["接触圧 [MPa]"][s] = pres
        out["貫通量 [mm]"][s] = np.where(act, np.maximum(-g, 0), 0)
        out["すべり距離 [mm]"][s] = np.where(status == ST_SLIDING, slide, 0.0)
        out["摩擦応力 [MPa]"][s] = tau
        out["隙間 [mm]"][s] = np.where(act, 0.0, np.where(g > 1e29, np.nan, np.maximum(g, 0)))
    return out

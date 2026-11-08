"""振動応答（Ansys の「調和応答（モード重ね合わせ）」「ランダム振動」に相当）。モーダル解析の結果を使う。

調和応答: 荷重（力・圧力・重力＝加速度の振幅）または基礎加振（加速度の振幅）を周波数 f で正弦波でかけたときの定常応答。
  u(ω) = Σ φ_i (φ_iᵀ F) / (ω_i² − ω² + 2 i ζ ω_i ω)
  周波数は範囲を等分した点に加え、固有振動数のまわりを細かく（Ansys の「解をクラスタ化」と同じ考え方）。
ランダム振動: 基礎の加速度 PSD [G²/Hz]（表）を 1 方向に入れたときの 1σ 応答（基礎に対する相対変位、応力）。
  モードどうしの相関も含めて周波数で積分（完全二次結合と同じ）。相当応力は Segalman の方法（Ansys と同じ）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .modal import ModalResult, _rigid_vectors
from .solver import BC, Result, build_load_and_constraints, material_D, nodal_stress

G = 9806.65   # mm/s²


def _modal_force(modal: ModalResult, mesh, material, bcs, base_acc=None):
    """各モードのモード荷重 φ_iᵀ F（荷重ベクトル F は bcs の力・圧力・重力、または基礎加振 -M r a）"""
    if base_acc is not None:
        # 基礎加振: 有効荷重 -M r a → φᵀ(-M r a) = -Γ·a（Γ は刺激係数）
        a = np.asarray(base_acc, float)
        return -(modal.participation[:, :3] @ a)
    loads = [b for b in bcs if b.kind in ("force", "pressure", "moment", "gravity")]
    F, _ = build_load_and_constraints(mesh, material, loads)
    if not np.any(F):
        raise ValueError("調和応答: 荷重（力・圧力・モーメント・重力）がありません")
    phi = modal.modes.reshape(modal.n_modes, -1)
    return phi @ F


def frequency_points(fmin, fmax, n, freqs, cluster=5, zeta=0.02):
    """等分の点＋固有振動数のまわり（半値幅の中）に cluster 点ずつ"""
    pts = list(np.linspace(fmin, fmax, n))
    for fn in freqs:
        if fmin < fn < fmax and fn > 0:
            for k in np.linspace(-1, 1, cluster):
                pts.append(fn * (1 + 2 * zeta * k))
    pts = np.unique(np.round(np.clip(pts, fmin, fmax), 9))
    return pts[pts > 0]


@dataclass
class HarmonicResult:
    mesh: object
    freqs: np.ndarray                 # 周波数 [Hz]
    q: np.ndarray                     # (nf, n_modes) 複素のモード座標
    modal: ModalResult
    info: dict = field(default_factory=dict)
    base_acc: np.ndarray | None = None

    def displacement(self, k):
        """周波数 k 番目の複素変位 (N,3)"""
        return np.tensordot(self.q[k], self.modal.modes, axes=(0, 0))

    def node_response(self, node, comp=None):
        """節点の変位振幅 (nf,) または成分 (nf,3)"""
        u = np.einsum("fm,mk->fk", self.q, self.modal.modes[:, node, :])
        return np.abs(u) if comp is None else np.abs(u[:, comp])

    def max_amplitude(self):
        """周波数ごとの最大の全変形振幅（全節点）"""
        return np.array([np.linalg.norm(np.abs(self.displacement(k)), axis=1).max() for k in range(len(self.freqs))])

    def at(self, f_hz, material, D=None, elem_mat=None, phase=0.0) -> Result:
        """周波数 f の応答（振幅と位相 phase [°] での瞬時値）を Result にする"""
        k = int(np.argmin(np.abs(self.freqs - f_hz)))
        uc = self.displacement(k)
        amp = np.abs(uc)
        inst = np.real(uc * np.exp(1j * np.radians(phase)))
        Dm = D if D is not None else material_D(material)
        st, sn = nodal_stress(self.mesh, inst.ravel(), Dm, elem_mat=elem_mat)
        r = Result(self.mesh, material, inst, st, sn, {}, {"解析": "調和応答", "周波数 [Hz]": float(self.freqs[k]),
                                                        "位相 [°]": phase})
        r.extra = {"変位振幅 [mm]": np.linalg.norm(amp, axis=1), "変位振幅 X [mm]": amp[:, 0],
                   "変位振幅 Y [mm]": amp[:, 1], "変位振幅 Z [mm]": amp[:, 2]}
        r.time = float(self.freqs[k])
        return r


def solve_harmonic(modal: ModalResult, mesh, material, bcs, fmin, fmax, n=50, zeta=0.02, base_acc=None,
                   log=print) -> HarmonicResult:
    """モード重ね合わせの調和応答。zeta: 減衰比（全モード一定、Ansys の「減衰比」）"""
    import time
    t0 = time.time()
    P = _modal_force(modal, mesh, material, bcs, base_acc)
    wn = 2 * np.pi * modal.freqs
    f = frequency_points(fmin, fmax, n, modal.freqs, zeta=zeta)
    w = 2 * np.pi * f
    H = 1.0 / (wn[None, :] ** 2 - w[:, None] ** 2 + 2j * zeta * wn[None, :] * w[:, None])
    q = H * P[None, :]
    if max(modal.freqs) < fmax:
        log(f"⚠ 求めたモードの最大振動数 {max(modal.freqs):.4g} Hz が周波数範囲の上限より低いです"
            "（Ansys と同じく、上限の 1.5 倍までのモードを含めると精度が上がります）")
    info = {"解析": "調和応答（モード重ね合わせ）", "周波数の点数": len(f), "減衰比": zeta,
            "使ったモード数": modal.n_modes, "合計時間 [s]": round(time.time() - t0, 3)}
    log(f"調和応答: {len(f)} 点（{fmin:g}–{fmax:g} Hz）、モード {modal.n_modes} 個")
    return HarmonicResult(mesh, f, q, modal, info, None if base_acc is None else np.asarray(base_acc, float))


# ---------------------------------------------------------------------------- ランダム振動

def psd_interp(table, f):
    """PSD の表 [[f Hz, 値], ...] を両対数で補間（範囲外は 0）"""
    tab = sorted(([float(a), float(b)] for a, b in table), key=lambda r: r[0])
    fs = np.log10([r[0] for r in tab])
    vs = np.log10([max(r[1], 1e-300) for r in tab])
    lf = np.log10(np.maximum(f, 1e-300))
    out = 10 ** np.interp(lf, fs, vs)
    out[(f < tab[0][0]) | (f > tab[-1][0])] = 0.0
    return out


def solve_random(modal: ModalResult, mesh, material, psd_table, direction=(0, 0, 1), zeta=0.02, elem_materials=None,
                 n_freq=2000, log=print) -> Result:
    """基礎加速度 PSD [G²/Hz] を direction に入れたときの 1σ 応答（相対変位・応力）。Result（応力は 1σ）を返す"""
    import time
    t0 = time.time()
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    fmin = min(r[0] for r in psd_table)
    fmax = max(r[0] for r in psd_table)
    # 周波数の点: 対数等分＋各モードのまわりを細かく
    f = np.unique(np.r_[np.geomspace(fmin, fmax, n_freq),
                        np.concatenate([fn * (1 + zeta * np.linspace(-6, 6, 61)) for fn in modal.freqs if fn > 0])])
    f = f[(f >= fmin) & (f <= fmax)]
    S = psd_interp(psd_table, f) * G ** 2            # (mm/s²)²/Hz
    wn = 2 * np.pi * modal.freqs
    w = 2 * np.pi * f
    gam = modal.participation[:, :3] @ d              # 刺激係数
    H = 1.0 / (wn[None, :] ** 2 - w[:, None] ** 2 + 2j * zeta * wn[None, :] * w[:, None])
    Hq = -H * gam[None, :]                            # (nf, m) 単位加速度あたりのモード座標
    df = np.gradient(f)
    # モード座標の共分散 C_ij = ∫ Re(Hq_i conj(Hq_j)) S df
    C = np.real(np.einsum("fi,fj,f->ij", Hq, np.conj(Hq), S * df))
    phi = modal.modes                                 # (m, N, 3)
    # 変位の 1σ（成分ごと）
    var_u = np.einsum("ij,ink,jnk->nk", C, phi, phi)
    sig_u = np.sqrt(np.maximum(var_u, 0))
    # モードの応力形
    if elem_materials is not None:
        mats, em = elem_materials
        D = np.stack([material_D(m) for m in mats])
    else:
        D, em = material_D(material), None
    Sm = np.stack([nodal_stress(mesh, phi[i].ravel(), D, elem_mat=em)[0] for i in range(modal.n_modes)])   # (m,N,6)
    var_s = np.einsum("ij,inp,jnp->np", C, Sm, Sm)     # 成分ごとの分散
    # Segalman の相当応力: σ_eq² = Σ_ij C_ij (s_iᵀ Q s_j)、Q はミーゼスの二次形式
    Q = np.array([[1, -.5, -.5, 0, 0, 0], [-.5, 1, -.5, 0, 0, 0], [-.5, -.5, 1, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3]], float)
    var_eq = np.einsum("ij,inp,pq,jnq->n", C, Sm, Q, Sm)
    stress = np.sqrt(np.maximum(var_s, 0))
    r = Result(mesh, material, sig_u, stress, np.zeros_like(stress), {},
               {"解析": "ランダム振動（1σ）", "入力の方向": d.round(4).tolist(), "減衰比": zeta,
                "入力の加速度 rms [G]": float(np.sqrt(np.trapezoid(psd_interp(psd_table, f), f))),
                "使ったモード数": modal.n_modes, "合計時間 [s]": round(time.time() - t0, 3)})
    r.extra = {"相当応力 1σ（Segalman） [MPa]": np.sqrt(np.maximum(var_eq, 0)),
               "変位 1σ（相対） [mm]": np.linalg.norm(sig_u, axis=1)}
    log(f"ランダム振動: 入力 {r.info['入力の加速度 rms [G]']:.4g} Grms → 最大変位 1σ {r.extra['変位 1σ（相対） [mm]'].max():.4g} mm、"
        f"相当応力 1σ {r.extra['相当応力 1σ（Segalman） [MPa]'].max():.4g} MPa")
    return r

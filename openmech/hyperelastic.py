"""超弾性（ゴム・エラストマー）— Total Lagrangian の大変形で使う

Ansys の Neo-Hookean / Mooney-Rivlin（2・3・5・9 パラメータのうち 2 パラメータ）/ Yeoh（3 次）/ Arruda-Boyce は
近似なし（ここでは Neo-Hookean・Mooney-Rivlin・Yeoh）。ひずみエネルギーは Ansys と同じ形（体積・偏差分離）:

  Neo-Hookean    W = μ/2 (Ī1 − 3) + 1/d (J − 1)²
  Mooney-Rivlin  W = C10 (Ī1 − 3) + C01 (Ī2 − 3) + 1/d (J − 1)²
  Yeoh           W = C10 (Ī1 − 3) + C20 (Ī1 − 3)² + C30 (Ī1 − 3)³ + 1/d (J − 1)²

  Ī1 = J^(−2/3) I1、Ī2 = J^(−4/3) I2。初期せん断弾性率 μ0 = 2(C10 + C01)、体積弾性率 K = 2/d（d = 0 は非圧縮 →
  ここでは K = 1000 μ0 に置き換える）。

第2ピオラ・キルヒホッフ応力 S = 2 ∂W/∂C を解析的に求め、材料接線 ∂S/∂E は複素ステップ微分で厳密に求める
（打ち切り誤差なし。ニュートン法が2次収束する）。
"""
from __future__ import annotations

import numpy as np

HYPER_MODELS = {"neo_hookean": "Neo-Hookean", "mooney_rivlin": "Mooney-Rivlin（2 パラメータ）",
                "yeoh": "Yeoh（3 次）"}
HYPER_PARAMS = {"neo_hookean": [("mu", "初期せん断弾性率 μ [MPa]")],
                "mooney_rivlin": [("C10", "C10 [MPa]"), ("C01", "C01 [MPa]")],
                "yeoh": [("C10", "C10 [MPa]"), ("C20", "C20 [MPa]"), ("C30", "C30 [MPa]")]}
VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def coefficients(h: dict):
    """(C10, C01, C20, C30, d) を返す"""
    m = h.get("model", "neo_hookean")
    g = lambda k: float(h.get(k, 0.0) or 0.0)  # noqa: E731
    if m == "neo_hookean":
        c10, c01, c20, c30 = g("mu") / 2, 0.0, 0.0, 0.0
    elif m == "mooney_rivlin":
        c10, c01, c20, c30 = g("C10"), g("C01"), 0.0, 0.0
    elif m == "yeoh":
        c10, c01, c20, c30 = g("C10"), 0.0, g("C20"), g("C30")
    else:
        raise ValueError(f"超弾性モデル「{m}」には対応していません")
    mu0 = 2 * (c10 + c01)
    if mu0 <= 0:
        raise ValueError("超弾性の定数から求めた初期せん断弾性率が 0 以下です")
    d = g("d")
    if d <= 0:
        d = 2.0 / (1000.0 * mu0)          # 非圧縮 → ほぼ非圧縮（ν ≈ 0.4995）
    return c10, c01, c20, c30, d


def initial_E_nu(h: dict):
    """微小ひずみでの等価な E, ν（線形解析・接触剛性・弱いばねの目安に使う）"""
    c10, c01, _, _, d = coefficients(h)
    mu, K = 2 * (c10 + c01), 2.0 / d
    return 9 * K * mu / (3 * K + mu), (3 * K - 2 * mu) / (2 * (3 * K + mu))


class HyperModel:
    def __init__(self, h: dict):
        self.c10, self.c01, self.c20, self.c30, self.d = coefficients(h)

    def _S_from_C(self, C, part="all"):
        """C (n,3,3)（複素可）→ S (n,3,3)。part: "all" / "iso"（偏差だけ）/ "vol"（体積だけ）"""
        I = np.eye(3)
        J2 = np.linalg.det(C)
        J = np.sqrt(J2)
        Ci = np.linalg.inv(C)
        I1 = np.trace(C, axis1=1, axis2=2)
        I2 = 0.5 * (I1 ** 2 - np.einsum("eij,eji->e", C, C))
        Jm23 = J ** (-2.0 / 3.0)
        Jm43 = Jm23 ** 2
        I1b = Jm23 * I1
        a = I1b - 3.0
        W1 = self.c10 + 2 * self.c20 * a + 3 * self.c30 * a ** 2
        W2 = self.c01
        WJ = 2.0 / self.d * (J - 1.0)
        dI1b = Jm23[:, None, None] * (I[None] - (I1 / 3.0)[:, None, None] * Ci)
        dI2b = Jm43[:, None, None] * (I1[:, None, None] * I[None] - C - (2.0 / 3.0 * I2)[:, None, None] * Ci)
        iso = 2 * (W1[:, None, None] * dI1b + W2 * dI2b)
        vol = (WJ * J)[:, None, None] * Ci
        return iso if part == "iso" else vol if part == "vol" else iso + vol

    def pressure(self, J):
        """コーシー応力の体積成分 σ = U'(J) I の U'(J)（U = (J−1)²/d）"""
        return 2.0 / self.d * (J - 1.0)

    def S_C(self, Ev, part="all"):
        """グリーン・ラグランジュひずみ Ev (n,6)（工学せん断）→ S (n,6), 接線 D = ∂S/∂E (n,6,6)"""
        n = len(Ev)
        h = 1e-30
        # 6 方向の複素摂動をまとめて評価: (n,7) 個 → 1 回の呼び出し
        Cb = np.zeros((n, 7, 3, 3), complex)
        for k, (i, j) in enumerate(VOIGT):
            v = Ev[:, k] if i == j else 0.5 * Ev[:, k]
            Cb[:, :, i, j] += 2 * v[:, None]
            if i != j:
                Cb[:, :, j, i] += 2 * v[:, None]
        Cb += np.eye(3)
        for k, (i, j) in enumerate(VOIGT):
            # E_k に ih を足す: 対角は C_ii += 2ih、せん断（工学）は C_ij = C_ji += ih
            if i == j:
                Cb[:, k + 1, i, i] += 2j * h
            else:
                Cb[:, k + 1, i, j] += 1j * h
                Cb[:, k + 1, j, i] += 1j * h
        Sm = self._S_from_C(Cb.reshape(-1, 3, 3), part).reshape(n, 7, 3, 3)
        Sv = np.stack([Sm[..., i, j] for i, j in VOIGT], axis=-1)       # (n,7,6)
        S = Sv[:, 0].real
        D = (Sv[:, 1:].imag / h).transpose(0, 2, 1)                     # D[:, a, k] = ∂S_a/∂E_k
        return S, 0.5 * (D + D.transpose(0, 2, 1))


def uniaxial_nominal(h: dict, lam):
    """非圧縮の単軸引張の公称応力（検証用の理論式）: P = 2(λ − λ⁻²)(W1 + W2/λ)"""
    c10, c01, c20, c30, _ = coefficients(h)
    lam = np.asarray(lam, float)
    a = lam ** 2 + 2 / lam - 3
    W1 = c10 + 2 * c20 * a + 3 * c30 * a ** 2
    return 2 * (lam - lam ** -2) * (W1 + c01 / lam)


def fit_uniaxial(model: str, data):
    """単軸引張の試験データ [[ストレッチ λ（または公称ひずみ+1）, 公称応力 MPa], ...] から定数を最小二乗で求める
    （Ansys Engineering Data のカーブフィットに相当。非圧縮を仮定。式が定数について線形なので厳密な最小二乗）。
    戻り値: (定数 dict, 相対 RMS 誤差)"""
    d = np.asarray(data, float).reshape(-1, 2)
    lam, P = d[:, 0], d[:, 1]
    ok = np.abs(lam - 1) > 1e-9
    lam, P = lam[ok], P[ok]
    if len(lam) < 2:
        raise ValueError("試験データが少なすぎます（2 点以上）")
    g = 2 * (lam - lam ** -2)
    a = lam ** 2 + 2 / lam - 3
    cols = {"neo_hookean": [g], "mooney_rivlin": [g, g / lam], "yeoh": [g, g * 2 * a, g * 3 * a ** 2]}[model]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, P, rcond=None)
    rms = float(np.sqrt(np.mean((A @ c - P) ** 2)) / max(np.sqrt(np.mean(P ** 2)), 1e-30))
    names = {"neo_hookean": ["C10"], "mooney_rivlin": ["C10", "C01"], "yeoh": ["C10", "C20", "C30"]}[model]
    out = dict(zip(names, map(float, c)))
    if model == "neo_hookean":
        out = {"mu": 2 * out["C10"]}
    return out, rms

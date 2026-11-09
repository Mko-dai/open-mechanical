"""材料ライブラリ。

単位系は mm・N・MPa・t（トン）で統一しています。
  ヤング率 E [MPa]、密度 [t/mm^3]（= kg/m^3 × 1e-12）、降伏応力 [MPa]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


ELASTIC_MODELS = {"isotropic": "等方弾性", "orthotropic": "直交異方性弾性", "anisotropic": "異方性弾性"}
PLASTIC_MODELS = {"none": "なし（線形弾性）", "bilinear_iso": "等方硬化（2直線）", "multilinear_iso": "等方硬化（多直線）",
                  "bilinear_kin": "移動硬化（2直線）", "multilinear_kin": "移動硬化（多直線）"}


@dataclass
class Material:
    """材料定数（Ansys Engineering Data 相当）

    弾性:
      isotropic   等方弾性 E, ν。temp_table に [[温度, E, ν], ...] があれば温度で補間（温度依存）
      orthotropic 直交異方性 ortho = {Ex,Ey,Ez, nuxy,nuyz,nuxz, Gxy,Gyz,Gxz}（材料座標系の軸方向）
      anisotropic 異方性 aniso = 6×6 の剛性マトリクス D [MPa]（成分順 xx,yy,zz,xy,yz,zx・工学せん断ひずみ）
    熱膨張: alpha [1/°C]（等方）、alpha_ortho = [αx,αy,αz]（直交異方性、0 なら alpha）、ゼロ熱ひずみ参照温度 ref_temp [°C]
    塑性（ミーゼス）: plasticity = bilinear_iso / multilinear_iso / bilinear_kin / multilinear_kin
      2直線: 降伏応力 yield_strength、接線係数 tangent_modulus [MPa]
      多直線: ml_table = [[塑性ひずみ, 応力 MPa], ...]（最初の点は 塑性ひずみ 0 = 降伏応力）
    """
    name: str
    E: float              # ヤング率 [MPa]
    nu: float             # ポアソン比
    density: float        # 密度 [kg/m^3]（表示用。計算時に t/mm^3 へ変換）
    yield_strength: float = 0.0   # 引張降伏応力 [MPa]
    tensile_strength: float = 0.0  # 引張強さ [MPa]
    elastic: str = "isotropic"
    temp_table: list = field(default_factory=list)
    ortho: dict = field(default_factory=dict)
    aniso: list = field(default_factory=list)
    alpha: float = 0.0
    alpha_ortho: list = field(default_factory=list)
    ref_temp: float = 22.0
    plasticity: str = "none"
    tangent_modulus: float = 0.0
    ml_table: list = field(default_factory=list)
    # 熱（Ansys Engineering Data の「等方熱伝導率」「比熱」）
    conductivity: float = 0.0     # 熱伝導率 [W/(m·K)]（0 = 未設定）
    specific_heat: float = 0.0    # 比熱 [J/(kg·K)]
    # 電気（静電場・圧電）
    permittivity: float = 1.0     # 比誘電率 εr
    piezo: dict = field(default_factory=dict)   # 圧電: {"e": 6×3 の圧電定数 e [C/m²]（応力型）, "eps_s": [ε11,ε22,ε33] 比誘電率（ひずみ一定）}
    # 疲労（S-N 曲線）: [[繰返し数, 応力振幅 MPa], ...]（両対数で補間）
    sn_table: list = field(default_factory=list)
    # 超弾性（hyperelastic.py）: {"model": "neo_hookean"|"mooney_rivlin"|"yeoh", "mu"/"C10"/"C01"/"C20"/"C30", "d"}
    hyper: dict = field(default_factory=dict)
    # クリープ（Ansys の Norton, TBOPT=10）: ε̇cr = C1 σ^C2 exp(−C3/T)  {"C1": [1/s/MPa^C2], "C2": 指数, "C3": [K]}
    creep: dict = field(default_factory=dict)
    # 複合材の層の強度（composite.py の破損評価）: {"Xt","Xc","Yt","Yc","S"} [MPa]（圧縮も正の値）
    composite_strength: dict = field(default_factory=dict)

    @property
    def density_t_mm3(self) -> float:
        return self.density * 1e-12

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        keys = set(Material.__dataclass_fields__)
        return Material(**{k: v for k, v in d.items() if k in keys})

    # ------------------------------------------------------------------ 計算用
    def E_nu_at(self, T=None):
        """温度 T での E, ν（温度依存の表があれば線形補間。範囲外は端の値＝Ansys と同じ）"""
        if self.hyper and self.hyper.get("model"):
            from .hyperelastic import initial_E_nu
            return initial_E_nu(self.hyper)
        tab = [r for r in (self.temp_table or []) if len(r) >= 3]
        if T is None or not tab:
            return float(self.E), float(self.nu)
        tab = sorted(tab, key=lambda r: r[0])
        Ts = [float(r[0]) for r in tab]
        return float(np.interp(T, Ts, [float(r[1]) for r in tab])), float(np.interp(T, Ts, [float(r[2]) for r in tab]))

    def D_local(self, T=None):
        """材料座標系での剛性マトリクス (6,6)"""
        if self.elastic == "anisotropic" and len(self.aniso) == 6:
            D = np.array(self.aniso, float)
            return 0.5 * (D + D.T)
        if self.elastic == "orthotropic" and self.ortho:
            o = {k: float(v) for k, v in self.ortho.items()}
            Ex, Ey, Ez = o["Ex"], o["Ey"], o["Ez"]
            nxy, nyz, nxz = o["nuxy"], o["nuyz"], o["nuxz"]
            S = np.zeros((6, 6))
            S[0, 0], S[1, 1], S[2, 2] = 1 / Ex, 1 / Ey, 1 / Ez
            S[0, 1] = S[1, 0] = -nxy / Ex
            S[1, 2] = S[2, 1] = -nyz / Ey
            S[0, 2] = S[2, 0] = -nxz / Ex
            S[3, 3], S[4, 4], S[5, 5] = 1 / o["Gxy"], 1 / o["Gyz"], 1 / o["Gxz"]
            return np.linalg.inv(S)
        E, nu = self.E_nu_at(T)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[np.arange(3), np.arange(3)] += 2 * mu
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def state(self, T=None, R=None) -> "MaterialState":
        """温度 T [°C]・材料座標系の回転 R (3,3: 行 = 材料軸の全体座標での向き) での計算用の材料"""
        D = self.D_local(T)
        Tm = strain_rotation(R) if R is not None else None
        if Tm is not None:
            D = Tm.T @ D @ Tm
        eps = np.zeros(6)
        if T is not None:
            a = [float(x) for x in (self.alpha_ortho or [])]
            if len(a) == 3 and any(a) and self.elastic != "isotropic":
                el = np.array([a[0], a[1], a[2], 0, 0, 0]) * (T - self.ref_temp)
                if R is not None:
                    # 材料座標のひずみ → 全体座標のひずみ: ε_g = T⁻¹ ε_l
                    el = np.linalg.solve(Tm, el)
                eps = el
            elif self.alpha:
                eps = np.array([1.0, 1.0, 1.0, 0, 0, 0]) * self.alpha * (T - self.ref_temp)
        E, nu = self.E_nu_at(T)
        if self.elastic != "isotropic":
            # 接触剛性・表示用の代表値
            E = float(np.mean(np.diag(np.linalg.inv(D))[:3] ** -1))
        st = MaterialState(self, D, eps, E, nu, T)
        st.R = None if R is None else np.asarray(R, float)
        return st


class MaterialState:
    """解析に使う材料（温度・向きを反映済み）。Material と同じ名前の属性を持つ"""

    def __init__(self, base: Material, D, eps_th, E, nu, T):
        self.base, self.Dm, self.eps_th, self.E, self.nu, self.T = base, D, eps_th, E, nu, T
        self.name = base.name if T is None else f"{base.name}（{T:g}°C）"
        self.density, self.yield_strength = base.density, base.yield_strength
        self.tensile_strength = base.tensile_strength
        self.plasticity = base.plasticity
        self.R = None

    @property
    def density_t_mm3(self):
        return self.density * 1e-12

    def __getattr__(self, name):
        # 熱伝導率・比熱・誘電率・圧電・S-N 曲線など、計算用に変換しない定数は元の材料のものを使う
        if name.startswith("__") or name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def plastic(self):
        """積分点の状態を持つ材料（塑性・クリープ）"""
        return self.plasticity != "none" or self.creeping

    @property
    def creeping(self):
        return bool(self.base.creep) and float(self.base.creep.get("C1", 0) or 0) > 0

    @property
    def hyperelastic(self):
        return bool(self.base.hyper) and bool(self.base.hyper.get("model"))

    def hardening(self):
        """(初期降伏応力, 応力-塑性ひずみ表 [(εp, σ)], 移動硬化か)"""
        b = self.base
        if b.plasticity == "none":
            return 1e30, [(0.0, 1e30)], False          # クリープのみ（降伏しない）
        kin = b.plasticity.endswith("_kin")
        if b.plasticity.startswith("bilinear"):
            E = self.E
            Et = min(float(b.tangent_modulus), 0.999 * E)
            H = E * Et / (E - Et) if Et > 0 else 0.0
            sy = float(b.yield_strength)
            return sy, [(0.0, sy), (1.0, sy + H)], kin
        tab = sorted(([float(r[0]), float(r[1])] for r in b.ml_table if len(r) >= 2), key=lambda r: r[0])
        if not tab:
            raise ValueError(f"材料「{b.name}」の多直線硬化の表が空です")
        if tab[0][0] > 0:
            tab.insert(0, [0.0, tab[0][1]])
        return tab[0][1], [tuple(r) for r in tab], kin

    def to_dict(self):
        return self.base.to_dict()


def strain_rotation(R):
    """ひずみ（工学せん断）ベクトルの座標変換 ε_local = T ε_global（成分順 xx,yy,zz,xy,yz,zx）"""
    R = np.asarray(R, float)
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    T = np.zeros((6, 6))
    for j, (k, l) in enumerate(idx):
        e = np.zeros((3, 3))
        if k == l:
            e[k, k] = 1.0
        else:
            e[k, l] = e[l, k] = 0.5   # 工学せん断ひずみ 1 = テンソル成分 1/2
        el = R @ e @ R.T
        for i, (a, c) in enumerate(idx):
            T[i, j] = el[a, a] if a == c else 2 * el[a, c]
    return T


def rotation_xyz(rx, ry, rz):
    """X→Y→Z の順の回転角 [度] から、材料軸（行）の全体座標での向き"""
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    Q = Rz @ Ry @ Rx          # 列 = 材料軸
    return Q.T


# Ansys Engineering Data の代表値に近い一般的な値
LIBRARY: dict[str, Material] = {m.name: m for m in [
    Material("構造用鋼 (Structural Steel)", 200000, 0.30, 7850, 250, 460),
    Material("SS400", 206000, 0.30, 7850, 245, 400),
    Material("S45C", 205000, 0.30, 7850, 345, 570),
    Material("ステンレス SUS304", 193000, 0.29, 7930, 205, 520),
    Material("アルミ合金 A5052", 70600, 0.33, 2680, 195, 260),
    Material("アルミ合金 A6061-T6", 68900, 0.33, 2700, 276, 310),
    Material("アルミ合金 A7075-T6", 71700, 0.33, 2810, 503, 572),
    Material("チタン合金 Ti-6Al-4V", 113800, 0.34, 4430, 880, 950),
    Material("ねずみ鋳鉄 FC250", 110000, 0.28, 7200, 0, 250),
    Material("銅 C1100", 117000, 0.34, 8900, 69, 220),
    Material("黄銅 C2801", 103000, 0.34, 8400, 150, 330),
    Material("ABS樹脂", 2300, 0.35, 1050, 40, 45),
    Material("POM（ジュラコン）", 2800, 0.35, 1410, 62, 65),
    Material("PA66（ナイロン）", 3000, 0.39, 1140, 80, 85),
    Material("PC（ポリカーボネート）", 2400, 0.37, 1200, 62, 65),
]}

# 熱伝導率 [W/(m·K)]・比熱 [J/(kg·K)]・線膨張係数 [1/°C]（一般的な値）
_THERMAL = {"構造用鋼 (Structural Steel)": (60.5, 434, 1.2e-5), "SS400": (51.6, 473, 1.17e-5),
            "S45C": (47.5, 486, 1.17e-5), "ステンレス SUS304": (16.3, 500, 1.73e-5),
            "アルミ合金 A5052": (137, 900, 2.38e-5), "アルミ合金 A6061-T6": (167, 896, 2.36e-5),
            "アルミ合金 A7075-T6": (130, 960, 2.36e-5), "チタン合金 Ti-6Al-4V": (6.7, 526, 8.6e-6),
            "ねずみ鋳鉄 FC250": (48, 460, 1.1e-5), "銅 C1100": (391, 385, 1.7e-5), "黄銅 C2801": (121, 380, 2.0e-5),
            "ABS樹脂": (0.17, 1400, 8.0e-5), "POM（ジュラコン）": (0.23, 1470, 1.1e-4),
            "PA66（ナイロン）": (0.25, 1700, 8.0e-5), "PC（ポリカーボネート）": (0.2, 1200, 6.8e-5)}
for _n, (_k, _c, _a) in _THERMAL.items():
    if _n in LIBRARY:
        LIBRARY[_n].conductivity, LIBRARY[_n].specific_heat = _k, _c
        if not LIBRARY[_n].alpha:
            LIBRARY[_n].alpha = _a
# 構造用鋼の S-N 曲線（Ansys Engineering Data の構造用鋼の代表的な形: 平均応力 0、両振り）
LIBRARY["構造用鋼 (Structural Steel)"].sn_table = [
    [10, 3999], [20, 2827], [50, 1896], [100, 1413], [200, 1069], [2000, 441], [10000, 262], [20000, 214],
    [1e5, 138], [2e5, 114], [1e6, 86.2]]

DEFAULT = "構造用鋼 (Structural Steel)"

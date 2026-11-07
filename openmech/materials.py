"""材料ライブラリ。

単位系は mm・N・MPa・t（トン）で統一しています。
  ヤング率 E [MPa]、密度 [t/mm^3]（= kg/m^3 × 1e-12）、降伏応力 [MPa]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Material:
    name: str
    E: float              # ヤング率 [MPa]
    nu: float             # ポアソン比
    density: float        # 密度 [kg/m^3]（表示用。計算時に t/mm^3 へ変換）
    yield_strength: float = 0.0   # 引張降伏応力 [MPa]
    tensile_strength: float = 0.0  # 引張強さ [MPa]

    @property
    def density_t_mm3(self) -> float:
        return self.density * 1e-12

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Material(**d)


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

DEFAULT = "構造用鋼 (Structural Steel)"

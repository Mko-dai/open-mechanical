"""操作履歴（Femtet の履歴と同じく、操作ごとに状態を記録して どの時点にも戻れる）"""
from __future__ import annotations

import datetime
import json

# プロジェクトの項目 → 操作の種類（自動でつける名前）
_SECTIONS = [("geometry", "形状"), ("mesh", "メッシュ設定"), ("materials", "材料定数"), ("material", "材料"),
             ("body_attrs", "ボディ属性"), ("body_assign", "ボディ属性・材料の割り当て"), ("bcs", "境界条件"),
             ("contacts", "接触"), ("settings", "解析設定"), ("results", "結果の項目"),
             ("share_topology", "節点共有"), ("parts", "パート"), ("env_temp", "環境温度"), ("title", "タイトル")]


def describe_change(old: dict | None, new: dict) -> str:
    """2 つの状態の差から操作の名前をつける"""
    if old is None:
        return "開始"
    changed = [name for key, name in _SECTIONS if old.get(key) != new.get(key)]
    if not changed:
        return "変更"
    fo, fn = old.get("geometry", {}).get("features", []), new.get("geometry", {}).get("features", [])
    if "形状" in changed and len(fn) == len(fo) + 1:
        from .geometry import FEATURES
        return "作成: " + FEATURES.get(fn[-1].get("type"), ("ブール演算",))[0]
    bo, bn = old.get("bcs", []), new.get("bcs", [])
    if changed == ["境界条件"] and len(bn) == len(bo) + 1:
        return "追加: " + _bc_name(bn[-1])
    if changed == ["境界条件"] and len(bn) == len(bo) - 1:
        return "削除: 境界条件"
    return "変更: " + "・".join(changed[:3])


def _bc_name(d):
    names = {"fixed": "固定支持", "displacement": "変位", "force": "力", "pressure": "圧力", "moment": "モーメント",
             "gravity": "重力", "thermal": "熱条件", "symmetry": "対称領域", "frictionless": "摩擦なし支持"}
    return d.get("name") or names.get(d.get("kind"), d.get("kind", ""))


class History:
    """状態（プロジェクトの辞書）の列。pos が現在の位置"""

    def __init__(self, limit: int = 300):
        self.items: list[dict] = []
        self.pos = -1
        self.limit = limit

    @staticmethod
    def _key(state: dict) -> str:
        return json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)

    def record(self, state: dict, label: str | None = None) -> bool:
        """状態が変わっていれば記録（戻った位置から先の履歴は捨てる）。記録したら True"""
        key = self._key(state)
        if self.pos >= 0 and self.items[self.pos]["key"] == key:
            return False
        old = json.loads(self.items[self.pos]["key"]) if self.pos >= 0 else None
        del self.items[self.pos + 1:]
        self.items.append({"key": key, "label": label or describe_change(old, state),
                           "time": datetime.datetime.now().strftime("%H:%M:%S")})
        if len(self.items) > self.limit:
            del self.items[0]
        self.pos = len(self.items) - 1
        return True

    def replace_current(self, state: dict):
        """戻った直後の自動調整（メッシュの再生成など）は、その時点の状態として上書き"""
        if self.pos >= 0:
            self.items[self.pos]["key"] = self._key(state)

    def state(self, i: int) -> dict:
        return json.loads(self.items[i]["key"])

    def can_undo(self):
        return self.pos > 0

    def can_redo(self):
        return self.pos < len(self.items) - 1

    def reset(self, state: dict, label="開始"):
        self.items, self.pos = [], -1
        self.record(state, label)

"""変数（パラメータ）と計算式。Femtet の「変数」と同じ考え方:

- 変数は「変数名」「式（または値）」「コメント」の表。式には他の変数・関数が使える
    例: L = 100, W = L/5, H = sqrt(W*10), R = min(W, H)/2
- 寸法・荷重・メッシュサイズの入力欄に式を書ける（数値の代わりに "L/2+5" など）
- 式はフィーチャーなら f["expr"][キー]、境界条件なら bc.exprs[成分] に保存し、
  数値の欄には計算した値を入れておく（式を持たない欄はこれまでどおり数値だけ）

安全のため eval は使わず、式を構文木（ast）で評価する（四則演算・べき乗・関数・変数のみ）。
"""
from __future__ import annotations

import ast
import itertools
import math
import re

import numpy as np


class ParamError(ValueError):
    pass


FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10,
    "abs": abs, "min": min, "max": max, "round": round, "floor": math.floor, "ceil": math.ceil,
    "int": lambda x: float(int(x)), "rad": math.radians, "deg": math.degrees,
    "radians": math.radians, "degrees": math.degrees, "pow": math.pow, "hypot": math.hypot,
    # 三角関数を度で（Femtet の sind/cosd 相当）
    "sind": lambda x: math.sin(math.radians(x)), "cosd": lambda x: math.cos(math.radians(x)),
    "tand": lambda x: math.tan(math.radians(x)),
    "iif": lambda c, a, b: a if c else b,     # 条件で切り替え（Excel の IF）
}
CONSTS = {"pi": math.pi, "e": math.e}
NAME_RE = re.compile(r"^[A-Za-z_぀-ヿ一-鿿][A-Za-z0-9_぀-ヿ一-鿿]*$")

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b}
_CMP = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


def _prep(expr: str) -> str:
    s = str(expr).strip()
    if s.startswith("="):
        s = s[1:]
    # 全角の記号・数字を半角に、^ はべき乗（Femtet・Excel と同じ）
    s = s.translate(str.maketrans("０１２３４５６７８９＋－×÷＊／（）．，＾　", "0123456789+-*/*/().,^ "))
    return s.replace("^", "**")


def evaluate(expr, env: dict | None = None) -> float:
    """式を評価して数値を返す。env: {変数名: 値}"""
    if isinstance(expr, (int, float, np.floating, np.integer)) and not isinstance(expr, bool):
        return float(expr)
    env = env or {}
    src = _prep(expr)
    if not src:
        raise ParamError("式が空です")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise ParamError(f"式の書き方が正しくありません: 「{expr}」") from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
            return float(_CMP[type(n.ops[0])](ev(n.left), ev(n.comparators[0])))
        if isinstance(n, ast.Name):
            if n.id in env:
                return float(env[n.id])
            if n.id in CONSTS:
                return CONSTS[n.id]
            raise ParamError(f"変数「{n.id}」が定義されていません")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            fn = FUNCS.get(n.func.id)
            if fn is None:
                raise ParamError(f"関数「{n.func.id}」は使えません")
            return float(fn(*[ev(a) for a in n.args]))
        raise ParamError(f"式に使えない要素があります: 「{expr}」")

    try:
        v = ev(tree)
    except ParamError:
        raise
    except ZeroDivisionError:
        raise ParamError(f"0 で割っています: 「{expr}」") from None
    except (ValueError, TypeError, OverflowError) as e:
        raise ParamError(f"計算できません: 「{expr}」（{e}）") from None
    if not math.isfinite(v):
        raise ParamError(f"計算結果が有限の値になりません: 「{expr}」")
    return v


def is_number(text) -> bool:
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return True
    try:
        float(str(text).strip())
        return True
    except ValueError:
        return False


def names_in(expr) -> set[str]:
    """式に出てくる変数名（関数名・定数は除く）"""
    try:
        tree = ast.parse(_prep(expr), mode="eval")
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in FUNCS and n.id not in CONSTS:
            out.add(n.id)
    return out


def check_name(name: str):
    if not name or not NAME_RE.match(name):
        raise ParamError(f"変数名「{name}」は使えません（英字・かな・漢字で始め、英数字と _ のみ）")
    if name in FUNCS or name in CONSTS:
        raise ParamError(f"変数名「{name}」は関数名・定数名と同じなので使えません")


def resolve(params: list, overrides: dict | None = None) -> dict:
    """変数の表 [{"name", "expr", "comment"}] を評価して {変数名: 値} を返す。

    overrides: {変数名: 値または式}（パラメトリック解析でその変数の式を置き換える）
    変数どうしの参照は順番によらず解決する（循環参照はエラー）。
    """
    overrides = overrides or {}
    exprs = {}
    for p in params:
        nm = str(p.get("name", "")).strip()
        check_name(nm)
        if nm in exprs:
            raise ParamError(f"変数「{nm}」が 2 回定義されています")
        exprs[nm] = overrides.get(nm, p.get("expr", 0))
    for nm in overrides:
        if nm not in exprs:
            raise ParamError(f"変数「{nm}」がありません")
    env: dict[str, float] = {}
    todo = dict(exprs)
    while todo:
        done = []
        for nm, ex in todo.items():
            deps = names_in(ex) if not is_number(ex) else set()
            unknown = deps - set(exprs)
            if unknown:
                raise ParamError(f"変数「{nm}」の式に未定義の変数があります: {', '.join(sorted(unknown))}")
            if deps <= set(env):
                try:
                    env[nm] = evaluate(ex, env)
                except ParamError as e:
                    raise ParamError(f"変数「{nm}」: {e}") from None
                done.append(nm)
        if not done:
            raise ParamError("変数が循環して参照されています: " + ", ".join(sorted(todo)))
        for nm in done:
            del todo[nm]
    return env


# ---------------------------------------------------------------------------- 欄への式の保存

def split_points(text) -> list[list[str]]:
    """「u,v; u,v; …」を括弧の中のカンマを区切らずに分ける"""
    rows, cur, depth, buf = [], [], 0, ""
    for ch in str(text).replace("\n", ";") + ";":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in ",;":
            if buf.strip():
                cur.append(buf.strip())
            buf = ""
            if ch == ";" and cur:
                rows.append(cur)
                cur = []
            continue
        buf += ch
    return rows


def eval_points(text, env) -> str:
    rows = split_points(text)
    out = []
    for r in rows:
        if len(r) != 2:
            raise ParamError(f"頂点の書式が正しくありません: 「{', '.join(r)}」（例: 0,0; L,0; L,H）")
        out.append(f"{evaluate(r[0], env):.10g},{evaluate(r[1], env):.10g}")
    return "; ".join(out)


def points_have_expr(text) -> bool:
    return any(not is_number(c) for r in split_points(text) for c in r)


def set_field(d: dict, key: str, raw, env: dict):
    """フィーチャー d の欄 key に数値または式 raw を入れる（式なら d["expr"][key] に保存して評価値を入れる）"""
    ex = d.setdefault("expr", {})
    if key == "points":
        text = str(raw)
        if points_have_expr(text):
            d[key] = eval_points(text, env)
            ex[key] = text
        else:
            d[key] = text
            ex.pop(key, None)
    elif isinstance(raw, str) and not is_number(raw):
        d[key] = evaluate(raw, env)
        ex[key] = raw.strip()
    else:
        d[key] = float(raw)
        ex.pop(key, None)
    if not ex:
        d.pop("expr", None)


def field_text(d: dict, key: str, default=0.0):
    """入力欄に表示する文字（式があれば式、無ければ数値）"""
    ex = (d.get("expr") or {}).get(key)
    if ex is not None:
        return ex
    v = d.get(key, default)
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def apply_to_dict(d: dict, env: dict, where=""):
    for key, ex in (d.get("expr") or {}).items():
        try:
            d[key] = eval_points(ex, env) if key == "points" else evaluate(ex, env)
        except ParamError as e:
            raise ParamError(f"{where}「{key}」: {e}") from None


def sweep_values(start, stop, n=None, step=None, values_text=""):
    """スイープする値の列（値の列挙 > 刻み > 分割数 の順に優先）"""
    if str(values_text).strip():
        return [evaluate(v) for r in split_points(values_text.replace(",", ";")) for v in r]
    a, b = float(start), float(stop)
    if step:
        st = abs(float(step)) * (1 if b >= a else -1)
        k = int(math.floor((b - a) / st + 1e-9))
        return [a + i * st for i in range(k + 1)]
    n = max(1, int(n or 1))
    return [a + (b - a) * i / n for i in range(n + 1)] if n > 0 else [a]


def sweep_cases(sweeps: dict, mode="all") -> list[dict]:
    """{変数名: [値...]} からケースの列を作る。mode="all" 全組み合わせ（総当たり）/ "each" 1 変数ずつ"""
    names = list(sweeps)
    if not names:
        return []
    if mode == "each":
        return [{nm: v} for nm in names for v in sweeps[nm]]
    if mode == "zip":
        n = min(len(v) for v in sweeps.values())
        return [{nm: sweeps[nm][i] for nm in names} for i in range(n)]
    return [dict(zip(names, combo)) for combo in itertools.product(*(sweeps[nm] for nm in names))]

"""最適化（Femtet の最適化・Ansys の DesignXplorer に相当）。パラメトリック解析の変数を自動で決める。

方法: ベイズ最適化（ガウス過程＋期待改善量 EI）。初期点はラテン超方格。制約は「出力 ≤ 上限 / ≥ 下限」で、
      制約もガウス過程で予測し、満たす確率を EI に掛ける（制約付き EI）。
目的: 出力 1 つの最小化 / 最大化。
使い方: optimize(evaluate, variables, objective, constraints, n_init, n_iter)
  evaluate(dict 変数→値) → dict 出力→値（失敗したら例外）
  variables:   [{"name": "t", "min": 1, "max": 5}]
  objective:   {"output": "質量 [kg]", "goal": "min"}
  constraints: [{"output": "相当応力（ミーゼス） [MPa] 最大", "max": 200}]
"""
from __future__ import annotations

import math
import time

import numpy as np


def _lhs(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


class GP:
    """ガウス過程回帰（RBF カーネル・長さスケールは対数尤度の格子探索）。入力は [0,1]^d に正規化済み"""

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu, self.sd = y.mean(), max(y.std(), 1e-12)
        self.y = (y - self.mu) / self.sd
        best = None
        for ls in (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0):
            for noise in (1e-6, 1e-3, 1e-2):
                try:
                    K = self._k(self.X, self.X, ls) + noise * np.eye(len(X))
                    Lc = np.linalg.cholesky(K)
                    a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, self.y))
                    ll = -0.5 * self.y @ a - np.log(np.diag(Lc)).sum()
                except np.linalg.LinAlgError:
                    continue
                if best is None or ll > best[0]:
                    best = (ll, ls, noise, Lc, a)
        _, self.ls, self.noise, self.L, self.a = best
        return self

    @staticmethod
    def _k(A, B, ls):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / ls ** 2)

    def predict(self, Xs):
        Ks = self._k(np.asarray(Xs, float), self.X, self.ls)
        m = Ks @ self.a
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(1.0 - (v ** 2).sum(0), 1e-12)
        return m * self.sd + self.mu, np.sqrt(var) * self.sd


def _norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


def _norm_pdf(z):
    return np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)


def optimize(evaluate, variables, objective, constraints=(), n_init=None, n_iter=20, seed=0, log=print,
             progress=None, cancel=None):
    """戻り値: {"best": 行, "history": [行...]}  行 = {"番号", "変数", "出力", "目的", "制約を満たす", "時間 [s]"}"""
    rng = np.random.default_rng(seed)
    d = len(variables)
    lo = np.array([float(v["min"]) for v in variables])
    hi = np.array([float(v["max"]) for v in variables])
    names = [v["name"] for v in variables]
    sign = 1.0 if objective.get("goal", "min") == "min" else -1.0
    n_init = n_init or max(2 * d + 1, 5)
    hist = []

    def run(u):
        x = lo + u * (hi - lo)
        vals = {n: float(v) for n, v in zip(names, x)}
        t0 = time.time()
        row = {"番号": len(hist) + 1, "変数": vals, "出力": {}, "目的": np.nan, "制約を満たす": False}
        try:
            out = evaluate(vals)
            row["出力"] = out
            row["目的"] = float(out[objective["output"]])
            ok = True
            for c in constraints:
                v = float(out[c["output"]])
                if "max" in c and v > float(c["max"]):
                    ok = False
                if "min" in c and v < float(c["min"]):
                    ok = False
            row["制約を満たす"] = ok
            row["状態"] = "完了"
        except Exception as e:  # noqa: BLE001
            row["状態"] = f"エラー: {e}"
        row["時間 [s]"] = round(time.time() - t0, 2)
        hist.append(row)
        log(f"  試行 {row['番号']}: " + ", ".join(f"{k} = {v:.4g}" for k, v in vals.items())
            + (f" → {objective['output']} = {row['目的']:.5g}" + ("" if row["制約を満たす"] else "（制約違反）")
               if row.get("状態") == "完了" else f" → {row['状態']}"))
        return row

    U = []
    total = n_init + n_iter
    for u in _lhs(n_init, d, rng):
        if cancel is not None and cancel():
            break
        U.append(u)
        run(u)
        if progress:
            progress(len(hist), total)
    for it in range(n_iter):
        if cancel is not None and cancel():
            break
        ok = [i for i, r in enumerate(hist) if r.get("状態") == "完了"]
        if len(ok) < 2:
            u = rng.random(d)
        else:
            X = np.array([U[i] for i in ok])
            y = np.array([sign * hist[i]["目的"] for i in ok])
            gp = GP().fit(X, y)
            cgps = []
            for c in constraints:
                cv = np.array([float(hist[i]["出力"][c["output"]]) for i in ok])
                cgps.append((c, GP().fit(X, cv)))
            feas = [i for i in ok if hist[i]["制約を満たす"]]
            fbest = min(sign * hist[i]["目的"] for i in feas) if feas else None
            cand = np.vstack([rng.random((4000, d)),
                              np.clip(X[np.argmin(y)] + 0.05 * rng.standard_normal((1000, d)), 0, 1)])
            m, s = gp.predict(cand)
            if fbest is None:
                ei = np.ones(len(cand))
            else:
                z = (fbest - m) / s
                ei = (fbest - m) * _norm_cdf(z) + s * _norm_pdf(z)
            pf = np.ones(len(cand))
            for c, g in cgps:
                cm, cs = g.predict(cand)
                if "max" in c:
                    pf *= _norm_cdf((float(c["max"]) - cm) / cs)
                if "min" in c:
                    pf *= _norm_cdf((cm - float(c["min"])) / cs)
            score = ei * pf if fbest is not None else pf * s
            u = cand[int(np.argmax(score))]
        U.append(u)
        run(u)
        if progress:
            progress(len(hist), total)
    feas = [r for r in hist if r.get("状態") == "完了" and r["制約を満たす"]]
    best = min(feas, key=lambda r: sign * r["目的"]) if feas else None
    if best:
        log("最適解: " + ", ".join(f"{k} = {v:.5g}" for k, v in best["変数"].items())
            + f" → {objective['output']} = {best['目的']:.5g}")
    else:
        log("制約を満たす解が見つかりませんでした")
    return {"best": best, "history": hist}


def optimize_project(project, variables, objective, constraints=(), n_init=None, n_iter=20, log=print,
                     progress=None, cancel=None, seed=0):
    """プロジェクトの変数を最適化（各試行でメッシュ→解析→出力の評価。出力の名前は parametric と同じ）"""
    from .parametric import evaluate_outputs, output_label
    outs = [{"field": objective["field"], "stat": objective.get("stat", "max")}]
    for c in constraints:
        outs.append({"field": c["field"], "stat": c.get("stat", "max")})
    obj = dict(objective, output=output_label(outs[0]))
    cons = [dict(c, output=output_label(o)) for c, o in zip(constraints, outs[1:])]

    def evaluate(vals):
        p = project.copy()
        p.apply_params(vals)
        mesh = p.mesh(log=lambda *a: None)
        res = p.run_all(mesh, log=lambda *a: None)
        return evaluate_outputs(p, mesh, res, outs)

    return optimize(evaluate, variables, obj, cons, n_init, n_iter, seed=seed, log=log, progress=progress,
                    cancel=cancel)

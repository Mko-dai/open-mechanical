"""周期対称のモーダル解析（Ansys の Cyclic Symmetry → Modal と同じ考え方）

1 セクタだけのモデルで、全周（N セクタ）の固有振動数を求める。 全周のモードは「調和指数 k」（節直径の数。
0 〜 N/2）ごとに分かれ、切断面の変位は

    u_high = e^{i k α} R(α) u_low        （α = 360°/N）

の関係になる（k = 0 はすべてのセクタが同じ動き、k = N/2 は隣どうしが逆向き、それ以外は周方向に進む波）。
この拘束を複素数の多点拘束として消去し、エルミート行列の固有値問題を解く。 0 < k < N/2 のモードは
全周モデルでは同じ振動数が 2 つずつ（向きの違う波）出るが、ここでは 1 つとして返す。
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .modal import ModalResult, _materials, _supports_only
from .solver import BC, assemble_stiffness, build_load_and_constraints


def solve_cyclic_modal(mesh, material, bcs: list[BC], n_modes: int = 6, harmonics="all", elem_materials=None,
                       log=print, progress=None) -> ModalResult:
    """調和指数ごとに n_modes 個ずつ求め、振動数の低い順に並べて返す（mr.harmonics に各モードの調和指数）。
    harmonics: "all"（0 〜 N/2）または 整数のリスト"""
    from .dynamics import consistent_mass
    from .remote import REMOTE_KINDS, extra_terms
    t0 = time.time()
    cyc = [b for b in bcs if b.kind == "cyclic"]
    if len(cyc) != 1:
        raise ValueError("周期対称のモーダルには、周期対称の条件を 1 つだけ設定してください")
    if any(b.kind in REMOTE_KINDS and b.kind != "cyclic" for b in bcs):
        raise ValueError("周期対称のモーダルは、リモート境界条件・ばね・ジョイントとはまだ組み合わせられません")
    nsec = int((cyc[0].options or {}).get("sectors", 0) or 0)
    alpha = 2 * np.pi / nsec
    mats, elem_mat, D, rho_e = _materials(mesh, material, elem_materials)
    K = assemble_stiffness(mesh, D, elem_mat=elem_mat).tocsr()
    M = consistent_mass(mesh, rho_e).tocsr()
    n = K.shape[0]
    _, fixed = build_load_and_constraints(mesh, mats[0], _supports_only([b for b in bcs if b.kind != "cyclic"]))
    ex = extra_terms(mesh, cyc, float(np.mean(np.abs(K.diagonal()))))
    for note in ex.notes:
        log(note)
    slaves = np.array(sorted(ex.mpc), dtype=np.int64)
    is_slave = np.zeros(n, bool)
    is_slave[slaves] = True
    keep = np.nonzero(~is_slave)[0]
    newidx = -np.ones(n, dtype=np.int64)
    newidx[keep] = np.arange(len(keep))
    axis_nodes = sorted({int(d) // 3 for d in ex.fixed})        # 軸上の節点（低い側と高い側の両方に入る）
    a = np.asarray((cyc[0].options or {}).get("axis", [0.0, 0.0, 1.0]), float)
    ax = int(np.argmax(np.abs(a)))
    ks = list(range(nsec // 2 + 1)) if harmonics in ("all", None) else sorted({int(k) % nsec for k in harmonics})
    ks = [k if k <= nsec // 2 else nsec - k for k in ks]
    lam_ref = float(K.diagonal().mean() / max(M.diagonal().mean(), 1e-300))
    found = []
    for kk, k in enumerate(sorted(set(ks))):
        ph = np.exp(1j * k * alpha)
        rows, cols, vals = [keep], [np.arange(len(keep))], [np.ones(len(keep), complex)]
        for sd in slaves:
            m_, c_ = ex.mpc[int(sd)]
            rows.append(np.full(len(m_), sd))
            cols.append(newidx[m_])
            vals.append(ph * c_)
        T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(keep)))
        fx = set(int(d) for d in fixed)
        for nd in axis_nodes:
            # 軸上の節点: k = 0 は軸方向だけ動ける、k = 1 は軸に垂直な方向だけ、k ≥ 2 は動けない
            for c in range(3):
                if (k == 0 and c != ax) or (k == 1 and c == ax) or k >= 2:
                    fx.add(3 * nd + c)
        free = np.ones(len(keep), bool)
        fr = newidx[np.fromiter(fx, dtype=np.int64)] if fx else np.zeros(0, np.int64)
        free[fr[fr >= 0]] = False
        Tf = T[:, np.nonzero(free)[0]]
        Kr = (Tf.conj().T @ K @ Tf).tocsc()
        Mr = (Tf.conj().T @ M @ Tf).tocsc()
        if k == 0 or 2 * k == nsec:
            Kr, Mr = Kr.real.tocsc(), Mr.real.tocsc()
        nm = min(int(n_modes), Kr.shape[0] - 2)
        w2, X = spla.eigsh(Kr, k=nm, M=Mr, sigma=-1e-6 * lam_ref, which="LM")
        o = np.argsort(w2)
        for j in o:
            u = np.asarray(Tf @ X[:, j]).ravel()
            u = u * np.exp(-1j * np.angle(u[np.argmax(np.abs(u))]))      # 一番大きい成分が実数になる向きに回す
            found.append((float(np.sqrt(max(w2[j], 0.0)) / (2 * np.pi)), k, (u.real / np.abs(u).max())))
        log(f"  調和指数 {k}: " + ", ".join(f"{np.sqrt(max(v, 0)) / (2 * np.pi):.5g}" for v in np.sort(w2)[:4]) + " Hz …")
        if progress:
            progress((kk + 1) / len(set(ks)), f"調和指数 {k}")
    found.sort(key=lambda r: r[0])
    freqs = np.array([f[0] for f in found])
    modes = np.array([f[2].reshape(-1, 3) for f in found])
    z = np.zeros((len(found), 6))
    info = {"解析": "モーダル（周期対称）", "セクタ数": nsec, "節点数": mesh.n_nodes, "要素数": mesh.n_elems,
            "モード数": len(found), "合計時間 [s]": round(time.time() - t0, 2)}
    mr = ModalResult(mesh, freqs, modes, z, z.copy(), float((rho_e * np.abs(mesh.element_volumes())).sum()), info)
    mr.harmonics = np.array([f[1] for f in found], dtype=int)
    return mr

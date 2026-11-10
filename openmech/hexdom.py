"""六面体優勢メッシュ（Ansys の「六面体ドミナント」と同じ考え方）。

形の内側に立方体の格子（六面体）を敷き、表面との間のすき間だけを四面体で埋める。
六面体と四面体のあいだは、格子の外に出ている四角形の面ごとにピラミッドを 1 つ外向きに載せてつなぐ
（頂点は、となりの空いているセルの中心。 へこんだ角では同じ頂点を共有するので、面はすべてきちんと合う）。

要素はすべて「縮退した六面体」（elements.DEGEN_*）として 1 つの配列に入れる（SOLID186 と同じ扱い）。
"""
from __future__ import annotations

import numpy as np

# 立方体の 6 面（頂点の番号は VTK の六面体）。 右ねじの向きがセルの内側（ピラミッドの頂点 = セルの中心の側）
_IN_FACES = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
_NEIGH = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
_CORNER = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def points_inside(P, X, tris):
    """点 P が、三角形で閉じた表面（節点 X、三角形 tris）の内側にあるか。 z 方向の半直線と交わる回数で数える。
    P は (x, y) が同じ点をまとめて調べるので、格子の点に向く"""
    P = np.asarray(P, float)
    A, B, C = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    xy, inv = np.unique(P[:, :2], axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = np.zeros(len(P), bool)
    lo2, hi2 = np.minimum(np.minimum(A, B), C)[:, :2], np.maximum(np.maximum(A, B), C)[:, :2]
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(xy) + 1))
    for k, (x, y) in enumerate(xy):
        idx = order[starts[k]:starts[k + 1]]
        cand = np.nonzero((lo2[:, 0] <= x) & (hi2[:, 0] >= x) & (lo2[:, 1] <= y) & (hi2[:, 1] >= y))[0]
        if not len(cand):
            continue
        a, b, c = A[cand], B[cand], C[cand]
        d = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        ok = np.abs(d) > 1e-300
        dd = np.where(ok, d, 1.0)
        s = ((x - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (y - a[:, 1])) / dd
        t = ((b[:, 0] - a[:, 0]) * (y - a[:, 1]) - (x - a[:, 0]) * (b[:, 1] - a[:, 1])) / dd
        hit = ok & (s >= 0) & (t >= 0) & (s + t <= 1)
        zc = (a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2]))[hit]
        out[idx] = (np.sum(zc[None, :] > P[idx, 2][:, None], axis=1) % 2) == 1
    return out


def clean_cells(keep):
    """六面体の集まりの表面が「きれいな面」になるよう、辺だけ・頂点だけで触れ合っているセルを取り除く"""
    keep = keep.copy()
    for _ in range(50):
        K = np.pad(keep, 1)
        bad = np.zeros_like(K)
        # 辺を共有する 4 セルのうち、対角の 2 つだけが残っている
        for ax in range(3):
            a, b = [i for i in range(3) if i != ax]

            def sh(da, db):
                sl = [slice(None)] * 3
                sl[a] = slice(da, K.shape[a] - 1 + da)
                sl[b] = slice(db, K.shape[b] - 1 + db)
                return tuple(sl)
            c00, c10, c01, c11 = K[sh(0, 0)], K[sh(1, 0)], K[sh(0, 1)], K[sh(1, 1)]
            d1 = c00 & c11 & ~c10 & ~c01
            d2 = c10 & c01 & ~c00 & ~c11
            bad[sh(1, 1)] |= d1
            bad[sh(1, 0)] |= d2
        # 頂点を共有する 8 セルのうち、面でつながっていないものがある（対角だけで触れている）
        n = np.array(K.shape) - 1
        blk = np.stack([K[i:n[0] + i, j:n[1] + j, k:n[2] + k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], -1)
        cnt = blk.sum(-1)
        for v in np.argwhere((cnt >= 2) & (cnt <= 6)):
            cells = blk[tuple(v)].reshape(2, 2, 2)
            pts = np.argwhere(cells)
            comp = {tuple(pts[0])}
            grew = True
            while grew:
                grew = False
                for p in pts:
                    if tuple(p) not in comp and any(np.abs(p - np.array(q)).sum() == 1 for q in comp):
                        comp.add(tuple(p))
                        grew = True
            if len(comp) < len(pts):
                for p in pts:
                    if tuple(p) not in comp:
                        bad[tuple(v + p)] = True
        # 空いているセルで、向かい合う 2 面だけが六面体に触れている（ピラミッドが頂点だけで触れ合う）
        Kp = np.pad(K, 1)
        c = (slice(1, -1),) * 3
        nbs = [Kp[2:, 1:-1, 1:-1], Kp[:-2, 1:-1, 1:-1], Kp[1:-1, 2:, 1:-1], Kp[1:-1, :-2, 1:-1],
               Kp[1:-1, 1:-1, 2:], Kp[1:-1, 1:-1, :-2]]
        tot = sum(x.astype(int) for x in nbs)
        for ax in range(3):
            slot = ~K & nbs[2 * ax] & nbs[2 * ax + 1] & (tot == 2)
            sl = [slice(None)] * 3
            sl[ax] = slice(1, None)
            dst = [slice(None)] * 3
            dst[ax] = slice(None, -1)
            bad[tuple(sl)] |= slot[tuple(dst)]                 # ＋側のとなりの六面体を外す
        bad = bad[1:-1, 1:-1, 1:-1] & keep
        if not bad.any():
            break
        keep &= ~bad
    return keep


def _surface_distance(P, X, tris, tree=None):
    """点 P から表面の三角形までの距離（近い三角形 12 個を調べる）"""
    from scipy.spatial import cKDTree
    from .contact import _project
    tp = X[tris]
    if tree is None:
        tree = cKDTree(tp.mean(axis=1))
    return _project(np.asarray(P, float), tp, tree, min(12, len(tris)))[2], tree


def _inside(P, X, tris, h):
    """表面から離れた点の内外。 半直線が三角形の辺をちょうど通らないよう、ほんの少しずらして調べる"""
    return points_inside(np.asarray(P, float) + np.array([3.71e-4, 4.13e-4, 0.0]) * h, X, tris)


def core_cells(X, tris, h, margin=0.85):
    """表面（節点 X・三角形 tris）の内側で、表面から margin × h より離れている格子のセル。
    戻り値: (keep (nx,ny,nz), xs, ys, zs = 格子の座標)"""
    lo, hi = X.min(0), X.max(0)
    axes = []
    for i in range(3):       # 格子は外形の真ん中にそろえる（両側のすき間が同じになる）
        L = hi[i] - lo[i]
        n = int(np.floor((L - 2 * margin * h) / h + 1e-9))
        if n < 1:
            return np.zeros((0, 0, 0), bool), np.zeros(0), np.zeros(0), np.zeros(0)
        axes.append(lo[i] + (L - n * h) / 2 + h * np.arange(n + 1))
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    dist, tree = _surface_distance(G, X, tris)
    far = np.abs(dist) > margin * h * (1 - 1e-9)
    ok = np.zeros(len(G), bool)
    if far.any():
        ok[far] = _inside(G[far], X, tris, h)
    ok = ok.reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    keep = np.ones(tuple(n - 1 for n in ok.shape), bool)
    for i, j, k in _CORNER:
        keep &= ok[i:ok.shape[0] - 1 + i, j:ok.shape[1] - 1 + j, k:ok.shape[2] - 1 + k]
    keep = clean_cells(keep)
    # ピラミッドの頂点（となりの空きセルの中心）が形の内側にあって、表面に近すぎないこと。 だめなら六面体を外す
    org = np.array([axes[0][0], axes[1][0], axes[2][0]])
    for _ in range(20):
        if not keep.any():
            break
        cells = np.argwhere(keep)
        K = np.pad(keep, 1)
        drop = np.zeros(len(cells), bool)
        for d in _NEIGH:
            ex = ~K[cells[:, 0] + 1 + d[0], cells[:, 1] + 1 + d[1], cells[:, 2] + 1 + d[2]]
            if not ex.any():
                continue
            ap = org + (cells[ex] + d + 0.5) * h
            okp = (np.abs(_surface_distance(ap, X, tris, tree)[0]) > 0.3 * h) & _inside(ap, X, tris, h)
            drop[np.nonzero(ex)[0][~okp]] = True
        if not drop.any():
            break
        keep[tuple(cells[drop].T)] = False
        keep = clean_cells(keep)
    return keep, *axes


def build_core(keep, xs, ys, zs):
    """セルの集まり → 節点・要素（縮退した六面体 8 列）・すき間との境の三角形。
    戻り値: (nodes, elems8, surf_tris, n_hex)。 surf_tris は nodes の番号"""
    nx, ny, nz = keep.shape
    h = float(xs[1] - xs[0])
    vid = -np.ones((nx + 1, ny + 1, nz + 1), np.int64)
    cells = np.argwhere(keep)
    cv = cells[:, None, :] + _CORNER[None]                      # (C,8,3)
    used = np.zeros(vid.shape, bool)
    used[cv[..., 0], cv[..., 1], cv[..., 2]] = True
    vid[used] = np.arange(int(used.sum()))
    ijk = np.argwhere(used)
    nodes = np.column_stack([xs[ijk[:, 0]], ys[ijk[:, 1]], zs[ijk[:, 2]]])
    conn = vid[cv[..., 0], cv[..., 1], cv[..., 2]]              # (C,8)
    K = np.pad(keep, 1)
    org = np.array([xs[0], ys[0], zs[0]])
    quads, apc = [], []
    for f, d in enumerate(_NEIGH):
        ex = ~K[cells[:, 0] + 1 + d[0], cells[:, 1] + 1 + d[1], cells[:, 2] + 1 + d[2]]
        if ex.any():
            quads.append(conn[ex][:, _IN_FACES[f][::-1]])      # 右ねじの向きが外（ピラミッドの頂点の側）
            apc.append(cells[ex] + d)
    quads, apc = np.vstack(quads), np.vstack(apc)
    ua, ai = np.unique(apc, axis=0, return_inverse=True)
    ap = len(nodes) + ai.reshape(-1)
    nodes = np.vstack([nodes, org + (ua + 0.5) * h])
    pyr = np.column_stack([quads, ap, ap, ap, ap])
    # ピラミッドの側面: 2 つのピラミッドが共有する面（へこんだ角）は中の面。 1 回だけ出る面がすき間との境
    side = np.vstack([np.column_stack([quads[:, k], quads[:, (k + 1) % 4], ap]) for k in range(4)])
    key = np.sort(side, axis=1)
    _, first, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return nodes, np.vstack([conn, pyr]), side[first[cnt == 1]], len(conn)


def to_quadratic(nodes, elems8, curved=None):
    """縮退した六面体（8 列）→ 20 列。 つぶれた辺の中間節点は、その節点自身にする。
    curved: {(小さい番号, 大きい番号): 中間節点の座標}（曲面の上の辺）"""
    from .elements import HEX20_EDGES
    E = np.array(HEX20_EDGES)
    pa, pb = elems8[:, E[:, 0]], elems8[:, E[:, 1]]             # (N,12)
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    real = lo != hi
    key = lo[real] * (int(elems8.max()) + 1) + hi[real]
    uk, inv = np.unique(key, return_inverse=True)
    n0 = len(nodes)
    a, b = uk // (int(elems8.max()) + 1), uk % (int(elems8.max()) + 1)
    mids = 0.5 * (nodes[a] + nodes[b])
    if curved:
        pos = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(a, b))}
        for kk, xyz in curved.items():
            k = pos.get(kk)
            if k is not None:
                mids[k] = xyz
    mid = lo.copy()
    mid[real] = n0 + inv.reshape(-1)
    return np.vstack([nodes, mids]), np.hstack([elems8, mid])


def gap_tets(gmsh, outer_X, outer_tris_by_surf, core_X, core_tris, h):
    """表面（CAD の面ごとの三角形）と六面体の集まりの表面にはさまれたすき間を、四面体で埋める。
    戻り値: (節点 (M,3)、四面体 (T,4))。 節点の最初の len(outer_X) 個は outer_X、その次の len(core_X) 個は core_X"""
    cur = gmsh.model.getCurrent()
    alg0 = gmsh.option.getNumber("Mesh.Algorithm3D")
    gmsh.model.add("hexdom_gap")
    try:
        no, nc = len(outer_X), len(core_X)
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, np.arange(1, no + nc + 1), np.vstack([outer_X, core_X]).ravel())
        tags = []
        for i, tr in enumerate(outer_tris_by_surf):
            if not len(tr):
                continue
            t = i + 1
            if t != 1:
                gmsh.model.addDiscreteEntity(2, t)
            gmsh.model.mesh.addElementsByType(t, 2, [], (np.asarray(tr) + 1).ravel())
            tags.append(t)
        if 1 not in tags:
            tags.append(1)
        tc = len(outer_tris_by_surf) + 1
        gmsh.model.addDiscreteEntity(2, tc)
        gmsh.model.mesh.addElementsByType(tc, 2, [], (np.asarray(core_tris) + no + 1).ravel())
        l1 = gmsh.model.geo.addSurfaceLoop(tags)
        l2 = gmsh.model.geo.addSurfaceLoop([tc])
        gmsh.model.geo.addVolume([l1, l2])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        nt, xyz, _ = gmsh.model.mesh.getNodes()
        nt = nt.astype(np.int64)
        X = np.zeros((int(nt.max()), 3))
        X[nt - 1] = xyz.reshape(-1, 3)
        _, en = gmsh.model.mesh.getElementsByType(4)
        tets = en.astype(np.int64).reshape(-1, 4) - 1
        if not len(tets):
            raise RuntimeError("すき間の四面体を作れませんでした")
        if not (np.allclose(X[:no], outer_X) and np.allclose(X[no:no + nc], core_X)):
            raise RuntimeError("すき間のメッシュで表面の節点が動きました")
        # 使われている節点だけ（表面・六面体側の節点はそのまま先頭に残す）
        usedm = np.zeros(len(X), bool)
        usedm[:no + nc] = True
        usedm[np.unique(tets)] = True
        remap = -np.ones(len(X), np.int64)
        remap[usedm] = np.arange(int(usedm.sum()))
        return X[usedm], remap[tets]
    finally:
        gmsh.model.remove()
        gmsh.model.setCurrent(cur)
        gmsh.option.setNumber("Mesh.Algorithm3D", alg0)


def assemble(outer_X, core_nodes, core_elems, core_tris, gapX, tets):
    """六面体の集まり（build_core）とすき間の四面体（gap_tets）を 1 つの節点番号にまとめる。
    戻り値: (nodes, elems8)。 節点の最初の len(outer_X) 個は表面の節点"""
    no = len(outer_X)
    on = np.unique(core_tris)                                   # 六面体側で、すき間と共有する節点
    cmap = -np.ones(len(core_nodes), np.int64)
    cmap[on] = no + np.arange(len(on))
    rest = np.nonzero(cmap < 0)[0]
    cmap[rest] = len(gapX) + np.arange(len(rest))
    nodes = np.vstack([gapX, core_nodes[rest]])
    te = tets[:, [0, 1, 2, 2, 3, 3, 3, 3]]
    return nodes, np.vstack([cmap[core_elems], te]), on


def gmsh_hexdom(gmsh, h, order, log=print):
    """いまの gmsh のモデル（ボディ 1 つ）を六面体優勢で切る。 戻り値: Mesh（面番号つき）。 六面体を置けなければ None"""
    from .mesh import Mesh
    gmsh.model.mesh.generate(2)
    if order == 2:
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.setOrder(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    t2i = np.full(int(tags.max()) + 1, -1, np.int64)
    t2i[tags.astype(np.int64)] = np.arange(len(tags))
    surfs = sorted(gmsh.model.getEntities(2), key=lambda e: e[1])
    tri_by, mid_by = [], []
    for _, st in surfs:
        tr, md = np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64)
        ets, _, ens = gmsh.model.mesh.getElements(2, st)
        for t, en in zip(ets, ens):
            if int(t) == 2:
                tr = t2i[en.astype(np.int64)].reshape(-1, 3)
            elif int(t) == 9:
                e6 = t2i[en.astype(np.int64)].reshape(-1, 6)
                tr, md = e6[:, :3], e6[:, 3:]
            elif len(en):
                raise ValueError("六面体優勢メッシュ: 表面を三角形で切れませんでした")
        tri_by.append(tr)
        mid_by.append(md)
    alltri = np.vstack(tri_by)
    cu = np.unique(alltri)
    om = -np.ones(len(coords), np.int64)
    om[cu] = np.arange(len(cu))
    outer_X = coords[cu]
    tri_by_l = [om[t] for t in tri_by]
    curved = {}
    for tr, md in zip(tri_by_l, mid_by):
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            if len(md):
                for i, j, m in zip(tr[:, a], tr[:, b], md[:, k]):
                    curved[(min(int(i), int(j)), max(int(i), int(j)))] = coords[m]
    otri = om[alltri]
    keep, xs, ys, zs = core_cells(outer_X, otri, h)
    if not keep.any():
        log("六面体優勢: 六面体を置ける場所がありません（形が薄い・要素サイズが大きい）。四面体で切ります")
        return None
    cn, ce, ct, n_hex = build_core(keep, xs, ys, zs)
    on_nodes = np.unique(ct)
    cl = -np.ones(len(cn), np.int64)
    cl[on_nodes] = np.arange(len(on_nodes))
    gapX, tets = gap_tets(gmsh, outer_X, tri_by_l, cn[on_nodes], cl[ct], h)
    nodes, e8, _ = assemble(outer_X, cn, ce, ct, gapX, tets)
    n_cell = int(keep.sum())
    log(f"六面体優勢: 六面体 {n_hex:,}・つなぎのピラミッド {len(ce) - n_hex:,}・すき間の四面体 {len(tets):,} 個"
        f"（六面体の体積は全体の {n_cell * h ** 3 / max(_tri_volume(outer_X, otri), 1e-300):.0%}）")
    if order == 2:
        nodes, elems = to_quadratic(nodes, e8, curved)
    else:
        elems = e8
    mesh = Mesh(nodes, elems)
    mesh.elem_body = np.zeros(len(elems), np.int64)
    mesh = mesh.split_body_interfaces(None)
    mesh.hexdom_counts = {"六面体": n_hex, "ピラミッド": len(ce) - n_hex, "四面体": len(tets)}
    # 面番号: CAD の面ごとの三角形（頂点 3 つ）と、境界の四角形（つぶれて三角形になっている）の頂点を突き合わせる
    key_of = {}
    for fid, tr in enumerate(tri_by_l):
        for k in map(tuple, np.sort(tr, axis=1)):
            key_of[k] = fid
    bq = mesh.boundary_quads_full
    corners = bq[:, ::2] if bq.shape[1] == 8 else bq
    co = mesh.node_origin[corners]
    qf = np.array([key_of.get(tuple(sorted(set(r.tolist()))), -1) for r in co])
    fids = qf[mesh.boundary_quad_of]
    if (fids < 0).any():
        mesh.group_faces()
    else:
        from .geometry import _surface_face_ids
        _surface_face_ids(mesh, fids, len(surfs))
    return mesh


def _tri_volume(X, tris):
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)

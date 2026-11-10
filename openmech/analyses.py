"""解析の種類ごとの実行（Project.run_all から呼ぶ）。どの解析も GUI で表示できる Result のリストを返す。

analysis_type:
  modal       モーダル          opts: n_modes, f_min, prestress（静解析の応力を含める）
  buckling    固有値座屈        opts: n_buckling
  thermal     定常熱 / 過渡熱   opts: thermal = ThermalSettings の内容
  harmonic    調和応答          opts: n_modes, f_min_h, f_max_h, n_freq, zeta, base_acc（基礎加振の加速度振幅 [G] 3 成分 or 空）
  random      ランダム振動      opts: n_modes, psd（[[Hz, G²/Hz], ...]）, psd_dir, zeta
  spectrum    応答スペクトル    opts: n_modes, spectrum（[[Hz, 加速度], ...]）, spec_unit（"G" / "mm/s2"）, spec_dir,
                                spec_method（"srss" / "cqc"）, zeta
  electrostatic 静電場
  piezo       圧電（静解析）
static でボルト予張力があるとき、または opts.thermal_import（熱解析の温度で熱応力）のときもここを通る。
"""
from __future__ import annotations

import numpy as np

from .solver import Result, material_D, nodal_stress, solve

EXTRA_TYPES = ("modal", "buckling", "thermal", "harmonic", "random", "spectrum", "electrostatic", "piezo")
ANALYSIS_JP = {"static": "静的構造", "modal": "モーダル", "buckling": "固有値座屈", "thermal": "熱",
               "harmonic": "調和応答", "random": "ランダム振動", "spectrum": "応答スペクトル", "electrostatic": "静電場",
               "piezo": "圧電",
               "transient": "過渡構造"}
G = 9806.65


def _field_result(mesh, material, extra: dict, info: dict, t=1.0, label=None) -> Result:
    z = np.zeros((mesh.n_nodes, 6))
    r = Result(mesh, material, np.zeros((mesh.n_nodes, 3)), z, z, {}, dict(info))
    r.extra = extra
    r.only_extra = True
    r.time = t
    if label:
        r.label = label
    return r


def _mode_results(mesh, material, mr, em_pair):
    """モード形 → Result（変形はモード形、応力は同じ倍率の相対値）"""
    if em_pair is not None:
        mats, em = em_pair
        D = np.stack([material_D(m) for m in mats])
    else:
        D, em = material_D(material), None
    out = []
    table = mr.table()
    for i in range(mr.n_modes):
        u = mr.modes[i]
        st, sn = nodal_stress(mesh, u.ravel(), D, elem_mat=em)
        r = Result(mesh, material, u, st, sn, {}, dict(mr.info))
        r.time = float(i + 1)
        if mr.kind == "buckling":
            r.info["座屈荷重係数"] = float(mr.load_factors[i])
            r.label = f"モード {i + 1}（荷重係数 {mr.load_factors[i]:.5g}）"
        else:
            r.info["振動数 [Hz]"] = float(mr.freqs[i])
            r.label = f"モード {i + 1}（{mr.freqs[i]:.5g} Hz）"
            if getattr(mr, "harmonics", None) is not None:
                r.info["調和指数"] = int(mr.harmonics[i])
                r.label = f"モード {i + 1}（{mr.freqs[i]:.5g} Hz、調和指数 {int(mr.harmonics[i])}）"
        r.modal_table = table
        r.mode_shape = True          # 変形・応力は相対値（大きさに意味はない）
        out.append(r)
    return out


def run_extra(project, mesh, em, log=print, progress=None):
    p = project
    o = dict(p.analysis_opts or {})
    t = p.analysis_type
    mat = p.material
    if t == "modal" and any(b.kind == "cyclic" for b in p.bcs) and o.get("cyclic_harmonics", "all") != "0":
        # 周期対称: 調和指数ごとに解く（"all" = 0〜N/2、または "0,1,2" のような指定）
        from .cyclic import solve_cyclic_modal
        p.require_connected(mesh, "周期対称のモーダル")
        h = o.get("cyclic_harmonics", "all")
        hs = "all" if h in ("all", "", None) else [int(x) for x in str(h).replace("、", ",").split(",") if x.strip()]
        mr = solve_cyclic_modal(mesh, mat, p.bcs, n_modes=int(o.get("n_modes", 6)), harmonics=hs, elem_materials=em,
                                log=log, progress=progress)
        return _mode_results(mesh, mat, mr, em)
    Kc = p.linear_contact_K(mesh, em, log) if t in ("modal", "buckling", "harmonic", "random", "spectrum") else None
    if t == "modal":
        from .modal import solve_modal
        Ks = None
        if o.get("prestress"):
            from .modal import gauss_stress_fn, stress_stiffness
            st = p.static_result(mesh, em, log)
            mats, eidx = em if em is not None else ([mat], np.zeros(mesh.n_elems, int))
            D = np.stack([material_D(m) for m in mats])
            Ks = stress_stiffness(mesh, gauss_stress_fn(mesh, st.u.ravel(), D, np.asarray(eidx)))
            log("プリストレス付きモーダル: 静解析の応力による剛性を含めます")
        if Kc is not None:
            Ks = Kc if Ks is None else Ks + Kc
        mr = solve_modal(mesh, mat, p.bcs, n_modes=int(o.get("n_modes", 6)), f_min=float(o.get("f_min", 0.0)),
                         elem_materials=em, log=log, progress=progress, K_sigma=Ks)
        return _mode_results(mesh, mat, mr, em)
    if t == "buckling":
        from .modal import solve_buckling
        from .remote import REMOTE_KINDS
        bad = [b.label() for b in p.bcs if b.kind in REMOTE_KINDS or b.kind == "submodel"]
        if bad:
            raise ValueError("固有値座屈では、次の境界条件にまだ対応していません: " + "、".join(bad))
        mr = solve_buckling(mesh, mat, p.bcs, n_modes=int(o.get("n_buckling", 3)), elem_materials=em, log=log,
                            progress=progress, K_add=Kc,
                            static=p.static_result(mesh, em, log) if Kc is not None else None)
        out = [mr.prestress] + _mode_results(mesh, mat, mr, em)
        mr.prestress.label = "静解析（与えた荷重）"
        mr.prestress.time = 0.0
        return out
    if t == "thermal":
        from .thermal import ThermalSettings, solve_thermal
        res = solve_thermal(mesh, mat, p.bcs, ThermalSettings.from_dict(o.get("thermal", {})), elem_materials=em,
                            log=log, progress=progress, contacts=p.ensure_contacts(mesh),
                            contact_tolerance=p.contact_tolerance)
        out = []
        for r in res:
            fr = _field_result(mesh, mat, r.fields(), r.info, r.time,
                               f"時間 {r.time:g} s" if o.get("thermal", {}).get("transient") else "定常")
            fr.thermal = r
            fr.reactions_heat = r.reactions
            out.append(fr)
        p._last_temperature = res[-1].T          # 熱応力に使う
        return out
    if t in ("harmonic", "random", "spectrum"):
        from .modal import solve_modal
        from .vibration import solve_harmonic, solve_random, solve_spectrum
        nm = int(o.get("n_modes", 20))
        mr = solve_modal(mesh, mat, p.bcs, n_modes=nm, elem_materials=em, log=log, K_sigma=Kc)
        if t == "harmonic":
            ba = o.get("base_acc")
            base = np.asarray(ba, float) * G if ba and any(ba) else None
            h = solve_harmonic(mr, mesh, mat, p.bcs, float(o.get("f_min_h", 1.0)), float(o.get("f_max_h", 1000.0)),
                               int(o.get("n_freq", 100)), zeta=float(o.get("zeta", 0.02)), base_acc=base, log=log)
            amp = h.max_amplitude()
            # 表示する周波数: 応答のピーク（大きい順に最大 8 個）＋指定の周波数
            pk = [k for k in range(1, len(amp) - 1) if amp[k] >= amp[k - 1] and amp[k] >= amp[k + 1]]
            pk = sorted(sorted(pk, key=lambda k: -amp[k])[:8])
            fs = [float(h.freqs[k]) for k in pk] + [float(x) for x in o.get("show_freqs", [])]
            if not fs:
                fs = [float(h.freqs[int(np.argmax(amp))])]
            mats_D = (np.stack([material_D(m) for m in em[0]]), np.asarray(em[1])) if em is not None else (None, None)
            out = []
            for f in sorted(set(fs)):
                r = h.at(f, mat, D=mats_D[0], elem_mat=mats_D[1])
                r.label = f"{r.info['周波数 [Hz]']:.5g} Hz"
                r.frf = (h.freqs, amp)
                r.harmonic = h
                out.append(r)
            return out
        if t == "spectrum":
            spec = o.get("spectrum") or [[1, 0.5], [10, 1.0], [100, 1.0], [1000, 0.5]]
            r = solve_spectrum(mr, mesh, mat, spec, direction=o.get("spec_dir", [0, 0, 1]),
                               unit=o.get("spec_unit", "G"), method=o.get("spec_method", "srss"),
                               zeta=float(o.get("zeta", 0.05)), elem_materials=em, log=log)
            r.label = r.info["組み合わせ"]
            return [r]
        psd = o.get("psd") or [[20, 0.01], [2000, 0.01]]
        r = solve_random(mr, mesh, mat, psd, direction=o.get("psd_dir", [0, 0, 1]), zeta=float(o.get("zeta", 0.02)),
                         elem_materials=em, log=log)
        r.label = "1σ"
        return [r]
    if t == "electrostatic":
        from .electric import solve_electrostatic
        r = solve_electrostatic(mesh, mat, p.bcs, elem_materials=em, log=log, contacts=p.ensure_contacts(mesh),
                                contact_tolerance=p.contact_tolerance)
        return [_field_result(mesh, mat, r.fields(), r.info)]
    if t == "piezo":
        from .electric import solve_piezo_static
        p.require_connected(mesh, "圧電解析")
        r, _ = solve_piezo_static(mesh, mat, p.bcs, elem_materials=em, log=log)
        return [r]
    # 静的構造（ボルト予張力・熱解析の温度の取り込み）
    if o.get("thermal_import"):
        from .thermal import ThermalSettings, solve_thermal, thermal_stress
        th = solve_thermal(mesh, mat, p.bcs, ThermalSettings.from_dict(o.get("thermal", {})), elem_materials=em,
                           log=log, contacts=p.ensure_contacts(mesh), contact_tolerance=p.contact_tolerance)
        T = th[-1].T
        fn = None
        if p.needs_nonlinear_solver(mesh):
            from .nonlinear import solve_static
            p.ensure_contacts(mesh)

            def fn(bcs, emm):
                return solve_static(mesh, mat, bcs, p.contacts, p.settings, elem_materials=emm, log=lambda *a: None,
                                    contact_tolerance=p.contact_tolerance)[-1]
        r = thermal_stress(mesh, mat, p.bcs, T, elem_materials=em, log=log, solve_fn=fn)
        r.extra = {"温度 [°C]": T}
        return [r]
    if any(b.kind == "bolt_pretension" for b in p.bcs):
        from .bolt import has_beam_pretension as _hbp, solve_pretension
        if _hbp(p.bcs):
            raise ValueError("ボルト予張力（ソリッドのボルト）とビーム接続の予張力は、同じ解析では併用できません")
        if p.needs_nonlinear_solver(mesh):
            from .nonlinear import solve_static

            def fn(bcs, emm):
                return solve_static(mesh, mat, bcs, p.contacts, p.settings, elem_materials=emm,
                                    log=lambda *a: None, contact_tolerance=p.contact_tolerance)[-1]
        else:
            def fn(bcs, emm):
                return solve(mesh, mat, bcs, log=lambda *a: None, elem_materials=emm)
        res = solve_pretension(mesh, mat, p.bcs, fn, elem_materials=em, log=log)
        for r in res:
            r.label = r.info.get("解析", "")
        return res
    from .bolt import has_beam_pretension
    if has_beam_pretension(p.bcs):
        from .bolt import solve_beam_pretension
        if p.needs_nonlinear_solver(mesh):
            from .nonlinear import solve_static

            def fnb(bcs):
                return solve_static(mesh, mat, bcs, p.contacts, p.settings, elem_materials=em,
                                    log=lambda *a: None, contact_tolerance=p.contact_tolerance)[-1]
        else:
            def fnb(bcs):
                return solve(mesh, mat, bcs, log=lambda *a: None, elem_materials=em)
        res = solve_beam_pretension(p.bcs, fnb, log=log)
        for r in res:
            r.label = r.info.get("解析", "")
        return res
    return [solve(mesh, mat, p.bcs, log=log, elem_materials=em)]

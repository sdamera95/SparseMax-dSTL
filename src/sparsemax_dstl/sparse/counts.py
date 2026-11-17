"""AD product counts of one control period's program over control periods, for three comparators.

The program of period k has the 42 T variables of the horizon's samples k + 1..k + T (28 states and 14 accelerations each),
the 28 T defect constraints and one constraint per imposed conjunct instance. Dense AD evaluates the whole constraint
Jacobian with min(28 T + m, 42 T) products and the Lagrangian Hessian with 42 T Hessian-vector products. The prescribed
pattern holds every operand of the instances' smoothed operators (what a structural detector returns for log-sum-exp and
sparsemax alike); the supports pattern keeps the operands in the sparsemax supports along the reference motion. A Jacobian's
products are the smaller of asdex's greedy row coloring (reverse products) and column coloring (forward products); a Hessian's
are asdex's greedy star coloring. The Lagrangian Hessian is the union of the rows' cliques and the acceleration cost's band.
"""
import hashlib
import warnings

import asdex
import numpy as np
import scipy.sparse as sps

from ..tasks import dual_arm as D
from .patterns import LEVELS, Reference, columns, control_band, defect_pattern, hessian_union, scene_rows

NX, NU, NQ = D.NX, D.NU, D.NQ
FAMILIES = ("until", "reach")
CAUSES = {-1: "inside", 0: "new instance", 1: "horizon end", 2: "progression", 4: "other", 5: "shift"}


def _colors(r, c, K, N, ncol, mode):
    """Colors per period of K block-diagonal (N, ncol) Jacobian patterns, asdex's row ("rev") or column ("fwd") coloring."""
    sp = asdex.SparsityPattern.from_coo(r.astype(np.int32), c.astype(np.int32), (K * N, K * ncol))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        col = np.asarray(asdex.jacobian_coloring_from_sparsity(sp, mode=mode).colors)
    return col.reshape(K, N if mode == "rev" else ncol).max(1) + 1


class StarColors:
    """asdex's star coloring of symmetric Boolean patterns, each distinct pattern colored once."""

    def __init__(self):
        self.done, self.hits = {}, 0

    def __call__(self, H):
        r, c = H.nonzero()
        if r.size == 0:
            return 0
        key = np.sort(r.astype(np.int64) * H.shape[1] + c)
        h = hashlib.sha1(key.tobytes()).hexdigest() + str(H.shape)
        if h in self.done:
            self.hits += 1
            return self.done[h]
        sp = asdex.SparsityPattern.from_coo((key // H.shape[1]).astype(np.int32), (key % H.shape[1]).astype(np.int32), H.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", asdex.DenseColoringWarning)
            k = int(asdex.hessian_coloring_from_sparsity(sp, mode="fwd_over_rev", symmetric=True).num_colors)
        self.done[h] = k
        return k


def persistence(ks, parts, cid, prev):
    """Per instance at each period whose predecessor was counted: whether its supports' pattern lies inside the pattern at
    k - 1 with each near margin (shifted by one sample, and unshifted), the cause of a change (CAUSES), the change of its
    smoothed values, and whether its supports' entries left or entered."""
    if prev is not None:
        ks = np.concatenate([[prev[0]], ks])
        parts = {k: np.concatenate([prev[1][k], v], 0) for k, v in parts.items()}
        off = 1
    else:
        off = 0
    t = np.flatnonzero(ks[1:] == ks[:-1] + 1) + 1
    t = t[t >= off] if prev is not None else t
    if t.size == 0:
        return {}
    key = np.where(parts["inst"] >= 0, cid[None] * 100000 + parts["read"], -1)
    now, before = key[t], key[t - 1]
    eq = (now[:, :, None] == before[:, None, :]) & (now[:, :, None] >= 0)
    match = np.where(eq.any(-1), eq.argmax(-1), -1)
    tt, s = np.nonzero(now >= 0)
    m = match[tt, s]
    new = parts["prefix"][t[tt], s, 0].astype(np.int64)  # (n, T, 2)
    old = parts["prefix"][t[tt] - 1, np.maximum(m, 0)].astype(np.int64)  # (n, modes, T, 2)
    old = np.where((m >= 0)[:, None, None, None], old, 0)
    rolled = np.roll(old, -1, axis=2)
    n_modes = old.shape[1] - 1  # the supports and the near margins, not every operand
    inside_mod = np.stack([np.all(new <= rolled[:, j], (1, 2)) for j in range(n_modes)], 1)
    inside_rel = np.stack([np.all(new <= old[:, j], (1, 2)) for j in range(n_modes)], 1)
    end = np.stack([np.any(new[:, -1] > rolled[:, j, -1], 1) for j in range(n_modes)], 1)
    nd = parts["next_decided"][t[tt] - 1, np.maximum(m, 0)]
    cause = np.where((m < 0)[:, None], 0, np.where(end, 1, np.where(nd[:, None], 2, 4)))
    cause = np.where(inside_mod, -1, cause)
    cause_rel = np.where(inside_rel, -1, np.where(inside_mod, 5, cause))
    p0 = old[:, 0]
    left = np.any(p0[:, 0] > 0, 1) | np.any(p0[:, 1:] > new[:, :-1], (1, 2))
    entered = np.any(new[:, :-1] > p0[:, 1:], (1, 2)) | np.any(new[:, -1] > 0, 1)
    return {"k": ks[t[tt]], "cid": cid[s], "read": parts["read"][t[tt], s], "matched": m >= 0, "inside_mod": inside_mod,
            "inside_rel": inside_rel, "cause_mod": cause, "cause_rel": cause_rel,
            "d_sm": parts["sm"][t[tt], s] - np.where(m >= 0, parts["sm"][t[tt] - 1, np.maximum(m, 0)], np.nan),
            "d_lse": parts["lse"][t[tt], s] - np.where(m >= 0, parts["lse"][t[tt] - 1, np.maximum(m, 0)], np.nan),
            "left": left, "entered": entered, "end_reached": parts["reaches_end"][t[tt], s] | parts["reaches_end"][t[tt] - 1, np.maximum(m, 0)]}


def count(scene, conjs, periods, hessian_periods, figure_period=None, chunk=10, reference=None):
    """Counts at `periods` (Jacobian) and `hessian_periods` (a subset; Hessian) along the scene's states; (arrays, meta)."""
    T = scene.T
    n, nv = NX * T, (NX + NU) * T
    periods = np.asarray(periods)
    def_r, def_c = defect_pattern(T)
    cost_r, cost_c = control_band(T)
    reference = Reference(scene, conjs) if reference is None else reference
    n_slots = [cj.imposed(periods, T)[0].shape[1] for cj in conjs]
    fam_slot = np.concatenate([np.full(s, cj.kind) for cj, s in zip(conjs, n_slots)])
    cid = np.concatenate([np.full(s, i) for i, s in enumerate(n_slots)])
    fams = [f for f in FAMILIES if (fam_slot == f).any()]
    fam_idx = np.zeros(fam_slot.size, np.int64)
    for i, f in enumerate(fams):  # conjunct kinds
        fam_idx[fam_slot == f] = i
    star = StarColors()
    rec, hrec, steps, pers, prev, figure = {}, {}, [], [], None, {}
    add = lambda d, key, v: d.setdefault(key, []).append(v)  # noqa: E731
    for c0 in range(0, periods.size, chunk):  # chunks of periods
        ks = periods[c0:c0 + chunk]
        K = ks.size
        parts = scene_rows(reference, ks, n_slots)
        on = parts["inst"] >= 0
        N = on.shape[1]
        pre = parts["prefix"]  # (K, N, modes, T, 2)
        st = (pre[:, :, 0].sum(3) > 0).sum(-1)
        share = (pre[:, :, -1].sum(3) > 0).sum(-1) / T
        kk_, ss_ = np.nonzero(on)
        steps.append(np.stack([st[on], share[on], fam_idx[ss_]], 1))
        for lev, mode in LEVELS:  # rows reading one arm's positions at one time step
            over = np.where(on[:, :, None, None], pre[:, :, mode] > 0, False).sum(1)  # (K, T, 2)
            top = over.reshape(K, -1).max(1)
            add(rec, ("overlap", lev), top)
            add(rec, ("overlap_median", lev), np.median(over.reshape(K, -1), 1))
            add(rec, ("overlap_step", lev), np.argmax(over.max(-1), 1))
            add(rec, ("overlap_last", lev), over[:, -1].max(-1) == top)
            add(rec, ("overlap_inner", lev), over[:, :-1].reshape(K, -1).max(1))
            add(rec, ("overlap_first", lev), over[:, 0].max(-1))
            add(rec, ("overlap_lastval", lev), over[:, -1].max(-1))
            add(rec, ("overlap_mid", lev), over[:, 1:-1].reshape(K, -1).max(1) if T > 2 else np.zeros(K, np.int64))
        has = pre[:, :, 0].sum(3) > 0  # (K, N, T) time steps of each row's supports
        for fset in ["all"] + fams:  # where the supports sit, all rows and by kind
            sel = on if fset == "all" else on & (fam_slot == fset)[None]
            add(rec, ("support_location", fset, "first"), (sel & has[..., 0]).sum(1))
            add(rec, ("support_location", fset, "last"), (sel & has[..., -1]).sum(1))
            add(rec, ("support_location", fset, "interior"), (sel & has.any(-1) & ~has[..., 0] & ~has[..., -1]).sum(1))
            add(rec, ("support_location", fset, "none"), (sel & ~has.any(-1)).sum(1))
        for fset in ["all"] + fams:  # the specification's rows, all and by kind
            sel = on if fset == "all" else on & (fam_slot == fset)[None]
            add(rec, (fset, "rows"), sel.sum(1))
            for lev, mode in LEVELS:
                p = np.where(sel[:, :, None, None], pre[:, :, mode], 0).reshape(K * N, T, 2)
                r, c = columns(p)
                t = r // N
                add(rec, (fset, lev, "nnz"), np.bincount(t, minlength=K))
                rc = _colors(r, c + t * nv, K, N, nv, "rev") * (sel.sum(1) > 0)
                cc = _colors(r, c + t * nv, K, N, nv, "fwd") * (sel.sum(1) > 0)
                add(rec, (fset, lev, "products"), np.minimum(rc, cc))
                add(rec, (fset, lev, "row_colors"), rc)
                add(rec, (fset, lev, "col_colors"), cc)
                add(rec, (fset, lev, "leaves"), np.where(sel, parts["leaves"][..., mode], 0).sum(1))
                v = np.where(sel, pre[:, :, mode].astype(np.int64).sum((2, 3)), 0)
                add(rec, (fset, lev, "row_vars_max"), v.max(1))
        m_dense = n + on.sum(1)
        add(rec, ("whole", "dense", "nnz"), m_dense * nv)
        add(rec, ("whole", "dense", "products"), np.minimum(m_dense, nv))
        for lev, mode in LEVELS:  # the whole Jacobian: the defects stacked above the specification's rows
            p = np.where(on[:, :, None, None], pre[:, :, mode], 0).reshape(K * N, T, 2)
            r, c = columns(p)
            t = r // N
            M = n + N
            rr = np.concatenate([(np.arange(K)[:, None] * M + def_r[None]).ravel(), t * M + n + r % N])
            cc = np.concatenate([(np.arange(K)[:, None] * nv + def_c[None]).ravel(), t * nv + c])
            rcw = _colors(rr, cc, K, M, nv, "rev")
            ccw = _colors(rr, cc, K, M, nv, "fwd")
            add(rec, ("whole", lev, "products"), np.minimum(rcw, ccw))
            add(rec, ("whole", lev, "row_colors"), rcw)
            add(rec, ("whole", lev, "col_colors"), ccw)
            add(rec, ("whole", lev, "nnz"), np.bincount(t, minlength=K) + def_r.size)
            # forward products on the position columns of chosen time steps, reverse products on the other entries: "ends"
            # the first and the last step, "piles" those and every step at which at least 3 rows read one arm's positions
            reads = (np.where(on[:, :, None, None], pre[:, :, mode], 0) > 0).sum(1).max(-1)  # (K, T)
            loc = cc - (rr // M) * nv
            pos = (loc < n) & ((loc % NX) < NQ)
            step = np.minimum(loc // NX, T - 1)
            for rule in ("ends", "piles"):  # column sets of the combination
                Fs = np.zeros((K, T), bool)
                Fs[:, [0, T - 1]] = True
                if rule == "piles":
                    Fs |= reads >= 3
                F = pos & Fs[rr // M, step]
                fc = _colors(rr[F], cc[F], K, M, nv, "fwd") * (np.bincount(rr[F] // M, minlength=K) > 0)
                rc = _colors(rr[~F], cc[~F], K, M, nv, "rev") * (np.bincount(rr[~F] // M, minlength=K) > 0)
                add(rec, ("whole", lev, rule + "_fwd"), fc)
                add(rec, ("whole", lev, rule + "_rev"), rc)
                add(rec, ("whole", lev, rule), fc + rc)
                add(rec, ("whole", lev, rule + "_steps"), Fs.sum(1))
        for kk in np.flatnonzero(np.isin(ks, hessian_periods)):  # Hessian periods: one star coloring per pattern
            for lev, mode in LEVELS:  # the whole Lagrangian Hessian: the rows' cliques and the cost's band
                r, c = columns(pre[kk][on[kk]][:, mode])
                Hs = hessian_union(r, c, max(int(on[kk].sum()), 1), nv)
                hr, hc = Hs.nonzero()
                key = np.unique(np.concatenate([hr.astype(np.int64) * nv + hc, cost_r * nv + cost_c, cost_c * nv + cost_r]))
                Hw = sps.csr_matrix((np.ones(key.size, np.int8), (key // nv, key % nv)), shape=(nv, nv))
                a_, b_ = Hw.nonzero()
                add(hrec, ("whole", lev, "upper"), int(np.sum(a_ <= b_)))
                add(hrec, ("whole", lev, "star"), star(Hw))
            for fset in ["all"] + fams:  # the specification's rows only
                sel = on[kk] if fset == "all" else on[kk] & (fam_slot == fset)
                for lev, mode in LEVELS:
                    r, c = columns(pre[kk][sel][:, mode])
                    Hm = hessian_union(r, c, max(int(sel.sum()), 1), nv)
                    hr, hc = Hm.nonzero()
                    add(hrec, (fset, lev, "upper"), int(np.sum(hr <= hc)))
                    add(hrec, (fset, lev, "star"), star(Hm))
            add(hrec, ("k",), int(ks[kk]))
        if figure_period is not None and figure_period in ks:  # the figure's period
            kk = int(np.flatnonzero(ks == figure_period)[0])
            order = np.argsort(fam_idx[on[kk]], kind="stable")
            for lev, mode in LEVELS:
                p = pre[kk][on[kk]][order][:, mode]
                r, c = columns(p)
                figure["jacobian_" + lev + "_rows"], figure["jacobian_" + lev + "_cols"] = r, c
                hr, hc = hessian_union(r, c, p.shape[0], nv).nonzero()
                figure["hessian_" + lev + "_rows"], figure["hessian_" + lev + "_cols"] = hr, hc
            figure["row_kind"] = fam_slot[on[kk]][order]
            figure["period"] = int(figure_period)
        pr = persistence(ks, parts, cid, prev)
        if pr:
            pers.append(pr)
        prev = (ks[-1], {k: v[-1:] for k, v in parts.items()})
    arrays = {"periods": periods, "instance_steps": np.concatenate(steps)}
    arrays.update({"_".join(k): np.concatenate(v) for k, v in rec.items()})
    arrays.update({"hessian_" + "_".join(k): np.asarray(v) for k, v in hrec.items()})
    if pers:
        arrays.update({"persistence_" + k: np.concatenate([p[k] for p in pers]) for k in pers[0]})
    arrays.update({"figure_" + k: np.asarray(v) for k, v in figure.items()})
    meta = {"T": T, "record_samples": max(cj.program.T for cj in conjs), "n": n, "nv": nv, "fams": fams,
            "n_defect_entries": def_r.size, "star_colorings": len(star.done), "star_cache_hits": star.hits,
            "conjuncts": {f: int(sum(cj.kind == f for cj in conjs)) for f in fams}, "slots": int(fam_slot.size)}
    return arrays, meta


def _q(v, p=50):
    return round(float(np.percentile(v, p)), 3) if np.size(v) else None


def _line(*xs):
    return " ".join(str(x) for x in xs)


def report(A, meta, title, last_sample):
    """Summary lines: medians over the counted periods, ratios as medians of per-period ratios; the lines marked "inside"
    repeat the whole program's counts over the periods whose horizon ends at or before `last_sample`."""
    T, n, nv, fams = meta["T"], meta["n"], meta["nv"], meta["fams"]
    k, kh = A["periods"], A["hessian_k"]
    out = []
    rows = A["all_rows"]
    m_dense = n + rows
    u = A["until_rows"] if "until_rows" in A else None
    out.append(_line(title, "| T", T, "| conjuncts", meta["conjuncts"], "| rows per period median", _q(rows), "| until rows per period median",
                     _q(u) if u is not None else 0, "min", int(u.min()) if u is not None else 0, "max", int(u.max()) if u is not None else 0,
                     "| periods", k.size, "Hessian periods", kh.size, "| record samples", meta["record_samples"],
                     "| distinct Hessian patterns colored", meta["star_colorings"], "reused", meta["star_cache_hits"]))
    out.append(_line("   dense AD: Jacobian", _q(m_dense), "x", nv, "=", _q(m_dense * nv), "entries, AD products", _q(np.minimum(m_dense, nv)),
                     "| Hessian", nv, "x", nv, "upper entries", nv * (nv + 1) // 2, ", Hessian-vector products", nv))
    spans = (("all periods", np.ones(k.size, bool), np.ones(kh.size, bool)),
             ("inside (k + T <= " + str(last_sample) + ")", k + T <= last_sample, kh + T <= last_sample))
    for label, mj, mh in spans:  # all periods, and those whose horizon ends inside the record
        if not mj.any():
            out.append("   " + label + ": no period")
            continue
        w1, w2 = A["whole_prescribed_nnz"][mj], A["whole_supports_nnz"][mj]
        p1, p2 = A["whole_prescribed_products"][mj], A["whole_supports_products"][mj]
        md = np.minimum(m_dense, nv)[mj]
        out.append(_line("   whole Jacobian,", label, "(" + str(int(mj.sum())), "periods) | entries prescribed", _q(w1), "supports", _q(w2),
                         "| fraction of dense prescribed", _q(w1 / (m_dense[mj] * nv)), "supports", _q(w2 / (m_dense[mj] * nv)),
                         "| AD products prescribed", _q(p1), "(rows", _q(A["whole_prescribed_row_colors"][mj]), "columns",
                         _q(A["whole_prescribed_col_colors"][mj]), ") supports", _q(p2), "(rows", _q(A["whole_supports_row_colors"][mj]),
                         "columns", _q(A["whole_supports_col_colors"][mj]), ") | ratios dense/supports", _q(md / p2), "prescribed/supports",
                         _q(p1 / p2), "dense/prescribed", _q(md / p1)))
        if not mh.any():
            out.append("   whole Hessian, " + label + ": no Hessian period")
            continue
        hu1, hu2 = A["hessian_whole_prescribed_upper"][mh], A["hessian_whole_supports_upper"][mh]
        hs1, hs2 = A["hessian_whole_prescribed_star"][mh], A["hessian_whole_supports_star"][mh]
        up = nv * (nv + 1) / 2
        out.append(_line("   whole Hessian,", label, "(" + str(int(mh.sum())), "periods) | upper entries prescribed", _q(hu1), "supports", _q(hu2),
                         "| fraction prescribed", _q(hu1 / up), "supports", _q(hu2 / up), "| star colors prescribed", _q(hs1), "supports", _q(hs2),
                         "| ratios dense/supports", _q(nv / hs2), "prescribed/supports", _q(hs1 / hs2), "dense/prescribed", _q(nv / hs1),
                         "| ratio of medians prescribed/supports", round(float(np.median(hs1) / np.median(hs2)), 3),
                         "| per period (k, prescribed, supports)", list(zip(kh[mh].tolist(), hs1.tolist(), hs2.tolist()))))
    for fset in ["all"] + fams:  # the specification's rows, all and by kind
        r_ = A[fset + "_rows"]
        m = r_ > 0
        if not m.any():
            continue
        a = lambda lev, q: A[fset + "_" + lev + "_" + q]  # noqa: E731
        n1, n2 = a("prescribed", "nnz")[m], a("supports", "nnz")[m]
        p1, p2 = a("prescribed", "products")[m], a("supports", "products")[m]
        p0 = np.minimum(r_[m], nv)
        out.append(_line("  ", "specification rows" if fset == "all" else "family " + fset, "| rows", _q(r_[m]), "| entries prescribed", _q(n1),
                         "supports", _q(n2), "| fraction of rows x variables prescribed", _q(n1 / (r_[m] * nv)), "supports", _q(n2 / (r_[m] * nv)),
                         "| AD products dense", _q(p0), "prescribed", _q(p1), "(rows", _q(a("prescribed", "row_colors")[m]), "columns",
                         _q(a("prescribed", "col_colors")[m]), ") supports", _q(p2), "(rows", _q(a("supports", "row_colors")[m]), "columns",
                         _q(a("supports", "col_colors")[m]), ") | ratios dense/supports", _q(p0 / np.maximum(p2, 1)), "prescribed/supports",
                         _q(p1 / np.maximum(p2, 1)), "| operand values prescribed", _q(a("prescribed", "leaves")[m]), "supports",
                         _q(a("supports", "leaves")[m]), "| largest variables of one row, median over periods prescribed",
                         _q(a("prescribed", "row_vars_max")[m]), "supports", _q(a("supports", "row_vars_max")[m]), "; overall prescribed",
                         int(a("prescribed", "row_vars_max").max()), "supports", int(a("supports", "row_vars_max").max())))
        hu1, hu2 = A["hessian_" + fset + "_prescribed_upper"], A["hessian_" + fset + "_supports_upper"]
        hs1, hs2 = A["hessian_" + fset + "_prescribed_star"], A["hessian_" + fset + "_supports_star"]
        out.append(_line("      its Hessian part | upper entries prescribed", _q(hu1), "supports", _q(hu2), "| star colors prescribed", _q(hs1),
                         "supports", _q(hs2), "| ratios dense (", nv, ")/supports", _q(nv / np.maximum(hs2, 1)), "prescribed/supports",
                         _q(hs1 / np.maximum(hs2, 1))))
    S = A["instance_steps"]
    for i, f in enumerate(fams):  # conjunct kinds
        m = S[:, 2] == i
        out.append(_line("   support length in time steps, family", f, "| rows", int(m.sum()), "median", _q(S[m, 0]), "p90", _q(S[m, 0], 90), "max",
                         int(S[m, 0].max(initial=0)), "| window share of the horizon median", _q(S[m, 1]), "max", _q(S[m, 1], 100)))
    for lev in ("prescribed", "supports"):
        top = A["overlap_" + lev]
        out.append(_line("   rows reading one arm's positions at one time step,", lev, "| largest: median over periods", _q(top), "overall",
                         int(top.max()), "| median over steps and arms: median over periods", _q(A["overlap_median_" + lev]),
                         "| horizon step of the largest (0 to", T - 1, "): median", _q(A["overlap_step_" + lev]), "| periods whose largest is at the last step",
                         round(float(A["overlap_last_" + lev].mean()), 3), "| largest without the last step: median over periods",
                         _q(A["overlap_inner_" + lev])))
    if "persistence_matched" in A:
        mm = A["persistence_matched"]
        out.append(_line("   persistence modulo T | pairs", int(mm.sum()), "inside at margin 0", round(float(A["persistence_inside_mod"][mm, 0].mean()), 3),
                         "at margin gamma", round(float(A["persistence_inside_mod"][mm, 2].mean()), 3)))
    md = np.minimum(m_dense, nv)
    best = {}
    for lev in ("prescribed", "supports"):  # the Jacobian's four counts: reverse, forward and the two combinations
        w = lambda q: A["whole_" + lev + "_" + q]  # noqa: E731
        rv, fw, e, p = w("row_colors"), w("col_colors"), w("ends"), w("piles")
        best[lev] = np.minimum(np.minimum(rv, fw), np.minimum(e, p))
        b = best[lev]
        out.append(_line("   whole Jacobian,", lev, "| reverse (row colors)", _q(rv), "forward (column colors)", _q(fw),
                         "| combination ends (first and last step forward):", _q(e), "= forward", _q(w("ends_fwd")), "+ reverse", _q(w("ends_rev")),
                         "| combination piles:", _q(p), "= forward", _q(w("piles_fwd")), "+ reverse", _q(w("piles_rev")), "over", _q(w("piles_steps")),
                         "steps", "| smallest of the four per period: median", _q(b), "; which is smallest (reverse, forward, ends, piles) in periods",
                         [int(np.sum(b == rv)), int(np.sum(b == fw)), int(np.sum(b == e)), int(np.sum(b == p))]))
    b1, b2 = best["prescribed"], best["supports"]
    out.append(_line("   whole Jacobian, best of the four counts | dense", _q(md), "prescribed", _q(b1), "supports", _q(b2), "| ratios dense/supports",
                     _q(md / b2), "prescribed/supports", _q(b1 / b2), "| prescribed reverse over supports best", _q(A["whole_prescribed_row_colors"] / b2)))
    for lev in ("prescribed", "supports"):  # rows reading one arm's positions at the first, last and interior steps
        o = lambda q: A["overlap_" + q + "_" + lev]  # noqa: E731
        out.append(_line("   rows reading one arm's positions,", lev, "| at the first step: median", _q(o("first")), "largest", int(o("first").max()),
                         "| at the last step: median", _q(o("lastval")), "largest", int(o("lastval").max()), "| largest at an interior step: median",
                         _q(o("mid")), "largest", int(o("mid").max())))
    for fset in ["all"] + fams:  # where the supports sit
        s = lambda q: A["support_location_" + fset + "_" + q]  # noqa: E731
        out.append(_line("   supports' time steps,", "all rows" if fset == "all" else "family " + fset, "| rows per period: touching the first step",
                         _q(s("first")), "touching the last step", _q(s("last")), "interior only", _q(s("interior")), "no entry", _q(s("none")),
                         "(medians over periods; largest first", int(s("first").max()), "last", int(s("last").max()), ")"))
    return out

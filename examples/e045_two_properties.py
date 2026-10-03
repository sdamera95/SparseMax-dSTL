"""E045: the two proved properties per method on the Until conjunct clear U[w, w+1] pick of the person scene (E039), at
the violating start trajectories (E034's four starts at entry depths 0.05 and 0.10, the hold extended to the horizon as
E038 built the 20 and 40 s starts), at the longest wait of each horizon (M003 note of 2026-10-01 14:49Z; entry E045).

    JAX_PLATFORMS=cpu python -m examples.e045_two_properties theory <H> <instance.npz> <out_prefix>
    JAX_PLATFORMS=cpu python -m examples.e045_two_properties cpu <H> <out_prefix>
    JAX_PLATFORMS=nojax python -m examples.e045_two_properties gpu <out.csv> <H>:<instance.npz>:<prefix> ...
    python -m examples.e045_two_properties table <gate_dir> <bounds.csv> <out_prefix> <H>:<prefix> ...

theory: the commands replayed in float64 MuJoCo C, the predicate values of tasks.workspace (scores: the smoothed
predicates the smoothed robustness reads; margins: the exact ones the certificate reads), saved to <prefix>_pred.npz,
and the theory's weight sum on the violating samples per method and eps from these values alone (predict(); no
derivative is taken), <prefix>_theory.csv.
cpu: the smoothed robustness of the Until conjunct (on the scores) and the exact one (on the margins), and the gradient
of the smoothed robustness with respect to every predicate value by reverse-mode automatic differentiation (JAX,
float64) with its sums (sums()), <prefix>_cpu.csv.
gpu: through the plant (the Warp chain, float32): the norm of the gradient of the smoothed robustness with respect to
the controls, from the whole score gradient and from its entries on clear at the violating samples only (every other
entry set to 0; one more backward pass through the predicates and the plant).

The violating samples are the samples the Until at t = 0 reads (0 <= t <= w + 1) where clear's predicate value (the score
the smoothed robustness reads and is differentiated against) is below 0; the columns *_margin use clear's exact value (margin)
instead, which is negative on 0 to 2 fewer samples
(clear is also negative after w + 1, at the handover inside the zone; the Until does not read those samples); the hold samples are the samples
with 2.0 s <= t <= w + 1 that are not violating (E038's hold). The Until at t = 0 is a maximum over the witnesses
k = kw_j (j = 1..J, J = 51) of rows (pick(k), clear(0), ..., clear(k)); sparsemax and the log-sum-exp reduce a row as
one minimum over m_j = k + 2 entries, the generalized mean robustness nests it (eq. 15: the prefix over d_j = k + 1 entries, then the
pair (pick(k), prefix)).
"""
import csv
import sys

import numpy as np

from examples import e034_until_demo as D
from examples import e038_horizon as X

ARMS = ("lse_plain", "lse", "sparsemax", "gm_pm01", "gm_pm10", "gm_exp")
EPS = (0.1, 0.2, 0.4)
NO_EPS = ("gm_pm01", "gm_pm10")


def settings():
    """(arm, eps) pairs; gm_pm01 and gm_pm10 have no parameter (eps NaN)."""
    return [(a, e) for a in ARMS for e in ((np.nan,) if a in NO_EPS else EPS)]


def starts(H, path):
    """The longest wait of H (set in this process), the instance, and one row per distinct start trajectory
    (depth, start) at that wait, sorted by depth and start."""
    w = X.horizon(H)
    I = D.load(path)
    rows = np.nonzero(np.isclose(I["run_wait"], w))[0]
    key = np.round(I["run_depth"][rows], 6) * 100 + I["run_start"][rows]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    idx = rows[first]
    if not (np.array_equal(I["V0"][rows], I["V0"][idx][inv]) and np.all(I["x0"][rows] == I["x0"][idx[0]])):
        raise ValueError("two runs of the same depth and start differ")
    return w, I, idx


def until_program(w, n_h):
    """The Until conjunct's program (E040's conjunct programs, pruned to t = 0) and its witness samples kw (J,)."""
    from examples.e040_conj import CONJ, conj_programs
    sc = D.scenario(w)
    prog = conj_programs(sc, n_h)[CONJ.index("order")]
    inner = [s for s in prog.steps if s.label.endswith(".inner")]
    if len(inner) != 1 or prog.steps[-1].length != 1:
        raise ValueError("expected one Until read at t = 0")
    return sc, prog, inner[0].count.astype(int) - 2


# ------------------------------------------------------------------
# the theory's weight sums

def _lse(a, axis=-1):
    top = np.max(a, axis, keepdims=True)
    return np.squeeze(top, axis) + np.log(np.sum(np.exp(a - top), axis))


def _simplex(s):
    """Euclidean projection onto the simplex of s (..., m) along the last axis (-inf entries excluded): (p, tau)."""
    u = -np.sort(-s, -1)
    fin = np.isfinite(u)
    cs = np.cumsum(np.where(fin, u, 0.0), -1)
    r = np.arange(1, s.shape[-1] + 1)
    kk = np.sum(fin & (1 + r * u > cs), -1)
    tau = (np.take_along_axis(cs, kk[..., None] - 1, -1)[..., 0] - 1) / kk
    return np.where(np.isfinite(s), np.maximum(s - tau[..., None], 0.0), 0.0), tau


def predict(c, p, neg, kw, hold, arm, eps, sel=None):
    """The theory's weight sum on the samples sel (n, T; default neg) of the Until's value at t = 0, per trajectory, where
    neg (n, T) marks the samples with clear below 0 (the branch the generalized mean robustness takes depends on it), from clear c and
    pick p (n, T) at the witness samples kw (J,); hold (T,) marks the hold samples. Returns a dict of (n,) arrays:
    law (the paper's or the derived law with the measured plug-in quantities), closed (the derivative of the
    implemented value written out by hand; equal to the automatic derivative when the derivation is right) and the
    plug-in quantities. The closed forms of the generalized mean robustness assume pick > 0 at every witness and every negative clear sample before
    the first witness (reported as pick_min_witness, neg_after_first)."""
    n, T = c.shape
    J = len(kw)
    t = np.arange(T)
    pre = t[None] <= kw[:, None]  # (J, T): the prefix of row j
    pw = p[:, kw]  # (n, J)
    sel = neg if sel is None else sel
    k = neg.sum(1).astype(float)
    ks = sel.sum(1).astype(float)
    d = kw + 1.0
    m = kw + 2.0
    nan = np.full(n, np.nan)
    out = {"k": k, "k_sel": ks, "pick_min_witness": pw.min(1), "neg_after_first": np.sum(neg & (t[None] > kw[0]), 1), "law": nan, "closed": nan}
    v = np.where(neg, -c, 0.0)  # depth of each negative sample
    rows = np.concatenate([pw[..., None], np.where(pre[None], c[:, None], np.inf)], -1)  # (n, J, T + 1); +inf: not in the row
    negr = np.concatenate([np.zeros((n, J, 1), bool), sel[:, None] & pre[None]], -1)  # the summed samples of each row
    if arm == "sparsemax":
        gam = 2 * eps / (1 - 1 / m)
        zh = np.min(np.where(negr, np.inf, rows), -1)  # (n, J): the smallest entry of the row that is not a negative clear sample
        S = np.sum(np.where(negr, zh[..., None] - rows, 0.0), -1)  # sum of the gaps of the negative samples below it
        ok = np.all(S >= gam[None], 1)
        P_, tau = _simplex(-rows / gam[None, :, None])  # the row's weights (lower minimum)
        W = np.sum(np.where(negr, P_, 0.0), -1)
        r = -gam * (tau + 0.5 * np.sum(P_ * P_, -1)) - gam / 2  # the row's value
        go = 2 * eps / (1 - 1 / J)
        q, _ = _simplex(r / go)  # the outer maximum's weights
        out.update(law=np.where(ok, 1.0, np.nan), closed=np.sum(q * W, 1), gamma=np.full(n, gam.max()), gamma_over_k=gam.max() / ks,
                   delta_mean=(S / ks[:, None]).min(1), cond_margin=(S - gam[None]).min(1))
    elif arm in ("lse", "lse_plain"):
        beta = np.log(m) / eps
        a = np.where(np.isfinite(rows), -beta[None, :, None] * np.where(np.isfinite(rows), rows, 0.0), -np.inf)
        L = _lse(a)
        W = np.sum(np.where(negr, np.exp(a - L[..., None]), 0.0), -1)
        r = -L / beta
        bo = np.log(J) / eps
        q = np.exp(bo * r - _lse(bo * r)[:, None])
        j = J // 2  # the two-level law at the middle witness
        cmin = np.min(np.where(sel, c, np.inf), 1)
        keff = np.sum(np.where(sel, m[j] ** (-(c - cmin[:, None]) / eps), 0.0), 1)
        hj = hold[None] & ~neg & ~sel & (t[None] <= kw[j])
        hc = np.where(hj, c, np.nan)
        N = hj.sum(1).astype(float)
        Delta = np.nanmedian(hc, 1) - cmin
        out.update(law=keff / (keff + N * m[j] ** (-Delta / eps)), closed=np.sum(q * W, 1), k_eff=keff, N=N, m=np.full(n, m[j]), Delta=Delta,
                   hold_min=np.nanmin(hc, 1), hold_max=np.nanmax(hc, 1))
    elif arm == "gm_pm01":
        dg = np.exp(np.mean(np.log(d)))
        out.update(law=ks / (2 * dg), closed=ks * np.exp(np.mean(np.log(1 / (2 * d)))), d_geo=np.full(n, dg))
    elif arm == "gm_pm10":
        G = np.sum(np.where(sel, v, 0.0) ** 9, 1) / np.sum(v ** 10, 1) ** 0.9
        c_ = 0.5 ** 0.1 * d ** -0.1
        Mc = np.mean(c_ ** -10.0) ** -0.1
        out.update(law=G * (2 * np.mean(d)) ** -0.1, closed=G * np.mean((Mc / c_) ** 11 * c_), G=G, d_mean=np.full(n, np.mean(d)))
    elif arm == "gm_exp":
        bp = np.log(d) / eps
        ex = np.exp(bp[None, :, None] * v[:, None])  # (n, J, T)
        A = np.sum(np.where(neg[:, None], ex, 0.0), -1)  # (n, J)
        E = np.sum(np.where(sel[:, None], ex, 0.0), -1) / (A + d - k[:, None])
        Pv = np.log((A + d - k[:, None]) / d) / bp  # |prefix value|
        b2 = np.log(2.0) / eps
        e2 = np.exp(b2 * Pv)
        pair = e2 / (1 + e2)
        x = -np.log((1 + e2) / 2) / b2
        bo = np.log(J) / eps
        q = np.exp(bo * x - _lse(bo * x)[:, None])
        j = J // 2
        out.update(law=pair[:, j] * E[:, j], closed=np.sum(q * pair * E, 1), E_mid=E[:, j], pair_mid=pair[:, j], d_mid=np.full(n, d[j]))
    return out


def sums(g, viol, hold):
    """Sums of a score gradient g (n, T, P): on clear (column 2) at the violating samples viol (n, T), on clear at the
    hold samples hold (n, T), on pick (column 0), over every entry; the number of nonzero entries (all, clear) and the
    number of largest entries carrying 90 % of the absolute sum."""
    gc = g[:, :, 2]
    a = np.abs(g.reshape(len(g), -1))
    cs = np.cumsum(-np.sort(-a, 1), 1)
    return {"w_viol": np.sum(np.where(viol, gc, 0.0), 1), "w_hold": np.sum(np.where(hold, gc, 0.0), 1), "w_pick": g[:, :, 0].sum(1),
            "w_total": g.sum((1, 2)), "nonzero": np.sum(a > 0, 1), "nonzero_clear": np.sum(np.abs(gc) > 0, 1),
            "n90": np.argmax(cs >= 0.9 * cs[:, -1:], 1) + 1}


# ------------------------------------------------------------------
# the CPU steps (float64)

def theory_main(H, path, prefix):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl.tasks import workspace as W
    jax.config.update("jax_enable_x64", True)
    w, I, idx = starts(H, path)
    sc, prog, kw = until_program(w, int(I["n_h"]))
    Xs = D.replay(I["V0"][idx], I["x0"][idx[0]].astype(np.float64))
    mx = mjx.put_model(E.plant.model, impl="jax")
    fixed = {"pick": jnp.asarray(I["pick"]), "handover": jnp.asarray(I["handover"]), "human_radii": jnp.asarray(I["hr"])}

    def both(x, hc):
        inst = dict(fixed, human_centres=hc)
        return W.margins(mx, E.plant, sc, inst, x), W.scores(mx, E.plant, sc, inst, x)
    f = jax.jit(jax.vmap(both))
    hc = jnp.asarray(I["hc"][idx], jnp.float64)
    Mg, Zs = (np.asarray(a) for a in f(jnp.asarray(Xs), hc))
    Mf, Zf = (np.asarray(a) for a in f(jnp.asarray(Xs), hc + 100.0))  # the person 100 m away: clear and pick must not change
    person = float(max(np.abs(Mf - Mg)[..., [0, 2]].max(), np.abs(Zf - Zs)[..., [0, 2]].max()))
    t = np.arange(Zs.shape[1]) * D.HS
    reads = np.arange(Zs.shape[1]) <= kw[-1]  # the samples the Until at t = 0 reads (clear on [0, w + 1])
    vm = (Mg[:, :, 2] < 0) & reads  # clear's exact value below 0
    neg = (Zs[:, :, 2] < 0) & reads  # clear's predicate value (the score the smoothed robustness reads) below 0
    viol = neg
    hold = (t >= D.T_HOLD - 1e-9) & (t <= w + 1 + 1e-9)
    np.savez(prefix + "_pred.npz", X64=Xs, margins=Mg, scores=Zs, viol=viol, viol_margin=vm, neg=neg, hold=hold, kw=kw, wait=w, horizon=H, plan=I["plan"][idx],
             depth=I["run_depth"][idx], start=I["run_start"][idx], instance=path, person_effect=person)
    rows = []
    for arm, e in settings():  # over the 14 (method, eps) settings
        P = predict(Zs[:, :, 2], Zs[:, :, 0], neg, kw, hold, arm, e)
        Pm = predict(Zs[:, :, 2], Zs[:, :, 0], neg, kw, hold, arm, e, sel=vm)
        for i in range(len(idx)):  # over the 8 start trajectories (table rows)
            row = {"horizon": H, "wait": w, "depth": float(I["run_depth"][idx[i]]), "start": int(I["run_start"][idx[i]]), "arm": arm, "eps": e,
                   "k_viol": int(viol[i].sum()), "k_viol_margin": int(vm[i].sum()), "viol_ne_margin": int(np.sum(viol[i] != vm[i])),
                   "viol_first_t": float(t[viol[i]].min()), "viol_last_t": float(t[viol[i]].max()), "hold_overlap": int(np.sum(viol[i] & hold)),
                   "n_witness": len(kw), "kw_first": int(kw[0]), "kw_last": int(kw[-1])}
            row.update({"pred_" + k_: float(np.asarray(v_)[i]) for k_, v_ in P.items()})
            row["predm_closed"] = float(Pm["closed"][i])
            rows.append(row)
    keys = list(dict.fromkeys(k_ for r_ in rows for k_ in r_))
    D._csv(prefix + "_theory.csv", keys, [[r_.get(k_, "") for k_ in keys] for r_ in rows])
    print("horizon", H, "wait", w, "T", Zs.shape[1], "witnesses", kw[0], kw[-1], "person effect on clear and pick", person)
    print("violating samples (negative clear score)", viol.sum(1).tolist(), "negative clear margin", vm.sum(1).tolist(), "deepest clear read", np.round(np.where(reads, Mg[:, :, 2], np.inf).min(1), 4).tolist(),
          "negative clear after w + 1", np.sum((Mg[:, :, 2] < 0) & ~reads, 1).tolist())
    print("hold clear min", np.round(np.where(hold & ~viol, Zs[:, :, 2], np.inf).min(1), 5).tolist(), "max",
          np.round(np.where(hold & ~viol, Zs[:, :, 2], -np.inf).max(1), 5).tolist(), "pick at witnesses min", np.round(Zs[:, kw, 0].min(1), 4).tolist())
    for r_ in rows[::8]:  # printout, the first trajectory of each setting
        print(r_["arm"], r_["eps"], "law", round(r_["pred_law"], 6), "closed", round(r_["pred_closed"], 6))


def cpu_main(H, prefix):
    import jax
    import jax.numpy as jnp

    from sparsemax_dstl import stl
    from sparsemax_dstl.core_study import methods
    jax.config.update("jax_enable_x64", True)
    z = np.load(prefix + "_pred.npz")
    w = X.horizon(H)
    if not np.isclose(float(z["wait"]), w):
        raise ValueError("the predicate file is for another horizon")
    _, prog, kw = until_program(w, int(D.load(str(z["instance"]))["n_h"]))
    if not np.array_equal(kw, z["kw"]):
        raise ValueError("witnesses differ")
    Mg, Zs, viol, vm = z["margins"], z["scores"], z["viol"], z["viol_margin"]
    hold = z["hold"][None] & ~viol & ~vm
    ex = np.asarray(jax.jit(jax.vmap(lambda y: stl.evaluate(prog, y, "exact")[-1][0]))(jnp.asarray(Mg)))
    exs = np.asarray(jax.jit(jax.vmap(lambda y: stl.evaluate(prog, y, "exact")[-1][0]))(jnp.asarray(Zs)))
    rows = []
    for arm, e in settings():  # over the 14 (method, eps) settings
        sem = methods.SEMANTICS[arm]
        par = None if np.isnan(e) else e
        f = jax.jit(jax.vmap(jax.value_and_grad(lambda y: stl.evaluate(prog, y, sem, par)[-1][0])))
        v, g = (np.asarray(a) for a in f(jnp.asarray(Zs)))
        s = sums(g, viol, hold)
        s["w_viol_margin"] = sums(g, vm, hold)["w_viol"]
        for i in range(len(v)):  # over the 8 start trajectories (table rows)
            row = {"horizon": H, "wait": w, "depth": float(z["depth"][i]), "start": int(z["start"][i]), "arm": arm, "eps": e, "value": float(v[i]),
                   "exact": float(ex[i]), "exact_scores": float(exs[i]), "value_minus_exact": float(v[i] - ex[i]), "lower_bound": int(v[i] <= ex[i]),
                   "nonneg": int(v[i] >= 0)}
            row.update({k_: float(s[k_][i]) for k_ in s})
            rows.append(row)
        print(arm, e, "value", np.round(v, 4).tolist(), "exact", np.round(ex, 4).tolist(), "w_viol", np.round(s["w_viol"], 5).tolist(),
              "w_hold", np.round(s["w_hold"], 5).tolist(), "nonzero", s["nonzero"].tolist(), flush=True)
    keys = list(rows[0])
    D._csv(prefix + "_cpu.csv", keys, [[r_[k_] for k_ in keys] for r_ in rows])


# ------------------------------------------------------------------
# through the plant (Warp, float32, GPU)

def gpu_main(out_csv, items, dev="cuda:0", only=None):
    import time

    import warp as wp

    from examples import e022_regime as R
    from sparsemax_dstl.stl.warp_backend import Evaluator, matched_param
    from sparsemax_dstl.warp_plant import Plant
    from sparsemax_dstl.warp_predicates import Predicates
    wp.init()
    S = settings() if only is None else [s_ for s_ in settings() if s_[0] + "/" + str(s_[1]) in only]
    nc = 2 * len(S)  # cotangents: per setting the whole score gradient and its violating part
    rows = []
    for item in items:  # over the horizons (four)
        H, path, prefix = item.split(":")
        H = int(H)
        t0 = time.perf_counter()
        w, I, idx = starts(H, path)
        sc, prog, kw = until_program(w, int(I["n_h"]))
        z = np.load(prefix + "_pred.npz")
        if not np.isclose(float(z["wait"]), w) or not np.array_equal(z["plan"], I["plan"][idx]):
            raise ValueError("the predicate file does not match the instance")
        viol = z["viol"]
        n, T = viol.shape
        plant = Plant(n * nc, T - 1, device=dev)
        Xw = plant.rollout(np.tile(I["x0"][idx], (nc, 1)), np.tile(I["V0"][idx], (nc, 1, 1)))
        copies = float(np.abs(Xw.reshape(nc, n, T, -1) - Xw[None, :n]).max())
        t_roll = time.perf_counter() - t0
        pred = Predicates(R.plant, sc, T, nworld=n, device=dev)
        pred.set_instance(np.broadcast_to(I["pick"], (n, 3)), np.broadcast_to(I["handover"], (n, 3)), I["hc"][idx], I["hr"])
        q = wp.array(np.ascontiguousarray(Xw[:n, :, :7].reshape(-1, 7)), dtype=float, device=dev, requires_grad=True)
        v = wp.array(np.ascontiguousarray(Xw[:n, :, 7:].reshape(-1, 7)), dtype=float, device=dev, requires_grad=True)
        tape = wp.Tape()
        Z = pred.scores(q, v, tape)
        Zn = Z.numpy()
        sign_flips = np.sum(((Zn[:, :, 2] < 0) & (np.arange(T) <= kw[-1])[None]) != viol, 1)  # at the samples the Until reads
        gZ = wp.zeros((n, T, pred.P), dtype=float, device=dev)
        C = np.zeros((nc, n, T, 14), np.float32)
        only = viol[..., None] & (np.arange(pred.P) == 2)[None, None]
        vals, ws = [], []
        for s, (arm, e) in enumerate(S):  # over the 14 settings (method structure)
            ev = Evaluator(prog, arm, matched_param(prog, arm, 0.2 if np.isnan(e) else e), n, wp.float32, dev, P=pred.P)
            r, G = ev.gradient(Z)
            Gn = G.numpy().copy()
            vals.append(r.numpy()[:, 0].copy())
            ws.append(np.sum(np.where(viol, Gn[:, :, 2], 0.0), 1))
            for h, Gs in enumerate((Gn, np.where(only, Gn, 0.0))):  # the whole gradient and its violating part
                gZ.assign(Gs.astype(np.float32))
                tape.backward(grads={Z: gZ})
                C[2 * s + h] = np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(n, T, 14)
                tape.zero()
        t1 = time.perf_counter()
        gV, _ = plant.vjp(C.reshape(n * nc, T, 14))
        t_vjp = time.perf_counter() - t1
        norm = np.sqrt(np.sum(gV.astype(np.float64) ** 2, (1, 2))).reshape(nc, n)
        for s, (arm, e) in enumerate(S):  # table rows: setting x trajectory
            for i in range(n):
                rows.append({"horizon": H, "wait": w, "depth": float(z["depth"][i]), "start": int(z["start"][i]), "arm": arm, "eps": e,
                             "value32": float(vals[s][i]), "w_viol32": float(ws[s][i]), "grad_norm": float(norm[2 * s, i]),
                             "grad_norm_viol": float(norm[2 * s + 1, i]), "part_viol": float(norm[2 * s + 1, i] / max(norm[2 * s, i], 1e-30)),
                             "sign_flips_f32": int(sign_flips[i]), "copies_max_diff": copies, "rollout_s": t_roll, "vjp_s": t_vjp})
        print("horizon", H, "worlds", n * nc, "T", T, "rollout and setup s", round(t_roll, 1), "vjp s", round(t_vjp, 1), "copies max diff", copies,
              "sign flips float32 vs float64", sign_flips.tolist(), flush=True)
        for s, (arm, e) in enumerate(S):  # printout per setting
            print(H, arm, e, "norm", np.round(norm[2 * s], 4).tolist(), "viol part", np.round(norm[2 * s + 1], 4).tolist(), flush=True)
        del plant, pred, tape
    keys = list(rows[0])
    D._csv(out_csv, keys, [[r_[k_] for k_ in keys] for r_ in rows])


# ------------------------------------------------------------------
# the table and the figure

def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def table_main(gate, bounds, out, items, gpu_csv=None):
    """<out>_rows.csv (every row: theory, CPU and GPU columns joined), <out>.md (one row per method and entry depth: the
    lower-bound columns and, at eps 0.2, the weight on the violating samples per horizon, measured (predicted), medians over
    the 4 starts) and <out>_weight.svg/.pdf (the weight against the horizon at eps 0.2, medians over the 4 starts at entry depth 0.05; x marks the law).
    bounds: comma-separated E045 recount CSVs of E043's records (bounds/above_counts*.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = lambda r: (int(float(r["horizon"])), r["arm"], r["eps"], float(r["depth"]), int(float(r["start"])))  # noqa: E731
    rows = {}
    for item in items:  # over horizons
        H, prefix = item.split(":")
        for src in ("_theory.csv", "_cpu.csv"):  # the two CPU files
            for r in _read(prefix + src):
                rows.setdefault(key(r), {}).update(r)
    if gpu_csv:
        for r in _read(gpu_csv):
            if key(r) in rows:
                rows[key(r)].update(r)
    allk = list(dict.fromkeys(k_ for r in rows.values() for k_ in r))
    D._csv(out + "_rows.csv", allk, [[r.get(k_, "") for k_ in allk] for r in rows.values()])
    rec = [r for path in bounds.split(",") for r in _read(path) if r["conjunct"] in ("root", "per_conjunct")]
    Hs = sorted({k_[0] for k_ in rows})
    names = {"lse_plain": "plain log-sum-exp", "lse": "sound log-sum-exp", "sparsemax": "sparsemax", "gm_pm01": "GMR (0,1)",
             "gm_pm10": "GMR (-10,10)", "gm_exp": "GMR exponential"}
    pick = lambda arm, r: float(r["pred_law"])  # noqa: E731  (the laws of PREDICTION.txt; pred_closed is the exact derivative)
    lines = ["| method | entry depth | smoothed <= exact (4 starts, 4 horizons, all eps) | smoothed >= 0 | E043 recorded iterates: checked, above exact, largest excess | "
             + " | ".join("weight on violating samples, " + str(h) + " s, eps 0.2: measured (law)" for h in Hs) + " | " + str(Hs[-1]) + " s / " + str(Hs[0]) + " s | control-gradient norm from violating samples, "
             + str(Hs[0]) + " s / " + str(Hs[-1]) + " s |", "|" + "---|" * (7 + len(Hs))]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for arm in ARMS:  # over the six methods (table rows)
        mine = [r for k_, r in rows.items() if k_[1] == arm]
        e02 = "nan" if arm in NO_EPS else "0.2"
        br = [r for r in rec if r["arm"] == arm]
        bt = (str(sum(int(r["pairs"]) for r in br)) + ", " + str(sum(int(r["above"]) for r in br)) + ", "
              + str(round(max(float(r["largest_excess"]) for r in br), 3))) if br else "none recorded"
        for dep in (0.05, 0.1):  # over the two entry depths
            md = [r for k_, r in rows.items() if k_[1] == arm and k_[3] == dep]
            lb = str(sum(int(r["lower_bound"]) for r in md)) + " of " + str(len(md))
            nn = str(sum(int(r["nonneg"]) for r in md)) + " of " + str(len(md))
            cells, meas = [], []
            for h in Hs:  # over the horizons (table columns)
                sel = [r for k_, r in rows.items() if k_[0] == h and k_[1] == arm and k_[2] == e02 and k_[3] == dep]
                mv = np.median([float(r["w_viol"]) for r in sel])
                pv = np.median([pick(arm, r) for r in sel])
                meas.append(mv)
                cells.append(str(round(mv, 4)) + " (" + str(round(pv, 4)) + ")")
            gn = ""
            if gpu_csv:
                g = [np.median([float(r["grad_norm_viol"]) for k_, r in rows.items() if k_[0] == h and k_[1] == arm and k_[2] == e02 and k_[3] == dep])
                     for h in (Hs[0], Hs[-1])]
                gn = str(round(g[0], 4)) + " / " + str(round(g[1], 4))
            lines.append("| " + names[arm] + " | " + str(dep) + " | " + lb + " | " + nn + " | " + (bt if dep == 0.05 else "as above") + " | "
                         + " | ".join(cells) + " | " + str(round(meas[-1] / meas[0], 3)) + " | " + gn + " |")
        allv = [np.median([float(r["w_viol"]) for k_, r in rows.items() if k_[0] == h and k_[1] == arm and k_[2] == e02 and k_[3] == 0.05]) for h in Hs]
        allp = [np.median([pick(arm, r) for k_, r in rows.items() if k_[0] == h and k_[1] == arm and k_[2] == e02 and k_[3] == 0.05]) for h in Hs]
        lab = names[arm] + (" (gradient of the sound one)" if arm == "lse_plain" else "")
        line, = ax.plot(Hs, allv, marker="o", linestyle="--" if arm == "lse_plain" else "-", label=lab, zorder=3 if arm == "lse_plain" else 2)
        ax.plot(Hs, allp, linestyle="none", marker="x", markersize=4, color="black", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Hs)
    ax.set_xticklabels([str(h) for h in Hs])
    ax.minorticks_off()
    ax.set_xlabel("horizon (s)")
    ax.set_ylabel("gradient weight on the violating samples")
    ax.legend(fontsize=7, title="eps 0.2, entry depth 0.05; x: the law", title_fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out + "_weight.svg")
    fig.savefig(out + "_weight.pdf")
    with open(out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "theory":
        theory_main(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif mode == "cpu":
        cpu_main(int(sys.argv[2]), sys.argv[3])
    elif mode == "gpu":  # optional dev=<warp device> and only=<arm>/<eps>,... (a smoke test on the CPU)
        opt = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
        gpu_main(sys.argv[2], [a for a in sys.argv[3:] if "=" not in a], opt.get("dev", "cuda:0"), opt["only"].split(",") if "only" in opt else None)
    elif mode == "table":
        gpu = [a for a in sys.argv[5:] if a.startswith("gpu=")]
        table_main(sys.argv[2], sys.argv[3], sys.argv[4], [a for a in sys.argv[5:] if not a.startswith("gpu=")], gpu[0][4:] if gpu else None)
    else:
        raise ValueError("unknown mode " + mode)

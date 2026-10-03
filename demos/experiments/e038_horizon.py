"""E038: the horizon sweep in the constrained problem on the until node (M003 requirement 2 and
note of 2026-10-01 06:25Z; D006 amendment of 06:25Z).

E034's demo scene (experiments/e034_until_demo.py, the person standing 2.0 m from the zone centre)
at horizons H of 10, 20, 40 (and 64) s, at the same sampling (0.02 s; 50 H samples) and physics
(2 ms). At every horizon the wait is the longest that fits, w = H - 1 - T_TR - F_LEN - DWELL
(7.22, 17.22, 37.22, 61.22 s): the robot holds at the pick pose, 0.5 cm outside the zone, from
the end of the entry (1.8 or 2.0 s) to w + 1, then hands over in the window [H - 0.98, H - 0.8] s
and dwells 0.8 s. The entry (its timing, depth, calibration and the four starts), the pick
window's length (1 s), the handover window's length and dwell, the rules and the physics are
E034's. At the until's inner minimum the extended hold adds hold samples (the near-deciding
samples, N) while the entry's samples (k) and the gap above them (Delta) are the same plan
segment at every horizon.

E034's module reads the horizon, the waits and the persons from its globals H, WAITS and PERSONS.
horizon(H) sets them in this process before E034's functions are called, so the plans, scenario,
programs, chain and certificate are E034's code at H; E034's defaults are unchanged in every
other process.

    JAX_PLATFORMS=cpu python -m experiments.e038_horizon starts <H> <out_dir> [<E034's start_dir at 10 s>]
    JAX_PLATFORMS=cpu python -m experiments.e038_horizon regime <H> <start_dir> <out_dir>
    python -m experiments.e038_horizon pretable <out.csv> <H>:<regime.csv>...
    JAX_PLATFORMS=cpu python -m experiments.e038_horizon grads <H> <start_dir> <out_dir> <arms> <eps> [notraces]
    JAX_PLATFORMS=nojax python -m experiments.e038_horizon gradnorm <H> <instance.npz> <out.csv>
    python -m experiments.e038_horizon gradtable <out_prefix> <H>:<grads.csv>:<gradnorm.csv>...
    python -m experiments.e038_horizon setup <H> <start_dir> <out.npz> <depths> <eps> <arm>:<starts>...
    JAX_PLATFORMS=nojax python -m experiments.e038_horizon run <H> <instance.npz> <iterations> <out.npz> <resume.npz|-> <runs|all> [stop]
    JAX_PLATFORMS=cpu python -m experiments.e038_horizon path <H> <instance.npz> <run.npz> <out.csv> <iterates> <runs|all>
    python -m experiments.e038_horizon tables <out_dir> <H>:<dir>:<tag>[,<tag>]...
    JAX_PLATFORMS=cpu python -m experiments.e038_horizon diag <H> <instance.npz> <run.npz> <out.csv> <runs> <iterates> [nocontrol]
    JAX_PLATFORMS=nojax python -m experiments.e038_horizon diaggpu <H> <out.csv> <iterates> <instance.npz>:<run.npz>:<runs>...

Lists are comma-separated; <arm>:<starts> gives each arm its starts (the plain arm on start 0, the
sound arms on 0,1,2,3), so one batch holds every arm of a horizon. The run axis is E034's: eps,
arm, depth, start (start fastest); one batch has one horizon.

path: the regime at the until node along the solver's path (E034's probes/path_regime.py, here
batched): the commands of the given iterates replayed in float64 MuJoCo C from the instance's
start state (the certificate's replay), the exact margins and the smoothed atoms, and at the
inner minimum of the witness with the largest outer weight, at the run's eps: the deepest entry,
the entry samples (inside the zone, t <= 2.0 s), the sound share on them, sparsemax's support and
its hold samples, the gap against gamma / k. until_batch is E034's until_rows in JAX, vectorized
over traces; tests/test_e038_horizon.py checks the two against each other.
"""
import csv
import sys

import numpy as np

from experiments import e034_until_demo as D

ROOT_PREDICTION = {  # M003 note 2026-10-01 06:25Z: root's predicted sound share at eps 0.2 (formula; after the 10 s correction)
    (0.2, 0.05, 10): "formula 0.30; measured 0.21-0.23", (0.2, 0.05, 20): "formula 0.20; about 0.15", (0.2, 0.05, 40): "formula 0.13; about 0.10",
    (0.2, 0.05, 64): "formula about 0.10 (0.09 at 80 s)", (0.2, 0.1, 10): "above 0.4", (0.2, 0.1, 20): "above 0.4", (0.2, 0.1, 40): "above 0.4",
    (0.2, 0.1, 64): "above 0.4", (0.4, 0.05, 20): "falls below 0.11-0.12", (0.4, 0.05, 40): "falls below 0.11-0.12",
    (0.4, 0.1, 20): "falls below 0.155-0.165", (0.4, 0.1, 40): "falls below 0.155-0.165"}


def longest_wait(H):
    return round(H - 1 - D.T_TR - D.F_LEN - D.DWELL, 2)


def horizon(H):
    """Set E034's horizon, wait and person in this process; returns the wait."""
    D.H = int(H)
    D.WAITS = (longest_wait(H),)
    D.PERSONS = ("out",)
    return D.WAITS[0]


def samples(H):
    return int(H) * 50 + 1


# ------------------------------------------------------------------
# the starts and the regime check

def zone_pick(H, X):
    """Exact zone and pick margins (B, T) of float64 states X (B, T, 14) in the scene at H."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from experiments import e022_regime as E
    from sparsemax_diffstl.tasks import workspace as W
    w = horizon(H)
    sc, I = D.scenario(w), D.instance(w, "out")
    with jax.enable_x64(True):
        mx = mjx.put_model(E.plant.model, impl="jax")
        inst = {k: jnp.asarray(I[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
        M = np.asarray(jax.jit(jax.vmap(lambda x: W.margins(mx, E.plant, sc, inst, x)))(jnp.asarray(X, jnp.float64)))
    return M[..., 2], M[..., 0]


def hold_check(H, start_dir, ref_dir=None):
    """Per plan: the entry (deepest zone margin and samples inside on [1, 2] s), the zone margin on
    the hold [2.0, w + 1] s (smallest and largest; the hold pose is 0.025 outside), the smallest
    zone margin up to w + 1 outside [1, 2] s, the smallest pick margin on the pick window, and,
    with ref_dir (E034's starts at 10 s), the largest difference of the commands and of the
    replayed states to E034's plan of the same depth and start at wait 7.22 s before 8.22 s."""
    w = horizon(H)
    z = np.load(start_dir + "/starts.npz")
    Mz, Mp = zone_pick(H, z["X64"])
    t = np.arange(Mz.shape[1]) * D.HS
    span = (t >= 1.0 - 1e-9) & (t <= 2.0 + 1e-9)
    hold = (t >= D.T_HOLD - 1e-9) & (t <= w + 1 + 1e-9)
    upto = t <= w + 1 + 1e-9
    pick = (t >= w - 1e-9) & upto
    inf = np.inf
    out = {"depth": z["depth"], "start": z["start"], "entry_deepest": np.where(span, Mz, inf).min(1), "entry_inside": np.sum(span & (Mz < 0), 1),
           "hold_zone_min": np.where(hold, Mz, inf).min(1), "hold_zone_max": np.where(hold, Mz, -inf).max(1),
           "zone_min_outside_entry": np.where(upto & ~span, Mz, inf).min(1), "pick_min": np.where(pick, Mp, inf).min(1),
           "vmax": np.abs(z["V0"]).max((1, 2))}
    if ref_dir is not None:
        r = np.load(ref_dir + "/starts.npz")
        key = lambda d, s: np.round(d, 6) * 100 + s  # noqa: E731
        sel = np.isclose(r["wait"], 7.22)
        rk = key(r["depth"][sel], r["start"][sel])
        j = np.nonzero(sel)[0][np.argsort(rk)][np.searchsorted(np.sort(rk), key(z["depth"], z["start"]))]
        if not np.array_equal(key(r["depth"][j], r["start"][j]), key(z["depth"], z["start"])):
            raise ValueError("a plan has no counterpart at 10 s")
        n = int(round(8.22 / D.HS))
        out["ref_plan"] = j
        out["dV_before_8.22"] = np.abs(z["V0"][:, :n] - r["V0"][j, :n]).max((1, 2))
        out["dX_before_8.22"] = np.abs(z["X64"][:, :n + 1] - r["X64"][j, :n + 1]).max((1, 2))
    keys = list(out)
    _csv(start_dir + "/hold.csv", keys, np.stack([np.asarray(out[k], float) for k in keys], -1).tolist())
    for k in keys:  # printout, one line per quantity
        print(k, np.round(np.asarray(out[k], float), 6).tolist())


# ------------------------------------------------------------------
# the pre-run table: the regime per horizon with P2's formula and root's prediction beside it

def pretable_main(out, items):
    rows = []
    for item in items:  # over horizons (a handful of files)
        H, path = item.split(":", 1)
        H = int(H)
        w = longest_wait(H)
        with open(path) as fh:
            reg = [r for r in csv.DictReader(fh) if r["person"] == "out" and np.isclose(float(r["wait"]), w)]
        for r in reg:  # over regime rows (table rows)
            eps, k, gap, td = float(r["eps"]), int(r["k_entry"]), float(r["gap"]), float(r["deciding_witness_s"])
            m = int(round(td / D.HS)) + 2
            N = int(round((td - D.T_HOLD) / D.HS)) + 1
            formula = k / (k + N * m ** (-gap / eps))
            rows.append([H, w, eps, float(r["depth"]), int(r["start"]), k, N, m, round(gap, 5), round(gap / eps, 4), round(float(r["share_entry_deciding"]), 4),
                         round(formula, 4), round(float(r["share_entry_deciding"]) / formula, 3), ROOT_PREDICTION.get((eps, float(r["depth"]), H), ""),
                         int(r["support_deciding"]), int(r["support_hold_deciding"]), round(float(r["gamma_over_k"]), 5), r["delta_ge_gamma_over_k"],
                         round(float(r["plain_minus_exact"]), 5), int(r["n_tie"]), round(float(r["root_lse_plain"]), 5), round(float(r["root_lse"]), 5),
                         round(float(r["root_sparsemax"]), 5), round(float(r["root_exact"]), 5), r["deciding"], round(float(r["hold_level"]), 5)])
    head = ["horizon", "wait", "eps", "depth", "start", "k_entry", "N_hold", "m", "gap", "gap_over_eps", "sound_share", "p2_formula", "share_over_formula",
            "root_prediction", "sparsemax_support", "sparsemax_support_hold", "gamma_over_k", "gap_ge_gamma_over_k", "plain_minus_exact", "tied_witnesses",
            "root_plain", "root_sound", "root_sparsemax", "root_exact", "deciding_rule", "hold_level"]
    _csv(out, head, rows)
    cells = {}
    for r in rows:  # group by (eps, depth, horizon) (table rows)
        cells.setdefault((r[2], r[3], r[0]), []).append(r)
    rng = lambda v: str(min(v)) if min(v) == max(v) else str(min(v)) + "-" + str(max(v))  # noqa: E731
    for key in sorted(cells):  # printout, one line per cell
        v = cells[key]
        print("eps", key[0], "depth", key[1], "H", key[2], "wait", v[0][1], "k", rng([r[5] for r in v]), "N", rng([r[6] for r in v]), "m", rng([r[7] for r in v]),
              "gap", rng([r[8] for r in v]), "share", rng([r[10] for r in v]), "formula", rng([r[11] for r in v]), "ratio", rng([r[12] for r in v]),
              "root:", v[0][13] or "-", "support", rng([r[14] for r in v]), "hold in support", rng([r[15] for r in v]), "gamma/k", rng([r[16] for r in v]),
              "plain-exact", rng([r[18] for r in v]), "ties", rng([r[19] for r in v]), "roots plain", rng([r[20] for r in v]), "sound", rng([r[21] for r in v]),
              "sparsemax", rng([r[22] for r in v]), "exact", rng([r[23] for r in v]), "decides", v[0][24])


# ------------------------------------------------------------------
# the gradient table (root's addition of 06:36Z): each arm's gradient with respect to every score

def grads_main(H, start_dir, out_dir, arms, eps_, traces=True):
    """Per start at the longest wait of H, per eps and arm, from the float64 replayed states
    (starts.npz X64): the gradient of the arm's value at the until node (the order conjunct) and at
    the root with respect to every per-sample score (T, P) by reverse-mode automatic
    differentiation through the JAX evaluator (float64, CPU), and, not normalized: its sum over the
    entry samples (zone column, zone margin below 0, t <= 2.0 s), over the hold samples (zone column,
    2.0 s <= t <= w + 1), over the zone column, over the pick column, over every score; the number of
    nonzero partial derivatives and the number carrying 90 % of their absolute sum, over every score
    and over the zone column; the arm's values at the until and the root and the exact ones. Writes
    grads<H>.csv and traces<H>.npz (the states, the exact margins and the smoothed scores of every
    start) in out_dir."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from experiments import e022_regime as E
    from sparsemax_diffstl import stl
    from sparsemax_diffstl.core_study import methods
    from sparsemax_diffstl.tasks import workspace as W
    jax.config.update("jax_enable_x64", True)
    w = horizon(H)
    z = np.load(start_dir + "/starts.npz")
    sel = np.nonzero(np.isclose(z["wait"], w))[0]
    X = np.asarray(z["X64"][sel], np.float64)
    sc, I = D.scenario(w), D.instance(w, "out")
    inst = {k: jnp.asarray(I[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
    mx = mjx.put_model(E.plant.model, impl="jax")
    Mg = np.asarray(jax.jit(jax.vmap(lambda x: W.margins(mx, E.plant, sc, inst, x)))(jnp.asarray(X)))
    Zs = np.asarray(jax.jit(jax.vmap(lambda x: W.scores(mx, E.plant, sc, inst, x)))(jnp.asarray(X)))
    if traces:
        np.savez(out_dir + "/traces" + str(H) + ".npz", X64=X, margins=Mg, scores=Zs, depth=z["depth"][sel], start=z["start"][sel], wait=w, horizon=H,
                 plan=sel, start_file=start_dir + "/starts.npz", columns="atoms in tasks.workspace.layout order: 0 pick, 1 handover, 2 zone, then speed, sep, slow")
    prog = E.core_program(sc, len(I["human_radii"]))
    root = prog.steps[prog.root]
    r_idx = np.asarray(root.index[0, :root.count[0]])

    def vals(Z, sem, e):
        v = stl.evaluate(prog, Z, sem, e)
        return v[-1][0], jnp.concatenate([v[j] for j in root.sources], -1)[r_idx][2]

    B, T, P = Zs.shape
    t = np.arange(T) * D.HS
    entry = (Mg[:, :, 2] < 0) & (t[None] <= D.T_HOLD + 1e-9)
    hold = (t >= D.T_HOLD - 1e-9) & (t <= w + 1 + 1e-9)
    ex_root, ex_until = (np.asarray(a) for a in jax.jit(jax.vmap(lambda Z: vals(Z, "exact", None)))(jnp.asarray(Mg)))

    def counts(g):
        a = np.abs(g.reshape(len(g), -1))
        c = np.cumsum(-np.sort(-a, 1), 1)
        return np.sum(a > 0, 1), np.argmax(c >= 0.9 * c[:, -1:], 1) + 1

    rows = []
    for e in eps_:  # over eps (two)
        for arm in arms:  # over arms (formula structure: one semantics each)
            sem = methods.SEMANTICS[arm]
            f = jax.jit(jax.vmap(lambda Z: (vals(Z, sem, e), jax.grad(lambda Y: vals(Y, sem, e)[1])(Z), jax.grad(lambda Y: vals(Y, sem, e)[0])(Z))))
            (v_root, v_until), g_until, g_root = jax.tree_util.tree_map(np.asarray, f(jnp.asarray(Zs)))
            cols = {"horizon": np.full(B, H), "wait": np.full(B, w), "eps": np.full(B, e), "arm": np.full(B, arm), "depth": z["depth"][sel], "start": z["start"][sel],
                    "k_entry": entry.sum(1), "n_hold": np.full(B, hold.sum()), "until_value": v_until, "until_exact": ex_until, "root_value": v_root, "root_exact": ex_root}
            for name, g in (("until", g_until), ("root", g_root)):  # the two nodes
                cols[name + "_sum_entry"] = np.sum(np.where(entry, g[:, :, 2], 0.0), 1)
                cols[name + "_sum_hold"] = np.sum(np.where(hold[None], g[:, :, 2], 0.0), 1)
                cols[name + "_sum_zone"] = g[:, :, 2].sum(1)
                cols[name + "_sum_pick"] = g[:, :, 0].sum(1)
                cols[name + "_sum_all"] = g.sum((1, 2))
                cols[name + "_nonzero_all"], cols[name + "_n90_all"] = counts(g)
                cols[name + "_nonzero_zone"], cols[name + "_n90_zone"] = counts(g[:, :, 2])
            keys = list(cols)
            rows += np.stack([np.asarray(cols[k], object) for k in keys], -1).tolist()
            print("eps", e, arm, "until entry", np.round(cols["until_sum_entry"], 4).tolist(), "hold", np.round(cols["until_sum_hold"], 4).tolist(),
                  "all", np.round(cols["until_sum_all"], 4).tolist(), "root entry", np.round(cols["root_sum_entry"], 4).tolist(),
                  "root all", np.round(cols["root_sum_all"], 4).tolist(), "n90 zone", cols["until_n90_zone"].tolist(), flush=True)
    _csv(out_dir + "/grads" + str(H) + ".csv", keys, rows)
    print("scores", Zs.shape)


def gradnorm_main(H, instance, out_csv):
    """Through the simulator at the start, every run of the instance: the smoothed root (float32,
    the solver's Warp chain on the GPU), the norm of its control gradient (Chain.pullback with weight
    1) over all controls and over the controls before 2.0 s (the entry), from 2.0 s to w + 1 (the
    hold) and after w + 1, the norm of the effort term's gradient lam 2 V / N (E033's solver), and the
    cosine of each run's control gradient with sparsemax's for the same eps, depth and start (over all
    controls and over the entry block). The gradients are saved next to the CSV (_grad.npz)."""
    horizon(H)
    I = D.load(instance)
    progs, chain, referee = D.build(I)
    V = np.asarray(I["V0"], np.float32)
    rho = chain.forward(V)
    gV = chain.pullback(np.ones(len(V), np.float32))
    N = V.shape[1]
    t = np.arange(N) * D.HS
    w = D.WAITS[0]
    blocks = {"entry": t < D.T_HOLD - 1e-9, "hold": (t >= D.T_HOLD - 1e-9) & (t < w + 1 - 1e-9), "after": t >= w + 1 - 1e-9}
    ge = D.LAM * 2.0 * V / N
    cols = {"horizon": np.full(len(V), H), "eps": I["run_eps"], "arm": I["run_method"], "depth": I["run_depth"], "start": I["run_start"], "rho": rho,
            "grad_norm": np.sqrt(np.sum(gV * gV, (1, 2))), "effort_grad_norm": np.sqrt(np.sum(ge * ge, (1, 2))), "lam": np.full(len(V), D.LAM)}
    for k, m in blocks.items():  # three blocks of control intervals
        cols["grad_norm_" + k] = np.sqrt(np.sum(gV[:, m] ** 2, (1, 2)))
    key = np.char.add(np.char.add(I["run_eps"].astype(str), "/"), np.char.add(I["run_depth"].astype(str), np.char.add("/", I["run_start"].astype(str))))
    ref = np.nonzero(I["run_method"] == "sparsemax")[0]
    j = ref[np.argsort(key[ref])][np.searchsorted(np.sort(key[ref]), key)] if len(ref) else np.arange(len(V))
    if len(ref) and not np.array_equal(key[j], key):
        raise ValueError("a run has no sparsemax counterpart")
    gn = gV.reshape(len(V), -1)
    cols["cosine_to_sparsemax"] = np.sum(gn * gn[j], 1) / np.maximum(np.linalg.norm(gn, axis=1) * np.linalg.norm(gn[j], axis=1), 1e-30)
    cols["cosine_to_sparsemax_entry"] = np.sum((gV * gV[j])[:, blocks["entry"]], (1, 2)) / np.maximum(cols["grad_norm_entry"] * cols["grad_norm_entry"][j], 1e-30)
    np.savez(out_csv[:-4] + "_grad.npz", gV=gV, rho=rho, run_eps=I["run_eps"], run_method=I["run_method"], run_depth=I["run_depth"], run_start=I["run_start"])
    keys = list(cols)
    _csv(out_csv, keys, np.stack([np.asarray(cols[k], object) for k in keys], -1).tolist())
    for i in range(len(V)):  # printout, one line per run
        print([cols[k][i] for k in keys])
    print("seconds", chain.times)


def gradtable_main(out_prefix, items):
    """The gradient table per horizon, eps, arm, depth and start: grads<H>.csv (the score gradients)
    joined with gradnorm<H>.csv (the control gradients through the simulator) on (eps, arm, depth,
    start); <out_prefix>.csv, and <out_prefix>.txt with ranges over the four starts."""
    rows, head = [], None
    for item in items:  # over horizons (a handful of files)
        H, gpath, npath = item.split(":")
        with open(gpath) as fh:
            g = list(csv.DictReader(fh))
        with open(npath) as fh:
            n = {(float(r["eps"]), r["arm"], float(r["depth"]), int(r["start"])): r for r in csv.DictReader(fh)}
        for r in g:  # over table rows
            c = n.get((float(r["eps"]), r["arm"], float(r["depth"]), int(r["start"])), {})
            extra = {k: c.get(k, "") for k in ("rho", "grad_norm", "grad_norm_entry", "grad_norm_hold", "grad_norm_after", "effort_grad_norm",
                                                "cosine_to_sparsemax", "cosine_to_sparsemax_entry")}
            extra["entry_part_of_unit_step"] = float(c["grad_norm_entry"]) / float(c["grad_norm"]) if c else ""
            row = dict(r, **{"control_" + k: v for k, v in extra.items()})
            head = head or list(row)
            rows.append([row[k] for k in head])
    _csv(out_prefix + ".csv", head, rows)
    cells = {}
    for r in rows:  # group by (eps, arm, depth, horizon) (table rows)
        d = dict(zip(head, r))
        cells.setdefault((float(d["eps"]), d["arm"], float(d["depth"]), int(d["horizon"])), []).append(d)

    def rng(v, nd=4):
        v = [round(float(x), nd) for x in v if x != ""]
        return "-" if not v else (str(min(v)) if min(v) == max(v) else str(min(v)) + " to " + str(max(v)))
    lines = ["Ranges over the four starts. Sums of partial derivatives of the arm's value (not normalized) over the zone column's",
             "entry samples (margin below 0, t <= 2.0 s), hold samples (2.0 s to w + 1) and over every score; n90: the number of",
             "partial derivatives on the zone column carrying 90 % of their absolute sum. Control gradients through the Warp chain",
             "(float32, GPU) at the start: the norm of d(root)/dV over all controls and its part on the controls before 2.0 s; the",
             "entry part of the unit step is their ratio; the effort term's gradient norm; the cosine with sparsemax's gradient."]
    for k in sorted(cells):  # one line per cell
        v = cells[k]
        f = lambda c, nd=4: rng([d[c] for d in v], nd)  # noqa: E731
        lines.append("eps " + str(k[0]) + " " + k[1] + " depth " + str(k[2]) + " H " + str(k[3]) + ": until value " + f("until_value") + " (exact "
                     + f("until_exact") + "), root " + f("root_value") + " (exact " + f("root_exact") + "); until sums: entry " + f("until_sum_entry")
                     + ", hold " + f("until_sum_hold") + ", all " + f("until_sum_all") + ", nonzero on zone " + f("until_nonzero_zone", 0) + ", n90 on zone "
                     + f("until_n90_zone", 0) + "; root sums: entry " + f("root_sum_entry") + ", hold " + f("root_sum_hold") + ", all " + f("root_sum_all")
                     + ", n90 over every score " + f("root_n90_all", 0) + "; control gradient norm " + f("control_grad_norm", 2) + ", on the entry block "
                     + f("control_grad_norm_entry", 2) + ", entry part of the unit step " + f("control_entry_part_of_unit_step", 3) + ", effort term "
                     + f("control_effort_grad_norm", 7) + ", cosine to sparsemax " + f("control_cosine_to_sparsemax", 3) + " (entry block "
                     + f("control_cosine_to_sparsemax_entry", 3) + ")")
    with open(out_prefix + ".txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------
# setup and the runs

def setup_main(H, start_dir, out, depths, eps_, arm_starts):
    """E034's setup with per-arm starts at the longest wait of H (the person standing clear)."""
    w = horizon(H)
    z = np.load(start_dir + "/starts.npz")
    rows = []
    for e in eps_:  # the run axis: eps, arm, depth, start (start fastest)
        for arm, starts in arm_starts:
            for d in depths:
                for s in starts:
                    p = int(np.nonzero(np.isclose(z["wait"], w) & np.isclose(z["depth"], d) & (z["start"] == s))[0][0])
                    rows.append(("out", e, w, arm, d, s, p))
    if z["V0"].shape[1] != samples(H) - 1:
        raise ValueError("the starts hold " + str(z["V0"].shape[1]) + " intervals, not " + str(samples(H) - 1))
    pi = np.asarray([r[6] for r in rows])
    I = D.instance(w, "out")
    hc = np.broadcast_to(I["human_centres"], (len(rows),) + I["human_centres"].shape).astype(np.float32)
    np.savez(out, x0=np.broadcast_to(z["x0"], (len(rows), 14)).astype(np.float32), V0=z["V0"][pi], hc=hc, pick=I["pick"], handover=I["handover"],
             hr=I["human_radii"], n_h=len(I["human_radii"]), start_dir=start_dir, plan=pi, horizon=int(H),
             run_person=np.asarray([r[0] for r in rows]), run_eps=np.asarray([r[1] for r in rows]), run_wait=np.asarray([r[2] for r in rows]),
             run_method=np.asarray([r[3] for r in rows]), run_depth=np.asarray([r[4] for r in rows]), run_start=np.asarray([r[5] for r in rows]))
    print("runs", len(rows), "horizon", H, "wait", w, "hc", hc.shape)
    for i, r in enumerate(rows):  # printout, one line per run
        print(i, r)


def run_main(H, path, iterations, out, resume, runs, stop=False):
    horizon(H)
    z = np.load(path)
    if int(z["horizon"]) != int(H) or z["hc"].shape[1] != samples(H):
        raise ValueError("the instance is for another horizon")
    D.run_main(path, iterations, out, resume, runs, stop)


# ------------------------------------------------------------------
# the regime along the solver's path

def until_batch(zone_s, pick_s, zone_m, pick_m, entry, eps, w, hold_from, batch=8):
    """E034's until_rows (experiments/e034_until_demo.py) for B traces at once: smoothed atoms
    zone_s, pick_s (B, T), exact margins zone_m, pick_m (B, T), entry masks (B, T), eps (B,).
    float64 JAX (lax.map over traces in groups of `batch`). Returns a dict of (B,) arrays: the
    deciding witness (largest outer weight of the sound arm), the inner arity m there, the sound
    share on the entry samples there, k, the deepest entry margin, the hold level and the gap, gamma
    over k, sparsemax's support and its hold and entry samples, the sound and plain values at the
    until and the exact one, the tied witnesses."""
    import jax
    import jax.numpy as jnp

    from sparsemax_diffstl.operators import sparsemax_weights
    T = zone_s.shape[1]
    wit = np.arange(int(round(w / D.HS)), int(round((w + 1) / D.HS)) + 1)
    m = wit + 2
    M = len(wit)
    cols = np.arange(T + 1)
    valid = cols[None, :] < m[:, None]
    t = np.arange(T) * D.HS
    hold = np.concatenate([[False], t >= hold_from - 1e-9])
    hold_w = jnp.asarray(hold[None] & valid)
    hold_med = (t >= hold_from - 1e-9) & (np.arange(T) <= wit[-1])
    big = 1e9

    def one(a):
        zs, ps, zm, pm, ent0, e = a
        Xs = jnp.where(valid, jnp.where(cols[None] == 0, ps[wit][:, None], jnp.concatenate([jnp.zeros(1), zs])[None]), big)
        Xm = jnp.where(valid, jnp.where(cols[None] == 0, pm[wit][:, None], jnp.concatenate([jnp.zeros(1), zm])[None]), big)
        inner_exact = Xm.min(1)
        beta = jnp.log(m) / e
        q = jnp.where(valid, jnp.exp(-beta[:, None] * (Xs - Xs.min(1, keepdims=True))), 0.0)
        soft = Xs.min(1) - jnp.log(q.sum(1)) / beta
        q = q / q.sum(1, keepdims=True)
        ent = jnp.concatenate([jnp.zeros(1, bool), ent0])
        gam = 2 * e / (1 - 1 / m)
        p = jax.vmap(lambda z_, g, v: sparsemax_weights(jnp.where(v, z_, -big), g))(-Xs, gam, valid)
        supp = p > 0
        bo = jnp.log(M) / e
        c = soft.max()
        plain = c + jnp.log(jnp.sum(jnp.exp(bo * (soft - c)))) / bo
        d = jnp.argmax(jnp.exp(bo * (soft - c)))
        k = ent0.sum()
        deep = jnp.where(k > 0, jnp.min(jnp.where(ent0, zm, jnp.inf)), jnp.nan)
        level = jnp.nanmedian(jnp.where(hold_med, zm, jnp.nan))
        share = (q * ent).sum(1)
        return {"deciding_witness_s": jnp.asarray(wit * D.HS)[d], "m": jnp.asarray(m)[d], "share_entry_deciding": jnp.where(k > 0, share[d], jnp.nan),
                "k_entry": k, "entry_depth": -deep, "hold_level": level, "gap": level - deep, "gamma_over_k": gam[d] / jnp.maximum(k, 1),
                "support_deciding": supp[d].sum(), "support_hold_deciding": (supp[d] & hold_w[d]).sum(), "support_entry_deciding": (supp[d] & ent).sum(),
                "sound_until": plain - jnp.log(M) / bo, "plain_until": plain, "exact_until": inner_exact.max(),
                "n_tie": jnp.sum(inner_exact >= inner_exact.max() - D.TIE)}

    with jax.enable_x64(True):
        f = lambda *a: jax.lax.map(one, a, batch_size=batch)  # noqa: E731
        out = jax.jit(f)(*(jnp.asarray(x) for x in (zone_s, pick_s, zone_m, pick_m)), jnp.asarray(entry), jnp.asarray(eps, jnp.float64))
        return {k: np.asarray(v) for k, v in out.items()}


def path_main(H, instance, run_path, out, iterates, runs="all"):
    """The regime at the until node along the path of the runs of run_path (module docstring)."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from experiments import e022_regime as E
    from sparsemax_diffstl.tasks import workspace as W
    w = horizon(H)
    zi, z = np.load(instance), np.load(run_path)
    sel = np.arange(len(z["run_method"])) if runs == "all" else np.asarray([int(i) for i in runs.split(",")])
    K = np.asarray([int(k) for k in iterates.split(",")])
    K = K[K < z["V"].shape[1]]
    V = z["V"][sel][:, K]  # (R, K, N, 7)
    x0 = np.asarray(zi["x0"][int(z["index"][sel[0]])], np.float64)
    if not np.all(zi["x0"][z["index"][sel]] == zi["x0"][int(z["index"][sel[0]])]):
        raise ValueError("the runs start from different states")
    X = D.replay(V.reshape((-1,) + V.shape[2:]), x0)  # (R K, T, 14), the certificate's replay
    sc, I = D.scenario(w), D.instance(w, "out")
    with jax.enable_x64(True):
        mx = mjx.put_model(E.plant.model, impl="jax")
        inst = {k: jnp.asarray(I[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
        f = jax.jit(lambda x: (W.margins(mx, E.plant, sc, inst, x)[:, jnp.asarray([0, 2])], W.scores(mx, E.plant, sc, inst, x)[:, jnp.asarray([0, 2])]))
        Mg, Zs = (np.asarray(a) for a in jax.lax.map(f, jnp.asarray(X), batch_size=8))
    t = np.arange(X.shape[1]) * D.HS
    entry = (Mg[:, :, 1] < 0) & (t[None] <= D.T_HOLD + 1e-9)
    eps = np.repeat(z["run_eps"][sel].astype(np.float64), len(K))
    u = until_batch(Zs[:, :, 1], Zs[:, :, 0], Mg[:, :, 1], Mg[:, :, 0], entry, eps, w, D.T_HOLD)
    hold = (t >= D.T_HOLD - 1e-9) & (t <= w + 1 + 1e-9)
    u["hold_zone_min"] = np.where(hold[None], Mg[:, :, 1], np.inf).min(1)
    rr, kk = np.repeat(sel, len(K)), np.tile(K, len(sel))
    tr = z["trace"]
    L = np.where(kk < tr.shape[1], tr[rr, np.minimum(kk, tr.shape[1] - 1), 0], np.nan)  # the merit at iterate k (before its step)
    head = ["horizon", "eps", "arm", "depth", "start", "run", "iterate", "certificate", "merit"] + list(u)
    rows = [[H, z["run_eps"][r], str(z["run_method"][r]), z["run_depth"][r], z["run_start"][r], int(z["index"][r]), int(k), z["referee64"][r, k], L[i]]
            + [u[c][i] for c in u] for i, (r, k) in enumerate(zip(rr, kk))]
    _csv(out, head, rows)
    for i, row in enumerate(rows):  # printout, one line per run and iterate
        print(row[1:5], "iterate", row[6], "certificate", round(float(row[7]), 4), "merit", round(float(row[8]), 5),
              "deepest", round(-float(u["entry_depth"][i]), 4), "inside", int(u["k_entry"][i]), "gap", round(float(u["gap"][i]), 4),
              "support", int(u["support_deciding"][i]), "hold in support", int(u["support_hold_deciding"][i]),
              "sound share", round(float(u["share_entry_deciding"][i]), 3), "hold zone min", round(float(u["hold_zone_min"][i]), 4))


# ------------------------------------------------------------------
# root's diagnosis of 06:53Z: sparsemax's path at eps 0.4 against 0.2 over the horizon

RULES = ("separation", "slowdown", "order", "handover")  # E022's CONJUNCTS order


def rule_columns(P, n_h):
    """Boolean (4, P): the score columns that feed each rule (tasks.workspace.layout): separation
    the sep atoms; slow-down the slow and speed atoms; order the zone and pick atoms; handover the
    handover atom. Every atom feeds exactly one rule."""
    from experiments import e022_regime as E
    nr = E.N_R
    c = np.arange(P)
    sep = (c >= 3 + nr) & (c < 3 + nr + nr * n_h)
    slow = (c >= 3 + nr + nr * n_h) | ((c >= 3) & (c < 3 + nr))
    return np.stack([sep, slow, (c == 0) | (c == 2), c == 1])


def diag_main(H, instance, run_path, out, runs, iterates, chunk=4, control=True):
    """For the given runs and recorded iterates (one arm each, at its eps):
    - from the certificate's float64 MuJoCo C replay: k (zone samples below 0, t <= 2.0 s), the hold
      level (median zone margin on [2.0, w + 1]) and the gap Delta to the deepest entry sample, gamma / k
      (gamma = 2 eps / (1 - 1/m), m the inner arity at the first witness), and by reverse-mode AD of
      the arm's until value and root with respect to every score (float64): the sums on the entry,
      hold and other zone samples, the hold samples with a nonzero until gradient (in the support),
      and the root's gradient sums per rule (its score columns);
    - through the simulator: the control gradient of the arm's root along the MJX float64 rollout of
      the same commands on the CPU (tasks.panda.rollout, the implicitly differentiated step), split by
      rule (the root's score cotangent restricted to the rule's columns, pulled back by one VJP) and
      by time segment (controls before 2.0 s, on [2.0, w + 1), after w + 1): the norms. Only order and
      handover are pulled back (the cost); separation and slow-down are 0 where the root's gradient on
      all their scores is exactly 0 and NaN otherwise (root_nonzero_<rule> counts those entries). The
      total is the sum of the order and handover parts.
    The traces are processed `chunk` at a time (lax.map); the rollout is tasks.panda.rollout's composition of
    interval_map with each interval under jax.checkpoint (the 40 s pullback otherwise needs about 47 GB per trace)."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from experiments import e022_regime as E
    from sparsemax_diffstl import stl
    from sparsemax_diffstl.core_study import methods
    from sparsemax_diffstl.tasks import panda as Pd
    from sparsemax_diffstl.tasks import workspace as W
    jax.config.update("jax_enable_x64", True)
    w = horizon(H)
    zi, z = np.load(instance), np.load(run_path)
    sel = np.asarray([int(i) for i in runs.split(",")])
    K = np.asarray([int(k) for k in iterates.split(",")])
    K = K[K < z["V"].shape[1]]
    arms = {str(a) for a in z["run_method"][sel]}
    if len(arms) != 1:
        raise ValueError("one arm per call")
    arm = arms.pop()
    V = np.asarray(z["V"][sel][:, K], np.float64).reshape((-1,) + z["V"].shape[2:])  # (R K, N, 7)
    eps = np.repeat(z["run_eps"][sel].astype(np.float64), len(K))
    x0 = np.asarray(zi["x0"][int(z["index"][sel[0]])], np.float64)
    X = D.replay(V, x0)
    sc, I = D.scenario(w), D.instance(w, "out")
    mx = mjx.put_model(E.plant.model, impl="jax")
    inst = {k: jnp.asarray(I[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
    prog = E.core_program(sc, len(I["human_radii"]))
    root = prog.steps[prog.root]
    r_idx = np.asarray(root.index[0, :root.count[0]])
    sem = methods.SEMANTICS[arm]

    def vals(Zt, e):
        v = stl.evaluate(prog, Zt, sem, e)
        return v[-1][0], jnp.concatenate([v[j] for j in root.sources], -1)[r_idx][2]

    margins = lambda x: W.margins(mx, E.plant, sc, inst, x)  # noqa: E731
    scores = lambda x: W.scores(mx, E.plant, sc, inst, x)  # noqa: E731
    umax = jnp.asarray(Pd.torque_limit(), jnp.float64)
    n_sub = Pd.substeps(D.H_CTRL)
    n_h = len(I["human_radii"])
    rc = jnp.asarray(rule_columns(3 + E.N_R + 2 * E.N_R * n_h, n_h)[2:])  # order and handover; see below for the other two

    def score_part(a):
        x, e = a
        Zt = scores(x)
        g_until = jax.grad(lambda Y: vals(Y, e)[1])(Zt)
        g_root = jax.grad(lambda Y: vals(Y, e)[0])(Zt)
        return margins(x), g_until, g_root

    f_h = jax.checkpoint(Pd.interval_map(mx, n_sub))  # tasks.panda.rollout with each interval recomputed in the backward (memory)

    def body(x, ui):
        y = f_h(x, ui)
        return y, y

    def rollout(u):
        _, Xr = jax.lax.scan(body, jnp.asarray(x0), u)
        return jnp.concatenate([jnp.asarray(x0)[None], Xr])

    def control_part(a):
        v, e = a
        Zt, pull = jax.vjp(lambda u: scores(rollout(u * umax)), v)
        G = jax.grad(lambda Y: vals(Y, e)[0])(Zt)
        return jax.vmap(lambda m: pull(jnp.where(m[None], G, 0.0))[0])(rc)  # (2, N, 7): order, handover

    Mg, Gu, Gr = (np.asarray(t) for t in jax.jit(lambda a, b: jax.lax.map(score_part, (a, b), batch_size=chunk))(jnp.asarray(X), jnp.asarray(eps)))
    if control:
        Gc = np.asarray(jax.jit(lambda a, b: jax.lax.map(control_part, (a, b), batch_size=chunk))(jnp.asarray(V), jnp.asarray(eps)))  # (B, 2, N, 7)
    else:  # the control split comes from diaggpu (the Warp chain); here NaN
        Gc = np.full((len(V), 2) + V.shape[1:], np.nan)
    P = Mg.shape[-1]
    T = Mg.shape[1]
    t = np.arange(T) * D.HS
    zm = Mg[:, :, 2]
    entry = (zm < 0) & (t[None] <= D.T_HOLD + 1e-9)
    hold = (t >= D.T_HOLD - 1e-9) & (t <= w + 1 + 1e-9)
    k = entry.sum(1)
    deep = np.where(entry, zm, np.inf).min(1)
    level = np.nanmedian(np.where(hold[None], zm, np.nan), 1)
    m0 = int(round(w / D.HS)) + 2
    gam = 2 * eps / (1 - 1 / m0)
    cols = rule_columns(P, n_h)
    if cols.shape[1] != rc.shape[1]:
        raise ValueError("the score layout differs from rule_columns")
    tc = np.arange(V.shape[1]) * D.HS
    seg = {"entry": tc < D.T_HOLD - 1e-9, "hold": (tc >= D.T_HOLD - 1e-9) & (tc < w + 1 - 1e-9), "after": tc >= w + 1 - 1e-9}
    out_cols = {"horizon": np.full(len(V), H), "arm": np.full(len(V), arm), "eps": eps, "depth": np.repeat(z["run_depth"][sel], len(K)),
                "start": np.repeat(z["run_start"][sel], len(K)), "run": np.repeat(z["index"][sel], len(K)), "iterate": np.tile(K, len(sel)),
                "certificate": z["referee64"][sel][:, K].ravel(), "k_entry": k, "deepest": np.where(k > 0, deep, np.nan), "hold_level": level,
                "gap": np.where(k > 0, level - deep, np.nan), "gamma_over_k": np.where(k > 0, gam / np.maximum(k, 1), np.nan),
                "until_sum_entry": np.sum(np.where(entry, Gu[:, :, 2], 0.0), 1), "until_sum_hold": np.sum(np.where(hold[None], Gu[:, :, 2], 0.0), 1),
                "until_sum_zone_other": np.sum(np.where(~entry & ~hold[None], Gu[:, :, 2], 0.0), 1), "until_sum_pick": Gu[:, :, 0].sum(1),
                "until_hold_in_support": np.sum(hold[None] & (Gu[:, :, 2] != 0), 1)}
    for i, r in enumerate(RULES):  # four rules (formula structure)
        out_cols["root_sum_" + r] = np.sum(np.where(cols[i][None, None], Gr, 0.0), (1, 2))
    # the control gradient by rule is pulled back for order and handover only; separation and slow-down get 0 where the
    # root's gradient on every one of their scores is exactly 0 (a zero cotangent pulls back to 0) and NaN otherwise
    for i, r in enumerate(RULES):  # four rules (formula structure)
        out_cols["root_nonzero_" + r] = np.sum(np.where(cols[i][None, None], Gr != 0, False), (1, 2))
    for i, r in enumerate(RULES):  # four rules (formula structure)
        if i < 2:
            zero = np.where(out_cols["root_nonzero_" + r] == 0, 0.0, np.nan)
            out_cols["control_" + r] = zero
            for sname in seg:  # three time segments
                out_cols["control_" + r + "_" + sname] = zero
            continue
        out_cols["control_" + r] = np.sqrt(np.sum(Gc[:, i - 2] ** 2, (1, 2)))
        for sname, m in seg.items():  # three time segments
            out_cols["control_" + r + "_" + sname] = np.sqrt(np.sum(Gc[:, i - 2][:, m] ** 2, (1, 2)))
    tot = Gc.sum(1)
    out_cols["control_total"] = np.sqrt(np.sum(tot ** 2, (1, 2)))
    for sname, m in seg.items():  # three time segments
        out_cols["control_total_" + sname] = np.sqrt(np.sum(tot[:, m] ** 2, (1, 2)))
    keys = list(out_cols)
    _csv(out, keys, np.stack([np.asarray(out_cols[c], object) for c in keys], -1).tolist())
    for i in range(len(V)):  # printout, one line per run and iterate
        print("eps", eps[i], "depth", out_cols["depth"][i], "iterate", out_cols["iterate"][i], "cert", round(float(out_cols["certificate"][i]), 4), "k", k[i],
              "gap", round(float(out_cols["gap"][i]), 4), "gamma/k", round(float(out_cols["gamma_over_k"][i]), 4), "hold in support",
              out_cols["until_hold_in_support"][i], "until entry", round(float(out_cols["until_sum_entry"][i]), 3), "hold", round(float(out_cols["until_sum_hold"][i]), 3),
              "root by rule", [round(float(out_cols["root_sum_" + r][i]), 3) for r in RULES], "control order entry/hold/after",
              [round(float(out_cols["control_order_" + s_][i]), 3) for s_ in seg], "handover entry/hold/after",
              [round(float(out_cols["control_handover_" + s_][i]), 3) for s_ in seg], flush=True)


def diaggpu_main(H, out_csv, iterates, sources):
    """The control-gradient split of diag_main through the solver's own chain (the fork's float32 rollout and
    pullback on the GPU, constrained_warp.Chain). sources: <instance.npz>:<run.npz>:<runs>; every (run, iterate)
    becomes one world with the recorded commands of that iterate. Per world: the smoothed root, and the root's
    control gradient split by rule (the root's score cotangent restricted to the rule's columns, pulled back
    through the atoms' tape and the plant; rules whose cotangent is exactly zero are not pulled back and get 0) and
    by time segment; check: the sum of the rule parts against the full pullback (Chain.forward and pullback)."""
    import warp as wp
    w = horizon(H)
    K = [int(k) for k in iterates.split(",")]
    parts = []
    for src in sources:  # a handful of (instance, record) pairs
        ipath, rpath, runs = src.split(":")
        zi, z = np.load(ipath), np.load(rpath)
        for r in [int(x) for x in runs.split(",")]:  # a handful of runs (table rows)
            j = int(np.nonzero(z["index"] == r)[0][0])
            for k in K:  # a handful of recorded iterates
                parts.append({"x0": zi["x0"][r], "V0": z["V"][j, k], "hc": zi["hc"][r], "run_person": zi["run_person"][r], "run_eps": zi["run_eps"][r],
                              "run_wait": zi["run_wait"][r], "run_method": zi["run_method"][r], "run_depth": zi["run_depth"][r], "run_start": zi["run_start"][r],
                              "plan": zi["plan"][r], "run": r, "iterate": k, "certificate": z["referee64"][j, k], "record": rpath})
    parts.sort(key=lambda d: (str(d["run_method"]), float(d["run_eps"])))
    tmp = out_csv[:-4] + "_instance.npz"
    zi0 = np.load(sources[0].split(":")[0])
    np.savez(tmp, **{k: np.stack([d[k] for d in parts]) for k in ("x0", "V0", "hc", "plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_start")},
             pick=zi0["pick"], handover=zi0["handover"], hr=zi0["hr"], n_h=zi0["n_h"], horizon=H)
    I = D.load(tmp)
    progs, chain, referee = D.build(I)
    V = np.asarray(I["V0"], np.float32)
    n, N = V.shape[:2]
    rho_full = chain.forward(V)
    g_full = chain.pullback(np.ones(n, np.float32))
    X = chain.plant.rollout(chain.x0, V)
    q, v = chain._states(X, True)
    tape = wp.Tape()
    Z = chain.pred.scores(q, v, tape)
    gZ = np.zeros((n, chain.T, chain.pred.P), np.float32)
    rho = np.zeros(n, np.float32)
    for a, b, ev in chain.ev:  # over (method, eps) groups
        r_, G = ev.gradient(Z[a:b])
        gZ[a:b] = G.numpy()
        rho[a:b] = r_.numpy()[:, 0]
    masks = rule_columns(chain.pred.P, int(I["n_h"]))
    tc = np.arange(N) * D.HS
    seg = {"entry": tc < D.T_HOLD - 1e-9, "hold": (tc >= D.T_HOLD - 1e-9) & (tc < w + 1 - 1e-9), "after": tc >= w + 1 - 1e-9}
    cols = {"horizon": np.full(n, H), "arm": I["run_method"], "eps": I["run_eps"], "depth": I["run_depth"], "start": I["run_start"],
            "run": np.asarray([d["run"] for d in parts]), "iterate": np.asarray([d["iterate"] for d in parts]),
            "certificate": np.asarray([d["certificate"] for d in parts]), "rho": rho, "rho_forward": rho_full}
    total = np.zeros_like(V)
    with wp.ScopedDevice(chain.device):
        for i, rule in enumerate(RULES):  # four rules (formula structure)
            cot = gZ * masks[i][None, None]
            cols["root_sum_" + rule] = cot.sum((1, 2))
            if not np.any(cot):
                gV = np.zeros_like(V)
            else:
                tape.zero()
                tape.backward(grads={Z: wp.array(cot, dtype=float)})
                C = np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(n, chain.T, -1)
                gV, _ = chain.plant.vjp(C)
            total += gV
            cols["control_" + rule] = np.sqrt(np.sum(gV ** 2, (1, 2)))
            for sname, m in seg.items():  # three time segments
                cols["control_" + rule + "_" + sname] = np.sqrt(np.sum(gV[:, m] ** 2, (1, 2)))
    cols["control_total"] = np.sqrt(np.sum(total ** 2, (1, 2)))
    for sname, m in seg.items():  # three time segments
        cols["control_total_" + sname] = np.sqrt(np.sum(total[:, m] ** 2, (1, 2)))
    cols["rule_sum_vs_full_rel"] = np.sqrt(np.sum((total - g_full) ** 2, (1, 2))) / np.maximum(np.sqrt(np.sum(g_full ** 2, (1, 2))), 1e-30)
    keys = list(cols)
    _csv(out_csv, keys, np.stack([np.asarray(cols[c], object) for c in keys], -1).tolist())
    for i in range(n):  # printout, one line per world
        print([cols[c][i] for c in ("horizon", "arm", "eps", "depth", "iterate", "certificate", "rho")], "order entry/hold/after",
              [round(float(cols["control_order_" + s_][i]), 3) for s_ in seg], "handover entry/hold/after", [round(float(cols["control_handover_" + s_][i]), 3) for s_ in seg],
              "sep", round(float(cols["control_separation"][i]), 3), "slow", round(float(cols["control_slowdown"][i]), 3), "check", float(cols["rule_sum_vs_full_rel"][i]))


# ------------------------------------------------------------------
# the tables against the horizon

def tables_main(out_dir, items):
    """From E034's per-batch tables (<tag>_attenuation.csv, <tag>_soundness.csv, <tag>_timing.csv, written
    by E034's tables and report modes) of each horizon, at that horizon's longest wait: the first exactly
    safe iterate per arm, depth and eps against the horizon (horizon_attenuation.csv), the plain arm's
    claims (horizon_soundness.csv) and the seconds per iterate (horizon_timing.csv)."""
    att, snd, tim = [], [], []
    for item in items:  # over horizons (a handful)
        H, d, tags = item.split(":")
        H = int(H)
        w = longest_wait(H)
        for tag in tags.split(","):  # over batches of that horizon
            for name, dst in (("_attenuation.csv", att), ("_soundness.csv", snd)):
                with open(d + "/" + tag + name) as fh:
                    dst += [dict(r, horizon=H, batch=tag) for r in csv.DictReader(fh) if np.isclose(float(r["wait"]), w)]
            with open(d + "/" + tag + "_timing.csv") as fh:
                tim += [dict(r, horizon=H, batch=tag) for r in csv.DictReader(fh)]
    key = lambda r: (float(r["eps"]), r["arm"], float(r["depth"]), int(r["horizon"]))  # noqa: E731
    att.sort(key=key)
    snd.sort(key=key)
    ha = ["eps", "arm", "depth", "horizon", "wait", "runs", "first_safe_by_start", "runs_safe", "first_safe_median", "runs_staying_safe", "exact_end_by_start",
          "sound_share_entry_regime", "batch"]
    _csv(out_dir + "/horizon_attenuation.csv", ha, [[r[k] for k in ha] for r in att])
    hs = ["eps", "arm", "depth", "horizon", "wait", "runs", "runs_claiming_while_unsafe", "first_claim_by_start", "iterates_claimed_while_unsafe_by_start",
          "deepest_exact_while_claiming", "runs_ending_unsafe", "runs_ending_claiming_while_unsafe", "sound_share_entry_regime", "batch"]
    _csv(out_dir + "/horizon_soundness.csv", hs, [[r[k] for k in hs] for r in snd])
    ht = ["horizon", "batch", "runs", "iterates"] + [k for k in tim[0] if k not in ("record", "runs", "iterates", "horizon", "batch")] + ["record"]
    _csv(out_dir + "/horizon_timing.csv", ht, [[r.get(k, "") for k in ht] for r in tim])
    for r in att:  # printout
        print("attenuation eps", r["eps"], r["arm"], "depth", r["depth"], "H", r["horizon"], "first safe", r["first_safe_by_start"], "median", r["first_safe_median"],
              "share", r["sound_share_entry_regime"])
    for r in snd:  # printout
        if r["arm"] == "lse_plain":
            print("soundness eps", r["eps"], "depth", r["depth"], "H", r["horizon"], "claiming while unsafe", r["runs_claiming_while_unsafe"], "of", r["runs"],
                  "first claim", r["first_claim_by_start"], "iterates", r["iterates_claimed_while_unsafe_by_start"], "deepest", r["deepest_exact_while_claiming"],
                  "ending unsafe", r["runs_ending_unsafe"], "ending claiming", r["runs_ending_claiming_while_unsafe"])
    for r in tim:  # printout
        print("timing H", r["horizon"], r["batch"], "runs", r["runs"], "seconds per iterate", round(float(r["total"]), 3))


def _csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    floats = lambda a: tuple(float(x) for x in a.split(","))  # noqa: E731
    if mode == "starts":
        horizon(int(sys.argv[2]))
        D.starts_main(sys.argv[3])
        hold_check(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif mode == "regime":
        horizon(int(sys.argv[2]))
        D.regime_main(sys.argv[3], sys.argv[4])
    elif mode == "grads":
        grads_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], tuple(sys.argv[5].split(",")), floats(sys.argv[6]), len(sys.argv) < 8 or sys.argv[7] != "notraces")
    elif mode == "gradnorm":
        gradnorm_main(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif mode == "gradtable":
        gradtable_main(sys.argv[2], sys.argv[3:])
    elif mode == "pretable":
        pretable_main(sys.argv[2], sys.argv[3:])
    elif mode == "setup":
        arm_starts = [(a.split(":")[0], tuple(int(s) for s in a.split(":")[1].split(","))) for a in sys.argv[7:]]
        setup_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], floats(sys.argv[5]), floats(sys.argv[6]), arm_starts)
    elif mode == "run":
        run_main(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], len(sys.argv) > 8 and sys.argv[8] == "stop")
    elif mode == "path":
        path_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] if len(sys.argv) > 7 else "all")
    elif mode == "tables":
        tables_main(sys.argv[2], sys.argv[3:])
    elif mode == "diag":
        diag_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], control=len(sys.argv) < 9 or sys.argv[8] != "nocontrol")
    elif mode == "diaggpu":
        diaggpu_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise ValueError("unknown mode " + mode)

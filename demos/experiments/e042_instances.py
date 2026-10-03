"""E042: E037's 16 instances on the formulation that E040's verdict selects, at 10 s and at the longest wait at 20 and 40 s
(entry E042; M003 note and D006 amendment of 2026-10-01 12:21Z). An adapter: the solvers, the scene, the check and the tables
are the modules it imports, unedited; it adds the starts at a longer horizon, the merge of instance files and the run on
E037's instance layout with either formulation.

    JAX_PLATFORMS=cpu python -m experiments.e042_instances starts <H> <start_dir_10s> <out_dir>
    JAX_PLATFORMS=cpu python -m experiments.e042_instances regime <H> <start_dir> <out_dir>
    JAX_PLATFORMS=cpu python -m experiments.e042_instances check <H> <start_dir> <regime.csv> <out_dir>
    JAX_PLATFORMS=cpu python -m experiments.e042_instances roots <H> <start_dir> <choice.csv> <regime.csv> <out_dir> <eps>
    python -m experiments.e042_instances setup <H> <start_dir> <choice.csv> <out.npz> <instances> <arms> <eps>
    python -m experiments.e042_instances merge <out.npz> <instance.npz>...
    JAX_PLATFORMS=nojax python -m experiments.e042_instances run <H> <conj|root> <instance.npz> <iterations> <out.npz>
    python -m experiments.e042_instances tables <out_dir> <tag> <pairs.csv> <run.npz>...
    python -m experiments.e042_instances horizon <out_dir> <tag> <H>:<runs.csv>...

Horizon. H is 10, 20 or 40 s; for H > 10, experiments.e038_horizon.horizon(H) sets E034's globals (H, WAITS = the longest
wait H - 2.78 s) in the process before anything else, as E038 and E039 did.

starts: E037's 16 instances at the longest wait of H, one plan per instance and depth: E037's keyframes with the same via,
hold and handover configurations and the same calibrated entry parameter `a` as the instance's 10 s start (no new
calibration), the hold extended to w + 1; the inverse kinematics of the entry configuration, the plan, the computed-torque
tracking (float32) and the float64 MuJoCo C replay as E037's starts mode. Checks per plan, E037's (with the person standing
clear): inverse kinematics residual, commands inside the torque box, the deepest zone margin on [T_VIA, T_HOLD] within
DEPTH_TOL of -depth, no other zone violation up to w + 1, the pick margin at least TRACK_MIN on the pick window, separation,
slow-down and handover at least TRACK_MIN; and the largest difference of the commands, the replayed states and the planned
states to the instance's 10 s start at wait 7.22 s before 8.22 s (the extension leaves the plan before w + 1 unchanged).

regime, check, roots, setup: E037's modes (experiments.e037_instances.regime_main, experiments.e037_person.check_main,
roots_main, setup_main) at H. For H > 10 the person is E039's long-horizon script: E037's VISIT with length 100 s (E039's
visit_h0.45_L100.0), so the hand is near the arm on the whole hold, from 2.3 s to w + 0.7 s, and far while the arm moves;
at 10 s this equals E037's VISIT at every wait (the visit starts at 2.3 s in both). The hand's place follows
RULE-hand-place.txt of gates/E037/2026-10-01T0849Z applied to the plans of H (two per instance). check also writes
prerun.csv: per instance and depth at the chosen place, the pass condition, the plain log-sum-exp's claim under four
constraints (each conjunct's value at least 0) at eps 0.2, and the until node's numbers at eps 0.2 and 0.4 from regime.csv.

merge: one instance file from several (same start directory and person radii), runs sorted by wait, eps, arm (ARMS6 order),
instance, depth, so that every (arm, eps, wait) is one contiguous block; `source` lists (file, row) per run.

run: conj, the four conjuncts as four constraints (sparsemax_diffstl.constrained_conj through experiments.e040_conj.build);
root, the smoothed root as one constraint (constrained_warp, E034's build). The record has E037's keys (run_instance, pick,
handover) and, for conj, E040's per-conjunct arrays (conj_r, conj_w, conj_r_next, conj_r_ref, conj_mu, conj_nu).

tables: experiments.e037_person.tables_main. horizon: per horizon (the longest wait only), depth and arm, the first iterate
with a non-negative certificate by instance, its median (never counted as K = 101), runs keeping the certificate, runs
with a false claim; sparsemax minus each other arm per instance (earlier, equal, later; the median difference and a 95%
percentile interval from N_BOOT resamples of the instances, one generator seeded BOOT_SEED per horizon, cells in the order
depth, arm); the plain arm's claims at iterate 0 while the certificate is negative.
"""
import csv
import json
import sys
import time

import numpy as np

from experiments import e034_until_demo as U
from experiments import e037_instances as X
from experiments import e037_person as P37
from experiments import e038_horizon as H38

LONG_VISIT = 100.0
W10 = 7.22
K_RECORD = 101
OTHERS = ("lse_plain", "lse", "gm_pm01", "gm_pm10", "gm_exp")


def set_horizon(H):
    """E034's globals at H (E038's horizon) and, for H > 10, E039's whole-hold visit; returns the longest wait."""
    if int(H) == 10:
        return W10
    w = H38.horizon(int(H))
    P37.VISIT = dict(P37.VISIT, length=LONG_VISIT)
    return w


# ------------------------------------------------------------------
# the starts at H

def starts_main(H, start_dir, out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from experiments import e022_regime as E
    from sparsemax_diffstl import stl
    from sparsemax_diffstl.tasks import panda as P
    from sparsemax_diffstl.tasks import workspace as W
    t_start = time.perf_counter()
    z = np.load(start_dir + "/starts.npz")
    sel = np.nonzero(np.isclose(z["wait"], W10))[0]
    w = set_horizon(H)
    n = len(sel)
    plant = E.plant
    m = plant.model
    sc = U.scenario(w)
    inst, depth, a, ray = z["instance"][sel], z["depth"][sel], z["a"][sel], z["ray"][sel]
    q0, x0 = z["q0"], z["x0"]
    q_hold, q_hand, q_via = z["q_hold"][inst], z["q_hand"][inst], z["q_via"][inst]
    lo, hi = m.jnt_range[:, 0] + W.JOINT_MARGIN, m.jnt_range[:, 1] - W.JOINT_MARGIN
    zone, r = np.asarray(sc.zone), sc.zone_radius
    person = U.instance(w, "out")
    with jax.enable_x64(True):  # inverse kinematics, margins and exact conjuncts in float64, as E037's starts
        mx64 = mjx.put_model(m, impl="jax")
        hum = {k: jnp.asarray(person[k]) for k in ("human_centres", "human_radii")}
        ik_j = jax.jit(lambda tg, q: W.ik(mx64, plant, sc.robot_spacing, tg, q, jnp.asarray(lo), jnp.asarray(hi)))
        margins_j = jax.jit(jax.vmap(lambda X_, p, hd: W.margins(mx64, plant, sc, {"pick": p, "handover": hd, **hum}, X_)))
        q_in, miss_in = (np.asarray(v) for v in ik_j(jnp.asarray(zone + (r - a[:, None] * r) * ray, jnp.float64), jnp.asarray(q_hold, jnp.float64)))
    entry_t = np.asarray([U.ENTRY[round(float(d), 2)] for d in depth])
    tk, qk = U.keyframes(np.full(n, w), np.broadcast_to(q0, (n, 7)), q_via, q_in, q_hold, q_hand, entry_t)
    t = np.arange(H38.samples(H)) * U.HS
    Xp = U.plan_states(tk, qk, t)
    um = np.asarray(P.torque_limit(), np.float32)
    mx32 = mjx.put_model(m, impl="jax")
    track = jax.jit(jax.vmap(lambda Qs: E.track(mx32, P.substeps(U.H_CTRL), jnp.asarray(x0, jnp.float32), Qs, jnp.asarray(um))))
    V = np.clip(np.asarray(track(jnp.asarray(Xp, jnp.float32))) / um, -1.0, 1.0).astype(np.float32)
    print("tracked", V.shape, "seconds", round(time.perf_counter() - t_start, 1), flush=True)
    X64 = U.replay(V, x0)
    with jax.enable_x64(True):
        Mg = np.asarray(margins_j(jnp.asarray(X64, jnp.float64), jnp.asarray(z["pick"][sel], jnp.float64), jnp.asarray(z["handover"][sel], jnp.float64)))
        prog = E.core_program(sc, len(person["human_radii"]))
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])
        conj = np.asarray(jax.jit(jax.vmap(lambda M_: (lambda v: jnp.concatenate([v[s] for s in root.sources], -1)[..., r_idx])(stl.evaluate(prog, M_, "exact"))))(jnp.asarray(Mg)))
    entry_win = (t >= U.T_VIA - 1e-9) & (t <= U.T_HOLD + 1e-9)
    upto = t <= w + 1 + 1e-9
    pickw = (t >= w - 1e-9) & upto
    deepest = -np.where(entry_win, Mg[:, :, 2], np.inf).min(1)
    k_in = np.sum(entry_win & (Mg[:, :, 2] < 0), 1)
    other_zone = np.where(upto & ~entry_win, Mg[:, :, 2], np.inf).min(1)
    pick_min = np.where(pickw, Mg[:, :, 0], np.inf).min(1)
    vmax = np.abs(V).max((1, 2))
    ci = {c: i for i, c in enumerate(E.CONJUNCTS)}
    nb = int(round((W10 + 1) / U.HS))  # 411 intervals: the plan and commands before 8.22 s do not depend on the wait
    cols = {"plan": np.arange(n), "plan10": sel, "instance": inst, "draw": z["draw"][sel], "depth": depth, "wait": np.full(n, w),
            "ik_miss": miss_in, "vmax": vmax, "deepest": deepest, "depth_error": np.abs(deepest - depth), "entry_samples": k_in,
            "entry_samples_10s": z["entry_samples"][sel], "other_zone_min": other_zone, "pick_min": pick_min,
            **{"c_" + c: conj[:, i] for c, i in ci.items()}, "deciding": np.asarray(E.CONJUNCTS)[np.argmin(conj, 1)],
            "dV_before_8.22": np.abs(V[:, :nb] - z["V0"][sel, :nb]).max((1, 2)), "dX_before_8.22": np.abs(X64[:, :nb + 1] - z["X64"][sel, :nb + 1]).max((1, 2)),
            "dXplan_before_8.22": np.abs(Xp[:, :nb + 1] - z["Xplan"][sel, :nb + 1]).max((1, 2))}
    checks = {"ik": miss_in <= W.IK_TOLERANCE, "torque": vmax < 1.0 - 1e-6, "depth": cols["depth_error"] <= X.DEPTH_TOL, "one_entry": other_zone >= 0,
              "pick": pick_min >= X.TRACK_MIN,
              "rules": np.minimum(np.minimum(conj[:, ci["separation"]], conj[:, ci["slowdown"]]), conj[:, ci["handover"]]) >= X.TRACK_MIN}
    cols.update({"ok_" + k: v for k, v in checks.items()})
    cols["ok"] = np.all(np.stack(list(checks.values())), 0)
    U._csv(out + "/starts_check.csv", list(cols), np.stack([np.asarray(v, object) for v in cols.values()], -1).tolist())
    np.savez(out + "/starts.npz", V0=V, X64=X64, Xplan=Xp, x0=x0, wait=np.full(n, w), depth=depth, instance=inst, draw=z["draw"][sel], a=a,
             pick=z["pick"][sel], handover=z["handover"][sel], ray=ray, q_hold=z["q_hold"], q_hand=z["q_hand"], q_via=z["q_via"], q0=q0, kept=z["kept"],
             entry_samples=k_in, vmax=vmax, conj=conj, start=np.zeros(n, int), plan10=sel, horizon=int(H),
             meta=X.dumps({"from": start_dir, "horizon": int(H), "wait": w, "seconds": time.perf_counter() - t_start}))
    for k in ("ok", "ik_miss", "vmax", "depth_error", "entry_samples", "other_zone_min", "pick_min", "dV_before_8.22", "dX_before_8.22", "dXplan_before_8.22"):  # printout
        print(k, np.round(np.asarray(cols[k], float), 6).tolist())
    print("deciding", sorted(set(cols["deciding"].tolist())), "order", np.round(conj[:, ci["order"]], 5).tolist())
    print("plans", n, "passing every check", int(cols["ok"].sum()), "horizon", H, "wait", w, "seconds", round(time.perf_counter() - t_start, 1))


# ------------------------------------------------------------------
# the pre-run check at H (E037's modes) and its summary

def check_main(H, start_dir, regime_csv, out):
    set_horizon(H)
    print("visit", P37.VISIT, "waits", U.WAITS, flush=True)
    P37.check_main(start_dir, regime_csv, out)
    prerun_summary(out, regime_csv)


def prerun_summary(out, regime_csv):
    with open(out + "/check.csv") as fh:
        rows = list(csv.DictReader(fh))
    with open(out + "/choice.csv") as fh:
        place = {int(r["instance"]): float(r["place"]) for r in csv.DictReader(fh)}
    with open(out + "/pairs.csv") as fh:
        pairs = {(int(r["instance"]), round(float(r["depth"]), 6)): r for r in csv.DictReader(fh)}
    with open(regime_csv) as fh:
        reg = list(csv.DictReader(fh))
    keep = ("share_entry_deciding", "k_entry", "gap", "gamma_over_k", "support_deciding", "support_entry_deciding", "support_hold_deciding",
            "root_lse_plain", "root_lse", "root_sparsemax", "plain_minus_exact")
    res = []
    for (i, dp), pr in sorted(pairs.items()):  # instances x depths (table rows)
        v = [r for r in rows if int(r["instance"]) == i and np.isclose(float(r["depth"]), dp) and np.isclose(float(r["place"]), place[i])]
        plain4 = min(min(float(r["plain_" + c]) for c in ("separation", "slowdown", "order", "handover")) for r in v)
        row = {"instance": i, "depth": dp, "place": place[i], "how": pr["how"], "pass": pr["pass"], "order_decides": pr["order_decides"],
               "root_exact": min(float(r["root_exact"]) for r in v), "sep_min": pr["sep_min"], "slow_min": pr["slow_min"],
               "plain_root_0.2": pr["plain_root_min"], "plain_conjunct_min_0.2": plain4, "plain_claim_four_0.2": plain4 >= 0,
               "hand_gap_plateau_min": pr["hand_gap_plateau_min"], "hand_gap_hold": pr["hand_gap_hold"], "near": pr["near"], "until_inside_0.2": pr["until_inside_0.2"]}
        for e in (0.2, 0.4):  # eps (columns)
            g = [r for r in reg if int(r["instance"]) == i and np.isclose(float(r["depth"]), dp) and np.isclose(float(r["eps"]), e)]
            row.update({k + "_" + str(e): g[0][k] for k in keep})
        res.append(row)
    U._csv(out + "/prerun.csv", list(res[0]), [list(r.values()) for r in res])
    for dp in U.DEPTHS:  # printout per depth
        v = [r for r in res if np.isclose(r["depth"], dp)]
        f = lambda k: [round(float(r[k]), 4) for r in v]  # noqa: E731
        print("depth", dp, "pass", sum(r["pass"] == "True" for r in v), "of", len(v), "plain four-constraint claim", sum(r["plain_claim_four_0.2"] for r in v),
              "until inside at 0.2", sum(r["until_inside_0.2"] == "True" for r in v))
        print("  plain root 0.2", f("plain_root_0.2"))
        print("  plain conjunct min 0.2", f("plain_conjunct_min_0.2"))
        print("  sound share 0.2", f("share_entry_deciding_0.2"), "0.4", f("share_entry_deciding_0.4"))
        print("  sparsemax support 0.2", [int(r["support_deciding_0.2"]) for r in v], "hold samples in it", [int(r["support_hold_deciding_0.2"]) for r in v])
        print("  hand gap on the hold", f("hand_gap_hold"), "separation min", f("sep_min"), "slow-down min", f("slow_min"))
    print("pairs passing", sum(r["pass"] == "True" for r in res), "of", len(res))


# ------------------------------------------------------------------
# instance files and the runs

def merge_main(out, paths):
    keys = ("x0", "V0", "hc", "pick", "handover", "plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_instance", "run_start")
    zs = [np.load(p) for p in paths]
    for z in zs[1:]:  # over files (a handful)
        if str(z["start_dir"]) != str(zs[0]["start_dir"]) or not np.array_equal(z["hr"], zs[0]["hr"]) or int(z["n_h"]) != int(zs[0]["n_h"]):
            raise ValueError("the files need the same start directory and person radii")
    cat = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    src = [(p, i) for p, z in zip(paths, zs) for i in range(len(z["run_method"]))]
    rank = np.asarray([P37.ARMS6.index(str(a)) for a in cat["run_method"]])
    order = np.lexsort((cat["run_depth"], cat["run_instance"], rank, cat["run_eps"], cat["run_wait"]))
    if len(set(zip(cat["run_eps"], cat["run_method"], cat["run_wait"], cat["run_depth"], cat["run_instance"]))) != len(order):
        raise ValueError("a run appears twice")
    places = {}
    for z in zs:  # over files
        places.update(json.loads(str(z["places"])))
    np.savez(out, hr=zs[0]["hr"], n_h=zs[0]["n_h"], start_dir=zs[0]["start_dir"], places=json.dumps(places), source=json.dumps([src[i] for i in order]),
             **{k: v[order] for k, v in cat.items()})
    print("runs", len(order), "from", paths, "instances", sorted(set(cat["run_instance"].tolist())), "arms", sorted(set(cat["run_method"].tolist())),
          "waits", sorted(set(cat["run_wait"].tolist())))


def run_main(H, form, path, iterations, out):
    import socket

    import warp as wp

    from sparsemax_diffstl import constrained_warp as CW
    set_horizon(H)
    t0 = time.perf_counter()
    I = X.load(path, "all")
    if I["hc"].shape[1] != H38.samples(H):
        raise ValueError("the instance holds " + str(I["hc"].shape[1]) + " samples, not " + str(H38.samples(H)))
    if form == "conj":
        from experiments import e040_conj as C40
        from sparsemax_diffstl import constrained_conj as CC
        chain, referee = C40.build(I)
        state = CC.init_state(I["V0"], U.ALPHA0, chain.K)
        solver = CC
    elif form == "root":
        _, chain, referee = U.build(I)
        state = CW.init_state(I["V0"], U.ALPHA0)
        solver = CW
    else:
        raise ValueError("form is conj or root")
    setup_seconds = time.perf_counter() - t0

    def log(k, rec, r64, wall):
        print("iterate", k, "seconds", round(wall, 2), "accepted", int(rec[:, 10].sum()), "of", len(rec), "rho", np.round(rec[:, 1], 4).tolist(),
              "referee64", np.round(r64, 4).tolist(), flush=True)

    res = solver.solve(chain, referee, state, iterations, U.LAM, U.DELTA, log=log)
    meta = {"host": socket.gethostname(), "device": str(chain.device), "warp": wp.config.version, "instance": path, "horizon": int(H), "form": form,
            "iterations": state["k"], "conjuncts": ("separation", "slowdown", "order", "handover") if form == "conj" else ("root",), "every": CW.EVERY,
            "lam": U.LAM, "delta": U.DELTA, "alpha0": U.ALPHA0, "trials": CW.TRIALS, "armijo_c": CW.ARMIJO_C, "mu0": CW.MU0, "mu_max": CW.MU_MAX,
            "viol_tol": CW.VIOL_TOL, "trace_keys": CW.TRACE_KEYS, "groups": I["groups"], "setup_seconds": setup_seconds, "seconds_keys": res["seconds_keys"]}
    extra = {"conj_" + k: v for k, v in res["conj"].items()} if form == "conj" else {}
    np.savez(out, meta=X.dumps(meta), index=I["index"], trace=res["trace"], V=np.concatenate([np.asarray(I["V0"], np.float32)[:, None], res["V"]], 1),
             referee64=res["referee64"], conjuncts64=res["conjuncts64"], seconds=res["seconds"], wall=res["wall"], alpha=state["alpha"], nu=state["nu"],
             mu=state["mu"], k=state["k"], **extra, **{k: I[k] for k in ("plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_instance",
                                                                          "run_start", "pick", "handover")})
    med = np.median(res["seconds"], 0)
    print("setup seconds", round(setup_seconds, 2), "seconds per iterate (median by component)", X.dumps(dict(zip(res["seconds_keys"], np.round(med, 4)))),
          "total", round(float(np.median(res["seconds"].sum(1))), 3))


# ------------------------------------------------------------------
# the table against the horizon

def horizon_main(out_dir, tag, items):
    rows, pairs, plain = [], [], []
    for item in items:  # horizons (a handful of files)
        H, path = item.split(":", 1)
        with open(path) as fh:
            R = list(csv.DictReader(fh))
        w = max(float(r["wait"]) for r in R)
        R = [r for r in R if np.isclose(float(r["wait"]), w)]
        rng = np.random.default_rng(X.BOOT_SEED)
        for dp in sorted(set(float(r["depth"]) for r in R)):  # depths
            cell = [r for r in R if np.isclose(float(r["depth"]), dp)]
            fs = {}
            for a in P37.ARMS6:  # arms
                v = sorted((r for r in cell if r["arm"] == a), key=lambda r: int(r["instance"]))
                if not v:
                    continue
                f = np.asarray([int(r["first_safe"]) for r in v])
                fs[a] = (np.asarray([int(r["instance"]) for r in v]), f)
                rows.append([int(H), w, dp, a, len(v), " ".join(str(x) for x in f), float(np.median(np.where(f < 0, K_RECORD, f))), int(np.sum(f < 0)),
                             sum(r["stays_safe"] == "True" for r in v), sum(int(r["false_claim_iterates"]) > 0 for r in v),
                             round(min(float(r["exact_end"]) for r in v), 4), round(float(np.median([float(r["exact_end"]) for r in v])), 4)])
                if a == "lse_plain":
                    plain.append([int(H), w, dp, len(v), sum(r["claim0_false"] == "True" for r in v), sum(int(r["false_claim_iterates"]) > 0 for r in v),
                                  sum(float(r["exact_end"]) < U.DELTA for r in v)])
            for b in OTHERS:  # sparsemax against each other arm
                if b not in fs or "sparsemax" not in fs:
                    continue
                if not np.array_equal(fs[b][0], fs["sparsemax"][0]):
                    raise ValueError("the two arms need the same instances")
                q = X.paired(fs["sparsemax"][1], fs[b][1], K_RECORD, rng)
                pairs.append([int(H), w, dp, b, q["n"], q["earlier"], q["equal"], q["later"], q["median"], q["lo"], q["hi"]])
    U._csv(out_dir + "/" + tag + "_arms.csv", ["horizon", "wait", "depth", "arm", "instances", "first_safe_by_instance", "first_safe_median_never_as_K", "never_safe",
                                               "runs_staying_safe", "runs_with_a_false_claim", "exact_end_min", "exact_end_median"], rows)
    U._csv(out_dir + "/" + tag + "_sparsemax_vs.csv", ["horizon", "wait", "depth", "arm_b", "instances", "sparsemax_earlier", "equal", "sparsemax_later",
                                                       "median_sparsemax_minus_b", "lo95", "hi95"], pairs)
    U._csv(out_dir + "/" + tag + "_plain.csv", ["horizon", "wait", "depth", "instances", "claim_at_0_while_negative", "any_claim_while_negative", "ending_negative"], plain)
    print("bootstrap seed", X.BOOT_SEED, "resamples", X.N_BOOT, "never safe counted as", K_RECORD)
    for r in rows:  # printout
        print("arm", r)
    for r in pairs:  # printout
        print("sparsemax_vs", r)
    for r in plain:  # printout
        print("plain", r)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "starts":
        starts_main(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif mode == "regime":
        set_horizon(int(sys.argv[2]))
        X.regime_main(sys.argv[3], sys.argv[4])
    elif mode == "check":
        check_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "roots":
        set_horizon(int(sys.argv[2]))
        P37.roots_main(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7]))
    elif mode == "setup":
        set_horizon(int(sys.argv[2]))
        print("visit", P37.VISIT, "waits", U.WAITS)
        P37.setup_main(sys.argv[3], sys.argv[4], sys.argv[5], tuple(int(x) for x in sys.argv[6].split(",")), tuple(sys.argv[7].split(",")), float(sys.argv[8]))
    elif mode == "merge":
        merge_main(sys.argv[2], sys.argv[3:])
    elif mode == "run":
        run_main(int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    elif mode == "tables":
        P37.tables_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif mode == "horizon":
        horizon_main(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        raise ValueError("unknown mode " + mode)

"""E039: the person back in the constrained study's scene, the right hand inside the zone during the wait
(M003 note and D006 amendment of 2026-10-01 06:52Z). The robot's plans are E034's 32 saved starts
(examples.e034_until_demo, gate E034/2026-10-01T0337Z/starts6); only the person changes.

    JAX_PLATFORMS=cpu python -m examples.e039_person_zone scan <start_dir> <geometries.json> <out_dir> <tag> [roots]
    JAX_PLATFORMS=cpu python -m examples.e039_person_zone check <start_dir> <geometries.json> <name> <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e039_person_zone setup <start_dir> <geometries.json> <name> <out.npz> <waits> <depths> <arms> <eps>
    JAX_PLATFORMS=nojax python -m examples.e034_until_demo run <instance.npz> ...   (the runs, unchanged)
    python -m examples.e039_person_zone tables <out_dir> <tag> <regime.csv> <run.npz>...

geometries.json is a list of dicts: "name", and either "person": "out" (E034's person standing 2.0 m away,
tasks.workspace.until_script) or the keys of tasks.workspace.until_work_script (phi, stand, depth, reach_z,
release, t_move, hover, hover_z).

Terms. The *gap* between the arm and the person is the smallest distance between the surfaces of the two
sphere covers the rules use (tasks.workspace: robot_spheres, human.spheres), |c_i - h_j| - r_i - r_j over
robot spheres i and person spheres j, in metres; the *hand gap* takes j = the hand's sphere only (the
distal sphere of the right forearm). Separation is violated when a gap is below d_min = 0.10 m; the
slow-down rule asks, per robot sphere, for a gap of at least d_slow = 0.30 m or a speed of the sphere's
centre of at most v_slow = 0.25 m/s. The exact rule values are the referee's (tasks.workspace.margins):
separation the smallest (d - R) / R, slow-down the smallest over spheres of max(min_j (d - S) / S, speed
score), R = r_i + r_j + d_min, S = r_i + r_j + d_slow, as the formula evaluates them.

Phases of a plan with wait w and entry depth delta (ENTRY of e034_until_demo): approach [0, 1.0) s; dip
[1.0, t_hold) s, the move to the zone and the entry, t_hold the arrival at the hold pose (2.0 s at depth
0.05, 1.8 s at 0.10); hold [t_hold, w + 1] s, at the pick pose 0.5 cm outside the zone; transfer
(w + 1, w + 1.76] s, from the pick pose to the handover pose inside the zone; handover (w + 1.76, 10] s,
the handover window, its 0.8 s dwell and the rest of the horizon at the handover pose.
"""
import csv
import json
import sys
import time

import numpy as np

from examples import e034_until_demo as D

PHASES = ("approach", "dip", "hold", "transfer", "handover")
HAND = 17  # index of the hand's sphere in the person's cover (the distal sphere of forearm_r; tests/test_person_zone.py)
BELTA = ("gm_pm01", "gm_pm10", "gm_exp")


def load_geometries(path):
    with open(path) as fh:
        return {g["name"]: g for g in json.load(fh)}


def instance(w, geo):
    from sparsemax_dstl.tasks import workspace as W
    if geo.get("person") == "out":
        return D.instance(w, "out")
    if geo.get("kind") == "visit":
        return W.until_visit_instance(D.scenario(w), D.DIRECTION, D.HANDOVER_RADIUS, geo)
    return W.until_work_instance(D.scenario(w), D.DIRECTION, D.HANDOVER_RADIUS, geo)


def phase_masks(w, depth, t):
    """Boolean (5, T) masks of PHASES at times t for wait w and entry depth."""
    th = D.ENTRY[round(float(depth), 2)][2]
    return np.stack([t < D.T_VIA, (t >= D.T_VIA) & (t < th), (t >= th) & (t <= w + 1 + 1e-9), (t > w + 1 + 1e-9) & (t <= w + 1.76 + 1e-9),
                     t > w + 1.76 + 1e-9])


def robot_centres(X):
    """Robot sphere centres (B, T, S_r, 3) and speeds (B, T, S_r) along float64 states X (B, T, 14)."""
    import jax
    import jax.numpy as jnp
    from functools import partial
    from mujoco import mjx

    from sparsemax_dstl.tasks import workspace as W
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    from sparsemax_dstl.tasks import workspace_program as Wp
    with jax.enable_x64(True):
        mx = mjx.put_model(Wp.plant.model, impl="jax")
        f = partial(Wm.points, mx, Wp.plant, W.Scenario().robot_spacing)
        one = lambda x: jax.jvp(f, (x[:7],), (x[7:],))  # noqa: E731
        (_, C), (_, Cd) = jax.jit(jax.vmap(jax.vmap(one)))(jnp.asarray(X, jnp.float64))
        return np.asarray(C), np.linalg.norm(np.asarray(Cd), axis=-1)


def gaps(C, rr, hc, hr):
    """Per sample the smallest surface gap (B, T) between the robot spheres C (B, T, S_r, 3) and the
    person's spheres hc (T, S_h, 3), and the same to the hand's sphere (B, T)."""
    d = np.linalg.norm(C[:, :, :, None] - hc[None, :, None], axis=-1) - rr[:, None] - hr[None, None]  # (B, T, S_r, S_h)
    return d.min((2, 3)), d[..., HAND].min(-1)


def rule_values(M, n_r, n_h):
    """Exact separation and slow-down per sample (..., T) from the referee's margins M (..., T, P)."""
    sep = M[..., 3 + n_r:3 + n_r + n_r * n_h].min(-1)
    slow = M[..., 3 + n_r + n_r * n_h:].reshape(M.shape[:-1] + (n_r, n_h)).min(-1)
    return sep, np.maximum(slow, M[..., 3:3 + n_r]).min(-1)


def scan_main(start_dir, geo_path, out, tag, roots=False, arm_names=None):
    """Per geometry, wait, depth and start: the gaps, separation and slow-down per phase (exact, float64
    replay of E034's starts), the four rules' exact values and the deciding rule; with roots, the roots of
    the plain log-sum-exp, the sound log-sum-exp and sparsemax at eps 0.4 and 0.2 and of the generalized mean arms."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.jax import methods
    from sparsemax_dstl.tasks import workspace as W
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    from sparsemax_dstl.tasks import workspace_program as Wp
    jax.config.update("jax_enable_x64", True)
    t0 = time.perf_counter()
    geos = list(load_geometries(geo_path).values())
    z = np.load(start_dir + "/starts.npz")
    plant = Wp.plant
    mx = mjx.put_model(plant.model, impl="jax")
    X = z["X64"]
    T = X.shape[1]
    t = np.arange(T) * D.HS
    C, speed = robot_centres(X)
    rr = W.robot_spheres(plant, W.Scenario().robot_spacing)["radius"]
    n_r = len(rr)
    arms = [(a, e) for a in ("lse_plain", "lse", "sparsemax") for e in D.EPS] + [("gm_pm01", None), ("gm_pm10", None)] + [("gm_exp", e) for e in D.EPS]
    if arm_names is not None:  # a subset of the arms (long horizons)
        arms = [(a, e) for a, e in arms if a in arm_names]
    rows = []
    for w in D.WAITS:  # over waits (four)
        sc = D.scenario(w)
        sel = np.nonzero(np.isclose(z["wait"], w))[0]
        insts = [instance(w, g) for g in geos]  # over geometries (scan settings)
        n_h = len(insts[0]["human_radii"])
        hc_all = np.stack([I["human_centres"] for I in insts])  # (G, T, S_h, 3)
        hr = insts[0]["human_radii"]
        prog = Wp.core_program(sc, n_h)
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])
        base = {k: jnp.asarray(insts[0][k]) for k in ("pick", "handover", "human_radii")}
        marg = jax.jit(jax.vmap(jax.vmap(lambda hc, x: Wm.margins(mx, plant, sc, dict(base, human_centres=hc), x), (None, 0)), (0, None)))
        scor = jax.jit(jax.vmap(jax.vmap(lambda hc, x: Wm.scores(mx, plant, sc, dict(base, human_centres=hc), x), (None, 0)), (0, None)))
        Mg = np.asarray(marg(jnp.asarray(hc_all), jnp.asarray(X[sel])))  # (G, B, T, P)
        G_, B_ = Mg.shape[:2]
        flat = lambda a: jnp.asarray(a.reshape((G_ * B_,) + a.shape[2:]))  # noqa: E731

        def ev(scores, sem, eps):
            f = jax.jit(jax.vmap(lambda M_: (lambda v: (v[-1][0], jnp.concatenate([v[j] for j in root.sources], -1)[r_idx]))(stl_jax.evaluate(prog, M_, sem, eps))))
            r, c = f(scores)
            return np.asarray(r).reshape(G_, B_), np.asarray(c).reshape(G_, B_, -1)
        r_ex, c_ex = ev(flat(Mg), "exact", None)
        sm = {}
        if roots:
            Zs = flat(np.asarray(scor(jnp.asarray(hc_all), jnp.asarray(X[sel]))))
            for a, e in arms:  # over arms and eps
                sm[(a, e)] = ev(Zs, methods.SEMANTICS[a], 0.4 if e is None else e)
            del Zs
        sep, slow = rule_values(Mg, n_r, n_h)  # (G, B, T)
        for g, geo in enumerate(geos):  # over geometries (table rows)
            gp, gh = gaps(C[sel], rr, hc_all[g], hr)
            for i, s_ in enumerate(sel):  # over the eight plans of this wait (table rows)
                ph = phase_masks(w, z["depth"][s_], t)
                row = {"geometry": geo["name"], "wait": w, "depth": float(z["depth"][s_]), "start": int(z["start"][s_]), "plan": int(s_),
                       "root_exact": float(r_ex[g, i]), **{"c_" + c: float(v) for c, v in zip(E.CONJUNCTS, c_ex[g, i])},
                       "deciding": E.CONJUNCTS[int(np.argmin(c_ex[g, i]))]}
                for k, name in enumerate(PHASES):  # over phases (five)
                    m = ph[k]
                    row["gap_" + name] = float(gp[i][m].min())
                    row["hand_gap_" + name] = float(gh[i][m].min())
                    row["sep_" + name] = float(sep[g, i][m].min())
                    row["slow_" + name] = float(slow[g, i][m].min())
                    row["tip_speed_max_" + name] = float(speed[s_][m].max())
                row["visit_start"], row["visit_end"] = insts[g].get("visit", (np.nan, np.nan))
                row["hand_gap_visit"] = float(gh[i][(t >= row["visit_start"] - 1e-9) & (t <= row["visit_end"] + 1e-9)].min()) if np.isfinite(row["visit_start"]) else np.nan
                row["hand_in_zone_at_0"] = bool(np.linalg.norm(hc_all[g, 0, HAND] - np.asarray(sc.zone)) < sc.zone_radius)
                row["hand_zone_distance_hold"] = float(np.linalg.norm(hc_all[g][ph[2]][:, HAND] - np.asarray(sc.zone), axis=-1).max())
                row["hand_zone_distance_handover"] = float(np.linalg.norm(hc_all[g][ph[4]][:, HAND] - np.asarray(sc.zone), axis=-1).min())
                for (a, e), (r_, c_) in sm.items():  # over arms (columns)
                    key = a + ("" if e is None else "_" + str(e))
                    row["root_" + key] = float(r_[g, i])
                    row["order_" + key] = float(c_[g, i][2])
                    row["sep_" + key] = float(c_[g, i][0])
                    row["slow_" + key] = float(c_[g, i][1])
                    row["hand_" + key] = float(c_[g, i][3])
                rows.append(row)
        print("wait", w, "seconds", round(time.perf_counter() - t0, 1), flush=True)
    D._csv(out + "/" + tag + ".csv", list(rows[0]), [list(r.values()) for r in rows])
    for geo in geos:  # summary per geometry (printout)
        v = [r for r in rows if r["geometry"] == geo["name"]]
        mn = lambda k: round(min(r[k] for r in v), 4)  # noqa: E731
        mx_ = lambda k: round(max(r[k] for r in v), 4)  # noqa: E731
        line = [geo["name"], "deciding", sorted({r["deciding"] for r in v}), "root exact", mn("root_exact"), mx_("root_exact")]
        line += ["hand gap dip/hold/transfer/handover", mn("hand_gap_dip"), mn("hand_gap_hold"), mn("hand_gap_transfer"), mn("hand_gap_handover")]
        line += ["gap dip/hold/transfer/handover", mn("gap_dip"), mn("gap_hold"), mn("gap_transfer"), mn("gap_handover")]
        line += ["sep min by phase", [mn("sep_" + p) for p in PHASES], "slow min by phase", [mn("slow_" + p) for p in PHASES]]
        line += ["c_sep", mn("c_separation"), "c_slow", mn("c_slowdown")]
        if roots:
            for e in D.EPS:  # over eps
                line += ["eps", e, "plain root min/max", mn("root_lse_plain_" + str(e)), mx_("root_lse_plain_" + str(e)),
                         "plain sep min", mn("sep_lse_plain_" + str(e)), "plain slow min", mn("slow_lse_plain_" + str(e)),
                         "plain order min", mn("order_lse_plain_" + str(e))]
        print(*line, flush=True)
    print("seconds", round(time.perf_counter() - t0, 1))


# ------------------------------------------------------------------
# setup and tables for the runs (the runs themselves are e034_until_demo's run mode)

def setup_main(start_dir, geo_path, name, out, waits, depths, arms, eps_, starts=None):
    """The instance file of e034_until_demo's run mode for the person setting `name` and the starts in start_dir
    (pick and handover targets from the starts file). Run axis: eps, wait, arm, depth, start (start fastest)."""
    geo = load_geometries(geo_path)[name]
    z = np.load(start_dir + "/starts.npz")
    starts = tuple(range(D.N_STARTS)) if starts is None else starts
    rows = []
    for e in eps_:  # the run axis, as e034_until_demo.setup_main
        for w in waits:
            for arm in arms:
                for d in depths:
                    for s in starts:
                        p = int(np.nonzero(np.isclose(z["wait"], w) & np.isclose(z["depth"], d) & (z["start"] == s))[0][0])
                        rows.append((name, e, w, arm, d, s, p))
    pi = np.asarray([r[6] for r in rows])
    insts = {w: instance(w, geo) for w in waits}
    hc = np.stack([insts[r[2]]["human_centres"] for r in rows]).astype(np.float32)
    I0 = insts[waits[0]]
    np.savez(out, x0=np.broadcast_to(z["x0"], (len(rows), 14)).astype(np.float32), V0=z["V0"][pi], hc=hc, pick=z["pick"], handover=z["handover"],
             hr=I0["human_radii"], n_h=len(I0["human_radii"]), start_dir=start_dir, plan=pi, geometry=json.dumps(geo), horizon=int(D.H),
             run_person=np.asarray([r[0] for r in rows]), run_eps=np.asarray([r[1] for r in rows]), run_wait=np.asarray([r[2] for r in rows]),
             run_method=np.asarray([r[3] for r in rows]), run_depth=np.asarray([r[4] for r in rows]), run_start=np.asarray([r[5] for r in rows]))
    print("runs", len(rows), "hc", hc.shape, "pick", z["pick"].tolist(), "handover", z["handover"].tolist())


def run_table(z):
    """Per run of a run record z (e034_until_demo run): the outcomes of the brief (E039) as a list of dicts."""
    from examples import e022_regime as E
    tr, ex, conj = z["trace"], z["referee64"], z["conjuncts64"]
    rs = np.concatenate([tr[:, :, 1], tr[:, -1:, 12]], 1)  # the arm's own smoothed robustness at iterates 0..K-1
    claim = rs >= D.DELTA
    bad = claim & (ex < D.DELTA)
    safe_ok = ex >= D.DELTA
    first_safe = np.where(safe_ok.any(1), np.argmax(safe_ok, 1), -1)
    tail = np.flip(np.logical_and.accumulate(np.flip(safe_ok, 1), 1), 1)
    stays = np.where(first_safe >= 0, tail[np.arange(len(ex)), np.maximum(first_safe, 0)], False)
    dec = np.argmin(conj, -1)  # (n, K) the rule deciding the certificate at each iterate
    sep, slow = conj[:, :, 0], conj[:, :, 1]
    out = []
    for r in range(len(ex)):  # over runs (table rows)
        c = int(np.argmax(claim[r])) if claim[r].any() else -1
        out.append({"person": str(z["run_person"][r]), "eps": float(z["run_eps"][r]), "arm": str(z["run_method"][r]), "wait": float(z["run_wait"][r]),
                    "depth": float(z["run_depth"][r]), "start": int(z["run_start"][r]), "run": int(z["index"][r]),
                    "exact_start": float(ex[r, 0]), "smooth_start": float(rs[r, 0]), "first_claim": c,
                    "exact_at_first_claim": float(ex[r, c]) if c >= 0 else np.nan, "rule_at_first_claim": E.CONJUNCTS[dec[r, c]] if c >= 0 else "",
                    "first_claim_false": bool(c >= 0 and ex[r, c] < D.DELTA), "false_claim_iterates": int(bad[r].sum()),
                    "deepest_exact_while_claiming": float(ex[r][claim[r]].min()) if claim[r].any() else np.nan,
                    "first_safe": int(first_safe[r]), "stays_safe": bool(stays[r]), "exact_end": float(ex[r, -1]), "claims_end": bool(claim[r, -1]),
                    "sep_min": float(sep[r].min()), "sep_min_iterate": int(np.argmin(sep[r])), "slow_min": float(slow[r].min()),
                    "slow_min_iterate": int(np.argmin(slow[r])), "iterates_sep_negative": int((sep[r] < 0).sum()), "iterates_slow_negative": int((slow[r] < 0).sum()),
                    "deciding_counts": "/".join(str(int((dec[r] == k).sum())) for k in range(len(E.CONJUNCTS))),
                    "deciding_by_iterate": "".join("SLOH"[k] for k in dec[r]), "iterates": ex.shape[1] - 1})
    return out


def tables_main(out_dir, tag, regime_csv, paths):
    """<tag>_runs.csv (run_table), <tag>_soundness.csv and <tag>_attenuation.csv per (eps, arm, wait, depth), starts in order,
    with the sound share on the entry samples from the regime table (range over the starts of that wait and depth), and
    <tag>_person.csv (the smallest exact separation and slow-down over the iterates per start)."""
    with open(regime_csv) as fh:
        reg = list(csv.DictReader(fh))
    share = {}
    for r in reg:  # over regime rows (table rows); the until rows do not depend on the person
        share.setdefault((float(r["eps"]), float(r["wait"]), float(r["depth"])), {})[int(r["start"])] = float(r["share_entry_deciding"])
    runs = [u for p in paths for u in run_table(np.load(p))]  # over run records (files)
    cells = {}
    for u in runs:  # group by cell (table rows)
        cells.setdefault((u["eps"], u["arm"], u["wait"], u["depth"]), []).append(u)
    snd, att, per = [], [], []
    j = lambda v, k: " ".join(str(round(u[k], 4)) if isinstance(u[k], float) else str(u[k]) for u in v)  # noqa: E731
    for key, v in sorted(cells.items()):  # over cells (table rows)
        v.sort(key=lambda u: u["start"])
        sh = list(share.get((key[0], key[2], key[3]), {0: np.nan}).values())
        shs = str(round(min(sh), 3)) + "-" + str(round(max(sh), 3))
        claims = [u for u in v if u["first_claim"] >= 0]
        snd.append(list(key) + [len(v), sum(u["first_claim_false"] for u in v), sum(u["false_claim_iterates"] > 0 for u in v), j(v, "first_claim"),
                                j(v, "exact_at_first_claim"), j(v, "false_claim_iterates"),
                                float(np.nanmin([u["deepest_exact_while_claiming"] for u in claims])) if claims else np.nan,
                                sum(u["exact_end"] < D.DELTA for u in v), sum(u["claims_end"] and u["exact_end"] < D.DELTA for u in v), shs])
        fs = np.array([u["first_safe"] for u in v], float)
        fs[fs < 0] = np.nan
        att.append(list(key) + [len(v), j(v, "first_safe"), int(np.isfinite(fs).sum()), float(np.nanmedian(fs)) if np.isfinite(fs).any() else np.nan,
                                sum(u["stays_safe"] for u in v), j(v, "exact_end"), shs])
        per.append(list(key) + [len(v), j(v, "sep_min"), j(v, "sep_min_iterate"), j(v, "slow_min"), j(v, "slow_min_iterate"),
                                sum(u["iterates_sep_negative"] > 0 for u in v), sum(u["iterates_slow_negative"] > 0 for u in v), j(v, "deciding_counts")])
    head = ["eps", "arm", "wait", "depth", "runs"]
    D._csv(out_dir + "/" + tag + "_soundness.csv", head + ["runs_first_claim_false", "runs_with_a_false_claim", "first_claim_by_start", "exact_at_first_claim_by_start",
                                                         "iterates_claimed_while_unsafe_by_start", "deepest_exact_while_claiming", "runs_ending_unsafe",
                                                         "runs_ending_claiming_while_unsafe", "sound_share_entry_regime"], snd)
    D._csv(out_dir + "/" + tag + "_attenuation.csv", head + ["first_safe_by_start", "runs_safe", "first_safe_median", "runs_staying_safe",
                                                           "exact_end_by_start", "sound_share_entry_regime"], att)
    D._csv(out_dir + "/" + tag + "_person.csv", head + ["sep_min_by_start", "sep_min_iterate_by_start", "slow_min_by_start", "slow_min_iterate_by_start",
                                                      "runs_sep_negative_somewhere", "runs_slow_negative_somewhere",
                                                      "iterates_decided_by_separation/slowdown/order/handover_by_start"], per)
    D._csv(out_dir + "/" + tag + "_runs.csv", list(runs[0]), [list(u.values()) for u in runs])
    for row in snd:  # printout
        print("soundness", row)
    for row in att:  # printout
        print("attenuation", row)
    for row in per:  # printout
        print("person", row)


if __name__ == "__main__":
    hs = [a for a in sys.argv if a.startswith("H=")]
    if hs:  # E038's horizon: E034's globals H and WAITS (the longest wait) set in this process
        from examples import e038_horizon
        e038_horizon.horizon(int(hs[0][2:]))
        sys.argv = [a for a in sys.argv if not a.startswith("H=")]
    mode = sys.argv[1]
    if mode == "scan":
        scan_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], len(sys.argv) > 6 and sys.argv[6] == "roots",
                  tuple(sys.argv[7].split(",")) if len(sys.argv) > 7 else None)
    elif mode == "setup":
        f = lambda a: tuple(float(x) for x in a.split(","))  # noqa: E731
        setup_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], f(sys.argv[6]), f(sys.argv[7]), tuple(sys.argv[8].split(",")), f(sys.argv[9]))
    elif mode == "tables":
        tables_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise ValueError("unknown mode " + mode)

"""E037 extension (root's brief of 2026-10-01 08:43Z; M003 note and D006 amendment of 08:42Z): E037's 16 instances and
their starts as built, on E039's person scene visit_h0.45_L10.0, at eps 0.2, with six arms.

    JAX_PLATFORMS=cpu python -m examples.e037_person check <start_dir> <regime.csv> <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e037_person roots <start_dir> <choice.csv> <regime.csv> <out_dir> <eps>
    python -m examples.e037_person setup <start_dir> <choice.csv> <out.npz> <instances> <arms> [<eps>]
    JAX_PLATFORMS=nojax python -m examples.e037_instances run <instance.npz> <iterations> <out.npz> all
    python -m examples.e037_person tables <out_dir> <tag> <pairs.csv> <run.npz>...

The person (tasks.workspace.until_visit_inputs, E039): 1.5 m from the zone centre while the arm moves; once the arm is at
rest at the hold pose the person steps in and holds the hand point `place` metres from the zone centre on the far side
(phi = 0), 0.10 m above the table, until shortly before the arm leaves the pick pose; then withdraws. VISIT is E039's
dict (gates/E039/2026-10-01T0655Z/scan/visit1.json); only its key depth (= -place) changes per instance, by the rule in
the gate's RULE-hand-place.txt (written before the check): E039's 0.45 m if separation and slow-down hold and the plain
log-sum-exp's root at eps 0.2 is positive on the instance's 8 starts, else the smallest of PLACES at which these hold
and the hand's surface gap to the arm on the visit's plateau is at most NEAR, else 0.45 m labelled "no place".

check: per wait, hand place (PLACES and 0.45) and start, from the float64 replay of the starts: the exact rules' values
at the root and the deciding rule, the plain root at eps 0.2, the exact separation and slow-down, the surface gap from
the person and from the hand to the arm per phase (E039's phases and gap definitions, examples.e039_person_zone),
in check.csv; the rule's choice per instance (choice.csv); per (instance, depth) at the chosen place the pass condition
(order rule deciding at -depth, separation and slow-down >= 0, plain root > 0, at every wait) and the labels near and
the two until-node conditions at eps 0.2 from E037's regime table (pairs.csv).

tables: per wait and depth, for the plain arm the instances with a claim at iterate 0 while the certificate is negative,
with any claim while the certificate is negative, and ending negative; for sparsemax against each sound arm the
per-instance difference sparsemax minus the arm of the first iterate with a non-negative certificate (a run without one
counts as K, one past the last iterate; counts earlier, equal, later; median with a resampling interval, E037's paired);
per arm the smallest exact separation and slow-down over every iterate.
"""
import csv
import json
import sys
import time

import numpy as np

from examples import e034_until_demo as U
from examples import e037_instances as X

VISIT = {"kind": "visit", "phi": 0.0, "stand": 0.9, "depth": -0.45, "reach_z": 0.1, "t_move": 0.2, "lead": 0.1, "t_rest": 2.1, "length": 10.0,
         "far": 0.6, "hover": 0.6, "hover_z": -0.15}
E039_PLACE = 0.45
PLACES = (0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
NEAR = 0.30
EPS_P = 0.2
ARMS6 = ("lse_plain", "lse", "sparsemax", "gm_pm01", "gm_pm10", "gm_exp")
BASELINES = ("lse", "gm_pm01", "gm_pm10", "gm_exp")


def visit(place):
    return dict(VISIT, depth=-float(place))


def person(w, place):
    from sparsemax_dstl.tasks import workspace as W
    return W.until_visit_inputs(U.scenario(w), visit(place))


# ------------------------------------------------------------------
# the pre-run check

def check_main(start_dir, regime_csv, out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from examples import e039_person_zone as P39
    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.jax import methods
    from sparsemax_dstl.tasks import workspace as W
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    from sparsemax_dstl.tasks import workspace_program as Wp
    jax.config.update("jax_enable_x64", True)
    t_start = time.perf_counter()
    z = np.load(start_dir + "/starts.npz")
    plant = Wp.plant
    mx = mjx.put_model(plant.model, impl="jax")
    Xs = z["X64"]
    t = np.arange(Xs.shape[1]) * U.HS
    C, _ = P39.robot_centres(Xs)
    rr = W.robot_spheres(plant, W.Scenario().robot_spacing)["radius"]
    n_r = len(rr)
    places = sorted(set(PLACES + (E039_PLACE,)))
    depth_idx = np.searchsorted(np.asarray(U.DEPTHS), np.round(z["depth"], 6))
    rows = []
    for w in U.WAITS:  # four waits (one program each)
        sc = U.scenario(w)
        prog = Wp.core_program(sc, len(person(w, E039_PLACE)["human_radii"]))
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])
        sel = np.nonzero(np.isclose(z["wait"], w))[0]
        masks = np.stack([P39.phase_masks(w, d, t) for d in U.DEPTHS])[depth_idx[sel]]  # (B, 5, T); two depths
        hr = jnp.asarray(person(w, E039_PLACE)["human_radii"])
        mg = jax.jit(jax.vmap(lambda x, p, hd, hc: Wm.margins(mx, plant, sc, {"pick": p, "handover": hd, "human_centres": hc, "human_radii": hr}, x), (0, 0, 0, None)))
        sg = jax.jit(jax.vmap(lambda x, p, hd, hc: Wm.scores(mx, plant, sc, {"pick": p, "handover": hd, "human_centres": hc, "human_radii": hr}, x), (0, 0, 0, None)))

        def children(M_, sem, e):
            v = stl_jax.evaluate(prog, M_, sem, e)
            return jnp.ravel(v[-1])[0], jnp.concatenate([v[j] for j in root.sources], -1)[r_idx]
        f_ex = jax.jit(jax.vmap(lambda M_: children(M_, "exact", None)))
        f_pl = jax.jit(jax.vmap(lambda M_: children(M_, methods.SEMANTICS["lse_plain"], EPS_P)))
        args = (jnp.asarray(Xs[sel]), jnp.asarray(z["pick"][sel]), jnp.asarray(z["handover"][sel]))
        for d in places:  # hand places (twelve configurations)
            P = person(w, d)
            hc = jnp.asarray(P["human_centres"])
            Mg = mg(*args, hc)
            r_ex, c_ex = (np.asarray(a) for a in f_ex(Mg))
            r_pl, c_pl = (np.asarray(a) for a in f_pl(sg(*args, hc)))
            sep, slow = P39.rule_values(np.asarray(Mg), n_r, len(P["human_radii"]))
            del Mg
            gp, gh = P39.gaps(C[sel], rr, np.asarray(P["human_centres"]), np.asarray(P["human_radii"]))
            v0, v1 = P["visit"]
            plateau = (t >= v0 - 1e-9) & (t <= v1 + 1e-9)
            ph_min = lambda a: np.where(masks, a[:, None], np.inf).min(-1)  # noqa: E731  (B, 5)
            cols = {"instance": z["instance"][sel], "draw": z["draw"][sel], "depth": z["depth"][sel], "wait": np.full(len(sel), w), "plan": sel,
                    "place": np.full(len(sel), d), "root_exact": r_ex, **{"c_" + c: c_ex[:, j] for j, c in enumerate(E.CONJUNCTS)},
                    "deciding": np.asarray(E.CONJUNCTS)[np.argmin(c_ex, 1)], "root_plain_0.2": r_pl,
                    **{"plain_" + c: c_pl[:, j] for j, c in enumerate(E.CONJUNCTS)}, "sep_min": sep.min(1), "slow_min": slow.min(1),
                    "hand_gap_plateau": np.where(plateau, gh, np.inf).min(1), "visit_start": np.full(len(sel), v0), "visit_end": np.full(len(sel), v1)}
            for k, name in enumerate(P39.PHASES):  # five phases (columns)
                cols["gap_" + name] = ph_min(gp)[:, k]
                cols["hand_gap_" + name] = ph_min(gh)[:, k]
                cols["sep_" + name] = ph_min(sep)[:, k]
                cols["slow_" + name] = ph_min(slow)[:, k]
            keys = list(cols)
            rows += [{k: (cols[k][i].item() if hasattr(cols[k][i], "item") else cols[k][i]) for k in keys} for i in range(len(sel))]
        print("wait", w, "seconds", round(time.perf_counter() - t_start, 1), flush=True)
    U._csv(out + "/check.csv", list(rows[0]), [list(r.values()) for r in rows])
    choice, pairs = choose(rows, regime_csv)
    U._csv(out + "/choice.csv", list(choice[0]), [list(r.values()) for r in choice])
    U._csv(out + "/pairs.csv", list(pairs[0]), [list(r.values()) for r in pairs])
    for r in choice:  # printout
        print("choice", r)
    for r in pairs:  # printout
        print("pair", r)
    print("pairs passing", sum(r["pass"] for r in pairs), "of", len(pairs), "seconds", round(time.perf_counter() - t_start, 1))


def choose(rows, regime_csv):
    """The rule of RULE-hand-place.txt over the check's rows (dicts); returns the choice per instance and the pass table per
    (instance, depth) at the chosen place, with the until-node labels at EPS_P from E037's regime table."""
    R = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    a_ok = (R["sep_min"] >= 0) & (R["slow_min"] >= 0) & (R["root_plain_0.2"] > 0)
    near = R["hand_gap_plateau"] <= NEAR
    insts = sorted(set(R["instance"].tolist()))
    choice = []
    for i in insts:  # instances (table rows)
        A = {d: bool(a_ok[(R["instance"] == i) & np.isclose(R["place"], d)].all()) for d in sorted(set(R["place"].tolist()))}
        N = {d: bool(near[(R["instance"] == i) & np.isclose(R["place"], d)].all()) for d in A}
        ok = [d for d in PLACES if A[d] and N[d]]
        d, how = (E039_PLACE, "e039") if A[E039_PLACE] else ((ok[0], "rule") if ok else (E039_PLACE, "no place"))
        choice.append({"instance": i, "A_0.45": A[E039_PLACE], "near_0.45": N[E039_PLACE], "place": d, "how": how,
                       "places_A_and_near": " ".join(str(x) for x in ok)})
    with open(regime_csv) as fh:
        reg = [r for r in csv.DictReader(fh) if float(r["eps"]) == EPS_P]
    pairs = []
    for c in choice:  # instances x depths (table rows)
        for dp in U.DEPTHS:
            m = (R["instance"] == c["instance"]) & np.isclose(R["depth"], dp) & np.isclose(R["place"], c["place"])
            rg = [r for r in reg if int(r["instance"]) == c["instance"] and np.isclose(float(r["depth"]), dp)]
            share = [float(r["share_entry_deciding"]) for r in rg if np.isclose(float(r["wait"]), max(U.WAITS))][0]
            hold = max(int(r["support_hold_deciding"]) for r in rg)
            order_ok = bool(np.all((R["deciding"][m] == "order") & np.isclose(R["root_exact"][m], -dp, atol=1e-4)))
            ok = order_ok and bool(np.all(R["sep_min"][m] >= 0) and np.all(R["slow_min"][m] >= 0) and np.all(R["root_plain_0.2"][m] > 0))
            pairs.append({"instance": c["instance"], "depth": dp, "place": c["place"], "how": c["how"], "pass": ok, "order_decides": order_ok,
                          "sep_min": float(R["sep_min"][m].min()), "slow_min": float(R["slow_min"][m].min()),
                          "plain_root_min": float(R["root_plain_0.2"][m].min()), "plain_root_max": float(R["root_plain_0.2"][m].max()),
                          "plain_sep_min": float(R["plain_separation"][m].min()), "plain_slow_min": float(R["plain_slowdown"][m].min()),
                          "hand_gap_plateau_min": float(R["hand_gap_plateau"][m].min()), "hand_gap_plateau_max": float(R["hand_gap_plateau"][m].max()),
                          "near": bool(np.all(near[m])), **{"gap_" + p: float(R["gap_" + p][m].min()) for p in ("approach", "dip", "hold", "transfer", "handover")},
                          **{"hand_gap_" + p: float(R["hand_gap_" + p][m].min()) for p in ("dip", "hold", "transfer", "handover")},
                          "share_longest_wait_0.2": share, "support_hold_max_0.2": hold, "until_inside_0.2": bool(share <= X.SHARE_MAX and hold == 0)})
    return choice, pairs


def roots_main(start_dir, choice_csv, regime_csv, out, eps):
    """At each instance's chosen place (choice.csv), per start: the exact root and deciding rule, and the roots and the
    root's four children of the plain log-sum-exp, the sound log-sum-exp and sparsemax at eps (roots_<eps>.csv); per
    (instance, depth): the plain root's range over the waits and the count of positive starts, and the until-node labels at
    eps from E037's regime table (pairs_<eps>.csv; pass = the order rule decides and the plain root is positive at every wait)."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.jax import methods
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    from sparsemax_dstl.tasks import workspace_program as Wp
    jax.config.update("jax_enable_x64", True)
    with open(choice_csv) as fh:
        place = {int(r["instance"]): float(r["place"]) for r in csv.DictReader(fh)}
    z = np.load(start_dir + "/starts.npz")
    plant = Wp.plant
    mx = mjx.put_model(plant.model, impl="jax")
    rows = []
    for w in U.WAITS:  # four waits (one program each)
        sc = U.scenario(w)
        sel = np.nonzero(np.isclose(z["wait"], w))[0]
        hc = np.stack([person(w, place[int(i)])["human_centres"] for i in z["instance"][sel]])  # per start (instances' places)
        hr = jnp.asarray(person(w, E039_PLACE)["human_radii"])
        prog = Wp.core_program(sc, len(hr))
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])
        args = (jnp.asarray(z["X64"][sel]), jnp.asarray(z["pick"][sel]), jnp.asarray(z["handover"][sel]), jnp.asarray(hc))
        inst = lambda p, hd, h: {"pick": p, "handover": hd, "human_centres": h, "human_radii": hr}  # noqa: E731
        Mg = jax.vmap(lambda x, p, hd, h: Wm.margins(mx, plant, sc, inst(p, hd, h), x))(*args)
        Zs = jax.vmap(lambda x, p, hd, h: Wm.scores(mx, plant, sc, inst(p, hd, h), x))(*args)

        def children(M_, sem, e):
            v = stl_jax.evaluate(prog, M_, sem, e)
            return jnp.ravel(v[-1])[0], jnp.concatenate([v[j] for j in root.sources], -1)[r_idx]
        r_ex, c_ex = (np.asarray(a) for a in jax.jit(jax.vmap(lambda M_: children(M_, "exact", None)))(Mg))
        cols = {"instance": z["instance"][sel], "depth": z["depth"][sel], "wait": np.full(len(sel), w), "place": np.asarray([place[int(i)] for i in z["instance"][sel]]),
                "eps": np.full(len(sel), eps), "root_exact": r_ex, "deciding": np.asarray(E.CONJUNCTS)[np.argmin(c_ex, 1)]}
        for a in ("lse_plain", "lse", "sparsemax"):  # three arms
            r_, c_ = (np.asarray(v) for v in jax.jit(jax.vmap(lambda M_, s=methods.SEMANTICS[a]: children(M_, s, eps)))(Zs))
            cols["root_" + a] = r_
            cols.update({a + "_" + c: c_[:, j] for j, c in enumerate(E.CONJUNCTS)})
        keys = list(cols)
        rows += [{k: (cols[k][i].item() if hasattr(cols[k][i], "item") else cols[k][i]) for k in keys} for i in range(len(sel))]
    tag = str(eps)
    U._csv(out + "/roots_" + tag + ".csv", list(rows[0]), [list(r.values()) for r in rows])
    with open(regime_csv) as fh:
        reg = [r for r in csv.DictReader(fh) if np.isclose(float(r["eps"]), eps)]
    pairs = []
    for i in sorted(place):  # instances x depths (table rows)
        for dp in U.DEPTHS:
            v = [r for r in rows if r["instance"] == i and np.isclose(r["depth"], dp)]
            rg = [r for r in reg if int(r["instance"]) == i and np.isclose(float(r["depth"]), dp)]
            share = [float(r["share_entry_deciding"]) for r in rg if np.isclose(float(r["wait"]), max(U.WAITS))][0]
            hold = max(int(r["support_hold_deciding"]) for r in rg)
            pl = [r["root_lse_plain"] for r in v]
            order_ok = all(r["deciding"] == "order" and abs(r["root_exact"] + dp) <= 1e-4 for r in v)
            pairs.append({"instance": i, "depth": dp, "place": place[i], "eps": eps, "order_decides": order_ok, "plain_root_min": min(pl), "plain_root_max": max(pl),
                          "plain_positive_waits": sum(x > 0 for x in pl), "pass": order_ok and min(pl) > 0,
                          "plain_sep_min": min(r["lse_plain_separation"] for r in v), "plain_slow_min": min(r["lse_plain_slowdown"] for r in v),
                          "plain_order_min": min(r["lse_plain_order"] for r in v), "share_longest_wait": share, "support_hold_max": hold,
                          "until_inside": bool(share <= X.SHARE_MAX and hold == 0)})
    U._csv(out + "/pairs_" + tag + ".csv", list(pairs[0]), [list(r.values()) for r in pairs])
    for r in pairs:  # printout
        print("pair", r)
    print("eps", eps, "pairs with the plain root positive at every wait", sum(r["pass"] for r in pairs), "of", len(pairs), "starts with a positive plain root",
          sum(r["plain_positive_waits"] for r in pairs), "of", 4 * len(pairs), "until inside", sum(r["until_inside"] for r in pairs))


# ------------------------------------------------------------------
# setup

def setup_main(start_dir, choice_csv, out, instances, arms, eps=EPS_P):
    """The run table of one batch (e037_instances' run mode reads it): run axis wait, arm, instance, depth (depth fastest);
    the person of each run is the visit at its instance's chosen place and its wait."""
    with open(choice_csv) as fh:
        place = {int(r["instance"]): float(r["place"]) for r in csv.DictReader(fh)}
    z = np.load(start_dir + "/starts.npz")
    waits, depths = U.WAITS, U.DEPTHS
    W_, A_, I_, D_ = np.meshgrid(np.arange(len(waits)), np.arange(len(arms)), np.arange(len(instances)), np.arange(len(depths)), indexing="ij")
    rw, ra, ri, rd = (np.asarray(waits)[W_.ravel()], np.asarray(arms)[A_.ravel()], np.asarray(instances)[I_.ravel()], np.asarray(depths)[D_.ravel()])
    key = np.stack([np.round(z["wait"], 6), np.round(z["depth"], 6), z["instance"]], 1)
    hit = np.all(np.stack([np.round(rw, 6), np.round(rd, 6), ri], 1)[:, None] == key[None], -1)
    if not np.all(hit.sum(1) == 1):
        raise ValueError("every run needs exactly one start")
    pi = np.argmax(hit, 1)
    n = len(pi)
    hcs = {(i, w): person(w, place[i])["human_centres"] for i in instances for w in waits}  # (instance, wait) configurations
    P0 = person(waits[0], place[instances[0]])
    np.savez(out, x0=np.broadcast_to(z["x0"], (n, 14)).astype(np.float32), V0=z["V0"][pi], hc=np.stack([hcs[(int(i), float(w))] for i, w in zip(ri, rw)]).astype(np.float32),
             pick=z["pick"][pi], handover=z["handover"][pi], hr=P0["human_radii"], n_h=len(P0["human_radii"]), start_dir=start_dir, plan=pi,
             run_person=np.asarray(["visit_" + str(place[int(i)]) for i in ri]), run_eps=np.full(n, float(eps)), run_wait=rw, run_method=ra,
             run_depth=rd, run_instance=ri, run_start=np.zeros(n, int), places=json.dumps(place))
    print("runs", n, "eps", eps, "instances", list(instances), "places", [place[i] for i in instances], "arms", list(arms))


# ------------------------------------------------------------------
# tables

def tables_main(out_dir, tag, pairs_csv, paths):
    with open(pairs_csv) as fh:
        lab = {(int(r["instance"]), round(float(r["depth"]), 6)): r for r in csv.DictReader(fh)}
    cols = {k: [] for k in ("eps", "arm", "wait", "depth", "instance")}
    outs, extra, K = [], [], None
    for p in paths:  # run records (files)
        z = np.load(p)
        tr, ex, conj = z["trace"], z["referee64"], z["conjuncts64"]
        rs = np.concatenate([tr[:, :, 1], tr[:, -1:, 12]], 1)
        K = ex.shape[1] if K is None else K
        if ex.shape[1] != K:
            raise ValueError("every record needs the same number of iterates")
        outs.append(X.run_outcomes(rs, ex))
        extra.append({"claim0_false": (rs[:, 0] >= U.DELTA) & (ex[:, 0] < U.DELTA), "sep_min": conj[:, :, 0].min(1), "slow_min": conj[:, :, 1].min(1)})
        for k, src in (("eps", "run_eps"), ("arm", "run_method"), ("wait", "run_wait"), ("depth", "run_depth"), ("instance", "run_instance")):
            cols[k].append(z[src])
    R = {k: np.concatenate(v) for k, v in cols.items()}
    R.update({k: np.concatenate([o[k] for o in outs]) for k in outs[0]})
    R.update({k: np.concatenate([o[k] for o in extra]) for k in extra[0]})
    R["arm"] = R["arm"].astype(str)
    ukey = "until_inside" if "until_inside" in next(iter(lab.values())) else "until_inside_0.2"
    R["until_inside"] = np.asarray([lab[(int(i), round(float(d), 6))][ukey] == "True" for i, d in zip(R["instance"], R["depth"])])
    R["pass"] = np.asarray([lab[(int(i), round(float(d), 6))]["pass"] == "True" for i, d in zip(R["instance"], R["depth"])])
    order = np.lexsort((R["instance"], R["depth"], R["wait"], R["arm"], R["eps"]))
    R = {k: v[order] for k, v in R.items()}
    if len(set(zip(R["eps"], R["arm"], R["wait"], R["depth"], R["instance"]))) != len(R["eps"]):
        raise ValueError("a run appears twice")
    names = list(R)
    U._csv(out_dir + "/" + tag + "_runs.csv", names, [[R[k][i] for k in names] for i in range(len(R["eps"]))])
    regions = (("all", None), ("until_inside", True), ("until_outside", False))
    rng = np.random.default_rng(X.BOOT_SEED)
    pr, pl, att, per = [], [], [], []
    for e in sorted(set(R["eps"])):  # eps
        for w in sorted(set(R["wait"])):  # waits
            for dp in sorted(set(R["depth"])):  # depths
                for reg, flag in regions:  # regions
                    base = (R["eps"] == e) & (R["wait"] == w) & (R["depth"] == dp) & ((R["until_inside"] == flag) if flag is not None else True)
                    s = base & (R["arm"] == "sparsemax")
                    for b in BASELINES:  # sparsemax against each sound arm
                        sb = base & (R["arm"] == b)
                        if not s.any() or not sb.any():
                            continue
                        if not np.array_equal(R["instance"][s], R["instance"][sb]):
                            raise ValueError("the two arms need the same instances")
                        q = X.paired(R["first_safe"][s], R["first_safe"][sb], K, rng)
                        pr.append([e, "sparsemax", b, w, dp, reg, q["n"], q["earlier"], q["equal"], q["later"], q["median"], q["lo"], q["hi"],
                                   int(np.sum(R["first_safe"][s] < 0)), int(np.sum(R["first_safe"][sb] < 0)),
                                   " ".join(str(x) for x in R["first_safe"][s]), " ".join(str(x) for x in R["first_safe"][sb])])
                    sp = base & (R["arm"] == "lse_plain")
                    if sp.any():
                        fc = sp & (R["false_claim_iterates"] > 0)
                        pl.append([e, w, dp, reg, int(sp.sum()), int(np.sum(sp & R["claim0_false"])), int(fc.sum()), int(np.sum(sp & (R["exact_end"] < U.DELTA))),
                                   int(np.sum(sp & R["false_claim_end"])), float(np.min(R["deepest_while_claiming"][fc])) if fc.any() else np.nan])
                    for a in ARMS6:  # arms
                        sa = base & (R["arm"] == a)
                        if not sa.any():
                            continue
                        fs = R["first_safe"][sa]
                        ee = R["exact_end"][sa]
                        att.append([e, a, w, dp, reg, int(sa.sum()), " ".join(str(x) for x in fs), int(np.sum(fs < 0)), float(np.median(np.where(fs < 0, K, fs))),
                                    int(np.sum(R["stays_safe"][sa])), int(np.sum(sa & (R["false_claim_iterates"] > 0))), float(ee.min()), float(np.median(ee)),
                                    float(ee.max()), int(np.sum(ee < U.DELTA)), " ".join(str(round(float(x), 4)) for x in ee)])
                        per.append([e, a, w, dp, reg, int(sa.sum()), float(R["sep_min"][sa].min()), float(R["slow_min"][sa].min()),
                                    int(np.sum(R["sep_min"][sa] < 0)), int(np.sum(R["slow_min"][sa] < 0))])
    U._csv(out_dir + "/" + tag + "_sparsemax_vs.csv", ["eps", "arm_a", "arm_b", "wait", "depth", "region", "instances", "a_earlier", "equal", "a_later",
                                                      "median_a_minus_b", "lo95", "hi95", "a_never_safe", "b_never_safe", "a_first_safe", "b_first_safe"], pr)
    U._csv(out_dir + "/" + tag + "_plain.csv", ["eps", "wait", "depth", "region", "instances", "claim_at_0_while_negative", "any_claim_while_negative",
                                                "ending_negative", "ending_claiming_while_negative", "deepest_certificate_while_claiming"], pl)
    U._csv(out_dir + "/" + tag + "_attenuation.csv", ["eps", "arm", "wait", "depth", "region", "instances", "first_safe_by_instance", "never_safe",
                                                      "first_safe_median_never_as_K", "runs_staying_safe", "runs_with_a_false_claim", "exact_end_min",
                                                      "exact_end_median", "exact_end_max", "runs_ending_negative", "exact_end_by_instance"], att)
    U._csv(out_dir + "/" + tag + "_person.csv", ["eps", "arm", "wait", "depth", "region", "instances", "separation_min", "slowdown_min",
                                                 "runs_separation_negative", "runs_slowdown_negative"], per)
    print("iterates recorded", K, "bootstrap seed", X.BOOT_SEED, "resamples", X.N_BOOT)
    for row in pr:  # printout
        if row[5] == "all":
            print("sparsemax_vs", row[:15])
    for row in pl:  # printout
        if row[3] == "all":
            print("plain", row)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check_main(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "roots":
        roots_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]))
    elif mode == "setup":
        setup_main(sys.argv[2], sys.argv[3], sys.argv[4], tuple(int(x) for x in sys.argv[5].split(",")), tuple(sys.argv[6].split(",")),
                   float(sys.argv[7]) if len(sys.argv) > 7 else EPS_P)
    elif mode == "tables":
        tables_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise ValueError("unknown mode " + mode)

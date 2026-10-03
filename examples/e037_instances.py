"""E037: E034's demo on the Until over task instances with paired starts (M003 note and D006
amendment of 2026-10-01 06:25Z; M003 requirement 5).

    JAX_PLATFORMS=cpu python -m examples.e037_instances draws <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e037_instances starts <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e037_instances regime <start_dir> <out_dir>
    python -m examples.e037_instances setup <start_dir> <out.npz> <eps> <instances> <arms> <person> [<waits> <depths>]
    JAX_PLATFORMS=nojax python -m examples.e037_instances run <instance.npz> <iterations> <out.npz> <runs|all>
    python -m examples.e037_instances tables <out_dir> <tag> <labels.csv> <run.npz>...

Lists are comma-separated. The run axis of setup is wait, arm, instance, depth (depth fastest);
each (arm, wait) block shares one STL evaluator, each wait has its own program (E034's build).

Instances. A seeded generator (draws, SEED, RANGES) gives per draw:
- the pick direction, the unit ray n from the zone centre at azimuth `az` (from +x, in the
  table plane) and elevation `el`; the pick target on it, HOLD outside the zone boundary
  (E034's until_targets for that ray);
- the handover target inside the zone, `hd_r` from the centre on the ray at azimuth az + `hd_az`
  and elevation `hd_el`;
- the via site, the pick target moved `via_out` further out along n, `via_up` up and `via_side`
  along the horizontal normal of the ray; the via configuration is its inverse kinematics from
  the home configuration plus `via_noise` (rad per joint, VIA_NOISE standard deviation; E034's
  SIGMA_VIA).
E034's draw is az = pi, el = 0.6, hd_az = 0, hd_el = 0.6, hd_r = 0.10, via_out = 0.15,
via_up = 0.10, via_side = 0, via_noise = 0. Everything else is E034's study scene and is taken
from examples.e034_until_demo (U): the zone, the rules, the windows per wait, the hold
distance, the keyframe times and the entry timing per depth (U.ENTRY, fixed for every
instance), the person standing clear ("out"), the solver and its constants.

starts: per draw and depth one plan (E034's keyframes and computed-torque tracking), its entry
parameter calibrated in the float64 MuJoCo C replay to the depth (CALIBRATE updates), and the
plan at every wait. A draw is kept when all its eight plans are tracked: every inverse kinematics
residual at most W.IK_TOLERANCE, the via configuration inside the joint ranges less
W.JOINT_MARGIN, no command at the torque limit, the deepest zone margin on [T_VIA, T_HOLD] within
DEPTH_TOL of -depth, no other zone violation up to w + 1, the pick margin at least TRACK_MIN on
the pick window and the exact separation, slow-down and handover conjuncts at least TRACK_MIN.
The first N_INSTANCES kept draws in draw order are the instances; the others are reported with
their reasons (candidates.csv).

regime: E034's check before any run, on every start (instance, depth, wait), vectorized over the
starts of a wait (until_batch is U.until_rows with a leading start axis), and the label of each
(instance, depth) at eps 0.4: inside when the sound share at the longest wait is at most
SHARE_MAX and no hold sample is in sparsemax's support at the deciding witness at any wait.

tables: per run, the first claim and the first exactly safe iterate (E034's definitions); per
(eps, arm, wait, depth) and region (inside, outside, all) the false claims over instances; per
pair of sound arms the per-instance difference of the first exactly safe iterate (a run never
safe within the record counts as K, one past the last iterate), the counts earlier, equal and
later, the median difference and a percentile interval from resampling instances (BOOT_SEED,
N_BOOT resamples).
"""
import csv
import json
import socket
import sys
import time

import numpy as np

from examples import e034_until_demo as U

SEED = 2026100137
N_DRAWS = 40
N_INSTANCES = 16
# ranges of the uniform draws (rad, rad, rad, rad, m, m, m, m); written to the gate by `draws` before any start
RANGES = {"az": (np.pi - 0.8, np.pi + 0.8), "el": (0.4, 0.9), "hd_az": (-0.5, 0.5), "hd_el": (0.7, 1.2), "hd_r": (0.08, 0.12),
          "via_out": (0.10, 0.20), "via_up": (0.05, 0.15), "via_side": (-0.05, 0.05)}
VIA_NOISE = 0.05
CALIBRATE = 5
DEPTH_TOL = 1e-4
TRACK_MIN = 0.5
PERSON = "out"
SHARE_MAX = 0.2
EPS_LABEL = 0.4
SOUND = ("lse", "sparsemax", "gm_pm01", "gm_pm10", "gm_exp")  # arms whose claims are sound by construction (P1; Gilpin's bound; Theorem 2 of Mehdipour et al.)
BOOT_SEED = 2026100138
N_BOOT = 10000


def dumps(obj):
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ------------------------------------------------------------------
# the generator

def draws(seed=SEED, n=N_DRAWS):
    """n draws of RANGES (uniform) and via_noise (n, 7). Draw i does not depend on n."""
    keys = list(RANGES)
    u = np.random.default_rng([seed, 0]).random((n, len(keys)))
    lo, hi = np.asarray([RANGES[k][0] for k in keys]), np.asarray([RANGES[k][1] for k in keys])
    v = lo + u * (hi - lo)
    d = {k: v[:, i] for i, k in enumerate(keys)}
    d["via_noise"] = VIA_NOISE * np.random.default_rng([seed, 1]).standard_normal((n, 7))
    return d


def unit(az, el):
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def targets(d, sc):
    """Per draw: the ray n, the pick target, the handover target and the via site (n, 3) each."""
    z = np.asarray(sc.zone, np.float64)
    n = unit(d["az"], d["el"])
    pick = z + (sc.zone_radius + sc.until_hold) * n
    hand = z + d["hd_r"][:, None] * unit(d["az"] + d["hd_az"], d["hd_el"])
    side = np.stack([-np.sin(d["az"]), np.cos(d["az"]), np.zeros_like(d["az"])], -1)
    via = pick + d["via_out"][:, None] * n + d["via_up"][:, None] * np.array([0.0, 0.0, 1.0]) + d["via_side"][:, None] * side
    return n, pick, hand, via


def draws_main(out):
    sc = U.scenario(U.WAITS[0])
    d = draws()
    n, pick, hand, via = targets(d, sc)
    with open(out + "/ranges.json", "w") as fh:
        fh.write(dumps({"seed": SEED, "draws": N_DRAWS, "instances": N_INSTANCES, "ranges": RANGES, "via_noise_sd": VIA_NOISE,
                        "hold": sc.until_hold, "zone": sc.zone, "zone_radius": sc.zone_radius, "person": PERSON, "entry": U.ENTRY,
                        "keyframes": [U.T_LEAVE, U.T_VIA], "waits": U.WAITS, "depths": U.DEPTHS, "calibrate": CALIBRATE,
                        "depth_tol": DEPTH_TOL, "track_min": TRACK_MIN, "share_max": SHARE_MAX, "eps_label": EPS_LABEL}) + "\n")
    rows = [[i] + [d[k][i] for k in RANGES] + list(pick[i]) + list(hand[i]) + list(via[i]) + list(d["via_noise"][i]) for i in range(N_DRAWS)]
    U._csv(out + "/draws.csv", ["draw"] + list(RANGES) + ["pick_" + c for c in "xyz"] + ["handover_" + c for c in "xyz"] + ["via_" + c for c in "xyz"]
           + ["via_noise_" + str(j) for j in range(7)], rows)
    print("draws", N_DRAWS, "seed", SEED, "pick z range", round(float(pick[:, 2].min()), 4), round(float(pick[:, 2].max()), 4),
          "handover z range", round(float(hand[:, 2].min()), 4), round(float(hand[:, 2].max()), 4))


# ------------------------------------------------------------------
# the starts

def starts_main(out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.tasks import panda as P
    from sparsemax_dstl.tasks import workspace as W
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    t_start = time.perf_counter()
    plant = E.plant
    m = plant.model
    sc = U.scenario(U.WAITS[0])
    d = draws()
    nd = N_DRAWS
    ray, g, g_hand, g_via = targets(d, sc)
    person = U.instance(U.WAITS[0], PERSON)
    lo, hi = m.jnt_range[:, 0] + W.JOINT_MARGIN, m.jnt_range[:, 1] - W.JOINT_MARGIN
    with jax.enable_x64(True):  # inverse kinematics, the replay's margins and the exact conjuncts in float64, as E034's starts
        mx64 = mjx.put_model(m, impl="jax")
        hum = {k: jnp.asarray(person[k]) for k in ("human_centres", "human_radii")}
        ik_j = jax.jit(lambda tg, q: Wm.ik(mx64, plant, sc.robot_spacing, tg, q, jnp.asarray(lo), jnp.asarray(hi)))
        margins_j = jax.jit(jax.vmap(lambda X, p, hd: Wm.margins(mx64, plant, sc, {"pick": p, "handover": hd, **hum}, X)))

    def ik(tg, q):
        with jax.enable_x64(True):
            return tuple(np.asarray(x) for x in ik_j(jnp.asarray(tg, jnp.float64), jnp.asarray(q, jnp.float64)))

    def margins(X, p, hd):
        with jax.enable_x64(True):
            return np.asarray(margins_j(jnp.asarray(X, jnp.float64), jnp.asarray(p, jnp.float64), jnp.asarray(hd, jnp.float64)))
    zone, r = np.asarray(sc.zone), sc.zone_radius
    q0 = m.key_qpos[0].copy()
    Q, miss = ik(np.concatenate([g, g_hand, g_via]), np.broadcast_to(q0, (3 * nd, 7)))
    Q, miss = Q.reshape(3, nd, 7), miss.reshape(3, nd)
    q_hold, q_hand, q_via = Q[0], Q[1], Q[2] + d["via_noise"]
    um = np.asarray(P.torque_limit(), np.float32)
    n_sub = P.substeps(U.H_CTRL)
    mx32 = mjx.put_model(m, impl="jax")
    x0 = np.concatenate([q0, np.zeros(7)])
    t = np.arange(U.H * 50 + 1) * U.HS
    track = jax.jit(jax.vmap(lambda Qs: E.track(mx32, n_sub, jnp.asarray(x0, jnp.float32), Qs, jnp.asarray(um))))
    entry_win = (t >= U.T_VIA - 1e-9) & (t <= U.T_HOLD + 1e-9)
    entry_t = np.asarray([U.ENTRY[x] for x in U.DEPTHS])  # (depths, 3)

    def build(a, waits, di, ii):
        """Commands (B, N, 7), planned states and the entry configuration's residual for entry
        parameters a (B,), waits (B,), depth indices di and draw indices ii (B,)."""
        q_in, miss_in = ik(zone + (r - a[:, None] * r) * ray[ii], q_hold[ii])
        tk, qk = U.keyframes(waits, np.broadcast_to(q0, (len(a), 7)), q_via[ii], q_in, q_hold[ii], q_hand[ii], entry_t[di])
        Xp = U.plan_states(tk, qk, t)
        V = np.asarray(track(jnp.asarray(Xp, jnp.float32)))
        return np.clip(V / um, -1.0, 1.0).astype(np.float32), Xp, miss_in

    # calibration at the shortest wait (the plan before w + 1 does not depend on the wait)
    ii, di = (x.ravel() for x in np.meshgrid(np.arange(nd), np.arange(len(U.DEPTHS)), indexing="ij"))
    delta = np.asarray(U.DEPTHS)[di]
    a = delta.copy()
    hist = []
    for it in range(CALIBRATE + 1):  # over calibration updates (sequential by definition)
        V, _, _ = build(a, np.full(len(a), U.WAITS[0]), di, ii)
        Mz = margins(U.replay(V, x0), g[ii], g_hand[ii])[:, :, 2]
        depth = -np.where(entry_win, Mz, np.inf).min(1)
        hist.append({"a": a, "depth": depth})
        print("calibration", it, "largest depth error", float(np.abs(depth - delta).max()), flush=True)
        if it < CALIBRATE:
            a = a + (delta - depth)
    a_cal = a.reshape(nd, len(U.DEPTHS))
    # every (draw, depth, wait)
    Iv, Dv, Wv = (x.ravel() for x in np.meshgrid(np.arange(nd), np.arange(len(U.DEPTHS)), np.arange(len(U.WAITS)), indexing="ij"))
    waits = np.asarray(U.WAITS)[Wv]
    V, Xp, miss_in = build(a_cal[Iv, Dv], waits, Dv, Iv)
    X = U.replay(V, x0)
    Mg = margins(X, g[Iv], g_hand[Iv])
    # exact conjuncts per wait (one program each)
    conj = np.zeros((len(V), len(E.CONJUNCTS)))
    for j, w in enumerate(U.WAITS):  # over the four waits (programs)
        prog = E.core_program(U.scenario(w), len(person["human_radii"]))
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])

        def children(M_, prog=prog, root=root, r_idx=r_idx):
            v = stl_jax.evaluate(prog, M_, "exact")
            return jnp.concatenate([v[s] for s in root.sources], -1)[..., r_idx]
        sel = Wv == j
        with jax.enable_x64(True):
            conj[sel] = np.asarray(jax.jit(jax.vmap(children))(jnp.asarray(Mg[sel], jnp.float64)))
    upto = t[None] <= waits[:, None] + 1 + 1e-9
    pickw = (t[None] >= waits[:, None] - 1e-9) & upto
    deepest = -np.where(entry_win, Mg[:, :, 2], np.inf).min(1)
    k_in = np.sum(entry_win & (Mg[:, :, 2] < 0), 1)
    other_zone = np.where(upto & ~entry_win, Mg[:, :, 2], np.inf).min(1)
    pick_min = np.where(pickw, Mg[:, :, 0], np.inf).min(1)
    vmax = np.abs(V).max((1, 2))
    ci = {c: i for i, c in enumerate(E.CONJUNCTS)}
    checks = {
        "ik": np.maximum(np.maximum(miss[0], miss[1])[Iv], np.maximum(miss[2][Iv], miss_in)) <= W.IK_TOLERANCE,
        "via_in_range": np.all((q_via >= lo) & (q_via <= hi), 1)[Iv],
        "torque": vmax < 1.0 - 1e-6,
        "depth": np.abs(deepest - np.asarray(U.DEPTHS)[Dv]) <= DEPTH_TOL,
        "one_entry": other_zone >= 0,
        "pick": pick_min >= TRACK_MIN,
        "rules": np.minimum(np.minimum(conj[:, ci["separation"]], conj[:, ci["slowdown"]]), conj[:, ci["handover"]]) >= TRACK_MIN,
    }
    ok_plan = np.all(np.stack(list(checks.values())), 0)
    ok_draw = ok_plan.reshape(nd, -1).all(1)
    kept = np.nonzero(ok_draw)[0][:N_INSTANCES]
    last = kept[-1] if len(kept) else -1
    reasons = {c: v.reshape(nd, -1).all(1) for c, v in checks.items()}
    cand = [[i, bool(ok_draw[i]), int(np.searchsorted(kept, i)) if i in kept else -1, " ".join(c for c in checks if not reasons[c][i]),
             float(vmax.reshape(nd, -1)[i].max()), float(np.abs(deepest - np.asarray(U.DEPTHS)[Dv]).reshape(nd, -1)[i].max()),
             float(other_zone.reshape(nd, -1)[i].min()), float(pick_min.reshape(nd, -1)[i].min()),
             " ".join(str(x) for x in k_in.reshape(nd, len(U.DEPTHS), -1)[i, :, 0]), float(np.max(miss[:, i])),
             float(conj[:, ci["handover"]].reshape(nd, -1)[i].min()), float(conj[:, ci["order"]].reshape(nd, -1)[i].max())] for i in range(nd)]
    U._csv(out + "/candidates.csv", ["draw", "tracked", "instance", "failed_checks", "vmax", "depth_error_max", "other_zone_min", "pick_min",
                                     "entry_samples_by_depth", "ik_miss_max", "handover_conjunct_min", "order_conjunct_max"], cand)
    replaced = int(np.sum(~ok_draw[:last + 1])) if len(kept) else nd
    print("draws", nd, "tracked", int(ok_draw.sum()), "instances", len(kept), "draws replaced before the last instance", replaced)
    for c in checks:  # over the checks (a handful)
        print("check", c, "draws failing", int(np.sum(~reasons[c])))
    sel = np.isin(Iv, kept)
    inst = np.searchsorted(kept, Iv[sel])
    np.savez(out + "/starts.npz", V0=V[sel], X64=X[sel], Xplan=Xp[sel], x0=x0, wait=waits[sel], depth=np.asarray(U.DEPTHS)[Dv[sel]], instance=inst,
             draw=Iv[sel], a=a_cal[Iv[sel], Dv[sel]], pick=g[Iv[sel]], handover=g_hand[Iv[sel]], ray=ray[Iv[sel]], q_hold=q_hold[kept], q_hand=q_hand[kept],
             q_via=q_via[kept], q0=q0, kept=kept, ok_draw=ok_draw, entry_samples=k_in[sel], vmax=vmax[sel], conj=conj[sel],
             meta=dumps({"calibration": hist, "seconds": time.perf_counter() - t_start, "seed": SEED, "draws": nd, "instances": len(kept),
                         "replaced": replaced, "ranges": RANGES, "via_noise_sd": VIA_NOISE, "entry": U.ENTRY, "person": PERSON}))
    print("kept draws", kept.tolist())
    print("entry samples by instance (depth 0.05, 0.10)", k_in[sel].reshape(len(kept), len(U.DEPTHS), -1)[:, :, 0].tolist())
    print("vmax", round(float(vmax[sel].max()), 3), "seconds", round(time.perf_counter() - t_start, 1))


# ------------------------------------------------------------------
# the regime check

def until_batch(zone_s, pick_s, zone_m, pick_m, w, eps, entry, hold_from):
    """U.until_rows for B starts at once: zone_s, pick_s, zone_m, pick_m, entry (B, T). Returns a
    dict of arrays (B,) with U.until_rows' keys."""
    import jax
    import jax.numpy as jnp

    from sparsemax_dstl.jax.operators import sparsemax_weights
    B, T = zone_s.shape
    k0, k1 = int(round(w / U.HS)), int(round((w + 1) / U.HS))
    wit = np.arange(k0, k1 + 1)
    m = wit + 2
    cols = np.arange(T + 1)
    valid = cols[None, :] < m[:, None]  # (M, T + 1); column 0 is pick(t'), column c >= 1 is zone(c - 1)
    big = 1e9

    def stack(z_, p_):
        zp = np.concatenate([np.zeros((B, 1)), z_], 1)
        X_ = np.where(cols[None, None] == 0, p_[:, wit][:, :, None], zp[:, None, :])
        return np.where(valid[None], X_, big)
    Xs, Xm = stack(zone_s, pick_s), stack(zone_m, pick_m)
    inner_exact = Xm.min(2)  # (B, M)
    beta = np.log(m) / eps
    q = np.where(valid[None], np.exp(-beta[None, :, None] * (Xs - Xs.min(2, keepdims=True))), 0.0)
    soft = Xs.min(2) - np.log(q.sum(2)) / beta[None]
    q /= q.sum(2, keepdims=True)
    ent = np.concatenate([np.zeros((B, 1), bool), entry], 1)
    hold = np.concatenate([[False], np.arange(T) * U.HS >= hold_from - 1e-9])
    hold_w = hold[None] & valid
    gam = 2 * eps / (1 - 1 / m)
    with jax.enable_x64(True):
        sw = jax.vmap(lambda z, g, v: sparsemax_weights(jnp.where(v, z, -big), g))
        p = np.asarray(jax.vmap(lambda z: sw(z, jnp.asarray(gam), jnp.asarray(valid)))(jnp.asarray(-Xs)))
    supp = p > 0
    M = len(wit)
    bo = np.log(M) / eps
    c = soft.max(1)
    plain = c + np.log(np.sum(np.exp(bo * (soft - c[:, None])), 1)) / bo
    sound = plain - np.log(M) / bo
    u = np.exp(bo * (soft - c[:, None]))
    u /= u.sum(1, keepdims=True)
    d = np.argmax(u, 1)
    b = np.arange(B)
    hm = hold[1:] & (np.arange(T) <= wit[-1])
    z_hold = np.median(zone_m[:, hm], 1)
    deep = np.where(entry.any(1), np.where(entry, zone_m, np.inf).min(1), np.nan)
    k = entry.sum(1)
    share = (q * ent[:, None, :]).sum(2)
    ex_max = inner_exact.max(1)
    return {"witnesses": np.full(B, M), "m_first": np.full(B, m[0]), "m_last": np.full(B, m[-1]), "exact_until": ex_max, "plain_until": plain,
            "sound_until": sound, "plain_minus_sound": plain - sound, "plain_minus_exact": plain - ex_max,
            "n_tie": np.sum(inner_exact >= ex_max[:, None] - U.TIE, 1), "n_effective": np.exp(bo * (plain - c)),
            "deciding_witness_s": wit[d] * U.HS, "share_entry_deciding": share[b, d], "share_entry_first": share[:, 0],
            "share_entry_last": share[:, -1], "share_entry_weighted": np.sum(u * share, 1), "share_pick_weighted": np.sum(u * q[:, :, 0], 1),
            "k_entry": k, "entry_depth": -deep, "hold_level": z_hold, "gap": z_hold - deep, "gamma_over_k": gam[d] / np.maximum(k, 1),
            "support_deciding": supp[b, d].sum(1), "support_hold_deciding": (supp[b, d] & hold_w[d]).sum(1),
            "support_entry_deciding": (supp[b, d] & ent).sum(1), "support_hold_any": (supp & hold_w[None]).sum((1, 2)),
            "sparsemax_entry_mass_deciding": (p[b, d] * ent).sum(1)}


def regime_main(start_dir, out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl import jax as stl_jax
    from sparsemax_dstl.jax import methods
    from sparsemax_dstl.tasks import workspace_mjx as Wm
    jax.config.update("jax_enable_x64", True)
    t_start = time.perf_counter()
    z = np.load(start_dir + "/starts.npz")
    plant = E.plant
    mx = mjx.put_model(plant.model, impl="jax")
    X = z["X64"]
    t = np.arange(X.shape[1]) * U.HS
    person = U.instance(U.WAITS[0], PERSON)
    hc, hr = jnp.asarray(person["human_centres"]), jnp.asarray(person["human_radii"])
    inst = lambda p, hd: {"pick": p, "handover": hd, "human_centres": hc, "human_radii": hr}  # noqa: E731
    rows = []
    for w in U.WAITS:  # over the four waits (one program each)
        sc = U.scenario(w)
        prog = E.core_program(sc, len(person["human_radii"]))
        root = prog.steps[prog.root]
        r_idx = np.asarray(root.index[0, :root.count[0]])
        sel = np.nonzero(np.isclose(z["wait"], w))[0]
        args = (jnp.asarray(X[sel]), jnp.asarray(z["pick"][sel]), jnp.asarray(z["handover"][sel]))
        Mg = np.asarray(jax.vmap(lambda x, p, hd: Wm.margins(mx, plant, sc, inst(p, hd), x))(*args))
        Zs = np.asarray(jax.vmap(lambda x, p, hd: Wm.scores(mx, plant, sc, inst(p, hd), x))(*args))
        ev = {}
        for name in ("exact",) + U.ARMS:  # over arms
            for eps in ((None,) if name == "exact" else U.EPS):  # over eps
                sem = "exact" if name == "exact" else methods.SEMANTICS[name]
                f = jax.jit(jax.vmap(lambda M_, e=eps, s=sem: (lambda v: (v[-1][:, 0] if v[-1].ndim > 1 else v[-1][0],
                                                                     jnp.concatenate([v[j] for j in root.sources], -1)[..., r_idx]))(stl_jax.evaluate(prog, M_, s, e))))
                ev[(name, eps)] = [np.asarray(x) for x in f(jnp.asarray(Mg if name == "exact" else Zs))]
        entry = (Mg[:, :, 2] < 0) & (t[None] <= U.T_HOLD + 1e-9)
        ex_root, ex_conj = ev[("exact", None)]
        for eps in U.EPS:  # over eps
            u = until_batch(Zs[:, :, 2], Zs[:, :, 0], Mg[:, :, 2], Mg[:, :, 0], w, eps, entry, U.T_HOLD)
            cols = {"instance": z["instance"][sel], "draw": z["draw"][sel], "wait": np.full(len(sel), w), "depth": z["depth"][sel], "plan": sel,
                    "eps": np.full(len(sel), eps), "root_exact": ex_root, **{"c_" + c: ex_conj[:, j] for j, c in enumerate(E.CONJUNCTS)},
                    "deciding": np.asarray(E.CONJUNCTS)[np.argmin(ex_conj, 1)]}
            for name in U.ARMS:  # over arms
                r_, c_ = ev[(name, eps)]
                cols["root_" + name], cols["order_" + name] = r_, c_[:, 2]
                cols["min_other_" + name] = np.delete(c_, 2, 1).min(1)
            cols.update(u)
            cols["order_plain_check"] = cols["order_lse_plain"] - u["plain_until"]
            cols["delta_ge_gamma_over_k"] = u["gap"] >= u["gamma_over_k"]
            keys = list(cols)
            rows += [{k: (cols[k][i].item() if hasattr(cols[k][i], "item") else cols[k][i]) for k in keys} for i in range(len(sel))]
    rows.sort(key=lambda r: (r["eps"] != 0.4, r["instance"], r["depth"], r["wait"]))
    with open(out + "/regime.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    labels = label_rows(rows)
    U._csv(out + "/labels.csv", list(labels[0]), [list(r.values()) for r in labels])
    for r in rows:  # text table, one line per row
        print("instance", r["instance"], "wait", r["wait"], "depth", r["depth"], "eps", r["eps"], "exact", round(r["root_exact"], 4), "deciding", r["deciding"],
              "root plain/sound/sparsemax", round(r["root_lse_plain"], 4), round(r["root_lse"], 4), round(r["root_sparsemax"], 4),
              "k", r["k_entry"], "gap", round(r["gap"], 4), "gamma/k", round(r["gamma_over_k"], 4), "support", r["support_deciding"],
              "support_hold", r["support_hold_deciding"], "support_hold_any", r["support_hold_any"], "share_entry", round(r["share_entry_deciding"], 3),
              "n_tie", r["n_tie"], "plain-exact", round(r["plain_minus_exact"], 4), "check", round(r["order_plain_check"], 6))
    for r in labels:  # the labels
        print("label", r)
    print("seconds", round(time.perf_counter() - t_start, 1))


def label_rows(rows, eps=EPS_LABEL, share_max=SHARE_MAX):
    """Per (instance, depth): the two conditions at eps over the regime rows (dicts), and the plain
    arm's root at the start at each eps and wait."""
    out = {}
    w_last = max(U.WAITS)
    for r in rows:  # over regime rows (table rows)
        key = (int(r["instance"]), float(r["depth"]))
        o = out.setdefault(key, {"instance": key[0], "depth": key[1], "draw": int(r["draw"]), "share_longest_wait": np.nan, "support_hold_max": 0,
                                 "support_hold_any_max": 0, "k_entry_min": 10 ** 9, "k_entry_max": 0, "gap_min": np.inf, "gamma_over_k_max": 0.0})
        if float(r["eps"]) == eps:
            if float(r["wait"]) == w_last:
                o["share_longest_wait"] = float(r["share_entry_deciding"])
            o["support_hold_max"] = max(o["support_hold_max"], int(r["support_hold_deciding"]))
            o["support_hold_any_max"] = max(o["support_hold_any_max"], int(r["support_hold_any"]))
            o["k_entry_min"], o["k_entry_max"] = min(o["k_entry_min"], int(r["k_entry"])), max(o["k_entry_max"], int(r["k_entry"]))
            o["gap_min"], o["gamma_over_k_max"] = min(o["gap_min"], float(r["gap"])), max(o["gamma_over_k_max"], float(r["gamma_over_k"]))
            o["share_wait_" + str(r["wait"])] = float(r["share_entry_deciding"])
        o["plain_root_eps" + str(r["eps"]) + "_wait" + str(r["wait"])] = float(r["root_lse_plain"])
        o["deciding_wait" + str(r["wait"])] = r["deciding"]
    labels = []
    for key in sorted(out):  # over (instance, depth) (table rows)
        o = out[key]
        o["share_ok"] = bool(o["share_longest_wait"] <= share_max)
        o["support_ok"] = bool(o["support_hold_max"] == 0)
        o["inside"] = o["share_ok"] and o["support_ok"]
        labels.append(o)
    return labels


# ------------------------------------------------------------------
# setup and the runs (no JAX in the run)

def setup_main(start_dir, out, eps, instances, arms, person=PERSON, waits=U.WAITS, depths=U.DEPTHS):
    """The run table of one batch; person is a key of W.UNTIL_PERSON (the robot's plans do not
    depend on it), its sphere centres taken at each run's wait."""
    z = np.load(start_dir + "/starts.npz")
    W_, A_, I_, D_ = np.meshgrid(np.arange(len(waits)), np.arange(len(arms)), np.arange(len(instances)), np.arange(len(depths)), indexing="ij")
    rw, ra, ri, rd = (np.asarray(waits)[W_.ravel()], np.asarray(arms)[A_.ravel()], np.asarray(instances)[I_.ravel()], np.asarray(depths)[D_.ravel()])
    key = np.stack([np.round(z["wait"], 6), np.round(z["depth"], 6), z["instance"]], 1)
    want = np.stack([np.round(rw, 6), np.round(rd, 6), ri], 1)
    hit = np.all(want[:, None] == key[None], -1)
    if not np.all(hit.sum(1) == 1):
        raise ValueError("every run needs exactly one start")
    pi = np.argmax(hit, 1)
    n = len(pi)
    hcw = np.stack([U.instance(w, person)["human_centres"] for w in waits])  # (waits, T, S_h, 3); over the waits (configurations)
    I0 = U.instance(waits[0], person)
    np.savez(out, x0=np.broadcast_to(z["x0"], (n, 14)).astype(np.float32), V0=z["V0"][pi], hc=hcw[W_.ravel()].astype(np.float32),
             pick=z["pick"][pi], handover=z["handover"][pi], hr=I0["human_radii"], n_h=len(I0["human_radii"]), start_dir=start_dir, plan=pi,
             run_person=np.full(n, person), run_eps=np.full(n, float(eps)), run_wait=rw, run_method=ra, run_depth=rd, run_instance=ri,
             run_start=np.zeros(n, int))
    print("runs", n, "eps", eps, "person", person, "instances", list(instances), "arms", list(arms))


def load(path, runs="all"):
    z = np.load(path)
    sel = np.arange(len(z["run_method"])) if runs == "all" else np.asarray([int(i) for i in runs.split(",")])
    I = {k: z[k][sel] for k in ("x0", "V0", "hc", "pick", "handover", "plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth",
                                "run_instance", "run_start")}
    I.update(hr=z["hr"], n_h=int(z["n_h"]), index=sel)
    key = [(str(m_), float(e), float(w)) for m_, e, w in zip(I["run_method"], I["run_eps"], I["run_wait"])]
    groups, a = [], 0
    for b in range(1, len(key) + 1):  # contiguous (method, eps, wait) blocks of the run table (a handful)
        if b == len(key) or key[b] != key[a]:
            groups.append((key[a][0], key[a][1], a, b, key[a][2]))
            a = b
    if len(set((g[0], g[1], g[4]) for g in groups)) != len(groups):
        raise ValueError("each (method, eps, wait) must be one contiguous block")
    I["groups"] = groups
    return I


def run_main(path, iterations, out, runs="all"):
    import warp as wp

    from sparsemax_dstl.warp import solver as CW
    t0 = time.perf_counter()
    I = load(path, runs)
    progs, chain, referee = U.build(I)
    setup_seconds = time.perf_counter() - t0
    state = CW.init_state(I["V0"], U.ALPHA0)

    def log(k, rec, r64, wall):
        print("iterate", k, "seconds", round(wall, 2), "accepted", int(rec[:, 10].sum()), "of", len(rec), "rho", np.round(rec[:, 1], 4).tolist(),
              "referee64", np.round(r64, 4).tolist(), flush=True)

    res = CW.solve(chain, referee, state, iterations, U.LAM, U.DELTA, log=log)
    meta = {"host": socket.gethostname(), "device": str(chain.device), "warp": wp.config.version, "instance": path, "runs": runs, "iterations": state["k"],
            "every": CW.EVERY, "lam": U.LAM, "delta": U.DELTA, "alpha0": U.ALPHA0, "trials": CW.TRIALS, "armijo_c": CW.ARMIJO_C, "mu0": CW.MU0,
            "mu_max": CW.MU_MAX, "alpha_min": CW.ALPHA_MIN, "viol_tol": CW.VIOL_TOL, "trace_keys": CW.TRACE_KEYS, "groups": I["groups"],
            "setup_seconds": setup_seconds, "seconds_keys": res["seconds_keys"]}
    np.savez(out, meta=dumps(meta), index=I["index"], trace=res["trace"], V=np.concatenate([np.asarray(I["V0"], np.float32)[:, None], res["V"]], 1),
             referee64=res["referee64"], conjuncts64=res["conjuncts64"], seconds=res["seconds"], wall=res["wall"], alpha=state["alpha"], nu=state["nu"],
             mu=state["mu"], k=state["k"], **{k: I[k] for k in ("plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_instance",
                                                                 "run_start", "pick", "handover")})
    med = np.median(res["seconds"], 0)
    print("setup seconds", round(setup_seconds, 2), "seconds per iterate (median by component)", dumps(dict(zip(res["seconds_keys"], np.round(med, 4)))),
          "total", round(float(np.median(res["seconds"].sum(1))), 3))


# ------------------------------------------------------------------
# tables and statistics

def run_outcomes(rs, ex, delta=U.DELTA):
    """Per run, from the arm's smoothed robustness rs (n, K) and the certificate ex (n, K) at
    iterates 0..K-1: first claim (-1 if none), iterates claimed while unsafe, the deepest
    certificate while claiming (nan if no claim), the first safe iterate (-1 if none), whether it
    stays safe after it, the certificate at the end, a claim at the end and a false claim there."""
    n, K = ex.shape
    claim = rs >= delta
    safe_ok = ex >= delta
    first_safe = np.where(safe_ok.any(1), np.argmax(safe_ok, 1), -1)
    tail = np.flip(np.logical_and.accumulate(np.flip(safe_ok, 1), 1), 1)
    return {"first_claim": np.where(claim.any(1), np.argmax(claim, 1), -1), "false_claim_iterates": np.sum(claim & ~safe_ok, 1),
            "deepest_while_claiming": np.where(claim.any(1), np.where(claim, ex, np.inf).min(1), np.nan), "first_safe": first_safe,
            "stays_safe": np.where(first_safe >= 0, tail[np.arange(n), np.maximum(first_safe, 0)], False), "exact_end": ex[:, -1],
            "claims_end": claim[:, -1], "false_claim_end": claim[:, -1] & ~safe_ok[:, -1]}


def paired(fa, fb, K, rng, n_boot=N_BOOT):
    """Per-instance differences of first safe iterates fa - fb (arrays (n,), -1 = never safe,
    counted as K). Returns counts earlier (fa < fb), equal, later, the median difference and a 95%
    percentile interval of the median from n_boot resamples of the instances."""
    a, b = np.where(fa < 0, K, fa), np.where(fb < 0, K, fb)
    d = (a - b).astype(float)
    n = len(d)
    if n == 0:
        return {"n": 0, "earlier": 0, "equal": 0, "later": 0, "median": np.nan, "lo": np.nan, "hi": np.nan}
    med = np.median(d[rng.integers(0, n, (n_boot, n))], 1)
    return {"n": n, "earlier": int(np.sum(d < 0)), "equal": int(np.sum(d == 0)), "later": int(np.sum(d > 0)), "median": float(np.median(d)),
            "lo": float(np.percentile(med, 2.5)), "hi": float(np.percentile(med, 97.5))}


def tables_main(out_dir, tag, labels_csv, paths):
    with open(labels_csv) as fh:
        lab = {(int(r["instance"]), round(float(r["depth"]), 6)): r["inside"] == "True" for r in csv.DictReader(fh)}
    cols = {k: [] for k in ("eps", "arm", "wait", "depth", "instance")}
    outs, K = [], None
    for p in paths:  # over run records (files)
        z = np.load(p)
        tr, ex = z["trace"], z["referee64"]
        rs = np.concatenate([tr[:, :, 1], tr[:, -1:, 12]], 1)
        K = ex.shape[1] if K is None else K
        if ex.shape[1] != K:
            raise ValueError("every record needs the same number of iterates")
        outs.append(run_outcomes(rs, ex))
        for k, src in (("eps", "run_eps"), ("arm", "run_method"), ("wait", "run_wait"), ("depth", "run_depth"), ("instance", "run_instance")):
            cols[k].append(z[src])
    R = {k: np.concatenate(v) for k, v in cols.items()}
    R.update({k: np.concatenate([o[k] for o in outs]) for k in outs[0]})
    R["arm"] = R["arm"].astype(str)
    R["inside"] = np.asarray([lab[(int(i), round(float(d), 6))] for i, d in zip(R["instance"], R["depth"])])
    order = np.lexsort((R["instance"], R["depth"], R["wait"], R["arm"], R["eps"]))
    R = {k: v[order] for k, v in R.items()}
    if len(set(zip(R["eps"], R["arm"], R["wait"], R["depth"], R["instance"]))) != len(R["eps"]):
        raise ValueError("a run appears twice")
    names = list(R)
    U._csv(out_dir + "/" + tag + "_runs.csv", names, [[R[k][i] for k in names] for i in range(len(R["eps"]))])
    regions = (("inside", True), ("outside", False), ("all", None))
    snd, att, pr = [], [], []
    cells = sorted(set(zip(R["eps"], R["arm"], R["wait"], R["depth"])))
    for e, arm, w, dp in cells:  # over cells (table rows)
        for reg, flag in regions:  # over regions
            s = (R["eps"] == e) & (R["arm"] == arm) & (R["wait"] == w) & (R["depth"] == dp) & ((R["inside"] == flag) if flag is not None else True)
            fc = s & (R["false_claim_iterates"] > 0)
            fs = R["first_safe"][s]
            fsk = np.where(fs < 0, K, fs).astype(float)
            snd.append([e, arm, w, dp, reg, int(s.sum()), int(fc.sum()), int(np.sum(s & R["false_claim_end"])), int(np.sum(s & (R["exact_end"] < U.DELTA))),
                        float(np.min(R["deepest_while_claiming"][fc])) if fc.any() else np.nan, " ".join(str(x) for x in R["first_claim"][s]),
                        " ".join(str(x) for x in R["instance"][s])])
            att.append([e, arm, w, dp, reg, int(s.sum()), " ".join(str(x) for x in fs), int(np.sum(fs < 0)), float(np.median(fsk)) if s.any() else np.nan,
                        int(np.sum(R["stays_safe"][s])), float(np.min(R["exact_end"][s])) if s.any() else np.nan])
    arms = sorted(set(R["arm"]) & set(SOUND), key=SOUND.index)
    rng = np.random.default_rng(BOOT_SEED)
    for e in sorted(set(R["eps"])):  # over eps
        for ia, A in enumerate(arms):  # over pairs of sound arms (a handful)
            for B in arms[ia + 1:]:
                for w in sorted(set(R["wait"])):  # over waits
                    for dp in sorted(set(R["depth"])):  # over depths
                        for reg, flag in regions:  # over regions
                            base = (R["eps"] == e) & (R["wait"] == w) & (R["depth"] == dp) & ((R["inside"] == flag) if flag is not None else True)
                            sa, sb = base & (R["arm"] == A), base & (R["arm"] == B)
                            if not np.array_equal(R["instance"][sa], R["instance"][sb]):
                                raise ValueError("the two arms need the same instances")
                            q = paired(R["first_safe"][sa], R["first_safe"][sb], K, rng)
                            pr.append([e, A, B, w, dp, reg, q["n"], q["earlier"], q["equal"], q["later"], q["median"], q["lo"], q["hi"],
                                       int(np.sum(R["first_safe"][sa] < 0)), int(np.sum(R["first_safe"][sb] < 0)),
                                       " ".join(str(x) for x in R["first_safe"][sa]), " ".join(str(x) for x in R["first_safe"][sb])])
    head = ["eps", "arm", "wait", "depth", "region", "instances"]
    U._csv(out_dir + "/" + tag + "_soundness.csv", head + ["instances_with_false_claim", "instances_false_claim_at_end", "instances_ending_unsafe",
                                                        "deepest_exact_while_claiming", "first_claim_by_instance", "instance_ids"], snd)
    U._csv(out_dir + "/" + tag + "_attenuation.csv", head + ["first_safe_by_instance", "never_safe", "first_safe_median_never_as_K",
                                                          "instances_staying_safe", "exact_end_min"], att)
    U._csv(out_dir + "/" + tag + "_paired.csv", ["eps", "arm_a", "arm_b", "wait", "depth", "region", "instances", "a_earlier", "equal", "a_later",
                                                 "median_a_minus_b", "lo95", "hi95", "a_never_safe", "b_never_safe", "a_first_safe", "b_first_safe"], pr)
    print("iterates recorded", K, "(a run never safe counts as", K, ") bootstrap seed", BOOT_SEED, "resamples", N_BOOT)
    for row in snd:  # printout
        if row[1] == "lse_plain" or row[6] > 0:
            print("soundness", row[:10])
    for row in pr:  # printout
        print("paired", row[:15])


if __name__ == "__main__":
    mode = sys.argv[1]
    strs = lambda a: tuple(a.split(","))  # noqa: E731
    floats = lambda a: tuple(float(x) for x in a.split(","))  # noqa: E731
    if mode == "draws":
        draws_main(sys.argv[2])
    elif mode == "starts":
        starts_main(sys.argv[2])
    elif mode == "regime":
        regime_main(sys.argv[2], sys.argv[3])
    elif mode == "setup":
        setup_main(sys.argv[2], sys.argv[3], float(sys.argv[4]), tuple(int(x) for x in sys.argv[5].split(",")), strs(sys.argv[6]), sys.argv[7],
                   *((floats(sys.argv[8]), floats(sys.argv[9])) if len(sys.argv) > 9 else ()))
    elif mode == "run":
        run_main(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif mode == "tables":
        tables_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise ValueError("unknown mode " + mode)

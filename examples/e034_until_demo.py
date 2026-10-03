"""E034: the demo on the Until with three smoothings, the plain log-sum-exp, the sound
log-sum-exp and sparsemax, on E033's single-shooting constrained solve (M003 note and D006
amendment of 2026-10-01 03:29Z).

    JAX_PLATFORMS=cpu python -m examples.e034_until_demo starts <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e034_until_demo regime <start_dir> <out_dir>
    JAX_PLATFORMS=cpu python -m examples.e034_until_demo setup <start_dir> <out.npz> <persons> <waits> <depths> <arms> <eps> [<starts>]
    JAX_PLATFORMS=nojax python -m examples.e034_until_demo run <instance.npz> <iterations> <out.npz> <resume.npz|-> <runs|all>
    python -m examples.e034_until_demo report <out_dir> <tag> <run.npz>...
    python -m examples.e034_until_demo tables <out_dir> <tag> <regime.csv> <run.npz>...

Lists are comma-separated. The run axis of setup is person, eps, wait, arm, depth, start (start
fastest); each (eps, wait, arm) block shares one STL evaluator, and each wait has its own program.

Scene (tasks.workspace.until_instance, the flag Scenario.until_hold = HOLD). H = 10 s, h = h_s =
0.02 s, 2 ms physics. The pick target lies on the zone ball's boundary, HOLD metres outside it,
on the ray DIRECTION from the zone centre (toward the robot and up); the handover target lies on
the same ray HANDOVER_RADIUS from the centre, inside the zone. Windows for a wait w: the order
conjunct's until over the pick window [w, w + 1] s; the handover F_[F0, F0 + F_LEN] G_[0, DWELL]
with F0 = w + 1 + T_TR. The longest wait at which the handover window and the dwell fit the
10 s horizon is 10 - 1 - T_TR - F_LEN - DWELL = 7.22 s. Separation and slow-down as they are.

The plan (one per wait, entry depth and start): joint keyframes with C1 cosine blends, from the
model's home configuration q0 (held until T_LEAVE) to a via configuration (the site VIA_OUT
further out on the ray and VIA_UP higher; start j > 0 perturbs it by SIGMA_VIA rad per joint) at
T_VIA, then an overshoot into the zone: the entry configuration (the site on the ray, a zone
radius times `a` inside the boundary; a < 0 is outside, where the tracking's overshoot alone
enters the zone) at ENTRY[delta][0], held for ENTRY[delta][1], then the hold configuration (the
site at the pick target) at ENTRY[delta][2], held until w + 1, then the handover
configuration by F0 - 0.04, held to the end. The plan is tracked by e022_regime.track (the
computed-torque PD law, float32, as E029's starts), and the commands are the start. The entry
parameter `a` is calibrated per depth and start so that the float64 MuJoCo C replay of the
commands enters the zone by the depth delta (zone margin minimum -delta): three secant-free
updates a <- a + (delta - measured depth). The person does not move the robot's plan, so one set
of starts serves both persons.

regime: the check before any run, on every start (person, wait, depth, start), from the float64
replay: the exact margins and the smoothed atoms (tasks.workspace.margins, scores), the program's
root and conjuncts under every arm at eps 0.4 and 0.2, and at the Until (closed prefix, t = 0):
for every witness t' in the pick window, the inner minimum over pick(t') and zone(0..t'); the
sound arm's weights on the entry samples (inside the zone, margin < 0), sparsemax's support at
gamma = 2 eps / (1 - 1/m) and whether it contains a hold sample (from T_HOLD to t'); the gap Delta
(median zone score on the hold minus the deepest entry score) against gamma / k, k the number of
entry samples; at the outer maximum the plain value minus the sound value, the plain value minus
the exact value, the number of witnesses tied with the exact maximum (within 1e-9) and the
effective number exp(beta (plain - largest smoothed inner value)).
"""
import csv
import json
import socket
import sys
import time
from fractions import Fraction

import numpy as np

H = 10
H_CTRL = Fraction(1, 50)
HS = 0.02
WAITS = (2.0, 4.0, 6.0, 7.22)
DEPTHS = (0.05, 0.10)
PERSONS = ("zone", "out")
HOLD = 0.005
ELEV = 0.6
DIRECTION = (-float(np.cos(ELEV)), 0.0, float(np.sin(ELEV)))
HANDOVER_RADIUS = 0.10
T_TR, F_LEN, DWELL = 0.8, 0.18, 0.8
VIA_OUT, VIA_UP = 0.15, 0.10
T_LEAVE, T_VIA = 0.3, 1.0
# per depth: (arrival at the entry configuration, time held there, arrival at the hold configuration), seconds;
# chosen by the regime check (gate E034/2026-10-01T0337Z, starts3 to starts6): sparsemax's support at the
# Until's inner minimum excludes the hold samples, and the sound share is at most 0.2 at the longest wait
ENTRY = {0.05: (1.5, 0.18, 2.0), 0.10: (1.4, 0.0, 1.8)}
T_HOLD = max(e[2] for e in ENTRY.values())  # the hold samples of the regime check start here
N_STARTS = 4
SIGMA_VIA = 0.05
START_SEED = 2026100103
CALIBRATE = 3
ARMS = ("lse_plain", "lse", "sparsemax")
EPS = (0.4, 0.2)
LAM, DELTA, ALPHA0 = 0.01, 0.0, 0.002  # E029 and E033, unchanged
TIE = 1e-9


def dumps(obj):
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def frac(x):
    return Fraction(str(round(float(x), 2)))


def scenario(w):
    from sparsemax_dstl.tasks import workspace as W
    f0 = frac(w) + 1 + frac(T_TR)
    return W.Scenario(H=Fraction(H), h_s=H_CTRL, pick=(frac(w) / H, (frac(w) + 1) / H), handover=(f0 / H, (f0 + frac(F_LEN)) / H),
                      dwell=frac(DWELL) / H, until_hold=HOLD)


def instance(w, person):
    from sparsemax_dstl.tasks import workspace as W
    return W.until_instance(scenario(w), DIRECTION, HANDOVER_RADIUS, person)


# ------------------------------------------------------------------
# the plans and the starts

def plan_states(tk, qk, t):
    """Sampled states (B, T, 14) of keyframes (times tk (B, K), configurations qk (B, K, 7)) with C1
    cosine blends between consecutive keyframes, at times t (T,)."""
    j = np.clip(np.sum(t[None, :, None] >= tk[:, None, :], -1) - 1, 0, tk.shape[1] - 2)  # (B, T)
    t0, t1 = np.take_along_axis(tk, j, 1), np.take_along_axis(tk, j + 1, 1)
    s = np.clip((t[None] - t0) / (t1 - t0), 0.0, 1.0)
    b, db = 0.5 - 0.5 * np.cos(np.pi * s), 0.5 * np.pi * np.sin(np.pi * s) / (t1 - t0)
    q0, q1 = np.take_along_axis(qk, j[..., None], 1), np.take_along_axis(qk, j[..., None] + 1, 1)
    return np.concatenate([q0 + b[..., None] * (q1 - q0), db[..., None] * (q1 - q0)], -1)


def keyframes(w, q0, q_via, q_in, q_hold, q_hand, entry):
    """Keyframe times (B, 9) and configurations (B, 9, 7) for waits w (B,), per-plan configurations
    (B, 7) and the entry timing (B, 3) of ENTRY."""
    B = len(w)
    f0 = np.asarray(w) + 1 + T_TR
    e = np.asarray(entry)
    tk = np.stack([np.zeros(B), np.full(B, T_LEAVE), np.full(B, T_VIA), e[:, 0], e[:, 0] + e[:, 1], e[:, 2],
                   np.asarray(w) + 1, f0 - 0.04, np.full(B, float(H))], -1)
    qk = np.stack([q0, q0, q_via, q_in, q_in, q_hold, q_hold, q_hand, q_hand], 1)
    return tk, qk


def replay(V, x0):
    """The float64 MuJoCo C states (B, T, 14) of normalized commands V (B, N, 7) from x0 (14,), as
    constrained_warp.Referee.states."""
    import os

    import mujoco
    from mujoco import rollout as mj_rollout

    from sparsemax_dstl.tasks import panda as P
    mjm = P.model()
    B = len(V)
    datas = [mujoco.MjData(mjm) for _ in range(min(B, len(os.sched_getaffinity(0))))]
    U = np.repeat(np.asarray(V, np.float64) * np.asarray(P.torque_limit(), np.float64), 10, axis=1)
    state0 = np.broadcast_to(np.concatenate([[0.0], x0]), (B, 15)).copy()
    st, _ = mj_rollout.rollout(mjm, datas, state0, U)
    return np.concatenate([np.broadcast_to(x0, (B, 1, 14)), st[:, 9::10, 1:]], 1)


def starts_main(out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl.tasks import panda as P
    from sparsemax_dstl.tasks import workspace as W
    t_start = time.perf_counter()
    plant = E.plant
    m = plant.model
    sc = scenario(WAITS[0])
    inst0 = instance(WAITS[0], "out")
    with jax.enable_x64(True):  # inverse kinematics and the replay's margins in float64, as tasks.workspace._instances
        mx64 = mjx.put_model(m, impl="jax")
        lo, hi = jnp.asarray(m.jnt_range[:, 0] + W.JOINT_MARGIN), jnp.asarray(m.jnt_range[:, 1] - W.JOINT_MARGIN)
        ik_j = jax.jit(lambda tg, q: W.ik(mx64, plant, sc.robot_spacing, tg, q, lo, hi))
        inst64 = {k: jnp.asarray(inst0[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
        margins_j = jax.jit(jax.vmap(lambda X: W.margins(mx64, plant, sc, inst64, X)))

    def ik(tg, q):
        with jax.enable_x64(True):
            return ik_j(jnp.asarray(tg, jnp.float64), jnp.asarray(q, jnp.float64))

    def margins(X):
        with jax.enable_x64(True):
            return margins_j(jnp.asarray(X, jnp.float64))
    zone, r, n = np.asarray(sc.zone), sc.zone_radius, np.asarray(DIRECTION) / np.linalg.norm(DIRECTION)
    g, g_hand = W.until_targets(sc, DIRECTION, HANDOVER_RADIUS)
    g_via = g + VIA_OUT * n + np.array([0.0, 0.0, VIA_UP])
    q0 = m.key_qpos[0].copy()
    Q, miss = ik(np.stack([g, g_hand, g_via]), np.broadcast_to(q0, (3, 7)))
    q_hold, q_hand, q_via0 = np.asarray(Q)
    rng = np.random.default_rng(START_SEED)
    q_via = q_via0 + np.concatenate([np.zeros((1, 7)), SIGMA_VIA * rng.standard_normal((N_STARTS - 1, 7))])
    um = np.asarray(P.torque_limit(), np.float32)
    n_sub = P.substeps(H_CTRL)
    mx32 = mjx.put_model(m, impl="jax")
    x0 = np.concatenate([q0, np.zeros(7)])
    t = np.arange(H * 50 + 1) * HS
    track = jax.jit(jax.vmap(lambda Qs: E.track(mx32, n_sub, jnp.asarray(x0, jnp.float32), Qs, jnp.asarray(um))))
    entry = (t >= T_VIA) & (t <= T_HOLD)

    def build(a, waits, entry_t):
        """Commands (B, N, 7) for entry parameters a (B,), waits (B,) and entry timings (B, 3), with the start's via."""
        g_in = zone + (r - a[:, None] * r) * n
        q_in, _ = ik(g_in, np.broadcast_to(q_hold, (len(a), 7)))
        tk, qk = keyframes(waits, np.broadcast_to(q0, (len(a), 7)), q_via[sidx], np.asarray(q_in), np.broadcast_to(q_hold, (len(a), 7)),
                           np.broadcast_to(q_hand, (len(a), 7)), entry_t)
        Xp = plan_states(tk, qk, t)
        U = np.asarray(track(jnp.asarray(Xp, jnp.float32)))
        return np.clip(U / um, -1.0, 1.0).astype(np.float32), Xp

    # calibration at the shortest wait (the plan and the tracking before w + 1 do not depend on the wait)
    D, S = np.meshgrid(np.asarray(DEPTHS), np.arange(N_STARTS), indexing="ij")
    delta, sidx = D.ravel(), S.ravel()
    a = delta.copy()
    hist = []
    for it in range(CALIBRATE + 1):  # over calibration updates (a handful, sequential by definition)
        V, _ = build(a, np.full(len(a), WAITS[0]), np.asarray([ENTRY[d] for d in delta]))
        X = replay(V, x0)
        Mz = np.asarray(margins(X))[:, :, 2]
        depth = -np.where(entry, Mz, np.inf).min(1)
        hist.append({"a": a.tolist(), "depth": depth.tolist()})
        print("calibration", it, "a", np.round(a, 5).tolist(), "depth", np.round(depth, 5).tolist(), "samples inside", np.sum(entry & (Mz < 0), 1).tolist(), flush=True)
        if it < CALIBRATE:
            a = a + (delta - depth)
    a_cal = a
    # every (wait, depth, start)
    Wv, Dv, Sv = (x.ravel() for x in np.meshgrid(np.asarray(WAITS), np.arange(len(DEPTHS)), np.arange(N_STARTS), indexing="ij"))
    sidx = Sv
    V, Xp = build(a_cal.reshape(len(DEPTHS), N_STARTS)[Dv, Sv], Wv, np.asarray([ENTRY[DEPTHS[d]] for d in Dv]))
    X = replay(V, x0)
    np.savez(out + "/starts.npz", V0=V, X64=X, Xplan=Xp, x0=x0, wait=Wv, depth=np.asarray(DEPTHS)[Dv], start=Sv, a=a_cal, q_hold=q_hold, q_hand=q_hand,
             q_via=q_via, q0=q0, ik_miss=np.asarray(miss), pick=g, handover=g_hand,
             meta=dumps({"calibration": hist, "seconds": time.perf_counter() - t_start, "hold": HOLD, "direction": DIRECTION, "handover_radius": HANDOVER_RADIUS,
                         "waits": WAITS, "depths": DEPTHS, "keyframes": [T_LEAVE, T_VIA], "entry": ENTRY, "t_tr": T_TR, "sigma_via": SIGMA_VIA,
                         "seed": START_SEED, "vmax": np.abs(V).max((1, 2)), "clipped_fraction": np.mean(np.abs(V) >= 1 - 1e-6, (1, 2))}))
    print("ik miss", np.asarray(miss).tolist(), "seconds", round(time.perf_counter() - t_start, 1))
    print("vmax per plan", np.round(np.abs(V).max((1, 2)), 3).tolist())


# ------------------------------------------------------------------
# the regime check

def until_rows(zone_s, pick_s, zone_m, pick_m, w, eps, entry, hold_from):
    """The Until at t = 0 from the smoothed atoms (zone_s, pick_s (T,)) and the exact margins: per
    witness t' in [w, w + 1] the inner minimum over pick(t'), zone(0..t') (a labeled host loop over
    the 51 witnesses would be a Python loop over samples, so the witnesses are stacked as a padded
    matrix). Returns a dict of the measurements described in the module docstring."""
    import jax
    import jax.numpy as jnp

    from sparsemax_dstl.operators import sparsemax_weights
    T = len(zone_s)
    k0, k1 = int(round(w / HS)), int(round((w + 1) / HS))
    wit = np.arange(k0, k1 + 1)  # witness samples
    m = wit + 2  # inner arity: pick(t') and zone(0..t')
    cols = np.arange(T + 1)
    valid = cols[None, :] < m[:, None]  # column 0 is pick(t'), column c >= 1 is zone(c - 1)
    zpad = np.concatenate([[0.0], zone_s])
    Xs = np.where(cols[None] == 0, pick_s[wit][:, None], zpad[None])
    Xm = np.where(cols[None] == 0, pick_m[wit][:, None], np.concatenate([[0.0], zone_m])[None])
    big = 1e9
    Xs, Xm = np.where(valid, Xs, big), np.where(valid, Xm, big)
    inner_exact = Xm.min(1)
    beta = np.log(m) / eps
    lq = -beta[:, None] * (Xs - Xs.min(1, keepdims=True))
    q = np.where(valid, np.exp(lq), 0.0)
    soft = Xs.min(1) - np.log(q.sum(1)) / beta  # the sound (= plain) inner minimum
    q /= q.sum(1, keepdims=True)
    ent = np.concatenate([[False], entry])
    hold = np.concatenate([[False], (np.arange(T) * HS >= hold_from - 1e-9)])
    hold_w = hold[None] & valid & (cols[None] <= m[:, None] - 1)
    gam = 2 * eps / (1 - 1 / m)
    with jax.enable_x64(True):
        p = np.asarray(jax.vmap(lambda z, g, v: sparsemax_weights(jnp.where(v, z, -big), g))(jnp.asarray(-Xs), jnp.asarray(gam), jnp.asarray(valid)))
    supp = p > 0
    # outer maximum over the witnesses
    M = len(wit)
    bo = np.log(M) / eps
    c = soft.max()
    plain = c + np.log(np.sum(np.exp(bo * (soft - c)))) / bo
    sound = plain - np.log(M) / bo
    u = np.exp(bo * (soft - c))
    u /= u.sum()
    d = int(np.argmax(u))
    z_hold = np.median(zone_m[hold[1:] & (np.arange(T) <= wit[-1])])
    deep = zone_m[entry].min() if entry.any() else np.nan
    k = int(entry.sum())
    share = (q * ent[None]).sum(1)
    return {"witnesses": M, "m_first": int(m[0]), "m_last": int(m[-1]), "exact_until": float(inner_exact.max()), "plain_until": float(plain),
            "sound_until": float(sound), "plain_minus_sound": float(plain - sound), "plain_minus_exact": float(plain - inner_exact.max()),
            "n_tie": int(np.sum(inner_exact >= inner_exact.max() - TIE)), "n_effective": float(np.exp(bo * (plain - c))),
            "deciding_witness_s": float(wit[d] * HS), "share_entry_deciding": float(share[d]), "share_entry_first": float(share[0]),
            "share_entry_last": float(share[-1]), "share_entry_weighted": float(np.sum(u * share)), "share_pick_weighted": float(np.sum(u * q[:, 0])),
            "k_entry": k, "entry_depth": float(-deep), "hold_level": float(z_hold), "gap": float(z_hold - deep), "gamma_over_k": float(gam[d] / max(k, 1)),
            "support_deciding": int(supp[d].sum()), "support_hold_deciding": int((supp[d] & hold_w[d]).sum()),
            "support_entry_deciding": int((supp[d] & ent).sum()), "support_hold_any": int((supp & hold_w).sum()),
            "sparsemax_entry_mass_deciding": float((p[d] * ent).sum())}


def regime_main(start_dir, out):
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    from examples import e022_regime as E
    from sparsemax_dstl import stl
    from sparsemax_dstl.core_study import methods
    from sparsemax_dstl.tasks import workspace as W
    jax.config.update("jax_enable_x64", True)
    z = np.load(start_dir + "/starts.npz")
    plant = E.plant
    mx = mjx.put_model(plant.model, impl="jax")
    X = z["X64"]
    t = np.arange(X.shape[1]) * HS
    rows = []
    for person in PERSONS:  # over the two persons
        for w in WAITS:  # over waits (four)
            sc = scenario(w)
            I = instance(w, person)
            inst = {k: jnp.asarray(I[k]) for k in ("pick", "handover", "human_centres", "human_radii")}
            prog = E.core_program(sc, len(I["human_radii"]))
            root = prog.steps[prog.root]
            r_idx = np.asarray(root.index[0, :root.count[0]])
            sel = np.nonzero(np.isclose(z["wait"], w))[0]
            Mg = np.asarray(jax.vmap(lambda x: W.margins(mx, plant, sc, inst, x))(jnp.asarray(X[sel])))
            Zs = np.asarray(jax.vmap(lambda x: W.scores(mx, plant, sc, inst, x))(jnp.asarray(X[sel])))
            ev = {}
            for name in ("exact",) + ARMS:  # over arms
                for eps in ((None,) if name == "exact" else EPS):  # over eps
                    sem = "exact" if name == "exact" else methods.SEMANTICS[name]
                    f = jax.jit(jax.vmap(lambda M_, e=eps, s=sem: (lambda v: (v[-1][:, 0] if v[-1].ndim > 1 else v[-1][0],
                                                                         jnp.concatenate([v[j] for j in root.sources], -1)[..., r_idx]))(stl.evaluate(prog, M_, s, e))))
                    ev[(name, eps)] = [np.asarray(x) for x in f(jnp.asarray(Mg if name == "exact" else Zs))]
            for i, s_ in enumerate(sel):  # over the starts of this wait (eight; table rows)
                entry = (Mg[i, :, 2] < 0) & (t <= T_HOLD)
                for eps in EPS:  # over eps
                    u = until_rows(Zs[i, :, 2], Zs[i, :, 0], Mg[i, :, 2], Mg[i, :, 0], w, eps, entry, T_HOLD)
                    row = {"person": person, "wait": w, "depth": float(z["depth"][s_]), "start": int(z["start"][s_]), "plan": int(s_), "eps": eps,
                           "root_exact": float(ev[("exact", None)][0][i]), **{"c_" + c: float(v) for c, v in zip(E.CONJUNCTS, ev[("exact", None)][1][i])},
                           "deciding": E.CONJUNCTS[int(np.argmin(ev[("exact", None)][1][i]))]}
                    for name in ARMS:  # over arms
                        row["root_" + name] = float(ev[(name, eps)][0][i])
                        row["order_" + name] = float(ev[(name, eps)][1][i][2])
                        row["min_other_" + name] = float(np.delete(ev[(name, eps)][1][i], 2).min())
                    row.update(u)
                    row["order_plain_check"] = row["order_lse_plain"] - u["plain_until"]
                    row["delta_ge_gamma_over_k"] = bool(u["gap"] >= u["gamma_over_k"])
                    rows.append(row)
    keys = list(rows[0])
    with open(out + "/regime.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:  # text table, one line per row
        print(r["person"], r["wait"], r["depth"], r["start"], "eps", r["eps"], "exact", round(r["root_exact"], 4), "deciding", r["deciding"],
              "conj", [round(r["c_" + c], 3) for c in E.CONJUNCTS], "root plain/sound/sparsemax", round(r["root_lse_plain"], 4), round(r["root_lse"], 4),
              round(r["root_sparsemax"], 4), "order plain/sound/sparsemax", round(r["order_lse_plain"], 4), round(r["order_lse"], 4), round(r["order_sparsemax"], 4),
              "k", r["k_entry"], "depth", round(r["entry_depth"], 5), "gap", round(r["gap"], 4), "gamma/k", round(r["gamma_over_k"], 4),
              "support", r["support_deciding"], "support_hold", r["support_hold_deciding"], "share_entry", round(r["share_entry_deciding"], 3),
              round(r["share_entry_weighted"], 3), "n_tie", r["n_tie"], "n_eff", round(r["n_effective"], 2), "plain-exact", round(r["plain_minus_exact"], 4),
              "plain-sound", round(r["plain_minus_sound"], 4), "check", round(r["order_plain_check"], 6))


# ------------------------------------------------------------------
# setup and the runs (no JAX in the run)

def setup_main(start_dir, out, persons, waits, depths, arms, eps_, starts=None):
    z = np.load(start_dir + "/starts.npz")
    starts = tuple(range(N_STARTS)) if starts is None else starts
    rows = []
    for person in persons:  # the run axis: person, eps, wait, arm, depth, start (start fastest)
        for e in eps_:
            for w in waits:
                for arm in arms:
                    for d in depths:
                        for s in starts:
                            p = int(np.nonzero(np.isclose(z["wait"], w) & np.isclose(z["depth"], d) & (z["start"] == s))[0][0])
                            rows.append((person, e, w, arm, d, s, p))
    pi = np.asarray([r[6] for r in rows])
    hc = np.stack([instance(r[2], r[0])["human_centres"] for r in rows]).astype(np.float32)
    I0 = instance(waits[0], persons[0])
    np.savez(out, x0=np.broadcast_to(z["x0"], (len(rows), 14)).astype(np.float32), V0=z["V0"][pi], hc=hc, pick=I0["pick"], handover=I0["handover"],
             hr=I0["human_radii"], n_h=len(I0["human_radii"]), start_dir=start_dir, plan=pi,
             run_person=np.asarray([r[0] for r in rows]), run_eps=np.asarray([r[1] for r in rows]), run_wait=np.asarray([r[2] for r in rows]),
             run_method=np.asarray([r[3] for r in rows]), run_depth=np.asarray([r[4] for r in rows]), run_start=np.asarray([r[5] for r in rows]))
    print("runs", len(rows), "hc", hc.shape)


def load(path, runs="all"):
    z = np.load(path)
    sel = np.arange(len(z["run_method"])) if runs == "all" else np.asarray([int(i) for i in runs.split(",")])
    I = {k: z[k][sel] for k in ("x0", "V0", "hc", "plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_start")}
    I.update(pick=z["pick"], handover=z["handover"], hr=z["hr"], n_h=int(z["n_h"]), index=sel)
    key = [(str(m_), float(e), float(w)) for m_, e, w in zip(I["run_method"], I["run_eps"], I["run_wait"])]
    groups, a = [], 0
    for b in range(1, len(key) + 1):  # contiguous (method, eps, wait) blocks of the run table (a handful)
        if b == len(key) or key[b] != key[a]:
            groups.append((key[a][0], key[a][1], a, b, key[a][2]))
            a = b
    I["groups"] = groups
    return I


def build(I, device="cuda:0"):
    """Programs per wait, Chain and Referee for the runs of I."""
    from examples import e022_regime as R
    from sparsemax_dstl import constrained_warp as CW
    n = len(I["x0"])
    progs = {w: R.core_program(scenario(w), I["n_h"]) for w in sorted({g[4] for g in I["groups"]})}
    sc = scenario(min(progs))
    groups = [(m_, e, a, b, progs[w]) for m_, e, a, b, w in I["groups"]]
    blocks = [(a, b, progs[w]) for _, _, a, b, w in I["groups"]]
    pick, hand = np.broadcast_to(I["pick"], (n, 3)), np.broadcast_to(I["handover"], (n, 3))
    chain = CW.Chain(R.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], groups, device=device)
    referee = CW.Referee(R.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], device=device, blocks=blocks)
    return progs, chain, referee


def run_main(path, iterations, out, resume, runs, stop=False):
    import warp as wp

    from sparsemax_dstl import constrained_warp as CW
    t0 = time.perf_counter()
    I = load(path, runs)
    progs, chain, referee = build(I)
    setup_seconds = time.perf_counter() - t0
    prev = np.load(resume) if resume != "-" else None
    if prev is not None:
        if not np.array_equal(prev["index"], I["index"]):
            raise ValueError("the resumed record holds other runs")
        state = {"V": prev["V"][:, -1].copy(), "alpha": prev["alpha"], "nu": prev["nu"], "mu": prev["mu"], "k": int(prev["k"])}
        ref = (prev["referee64"][:, -1], prev["conjuncts64"][:, -1])
    else:
        state = CW.init_state(I["V0"], ALPHA0)
        ref = None

    def log(k, rec, r64, wall):
        print("iterate", k, "seconds", round(wall, 2), "accepted", int(rec[:, 10].sum()), "of", len(rec), "rho", np.round(rec[:, 1], 4).tolist(),
              "referee64", np.round(r64, 4).tolist(), flush=True)

    res = CW.solve(chain, referee, state, iterations, LAM, DELTA, ref=ref, log=log, stop_certified=stop)
    meta = {"host": socket.gethostname(), "device": str(chain.device), "warp": wp.config.version, "instance": path, "runs": runs, "iterations": state["k"],
            "every": CW.EVERY, "lam": LAM, "delta": DELTA, "alpha0": ALPHA0, "trials": CW.TRIALS, "armijo_c": CW.ARMIJO_C, "mu0": CW.MU0, "mu_max": CW.MU_MAX,
            "alpha_min": CW.ALPHA_MIN, "viol_tol": CW.VIOL_TOL, "trace_keys": CW.TRACE_KEYS, "groups": I["groups"], "setup_seconds": setup_seconds, "stop_certified": stop,
            "seconds_keys": res["seconds_keys"], "resumed_from": ([resume] if resume != "-" else []) + (json.loads(str(prev["meta"]))["resumed_from"] if prev is not None else [])}
    join = lambda k, new: np.concatenate([prev[k], new], 1) if prev is not None else new  # noqa: E731
    V_all = join("V", res["V"]) if prev is not None else np.concatenate([np.asarray(I["V0"], np.float32)[:, None], res["V"]], 1)
    np.savez(out, meta=dumps(meta), index=I["index"], trace=join("trace", res["trace"]), V=V_all,
             referee64=join("referee64", res["referee64"][:, 1:]) if prev is not None else res["referee64"],
             conjuncts64=join("conjuncts64", res["conjuncts64"][:, 1:]) if prev is not None else res["conjuncts64"],
             seconds=np.concatenate([prev["seconds"], res["seconds"]]) if prev is not None else res["seconds"],
             wall=np.concatenate([prev["wall"], prev["wall"][-1] + res["wall"][1:]]) if prev is not None else res["wall"],
             alpha=state["alpha"], nu=state["nu"], mu=state["mu"], k=state["k"],
             **{k: I[k] for k in ("plan", "run_person", "run_eps", "run_wait", "run_method", "run_depth", "run_start")})
    med = np.median(res["seconds"], 0)
    print("setup seconds", round(setup_seconds, 2), "seconds per iterate (median by component)", dumps(dict(zip(res["seconds_keys"], np.round(med, 4)))),
          "total", round(float(np.median(res["seconds"].sum(1))), 3))


# ------------------------------------------------------------------
# outcomes

def outcomes(rho_smooth, exact, conj, delta=DELTA):
    """Per run, from the arm's own smoothed robustness rho_smooth (n, K) and the float64 certificate
    exact (n, K) at the same iterates, with the root conjunction's children conj (n, K, c):
    claim, the first iterate with rho_smooth >= delta (-1 if none), the certificate there and the
    child that decides it (the smallest; -1 if no claim), false_safe (the certificate there below
    delta); safe, the first iterate with exact >= delta (-1 if none); the same claim record at the
    last iterate; and lead, safe minus claim when both exist (positive: the claim came first)."""
    n, K = exact.shape
    ok = rho_smooth >= delta
    claim = np.where(ok.any(1), np.argmax(ok, 1), -1)
    safe_ok = exact >= delta
    safe = np.where(safe_ok.any(1), np.argmax(safe_ok, 1), -1)
    at = np.maximum(claim, 0)
    ex_claim = np.where(claim >= 0, exact[np.arange(n), at], np.nan)
    row_claim = np.where(claim >= 0, np.argmin(conj[np.arange(n), at], -1), -1)
    return {"claim": claim, "exact_at_claim": ex_claim, "row_at_claim": row_claim, "false_safe": (claim >= 0) & (ex_claim < delta), "safe": safe,
            "claim_end": ok[:, -1], "exact_end": exact[:, -1], "rho_end": rho_smooth[:, -1], "row_end": np.argmin(conj[:, -1], -1),
            "false_safe_end": ok[:, -1] & (exact[:, -1] < delta), "lead": np.where((claim >= 0) & (safe >= 0), safe - claim, np.where(claim >= 0, K - claim, 0))}


def report_main(out_dir, tag, paths):
    from examples import e022_regime as E
    fin, rep, tim = [], [], []
    for p in paths:  # over run records (files)
        z = np.load(p)
        tr, rho, conj = z["trace"], z["referee64"], z["conjuncts64"]
        n, K = rho.shape
        r_s = np.concatenate([tr[:, :, 1], tr[:, -1:, 12]], 1)  # the smoothed robustness at iterates 0..K-1 (the last from the accepted trial)
        o = outcomes(r_s, rho, conj)
        wall = z["wall"]
        for r in range(n):  # over runs (table rows)
            key = [str(z["run_person"][r]), float(z["run_eps"][r]), float(z["run_wait"][r]), str(z["run_method"][r]), float(z["run_depth"][r]), int(z["run_start"][r])]
            fin.append(key + [int(z["index"][r]), rho[r, 0], r_s[r, 0], int(o["claim"][r]), o["exact_at_claim"][r],
                              E.CONJUNCTS[o["row_at_claim"][r]] if o["row_at_claim"][r] >= 0 else "", bool(o["false_safe"][r]), int(o["safe"][r]),
                              bool(o["claim_end"][r]), o["exact_end"][r], o["rho_end"][r], E.CONJUNCTS[o["row_end"][r]], bool(o["false_safe_end"][r]),
                              int(o["lead"][r]), K - 1, wall[o["safe"][r]] if o["safe"][r] >= 0 else np.nan, n, p])
            rep += [key + [k, rho[r, k], r_s[r, k]] + list(conj[r, k]) for k in range(K)]
        sec = z["seconds"]
        tim.append([p, n, len(sec)] + list(np.median(sec, 0)) + [float(np.median(sec.sum(1))), float(wall[-1])])
    head = ["person", "eps", "wait", "arm", "depth", "start"]
    _csv(out_dir + "/" + tag + "_final.csv", head + ["run", "exact_start", "smooth_start", "claim_iterate", "exact_at_claim", "row_at_claim", "false_safe",
                                                     "safe_iterate", "claim_at_end", "exact_end", "smooth_end", "row_end", "false_safe_at_end",
                                                     "iterates_claim_before_safe", "iterations", "wall_seconds_to_safe_in_batch", "batch_runs", "record"], fin)
    _csv(out_dir + "/" + tag + "_replay.csv", head + ["iteration", "exact", "smooth"] + ["c_" + c for c in E.CONJUNCTS], rep)
    keys = json.loads(str(np.load(paths[0])["meta"]))["seconds_keys"]
    _csv(out_dir + "/" + tag + "_timing.csv", ["record", "runs", "iterates"] + list(keys) + ["total", "wall_seconds"], tim)
    cells = {}
    for row in fin:  # group the runs by (person, eps, wait, arm, depth), starts in order (table rows)
        cells.setdefault(tuple(row[:5]), []).append(row)
    summ = []
    for key, v in sorted(cells.items()):  # over cells (table rows)
        v.sort(key=lambda r: r[5])
        claims = [r for r in v if r[9] >= 0]
        fs = [r for r in claims if r[12]]
        safe = np.array([r[13] for r in v], float)
        safe[safe < 0] = np.nan
        summ.append(list(key) + [len(v), len(claims), len(fs), len(fs) / len(v), float(np.min([r[10] for r in fs])) if fs else np.nan,
                                 " ".join(str(r[9]) for r in v), " ".join(str(r[13]) for r in v), int(np.isfinite(safe).sum()),
                                 float(np.nanmedian(safe)) if np.isfinite(safe).any() else np.nan, sum(r[18] for r in v),
                                 float(np.min([r[15] for r in v]))])
    _csv(out_dir + "/" + tag + "_summary.csv", head[:5] + ["runs", "claims", "false_safe_claims", "false_safe_fraction", "deepest_hidden_violation",
                                                           "claim_iterate_by_start", "safe_iterate_by_start", "safe_runs", "safe_iterate_median",
                                                           "false_safe_at_end", "exact_end_min"], summ)
    for row in summ:  # printout
        print("cell", row)
    for row in fin:  # printout
        print(row[:6], "claim", row[9], "exact there", None if np.isnan(row[10]) else round(row[10], 4), row[11], "false_safe", row[12], "safe", row[13],
              "end exact", round(row[15], 4), "smooth", round(row[16], 4), "false_safe_end", row[18])


def tables_main(out_dir, tag, regime_csv, paths):
    """The grid's tables per (eps, arm, wait, depth), starts in order: <tag>_soundness.csv (runs that
    claim safety while the exact robustness is negative, the first claim iterate, the number of
    iterates claimed while unsafe, the deepest exact value while claiming, how the run ends) and
    <tag>_attenuation.csv (the first exactly safe iterate, the four counts and their median, whether
    the run stays safe after it, the exact value at the end), each with the sound arm's share on the
    entry samples from the regime table (range over the starts of that wait and depth)."""
    with open(regime_csv) as fh:
        reg = [r for r in csv.DictReader(fh) if r["person"] == "out"]
    share = {}
    for r in reg:  # over regime rows (table rows)
        share.setdefault((float(r["eps"]), float(r["wait"]), float(r["depth"])), []).append(float(r["share_entry_deciding"]))
    runs = []
    for p in paths:  # over run records (files)
        z = np.load(p)
        tr, ex = z["trace"], z["referee64"]
        rs = np.concatenate([tr[:, :, 1], tr[:, -1:, 12]], 1)
        claim = rs >= DELTA
        bad = claim & (ex < DELTA)
        safe_ok = ex >= DELTA
        first_safe = np.where(safe_ok.any(1), np.argmax(safe_ok, 1), -1)
        tail = np.flip(np.logical_and.accumulate(np.flip(safe_ok, 1), 1), 1)
        stays = np.where(first_safe >= 0, tail[np.arange(len(ex)), np.maximum(first_safe, 0)], False)
        for r in range(len(ex)):  # over runs (table rows)
            if str(z["run_person"][r]) != "out":  # the study's scene (root, 04:36Z)
                continue
            runs.append({"eps": float(z["run_eps"][r]), "arm": str(z["run_method"][r]), "wait": float(z["run_wait"][r]), "depth": float(z["run_depth"][r]),
                         "start": int(z["run_start"][r]), "first_claim": int(np.argmax(claim[r])) if claim[r].any() else -1,
                         "false_claim_iterates": int(bad[r].sum()), "deepest_while_claiming": float(ex[r][claim[r]].min()) if claim[r].any() else np.nan,
                         "first_safe": int(first_safe[r]), "stays_safe": bool(stays[r]), "exact_end": float(ex[r, -1]),
                         "claims_end": bool(claim[r, -1]), "iterates": ex.shape[1] - 1})
    cells = {}
    for u in runs:  # group by cell, starts in order (table rows)
        cells.setdefault((u["eps"], u["arm"], u["wait"], u["depth"]), []).append(u)
    snd, att = [], []
    for key, v in sorted(cells.items()):  # over cells (table rows)
        v.sort(key=lambda u: u["start"])
        sh = share.get((key[0], key[2], key[3]), [np.nan])
        shs = str(round(min(sh), 3)) + "-" + str(round(max(sh), 3))
        fs = [u for u in v if u["false_claim_iterates"] > 0]
        snd.append(list(key) + [len(v), len(fs), " ".join(str(u["first_claim"]) for u in v), " ".join(str(u["false_claim_iterates"]) for u in v),
                                float(np.nanmin([u["deepest_while_claiming"] for u in v])) if any(u["first_claim"] >= 0 for u in v) else np.nan,
                                sum(u["exact_end"] < DELTA for u in v), sum(u["claims_end"] and u["exact_end"] < DELTA for u in v), shs])
        fsafe = np.array([u["first_safe"] for u in v], float)
        fsafe[fsafe < 0] = np.nan
        att.append(list(key) + [len(v), " ".join(str(u["first_safe"]) for u in v), int(np.isfinite(fsafe).sum()),
                                float(np.nanmedian(fsafe)) if np.isfinite(fsafe).any() else np.nan, sum(u["stays_safe"] for u in v),
                                " ".join(str(round(u["exact_end"], 3)) for u in v), shs])
    head = ["eps", "arm", "wait", "depth", "runs"]
    _csv(out_dir + "/" + tag + "_soundness.csv", head + ["runs_claiming_while_unsafe", "first_claim_by_start", "iterates_claimed_while_unsafe_by_start",
                                                       "deepest_exact_while_claiming", "runs_ending_unsafe", "runs_ending_claiming_while_unsafe",
                                                       "sound_share_entry_regime"], snd)
    _csv(out_dir + "/" + tag + "_attenuation.csv", head + ["first_safe_by_start", "runs_safe", "first_safe_median", "runs_staying_safe",
                                                         "exact_end_by_start", "sound_share_entry_regime"], att)
    _csv(out_dir + "/" + tag + "_runs.csv", list(runs[0]), [list(u.values()) for u in runs])
    for row in snd:  # printout
        print("soundness", row)
    for row in att:  # printout
        print("attenuation", row)


def _csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    mode = sys.argv[1]
    strs = lambda a: tuple(a.split(","))  # noqa: E731
    floats = lambda a: tuple(float(x) for x in a.split(","))  # noqa: E731
    if mode == "starts":
        starts_main(sys.argv[2])
    elif mode == "regime":
        regime_main(sys.argv[2], sys.argv[3])
    elif mode == "setup":
        setup_main(sys.argv[2], sys.argv[3], strs(sys.argv[4]), floats(sys.argv[5]), floats(sys.argv[6]), strs(sys.argv[7]), floats(sys.argv[8]),
                   tuple(int(x) for x in sys.argv[9].split(",")) if len(sys.argv) > 9 else None)
    elif mode == "run":
        run_main(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], len(sys.argv) > 7 and sys.argv[7] == "stop")
    elif mode == "report":
        report_main(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "tables":
        tables_main(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        raise ValueError("unknown mode " + mode)

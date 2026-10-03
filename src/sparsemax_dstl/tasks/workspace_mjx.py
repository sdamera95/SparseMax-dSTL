"""The manipulator beside a person in MJX: the predicate values of workspace.layout() and their exact margins, the
rollout at the physics step, inverse kinematics of the site, and the seeded instances."""
from functools import cache, partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from ..jax import mjx_implicit
from . import human as Hm
from .workspace import (FILTER_STRIDE, HANDOVER_MIN_Z, HELDOUT_SEED, IK_ITERATIONS, IK_TOLERANCE, JOINT_MARGIN, N_CANDIDATES,
                        N_HELDOUT, N_TUNING, PER_INSTANCE, PICK_BOX, Plant, Q_SPREAD, REGIME_CLEAR, REGIME_RANGES, SCRIPT_RANGES,
                        SEP_MARGIN, Scenario, TUNING_SEED, ZONE_MARGIN, human_inputs, regime_episode, regime_script_of,
                        robot_spheres, script_of)


# ------------------------------------------------------------------
# kinematics

def points(mx, plant, spacing, q):
    """Site position (3,) and robot sphere centres (S, 3) at configuration q, from MJX kinematics."""
    s = robot_spheres(plant, spacing)
    site = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_SITE, plant.site)
    d = mjx.kinematics(mx, mjx.make_data(mx).replace(qpos=q))
    C = d.xpos[s["body"]] + jnp.einsum("lij,lj->li", d.xmat[s["body"]], jnp.asarray(s["centre"], q.dtype))
    return d.site_xpos[site], C


# ------------------------------------------------------------------
# atoms

def _n_plus(x, eps):
    return jnp.sqrt(jnp.sum(x * x, -1) + eps ** 2)


def _n_minus(x, eps):
    s = jnp.sum(x * x, -1)
    return s / (jnp.sqrt(s + eps ** 2) + eps) if eps > 0 else jnp.sqrt(s)


def _atoms(mx, plant, sc, pick, hand, hc, hr, x, eps_l, eps_v):
    nq = plant.model.nq
    q, qd = x[:nq], x[nq:]
    (p, C), (_, Cd) = jax.jvp(partial(points, mx, plant, sc.robot_spacing), (q,), (qd,))
    r = jnp.asarray(robot_spheres(plant, sc.robot_spacing)["radius"], x.dtype)
    zone = jnp.asarray(sc.zone, x.dtype)
    rg, rz, vs = sc.r_goal, sc.zone_radius, sc.v_slow
    goal = lambda g: (rg - _n_plus(p - g, eps_l)) / rg
    zone_atom = (_n_minus(p - zone, eps_l) - rz) / rz
    speed = (vs - _n_plus(Cd, eps_v)) / vs
    dist = _n_minus(C[:, None] - hc[None], eps_l)  # (S_r, S_h)
    R = r[:, None] + hr[None] + sc.d_min
    S = r[:, None] + hr[None] + sc.d_slow
    return jnp.concatenate([goal(pick)[None], goal(hand)[None], zone_atom[None], speed,
                            ((dist - R) / R).ravel(), ((dist - S) / S).ravel()])


def _trace(plant, sc, inst, X, eps_l, eps_v, mx):
    dt = X.dtype
    f = partial(_atoms, mx, plant, sc, jnp.asarray(inst["pick"], dt), jnp.asarray(inst["handover"], dt))
    hr = jnp.asarray(inst["human_radii"], dt)
    return jax.vmap(lambda hc, x: f(hc, hr, x, eps_l, eps_v))(jnp.asarray(inst["human_centres"], dt), X)


def scores(mx, plant, sc, inst, X):
    """Atoms (T, P) along states X (T, 2 nq), sample k with the human spheres of sample k.
    inst carries pick (3,), handover (3,), human_centres (T, S_h, 3) and human_radii (S_h,)."""
    return _trace(plant, sc, inst, X, sc.eps_length, sc.eps_speed, mx)


def margins(mx, plant, sc, inst, X):
    """The referee's unsmoothed normalized margins, in the layout of scores(); gradients stop
    at X. The referee uses the sphere covers, the geometry the atoms use, so each margin is at
    least its atom and the referee argument of the D006 amendment holds entry by entry."""
    return _trace(plant, sc, inst, jax.lax.stop_gradient(X), 0.0, 0.0, mx)


# ------------------------------------------------------------------
# dynamics at the STL sampling interval

def physics_rollout(mx, n_sub, x0, U, step=mjx_implicit.step):
    """Every physics state (N n_sub + 1, 2 nq) from x0 under U (N, nu), each command held for
    n_sub physics steps, the solver warm start reset at every interval as in
    tasks.panda.interval_map."""
    nq = x0.shape[-1] // 2

    def interval(x, u):
        d = mjx.make_data(mx).replace(qpos=x[:nq], qvel=x[nq:], ctrl=u)

        def sub(d, _):
            d = step(mx, d)
            return d, jnp.concatenate([d.qpos, d.qvel])

        _, Y = jax.lax.scan(sub, d, None, length=n_sub)
        return Y[-1], Y

    _, Y = jax.lax.scan(interval, x0, U)
    return jnp.concatenate([x0[None], Y.reshape(-1, x0.shape[-1])])


# ------------------------------------------------------------------
# instances

def ik(mx, plant, spacing, targets, q_init, lo, hi, iterations=IK_ITERATIONS, damping=1e-3):
    """Site-position inverse kinematics by damped least squares, vectorized over targets (B, 3):
    q <- clip(q + J^T (J J^T + damping I)^-1 (target - p), lo, hi), from q_init (B, nq)."""
    site = lambda q: points(mx, plant, spacing, q)[0]

    def one(target, q):
        def step(q, _):
            J = jax.jacfwd(site)(q)
            dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(3), target - site(q))
            return jnp.clip(q + dq, lo, hi), None

        q, _ = jax.lax.scan(step, q, None, length=iterations)
        return q, jnp.linalg.norm(site(q) - target)

    return jax.vmap(one)(targets, q_init)


def instances(seed, n, sc=Scenario(), plant=Plant()):
    """The first n valid instances of a seeded candidate stream, for the scenario sc.

    Each candidate draws:
    - a start configuration q0, the model's home keyframe plus uniform noise of Q_SPREAD rad per
      joint, JOINT_MARGIN inside the joint ranges;
    - a pick target uniformly in PICK_BOX (x, |y| with a random side, z), beside the zone;
    - a handover target uniformly in the part of the zone ball within zone_radius - ZONE_MARGIN
      of its centre and at least HANDOVER_MIN_Z above the table;
    - a human script uniformly in SCRIPT_RANGES.
    The pick and handover configurations solve site-position inverse kinematics (ik) from the
    home keyframe, JOINT_MARGIN inside the joint ranges, and the poses are the site positions
    there, so both are reachable. A
    candidate is valid when:
    - both IK solves end within IK_TOLERANCE of their targets;
    - the start site and the pick pose lie at least ZONE_MARGIN outside the zone ball;
    - every robot sphere clears every human sphere by d_min + SEP_MARGIN beyond the radii: at q0
      over [0, a H], at the pick configuration over the pick window, and at the handover
      configuration from c H to H (the robot may hold the part there to the end), at every
      FILTER_STRIDE-th STL sample.
    These are necessary conditions at the configurations the task must visit; that a whole
    trajectory satisfies the specification is shown by a witness path (examples.e019_witness).
    Kinematics run in float64. instances(seed, k) is a prefix of instances(seed, n).
    """
    out = _instances(seed, sc, plant)
    if out["accepted"] < n:
        raise ValueError("only " + str(out["accepted"]) + " of " + str(N_CANDIDATES) + " candidates are valid")
    return {k: (v[:n].copy() if k in PER_INSTANCE else v) for k, v in out.items()}


@cache
def _instances(seed, sc, plant):
    m = plant.model
    rng = np.random.default_rng([seed, 0])
    lo, hi = m.jnt_range[:, 0] + JOINT_MARGIN, m.jnt_range[:, 1] - JOINT_MARGIN
    n = N_CANDIDATES
    q0 = np.clip(m.key_qpos[0] + rng.uniform(-Q_SPREAD, Q_SPREAD, (n, m.nq)), lo, hi)
    (x0, x1), (y0, y1), (z0, z1) = PICK_BOX
    pick_t = np.stack([rng.uniform(x0, x1, n), rng.choice([-1.0, 1.0], n) * rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)], -1)
    r = (sc.zone_radius - ZONE_MARGIN) * rng.random(n) ** (1 / 3)
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])  # the upper half ball
    hand_t = np.asarray(sc.zone) + r[:, None] * v / np.linalg.norm(v, axis=-1, keepdims=True)
    draws = {k: rng.uniform(a, b, n) for k, (a, b) in SCRIPT_RANGES.items()}
    with jax.enable_x64(True):
        mx = mjx.put_model(m, impl="jax")
        home = jnp.broadcast_to(jnp.asarray(m.key_qpos[0]), (2 * n, m.nq))
        Qg, miss = jax.jit(partial(ik, mx, plant, sc.robot_spacing, lo=jnp.asarray(lo), hi=jnp.asarray(hi)))(
            jnp.asarray(np.concatenate([pick_t, hand_t])), home)
        Qg, miss = np.asarray(Qg).reshape(2, n, m.nq).transpose(1, 0, 2), np.asarray(miss).reshape(2, n).T
        Q = np.concatenate([q0[:, None], Qg], 1)
        p, C = jax.jit(jax.vmap(jax.vmap(partial(points, mx, plant, sc.robot_spacing))))(jnp.asarray(Q))
        p, C = np.asarray(p), np.asarray(C)
    radial = np.linalg.norm(p - np.array(sc.zone), axis=-1)  # (C, 3): distance from the zone centre
    ok = np.all(miss <= IK_TOLERANCE, -1) & (hand_t[:, 2] >= HANDOVER_MIN_Z)
    ok &= (radial[:, 0] >= sc.zone_radius + ZONE_MARGIN) & (radial[:, 1] >= sc.zone_radius + ZONE_MARGIN)
    idx = np.nonzero(ok)[0]
    # separation at the three configurations over their windows (human motion per candidate)
    H, hs = float(sc.H), float(sc.h_s)
    t = np.arange(sc.samples) * hs
    spans = [(0.0, float(sc.pick[0]) * H), (float(sc.pick[0]) * H, float(sc.pick[1]) * H), (float(sc.handover[0]) * H, H)]
    rr = robot_spheres(plant, sc.robot_spacing)["radius"]
    hum = human_inputs(script_of(sc, {k: v[idx] for k, v in draws.items()}), sc)  # (V, T, S_h, 3) for V candidates
    gap = np.full(len(idx), np.inf)
    for k, (a, b) in enumerate(spans):  # the three configurations
        sel = (t >= a) & (t <= b) & (np.arange(len(t)) % FILTER_STRIDE == 0)
        X, Y = C[idx, k], hum["human_centres"][:, sel]  # (V, S_r, 3), (V, T', S_h, 3)
        d2 = np.sum(X * X, -1)[:, None, :, None] + np.sum(Y * Y, -1)[:, :, None] - 2 * np.einsum("vic,vtjc->vtij", X, Y)
        d = np.sqrt(np.maximum(d2, 0.0))  # (V, T', S_r, S_h)
        gap = np.minimum(gap, np.min(d - rr[:, None] - hum["human_radii"] - sc.d_min, (1, 2, 3)))
    sel = gap >= SEP_MARGIN
    keep = idx[sel]
    return {"q0": q0[keep], "q_pick": Qg[keep, 0], "q_handover": Qg[keep, 1], "pick": p[keep, 1], "handover": p[keep, 2],
            "start": p[keep, 0], "script": np.stack([draws[k][keep] for k in SCRIPT_RANGES], -1),
            "human_ends": hum["human_ends"][sel], "human_centres": hum["human_centres"][sel],
            "script_keys": tuple(SCRIPT_RANGES), "human_capsule_radii": hum["human_capsule_radii"],
            "human_radii": hum["human_radii"], "human_owner": hum["human_owner"],
            "candidate": keep, "accepted": len(keep), "candidates": n}


def tuning_set(sc=Scenario()):
    return instances(TUNING_SEED, N_TUNING, sc)


def heldout_set(sc=Scenario()):
    return instances(HELDOUT_SEED, N_HELDOUT, sc)


# ------------------------------------------------------------------
# the designed regime: one close pass of the hand and a wait at a standoff

@cache
def _regime_candidates(seed, sc, plant):
    """The candidate stream of _instances (the same draws in the same order, so candidate c is
    the same start, pick and handover), with the IK solutions and sphere centres."""
    m = plant.model
    rng = np.random.default_rng([seed, 0])
    lo, hi = m.jnt_range[:, 0] + JOINT_MARGIN, m.jnt_range[:, 1] - JOINT_MARGIN
    n = N_CANDIDATES
    q0 = np.clip(m.key_qpos[0] + rng.uniform(-Q_SPREAD, Q_SPREAD, (n, m.nq)), lo, hi)
    (x0, x1), (y0, y1), (z0, z1) = PICK_BOX
    pick_t = np.stack([rng.uniform(x0, x1, n), rng.choice([-1.0, 1.0], n) * rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)], -1)
    r = (sc.zone_radius - ZONE_MARGIN) * rng.random(n) ** (1 / 3)
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    hand_t = np.asarray(sc.zone) + r[:, None] * v / np.linalg.norm(v, axis=-1, keepdims=True)
    draws = {k: rng.uniform(a, b, n) for k, (a, b) in SCRIPT_RANGES.items()}
    with jax.enable_x64(True):
        mx = mjx.put_model(m, impl="jax")
        home = jnp.broadcast_to(jnp.asarray(m.key_qpos[0]), (2 * n, m.nq))
        Qg, miss = jax.jit(partial(ik, mx, plant, sc.robot_spacing, lo=jnp.asarray(lo), hi=jnp.asarray(hi)))(
            jnp.asarray(np.concatenate([pick_t, hand_t])), home)
        Qg, miss = np.asarray(Qg).reshape(2, n, m.nq).transpose(1, 0, 2), np.asarray(miss).reshape(2, n).T
        Q = np.concatenate([q0[:, None], Qg], 1)
        p, C = jax.jit(jax.vmap(jax.vmap(partial(points, mx, plant, sc.robot_spacing))))(jnp.asarray(Q))
    return {"q0": q0, "Qg": Qg, "miss": miss, "hand_t": hand_t, "draws": draws, "p": np.asarray(p), "C": np.asarray(C)}


def _min_margins(C, hc, R, S, ok, batch=16):
    """Per candidate, the smallest (d - R) / S over robot spheres C (V, K, S_r, 3) at K
    configurations and human spheres hc (V, T, S_h, 3), d the centre distance, over the entries
    where ok (V, K, T, S_h) holds; R and S (S_r, S_h). float64, lax.map over candidates."""
    with jax.enable_x64(True):
        R, S = jnp.asarray(R), jnp.asarray(S)

        def one(args):
            c, h, o = args  # (K, S_r, 3), (T, S_h, 3), (K, T, S_h)
            d = jnp.linalg.norm(c[:, None, :, None] - h[None, :, None], axis=-1)  # (K, T, S_r, S_h)
            return jnp.min(jnp.where(o[:, :, None, :], (d - R) / S, jnp.inf))

        return np.asarray(jax.lax.map(one, (jnp.asarray(C), jnp.asarray(hc), jnp.asarray(ok)), batch_size=batch))


def regime_instances(seed, n, sc=Scenario(), ranges=None, plant=Plant()):
    """The first n valid instances of the candidate stream of _instances, with the human of
    regime_script_of (the pass and the wait drawn in ranges, REGIME_RANGES by default, from the
    stream [seed, 2]). A candidate is valid when:
    - it passes the checks of instances() (IK, zone margins, and every robot sphere clearing
      every human sphere by d_min + SEP_MARGIN at the three configurations over their windows,
      every FILTER_STRIDE-th sample), with this human;
    - the pass and standoff points lie within 0.98 of the human's arm length from the standing
      shoulder, so the stated distances hold;
    - at the three configurations over their windows, every FILTER_STRIDE-th sample, every
      separation margin (score units) is at least standoff_margin + REGIME_CLEAR, except those
      of the right forearm's spheres during the designed episode (regime_episode). So on the
      reference path the pass is the separation minimum, the wait the next level, and every
      other entry lies REGIME_CLEAR above the wait.
    Returns the arrays of instances() plus regime (dict of (n,) draws), anchor (n,) the anchor
    sphere index and R (n,)."""
    ranges = REGIME_RANGES if ranges is None else ranges
    out = _regime_instances(seed, sc, plant, tuple(sorted((k, tuple(v)) for k, v in ranges.items())))
    if out["accepted"] < n:
        raise ValueError("only " + str(out["accepted"]) + " of " + str(N_CANDIDATES) + " candidates are valid")
    per = PER_INSTANCE + ("anchor", "R", "margin_min", "margin_other")
    res = {k: (v[:n].copy() if k in per else v) for k, v in out.items() if k != "regime"}
    res["regime"] = {k: v[:n].copy() for k, v in out["regime"].items()}
    return res


@cache
def _regime_instances(seed, sc, plant, ranges):
    cand = _regime_candidates(seed, sc, plant)
    rng = np.random.default_rng([seed, 2])
    regime = {k: rng.uniform(a, b, N_CANDIDATES) for k, (a, b) in ranges}
    radial = np.linalg.norm(cand["p"] - np.asarray(sc.zone), axis=-1)
    ok = np.all(cand["miss"] <= IK_TOLERANCE, -1) & (cand["hand_t"][:, 2] >= HANDOVER_MIN_Z)
    ok &= (radial[:, 0] >= sc.zone_radius + ZONE_MARGIN) & (radial[:, 1] >= sc.zone_radius + ZONE_MARGIN)
    idx = np.nonzero(ok)[0]
    rs = robot_spheres(plant, sc.robot_spacing)
    reg = {k: v[idx] for k, v in regime.items()}
    script, a, R = regime_script_of(sc, {k: v[idx] for k, v in cand["draws"].items()}, reg, cand["C"][idx, 2], rs["radius"], cand["p"][idx, 2])
    hum = human_inputs(script, sc)
    H, hs = float(sc.H), float(sc.h_s)
    t = np.arange(sc.samples) * hs
    hand_from = dict(ranges)["hand_from"][0] if "hand_from" in dict(ranges) else float(sc.handover[0]) * H  # E029 key
    spans = [(0.0, float(sc.pick[0]) * H), (float(sc.pick[0]) * H, float(sc.pick[1]) * H), (hand_from, H)]
    inside = np.stack([(t >= lo_) & (t <= hi_) & (np.arange(len(t)) % FILTER_STRIDE == 0) for lo_, hi_ in spans])  # (3, T)
    hr = hum["human_radii"]
    Rp = rs["radius"][:, None] + hr[None] + sc.d_min  # (S_r, S_h)
    ok_all = np.broadcast_to(inside[None, :, :, None], (len(idx),) + inside.shape + (len(hr),))
    arm = hum["human_owner"] == Hm.NAMES.index("forearm_r")
    ok_other = ok_all & ~(regime_episode(script, t)[:, None, :, None] & arm)
    C = cand["C"][idx]
    gap = _min_margins(C, hum["human_centres"], Rp, np.ones_like(Rp), ok_all)  # metres beyond d_min and the radii
    low = _min_margins(C, hum["human_centres"], Rp, Rp, ok_all)  # score units, every pair
    other = _min_margins(C, hum["human_centres"], Rp, Rp, ok_other)  # score units, outside the episode
    _, p_pass, p_stand = Hm.pass_points(script)
    shoulder = Hm.standing_shoulder(script)
    reach = 0.98 * (Hm.UPPER + Hm.FORE)
    within = (np.linalg.norm(p_pass - shoulder, axis=-1) <= reach) & (np.linalg.norm(p_stand - shoulder, axis=-1) <= reach)
    need = reg["pass_margin"] + reg["clear_above_pass"] if "clear_above_pass" in reg else reg["standoff_margin"] + REGIME_CLEAR
    if "pre_margin" in reg:  # the trapezoid's pre point must lie within reach as well
        n, _, _ = Hm.pass_points(script)
        within &= np.linalg.norm(np.asarray(script.anchor) + np.asarray(script.pre_gap)[..., None] * n - shoulder, axis=-1) <= reach
    sel = (gap >= SEP_MARGIN) & within & (other >= need)
    keep = idx[sel]
    return {"q0": cand["q0"][keep], "q_pick": cand["Qg"][keep, 0], "q_handover": cand["Qg"][keep, 1],
            "pick": cand["p"][keep, 1], "handover": cand["p"][keep, 2], "start": cand["p"][keep, 0],
            "script": np.stack([cand["draws"][k][keep] for k in SCRIPT_RANGES], -1), "script_keys": tuple(SCRIPT_RANGES),
            "human_ends": hum["human_ends"][sel], "human_centres": hum["human_centres"][sel],
            "human_capsule_radii": hum["human_capsule_radii"], "human_radii": hr, "human_owner": hum["human_owner"],
            "regime": {k: v[sel] for k, v in reg.items()}, "anchor": a[sel], "R": R[sel], "margin_min": low[sel],
            "margin_other": other[sel], "candidate": keep, "accepted": len(keep), "candidates": N_CANDIDATES,
            "rejected": {"separation": int(np.sum(gap < SEP_MARGIN)), "reach": int(np.sum(~within)),
                         "regime": int(np.sum(other < need)), "ik_or_zone": int(N_CANDIDATES - len(idx))}}

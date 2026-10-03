"""E022: the designed regime in the shared-workspace scenario, on the witness and on the
optimization path (M003, the core paper's open question, requirements 1, 2 and 4).

    python -m examples.e022_regime witness <out.npz> <n_inst> <configs>
    python -m examples.e022_regime run <witness.npz> <H> <D> <h_s> <iterations> <out.npz> [<lr> <both|witness>]
    python -m examples.e022_regime report <out_dir> <witness.npz> <run.npz>...

<configs> is a comma-separated list of H:D:h_s (seconds; h_s as a fraction such as 1/50).

Scenario. tasks.workspace with the task windows fixed in seconds (pick [2, 4], handover
[4.8, 6.4], dwell 0.8, passed as fractions of H) and the human of
workspace.regime_script_of: the pass (closest approach at t_pass = 6 s, margin
pass_margin of the anchor pair at the handover configuration) and the wait (the hand held
at the standoff, margin standoff_margin, for D seconds from 6.9 s). The ranges are
workspace.REGIME_RANGES with the wait set to D. Instances are the tuning seed's candidates
that are valid in every listed configuration (the first n_inst of them).

witness (float64, run it with JAX_ENABLE_X64=true): E019's kinematic witness
(examples.e019_witness: piecewise-linear joint keyframes every KEY s, the loss on the
separation, speed-where-near, zone-before-pick hinges and the joint speed limits, Adam),
here with the hinges' slack SLACK = 0.2 score units instead of E019's 0.02, so that on a
witness every conjunct other than the separation's pass lies above the pass margin 0.1,
solved on [0, WIT_H] with the pick configuration at T_PICK and the handover configuration
from T_HAND on; the keyframes after T_HAND are fixed, so the witness extends to any horizon
by holding the handover configuration. For every configuration it reports the referee (the
exact robustness of the core conjunction on workspace.margins, every STL sample) and each
conjunct, and the node table of the witness trajectory's scores.

run (float32): single-shooting Adam as in the core study (D006 item 13): the normalized
controls V = U / u_max, the objective rho_smooth - LAM E(V) with E the time-averaged squared
normalized torque, Adam (optax) at learning rate LR, every update projected onto |V| <= 1,
for each optimizing method in METHODS at the per-node error EPS_RUN (the matched wrappers of
core_study.methods; budget B = 5 EPS_RUN on the path depth 5). The rollout is
workspace.physics_rollout over blocks of BLOCK intervals, each block rematerialized in
reverse mode (values equal those of one call). The STL program is E005's compiled core
formula with the rows that the root at t = 0 does not read removed (prune; values and
gradients equal, checked here). Runs per method: for each instance, one from the study's
small random start (N(0, 0.01^2) normalized torque on 1/25 s cells) and one from the
tracked witness (TRACK: a computed-torque PD law tracking the witness in MJX, recorded as an
open-loop command). Every iteration records J, rho_smooth, the referee and E. At iterations
0, REC, 2 REC, ... it records, at every eps in EPS and for both matched semantics:
- the node table (host, float64): along the path from the root to a leaf that follows the
  exact extremum entry of the scores (the deciding conjunct first), and along the separation
  conjunct's path when another conjunct decides, per node its arity m, the
  sparsemax support size k at gamma = 2 eps / (1 - 1/m), the count of entries within eps of
  the extremum, the gap quantiles and gap counts of (|z - z*|) / eps, the lse mass
  W_L = sum of softmax(+-beta z), beta = log m / eps, over the sparsemax support (the
  sparsemax mass there, W_Q, is 1 by definition), and for nodes over time (m = T) both
  masses on the pass window and on the wait window;
- the composed leaf masses d rho / d z of each method on the leaf groups (separation atoms
  in the pass window, in the wait window, elsewhere; slow-down and speed atoms; zone; pick;
  handover), and the lse leaf mass on the sparsemax leaf support;
- after the pullback: both control gradients' norms, their cosine over all controls and
  over the controls before the pass window, and the fraction of controls where their signs
  agree.
The node inputs are the exact values of the scores (a common input vector, as in the draft's
allocation result); the composed masses and control gradients are each method's own.
"""
import csv
import json
import sys
import time
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mujoco import mjx

from sparsemax_dstl import stl
from sparsemax_dstl.core_study import methods
from sparsemax_dstl.operators import sparsemax_weights
from sparsemax_dstl.stl import Program, Step
from sparsemax_dstl.tasks import human as Hm
from sparsemax_dstl.tasks import panda as P
from sparsemax_dstl.tasks import workspace as W

SEED = W.TUNING_SEED


def dumps(obj, **kw):
    """json.dumps with NumPy scalars and arrays as Python numbers and lists."""
    return json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), **kw)
EPS = (0.05, 0.1, 0.2, 0.4)
EPS_RUN = 0.2
METHODS = ("sparsemax", "lse")
LAM = 0.1
LR = 0.01
REC = 10
BLOCK = 50
H_CTRL = Fraction(1, 50)
T_PICK, T_HAND, KEY, WIT_H = 3.0, 5.0, 0.2, 5.4
WIT_STEPS, WIT_LR, SLACK = 3000, 0.02, 0.2  # E019: 600 to 2500 steps, slack 0.02
N_USE = 2  # instances used by the runs: the first N_USE witnessed ones
TRACK = 20.0  # rad/s, the PD law's natural frequency
CONJUNCTS = ("separation", "slowdown", "order", "handover")  # the core And's children, in workspace.specs order
GROUPS = ("sep_pass", "sep_wait", "sep_other", "slowdown", "zone", "pick", "handover")
QUANTILES = (0.0, 0.01, 0.1, 0.5)
# Franka FER joint velocity limits, rad/s (frankarobotics.github.io/docs/robot_specifications.html,
# "Limits for Franka Emika Robot (FER)"), the values of extension_study.nlp.QDOT_MAX, defined here
# so that this core-paper script imports nothing of the extension paper (E024 lead, root's
# authorization of 2026-09-30 07:25Z).
QDOT_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
BINS = (0.5, 1.0, 2.0, 4.0)  # gap counts below each value of |z - z*| / eps
plant = W.Plant()
N_R = len(W.robot_spheres(plant, W.Scenario().robot_spacing)["body"])


def scenario(H, h_s=H_CTRL):
    H = Fraction(H)
    return W.Scenario(H=H, h_s=Fraction(h_s), pick=(2 / H, 4 / H), handover=(Fraction(24, 5) / H, Fraction(32, 5) / H),
                      dwell=Fraction(4, 5) / H)


def ranges(D):
    rg = dict(W.REGIME_RANGES)
    rg["wait"] = (float(D), float(D))
    return rg


def parse(configs):
    out = []
    for c in configs.split(","):  # over configurations
        H, D, hs = c.split(":")
        out.append((int(H), float(D), Fraction(hs)))
    return out


def instance_set(sc, rg):
    n = W.regime_instances(SEED, 1, sc, rg)["accepted"]
    return W.regime_instances(SEED, n, sc, rg)


def pick(I, cands):
    k = [int(np.nonzero(I["candidate"] == c)[0][0]) for c in cands]  # over chosen instances
    out = {key: I[key][k] for key in ("q0", "q_pick", "q_handover", "pick", "handover", "human_centres", "anchor", "R",
                                      "margin_min", "margin_other", "candidate")}
    out["regime"] = {key: v[k] for key, v in I["regime"].items()}
    out["human_radii"] = I["human_radii"]
    return out


def windows(sc, D):
    """Boolean (T,) masks of the pass window and the wait hold, from REGIME_RANGES."""
    t = np.arange(sc.samples) * float(sc.h_s)
    rg = W.REGIME_RANGES
    tp, w, tm = rg["t_pass"][0], rg["pass_width"][0], Hm.Script().t_move
    hold = tp + w / 2 + tm
    return (t >= tp - w / 2) & (t <= tp + w / 2), (t >= hold) & (t <= hold + D)


# ------------------------------------------------------------------
# the program: E005's compiled core formula, rows the root does not read removed

def reachable(program):
    """Boolean per step: the rows the root entry at t = 0 reads through valid window entries."""
    steps = program.steps
    reach = [np.zeros(s.length, bool) for s in steps]
    reach[-1][0] = True
    for s in reversed(range(len(steps))):  # over formula steps, consumers before sources
        step = steps[s]
        if step.kind == "atom" or not reach[s].any():
            continue
        sizes = [steps[j].length for j in step.sources]
        valid = np.arange(step.index.shape[1])[None, :] < step.count[:, None]
        hit = np.zeros(sum(sizes), bool)
        hit[step.index[reach[s][:, None] & valid]] = True
        offsets = np.cumsum([0] + sizes)
        for j, a, b in zip(step.sources, offsets[:-1], offsets[1:]):  # over a step's children
            reach[j] |= hit[a:b]
    return reach


def prune(program):
    """The same program with every reduction step restricted to the rows the root at t = 0
    reads (atom steps kept whole); the root entry's value and its derivatives are unchanged."""
    reach = reachable(program)
    new, pos = [], []
    for s, step in enumerate(program.steps):  # over formula steps, sources before consumers
        if step.kind == "atom":
            new.append(step)
            pos.append(np.arange(step.length))
            continue
        keep = np.nonzero(reach[s])[0]
        offs = np.cumsum([0] + [new[j].length for j in step.sources])
        m = np.concatenate([np.where(pos[j] >= 0, pos[j] + o, -1) for j, o in zip(step.sources, offs[:-1])])
        cnt = step.count[keep]
        valid = np.arange(step.index.shape[1])[None, :] < cnt[:, None]
        idx = m[step.index[keep]]
        if len(keep) == 0 or np.any(idx[valid] < 0):
            raise ValueError("pruning lost a row that a kept row reads")
        new.append(Step(step.kind, len(keep), step.label, step.sources, np.where(valid, idx, 0).astype(np.int32),
                        cnt.astype(np.int32)))
        p = -np.ones(step.length, int)
        p[keep] = np.arange(len(keep))
        pos.append(p)
    return Program(program.formula, tuple(new), program.T, program.boundary, program.n_predicates)


def core_program(sc, n_h, pruned=True):
    prog = stl.compile_formula(W.specs(sc, N_R, n_h)[2], sc.samples)
    return prune(prog) if pruned else prog


def leaf_groups(sc, D, n_h):
    """Boolean (len(GROUPS), T, P) masks of the leaf groups."""
    T, P_ = sc.samples, 3 + N_R + 2 * N_R * n_h
    sep0, slow0 = 3 + N_R, 3 + N_R + N_R * n_h
    col = np.arange(P_)
    sep = (col >= sep0) & (col < slow0)
    pw, ww = windows(sc, D)
    g = [pw[:, None] & sep, ww[:, None] & sep, ~(pw | ww)[:, None] & sep,
         np.broadcast_to((col >= slow0) | ((col >= 3) & (col < sep0)), (T, P_)),
         np.broadcast_to(col == 2, (T, P_)), np.broadcast_to(col == 0, (T, P_)), np.broadcast_to(col == 1, (T, P_))]
    return np.stack(g)


# ------------------------------------------------------------------
# node tables on the host, float64

def path_nodes(prog, vals, first=None):
    """(step, row, inputs, kind) of every extremum node on the path from the root (row 0) to a
    leaf, following the exact extremum entry (at the root, entry `first` if given); and the
    leaf (atom index, time)."""
    out = []
    s, r = prog.root, 0
    while prog.steps[s].kind != "atom":  # over the path's nodes (formula structure)
        step = prog.steps[s]
        src = np.concatenate([np.asarray(vals[j], np.float64) for j in step.sources])
        idx = step.index[r, :step.count[r]]
        x = src[idx]
        k = int(np.argmin(x) if step.kind == "min" else np.argmax(x))
        if first is not None and s == prog.root:
            k = first
        out.append((s, r, x, step.kind))
        sizes = np.cumsum([0] + [prog.steps[j].length for j in step.sources])
        j = int(np.searchsorted(sizes, idx[k], side="right") - 1)
        s, r = step.sources[j], int(idx[k] - sizes[j])
    return out, (prog.steps[s].atom, r)


def node_row(x, kind, eps, T, pw, ww):
    """Allocation statistics of one node at per-node error eps (see the module docstring)."""
    z = -x if kind == "min" else x
    m = len(z)
    gap = (z.max() - z) / eps
    row = {"m": m, "n_within_eps": int(np.sum(gap <= 1.0))}
    row.update({"gap_q" + str(q): float(np.quantile(gap, q)) for q in QUANTILES})
    row.update({"gap_lt" + str(b): int(np.sum(gap < b)) for b in BINS})
    if m == 1:
        row.update(k=1, W_L=1.0, WQ_pass=np.nan, WL_pass=np.nan, WQ_wait=np.nan, WL_wait=np.nan)
        return row
    gamma, beta = 2 * eps / (1 - 1 / m), np.log(m) / eps
    with jax.enable_x64(True):
        p = np.asarray(sparsemax_weights(jnp.asarray(z, jnp.float64), gamma))
    q = np.exp(beta * (z - z.max()))
    q /= q.sum()
    S = p > 0
    row.update(k=int(S.sum()), W_L=float(q[S].sum()))
    if m == T:
        row.update(WQ_pass=float(p[pw].sum()), WL_pass=float(q[pw].sum()), WQ_wait=float(p[ww].sum()), WL_wait=float(q[ww].sum()))
    else:
        row.update(WQ_pass=np.nan, WL_pass=np.nan, WQ_wait=np.nan, WL_wait=np.nan)
    return row


def node_table(prog, exact_fn, Z, sc, D):
    """Rows (list of dicts) for every node on the exact path, at every eps, and the conjunct values."""
    vals = exact_fn(Z)
    nodes, leaf = path_nodes(prog, vals)
    paths = [("deciding", nodes)]
    if int(np.argmin(nodes[0][2])) != 0:
        paths.append(("separation", path_nodes(prog, vals, first=0)[0]))  # the separation conjunct's path as well
    pw, ww = windows(sc, D)
    rows = []
    for name, path in paths:  # the deciding path, and the separation path when it differs
        for depth, (s, r, x, kind) in enumerate(path):  # over the path's nodes (formula structure)
            for eps in EPS:  # over the per-node errors
                row = {"path": name, "depth": depth, "label": prog.steps[s].label, "row": r, "kind": kind, "eps": eps}
                row.update(node_row(x, kind, eps, sc.samples, pw, ww))
                rows.append(row)
    conj = nodes[0][2]
    return rows, conj, leaf


def node_table_masks(prog, vals, T, pw, ww, eps_list=EPS):
    """node_table with the step values vals (evaluate's list, host arrays) given directly and the
    pass and wait sample masks pw, ww (T,) given by the caller (E024: the trapezoid pass's hold
    samples and the standoff's hold samples); rows at every eps of eps_list."""
    nodes, leaf = path_nodes(prog, vals)
    paths = [("deciding", nodes)]
    if int(np.argmin(nodes[0][2])) != 0:
        paths.append(("separation", path_nodes(prog, vals, first=0)[0]))
    rows = []
    for name, path in paths:  # the deciding path, and the separation path when it differs
        for depth, (s, r, x, kind) in enumerate(path):  # over the path's nodes (formula structure)
            for eps in eps_list:  # over the per-node errors
                row = {"path": name, "depth": depth, "label": prog.steps[s].label, "row": r, "kind": kind, "eps": eps}
                row.update(node_row(x, kind, eps, T, pw, ww))
                rows.append(row)
    return rows, nodes[0][2], leaf


def conjuncts(prog, exact_fn, Z):
    """The core And's inputs at t = 0 (separation, slow-down, order, handover), exact semantics."""
    return path_nodes(prog, exact_fn(Z))[0][0][2]


# ------------------------------------------------------------------
# the kinematic witness (E019's construction)

def kin_setup(sc):
    K = int(round(WIT_H / KEY))
    per = int(round(KEY / float(sc.h_s)))
    kp, kh = int(round(T_PICK / KEY)), int(round(T_HAND / KEY))
    free = np.array([k not in (0, kp) and k < kh for k in range(K + 1)])
    return K, per, kp, kh, free


def keyframes(Wf, inst, K, kp, kh, free):
    Q = jnp.where(jnp.arange(K + 1)[:, None] < kh, inst["q_pick"], inst["q_handover"])
    Q = Q.at[0].set(inst["q0"]).at[kp].set(inst["q_pick"])
    return jnp.where(jnp.asarray(free)[:, None], Wf, Q)


def kin_states(Q, per, T):
    """Sampled states (T, 14): q linear between keyframes, qdot constant on each segment, then
    the last keyframe held at rest up to T samples."""
    frac = jnp.arange(per) / per
    q = (Q[:-1, None] + frac[None, :, None] * (Q[1:] - Q[:-1])[:, None]).reshape(-1, Q.shape[-1])
    qd = jnp.repeat((Q[1:] - Q[:-1]) / KEY, per, axis=0)
    n = T - q.shape[0]
    q = jnp.concatenate([q, jnp.broadcast_to(Q[-1], (n, Q.shape[-1]))])
    qd = jnp.concatenate([qd, jnp.zeros((n, Q.shape[-1]), Q.dtype)])
    return jnp.concatenate([q, qd], -1)


def solve_witness(I, mx):
    """Keyframes (n, K + 1, 7) for the instances I on [0, WIT_H] at h_s = 0.02 s."""
    sc = scenario(16)
    K, per, kp, kh, free = kin_setup(sc)
    T = K * per + 1
    n_h = len(I["human_radii"])
    lo, hi = jnp.asarray(plant.model.jnt_range[:, 0]), jnp.asarray(plant.model.jnt_range[:, 1])
    hs = float(sc.h_s)

    def loss(Wf, inst):
        Q = keyframes(Wf, inst, K, kp, kh, free)
        Z = W.scores(mx, plant, sc, inst, kin_states(Q, per, T))
        speed = Z[:, 3:3 + N_R]
        sep = Z[:, 3 + N_R:3 + N_R + N_R * n_h]
        slow = Z[:, 3 + N_R + N_R * n_h:].reshape(-1, N_R, n_h).min(-1)
        pre = jnp.arange(T) * hs <= T_PICK
        hinge = lambda v: jnp.sum(jax.nn.relu(SLACK - v) ** 2)
        vel = jnp.abs(jnp.diff(Q, axis=0)) / KEY - jnp.asarray(QDOT_MAX)
        near = jax.lax.stop_gradient(slow) < SLACK
        return hinge(sep) + hinge(jnp.where(near, speed, 1.0)) + hinge(jnp.where(pre, Z[:, 2], 1.0)) + jnp.sum(jax.nn.relu(vel) ** 2)

    def solve(Wf, inst):
        opt = optax.adam(WIT_LR)

        def step(carry, _):
            Wf, st = carry
            u, st = opt.update(jax.grad(loss)(Wf, inst), st)
            return (jnp.clip(Wf + u, lo, hi), st), None

        return jax.lax.scan(step, (Wf, opt.init(Wf)), None, length=WIT_STEPS)[0][0]

    inst = {k: jnp.asarray(I[k][:, :T] if k == "human_centres" else I[k]) for k in ("q0", "q_pick", "q_handover", "pick", "handover", "human_centres")}
    tk = jnp.arange(K + 1) * KEY
    line = jnp.clip(tk / T_PICK, 0, 1)[None, :, None]
    between = jnp.clip((tk - T_PICK) / (T_HAND - T_PICK), 0, 1)[None, :, None]
    W0 = jnp.where((tk <= T_PICK)[None, :, None], inst["q0"][:, None] + line * (inst["q_pick"] - inst["q0"])[:, None],
                   inst["q_pick"][:, None] + between * (inst["q_handover"] - inst["q_pick"])[:, None])
    hr = jnp.asarray(I["human_radii"])
    Wf = jax.jit(jax.vmap(lambda Wf, inst: solve(Wf, {**inst, "human_radii": hr})))(W0, inst)
    return np.asarray(jax.vmap(lambda Wf, inst: keyframes(Wf, inst, K, kp, kh, free))(Wf, inst))


def witness_states(Q, sc):
    K, per, _, _, _ = kin_setup(sc)
    return kin_states(jnp.asarray(Q), per, sc.samples)


def witness_main(out, n_inst, configs):
    if configs[0][2] != H_CTRL:
        raise ValueError("the first configuration must sample at 1/50 s (the witness is solved there)")
    mx = mjx.put_model(plant.model, impl="jax")
    sets = {c: instance_set(scenario(c[0], c[2]), ranges(c[1])) for c in configs}
    common = sorted(set.intersection(*[set(I["candidate"].tolist()) for I in sets.values()]))
    cands = common[:n_inst]
    t0 = time.perf_counter()
    Q = solve_witness(pick(sets[configs[0]], cands), mx)
    np.savez(out, Q=Q, candidates=np.array(cands))  # replaced below by the full record
    res = {"candidates": np.array(cands), "keyframes": Q, "solve_seconds": time.perf_counter() - t0,
           "accepted": {str(c): int(I["accepted"]) for c, I in sets.items()}, "rejected": {str(c): I["rejected"] for c, I in sets.items()},
           "common": len(common), "configs": [[c[0], c[1], str(c[2])] for c in configs]}
    tables = {}
    for c in configs:  # over configurations
        sc = scenario(c[0], c[2])
        I = pick(sets[c], cands)
        n_h = len(I["human_radii"])
        prog = core_program(sc, n_h)
        rows_ = core_program(sc, n_h, pruned=False)
        exact = jax.jit(lambda Z: stl.evaluate(prog, Z, "exact"))
        ref_p = jax.jit(lambda M: stl.robustness(prog, M, "exact")[0])
        ref_u = jax.jit(lambda M: stl.robustness(rows_, M, "exact")[0])
        referee, per_inst = [], []
        for i in range(len(cands)):  # over the chosen instances (a handful)
            inst = {"pick": I["pick"][i], "handover": I["handover"][i], "human_centres": I["human_centres"][i], "human_radii": I["human_radii"]}
            X = witness_states(Q[i], sc)
            M = W.margins(mx, plant, sc, inst, X)
            Z = W.scores(mx, plant, sc, inst, X)
            rho, full = float(ref_p(M)), float(ref_u(M))
            table, conj, leaf = node_table(prog, exact, Z, sc, c[1])
            conj_ref = np.asarray(conjuncts(prog, exact, M))
            referee.append({"candidate": int(cands[i]), "referee": rho, "referee_unpruned": full,
                            "conjuncts": dict(zip(CONJUNCTS, np.round(conj_ref, 5).tolist())), "leaf": [int(leaf[0]), int(leaf[1])],
                            "anchor_margin_designed": [float(I["regime"]["pass_margin"][i]), float(I["regime"]["standoff_margin"][i])],
                            "margin_min_filter": float(I["margin_min"][i]), "margin_other_filter": float(I["margin_other"][i])})
            per_inst.append(table)
        res[str(c)] = referee
        tables[str(c)] = per_inst
    ok = [i for i in range(len(cands)) if all(res[str(c)][i]["referee"] >= 0 for c in configs)]  # over instances
    use = [cands[i] for i in ok[:N_USE]]
    res["witnessed"] = [cands[i] for i in ok]
    res["use"] = use
    np.savez(out, Q=Q[ok[:N_USE]], candidates=np.array(use), Q_all=Q, candidates_all=np.array(cands), meta=dumps(res), tables=dumps(tables))
    print(dumps(res, indent=1))


# ------------------------------------------------------------------
# optimization runs, float32

def rollout(mx, n_sub, x0, U):
    """workspace.physics_rollout over blocks of BLOCK intervals, each block rematerialized."""
    blk = U.reshape(U.shape[0] // BLOCK, BLOCK, U.shape[-1])

    def body(x, Ub):
        Y = W.physics_rollout(mx, n_sub, x, Ub)
        return Y[-1], Y[1:]

    _, Y = jax.lax.scan(jax.checkpoint(body), x0, blk)
    return jnp.concatenate([x0[None], Y.reshape(-1, x0.shape[-1])])


def track(mx, n_sub, x0, Qs, u_max):
    """Open-loop commands (N, 7) of a computed-torque PD law tracking the witness samples at the
    control boundaries Qs (N + 1, 14): u = M(q) (w^2 (q_ref' - q) + 2 w (qd_ref - qd)), with q_ref'
    the reference at the end of the interval, clipped to the command box."""
    nq = x0.shape[-1] // 2

    def interval(x, ref):
        d = mjx.forward(mx, mjx.make_data(mx).replace(qpos=x[:nq], qvel=x[nq:]))
        M = mjx.full_m(mx, d)
        u = M @ (TRACK ** 2 * (ref[:nq] - x[:nq]) + 2 * TRACK * (ref[nq:] - x[nq:]))
        u = jnp.clip(u, -u_max, u_max)
        return W.physics_rollout(mx, n_sub, x, u[None])[-1], u

    return jax.lax.scan(interval, x0, jnp.concatenate([Qs[1:, :nq], Qs[:-1, nq:]], -1))[1]


def random_start(seed, N, cell=Fraction(1, 25)):
    """Normalized commands (N, 7): N(0, 0.01^2) per joint on cells of `cell` seconds, held at H_CTRL."""
    per = int(cell / H_CTRL)
    rng = np.random.default_rng([seed, 1])
    Wn = np.clip(rng.standard_normal((-(-N // per), 7)) * P.INIT_SIGMA, -1, 1)
    return np.repeat(Wn, per, axis=0)[:N]


def run_main(wit, H, D, hs, iterations, out, lr=LR, starts="both"):
    wz = np.load(wit)
    cands, Qk = wz["candidates"].tolist(), wz["Q"]
    sc = scenario(H, hs)
    rg = ranges(D)
    I = pick(instance_set(sc, rg), cands)
    n_h = len(I["human_radii"])
    mx = mjx.put_model(plant.model, impl="jax")
    dt = jnp.float32
    n_sub, stride = P.substeps(0.02), W.sample_stride(sc)
    N = P.intervals(sc.H, H_CTRL)
    u_max = jnp.asarray(P.torque_limit(), dt)
    prog = core_program(sc, n_h)
    groups = jnp.asarray(leaf_groups(sc, D, n_h), dt)
    pw, _ = windows(sc, D)
    t_ctrl = np.arange(N) * float(H_CTRL)
    before = jnp.asarray(t_ctrl < float(W.REGIME_RANGES["t_pass"][0]) - float(W.REGIME_RANGES["pass_width"][0]) / 2)
    depth = int(methods.path_depth(prog)[0])
    n = len(cands)
    insts = {"pick": jnp.asarray(I["pick"], dt), "handover": jnp.asarray(I["handover"], dt),
             "human_centres": jnp.asarray(I["human_centres"], dt), "x0": jnp.asarray(np.concatenate([I["q0"], np.zeros_like(I["q0"])], -1), dt)}
    hr = jnp.asarray(I["human_radii"], dt)

    def inst_of(a):
        return {"pick": a["pick"], "handover": a["handover"], "human_centres": a["human_centres"], "human_radii": hr}

    def ZX(V, a):
        X = rollout(mx, n_sub, a["x0"], V * u_max)[::stride]
        return jax.lax.optimization_barrier(W.scores(mx, plant, sc, inst_of(a), X)), X

    # starts: the tracked witness and the random start, per instance
    Qs = jnp.asarray(np.stack([np.asarray(witness_states(Qk[i], scenario(H, H_CTRL))) for i in range(n)]), dt)  # over instances
    t0 = time.perf_counter()
    Ut = jax.jit(jax.vmap(lambda x0, Q: track(mx, n_sub, x0, Q, u_max)))(insts["x0"], Qs)
    track_s = time.perf_counter() - t0
    V0 = jnp.concatenate([Ut / u_max, jnp.asarray(np.stack([random_start(int(c), N) for c in cands]), dt)])  # (2n, N, 7)
    runs = {k: jnp.concatenate([v, v]) for k, v in insts.items()}
    start = ["witness"] * n + ["random"] * n
    if starts == "witness":  # the tracked-witness starts only
        V0, runs, start = V0[:n], insts, start[:n]
    R = V0.shape[0]

    # check (H <= 32 s): the pruned program equals the unpruned one at the first tracked start,
    # value and leaf gradient, for both matched semantics
    check = {}
    if H <= 32:
        Z0 = jax.jit(ZX)(V0[0], {k: v[0] for k, v in runs.items()})[0]
        full = core_program(sc, n_h, pruned=False)
        for name in METHODS:  # the two matched semantics
            sem = methods.SEMANTICS[name]
            a = jax.jit(jax.value_and_grad(lambda Z: stl.robustness(prog, Z, sem, EPS_RUN)[0]))(Z0)
            b = jax.jit(jax.value_and_grad(lambda Z: stl.robustness(full, Z, sem, EPS_RUN)[0]))(Z0)
            check[name] = [float(a[0]), float(b[0]), float(jnp.max(jnp.abs(a[1] - b[1])))]
        del full

    # diagnostics in three compiled pieces (one program graph per semantics, eps batched), runs one
    # after another inside each (lax.map), for compile time and memory
    ne = len(EPS)

    def forward(V, a):
        Z, X = ZX(V, a)
        return Z, W.margins(mx, plant, sc, inst_of(a), X)

    def leafgrads(Z):
        vs, Gs = [], []
        for nm in METHODS:  # the two matched semantics: sparsemax rows first, then lse
            f = lambda Z, e: stl.robustness(prog, Z, methods.SEMANTICS[nm], e)[0]
            v, G = jax.vmap(jax.value_and_grad(f), in_axes=(None, 0))(Z, jnp.asarray(EPS, dt))
            vs.append(v)
            Gs.append(G)
        return jnp.concatenate(vs), jnp.concatenate(Gs)  # (2 ne,), (2 ne, T, P)

    def pullback(V, a, G):
        _, pull = jax.vjp(lambda V: ZX(V, a)[0], V)
        return jax.vmap(lambda G: pull(G)[0])(G).reshape(G.shape[0], -1)  # (2 ne, N 7)

    def stats(G, g):
        gq, gl = g[:ne], g[ne:]
        norm = jnp.linalg.norm(g, axis=-1)
        cos = jnp.sum(gq * gl, -1) / (norm[:ne] * norm[ne:])
        pre = jnp.repeat(before, 7)
        cos_pre = jnp.sum(gq * gl * pre, -1) / jnp.sqrt(jnp.sum(gq * gq * pre, -1) * jnp.sum(gl * gl * pre, -1))
        both = (gq != 0) & (gl != 0)
        agree = jnp.sum(both & (jnp.sign(gq) == jnp.sign(gl)), -1) / jnp.maximum(jnp.sum(both, -1), 1)
        mass = jnp.einsum("etp,gtp->eg", G, groups)
        wl_leaf = jnp.sum(jnp.where(G[:ne] > 0, G[ne:], 0.0), (-2, -1))
        return {"norm": norm, "cos": cos, "cos_pre": cos_pre, "sign_agree": agree, "mass": mass, "wl_leaf": wl_leaf}

    each = lambda f: jax.jit(lambda *args: jax.lax.map(lambda x: f(*x), args))
    fwd_f, leaf_f, pull_f, stat_f = each(forward), each(leafgrads), each(pullback), each(stats)
    t0 = time.perf_counter()
    Zs, Ms = fwd_f(V0, runs)
    vals, Gs = leaf_f(Zs)
    fwd_c = fwd_f.lower(V0, runs).compile()
    pull_c = pull_f.lower(V0, runs, Gs).compile()
    diag_compile_s = time.perf_counter() - t0

    def diag_c(V, runs):
        Z, M = fwd_c(V, runs)
        v, G = leaf_f(Z)
        return {"Z": Z, "M": M, "values": v} | stat_f(G, pull_c(V, runs, G))

    exact = jax.jit(lambda Z: stl.evaluate(prog, Z, "exact"))
    res = {"H": H, "D": D, "h_s": str(hs), "samples": sc.samples, "intervals": N, "candidates": cands, "starts": start,
           "path_depth": depth, "prune_check": check, "track_seconds": track_s, "iterations": iterations, "lr": lr,
           "diag_compile_seconds": diag_compile_s, "methods": {}}
    arrays = {}
    for name in METHODS:  # the optimizing methods
        sem = methods.SEMANTICS[name]

        def J(V, a):
            Z, X = ZX(V, a)
            rho = stl.robustness(prog, Z, sem, EPS_RUN)[0]
            E = jnp.sum(V ** 2) / V.shape[0]
            ref = stl.robustness(prog, W.margins(mx, plant, sc, inst_of(a), jax.lax.stop_gradient(X)), "exact")[0]
            return rho - LAM * E, (rho, ref, E)

        grad = jax.value_and_grad(J, has_aux=True)
        adam = optax.adam(lr)

        def segment(V, st, a):
            def it(carry, _):
                V, st = carry
                (j, (rho, ref, E)), g = grad(V, a)
                u, st = adam.update(-g, st)
                return (jnp.clip(V + u, -1.0, 1.0), st), jnp.stack([j, rho, ref, E, jnp.linalg.norm(g)])
            (V, st), rec = jax.lax.scan(it, (V, st), None, length=REC)
            return V, st, rec

        seg = jax.jit(jax.vmap(segment))
        V, st = V0, jax.vmap(adam.init)(V0)
        t0 = time.perf_counter()
        seg_c = seg.lower(V, st, runs).compile()
        compile_s = time.perf_counter() - t0
        recs, diags, iters, times, Vs, dtimes = [], [], [], [], [], []
        for k in range(0, iterations + 1, REC):  # over recording points (segments of REC iterations)
            t0 = time.perf_counter()
            d = jax.device_get(diag_c(V, runs))
            dtimes.append(time.perf_counter() - t0)
            tables = []
            for r in range(R):  # over runs (host node tables, a handful)
                rows, conj, leaf = node_table(prog, exact, d["Z"][r], sc, D)
                ref_conj = np.asarray(conjuncts(prog, exact, d["M"][r]))
                tables.append({"rows": rows, "conjuncts": conj.tolist(), "referee_conjuncts": ref_conj.tolist(), "leaf": list(leaf)})
            diags.append({key: np.asarray(v) for key, v in d.items() if key not in ("Z", "M")} | {"tables": tables})
            iters.append(k)
            Vs.append(np.asarray(V))
            if k == iterations:
                break
            t0 = time.perf_counter()
            V, st, rec = seg_c(V, st, runs)
            jax.block_until_ready(V)
            times.append(time.perf_counter() - t0)
            recs.append(np.asarray(rec))
            print(name, k + REC, "s", round(times[-1], 1), "referee", np.round(np.asarray(rec)[:, -1, 2], 3).tolist(), flush=True)
        res["methods"][name] = {"compile_seconds": compile_s, "segment_seconds": times, "diag_seconds": dtimes, "recorded": iters,
                                "diag": [{key: (v.tolist() if key != "tables" else v) for key, v in dg.items()} for dg in diags]}
        arrays[name + "_trace"] = np.concatenate(recs, 1)  # (R, iterations, 5): J, rho, referee, E, |g|
        arrays[name + "_V"] = np.stack(Vs, 1)  # (R, recordings, N, 7)
    np.savez(out, meta=dumps(res), **arrays)
    print(dumps({"prune_check": check, "compile": {m: res["methods"][m]["compile_seconds"] for m in METHODS},
                      "segment_mean": {m: float(np.mean(res["methods"][m]["segment_seconds"])) for m in METHODS}}))


# ------------------------------------------------------------------
# report: CSV tables from the witness and run files

NODE_KEYS = ("path", "depth", "label", "row", "kind", "eps", "m", "k", "n_within_eps") + tuple("gap_q" + str(q) for q in QUANTILES) + \
    tuple("gap_lt" + str(b) for b in BINS) + ("W_L", "WQ_pass", "WL_pass", "WQ_wait", "WL_wait")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def report_main(out_dir, wit, run_files):
    wz = np.load(wit)
    meta, tables = json.loads(str(wz["meta"])), json.loads(str(wz["tables"]))
    ref_rows, node_rows = [], []
    for c in meta["configs"]:  # over configurations
        key = str((c[0], c[1], Fraction(c[2])))
        for i, r in enumerate(meta[key]):  # over instances
            conj = r["conjuncts"]
            ref_rows.append([c[0], c[1], c[2], r["candidate"], r["referee"], r["referee_unpruned"]] + [conj[n] for n in CONJUNCTS] +
                            [min(conj, key=conj.get)])
            for row in tables[key][i]:  # over node rows
                node_rows.append([c[0], c[1], c[2], r["candidate"]] + [row[k] for k in NODE_KEYS])
    write_csv(out_dir + "/witness_referee.csv", ["H", "D", "h_s", "candidate", "referee", "referee_unpruned"] + list(CONJUNCTS) + ["deciding"], ref_rows)
    write_csv(out_dir + "/witness_nodes.csv", ["H", "D", "h_s", "candidate"] + list(NODE_KEYS), node_rows)
    trace, nodes, grads = [], [], []
    for f in run_files:  # over run files
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        cfg = [m["H"], m["D"], m["h_s"], m.get("lr", LR)]
        for name in METHODS:  # over optimizing methods
            tr = z[name + "_trace"]
            for r in range(tr.shape[0]):  # over runs
                who = [m["starts"][r], m["candidates"][r % len(m["candidates"])]]
                for k in range(tr.shape[1]):  # over iterations (text output)
                    trace.append(cfg + [name] + who + [k] + tr[r, k].tolist())
            for it, dg in zip(m["methods"][name]["recorded"], m["methods"][name]["diag"]):  # over recorded iterates
                for r, tb in enumerate(dg["tables"]):  # over runs
                    who = [m["starts"][r], m["candidates"][r % len(m["candidates"])]]
                    rc = dict(zip(CONJUNCTS, tb["referee_conjuncts"]))
                    decide = CONJUNCTS[int(np.argmin(tb["conjuncts"]))]
                    for row in tb["rows"]:  # over node rows
                        nodes.append(cfg + [name] + who + [it, decide, min(rc.values())] + [row[k] for k in NODE_KEYS])
                    for e, eps in enumerate(EPS):  # over per-node errors
                        ne = len(EPS)
                        grads.append(cfg + [name] + who + [it, decide, eps, dg["values"][r][e], dg["values"][r][ne + e], dg["norm"][r][e],
                                                          dg["norm"][r][ne + e], dg["cos"][r][e], dg["cos_pre"][r][e], dg["sign_agree"][r][e],
                                                          dg["wl_leaf"][r][e]] + dg["mass"][r][e] + dg["mass"][r][ne + e])
    head = ["H", "D", "h_s", "lr", "method", "start", "candidate"]
    # summary at the separation conjunct's time node (depth 1 of its path), per configuration,
    # method, start and eps, over the recorded iterates: how often the separation decides, and
    # the medians of its allocation statistics over the iterates where it decides (and over all)
    ni = len(head)
    col = {k: ni + 3 + j for j, k in enumerate(NODE_KEYS)}
    sep = [r for r in nodes if r[col["depth"]] == 1 and (r[col["path"]] == "separation" or r[ni + 1] == "separation")]
    summary = []
    keys = sorted({tuple(r[:ni]) + (r[col["eps"]],) for r in sep}, key=str)
    for key in keys:  # over (configuration, method, start, candidate, eps)
        rs = [r for r in sep if tuple(r[:ni]) + (r[col["eps"]],) == key]
        dec = [r for r in rs if r[ni + 1] == "separation"]
        med = lambda rows, k: float(np.median([r[col[k]] for r in rows])) if rows else np.nan
        crit = [r for r in dec if r[col["WQ_pass"]] >= 0.9 and r[col["WL_pass"]] <= 0.2]
        summary.append(list(key) + [len(rs), len(dec), len(crit)] + [med(dec, k) for k in ("k", "n_within_eps", "W_L", "WQ_pass", "WL_pass", "WL_wait")] +
                       [med(rs, k) for k in ("k", "W_L", "WQ_pass", "WL_pass")])
    write_csv(out_dir + "/summary_separation_node.csv", head + ["eps", "recorded", "separation_decides", "criterion_met", "k_dec", "within_eps_dec",
                                                                "WL_support_dec", "WQ_pass_dec", "WL_pass_dec", "WL_wait_dec", "k_all", "WL_support_all",
                                                                "WQ_pass_all", "WL_pass_all"], summary)
    write_csv(out_dir + "/run_trace.csv", head + ["iteration", "J", "rho_smooth", "referee", "effort", "grad_norm"], trace)
    write_csv(out_dir + "/run_nodes.csv", head + ["iteration", "deciding", "referee"] + list(NODE_KEYS), nodes)
    write_csv(out_dir + "/run_grads.csv", head + ["iteration", "deciding", "eps", "rho_Q", "rho_L", "norm_Q", "norm_L", "cos", "cos_before_pass",
                                                  "sign_agree", "WL_leaf"] + ["Q_" + g for g in GROUPS] + ["L_" + g for g in GROUPS], grads)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "witness":
        witness_main(sys.argv[2], int(sys.argv[3]), parse(sys.argv[4]))
    elif mode == "run":
        extra = (float(sys.argv[8]), sys.argv[9]) if len(sys.argv) > 8 else ()
        run_main(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), Fraction(sys.argv[5]), int(sys.argv[6]), sys.argv[7], *extra)
    elif mode == "report":
        report_main(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        raise ValueError("unknown mode " + mode)

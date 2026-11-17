"""The three problems of the sparse-differentiation examples on the dual-arm manipulator: the five-cell zone, the hold-up
and the 64-cell circle. Each constructor returns a Scene: the episode's states, the cells, the waypoints, the scheduled
visits, and the until and reach conjuncts with their windows and smoothing parameters.

Both arms move without stopping on periodic loops in joint space. Cells are balls of radius 0.03 m that only one arm may
occupy at a time. A visit of cell z by arm v (o the other arm) is a run of samples at which v's exact outside predicate is
below the margin 0.02 and that holds a sample inside the cell; it gives the until conjunct
    out_{o,z} U_[0, W_i - 1] out_{v,z}   at s_i = e_i + 1,
with e_i the visit's first sample inside and x_i its end (the first sample outside by the margin). A passage of a waypoint p
by arm a at sample s after the passage of the previous waypoint at s' gives the reach conjunct F_[0, s - s' + 10] g_{a,p}
at s'. The schedule (the visits and the passages) is read from the first 1000 samples of a motion on which every conjunct
holds in exact semantics with the margin.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..stl import Atom, Eventually, Until, horizon, to_nnf
from . import dual_arm as D

N_EPISODE = 1000  # samples over which the schedule is read
LENGTH = N_EPISODE + 426  # samples of an episode's states
MARGIN = 0.02  # every conjunct's margin, and the margin that ends a visit
SMOOTHING_LENGTH = 0.02  # m: a conjunct's gamma is this length over its predicates' normalization length
CELL_RADIUS = 0.03
REACH_SLACK = 10  # samples added to a reach conjunct's deadline
MAX_STAY = 60  # samples an arm may stay inside a cell for the cell to be kept
PERIODS = np.arange(991)
TIMED_PERIOD_RULE = {"periods": (100, 800), "horizon": 100}

# the five-cell zone: four waypoints per arm, the loop of 3.2 s through them, the left arm half a loop behind
ZONE_TARGETS = {"home_r": (0.30, -0.34, 0.74), "pick": (0.40, -0.30, 0.68), "place": (0.45, 0.0, 0.72),
                "wait_r": (0.30, -0.38, 0.70), "home_l": (0.30, 0.34, 0.74), "wait_l": (0.30, 0.38, 0.70),
                "wait_l2": (0.33, 0.44, 0.74)}
ZONE_LOOPS = (("home_r", "pick", "place", "wait_r"), ("home_l", "wait_l", "place", "wait_l2"))
ZONE_WAYPOINTS = ((0, "home_r"), (0, "pick"), (0, "place"), (0, "wait_r"), (1, "home_l"), (1, "wait_l"), (1, "place"))
ZONE_PERIOD = 3.2
ZONE_PHASE = (0.0, 0.5)
ZONE_LIFT = 0.03  # the loops pass this far above each waypoint other than the place
ZONE_SLACK = 30  # samples added to a visit's scheduled stay in its until's window
ZONE_CELLS, ZONE_SEED = 6, 11  # cells tried and the seed of their placement
HOLD_UP_LAG = 137  # samples by which the left arm runs behind in the hold-up

# the 64-cell circle: both hands on a circle of radius 0.16 m in the plane between the arms, one turn in 3 s
CIRCLE_CENTRE = (0.42, 0.0, 0.74)
CIRCLE_RADIUS = 0.16
CIRCLE_PERIOD = 3.0
CIRCLE_LAG = 75
CIRCLE_KNOTS = 32  # joint waypoints of a turn
CIRCLE_WINDOW = 100
CIRCLE_NAMES = ("front", "top", "back", "bottom")  # waypoints at 0, 90, 180 and 270 degrees
CIRCLE_TIP = (22, 46)  # the spheres at the fingers, held at y = -0.02 m (right) and 0.02 m (left)
CIRCLE_HAND_Y, CIRCLE_TIP_Y = 0.10, 0.02
R_PASS = 0.05  # a passage of a circle waypoint is a closest approach within this distance (m)


class Scene:
    """One problem: its states, cells, waypoints, scheduled visits and conjuncts (attributes listed in __init__)."""

    def __init__(self, name, T, states, schedule, cells, waypoints, waypoint_arms, waypoint_names, visits, conjuncts):
        self.name, self.T = name, T
        self.states, self.schedule = states, schedule
        self.cells, self.cell_radius = cells, CELL_RADIUS
        self.waypoints, self.waypoint_arms, self.waypoint_names = waypoints, waypoint_arms, waypoint_names
        self.visits, self.conjuncts = visits, conjuncts
        self.n_episode, self.margin, self.smoothing_length = N_EPISODE, MARGIN, SMOOTHING_LENGTH
        self.periods, self.timed_period_rule = PERIODS, TIMED_PERIOD_RULE
        n_wp = len(waypoints)
        self.n_predicates = n_wp + len(cells) * D.DISTAL.size
        self.lengths = np.concatenate([np.full(n_wp, D.R_GOAL), np.full(len(cells) * D.DISTAL.size, CELL_RADIUS)])
        self.joints = D.predicate_joints(waypoint_arms, len(cells))

    def outside_index(self, z, arm):
        """Predicate indices (17,) of "arm outside cell z"."""
        return len(self.waypoints) + z * D.DISTAL.size + np.flatnonzero(D.ARM_OF[D.DISTAL] == arm)

    def waypoint_index(self, arm, name):
        return [(int(a), n) for a, n in zip(self.waypoint_arms, self.waypoint_names)].index((arm, name))

    def predicates(self, x, exact=False):
        """The predicate vector (n_predicates,) of one state x (28,): the waypoint predicates, then the outside predicates."""
        eps = 0.0 if exact else D.EPS_LENGTH
        cells = jnp.asarray(self.cells, x.dtype)
        return jnp.concatenate([D.at_waypoint(x, self.waypoints, self.waypoint_arms, eps), D.outside(x, cells, CELL_RADIUS, eps)])


# ------------------------------------------------------------------
# geometry along a motion

def hand_and_spheres(X):
    """Hand points (L, 2, 3) and sphere centres (L, 52, 3) along states X (L, 28)."""
    with jax.enable_x64(True):
        hand, C = jax.jit(jax.vmap(D.points))(jnp.asarray(X[:, :D.NQ]))
        return np.asarray(hand), np.asarray(C)


def outside_exact(C, cells):
    """(L, n_cells, 2) per arm and cell the smallest exact outside predicate over the arm's distal spheres."""
    d = (np.linalg.norm(C[:, D.DISTAL][:, None] - cells[None, :, None], axis=-1) - D.RADIUS[D.DISTAL] - CELL_RADIUS) / CELL_RADIUS
    return np.stack([d[:, :, D.ARM_OF[D.DISTAL] == a].min(-1) for a in (0, 1)], -1)


def _longest_run(b):
    """Longest run of True along axis -2 of b (..., L, n)."""
    idx = np.arange(b.shape[-2])[:, None]
    last = np.maximum.accumulate(np.where(~b, idx, -1), axis=-2)
    return np.where(b, idx - last, 0).max(-2)


def _admissible(inside):
    """Whether a cell with inside (..., L, 2) is entered by both arms, never together, and no stay exceeds MAX_STAY."""
    both = (inside[..., 0] & inside[..., 1]).any(-1)
    return inside[..., 0].any(-1) & inside[..., 1].any(-1) & ~both & (_longest_run(inside).max(-1) <= MAX_STAY)


def place_cells(X, n, seed=ZONE_SEED):
    """Cells on the hands' paths of the first N_EPISODE samples of X: candidates are drawn as hand points of the right and
    left arm in turn, at most 200 n draws; a candidate is kept when the cell is admissible and at least 0.03 m from every
    kept cell, until n are kept or the draws run out."""
    hand, C = hand_and_spheres(X[:N_EPISODE])
    draws = np.random.default_rng(seed).integers(N_EPISODE, size=200 * n)
    cand = hand[draws]  # (draws, 2 arms, 3)
    arm = D.ARM_OF[D.DISTAL]
    with jax.enable_x64(True):
        Cd = jnp.asarray(C[:, D.DISTAL])
        r = jnp.asarray(D.RADIUS[D.DISTAL])

        def inside_of(c):  # (L, 2): whether a distal sphere of each arm is inside the cell at c
            d = jnp.sqrt(jnp.sum((Cd - c) ** 2, -1)) - r - CELL_RADIUS
            return jnp.stack([(d[:, arm == b] < 0).any(-1) for b in (0, 1)], -1)

        inside = np.asarray(jax.jit(jax.vmap(inside_of))(jnp.asarray(cand.reshape(-1, 3))))
    inside = inside.reshape(cand.shape[:2] + inside.shape[1:])  # (draws, candidate's arm, L, 2)
    ok = _admissible(inside)  # (draws, candidate's arm)
    out, t = [], 0
    while len(out) < n:  # cells kept (scene construction)
        a = len(out) % 2
        far = np.all(np.linalg.norm(cand[t:, a, None] - np.array(out).reshape(-1, 3)[None], axis=-1) > 0.03, -1)
        hit = np.flatnonzero(ok[t:, a] & far)
        if hit.size == 0:
            break
        out.append(cand[t + hit[0], a])
        t += hit[0] + 1
    return np.array(out)


def visits(out, m=None, window=None):
    """The scheduled visits from out (L, n_cells, 2) (outside_exact): per kept visit the cell, visitor, entry (first sample
    inside), exit (first sample outside by the margin), anchor = entry + 1 and window, either fixed or the stay plus m."""
    o = np.moveaxis(out[:N_EPISODE], 0, -1)  # (cells, 2, L)
    near = o < MARGIN
    pad = np.zeros(near.shape[:2] + (1,), bool)
    d = np.diff(np.concatenate([pad, near, pad], -1).astype(np.int8), axis=-1)
    c0, a0, t0 = np.nonzero(d == 1)
    _, _, t1 = np.nonzero(d == -1)
    ins = np.concatenate([np.zeros(o.shape[:2] + (1,), np.int64), np.cumsum(o < 0, -1)], -1)
    has = ins[c0, a0, t1] - ins[c0, a0, t0] > 0
    c0, a0, t0, t1 = c0[has], a0[has], t0[has], t1[has]
    L = o.shape[-1]
    first_inside = np.minimum.accumulate(np.where(o < 0, np.arange(L), L)[..., ::-1], axis=-1)[..., ::-1]
    e = first_inside[c0, a0, t0]
    s = e + 1
    W = np.full(s.shape, window) if window is not None else (t1 - s) + m
    keep = ((t1 - e) > 1) & (s + W < N_EPISODE)
    return {"cell": c0[keep], "visitor": a0[keep], "entry": e[keep], "exit": t1[keep], "anchor": s[keep], "window": W[keep]}


def _gamma(lengths, nodes):
    idx = np.concatenate([n["preds"] for n in nodes])
    return SMOOTHING_LENGTH / float(np.median(lengths[idx]))


def until_conjuncts(vis, outside_index, lengths):
    """One until conjunct per cell, visitor and window, holding those visits' anchors."""
    res = []
    for z in np.unique(vis["cell"]).tolist():  # cells (scene structure)
        for v in (0, 1):  # visitors
            sel = (vis["cell"] == z) & (vis["visitor"] == v)
            for w in np.unique(vis["window"][sel]):  # windows of this cell and visitor
                nodes = [{"kind": "and", "preds": outside_index(z, 1 - v)}, {"kind": "and", "preds": outside_index(z, v)}]
                f = Until((0, int(w) - 1), Atom(0), Atom(1))
                res.append({"name": "cell_" + str(z) + "_visit_" + "rl"[v] + "_W" + str(int(w)), "kind": "until", "formula": f,
                            "nodes": nodes, "anchors": vis["anchor"][sel & (vis["window"] == w)], "window": int(w),
                            "gamma": _gamma(lengths, nodes)})
    return res


def reach_conjunct(a, name, s_prev, s, index, lengths):
    nodes = [{"kind": "leaf", "preds": np.array([index])}]
    f = Eventually((0, int(s - s_prev + REACH_SLACK)), Atom(0))
    return {"name": "reach_" + "rl"[a] + "_" + name + "_" + str(int(s)), "kind": "reach", "formula": f, "nodes": nodes,
            "anchors": np.array([s_prev]), "window": horizon(to_nnf(f)) + 1, "gamma": _gamma(lengths, nodes)}


def passages(hand, target, within):
    """Samples at which a hand point (L, 3) passes closest to a target: local minima of the distance below `within`."""
    d = np.linalg.norm(hand - np.asarray(target), axis=-1)
    m = (d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:]) & (d[1:-1] < within)
    return np.flatnonzero(m) + 1


def _scene(name, T, states, schedule, cells, waypoints, arms, names, vis, reach):
    """A Scene whose until conjuncts come from the visits and whose reach conjuncts are (arm, name, s_prev, s)."""
    sc = Scene(name, T, states, schedule, cells, waypoints, arms, names, vis, [])
    sc.conjuncts = until_conjuncts(vis, sc.outside_index, sc.lengths)
    sc.conjuncts += [reach_conjunct(a, n, s0, s, sc.waypoint_index(a, n), sc.lengths) for a, n, s0, s in reach]
    return sc


# ------------------------------------------------------------------
# the five-cell zone and the hold-up

def zone_configurations():
    """Joint configurations (7,) per arm and loop waypoint, by inverse kinematics from START; the loop passes ZONE_LIFT above
    each waypoint other than the place."""
    lift = lambda n: np.asarray(ZONE_TARGETS[n]) + (0.0 if n == "place" else ZONE_LIFT) * np.eye(3)[2]  # noqa: E731
    targets = np.array([[lift(a), lift(b)] for a, b in zip(*ZONE_LOOPS)])
    with jax.enable_x64(True):
        lo, hi = (jnp.asarray(v) for v in D.joint_bounds())
        q, _ = D.ik(jnp.asarray(targets), jnp.broadcast_to(jnp.asarray(D.START), (len(targets), D.NQ)), lo, hi)
        q = np.asarray(q)
    return {(a, ZONE_LOOPS[a][i]): q[i, a * D.NJ:(a + 1) * D.NJ] for a in (0, 1) for i in range(len(targets))}


def zone_reference(n, shift, wp):
    """States (n, 28) of the zone's loops with arm a at sample s where its own loop has it at sample s - shift[a]: each arm on
    the shape-preserving piecewise cubic through its loop's configurations, knots spaced by joint-space distance."""
    t = np.arange(n) * D.H_STEP
    X = np.zeros((n, D.NX))
    for a in (0, 1):  # arms
        Q = np.stack([wp[(a, w)] for w in ZONE_LOOPS[a]])
        d = np.linalg.norm(np.diff(np.concatenate([Q, Q[:1]]), axis=0), axis=-1)
        tk = np.concatenate([[0.0], np.cumsum(d)])[:-1] / d.sum() * ZONE_PERIOD
        reps = np.arange(-6, 6)
        S = PchipInterpolator((tk[None] + ZONE_PERIOD * reps[:, None]).ravel(), np.tile(Q, (reps.size, 1)))
        tau = (t - shift[a] * D.H_STEP + ZONE_PHASE[a] * ZONE_PERIOD) % ZONE_PERIOD
        X[:, a * D.NJ:(a + 1) * D.NJ] = S(tau)
        X[:, D.NQ + a * D.NJ:D.NQ + (a + 1) * D.NJ] = S(tau, 1)
    return X


def zone_reaches(hand):
    """(arm, name, s_prev, s) for consecutive passages of the loop waypoints in the loop's order, for waypoints with a
    predicate."""
    out = []
    for a in (0, 1):  # arms
        raw = sorted((int(s), w) for w in ZONE_LOOPS[a] for s in passages(hand[:N_EPISODE, a], ZONE_TARGETS[w], ZONE_LIFT + 0.02))
        seq, nxt = [], None
        for s, w in raw:  # passages in time order, kept in the loop's cyclic order
            if nxt is None or w == nxt:
                seq.append((s, w))
                nxt = ZONE_LOOPS[a][(ZONE_LOOPS[a].index(w) + 1) % len(ZONE_LOOPS[a])]
        out += [(a, w, s0, s) for (s0, _), (s, w) in zip(seq[:-1], seq[1:]) if (a, w) in ZONE_WAYPOINTS and s + REACH_SLACK < N_EPISODE]
    return out


def _zone_scene(name, T, plan_shift, schedule_shift, states):
    wp = zone_configurations()
    cells = place_cells(zone_reference(LENGTH, (0, 0), wp), ZONE_CELLS)
    schedule = zone_reference(LENGTH, schedule_shift, wp)
    states = zone_reference(LENGTH, plan_shift, wp) if states is None else np.asarray(states, np.float64)
    hand, C = hand_and_spheres(schedule)
    vis = visits(outside_exact(C, cells), m=ZONE_SLACK)
    arms = np.array([a for a, _ in ZONE_WAYPOINTS])
    names = tuple(n for _, n in ZONE_WAYPOINTS)
    waypoints = np.array([ZONE_TARGETS[n] for n in names])
    return _scene(name, T, states, schedule, cells, waypoints, arms, names, vis, zone_reaches(hand))


def five_cell_zone(T, states=None):
    """The five-cell zone at horizon T: both arms cross five cells once per loop of 3.2 s, half a loop apart; W_i is the
    scheduled stay plus 30 samples."""
    return _zone_scene("five-cell zone", T, (0, 0), (0, 0), states)


def hold_up(T, delay, states=None):
    """The hold-up at horizon T: the five-cell zone's cells with the left arm HOLD_UP_LAG samples behind; the right arm is
    held up by `delay` samples and the schedule moves with it, so the left arm arrives `delay` samples early."""
    return _zone_scene("hold-up", T, (delay, HOLD_UP_LAG), (delay, HOLD_UP_LAG + delay), states)


# ------------------------------------------------------------------
# the 64-cell circle

def circle_points(phi):
    """Points (..., 3) of the circle at angles phi (rad), 0 the front point, 90 degrees the top."""
    return np.asarray(CIRCLE_CENTRE) + CIRCLE_RADIUS * np.stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)], -1)


def ik_tip(targets, tip_y, q_init, lo, hi, iterations=300, damping=1e-3):
    """ik with one more equation per arm, the y coordinate of the finger sphere at tip_y (B, 2): damped least squares on 8
    equations. Returns q (B, 14) and the largest error left per arm (B, 2)."""
    def f(q):
        hand, C = D.points(q)
        return jnp.concatenate([hand.ravel(), jnp.stack([C[CIRCLE_TIP[0], 1], C[CIRCLE_TIP[1], 1]])])

    def one(target, ty, q):
        g = jnp.concatenate([target.ravel(), ty])

        def step(q, _):
            J = jax.jacfwd(f)(q)
            dq = J.T @ jnp.linalg.solve(J @ J.T + damping * jnp.eye(8, dtype=q.dtype), g - f(q))
            return jnp.clip(q + dq, lo, hi), None

        q, _ = jax.lax.scan(step, q, None, length=iterations)
        e = jnp.abs(f(q) - g)
        return q, jnp.maximum(jnp.linalg.norm((f(q) - g)[:6].reshape(2, 3), axis=-1), e[6:])

    return jax.vmap(one)(targets, tip_y, q_init)


def circle_configurations():
    """Joint waypoints (CIRCLE_KNOTS, 14) at equal angles, both hands on the circle shifted to y = -0.10 m and 0.10 m and
    pointing towards y = 0, by ik_tip by continuation (two passes from the solution at the first angle)."""
    p = circle_points(2 * np.pi * np.arange(CIRCLE_KNOTS) / CIRCLE_KNOTS)
    h = np.stack([p - np.array([0.0, CIRCLE_HAND_Y, 0.0]), p + np.array([0.0, CIRCLE_HAND_Y, 0.0])], 1)
    ty = np.broadcast_to(np.array([-CIRCLE_TIP_Y, CIRCLE_TIP_Y]), (CIRCLE_KNOTS, 2)).copy()
    with jax.enable_x64(True):
        lo, hi = (jnp.asarray(v) for v in D.joint_bounds())
        h, ty = jnp.asarray(h), jnp.asarray(ty)
        q0, _ = ik_tip(h[:1], ty[:1], jnp.asarray(D.START)[None], lo, hi, iterations=1000)

        def cont(q, t):
            q, m = ik_tip(t[0][None], t[1][None], q[None], lo, hi)
            return q[0], (q[0], m[0])

        qe, _ = jax.lax.scan(cont, q0[0], (h, ty))
        _, (q, _) = jax.lax.scan(cont, qe, (h, ty))
        return np.asarray(q)


def circle_reference(n, shift, q):
    """States (n, 28): each arm's joints on the periodic cubic spline through q at equal knot times, arm a at sample s where
    the turn has it at sample s - shift[a]."""
    turn = int(round(CIRCLE_PERIOD / D.H_STEP))
    t = np.arange(n) * D.H_STEP
    S = CubicSpline(np.arange(len(q) + 1) * CIRCLE_PERIOD / len(q), np.concatenate([q, q[:1]]), bc_type="periodic")
    X = np.zeros((n, D.NX))
    for a in (0, 1):  # arms
        tau = (t - shift[a] / turn * CIRCLE_PERIOD) % CIRCLE_PERIOD
        J = slice(a * D.NJ, (a + 1) * D.NJ)
        X[:, J] = S(tau)[:, J]
        X[:, D.NQ + a * D.NJ:D.NQ + (a + 1) * D.NJ] = S(tau, 1)[:, J]
    return X


def circle_cells(C, n):
    """Centres (n, 3) at equal angles on the circle and the mask of the admissible cells over the first N_EPISODE samples."""
    z = circle_points(2 * np.pi * np.arange(n) / n)
    d = np.linalg.norm(C[:, D.DISTAL][:, None] - z[None, :, None], axis=-1) - D.RADIUS[D.DISTAL] - CELL_RADIUS  # (L, n, 34)
    inside = np.stack([(d[:N_EPISODE, :, D.ARM_OF[D.DISTAL] == b] < 0).any(-1) for b in (0, 1)], -1)  # (L, n, 2)
    return z, _admissible(np.moveaxis(inside, 0, 1))


def circle(T, cells=64, delay=0, states=None):
    """The 64-cell circle at horizon T: each arm passes the cells on the circle once per turn of 3 s, the left arm half a
    turn behind; W_i = 100. `cells` cells are placed at equal angles and the admissible ones kept. With a delay the right
    arm is held up by `delay` samples and the schedule moves with it, so the left arm arrives `delay` samples early; the
    cells stay those of the undelayed motion."""
    q = circle_configurations()
    _, C0 = hand_and_spheres(circle_reference(LENGTH, (0, CIRCLE_LAG), q))
    z, keep = circle_cells(C0, cells)
    z = z[keep]
    schedule = circle_reference(LENGTH, (delay, CIRCLE_LAG + delay), q)
    states = circle_reference(LENGTH, (delay, CIRCLE_LAG), q) if states is None else np.asarray(states, np.float64)
    hand, C = hand_and_spheres(schedule)
    vis = visits(outside_exact(C, z), window=CIRCLE_WINDOW)
    tg = circle_points(np.pi / 2 * np.arange(len(CIRCLE_NAMES)))
    waypoints = np.concatenate([tg - np.array([0.0, CIRCLE_HAND_Y, 0.0]), tg + np.array([0.0, CIRCLE_HAND_Y, 0.0])])
    arms = np.repeat([0, 1], len(CIRCLE_NAMES))
    names = CIRCLE_NAMES * 2
    reach = []
    for a in (0, 1):  # arms
        seq = sorted((int(s), n) for n in CIRCLE_NAMES for s in passages(hand[:, a], waypoints[a * len(CIRCLE_NAMES) + CIRCLE_NAMES.index(n)], R_PASS))
        reach += [(a, n, s0, s1) for (s0, _), (s1, n) in zip(seq[:-1], seq[1:]) if s1 + REACH_SLACK < N_EPISODE]
    return _scene("64-cell circle", T, states, schedule, z, waypoints, arms, names, vis, reach)


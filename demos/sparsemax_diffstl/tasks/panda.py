"""Panda task environment for both robot studies (E007, protocol D006).

Status (D006 amendment of 2026-09-29): the study scenario is the human-robot shared workspace
of tasks.workspace (E019), which uses this module's plant, kinematics and dynamics. The random
virtual obstacle spheres, the instances and the specifications below are retired for study use
and remain as test fixtures.

State and control. The state is x = (q, qdot) in R^14 and the control u in R^7 is the
gravity-compensated joint torque command, held constant over each sampling interval h.
Physics runs at DT = 0.002 s with implicitfast, the Newton solver, 100 iterations and
tolerance 1e-8 (Menagerie's options, checked in the tests); h is a whole number of physics
steps, so changing h changes sampling and nothing about the physics.

Torque. The model carries MuJoCo's gravity compensation routed through the actuators: body
gravcomp = 1 on every body, actuatorgravcomp on every joint, and a joint-level actuator force
range equal to the actuator's force range. At every physics step MuJoCo (C, MJX and
MuJoCo-Warp alike) applies

    tau = clip(clip(u, -tau_max, tau_max) + tau_g(q), -tau_max, tau_max),
    tau_g(q) = -sum_b J_b(q)^T m_b g,

with tau_max = (87, 87, 87, 87, 12, 12, 12) N m, so the command and the total joint torque both
stay within Menagerie's force ranges. Joint limits stay in the dynamics; robot geoms carry no
contact rows.

Signals. Predicate scores (atoms) are functions of one state, positive when satisfied, stacked
on the last axis: scores(mx, inst, X) maps (..., T, 14) to (..., T, P), the layout of the STL
layer. The order is score_layout(): goals, speed, then clearances link-major. Each atom is a
distance-like margin divided by its own threshold, smoothed at zero (E014, user ruling
2026-09-28, option C):

    goal       (R_GOAL - n+(p - c)) / R_GOAL
    speed      (V_MAX - n+(pdot)) / V_MAX
    clearance  (n-(c_l - o) - R) / R,          R = r_l + R_OBS

    n+(x) = sqrt(|x|^2 + eps^2) >= |x|,   n-(x) = sqrt(|x|^2 + eps^2) - eps <= |x|,

with eps = EPS_LENGTH for positions and EPS_SPEED for the velocity. So each atom is at most
its exact normalized margin, a positive atom implies the exact distance or speed condition,
one unit is the atom's own threshold, and the loss is at most eps / threshold. Every atom is
infinitely differentiable; its gradient in the physical vector has norm at most
1 / threshold and its Hessian norm at most 1 / (eps threshold), reached at x = 0. n- is
computed as |x|^2 / (n+(x) + eps), the same function without cancellation near zero.

p is the attachment_site position and c_l the centre of the bounding sphere of one collision
geom of a link (link_spheres), both from MJX kinematics; pdot is the forward-mode derivative
of p along qdot. None of these paths touches the constraint solver. lipschitz() gives each
atom's declared bound in the infinity norm of the state, the form E010's guard consumes.

Referee. margins(mx, inst, X) gives the unsmoothed normalized margins in the same layout,
eps = 0 in the norms (root decision on E014, 2026-09-28):

    goal (R_GOAL - |p - c|) / R_GOAL,  speed (V_MAX - |pdot|) / V_MAX,  clearance (|c_l - o| - R) / R.

They are the referee's scores: every exact robustness reported as an outcome reads them,
while every optimized objective reads scores(). Each margin is at least its atom, and every
Panda formula uses its atoms without negation, so the referee's robustness is at least the
exact robustness on the atoms. margins() stops gradients at its input, so no derivative is
ever taken through it; the norm is not differentiable at zero length.

Dynamics. interval_map is f_h, rollout composes it, interval_jacobians gives D_x f_h and
D_u f_h by forward mode. All three take the physics step as an argument; the default is
E004's mjx_implicit.step, whose values equal mjx.step and whose derivatives differentiate
the converged constraint solve.

Specifications. core_spec (phi_H) and extension_specs build E005 formulas over the score
layout, with windows rounded by window().
"""

import math
from fractions import Fraction
from functools import cache, partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from .. import mjx_implicit, plants
from ..stl import Always, And, Atom, Eventually, Predicate

DT = 0.002
REPLAY_DT = 0.0005
NQ = 7
SITE = "attachment_site"
BASE = "link0"
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6", "link7")

R_GOAL = 0.05
R_OBS = 0.05
V_MAX = 0.1
# eps is 1% of the smallest threshold of its physical quantity (E014): R_GOAL for lengths,
# V_MAX for the speed.
EPS_LENGTH = 5e-4
EPS_SPEED = 1e-3

TUNING_SEED = 2026092801
HELDOUT_SEED = 2026092802
N_TUNING = 10
N_HELDOUT = 30
N_INIT = 5
Q_SPREAD = np.array([0.8, 0.5, 0.5, 0.6, 0.6, 0.5, 0.6])
JOINT_MARGIN = 0.1
GOAL_MIN_Z = 0.15
MIN_SEPARATION = 0.25
OBS_FRACTION = (0.35, 0.65)
OBS_OFFSET = 0.06
CLEAR_MARGIN = 0.02
N_CANDIDATES = 4096
INIT_SIGMA = 0.01
INIT_CELL = Fraction(1, 25)
H_MAX = 16


# ------------------------------------------------------------------
# model

@cache
def model(timestep=DT):
    """The task plant, compiled from model_spec(timestep). Cached; do not mutate the result."""
    return model_spec(timestep).compile()


def model_spec(timestep=DT):
    """plants.panda(contacts=False) plus gravity compensation through the actuators and
    joint-level actuator force ranges equal to the actuator force ranges."""
    spec = plants.panda_spec(contacts=False)
    spec.option.timestep = timestep
    force = {a.target: np.array(a.forcerange) for a in spec.actuators}
    for b in spec.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    for j in spec.joints:
        j.actgravcomp = True
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = force[j.name]
    return spec


def torque_limit():
    return model().actuator_ctrlrange[:, 1].copy()


def _body(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _chain(m, body):
    """Degrees of freedom of the joints between the world and body, in increasing order."""
    dofs = []
    while body > 0:
        dofs += range(m.body_dofadr[body], m.body_dofadr[body] + m.body_dofnum[body])
        body = m.body_parentid[body]
    return tuple(sorted(dofs))


# ------------------------------------------------------------------
# link spheres

def collision_geoms(m, body):
    """Collision geoms of a body; Menagerie's collision class is geom group 3."""
    return np.nonzero((m.geom_bodyid == body) & (m.geom_group == 3))[0]


def geom_vertices(m, g):
    """Vertices of mesh geom g in its body's frame."""
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.geom_pos[g] + v @ R.reshape(3, 3).T


def _enclosing_centre(V, iterations=10000):
    """Badoiu-Clarkson iterations toward the minimum enclosing ball of V (N, 3), in float64.
    Only the centre is used; the radius is measured afterwards, so containment does not
    depend on how close this gets to the minimum."""
    # ensure_compile_time_eval: computed now even when the first caller is inside a jit trace,
    # so np.asarray receives a value, not a tracer
    with jax.enable_x64(True), jax.ensure_compile_time_eval():
        V = jnp.asarray(V)

        def body(k, c):
            far = V[jnp.argmax(jnp.sum((V - c) ** 2, -1))]
            return c + (far - c) / (k + 2)

        c = jax.lax.fori_loop(0, iterations, body, (V.min(0) + V.max(0)) / 2)
        return np.asarray(c)


@cache
def link_spheres():
    """One bounding sphere per collision geom of BASE and LINKS, in body then geom order, in
    each body's frame; the first row is BASE's single geom.

    Each centre is rounded to 0.1 mm and each radius is the largest vertex distance from that
    centre plus at least 1 micrometre, rounded up to 0.1 mm. A ball is convex, so containing
    every vertex of a mesh means containing its convex hull, which contains the mesh and is
    the shape MuJoCo collides for a mesh geom. A link's spheres together therefore contain its
    collision geometry. Only the LINKS rows enter clearance predicates; BASE is fixed and is
    used in instance generation.
    """
    m = model()
    geoms = np.concatenate([collision_geoms(m, _body(m, n)) for n in (BASE,) + LINKS])
    assert len(collision_geoms(m, _body(m, BASE))) == 1
    centre, radius = [], []
    for g in geoms:
        V = geom_vertices(m, g)
        c = np.round(_enclosing_centre(V), 4)
        r = math.ceil((np.sqrt(np.max(np.sum((V - c) ** 2, -1))) + 1e-6) * 1e4) / 1e4
        centre.append(c)
        radius.append(r)
    names = tuple(m.mesh(m.geom_dataid[g]).name for g in geoms)
    return {"names": names, "geom": geoms, "body": m.geom_bodyid[geoms], "centre": np.array(centre),
            "radius": np.array(radius)}


# ------------------------------------------------------------------
# kinematics and scores

def frames(mx, q):
    """World positions of the attachment site (3,) and of every sphere centre in link_spheres()
    order (S, 3), from MJX kinematics."""
    s = link_spheres()
    site = mujoco.mj_name2id(model(), mujoco.mjtObj.mjOBJ_SITE, SITE)
    d = mjx.kinematics(mx, mjx.make_data(mx).replace(qpos=q))
    C = d.xpos[s["body"]] + jnp.einsum("lij,lj->li", d.xmat[s["body"]], jnp.asarray(s["centre"], q.dtype))
    return d.site_xpos[site], C


def score_layout(n_goals=2, n_obstacles=2):
    """Names and declared state dependencies of the scores, in the order scores() returns.

    A dependency set lists the state coordinates on the kinematic chain of the score's body:
    joint positions for positions, and also joint velocities for the speed. Coordinates
    outside it have exactly zero derivative.
    """
    m = model()
    s = link_spheres()
    site = _chain(m, _body(m, "attachment"))
    names = ["goal" + str(i) for i in range(n_goals)] + ["speed"]
    deps = [site] * n_goals + [site + tuple(NQ + i for i in site)]
    for name, body in zip(s["names"][1:], s["body"][1:]):
        for o in range(n_obstacles):
            names.append("clear_" + name + "_obs" + str(o))
            deps.append(_chain(m, body))
    return tuple(names), tuple(deps)


def n_plus(x, eps):
    """sqrt(|x|^2 + eps^2) over the last axis, at least |x|."""
    return jnp.sqrt(jnp.sum(x * x, -1) + eps ** 2)


def n_minus(x, eps):
    """sqrt(|x|^2 + eps^2) - eps over the last axis, at most |x|, written as
    |x|^2 / (sqrt(|x|^2 + eps^2) + eps) to avoid cancellation near zero."""
    return jnp.sum(x * x, -1) / (n_plus(x, eps) + eps)


def goal_atom(e):
    """(R_GOAL - n+(e)) / R_GOAL for e = p - c."""
    return (R_GOAL - n_plus(e, EPS_LENGTH)) / R_GOAL


def speed_atom(v):
    """(V_MAX - n+(v)) / V_MAX for the site velocity v."""
    return (V_MAX - n_plus(v, EPS_SPEED)) / V_MAX


def clear_atom(e, R):
    """(n-(e) - R) / R for e = c_l - o and R = r_l + R_OBS."""
    return (n_minus(e, EPS_LENGTH) - R) / R


def _scores(mx, goals, obstacles, x):
    q, qd = x[:NQ], x[NQ:]
    (p, C), (v, _) = jax.jvp(partial(frames, mx), (q,), (qd,))
    C = C[1:]
    R = jnp.asarray(link_spheres()["radius"][1:] + R_OBS, x.dtype)
    goal = goal_atom(p - goals)
    speed = speed_atom(v)
    clear = clear_atom(C[:, None] - obstacles[None], R[:, None])
    return jnp.concatenate([goal, speed[None], clear.reshape(-1)])


def scores(mx, inst, X):
    """Scores (..., T, P) of states X (..., T, 14) for one instance's goals and obstacles."""
    goals = jnp.asarray(inst["goals"], X.dtype)
    obstacles = jnp.asarray(inst["obstacles"], X.dtype)
    return jnp.vectorize(partial(_scores, mx, goals, obstacles), signature="(n)->(p)")(X)


def _norm(e):
    return jnp.sqrt(jnp.sum(e * e, -1))


def _margins(mx, goals, obstacles, x):
    q, qd = x[:NQ], x[NQ:]
    (p, C), (v, _) = jax.jvp(partial(frames, mx), (q,), (qd,))
    C = C[1:]
    R = jnp.asarray(link_spheres()["radius"][1:] + R_OBS, x.dtype)
    goal = (R_GOAL - _norm(p - goals)) / R_GOAL
    speed = (V_MAX - _norm(v)) / V_MAX
    clear = (_norm(C[:, None] - obstacles[None]) - R[:, None]) / R[:, None]
    return jnp.concatenate([goal, speed[None], clear.reshape(-1)])


def margins(mx, inst, X):
    """The referee's unsmoothed normalized margins (..., T, P) of states X (..., T, 14), in the
    layout of scores(); gradients stop at X."""
    X = jax.lax.stop_gradient(X)
    goals = jnp.asarray(inst["goals"], X.dtype)
    obstacles = jnp.asarray(inst["obstacles"], X.dtype)
    return jnp.vectorize(partial(_margins, mx, goals, obstacles), signature="(n)->(p)")(X)


def predicates(mx, inst):
    """The scores as E005 predicates, one per score_layout() entry: Predicate(fn, deps, name)
    with fn a function of one state. scores() computes them all from one kinematics pass."""
    goals, obstacles = jnp.asarray(inst["goals"]), jnp.asarray(inst["obstacles"])
    names, deps = score_layout(len(goals), len(obstacles))

    def entry(i):
        return lambda x: _scores(mx, goals.astype(x.dtype), obstacles.astype(x.dtype), x)[i]

    return tuple(Predicate(entry(i), dep, name) for i, (name, dep) in enumerate(zip(names, deps)))


def reach_bounds(m, body, offset):
    """Upper bounds D_j >= |x - o_j| over all configurations, for the point x at offset (body
    frame) on body, and o_j the anchor of the j-th joint of the body's chain, in chain order.

    D_j is the joint anchor's offset in its body plus the lengths of the body offsets after
    that body up to body, plus |offset|: the triangle inequality along the chain, since
    rotations preserve lengths. Every joint on the chain must be a hinge."""
    path = []
    b = body
    while b > 0:
        path.append(b)
        b = m.body_parentid[b]
    path = path[::-1]  # world -> body
    step = np.array([np.linalg.norm(m.body_pos[b]) for b in path])
    tail = np.concatenate([np.cumsum(step[::-1])[::-1][1:], [0.0]]) + np.linalg.norm(offset)
    D = []
    for i, b in enumerate(path):  # over the bodies of the chain (model structure)
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            D.append(np.linalg.norm(m.jnt_pos[j]) + tail[i])
    return np.array(D)


def lipschitz(n_goals=2, n_obstacles=2, qdot_max=None):
    """Declared bounds l_mu with |g_mu(x) - g_mu(x')| <= l_mu |x - x'|_inf, one per score in
    score_layout() order, the form E010's guard (sparsead.GuardedJacobian) consumes.

    For a point x on a chain of hinges with axes a_j and anchors o_j, the Jacobian column is
    a_j x (x - o_j), so |J dq| <= sum_j D_j |dq|_inf with D_j from reach_bounds. n+ and n- are
    1-Lipschitz in the Euclidean norm, so

        goal       l = sum_j D_j(site) / R_GOAL,
        clearance  l = sum_j D_j(c_l) / (r_l + R_OBS),

    for all states. The speed reads pdot = J(q) qdot, and d J_j / d q_k has norm at most
    D_max(j,k) (the more distal of the two joints), so on the box |qdot| <= qdot_max, which is
    convex and so holds every segment between its points,

        speed      l = (sum_j D_j + sum_{j,k} qdot_max_j D_max(j,k)) / V_MAX.

    Without qdot_max the speed has no global bound and gets inf."""
    m = model()
    s = link_spheres()
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
    D = reach_bounds(m, m.site_bodyid[site], m.site_pos[site])
    out = [D.sum() / R_GOAL] * n_goals
    if qdot_max is None:
        out.append(np.inf)
    else:
        k = np.arange(len(D))
        out.append((D.sum() + np.sum(np.asarray(qdot_max)[:, None] * D[np.maximum.outer(k, k)])) / V_MAX)
    for b, c, r in zip(s["body"][1:], s["centre"][1:], s["radius"][1:]):
        out += [reach_bounds(m, b, c).sum() / (r + R_OBS)] * n_obstacles
    return np.array(out)


# ------------------------------------------------------------------
# time grid

def _exact(t):
    """A time as an exact fraction, reading a float by its decimal representation."""
    return t if isinstance(t, (Fraction, int)) else Fraction(str(t))


def _whole(ratio, what):
    if ratio.denominator != 1 or ratio < 1:
        raise ValueError(what + " must be a positive whole number, got " + str(ratio))
    return int(ratio)


def substeps(h, dt=DT):
    """Physics steps per sampling interval."""
    return _whole(_exact(h) / _exact(dt), "h / dt")


def intervals(H, h):
    """Number of sampling intervals N = H / h; a trajectory has N + 1 samples."""
    return _whole(_exact(H) / _exact(h), "H / h")


def window(a, b, h):
    """Sample indices [ceil(a/h), floor(b/h)] of a window [a, b] in seconds (D006 item 7)."""
    a, b, h = _exact(a), _exact(b), _exact(h)
    return math.ceil(a / h), math.floor(b / h)


# ------------------------------------------------------------------
# specifications, in E005's formula interface

REACH = ((Fraction(1, 4), Fraction(1, 2)), (Fraction(3, 5), Fraction(4, 5)))
DWELL = Fraction(1, 10)


def core_spec(H, h, n_obstacles=2):
    """phi_H of smd:eq:robot_spec (D006 item 12) over the score layout, sampled at h:

        G_[0,H] a  and  F_[H/4,H/2] goal0  and  F_[3H/5,4H/5] G_[0,H/10] goal1,

    with a the conjunction of every clearance score, in layout order, and windows rounded
    by window(). Compile it with T = intervals(H, h) + 1 samples; its horizon is H/h, so
    the robustness trace has one entry, the value at t = 0."""
    H = _exact(H)
    names, _ = score_layout(2, n_obstacles)
    a = And(*[Atom(i) for i, n in enumerate(names) if n.startswith("clear_")])
    return And(Always(window(0, H, h), a),
               Eventually(window(REACH[0][0] * H, REACH[0][1] * H, h), Atom(names.index("goal0"))),
               Eventually(window(REACH[1][0] * H, REACH[1][1] * H, h),
                          Always(window(0, DWELL * H, h), Atom(names.index("goal1")))))


def extension_specs(H, h, n_obstacles=2):
    """The extension's specifications, one constraint row each (D006 item 18), sampled at h:
    for each goal g the reach-and-dwell template sad:eq:task,

        F_[a_g H, b_g H] G_[0, DWELL H] (goal_g and speed),

    with the core study's windows REACH, then G_[0,H] clear for each declared sphere and
    obstacle, in layout order. Returns (names, formulas); compile each with
    T = intervals(H, h) + 1."""
    H = _exact(H)
    names, _ = score_layout(2, n_obstacles)
    speed = Atom(names.index("speed"))
    rows, out = [], []
    for g, (a, b) in enumerate(REACH):  # over goals (formula structure)
        rows.append("reach_dwell_goal" + str(g))
        out.append(Eventually(window(a * H, b * H, h),
                              Always(window(0, DWELL * H, h), And(Atom(names.index("goal" + str(g))), speed))))
    for i, n in enumerate(names):  # over predicates (formula structure)
        if n.startswith("clear_"):
            rows.append("always_" + n)
            out.append(Always(window(0, H, h), Atom(i)))
    return tuple(rows), tuple(out)


# ------------------------------------------------------------------
# dynamics

def interval_map(mx, n_sub, step=mjx_implicit.step):
    """f_h(x, u): the state after n_sub physics steps from x under the constant command u.

    step(m, d) -> d is the physics step. The default is E004's mjx_implicit.step: its values
    are stock MJX's, and its derivatives differentiate the converged constraint solve
    (root's direction, 2026-09-28). mjx.step, the value reference, and E003's mjx_scan.step
    differentiate the executed solver iterations instead, which misses a step's dependence
    on its inputs when the solve returns its warm start without iterating. Each call starts
    from fresh data, so the solver warmstart is zero at the start of every interval and
    carried across its substeps.
    """
    def f(x, u):
        d = mjx.make_data(mx).replace(qpos=x[:NQ], qvel=x[NQ:], ctrl=u)
        d, _ = jax.lax.scan(lambda d, _: (step(mx, d), None), d, length=n_sub)
        return jnp.concatenate([d.qpos, d.qvel])

    return f


def rollout(mx, n_sub, x0, U, step=mjx_implicit.step):
    """Single shooting: states (N + 1, 14) at the sample times from x0 (14,) and U (N, 7).
    It is the composition of interval_map, so it equals multiple shooting with zero defects."""
    f = interval_map(mx, n_sub, step)

    def body(x, u):
        y = f(x, u)
        return y, y

    _, X = jax.lax.scan(body, x0, U)
    return jnp.concatenate([x0[None], X])


def interval_jacobians(mx, n_sub, x, u, step=mjx_implicit.step):
    """D_x f_h (14, 14) and D_u f_h (14, 7) by forward mode, through step (see interval_map)."""
    return jax.jacfwd(interval_map(mx, n_sub, step), argnums=(0, 1))(x, u)


def solver_gradient(mx, d):
    """Scaled norm of the constraint solver's cost gradient M qacc - qfrc_smooth - qfrc_constraint
    for data returned by a step, the quantity MJX compares with opt.tolerance."""
    g = mjx.mul_m(mx, d, d.qacc) - d.qfrc_smooth - d.qfrc_constraint
    return jnp.linalg.norm(g) / (mx.stat.meaninertia * max(1, mx.nv))


# ------------------------------------------------------------------
# instances and initializations

def _config_box():
    m = model()
    lo, hi = m.jnt_range.T
    home = m.key_qpos[0]
    return np.maximum(home - Q_SPREAD, lo + JOINT_MARGIN), np.minimum(home + Q_SPREAD, hi - JOINT_MARGIN)


def instances(seed, n):
    """The first n valid task instances of a seeded candidate stream (copies of a cached draw).

    Each candidate draws three configurations uniformly from _config_box(): the start q0 and
    two goal configurations. The goals are the attachment-site positions at the goal
    configurations, so each is reachable by construction. Obstacle 0 lies on the segment from
    the start position to goal 0 and obstacle 1 on the segment from goal 0 to goal 1, at a
    fraction drawn from OBS_FRACTION, plus an offset drawn from [-OBS_OFFSET, OBS_OFFSET]^3.
    A candidate is valid when the goals are at least GOAL_MIN_Z high, the start and both goals
    are pairwise MIN_SEPARATION apart, every sphere of link_spheres() clears every obstacle by
    CLEAR_MARGIN at all three configurations, and every goal ball clears every obstacle by
    CLEAR_MARGIN. Kinematics run in float64 whatever the global precision, so the result does
    not depend on it. instances(seed, k) is a prefix of instances(seed, n) for k <= n.
    """
    out = _instances(seed)
    if out["accepted"] < n:
        raise ValueError("only " + str(out["accepted"]) + " of " + str(N_CANDIDATES) + " candidates are valid")
    return {k: v[:n].copy() if isinstance(v, np.ndarray) else v for k, v in out.items()}


@cache
def _instances(seed):
    s = link_spheres()
    rng = np.random.default_rng([seed, 0])
    lo, hi = _config_box()
    Q = lo + (hi - lo) * rng.random((N_CANDIDATES, 3, NQ))
    lam = rng.uniform(*OBS_FRACTION, (N_CANDIDATES, 2, 1))
    offset = rng.uniform(-OBS_OFFSET, OBS_OFFSET, (N_CANDIDATES, 2, 3))
    with jax.enable_x64(True):
        mx = mjx.put_model(model(), impl="jax")
        p, C = jax.jit(jax.vmap(jax.vmap(partial(frames, mx))))(jnp.asarray(Q))
        p, C = np.asarray(p), np.asarray(C)
    start, goals = p[:, 0], p[:, 1:]
    O = np.stack([start, goals[:, 0]], 1) + lam * (np.stack([goals[:, 0], goals[:, 1]], 1) - np.stack([start, goals[:, 0]], 1)) + offset

    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    ok = np.all(goals[..., 2] >= GOAL_MIN_Z, -1)
    ok &= dist(start, goals[:, 0]) >= MIN_SEPARATION
    ok &= dist(start, goals[:, 1]) >= MIN_SEPARATION
    ok &= dist(goals[:, 0], goals[:, 1]) >= MIN_SEPARATION
    gap = dist(C[:, :, :, None], O[:, None, None]) - (s["radius"][:, None] + R_OBS + CLEAR_MARGIN)
    ok &= np.all(gap >= 0, (1, 2, 3))
    ok &= np.all(dist(goals[:, :, None], O[:, None]) >= R_GOAL + R_OBS + CLEAR_MARGIN, (1, 2))
    idx = np.nonzero(ok)[0]
    return {"q0": Q[idx, 0], "q_goals": Q[idx, 1:], "goals": goals[idx], "obstacles": O[idx],
            "accepted": len(idx), "candidates": N_CANDIDATES}


def tuning_set():
    return instances(TUNING_SEED, N_TUNING)


def heldout_set():
    return instances(HELDOUT_SEED, N_HELDOUT)


def initial_state(inst):
    """x0 = (q0, 0) for every instance: (..., 14)."""
    q0 = np.asarray(inst["q0"])
    return np.concatenate([q0, np.zeros_like(q0)], -1)


def initial_controls(seed, n, H, h, n_init=N_INIT):
    """Paired control initializations (n, n_init, H / h, 7).

    Torques are piecewise constant on cells of INIT_CELL seconds, drawn i.i.d. from
    N(0, (INIT_SIGMA tau_max)^2) per joint and clipped to the command limits, over H_MAX
    seconds; they are then held at the sampling interval h and cut to the horizon H. The
    initial command, as a function of time, is therefore the same for every h that divides
    INIT_CELL, and the one for a shorter horizon is a prefix of the one for a longer one.
    """
    tau = torque_limit()
    rng = np.random.default_rng([seed, 1])
    W = rng.standard_normal((n, n_init, intervals(H_MAX, INIT_CELL), NQ)) * INIT_SIGMA * tau
    W = np.clip(W, -tau, tau)
    return np.repeat(W, intervals(INIT_CELL, h), axis=2)[:, :, :intervals(H, h)]

"""The Panda in MJX: bounding spheres of the links, the predicate values and their exact margins, the dynamics over a
sampling interval with the implicitly differentiated step, and the seeded instances."""
import math
from functools import cache, partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from ..jax import mjx_implicit
from ..jax.predicates import Predicate
from ..stl import Always, And, Atom, Eventually
from .panda import (BASE, CLEAR_MARGIN, DWELL, EPS_LENGTH, EPS_SPEED, GOAL_MIN_Z, HELDOUT_SEED, LINKS, MIN_SEPARATION, NQ,
                    N_CANDIDATES, N_HELDOUT, N_TUNING, OBS_FRACTION, OBS_OFFSET, REACH, R_GOAL, R_OBS, SITE, TUNING_SEED, V_MAX,
                    _body, _chain, _config_box, _exact, collision_geoms, geom_vertices, model, reach_bounds, window)


# ------------------------------------------------------------------
# link spheres

def _enclosing_centre(V, iterations=10000):
    """Centre for a bounding sphere of V (N, 3), in float64: from the centre of the bounding box, step k moves
    1 / (k + 2) of the way to the farthest point."""
    # evaluated at trace time, so that np.asarray receives a value and not a tracer under jit
    with jax.enable_x64(True), jax.ensure_compile_time_eval():
        V = jnp.asarray(V)

        def body(k, c):
            far = V[jnp.argmax(jnp.sum((V - c) ** 2, -1))]
            return c + (far - c) / (k + 2)

        c = jax.lax.fori_loop(0, iterations, body, (V.min(0) + V.max(0)) / 2)
        return np.asarray(c)


@cache
def link_spheres():
    """One bounding sphere per collision geom of BASE and LINKS, in body then geom order, in each body's frame (the
    first row is BASE's single geom); radius: the largest vertex distance plus 1 micrometre, rounded up to 0.1 mm."""
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
    """Names and state dependencies of the scores, in the order scores() returns: the state coordinates on the
    kinematic chain of each score's body, joint positions and, for the speed, also joint velocities."""
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
    """The margins of scores() with the Euclidean norm in place of the smooth norms, (..., T, P) in the same layout;
    gradients stop at X."""
    X = jax.lax.stop_gradient(X)
    goals = jnp.asarray(inst["goals"], X.dtype)
    obstacles = jnp.asarray(inst["obstacles"], X.dtype)
    return jnp.vectorize(partial(_margins, mx, goals, obstacles), signature="(n)->(p)")(X)


def predicates(mx, inst):
    """The scores as Predicate(fn, deps, name), one per score_layout() entry, with fn a function of one state."""
    goals, obstacles = jnp.asarray(inst["goals"]), jnp.asarray(inst["obstacles"])
    names, deps = score_layout(len(goals), len(obstacles))

    def entry(i):
        return lambda x: _scores(mx, goals.astype(x.dtype), obstacles.astype(x.dtype), x)[i]

    return tuple(Predicate(entry(i), dep, name) for i, (name, dep) in enumerate(zip(names, deps)))


def lipschitz(n_goals=2, n_obstacles=2, qdot_max=None):
    """Lipschitz bounds of the scores in the infinity norm of the state, in score_layout() order, from reach_bounds.
    The speed bound holds on the box |qdot| <= qdot_max and is inf without qdot_max."""
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
# specifications

def core_spec(H, h, n_obstacles=2):
    """G_[0,H] a and F_[H/4,H/2] goal0 and F_[3H/5,4H/5] G_[0,H/10] goal1 over the score layout, sampled at h, with a
    the conjunction of every clearance score."""
    H = _exact(H)
    names, _ = score_layout(2, n_obstacles)
    a = And(*[Atom(i) for i, n in enumerate(names) if n.startswith("clear_")])
    return And(Always(window(0, H, h), a),
               Eventually(window(REACH[0][0] * H, REACH[0][1] * H, h), Atom(names.index("goal0"))),
               Eventually(window(REACH[1][0] * H, REACH[1][1] * H, h),
                          Always(window(0, DWELL * H, h), Atom(names.index("goal1")))))


def extension_specs(H, h, n_obstacles=2):
    """(names, formulas), sampled at h: F_[a_g H, b_g H] G_[0, DWELL H] (goal_g and speed) for each goal g with the
    windows REACH, then G_[0,H] of each clearance score."""
    H = _exact(H)
    names, _ = score_layout(2, n_obstacles)
    speed = Atom(names.index("speed"))
    rows, out = [], []
    for g, (a, b) in enumerate(REACH):
        rows.append("reach_dwell_goal" + str(g))
        out.append(Eventually(window(a * H, b * H, h),
                              Always(window(0, DWELL * H, h), And(Atom(names.index("goal" + str(g))), speed))))
    for i, n in enumerate(names):
        if n.startswith("clear_"):
            rows.append("always_" + n)
            out.append(Always(window(0, H, h), Atom(i)))
    return tuple(rows), tuple(out)


# ------------------------------------------------------------------
# dynamics

def interval_map(mx, n_sub, step=mjx_implicit.step):
    """f_h(x, u): the state after n_sub physics steps of step(m, d) -> d from x under the constant command u. Each
    call starts from new data, so the constraint solver's warm start is zero at the start of the interval."""
    def f(x, u):
        d = mjx.make_data(mx).replace(qpos=x[:NQ], qvel=x[NQ:], ctrl=u)
        d, _ = jax.lax.scan(lambda d, _: (step(mx, d), None), d, length=n_sub)
        return jnp.concatenate([d.qpos, d.qvel])

    return f


def rollout(mx, n_sub, x0, U, step=mjx_implicit.step):
    """Single shooting: states (N + 1, 14) at the sample times from x0 (14,) and U (N, 7)."""
    f = interval_map(mx, n_sub, step)

    def body(x, u):
        y = f(x, u)
        return y, y

    _, X = jax.lax.scan(body, x0, U)
    return jnp.concatenate([x0[None], X])


def interval_jacobians(mx, n_sub, x, u, step=mjx_implicit.step):
    """D_x f_h (14, 14) and D_u f_h (14, 7) of interval_map by forward mode."""
    return jax.jacfwd(interval_map(mx, n_sub, step), argnums=(0, 1))(x, u)


def solver_gradient(mx, d):
    """Norm of M qacc - qfrc_smooth - qfrc_constraint for data returned by a step, divided by meaninertia max(1, nv)."""
    g = mjx.mul_m(mx, d, d.qacc) - d.qfrc_smooth - d.qfrc_constraint
    return jnp.linalg.norm(g) / (mx.stat.meaninertia * max(1, mx.nv))


# ------------------------------------------------------------------
# instances

def instances(seed, n):
    """The first n valid instances of a seeded candidate stream: a start configuration, two goals at the site positions
    of drawn configurations, and two obstacles near the segments between them; a prefix of any longer draw."""
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

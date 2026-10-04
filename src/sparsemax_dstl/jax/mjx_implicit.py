"""MJX step and forward whose constraint solve is differentiated by the implicit function theorem at the
solver's optimality condition G(qacc) = 0 (Appendix III of the paper). For mujoco-mjx 3.12.0 and impl="jax"."""

from importlib.metadata import version

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

MJX_VERSION = "3.12.0"
if mujoco.__version__ != MJX_VERSION or version("mujoco-mjx") != MJX_VERSION:
    raise ImportError(
        "sparsemax_dstl.jax.mjx_implicit relies on private mujoco-mjx " + MJX_VERSION
        + " internals, but mujoco " + mujoco.__version__ + " and mujoco-mjx "
        + version("mujoco-mjx") + " are installed"
    )

from mujoco.mjx._src import forward as mjx_forward  # noqa: E402
from mujoco.mjx._src import math as mjx_math  # noqa: E402
from mujoco.mjx._src import scan, sensor, solver, support  # noqa: E402
from mujoco.mjx._src.types import (  # noqa: E402
    ConeType,
    DataJAX,
    Impl,
    IntegratorType,
    ModelJAX,
    OptionJAX,
    SolverType,
)


def _check(m, d):
    if (
        m.impl != Impl.JAX
        or not isinstance(m._impl, ModelJAX)
        or not isinstance(m.opt._impl, OptionJAX)
        or not isinstance(d._impl, DataJAX)
    ):
        raise ValueError("mjx_implicit supports only models and data built with impl='jax'")
    if m.opt.solver not in (SolverType.CG, SolverType.NEWTON):
        raise NotImplementedError("mjx_implicit supports the CG and NEWTON solvers, not " + str(m.opt.solver))
    if m.opt.cone not in (ConeType.PYRAMIDAL, ConeType.ELLIPTIC):
        raise NotImplementedError("mjx_implicit supports pyramidal and elliptic cones, not " + str(m.opt.cone))


def stationarity(m, d, qacc):
    """G = M qacc - qfrc_smooth - qfrc_constraint, the gradient of the solver's cost at qacc, with efc_force
    and qfrc_constraint there; d is the data the solver receives."""
    ctx = solver.Context.create(m, d.replace(qacc=qacc), grad=False)
    return ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint, ctx.efc_force, ctx.qfrc_constraint


def _cone_hessian(ctx):
    """Hessian of the elliptic middle-zone cost in Jaref, one 6x6 block per contact, zero outside that zone."""
    mu, u, fri, dm = ctx.fri[:, 0], ctx.u, ctx.fri, ctx.dm
    n = u[:, 0]
    t = jax.vmap(mjx_math.norm)(u[:, 1:])
    middle = (t > 0) & (n < mu * t) & (mu * n + t > 0)
    t = jnp.maximum(t, mujoco.mjMINVAL)
    ttt = jnp.maximum(t * t * t, mujoco.mjMINVAL)
    h = (mu * n / ttt)[:, None, None] * u[:, :, None] * u[:, None, :]
    h = h + (mu * mu - mu * n / t)[:, None, None] * jnp.eye(6, dtype=u.dtype)
    edge = jnp.concatenate([jnp.ones_like(mu)[:, None], -(mu / t)[:, None] * u[:, 1:]], axis=1)
    h = h.at[:, 0, :].set(edge).at[:, :, 0].set(edge)
    h = h * dm[:, None, None] * fri[:, :, None] * fri[:, None, :]
    return jnp.where(middle[:, None, None], h, 0.0)


def hessian(m, d, qacc):
    """H = M + J^T C J, the Jacobian of G in qacc with the constraint state at qacc held fixed: C is
    diag(efc_D) on the rows MJX marks active plus the elliptic middle-zone blocks."""
    ctx = solver.Context.create(m, d.replace(qacc=qacc), grad=False)
    efc_j = d._impl.efc_J
    h = support.full_m(m, d) + (efc_j.T * (d._impl.efc_D * ctx.active)) @ efc_j
    cone = d._impl.contact.dim > 1
    if m.opt.cone == ConeType.ELLIPTIC and cone.any():
        addr = d._impl.contact.efc_address[cone]
        dim = d._impl.contact.dim[cone]
        k = np.arange(6)
        rows = np.where(k < dim[:, None], addr[:, None] + k, efc_j.shape[0])
        jc = jnp.concatenate([efc_j, jnp.zeros_like(efc_j[:1])])[rows]
        h = h + jnp.einsum("cin,cij,cjm->nm", jc, _cone_hessian(ctx), jc)
    return 0.5 * (h + h.T)


@jax.custom_jvp
def solve(m, d):
    """(qacc, qfrc_constraint, efc_force) of MJX's solver.solve; the derivative rule is _solve_jvp."""
    d = solver.solve(m, d)
    return d.qacc, d.qfrc_constraint, d._impl.efc_force


@solve.defjvp
def _solve_jvp(primals, tangents):
    m, d = primals
    m_dot, d_dot = tangents
    out = solve(m, d)
    qacc = out[0]
    # tangents of G and of the two forces at fixed qacc; then qacc_dot = -H^{-1} g_dot
    _, (g_dot, f_dot, q_dot) = jax.jvp(lambda m, d: stationarity(m, d, qacc), (m, d), (m_dot, d_dot))
    factor = jax.scipy.linalg.cho_factor(hessian(m, d, qacc))
    qacc_dot = -jax.scipy.linalg.cho_solve(factor, g_dot)
    # tangents of the two forces through qacc
    _, (_, fa_dot, qa_dot) = jax.jvp(lambda a: stationarity(m, d, a), (qacc,), (qacc_dot,))
    return out, (qacc_dot, q_dot + qa_dot, f_dot + fa_dot)


def forward(m, d):
    """The stages of mjx.forward with this module's solve in place of solver.solve."""
    _check(m, d)
    d = mjx_forward.fwd_position(m, d)
    d = sensor.sensor_pos(m, d)
    d = mjx_forward.fwd_velocity(m, d)
    d = sensor.sensor_vel(m, d)
    d = mjx_forward.fwd_actuation(m, d)
    d = mjx_forward.fwd_acceleration(m, d)
    if d._impl.efc_J.size == 0:
        return d.replace(qacc=d.qacc_smooth)
    qacc, qfrc_constraint, efc_force = solve(m, d)
    d = d.tree_replace({"qacc": qacc, "qfrc_constraint": qfrc_constraint, "_impl.efc_force": efc_force})
    return sensor.sensor_acc(m, d)


def _rungekutta4(m, d):
    """MJX's rungekutta4 with this module's forward."""
    d0 = d
    a, b = mjx_forward._RK4_A, mjx_forward._RK4_B
    c = jnp.tril(a).sum(axis=0)
    times = d.time + c * m.opt.timestep
    kqvel = d.qvel
    qvel, qacc, act_dot = jax.tree_util.tree_map(lambda k: b[0] * k, (kqvel, d.qacc, d.act_dot))
    integrate = lambda *args: mjx_forward._integrate_pos(*args, dt=m.opt.timestep)

    def stage(carry, x):
        qvel, qacc, act_dot, kqvel, d = carry
        ai, bi, ti = x
        dqvel, dqacc, dact_dot = jax.tree_util.tree_map(lambda k: ai * k, (kqvel, d.qacc, d.act_dot))
        kqpos = scan.flat(m, integrate, "jqv", "q", m.jnt_type, d0.qpos, dqvel)
        kact = d0.act + dact_dot * m.opt.timestep
        kqvel = d0.qvel + dqacc * m.opt.timestep
        d = forward(m, d.replace(qpos=kqpos, qvel=kqvel, act=kact, time=ti))
        qvel += bi * kqvel
        qacc += bi * d.qacc
        act_dot += bi * d.act_dot
        return (qvel, qacc, act_dot, kqvel, d), None

    abt = jnp.vstack([jnp.diag(a), b[1:4], times]).T
    out, _ = jax.lax.scan(stage, (qvel, qacc, act_dot, kqvel, d), abt, unroll=3)
    qvel, qacc, act_dot, _, d1 = out
    d = d1.replace(qpos=d0.qpos, qvel=d0.qvel, act=d0.act, time=d0.time)
    return mjx_forward._advance(m, d, act_dot, qacc, qvel)


def step(m, d):
    """mjx.step with this module's forward, for the Euler, implicitfast and RK4 integrators."""
    d = forward(m, d)
    if m.opt.integrator == IntegratorType.EULER:
        return mjx_forward.euler(m, d)
    if m.opt.integrator == IntegratorType.IMPLICITFAST:
        return mjx_forward.implicit(m, d)
    if m.opt.integrator == IntegratorType.RK4:
        return _rungekutta4(m, d)
    raise NotImplementedError("mjx_implicit does not support integrator " + str(m.opt.integrator))

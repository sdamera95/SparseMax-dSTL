"""sparsemax_dstl.warp_plant (E027): the Panda on the MuJoCo-Warp adjoint fork.

Needs a CUDA device (graph capture); run with CUDA_VISIBLE_DEVICES pinned to one GPU. The
derivative check against mjx_implicit runs JAX in float64 on the CPU.
"""
import numpy as np
import pytest
import warp as wp

from sparsemax_dstl.tasks import panda as P

pytestmark = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="needs a CUDA device")


def _start(n, T, seed=0):
    rng = np.random.default_rng(seed)
    q = P.model().key_qpos[0] + 0.2 * rng.standard_normal((n, 7))
    x0 = np.concatenate([q, 0.1 * rng.standard_normal((n, 7))], 1)
    return x0, np.clip(0.3 * rng.standard_normal((n, T, 7)), -1, 1)


@pytest.fixture(scope="module")
def WP():
    from sparsemax_dstl import warp_plant
    return warp_plant


def test_rollout_matches_mujoco_c(WP):
    import mujoco
    from mujoco import rollout
    x0, V = _start(2, 5)
    X = WP.Plant(2, 5).rollout(x0, V)
    m = P.model()
    init = np.concatenate([np.zeros((2, 1)), x0], 1)
    st, _ = rollout.rollout(m, mujoco.MjData(m), init, np.repeat(V * P.torque_limit(), 10, 1))
    Xc = np.concatenate([x0[:, None], st[:, 9::10, 1:15]], 1)
    assert np.abs(X - Xc).max() < 1e-4


def test_graph_replay_matches_eager(WP):
    x0, V = _start(1, 3)
    C = np.random.default_rng(1).standard_normal((1, 4, 14))
    out = []
    for graph in (True, False):  # captured and eager launch sequences
        pl = WP.Plant(1, 3, graph=graph)
        X = pl.rollout(x0, V)
        out.append((X, *pl.vjp(C)))
    for a, b in zip(*out):  # states, control and initial-state cotangents
        assert np.abs(a - b).max() <= 1e-6 * max(1.0, np.abs(b).max())


def test_vjp_is_the_chain_of_interval_vjps(WP):
    """The rollout's VJP with cotangents on every sample equals back-propagating interval by
    interval through one-interval rollouts from the rollout's own sample states (checkpointed
    backward across intervals, cotangent injection at the boundaries, the warm start reset at
    each interval)."""
    x0, V = _start(1, 3)
    C = np.random.default_rng(2).standard_normal((1, 4, 14))
    pl = WP.Plant(1, 3)
    X = pl.rollout(x0, V)
    gV, gx0 = pl.vjp(C)
    one = WP.Plant(1, 1)
    g = C[:, 3]
    gVs = []
    for t in (2, 1, 0):  # interval-by-interval reference (a brute-force test oracle over three intervals)
        one.rollout(X[:, t], V[:, t:t + 1])
        gv, gx = one.vjp(np.stack([np.zeros_like(g), g], 1))
        gVs.append(gv[:, 0])
        g = gx + C[:, t]
    assert np.allclose(gV[0], np.stack(gVs[::-1], 1)[0], rtol=1e-4, atol=1e-6)
    assert np.allclose(gx0, g, rtol=1e-4, atol=1e-6)


def test_clamp_leaf(WP):
    """At home, joint 1's axis is vertical, so tau_g,1 = 0, and joint 2 carries gravity
    (tau_g,2 = -25.2 N m, MuJoCo C). A command at -u_max on joint 2 saturates the joint-level
    clamp (s < -u_max): its control column is exactly zero. A command exactly at +u_max on joint 1 (s = u_max, the boundary) gets the
    derivative from inside the range: equal to the column at 0.9999 u_max."""
    q = P.model().key_qpos[0]
    x0 = np.repeat(np.concatenate([q, np.zeros(7)])[None], 14 * 2, 0)
    V = np.zeros((28, 1, 7))
    V[:, 0, 1] = -1.0
    V[:14, 0, 0] = 1.0
    V[14:, 0, 0] = 0.9999
    C = np.zeros((28, 2, 14))
    C[:, 1] = np.tile(np.eye(14), (2, 1))
    pl = WP.Plant(28, 1)
    pl.rollout(x0, V)
    gV, _ = pl.vjp(C)
    Ju = gV[:, 0].reshape(2, 14, 7)
    assert np.all(Ju[:, :, 1] == 0.0)
    assert np.abs(Ju[0, :, 0]).max() > 0
    assert np.abs(Ju[0, :, 0] - Ju[1, :, 0]).max() <= 1e-3 * np.abs(Ju[1, :, 0]).max()


def test_against_mjx_implicit(WP):
    """One interval at two states, all 14 unit cotangents, against mjx_implicit in float64."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    x0, V = _start(2, 1, seed=3)
    pl = WP.Plant(28, 1)
    pl.rollout(np.repeat(x0, 14, 0), np.repeat(V, 14, 0))
    C = np.zeros((28, 2, 14))
    C[:, 1] = np.tile(np.eye(14), (2, 1))
    gV, gx = pl.vjp(C)
    um = P.torque_limit()
    with jax.enable_x64(True), jax.default_device(jax.devices("cpu")[0]):
        mx = mjx.put_model(P.model(), impl="jax")
        f = P.interval_map(mx, 10)

        def rows(x, u):
            _, pull = jax.vjp(f, x, u)
            return jax.vmap(pull)(jnp.eye(14))

        Jx, Ju = jax.jit(jax.vmap(rows))(jnp.asarray(x0), jnp.asarray(V[:, 0] * um))
    Jx_f, Ju_f = gx.reshape(2, 14, 14), (gV[:, 0] / um).reshape(2, 14, 7)
    assert np.abs(Jx_f - np.asarray(Jx)).max() <= 1e-4 * np.abs(np.asarray(Jx)).max()
    assert np.abs(Ju_f - np.asarray(Ju)).max() <= 1e-4 * np.abs(np.asarray(Ju)).max()

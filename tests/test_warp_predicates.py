"""Tests of the Warp predicates (sparsemax_dstl.warp.predicates) against tasks.workspace_mjx.scores on the CPU in float32 and float64."""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from mujoco import mjx

from sparsemax_dstl.tasks import workspace as W
from sparsemax_dstl.tasks import workspace_mjx as Wm
from sparsemax_dstl.warp.predicates import Predicates

wp.config.log_level = wp.LOG_WARNING
DTYPES = {np.float32: wp.float32, np.float64: wp.float64}
# values (absolute); gradients with the zero-length atoms removed from the cotangent and on the full cotangent (relative to the largest
# reference entry). float32 is looser on the full cotangent: the smoothed atoms have a large Hessian at zero length, which amplifies rounding differences
TOL = {np.float32: (2e-5, 1e-4, 1e-3), np.float64: (1e-12, 1e-10, 1e-10)}
B, T, NH = 2, 5, 5


def x64(dtype):
    return jax.enable_x64(True) if dtype == np.float64 else contextlib.nullcontext()


def case(mx, plant, sc, rng):
    m = plant.model
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    q = rng.uniform(lo, hi, (B, T, 7))
    qd = rng.standard_normal((B, T, 7))
    qd[1, 0] = 0.0
    hc = rng.uniform([-0.2, -0.5, 0.0], [0.8, 0.5, 1.0], (B, T, NH, 3))
    hr = rng.uniform(0.04, 0.1, NH)
    goals = rng.uniform([0.2, -0.5, 0.1], [0.7, 0.5, 0.6], (2, B, 3))
    p, C = jax.vmap(lambda q: Wm.points(mx, plant, sc.robot_spacing, q))(jnp.asarray(q[1, 1:3]))
    goals[0, 1] = np.asarray(p[0])
    hc[1, 2, 0] = np.asarray(C[1, 4])
    return np.concatenate([q, qd], -1), hc, hr, goals


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_atoms_and_tape_gradients(dtype):
    plant, sc = W.Plant(), W.Scenario()
    rng = np.random.default_rng(30)
    with x64(dtype):
        mx = mjx.put_model(plant.model, impl="jax")
        X, hc, hr, goals = (np.asarray(a, dtype) for a in case(mx, plant, sc, rng))

        def ref(X, hc, pick, hand):
            return Wm.scores(mx, plant, sc, {"pick": pick, "handover": hand, "human_centres": hc, "human_radii": jnp.asarray(hr)}, X)

        Zr, pull = jax.vjp(lambda X: jax.vmap(ref)(X, jnp.asarray(hc), jnp.asarray(goals[0]), jnp.asarray(goals[1])), jnp.asarray(X))
        G = rng.standard_normal(Zr.shape).astype(dtype)
        Gg = G.copy()
        Gg[1, 1, 0] = 0.0  # the zero-length pick atom
        n_r = (Zr.shape[-1] - 3) // (1 + 2 * NH)
        Gg[1, 2, [3 + n_r + 4 * NH, 3 + n_r + n_r * NH + 4 * NH]] = 0.0  # robot sphere 4 and human sphere 0, centres equal
        gr, grg = (np.asarray(pull(jnp.asarray(g))[0]) for g in (G, Gg))
    dt = DTYPES[dtype]
    pr = Predicates(plant, sc, T, nworld=B, dtype=dt, device="cpu")
    pr.set_instance(goals[0], goals[1], hc, hr)

    def warp_vjp(G):
        q = wp.array(X[..., :7].reshape(B * T, 7), dtype=dt, device="cpu", requires_grad=True)
        v = wp.array(X[..., 7:].reshape(B * T, 7), dtype=dt, device="cpu", requires_grad=True)
        tape = wp.Tape()
        Z = pr.scores(q, v, tape)
        tape.backward(grads={Z: wp.array(G, dtype=dt, device="cpu")})
        return Z.numpy(), np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(X.shape)

    Z, gw = warp_vjp(G)
    _, gwg = warp_vjp(Gg)
    tv, tg, t0 = TOL[dtype]
    assert Z.shape == Zr.shape
    assert np.abs(Z - np.asarray(Zr)).max() <= tv
    assert np.abs(gwg - grg).max() <= tg * np.abs(grg).max()
    assert np.abs(gw - gr).max() <= t0 * np.abs(gr).max()
    assert np.all(np.isfinite(gw))


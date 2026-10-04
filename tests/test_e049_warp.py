"""Tests of the Warp chain (tasks.planar_warp) against the JAX chain (tasks.planar_al_jax) on the CPU in float64."""
import jax
import numpy as np
import pytest

from sparsemax_dstl.tasks import planar_disk as D
from sparsemax_dstl.tasks.planar_al_jax import JaxChain
from sparsemax_dstl.tasks.planar_warp import WarpChain

jax.config.update("jax_enable_x64", True)

T = 120
N = 2
EPS = 0.1
Z0 = (1.5, 1.0, np.pi / 2)
REG = D.make_regions((3.0, 4.5, 1.4), (5.3, 7.3, 1.0), (8.3, 2.5, 1.0), (8.8, 6.8, 0.5))
SPEC, CONJ, _ = D.specification(40, 60, 80, 95, T)
METHODS = ("lse_plain", "gm_pm01", "gm_pm10", "sparsemax")


def commands():
    rng = np.random.default_rng(0)
    rand = rng.uniform(-0.5, 0.5, (N, T - 1, 2))
    s = np.arange(T - 1) / (T - 1)
    smooth = np.stack([np.full((N, T - 1), 0.9), 0.4 * np.sin(2 * np.pi * s + np.array([[0.0], [1.0]]))], -1)
    trials = rng.uniform(-1.0, 1.0, (N, 7, T - 1, 2))
    return {"random": rand, "smooth": smooth}, trials, rng.uniform(0.0, 2.0, (N, 4))


@pytest.mark.parametrize("K", (4, 1))
@pytest.mark.parametrize("method", METHODS)
def test_warp_matches_jax(method, K):
    conj = CONJ if K == 4 else (SPEC,)
    jc = JaxChain(conj, T, method, EPS, Z0, REG, N)
    wc = WarpChain(conj, T, method, EPS, Z0, REG, N)
    Vs, Va, w = commands()
    w = w[:, :K]
    for V in Vs.values():
        r_j, r_w = jc.forward(V), wc.forward(V)
        print(method, K, "r", np.abs(r_j - r_w).max(), "pullback", np.abs(jc.pullback(w) - wc.pullback(w)).max())
        assert np.abs(r_j - r_w).max() < 1e-9
        assert np.abs(jc.pullback(w) - wc.pullback(w)).max() < 1e-9
        assert np.abs(jc.C).max() > 1e-3  # the gradients compared are not all zero
    v_j, v_w = jc.values(Va), wc.values(Va)
    print(method, K, "values", np.abs(v_j - v_w).max())
    assert v_w.shape == (N, 7, K)
    assert np.abs(v_j - v_w).max() < 1e-9


@pytest.mark.parametrize("method", ("lse_plain", "sparsemax"))
def test_warp_gradient_finite_difference(method):
    """Test check only: (r_0(V + h D) - r_0(V - h D)) / (2 h) against <C_0, D> for the until conjunct."""
    wc = WarpChain(CONJ, T, method, EPS, Z0, REG, N)
    V = commands()[0]["smooth"]
    D_ = np.random.default_rng(1).normal(size=V.shape)
    wc.forward(V)
    ad = np.sum(wc.C[:, 0] * D_, (1, 2))
    h = 1e-6
    pts = np.concatenate([(V + h * D_)[:, None], (V - h * D_)[:, None], np.repeat(V[:, None], 5, 1)], 1)
    r = wc.values(pts)[..., 0]
    fd = (r[:, 0] - r[:, 1]) / (2 * h)
    print(method, "ad", ad, "fd", fd)
    assert np.allclose(fd, ad, rtol=1e-5, atol=1e-7)

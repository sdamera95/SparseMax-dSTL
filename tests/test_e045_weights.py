"""Weight sums of the Until's gradient and their closed forms (examples.e045_two_properties) against reverse-mode automatic differentiation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import e045_two_properties as E
from sparsemax_dstl import jax as stl_jax
from sparsemax_dstl import stl
from sparsemax_dstl.jax import methods

jax.config.update("jax_enable_x64", True)

T, A, B = 40, 20, 25
KW = np.arange(A, B + 1)
HOLD = np.arange(T) >= 10


def signal(depths, gap):
    """Signal (1, T, 3): column 0 is 0.5, column 2 is 0.3 with -depths from sample 3 and gap from sample 10."""
    c = np.full(T, 0.3)
    c[10:] = gap
    c[3:3 + len(depths)] = -np.asarray(depths)
    return np.stack([np.full(T, 0.5), np.ones(T), c], -1)[None]


def ad(Z, arm, eps):
    prog = stl.compile_formula(stl.Until((A, B), stl.Atom(2), stl.Atom(0)), T)
    sem = methods.SEMANTICS[arm]
    f = jax.vmap(jax.value_and_grad(lambda y: stl_jax.evaluate(prog, y, sem, eps)[-1][0]))
    v, g = f(jnp.asarray(Z))
    return np.asarray(v), np.asarray(g)


def test_sums_on_a_known_gradient():
    g = np.zeros((1, T, 3))
    g[0, 3, 2], g[0, 12, 2], g[0, 22, 0] = 0.25, 0.5, 0.25
    viol = (np.arange(T) == 3)[None]
    s = E.sums(g, viol, HOLD[None] & ~viol)
    assert s["w_viol"][0] == 0.25 and s["w_hold"][0] == 0.5 and s["w_pick"][0] == 0.25 and s["w_total"][0] == 1.0
    assert s["nonzero"][0] == 3 and s["nonzero_clear"][0] == 2 and s["n90"][0] == 3


def test_sparsemax_puts_weight_one_on_tied_violating_samples():
    Z = signal([0.1] * 5, 0.02)
    viol = Z[..., 2] < 0
    _, g = ad(Z, "sparsemax", 0.05)
    s = E.sums(g, viol, HOLD[None] & ~viol)
    P = E.predict(Z[..., 2], Z[..., 0], viol, KW, HOLD, "sparsemax", 0.05)
    assert abs(s["w_viol"][0] - 1.0) < 1e-12 and s["w_hold"][0] == 0.0
    assert P["law"][0] == 1.0 and P["cond_margin"][0] > 0
    assert abs(P["closed"][0] - 1.0) < 1e-12


def test_gmr_pm01_law():
    Z = signal([0.1, 0.05, 0.08], 0.02)
    viol = Z[..., 2] < 0
    _, g = ad(Z, "gm_pm01", None)
    want = 3 / (2 * np.exp(np.mean(np.log(KW + 1.0))))
    assert abs(E.sums(g, viol, HOLD[None])["w_viol"][0] - want) < 1e-12


@pytest.mark.parametrize("arm,eps", [("sparsemax", 0.05), ("sparsemax", 2.0), ("lse", 0.1), ("lse", 0.4), ("lse_plain", 0.2),
                                     ("gm_pm01", np.nan), ("gm_pm10", np.nan), ("gm_exp", 0.1), ("gm_exp", 0.4)])
def test_closed_forms_equal_automatic_derivative(arm, eps):
    Z = signal([0.02, 0.06, 0.1, 0.07, 0.03], 0.01)
    viol = Z[..., 2] < 0
    _, g = ad(Z, arm, None if np.isnan(eps) else eps)
    P = E.predict(Z[..., 2], Z[..., 0], viol, KW, HOLD, arm, eps)
    assert abs(E.sums(g, viol, HOLD[None] & ~viol)["w_viol"][0] - P["closed"][0]) < 1e-10


@pytest.mark.parametrize("arm,eps", [("sparsemax", 0.05), ("sparsemax", 2.0), ("lse", 0.2), ("gm_pm01", np.nan), ("gm_pm10", np.nan), ("gm_exp", 0.2)])
def test_closed_forms_on_a_selection(arm, eps):
    # a slightly negative clear sample (t = 8) that is not summed: the weight sum covers the selected samples only
    Z = signal([0.02, 0.06, 0.1, 0.07, 0.03], 0.01)
    Z[0, 8, 2] = -0.001
    neg = Z[..., 2] < 0
    sel = neg & (np.arange(T) != 8)[None]
    _, g = ad(Z, arm, None if np.isnan(eps) else eps)
    P = E.predict(Z[..., 2], Z[..., 0], neg, KW, HOLD, arm, eps, sel=sel)
    assert abs(E.sums(g, sel, HOLD[None] & ~neg)["w_viol"][0] - P["closed"][0]) < 1e-10

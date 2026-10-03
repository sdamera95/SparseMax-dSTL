"""Generalized mean robustness (E036): the arms gm_pm01 and gm_exp against the paper's equations.

Source: Mehdipour, Vasile and Belta, Generalized mean robustness for signal temporal
logic, IEEE TAC 70(3), 2025 (docs/references/Generalized_Mean_Robustness_for_Signal_Temporal_Logic.pdf).
With $[x]_- = \\min(x, 0)$, $[x]_+ = \\max(x, 0)$ and $d$ the number of valid entries,

    conjunction (eq. 12, eq. 17):  $\\wedge(x) = F_c(x)$ if $\\min_i x_i > 0$, else $-F_g(-[x]_-)$,
    disjunction (De Morgan):       $\\vee(x) = -\\wedge(-x)$, so $\\vee(x) = 0$ when $\\max_i x_i = 0$,

with $F_c = M_p$ and $F_g = M_q$ for the power mean robustness of order $(p, q)$, where
$M_p(x) = (\\frac{1}{d} \\sum_i x_i^p)^{1/p}$ (eq. 3) and $M_0$ the geometric mean: gm_pm01 is
$(p, q) = (0, 1)$, gm_pm10 is $(p, q) = (-10, 10)$, and semantics.gm_power(p, q) gives any
order with $q \\ge 1$. For gm_exp $c(x) = -e^{-\\beta x}$, $g(x) = e^{\\beta x}$, so that
$F_c(x) = -\\frac{1}{\\beta} \\log \\frac{1}{d} \\sum_i e^{-\\beta x_i}$ and
$F_g(y) = \\frac{1}{\\beta} \\log \\frac{1}{d} \\sum_i e^{\\beta y_i}$. The formula recursion is eq. 15,
with the closed prefix of the Until nested:
$\\eta(\\varphi U_{[a,b]} \\psi, t) = \\vee_{k \\in [a,b]} \\wedge(\\eta(\\psi, t+k), \\wedge_{s \\in [t, t+k]} \\eta(\\varphi, s))$.
The matched gm_exp uses $\\beta_v = \\log(m_v) / \\epsilon$ at a node with $m_v > 1$ entries and
returns the entry itself at a node with one entry.

The oracles here are written from these equations only: conj_oracle and walk are
brute-force oracles (plain Python, one entry and one sample at a time, the values in
40-digit decimal arithmetic), and literal_conj is a literal JAX transcription without
numerical stabilization, used as the gradient reference.
"""
import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from sparsemax_diffstl import stl
from sparsemax_diffstl.core_study import methods
from sparsemax_diffstl.stl import semantics
from sparsemax_diffstl.stl.warp_backend import matched_param, robustness_warp

wp.config.log_level = wp.LOG_WARNING
ARMS = ("gm_pm01", "gm_pm10", "gm_exp")
ORDERS = {"gm_pm01": (0, 1), "gm_pm10": (-10, 10)}
BETAS = (10.0, math.log(500) / 0.4)
EPS = 0.3


def close(a, b, atol=1e-12, rtol=1e-10):
    """Entrywise |a - b| <= atol + rtol |b|. The float64 defaults: the reductions take exp and log
    of arguments up to beta max|x| (about 50 here), each of which commits a few units of
    roundoff 2.2e-16 relative to the argument, so absolute errors stay below 1e-13."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))


def same_sign(a, b):
    return bool(np.array_equal(np.sign(a), np.sign(b)))


# ------------------------------------------------------------------
# brute-force oracles from the equations

def power_mean_oracle(x, p):
    """Brute-force oracle of M_p (eq. 3) for decimals x >= 0; M_0 is the geometric mean."""
    d = Decimal(len(x))
    s = Decimal(0)
    if p == 0:
        for v in x:
            s += v.ln()
        return (s / d).exp()
    for v in x:
        s += v ** p
    return Decimal(0) if s == 0 else ((s / d).ln() / p).exp()


def conj_oracle(xs, arm, beta=None):
    """Brute-force oracle of the d-ary conjunction (eq. 12 for the power means, arm a name in
    ORDERS or a pair (p, q); eq. 17 with the exponential member for gm_exp), one entry at a
    time in 40-digit decimals."""
    with localcontext() as ctx:
        ctx.prec = 40
        x = [v if isinstance(v, Decimal) else Decimal(float(v)) for v in xs]
        d = Decimal(len(x))
        order = ORDERS.get(arm, arm) if arm != "gm_exp" else None
        if min(x) > 0:
            if order is not None:
                return power_mean_oracle(x, Decimal(order[0]))  # M_p(x)
            b = Decimal(float(beta))
            s = Decimal(0)
            for v in x:
                s += -(-b * v).exp()  # c(x_i)
            return -(-(s / d)).ln() / b  # c^{-1}(mean c(x)), c^{-1}(y) = -log(-y)/beta
        y = [-min(v, Decimal(0)) for v in x]  # -[x_i]_-
        if order is not None:
            return -power_mean_oracle(y, Decimal(order[1]))  # -M_q(-[x]_-)
        b = Decimal(float(beta))
        s = Decimal(0)
        for v in y:
            s += (b * v).exp()
        return -((s / d).ln() / b)  # -F_g(-[x]_-)


def disj_oracle(xs, arm, beta=None):
    """Brute-force oracle of the disjunction by De Morgan's law: disj(x) = -conj(-x)."""
    return -conj_oracle([-(v if isinstance(v, Decimal) else Decimal(float(v))) for v in xs], arm, beta)


def walk(f, t, leaf, red):
    """Brute-force oracle of eq. 15: the value of the NNF formula f at sample t, looping over the
    window entries, the Until's witnesses and its closed prefix. leaf(atom, t) gives an atom's
    value and red(values, kind) a conjunction ("min") or disjunction ("max") of a list."""
    if isinstance(f, stl.Atom):
        return leaf(f, t)
    if isinstance(f, stl.And):
        return red([walk(c, t, leaf, red) for c in f.children], "min")
    if isinstance(f, stl.Or):
        return red([walk(c, t, leaf, red) for c in f.children], "max")
    a, b = f.interval
    if isinstance(f, stl.Always):
        return red([walk(f.child, t + k, leaf, red) for k in range(a, b + 1)], "min")
    if isinstance(f, stl.Eventually):
        return red([walk(f.child, t + k, leaf, red) for k in range(a, b + 1)], "max")
    inner, outer = ("min", "max") if isinstance(f, stl.Until) else ("max", "min")  # Release is the dual
    vals = []
    for k in range(a, b + 1):
        prefix = red([walk(f.left, s, leaf, red) for s in range(t, t + k + 1)], inner)
        vals.append(red([walk(f.right, t + k, leaf, red), prefix], inner))
    return red(vals, outer)


def decimal_reduce(arm, eps=None, beta=None):
    """Reduction for walk: the matched parameter beta = log(m)/eps when eps is given, else the
    fixed beta; a node with one entry returns the entry (eq. 12 and 17 give the same value)."""
    def red(vals, kind):
        if len(vals) == 1:
            return vals[0]
        b = math.log(len(vals)) / eps if eps is not None else beta
        return conj_oracle(vals, arm, b) if kind == "min" else disj_oracle(vals, arm, b)
    return red


def oracle_trace(f, z, arm, eps=None, beta=None):
    """Brute-force oracle: eq. 15 at every root sample of every trace of z (B, T, P), as float64."""
    def leaf(g, t):
        v = Decimal(float(z[b, t, g.index]))
        return -v if g.negated else v

    red = decimal_reduce(arm, eps, beta)
    L = z.shape[1] - stl.horizon(f)
    out = np.zeros((z.shape[0], L))
    for b in range(z.shape[0]):  # traces, brute-force oracle
        for t in range(L):  # samples, brute-force oracle
            out[b, t] = float(walk(f, t, leaf, red))
    return out


# ------------------------------------------------------------------
# literal JAX transcription (gradient reference)

def literal_conj(x, arm, beta):
    """eq. 12 / eq. 17 written literally in JAX, without stabilization: M_0 as exp(mean(log x)),
    F_c as c^{-1}(mean c(x)), the branch by jnp.where with a safe input to the unused branch."""
    pos = jnp.min(x) > 0
    xs = jnp.where(pos, x, 1.0)
    neg = jnp.where(pos, 1.0, -jnp.minimum(x, 0.0))  # -[x]_-, safe when the branch is unused
    if arm != "gm_exp":
        p, q = ORDERS.get(arm, arm)
        a = jnp.exp(jnp.mean(jnp.log(xs))) if p == 0 else jnp.mean(xs ** p) ** (1 / p)
        b = -jnp.mean(neg ** q) ** (1 / q)
    else:
        a = -jnp.log(-jnp.mean(-jnp.exp(-beta * xs))) / beta
        b = -jnp.log(jnp.mean(jnp.exp(beta * neg))) / beta
    return jnp.where(pos, a, b)


def literal_disj(x, arm, beta):
    return -literal_conj(-x, arm, beta)


def literal_trace(f, z, arm, eps):
    """Literal transcription of eq. 15 for one trace z (T, P) with the matched parameters, built by
    the brute-force walk over windows and samples; returns the root trace."""
    def leaf(g, t):
        return -z[t, g.index] if g.negated else z[t, g.index]

    def red(vals, kind):
        if len(vals) == 1:
            return vals[0]
        x, b = jnp.stack(vals), math.log(len(vals)) / eps
        return literal_conj(x, arm, b) if kind == "min" else literal_disj(x, arm, b)

    return jnp.stack([walk(f, t, leaf, red) for t in range(z.shape[0] - stl.horizon(f))])  # samples, oracle


def reducer(arm, kind):
    """The semantics module's function for a fixed beta (the power means ignore beta); a pair
    (p, q) goes through semantics.gm_power."""
    if isinstance(arm, tuple):
        return semantics.gm_power(*arm)[1 if kind == "min" else 0]
    return {("gm_pm01", "min"): semantics.gm_pm01_min, ("gm_pm01", "max"): semantics.gm_pm01_max,
            ("gm_pm10", "min"): semantics.gm_pm10_min, ("gm_pm10", "max"): semantics.gm_pm10_max,
            ("gm_exp", "min"): semantics.gm_exp_min, ("gm_exp", "max"): semantics.gm_exp_max}[(arm, kind)]


def oracle(arm, kind, x, beta):
    return float(conj_oracle(x, arm, beta) if kind == "min" else disj_oracle(x, arm, beta))


# ------------------------------------------------------------------
# 1. hand vectors against the oracle

_rng = np.random.default_rng(0)
VECTORS = {
    "positive": [0.3, 1.2, 0.05, 2.0],
    "mixed": [0.4, -0.25, 1.1, -0.6, 0.02],
    "negative": [-0.3, -1.2, -0.05, -2.0],
    "tie": [0.2, 0.2, 0.7, 1.5],
    "tie_negative": [-0.4, -0.4, 0.3],
    "zero_min": [0.0, 0.5, 1.5],
    "zero_max": [0.0, -0.5, -1.5],
    "near_zero": [1e-6, -1e-6, 0.3],
    "near_zero_positive": [1e-6, 0.4, 0.9],
    "near_zero_negative": [-1e-6, -0.4, -0.9],
    "long": list(_rng.standard_normal(500)),
    "long_positive": list(np.abs(_rng.standard_normal(500)) + 0.01),
}


@pytest.mark.parametrize("name", list(VECTORS))
@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize("arm,beta", [("gm_pm01", None), ("gm_pm10", None), ((-2, 2), None), ("gm_exp", BETAS[0]),
                                       ("gm_exp", BETAS[1])])
def test_vectors_against_oracle(name, kind, arm, beta):
    x = VECTORS[name]
    ref = oracle(arm, kind, x, beta)
    with jax.enable_x64(True):
        v = float(reducer(arm, kind)(jnp.asarray(x, jnp.float64), beta))
    assert close(v, ref), (v, ref)
    assert same_sign(v, ref)


def test_zero_entries():
    # an entry exactly 0 with the others positive: conj = 0 (the else branch), disj > 0;
    # max exactly 0: disj = 0 by De Morgan
    with jax.enable_x64(True):
        for arm, beta in (("gm_pm01", None), ("gm_pm10", None), ((-2, 2), None), ("gm_exp", 10.0)):  # arms
            assert float(reducer(arm, "min")(jnp.asarray([0.0, 0.5, 1.5]), beta)) == 0.0
            assert float(reducer(arm, "max")(jnp.asarray([0.0, 0.5, 1.5]), beta)) > 0.0
            assert float(reducer(arm, "max")(jnp.asarray([0.0, -0.5, -1.5]), beta)) == 0.0
            assert float(reducer(arm, "min")(jnp.asarray([0.0, -0.5, -1.5]), beta)) < 0.0


def test_hand_values():
    # values written out by hand from eq. 12, 13 (De Morgan) and 17
    b = 10.0
    with jax.enable_x64(True):
        def v(arm, kind, x, beta=None):
            return float(reducer(arm, kind)(jnp.asarray(x, jnp.float64), beta))

        assert close(v("gm_pm01", "min", [0.5, 2.0]), 1.0)  # sqrt(0.5 * 2)
        assert close(v("gm_pm01", "min", [-0.3, 0.6, -0.1]), -0.4 / 3)  # (-0.3 + 0 - 0.1) / 3
        assert close(v("gm_pm01", "max", [-0.5, -2.0]), -1.0)  # -sqrt(0.5 * 2)
        assert close(v("gm_pm01", "max", [0.3, -0.6, 0.1]), 0.4 / 3)  # (0.3 + 0 + 0.1) / 3
        assert close(v("gm_pm01", "max", [0.0, 1.0, 2.0]), 1.0)  # (0 + 1 + 2) / 3
        assert close(v("gm_pm01", "min", [0.25, 1.0, 4.0]), 1.0)  # cube root of 1
        assert close(v("gm_exp", "min", [0.7, 0.7], b), 0.7)  # a mean of equal values
        # mixed: -(1/b) log((e^{0.3 b} + e^0) / 2)
        assert close(v("gm_exp", "min", [-0.3, 0.6], b), -math.log((math.exp(3.0) + 1.0) / 2) / b)
        # all positive: -(1/b) log((e^{-0.2 b} + e^{-0.5 b}) / 2)
        assert close(v("gm_exp", "min", [0.2, 0.5], b), -math.log((math.exp(-2.0) + math.exp(-5.0)) / 2) / b)
        # all negative disjunction: -F_c(-x) = (1/b) log((e^{-0.2 b} + e^{-0.5 b}) / 2)
        assert close(v("gm_exp", "max", [-0.2, -0.5], b), math.log((math.exp(-2.0) + math.exp(-5.0)) / 2) / b)
        # mixed disjunction: F_g([x]_+) = (1/b) log((e^{0.4 b} + e^0 + e^0) / 3)
        assert close(v("gm_exp", "max", [0.4, -0.3, -1.0], b), math.log((math.exp(4.0) + 2.0) / 3) / b)
        # (p, q) = (-10, 10): M_{-10}(2, 2, 2) = 2; mixed -((0.3^10 + 0 + 0.1^10) / 3)^{1/10}
        assert close(v("gm_pm10", "min", [2.0, 2.0, 2.0]), 2.0)
        assert close(v("gm_pm10", "min", [-0.3, 0.6, -0.1]), -((0.3 ** 10 + 0.1 ** 10) / 3) ** 0.1)
        # all negative disjunction: -M_{-10}(1, 2) = -((1 + 2^{-10}) / 2)^{-1/10}
        assert close(v("gm_pm10", "max", [-1.0, -2.0]), -((1 + 2.0 ** -10) / 2) ** -0.1)
        # (p, q) = (-2, 2): M_{-2}(1, 4) = ((1 + 1/16) / 2)^{-1/2} = sqrt(32/17); disjunction
        # of (0.3, -0.4, 0.4) is M_2(0.3, 0, 0.4) = sqrt((0.09 + 0 + 0.16) / 3)
        assert close(v((-2, 2), "min", [1.0, 4.0]), math.sqrt(32 / 17))
        assert close(v((-2, 2), "max", [0.3, -0.4, 0.4]), math.sqrt(0.25 / 3))


# ------------------------------------------------------------------
# 2. masks

PAD = np.array([[0.5, -0.2, 0.9, -100.0, 0.0],
                [0.3, 0.4, 1e3, -5.0, 0.0],
                [-0.1, -0.7, -0.2, 0.05, -3.0],
                [0.8, 0.0, -2.0, 7.0, 1e-9]])
COUNTS = np.array([3, 2, 4, 1])


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("kind", ["min", "max"])
def test_mask_equals_unpadded(arm, kind):
    beta = 10.0
    f = reducer(arm, kind)
    mask = np.arange(5)[None, :] < COUNTS[:, None]
    with jax.enable_x64(True):
        z = jnp.asarray(PAD)
        v = np.asarray(f(z, beta, mask))
        assert close(v[0], f(z[0, :3], beta))
        assert close(v[1], f(z[1, :2], beta))
        assert close(v[2], f(z[2, :4], beta))
        assert close(v[3], f(z[3, :1], beta))
        # gradients: zero on the padding, equal to the unpadded gradients on the valid entries
        g = np.asarray(jax.grad(lambda x: jnp.sum(f(x, beta, mask)))(z))
        assert np.all(np.isfinite(g)) and np.all(g[~mask] == 0.0)
        assert close(g[0, :3], jax.grad(lambda x: f(x, beta))(z[0, :3]))
        assert close(g[2, :4], jax.grad(lambda x: f(x, beta))(z[2, :4]))
        # a one-dimensional mask broadcasts over the rows
        m1 = np.arange(5) < 2
        assert close(f(z, beta, m1), f(z[:, :2], beta))


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("kind", ["min", "max"])
def test_mask_matched(arm, kind):
    # methods.SEMANTICS: beta = log(m)/eps per row from the row's valid count; one entry is itself
    pair = methods.SEMANTICS[arm]
    f = pair[1] if kind == "min" else pair[0]
    mask = np.arange(5)[None, :] < COUNTS[:, None]
    with jax.enable_x64(True):
        z = jnp.asarray(PAD)
        v = np.asarray(f(z, EPS, mask))
        assert close(v[0], oracle(arm, kind, PAD[0, :3], math.log(3) / EPS))
        assert close(v[1], oracle(arm, kind, PAD[1, :2], math.log(2) / EPS))
        assert close(v[2], oracle(arm, kind, PAD[2, :4], math.log(4) / EPS))
        assert close(v[3], PAD[3, 0])  # one valid entry: the entry itself
        assert close(f(z[0, :3], EPS), v[0])


# ------------------------------------------------------------------
# 3. float32

def test_float32_sign():
    """500 entries in float32: one violating entry -1e-6 among 499 entries at +0.05 keeps a
    negative conjunction; with that entry at +1e-6 the conjunction is positive.

    Tolerance against float64: relative 2e-4. The float32 unit roundoff is 1.2e-7. The
    largest float32 error here is the geometric mean's sum of 500 logarithms of magnitude
    about 3 (log 0.05): recursive summation commits at most 500 roundoffs of the running
    sum, an absolute error of at most 500 * 3 * 1.2e-7 = 1.8e-4 in the mean logarithm, which
    the exponential turns into the same relative error. The violating case's exact values are
    about -2e-9 (gm_pm01: -1e-6/500, the same order for gm_exp), so the tolerance also rules
    out a float32 evaluation whose cancellation leaves only noise. For gm_pm10 the violating
    value is -(1e-60 / 500)^{1/10}, about -5.4e-7, and the positive one about 1.9e-6: x^{10}
    underflows and x^{-10} overflows in float32 at x = 1e-6, so only a log-domain evaluation
    keeps the sign. There the logarithms have magnitude up to 10 |log 1e-6| = 138, each with
    an absolute error near 138 * 1.2e-7 = 1.7e-5, which division by 10 turns into a relative
    error near 2e-6 of the mean.
    """
    x = np.full(500, 0.05)
    x[123] = -1e-6
    xp = x.copy()
    xp[123] = 1e-6
    for arm, beta in (("gm_pm01", None), ("gm_pm10", None), ("gm_exp", BETAS[0]), ("gm_exp", BETAS[1])):  # cases
        conj, disj = reducer(arm, "min"), reducer(arm, "max")
        neg = conj(jnp.asarray(x, jnp.float32), beta)
        pos = conj(jnp.asarray(xp, jnp.float32), beta)
        assert neg.dtype == jnp.float32 and pos.dtype == jnp.float32
        assert float(neg) < 0.0, (arm, beta, float(neg))
        assert float(pos) > 0.0, (arm, beta, float(pos))
        # the disjunction of the negated vectors, by De Morgan
        assert float(disj(jnp.asarray(-x, jnp.float32), beta)) > 0.0
        assert float(disj(jnp.asarray(-xp, jnp.float32), beta)) < 0.0
        with jax.enable_x64(True):
            neg64 = float(conj(jnp.asarray(x, jnp.float64), beta))
            pos64 = float(conj(jnp.asarray(xp, jnp.float64), beta))
        assert close(neg64, oracle(arm, "min", x, beta)) and close(pos64, oracle(arm, "min", xp, beta))
        assert close(float(neg), neg64, 0.0, 2e-4), (arm, beta, float(neg), neg64)
        assert close(float(pos), pos64, 0.0, 2e-4), (arm, beta, float(pos), pos64)


def test_float32_pm10_range():
    """gm_pm10 in float32 on 500 entries spanning 1e-3 to 4 (x^{-10} up to 1e30, x^{10} up to
    1e6), all positive and with a third of them negated: finite, same sign as float64, within
    relative 1e-4 (logarithms of magnitude up to 10 |log 1e-3| = 69 with absolute error near
    69 * 1.2e-7 = 8e-6 each; divided by 10 that is under 1e-6 relative per term, and the 500-term
    log-sum-exp adds at most a few float32 roundoffs relative to its value)."""
    rng = np.random.default_rng(3)
    x = np.exp(rng.uniform(np.log(1e-3), np.log(4.0), 500))
    xm = x * np.where(np.arange(500) % 3 == 0, -1.0, 1.0)
    for z in (x, xm):  # two vectors
        for kind in ("min", "max"):  # conjunction and disjunction
            f = reducer("gm_pm10", kind)
            v = float(f(jnp.asarray(z, jnp.float32)))
            with jax.enable_x64(True):
                v64 = float(f(jnp.asarray(z, jnp.float64)))
            assert close(v64, oracle("gm_pm10", kind, z, None))
            assert np.isfinite(v) and same_sign(v, v64)
            assert close(v, v64, 0.0, 1e-4), (kind, v, v64)


# ------------------------------------------------------------------
# 4. sign equivalence with the exact robustness (Theorem 2)

A0, A1, A2 = stl.Atom(0), stl.Atom(1), stl.Atom(2)
FORMULAS = {
    "until": stl.Until((1, 3), A0, A1),
    "and_until_always": stl.And(stl.Until((0, 2), A0, A1), stl.Always((0, 3), A2), stl.Atom(1, True)),
    "or_until_eventually": stl.Or(stl.Until((1, 2), A0, A1), stl.Eventually((0, 3), A2)),
    "always_of_until": stl.Always((0, 2), stl.Until((0, 2), A0, stl.Or(A1, stl.Atom(2, True)))),
}


@pytest.mark.parametrize("name", list(FORMULAS))
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_sign_equivalence(name, arm, scale):
    f = FORMULAS[name]
    prog = stl.compile_formula(f, stl.horizon(f) + 4, boundary="strict")
    rng = np.random.default_rng(len(name))
    z = scale * rng.standard_normal((256, prog.T, 3))
    assert np.all(z != 0)
    with jax.enable_x64(True):
        rho = np.asarray(stl.robustness(prog, z))
        eta = np.asarray(stl.robustness(prog, z, methods.SEMANTICS[arm], EPS))
    assert eta.shape == rho.shape and np.all(np.isfinite(eta))
    assert np.all(rho != 0) and same_sign(eta, rho)
    assert np.any(rho > 0) and np.any(rho < 0)


# ------------------------------------------------------------------
# 5. the nested Until (eq. 15) against the brute-force oracle

@pytest.mark.parametrize("name", list(FORMULAS) + ["release"])
@pytest.mark.parametrize("arm", ARMS)
def test_formula_against_oracle(name, arm):
    f = FORMULAS.get(name, stl.Release((1, 2), A0, A1))
    prog = stl.compile_formula(f, stl.horizon(f) + 3, boundary="strict")
    z = np.random.default_rng(7).standard_normal((2, prog.T, 3))
    with jax.enable_x64(True):
        eta = np.asarray(stl.robustness(prog, z, methods.SEMANTICS[arm], EPS))
        fixed = np.asarray(stl.robustness(prog, z, arm, 10.0))  # REDUCTIONS by name, fixed beta
    ref = oracle_trace(f, z, arm, eps=EPS)
    assert close(eta, ref), np.max(np.abs(eta - ref))
    ref = oracle_trace(f, z, arm, beta=10.0)
    assert close(fixed, ref), np.max(np.abs(fixed - ref))


def test_until_is_nested():
    """Until([1, 1], phi, psi) at t = 0 over T = 2 samples is conj2(psi(1), conj(phi(0), phi(1))).
    With phi = (1, -0.5) and psi(1) = -1, gm_pm01 gives the prefix (0 - 0.5)/2 = -0.25 and the
    root (-1 - 0.25)/2 = -0.625; one flat conjunction of (psi(1), phi(0), phi(1)) would give
    (-1 + 0 - 0.5)/3 = -0.5."""
    f = stl.Until((1, 1), A0, A1)
    prog = stl.compile_formula(f, 2, boundary="strict")
    z = np.zeros((1, 2, 2))
    z[0, :, 0] = [1.0, -0.5]
    z[0, :, 1] = [0.7, -1.0]
    with jax.enable_x64(True):
        pm = float(stl.robustness(prog, z, methods.SEMANTICS["gm_pm01"], EPS)[0, 0])
        p10 = float(stl.robustness(prog, z, methods.SEMANTICS["gm_pm10"], EPS)[0, 0])
        ex = float(stl.robustness(prog, z, methods.SEMANTICS["gm_exp"], EPS)[0, 0])
    assert close(pm, -0.625)
    # gm_pm10: prefix -(0.5^10 / 2)^{1/10}, root -((1 + prefix^10) / 2)^{1/10}; flat -((1 + 0.5^10) / 3)^{1/10}
    pre = -(0.5 ** 10 / 2) ** 0.1
    nested, flat = -((1 + pre ** 10) / 2) ** 0.1, -((1 + 0.5 ** 10) / 3) ** 0.1
    assert abs(nested - flat) > 1e-2
    assert close(p10, nested), (p10, nested, flat)
    nested = float(conj_oracle([Decimal(-1), conj_oracle([1.0, -0.5], "gm_exp", math.log(2) / EPS)], "gm_exp",
                               math.log(2) / EPS))
    flat = float(conj_oracle([-1.0, 1.0, -0.5], "gm_exp", math.log(3) / EPS))
    assert abs(nested - flat) > 1e-2  # the case separates the two readings
    assert close(ex, nested), (ex, nested, flat)


# ------------------------------------------------------------------
# 6. gradients

def vector_program(kind, d):
    """Always[0, d-1] p0 (a conjunction of d entries) or Eventually[0, d-1] p0, over T = d."""
    f = stl.Always((0, d - 1), A0) if kind == "min" else stl.Eventually((0, d - 1), A0)
    return stl.compile_formula(f, d, boundary="strict")


def warp_vjp(prog, z, arm, param, seed):
    s = wp.array(z, dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    rho = robustness_warp(prog, s, arm, param, tape=tape)
    tape.backward(grads={rho: wp.array(seed, dtype=wp.float64, device="cpu")})
    return rho.numpy(), s.grad.numpy()


GRAD_VECTORS = {"mixed": [0.3, -0.2, 0.8, -0.05, 1.1], "positive": [0.3, 0.2, 0.8, 0.05, 1.1]}


@pytest.mark.parametrize("name", list(GRAD_VECTORS))
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("kind", ["min", "max"])
def test_vector_gradients(name, arm, kind):
    x = np.asarray(GRAD_VECTORS[name]) * (1 if kind == "min" else -1)  # the max reads the negated vector
    beta = 10.0
    lit = literal_conj if kind == "min" else literal_disj
    with jax.enable_x64(True):
        g_lit = np.asarray(jax.grad(lambda u: lit(u, arm, beta))(jnp.asarray(x)))
        g_jax = np.asarray(jax.grad(lambda u: reducer(arm, kind)(u, beta))(jnp.asarray(x)))
        v_lit = float(lit(jnp.asarray(x), arm, beta))
    assert close(v_lit, oracle(arm, kind, x, beta))
    assert close(g_jax, g_lit), (g_jax, g_lit)
    # Warp: the root of a single d-entry window; for gm_exp eps = log(d)/beta gives beta in the kernel
    d = len(x)
    prog = vector_program(kind, d)
    param = math.log(d) / beta if arm == "gm_exp" else 1.0
    v, g = warp_vjp(prog, x.reshape(1, d, 1), arm, param, np.ones((1, 1)))
    assert close(v[0, 0], v_lit)
    assert close(g[0, :, 0], g_lit), (g[0, :, 0], g_lit)


def test_finite_difference():
    # the minimal extra check: central differences, step 1e-6, float64, on one mixed vector.
    # Truncation error h^2 |f'''| / 6 with |f'''| about beta^2 = 100 is 2e-11; rounding
    # error about 2.2e-16 / 1e-6 = 2e-10; tolerance 1e-8.
    x = np.asarray([0.3, -0.2, 0.8, -0.05, 1.1])
    h = 1e-6
    with jax.enable_x64(True):
        f = semantics.gm_exp_min
        E = h * jnp.eye(5)
        fd = (f(jnp.asarray(x) + E, 10.0) - f(jnp.asarray(x) - E, 10.0)) / (2 * h)
        g = jax.grad(lambda u: f(u, 10.0))(jnp.asarray(x))
    assert close(g, fd, 1e-8, 0.0), (g, fd)


@pytest.mark.parametrize("arm", ARMS)
def test_formula_gradients(arm):
    f = stl.Or(stl.Until((1, 3), A0, A1), stl.Always((0, 2), A2))
    prog = stl.compile_formula(f, stl.horizon(f) + 3, boundary="strict")
    rng = np.random.default_rng(8)
    z = np.sign(rng.standard_normal((3, prog.T, 3))) * rng.uniform(0.1, 1.0, (3, prog.T, 3))
    seed = rng.standard_normal((3, prog.steps[-1].length))
    with jax.enable_x64(True):
        lit = jax.vmap(lambda u: literal_trace(f, u, arm, EPS))
        v_lit, vjp = jax.vjp(lit, jnp.asarray(z))
        g_lit = np.asarray(vjp(jnp.asarray(seed))[0])
        v_jax, vjp = jax.vjp(lambda u: stl.robustness(prog, u, methods.SEMANTICS[arm], EPS), jnp.asarray(z))
        g_jax = np.asarray(vjp(jnp.asarray(seed))[0])
    assert close(v_lit, oracle_trace(f, z, arm, eps=EPS))
    assert close(v_jax, v_lit)
    assert close(g_jax, g_lit), np.max(np.abs(g_jax - g_lit))
    v, g = warp_vjp(prog, z, arm, matched_param(prog, arm, EPS), seed)
    assert close(v, v_lit)
    assert close(g, g_lit), np.max(np.abs(g - g_lit))


# ------------------------------------------------------------------
# 7. JAX against Warp

F7 = stl.And(stl.Always((0, 2), stl.Eventually((0, 1), A2)), stl.Until((1, 3), A0, A1))


def f7_case(seed, B):
    prog = stl.compile_formula(F7, stl.horizon(F7) + 3, boundary="strict")
    rng = np.random.default_rng(seed)
    # magnitudes in [0.1, 1] keep every node value away from the branch point 0
    z = np.sign(rng.standard_normal((B, prog.T, 3))) * rng.uniform(0.1, 1.0, (B, prog.T, 3))
    return prog, z, rng.standard_normal((B, prog.steps[-1].length))


@pytest.mark.parametrize("arm", ARMS)
def test_jax_warp_cpu(arm):
    prog, z, seed = f7_case(9, 4)
    with jax.enable_x64(True):
        j, vjp = jax.vjp(lambda u: stl.robustness(prog, u, methods.SEMANTICS[arm], EPS), jnp.asarray(z))
        gj = np.asarray(vjp(jnp.asarray(seed))[0])
    scalar = EPS if arm == "gm_exp" else 1.0
    for param in (matched_param(prog, arm, EPS), scalar):  # per-row and scalar parameters
        v, g = warp_vjp(prog, z, arm, param, seed)
        assert close(v, j, 1e-10, 0.0), np.max(np.abs(v - j))
        assert close(g, gj, 1e-10, 0.0), np.max(np.abs(g - gj))
    assert close(j, oracle_trace(F7, z, arm, eps=EPS))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs a CUDA device for Warp")
@pytest.mark.parametrize("arm", ARMS)
def test_jax_warp_cuda_float32(arm):
    """float32 Warp on cuda:0 against float64 JAX: 1024 float32 roundoffs (1.2e-4) relative to
    max(1, largest reference entry), the bound tests/test_stl_warp.py uses for exp and log from
    different libraries and different summation orders; the float64 reference adds nothing."""
    tol = 1024 * np.finfo(np.float32).eps
    prog, z, seed = f7_case(10, 8)
    with jax.enable_x64(True):
        j, vjp = jax.vjp(lambda u: stl.robustness(prog, u, methods.SEMANTICS[arm], EPS), jnp.asarray(z))
        j, gj = np.asarray(j), np.asarray(vjp(jnp.asarray(seed))[0])
    s = wp.array(z.astype(np.float32), dtype=wp.float32, device="cuda:0", requires_grad=True)
    tape = wp.Tape()
    rho = robustness_warp(prog, s, arm, matched_param(prog, arm, EPS), tape=tape)
    tape.backward(grads={rho: wp.array(seed.astype(np.float32), dtype=wp.float32, device="cuda:0")})
    v, g = rho.numpy(), s.grad.numpy()
    assert same_sign(v, j)
    assert close(v, j, tol * max(1.0, np.max(np.abs(j))), 0.0), np.max(np.abs(v - j))
    assert close(g, gj, tol * max(1.0, np.max(np.abs(gj))), 0.0), np.max(np.abs(g - gj))

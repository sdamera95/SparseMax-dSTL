"""Progression of a specification at a margin over a recorded prefix against the exact oracle, the incremental state against
the full recomputation, and the recount and matched log-sum-exp operators. CPU, float64."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparsemax_dstl import stl
from sparsemax_dstl.jax import evaluator as ev
from sparsemax_dstl.jax import operators as op
from sparsemax_dstl.jax import progression as jpg
from sparsemax_dstl.stl import oracle
from sparsemax_dstl.stl import progression as pg


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def cases(seed, n, T=12):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:  # random instances (a brute-force test generator)
        f = oracle.random_formula(rng, 3, int(rng.integers(1, 4)), max_bound=3)
        if stl.horizon(stl.to_nnf(f)) > T - 1 or stl.compile_formula(f, T).steps[-1].kind == "atom":
            continue
        out.append((f, rng.normal(size=(T, 3)), rng.normal(size=(T, 3)), rng))
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_residual_verdict_equals_oracle(seed):
    """Open root rows: the exact residual value is at least delta exactly when the oracle's robustness of the joined trace
    is; decided rows: ok is that verdict. The smoothed residual values are at most the exact one."""
    checked = 0
    for f, rec, pred, rng in cases(seed, 30):  # random instances (brute-force oracle test)
        T = rec.shape[0]
        prog = stl.compile_formula(f, T)
        nnf = stl.to_nnf(f)
        ks = np.arange(-1, T)
        joined = np.where(np.arange(T)[None, :, None] <= ks[:, None, None], rec[None], pred[None])
        delta = float(rng.normal(scale=0.5))
        dec = pg.decisions(prog, rec, delta)
        res = pg.residual(prog, dec, ks)
        masks = pg.evaluation_masks(prog, res)
        exact = np.asarray(jpg.evaluate(prog, joined, masks, "exact")[-1])
        smooth = np.asarray(jpg.evaluate(prog, joined, masks, "sparsemax", 0.5)[-1])
        lse = np.asarray(jpg.evaluate(prog, joined, masks, "lse_matched", 0.5)[-1])
        is_open, (d, ok) = res[-1][0], dec[-1]
        truth = np.array([[oracle.rho(nnf, joined[i], t) >= delta for t in range(d.size)] for i in range(ks.size)])  # oracle
        assert np.array_equal((exact >= delta)[is_open], truth[is_open])
        assert np.all(smooth[is_open] <= exact[is_open] + 1e-12) and np.all(lse[is_open] <= exact[is_open] + 1e-12)
        decided = ~is_open
        assert np.all(np.broadcast_to(d, decided.shape)[decided] <= np.broadcast_to(ks[:, None], decided.shape)[decided])
        assert np.array_equal(np.broadcast_to(ok, decided.shape)[decided], truth[decided])
        checked += int(is_open.sum())
    assert checked > 100


def brute_reach(prog, root, row, follow):
    """Rows reached from one root row, by a recursive walk over the followed entries (a brute-force test oracle)."""
    out = [np.zeros(s.length, bool) for s in prog.steps]

    def walk(i, r):
        out[i][r] = True
        s = prog.steps[i]
        if s.kind == "atom":
            return
        offs = np.cumsum([0] + [prog.steps[k].length for k in s.sources])
        for w in np.flatnonzero(follow[i][r]):
            pos = s.index[r, w]
            j = int(np.searchsorted(offs, pos, side="right")) - 1
            walk(s.sources[j], pos - offs[j])

    walk(root, row)
    return out


def test_reach_and_leaves():
    """The batched reach equals a recursive walk; every leaf an open root reaches through the residual lies at a predicted
    sample, and every open row keeps an operand."""
    for f, rec, pred, rng in cases(7, 20):  # random instances (brute-force oracle test)
        prog = stl.compile_formula(f, rec.shape[0])
        k = int(rng.integers(0, rec.shape[0] - 1))
        res = pg.residual(prog, pg.decisions(prog, rec, 0.1), k)
        for r in res:
            if r is not None:
                assert np.all(r[1].sum(-1)[r[0]] >= 1)
        roots = np.flatnonzero(res[-1][0])
        if roots.size == 0:
            continue
        follow = [None if r is None else r[1] for r in res]
        reached = pg.reach(prog, prog.root, roots[None], [None if m is None else m[None] for m in follow])
        for j, t in enumerate(roots):  # roots, brute-force comparison
            ref = brute_reach(prog, prog.root, t, follow)
            for i, (st, m) in enumerate(zip(prog.steps, reached)):
                assert np.array_equal(np.zeros(st.length, bool) if m is None else m[0, j], ref[i])
                if st.kind == "atom" and ref[i].any():
                    assert np.all(np.flatnonzero(ref[i]) > k)


def test_incremental_state_equals_recomputation():
    """Progressions updated one sample at a time (a copy taken halfway) equals decisions() and residual() on the record with
    the later samples NaN, for two members of one program with different columns and margins."""
    for f, rec, pred, rng in cases(11, 10, T=14):  # random instances
        T = rec.shape[0]
        prog = stl.compile_formula(f, T)
        cols = [np.arange(3), np.array([2, 0, 1])]
        delta = np.array([0.1, -0.2])
        state = pg.Progressions([prog, prog], cols, delta)
        for t in range(T):  # recorded samples, one update each
            if t == T // 2:
                state = state.copy()
            state.update(rec[t])
            for m, s in enumerate(state.states(t)):
                full = np.where(np.arange(T)[:, None] <= t, rec[:, cols[m]], np.nan)
                dec = pg.decisions(prog, full, delta[m])
                res = pg.residual(prog, dec, t)
                for (d, ok), (d2, ok2) in zip(s["dec"], dec):
                    assert np.array_equal(d, d2) and np.array_equal(ok, ok2)
                for r, r2 in zip(s["residual"], res):
                    assert (r is None) == (r2 is None)
                    if r is not None:
                        assert np.array_equal(r[0], r2[0]) and np.array_equal(r[1], r2[1])


def test_masking_is_rebuilding():
    """A masked lower maximum is the node rebuilt over its remaining operands: the offset gamma/(2m) counts the mask."""
    z = np.array([0.3, 0.0])
    assert abs(float(op.lower_max(z, 1.0)) - 0.1725) < 1e-12
    assert abs(float(op.lower_max(z, 1.0, np.array([True, False]))) - 0.3) < 1e-12
    z = np.array([0.0, -5.0, -5.0])
    gain = float(op.lower_max(z, 1.0, np.array([True, True, False]))) - float(op.lower_max(z, 1.0))
    assert abs(gain - 0.5 * (1 / 2 - 1 / 3)) < 1e-12


def inputs(seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(500, 9)) * rng.choice([0.01, 0.3, 3.0], size=(500, 1))
    Z[:60, 3] = Z[:60, 1]
    M = rng.random((500, 9)) < 0.7
    M[:, 0] = True
    M[:25, 1:] = False
    return Z, M, rng.normal(size=Z.shape)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_recount_equals_sort(kind):
    """The recount's threshold gives the sort's values, derivatives and weights."""
    Z, M, dZ = inputs(3)
    recount, sort = (op.lower_max_recount, op.lower_max) if kind == "max" else (op.lower_min_recount, op.lower_min)
    a, ta = jax.jvp(lambda z: recount(z, 0.4, M), (jnp.asarray(Z),), (jnp.asarray(dZ),))
    b, tb = jax.jvp(lambda z: sort(z, 0.4, M), (jnp.asarray(Z),), (jnp.asarray(dZ),))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ta, tb, rtol=0, atol=1e-12)
    v, w = op.extremum_and_weights(jnp.asarray(Z), 0.4, M, kind)
    np.testing.assert_allclose(v, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w, op.sparsemax_weights(Z if kind == "max" else -Z, 0.4, M), rtol=0, atol=1e-12)


def test_matched_log_sum_exp_has_the_sparsemax_error_bound():
    """max z - gamma/2 (1 - 1/m) <= lse_max_matched <= max z over m valid entries, likewise for the minimum, both exact over
    one entry; the "lse_matched" reductions are these functions."""
    Z, M, _ = inputs(5)
    gamma = 0.3
    m = M.sum(-1)
    band = gamma / 2 * (1 - 1 / m)
    hi = np.max(np.where(M, Z, -np.inf), -1)
    lo = np.min(np.where(M, Z, np.inf), -1)
    up = np.asarray(ev.lse_max_matched(jnp.asarray(Z), gamma, M))
    dn = np.asarray(ev.lse_min_matched(jnp.asarray(Z), gamma, M))
    assert np.all(up <= hi + 1e-12) and np.all(up >= hi - band - 1e-12)
    assert np.all(dn <= lo + 1e-12) and np.all(dn >= lo - band - 1e-12)
    assert np.array_equal(up[m == 1], hi[m == 1]) and np.array_equal(dn[m == 1], lo[m == 1])
    assert ev.REDUCTIONS["lse_matched"] == (ev.lse_max_matched, ev.lse_min_matched)
    assert abs(ev.matched_beta(4, gamma) * gamma / 2 * (1 - 1 / 4) - np.log(4)) < 1e-12

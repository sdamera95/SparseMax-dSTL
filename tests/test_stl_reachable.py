"""E020: reachable-row compilation. Pruned programs against unpruned ones."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparsemax_dstl import jax as stl_jax
from sparsemax_dstl import stl
from sparsemax_dstl.jax import methods, read, robustness
from sparsemax_dstl.jax.baselines import DGMSR, GILPIN
from sparsemax_dstl.stl import Always, And, Atom, Eventually, Or, Release, Until, compile_formula, oracle
from sparsemax_dstl.stl.program import locate

SEMANTICS = [("exact", None), ("sparsemax", 0.5), ("lse", 4.0), (methods.SEMANTICS["sparsemax"], 0.2),
             (methods.SEMANTICS["lse"], 0.2), (GILPIN, (5.0, 5.0)), (DGMSR, (0.05, 2.0))]
# Pruned and unpruned rows hold the same entries, so they agree to rounding in either precision.
TOL = 1e-10 if jax.config.jax_enable_x64 else 1e-4


def entries(program):
    return int(sum(s.count.sum() for s in program.steps if s.kind != "atom"))


def random_reads(rng, full):
    """1 to 3 random reads of subformulas the program holds, some of them witness rows."""
    nodes = list(full.nodes)
    reads = []
    for _ in range(int(rng.integers(1, 4))):  # reads
        g = nodes[int(rng.integers(len(nodes)))]
        n = full.steps[full.nodes[g]].length
        t = np.sort(rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False))
        if g in full.inner and rng.random() < 0.5:
            a, b = g.interval
            reads.append((g, t, rng.integers(a, b + 1, t.size)))
        else:
            reads.append((g, t))
    return reads


def unpruned_reads(full, reads):
    locs = [locate(full, *r) for r in reads]  # reads
    return lambda vals: jnp.concatenate([vals[s][..., rows] for s, rows in locs], -1)


def test_root_reads_equal_unpruned_rows_and_shrink():
    f = And(Always((0, 6), Atom(0)), Until((2, 5), Atom(1), Atom(2)), Eventually((1, 3), Always((0, 2), Atom(2))))
    T = 20
    z = np.random.default_rng(0).normal(size=(T, 3))
    full = compile_formula(f, T)
    pr = compile_formula(f, T, reads=[(f, [0])])
    assert entries(pr) < entries(full)
    for sem, param in SEMANTICS:  # semantics
        a = np.asarray(robustness(full, z, sem, param))[0]
        b = np.asarray(robustness(pr, z, sem, param))
        assert b.shape == (1,)
        np.testing.assert_allclose(b[0], a, rtol=TOL, atol=TOL)


def derivatives(F):
    """Jitted value, gradient (reverse mode) and Hessian (forward over reverse) of F."""
    return jax.jit(lambda z: (F(z), jax.jacrev(F)(z), jax.jacfwd(jax.jacrev(F))(z)))


@pytest.mark.parametrize("boundary", ["strict", "clip"])
def test_random_reads_values_and_derivatives(boundary):
    # A few formulas and four semantics here; the gate script runs every semantics on more.
    rng = np.random.default_rng(11 if boundary == "strict" else 12)
    sems = [SEMANTICS[i] for i in (0, 1, 4, 6)] if boundary == "strict" else SEMANTICS[:1]
    for _ in range(4):  # random formulas
        f = oracle.random_formula(rng, 3, int(rng.integers(1, 4)))
        T = stl.horizon(f) + int(rng.integers(1, 4))
        full = compile_formula(f, T, boundary)
        reads = random_reads(rng, full)
        pr = compile_formula(f, T, boundary, reads)
        gather = unpruned_reads(full, reads)
        z = jnp.asarray(rng.normal(size=(T, 3)))
        for sem, param in sems:  # semantics
            F = lambda z: gather(stl_jax.evaluate(full, z, sem, param))
            G = lambda z: read(pr, stl_jax.evaluate(pr, z, sem, param))
            if boundary == "clip":
                np.testing.assert_array_equal(np.asarray(G(z)), np.asarray(F(z)))
                continue
            for a, b in zip(derivatives(F)(z), derivatives(G)(z)):  # value, gradient, Hessian
                np.testing.assert_allclose(b, a, rtol=TOL, atol=TOL)


def test_clip_empty_windows_keep_padding_rows():
    # At t = 5 of T = 6, G_[2,3] and the Until's witnesses read nothing; the pruned program
    # still gives the identities of the extrema there, and F_[0,1] keeps one padding row.
    f = Or(Always((2, 3), Eventually((0, 1), Atom(0))), Until((1, 2), Atom(1), Atom(2)))
    z = np.random.default_rng(3).normal(size=(6, 3))
    full = compile_formula(f, 6, "clip")
    pr = compile_formula(f, 6, "clip", reads=[(f, [5]), (f.children[0], [5]), (f.children[1], [5], [2])])
    got = np.asarray(read(pr, stl_jax.evaluate(pr, z)))
    np.testing.assert_array_equal(got, [np.inf, np.inf, np.inf])
    assert pr.steps[pr.nodes[f.children[0].child]].length == 1
    np.testing.assert_array_equal(np.asarray(robustness(full, z))[5], np.inf)


def test_witness_reads_and_locate():
    f = Release((1, 3), Atom(0), Or(Atom(1), Atom(2)))
    T = 12
    z = np.random.default_rng(4).normal(size=(T, 3)).astype(np.float32)  # exact in both precisions
    t, k = np.array([0, 2, 5]), np.array([1, 3, 2])
    pr = compile_formula(f, T, reads=[(f, t, k)])
    assert f not in pr.nodes and f in pr.inner  # only the inner step is built
    got = np.asarray(read(pr, stl_jax.evaluate(pr, z)))
    ref = [max(max(z[tt + kk, 1], z[tt + kk, 2]), max(z[tt:tt + kk + 1, 0])) for tt, kk in zip(t, k)]  # oracle
    np.testing.assert_array_equal(got, ref)
    full = compile_formula(f, T)
    s, rows = locate(full, f, t, k)
    np.testing.assert_array_equal(np.asarray(stl_jax.evaluate(full, z)[s])[rows], ref)


def test_reads_are_checked():
    f = Always((0, 3), Atom(0))
    with pytest.raises(ValueError):
        compile_formula(f, 5, reads=[(f, [2])])  # G_[0,3] exists only at t < 2
    with pytest.raises(ValueError):
        compile_formula(f, 5, reads=[(Atom(1), [0])])  # not a subformula
    with pytest.raises(ValueError):
        compile_formula(f, 5, reads=[(f, [0], [1])])  # witnesses need an Until or Release
    pr = compile_formula(f, 5, reads=[(f, [0])])
    with pytest.raises(ValueError):
        locate(pr, f, [1])  # pruned away

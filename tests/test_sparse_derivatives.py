"""The specification-prescribed patterns, the supports' structures and the structural Jacobian and Hessian of one control
period's program: against stored values of the reference implementation on the five-cell zone at 100 time steps, and
against dense automatic differentiation of the same program at 20 time steps."""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("asdex")
pytest.importorskip("scipy")
from sparsemax_dstl.mpc import conjuncts as MC  # noqa: E402
from sparsemax_dstl.mpc import program as MP  # noqa: E402
from sparsemax_dstl.sparse import assembly as SA  # noqa: E402
from sparsemax_dstl.sparse import coloring as SC  # noqa: E402
from sparsemax_dstl.sparse import patterns as SP  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"


def program(T, k):
    with jax.enable_x64(True):
        sc = S.five_cell_zone(T)
        conjs = MC.conjuncts(sc, MC.record_predicates(sc, sc.states))
        return MP.PeriodProgram(sc, conjs, k)


@pytest.fixture(scope="module")
def zone():
    f = np.load(DATA / "sparse_five_cell_zone_T100.npz")
    return f, program(100, int(np.load(DATA / "mpc_five_cell_zone_T100.npz")["period"]))


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)), initial=0.0))


def test_prescribed_patterns(zone):
    f, pb = zone
    J, H = SP.prescribed(pb)
    assert np.array_equal(J[0], f["prescribed_jacobian_rows"]) and np.array_equal(J[1], f["prescribed_jacobian_cols"])
    assert H[0].size == int(f["prescribed_hessian_upper"])
    assert hashlib.sha256(H[0].astype(np.int64).tobytes() + H[1].astype(np.int64).tobytes()).hexdigest() == str(f["prescribed_hessian_sha256"])


def test_supports_derivatives_at_the_start_point(zone):
    """The supports' structures declared at the start point and the Jacobian and Hessian (sigma 1, multipliers 1) on them."""
    f, pb = zone
    with jax.enable_x64(True):
        w = jnp.asarray(pb.reference())
        rt = SA.StructuralDerivatives(pb)
        rt.declare(np.asarray(rt.prefix(w)), jnp.float64)
        J, jesc = rt.jacobian(w)
        H, hesc = rt.hessian(w, 1.0, jnp.ones(pb.m))
    assert not jesc and not hesc
    assert np.array_equal(rt.jac_structure[0], f["jacobian_rows"]) and np.array_equal(rt.jac_structure[1], f["jacobian_cols"])
    assert np.array_equal(rt.hess_structure[0], f["hessian_rows"]) and np.array_equal(rt.hess_structure[1], f["hessian_cols"])
    assert rel(J, f["jacobian_values"]) <= 1e-12 and rel(H, f["hessian_values"]) <= 1e-12


def test_structural_derivatives_equal_dense_ad():
    """At 20 time steps: the structural Jacobian and Hessian equal dense AD on the declared structures, the prescribed
    structures hold every entry dense AD finds, and dense AD is zero off the supports' structures."""
    pb = program(20, 100)
    assert pb.n_rows > 0
    with jax.enable_x64(True):
        rng = np.random.default_rng(0)
        w = jnp.asarray(pb.reference() + 1e-3 * rng.normal(size=pb.n))
        lam = jnp.asarray(rng.uniform(0.5, 1.5, pb.m))
        rt = SA.StructuralDerivatives(pb)
        rt.declare(np.asarray(rt.prefix(w)), jnp.float64)
        J, _ = rt.jacobian(w)
        H, _ = rt.hessian(w, 0.7, lam)
        dj, dh = SC.dense_derivatives(pb)
        Jd, Hd = np.asarray(dj(w)), np.asarray(dh(w, jnp.asarray(0.7), lam))
    jr, jc = rt.jac_structure
    hr, hc = rt.hess_structure
    assert rel(J, Jd[jr, jc]) <= 1e-10 and rel(H, Hd[hr, hc]) <= 1e-10
    off_j, off_h = Jd.copy(), np.triu(Hd)
    off_j[jr, jc] = 0.0
    off_h[hr, hc] = 0.0
    assert np.abs(off_j).max() == 0.0 and np.abs(off_h).max() == 0.0
    Jp, Hp = SP.prescribed(pb)
    on_j, on_h = np.zeros(Jd.shape, bool), np.zeros(Hd.shape, bool)
    on_j[Jp] = True
    on_h[Hp] = True
    assert not np.any((Jd != 0) & ~on_j) and not np.any((np.triu(Hd) != 0) & ~on_h)

"""The product counts of the five-cell zone at horizon 100 against stored counts (tests/data): the per-period Jacobian and
Hessian counts of the prescribed and supports patterns at periods 100 and 101, the persistence of the supports between
them, the figure period's patterns, and the summary lines of the report."""
from pathlib import Path

import jax
import numpy as np
import pytest

pytest.importorskip("scipy")
from sparsemax_dstl.mpc import conjuncts as MC  # noqa: E402
from sparsemax_dstl.sparse import counts as SC  # noqa: E402
from sparsemax_dstl.tasks import dual_arm as D  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

DATA = Path(__file__).parent / "data"
TITLE = "five-cell zone, horizon 100"


@pytest.fixture(scope="module")
def stored():
    return dict(np.load(DATA / "sparse_counts_five_cell_zone_T100.npz"))


@pytest.fixture(scope="module")
def counted():
    sc = S.five_cell_zone(100)
    conjs = MC.conjuncts(sc, MC.record_predicates(sc, sc.states, D.EPS_LENGTH))
    with jax.enable_x64(True):
        return SC.count(sc, conjs, np.array([100, 101]), np.array([100]), figure_period=100)


def test_counts_per_period(stored, counted):
    A, meta = counted
    assert meta["conjuncts"] == {"until": 15, "reach": 42}
    assert meta["record_samples"] == 998
    for k, v in stored.items():  # stored arrays
        if k.startswith(("figure_", "report_")):
            continue
        if np.issubdtype(v.dtype, np.floating):
            np.testing.assert_array_equal(np.isnan(A[k]), np.isnan(v), err_msg=k)
            np.testing.assert_allclose(A[k], v, rtol=1e-12, atol=0, err_msg=k)
        else:
            np.testing.assert_array_equal(A[k], v, err_msg=k)
    assert np.array_equal(A["whole_dense_products"], np.minimum(D.NX * 100 + A["all_rows"], meta["nv"]))


def test_figure_patterns(stored, counted):
    A, _ = counted
    assert int(A["figure_period"]) == int(stored["figure_period"])
    assert np.array_equal(A["figure_row_kind"], stored["figure_row_kind"])
    for lev in ("prescribed", "supports"):  # the two patterns
        assert np.array_equal(A["figure_jacobian_" + lev + "_rows"], stored["figure_jacobian_" + lev + "_rows"])
        assert np.array_equal(A["figure_jacobian_" + lev + "_cols"], stored["figure_jacobian_" + lev + "_cols"])
        assert A["figure_hessian_" + lev + "_rows"].size == int(stored["figure_hessian_" + lev + "_entries"])
    assert np.array_equal(A["figure_hessian_supports_rows"], stored["figure_hessian_supports_rows"])
    assert np.array_equal(A["figure_hessian_supports_cols"], stored["figure_hessian_supports_cols"])


def test_report_lines(stored):
    meta = {"T": 100, "record_samples": 998, "n": D.NX * 100, "nv": (D.NX + D.NU) * 100, "fams": ["until", "reach"],
            "conjuncts": {"until": 15, "reach": 42}, "star_colorings": int(stored["report_meta_star_colorings"]),
            "star_cache_hits": int(stored["report_meta_star_reused"])}
    lines = SC.report(stored, meta, TITLE, 999)
    assert lines == stored["report_lines"].tolist()

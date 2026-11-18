# One solve on the five-cell zone at any horizon, beside the paper's (results/solves.csv, results/further_solves.csv):
# ours on the supports' structure (U), widened by 2 or 5 samples (D2, D5) or prescribed (P); Sound-LSE (lse) or GMR (gmr)
# in CasADi with L-BFGS; from the previous control period's solution or from the reference motion. Solved twice, the second
# timed. At 400 time steps about 2 minutes each.
#   python scripts/solves.py <T> <U|D2|D5|P|lse|gmr> <previous|reference>
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # MUMPS and its BLAS on one thread, as the paper's solves ran
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from common import build, casadi_solve, ours_solve, paper_rows, timed_period  # noqa: E402
from sparsemax_dstl.sparse import patterns as SP  # noqa: E402
from sparsemax_dstl.tasks import dual_arm_scenes as S  # noqa: E402

T, which, start = int(sys.argv[1]), sys.argv[2], sys.argv[3]
scene, conjs = build(S.five_cell_zone(T))
k = timed_period(scene, conjs)
if which in ("lse", "gmr"):
    out = casadi_solve(scene, conjs, k, which, start, twice=True)
    m = dict(comparator="Sound-LSE" if which == "lse" else "GMR", variant="L-BFGS")
else:
    arm, d = which[0], int(which[1:] or 0)
    out = ours_solve(scene, conjs, k, arm, d, start, SP.Reference(scene, conjs), twice=True)
    variant = {"U": "supports' structure", "D2": "structure widened by 2 samples", "D5": "structure widened by 5 samples", "P": "prescribed structure"}[which]
    m = dict(comparator="SparseMax (ours)", variant=variant)
m.update(scene="five-cell zone", start="previous solution" if start == "previous" else "reference motion", horizon=T)
print("T", T, "control period", k, "|", out["status"], out["updates"], "updates, objective", out["objective"], ",", round(out["seconds"], 3), "s")
rows = paper_rows("further_solves", **m)
print("paper:", ", ".join(r["quantity"] + " " + r["value"] for r in rows))

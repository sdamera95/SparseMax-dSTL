# One loop of the re-planning table at any horizon (a hold-up of 14 samples from sample 113 at 100 time steps, of 16 samples
# from sample 100 otherwise), ours or Sound-LSE, a new plan every P samples, beside the paper's (results/replanning.csv).
# At 400 time steps (8 s) about 6.5 minutes.
#   python scripts/loops.py <T> <P> <ours|lse>
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # MUMPS and its BLAS on one thread, as the paper's solves ran
import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from common import lse_loop, ours_loop, paper_rows  # noqa: E402

T, P, side = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
delay, k0 = (14, 113) if T == 100 else (16, 100)
every = ", a new plan every " + str(P) + " samples"
if side == "ours":
    out, _ = ours_loop(delay, k0, T, 10, P)
    m = dict(comparator="SparseMax (ours)", variant="supports' structure, one program per horizon" + every)
else:
    out = lse_loop(delay, k0, T, 10, P)
    m = dict(comparator="Sound-LSE", variant="L-BFGS" + every)
print("T", T, "P", P, side, "| updates", [s["updates"] for s in out[1:]], "| median seconds", round(float(np.median([s["seconds"] for s in out[1:]])), 3))
for r in paper_rows("replanning", horizon=T, **m):
    print("paper", r["quantity"], r["value"], r["values"])

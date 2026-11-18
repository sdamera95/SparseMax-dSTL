# The medians of the AD products over the counts table's control periods at any horizon, beside the paper's
# (results/counts.csv). At 400 time steps about 10 minutes on 4 CPUs.
#   python scripts/counts.py <five-cell zone|64-cell circle> <T>
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")  # MUMPS and its BLAS on one thread, as the paper's solves ran
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from common import counts  # noqa: E402

counts(sys.argv[1], int(sys.argv[2]))

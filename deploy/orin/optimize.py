# The manipulator optimization of Section V-B of the paper on the GPU: eight violating initial trajectories at
# H = 10 s under SparseMax at eps = 0.2, one constraint per conjunct (Equation (4)). Prints the exact robustness of
# the eight trajectories before the first update and after each one, from a float64 replay with MuJoCo, and the
# median seconds per update by component.
#     python deploy/orin/optimize.py [updates]      (a multiple of 10; default 10)
import sys

import numpy as np
import warp as wp

from sparsemax_dstl.tasks import workspace_scene as scene
from sparsemax_dstl.tasks import workspace_warp as manipulator
from sparsemax_dstl.warp import solver_conjuncts

wp.config.log_level = wp.LOG_WARNING
updates = int(sys.argv[1]) if len(sys.argv) > 1 else 10
R = scene.runs(scene.load("examples/data/manipulator_H10_w2.npz"), ("sparsemax",), 0.2)
chain, referee = manipulator.conjunct_chain(R)
state = solver_conjuncts.init_state(R["V0"], scene.ALPHA0, chain.K)


def show(k, rec, rho, wall):
    print(k, rho.round(4).tolist(), flush=True)


ref = referee(state["V"])
print(0, ref[0].round(4).tolist(), flush=True)
res = solver_conjuncts.solve(chain, referee, state, updates, scene.LAM, scene.DELTA, ref=ref, log=show)
print(wp.get_device("cuda:0").name)
print(dict(zip(res["seconds_keys"], np.median(res["seconds"], 0).round(3).tolist())), "total", np.median(res["seconds"].sum(1)).round(3))

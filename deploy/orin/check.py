# Every measure of the Warp evaluator on the GPU, on one until specification:
# the robustness and the gradient weight on the violating samples.
import numpy as np
import warp as wp

from sparsemax_dstl import Atom, Until, compile_formula
from sparsemax_dstl.warp import Evaluator

wp.config.log_level = wp.LOG_WARNING

# stay out of the zone (predicate 0) until the pick (predicate 1), over 41 samples;
# the trajectory is inside the zone on samples 10 to 13, and the pick happens at sample 30
program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)
t = np.arange(41)
scores = np.stack([np.where((t >= 10) & (t < 14), -0.05, 0.2), np.where(t >= 30, 0.1, -1.0)], -1)[None]
x = wp.array(scores, dtype=wp.float32, device="cuda:0")

def show(measure, param=None):
    rho, grad = Evaluator(program, measure, param, B=1, dtype=wp.float32, device="cuda:0").gradient(x)
    print(measure, round(float(rho.numpy()[0, 0]), 6), round(float(grad.numpy()[0, 10:14, 0].sum()), 4))

print(wp.get_device("cuda:0").name)
show("exact")
show("lse_plain", 20.0)
show("lse", 20.0)
show("gm_pm01")
show("gm_pm10")
show("gm_exp", 0.1)
show("sparsemax", 0.1)

<p align="center">
  <img src="assets/README%20lockup@2x.png" alt="SparseMax-∂STL: Sound, sparse smoothing for differentiable STL" width="960">
</p>

# SparseMax-∂STL

Code of the paper

> S. S. Damera, R. Matheu, J. S. Baras and C. Belta, "SparseMax-∂STL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization."

The paper introduces lower approximations of the minimum and the maximum whose gradients are sparsemax projections, and builds from them a smoothed robustness for Signal Temporal Logic (STL) specifications. This repository holds that robustness as a Python package, `sparsemax_dstl`, with one evaluator in JAX and one in Warp, together with the smoothings the paper compares it with (the plain and the sound log-sum-exp, the generalized-mean robustness), the exact robustness, and the scripts that produce the numbers of the paper's two examples.

## Contents

| Part of the paper | Code |
|---|---|
| The sparsemax lower approximations of the maximum and the minimum, and their weights (Section IV) | `src/sparsemax_dstl/operators.py` |
| STL specifications, their compilation into an evaluation graph over the samples, and the robustness under every measure (Sections III and IV) | `src/sparsemax_dstl/stl/`: JAX in `semantics.py`, Warp with its adjoint in `warp_backend.py` |
| Planar unicycle: Table I and Fig. 4 (Section V-A) | `src/sparsemax_dstl/tasks/planar*.py`, `examples/` |
| Manipulator beside a person: Tables II and III (Section V-B) | `src/sparsemax_dstl/tasks/`, `warp_plant.py`, `warp_predicates.py`, `constrained_warp.py`, `constrained_conj.py`, `examples/` |

[examples/README.md](examples/README.md) describes every module and lists the names of the measures in the code.

## Install

[uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```bash
git clone https://github.com/sdamera95/SparseMax-dSTL.git
cd SparseMax-dSTL
uv sync
./scripts/fetch_menagerie.sh
```

`uv sync` installs the versions of `uv.lock`: JAX 0.11.2, MuJoCo and MJX 3.12.0, Warp 1.17.0, and MuJoCo Warp from the `adjoint` branch of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at commit `357a75d`, which adds the reverse-mode derivatives the manipulator example needs. That source is declared in `[tool.uv.sources]` of `pyproject.toml`, which `pip` does not read, so install with uv. The second command clones the Franka Panda model of MuJoCo Menagerie into `third_party/`; the manipulator example and its tests use it.

JAX is installed with CUDA on x86-64 Linux and for the CPU elsewhere. The optimization of the manipulator needs a CUDA GPU; everything else runs on the CPU.

## Use

A specification is written over predicates, which are referred to by their index, and compiled for a number of samples. The robustness is then a function of the predicate values, an array of shape `(T, P)` that is positive where a predicate holds.

```python
import jax
import jax.numpy as jnp
from sparsemax_dstl.stl import Atom, Until, compile_formula, robustness

# stay out of the zone (predicate 0) until the pick (predicate 1), over 41 samples
program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)

# the trajectory is inside the zone on samples 10 to 13, and the pick happens at sample 30
t = jnp.arange(41)
scores = jnp.stack([jnp.where((t >= 10) & (t < 14), -0.05, 0.2), jnp.where(t >= 30, 0.1, -1.0)], -1)

def rho(scores, measure, param=None):
    return robustness(program, scores, measure, param)[0]

print(rho(scores, "exact"))            # -0.05
print(rho(scores, "sparsemax", 0.1))   # -0.090825945

w = jax.grad(rho)(scores, "sparsemax", 0.1)
print(w[10:14, 0].sum(), int((w != 0).sum()))   # 1.0000001 4
```

`param` is $\gamma$ for `"sparsemax"` and $\beta$ for the log-sum-exp measures. The last line prints the gradient weight on the four samples inside the zone and the number of nonzero entries of the gradient. `robustness_warp` in `sparsemax_dstl.stl.warp_backend` evaluates the same compiled program in Warp, on a batch of predicate values of shape `(B, T, P)`.

## The paper's examples

[examples/README.md](examples/README.md) lists every command, what it writes, and the table or figure it corresponds to.

| Example | In the paper | Runs on | Time |
|---|---|---|---|
| Planar unicycle | Table I, Fig. 4 | CPU, float64 | about 5 minutes |
| Manipulator, the until conjunct on the violating initial trajectories | Table II | CPU, then one GPU stage in float32 | about 7 minutes, 2 of them on the GPU |
| Manipulator, the optimization on the torques | Table III | one GPU, float32; the exact robustness of every iterate from a float64 replay with MuJoCo | 17, 29 and 55 minutes per file at horizons of 10, 20 and 40 s; 4.8 hours for the nine files |

The times are from a workstation with one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). The GPU stages are not bit-reproducible: two runs of the Table II stage on that GPU differ by up to $6 \times 10^{-5}$ relative in the norm of the torque gradient, and the printed table is the same.

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q --ignore=tests/test_warp_predicates.py
JAX_PLATFORMS=cpu uv run pytest -q tests/test_warp_predicates.py
```

396 tests, about 7 minutes. The 8 that need a CUDA device are skipped without one. `tests/test_warp_predicates.py` runs in a process of its own because it has a float32 case and other test files switch JAX to float64 for the whole process.

## Citation

```bibtex
@misc{damera2026sparsemax,
  title  = {{SparseMax}-$\partial${STL}: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization},
  author = {Damera, Sai Sandeep and Matheu, Ryan and Baras, John S. and Belta, Calin},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).

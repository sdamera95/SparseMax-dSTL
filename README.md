<p align="center">
  <img src="assets/README%20lockup@2x.png" alt="SparseMax-∂STL: Sound, sparse smoothing for differentiable STL" width="960">
</p>

# SparseMax-∂STL

Differentiable Signal Temporal Logic (STL) robustness in [Warp](https://github.com/NVIDIA/warp) and in JAX with the SparseMax smoothing, and sparse derivatives of STL-constrained nonlinear programs for model predictive control. The package holds the code of two papers:

- [SparseMax-∂STL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization](https://arxiv.org/abs/2610.23837), with its notebooks in [notebooks/sparsemax-robustness](notebooks/sparsemax-robustness/).
- Specification-Aware Sparse Differentiation: Planning under Signal Temporal Logic with Sparsemax Robustness (L4DC 2027 submission), with its notebooks in [notebooks/sparse-differentiation](notebooks/sparse-differentiation/).

## SparseMax robustness

A gradient-based solver that enforces an STL specification uses a smooth approximation of its robustness. In safety-critical trajectory optimization the approximation must be sound, nonnegative only on trajectories that satisfy the specification, and at each temporal operator the weight of its gradient on the violating samples must stay bounded away from zero however long the horizon. The log-sum-exp (LSE), its sound variant and the generalized-mean robustness (GMR) [2] do not have both. SparseMax replaces the minimum and the maximum by lower approximations whose gradients are sparsemax projections: the robustness lies below the exact one with an error bound independent of the number of samples, and its gradient is exactly zero on every sample beyond a data-dependent threshold. On a seven-joint manipulator beside a person, as the horizon grows from 10 to 80 s, the gradient weight of the sound LSE on the violating samples falls from 0.22 to 0.04, while that of SparseMax stays at 1.

The package evaluates the robustness and its gradient with respect to the predicate values in two backends. In Warp, a specification is compiled once into an evaluation graph and evaluated by kernels with hand-written adjoints for a batch of trajectories; on a CUDA device the forward and backward passes are recorded as CUDA graphs. Paired with MuJoCo Warp, a trajectory optimization runs with no call outside Warp, on a workstation and on a Jetson AGX Orin ([deploy/orin](deploy/orin/README.md)). In JAX, the same measures run on the same graph, and SparseMax enters as two operators, `lower_max` and `lower_min`, that carry their own derivative.

The evaluation graph adopts the masking formulation of STLCG++ [1], the method of the JAX library [stljax](https://github.com/UW-CTRL/stljax): each temporal operator is a masked reduction over a window of samples, so a specification unrolls into a static list of maxima and minima. This package does not depend on stljax. The until has a closed prefix: a witness at offset $j$ requires $\psi$ at sample $t+j$ and $\varphi$ at every sample from $t$ through $t+j$.

| Name in the code | Measure | Parameter |
|---|---|---|
| `exact` | the exact robustness | none |
| `lse_plain` | the plain log-sum-exp | $\beta$ |
| `lse` | the sound log-sum-exp, a lower bound of the exact robustness | $\beta$ |
| `gm_pm01`, `gm_pm10` | the GMR [2] of order $(0, 1)$ and $(-10, 10)$ | none |
| `gm_exp` | the GMR [2] with exponential generating functions | the error per node $\varepsilon$ in Warp ($\beta = \log m / \varepsilon$ over $m$ entries), $\beta$ in JAX |
| `sparsemax` | SparseMax, a lower bound of the exact robustness | $\gamma$ |

## Sparse derivatives for STL-constrained control

A model predictive controller under an STL specification solves, at every control period, a nonlinear program whose constraints are the dynamics and one smoothed robustness constraint per conjunct of the specification. Automatic differentiation without knowledge of sparsity computes the constraint Jacobian and the Lagrangian Hessian that an interior-point solver needs in full. The dynamics and the specification prescribe their sparsity pattern through the STL syntax, the windows of the temporal operators and the variables that the predicates read; CasADi recovers the same pattern by structural detection. Under SparseMax the derivative is exactly zero at every operand at or below the threshold, a sparsity that depends on the trajectory and that structural detection cannot see.

The package builds the prescribed patterns from the specification's syntax tree, restricts them to the SparseMax supports at the current iterate, and computes both matrices from compressed automatic differentiation products in JAX with the colorings of [asdex](https://github.com/adrhill/asdex), or assembles them on the supports. IPOPT receives them through CasADi and, at every accepted iterate, a margin test checks that the supports still lie in the declared structure. On a dual-arm coordination task, in which two arms share a hand-off zone divided into cells and one until constraint per scheduled visit keeps the other arm out of a cell until the visitor has left it, these exact-Hessian solves are up to 27 times faster than those with the sound LSE in CasADi and IPOPT.

## Install

[uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```bash
git clone https://github.com/sdamera95/SparseMax-dSTL.git
cd SparseMax-dSTL
uv sync
./scripts/fetch_menagerie.sh
```

`uv sync` installs the versions of `uv.lock` (Warp 1.17.0, JAX 0.11.2, MuJoCo and MJX 3.12.0, asdex 0.6.0, CasADi 3.8.1) and MuJoCo Warp from [third_party/mujoco_warp](third_party/mujoco_warp/SOURCE.md), the `adjoint` branch of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at commit `357a75d` with two changes to its contact derivatives. Both sources are declared in `[tool.uv.sources]` of `pyproject.toml`, which `pip` does not read, so install with uv. The second command fetches the Franka Panda model of MuJoCo Menagerie for the manipulator example.

The Warp backend alone needs no JAX: `uv sync --no-dev` installs NumPy, Warp, MuJoCo and MuJoCo Warp. Three extras add to it:

| Extra | Adds | For |
|---|---|---|
| `jax` | JAX (with CUDA on x86-64 Linux) and MJX | the JAX backend |
| `sparse` | the `jax` extra, asdex, CasADi with IPOPT, SciPy | `sparsemax_dstl.sparse`, `sparsemax_dstl.mpc`, the dual-arm manipulator |
| `notebooks` | matplotlib, Jupyter's kernel and `nbconvert` | the notebooks |

The manipulator example needs a CUDA GPU; everything else runs on the CPU.

## Use

A specification is written over predicates, referred to by their index, and compiled for a number of samples. The robustness is a function of the predicate values, which are positive where a predicate holds.

```python
import numpy as np
import warp as wp
from sparsemax_dstl.stl import Atom, Until, compile_formula
from sparsemax_dstl.warp import Evaluator

# stay out of the zone (predicate 0) until the pick (predicate 1), over 41 samples
program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)

# predicate values of shape (B, T, P): the trajectory is inside the zone
# on samples 10 to 13, and the pick happens at sample 30
t = np.arange(41)
scores = np.stack([np.where((t >= 10) & (t < 14), -0.05, 0.2), np.where(t >= 30, 0.1, -1.0)], -1)[None]

device = "cuda:0" if wp.is_cuda_available() else "cpu"
x = wp.array(scores, dtype=wp.float32, device=device)

exact = Evaluator(program, "exact", None, B=1, dtype=wp.float32, device=device)
print(exact.value(x).numpy())                   # [[-0.05]]

sparsemax = Evaluator(program, "sparsemax", 0.1, B=1, dtype=wp.float32, device=device)
rho, grad = sparsemax.gradient(x)
print(rho.numpy())                              # [[-0.09082595]]
w = grad.numpy()[0]
print(w[10:14, 0].sum(), int((w != 0).sum()))   # 0.9999998 4
```

The SparseMax robustness lies below the exact one, and its whole gradient sits on the four samples inside the zone. The same in JAX:

```python
import jax
import jax.numpy as jnp
from sparsemax_dstl.jax import robustness
from sparsemax_dstl.stl import Atom, Until, compile_formula

program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)
t = jnp.arange(41)
scores = jnp.stack([jnp.where((t >= 10) & (t < 14), -0.05, 0.2), jnp.where(t >= 30, 0.1, -1.0)], -1)

def rho(scores, measure, param=None):
    return robustness(program, scores, measure, param)[0]

print(rho(scores, "exact"))            # -0.05
print(rho(scores, "sparsemax", 0.1))   # -0.090825945
w = jax.grad(rho)(scores, "sparsemax", 0.1)
print(w[10:14, 0].sum(), int((w != 0).sum()))   # 1.0000001 4
```

## Reproducing the papers

Each paper has a folder of notebooks, stored with their outputs, whose README lists what each notebook computes, the table or figure of the paper it reproduces, and its run time.

| Paper | Folder | Contents |
|---|---|---|
| SparseMax-∂STL | [notebooks/sparsemax-robustness](notebooks/sparsemax-robustness/) | the planar unicycle, the manipulator beside a person, and the JAX backend against the Warp backend |
| Specification-aware sparse differentiation | [notebooks/sparse-differentiation](notebooks/sparse-differentiation/) | the sparsity patterns and derivative products, the derivative times, the single solves, the re-planning loops and the closed loop on a rigid-body simulation; scripts for the longest computations; the measured numbers of every table and figure |

## Layout

| Path | Content |
|---|---|
| `src/sparsemax_dstl/stl/` | specifications, their compilation into the evaluation graph, and the rows a recorded prefix decides |
| `src/sparsemax_dstl/warp/` | the Warp evaluator and its adjoints, the MuJoCo Warp rollout with its reverse pass, predicates as kernels, and a first-order augmented Lagrangian solver |
| `src/sparsemax_dstl/jax/` | the JAX evaluator, the SparseMax operators, and the evaluation under masks with the weights of every reduction |
| `src/sparsemax_dstl/sparse/` | the prescribed patterns, the supports and the pattern they leave, the colorings, the derivatives on a declared structure, and the product counts |
| `src/sparsemax_dstl/mpc/` | the program of one control period, IPOPT through CasADi with the margin test, one program per horizon, the CasADi baselines, and the rigid-body simulation of both arms |
| `src/sparsemax_dstl/tasks/` | the planar unicycle, the manipulator beside a person, and the dual-arm manipulator with its scenes |
| `notebooks/` | one folder per paper |
| `deploy/orin/` | the install and optimization scripts for a Jetson AGX Orin, and the measurements on it |
| `third_party/mujoco_warp/` | MuJoCo Warp with reverse-mode derivatives |

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q
```

488 tests, about 9 minutes on 16 cores of an AMD Ryzen 9 7950X. The 10 that need a CUDA device for Warp are skipped without one.

## References

[1] P. Kapoor, K. Mizuta, E. Kang and K. Leung, "STLCG++: A masking approach for differentiable signal temporal logic specification," IEEE Robotics and Automation Letters, 2025. Library: [stljax](https://github.com/UW-CTRL/stljax).

[2] N. Mehdipour, C.-I. Vasile and C. Belta, "Generalized mean robustness for signal temporal logic," IEEE Transactions on Automatic Control, vol. 70, no. 3, pp. 1949–1956, 2025.

## Citation

```bibtex
@misc{damera2026sparsemax,
  title = {{SparseMax}-$\partial${STL}: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization},
  author = {Damera, Sai Sandeep and Matheu, Ryan and Baras, John S. and Belta, Calin},
  year = {2026},
  eprint = {2610.23837},
  archivePrefix = {arXiv},
  primaryClass = {eess.SY},
  doi = {10.48550/arXiv.2610.23837},
  url = {https://arxiv.org/abs/2610.23837}
}
```

## License

MIT, see [LICENSE](LICENSE).

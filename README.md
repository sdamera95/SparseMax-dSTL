<p align="center">
  <img src="assets/README%20lockup@2x.png" alt="SparseMax-∂STL: Sound, sparse smoothing for differentiable STL" width="960">
</p>

# SparseMax-∂STL

Differentiable Signal Temporal Logic (STL) in [Warp](https://github.com/NVIDIA/warp) and in JAX.

**Warp.** The robustness of an STL specification and its gradient with respect to the predicate values, written as Warp kernels, for a batch of sampled trajectories. A specification is compiled once into an evaluation graph, and the kernels evaluate it under the exact robustness, the plain and the sound log-sum-exp (LSE), the generalized-mean robustness (GMR) [3], and SparseMax [2]. Every measure has a hand-written adjoint, and on a CUDA device the forward and backward passes are recorded as CUDA graphs. Paired with a Warp simulator, MuJoCo Warp in the manipulator example, a trajectory optimization under an STL specification runs with no call outside Warp, and it deploys on NVIDIA Jetson AGX boards: we ran it on a Jetson AGX Orin 64 GB ([measurements below](#on-nvidia-jetson-agx)).

**JAX.** The same measures on the same compiled graph, differentiated by JAX. SparseMax enters as two operators, `lower_max` and `lower_min`, that carry their own derivative.

The evaluation graph adopts the masking formulation of STLCG++ [1], the method of the JAX library [stljax](https://github.com/UW-CTRL/stljax): each temporal operator is a masked reduction over a window of samples, so a specification unrolls into a static list of maxima and minima. This package does not depend on stljax. The until has a closed prefix: a witness at offset $j$ requires $\psi$ at sample $t+j$ and $\varphi$ at every sample from $t$ through $t+j$. A node is evaluated only at the samples where its whole window exists.

## Measures

| Name in the code | Measure | Parameter |
|---|---|---|
| `exact` | the exact robustness | none |
| `lse_plain` | the plain log-sum-exp | $\beta$ |
| `lse` | the sound log-sum-exp, a lower bound of the exact robustness | $\beta$ |
| `gm_pm01`, `gm_pm10` | the generalized-mean robustness [3] of order $(0, 1)$ and $(-10, 10)$ | none |
| `gm_exp` | the generalized-mean robustness [3] with exponential generating functions | the error per node $\varepsilon$ in Warp, $\beta$ in JAX |
| `sparsemax` | SparseMax [2], a lower bound of the exact robustness | $\gamma$ |

Both implementations take these names. For `gm_exp` the Warp evaluator takes the error per node $\varepsilon$ and uses $\beta = \log m / \varepsilon$ for a conjunction or disjunction of $m$ entries, and the JAX evaluator takes $\beta$; `sparsemax_dstl.jax.methods.SEMANTICS["gm_exp"]` is the JAX form that takes $\varepsilon$.

## Install

[uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```bash
git clone https://github.com/sdamera95/SparseMax-dSTL.git
cd SparseMax-dSTL
uv sync
./scripts/fetch_menagerie.sh
```

`uv sync` installs the versions of `uv.lock`: Warp 1.17.0, JAX 0.11.2, MuJoCo and MJX 3.12.0, and MuJoCo Warp from [third_party/mujoco_warp](third_party/mujoco_warp/SOURCE.md). That folder is the `adjoint` branch of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at commit `357a75d`, which adds the reverse-mode derivatives the manipulator example needs, with two files changed: the adjoint of box-box contact geometry is added, and a contact stays in the reverse pass when any of its constraint rows is active. The folder is declared in `[tool.uv.sources]` of `pyproject.toml`, which `pip` does not read, so install with uv. The second command clones the Franka Panda model of MuJoCo Menagerie into `third_party/`; the manipulator example and its tests use it.

`uv sync` installs both backends and what the notebooks and the tests use. The Warp backend alone needs no JAX:

```bash
uv sync --no-dev
```

That environment holds NumPy, Warp, MuJoCo and MuJoCo Warp. `sparsemax_dstl.warp`, the specification layer and the Warp side of both examples import and run in it. Two extras add to it, as in `uv sync --no-dev --extra notebooks`:

| Extra | Adds | For |
|---|---|---|
| `notebooks` | matplotlib and Jupyter's kernel and `nbconvert` | `examples/unicycle.ipynb` and `examples/manipulator.ipynb` |
| `jax` | JAX, with CUDA on x86-64 Linux and for the CPU elsewhere, and MJX | the JAX backend, the MJX plant and `examples/jax_backend.ipynb` |

The manipulator example needs a CUDA GPU; everything else runs on the CPU.

## Use

A specification is written over predicates, which are referred to by their index, and compiled for a number of samples. The robustness is then a function of the predicate values, which are positive where a predicate holds.

### Warp

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

An `Evaluator` is built for one specification, measure, parameter, batch size and device, and allocates its arrays once. `value` returns the robustness, of shape `(B, L)`, and `gradient` returns it with the gradient with respect to the predicate values, both as Warp arrays on the device. The last line prints the gradient weight on the four samples inside the zone and the number of nonzero entries of the gradient. `robustness_warp` in the same module records the evaluation on a `wp.Tape`, so the gradient continues into the kernels that computed the predicate values.

### JAX

```python
import jax
import jax.numpy as jnp
from sparsemax_dstl.jax import robustness
from sparsemax_dstl.stl import Atom, Until, compile_formula

program = compile_formula(Until((0, 40), Atom(0), Atom(1)), T=41)

# predicate values of shape (T, P)
t = jnp.arange(41)
scores = jnp.stack([jnp.where((t >= 10) & (t < 14), -0.05, 0.2), jnp.where(t >= 30, 0.1, -1.0)], -1)

def rho(scores, measure, param=None):
    return robustness(program, scores, measure, param)[0]

print(rho(scores, "exact"))            # -0.05
print(rho(scores, "sparsemax", 0.1))   # -0.090825945

w = jax.grad(rho)(scores, "sparsemax", 0.1)
print(w[10:14, 0].sum(), int((w != 0).sum()))   # 1.0000001 4
```

## On NVIDIA Jetson AGX

The Warp backend installs on a Jetson without JAX and runs on its GPU:

```bash
./deploy/orin/setup.sh
```

The script installs NumPy, Warp, MuJoCo and MuJoCo Warp with `uv sync --locked --no-dev` and evaluates every measure on the board's GPU. [deploy/orin/README.md](deploy/orin/README.md) has the commands for the manipulator optimization on the board and the measurements in full. On a Jetson AGX Orin 64 GB against a workstation with one NVIDIA RTX PRO 6000 Blackwell Max-Q, the same commit and the same environment without JAX on both:

| | Workstation | Jetson AGX Orin 64 GB |
|---|---|---|
| one update of the manipulator optimization with one constraint per conjunct, eight trajectories, horizon 10 s | 6.7 s | 9.4 s |
| of which the specification and its gradient | 0.02 s | 0.11 s |

Over ten updates the exact robustness of the manipulator's trajectories differs between the two machines by at most $1.2 \times 10^{-5}$ and has the same sign after every update.

The paper ran the manipulator optimization on the same board with Warp built for CUDA 13, as the same code that runs on the workstation: the MuJoCo Warp simulation and its reverse pass, the predicates, the robustness, its gradient and the solver. It reports (Appendix II-G, Table XII), for one run at a horizon of 10 s:

| | Workstation, NVIDIA RTX PRO 6000 | Jetson AGX Orin 64 GB |
|---|---|---|
| one physics step of one world, forward and reverse | 236 and 800 µs | 298 and 1106 µs |
| one update of the optimization | 6.2 s | 8.6 to 8.7 s |
| compilation of the kernels, once | | 80 s |

The first satisfying update is the same on both machines, and their exact robustness differs by at most $2.8 \times 10^{-6}$ over 30 updates. On the workstation the specification, its gradient, the exact robustness and the solver together take under 0.05 s of an update; the rest is the simulation and its reverse pass.

## Layout

| Path | Content |
|---|---|
| `src/sparsemax_dstl/stl/formula.py`, `program.py` | specifications and their compilation into the evaluation graph |
| `src/sparsemax_dstl/warp/evaluator.py` | the Warp evaluator and its adjoints |
| `src/sparsemax_dstl/jax/evaluator.py`, `operators.py` | the JAX evaluator and the SparseMax operators |
| `src/sparsemax_dstl/warp/plant.py`, `predicates.py` | the MuJoCo Warp rollout with its reverse-mode gradient, and predicates as Warp kernels |
| `src/sparsemax_dstl/warp/solver.py`, `solver_conjuncts.py` | the first-order augmented Lagrangian solver under single shooting |
| `src/sparsemax_dstl/tasks/` | the planar unicycle and the manipulator beside a person |
| `third_party/mujoco_warp/` | MuJoCo Warp with reverse-mode derivatives, the copy that `uv sync` installs |
| `examples/` | three notebooks on the paper's examples, and the initial trajectories of the manipulator in `examples/data/` |
| `deploy/orin/` | the install script and the optimization script for a Jetson AGX Orin, and the measurements on it |

## The paper's examples

Three notebooks, stored with their outputs:

| Notebook | In the paper | Runs on | Time |
|---|---|---|---|
| [unicycle.ipynb](examples/unicycle.ipynb) | Planar unicycle (Section V-A): Table I and Fig. 4 | Warp, CPU, float64, no JAX | 2.5 minutes |
| [manipulator.ipynb](examples/manipulator.ipynb) | Manipulator beside a person (Section V-B): Table II, and 20 of the 100 updates of Table III at four of its 16 pick and handover locations | Warp and MuJoCo Warp, no JAX; one CUDA GPU with 14 GB free, float32, with the exact robustness from a float64 replay with MuJoCo | 5.5 minutes |
| [jax_backend.ipynb](examples/jax_backend.ipynb) | The unicycle's specification in the JAX evaluator, and the optimization of Table I on the JAX backend against the Warp backend | JAX and Warp, CPU, float64 | 3.5 minutes |

Open them in Jupyter or an editor with a Jupyter kernel, or execute one from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace examples/unicycle.ipynb
```

The times are from a workstation with an AMD Ryzen 9 7950X and one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). In Table I the returned robustness is the JAX backend's; the Warp backend reproduces 31 of the table's 32 printed entries and returns $-0.0586$ where it prints $-0.058$, for the plain LSE at $c = 0$ (JAX: $-0.0584$). Runs on the GPU do not repeat bit for bit: two runs of the Table II computation on that GPU differ by up to $6 \times 10^{-5}$ relative in the norm of the torque gradient, and the printed table is the same; between two executions of the manipulator notebook the first satisfying update differed in one of the 40 runs (10 and 12).

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q --ignore=tests/test_warp_predicates.py
JAX_PLATFORMS=cpu uv run pytest -q tests/test_warp_predicates.py
```

383 tests, about 6 minutes. Those that need a CUDA device are skipped without one. `tests/test_warp_predicates.py` runs in a process of its own because it has a float32 case and other test files switch JAX to float64 for the whole process.

## References

[1] P. Kapoor, K. Mizuta, E. Kang and K. Leung, "STLCG++: A masking approach for differentiable signal temporal logic specification," IEEE Robotics and Automation Letters, 2025. Library: [stljax](https://github.com/UW-CTRL/stljax).

[2] S. S. Damera, R. Matheu, J. S. Baras and C. Belta, "SparseMax-∂STL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization."

[3] N. Mehdipour, C.-I. Vasile and C. Belta, "Generalized mean robustness for signal temporal logic," IEEE Transactions on Automatic Control, vol. 70, no. 3, pp. 1949–1956, 2025.

```bibtex
@misc{damera2026sparsemax,
  title  = {{SparseMax}-$\partial${STL}: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization},
  author = {Damera, Sai Sandeep and Matheu, Ryan and Baras, John S. and Belta, Calin},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).

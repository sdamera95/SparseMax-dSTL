<p align="center">
  <img src="assets/README%20lockup@2x.png" alt="SparseMax-∂STL: Sound, sparse smoothing for differentiable STL" width="960">
</p>

# SparseMax-∂STL

Differentiable Signal Temporal Logic (STL) in [Warp](https://github.com/NVIDIA/warp) and in JAX, and sparse derivatives of STL-constrained nonlinear programs for model predictive control. The package holds the code of two papers:

- [SparseMax-∂STL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization](https://arxiv.org/abs/2610.23837): the SparseMax robustness, its Warp and JAX evaluators, and the planar unicycle and the manipulator beside a person ([its examples](#the-sparsemax-stl-papers-examples)).
- Specification-Aware Sparse Differentiation: Planning under Signal Temporal Logic with Sparsemax Robustness (L4DC 2027 submission): the sparsity patterns that the STL specification and the dynamics prescribe for the constraint Jacobian and the Lagrangian Hessian, their restriction to the SparseMax supports, and a model predictive controller on a dual-arm manipulator ([its section](#sparse-derivatives-and-model-predictive-control)).

**Warp.** The robustness of an STL specification and its gradient with respect to the predicate values, written as Warp kernels, for a batch of sampled trajectories. A specification is compiled once into an evaluation graph, and the kernels evaluate it under the exact robustness, the plain and the sound log-sum-exp (LSE), the generalized-mean robustness (GMR) [2], and [SparseMax](https://arxiv.org/abs/2610.23837). Every measure has a hand-written adjoint, and on a CUDA device the forward and backward passes are recorded as CUDA graphs. Paired with a Warp simulator, MuJoCo Warp in the manipulator example, a trajectory optimization under an STL specification runs with no call outside Warp, and it deploys on NVIDIA Jetson AGX boards: we ran it on a Jetson AGX Orin 64 GB ([measurements below](#on-nvidia-jetson-agx)).

**JAX.** The same measures on the same compiled graph, differentiated by JAX. SparseMax enters as two operators, `lower_max` and `lower_min`, that carry their own derivative.

**Sparse derivatives.** The nonlinear program of one control period of a model predictive controller, with the dynamics as defect constraints and one smoothed robustness constraint per conjunct of the specification, its constraint Jacobian and Lagrangian Hessian from compressed automatic differentiation products on the patterns that the specification prescribes, and IPOPT through CasADi with these exact derivatives (`sparsemax_dstl.sparse`, `sparsemax_dstl.mpc`).

The evaluation graph adopts the masking formulation of STLCG++ [1], the method of the JAX library [stljax](https://github.com/UW-CTRL/stljax): each temporal operator is a masked reduction over a window of samples, so a specification unrolls into a static list of maxima and minima. This package does not depend on stljax. The until has a closed prefix: a witness at offset $j$ requires $\psi$ at sample $t+j$ and $\varphi$ at every sample from $t$ through $t+j$. A node is evaluated only at the samples where its whole window exists.

## Measures

| Name in the code | Measure | Parameter |
|---|---|---|
| `exact` | the exact robustness | none |
| `lse_plain` | the plain log-sum-exp | $\beta$ |
| `lse` | the sound log-sum-exp, a lower bound of the exact robustness | $\beta$ |
| `gm_pm01`, `gm_pm10` | the generalized-mean robustness [2] of order $(0, 1)$ and $(-10, 10)$ | none |
| `gm_exp` | the generalized-mean robustness [2] with exponential generating functions | the error per node $\varepsilon$ in Warp, $\beta$ in JAX |
| `sparsemax` | [SparseMax](https://arxiv.org/abs/2610.23837), a lower bound of the exact robustness | $\gamma$ |

Both implementations take these names. For `gm_exp` the Warp evaluator takes the error per node $\varepsilon$ and uses $\beta = \log m / \varepsilon$ for a conjunction or disjunction of $m$ entries, and the JAX evaluator takes $\beta$; `sparsemax_dstl.jax.methods.SEMANTICS["gm_exp"]` is the JAX form that takes $\varepsilon$.

## Install

[uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```bash
git clone https://github.com/sdamera95/SparseMax-dSTL.git
cd SparseMax-dSTL
uv sync
./scripts/fetch_menagerie.sh
```

`uv sync` installs the versions of `uv.lock`: Warp 1.17.0, JAX 0.11.2, MuJoCo and MJX 3.12.0, asdex 0.6.0, CasADi 3.8.1, and MuJoCo Warp from [third_party/mujoco_warp](third_party/mujoco_warp/SOURCE.md). That folder is the `adjoint` branch of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at commit `357a75d`, which adds the reverse-mode derivatives the manipulator example needs, with two files changed: the adjoint of box-box contact geometry is added, and a contact stays in the reverse pass when any of its constraint rows is active. The folder and the asdex commit are declared in `[tool.uv.sources]` of `pyproject.toml`, which `pip` does not read, so install with uv. The second command clones the Franka Panda model of MuJoCo Menagerie into `third_party/`; the manipulator example and its tests use it.

`uv sync` installs both backends, the sparse derivatives and what the notebooks and the tests use. The Warp backend alone needs no JAX:

```bash
uv sync --no-dev
```

That environment holds NumPy, Warp, MuJoCo and MuJoCo Warp. `sparsemax_dstl.warp`, the specification layer and the Warp side of both examples of the SparseMax-∂STL paper import and run in it. Three extras add to it, as in `uv sync --no-dev --extra notebooks`:

| Extra | Adds | For |
|---|---|---|
| `notebooks` | matplotlib and Jupyter's kernel and `nbconvert` | `notebooks/sparsemax-robustness/unicycle.ipynb` and `notebooks/sparsemax-robustness/manipulator.ipynb` |
| `jax` | JAX, with CUDA on x86-64 Linux and for the CPU elsewhere, and MJX | the JAX backend, the MJX plant and `notebooks/sparsemax-robustness/jax_backend.ipynb` |
| `sparse` | the `jax` extra, asdex, CasADi (with IPOPT and MUMPS) and SciPy | `sparsemax_dstl.sparse`, `sparsemax_dstl.mpc`, the dual-arm manipulator and, with `notebooks`, `notebooks/sparse-differentiation/` |

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

The script installs NumPy, Warp, MuJoCo and MuJoCo Warp with `uv sync --locked --no-dev` and evaluates every measure on the board's GPU. [deploy/orin/README.md](deploy/orin/README.md) has the commands for the manipulator optimization on the board and the measurements in full. On a Jetson AGX Orin 64 GB against a workstation with one NVIDIA RTX PRO 6000 Blackwell Max-Q, the same code and the same environment without JAX on both:

| | Workstation | Jetson AGX Orin 64 GB |
|---|---|---|
| one update of the manipulator optimization with one constraint per conjunct, eight trajectories, horizon 10 s | 6.6 s | 9.7 s |
| of which the specification and its gradient | 0.02 s | 0.11 s |

Over ten updates the exact robustness of the eight trajectories, printed to four decimals, differs between the two machines in at most the last digit and has the same sign after every update.

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
| `src/sparsemax_dstl/stl/progression.py` | which rows of a compiled specification a recorded prefix decides at a margin, and the operands it leaves open |
| `src/sparsemax_dstl/warp/evaluator.py` | the Warp evaluator and its adjoints |
| `src/sparsemax_dstl/jax/evaluator.py`, `operators.py` | the JAX evaluator and the SparseMax operators |
| `src/sparsemax_dstl/jax/progression.py` | the evaluation under the masks of the open operands, with the SparseMax weights of every reduction |
| `src/sparsemax_dstl/warp/plant.py`, `predicates.py` | the MuJoCo Warp rollout with its reverse-mode gradient, and predicates as Warp kernels |
| `src/sparsemax_dstl/warp/solver.py`, `solver_conjuncts.py` | the first-order augmented Lagrangian solver under single shooting |
| `src/sparsemax_dstl/tasks/` | the planar unicycle and the manipulator beside a person |
| `src/sparsemax_dstl/tasks/dual_arm.py`, `dual_arm_scenes.py`, `data/` | the dual-arm manipulator (two seven-joint arms, their forward kinematics and collision spheres, the double integrator in joint space, the predicates), the five-cell zone, the hold-up and the 64-cell circle, and the arms' URDFs without their meshes |
| `src/sparsemax_dstl/sparse/patterns.py` | the constraint Jacobian and Lagrangian Hessian patterns that the dynamics and the specification prescribe for one control period |
| `src/sparsemax_dstl/sparse/supports.py` | the SparseMax supports at a point and the pattern they leave |
| `src/sparsemax_dstl/sparse/coloring.py` | asdex's colorings of a declared pattern, and dense automatic differentiation of a whole program |
| `src/sparsemax_dstl/sparse/assembly.py` | the constraint Jacobian and the Lagrangian Hessian on a declared structure |
| `src/sparsemax_dstl/sparse/counts.py` | the derivative products of dense differentiation, the prescribed pattern and the supports over control periods |
| `src/sparsemax_dstl/mpc/conjuncts.py`, `program.py` | the conjunct instances over the receding horizon, and the nonlinear program of one control period |
| `src/sparsemax_dstl/mpc/ipopt.py`, `horizon.py` | IPOPT through CasADi with these derivatives, with the test of the supports at every accepted iterate; one program per horizon, compiled once, with each period's state and instances as parameters |
| `src/sparsemax_dstl/mpc/baselines.py` | the same program in CasADi's scalar expressions under the sound log-sum-exp or the GMR, with CasADi's own sparsity detection and derivatives, and dense differentiation in JAX |
| `src/sparsemax_dstl/mpc/simulation.py` | the rigid-body simulation of both arms in MuJoCo under joint drives, and the exact robustness of every instance on an executed motion |
| `third_party/mujoco_warp/` | MuJoCo Warp with reverse-mode derivatives, the copy that `uv sync` installs |
| `notebooks/sparsemax-robustness/` | three notebooks on the SparseMax-∂STL paper's examples, and the initial trajectories of the manipulator in `data/` |
| `notebooks/sparse-differentiation/` | four notebooks on the sparse-differentiation paper, its scripts, figures and measured results |
| `deploy/orin/` | the install script and the optimization script for a Jetson AGX Orin, and the measurements on it |

## The SparseMax-∂STL paper's examples

Three notebooks, stored with their outputs:

| Notebook | In the paper | Runs on | Time |
|---|---|---|---|
| [unicycle.ipynb](notebooks/sparsemax-robustness/unicycle.ipynb) | Planar unicycle (Section V-A): Table I and Fig. 4 | Warp, CPU, float64, no JAX | 2.5 minutes |
| [manipulator.ipynb](notebooks/sparsemax-robustness/manipulator.ipynb) | Manipulator beside a person (Section V-B): Table II, and 20 of the 100 updates of Table III at four of its 16 pick and handover locations | Warp and MuJoCo Warp, no JAX; one CUDA GPU with 14 GB free, float32, with the exact robustness from a float64 replay with MuJoCo | 5.5 minutes |
| [jax_backend.ipynb](notebooks/sparsemax-robustness/jax_backend.ipynb) | The unicycle's specification in the JAX evaluator, and the optimization of Table I on the JAX backend against the Warp backend | JAX and Warp, CPU, float64 | 3.5 minutes |

Open them in Jupyter or an editor with a Jupyter kernel, or execute one from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/sparsemax-robustness/unicycle.ipynb
```

The times are from a workstation with an AMD Ryzen 9 7950X and one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). In Table I the returned robustness is the JAX backend's; the Warp backend reproduces 31 of the table's 32 printed entries and returns $-0.0586$ where it prints $-0.058$, for the plain LSE at $c = 0$ (JAX: $-0.0584$). Runs on the GPU do not repeat bit for bit: two runs of the Table II computation on that GPU differ by up to $6 \times 10^{-5}$ relative in the norm of the torque gradient, and the printed table is the same; between two executions of the manipulator notebook the first satisfying update differed in one of the 40 runs (10 and 12).

## Sparse derivatives and model predictive control

The code of "Specification-Aware Sparse Differentiation: Planning under Signal Temporal Logic with Sparsemax Robustness" (L4DC 2027 submission).

### What it computes

Two seven-joint arms pass through a shared hand-off zone divided into cells, balls of radius 0.03 m that only one arm may occupy at a time. The specification is a conjunction of one until per scheduled visit, which keeps the other arm outside the cell until the visitor has left it within a window of $W_i$ samples, and one eventually per waypoint, which requires the hand at the waypoint by a deadline. A model predictive controller re-plans the joint motion of both arms: at every control period $k$ it solves a nonlinear program over the next $T$ time steps of 0.02 s, whose variables are the states (joint positions and velocities, 28) and the joint accelerations (14) of every time step, whose constraints are the defects of the double integrator, the joint, velocity and acceleration limits, rest at the last time step, and one constraint per conjunct instance that the record has not yet decided, its SparseMax robustness at least the margin 0.02, and whose cost penalizes the deviation from a reference motion, the accelerations and their change. Three problems are built by `sparsemax_dstl.tasks.dual_arm_scenes`: the five-cell zone (both arms cross five cells once per loop of 3.2 s, half a loop apart), the hold-up (the five-cell zone with the right arm held up by a number of samples and its schedule moved with it), and the 64-cell circle (both hands pass 64 cells on a circle once per turn of 3 s, with $W_i = 100$).

The code computes, for the program of one control period:

- the sparsity patterns of the constraint Jacobian and of the Lagrangian Hessian that the dynamics and the specification prescribe, from the specification's syntax tree in negation normal form, the windows of its temporal operators and the joints that each predicate reads (`sparse/patterns.py`); this is the pattern that a structural detector such as CasADi's finds for the program;
- the supports of the SparseMax smoothing at the current iterate, the operands with a nonzero weight, and the pattern they leave inside the prescribed one, declared with a margin (`sparse/supports.py`);
- the constraint Jacobian and the Lagrangian Hessian on the declared structure, from compressed products of automatic differentiation in JAX with asdex's colorings or assembled on the supports (`sparse/coloring.py`, `sparse/assembly.py`);
- a solve by IPOPT through CasADi with these exact derivatives, which tests at every accepted iterate, before the Hessian is evaluated, whether the supports still lie in the declared structure and, when they do not, declares the structure again on their union and continues from that iterate (`mpc/ipopt.py`); and one program per horizon, compiled once, with each control period's measured state, reference and conjunct instances as parameters (`mpc/horizon.py`).

The closed loop runs on a rigid-body simulation of both arms in MuJoCo with joint drives that track each plan, and checks the exact robustness of every until and reach instance on the executed motion (`mpc/simulation.py`).

### Comparators

Every table of the paper compares these, on the same program up to the smoothing, with the same cost, bounds and IPOPT options:

| Name | What it is |
|---|---|
| SparseMax, our patterns (JAX and asdex) | the SparseMax robustness in every constraint; the constraint Jacobian and the exact Lagrangian Hessian on the supports' structure, given to IPOPT through CasADi |
| Sound-LSE in CasADi with IPOPT | the sound log-sum-exp robustness in every constraint, with the parameter matched to SparseMax's; the program in CasADi's scalar expressions, whose sparsity CasADi detects and colors, CasADi's default evaluation, IPOPT with a quasi-Newton Hessian approximation (L-BFGS) or CasADi's exact Hessian |
| GMR in CasADi with IPOPT | the generalized-mean robustness [2], built so that all subformulas and time points contribute to the score and sound for every order (its sign is the sign of the exact robustness), as the power mean robustness of order $(0, 1)$, the setting of its authors' control example; its equations as printed, with the until nested as its authors define it; in every constraint at the margin, in the same CasADi program and with the same IPOPT settings as Sound-LSE, where its authors maximize it in the cost under a margin constraint |
| dense AD (JAX) | the whole constraint Jacobian by one reverse product per constraint and the whole Lagrangian Hessian by one forward-over-reverse product per variable |

### Install

`uv sync` installs what this section needs. In an environment without the development group, the extras are `sparse` and, for the notebooks, `notebooks`:

```bash
uv sync --no-dev --extra sparse --extra notebooks
```

Everything here runs on the CPU in double precision.

### Notebooks

Four notebooks in `notebooks/sparse-differentiation/`, stored with their outputs. Each prints its own values beside the paper's, which it reads from `results/`, and writes its figures (SVG and PDF) and arrays to `out/`.

| Notebook | What it prints | In the paper | Time and memory |
|---|---|---|---|
| [patterns.ipynb](notebooks/sparse-differentiation/patterns.ipynb) | the medians over control periods of the derivative products of dense differentiation, the prescribed pattern and the supports at 100 and 200 time steps on both scenes; the constraint Jacobian products of the 64-cell circle at the control period of the derivative times; the entries and products of the two patterns of the pattern figure | the product columns of Table 2; Fig. 3; Fig. 6 | 15 min 30 s, 8.1 GiB |
| [derivatives.ipynb](notebooks/sparse-differentiation/derivatives.ipynb) | one constraint Jacobian and one Lagrangian Hessian on the supports' structure, and CasADi's constraint Jacobian of Sound-LSE and of GMR, at 100 and 200 time steps on both scenes; the paper's times of dense AD and of CasADi's exact Hessian | the time columns of Table 2; Table 5; Fig. 2 (b) and (c) | 9 min 24 s, 7.6 GiB |
| [solves.ipynb](notebooks/sparse-differentiation/solves.ipynb) | the solves from the previous control period's solution with Sound-LSE and GMR under L-BFGS and with ours at 100 and 200 time steps; ours on the structure widened by 2 and 5 samples and on the prescribed structure, from the previous solution and from the reference motion; the first solve after a hold-up of 12, 16 and 20 samples; the ten control periods after a hold-up of 16 samples; the 64-cell circle at 200 time steps | Table 1; Table 3 | 40 min 16 s, 13.3 GiB |
| [replanning.ipynb](notebooks/sparse-differentiation/replanning.ipynb) | the re-planning loops after a hold-up over horizons of 2 s and 4 s with a new plan every 0.5 s and every 0.02 s; the closed loop on the rigid-body simulation with the exact robustness of every until and reach instance on the executed motion and the arms' clearance; the same check of the 64-cell circle's executed motion in `data/` | Table 6; Fig. 2 (a) | 15 min 15 s, 4.7 GiB |

```bash
cd notebooks/sparse-differentiation
uv run jupyter nbconvert --to notebook --execute --inplace solves.ipynb
```

The last column is the wall-clock time and peak memory of one execution of each notebook on 8 cores of an AMD Ryzen 9 7950X.

### Scripts

The computations that take longest are scripts in `notebooks/sparse-differentiation/scripts/`. Each prints its value beside the paper's:

```bash
cd notebooks/sparse-differentiation
uv run python scripts/counts.py "five-cell zone" 400
```

The wall-clock time and peak memory of one run of each script on 4 cores of an AMD Ryzen 9 7950X:

| Script | Computes | Run of the script |
|---|---|---|
| `counts.py <scene> <T>` | the medians of the derivative products over control periods (Table 2) | five-cell zone, 100 time steps: 68 s, 1.6 GiB |
| `solves.py <T> <U\|D2\|D5\|P\|lse\|gmr> <previous\|reference>` | one solve on the five-cell zone (Tables 1 and 3) | 100 time steps: 10 s and 1.4 GiB for Sound-LSE from the previous solution, 32 s and 1.9 GiB for ours on the prescribed structure from the reference motion |
| `dense.py <scene> <T>` | dense AD of one control period's program (Table 2) | five-cell zone, 100 time steps: 39 s, 1.5 GiB |
| `casadi_exact_hessian.py <T> <hessian\|solve>` | CasADi's exact Lagrangian Hessian of Sound-LSE: its construction and one evaluation, or the solve from the previous solution with it (Tables 1 and 2) | `hessian`, 100 time steps: 6 min 17 s, 13.5 GiB |
| `loops.py <T> <P> <ours\|lse>` | one re-planning loop after a hold-up, a new plan every $P$ samples (Table 6) | 100 time steps, $P = 25$: 74 s and 3.3 GiB for ours, 45 s and 1.9 GiB for Sound-LSE |

### Results

Every time below comes from a workstation with an AMD Ryzen 9 7950X, one job at a time pinned to 12 of its 16 cores, the frequency governor at performance, in double precision. Our derivatives run on 12 JAX threads; CasADi's default evaluation runs on one thread; both sides call IPOPT 3.14.19 with the linear solver MUMPS 5.8.2 on one thread through CasADi 3.8.1. Each control period's program is built before its timed solve.

Derivative products of one control period, as medians over control periods: the constraint Jacobian by one reverse product per constraint (dense AD) or by a greedy row coloring, the Lagrangian Hessian by one product per variable (dense AD) or by a star coloring. The prescribed pattern is the pattern that CasADi detects for Sound-LSE and for GMR.

| Scene | $T$ | Jacobian: dense AD | prescribed | SparseMax supports | Hessian: dense AD | prescribed | SparseMax supports |
|---|---|---|---|---|---|---|---|
| five-cell zone | 100 | 2,806 | 7 | 4 | 4,200 | 1,011.5 | 59.5 |
| | 200 | 5,612 | 8 | 4 | 8,400 | 1,204 | 63 |
| | 400 | 11,222 | 8 | 4 | 16,800 | 1,204 | 63 |
| 64-cell circle | 100 | 2,823 | 25 | 9 | 4,200 | 1,393 | 203 |
| | 200 | 5,713 | 95.5 | 13 | 8,400 | 2,086 | 266 |
| | 400 | 11,493 | 102 | 13 | 16,800 | 2,786 | 266 |

Single solves on the five-cell zone from the previous control period's solution, as updates of the iterate and seconds. The factor divides the shorter Sound-LSE time by ours. Sound-LSE with L-BFGS and ours are medians of 4 runs, the others one run.

| $T$ | Sound-LSE, L-BFGS | Sound-LSE, CasADi's exact Hessian | GMR, L-BFGS | SparseMax, ours | Factor |
|---|---|---|---|---|---|
| 100 | 67, 0.79 | 3, 1.19 | 75, 0.95 | 3, 0.12 | 6.4 |
| 200 | 144, 5.53 | 3, 4.13 | 135, 5.16 | 3, 0.22 | 18.4 |
| 400 | 142, 10.81 | 3, 11.27 | 175, 14.48 | 3, 0.40 | 26.8 |

After the right arm is held up by 14 samples, re-planning every 0.5 s over a horizon of 2 s, our whole control period takes 0.11 to 0.40 s (median 0.16 s) in the ten control periods after the first solve, and Sound-LSE's solve alone 0.46 to 1.35 s (median 0.68 s); over a horizon of 4 s after a hold-up of 16 samples the medians are 0.36 s and 5.99 s. In closed loop on the rigid-body simulation, re-planning every 0.5 s over 2 s for 21 control periods, every control period after the first takes 0.09 to 0.47 s in six runs, against 0.49 to 1.45 s for Sound-LSE's solve alone; all 33 until and all 21 reach instances decided in the run hold in exact semantics on the executed motion, the until instances with a smallest exact robustness of 0.30, and the arms' collision spheres stay at least 38 mm apart.

`notebooks/sparse-differentiation/results/` holds the measured numbers of every table and figure of the paper, one row per cell with the machine, the cores, the threads, the frequency governor and the statistic: `solves.csv` (Table 1), `counts.csv` (Table 2), `further_solves.csv` (Table 3 and the 64-cell circle's solves), `spread.csv` (Table 4), `threads.csv` (Table 5), `replanning.csv` (Table 6), `orin.csv` (Table 7, the CPU of a Jetson AGX Orin 64 GB), `closed_loop.csv` (Fig. 2 (a) and the closed loops' exact robustness), `patterns.csv` (Fig. 3) and `products_horizon.csv` (Fig. 6).

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

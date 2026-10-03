# Demos

Code of the two numerical examples of the paper ([`../manuscripts/SparseMax-dSTL-submitted.pdf`](../manuscripts/SparseMax-dSTL-submitted.pdf), Section V): the planar unicycle and the manipulator beside a person. The STL evaluators are written in JAX and in Warp. The unicycle runs on both. The manipulator is simulated with MuJoCo Warp and differentiated in reverse mode.

## Contents

| Path | Content |
|---|---|
| `sparsemax_diffstl/operators.py` | the sparsemax lower approximations of the maximum and the minimum, and their weights |
| `sparsemax_diffstl/stl/` | STL specifications (`formula.py`), their compilation into a program over the samples (`program.py`), the robustness under every measure in JAX (`semantics.py`) and in Warp with its adjoint (`warp_backend.py`), and a brute-force evaluator used by the tests (`oracle.py`) |
| `sparsemax_diffstl/core_study/methods.py` | the measures at an equal worst-case error per node |
| `sparsemax_diffstl/tasks/planar*.py` | the unicycle example: regions, specification, fixed trajectories, the JAX and Warp chains, the solver |
| `sparsemax_diffstl/tasks/panda.py`, `workspace.py`, `human.py` | the manipulator example: the Panda, the person, the predicates, the specification |
| `sparsemax_diffstl/warp_plant.py`, `warp_predicates.py` | the MuJoCo Warp rollout with its reverse-mode gradient, and the predicates as Warp kernels |
| `sparsemax_diffstl/constrained_warp.py`, `constrained_conj.py` | the first-order augmented Lagrangian solver under single shooting, with one constraint per conjunct |
| `experiments/` | the scripts that produce the numbers of the paper's tables |
| `tests/` | tests of the modules above |
| `data/` | the parameters of the person's motion that the manipulator scripts read |

Names of the measures in the code:

| Name | Measure |
|---|---|
| `exact` | the exact robustness |
| `lse_plain` | the plain log-sum-exp (LSE) |
| `lse` (`lse_sound` in the unicycle scripts) | the sound LSE |
| `gm_pm01`, `gm_pm10` | the generalized-mean robustness (GMR) of order (0,1) and (-10,10) |
| `sparsemax` | sparsemax |

`gm_exp`, the member of the generalized-mean robustness with exponential generating functions, runs beside the others and is not reported in the paper.

## Setup

[uv](https://docs.astral.sh/uv/) and Python 3.12 or later. The optimization of the manipulator needs a CUDA GPU; everything else runs on the CPU.

```bash
cd demos
uv sync
./scripts/fetch_menagerie.sh
```

`uv sync` installs the versions of `uv.lock`: JAX 0.11.2, MuJoCo 3.12.0, Warp 1.17.0, and MuJoCo Warp from the `adjoint` branch of [etaoxing/mujoco_warp](https://github.com/etaoxing/mujoco_warp) at commit `357a75d`, which adds the reverse-mode derivatives. The second command clones the Franka Panda model of MuJoCo Menagerie into `third_party/`.

JAX is installed for the CPU. Every command below runs JAX on the CPU (`JAX_PLATFORMS=cpu`) or not at all (`JAX_PLATFORMS=nojax`, under which any JAX computation raises an error); the GPU work is Warp's.

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q --ignore=tests/test_warp_predicates.py
JAX_PLATFORMS=cpu uv run pytest -q tests/test_warp_predicates.py
```

`tests/test_warp_predicates.py` has a float32 case and runs in a process of its own, because other test files switch JAX to float64 for the whole process. The tests that need a CUDA device are skipped without one.

## Planar unicycle (Section V-A)

CPU, float64, about five minutes in total.

```bash
export JAX_PLATFORMS=cpu PYTHONPATH=.
uv run python experiments/e049_disk.py out/unicycle/values
uv run python experiments/e049_sound.py out/unicycle/sound out/unicycle/values/table.csv
uv run python experiments/e049_al.py out/unicycle/solve solve jax
uv run python experiments/e049_al.py out/unicycle/solve solve warp
uv run python experiments/e049_al.py out/unicycle/solve report
```

| Command | Output | In the paper |
|---|---|---|
| `e049_disk.py` | `values/table.md`: the value of every measure on the fixed trajectories $S_1$ and $S_2$, and the weight on the samples of $S_1$ inside Red at waits of 3 to 130 s | Table I, columns "value" and "weight on $S_1$", without the sound LSE |
| `e049_sound.py` | `sound/table.md`: the same with the sound LSE; `sound/node_errors.csv`: the error at each node of the until | Table I, the row of the sound LSE; the errors quoted in the text |
| `e049_al.py ... solve jax`, `solve warp` | the optimization from $S_1$ and $S_2$ at $c=0$ and $c=0.2$ m under every measure, 400 updates, on the JAX chain and on the Warp chain | |
| `e049_al.py ... report` | `solve/table.md`: exact robustness and effort of every returned trajectory, and the difference between the two chains; `solve/synthesis.svg` and `.pdf` | Table I, columns "returned $\rho$"; Fig. 4(b) |

The rows named "authors' form" maximize the generalized-mean robustness in the cost under a margin constraint, as its authors do, instead of using it as a constraint.

## Manipulator beside a person (Section V-B)

All stages but two run on the CPU. The two that need the GPU are marked.

### The until conjunct on the violating initial trajectories (Table II)

```bash
export JAX_PLATFORMS=cpu
M=out/manipulator
A=lse_plain,lse,sparsemax,gm_pm01,gm_pm10,gm_exp

# the scripts do not create their output folders
mkdir -p $M/starts10 $M/starts20 $M/starts40 $M/starts80 $M/until

# four initial trajectories per wait and overshoot depth at 10 s, then the same with the wait extended to 20, 40 and 80 s
uv run python -m experiments.e034_until_demo starts $M/starts10
uv run python -m experiments.e038_horizon starts 20 $M/starts20 $M/starts10
uv run python -m experiments.e038_horizon starts 40 $M/starts40 $M/starts10
uv run python -m experiments.e038_horizon starts 80 $M/starts80 $M/starts10

# the scene with the person, per horizon
uv run python -m experiments.e039_person_zone setup $M/starts10 data/visit1.json visit_h0.45_L10.0 $M/until/H10_instance.npz 2.0,4.0,6.0,7.22 0.05,0.1 lse_plain,lse,sparsemax 0.2
uv run python -m experiments.e039_person_zone setup $M/starts20 data/visitH20.json visit_h0.45_L100.0 $M/until/H20_instance.npz 17.22 0.05,0.1 $A 0.2 H=20
uv run python -m experiments.e039_person_zone setup $M/starts40 data/visitH40.json visit_h0.45_L100.0 $M/until/H40_instance.npz 37.22 0.05,0.1 $A 0.2 H=40
uv run python -m experiments.e039_person_zone setup $M/starts80 data/visitH40.json visit_h0.45_L100.0 $M/until/H80_instance.npz 77.22 0.05,0.1 sparsemax 0.2 H=80

# values and weights (JAX, float64)
for H in 10 20 40 80; do
  uv run python -m experiments.e045_two_properties theory $H $M/until/H${H}_instance.npz $M/until/h$H
  uv run python -m experiments.e045_two_properties cpu $H $M/until/h$H
done

# GPU: the torque gradient through the plant (Warp, float32)
JAX_PLATFORMS=nojax uv run python -m experiments.e045_two_properties gpu $M/until/gpu.csv \
  10:$M/until/H10_instance.npz:$M/until/h10 20:$M/until/H20_instance.npz:$M/until/h20 \
  40:$M/until/H40_instance.npz:$M/until/h40 80:$M/until/H80_instance.npz:$M/until/h80

uv run python experiments/table_manipulator.py $M/until/gpu.csv $M/until/h10 $M/until/h20 $M/until/h40 $M/until/h80
```

The last command prints Table II. With `-` in place of `gpu.csv` it prints the table without the two columns of the torque gradient, from the CPU stages alone.

### The optimization on the torques (Table III)

```bash
export JAX_PLATFORMS=cpu
M=out/manipulator
A=lse_plain,lse,sparsemax,gm_pm01,gm_pm10,gm_exp

mkdir -p $M/starts16 $M/check $M/grid $M/starts16_20 $M/starts16_40 $M/regime20 $M/regime40 $M/check20 $M/check40 $M/tables

# 16 pick and handover locations, two violating initial trajectories each, at 10 s; the person's hand per location
uv run python -m experiments.e037_instances starts $M/starts16
uv run python -m experiments.e037_instances regime $M/starts16 $M/starts16
uv run python -m experiments.e037_person check $M/starts16 $M/starts16/regime.csv $M/check
for k in 0 1 2 3 4 5 6 7; do
  uv run python -m experiments.e037_person setup $M/starts16 $M/check/choice.csv $M/grid/p_b${k}_instance.npz $((2*k)),$((2*k+1)) $A
done
for k in 0 1 2 3; do
  uv run python -m experiments.e042_instances merge $M/grid/t1_b${k}_instance.npz $M/grid/p_b$((2*k))_instance.npz $M/grid/p_b$((2*k+1))_instance.npz
done

# the same locations at 20 and 40 s
for H in 20 40; do
  uv run python -m experiments.e042_instances starts $H $M/starts16 $M/starts16_$H
  uv run python -m experiments.e042_instances regime $H $M/starts16_$H $M/regime$H
  uv run python -m experiments.e042_instances check $H $M/starts16_$H $M/regime$H/regime.csv $M/check$H
done
uv run python -m experiments.e042_instances setup 20 $M/starts16_20 $M/check20/choice.csv $M/grid/h20_b0_instance.npz 0,1,2,3,4,5,6,7 $A 0.2
uv run python -m experiments.e042_instances setup 20 $M/starts16_20 $M/check20/choice.csv $M/grid/h20_b1_instance.npz 8,9,10,11,12,13,14,15 $A 0.2
uv run python -m experiments.e042_instances setup 40 $M/starts16_40 $M/check40/choice.csv $M/grid/h40_a0_instance.npz 0,1,2,3,4,5,6,7 lse,sparsemax,gm_pm10,gm_pm01 0.2
uv run python -m experiments.e042_instances setup 40 $M/starts16_40 $M/check40/choice.csv $M/grid/h40_a1_instance.npz 8,9,10,11,12,13,14,15 lse,sparsemax,gm_pm10,gm_pm01 0.2
uv run python -m experiments.e042_instances setup 40 $M/starts16_40 $M/check40/choice.csv $M/grid/h40_a2_instance.npz 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 lse_plain,gm_exp 0.2

# GPU: 100 updates per optimization, four constraints (Warp, no JAX)
export JAX_PLATFORMS=nojax
for b in t1_b0 t1_b1 t1_b2 t1_b3; do uv run python -m experiments.e042_instances run 10 conj $M/grid/${b}_instance.npz 100 $M/grid/$b.npz; done
for b in h20_b0 h20_b1; do uv run python -m experiments.e042_instances run 20 conj $M/grid/${b}_instance.npz 100 $M/grid/$b.npz; done
for b in h40_a0 h40_a1 h40_a2; do uv run python -m experiments.e042_instances run 40 conj $M/grid/${b}_instance.npz 100 $M/grid/$b.npz; done

uv run python experiments/table_outcomes.py $M/tables $M/grid
```

The last command writes `tables/table_outcomes.md`, whose rows hold the entries of Table III: the first update with a nonnegative exact robustness (`i_sat`), the first update at which the solver's own constraints hold (`i_acc`), the exact robustness after 100 updates and the effort relative to the initial trajectory, as medians with quartiles over the 16 locations.

The paper's runs took about 17 minutes per file at 10 s, 29 minutes at 20 s and 55 minutes at 40 s on one NVIDIA RTX PRO 6000 Blackwell Max-Q, 4.8 hours for the nine files. The 40 s files are split as above because one file with all six measures on eight locations did not fit the memory of that GPU (97887 MiB).

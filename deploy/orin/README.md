# NVIDIA Jetson AGX Orin

The Warp backend runs on the board without JAX. Every command runs from the repository root.

## Install

```bash
./deploy/orin/setup.sh
```

The script installs the base dependencies of `uv.lock` with `uv sync --locked --no-dev` (NumPy, Warp, MuJoCo and MuJoCo Warp, 12 packages with their dependencies), fetches the Franka Panda model, and runs `deploy/orin/check.py`. That prints the name of the GPU and, for each measure, the robustness of one until specification and the gradient weight on its violating samples:

```
Orin
exact -0.05 1.0
lse_plain -0.002631 0.9378
lse -0.188309 0.9378
gm_pm01 -0.124832 0.6811
gm_pm10 -0.042715 0.8543
gm_exp -0.041748 0.2238
sparsemax -0.090826 1.0
```

Tested on a Jetson AGX Orin 64 GB with L4T R39.2.1, CUDA 13.2 and uv 0.12.12. The Warp 1.17.0 wheel of PyPI is built with the CUDA 12.9 toolkit and runs under the board's CUDA 13.2 driver. The wheel `warp_lang-1.17.0+cu13` of Warp's GitHub release starts the GPU there as well; the run in the paper used it.

## The manipulator optimization

Ten updates of the optimization of Section V-B of the paper with one constraint per conjunct, from eight violating initial trajectories of the 10 s task under SparseMax at $\varepsilon = 0.2$. The trajectories are in `examples/data/manipulator_H10_w2.npz` (the pick window opens at 2 s; exact robustness $-0.05$ and $-0.10$), and `examples/manipulator.ipynb` runs the same eight on a workstation.

```bash
uv run --locked --no-dev python deploy/orin/optimize.py
```

The script prints the exact robustness of the eight trajectories before the first update and after each one, from a float64 replay with MuJoCo, then the name of the GPU and the median seconds per update by component. After ten updates the exact robustness of the eight trajectories is between 0.17 and 0.20. The number of updates is an optional argument, a multiple of ten.

## Measured

One Jetson AGX Orin 64 GB against one workstation GPU (NVIDIA RTX PRO 6000 Blackwell Max-Q), the same environment without JAX on both. A process of another project was resident on the board's GPU; it showed no load before the runs, and its share during them could not be separated.

| | Workstation | Jetson AGX Orin |
|---|---|---|
| the optimization above, seconds per update | 6.7 | 9.4 |
| of which the simulation, its reverse pass and the trial simulations (`rollout`, `plant_backward`, `trial_rollout`) | 1.18, 4.16, 1.28 | 1.53, 5.90, 1.70 |
| of which the specification and its gradient (`stl`) | 0.02 | 0.11 |

On the two machines the exact robustness differs by $2 \times 10^{-8}$ after the first update and by at most $1.2 \times 10^{-5}$ over the ten, and has the same sign after every update.

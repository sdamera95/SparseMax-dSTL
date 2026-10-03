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

The scene file is written by two CPU stages that use the JAX backend. Make it on a machine with the full install and copy it to the board:

```bash
mkdir -p out/starts10 out/until
JAX_PLATFORMS=cpu uv run python -m examples.e034_until_demo starts out/starts10
JAX_PLATFORMS=cpu uv run python -m examples.e039_person_zone setup out/starts10 examples/data/visit1.json visit_h0.45_L10.0 out/until/H10_instance.npz 2.0,4.0,6.0,7.22 0.05,0.1 lse_plain,lse,sparsemax 0.2
```

On the board, ten updates of the optimization with one constraint per conjunct, on eight violating initial trajectories of the 10 s task under SparseMax at $\varepsilon = 0.2$ (runs 16 to 23 of the file, exact robustness $-0.05$ and $-0.10$):

```bash
uv run --locked --no-dev python -m examples.e040_conj run H10_instance.npz 10 out.npz 16,17,18,19,20,21,22,23
```

Each update prints the smoothed robustness and the exact robustness of every trajectory, the latter from a float64 replay with MuJoCo. After ten updates the exact robustness of the eight trajectories is between 0.17 and 0.20.

## Measured

One Jetson AGX Orin 64 GB against one workstation GPU (NVIDIA RTX PRO 6000 Blackwell Max-Q), the same commit and the same environment without JAX on both. A process of another project was resident on the board's GPU; it showed no load before the runs, and its share during them could not be separated.

| | Workstation | Jetson AGX Orin |
|---|---|---|
| the manipulator optimization above, seconds per update | 6.7 | 9.4 |
| of which the simulation, its reverse pass and the trial simulations | 1.18, 4.16, 1.28 | 1.53, 5.90, 1.70 |
| of which the specification and its gradient | 0.02 | 0.11 |
| the unicycle optimization on the Warp chain (`examples/e049_al.py ... solve warp`), CPU, float64 | 183 s | 539 s |

The two machines agree as follows.

- Manipulator: the exact robustness differs by $2 \times 10^{-8}$ after the first update and by at most $1.2 \times 10^{-5}$ over the ten, and has the same sign on both machines after every update.
- Unicycle, 28 optimizations of 400 updates: the exact robustness of the returned trajectories differs by at most 0.003, and its sign is the same in 27. The one that differs is a GMR $(-10, 10)$ run at margin 0 whose exact robustness is within $4 \times 10^{-4}$ of zero on both machines.

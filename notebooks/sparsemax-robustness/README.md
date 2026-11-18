# SparseMax-∂STL: the paper's examples

The notebooks of [SparseMax-∂STL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization](https://arxiv.org/abs/2610.23837), stored with their outputs. `data/` holds the manipulator's initial trajectories: at horizons of 10, 20, 40 and 80 s, at its pick and handover locations (`_locations`), and with the pick window opening at 2 s (`_w2`).

| Notebook | In the paper | Runs on | Time |
|---|---|---|---|
| [unicycle.ipynb](unicycle.ipynb) | the planar unicycle (Section V-A): Table I and Fig. 4 | Warp, CPU, float64, no JAX | 2.5 minutes |
| [manipulator.ipynb](manipulator.ipynb) | the manipulator beside a person (Section V-B): Table II, and 20 of the 100 updates of Table III at four of its 16 pick and handover locations | Warp and MuJoCo Warp, no JAX; one CUDA GPU with 14 GB free, float32, the exact robustness from a float64 replay with MuJoCo | 5.5 minutes |
| [jax_backend.ipynb](jax_backend.ipynb) | the unicycle's specification in the JAX evaluator, and the optimization of Table I on the JAX backend against the Warp backend | JAX and Warp, CPU, float64 | 3.5 minutes |

From the repository root:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/sparsemax-robustness/unicycle.ipynb
```

The times are from a workstation with an AMD Ryzen 9 7950X and one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB).

In Table I the returned robustness is the JAX backend's; the Warp backend reproduces 31 of the table's 32 printed entries and returns $-0.0586$ where it prints $-0.058$, for the plain LSE at $c = 0$ (JAX: $-0.0584$). Runs on the GPU do not repeat bit for bit: two runs of the Table II computation differ by up to $6 \times 10^{-5}$ relative in the norm of the torque gradient, with the same printed table, and between two executions of the manipulator notebook the first satisfying update differed in one of the 40 runs (10 and 12).

The manipulator optimization on a Jetson AGX Orin, and the paper's measurements there, are in [deploy/orin](../../deploy/orin/README.md).

# Specification-aware sparse differentiation: the paper's experiments

The notebooks, scripts and measured results of "Specification-Aware Sparse Differentiation: Planning under Signal Temporal Logic with Sparsemax Robustness" (L4DC 2027 submission). Everything here runs on the CPU in double precision; with the development group of `uv sync` nothing more needs installing.

## The task

Two seven-joint arms share a hand-off zone divided into cells, balls of radius 0.03 m that one arm at a time may occupy. The specification has one until per scheduled visit, which keeps the other arm out of the cell until the visitor has left it, and one eventually per waypoint, which brings the hand to the waypoint by a deadline. At every control period the controller solves a nonlinear program over the next $T$ time steps of 0.02 s: the states and joint accelerations of both arms, the defects of the double integrator, the joint, velocity and acceleration limits, and one constraint per conjunct instance not yet decided, its smoothed robustness at least 0.02. `sparsemax_dstl.tasks.dual_arm_scenes` builds the three problems: the five-cell zone, the hold-up (the five-cell zone with the right arm held up and its schedule moved with it) and the 64-cell circle.

The comparators, on the same program up to the smoothing, with the same cost, bounds and IPOPT options:

| Name | What it is |
|---|---|
| SparseMax, our patterns | SparseMax in every constraint; the constraint Jacobian and the exact Lagrangian Hessian on the supports' structure, given to IPOPT through CasADi |
| Sound-LSE in CasADi | the sound log-sum-exp with the parameter matched to SparseMax's error bound; the program in CasADi's scalar expressions, whose sparsity CasADi detects and colors; CasADi's default evaluation; IPOPT with a quasi-Newton Hessian approximation (L-BFGS) or CasADi's exact Hessian |
| GMR in CasADi | the generalized-mean robustness [1], sound for every order (its sign is the sign of the exact robustness), as the power mean robustness of order $(0, 1)$, the setting of its authors' control example, with its equations as printed and the until nested as its authors define it; in every constraint at the margin, in the same program and IPOPT settings as Sound-LSE (its authors maximize it in the cost under a margin constraint) |
| dense AD in JAX | the whole constraint Jacobian by one reverse product per constraint, and the whole Lagrangian Hessian by one forward-over-reverse product per variable |

## Notebooks

Stored with their outputs. Each prints its values beside the paper's, read from `results/`, and writes its figures (SVG and PDF) to `out/`.

| Notebook | Computes | In the paper | Time and memory |
|---|---|---|---|
| [patterns.ipynb](patterns.ipynb) | the derivative products of dense AD, the prescribed pattern and the supports on both scenes; the patterns of the pattern figures | Table 2 (products), Figs. 3 and 6 | 15 min 30 s, 8.1 GiB |
| [derivatives.ipynb](derivatives.ipynb) | one constraint Jacobian and one Lagrangian Hessian, ours against CasADi's | Table 2 (times), Table 5, Fig. 2 (b) and (c) | 9 min 24 s, 7.6 GiB |
| [solves.ipynb](solves.ipynb) | single solves from the previous solution and from the reference motion, after a hold-up, over ten control periods, and on the 64-cell circle | Tables 1 and 3 | 40 min 16 s, 13.3 GiB |
| [replanning.ipynb](replanning.ipynb) | the re-planning loops after a hold-up, and the closed loop on a rigid-body simulation of both arms in MuJoCo with the exact robustness of every instance on the executed motion | Table 6, Fig. 2 (a) | 15 min 15 s, 4.7 GiB |

```bash
cd notebooks/sparse-differentiation
uv run jupyter nbconvert --to notebook --execute --inplace solves.ipynb
```

The times are one execution of each notebook on 8 cores of an AMD Ryzen 9 7950X.

## Scripts

The longest computations, each printing its value beside the paper's. Times of one run on 4 cores of the same machine:

| Script | Computes | One run |
|---|---|---|
| `scripts/counts.py <scene> <T>` | the derivative products over control periods (Table 2) | five-cell zone, $T = 100$: 68 s, 1.6 GiB |
| `scripts/solves.py <T> <U\|D2\|D5\|P\|lse\|gmr> <previous\|reference>` | one solve on the five-cell zone (Tables 1 and 3) | $T = 100$: 10 s and 1.4 GiB (Sound-LSE from the previous solution), 32 s and 1.9 GiB (ours, prescribed structure, from the reference motion) |
| `scripts/dense.py <scene> <T>` | dense AD of one control period's program (Table 2) | five-cell zone, $T = 100$: 39 s, 1.5 GiB |
| `scripts/casadi_exact_hessian.py <T> <hessian\|solve>` | CasADi's exact Lagrangian Hessian of Sound-LSE, or the solve with it (Tables 1 and 2) | `hessian`, $T = 100$: 6 min 17 s, 13.5 GiB |
| `scripts/loops.py <T> <P> <ours\|lse>` | one re-planning loop after a hold-up, a new plan every $P$ samples (Table 6) | $T = 100$, $P = 25$: 74 s and 3.3 GiB (ours), 45 s and 1.9 GiB (Sound-LSE) |

`figures/` redraws the paper's figures, and `data/` holds the 64-cell circle's executed motion that `replanning.ipynb` checks.

## Measured results

`results/` holds the measured numbers of every table and figure, one row per cell with the machine, the cores, the threads, the frequency governor and the statistic: `solves.csv` (Table 1), `counts.csv` (Table 2), `further_solves.csv` (Table 3), `spread.csv` (Table 4), `threads.csv` (Table 5), `replanning.csv` (Table 6), `orin.csv` (Table 7, the CPU of a Jetson AGX Orin 64 GB), `closed_loop.csv` (Fig. 2 (a)), `patterns.csv` (Fig. 3) and `products_horizon.csv` (Fig. 6).

The paper's times come from a workstation with an AMD Ryzen 9 7950X, one job at a time on 12 of its 16 cores, the frequency governor at performance. Our derivatives run on 12 JAX threads and CasADi's default evaluation on one thread; both sides call IPOPT 3.14.19 with MUMPS 5.8.2 on one thread through CasADi 3.8.1, and each control period's program is built before its timed solve.

## Reference

[1] N. Mehdipour, C.-I. Vasile and C. Belta, "Generalized mean robustness for signal temporal logic," IEEE Transactions on Automatic Control, vol. 70, no. 3, pp. 1949–1956, 2025.

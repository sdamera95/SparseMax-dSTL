"""E049 round 2: the Warp chain of the planar unicycle example, the counterpart of planar_al.JaxChain with
the same interface, so that planar_al's update runs on either.

The unicycle step (planar.rollout) is one Warp kernel over the runs, launched once per time step on a
wp.Tape; step t reads the state at sample t and the normalized command V[:, t], multiplies the command
by U_MAX inside the kernel (so the taped gradient is with respect to V) and writes sample t + 1 into its
own time slice of the state array. The time recursion is sequential, so the launch per step is a Python
loop over time steps, the labelled exception to the no-loop rule (as warp_plant does for the
manipulator). A second taped kernel writes the 8 leaf scores of planar_disk.scores from the positions.
Each conjunct is its own program (the conjunct as the root, read at t = 0) with its own
stl.warp_backend.Evaluator; its gradient with respect to the scores is fed to tape.backward and the
gradient of V is read as C_c = d r_c / d V. The constants h and U_MAX and the regions are passed to the
kernels as arrays of the run's dtype, so float64 runs see the float64 values (a Python float captured by
a kernel would be a float32 constant).

Nothing here loads MuJoCo; constrained_warp, constrained_conj, warp_plant and warp_predicates are not
imported.
"""
import time
from typing import Any

import numpy as np
import warp as wp

from ..stl import compile_formula
from ..stl.warp_backend import Evaluator, matched_param
from . import planar_disk as D

NPRED = 8


@wp.kernel
def _step(Z: wp.array3d(dtype=Any), V: wp.array3d(dtype=Any), c: wp.array(dtype=Any), t: int):
    # c = (h, v_max, w_max); planar.rollout: x+ = x + h v cos th, y+ = y + h v sin th, th+ = th + h w
    b = wp.tid()
    x = Z[b, t, 0]
    y = Z[b, t, 1]
    th = Z[b, t, 2]
    v = V[b, t, 0] * c[1]
    w = V[b, t, 1] * c[2]
    Z[b, t + 1, 0] = x + c[0] * v * wp.cos(th)
    Z[b, t + 1, 1] = y + c[0] * v * wp.sin(th)
    Z[b, t + 1, 2] = th + c[0] * w


@wp.kernel
def _scores(Z: wp.array3d(dtype=Any), R: wp.array2d(dtype=Any), S: wp.array3d(dtype=Any)):
    # planar_disk.scores: r_i - |p - c_i| for the four disks, then x, 10 - x, y, 10 - y
    b, t = wp.tid()
    x = Z[b, t, 0]
    y = Z[b, t, 1]
    for i in range(4):  # the four regions, unrolled by Warp
        dx = x - R[i, 0]
        dy = y - R[i, 1]
        S[b, t, i] = R[i, 2] - wp.sqrt(dx * dx + dy * dy)
    ten = type(x)(10.0)
    S[b, t, 4] = x
    S[b, t, 5] = ten - x
    S[b, t, 6] = y
    S[b, t, 7] = ten - y


class WarpChain:
    """The unicycle rollout and the disk predicates as taped Warp kernels and the Warp STL evaluator,
    one program per conjunct (each the conjunct as its own root, read at t = 0). Interface and
    semantics of planar_al.JaxChain: forward(V) -> r (n, K) keeping C (n, K, T - 1, 2) = d r_c / d V;
    pullback(w) -> sum_c w_c C_c; values(Va) -> r (n, lanes, K); times (seconds). method is a
    Warp-backend name ('lse_plain', 'gm_pm01', 'gm_pm10', 'sparsemax', 'exact')."""

    def __init__(self, conjuncts, T, method, eps, z0, regions, n, trials=6, device="cpu", dtype=wp.float64):
        self.device = wp.get_device(device)
        self.dtype = dtype
        self.n, self.T, self.lanes, self.K = n, T, trials + 1, len(conjuncts)
        L = self.lanes
        progs = [compile_formula(c, T, reads=[(c, [0])]) for c in conjuncts]  # over the conjuncts
        params = [None if method == "exact" else matched_param(p, method, eps) for p in progs]
        npdt = wp.dtype_to_numpy(dtype)
        self.z0 = np.asarray(z0, npdt)
        dev = self.device
        self.c = wp.array(np.array([D.H, D.U_MAX[0], D.U_MAX[1]], npdt), dtype=dtype, device=dev)
        self.R = wp.array(np.asarray(regions, npdt), dtype=dtype, device=dev)
        # gradient buffers (n runs) and value buffers (n * lanes runs)
        self.Z = wp.zeros((n, T, 3), dtype=dtype, device=dev, requires_grad=True)
        self.V = wp.zeros((n, T - 1, 2), dtype=dtype, device=dev, requires_grad=True)
        self.S = wp.zeros((n, T, NPRED), dtype=dtype, device=dev, requires_grad=True)
        self.Zt = wp.zeros((n * L, T, 3), dtype=dtype, device=dev)
        self.Vt = wp.zeros((n * L, T - 1, 2), dtype=dtype, device=dev)
        self.St = wp.zeros((n * L, T, NPRED), dtype=dtype, device=dev)
        self.ev = [Evaluator(p, method, q, n, dtype, dev, P=NPRED, graphs=False) for p, q in zip(progs, params)]
        self.ev_t = [Evaluator(p, method, q, n * L, dtype, dev, P=NPRED, graphs=False) for p, q in zip(progs, params)]
        self.C = None
        self.times = {}

    def _load(self, Z, V, V_np):
        """Puts the start state into sample 0 of Z and the commands V_np (B, T - 1, 2) into V (off the tape)."""
        npdt = wp.dtype_to_numpy(self.dtype)
        Z0 = np.zeros((len(V_np), self.T, 3), npdt)
        Z0[:, 0] = self.z0
        Z.assign(Z0)
        V.assign(np.ascontiguousarray(V_np, dtype=npdt))

    def _rollout(self, Z, V, S):
        """The states into Z and the scores into S, on the active tape if any."""
        B = Z.shape[0]
        for t in range(self.T - 1):  # over time steps: the sequential recursion (labelled exception)
            wp.launch(_step, dim=B, inputs=[Z, V, self.c, t], device=self.device)
        wp.launch(_scores, dim=(B, self.T), inputs=[Z, self.R], outputs=[S], device=self.device)

    def forward(self, V):
        t0 = time.perf_counter()
        self._load(self.Z, self.V, V)
        tape = wp.Tape()
        with tape:
            self._rollout(self.Z, self.V, self.S)
        r = np.empty((self.n, self.K))
        C = []
        for c, ev in enumerate(self.ev):  # over the conjuncts (formula structure)
            rho, G = ev.gradient(self.S)
            r[:, c] = rho.numpy()[:, 0]
            # a new tape's zero() does not reach arrays it has not yet run backward on, so the
            # gradients of the taped arrays are cleared here (else the last forward's would add in)
            self.Z.grad.zero_()
            self.V.grad.zero_()
            tape.backward(grads={self.S: G})
            C.append(self.V.grad.numpy().copy())  # on the CPU numpy() is a view of the buffer
        self.C = np.stack(C, 1)
        self.times["forward"] = self.times.get("forward", 0.0) + time.perf_counter() - t0
        return r

    def pullback(self, w):
        return np.einsum("nk,nkij->nij", np.asarray(w), self.C)

    def values(self, Va):
        t0 = time.perf_counter()
        n, L = Va.shape[:2]
        self._load(self.Zt, self.Vt, Va.reshape((n * L,) + Va.shape[2:]))
        self._rollout(self.Zt, self.Vt, self.St)
        out = np.stack([ev.value(self.St).numpy()[:, 0] for ev in self.ev_t], -1)  # over the conjuncts
        self.times["values"] = self.times.get("values", 0.0) + time.perf_counter() - t0
        return out.reshape(n, L, self.K)

"""The Warp chain of the planar unicycle example: the rollout and the disk predicates as taped Warp kernels with the
Warp STL evaluator, with the interface of planar_al_jax.JaxChain."""
import time
from typing import Any

import numpy as np
import warp as wp

from ..stl import compile_formula
from ..warp.evaluator import Evaluator, matched_param
from . import planar_disk as D

NPRED = 8


@wp.kernel
def _step(Z: wp.array3d(dtype=Any), V: wp.array3d(dtype=Any), c: wp.array(dtype=Any), t: int):
    # c = (h, v_max, w_max); V holds the commands divided by their limits
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
    # R[i] = (centre x, centre y, radius): r_i - |p - c_i| for the four disks, then x, 10 - x, y, 10 - y
    b, t = wp.tid()
    x = Z[b, t, 0]
    y = Z[b, t, 1]
    for i in range(4):
        dx = x - R[i, 0]
        dy = y - R[i, 1]
        S[b, t, i] = R[i, 2] - wp.sqrt(dx * dx + dy * dy)
    ten = type(x)(10.0)
    S[b, t, 4] = x
    S[b, t, 5] = ten - x
    S[b, t, 6] = y
    S[b, t, 7] = ten - y


class WarpChain:
    """Taped Warp kernels of the rollout and the disk predicates and the Warp STL evaluator, one program per conjunct;
    forward(V) -> r at t = 0, keeping C = d r_c / d V; pullback(w) -> sum_c w_c C_c; values(Va) -> r per lane."""

    def __init__(self, conjuncts, T, method, eps, z0, regions, n, trials=6, device="cpu", dtype=wp.float64):
        self.device = wp.get_device(device)
        self.dtype = dtype
        self.n, self.T, self.lanes, self.K = n, T, trials + 1, len(conjuncts)
        L = self.lanes
        progs = [compile_formula(c, T, reads=[(c, [0])]) for c in conjuncts]
        params = [None if method == "exact" else matched_param(p, method, eps) for p in progs]
        npdt = wp.dtype_to_numpy(dtype)
        self.z0 = np.asarray(z0, npdt)
        dev = self.device
        # constants as arrays of the run's dtype; a Python float captured by a kernel is a float32 constant
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
        for t in range(self.T - 1):  # step t writes sample t + 1
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
        for c, ev in enumerate(self.ev):
            rho, G = ev.gradient(self.S)
            r[:, c] = rho.numpy()[:, 0]
            # backward adds to the gradients, so those of the previous conjunct and call are cleared
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
        out = np.stack([ev.value(self.St).numpy()[:, 0] for ev in self.ev_t], -1)
        self.times["values"] = self.times.get("values", 0.0) + time.perf_counter() - t0
        return out.reshape(n, L, self.K)

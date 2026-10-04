"""The JAX chain of the planar unicycle example, the counterpart of planar_warp.WarpChain with the same interface."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import evaluate, read
from ..stl import compile_formula
from . import planar_disk as D
from . import planar_disk_jax as Dj
from . import planar_jax as P0
from .planar_al import Update


class JaxChain:
    """The unicycle rollout, the disk predicates and the JAX STL evaluator, one program per conjunct;
    forward(V) -> r at t = 0, keeping C = d r_c / d V; pullback(w) -> sum_c w_c C_c; values(Va) -> r per lane."""

    def __init__(self, conjuncts, T, method, eps, z0, regions, n, trials=Update.TRIALS):
        self.n, self.T, self.lanes, self.K = n, T, trials + 1, len(conjuncts)
        progs = [compile_formula(c, T, reads=[(c, [0])]) for c in conjuncts]
        sem = Dj.matched(method, eps)
        z0 = jnp.asarray(z0, jnp.float64)
        umax = jnp.asarray(D.U_MAX)

        def r_one(V):
            z = P0.rollout(z0, V * umax)
            S = Dj.scores(z[:, :2], regions)
            return jnp.stack([read(p, evaluate(p, S, sem))[0] for p in progs])

        self._val = jax.jit(jax.vmap(r_one))
        self._fwd = jax.jit(jax.vmap(lambda V: (r_one(V), jax.jacrev(r_one)(V))))
        self.C = None
        self.times = {}

    def forward(self, V):
        t0 = time.perf_counter()
        r, C = self._fwd(jnp.asarray(V))
        self.C = np.asarray(C)
        self.times["forward"] = self.times.get("forward", 0.0) + time.perf_counter() - t0
        return np.asarray(r)

    def pullback(self, w):
        return np.einsum("nk,nkij->nij", np.asarray(w), self.C)

    def values(self, Va):
        t0 = time.perf_counter()
        n, L = Va.shape[:2]
        out = np.asarray(self._val(jnp.asarray(Va.reshape((n * L,) + Va.shape[2:])))).reshape(n, L, self.K)
        self.times["values"] = self.times.get("values", 0.0) + time.perf_counter() - t0
        return out

"""Predicates: differentiable score functions of one state, positive when satisfied.

A predicate declares the state coordinates its score depends on, A_mu, which
the specification-derived sparsity patterns use. The STL layer never sees the
functions, only their score traces, so any predicate composes with any
semantics through the chain rule.
"""
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Predicate:
    fn: Callable  # state (n_x,) -> scalar score
    deps: tuple  # state coordinates the score may depend on
    name: str = ""


def score_traces(predicates, X):
    """Scores of every predicate along a trajectory: (..., T, n_x) -> (..., T, P)."""
    return jnp.stack([jnp.vectorize(p.fn, signature="(n)->()")(X) for p in predicates], axis=-1)


def dependency_matrix(predicates, n_x):
    """Boolean (P, n_x) matrix of declared dependencies."""
    D = np.zeros((len(predicates), n_x), dtype=bool)
    for i, p in enumerate(predicates):
        D[i, list(p.deps)] = True
    return D


def undeclared_gradient(predicates, X):
    """Largest |d g / d x_j| over sampled states X (N, n_x) and undeclared j.

    A nonzero value refutes a declaration; zero on samples does not prove one.
    """
    D = dependency_matrix(predicates, X.shape[-1])
    worst = []
    for i, p in enumerate(predicates):
        g = jax.vmap(jax.grad(p.fn))(X)
        worst.append(jnp.max(jnp.abs(jnp.where(D[i], 0, g))))
    return jnp.max(jnp.stack(worst))

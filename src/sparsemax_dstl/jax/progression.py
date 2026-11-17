"""Evaluation of a compiled specification under per-step masks (the residual of sparsemax_dstl.stl.progression), the
sparsemax weights of its reduction steps and the operands near their thresholds. Batch axes come first in scores and masks."""
import jax.numpy as jnp

from .evaluator import reductions
from .operators import sparsemax_weights


def _window(step, values):
    src = values[step.sources[0]] if len(step.sources) == 1 else jnp.concatenate([values[s] for s in step.sources], -1)
    return src[..., step.index]


def evaluate(program, scores, masks, semantics="exact", param=None):
    """Step outputs in program order; masks per step None (atoms) or (..., rows, width) Boolean with at least one valid
    entry per row; semantics a name of REDUCTIONS or a pair (max_reduce, min_reduce)."""
    max_reduce, min_reduce = reductions(semantics)
    scores = jnp.asarray(scores)
    out = []
    for step, m in zip(program.steps, masks):
        if step.kind == "atom":
            out.append(step.sign * scores[..., step.atom])
            continue
        reduce = max_reduce if step.kind == "max" else min_reduce
        out.append(reduce(_window(step, out), param, jnp.asarray(m)))
    return out


def weights(program, values, masks, gamma):
    """Per step (None for atoms): the oriented window values zeta (z at a maximum, -z at a minimum), their sparsemax
    weights p under the mask, and the threshold theta = (sum of zeta over the support - gamma) / |support|."""
    out = []
    for step, m in zip(program.steps, masks):
        if step.kind == "atom":
            out.append(None)
            continue
        z = _window(step, values)
        zeta = z if step.kind == "max" else -z
        m = jnp.asarray(m)
        p = sparsemax_weights(zeta, gamma, m)
        on = p > 0
        theta = (jnp.sum(jnp.where(on, zeta, 0), -1) - gamma) / jnp.maximum(jnp.sum(on, -1), 1)
        out.append({"zeta": zeta, "p": p, "theta": theta})
    return out


def near(program, wts, masks, kappa):
    """Per step (None for atoms): the operands with zeta > theta - kappa under the mask, and the support."""
    out = []
    for step, w, m in zip(program.steps, wts, masks):
        if step.kind == "atom":
            out.append(None)
            continue
        out.append(jnp.asarray(m) & (w["zeta"] > w["theta"][..., None] - kappa) | (w["p"] > 0))
    return out

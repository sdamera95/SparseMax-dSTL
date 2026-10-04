"""The manipulator example on Warp and MuJoCo Warp: the solver's chain with one constraint per conjunct, the predicate
values on a float64 replay, and the torque gradient of the until conjunct through the plant (Table II)."""
import time

import numpy as np
import warp as wp

from ..warp import solver, solver_conjuncts
from ..warp.evaluator import Evaluator, matched_param
from ..warp.plant import Plant
from ..warp.predicates import Predicates
from . import panda
from . import workspace_program as Wp
from .workspace_scene import scenario


def conjunct_chain(I, single=False, device="cuda:0"):
    """The ConjChain with one constraint per conjunct (single: the whole specification as the one constraint) and the
    Referee for the runs of I."""
    n = len(I["x0"])
    waits = sorted({g[4] for g in I["groups"]})
    progs = {w: Wp.core_program(scenario(I["H"], w), I["n_h"]) for w in waits}
    cps = {w: ((progs[w],) if single else Wp.conj_programs(scenario(I["H"], w), I["n_h"])) for w in waits}
    sc = scenario(I["H"], min(progs))
    groups = [(m_, e, a, b, cps[w]) for m_, e, a, b, w in I["groups"]]
    blocks = [(a, b, progs[w]) for _, _, a, b, w in I["groups"]]
    pick, hand = np.broadcast_to(I["pick"], (n, 3)), np.broadcast_to(I["handover"], (n, 3))
    chain = solver_conjuncts.ConjChain(Wp.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], groups, device=device)
    referee = solver.Referee(Wp.plant, sc, progs[min(progs)], I["x0"], pick, hand, I["hc"], I["hr"], device=device, blocks=blocks)
    return chain, referee


def replay_predicates(I, sc, device="cuda:0"):
    """The predicate values (n, T, P) in float64 on the float64 MuJoCo replay of the torques of I: with the smoothed
    norms, as the smoothed measures read them, and with the exact norms."""
    n = len(I["V0"])
    X = panda.replay(I["V0"], I["x0"][0].astype(np.float64))
    pred = Predicates(Wp.plant, sc, sc.samples, nworld=n, dtype=wp.float64, device=device)
    pred.set_instance(I["pick"], I["handover"], I["hc"], I["hr"])
    q = wp.array(np.ascontiguousarray(X[..., :7].reshape(-1, 7)), dtype=wp.float64, device=device)
    v = wp.array(np.ascontiguousarray(X[..., 7:].reshape(-1, 7)), dtype=wp.float64, device=device)
    smoothed = pred.scores(q, v).numpy()
    pred.consts[-2:] = [wp.float64(0.0), wp.float64(0.0)]
    return smoothed, pred.scores(q, v).numpy()


def until_torque_gradient(I, sc, prog, kw, viol, settings, device="cuda:0"):
    """Through the plant in float32, per (measure, eps) of settings and trajectory of I: the until conjunct's value and its
    weight on the samples viol (n, T), and the norm of its torque gradient, whole and through those samples alone."""
    t0 = time.perf_counter()
    S = settings
    nc = 2 * len(S)  # cotangents: per setting the whole score gradient and its violating part
    n, T = viol.shape
    plant = Plant(n * nc, T - 1, device=device)
    Xw = plant.rollout(np.tile(I["x0"], (nc, 1)), np.tile(I["V0"], (nc, 1, 1)))
    copies = float(np.abs(Xw.reshape(nc, n, T, -1) - Xw[None, :n]).max())
    t_roll = time.perf_counter() - t0
    pred = Predicates(Wp.plant, sc, T, nworld=n, device=device)
    pred.set_instance(np.broadcast_to(I["pick"], (n, 3)), np.broadcast_to(I["handover"], (n, 3)), I["hc"], I["hr"])
    q = wp.array(np.ascontiguousarray(Xw[:n, :, :7].reshape(-1, 7)), dtype=float, device=device, requires_grad=True)
    v = wp.array(np.ascontiguousarray(Xw[:n, :, 7:].reshape(-1, 7)), dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    Z = pred.scores(q, v, tape)
    Zn = Z.numpy()
    sign_flips = np.sum(((Zn[:, :, 2] < 0) & (np.arange(T) <= kw[-1])[None]) != viol, 1)  # at the samples the until reads
    gZ = wp.zeros((n, T, pred.P), dtype=float, device=device)
    C = np.zeros((nc, n, T, 14), np.float32)
    only = viol[..., None] & (np.arange(pred.P) == 2)[None, None]
    vals, ws = [], []
    for s, (measure, e) in enumerate(S):  # over the settings (measures)
        ev = Evaluator(prog, measure, matched_param(prog, measure, 0.2 if np.isnan(e) else e), n, wp.float32, device, P=pred.P)
        r, G = ev.gradient(Z)
        Gn = G.numpy().copy()
        vals.append(r.numpy()[:, 0].copy())
        ws.append(np.sum(np.where(viol, Gn[:, :, 2], 0.0), 1))
        for h, Gs in enumerate((Gn, np.where(only, Gn, 0.0))):  # the whole gradient and its violating part
            gZ.assign(Gs.astype(np.float32))
            tape.backward(grads={Z: gZ})
            C[2 * s + h] = np.concatenate([q.grad.numpy(), v.grad.numpy()], -1).reshape(n, T, 14)
            tape.zero()
    t1 = time.perf_counter()
    gV, _ = plant.vjp(C.reshape(n * nc, T, 14))
    t_vjp = time.perf_counter() - t1
    norm = np.sqrt(np.sum(gV.astype(np.float64) ** 2, (1, 2))).reshape(nc, n)
    return {"value32": np.stack(vals), "w_viol32": np.stack(ws), "grad_norm": norm[0::2], "grad_norm_viol": norm[1::2],
            "sign_flips_f32": sign_flips, "copies_max_diff": copies, "rollout_s": t_roll, "vjp_s": t_vjp}

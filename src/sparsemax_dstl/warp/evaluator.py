"""Exact and lower log-sum-exp STL robustness in Warp, with hand-written adjoints.

Evaluates the same Program as semantics.evaluate. All step outputs live in one
flat buffer vals of shape (B, N); step v writes vals[:, offsets[v]:offsets[v] +
length_v]. A reduction step reads its entries through a global address table
built on the host, addr[r, k] = position in vals of window entry k of row r.

Warp's generated adjoints are wrong for reductions written as dynamic loops
(the loop mutates an accumulator, which its codegen flags as possibly not
differentiable), so no kernel in this module is recorded on a tape. Each
forward launch is followed by tape.record_func with a hand-written adjoint
kernel; the tape replays those closures in reverse program order, so the
adjoint of a step's output is complete before that step's adjoint runs.

With y = s z (s = 1 for a maximum, s = -1 for a minimum), c = max_k y_k, and m
valid entries, a reduction returns

    exact      s c
    lse        s (c + log sum_k exp(beta (y_k - c)) / beta) - [max] log(m) / beta
    lse_plain  s (c + log sum_k exp(beta (y_k - c)) / beta)

lse_plain is the plain log-sum-exp in common use: the same as lse at a minimum,
and without the shift log(m) / beta at a maximum. Its maximum lies between the
exact maximum and the exact maximum plus log(m) / beta, so it is not a lower
bound and can report a violated specification as satisfied.

Adjoint weights: exact splits the output adjoint equally among all valid
entries equal to the extremum (the jnp.max / jnp.min convention); lse and
lse_plain use w_k = exp(beta (y_k - c)) / sum_j exp(beta (y_j - c)) (the shift
is a constant, so the two have the same weights).

sparsemax (param gamma) computes M_gamma(y) = c + theta + gamma/2 sum_k p_k^2,
with p_k = (y_k - c - theta)_+ / gamma and sum_k (y_k - c - theta)_+ = gamma,
and returns M + gamma/(2m) for a maximum and -M(-z) - gamma/2 for a minimum
(operators.lower_max and lower_min). The threshold theta of u_k = y_k - c solves
g(theta) = gamma for g(tau) = sum_k (u_k - tau)_+, which decreases from
g(-gamma) >= gamma to g(0) = 0. _threshold halves the bracket [-gamma, 0]
BISECTIONS times, each a pass over the node's values, and then recomputes theta
in closed form: with S the entries above the bracket's midpoint, theta = (sum_S
u - gamma) / |S|, and S is recounted as the entries above theta (later recounts
only raise the cut) until its size repeats, so that S = {k : u_k > theta}. The
sum runs over S in index order, as the search below summed it, so where the two
supports agree theta equals the search's bit for bit (E028). A pass costs O(m),
the threshold O(m (BISECTIONS + recounts)). _threshold_search, the O(m^2)
search this module used before E028, stays as a brute-force test oracle: entry
i is in the support exactly when gamma + k_i u_i > S_i, where k_i and S_i
count and sum the entries u_j >= u_i. The adjoint returns the weights p_k
directly and never differentiates the threshold.

gm_pm01 and gm_exp (E036) are the generalized mean robustness (Mehdipour,
Vasile, Belta, IEEE TAC 2025, eq. 12, 13, 15, 17; stl.semantics states the
definition and the numerics, which _gm_forward follows step for step). With
u_k = -s z_k a reduction returns -s conj(u) (a minimum is the conjunction, a
maximum its De Morgan dual), and on the inner step of an Until (label ending
".inner") it returns -s conj2(u_0, conj(u_1, ..., u_{n-1})), eq. 15's nesting:
u_0 is psi at the witness and u_1.. the closed prefix of phi. gm_pm01 and
gm_pm10 are the power mean robustness of order (p, q) = (0, 1) and (-10, 10)
(ORDER; code 4, any p, q >= 1), with no parameter (any positive value; 0 per row
still means exact). The parameter of gm_exp (code 5) is the per-node error eps,
not beta: each conjunction of m > 1 entries uses beta = log(m) / eps (the prefix
m = n - 1, the pair m = 2). Adjoint (_gm_adjoint): d out / d z_k = d conj / d u_k,
which is (1/m) (u_k / M_p)^(p - 1) in the first branch of the power mean
(G / (m u_k) for the geometric mean G), (1/m) (v_k / M_q)^(q - 1) on each
negative entry of the second (1/m at q = 1), v = -[u]_-, the softmin weights
exp(-beta (u_k - min u)) / sum for F_c, and exp(beta v_k) / sum_j exp(beta v_j)
on each negative entry for -F_g(v); the nested row multiplies the prefix's
weights by d conj2 / d prefix. Warp 1.17 has no
expm1 or log1p; _expm1 and _log1p are Kahan's formulas from exp and log.

Parameters (E028 follow-up). param is one positive number for every reduction
node, or a 1-D array with one value per reduction row of the program (the rows
of the reduction steps in program order, matched_param builds the matched
protocol's); a row whose value is 0 is evaluated exactly, as the matched
protocol evaluates nodes of arity 1. The address tables, the step offsets and
uploaded per-row parameter arrays are built once per program and device and
kept while the program object lives (rebuilt for a new program object); the
first call with a new program or parameter array uploads them from the host, so
it must run outside a CUDA graph capture, and a parameter array must not be
changed in place after its first use. All atom steps run as one launch (and one
adjoint launch), before the reductions.
Evaluator holds every buffer of one program, parameter, batch size, dtype and
device, and on a CUDA device replays the forward and the forward-plus-backward
launch sequences as CUDA graphs.
"""
import weakref
from functools import partial
from typing import Any

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

SEMANTICS = ("exact", "lse", "sparsemax", "lse_plain", "gm_pm01", "gm_pm10", "gm_exp")
SMOOTH = {"exact": 0, "lse": 1, "sparsemax": 2, "lse_plain": 3, "gm_pm01": 4, "gm_pm10": 4, "gm_exp": 5}
GM = (4, 5)
ORDER = {"gm_pm01": (0.0, 1.0), "gm_pm10": (-10.0, 10.0)}  # (p, q) of the power mean settings (code 4)
BISECTIONS = wp.constant(40)


@wp.kernel
def _atoms_forward(scores: wp.array3d(dtype=Any), atom: wp.array(dtype=wp.int32), sign: wp.array(dtype=wp.int32),
                   aoff: wp.array(dtype=wp.int32), vals: wp.array2d(dtype=Any)):
    # every atom step in one launch: step a writes sign_a * scores[:, :, atom_a] at its offset
    b, a, t = wp.tid()
    v = scores[b, t, atom[a]]
    vals[b, aoff[a] + t] = type(v)(sign[a]) * v


@wp.kernel
def _atoms_adjoint(vals_grad: wp.array2d(dtype=Any), atom: wp.array(dtype=wp.int32), sign: wp.array(dtype=wp.int32),
                   aoff: wp.array(dtype=wp.int32), scores_grad: wp.array3d(dtype=Any)):
    b, a, t = wp.tid()
    g = vals_grad[b, aoff[a] + t]
    wp.atomic_add(scores_grad, b, t, atom[a], type(g)(sign[a]) * g)


@wp.func
def _threshold(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), b: int, r: int, n: int, s: Any,
               c: Any, gamma: Any):
    # sparsemax threshold of u_k = y_k - c, where c = max_k y_k: bisection on
    # g(tau) = sum_k (u_k - tau)_+ over [-gamma, 0], then the exact recompute
    zero = type(gamma)(0.0)
    half = type(gamma)(0.5)
    lo = -gamma
    hi = zero
    for it in range(BISECTIONS):
        mid = half * (lo + hi)
        g = zero
        for k in range(n):
            g += wp.max(s * vals[b, addr[r, k]] - c - mid, zero)
        if g > gamma:
            lo = mid
        else:
            hi = mid
    # theta_S = (sum_S u - gamma) / |S| is at most the threshold for every nonempty S,
    # so the first recount, at theta of the midpoint's set, contains the support; later
    # recounts only raise the cut and stop when the size repeats
    tau = half * (lo + hi)
    theta = tau
    prev = int(-1)
    for it in range(n + 2):
        size = int(0)
        total = zero
        for k in range(n):
            u = s * vals[b, addr[r, k]] - c
            if u > tau:
                size += 1
                total += u
        theta = (total - gamma) / type(gamma)(size)
        if size == prev:
            break
        if prev < 0:
            tau = theta
        else:
            tau = wp.max(tau, theta)
        prev = size
    return theta


@wp.func
def _threshold_search(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), b: int, r: int, n: int,
                      s: Any, c: Any, gamma: Any):
    # brute-force test oracle, O(n^2): the threshold search of E006, kept for tests and timing
    size = int(0)
    total = type(gamma)(0.0)
    for i in range(n):
        yi = s * vals[b, addr[r, i]] - c
        cnt = int(0)
        acc = type(gamma)(0.0)
        for j in range(n):
            yj = s * vals[b, addr[r, j]] - c
            if yj >= yi:
                cnt += 1
                acc += yj
        if gamma + type(gamma)(cnt) * yi > acc:
            size += 1
            total += yi
    return (total - gamma) / type(gamma)(size)


@wp.kernel
def _reduce_forward(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                    off: int, s: Any, is_max: int, smooth: int, beta: Any, empty: Any, oracle: int,
                    params: wp.array(dtype=Any), roff: int, per_row: int):
    b, r = wp.tid()
    n = count[r]
    p = beta
    sm = smooth
    if per_row == 1:
        p = params[roff + r]
        if p == type(beta)(0.0):
            sm = 0  # parameter 0: the row is evaluated exactly
    c = empty
    for k in range(n):
        c = wp.max(c, s * vals[b, addr[r, k]])
    out = s * c
    if (sm == 1 or sm == 3) and n > 0:
        total = type(beta)(0.0)
        for k in range(n):
            total += wp.exp(p * (s * vals[b, addr[r, k]] - c))
        out = s * (c + wp.log(total) / p)
        if is_max == 1 and sm == 1:  # lse_plain (3) keeps the unshifted value
            out -= wp.log(type(beta)(n)) / p
    if sm == 2 and n > 0:
        theta = type(beta)(0.0)
        if oracle == 1:
            theta = _threshold_search(vals, addr, b, r, n, s, c, p)
        else:
            theta = _threshold(vals, addr, b, r, n, s, c, p)
        sq = type(beta)(0.0)
        for k in range(n):
            q = wp.max(s * vals[b, addr[r, k]] - c - theta, type(beta)(0.0)) / p
            sq += q * q
        out = s * (c + theta + p * sq / type(beta)(2.0))
        if is_max == 1:
            out += p / (type(beta)(2.0) * type(beta)(n))
        else:
            out -= p / type(beta)(2.0)
    vals[b, off + r] = out


@wp.kernel
def _reduce_adjoint(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                    off: int, s: Any, smooth: int, beta: Any, empty: Any, oracle: int,
                    params: wp.array(dtype=Any), roff: int, per_row: int, vals_grad: wp.array2d(dtype=Any)):
    b, r = wp.tid()
    n = count[r]
    p = beta
    sm = smooth
    if per_row == 1:
        p = params[roff + r]
        if p == type(beta)(0.0):
            sm = 0
    ybar = vals_grad[b, off + r]
    c = empty
    for k in range(n):
        c = wp.max(c, s * vals[b, addr[r, k]])
    if sm == 2:
        # custom adjoint: the sparsemax weights p_k, not the derivative of the threshold
        if n > 0:
            theta = type(beta)(0.0)
            if oracle == 1:
                theta = _threshold_search(vals, addr, b, r, n, s, c, p)
            else:
                theta = _threshold(vals, addr, b, r, n, s, c, p)
            for k in range(n):
                q = wp.max(s * vals[b, addr[r, k]] - c - theta, type(beta)(0.0)) / p
                if q > type(beta)(0.0):
                    wp.atomic_add(vals_grad, b, addr[r, k], ybar * q)
        return
    total = type(beta)(0.0)
    for k in range(n):
        y = s * vals[b, addr[r, k]]
        if sm == 1 or sm == 3:
            total += wp.exp(p * (y - c))
        elif y == c:
            total += type(beta)(1.0)
    for k in range(n):
        y = s * vals[b, addr[r, k]]
        if sm == 1 or sm == 3:
            wp.atomic_add(vals_grad, b, addr[r, k], ybar * wp.exp(p * (y - c)) / total)
        elif y == c:
            wp.atomic_add(vals_grad, b, addr[r, k], ybar / total)


@wp.func
def _expm1(x: Any):
    # exp(x) - 1 without cancellation near 0 (Kahan), used only for x <= 1
    u = wp.exp(x)
    one = type(x)(1.0)
    r = x
    if u != one:
        r = (u - one) * x / wp.log(u)
    return r


@wp.func
def _log1p(x: Any):
    # log(1 + x) without cancellation near 0 (Kahan), x >= 0
    one = type(x)(1.0)
    u = one + x
    r = x
    if u != one:
        r = wp.log(u) * x / (u - one)
    return r


@wp.func
def _gm_conj(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), b: int, r: int, k0: int, n: int, s: Any,
             code: int, eps: Any, pp: Any, qq: Any):
    # conjunction of u_k = -s vals[b, addr[r, k]] over k0 <= k < n (stl.semantics.gm_conj)
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    m = n - k0
    fm = type(eps)(m)
    lo = -s * vals[b, addr[r, k0]]
    for k in range(k0 + 1, n):
        lo = wp.min(lo, -s * vals[b, addr[r, k]])
    out = lo
    if m > 1:
        acc = zero
        if code == 4:
            if lo > zero:
                if pp == zero:
                    for k in range(k0, n):
                        acc += wp.log(-s * vals[b, addr[r, k]])
                    out = wp.exp(acc / fm)
                else:
                    c = pp * wp.log(-s * vals[b, addr[r, k0]])
                    for k in range(k0 + 1, n):
                        c = wp.max(c, pp * wp.log(-s * vals[b, addr[r, k]]))
                    for k in range(k0, n):
                        acc += wp.exp(pp * wp.log(-s * vals[b, addr[r, k]]) - c)
                    out = wp.exp((c + wp.log(acc / fm)) / pp)
            elif qq == one:
                for k in range(k0, n):
                    acc += wp.min(-s * vals[b, addr[r, k]], zero)
                out = acc / fm
            elif lo < zero:
                for k in range(k0, n):
                    if -s * vals[b, addr[r, k]] < zero:
                        acc += wp.exp(qq * wp.log(s * vals[b, addr[r, k]] / (-lo)))
                out = lo * wp.exp(wp.log(acc / fm) / qq)
            else:
                out = zero
        else:
            beta = wp.log(fm) / eps
            if lo > zero:
                for k in range(k0, n):
                    acc += wp.exp(-beta * (-s * vals[b, addr[r, k]] - lo))
                out = lo - wp.log(acc / fm) / beta
            else:
                c = -lo
                if beta * c <= one:
                    for k in range(k0, n):
                        acc += _expm1(beta * wp.max(s * vals[b, addr[r, k]], zero))
                    out = -_log1p(acc / fm) / beta
                else:
                    for k in range(k0, n):
                        acc += wp.exp(beta * (wp.max(s * vals[b, addr[r, k]], zero) - c))
                    out = -(c + wp.log(acc / fm) / beta)
    return out


@wp.func
def _gm_conj2(x0: Any, x1: Any, code: int, eps: Any, pp: Any, qq: Any):
    # conjunction of the two entries (x0, x1), m = 2
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    two = type(eps)(2.0)
    lo = wp.min(x0, x1)
    out = zero
    if code == 4:
        if lo > zero:
            if pp == zero:
                out = wp.exp((wp.log(x0) + wp.log(x1)) / two)
            else:
                l0 = pp * wp.log(x0)
                l1 = pp * wp.log(x1)
                c = wp.max(l0, l1)
                out = wp.exp((c + wp.log((wp.exp(l0 - c) + wp.exp(l1 - c)) / two)) / pp)
        elif qq == one:
            out = (wp.min(x0, zero) + wp.min(x1, zero)) / two
        elif lo < zero:
            t = zero
            if x0 < zero:
                t += wp.exp(qq * wp.log(x0 / lo))
            if x1 < zero:
                t += wp.exp(qq * wp.log(x1 / lo))
            out = lo * wp.exp(wp.log(t / two) / qq)
    else:
        beta = wp.log(two) / eps
        if lo > zero:
            out = lo - wp.log((wp.exp(-beta * (x0 - lo)) + wp.exp(-beta * (x1 - lo))) / two) / beta
        else:
            v0 = wp.max(-x0, zero)
            v1 = wp.max(-x1, zero)
            c = wp.max(v0, v1)
            if beta * c <= one:
                out = -_log1p((_expm1(beta * v0) + _expm1(beta * v1)) / two) / beta
            else:
                out = -(c + wp.log((wp.exp(beta * (v0 - c)) + wp.exp(beta * (v1 - c))) / two) / beta)
    return out


@wp.func
def _gm_conj2_d(x0: Any, x1: Any, code: int, eps: Any, pp: Any, qq: Any):
    # d conj2(x0, x1) / d x0; d / d x1 is _gm_conj2_d(x1, x0) (conj2 is symmetric)
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    two = type(eps)(2.0)
    lo = wp.min(x0, x1)
    d = zero
    if code == 4:
        if lo > zero:
            M = _gm_conj2(x0, x1, code, eps, pp, qq)
            d = wp.exp((pp - one) * (wp.log(x0) - wp.log(M))) / two
        elif x0 < zero:
            if qq == one:
                d = type(eps)(0.5)
            else:
                M = -_gm_conj2(x0, x1, code, eps, pp, qq)
                d = wp.exp((qq - one) * (wp.log(-x0) - wp.log(M))) / two
    else:
        beta = wp.log(two) / eps
        if lo > zero:
            e0 = wp.exp(-beta * (x0 - lo))
            d = e0 / (e0 + wp.exp(-beta * (x1 - lo)))
        elif x0 < zero:
            v0 = wp.max(-x0, zero)
            v1 = wp.max(-x1, zero)
            c = wp.max(v0, v1)
            e0 = wp.exp(beta * (v0 - c))
            d = e0 / (e0 + wp.exp(beta * (v1 - c)))
    return d


@wp.func
def _gm_conj_adjoint(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), b: int, r: int, k0: int, n: int,
                     s: Any, code: int, eps: Any, pp: Any, qq: Any, scale: Any, vals_grad: wp.array2d(dtype=Any)):
    # vals_grad[addr[r, k]] += scale d conj(u) / d u_k for k0 <= k < n (the -s of u and of the output cancel)
    zero = type(eps)(0.0)
    one = type(eps)(1.0)
    m = n - k0
    fm = type(eps)(m)
    lo = -s * vals[b, addr[r, k0]]
    for k in range(k0 + 1, n):
        lo = wp.min(lo, -s * vals[b, addr[r, k]])
    if m == 1:
        wp.atomic_add(vals_grad, b, addr[r, k0], scale)
    elif code == 4:
        if lo > zero:
            lM = wp.log(_gm_conj(vals, addr, b, r, k0, n, s, code, eps, pp, qq))
            for k in range(k0, n):
                wp.atomic_add(vals_grad, b, addr[r, k], scale * wp.exp((pp - one) * (wp.log(-s * vals[b, addr[r, k]]) - lM)) / fm)
        elif qq == one:
            for k in range(k0, n):
                if -s * vals[b, addr[r, k]] < zero:
                    wp.atomic_add(vals_grad, b, addr[r, k], scale / fm)
        elif lo < zero:
            lM = wp.log(-_gm_conj(vals, addr, b, r, k0, n, s, code, eps, pp, qq))
            for k in range(k0, n):
                if -s * vals[b, addr[r, k]] < zero:
                    wp.atomic_add(vals_grad, b, addr[r, k], scale * wp.exp((qq - one) * (wp.log(s * vals[b, addr[r, k]]) - lM)) / fm)
    else:
        beta = wp.log(fm) / eps
        tot = zero
        if lo > zero:
            for k in range(k0, n):
                tot += wp.exp(-beta * (-s * vals[b, addr[r, k]] - lo))
            for k in range(k0, n):
                wp.atomic_add(vals_grad, b, addr[r, k], scale * wp.exp(-beta * (-s * vals[b, addr[r, k]] - lo)) / tot)
        else:
            c = -lo
            for k in range(k0, n):
                tot += wp.exp(beta * (wp.max(s * vals[b, addr[r, k]], zero) - c))
            for k in range(k0, n):
                if -s * vals[b, addr[r, k]] < zero:
                    wp.atomic_add(vals_grad, b, addr[r, k], scale * wp.exp(beta * (s * vals[b, addr[r, k]] - c)) / tot)


@wp.kernel
def _gm_forward(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                off: int, s: Any, code: int, eps: Any, empty: Any, params: wp.array(dtype=Any), roff: int, per_row: int,
                nested: int, pp: Any, qq: Any):
    b, r = wp.tid()
    n = count[r]
    e = eps
    exact = int(0)
    if per_row == 1:
        e = params[roff + r]
        if e == type(eps)(0.0):
            exact = 1
    out = empty
    if exact == 1:
        c = empty
        for k in range(n):
            c = wp.max(c, s * vals[b, addr[r, k]])
        out = s * c
    elif nested == 1:
        p = _gm_conj(vals, addr, b, r, 1, n, s, code, e, pp, qq)
        out = -s * _gm_conj2(-s * vals[b, addr[r, 0]], p, code, e, pp, qq)
    else:
        out = -s * _gm_conj(vals, addr, b, r, 0, n, s, code, e, pp, qq)
    vals[b, off + r] = out


@wp.kernel
def _gm_adjoint(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                off: int, s: Any, code: int, eps: Any, empty: Any, params: wp.array(dtype=Any), roff: int, per_row: int,
                nested: int, pp: Any, qq: Any, vals_grad: wp.array2d(dtype=Any)):
    b, r = wp.tid()
    n = count[r]
    e = eps
    exact = int(0)
    if per_row == 1:
        e = params[roff + r]
        if e == type(eps)(0.0):
            exact = 1
    ybar = vals_grad[b, off + r]
    if exact == 1:
        c = empty
        for k in range(n):
            c = wp.max(c, s * vals[b, addr[r, k]])
        total = type(eps)(0.0)
        for k in range(n):
            if s * vals[b, addr[r, k]] == c:
                total += type(eps)(1.0)
        for k in range(n):
            if s * vals[b, addr[r, k]] == c:
                wp.atomic_add(vals_grad, b, addr[r, k], ybar / total)
    elif nested == 1:
        u0 = -s * vals[b, addr[r, 0]]
        p = _gm_conj(vals, addr, b, r, 1, n, s, code, e, pp, qq)
        wp.atomic_add(vals_grad, b, addr[r, 0], ybar * _gm_conj2_d(u0, p, code, e, pp, qq))
        _gm_conj_adjoint(vals, addr, b, r, 1, n, s, code, e, pp, qq, ybar * _gm_conj2_d(p, u0, code, e, pp, qq), vals_grad)
    else:
        _gm_conj_adjoint(vals, addr, b, r, 0, n, s, code, e, pp, qq, ybar, vals_grad)


@wp.kernel
def _copy_forward(vals: wp.array2d(dtype=Any), off: int, rho: wp.array2d(dtype=Any)):
    b, t = wp.tid()
    rho[b, t] = vals[b, off + t]


@wp.kernel
def _copy_adjoint(rho_grad: wp.array2d(dtype=Any), off: int, vals_grad: wp.array2d(dtype=Any)):
    b, t = wp.tid()
    wp.atomic_add(vals_grad, b, off + t, rho_grad[b, t])


def layout(program):
    """Step offsets into the flat buffer, and each reduction step's (R, M) global address table."""
    lengths = np.array([s.length for s in program.steps])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    tables = []
    for step in program.steps:
        if step.kind == "atom":
            tables.append(None)
            continue
        src = np.array(step.sources)
        starts = np.concatenate([[0], np.cumsum(lengths[src])])
        j = np.searchsorted(starts, step.index, side="right") - 1
        tables.append((offsets[src][j] + step.index - starts[j]).astype(np.int32))
    return offsets, tables


def matched_param(program, semantics, eps):
    """Per-row parameters of the matched protocol (core_study.methods, D006 item 14), float64,
    one per reduction row in program order: at arity m > 1, gamma = 2 eps / (1 - 1/m) for
    sparsemax and beta = log(m) / eps for lse and lse_plain, so that every node's local error
    is eps (for lse_plain, a maximum overestimates by up to eps instead); 0 at
    m = 1, which the kernels evaluate exactly. m is the row's count of valid entries, the
    arity core_study.methods reads from the row's mask. gm_exp (E036) gets eps on every row with
    m > 1 (the kernel sets beta = log(m) / eps for each conjunction it forms, the nested Until's
    prefix and pair included) and gm_pm01 and gm_pm10 get 1; all 0 at m = 1."""
    m = np.concatenate([st.count for st in program.steps if st.kind != "atom"]).astype(np.float64)
    safe = np.maximum(m, 2)
    if semantics == "sparsemax":
        p = 2 * eps / (1 - 1 / safe)
    elif semantics in ("lse", "lse_plain"):
        p = np.log(safe) / eps
    elif semantics == "gm_exp":  # the kernel takes eps and computes beta = log(m) / eps per conjunction
        p = np.full(len(m), float(eps))
    elif semantics in ORDER:  # gm_pm01, gm_pm10: no parameter; 1 marks a row evaluated by the measure
        p = np.ones(len(m))
    else:
        raise ValueError("no matched parameter for " + repr(semantics))
    return np.where(m > 1, p, 0.0)


_PLANS = weakref.WeakKeyDictionary()


def _plan(program, device):
    """Step offsets, buffer size, reduction-row offsets and device address tables of a program on
    a device, built on the first call and kept with the program object."""
    per = _PLANS.setdefault(program, {})
    key = str(device)
    if key not in per:
        offsets, tables = layout(program)
        red = np.array([st.kind != "atom" for st in program.steps])
        rows = np.where(red, [st.length for st in program.steps], 0)
        atoms = [(st.atom, int(st.sign), off) for st, off in zip(program.steps, offsets) if st.kind == "atom"]
        per[key] = {"offsets": offsets, "N": int(sum(st.length for st in program.steps)), "R": int(rows.sum()),
                    "atoms": [wp.array(np.array(c, dtype=np.int32), dtype=wp.int32, device=device) for c in zip(*atoms)],
                    "roffs": np.concatenate([[0], np.cumsum(rows)[:-1]]), "params": [], "dummy": {},
                    "tables": [None if t is None else (wp.array(t, dtype=wp.int32, device=device),
                                                       wp.array(st.count.astype(np.int32), dtype=wp.int32, device=device))
                               for st, t in zip(program.steps, tables)]}
    return per[key]


def _param_array(plan, param, dt, device):
    """The device copy of a per-row parameter array (uploaded once per array object, dtype and
    device; the last four are kept), or a one-entry placeholder for a scalar parameter."""
    if np.ndim(param) == 0:
        if dt not in plan["dummy"]:
            plan["dummy"][dt] = wp.zeros(1, dtype=dt, device=device)
        return plan["dummy"][dt]
    for p, d, a in plan["params"]:
        if p is param and d == dt:
            return a
    a = wp.array(np.asarray(param, dtype=np.float64), dtype=dt, device=device)
    plan["params"] = plan["params"][-3:] + [(param, dt, a)]
    return a


def _check(program, scores, semantics, param):
    if semantics not in SEMANTICS:
        raise ValueError("unknown semantics " + repr(semantics))
    if semantics != "exact":
        if np.ndim(param) == 1:
            R = sum(st.length for st in program.steps if st.kind != "atom")
            p = np.asarray(param, dtype=np.float64)
            if len(p) != R or not np.all(np.isfinite(p) & (p >= 0)):
                raise ValueError("a per-row parameter needs " + str(R) + " finite nonnegative values")
        elif not (param is None and semantics in ORDER) and (param is None or not param > 0):
            raise ValueError(semantics + " semantics need a positive parameter")
        if np.any([s.kind != "atom" and np.any(s.count == 0) for s in program.steps]):
            raise ValueError("smooth semantics need nonempty windows; compile with boundary='strict'")
    if scores.ndim != 3 or scores.shape[1] != program.T or scores.shape[2] < program.n_predicates:
        raise ValueError("scores must have shape (B, T, P) with T = " + str(program.T))


def _launches(program, plan, scores, vals, semantics, param, oracle):
    """The forward launches, each with its adjoint launch and the arrays the adjoint touches: one
    launch for every atom step, then one per reduction step in program order (atom steps read
    only the scores, so they may all run first). The adjoints read vals.grad and scores.grad when
    they run."""
    dt, dev = scores.dtype, scores.device
    B, T = scores.shape[0], program.T
    smooth = SMOOTH[semantics]
    per_row = int(smooth > 0 and np.ndim(param) == 1)
    beta = dt(1.0 if smooth == 0 or per_row or param is None else param)
    params = _param_array(plan, param if per_row else 0.0, dt, dev)
    out = []
    A = sum(st.kind == "atom" for st in program.steps)
    if A:
        out.append((partial(wp.launch, _atoms_forward, dim=(B, A, T), inputs=[scores] + plan["atoms"], outputs=[vals],
                            device=dev, record_tape=False),
                    _atoms_backward(vals, scores, plan["atoms"], B, A, T, dev), [vals, scores]))
    for i, step in enumerate(program.steps):
        off = int(plan["offsets"][i])
        if step.kind == "atom":
            continue
        addr, count = plan["tables"][i]
        is_max = 1 if step.kind == "max" else 0
        s, empty = dt(1.0 if is_max else -1.0), dt(-np.inf)
        if smooth in GM:
            pq = ORDER.get(semantics, (0.0, 1.0))
            args = [addr, count, off, s, smooth, beta, empty, params, int(plan["roffs"][i]), per_row, int(step.label.endswith(".inner")),
                    dt(pq[0]), dt(pq[1])]
            out.append((partial(wp.launch, _gm_forward, dim=(B, step.length), inputs=[vals] + args, device=dev, record_tape=False),
                        _gm_backward(vals, args, B, step.length, dev), [vals]))
            continue
        tail = [int(oracle), params, int(plan["roffs"][i]), per_row]
        out.append((partial(wp.launch, _reduce_forward, dim=(B, step.length),
                            inputs=[vals, addr, count, off, s, is_max, smooth, beta, empty] + tail, device=dev,
                            record_tape=False),
                    _reduce_backward(vals, [addr, count, off, s, smooth, beta, empty] + tail, B, step.length, dev),
                    [vals]))
    return out


def evaluate_warp(program, scores, semantics="exact", param=None, tape=None, oracle=False):
    """All step outputs in one (B, N) buffer, and the step offsets into it.

    With a tape, every step records its hand-written adjoint; scores needs
    requires_grad=True, and vals.grad accumulates the step output adjoints.
    oracle=True finds the sparsemax threshold by the O(m^2) brute-force search
    (test oracle and timing reference); the default is the bisection. param is
    a positive number or one value per reduction row (module docstring).
    """
    _check(program, scores, semantics, param)
    plan = _plan(program, scores.device)
    vals = wp.zeros((scores.shape[0], plan["N"]), dtype=scores.dtype, device=scores.device,
                    requires_grad=tape is not None)
    for forward, backward, arrays in _launches(program, plan, scores, vals, semantics, param, oracle):
        forward()
        if tape is not None:
            tape.record_func(backward=backward, arrays=arrays)
    return vals, plan["offsets"].copy()


def _atoms_backward(vals, scores, tables, B, A, T, dev):
    def backward():
        wp.launch(_atoms_adjoint, dim=(B, A, T), inputs=[vals.grad] + tables, outputs=[scores.grad], device=dev,
                  record_tape=False)
    return backward


def _reduce_backward(vals, args, B, R, dev):
    def backward():
        wp.launch(_reduce_adjoint, dim=(B, R), inputs=[vals] + args, outputs=[vals.grad], device=dev,
                  record_tape=False)
    return backward


def _gm_backward(vals, args, B, R, dev):
    def backward():
        wp.launch(_gm_adjoint, dim=(B, R), inputs=[vals] + args, outputs=[vals.grad], device=dev, record_tape=False)
    return backward


def robustness_warp(program, scores, semantics="exact", param=None, tape=None, oracle=False):
    """Robustness trace (B, L_root) of the formula, in its own array so a tape can seed it."""
    vals, offsets = evaluate_warp(program, scores, semantics, param, tape, oracle)
    B, dev = scores.shape[0], scores.device
    L = program.steps[-1].length
    off = int(offsets[-1])
    rho = wp.zeros((B, L), dtype=scores.dtype, device=dev, requires_grad=tape is not None)
    wp.launch(_copy_forward, dim=(B, L), inputs=[vals, off], outputs=[rho], device=dev, record_tape=False)
    if tape is not None:
        def backward():
            wp.launch(_copy_adjoint, dim=(B, L), inputs=[rho.grad, off], outputs=[vals.grad], device=dev,
                      record_tape=False)
        tape.record_func(backward=backward, arrays=[rho, vals])
    return rho


class Evaluator:
    """robustness_warp and its gradient for one program, semantics, parameter, batch size B,
    dtype and device, with the tables, the per-row parameters and every buffer allocated once.
    scores (B, T, P), rho (B, L) and seed (B, L, ones until set) are its arrays; value() and
    gradient() copy their arguments into them when given. On a CUDA device with graphs=True the
    forward and the forward-plus-backward launch sequences are captured as CUDA graphs at
    construction and replayed per call; otherwise they are launched one by one."""

    def __init__(self, program, semantics, param, B, dtype, device, P=None, graphs=True, oracle=False):
        device = wp.get_device(device)
        P = program.n_predicates if P is None else P
        self.scores = wp.zeros((B, program.T, P), dtype=dtype, device=device, requires_grad=True)
        _check(program, self.scores, semantics, param)
        plan = _plan(program, device)
        L, off = program.steps[-1].length, int(plan["offsets"][-1])
        self.vals = wp.zeros((B, plan["N"]), dtype=dtype, device=device, requires_grad=True)
        self.rho = wp.zeros((B, L), dtype=dtype, device=device, requires_grad=True)
        self.seed = wp.ones((B, L), dtype=dtype, device=device)
        steps = _launches(program, plan, self.scores, self.vals, semantics, param, oracle)
        copy = partial(wp.launch, _copy_forward, dim=(B, L), inputs=[self.vals, off], outputs=[self.rho],
                       device=device, record_tape=False)
        copy_adj = partial(wp.launch, _copy_adjoint, dim=(B, L), inputs=[self.rho.grad, off],
                           outputs=[self.vals.grad], device=device, record_tape=False)
        self._forward = [f for f, _, _ in steps] + [copy]
        self._backward = [self.vals.grad.zero_, self.scores.grad.zero_, partial(wp.copy, self.rho.grad, self.seed),
                          copy_adj] + [g for _, g, _ in reversed(steps)]
        self._run(self._forward + self._backward)  # loads the modules before any capture
        wp.synchronize_device(device)
        self._graphs = None
        if graphs and device.is_cuda:
            self._graphs = (self._capture(device, self._forward), self._capture(device, self._forward + self._backward))

    @staticmethod
    def _run(ops):
        for op in ops:  # over launches (formula steps)
            op()

    def _capture(self, device, ops):
        with wp.ScopedCapture(device=device) as capture:
            self._run(ops)
        return capture.graph

    def value(self, scores=None):
        """rho, from scores (copied into self.scores) or from self.scores as it stands."""
        if scores is not None:
            wp.copy(self.scores, scores)
        if self._graphs is None:
            self._run(self._forward)
        else:
            wp.capture_launch(self._graphs[0])
        return self.rho

    def gradient(self, scores=None, seed=None):
        """rho and the gradient of <seed, rho> with respect to the scores (self.scores.grad)."""
        if scores is not None:
            wp.copy(self.scores, scores)
        if seed is not None:
            wp.copy(self.seed, seed)
        if self._graphs is None:
            self._run(self._forward + self._backward)
        else:
            wp.capture_launch(self._graphs[1])
        return self.rho, self.scores.grad

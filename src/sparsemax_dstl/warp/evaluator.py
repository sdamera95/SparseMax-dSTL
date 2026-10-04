"""STL robustness of a compiled Program in Warp with hand-written adjoints: exact, plain and sound
log-sum-exp, sparsemax lower extrema (Eq. 7) and generalized-mean robustness (Appendix I-A)."""
import weakref
from functools import partial
from typing import Any

import numpy as np
import warp as wp

# no generated adjoints: every forward launch is paired with a hand-written adjoint kernel
wp.set_module_options({"enable_backward": False})

SEMANTICS = ("exact", "lse", "sparsemax", "lse_plain", "gm_pm01", "gm_pm10", "gm_exp")
SMOOTH = {"exact": 0, "lse": 1, "sparsemax": 2, "lse_plain": 3, "gm_pm01": 4, "gm_pm10": 4, "gm_exp": 5}
GM = (4, 5)
ORDER = {"gm_pm01": (0.0, 1.0), "gm_pm10": (-10.0, 10.0)}  # (p, q) of the power mean settings (code 4)
BISECTIONS = wp.constant(40)


@wp.kernel
def _atoms_forward(scores: wp.array3d(dtype=Any), atom: wp.array(dtype=wp.int32), sign: wp.array(dtype=wp.int32),
                   aoff: wp.array(dtype=wp.int32), vals: wp.array2d(dtype=Any)):
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
    # sparsemax threshold theta of u_k = s z_k - c (Eq. 6): sum_k (u_k - theta)_+ = gamma, theta in
    # [-gamma, 0]; bisection, then the closed form on the support
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
    # theta = (sum_S u - gamma) / |S| over S = {k : u_k > tau}; S is recounted at tau = theta, which
    # after the first recount is only raised, until its size repeats
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
    # brute-force test oracle, O(n^2): entry i is in the support when gamma + k_i u_i > S_i, with k_i
    # and S_i the count and the sum of the entries u_j >= u_i
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
    # s = 1 at a maximum and -1 at a minimum; c = max_k s z_k over the n valid entries of the row
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
        if is_max == 1 and sm == 1:  # Eq. 14; lse_plain (3) keeps the unshifted value
            out -= wp.log(type(beta)(n)) / p
    # Eq. 7: M_gamma(z) + gamma / (2 n) at a maximum, -M_gamma(-z) - gamma / 2 at a minimum
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
        # the adjoint is the sparsemax weights (Eq. 6); the threshold is not differentiated
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
    # log-sum-exp: softmax weights; exact: the entries equal to the extremum share the adjoint equally
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
    # exp(x) - 1 without cancellation near 0; called with x <= 1
    u = wp.exp(x)
    one = type(x)(1.0)
    r = x
    if u != one:
        r = (u - one) * x / wp.log(u)
    return r


@wp.func
def _log1p(x: Any):
    # log(1 + x) without cancellation near 0; called with x >= 0
    one = type(x)(1.0)
    u = one + x
    r = x
    if u != one:
        r = wp.log(u) * x / (u - one)
    return r


@wp.func
def _gm_conj(vals: wp.array2d(dtype=Any), addr: wp.array2d(dtype=wp.int32), b: int, r: int, k0: int, n: int, s: Any,
             code: int, eps: Any, pp: Any, qq: Any):
    # generalized-mean conjunction of u_k = -s vals[b, addr[r, k]] over k0 <= k < n: code 4 the power
    # means of orders (pp, qq), code 5 the exponential generators with beta = log(m) / eps
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
    # -s conj(-s z): a minimum is the conjunction, a maximum its dual. nested (the inner step of an until
    # or release, whose row (t, k) holds psi(t+k), phi(t), ..., phi(t+k)): conj2(entry 0, conj(entries 1..))
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
    """Per-row parameters in program order, 0 at arity m = 1: by Eq. 15, gamma = 2 eps / (1 - 1/m) for
    sparsemax and beta = log(m) / eps for lse and lse_plain; eps for gm_exp; 1 for gm_pm01 and gm_pm10."""
    m = np.concatenate([st.count for st in program.steps if st.kind != "atom"]).astype(np.float64)
    safe = np.maximum(m, 2)
    if semantics == "sparsemax":
        p = 2 * eps / (1 - 1 / safe)
    elif semantics in ("lse", "lse_plain"):
        p = np.log(safe) / eps
    elif semantics == "gm_exp":  # the kernel takes eps and computes beta = log(m) / eps per conjunction
        p = np.full(len(m), float(eps))
    elif semantics in ORDER:  # gm_pm01, gm_pm10: no parameter; 1 marks a row that is not exact
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
    """The forward launches, each with its adjoint launch and the arrays the adjoint touches: one launch
    for all atom steps, then one per reduction step in program order."""
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
    """Step outputs in one (B, N) buffer and their offsets; with a tape, every launch registers its adjoint.
    param: beta (lse, lse_plain), gamma (sparsemax), eps (gm_exp); one number or one per row (0: exact)."""
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
    """robustness_warp and its gradient for one program, semantics, parameter, batch size B, dtype and
    device, with every buffer allocated once; on CUDA, graphs=True replays the launches as CUDA graphs."""

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

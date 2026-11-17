"""IPOPT through CasADi on one control period's program with this package's derivatives.

The constraint Jacobian's structure given to IPOPT is the prescribed one (every support's entries lie in it; the values off
the supports are exactly zero). The Lagrangian Hessian's structure, by arm:
    U  the supports' structure at the start point; at every accepted iterate the supports are tested against it before IPOPT
       asks for that iterate's Hessian, and when they leave it IPOPT stops there and continues from that iterate on a new
       solver (warm start, mu_init its last barrier parameter) declared on the union of the old structure and the supports;
    D  as U with the supports widened in time by d samples (per row, every (sample, arm) within d samples of a sample of its
       supports takes the largest joint prefix among them, never beyond the row's prescribed pattern);
    P  the prescribed structure, declared once; the derivatives' own structure grows by union inside a Hessian call whose test
       fails, and their values are written onto the prescribed one.
At an accepted iterate the Jacobian and the Hessian share one evaluation of the forward pass with the supports.
"""
import time

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from ..sparse import patterns as SP
from ..sparse.assembly import StructuralDerivatives
from ..tasks import dual_arm as D

NX, NU, NQ, NJ = D.NX, D.NU, D.NQ, D.NJ
ACCEPT_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
ACCEPT_VIOLATION = 1e-6  # IPOPT's constr_viol_tol in the problem's units
FUNCS = ("nlp_f", "nlp_g", "nlp_grad_f", "nlp_jac_g", "nlp_hess_l")


def ipopt_options(hessian, warm=False, **extra):
    """CasADi nlpsol options for IPOPT: tol and constr_viol_tol 1e-6, at most 1000 updates, IPOPT's default linear solver
    (MUMPS); hessian "exact" or "limited-memory" (L-BFGS, history 10); warm adds IPOPT's warm start (warm_start_init_point,
    warm_start_bound_push, warm_start_mult_bound_push and mu_init 1e-6); extra: further IPOPT options."""
    o = {"hessian_approximation": hessian, "max_iter": 1000, "tol": 1e-6, "constr_viol_tol": 1e-6, "print_level": 0}
    if hessian == "limited-memory":
        o["limited_memory_max_history"] = 10
    if warm:
        o.update({"warm_start_init_point": "yes", "warm_start_bound_push": 1e-6, "warm_start_mult_bound_push": 1e-6, "mu_init": 1e-6})
    o.update(extra)
    out = {"ipopt." + k: v for k, v in o.items()}
    out.update({"print_time": False, "record_time": True, "error_on_fail": False})
    return out


def ccs(rows, cols, n_rows, n_cols):
    """CasADi Sparsity of (rows, cols) and the order taking row-major values to its column-major nonzeros."""
    order = np.lexsort((np.asarray(rows), np.asarray(cols)))
    sp = ca.Sparsity.triplet(int(n_rows), int(n_cols), [int(v) for v in np.asarray(rows)[order]], [int(v) for v in np.asarray(cols)[order]])
    return sp, order


def violation(x, g, lbx, ubx, lbg, ubg):
    """The largest bound or constraint violation."""
    return float(max(np.max(np.maximum(lbg - g, 0) + np.maximum(g - ubg, 0)), np.max(np.maximum(lbx - x, 0) + np.maximum(x - ubx, 0))))


class Function(ca.Callback):
    """A CasADi function of the given input and output sparsities; fn maps the first input (x) to the outputs' nonzeros. The
    other inputs are unused. jac_sparsity, when given, is declared as the Jacobian of output 0 in input 0."""

    def __init__(self, name, sp_in, sp_out, fn, jac_sparsity=None):
        ca.Callback.__init__(self)
        self.sp_in, self.sp_out, self.fn, self.jac_sp = sp_in, sp_out, fn, jac_sparsity
        self.construct(name, {})

    def get_n_in(self):
        return len(self.sp_in)

    def get_n_out(self):
        return len(self.sp_out)

    def get_sparsity_in(self, i):
        return self.sp_in[i]

    def get_sparsity_out(self, i):
        return self.sp_out[i]

    def eval(self, arg):
        return [ca.DM(sp, v) for sp, v in zip(self.sp_out, self.fn(arg[0]))]

    def has_jac_sparsity(self, oind, iind):
        return self.jac_sp is not None and oind == 0 and iind == 0

    def get_jac_sparsity(self, oind, iind, symmetric):
        return self.jac_sp


class Hessian(Function):
    """hess_lag: inputs (x, p, lam_f, lam_g), output the declared upper triangle; fn receives (x, lam_f, lam_g)."""

    def eval(self, arg):
        return [ca.DM(self.sp_out[0], self.fn(arg[0], arg[2], arg[3]))]


class Iteration(ca.Callback):
    """IPOPT's iteration callback, called at the start point and every accepted iterate; fn(x) True stops IPOPT there with
    User_Requested_Stop, before that iterate's Hessian is evaluated."""

    def __init__(self, n, m, fn):
        ca.Callback.__init__(self)
        self.n, self.m, self.fn = n, m, fn
        self.construct("iteration", {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return "stop"

    def get_sparsity_in(self, i):
        name = ca.nlpsol_out(i)
        if name == "f":
            return ca.Sparsity.scalar()
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self.n)
        if name in ("g", "lam_g"):
            return ca.Sparsity.dense(self.m)
        return ca.Sparsity(0, 0)

    def eval(self, arg):
        return [1 if self.fn(np.asarray(arg[0], np.float64).ravel()) else 0]


# ------------------------------------------------------------------
# declared structures

def instance_keys(pb):
    """(conjunct index, root row) of every instance row, in row order."""
    out = []
    for ci, cj in enumerate(pb.conjs):
        roots, _ = cj.imposed([pb.k], pb.T)
        out += [(ci, int(r)) for r in roots[0][roots[0] >= 0]]
    return out


def prefix_of(pb, J):
    """(n_rows, T, 2) the joint prefix lengths of the instance rows' position entries of a Jacobian structure J."""
    spec = J[0] >= pb.n_def
    r, c = J[0][spec] - pb.n_def, J[1][spec]
    rem = c % NX
    pos = rem < NQ
    pre = np.zeros((pb.n_rows, pb.T, 2), np.int64)
    np.maximum.at(pre, (r[pos], (c // NX)[pos], (rem // NJ)[pos]), (rem % NJ)[pos] + 1)
    return pre


def widen(pre, d, cap):
    """The prefix widened in time by d samples within each row (the largest joint prefix within d samples), at most cap."""
    out = np.array(pre)
    for s in range(1, d + 1):
        out[:, s:] = np.maximum(out[:, s:], pre[:, :-s])
        out[:, :-s] = np.maximum(out[:, :-s], pre[:, s:])
    return np.minimum(out, cap)


# ------------------------------------------------------------------
# the solver

class Solver:
    """IPOPT on a PeriodProgram pb with arm "U", "D" (widened by d samples) or "P" (module docstring); reference: the scene's
    patterns.Reference, built here when None; options: further IPOPT options (for example max_iter). The program's JAX work
    runs in float64."""

    def __init__(self, pb, arm="U", d=0, max_segments=50, reference=None, **options):
        if arm not in ("U", "D", "P"):
            raise ValueError("arm is 'U', 'D' or 'P'")
        self.pb, self.arm, self.d, self.max_segments, self.options = pb, arm, int(d), max_segments, options
        with jax.enable_x64(True):
            self.J_pre, self.H_pre = SP.prescribed(pb, reference)
            self.cap = prefix_of(pb, self.J_pre)
            self.rt = StructuralDerivatives(pb)
            self.keys = instance_keys(pb)
            self._f = jax.jit(pb.objective)
            self._grad = jax.jit(jax.grad(pb.objective))
            self._exact = jax.jit(pb.exact_rows)
            self._prefix = self.rt.prefix
            rows = self.rt.supports.rows
            self._g = jax.jit(lambda w: jnp.concatenate([pb.defects(*pb.split(w)), rows(w) - pb.delta]))
        self.reset()

    def reset(self):
        self.clock = {"f": 0.0, "g": 0.0, "grad_f": 0.0, "jacobian": 0.0, "hessian": 0.0, "check": 0.0, "declare_in_callback": 0.0, "convert": 0.0}
        self.count = {"jacobian": 0, "hessian": 0, "check": 0, "check_reused": 0, "invalid_hessian": 0, "union_in_callback": 0, "g_reused": 0}
        self.last, self.last_g, self._pos = (None, None), (None, None), {}

    def _declared(self, pre):
        return widen(pre, self.d, self.cap) if self.arm == "D" else np.asarray(pre)

    def declare(self, pre):
        t = time.perf_counter()
        self.rt.declare(np.asarray(pre, np.int64), jnp.float64, self.J_pre)
        return time.perf_counter() - t

    # ------------------------------------------------------------------
    # callbacks

    def _x(self, x):
        xn = np.asarray(x, np.float64).ravel()
        return xn, xn.tobytes()

    def _f_cb(self, x):
        t = time.perf_counter()
        v = float(self._f(jnp.asarray(self._x(x)[0])))
        self.clock["f"] += time.perf_counter() - t
        return [v]

    def _g_cb(self, x):
        t = time.perf_counter()
        xn, key = self._x(x)
        v = np.asarray(self._g(jnp.asarray(xn)))
        self.last_g = (key, v)
        self.clock["g"] += time.perf_counter() - t
        return [v]

    def _grad_cb(self, x):
        t = time.perf_counter()
        w = jnp.asarray(self._x(x)[0])
        out = [float(self._f(w)), np.asarray(self._grad(w))]
        self.clock["grad_f"] += time.perf_counter() - t
        return out

    def _jac_cb(self, x):
        t = time.perf_counter()
        xn, key = self._x(x)
        w = jnp.asarray(xn)
        if self.last_g[0] == key:
            g = self.last_g[1]
            self.count["g_reused"] += 1
        else:
            g = np.asarray(self._g(w))
        v, esc = self.rt.jacobian_at(w, key)
        self.last = (key, esc)
        self.count["jacobian"] += 1
        c = time.perf_counter()
        out = [g, np.asarray(v)[self.obj["J_order"]]]
        self.clock["convert"] += time.perf_counter() - c
        self.clock["jacobian"] += time.perf_counter() - t
        return out

    def _hess_cb(self, x, lam_f, lam_g):
        t = time.perf_counter()
        xn, key = self._x(x)
        w = jnp.asarray(xn)
        lam = jnp.asarray(np.asarray(lam_g, np.float64).ravel())
        v, esc = self.rt.hessian_at(w, key, float(lam_f), lam)
        if esc and self.arm == "P":
            t1 = time.perf_counter()
            self.rt.declare(np.maximum(np.asarray(self.rt.declared), np.asarray(self._prefix(w))), jnp.float64, self.J_pre)
            self.clock["declare_in_callback"] += time.perf_counter() - t1
            self.count["union_in_callback"] += 1
            v, esc = self.rt.hessian_at(w, key, float(lam_f), lam)
        if esc:
            self.count["invalid_hessian"] += 1
        self.count["hessian"] += 1
        c = time.perf_counter()
        v = np.asarray(v)
        if self.arm == "P":
            ver = self.rt.version
            if ver not in self._pos:
                hs = self.rt.hess_structure
                self._pos = {ver: SP.positions(self.H_pre, hs[0], hs[1], self.pb.n)}
            full = np.zeros(self.H_pre[0].size)
            full[self._pos[ver]] = v
            v = full
        out = v[self.obj["H_order"]]
        self.clock["convert"] += time.perf_counter() - c
        self.clock["hessian"] += time.perf_counter() - t
        return out

    def _stop(self, x):
        if self.arm not in ("U", "D"):
            return False
        t = time.perf_counter()
        xn, key = self._x(x)
        if self.last[0] == key:
            esc = self.last[1]
            self.count["check_reused"] += 1
        else:
            esc = bool(np.any(np.asarray(self._prefix(jnp.asarray(xn))) > np.asarray(self.rt.declared)))
        self.count["check"] += 1
        self.clock["check"] += time.perf_counter() - t
        return bool(esc)

    def _object(self, warm, mu):
        pb = self.pb
        t = time.perf_counter()
        n, m = pb.n, pb.m
        J_sp, J_order = ccs(self.J_pre[0], self.J_pre[1], m, n)
        o = {"J_order": J_order}
        x_sp, p_sp = ca.Sparsity.dense(n, 1), ca.Sparsity.dense(0, 1)
        f_sp, g_sp = ca.Sparsity.dense(1, 1), ca.Sparsity.dense(m, 1)
        fns = {"F": Function("F", [x_sp], [f_sp], self._f_cb), "G": Function("G", [x_sp], [g_sp], self._g_cb, J_sp),
               "grad_f": Function("grad_f", [x_sp, p_sp], [f_sp, ca.Sparsity.dense(n, 1)], self._grad_cb),
               "jac_g": Function("jac_g", [x_sp, p_sp], [g_sp, J_sp], self._jac_cb), "iteration": Iteration(n, m, self._stop)}
        extra = dict(self.options) if mu is None else dict(self.options, mu_init=mu)
        opts = ipopt_options("exact", warm, **extra)
        opts.update({"grad_f": fns["grad_f"], "jac_g": fns["jac_g"], "iteration_callback": fns["iteration"], "no_nlp_grad": True})
        H = self.H_pre if self.arm == "P" else self.rt.hess_structure
        H_sp, H_order = ccs(H[0], H[1], n, n)
        o.update({"H": H, "H_order": H_order})
        fns["hess_lag"] = Hessian("hess_lag", [x_sp, p_sp, f_sp, g_sp], [H_sp], self._hess_cb)
        opts["hess_lag"] = fns["hess_lag"]
        xs = ca.MX.sym("x", n)
        o["S"] = ca.nlpsol("period", "ipopt", {"x": xs, "f": fns["F"](xs), "g": fns["G"](xs)}, opts)
        o["fns"], o["opts"] = fns, {a: b for a, b in opts.items() if a.startswith("ipopt.")}
        self.obj = o
        return time.perf_counter() - t

    def solve(self, w0, lam_g0=None, lam_x0=None, warm=False, carry=None):
        """One solve from w0, cold with zero multipliers or with IPOPT's warm start from the given multipliers. carry: a
        declared prefix to start from (joined with the supports at w0). Returns a summary dict; the solution, its constraint
        multipliers and bound multipliers are kept as x, lam_g, lam_x."""
        with jax.enable_x64(True):
            return self._solve(w0, lam_g0, lam_x0, warm, carry)

    def _solve(self, w0, lam_g0, lam_x0, warm, carry):
        pb = self.pb
        self.reset()
        lbx, ubx, lbg, ubg = pb.bounds()
        x = np.asarray(w0, np.float64)
        t = time.perf_counter()
        sup = np.asarray(self._prefix(jnp.asarray(x)))
        start = self._declared(sup) if carry is None else np.maximum(np.minimum(np.asarray(carry), self.cap), self._declared(sup))
        self.declare(start)
        out = {"declare_s": time.perf_counter() - t, "segments": [], "build_s": 0.0, "solve_wall_s": 0.0, "updates": 0,
               "function_s": {f: 0.0 for f in FUNCS}, "calls": {f: 0 for f in FUNCS}, "casadi_total_s": 0.0,
               "hessian_upper_start": int(self.rt.hess_structure[0].size) if self.arm != "P" else int(self.H_pre[0].size)}
        lam_g = np.zeros(pb.m) if lam_g0 is None else np.asarray(lam_g0, np.float64)
        lam_x = np.zeros(pb.n) if lam_x0 is None else np.asarray(lam_x0, np.float64)
        mu = None
        for seg in range(self.max_segments):
            out["build_s"] += self._object(warm or seg > 0, mu)
            t = time.perf_counter()
            r = self.obj["S"](x0=x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, lam_g0=lam_g, lam_x0=lam_x)
            wall = time.perf_counter() - t
            st = self.obj["S"].stats()
            x, lam_g, lam_x = np.asarray(r["x"]).ravel(), np.asarray(r["lam_g"]).ravel(), np.asarray(r["lam_x"]).ravel()
            it = st.get("iterations", {})
            mu = float(it["mu"][-1]) if it and len(it.get("mu", [])) else None
            for f in FUNCS:
                out["function_s"][f] += float(st.get("t_wall_" + f, 0.0))
                out["calls"][f] += int(st.get("n_call_" + f, 0))
            out["casadi_total_s"] += float(st["t_wall_total"])
            out["solve_wall_s"] += wall
            out["updates"] += int(st["iter_count"])
            out["segments"].append({"status": st["return_status"], "updates": int(st["iter_count"]), "wall_s": wall, "hess_upper": int(self.obj["H"][0].size)})
            if st["return_status"] != "User_Requested_Stop":
                break
            t = time.perf_counter()
            new = np.maximum(np.asarray(self.rt.declared), self._declared(np.asarray(self._prefix(jnp.asarray(x)))))
            self.declare(new)
            out["declare_s"] += time.perf_counter() - t
        out["ipopt_own_s"] = out["casadi_total_s"] - sum(out["function_s"].values()) - self.clock["check"]
        g = np.asarray(self._g(jnp.asarray(x)))
        rows = g[pb.n_def:]
        ex = np.asarray(self._exact(x[:NX * pb.T].reshape(pb.T, NX)))
        v = violation(x, g, lbx, ubx, lbg, ubg)
        out.update({"status": out["segments"][-1]["status"], "violation": v, "objective": float(self._f(jnp.asarray(x))),
                    "smallest_row": float(rows.min()) if rows.size else np.inf,
                    "accepted": bool(out["segments"][-1]["status"] in ACCEPT_STATUS and v <= ACCEPT_VIOLATION),
                    "exact_robustness": {"smallest": float(ex.min()) if ex.size else np.inf, "below_margin": int(np.sum(ex < pb.delta)), "of": int(ex.size), "per_row": ex.tolist()},
                    "hessian_upper_end": int(self.rt.hess_structure[0].size) if self.arm != "P" else int(self.H_pre[0].size),
                    "solver_objects": len(out["segments"]), "clock_s": dict(self.clock), "count": dict(self.count), "part_a_reused": self.rt.reused,
                    "options": self.obj["opts"], "arm": self.arm, "d": self.d})
        self.x, self.lam_g, self.lam_x = x, lam_g, lam_x
        return out


def shift_start(prev, pb):
    """The start of period pb.k from the solution of period pb.k - 1 (the solver prev): (w, lam_g, lam_x, carried prefix).
    The samples move one step earlier, the last state is held at zero velocity and the last acceleration is zero; the
    defects' and bounds' multipliers shift likewise; an instance row's multiplier and declared prefix are carried by its
    (conjunct, root row), zero for a new instance."""
    T, n_def = pb.T, pb.n_def
    X, U = prev.x[:NX * T].reshape(T, NX), prev.x[NX * T:].reshape(T, NU)
    last = np.concatenate([X[-1, :NQ], np.zeros(NQ)])
    w = np.concatenate([np.concatenate([X[1:], last[None]]).ravel(), np.concatenate([U[1:], np.zeros((1, NU))]).ravel()])
    ld = prev.lam_g[:n_def].reshape(T, NX)
    lam_g = np.zeros(pb.m)
    lam_g[:n_def] = np.concatenate([ld[1:], np.zeros((1, NX))]).ravel()
    old = {key: i for i, key in enumerate(prev.keys)}
    pre_old = np.asarray(prev.rt.declared) if prev.arm in ("U", "D") else None
    carry = np.zeros((pb.n_rows, T, 2), np.int64)
    for i, key in enumerate(instance_keys(pb)):
        if key in old:
            lam_g[n_def + i] = prev.lam_g[prev.pb.n_def + old[key]]
            if pre_old is not None:
                carry[i, :-1] = pre_old[old[key], 1:]
    lx, lu = prev.lam_x[:NX * T].reshape(T, NX), prev.lam_x[NX * T:].reshape(T, NU)
    lam_x = np.concatenate([np.concatenate([lx[1:], lx[-1:]]).ravel(), np.concatenate([lu[1:], np.zeros((1, NU))]).ravel()])
    return w, lam_g, lam_x, (carry if pre_old is not None else None)

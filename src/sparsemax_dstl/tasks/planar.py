"""Planar unicycle example of E049: the system, the regions, the STL specification and the two
fixed trajectories S1 and S2, and the smoothings matched at one worst-case error per node.

System (sampling period $h$ in seconds, state $(x, y, \\theta)$, input $(v, \\omega)$):

    $x^+ = x + h v \\cos\\theta$, $y^+ = y + h v \\sin\\theta$, $\\theta^+ = \\theta + h \\omega$,
    $|v| \\le$ V_MAX, $|\\omega| \\le$ W_MAX.

Regions are axis-aligned boxes $[x_1, x_2] \\times [y_1, y_2]$ written as the conjunction of the four
linear predicates $x - x_1$, $x_2 - x$, $y - y_1$, $y_2 - y$ (each $\\ge 0$), as in Example 4 of
Mehdipour, Vasile and Belta (IEEE TAC 70(3), 2025). Predicate $4 i + j$ is side $j$ of region $i$ in
the order RED, GREEN, BLUE, OBSTACLE, BOUNDARY.

Specification (intervals in samples):

    $(\\neg \\mathrm{Red}\\ U_{[a_1, b_1]}\\ \\mathrm{Green}) \\wedge F_{[a_2, b_2]} G_{[0, 2/h]} \\mathrm{Blue}
     \\wedge G_{[0, T-1]} \\neg \\mathrm{Obstacle} \\wedge G_{[0, T-1]} \\mathrm{Boundary}$,

in words: stay out of Red until Green has been visited within $[a_1, b_1]$; be in Blue for 2 s
starting within $[a_2, b_2]$; never enter the Obstacle; stay inside the workspace.

Trajectories are smooth curves (quintic Hermite pieces and circular arcs joined with equal
position, heading and curvature) sampled with smooth rest-to-rest speed profiles; the inputs are
the speed and turn rate that move the unicycle through those samples (the heading of each moving
step is its chord direction), so simulating them reproduces the samples. The forward speed is
never negative, there is no turn in place, and every input lies in the box.

Matched smoothings (`matched`): at a node whose row has $m$ valid entries the plain log-sum-exp
uses $\\beta = \\log m / \\varepsilon$ and the sparsemax extrema use
$\\gamma = 2 \\varepsilon / (1 - 1/m)$, so that each node's worst-case error is $\\varepsilon$; a row
with one entry is exact under both. The generalized-mean robustness of order $(p, q)$ is
`semantics.gm_power(p, q)` unchanged; it has no error parameter.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..jax.evaluator import evaluate, exact_max, exact_min, gm_power, lse_min, lse_plain_max, read
from ..jax.operators import lower_max, lower_min
from ..stl import Always, And, Atom, Eventually, Not, Until, compile_formula

H = 0.1  # sampling period, s
V_MAX = 1.0  # m/s
W_MAX = np.pi / 2  # rad/s
Z0 = (1.0, 1.0, np.pi / 2)  # start: (1, 1), facing +y

RED = (4.0, 6.0, 1.5, 8.5)
GREEN = (1.5, 3.5, 7.0, 9.0)
BLUE = (7.5, 9.5, 1.5, 3.5)
OBSTACLE = (6.4, 7.4, 3.9, 5.45)
BOUNDARY = (0.0, 10.0, 0.0, 10.0)
REGIONS = (RED, GREEN, BLUE, OBSTACLE, BOUNDARY)
NAMES = ("Red", "Green", "Blue", "Obstacle", "Boundary")



# ------------------------------------------------------------------
# predicates and specification

def scores(xy):
    """Leaf-score array (..., T, 20) of the four linear predicates of every region."""
    x, y = xy[..., 0:1], xy[..., 1:2]
    lo = np.array([r[0] for r in REGIONS]), np.array([r[1] for r in REGIONS])
    yl = np.array([r[2] for r in REGIONS]), np.array([r[3] for r in REGIONS])
    s = jnp.stack([x - lo[0], lo[1] - x, y - yl[0], yl[1] - y], -1)  # (..., T, 5, 4)
    return s.reshape(s.shape[:-2] + (20,))


def box(i):
    return And(*(Atom(4 * i + j) for j in range(4)))  # formula structure


def specification(a1, b1, a2, b2, T):
    """The STL specification and its until subformula; intervals in samples, T samples."""
    hold = int(round(2 / H))
    until = Until((a1, b1), Not(box(0)), box(1))
    spec = And(until, Eventually((a2, b2), Always((0, hold), box(2))), Always((0, T - 1), Not(box(3))),
               Always((0, T - 1), box(4)))
    return spec, until


# ------------------------------------------------------------------
# dynamics

def rollout(z0, u):
    """States (T+1, 3) from z0 under inputs u (T, 2), unicycle with period H."""
    def step(z, ui):
        x, y, th = z
        z1 = jnp.stack([x + H * ui[0] * jnp.cos(th), y + H * ui[0] * jnp.sin(th), th + H * ui[1]])
        return z1, z1
    z0 = jnp.asarray(z0, jnp.result_type(float))
    _, zs = jax.lax.scan(step, z0, jnp.asarray(u, z0.dtype))
    return jnp.concatenate([z0[None], zs], 0)


# ------------------------------------------------------------------
# smooth paths: pieces with matched position, heading and curvature at every junction

def hermite(p0, th0, k0, p1, th1, k1, scale=1.0, n=4000):
    """Quintic Hermite curve from (p0, heading th0, curvature k0) to (p1, th1, k1), dense points (n, 2).
    The end derivatives are L T and L^2 k N (T the unit tangent, N the left normal, L = scale times the
    chord), so heading and curvature match the given values at both ends."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = scale * np.hypot(*(p1 - p0))
    t0, t1 = np.array([np.cos(th0), np.sin(th0)]), np.array([np.cos(th1), np.sin(th1)])
    n0, n1 = np.array([-t0[1], t0[0]]), np.array([-t1[1], t1[0]])
    u = np.linspace(0, 1, n)[:, None]
    h = [1 - 10 * u**3 + 15 * u**4 - 6 * u**5, u - 6 * u**3 + 8 * u**4 - 3 * u**5, 0.5 * u**2 - 1.5 * u**3 + 1.5 * u**4 - 0.5 * u**5,
         0.5 * u**3 - u**4 + 0.5 * u**5, -4 * u**3 + 7 * u**4 - 3 * u**5, 10 * u**3 - 15 * u**4 + 6 * u**5]
    return h[0] * p0 + h[1] * L * t0 + h[2] * L**2 * k0 * n0 + h[3] * L**2 * k1 * n1 + h[4] * L * t1 + h[5] * p1


def arc(center, radius, phi0, phi1, n=4000):
    """Circular arc, counterclockwise for phi1 > phi0; dense points (n, 2)."""
    phi = np.linspace(phi0, phi1, n)
    return np.asarray(center, float) + radius * np.stack([np.cos(phi), np.sin(phi)], 1)


def leg(points, v_peak, up=0.3, down=0.3):
    """Positions at the samples along the dense polyline points, from rest to rest. The speed profile is
    a plateau at about v_peak (m/s) with smoothstep ramps over the fractions up and down of the leg's
    duration; the number of steps is the smallest that keeps the plateau at or below v_peak."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.ceil(s_dense[-1] / (v_peak * H * (1 - (up + down) / 2))))
    tau = (np.arange(n) + 0.5) / n
    r = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))  # noqa: E731
    shape = r(tau / up) * r((1 - tau) / down)
    s = np.concatenate([[0.0], np.cumsum(shape)]) * s_dense[-1] / shape.sum()
    return np.stack([np.interp(s, s_dense, points[:, 0]), np.interp(s, s_dense, points[:, 1])], 1)


def inputs_from_positions(p, th0):
    """Inputs (len(p) - 1, 2) that move the unicycle through the sample positions p from heading th0:
    the heading of each moving step is the direction of its chord, held while the robot stands."""
    d = np.diff(p, axis=0)
    v = np.hypot(d[:, 0], d[:, 1]) / H
    moving = v > 1e-12
    ang = np.arctan2(d[:, 1], d[:, 0])
    idx = np.maximum.accumulate(np.where(moving, np.arange(len(d)), -1))
    th = np.where(idx >= 0, ang[np.maximum(idx, 0)], th0)
    th = np.concatenate([[th0], th])
    dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
    # th[t + 1] is the heading step t needs; the turn input of step t sets the heading of step t + 1, so the
    # heading equals every chord direction from step 1 on (step 0 uses th0, the start heading)
    w = np.concatenate([[dth[0] + dth[1]], dth[2:], [0.0]]) / H
    return np.stack([v, w], 1)


# ------------------------------------------------------------------
# the two trajectories

R_CLIP = 1.0  # radius of S1's arc at Red's edge, m
APEX_Y = 4.6
GREEN_STOP = (2.8, 8.0, np.pi / 3)  # position and heading of the stop in Green
BLUE_STOP = (8.5, 2.5, -np.pi / 2)  # position and heading of the stop in Blue
V_CRUISE = 0.8  # m/s, plateau speed of the long drives


def trajectories(wait, depth=0.06, level=0.04, down=0.12):
    """S1, S2 and the specification's timing for one length of the wait (samples standing at the wait spot).

    S1 starts at (1, 1) facing +y, drives a gentle S-curve towards Red's left edge and enters Red on a
    circular arc of radius R_CLIP that touches the line x = 4 + depth at y = APEX_Y; it continues on the
    same arc out of Red, slowing over the last fraction down of the drive, and comes to rest where the
    arc is level metres outside the edge; it stands there for wait samples; it drives an S-curve to its stop in Green
    (facing north-east), stands there until 0.5 s after Green's window closes, and sweeps clockwise over
    the top of Red and down to its stop in Blue, passing the Obstacle's upper-right corner. S2 drives a
    gentle curve from the start to the same stop in Green, stands there for the same time, and drives
    the same sweep to Blue. Green's window opens 0.5 s after S1 reaches Green and lasts 2 s; Blue's
    start window is the 6 s centred on the arrival in Blue. Returns (u1, u2, timing); both start from Z0."""
    xa = 4.0 + depth
    c = (xa - R_CLIP, APEX_Y)
    phi_e = -np.arccos(1 - (depth + 0.15) / R_CLIP)  # the arc starts 0.15 m outside the edge
    phi_w = np.arccos(1 - (depth + level) / R_CLIP)
    entry = np.array(c) + R_CLIP * np.array([np.cos(phi_e), np.sin(phi_e)])
    wait_pt = np.array(c) + R_CLIP * np.array([np.cos(phi_w), np.sin(phi_w)])
    k = 1 / R_CLIP
    a = np.concatenate([hermite(Z0[:2], Z0[2], 0.0, entry, phi_e + np.pi / 2, k), arc(c, R_CLIP, phi_e, phi_w)[1:]])
    g = hermite(wait_pt, phi_w + np.pi / 2, k, GREEN_STOP[:2], GREEN_STOP[2], 0.0)
    pa = leg(a, V_CRUISE, up=0.2, down=down)
    pg = leg(g, V_CRUISE)
    p1 = np.concatenate([pa, np.repeat(pa[-1:], wait - 1, 0), pg])
    t_green = len(p1) - 1
    a1 = t_green + 5
    b1 = a1 + int(round(2 / H))
    leave = b1 + 5
    pblue = leg(sweep(), V_CRUISE)
    t_blue = leave + len(pblue) - 1
    a2, b2 = t_blue - 30, t_blue + 30
    T = b2 + int(round(2 / H)) + 1
    p1 = np.concatenate([p1, np.repeat(p1[-1:], leave - t_green, 0), pblue[1:]])
    p1 = np.concatenate([p1, np.repeat(p1[-1:], T - len(p1), 0)])
    p2 = leg(hermite(Z0[:2], Z0[2], 0.0, GREEN_STOP[:2], GREEN_STOP[2], 0.0), V_CRUISE)
    p2 = np.concatenate([p2, np.repeat(p2[-1:], leave - (len(p2) - 1), 0), pblue[1:]])
    p2 = np.concatenate([p2, np.repeat(p2[-1:], T - len(p2), 0)])
    timing = {"wait": wait, "depth": depth, "level": level, "t_green": t_green, "a1": a1, "b1": b1,
              "leave": leave, "t_blue": t_blue, "a2": a2, "b2": b2, "T": T}
    return inputs_from_positions(p1, Z0[2]), inputs_from_positions(p2, Z0[2]), timing


def sweep():
    """Dense points of the drive from the stop in Green to the stop in Blue."""
    return hermite(GREEN_STOP[:2], GREEN_STOP[2], 0.0, BLUE_STOP[:2], BLUE_STOP[2], 0.0, scale=1.2)


# ------------------------------------------------------------------
# matched smoothings

def _per_row(fn, z, param, mask):
    """Apply fn(z_row, param_row, mask_row) to every row of the last two axes."""
    valid = jnp.broadcast_to(jnp.ones((), bool) if mask is None else mask, z.shape)
    zr, vr = z.reshape(-1, z.shape[-1]), valid.reshape(-1, z.shape[-1])
    pr = jnp.broadcast_to(param, z.shape[:-1]).reshape(-1)
    return jax.vmap(fn)(zr, pr, vr).reshape(z.shape[:-1])


def _count(z, mask):
    valid = jnp.broadcast_to(jnp.ones((), bool) if mask is None else mask, z.shape)
    return jnp.maximum(jnp.sum(valid, -1), 2).astype(z.dtype)  # one entry: exact for any parameter


def matched(name, eps):
    """(max_reduce, min_reduce) at worst-case error eps per node, for name in
    'exact', 'lse_plain', 'sparsemax', 'gm01', 'gm10'."""
    if name == "exact":
        return exact_max, exact_min
    if name == "gm01":
        return gm_power(0.0, 1.0)
    if name == "gm10":
        return gm_power(-10.0, 10.0)
    if name == "lse_plain":
        def mx(z, param=None, mask=None):
            return _per_row(lse_plain_max, z, jnp.log(_count(z, mask)) / eps, mask)

        def mn(z, param=None, mask=None):
            return _per_row(lse_min, z, jnp.log(_count(z, mask)) / eps, mask)
        return mx, mn
    if name == "sparsemax":
        def mx(z, param=None, mask=None):
            return _per_row(lower_max, z, 2 * eps / (1 - 1 / _count(z, mask)), mask)

        def mn(z, param=None, mask=None):
            return _per_row(lower_min, z, 2 * eps / (1 - 1 / _count(z, mask)), mask)
        return mx, mn
    raise ValueError("unknown smoothing " + repr(name))


METHODS = ("exact", "lse_plain", "gm01", "gm10", "sparsemax")


def until_on_operands(phi, psi, a1, b1, sem):
    """Value at t = 0 of phi U_[a1, b1] psi for operand traces phi, psi (T,) under the pair sem."""
    T = phi.shape[-1]
    prog = compile_formula(Until((a1, b1), Atom(0), Atom(1)), T, reads=[(Until((a1, b1), Atom(0), Atom(1)), [0])])
    return read(prog, evaluate(prog, jnp.stack([phi, psi], -1), sem))[..., 0]


def operand_traces(S, sem):
    """Traces (T,) of not Red and Green under the pair sem, from leaf scores S (T, 20)."""
    T = S.shape[-2]
    p1 = compile_formula(Not(box(0)), T)
    p2 = compile_formula(box(1), T)
    return evaluate(p1, S, sem)[-1], evaluate(p2, S, sem)[-1]


def spec_values(S, timing, sem):
    """(specification at t = 0, until at t = 0) under the pair sem."""
    spec, until = specification(timing["a1"], timing["b1"], timing["a2"], timing["b2"], timing["T"])
    prog = compile_formula(spec, timing["T"], reads=[(spec, [0]), (until, [0])])
    out = read(prog, evaluate(prog, S, sem))
    return out[..., 0], out[..., 1]


def until_weight(S, timing, sem, viol):
    """Sum over the samples in viol (bool, T) of the derivative of the until value at t = 0 with
    respect to the value of not Red at that sample, by reverse-mode AD; also the until value."""
    phi, psi = operand_traces(S, sem)
    f = lambda ph: until_on_operands(ph, psi, timing["a1"], timing["b1"], sem)  # noqa: E731
    val, g = jax.value_and_grad(f)(phi)
    return val, jnp.sum(jnp.where(viol, g, 0.0)), g

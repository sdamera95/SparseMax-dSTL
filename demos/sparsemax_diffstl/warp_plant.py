"""The Panda through the MuJoCo-Warp adjoint fork: a taped rollout with CUDA-graph steps (E027).

The plant is the etaoxing mujoco_warp adjoint fork, the project's mujoco-warp (pyproject
[tool.uv.sources], the clone third_party/mujoco_warp_adjoint at 357a75d). Under a wp.Tape its
out-of-place step(m, d, d_out) records one analytic adjoint: the implicit function theorem at
the converged constraint solve (H lam = adj qacc, one Cholesky solve) and hand-written adjoints
of the smooth dynamics and of the Euler and implicitfast integrators (fork
mujoco_warp/_src/adjoint.py). The model is tasks.panda.model() unchanged: gravity compensation
through the actuators, joint-level actuator force ranges, joint limits, 2 ms physics,
implicitfast, Newton.

THE TWO LEAVES THIS MODULE SUPPLIES. The fork refuses gravity compensation routed through the
actuators (fork smooth_adjoint.py:80-81, "gravcomp routed to an actuator (jnt_actgravcomp;
force-limit clamp)"), which is how the task model applies it, and its control adjoint
(forward_adjoint.py:117-134) is gain * moment . lam without the clamps. For a joint j driven by
its own actuator a (gear 1, fixed gain g, no bias; checked by supported()), MuJoCo applies at
every physics step (MuJoCo C engine_forward.c mj_fwdActuation, fork forward.py:1235-1263)

    tau_j = clip(clip(g clip(u_a, ctrlrange), forcerange) + tau_g,j(q), actfrcrange) = clip(s_j, actfrcrange),

so, with lam the adjoint of the generalized force that the fork's backward already solves for
(the leaf its own control adjoint reads),

    adj u_a  = g lam_j 1[u_a inside ctrlrange] 1[g u_a inside forcerange] 1[s_j inside actfrcrange],
    adj q   += (d tau_g / d q)^T (lam * 1[s inside actfrcrange]).

step_backward() below runs the fork's own step backward, then multiplies the fork's control
adjoint by the indicator and adds the position term through the fork's own passive-gravcomp
leaf (smooth_adjoint.gravcomp_qpos_vjp) with the masked lam, on a model view whose
jnt_actgravcomp is zero so that the leaf seeds every dof. The fork's support check runs on that
view, so every other unsupported-feature check still applies, and the task model is then
registered as supported. Nothing in the fork is edited.

Boundary convention (a choice: at a clamp boundary the derivative does not exist): a clamp is
active only strictly outside its range. A command exactly at its bound, where the box of the
normalized controls puts it, gets the derivative from inside the range. MJX's jnp.clip gives 1/2
per clip at a tie instead.

ROLLOUT. Plant(B, T_max) holds B worlds. rollout(x0 (B, 14), V (B, T, 7)) applies u = V u_max
(V in the pilot's normalized units, u_max = tasks.panda.torque_limit()) held over n_sub physics
steps per interval and returns the states X (B, T + 1, 14) at the interval boundaries. As in
tasks.panda.interval_map, the solver warm start is zero at the start of every interval and
carried across its substeps, so the rollout is the composition of the interval map f_h(x, u).
vjp(C (B, T + 1, 14)) returns (gV (B, T, 7), gx0 (B, 14)), the vector-Jacobian product of the
last rollout for the state cotangents C.

The forward stores (qpos, qvel, qacc_warmstart) before every physics step. The backward
restores step k, re-runs it on a fresh wp.Tape and back-propagates the running state cotangent
(checkpointing of every physics step, as the fork's contrib/diffsim/_rollout.py and grip-mppi's
newton_grad.rollout do). The step index lives in a device counter, so one forward step and one
backward step are each a fixed launch sequence, captured once as a CUDA graph and replayed; the
Python loops replay graphs, one launch per physics step, and run no numerics. The pattern is
adapted from grip-mppi src/grip/newton_grad/rollout.py (branch fork-adjoint, BSD-3), not imported.
"""

import dataclasses

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import adjoint as mjw_adjoint
from mujoco_warp._src import adjoint_util
from mujoco_warp._src import forward as mjw_forward
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src.types import BiasType, DisableBit, DynType, GainType, JointType, TrnType, vec10

NJMAX = 32
NCONMAX = 8
_ENABLED = {"done": False}
_CONTEXTS = {}  # id(BackwardContext) -> the leaves' scratch arrays and model view


# ------------------------------------------------------------------
# the leaves

@wp.kernel(enable_backward=False)
def _clamp_masks(
    ctrllimited: wp.array(dtype=bool), ctrlrange: wp.array2d(dtype=wp.vec2), forcelimited: wp.array(dtype=bool),
    forcerange: wp.array2d(dtype=wp.vec2), gainprm: wp.array2d(dtype=vec10), act_dof: wp.array(dtype=int),
    dof_jntid: wp.array(dtype=int), jnt_actfrclimited: wp.array(dtype=bool), jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
    jnt_actgravcomp: wp.array(dtype=int), gravity_enabled: int, ctrl: wp.array2d(dtype=float),
    qfrc_gravcomp: wp.array2d(dtype=float), cmask: wp.array2d(dtype=float), jmask: wp.array2d(dtype=float)):
    w, a = wp.tid()
    i = act_dof[a]
    j = dof_jntid[i]
    inside = float(1.0)
    u = ctrl[w, a]
    if ctrllimited[a]:
        r = ctrlrange[w % ctrlrange.shape[0], a]
        if u < r[0] or u > r[1]:
            inside = 0.0
        u = wp.clamp(u, r[0], r[1])
    f = gainprm[w % gainprm.shape[0], a][0] * u
    if forcelimited[a]:
        r = forcerange[w % forcerange.shape[0], a]
        if f < r[0] or f > r[1]:
            inside = 0.0
        f = wp.clamp(f, r[0], r[1])
    s = f
    if gravity_enabled != 0 and jnt_actgravcomp[j] != 0:
        s = s + qfrc_gravcomp[w, i]
    jm = float(1.0)
    if jnt_actfrclimited[j]:
        r = jnt_actfrcrange[w % jnt_actfrcrange.shape[0], j]
        if s < r[0] or s > r[1]:
            jm = 0.0
    jmask[w, i] = jm
    cmask[w, a] = inside * jm


@wp.kernel(enable_backward=False)
def _mask_lam(lam: wp.array2d(dtype=float), jmask: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, i = wp.tid()
    out[w, i] = lam[w, i] * jmask[w, i]


@wp.kernel(enable_backward=False)
def _mul(x: wp.array2d(dtype=float), mask: wp.array2d(dtype=float)):
    w, i = wp.tid()
    x[w, i] = x[w, i] * mask[w, i]


@wp.kernel(enable_backward=False)
def _sub(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    w, i = wp.tid()
    x[w, i] = x[w, i] - y[w, i]


def step_backward(m, d, d_out):
    """The fork's step backward plus the actuator gravity-compensation and clamp leaves (module
    docstring). Registered for every taped out-of-place step by enable()."""
    bc = mjw_adjoint._ACTIVE_BACKWARD_CONTEXT.get()
    if bc is None or id(bc) not in _CONTEXTS:
        raise RuntimeError("step backward outside mujoco_warp.backward_context(bc) with bc from warp_plant.context()")
    sc = _CONTEXTS[id(bc)]
    gravity = int(not (int(m.opt.disableflags) & DisableBit.GRAVITY))
    # the masks read the step's input state: d.ctrl and d_out.qfrc_gravcomp (computed at d's
    # configuration by the step's forward), before any adjoint launch touches d_out
    wp.launch(_clamp_masks, dim=(d.nworld, m.nu),
              inputs=[m.actuator_ctrllimited, m.actuator_ctrlrange, m.actuator_forcelimited, m.actuator_forcerange,
                      m.actuator_gainprm, sc["act_dof"], m.dof_jntid, m.jnt_actfrclimited, m.jnt_actfrcrange,
                      m.jnt_actgravcomp, gravity, d.ctrl, d_out.qfrc_gravcomp],
              outputs=[sc["cmask"], sc["jmask"]])
    mjw_adjoint.step_backward(m, d, d_out, bc)
    lam = bc.solver_ctx.search
    wp.launch(_mask_lam, dim=(d.nworld, m.nv), inputs=[lam, sc["jmask"]], outputs=[sc["lam"]])
    view = dataclasses.replace(d_out, qpos=d.qpos, qvel=d.qvel)
    res = smooth_adjoint.gravcomp_qpos_vjp(sc["model_view"], view, sc["lam"], bc=bc)
    wp.launch(_sub, dim=d.qpos.shape, inputs=[d.qpos.grad, res])
    wp.launch(_mul, dim=d.ctrl.shape, inputs=[d.ctrl.grad, sc["cmask"]])


def enable():
    """Turn on the fork's analytic backward with this module's step backward (idempotent).

    mujoco_warp.enable_grad() rebinds wp.sqrt process-wide and registers quaternion adjoints;
    both change the content hash of Warp modules declared afterwards, so modules loaded before
    the call are unloaded and reload consistently on next use (grip-mppi E014)."""
    if _ENABLED["done"]:
        return
    from warp._src import context as wp_context
    mjw.enable_grad()
    mjw_forward.register_step_backward(step_backward, mjw_adjoint.step_backward_arrays)
    for mod in [mod for mod in wp_context.user_modules.values() if getattr(mod, "execs", None)]:
        mod.unload()
    _ENABLED["done"] = True


def supported(mjm, m):
    """Check the model against the fork's support test with jnt_actgravcomp zeroed (every other
    test of the fork applies) and against what step_backward's leaves assume; register m with
    the fork's support cache. Returns (the model view the gravcomp leaf uses, actuator -> dof)."""
    if mjm.nu != mjm.nv:
        raise NotImplementedError("one actuator per dof expected")
    if np.any(mjm.actuator_trntype != TrnType.JOINT) or np.any(mjm.actuator_gaintype != GainType.FIXED) \
            or np.any(mjm.actuator_biastype != BiasType.NONE) or np.any(mjm.actuator_dyntype != DynType.NONE):
        raise NotImplementedError("the clamp leaf assumes joint transmissions, fixed gains, no bias, no dynamics")
    jid = mjm.actuator_trnid[:, 0]
    if np.any((mjm.jnt_type[jid] != JointType.HINGE) & (mjm.jnt_type[jid] != JointType.SLIDE)):
        raise NotImplementedError("the clamp leaf assumes hinge or slide joints")
    if np.any(mjm.actuator_gear[:, 0] != 1.0) or np.any(mjm.actuator_gear[:, 1:] != 0.0):
        raise NotImplementedError("the clamp leaf assumes gear 1")
    act_dof = mjm.jnt_dofadr[jid]
    if sorted(act_dof.tolist()) != list(range(mjm.nv)):
        raise NotImplementedError("actuators must drive distinct dofs")
    view = dataclasses.replace(m, jnt_actgravcomp=wp.zeros_like(m.jnt_actgravcomp))
    smooth_adjoint.assert_smooth_supported(view)
    smooth_adjoint._SUPPORTED_CACHE[id(m)] = (m, True)
    return view, act_dof


def context(mjm, m, d):
    """mujoco_warp.create_backward_context plus this module's scratch for the leaves."""
    view, act_dof = supported(mjm, m)
    bc = mjw.create_backward_context(m, d)
    _CONTEXTS[id(bc)] = {"model_view": view, "act_dof": wp.array(act_dof.astype(np.int32), dtype=int),
                         "cmask": wp.zeros((d.nworld, mjm.nu), dtype=float), "jmask": wp.zeros((d.nworld, mjm.nv), dtype=float),
                         "lam": wp.zeros_like(bc.solver_ctx.search), "bc": bc}
    return bc


# ------------------------------------------------------------------
# device-indexed step kernels

@wp.kernel(enable_backward=False)
def _boundary(k_dev: wp.array(dtype=int), n_sub: int, warm: wp.array2d(dtype=float)):
    w, i = wp.tid()
    if k_dev[0] % n_sub == 0:
        warm[w, i] = 0.0


@wp.kernel(enable_backward=False)
def _save(k_dev: wp.array(dtype=int), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
          warm: wp.array2d(dtype=float), hq: wp.array3d(dtype=float), hv: wp.array3d(dtype=float),
          hw: wp.array3d(dtype=float)):
    w, i = wp.tid()
    k = k_dev[0]
    hq[k, w, i] = qpos[w, i]
    hv[k, w, i] = qvel[w, i]
    hw[k, w, i] = warm[w, i]


@wp.kernel(enable_backward=False)
def _restore(k_dev: wp.array(dtype=int), hq: wp.array3d(dtype=float), hv: wp.array3d(dtype=float),
             hw: wp.array3d(dtype=float), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
             warm: wp.array2d(dtype=float)):
    w, i = wp.tid()
    k = k_dev[0]
    qpos[w, i] = hq[k, w, i]
    qvel[w, i] = hv[k, w, i]
    warm[w, i] = hw[k, w, i]


@wp.kernel(enable_backward=False)
def _load_ctrl(k_dev: wp.array(dtype=int), n_sub: int, table: wp.array3d(dtype=float), umax: wp.array(dtype=float),
               ctrl: wp.array2d(dtype=float)):
    w, a = wp.tid()
    ctrl[w, a] = table[k_dev[0] / n_sub, w, a] * umax[a]


@wp.kernel(enable_backward=False)
def _advance(k_dev: wp.array(dtype=int), delta: int):
    k_dev[0] = k_dev[0] + delta


@wp.kernel(enable_backward=False)
def _collect_state(k_dev: wp.array(dtype=int), n_sub: int, gq: wp.array2d(dtype=float), gv: wp.array2d(dtype=float),
                   cot: wp.array3d(dtype=float), Gq: wp.array2d(dtype=float), Gv: wp.array2d(dtype=float)):
    # after back-propagating step k: the cotangent of state k, plus the sample's own cotangent
    # when state k is an interval boundary
    w, i = wp.tid()
    k = k_dev[0]
    nq = Gq.shape[1]
    x = gq[w, i]
    y = gv[w, i]
    if k % n_sub == 0:
        x = x + cot[k / n_sub, w, i]
        y = y + cot[k / n_sub, w, nq + i]
    Gq[w, i] = x
    Gv[w, i] = y


@wp.kernel(enable_backward=False)
def _collect_ctrl(k_dev: wp.array(dtype=int), n_sub: int, gctrl: wp.array2d(dtype=float), umax: wp.array(dtype=float),
                  gtable: wp.array3d(dtype=float)):
    w, a = wp.tid()
    t = k_dev[0] / n_sub
    gtable[t, w, a] = gtable[t, w, a] + gctrl[w, a] * umax[a]


@wp.kernel(enable_backward=False)
def _gather(n_sub: int, hq: wp.array3d(dtype=float), hv: wp.array3d(dtype=float), X: wp.array3d(dtype=float)):
    w, t, i = wp.tid()
    nq = hq.shape[2]
    if i < nq:
        X[w, t, i] = hq[t * n_sub, w, i]
    else:
        X[w, t, i] = hv[t * n_sub, w, i - nq]


# ------------------------------------------------------------------
# the plant

class Plant:
    """B worlds of one MuJoCo model on the fork with a differentiable rollout (module docstring).

    Args:
        nworld: B.
        max_intervals: the longest rollout T; the history and the graphs are sized for it once.
        mjm: the MjModel; tasks.panda.model() when None. nq must equal nv (hinge and slide joints).
        n_sub: physics steps per interval (10 = h of 0.02 s at 2 ms).
        umax: the normalization of the controls; tasks.panda.torque_limit() when None.
        graph: capture one forward and one backward physics step as CUDA graphs; False runs the
            same launch sequences eagerly (CPU devices always run eagerly).
        device: a Warp device; CUDA graph capture needs CUDA ordinal 0 (pin CUDA_VISIBLE_DEVICES).
    """

    def __init__(self, nworld, max_intervals, mjm=None, n_sub=10, umax=None, graph=True, device="cuda:0"):
        from .tasks import panda
        enable()
        self.mjm = panda.model() if mjm is None else mjm
        if self.mjm.nq != self.mjm.nv:
            raise NotImplementedError("nq must equal nv")
        self.device = wp.get_device(device)
        self.B, self.T_max, self.n_sub = int(nworld), int(max_intervals), int(n_sub)
        self.nq = self.mjm.nq
        self.N_max = self.T_max * self.n_sub
        umax = panda.torque_limit() if umax is None else np.asarray(umax)
        B, nq, nu = self.B, self.nq, self.mjm.nu
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.make_data(self.mjm, nworld=B, nconmax=NCONMAX, njmax=NJMAX)
            self.d.contact.geomcollisionid.zero_()  # allocated uninitialized (fork io.py:1967); grip-mppi E014
            self.d_out = adjoint_util._clone_nograd(self.d)
            self.bc = context(self.mjm, self.m, self.d)
            z = lambda *s: wp.zeros(s, dtype=float)  # noqa: E731
            self.umax = wp.array(umax.astype(np.float32), dtype=float)
            self.k_dev = wp.zeros(1, dtype=int)
            self.hq, self.hv, self.hw = z(self.N_max + 1, B, nq), z(self.N_max + 1, B, nq), z(self.N_max + 1, B, nq)
            self.table, self.gtable = z(self.T_max, B, nu), z(self.T_max, B, nu)
            self.cot = z(self.T_max + 1, B, 2 * nq)
            self.Gq, self.Gv = z(B, nq), z(B, nq)
        self.graph = bool(graph) and self.device.is_cuda
        self._fwd_graph = self._bwd_graph = None
        self._keep = []
        self._T = None
        self._capture()

    # ------------------------------------------------------------------
    # one physics step forward and backward (the captured launch sequences)

    def _fwd(self):
        d, B, nq = self.d, self.B, self.nq
        wp.launch(_boundary, dim=(B, nq), inputs=[self.k_dev, self.n_sub, d.qacc_warmstart])
        wp.launch(_save, dim=(B, nq), inputs=[self.k_dev, d.qpos, d.qvel, d.qacc_warmstart], outputs=[self.hq, self.hv, self.hw])
        wp.launch(_load_ctrl, dim=(B, self.mjm.nu), inputs=[self.k_dev, self.n_sub, self.table, self.umax], outputs=[d.ctrl])
        mjw.step(self.m, d, self.d_out)
        mjw_forward._copy_state(self.d_out, d)
        wp.launch(_advance, dim=1, inputs=[self.k_dev, 1])

    def _bwd(self):
        d, d_out, B, nq = self.d, self.d_out, self.B, self.nq
        wp.launch(_restore, dim=(B, nq), inputs=[self.k_dev, self.hq, self.hv, self.hw], outputs=[d.qpos, d.qvel, d.qacc_warmstart])
        wp.launch(_load_ctrl, dim=(B, self.mjm.nu), inputs=[self.k_dev, self.n_sub, self.table, self.umax], outputs=[d.ctrl])
        tape = wp.Tape()
        with tape:
            mjw.step(self.m, d, d_out)
        tape.zero()
        wp.copy(d_out.qpos.grad, self.Gq)
        wp.copy(d_out.qvel.grad, self.Gv)
        tape.backward()
        wp.launch(_collect_state, dim=(B, nq), inputs=[self.k_dev, self.n_sub, d.qpos.grad, d.qvel.grad, self.cot],
                  outputs=[self.Gq, self.Gv])
        wp.launch(_collect_ctrl, dim=(B, self.mjm.nu), inputs=[self.k_dev, self.n_sub, d.ctrl.grad, self.umax], outputs=[self.gtable])
        wp.launch(_advance, dim=1, inputs=[self.k_dev, -1])
        return tape

    def _reset(self):
        q0 = self.mjm.qpos0.astype(np.float32)
        self.d.qpos.assign(np.repeat(q0[None], self.B, 0))
        self.d.qvel.zero_()
        self.d.qacc_warmstart.zero_()
        self.k_dev.zero_()

    def _capture(self):
        """Run one forward and one backward step eagerly (compiles every module and runs the host
        checks outside a capture), then capture each as a graph."""
        with wp.ScopedDevice(self.device), mjw.backward_context(self.bc):
            self._reset()
            self._fwd()
            self.k_dev.zero_()
            self._bwd().reset()
            wp.synchronize_device(self.device)
            if self.graph:
                self._reset()
                with wp.ScopedCapture(device=self.device, force_module_load=False) as cap:
                    self._fwd()
                self._fwd_graph = cap.graph
                self._reset()
                with wp.ScopedCapture(device=self.device, force_module_load=False) as cap:
                    self._keep.append(self._bwd())
                self._bwd_graph = cap.graph
                # instantiate both executables now, on this thread (grip-mppi rollout.py)
                self._reset()
                wp.capture_launch(self._fwd_graph)
                self._reset()
                wp.capture_launch(self._bwd_graph)
            self._reset()
            self.gtable.zero_()
            wp.synchronize_device(self.device)

    # ------------------------------------------------------------------
    # public API

    def set_mocap(self, pos):
        """Mocap body positions (B, nmocap, 3), held over the following rollouts."""
        self.d.mocap_pos.assign(np.asarray(pos, np.float32).reshape(self.B, self.mjm.nmocap, 3))

    def rollout(self, x0, V):
        """States X (B, T + 1, 14) at the interval boundaries from x0 (B, 14) under the normalized
        controls V (B, T, 7); numpy in, numpy out, float32."""
        x0 = np.asarray(x0, np.float32).reshape(self.B, 2 * self.nq)
        V = np.asarray(V, np.float32).reshape(self.B, -1, self.mjm.nu)
        T = V.shape[1]
        if not 1 <= T <= self.T_max:
            raise ValueError("horizon " + str(T) + " outside 1.." + str(self.T_max))
        N = T * self.n_sub
        with wp.ScopedDevice(self.device):
            tab = np.zeros((self.T_max, self.B, self.mjm.nu), np.float32)
            tab[:T] = np.swapaxes(V, 0, 1)
            self.table.assign(tab)
            self.d.qpos.assign(x0[:, :self.nq])
            self.d.qvel.assign(x0[:, self.nq:])
            self.d.qacc_warmstart.zero_()
            self.d.time.zero_()
            self.k_dev.zero_()
            for _ in range(N):  # time recursion: one graph replay per physics step
                if self._fwd_graph is not None:
                    wp.capture_launch(self._fwd_graph)
                else:
                    self._fwd()
            wp.launch(_save, dim=(self.B, self.nq), inputs=[self.k_dev, self.d.qpos, self.d.qvel, self.d.qacc_warmstart],
                      outputs=[self.hq, self.hv, self.hw])
            X = wp.empty((self.B, T + 1, 2 * self.nq), dtype=float)
            wp.launch(_gather, dim=(self.B, T + 1, 2 * self.nq), inputs=[self.n_sub, self.hq, self.hv], outputs=[X])
            X = X.numpy()
        if not np.isfinite(X).all():
            raise FloatingPointError("non-finite state in the rollout")
        self._T = T
        return X

    def vjp(self, C):
        """(gV (B, T, 7), gx0 (B, 14)): the vector-Jacobian product of the last rollout for the
        state cotangents C (B, T + 1, 14); gV is in the normalized control units."""
        if self._T is None:
            raise RuntimeError("vjp() before rollout()")
        T, N, nq = self._T, self._T * self.n_sub, self.nq
        C = np.asarray(C, np.float32).reshape(self.B, T + 1, 2 * nq)
        with wp.ScopedDevice(self.device), mjw.backward_context(self.bc):
            cot = np.zeros((self.T_max + 1, self.B, 2 * nq), np.float32)
            cot[:T + 1] = np.swapaxes(C, 0, 1)
            self.cot.assign(cot)
            self.Gq.assign(C[:, T, :nq])
            self.Gv.assign(C[:, T, nq:])
            self.gtable.zero_()
            self.k_dev.fill_(N - 1)
            for _ in range(N):  # reverse time recursion: one graph replay per physics step
                if self._bwd_graph is not None:
                    wp.capture_launch(self._bwd_graph)
                else:
                    self._bwd().reset()
            gV = np.swapaxes(self.gtable.numpy()[:T], 0, 1).copy()
            gx0 = np.concatenate([self.Gq.numpy(), self.Gv.numpy()], -1)
        if not (np.isfinite(gV).all() and np.isfinite(gx0).all()):
            raise FloatingPointError("non-finite cotangent in the backward")
        return gV, gx0

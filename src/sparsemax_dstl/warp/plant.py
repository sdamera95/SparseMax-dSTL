"""Batched rollout of a torque-driven MuJoCo model and its vector-Jacobian product on the mujoco_warp adjoint
fork, with the adjoints of the actuator clamps and of gravity compensation applied through the actuators."""

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
_CONTEXTS = {}  # id(BackwardContext) -> scratch arrays and model view of step_backward


# ------------------------------------------------------------------
# adjoints of the actuator clamps and of gravity compensation through the actuators

@wp.kernel(enable_backward=False)
def _clamp_masks(
    ctrllimited: wp.array(dtype=bool), ctrlrange: wp.array2d(dtype=wp.vec2), forcelimited: wp.array(dtype=bool),
    forcerange: wp.array2d(dtype=wp.vec2), gainprm: wp.array2d(dtype=vec10), act_dof: wp.array(dtype=int),
    dof_jntid: wp.array(dtype=int), jnt_actfrclimited: wp.array(dtype=bool), jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
    jnt_actgravcomp: wp.array(dtype=int), gravity_enabled: int, ctrl: wp.array2d(dtype=float),
    qfrc_gravcomp: wp.array2d(dtype=float), cmask: wp.array2d(dtype=float), jmask: wp.array2d(dtype=float)):
    # cmask[w, a] = 1 when the control, the actuator force and the joint's total actuator force (with gravity
    # compensation) all lie inside their ranges, bounds included; jmask[w, i] is the last of the three tests
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
    """The fork's step backward, then the control adjoint masked by the actuator clamps and the qpos adjoint
    of gravity compensation through the actuators; enable() registers it as the fork's step backward."""
    bc = mjw_adjoint._ACTIVE_BACKWARD_CONTEXT.get()
    if bc is None or id(bc) not in _CONTEXTS:
        raise RuntimeError("step backward outside mujoco_warp.backward_context(bc) with bc from warp_plant.context()")
    sc = _CONTEXTS[id(bc)]
    gravity = int(not (int(m.opt.disableflags) & DisableBit.GRAVITY))
    # before the fork's backward: the masks read d.ctrl and d_out.qfrc_gravcomp as the forward step left them
    wp.launch(_clamp_masks, dim=(d.nworld, m.nu),
              inputs=[m.actuator_ctrllimited, m.actuator_ctrlrange, m.actuator_forcelimited, m.actuator_forcerange,
                      m.actuator_gainprm, sc["act_dof"], m.dof_jntid, m.jnt_actfrclimited, m.jnt_actfrcrange,
                      m.jnt_actgravcomp, gravity, d.ctrl, d_out.qfrc_gravcomp],
              outputs=[sc["cmask"], sc["jmask"]])
    mjw_adjoint.step_backward(m, d, d_out, bc)
    lam = bc.solver_ctx.search
    wp.launch(_mask_lam, dim=(d.nworld, m.nv), inputs=[lam, sc["jmask"]], outputs=[sc["lam"]])
    # qpos.grad += d(lam . qfrc_gravcomp) / d qpos with lam masked by jmask; model_view has jnt_actgravcomp
    # zeroed so that gravcomp_qpos_vjp covers every dof
    view = dataclasses.replace(d_out, qpos=d.qpos, qvel=d.qvel)
    res = smooth_adjoint.gravcomp_qpos_vjp(sc["model_view"], view, sc["lam"], bc=bc)
    wp.launch(_sub, dim=d.qpos.shape, inputs=[d.qpos.grad, res])
    wp.launch(_mul, dim=d.ctrl.shape, inputs=[d.ctrl.grad, sc["cmask"]])


def enable():
    """Switch on the fork's reverse mode with this module's step backward (idempotent)."""
    if _ENABLED["done"]:
        return
    from warp._src import context as wp_context
    mjw.enable_grad()
    mjw_forward.register_step_backward(step_backward, mjw_adjoint.step_backward_arrays)
    # enable_grad() rebinds wp.sqrt, so the modules loaded before it are unloaded and rebuild on next use
    for mod in [mod for mod in wp_context.user_modules.values() if getattr(mod, "execs", None)]:
        mod.unload()
    _ENABLED["done"] = True


def supported(mjm, m):
    """Check the model against the assumptions of step_backward and against the fork's support test with
    jnt_actgravcomp zeroed, and mark it supported. Returns the zeroed model view and each actuator's dof."""
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
    """mujoco_warp.create_backward_context plus the scratch arrays of step_backward."""
    view, act_dof = supported(mjm, m)
    bc = mjw.create_backward_context(m, d)
    _CONTEXTS[id(bc)] = {"model_view": view, "act_dof": wp.array(act_dof.astype(np.int32), dtype=int),
                         "cmask": wp.zeros((d.nworld, mjm.nu), dtype=float), "jmask": wp.zeros((d.nworld, mjm.nv), dtype=float),
                         "lam": wp.zeros_like(bc.solver_ctx.search), "bc": bc}
    return bc


# ------------------------------------------------------------------
# kernels of one physics step; k_dev[0] is the index of the physics step

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
    """nworld worlds of one MuJoCo model (nq = nv; the Panda of tasks.panda by default) with a differentiable
    rollout of at most max_intervals intervals of n_sub physics steps; controls are in units of umax."""

    def __init__(self, nworld, max_intervals, mjm=None, n_sub=10, umax=None, graph=True, device="cuda:0"):
        from ..tasks import panda
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
            self.d.contact.geomcollisionid.zero_()  # make_data leaves it uninitialized
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
        """Run a forward and a backward step eagerly (loading every module), then capture each as a graph."""
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
                # the first launch of a graph instantiates its executable
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
        """States X (B, T + 1, 2 nq) at the interval boundaries from x0 (B, 2 nq) under the normalized controls
        V (B, T, nu); float32 NumPy in and out."""
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
        """(gV (B, T, nu), gx0 (B, 2 nq)): the vector-Jacobian product of the last rollout with the state
        cotangents C (B, T + 1, 2 nq); gV is per normalized control."""
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

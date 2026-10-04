"""The predicates of the manipulator example (Section V-B) as Warp kernels differentiated by Warp's
tape, with the kinematics and point velocities of a serial hinge chain from MuJoCo's model arrays."""
from contextlib import nullcontext
from functools import cache

import mujoco
import numpy as np
import warp as wp

from ..tasks import workspace as Wk

MAX_BODIES = 16  # the unrolled chain loop covers bodies 1..15 (Warp's max_unroll is 16)


@cache
def _kernels(dtype):
    vec3 = wp.types.vector(3, dtype)
    quat = wp.types.quaternion(dtype)

    @wp.func
    def n_minus(e: vec3, eps: dtype):
        # sqrt(|e|^2 + eps^2) - eps
        s = wp.dot(e, e)
        return s / (wp.sqrt(s + eps * eps) + eps)

    @wp.kernel(enable_backward=True, module="unique")
    def chain(qpos: wp.array2d(dtype=dtype), qvel: wp.array2d(dtype=dtype), qpos0: wp.array(dtype=dtype),
              body_pos: wp.array(dtype=vec3), body_quat: wp.array(dtype=quat), body_jnt: wp.array(dtype=int),
              jnt_pos: wp.array(dtype=vec3), jnt_axis: wp.array(dtype=vec3), jnt_qadr: wp.array(dtype=int),
              jnt_dadr: wp.array(dtype=int), nbody: int,
              xpos: wp.array2d(dtype=vec3), xquat: wp.array2d(dtype=quat), omega: wp.array2d(dtype=vec3),
              mom: wp.array2d(dtype=vec3)):
        # om = sum_j qd_j axis_j and mm = sum_j qd_j axis_j x anchor_j over the hinges up to body b;
        # a point c of body b has velocity om x c - mm
        w = wp.tid()
        zero = dtype(0.0)
        pos = vec3(zero, zero, zero)
        rot = quat(zero, zero, zero, dtype(1.0))
        om = vec3(zero, zero, zero)
        mm = vec3(zero, zero, zero)
        xpos[w, 0] = pos
        xquat[w, 0] = rot
        omega[w, 0] = om
        mom[w, 0] = mm
        for b in range(1, 16):  # constant range: unrolled
            if b < nbody:
                pos = pos + wp.quat_rotate(rot, body_pos[b])
                rot = rot * body_quat[b]
                j = body_jnt[b]
                if j >= 0:
                    anchor = pos + wp.quat_rotate(rot, jnt_pos[j])
                    axis = wp.quat_rotate(rot, jnt_axis[j])
                    qd = qvel[w, jnt_dadr[j]]
                    om = om + axis * qd
                    mm = mm + wp.cross(axis, anchor) * qd
                    rot = rot * wp.quat_from_axis_angle(jnt_axis[j], qpos[w, jnt_qadr[j]] - qpos0[jnt_qadr[j]])
                    pos = anchor - wp.quat_rotate(rot, jnt_pos[j])
                rot = wp.normalize(rot)
                xpos[w, b] = pos
                xquat[w, b] = rot
                omega[w, b] = om
                mom[w, b] = mm

    @wp.kernel(enable_backward=True, module="unique")
    def points(xpos: wp.array2d(dtype=vec3), xquat: wp.array2d(dtype=quat), omega: wp.array2d(dtype=vec3),
               mom: wp.array2d(dtype=vec3), pt_body: wp.array(dtype=int), pt_off: wp.array(dtype=vec3),
               P: wp.array2d(dtype=vec3), Pd: wp.array2d(dtype=vec3)):
        w, i = wp.tid()
        b = pt_body[i]
        c = xpos[w, b] + wp.quat_rotate(xquat[w, b], pt_off[i])
        P[w, i] = c
        Pd[w, i] = wp.cross(omega[w, b], c) - mom[w, b]

    @wp.kernel(enable_backward=True, module="unique")
    def atoms(P: wp.array2d(dtype=vec3), Pd: wp.array2d(dtype=vec3), hc: wp.array3d(dtype=vec3),
              hr: wp.array(dtype=dtype), rr: wp.array(dtype=dtype), goals: wp.array2d(dtype=vec3), zone: vec3,
              r_goal: dtype, r_zone: dtype, v_slow: dtype, d_min: dtype, d_slow: dtype, eps_l: dtype, eps_v: dtype,
              T: int, n_r: int, n_h: int, Z: wp.array3d(dtype=dtype)):
        # atom a: 0 pick, 1 handover, 2 zone, then n_r speeds, n_r n_h separations (d_min) and
        # n_r n_h slow-down distances (d_slow); P[:, 0] is the site, P[:, 1:] the robot spheres
        b, t, a = wp.tid()
        w = b * T + t
        if a < 2:
            e = P[w, 0] - goals[b, a]
            Z[b, t, a] = (r_goal - wp.sqrt(wp.dot(e, e) + eps_l * eps_l)) / r_goal
        elif a == 2:
            Z[b, t, a] = (n_minus(P[w, 0] - zone, eps_l) - r_zone) / r_zone
        elif a < 3 + n_r:
            v = Pd[w, a - 2]
            Z[b, t, a] = (v_slow - wp.sqrt(wp.dot(v, v) + eps_v * eps_v)) / v_slow
        else:
            k = a - 3 - n_r
            kind = k // (n_r * n_h)
            k = k - kind * n_r * n_h
            i = k // n_h
            j = k - i * n_h
            d = n_minus(P[w, 1 + i] - hc[b, t, j], eps_l)
            R = rr[i] + hr[j] + d_min
            if kind == 1:
                R = rr[i] + hr[j] + d_slow
            Z[b, t, a] = (d - R) / R

    return {"chain": chain, "points": points, "atoms": atoms, "vec3": vec3, "quat": quat}


def chain_arrays(mjm):
    """Model arrays of a serial chain (at most 16 bodies, at most one hinge per body, nq = nv) for
    the chain kernel, with quaternions reordered to Warp's (x, y, z, w)."""
    nb = mjm.nbody
    if nb > MAX_BODIES or mjm.nq != mjm.nv or np.any(mjm.jnt_type != mujoco.mjtJoint.mjJNT_HINGE):
        raise NotImplementedError("a chain of at most " + str(MAX_BODIES) + " bodies with hinge joints is expected")
    if np.any(mjm.body_parentid[1:] != np.maximum(np.arange(nb - 1), 0)) or np.any(mjm.body_jntnum > 1) \
            or np.any(mjm.body_mocapid >= 0):
        raise NotImplementedError("a serial chain without mocap bodies, at most one joint per body, is expected")
    wxyz = lambda q: np.concatenate([q[:, 1:], q[:, :1]], -1)  # noqa: E731
    return {"qpos0": mjm.qpos0, "body_pos": mjm.body_pos, "body_quat": wxyz(mjm.body_quat),
            "body_jnt": np.where(mjm.body_jntnum > 0, mjm.body_jntadr, -1), "jnt_pos": mjm.jnt_pos,
            "jnt_axis": mjm.jnt_axis, "jnt_qadr": mjm.jnt_qposadr, "jnt_dadr": mjm.jnt_dofadr, "nbody": nb}


class Predicates:
    """The predicate values Z (B, T, P) of nworld = B worlds with T samples each, for a
    tasks.workspace.Plant and Scenario; set_instance() gives the targets and the person's spheres."""

    def __init__(self, plant, sc, T, nworld=1, dtype=wp.float32, device="cuda:0"):
        self.k = _kernels(dtype)
        self.dtype, self.device = dtype, wp.get_device(device)
        self.B, self.T = int(nworld), int(T)
        mjm = plant.model
        self.nq, self.nbody = mjm.nq, mjm.nbody
        rs = Wk.robot_spheres(plant, sc.robot_spacing)
        site = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, plant.site)
        self.n_r = len(rs["body"])
        np_dt = np.float32 if dtype == wp.float32 else np.float64
        ca = chain_arrays(mjm)
        vec3, quat = self.k["vec3"], self.k["quat"]
        with wp.ScopedDevice(self.device):
            arr = lambda x, t: wp.array(np.asarray(x, np_dt), dtype=t)  # noqa: E731
            self.chain_in = [arr(ca["qpos0"], dtype), arr(ca["body_pos"], vec3), arr(ca["body_quat"], quat),
                             wp.array(ca["body_jnt"].astype(np.int32), dtype=int), arr(ca["jnt_pos"], vec3),
                             arr(ca["jnt_axis"], vec3), wp.array(ca["jnt_qadr"].astype(np.int32), dtype=int),
                             wp.array(ca["jnt_dadr"].astype(np.int32), dtype=int), int(ca["nbody"])]
            self.pt_body = wp.array(np.concatenate([[mjm.site_bodyid[site]], rs["body"]]).astype(np.int32), dtype=int)
            self.pt_off = arr(np.concatenate([mjm.site_pos[site][None], rs["centre"]]), vec3)
            self.rr = arr(rs["radius"], dtype)
        self.consts = [vec3(*[np_dt(v) for v in sc.zone])] + [dtype(v) for v in (
            sc.r_goal, sc.zone_radius, sc.v_slow, sc.d_min, sc.d_slow, sc.eps_length, sc.eps_speed)]
        self.np_dt = np_dt

    def set_instance(self, pick, handover, human_centres, human_radii):
        """pick, handover (B, 3); human_centres (B, T, S_h, 3); human_radii (S_h,)."""
        B, T = self.B, self.T
        hc = np.asarray(human_centres, self.np_dt).reshape(B, T, -1, 3)
        self.n_h = hc.shape[2]
        self.P = 3 + self.n_r + 2 * self.n_r * self.n_h
        goals = np.stack([np.asarray(pick, self.np_dt).reshape(B, 3), np.asarray(handover, self.np_dt).reshape(B, 3)], 1)
        with wp.ScopedDevice(self.device):
            self.hc = wp.array(hc, dtype=self.k["vec3"])
            self.hr = wp.array(np.asarray(human_radii, self.np_dt), dtype=self.dtype)
            self.goals = wp.array(goals, dtype=self.k["vec3"])

    def frames(self, q, v, tape=None):
        """Body positions, orientations and the velocity sums om and mm, each (B T, nbody), from q, v
        (B T, nq); the launch goes on the tape if one is given."""
        W, nb, g = q.shape[0], self.nbody, tape is not None
        with wp.ScopedDevice(self.device):
            out = [wp.zeros((W, nb), dtype=t, requires_grad=g) for t in (self.k["vec3"], self.k["quat"], self.k["vec3"], self.k["vec3"])]
            with tape if g else nullcontext():
                wp.launch(self.k["chain"], dim=W, inputs=[q, v] + self.chain_in, outputs=out)
        return out

    def scores(self, q, v, tape=None):
        """Atoms Z (B, T, P) from the sampled states q, v (B T, nq), world-major. With a tape (q and
        v with requires_grad) the three launches go on it and Z has a gradient."""
        g = tape is not None
        W = self.B * self.T
        if q.shape[0] != W:
            raise ValueError("q must have B T = " + str(W) + " rows")
        F = self.frames(q, v, tape)
        with wp.ScopedDevice(self.device):
            npt = self.pt_body.shape[0]
            P = wp.zeros((W, npt), dtype=self.k["vec3"], requires_grad=g)
            Pd = wp.zeros((W, npt), dtype=self.k["vec3"], requires_grad=g)
            Z = wp.zeros((self.B, self.T, self.P), dtype=self.dtype, requires_grad=g)
            with tape if g else nullcontext():
                wp.launch(self.k["points"], dim=(W, npt), inputs=F + [self.pt_body, self.pt_off], outputs=[P, Pd])
                wp.launch(self.k["atoms"], dim=(self.B, self.T, self.P),
                          inputs=[P, Pd, self.hc, self.hr, self.rr, self.goals] + self.consts + [self.T, self.n_r, self.n_h],
                          outputs=[Z])
        return Z

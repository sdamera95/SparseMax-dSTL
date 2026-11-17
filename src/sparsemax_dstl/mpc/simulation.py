"""Rigid-body simulation of both arms in MuJoCo under joint drives, and the checks of an executed motion: the exact
robustness of every until and reach instance, the tracking error and the arms' clearance.

The model is MuJoCo's import of the two arms' URDFs, both arms in one model with joint frames equal to those of
tasks/dual_arm.py. Each joint has a position drive KP (target - q) and a velocity drive -KV qd whose sum is clipped at
the joint to the torque limit, an armature, no damping and no friction; each hand is one rigid payload on link 7.
Gravity acts, contacts are off. One sample of 0.02 s is SUBSTEPS physics steps of DT_PHYSICS holding the sample's
joint-position targets.
"""
import hashlib

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import rollout

from .. import stl
from ..jax import evaluate, read
from ..tasks import dual_arm as D
from ..tasks import dual_arm_scenes as S

ARMS = ("right", "left")

# joint drives and inertial parameters of the simulation model
DT_PHYSICS = 0.001  # s
SUBSTEPS = 20  # physics steps per sample of D.H_STEP
KP = 2000.0  # N m / rad
KV = 20.0  # N m s / rad
ARMATURE = 0.1  # kg m^2
EFFORT = np.array([80.0, 80.0, 36.0, 36.0, 36.0, 36.0, 36.0] * 2)  # N m, right joints 1 to 7, then left
GRAVITY = (0.0, 0.0, -9.81)
# centres of mass of links 1 to 7 (m): COM_OFFSET + COM_BODY in the link frame
COM_OFFSET = np.array([
    [[0.0, -0.075, 0.0], [-0.0328, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0325, 0.0], [0.0, 0.0, 0.0], [0.0, -0.022, 0.0],
     [-0.0225, 0.0, 0.0]],
    [[0.0, 0.075, 0.0], [-0.0328, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0325, 0.0], [0.0, 0.0, 0.0], [0.0, 0.022, 0.0],
     [-0.0225, 0.0, 0.0]]])
COM_BODY = np.array([
    [[0.000169601, 0.005361189, -0.0], [0.000146672, -0.0, -0.082006417], [0.006792617, -0.000117253, -0.129977554],
     [-0.009006311, -0.000240346, -0.07432998], [-0.006281223, -8.606e-06, -0.055255286], [0.004593143, 0.00676485, -0.058702651],
     [1e-09, -0.018241338, -0.024628798]],
    [[0.000169601, -0.005361189, -0.0], [0.000146672, 0.0, -0.082006417], [0.006792617, -0.000117253, -0.129977554],
     [-0.009006311, -0.000240346, -0.07432998], [-0.006281223, -8.606e-06, -0.055255286], [0.004646649, -0.006749867, -0.058559567],
     [-1e-09, 0.018241338, -0.024628798]]])
COM = COM_OFFSET + COM_BODY
# each hand as one rigid payload in its mount frame: mass (kg), centre of mass (m), inertia about it (kg m^2)
HAND_MASS = np.array([0.4383491539999999, 0.438347741])
HAND_COM = np.array([[0.00497437785, 0.010111325281, 0.06997894447], [0.004897282402, -0.007341579352, 0.074881721336]])
HAND_INERTIA = np.array([
    [[1.335248838592e-03, -1.436570082654e-04, -1.083040809435e-06], [-1.436570082654e-04, 1.003116816779e-03, 8.244722150795e-05],
     [-1.083040809435e-06, 8.244722150795e-05, 5.862137416534e-04]],
    [[1.307051871756e-03, 1.366973227312e-04, -2.347391737305e-05], [1.366973227312e-04, 9.800072059303e-04, -6.409099710237e-05],
     [-2.347391737305e-05, -6.409099710237e-05, 5.747618364742e-04]]])

LINKS = tuple(tuple(s + n for n in ("_Shoulder_Pitch_Link", "_Shoulder_Roll_Link", "_Shoulder_Yaw_Link", "_Elbow_Pitch_Link",
                                    "_Wrist_Yaw_Link", "_Wrist_Pitch_Link", "_Wrist_Roll_Link")) for s in ("Right", "Left"))
JOINTS = tuple(tuple(n.replace("_Link", "_Joint") for n in arm) for arm in LINKS)
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def urdf_path(arm):
    """An arm's URDF, checked against its sha256."""
    p = D.urdf_path(arm)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != D.URDF_SHA256[arm]:
        raise ValueError(str(p) + " has sha256 " + h + ", expected " + D.URDF_SHA256[arm])
    return p


def spec():
    """The uncompiled MjSpec of the model."""
    s = mujoco.MjSpec()
    s.modelname = "dual_arm_simulation"
    o = s.option
    o.timestep = DT_PHYSICS
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.iterations = 100
    o.ls_iterations = 50
    o.tolerance = 1e-6
    o.gravity = GRAVITY
    o.disableflags = o.disableflags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    for a, side in enumerate(ARMS):  # the two arms
        child = mujoco.MjSpec.from_file(str(urdf_path(side)))
        s.worldbody.add_frame(pos=D.PLACEMENT[a]).attach_body(child.body(LINKS[a][0]), "", "")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, D.MOUNT_ROT.ravel())
    for a, side in enumerate(ARMS):  # the two arms
        for k, name in enumerate(LINKS[a]):  # links 1 to 7
            s.body(name).ipos = COM[a, k]
        I = HAND_INERTIA[a]
        s.body(LINKS[a][-1]).add_body(name=side + "_hand", pos=D.MOUNT_POS, quat=quat, mass=HAND_MASS[a], ipos=HAND_COM[a],
                                      fullinertia=[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], explicitinertial=True)
    names = [n for arm in JOINTS for n in arm]
    for i, n in enumerate(names):  # the 14 joints
        j = s.joint(n)
        j.armature = ARMATURE
        j.damping = [0.0, 0.0, 0.0]
        j.frictionloss = 0.0
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = [-EFFORT[i], EFFORT[i]]
    for n in names:  # position drives first: ctrl[:14] are the targets
        s.add_actuator(name=n + "_position", target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                       gainprm=[KP] + [0.0] * 9, biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0.0, -KP] + [0.0] * 8)
    for n in names:  # then velocity drives, ctrl[14:] = 0
        s.add_actuator(name=n + "_velocity", target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                       gainprm=[KV] + [0.0] * 9, biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0.0, 0.0, -KV] + [0.0] * 7)
    return s


def model():
    return spec().compile()


class Simulation:
    """Both arms started at the state x0 (28,), at rest targets x0's positions; run() advances it."""

    def __init__(self, x0):
        self.m = model()
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for arm in JOINTS for n in arm]
        self.qadr, self.vadr = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = x0[:D.NQ]
        d.qvel[self.vadr] = x0[D.NQ:]
        d.ctrl[:D.NQ] = x0[:D.NQ]
        mujoco.mj_forward(m, d)
        self.s = np.empty(mujoco.mj_stateSize(m, FULL))
        mujoco.mj_getState(m, d, self.s, FULL)
        self.iq, self.iv = 1 + self.qadr, 1 + m.nq + self.vadr

    def run(self, Q):
        """Hold each row of the joint-position targets Q (P, 14) over one sample; the states (P, 28) at the samples' ends."""
        Q = np.asarray(Q, np.float64)
        ctrl = np.zeros((len(Q) * SUBSTEPS, self.m.nu))
        ctrl[:, :D.NQ] = np.repeat(Q, SUBSTEPS, 0)
        st, _ = rollout.rollout(self.m, self.d, self.s[None], ctrl[None])
        self.s = np.array(st[0, -1])
        ends = st[0, SUBSTEPS - 1::SUBSTEPS]
        return np.concatenate([ends[:, self.iq], ends[:, self.iv]], 1)


# ------------------------------------------------------------------
# checks of an executed motion

def predicate_traces(scene, X, smoothing_length=0.0):
    """(L, P) the scene's predicates along the states X (L, 28), outside predicates with the given smoothing length of the
    norm (0: exact), waypoint predicates exact."""
    with jax.enable_x64(True):
        cells, wp = jnp.asarray(scene.cells), jnp.asarray(scene.waypoints)

        def one(x):
            return jnp.concatenate([D.at_waypoint(x, wp, scene.waypoint_arms, 0.0),
                                    D.outside(x, cells, scene.cell_radius, smoothing_length)])
        return np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(X, jnp.float64)))


def instances(scene, smoothing_length=0.0):
    """The exact robustness of every instance of the scene's conjuncts on scene.states: a list of dicts with the conjunct's
    name and kind, the anchor (first sample read), the last sample read and the value."""
    P = predicate_traces(scene, scene.states, smoothing_length)
    L = len(P)
    out = []
    with jax.enable_x64(True):
        for c in scene.conjuncts:  # conjuncts
            nodes = [P[:, n["preds"]].min(-1) if n["kind"] == "and" else P[:, n["preds"][0]] for n in c["nodes"]]
            prog = stl.compile_formula(c["formula"], L, reads=[(c["formula"], c["anchors"])])
            v = np.asarray(read(prog, evaluate(prog, jnp.asarray(np.stack(nodes, -1)), "exact")))
            out += [{"conjunct": c["name"], "kind": c["kind"], "anchor": int(s), "last": int(s) + c["window"] - 1, "exact": float(r)}
                    for s, r in zip(c["anchors"], v)]
    return out


def tracking(q_exec, q_target):
    """Largest and root-mean-square joint-position difference (rad) between executed positions and targets, both (n, 14),
    and the largest per joint."""
    e = np.asarray(q_exec) - np.asarray(q_target)
    return {"largest_rad": float(np.abs(e).max()), "rms_rad": float(np.sqrt(np.mean(e ** 2))), "largest_per_joint_rad": np.abs(e).max(0).tolist()}


def clearance(X):
    """(n,) the smallest surface distance (m) between the two arms' spheres of D.ARM_PAIRS at each state of X (n, 28)."""
    C = S.hand_and_spheres(np.asarray(X))[1]
    i, j = D.ARM_PAIRS[:, 0], D.ARM_PAIRS[:, 1]
    return (np.linalg.norm(C[:, i] - C[:, j], axis=-1) - D.RADIUS[i] - D.RADIUS[j]).min(1)

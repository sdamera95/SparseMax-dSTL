"""MuJoCo models of the examples: a sliding block and the Franka Panda of MuJoCo Menagerie, read
from third_party/ (scripts/fetch_menagerie.sh fetches it) or from the path MUJOCO_MENAGERIE."""

import os
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE_COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
MENAGERIE = Path(os.environ.get("MUJOCO_MENAGERIE", Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie"))


def panda(contacts=True):
    """The Panda without its hand, driven by joint torques within Menagerie's force ranges.
    contacts=False clears contype and conaffinity on every geom."""
    return panda_spec(contacts).compile()


def panda_spec(contacts=True):
    """The uncompiled MjSpec that panda() compiles, for callers that add to the model."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_emika_panda" / "panda_nohand.xml"))
    for a in spec.actuators:
        lo, hi = a.forcerange
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm = np.eye(1, 10).ravel()
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.biasprm = np.zeros(10)
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [lo, hi]
    if not contacts:
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0
    return spec


SLIDING_BLOCK = """
<mujoco model="sliding block">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" friction="0.8 0.005 0.0001"/>
    <body name="block" pos="0 0 0.05">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <geom name="block" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="x" ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


def sliding_block():
    """A 1 kg box resting on a frictional plane, pushed along x by a force motor."""
    return mujoco.MjModel.from_xml_string(SLIDING_BLOCK)

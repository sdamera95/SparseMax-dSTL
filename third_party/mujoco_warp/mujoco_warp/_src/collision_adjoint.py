# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modified from the adjoint fork at commit 357a75d: box-box contact adjoints added.
"""VJPs for collision geometry and contact position dependence."""

from typing import Tuple

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src import math
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec8i
from mujoco_warp._src.warp_util import event_scope

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
wp.set_module_options({"enable_backward": True})


# Store the primal capsule-box witness needed by the differentiable replay.
@wp.kernel(enable_backward=False)
def _capsule_box_freeze(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  tseg_out: wp.array[wp.vec2],
  feat_out: wp.array[wp.vec3i],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  g0 = geoms[0]
  g1 = geoms[1]
  if geom_type[g0] != GeomType.CAPSULE or geom_type[g1] != GeomType.BOX:
    return
  w = contact_worldid_in[cid]
  gw = w % geom_size.shape[0]
  m0 = geom_xmat_in[w, g0]
  axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
  ts, ft = collision_primitive_core.capsule_box_witness(
    geom_xpos_in[w, g0], axis0, geom_size[gw, g0][1], geom_xpos_in[w, g1], geom_xmat_in[w, g1], geom_size[gw, g1]
  )
  tseg_out[cid] = ts
  feat_out[cid] = ft


# forward-only re-run of the box-box narrowphase's DISCRETE decisions, per contact. Mirrors
# collision_primitive_core.box_box (keep in sync); not AD-safe, so it is only ever launched with
# backward disabled. What it returns is the witness the backward needs, packed as one vec8i:
#   [0] axis_code  which separating axis won the search: -1 no contact at all, 0-11 a face of one
#                  of the two boxes (the face-vertex case), 12-20 the cross product of one edge of
#                  each box (the edge-edge case)
#   [1] face-vertex: clcorner, which corner of the incident box lies deepest under the reference
#                  face. edge-edge: cle1, which corner of box 1 the contacting edge runs from
#   [2] face-vertex: a1, the first incident axis lying along the reference face (-1 if unused).
#                  edge-edge: cle2, box 2's corner code
#   [3] face-vertex: a2, the second such axis. edge-edge: inv, 1 when the winning axis points
#                  from box 2 towards box 1, else 0
#   [4] [5] edge-edge: ax1, ax2 -- box 2's two axes that do not run along its contacting edge, in
#                  the order the forward's own size comparison left them. -1 on a face-vertex axis
#   [6] edge-edge: pax2, the box-1 axis whose face the forward projects everything onto. -1 on a
#                  face-vertex axis
#   [7] how many candidate points the emission produced (before the depth filter on a face-vertex
#                  axis). Carried only so a probe can watch the 8-slot emission buffer; the
#                  backward ignores it
#   codes          per emitted contact slot, WHICH point source produced it and with which index.
#                  Face-vertex: 64 + 4*edge + 2*clip_axis + clip_sign for an incident edge clipped
#                  against the reference face's boundary, 128 + corner for a reference-face corner
#                  inside the incident face, 192 + corner for an incident-face corner inside the
#                  reference face. Edge-edge: 256 + 4*edge + 2*clip_axis + clip_sign for an edge
#                  of box 2's contacting face clipped against box 1's face, 320 + corner for a
#                  box-1 face corner inside the projected face, 384 + corner for a box-2 face
#                  corner standing over box 1's face. -1 for a slot no point reached.
# On one such witness every emitted contact's distance, position and normal is a single closed
# form in the two poses (_box_box_point), which is what makes the backward differentiable.
@wp.func
def _box_box_witness(
  # In:
  box1_pos: wp.vec3,
  box1_rot: wp.mat33,
  box1_size: wp.vec3,
  box2_pos: wp.vec3,
  box2_rot: wp.mat33,
  box2_size: wp.vec3,
  margin: float,
) -> Tuple[vec8i, vec8i]:
  codes = vec8i()
  for i in range(8):
    codes[i] = -1
  none_state = vec8i(-1, 0, -1, -1, -1, -1, -1, 0)

  pos21 = wp.transpose(box1_rot) @ (box2_pos - box1_pos)
  pos12 = wp.transpose(box2_rot) @ (box1_pos - box2_pos)
  rot21 = wp.transpose(box1_rot) @ box2_rot
  rot12 = wp.transpose(rot21)
  rot21abs = wp.matrix_from_rows(wp.abs(rot21[0]), wp.abs(rot21[1]), wp.abs(rot21[2]))
  rot12abs = wp.transpose(rot21abs)
  plen2 = rot21abs @ box2_size
  plen1 = rot12abs @ box1_size

  s_sum_3 = 3.0 * (box1_size + box2_size)
  separation = wp.float32(margin + s_sum_3[0] + s_sum_3[1] + s_sum_3[2])
  axis_code = wp.int32(-1)

  # face normals of both boxes
  for i in range(3):
    c1 = -wp.abs(pos21[i]) + box1_size[i] + plen2[i]
    c2 = -wp.abs(pos12[i]) + box2_size[i] + plen1[i]
    if c1 < -margin or c2 < -margin:
      return none_state, codes
    if c1 < separation:
      separation = c1
      axis_code = i + 3 * wp.int32(pos21[i] < 0) + 0
    if c2 < separation:
      separation = c2
      axis_code = i + 3 * wp.int32(pos12[i] < 0) + 6

  # cross products of the boxes' edge directions
  clnorm = wp.vec3(0.0)
  cle1 = wp.int32(0)
  cle2 = wp.int32(0)
  inv = wp.int32(0)
  for i in range(3):
    for j in range(3):
      if i == 0:
        cross_axis = wp.vec3(0.0, -rot12[j, 2], rot12[j, 1])
      elif i == 1:
        cross_axis = wp.vec3(rot12[j, 2], 0.0, -rot12[j, 0])
      else:
        cross_axis = wp.vec3(-rot12[j, 1], rot12[j, 0], 0.0)

      cross_length = wp.length(cross_axis)
      if cross_length < MJ_MINVAL:
        continue
      cross_axis /= cross_length
      box_dist = wp.dot(pos21, cross_axis)
      c3 = wp.float32(0.0)
      for k in range(3):
        if k != i:
          c3 += box1_size[k] * wp.abs(cross_axis[k])
        if k != j:
          c3 += box2_size[k] * rot21abs[i, 3 - k - j] / cross_length
      c3 -= wp.abs(box_dist)
      if c3 < -margin:
        return none_state, codes
      if c3 < separation * (1.0 - 1e-12):
        separation = c3
        cle1 = 0
        cle2 = 0
        for k in range(3):
          if k != i and (int(cross_axis[k] > 0) ^ int(box_dist < 0)):
            cle1 += 1 << k
          if k != j:
            if int(rot21[i, 3 - k - j] > 0) ^ int(box_dist < 0) ^ int((k - j + 3) % 3 == 1):
              cle2 += 1 << k
        axis_code = 12 + i * 3 + j
        clnorm = cross_axis
        inv = wp.int32(box_dist < 0)

  if axis_code == -1:
    return none_state, codes

  if axis_code >= 12:
    # ------------------------------------------------------------------
    # edge-edge: box 2's contacting face is projected along the winning axis onto box 1's face,
    # and the emission clips the projected quadrilateral against that face's rectangle
    edge1 = (axis_code - 12) // 3
    edge2 = (axis_code - 12) % 3
    ax1 = wp.int32(1 - (edge2 & 1))
    ax2 = wp.int32(2 - (edge2 & 2))
    pax1 = wp.int32(1 - (edge1 & 1))
    pax2 = wp.int32(2 - (edge1 & 2))
    if rot21abs[edge1, ax1] < rot21abs[edge1, ax2]:
      swap = ax1
      ax1 = ax2
      ax2 = swap
    if rot12abs[edge2, pax1] < rot12abs[edge2, pax2]:
      swap = pax1
      pax1 = pax2
      pax2 = swap

    rotmore = collision_primitive_core._compute_rotmore(wp.where(cle1 & (1 << pax2), pax2, pax2 + 3))
    pe = rotmore @ pos21
    rnorm = rotmore @ clnorm
    re = rotmore @ rot21
    rte = wp.transpose(re)
    se = wp.abs(wp.transpose(rotmore) @ box1_size)
    lx = se[0]
    ly = se[1]
    pe[2] -= se[2]

    quad = collision_primitive_core.mat43f()
    edir = rte[edge2] * box2_size[edge2]
    quad[0] = (
      pe
      + rte[ax1] * box2_size[ax1] * wp.where(cle2 & (1 << ax1), 1.0, -1.0)
      + rte[ax2] * box2_size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )
    quad[1] = quad[0] - edir
    quad[0] += edir
    quad[2] = (
      pe
      + rte[ax1] * box2_size[ax1] * wp.where(cle2 & (1 << ax1), -1.0, 1.0)
      + rte[ax2] * box2_size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )
    quad[3] = quad[2] - edir
    quad[2] += edir

    axi_lp = quad[0]
    axi_cn1 = quad[1] - quad[0]
    axi_cn2 = quad[2] - quad[0]

    if wp.abs(rnorm[2]) < MJ_MINVAL:
      return none_state, codes
    innorm = wp.where(inv, -1.0, 1.0) / rnorm[2]

    pu = collision_primitive_core.mat43f()
    for i in range(4):
      pu[i] = quad[i]
      c_scl = quad[i, 2] * wp.where(inv, -1.0, 1.0) * innorm
      quad[i] -= rnorm * c_scl

    pts_lp = quad[0]
    pts_cn1 = quad[1] - quad[0]
    pts_cn2 = quad[2] - quad[0]

    n = wp.int32(0)
    for i in range(4):
      for q in range(2):
        la = pts_lp[q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[q], pts_cn2[q]))
        lb = wp.where(i == 0 or i == 3, pts_cn1[q], pts_cn2[q])
        lc = pts_lp[1 - q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[1 - q], pts_cn2[1 - q]))
        ld = wp.where(i == 0 or i == 3, pts_cn1[1 - q], pts_cn2[1 - q])
        lua = axi_lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, axi_cn1, axi_cn2))
        lub = wp.where(i == 0 or i == 3, axi_cn1, axi_cn2)
        if wp.abs(lb) > MJ_MINVAL:
          br = 1.0 / lb
          for j in range(-1, 2, 2):
            if n == 8:
              break
            l = se[q] * wp.float32(j)
            c1 = (l - la) * br
            if c1 < 0 or c1 > 1:
              continue
            c2 = lc + ld * c1
            if wp.abs(c2) > se[1 - q]:
              continue
            if (lua[2] + lub[2] * c1) * innorm > margin:
              continue
            codes[n] = 256 + 4 * i + 2 * q + (j + 1) // 2
            n += 1
    nl = n

    ax = pts_cn1[0]
    bx = pts_cn2[0]
    ay = pts_cn1[1]
    by = pts_cn2[1]
    C = math.safe_div(1.0, ax * by - bx * ay)
    for i in range(4):
      if n == 8:
        break
      llx = wp.where(i // 2, lx, -lx)
      lly = wp.where(i % 2, ly, -ly)
      x = llx - pts_lp[0]
      y = lly - pts_lp[1]
      u = (x * by - y * bx) * C
      v = (y * ax - x * ay) * C
      if nl == 0:
        if (u < 0 or u > 1) and (v < 0 or v > 1):
          continue
      elif u < 0 or v < 0 or u > 1 or v > 1:
        continue
      u = wp.clamp(u, 0.0, 1.0)
      v = wp.clamp(v, 0.0, 1.0)
      w = 1.0 - u - v
      vtmp = pu[0] * w + pu[1] * u + pu[2] * v
      corner = wp.vec3(llx, lly, 0.0)
      tc1 = wp.length_sq(corner - vtmp)
      if vtmp[2] > 0 and tc1 > margin * margin:
        continue
      codes[n] = 320 + i
      n += 1
    nf = n

    for i in range(4):
      if n >= 8:
        break
      x = pu[i, 0]
      y = pu[i, 1]
      if nl == 0 and nf != 0:
        if (x < -lx or x > lx) and (y < -ly or y > ly):
          continue
      elif x < -lx or x > lx or y < -ly or y > ly:
        continue
      c1 = wp.float32(0)
      for j in range(2):
        if pu[i, j] < -se[j]:
          c1 += (pu[i, j] + se[j]) * (pu[i, j] + se[j])
        elif pu[i, j] > se[j]:
          c1 += (pu[i, j] - se[j]) * (pu[i, j] - se[j])
      c1 += pu[i, 2] * innorm * pu[i, 2] * innorm
      if pu[i, 2] > 0 and c1 > margin * margin:
        continue
      codes[n] = 384 + i
      n += 1

    return vec8i(axis_code, cle1, cle2, inv, ax1, ax2, pax2, n), codes

  # ------------------------------------------------------------------
  # face-vertex: the reference face's frame, the deepest incident corner, the two incident axes
  # that lie along the face, and the emitted points in the forward's own order
  face_idx = axis_code % 6
  box_idx = axis_code // 6
  rotmore = collision_primitive_core._compute_rotmore(face_idx)

  r = rotmore @ wp.where(box_idx, rot12, rot21)
  p = rotmore @ wp.where(box_idx, pos12, pos21)
  ss = wp.abs(rotmore @ wp.where(box_idx, box2_size, box1_size))
  s = wp.where(box_idx, box1_size, box2_size)
  rt = wp.transpose(r)

  lx = ss[0]
  ly = ss[1]
  hz = ss[2]
  p[2] -= hz

  clcorner = wp.int32(0)
  for i in range(3):
    if r[2, i] < 0:
      clcorner += 1 << i

  lp = p
  for i in range(wp.static(3)):
    lp += rt[i] * s[i] * wp.where(clcorner & 1 << i, 1.0, -1.0)

  dirs = wp.int32(0)
  a1 = wp.int32(-1)
  a2 = wp.int32(-1)
  cn1 = wp.vec3(0.0)
  cn2 = wp.vec3(0.0)
  for i in range(3):
    if wp.abs(r[2, i]) < 0.5:
      if not dirs:
        cn1 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
        a1 = i
      else:
        cn2 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
        a2 = i
      dirs += 1

  points = collision_primitive_core.mat83f()
  cand = vec8i()
  for i in range(8):
    cand[i] = -1

  k = dirs * dirs
  n = wp.int32(0)

  for i in range(k):
    for q in range(2):
      lav = lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, cn1, cn2))
      lbv = wp.where(i == 0 or i == 3, cn1, cn2)
      if wp.abs(lbv[q]) > MJ_MINVAL:
        br = 1.0 / lbv[q]
        for j in range(-1, 2, 2):
          l = ss[q] * wp.float32(j)
          c1 = (l - lav[q]) * br
          if c1 < 0 or c1 > 1:
            continue
          c2 = lav[1 - q] + lbv[1 - q] * c1
          if wp.abs(c2) > ss[1 - q]:
            continue
          if n < 8:
            points[n] = lav + c1 * lbv
            cand[n] = 64 + 4 * i + 2 * q + (j + 1) // 2
          n += 1

  if dirs == 2:
    ax = cn1[0]
    bx = cn2[0]
    ay = cn1[1]
    by = cn2[1]
    C = math.safe_div(1.0, ax * by - bx * ay)
    for i in range(4):
      llx = wp.where(i // 2, lx, -lx)
      lly = wp.where(i % 2, ly, -ly)
      x = llx - lp[0]
      y = lly - lp[1]
      u = (x * by - y * bx) * C
      v = (y * ax - x * ay) * C
      if u > 0 and v > 0 and u < 1 and v < 1:
        if n < 8:
          points[n] = wp.vec3(llx, lly, lp[2] + u * cn1[2] + v * cn2[2])
          cand[n] = 128 + i
        n += 1

  for i in range(1 << dirs):
    tmpv = lp + wp.float32(i & 1) * cn1 + wp.float32((i & 2) != 0) * cn2
    if tmpv[0] > -lx and tmpv[0] < lx and tmpv[1] > -ly and tmpv[1] < ly:
      if n < 8:
        points[n] = tmpv
        cand[n] = 192 + i
      n += 1

  # the forward's depth filter decides which candidates become contact slots, in order. The
  # compaction and the halving it does beside it only touch indices this loop has already passed,
  # so leaving them out cannot change which candidates survive.
  m = n
  n = wp.int32(0)
  for i in range(8):
    if i >= m:
      break
    if points[i][2] > margin:
      continue
    if n < 8:
      codes[n] = cand[i]
    n += 1

  return vec8i(axis_code, clcorner, a1, a2, -1, -1, -1, m), codes


# differentiably re-derive ONE emitted box-box contact with the witness frozen. Every runtime
# index is a compare-select over constant indices, so the adjoint never indexes a local vector at
# runtime (the same rule the plane-box and plane-cylinder slot selections follow).
@wp.func
def _box_box_point(
  # In:
  box1_pos: wp.vec3,
  box1_rot: wp.mat33,
  box1_size: wp.vec3,
  box2_pos: wp.vec3,
  box2_rot: wp.mat33,
  box2_size: wp.vec3,
  wit: vec8i,  # frozen witness (from _box_box_witness)
  code: int,  # this slot's point source
) -> Tuple[float, wp.vec3, wp.vec3]:
  pos21 = wp.transpose(box1_rot) @ (box2_pos - box1_pos)
  pos12 = wp.transpose(box2_rot) @ (box1_pos - box2_pos)
  rot21 = wp.transpose(box1_rot) @ box2_rot
  rot12 = wp.transpose(rot21)

  axis_code = wit[0]
  src = code // 64
  idx = code - 64 * src
  dist = float(0.0)
  pos = wp.vec3(0.0)
  normal = wp.vec3(0.0)
  zero3 = wp.vec3(0.0)

  if axis_code >= 12:
    # ------------------------------------------------------------------
    # edge-edge
    edge1 = (axis_code - 12) // 3
    edge2 = (axis_code - 12) % 3
    cle1 = wit[1]
    cle2 = wit[2]
    inv = wit[3]
    ax1 = wit[4]
    ax2 = wit[5]
    pax2 = wit[6]

    # the winning cross axis, re-derived from the frozen edge pair: this is the forward's clnorm,
    # and it is smooth in the two poses once the edge pair is fixed
    r12row = wp.where(edge2 == 0, rot12[0], wp.where(edge2 == 1, rot12[1], rot12[2]))
    cross_axis = wp.where(
      edge1 == 0,
      wp.vec3(0.0, -r12row[2], r12row[1]),
      wp.where(
        edge1 == 1,
        wp.vec3(r12row[2], 0.0, -r12row[0]),
        wp.vec3(-r12row[1], r12row[0], 0.0),
      ),
    )
    cross_axis /= wp.length(cross_axis)

    rotmore = collision_primitive_core._compute_rotmore(wp.where(cle1 & (1 << pax2), pax2, pax2 + 3))
    pe = rotmore @ pos21
    rnorm = rotmore @ cross_axis
    re = rotmore @ rot21
    rte = wp.transpose(re)
    se = wp.abs(wp.transpose(rotmore) @ box1_size)
    lx = se[0]
    ly = se[1]
    hz = se[2]
    pf = pe - wp.vec3(0.0, 0.0, hz)

    rt_a1 = wp.where(ax1 == 0, rte[0], wp.where(ax1 == 1, rte[1], rte[2]))
    rt_a2 = wp.where(ax2 == 0, rte[0], wp.where(ax2 == 1, rte[1], rte[2]))
    rt_e2 = wp.where(edge2 == 0, rte[0], wp.where(edge2 == 1, rte[1], rte[2]))
    sz_a1 = wp.where(ax1 == 0, box2_size[0], wp.where(ax1 == 1, box2_size[1], box2_size[2]))
    sz_a2 = wp.where(ax2 == 0, box2_size[0], wp.where(ax2 == 1, box2_size[1], box2_size[2]))
    sz_e2 = wp.where(edge2 == 0, box2_size[0], wp.where(edge2 == 1, box2_size[1], box2_size[2]))

    edir = rt_e2 * sz_e2
    q0 = (
      pf
      + rt_a1 * sz_a1 * wp.where(cle2 & (1 << ax1), 1.0, -1.0)
      + rt_a2 * sz_a2 * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )
    q2 = (
      pf
      + rt_a1 * sz_a1 * wp.where(cle2 & (1 << ax1), -1.0, 1.0)
      + rt_a2 * sz_a2 * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )
    pu0 = q0 + edir
    pu1 = q0 - edir
    pu2 = q2 + edir
    pu3 = q2 - edir

    axi_lp = pu0
    axi_cn1 = pu1 - pu0
    axi_cn2 = pu2 - pu0

    sgn = wp.where(inv != 0, -1.0, 1.0)
    innorm = sgn / rnorm[2]
    pp0 = pu0 - rnorm * (pu0[2] * sgn * innorm)
    pp1 = pu1 - rnorm * (pu1[2] * sgn * innorm)
    pp2 = pu2 - rnorm * (pu2[2] * sgn * innorm)
    pts_lp = pp0
    pts_cn1 = pp1 - pp0
    pts_cn2 = pp2 - pp0

    pt = wp.vec3(0.0)
    if src == 4:  # an edge of box 2's face clipped against box 1's face rectangle
      i = idx // 4
      q = (idx - 4 * i) // 2
      jhi = idx - 4 * i - 2 * q
      lpq = wp.where(q == 0, pts_lp[0], pts_lp[1])
      c1q = wp.where(q == 0, pts_cn1[0], pts_cn1[1])
      c2q = wp.where(q == 0, pts_cn2[0], pts_cn2[1])
      lpo = wp.where(q == 0, pts_lp[1], pts_lp[0])
      c1o = wp.where(q == 0, pts_cn1[1], pts_cn1[0])
      c2o = wp.where(q == 0, pts_cn2[1], pts_cn2[0])
      la = lpq + wp.where(i < 2, 0.0, wp.where(i == 2, c1q, c2q))
      lb = wp.where(i == 0 or i == 3, c1q, c2q)
      lc = lpo + wp.where(i < 2, 0.0, wp.where(i == 2, c1o, c2o))
      ld = wp.where(i == 0 or i == 3, c1o, c2o)
      lua = axi_lp + wp.where(i < 2, zero3, wp.where(i == 2, axi_cn1, axi_cn2))
      lub = wp.where(i == 0 or i == 3, axi_cn1, axi_cn2)
      seq = wp.where(q == 0, se[0], se[1])
      br = 1.0 / lb
      l = seq * wp.where(jhi == 1, 1.0, -1.0)
      c1 = (l - la) * br
      c2 = lc + ld * c1
      base = lua * 0.5 + c1 * lub * 0.5
      ptx = base[0] + wp.where(q == 0, 0.5 * l, 0.5 * c2)
      pty = base[1] + wp.where(q == 0, 0.5 * c2, 0.5 * l)
      pt = wp.vec3(ptx, pty, base[2])
      dist = pt[2] * innorm * 2.0
    elif src == 5:  # a corner of box 1's face inside the projected face of box 2
      ax = pts_cn1[0]
      bx = pts_cn2[0]
      ay = pts_cn1[1]
      by = pts_cn2[1]
      C = math.safe_div(1.0, ax * by - bx * ay)
      llx = wp.where(idx // 2, lx, -lx)
      lly = wp.where(idx % 2, ly, -ly)
      x = llx - pts_lp[0]
      y = lly - pts_lp[1]
      u = wp.clamp((x * by - y * bx) * C, 0.0, 1.0)
      v = wp.clamp((y * ax - x * ay) * C, 0.0, 1.0)
      w = 1.0 - u - v
      vtmp = pu0 * w + pu1 * u + pu2 * v
      corner = wp.vec3(llx, lly, 0.0)
      tc1 = wp.length_sq(corner - vtmp)
      pt = 0.5 * (corner + vtmp)
      dist = wp.sqrt(tc1) * wp.where(vtmp[2] < 0.0, -1.0, 1.0)
    else:  # src == 6: a corner of box 2's face standing over box 1's face
      pui = wp.where(idx == 0, pu0, wp.where(idx == 1, pu1, wp.where(idx == 2, pu2, pu3)))
      c1 = wp.float32(0)
      if pui[0] < -se[0]:
        c1 += (pui[0] + se[0]) * (pui[0] + se[0])
      elif pui[0] > se[0]:
        c1 += (pui[0] - se[0]) * (pui[0] - se[0])
      if pui[1] < -se[1]:
        c1 += (pui[1] + se[1]) * (pui[1] + se[1])
      elif pui[1] > se[1]:
        c1 += (pui[1] - se[1]) * (pui[1] - se[1])
      c1 += pui[2] * innorm * pui[2] * innorm
      tx = pui[0]
      if pui[0] < -se[0]:
        tx = -se[0] * 0.5
      elif pui[0] > se[0]:
        tx = +se[0] * 0.5
      ty = pui[1]
      if pui[1] < -se[1]:
        ty = -se[1] * 0.5
      elif pui[1] > se[1]:
        ty = +se[1] * 0.5
      pt = (wp.vec3(tx, ty, 0.0) + pui) * 0.5
      dist = wp.sqrt(c1) * wp.where(pui[2] < 0.0, -1.0, 1.0)

    rw = box1_rot @ wp.transpose(rotmore)
    normal = (wp.where(inv != 0, -1.0, 1.0) * rw) @ rnorm
    pos = rw @ (pt + wp.vec3(0.0, 0.0, hz)) + box1_pos
    return dist, pos, normal

  # ------------------------------------------------------------------
  # face-vertex
  clcorner = wit[1]
  a1 = wit[2]
  a2 = wit[3]
  face_idx = axis_code % 6
  box_idx = axis_code // 6
  rotmore = collision_primitive_core._compute_rotmore(face_idx)

  r = rotmore @ wp.where(box_idx, rot12, rot21)
  p = rotmore @ wp.where(box_idx, pos12, pos21)
  ss = wp.abs(rotmore @ wp.where(box_idx, box2_size, box1_size))
  s = wp.where(box_idx, box1_size, box2_size)
  rt = wp.transpose(r)

  lx = ss[0]
  ly = ss[1]
  hz = ss[2]
  pf = p - wp.vec3(0.0, 0.0, hz)

  # the deepest incident corner, and the two edges leaving it along the face
  w0 = rt[0] * s[0]
  w1 = rt[1] * s[1]
  w2 = rt[2] * s[2]
  lp = (
    pf
    + w0 * wp.where((clcorner & 1) != 0, 1.0, -1.0)
    + w1 * wp.where((clcorner & 2) != 0, 1.0, -1.0)
    + w2 * wp.where((clcorner & 4) != 0, 1.0, -1.0)
  )
  e0 = w0 * wp.where((clcorner & 1) != 0, -2.0, 2.0)
  e1 = w1 * wp.where((clcorner & 2) != 0, -2.0, 2.0)
  e2 = w2 * wp.where((clcorner & 4) != 0, -2.0, 2.0)
  cn1 = wp.where(a1 == 0, e0, wp.where(a1 == 1, e1, wp.where(a1 == 2, e2, zero3)))
  cn2 = wp.where(a2 == 0, e0, wp.where(a2 == 1, e1, wp.where(a2 == 2, e2, zero3)))

  pt = wp.vec3(0.0)
  if src == 1:  # an incident edge clipped against the reference face's boundary
    i = idx // 4
    q = (idx - 4 * i) // 2
    jhi = idx - 4 * i - 2 * q
    lav = lp + wp.where(i < 2, zero3, wp.where(i == 2, cn1, cn2))
    lbv = wp.where(i == 0 or i == 3, cn1, cn2)
    ssq = wp.where(q == 0, ss[0], ss[1])
    lavq = wp.where(q == 0, lav[0], lav[1])
    lbvq = wp.where(q == 0, lbv[0], lbv[1])
    l = ssq * wp.where(jhi == 1, 1.0, -1.0)
    br = 1.0 / lbvq
    c1 = (l - lavq) * br
    pt = lav + c1 * lbv
  elif src == 2:  # a reference-face corner inside the incident face
    ax = cn1[0]
    bx = cn2[0]
    ay = cn1[1]
    by = cn2[1]
    C = math.safe_div(1.0, ax * by - bx * ay)
    llx = wp.where(idx // 2, lx, -lx)
    lly = wp.where(idx % 2, ly, -ly)
    x = llx - lp[0]
    y = lly - lp[1]
    u = (x * by - y * bx) * C
    v = (y * ax - x * ay) * C
    pt = wp.vec3(llx, lly, lp[2] + u * cn1[2] + v * cn2[2])
  else:  # src == 3: an incident-face corner inside the reference face
    pt = lp + wp.float32(idx & 1) * cn1 + wp.float32((idx & 2) != 0) * cn2

  dist = pt[2]
  local = wp.vec3(pt[0], pt[1], pt[2] * 0.5 + hz)  # the contact point is the midpoint
  rw = wp.where(box_idx, box2_rot, box1_rot) @ wp.transpose(rotmore)
  pw = wp.where(box_idx, box2_pos, box1_pos)
  normal = wp.where(box_idx, -1.0, 1.0) * wp.transpose(rw)[2]
  pos = rw @ local + pw
  return dist, pos, normal


# store per box-box contact the frozen witness and this slot's point source (_box_box_witness)
@wp.kernel(enable_backward=False)
def _box_box_freeze(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_geomcollisionid_in: wp.array[int],
  contact_includemargin_in: wp.array[float],
  nacon_in: wp.array[int],
  # Out:
  feat_out: wp.array[vec8i],
  code_out: wp.array[int],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  g0 = geoms[0]
  g1 = geoms[1]
  if geom_type[g0] != GeomType.BOX or geom_type[g1] != GeomType.BOX:
    return
  slot = contact_geomcollisionid_in[cid]
  if slot < 0 or slot >= 8:
    return
  w = contact_worldid_in[cid]
  gw = w % geom_size.shape[0]
  ft, codes = _box_box_witness(
    geom_xpos_in[w, g0],
    geom_xmat_in[w, g0],
    geom_size[gw, g0],
    geom_xpos_in[w, g1],
    geom_xmat_in[w, g1],
    geom_size[gw, g1],
    contact_includemargin_in[cid],
  )
  feat_out[cid] = ft
  code_out[cid] = codes[slot]


# ------------------------------------------------------------------
# the witness stamp
#
# A contact whose geom pair has no geometry adjoint here contributes ZERO to dqpos through its
# geometry, and the forward gives no sign of it -- that is how the missing box-box branch went
# unnoticed. Every call of contact_qpos_vjp now writes one integer per contact slot saying what
# happened, and keeps the array so a caller can read it without touching the step itself:
#   -1                    the slot is not a live contact
#    0                    a geometry adjoint ran for this contact
#    1                    box-box: the branch is there but this slot carried no witness, so the
#                         contact was skipped (the mirror and the forward routine disagreed)
#    100 + 16*t0 + t1     no branch at all for this geom-type pair (t0, t1 are the two GeomTypes)
_COVER_OK = 0
_COVER_NO_WITNESS = 1
_COVER_NO_BRANCH = 100
_LAST_COVER = None


def last_uncovered_contacts():
  """(n_live, n_uncovered, {reason: count}) for the most recent contact_qpos_vjp call.

  `reason` is the integer above; a value at or over 100 decodes as (value - 100) // 16 and
  (value - 100) % 16 = the two GeomTypes of the pair that has no branch. None before the first
  call. Reading this costs one device-to-host copy and is meant for probes, not for the step.
  """
  if _LAST_COVER is None:
    return None
  import numpy as _np

  c = _np.asarray(_LAST_COVER.numpy())
  live = c >= 0
  bad = c > 0
  vals, cnts = _np.unique(c[bad], return_counts=True)
  return int(live.sum()), int(bad.sum()), {int(v): int(n) for v, n in zip(vals, cnts)}


# Frozen-witness narrowphase replay using collision_primitive_core's differentiable leaves.
# Twelve primitive pairs are dispatched:
# plane-{sphere,capsule,ellipsoid,cylinder,box}, sphere-{sphere,box,capsule,cylinder},
# capsule-capsule, capsule-box, and box-box on either kind of separating axis (face-vertex and
# edge-edge). What still falls through: plane-convex/mesh; dqpos stays silently 0 for that pair,
# and for any pair with no branch here at all. cover_out is the witness stamp that makes such a
# contact visible instead of silent -- see _COVER_* above. Capsule-box freezes the minimizing
# feature; capsule-capsule supports the non-parallel slot-0 regime.
@wp.kernel(enable_backward=True)
def _narrowphase_recompute(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_geomcollisionid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  capsule_box_tseg: wp.array[wp.vec2],  # frozen capsule-box segment params (_capsule_box_freeze)
  capsule_box_feat: wp.array[wp.vec3i],  # frozen capsule-box discrete feature state
  box_box_feat: wp.array[vec8i],  # frozen box-box witness (_box_box_freeze)
  box_box_code: wp.array[int],  # frozen box-box point source for this contact slot
  # Out:
  cpos_out: wp.array[wp.vec3],
  dist_out: wp.array[float],
  frame_out: wp.array[wp.mat33],
  cover_out: wp.array[int],  # per-contact witness stamp: see _COVER_OK / _COVER_NO_WITNESS
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  w = contact_worldid_in[cid]
  slot = contact_geomcollisionid_in[cid]
  gw = w % geom_size.shape[0]
  g0 = geoms[0]
  g1 = geoms[1]
  p0 = geom_xpos_in[w, g0]
  m0 = geom_xmat_in[w, g0]
  s0 = geom_size[gw, g0]
  p1 = geom_xpos_in[w, g1]
  m1 = geom_xmat_in[w, g1]
  s1 = geom_size[gw, g1]
  type0 = geom_type[g0]
  type1 = geom_type[g1]
  cover_out[cid] = 0  # overwritten below wherever no branch runs
  if type0 == GeomType.PLANE and type1 == GeomType.SPHERE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos = collision_primitive_core.plane_sphere(n0, p0, p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n0)
  elif type0 == GeomType.SPHERE and type1 == GeomType.SPHERE:
    dist, pos, n = collision_primitive_core.sphere_sphere(p0, s0[0], p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.SPHERE and type1 == GeomType.BOX:
    dist, pos, n = collision_primitive_core.sphere_box(p0, s0[0], p1, m1, s1)
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.PLANE and type1 == GeomType.CAPSULE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dvec, pmat, frame = collision_primitive_core.plane_capsule(n0, p0, p1, axis, s1[0], s1[1])  # two caps, shared frame
    if slot == 0:
      dist_out[cid] = dvec[0]
      cpos_out[cid] = wp.vec3(pmat[0, 0], pmat[0, 1], pmat[0, 2])
    else:
      dist_out[cid] = dvec[1]
      cpos_out[cid] = wp.vec3(pmat[1, 0], pmat[1, 1], pmat[1, 2])
    frame_out[cid] = frame
  elif type0 == GeomType.SPHERE and type1 == GeomType.CAPSULE:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dist, pos, n = collision_primitive_core.sphere_capsule(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.SPHERE and type1 == GeomType.CYLINDER:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # cylinder local z-axis
    dist, pos, n = collision_primitive_core.sphere_cylinder(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.PLANE and type1 == GeomType.ELLIPSOID:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos, n = collision_primitive_core.plane_ellipsoid(n0, p0, p1, m1, s1)  # returns normal = plane normal
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.CAPSULE and type1 == GeomType.CAPSULE:
    # unique local names per multi-contact branch: Warp codegen scopes locals to the whole
    # function, so reusing one name with a different vec/mat type across branches is a
    # type-conflict error.
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis1 = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    # margin only gates write_contact's slot assignment in the forward; pass a large value so the
    # (frozen) active slot is always populated. non-parallel (crossed) axes -> slot 0; the
    # parallel slot-1 assignment is margin-dependent (see the coverage comment above).
    cc_dist, cc_pos, cc_nrm = collision_primitive_core.capsule_capsule(p0, axis0, s0[0], s0[1], p1, axis1, s1[0], s1[1], 1.0e6)
    for i in range(2):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cc_dist[i]
        cpos_out[cid] = wp.vec3(cc_pos[i, 0], cc_pos[i, 1], cc_pos[i, 2])
        frame_out[cid] = math.make_frame(wp.vec3(cc_nrm[i, 0], cc_nrm[i, 1], cc_nrm[i, 2]))  # per-slot normal
  elif type0 == GeomType.PLANE and type1 == GeomType.CYLINDER:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    cyl_dist, cyl_pos, cyl_n = collision_primitive_core.plane_cylinder(
      n0, p0, p1, axis, s1[0], s1[1]
    )  # 4 contacts, shared normal
    for i in range(4):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cyl_dist[i]
        cpos_out[cid] = wp.vec3(cyl_pos[i, 0], cyl_pos[i, 1], cyl_pos[i, 2])
    frame_out[cid] = math.make_frame(cyl_n)
  elif type0 == GeomType.PLANE and type1 == GeomType.BOX:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    box_dist, box_pos, box_n = collision_primitive_core.plane_box(
      n0, p0, p1, m1, s1
    )  # 8 corners (slot = corner id), shared normal
    for i in range(8):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = box_dist[i]
        cpos_out[cid] = wp.vec3(box_pos[i, 0], box_pos[i, 1], box_pos[i, 2])
    frame_out[cid] = math.make_frame(box_n)
  elif type0 == GeomType.CAPSULE and type1 == GeomType.BOX:
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    cb_dist, cb_pos, cb_n = collision_primitive_core.capsule_box_from_witness(
      p0, axis0, s0[0], s0[1], p1, m1, s1, slot, capsule_box_tseg[cid], capsule_box_feat[cid]
    )
    dist_out[cid] = cb_dist
    cpos_out[cid] = cb_pos
    frame_out[cid] = math.make_frame(cb_n)
  elif type0 == GeomType.BOX and type1 == GeomType.BOX:
    # frozen-witness narrowphase: the separating-axis search and the point-emission tests are not
    # AD-safe, so _box_box_freeze froze which axis won, which corners and axes the emission used,
    # and which emitted point this slot is. On that witness the contact's distance, position and
    # normal are one closed form in the two poses, which _box_box_point re-derives. Both kinds of
    # axis are covered: a face of one box against a corner of the other, and a cross product of
    # one edge of each. A slot the witness did not name is stamped instead of silently skipped.
    bb_feat = box_box_feat[cid]
    bb_code = box_box_code[cid]
    if bb_feat[0] >= 0 and bb_code >= 0:
      bb_dist, bb_pos, bb_n = _box_box_point(p0, m0, s0, p1, m1, s1, bb_feat, bb_code)
      dist_out[cid] = bb_dist
      cpos_out[cid] = bb_pos
      frame_out[cid] = math.make_frame(bb_n)
    else:
      cover_out[cid] = 1
  else:
    # no geometry adjoint for this geom pair: dqpos stays 0 through it, and the stamp says so
    cover_out[cid] = 100 + 16 * type0 + type1


# gather dr/defc_pos (efc-row indexed) to the per-contact normal-row distance adjoint
@wp.kernel(enable_backward=False)
def _gather_efc_to_contact(
  # Data in:
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  res_efc_pos: wp.array2d[float],  # dr/defc_pos * lam (per world, per efc row)
  # Out:
  adj_dist_out: wp.array[float],  # per-contact seed for dist_out.grad (defc_pos/ddist = 1)
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0:
    return
  adj_dist_out[cid] = res_efc_pos[contact_worldid_in[cid], e0]


# chain narrowphase geom-pose adjoints to the per-dof tangent gradient via support.jac_dof
@wp.kernel(enable_backward=False)
def _geom_pose_dof_vjp(
  # Model:
  ngeom: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  res_geom_xpos: wp.array2d[wp.vec3],  # adj(geom_xpos) from the narrowphase backward
  res_geom_xmat: wp.array2d[wp.mat33],  # adj(geom_xmat)
  # Out:
  res_dof_out: wp.array2d[float],  # per-dof tangent gradient d(contact)/d(dof k)
):
  w, k = wp.tid()
  acc = float(0.0)
  for g in range(ngeom):
    rgp = res_geom_xpos[w, g]
    rgm = res_geom_xmat[w, g]
    body = geom_bodyid[g]
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      geom_xpos_in[w, g],
      body,
      k,
      w,
    )
    tau = adjoint_util._adj_rotation(geom_xmat_in[w, g], rgm)
    acc += wp.dot(jacp, rgp) + wp.dot(jacr, tau)
  res_dof_out[w, k] += acc


@event_scope
def contact_qpos_vjp(
  m: Model,
  d_out: Data,
  qpos_in: wp.array2d[float],
  res_contact_pos: wp.array,
  res_contact_frame: wp.array,
  res_efc_pos: wp.array2d[float],
  res_subtree_com: wp.array2d[wp.vec3],
  res_cdof: wp.array2d[wp.spatial_vector],
  res_qpos: wp.array2d[float],
  bc: BackwardContext | None = None,
):
  """Accumulates the contact residual's dqpos into res_qpos from the exposed input-adjoints.

  A frozen-witness narrowphase replay yields adj(geom poses); analytic Jacobian VJPs then chain
  everything into one per-dof tangent buffer that _dof_to_qpos lifts.

  Args:
    m: The model.
    d_out: The step-output data holding the frozen contacts and kinematics.
    qpos_in: Input/linearization qpos (d.qpos), not the integrated d_out.qpos.
    res_contact_pos: dr/dcontact_pos * lam, per-contact vec3.
    res_contact_frame: dr/dcontact_frame * lam, per-contact mat33.
    res_efc_pos: dr/defc_pos * lam, (nworld, njmax).
    res_subtree_com: dr/dsubtree_com * lam, (nworld, nbody).
    res_cdof: dr/dcdof * lam, (nworld, nv).
    res_qpos: Accumulated output; the caller's writeback applies the IFT minus.
    bc: Reusable backward workspace.
  """
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d_out)
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  nconmax = d_out.contact.pos.shape[0]
  contact_segment, contact_feature = bc.contact_segment, bc.contact_feature
  contact_adj_dist = bc.contact_adj_dist
  res_geom_xpos, res_geom_xmat = bc.res_geom_xpos, bc.res_geom_xmat
  contact_res_dof, contact_ceff = bc.contact_res_dof, bc.contact_ceff
  contact_segment.zero_()
  contact_feature.zero_()
  contact_adj_dist.zero_()
  res_geom_xpos.zero_()
  res_geom_xmat.zero_()
  contact_res_dof.zero_()
  contact_ceff.zero_()

  wp.launch(
    _capsule_box_freeze,
    dim=nconmax,
    inputs=[m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom, d_out.contact.worldid, d_out.nacon],
    outputs=[contact_segment, contact_feature],
  )
  # frozen box-box witness (forward-only; see _box_box_freeze). -1 means "no witness", so the
  # fill matters: a zeroed entry would read as a valid axis and point source.
  bb_feat = wp.full(nconmax, vec8i(-1, 0, -1, -1, -1, -1, -1, 0), dtype=vec8i)
  bb_code = wp.full(nconmax, -1, dtype=int)
  wp.launch(
    _box_box_freeze,
    dim=nconmax,
    inputs=[
      m.geom_type,
      m.geom_size,
      d_out.geom_xpos,
      d_out.geom_xmat,
      d_out.contact.geom,
      d_out.contact.worldid,
      d_out.contact.geomcollisionid,
      d_out.contact.includemargin,
      d_out.nacon,
    ],
    outputs=[bb_feat, bb_code],
  )
  pairs = [
    (m.geom_type, None),
    (m.geom_size, None),
    (d_out.geom_xpos, res_geom_xpos),
    (d_out.geom_xmat, res_geom_xmat),
    (d_out.contact.geom, None),
    (d_out.contact.worldid, None),
    (d_out.contact.geomcollisionid, None),
    (d_out.nacon, None),
    (contact_segment, None),
    (contact_feature, None),
    (bb_feat, None),
    (bb_code, None),
  ]
  # the witness stamp: -1 on a slot that is not a live contact, 0 where a geometry adjoint ran,
  # and a reason code otherwise (read it with last_uncovered_contacts()).
  cover_o = wp.full(nconmax, -1, dtype=int)
  global _LAST_COVER
  _LAST_COVER = cover_o
  # a reverse launch writes no outputs, so the stamp takes one forward launch of the replay; its
  # geometry goes to the backward workspace's scratch contacts, never to the step's own d_out
  wp.launch(
    _narrowphase_recompute,
    dim=nconmax,
    inputs=[value for value, _ in pairs],
    outputs=[bc.scratch.contact.pos, bc.scratch.contact.dist, bc.scratch.contact.frame, cover_o],
  )
  wp.launch(
    _gather_efc_to_contact,
    dim=nconmax,
    inputs=[d_out.contact.efc_address, d_out.contact.worldid, d_out.nacon, res_efc_pos],
    outputs=[contact_adj_dist],
  )
  adjoint_util._launch_vjp(
    _narrowphase_recompute,
    nconmax,
    pairs,
    [d_out.contact.pos, d_out.contact.dist, d_out.contact.frame, cover_o],
    [res_contact_pos, contact_adj_dist, res_contact_frame, None],
  )
  wp.launch(
    _geom_pose_dof_vjp,
    dim=(nworld, nv),
    inputs=[
      m.ngeom,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.geom_bodyid,
      m.body_isdofancestor,
      d_out.geom_xpos,
      d_out.geom_xmat,
      d_out.subtree_com,
      d_out.cdof,
      res_geom_xpos,
      res_geom_xmat,
    ],
    outputs=[contact_res_dof],
  )
  smooth_kinematics_adjoint.kinematics_qpos_backward(
    m,
    d_out,
    qpos_in,
    res_cdof,
    res_subtree_com,
    contact_res_dof,
    contact_ceff,
    res_qpos,
  )

"""XgoDuck VelStand: MicroDuck's walk + protective-fall layer, retargeted.

Composition, in this order and for a reason:

1. ``make_microduck_velocity_env_cfg`` — the upstream walk recipe, verbatim.
2. ``retarget_xgoduck_walk``          — the XgoDuck walk layer. The SAME call
   the walk-only task makes, so the VelStand student and its warm-start teacher
   cannot drift apart.
3. ``add_velstand_layer``             — recovery economics, protective-fall
   costs, body control, prone curriculum, BC teachers.

Everything robot-specific is in ``XGODUCK_VELSTAND_PROFILE``; the layer itself
is shared. Numbers below are XgoDuck's, derived as noted — MicroDuck's values
live in ``microduck_velstand_env_cfg`` and are NOT reusable here (its stand
height is 0.115 m, this robot's is 0.139158 m).

Teachers (see ``XgoduckVelStandRlCfg``): the shipped stand-up checkpoint teaches
fallen frames, the shipped walk checkpoint anchors upright frames. Both are 61D
in / 14D out with the same 512-256-128 actor, so they load into the student
actor as-is.
"""

from __future__ import annotations

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_STAND_Z,
    XGODUCK_VELSTAND_ROBOT_CFG,
    make_xgoduck_contact_sensors,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    VelStandRobotProfile,
    add_velstand_layer,
)
from mjlab_microduck.tasks.xgoduck_walk_retarget import retarget_xgoduck_walk

# Sit keyframe, by servo index (same joint order as MicroDuck: left leg 0-4,
# head 5-8, right leg 9-13). Values are XGODUCK_SIT_Z's own definition —
# "knee ±1.35, hip_pitch ±0.4079, ankle/hip_roll 0" — which happens to be the
# same pose as MicroDuck's sit, on the same indices.
XGODUCK_SIT_JOINT_OVERRIDES = {
    1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0,
    10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0,
}

XGODUCK_VELSTAND_PROFILE = VelStandRobotProfile(
    name="XgoDuck",
    robot_cfg=XGODUCK_VELSTAND_ROBOT_CFG,
    make_contact_sensors=make_xgoduck_contact_sensors,
    # Standing trunk height (the retargeted HOME), not MicroDuck's 0.116.
    nominal_z=XGODUCK_STAND_Z,
    height_ceiling=XGODUCK_STAND_Z,
    com_max_height=XGODUCK_STAND_Z + 0.010,   # standup's retarget uses the same +1 cm
    # Fractions of the standing height, carried over from MicroDuck
    # (0.09/0.115 = 0.783, 0.08/0.115 = 0.696, 0.09 and 0.11 of 0.115):
    # a "genuinely up" trunk, the fallen gate, and the head-droop gates.
    recovered_up_z=0.109,
    term_gate_z=0.097,
    head_gate_low=0.109,
    head_gate_high=0.133,
    # Measured on THIS plant, not scaled: scripts/measure_prone_z.py --full-contact
    # settles the robot from a 90° drop in each orientation and reports the trunk
    # rest height — face-down 0.0571, face-up 0.0468, on-side 0.0484 m. MicroDuck's
    # band is rest+6 mm .. rest+46 mm, applied to the HIGHEST rest so no
    # orientation spawns with the head through the floor.
    prone_z_min=0.063,
    prone_z_max=0.103,
    sit_joint_overrides=XGODUCK_SIT_JOINT_OVERRIDES,
    # None, not MicroDuck's table: ANCHOR_YAW_REMAP is the inverse response of
    # MicroDuck's walk anchor (441tzs6d@3750), calibrated per (|v|, |wz|) cell.
    # Fed to a different robot it would over- or under-drive the turn. This
    # walk anchor already turns at 75% of the commanded rate in place (measured
    # by verify_velocity.py), i.e. it does not have the dead zone the table was
    # built to cancel.
    yaw_remap=None,
)


def make_xgoduck_velstand_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    cfg = retarget_xgoduck_walk(cfg, play=play, rough=rough)
    return add_velstand_layer(cfg, play=play, profile=XGODUCK_VELSTAND_PROFILE)

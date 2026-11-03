"""The XgoDuck walk layer: MicroDuck's velocity recipe, retargeted.

Extracted from ``tasks/__init__.py`` so that the walk-only tasks and the
VelStand task (which starts from the same walk recipe and then adds the
protective-fall layer on top) share ONE definition. Applying this to a cfg
twice, or letting the two copies drift, would silently train the VelStand
student on a different walk reward than its warm-start teacher.

Only XgoDuck-specific overrides belong here; everything upstream (reward
terms, curricula, DR) stays in ``microduck_velocity_env_cfg``.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.xgoduck_constants import XGODUCK_WALK_ROBOT_CFG

from . import mdp as microduck_mdp


def retarget_xgoduck_walk(cfg, *, play: bool = False, rough: bool = False):
    """Apply the XgoDuck walk overrides in place and return ``cfg``."""
    # MicroDuck's legacy head list includes bearing_roll, a right hip on XgoDuck.
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"].body_names = (
            "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
        )
    cfg.scene.entities = {"robot": deepcopy(XGODUCK_WALK_ROBOT_CFG)}
    # reset_root_state_uniform ADDS these offsets to InitialStateCfg.pos.
    # HOME already leaves 1 mm below the soles; do not add the height again
    # or sample negative offsets that push the feet into the floor.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
    if rough:
        # terrain_levels assumes increasing difficulty along rows.
        cfg.scene.terrain.terrain_generator.curriculum = not play
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.5)
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
        }
    actor = cfg.observations["actor"].terms
    actor["base_ang_vel"].noise.n_min = -0.06
    actor["base_ang_vel"].noise.n_max = 0.06
    actor["projected_gravity"].noise.n_min = -0.03
    actor["projected_gravity"].noise.n_max = 0.03
    actor["joint_pos"].noise.n_min = -0.01
    actor["joint_pos"].noise.n_max = 0.01
    cfg.rewards["pose"].weight = 1.3
    cfg.rewards["track_linear_velocity"].weight = 3.0
    # 0.25 pays exp(-(0.12/0.25)^2) = 79% at the 0.12 m/s tracking error the
    # 3000-iteration run plateaued on, so the gradient ran out while speed
    # regulation was still soft (measured by verify_velocity.py: 22% undershoot
    # at 0.34 m/s, 0.06 m/s of drift when told to stand). At 0.15 the same
    # 0.12 m/s error is worth 53% and keeps pushing.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.15
    # Yaw: leave the stock 0.707 ALONE and add a fine term beside it. Tightening
    # the stock std is the obvious move and it backfires — a Gaussian has no
    # gradient past its own std, so a 0.4 one is nearly flat at this policy's
    # ~0.5 rad/s error. Measured here 2026-11-03 (1500 iters from model_2999):
    # turn-in-place fell from 54% to 28% of the commanded rate while linear
    # tracking improved. That is the same failure the velstand yaw dead-zone note
    # records for a std-0.35 replacement (0.42 -> 0.16 rad/s). Keep the loose
    # term's far-error gradient and let track_yaw_fine price the small errors.
    cfg.rewards["track_yaw_fine"] = RewardTermCfg(
        func=microduck_mdp.track_yaw_rate_fine,
        weight=2.0,
        params={
            "std": 0.35,
            "command_name": cfg.rewards["track_angular_velocity"].params["command_name"],
        },
    )
    cfg.rewards["foot_swing_height"].weight = -5.0
    # Keep the hips near HOME at low commanded speed, while allowing turns.
    cfg.rewards["hip_yaw_home"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_l1_when_standing,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("left_hip_yaw", "right_hip_yaw"),
            ),
            "command_name": "twist",
            "linear_threshold": 0.03,
            "angular_threshold": 1e-6,
        },
    )
    # Sample straight reverse commands in 15% of moving training scenarios.
    if not play:
        cfg.commands["twist"].rel_backward_envs = 0.15
        cfg.commands["twist"].backward_speed_range = (-0.2, -0.1)
    # Hip yaw inward stop is 0.22 rad.
    _std_s = dict(cfg.rewards["pose"].params["std_standing"])
    _std_w = dict(cfg.rewards["pose"].params["std_walking"])
    _std_r = dict(cfg.rewards["pose"].params["std_running"])
    _std_s[r".*hip_yaw.*"] = 0.05
    _std_w[r".*hip_yaw.*"] = 0.15
    _std_r[r".*hip_yaw.*"] = 0.25
    cfg.rewards["pose"].params["std_standing"] = _std_s
    cfg.rewards["pose"].params["std_walking"] = _std_w
    cfg.rewards["pose"].params["std_running"] = _std_r
    if play:
        # Match the requested Walk playback posture; the training curriculum
        # must not re-enable random head commands after entering play.
        cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
        cfg.commands["head_pose"].zero_command_prob = 1.0
        cfg.curriculum.pop("head_pose_range", None)
    return cfg

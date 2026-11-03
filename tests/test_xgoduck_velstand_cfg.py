"""XgoDuck VelStand: the shared recovery layer, driven by XgoDuck numbers.

Two things are worth testing here and nothing else is:

1. The layer is SHARED, so the XgoDuck walk layer inside VelStand must equal the
   walk task's, term for term — the student is warm-started from that teacher and
   BC-anchored to it, so a drifted reward is a silently broken run.
2. The robot-specific numbers must actually be XgoDuck's. MicroDuck's 0.116 m
   standing height applied to a 0.139 m robot puts every recovery gate 20% low,
   which is the failure ``docs/XGODUCK.md`` already records for STAND_Z.
"""

from pathlib import Path

import pytest

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_STAND_Z,
    XGODUCK_VELSTAND_ROBOT_CFG,
)
from mjlab_microduck.tasks import XgoduckVelStandRlCfg, _xgoduck_velocity_cfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    MICRODUCK_VELSTAND_PROFILE,
    make_microduck_velstand_env_cfg,
)
from mjlab_microduck.tasks.xgoduck_velstand_env_cfg import (
    XGODUCK_VELSTAND_PROFILE,
    make_xgoduck_velstand_env_cfg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_task_is_registered():
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-VelStand-Flat-XgoDuck")
    assert "fallen_too_long" in cfg.terminations
    assert "recovery_success" in cfg.rewards
    assert "body_pose_tracking" in cfg.rewards


def test_walk_layer_is_identical_to_the_walk_task():
    """VelStand starts from the walk; its walk terms must not drift from it."""
    walk = _xgoduck_velocity_cfg()
    velstand = make_xgoduck_velstand_env_cfg()

    for name in ("track_linear_velocity", "track_angular_velocity", "track_yaw_fine"):
        assert velstand.rewards[name].weight == pytest.approx(walk.rewards[name].weight)
        assert velstand.rewards[name].params == walk.rewards[name].params

    assert velstand.rewards["foot_swing_height"].weight == pytest.approx(
        walk.rewards["foot_swing_height"].weight
    )
    assert velstand.rewards["hip_yaw_home"].weight == pytest.approx(
        walk.rewards["hip_yaw_home"].weight
    )
    assert velstand.commands["twist"].rel_backward_envs == walk.commands["twist"].rel_backward_envs
    for group, terms in walk.observations.items():
        for term, term_cfg in terms.terms.items():
            got = velstand.observations[group].terms[term].noise
            want = term_cfg.noise
            if want is None:
                assert got is None
                continue
            assert (got.n_min, got.n_max) == (want.n_min, want.n_max)


def test_robot_is_the_full_contact_spec_on_the_walk_plant():
    robot = make_xgoduck_velstand_env_cfg().scene.entities["robot"]
    assert robot.collisions == XGODUCK_VELSTAND_ROBOT_CFG.collisions
    # Walk plant, not the standup one: 3-6 lags of command delay, which is what
    # the warm-started walk was trained with (standup runs 1 lag).
    actuator = robot.articulation.actuators[0]
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (3, 6)


def test_heights_are_xgoduck_not_microduck():
    cfg = make_xgoduck_velstand_env_cfg()

    assert cfg.rewards["body_pose_tracking"].params["nominal_height"] == pytest.approx(
        XGODUCK_STAND_Z
    )
    assert cfg.rewards["height_progress"].params["ceiling"] == pytest.approx(XGODUCK_STAND_Z)
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == pytest.approx(
        XGODUCK_STAND_Z + 0.010
    )
    # Gates must stay ordered and strictly below the standing height.
    term = cfg.terminations["fallen_too_long"].params["gate_z_below"]
    up = cfg.rewards["recovery_success"].params["up_z"]
    assert 0.0 < term < up < XGODUCK_STAND_Z
    assert cfg.rewards["fallen_tax"].params["release_z_above"] == pytest.approx(up)

    head = cfg.rewards["head_pose_bias"].params
    assert head["gate_height_low"] == pytest.approx(up)
    assert head["gate_height_high"] < XGODUCK_STAND_Z


def test_prone_band_sits_above_the_measured_rest_height():
    """scripts/measure_prone_z.py --full-contact: rest 0.0468-0.0571 m."""
    p = make_xgoduck_velstand_env_cfg().events["random_prone_init"].params
    assert p["prone_z_min"] > 0.0571, "spawn would interpenetrate the face-down rest pose"
    assert p["prone_z_max"] < XGODUCK_STAND_Z, "prone spawn must be below standing"


def test_contact_sensors_are_the_partitioned_trio():
    names = [s.name for s in make_xgoduck_velstand_env_cfg().scene.sensors]
    for name in ("servo_ground_contact", "head_ground_contact", "trunk_ground_contact"):
        assert name in names
    # MicroDuck's trio must not be reused: its servo pattern matches `xl330`
    # housing meshes that the XgoDuck export does not have.
    assert MICRODUCK_VELSTAND_PROFILE.make_contact_sensors() != (
        XGODUCK_VELSTAND_PROFILE.make_contact_sensors()
    )


def test_microduck_profile_is_untouched():
    cfg = make_microduck_velstand_env_cfg()
    assert cfg.rewards["body_pose_tracking"].params["nominal_height"] == pytest.approx(0.116)
    assert MICRODUCK_VELSTAND_PROFILE.prone_z_min == 0.05
    assert MICRODUCK_VELSTAND_PROFILE.yaw_remap is not None


def test_bc_teachers_are_the_shipped_checkpoints():
    bc = XgoduckVelStandRlCfg.algorithm.bc_cfg
    assert bc is not None
    for key in ("checkpoint_path", "anchor_checkpoint_path"):
        path = Path(bc[key])
        assert path.is_absolute()
        assert path.exists(), f"{key} does not resolve: {path}"
    assert Path(bc["checkpoint_path"]).parts[-2:] == ("xgoduck_standup", "model_2999.pt")
    assert Path(bc["anchor_checkpoint_path"]).parts[-2:] == (
        "xgoduck_velocity",
        "model_5497.pt",
    )
    # No wandb/Hub fallback: the runs must not reach for MicroDuck's experts.
    assert bc["wandb_run_path"] is None and bc["hf_checkpoint"] is None
    assert bc["anchor_wandb_run_path"] is None and bc["anchor_hf_checkpoint"] is None
    # Calibrated on MicroDuck's anchor, so it must not be applied here.
    assert bc["anchor_yaw_remap"] is None


def test_yaw_remap_only_changes_the_turn_in_place_bucket():
    """Disabling the remap must not change the reward stack, only the commands."""
    cfg = make_xgoduck_velstand_env_cfg()
    walk = _xgoduck_velocity_cfg()
    assert cfg.commands["twist"].rel_turn_in_place_envs == walk.commands["twist"].rel_turn_in_place_envs
    assert cfg.commands["twist"].turn_in_place_min_frac == walk.commands["twist"].turn_in_place_min_frac

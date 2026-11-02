"""XGODUCK collisions: every link vs ground; only head_roll vs base self-collides."""

import re

import mujoco

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_ALLCOLLISIONS_XML,
    XGODUCK_COLLISION,
    XGODUCK_HEAD_GEOM_PATTERN,
    XGODUCK_SERVO_GEOM_PATTERN,
    XGODUCK_SERVO_SHELL_GEOMS,
    XGODUCK_TRUNK_GEOM_PATTERN,
    XGODUCK_WALK_XML,
    make_xgoduck_contact_sensors,
)

# Standup/allcollisions: every link has a named *_collision mesh for floor
# contact. Bit 0 = world/floor, bit 1 = head-base self-collision.
STANDUP_WORLD_COLLISION_GEOMS = {
    "trunk_collision": "trunk_base",
    "left_hip_yaw_collision": "yaw2roll",
    "left_hip_roll_collision": "hip_l",
    "left_thigh_collision": "upper_leg_left",
    "left_shin_collision": "leg",
    "left_foot_collision": "ankle_left",
    "neck_collision": "neck",
    "head_pitch_collision": "neck_pitch",
    "head_yaw_collision": "yaw_roll_motion",
    "head_collision": "jaw_soft",
    "right_hip_yaw_collision": "bearing_roll",
    "right_hip_roll_collision": "hip_l_2",
    "right_thigh_collision": "upper_leg_right",
    "right_shin_collision": "leg_2",
    "right_foot_collision": "ankle_right",
}
SELF_COLLISION_PAIR = frozenset({"trunk_collision", "head_collision"})
SELF_COLLISION_GEOMS = frozenset(SELF_COLLISION_PAIR)


def _compile_with_collision(xml_path, collision_cfg):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    collision_cfg.edit_spec(spec)
    # Match Entity.spec_fn: unnamed geoms must not leak world contacts.
    for geom in spec.geoms:
        if not geom.name:
            geom.contype = 0
            geom.conaffinity = 0
    return spec.compile()


def _pair_collides(model, i, j) -> bool:
    return bool(
        (int(model.geom_contype[i]) & int(model.geom_conaffinity[j]))
        or (int(model.geom_contype[j]) & int(model.geom_conaffinity[i]))
    )


def test_xgoduck_standup_all_links_hit_ground_only_head_and_base_self_collide():
    model = _compile_with_collision(XGODUCK_ALLCOLLISIONS_XML, XGODUCK_COLLISION)

    active = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        ct = int(model.geom_contype[i])
        ca = int(model.geom_conaffinity[i])
        if ct or ca:
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
            assert name, f"unnamed active geom on body {body!r}"
            assert name in STANDUP_WORLD_COLLISION_GEOMS, (
                f"unexpected active geom {name!r} on body {body!r}"
            )
            active[name] = (i, body, ct, ca)

    missing = set(STANDUP_WORLD_COLLISION_GEOMS) - set(active)
    assert not missing, f"missing world-collision geoms: {sorted(missing)}"

    for name, expected_body in STANDUP_WORLD_COLLISION_GEOMS.items():
        _, body, ct, ca = active[name]
        assert body == expected_body, f"{name} on {body!r}, expected {expected_body!r}"
        if name in SELF_COLLISION_GEOMS:
            # Floor (bit 0) + head↔base (bit 1).
            assert ct == 3 and ca == 2, f"{name} bits ct={ct} ca={ca}"
        else:
            # Floor only; must not participate in robot-robot contacts.
            assert ct == 1 and ca == 0, f"{name} bits ct={ct} ca={ca}"

    saw_self = False
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j) or ""
            collides = _pair_collides(model, i, j)
            if frozenset({n1, n2}) == SELF_COLLISION_PAIR:
                assert collides, "head_roll and base must self-collide"
                saw_self = True
            else:
                assert not collides, f"unexpected self-collision {n1!r} vs {n2!r}"
    assert saw_self


def test_xgoduck_walk_has_no_thigh_world_collision():
    model = _compile_with_collision(XGODUCK_WALK_XML, XGODUCK_COLLISION)
    for name in ("left_thigh_collision", "right_thigh_collision", "trunk_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid < 0


def test_xgoduck_standup_task_enables_head_base_self_collision():
    cfg = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    col = cfg.scene.entities["robot"].collisions[0]
    assert col.contype["^(trunk_collision|head_collision)$"] == 3
    assert col.conaffinity["^(trunk_collision|head_collision)$"] == 2


# ── Servo shells (the handle the protective-fall costs need) ─────────────────
# XgoDuck has no separate servo housing mesh the way MicroDuck has `xl330`: each
# link's single collision hull encloses its servo. The three protection costs
# therefore have to partition the 15 collision geoms by name, and the partition
# is the thing worth testing — a shared geom means one contact billed twice, a
# missing one means an unprotected servo.


def _geom_body(model, geom_name: str) -> str:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"no geom named {geom_name!r}"
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])


def _body_hinge_joints(model, body_name: str) -> list[str]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert bid >= 0, f"no body named {body_name!r}"
    out = []
    for k in range(int(model.body_jntnum[bid])):
        jid = int(model.body_jntadr[bid]) + k
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            out.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return out


def test_servo_head_trunk_sets_partition_the_collision_geoms():
    """Every collision geom belongs to exactly one protection cost."""
    servo = set(XGODUCK_SERVO_SHELL_GEOMS)
    head = {n for n in STANDUP_WORLD_COLLISION_GEOMS if re.fullmatch(XGODUCK_HEAD_GEOM_PATTERN, n)}
    trunk = {n for n in STANDUP_WORLD_COLLISION_GEOMS if re.fullmatch(XGODUCK_TRUNK_GEOM_PATTERN, n)}
    feet = {n for n in STANDUP_WORLD_COLLISION_GEOMS if n.endswith("_foot_collision")}

    assert trunk == {"trunk_collision"}
    assert feet == {"left_foot_collision", "right_foot_collision"}
    assert head == {
        "neck_collision", "head_pitch_collision", "head_yaw_collision", "head_collision",
    }, f"head pattern picked up {sorted(head)}"
    assert len(servo) == 8

    # Disjoint, and together they account for all 15 hulls with nothing left over.
    groups = {"servo": servo, "head": head, "trunk": trunk, "feet": feet}
    for a in groups:
        for b in groups:
            if a < b:
                assert not (groups[a] & groups[b]), f"{a} and {b} overlap: {groups[a] & groups[b]}"
    assert set().union(*groups.values()) == set(STANDUP_WORLD_COLLISION_GEOMS)


def test_servo_shell_geoms_are_actuated_links_and_never_the_soles():
    """The set must be exactly the leg servos whose hulls a fall drives down."""
    model = _compile_with_collision(XGODUCK_ALLCOLLISIONS_XML, XGODUCK_COLLISION)

    for name in XGODUCK_SERVO_SHELL_GEOMS:
        body = _geom_body(model, name)
        joints = _body_hinge_joints(model, body)
        assert len(joints) == 1, f"{name} sits on {body!r}, which has joints {joints}"
        assert not name.endswith("_foot_collision"), "the soles must not be billed as servo impacts"

    # 14 actuated joints = 8 servo shells + 2 ankles (soles) + 4 head/neck.
    servo_bodies = {_geom_body(model, n) for n in XGODUCK_SERVO_SHELL_GEOMS}
    actuated = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(model.nbody)
        if _body_hinge_joints(model, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
    }
    assert len(actuated) == 14, f"expected 14 actuated bodies, got {sorted(actuated)}"
    assert servo_bodies <= actuated
    assert len(servo_bodies) == 8

    excluded = actuated - servo_bodies
    assert excluded == {
        "ankle_left", "ankle_right",              # feet
        "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",   # head, see head_impact
    }, f"unexpected exclusion set {sorted(excluded)}"


def test_contact_sensor_patterns_match_only_their_own_geoms():
    """A pattern that over-matches silently double-bills a contact."""
    all_geoms = list(STANDUP_WORLD_COLLISION_GEOMS)
    sensors = {s.name: s for s in make_xgoduck_contact_sensors()}
    assert set(sensors) == {"servo_ground_contact", "head_ground_contact", "trunk_ground_contact"}

    matched = {}
    for sensor_name, pattern in (
        ("servo_ground_contact", XGODUCK_SERVO_GEOM_PATTERN),
        ("head_ground_contact", XGODUCK_HEAD_GEOM_PATTERN),
        ("trunk_ground_contact", XGODUCK_TRUNK_GEOM_PATTERN),
    ):
        hits = {n for n in all_geoms if re.fullmatch(pattern, n)}
        assert hits, f"{sensor_name} pattern {pattern!r} matched nothing"
        matched[sensor_name] = hits

    assert matched["servo_ground_contact"] == set(XGODUCK_SERVO_SHELL_GEOMS)
    assert not any(n.endswith("_foot_collision") for n in matched["servo_ground_contact"])

    # The sensor cfg must carry the same pattern it was validated against.
    assert sensors["servo_ground_contact"].primary.pattern == XGODUCK_SERVO_GEOM_PATTERN
    for sensor in sensors.values():
        assert sensor.secondary.pattern == "terrain"
        assert sensor.fields == ("force",)

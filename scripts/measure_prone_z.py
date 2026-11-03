"""Measure where the trunk actually rests, per fallen orientation.

The prone spawn band (``prone_z_min/max`` in the VelStand layer) is not a taste
setting: spawn it too high and every prone episode opens with a 15 cm free-fall
(upstream bug, see microduck_velstand_env_cfg), spawn it too low and the head /
neck interpenetrates the floor at 90 deg. So the band is measured on the real
plant: drop the robot from a clearance height in each orientation with HOME
joint targets, let it settle, and read the trunk z.

MicroDuck's band (0.05-0.09 for a 0.044 m face-down rest) is the template:
rest + 6 mm .. rest + 46 mm.

    .venv/Scripts/python.exe scripts/measure_prone_z.py
"""

from __future__ import annotations

import argparse

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def _quat(kind: str, yaw: float, sign: float = 1.0) -> list[float]:
    """Same construction as mdp.set_random_prone_orientation, yaw applied."""
    import math

    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    s = 2.0 ** -0.5
    if kind == "face_down":
        return [s * cy, -s * sy, s * cy, s * sy]
    if kind == "face_up":
        return [s * cy, s * sy, -s * cy, s * sy]
    return [s * cy, sign * s * cy, sign * s * sy, s * sy]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-Velocity-Flat-XgoDuck")
    p.add_argument("--drop-z", type=float, default=0.25, help="spawn clearance")
    p.add_argument("--settle-steps", type=int, default=300, help="50 Hz steps")
    p.add_argument(
        "--full-contact",
        action="store_true",
        help="swap in the all-collisions robot (a walk-only model has feet-only "
        "collision geoms, so the trunk would fall through the floor)",
    )
    args = p.parse_args()

    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 24
    if args.full_contact:
        from mjlab_microduck.robot.xgoduck_constants import XGODUCK_VELSTAND_ROBOT_CFG

        cfg.scene.entities = {"robot": XGODUCK_VELSTAND_ROBOT_CFG}
    # A fall must not terminate: the walk recipe's fell_over fires the instant
    # the robot is pitched 90 deg, and the reset would overwrite the pose we are
    # trying to measure (this is what made three earlier runs report the
    # standing height for every orientation).
    cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

    env_ids = torch.arange(env.num_envs, device=env.device)
    kinds = ["face_down", "face_up", "side"] * (env.num_envs // 3)
    robot = env.scene["robot"]
    # write_root_state_to_sim, NOT env.sim.data.qpos: the env re-writes the sim
    # state from the entity every step (scene.write_data_to_sim), so a raw qpos
    # poke is silently discarded and the robot just stands there.
    pose = torch.zeros((env.num_envs, 7), device=env.device)
    pose[:, 2] = args.drop_z
    for i, kind in enumerate(kinds):
        pose[i, 3:7] = torch.tensor(_quat(kind, 0.0), device=env.device)
    robot.write_root_state_to_sim(
        torch.cat([pose, torch.zeros((env.num_envs, 6), device=env.device)], dim=1)
    )
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    zero_action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    # Some reward terms log into env.extras["log"]; the trainer creates it, a
    # bare env does not.
    env.extras.setdefault("log", {})
    # The first step RESETS every env (mjlab defers the initial reset), which
    # would discard the pose written above — burn that step first.
    env.step(zero_action)
    robot.write_root_state_to_sim(
        torch.cat([pose, torch.zeros((env.num_envs, 6), device=env.device)], dim=1)
    )
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )
    for _ in range(args.settle_steps):
        env.step(zero_action)

    z = env.sim.data.qpos[env_ids, 2]
    for kind in ("face_down", "face_up", "side"):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        vals = z[idx]
        print(
            f"{kind:10s} rest z = {vals.mean().item():.4f} m "
            f"(min {vals.min().item():.4f}  max {vals.max().item():.4f})"
        )
    print(
        f"\nsuggested band: rest+0.006 .. rest+0.046 "
        f"(lowest rest = {z.min().item():.4f})"
    )


if __name__ == "__main__":
    main()

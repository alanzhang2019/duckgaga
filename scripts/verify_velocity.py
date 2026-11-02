"""Verify a trained velocity policy against the commands it was actually given.

`watch_run.py` reads `Metrics/twist/error_vel_xy`, but that number is only
interpretable next to the command magnitude: this task samples lin_vel_x in
+-0.4 and lin_vel_y in +-0.3, so the mean commanded speed is only ~0.3 m/s and
an error of 0.30 could mean "tracks well" or "stands still and ignores the
command" — both score the same. This script therefore reports the error next to
a **do-nothing baseline** (`mean |cmd_xy|`, the error a policy that always
outputs zero velocity would get). The policy is only tracking if its error is
clearly BELOW that baseline, and the per-bucket table shows where it breaks.

Velocities are read from `qvel` and rotated into the body frame by hand rather
than from `root_link_lin_vel_b`: `data` caches transforms that are only
refreshed by `sim.forward()`, the same trap that made `verify_standup.py`
report a standing robot as 52 deg tilted.

Usage:
  .venv/Scripts/python.exe scripts/verify_velocity.py \
      --checkpoint logs/rsl_rl/xgoduck_velocity/<run>/model_2999.pt \
      --num-envs 32 --steps 600 --video
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

# A policy that beats the do-nothing baseline by less than this is not
# tracking the command, whatever the reward curve says.
MIN_TRACKING_GAIN = 0.35
BUCKETS = [(0.00, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 10.0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-Velocity-Flat-XgoDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=600, help="policy steps (50 Hz)")
    p.add_argument("--warmup", type=int, default=50,
                   help="steps excluded from the stats (reset transient)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-length", type=int, default=300)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--device", default=None)
    return p.parse_args()


def quat_to_mat(quat):
    """(w, x, y, z) -> 3x3 rotation matrix, batched."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def body_velocity(env):
    """(lin_vel_xy, ang_vel_z) of the trunk in the body frame, from qvel.

    MuJoCo stores free-joint velocity in the WORLD frame, so rotate by R^T.
    """
    robot = env.scene["robot"]
    idx = robot.data.indexing
    qpos = robot.data.data.qpos
    qvel = robot.data.data.qvel
    v_adr = idx.free_joint_v_adr
    q_adr = idx.free_joint_q_adr

    rot = quat_to_mat(qpos[:, q_adr[3:7]])
    v_w = qvel[:, v_adr[0:3]]
    w_w = qvel[:, v_adr[3:6]]
    v_b = torch.einsum("nji,nj->ni", rot, v_w)
    w_b = torch.einsum("nji,nj->ni", rot, w_w)
    return v_b[:, :2], w_b[:, 2]


def trunk_state(env):
    """Trunk height above the terrain origin and cos(tilt) — sanity, not tracking."""
    robot = env.scene["robot"]
    idx = robot.data.indexing
    qpos = robot.data.data.qpos
    z = qpos[:, idx.free_joint_q_adr[2]] - env.scene.terrain.env_origins[:, 2]
    quat = qpos[:, idx.free_joint_q_adr[3:7]]
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return torch.nan_to_num(z, nan=0.0), torch.nan_to_num(cos_tilt, nan=1.0)


def main():
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.video:
        env_cfg.viewer.height = args.video_height
        env_cfg.viewer.width = args.video_width

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    print(f"[verify] checkpoint {ckpt}")

    env = ManagerBasedRlEnv(
        cfg=env_cfg, device=device, render_mode="rgb_array" if args.video else None
    )
    if args.video:
        env = VideoRecorder(
            env,
            video_folder=ckpt.parent / "videos" / "verify",
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd_term = env.command_manager.get_term("twist")
    obs, _ = vec_env.reset()

    n = args.num_envs
    steps = args.steps
    cmds = torch.zeros(steps, n, 3, device=device)
    vels = torch.zeros(steps, n, 2, device=device)
    wzs = torch.zeros(steps, n, device=device)
    zs = torch.zeros(steps, n, device=device)
    tilts = torch.zeros(steps, n, device=device)
    fell = torch.zeros(steps, n, dtype=torch.bool, device=device)

    for t in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, extras = vec_env.step(actions)
        cmds[t] = cmd_term.command
        vels[t], wzs[t] = body_velocity(env)
        zs[t], tilts[t] = trunk_state(env)
        # `fell_over` is a termination term; anything that ends an episode
        # before its time limit counts as a fall for this report.
        timeout = extras.get("time_outs") if isinstance(extras, dict) else None
        fell[t] = dones.bool() & ~(timeout.bool() if timeout is not None else False)

    vec_env.close()

    # Warmup window: the first steps after reset are a settling transient and
    # would flatter the error for the do-nothing comparison as well.
    sl = slice(args.warmup, steps)
    cmd = cmds[sl]
    vel = vels[sl]
    wz = wzs[sl]
    cmd_xy = cmd[:, :, :2]
    err_xy = torch.norm(cmd_xy - vel, dim=-1)          # per env per step
    err_wz = (cmd[:, :, 2] - wz).abs()
    cmd_mag = torch.norm(cmd_xy, dim=-1)

    # The number this whole script exists for: what a policy that outputs
    # nothing would score on the same commands.
    baseline_xy = cmd_mag.mean().item()
    baseline_wz = cmd[:, :, 2].abs().mean().item()
    got_xy = err_xy.mean().item()
    got_wz = err_wz.mean().item()

    print()
    print(f"commands   lin_xy {cmd_mag.mean():.3f} m/s (max {cmd_mag.max():.3f})"
          f"   yaw {cmd[:, :, 2].abs().mean():.3f} rad/s"
          f"   (|vx|<={cmd[:, :, 0].abs().max():.2f},"
          f" |vy|<={cmd[:, :, 1].abs().max():.2f})")
    print(f"tracking   lin_xy err {got_xy:.3f} m/s   yaw err {got_wz:.3f} rad/s"
          f"   ({n} envs x {steps - args.warmup} steps)")
    print(f"baseline   do-nothing lin_xy err {baseline_xy:.3f} m/s"
          f"   yaw {baseline_wz:.3f} rad/s   <- error of always commanding zero")
    gain_xy = 1.0 - got_xy / max(baseline_xy, 1e-6)
    gain_wz = 1.0 - got_wz / max(baseline_wz, 1e-6)
    print(f"gain       lin_xy {gain_xy:+.0%} better than doing nothing"
          f"   yaw {gain_wz:+.0%}")

    print()
    print("by commanded speed (m/s)      n     cmd      got      err   err/cmd")
    for lo, hi in BUCKETS:
        m = (cmd_mag >= lo) & (cmd_mag < hi)
        if m.sum() == 0:
            continue
        c = cmd_mag[m].mean().item()
        v = torch.norm(vel, dim=-1)[m].mean().item()
        e = err_xy[m].mean().item()
        print(f"  {lo:.2f} .. {hi:<5.2f}          {int(m.sum()):6d}  {c:6.3f}"
              f"  {v:7.3f}  {e:7.3f}   {e / max(c, 1e-6):5.0%}")

    # Signed bias: a policy that systematically undershoots shows up here as
    # got << cmd even when the error looks acceptable.
    print()
    print("signed     vx cmd {:.3f} -> got {:.3f}   vy cmd {:.3f} -> got {:.3f}"
          .format(cmd[:, :, 0].mean().item(), vel[:, :, 0].mean().item(),
                  cmd[:, :, 1].mean().item(), vel[:, :, 1].mean().item()))
    print("           wz cmd {:.3f} -> got {:.3f}"
          .format(cmd[:, :, 2].mean().item(), wz.mean().item()))

    # Turn-in-place: zero linear command, non-zero yaw command. The clearest
    # single test of "does it follow the command or just walk forward".
    tip = (cmd_mag < 0.05) & (cmd[:, :, 2].abs() > 0.3)
    if tip.sum() > 0:
        print()
        print("in place   {} samples   wz cmd {:.2f} -> got {:.2f}   "
              "linear drift {:.3f} m/s".format(
                  int(tip.sum()), cmd[:, :, 2][tip].abs().mean().item(),
                  wz[tip].abs().mean().item(),
                  torch.norm(vel, dim=-1)[tip].mean().item()))

    print()
    z, cos_tilt = trunk_state(env)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
    print(f"stability  height {zs[sl].mean():.4f} m   tilt {tilt_deg.mean():.1f} deg"
          f"   falls {fell[sl].float().mean():.1%} of steps")

    if args.video:
        print(f"video      {ckpt.parent / 'videos' / 'verify'}")

    print()
    ok = gain_xy >= MIN_TRACKING_GAIN
    print(f"VERDICT  {'PASS' if ok else 'CHECK'} — lin_xy error {got_xy:.3f} vs"
          f" do-nothing baseline {baseline_xy:.3f} ({gain_xy:+.0%} better,"
          f" need >={MIN_TRACKING_GAIN:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

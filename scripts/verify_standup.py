"""Verify a trained policy numerically AND visually, in one pass.

`watch_run.py` reads the reward curves, but a rising reward is not proof the
robot stands: `standing_composite` is a product of sub-terms, so it can jump
superlinearly while the actual pose is still wrong. This script steps the
policy in play mode and measures the physical quantities the reward is a proxy
for — trunk height and tilt — then writes a video of the same rollout.

Usage:
  .venv/Scripts/python.exe scripts/verify_standup.py \
      --checkpoint logs/rsl_rl/xgoduck_standup/<run>/model_2999.pt \
      --num-envs 16 --steps 600 --video
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

# Fallback only: the real target is read from the task's own reward config,
# because the XgoDuck retarget rewrites `target_height` (MicroDuck 0.115 m →
# XgoDuck 0.1392 m). Hardcoding either one misreports the other by 20%.
STAND_Z_FALLBACK = 0.115
UPRIGHT_TOL_DEG = 15.0
STAND_Z = STAND_Z_FALLBACK  # replaced in main() from env_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-StandUp-Flat-XgoDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=600, help="policy steps (50 Hz)")
    p.add_argument("--warmup", type=int, default=100,
                   help="steps excluded from the settled-window stats")
    p.add_argument("--tail", type=int, default=0,
                   help="settled-window length in steps (default: steps/6). "
                        "Keep steps below the episode length, or every env "
                        "resets mid-rollout and the window straddles the reset.")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-length", type=int, default=300)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--device", default=None)
    return p.parse_args()


def root_pose_from_qpos(env):
    """Root position and (w, x, y, z) quaternion, read straight from qpos.

    NOT from `root_link_pos_w` / `root_link_quat_w`: those read `xpos`/`xquat`,
    which are only refreshed by `sim.forward()`. Right after `env.step()` the
    cached values can lag a step behind, which is enough to report a standing
    robot as 52 deg tilted (observed — the same trap `set_ground_state` calls
    out in mdp.py).
    """
    robot = env.scene["robot"]
    idx = robot.data.indexing.free_joint_q_adr
    qpos = robot.data.data.qpos
    pos = qpos[:, idx[0:3]]
    quat = qpos[:, idx[3:7]]
    return pos, quat


def trunk_z(env):
    """Trunk height above the terrain origin, in metres."""
    pos, _ = root_pose_from_qpos(env)
    return torch.nan_to_num(
        pos[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )


def cos_tilt(env):
    """cos(angle between the trunk's z axis and world up): 1 = upright."""
    _, quat = root_pose_from_qpos(env)
    return torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0
    )


def main():
    global STAND_Z
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    # The task's own notion of "standing", after any retargeting.
    try:
        STAND_Z = float(env_cfg.rewards["height_stand"].params["target_height"])
    except (KeyError, AttributeError, TypeError):
        STAND_Z = STAND_Z_FALLBACK
    print(f"[verify] target stand height {STAND_Z:.4f} m (from reward cfg)")
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

    obs, _ = vec_env.reset()
    n = args.num_envs
    z0 = trunk_z(env).clone()
    tilt0 = torch.rad2deg(torch.acos(cos_tilt(env).clamp(-1.0, 1.0))).clone()

    zs = torch.zeros(args.steps, n, device=device)
    tilts = torch.zeros(args.steps, n, device=device)
    for t in range(args.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)
        zs[t] = trunk_z(env)
        tilts[t] = torch.rad2deg(torch.acos(cos_tilt(env).clamp(-1.0, 1.0)))

    # Cross-check against the cached xquat the reward terms read. A large gap
    # means the cached transform lagged, not that the robot is crooked.
    q_cached = env.scene["robot"].data.root_link_quat_w
    cached_tilt = torch.rad2deg(
        torch.acos((1.0 - 2.0 * (q_cached[:, 1] ** 2 + q_cached[:, 2] ** 2)).clamp(-1.0, 1.0))
    )
    qpos_tilt = torch.rad2deg(torch.acos(cos_tilt(env).clamp(-1.0, 1.0)))

    vec_env.close()

    # Settled window: the last stretch of the rollout, after the get-up
    # transient. A third of a 600-step run is 200 steps = 4 s, which was still
    # inside the transient for the harder face-up starts — the policy stood up
    # but the window closed before it got there.
    tail = args.tail or max(args.steps // 6, 50)
    win = zs[-tail:]
    win_t = tilts[-tail:]
    z_env = win.mean(dim=0)
    tilt_env = win_t.mean(dim=0)

    # Reset-robust: the fraction of settled STEPS in a standing pose, averaged
    # over envs. A mean over the window would be dragged down by any env that
    # timed out mid-window and restarted from a fallen pose.
    standing = (win >= STAND_Z - 0.02) & (win_t <= UPRIGHT_TOL_DEG)
    frac_standing = standing.float().mean().item()
    env_frac = standing.float().mean(dim=0)

    def q(x, frac):
        return torch.quantile(x, frac).item()

    print()
    print(f"start    z {z0.mean():.4f} m   tilt {tilt0.mean():5.1f} deg"
          f"   (target z {STAND_Z:.3f} m, tilt 0)")
    print()
    print("timeline  step   z(mean)  tilt(mean)  standing")
    for t in range(0, args.steps, max(args.steps // 10, 1)):
        seg = slice(t, t + max(args.steps // 10, 1))
        st = ((zs[seg] >= STAND_Z - 0.02) & (tilts[seg] <= UPRIGHT_TOL_DEG))
        print(f"          {t:4d}   {zs[seg].mean():7.4f}  {tilts[seg].mean():9.1f}"
              f"  {st.float().mean():7.0%}")
    print()
    print(f"settled  z {z_env.mean():.4f} m  [{q(z_env, 0.0):.4f} .. {q(z_env, 1.0):.4f}]"
          f"   tilt {tilt_env.mean():5.1f} deg"
          f"  [{q(tilt_env, 0.0):.1f} .. {q(tilt_env, 1.0):.1f}]"
          f"   (last {tail} steps)")
    print(f"height   mean {z_env.mean() / STAND_Z:.1%} of target"
          f"   worst env {q(z_env, 0.0) / STAND_Z:.1%}")
    print(f"standing {frac_standing:.0%} of settled steps   "
          f"{int((env_frac >= 0.8).sum())}/{n} envs standing >=80% of the time")
    print(f"upright  {(win_t <= UPRIGHT_TOL_DEG).float().mean():.0%} of settled steps"
          f"   best env {q(tilt_env, 0.0):.1f} deg, worst {q(tilt_env, 1.0):.1f} deg")

    # Instability check: a policy that balances by oscillating shows up as a
    # large step-to-step height swing even when the mean looks right.
    swing = (win[1:] - win[:-1]).abs().mean().item()
    print(f"swing    {swing * 100:.2f} cm per step (mean |Δz|)")
    print(f"xcheck   qpos tilt {qpos_tilt.mean():.1f} deg vs cached xquat"
          f" {cached_tilt.mean():.1f} deg  (should agree)")
    if args.video:
        print(f"video    {ckpt.parent / 'videos' / 'verify'}")

    print()
    ok = frac_standing
    print(f"VERDICT  {'PASS' if ok >= 0.8 else 'CHECK'} — {ok:.0%} of settled steps"
          f" in a standing pose")
    return 0 if ok >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())

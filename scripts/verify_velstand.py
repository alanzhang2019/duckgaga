"""Measure a VelStand policy on the two skills it is supposed to have.

VelStand adds fall recovery on top of the walk recipe, which makes every
single number in the training log ambiguous:

  * `Metrics/twist/error_vel_xy` is computed over ALL frames, including the
    frames where the robot is lying on its trunk and a velocity command is
    physically meaningless. With `prone_init_prob` at 0.45 it mostly measures
    "how long did it spend on the floor", not tracking.
  * `Episode_Termination/fallen_too_long == 0` does not mean "never falls". It
    means "never stays below the gate for 8 s". A robot crouching at 0.111 m
    with 34 deg of tilt is under neither gate and scores zero on both.
  * The play cfg runs `topple_push` at its curriculum FINAL value (1.2 m/s of
    trunk velocity every 0.5-1 s). That is not a walk test — the velocity
    teacher, which has no recovery skill at all, reads 0.080 m / 68.7 deg in
    that environment. Any comparison made there measures push tolerance.

So this script runs two separate, clean phases on the same checkpoint:

  walk      topple_push removed, standing start. Tracking is reported next to
            the do-nothing baseline, exactly like verify_velocity.py (it
            imports that script's measurement helpers so the definitions
            cannot drift apart).
  recovery  `prone_prob = 1.0`, no pushes. Every env starts on the floor; the
            script reports how long each one takes to become upright and
            whether it stays there.

Usage:
  .venv/Scripts/python.exe scripts/verify_velstand.py \
      --checkpoint logs/rsl_rl/xgoduck_velstand/<run>/model_2999.pt \
      --num-envs 48 --seed 7

  # walking only / recovery only
  ... --skip-recovery
  ... --skip-walk
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# Same directory, so `import verify_velocity` resolves when this is run as a
# script. Importing (rather than copying) the helpers is deliberate: the whole
# point of the baseline comparison is that both scripts measure identically.
from verify_velocity import BUCKETS, body_velocity, trunk_state

UPRIGHT_TILT_DEG = 25.0   # the layer's own "genuinely upright" gate
UPRIGHT_Z_FRAC = 0.90     # of nominal standing height


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-VelStand-Flat-XgoDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=48)
    p.add_argument("--steps", type=int, default=600, help="walk steps (50 Hz)")
    p.add_argument("--recovery-steps", type=int, default=600,
                   help="12 s window, matching the upstream rise battery")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--push", type=float, default=None,
                   help="override topple_push magnitude (m/s). Default: remove "
                        "the event entirely, so this is a walk test.")
    p.add_argument("--skip-walk", action="store_true")
    p.add_argument("--skip-recovery", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_env(task, num_envs, seed, device, *, prone_prob, push):
    """VelStand env set up for measurement rather than for training.

    Terminations are removed on purpose: every one of them ends the episode
    with a reset, and a reset overwrites the pose being measured (the same trap
    that made `measure_prone_z.py` report the standing height for a robot lying
    on its face). Episodes therefore never end and the measurement window is
    under our control.
    """
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.terminations = {}
    cfg.episode_length_s = 3600.0

    if push is None:
        cfg.events.pop("topple_push", None)
    if prone_prob > 0.0:
        # A rise test must not be interrupted: `push_robot` fires every 0.5-1 s
        # at +-0.2 m/s, and a policy that stands up in 2 s and is then shoved
        # over at 6 s scores as "never recovered" under the stay-upright-to-the-
        # end rule below. MicroDuck's rise battery measures the rise alone too.
        cfg.events.pop("push_robot", None)
    else:
        cfg.events["topple_push"].params["velocity_range"] = {
            "x": (-push, push), "y": (-push, push)
        }

    # The prone share is driven by a curriculum term that writes into the event
    # params on every step, so setting `prone_prob` alone is silently undone at
    # the first reset (which is exactly what a measurement run hits). Drop the
    # term, then set the value it would have overwritten.
    cfg.curriculum.pop("prone_init_prob", None)
    cfg.curriculum.pop("topple_push_range", None)
    prone = cfg.events["random_prone_init"].params
    prone["prone_prob"] = prone_prob
    if prone_prob > 0.0:
        # Recovery test: start on the floor in any orientation, nothing else.
        prone["crouch_prob"] = 0.0
        prone["sit_prob"] = 0.0
        prone["so3_prob"] = 0.0

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    # Some reward terms write into env.extras["log"]; the trainer creates it.
    env.extras.setdefault("log", {})
    stand_z = env.cfg.rewards["body_pose_tracking"].params["nominal_height"]
    return env, stand_z


def load_policy(env, task, agent_cfg, checkpoint, device):
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=device)
    return vec_env, runner.get_inference_policy(device=device)


def measure_walk(args, agent_cfg, device):
    env, stand_z = build_env(args.task, args.num_envs, args.seed, device,
                             prone_prob=0.0, push=args.push)
    vec_env, policy = load_policy(env, args.task, agent_cfg,
                                 Path(args.checkpoint).resolve(), device)
    cmd_term = env.command_manager.get_term("twist")
    obs, _ = vec_env.reset()

    n, steps = args.num_envs, args.steps
    cmds = torch.zeros(steps, n, 3, device=device)
    vels = torch.zeros(steps, n, 2, device=device)
    wzs = torch.zeros(steps, n, device=device)
    zs = torch.zeros(steps, n, device=device)
    cos_ts = torch.zeros(steps, n, device=device)

    for t in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)
        cmds[t] = cmd_term.command
        vels[t], wzs[t] = body_velocity(env)
        zs[t], cos_ts[t] = trunk_state(env)

    sl = slice(args.warmup, steps)
    cmd, vel, wz = cmds[sl], vels[sl], wzs[sl]
    cmd_mag = torch.norm(cmd[:, :, :2], dim=-1)
    err_xy = torch.norm(cmd[:, :, :2] - vel, dim=-1)
    err_wz = (cmd[:, :, 2] - wz).abs()

    base_xy, base_wz = cmd_mag.mean().item(), cmd[:, :, 2].abs().mean().item()
    got_xy, got_wz = err_xy.mean().item(), err_wz.mean().item()
    gain_xy = 1.0 - got_xy / max(base_xy, 1e-6)
    gain_wz = 1.0 - got_wz / max(base_wz, 1e-6)
    tilt = torch.rad2deg(torch.acos(cos_ts[sl].clamp(-1, 1)))

    print()
    print("--- walk" + ("" if args.push is None else f" (topple_push +-{args.push})") + " ---")
    print(f"commands   lin_xy {base_xy:.3f} m/s (max {cmd_mag.max():.3f})"
          f"   yaw {base_wz:.3f} rad/s")
    print(f"tracking   lin_xy err {got_xy:.3f} m/s   yaw err {got_wz:.3f} rad/s"
          f"   ({n} envs x {steps - args.warmup} steps)")
    print(f"baseline   do-nothing lin_xy err {base_xy:.3f} m/s   yaw {base_wz:.3f} rad/s")
    print(f"gain       lin_xy {gain_xy:+.0%}   yaw {gain_wz:+.0%}")

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

    tip = (cmd_mag < 0.05) & (cmd[:, :, 2].abs() > 0.3)
    if tip.sum() > 0:
        print(f"in place   wz cmd {cmd[:, :, 2][tip].abs().mean():.2f} -> got"
              f" {wz[tip].abs().mean():.2f}   linear drift"
              f" {torch.norm(vel, dim=-1)[tip].mean():.3f} m/s")

    print(f"stability  height {zs[sl].mean():.4f} m ({zs[sl].mean() / stand_z:.0%} of"
          f" {stand_z:.4f})   tilt {tilt.mean():.1f} deg"
          f"   upright {100.0 * (tilt < UPRIGHT_TILT_DEG).float().mean():.0f}% of steps")

    vec_env.close()
    return gain_xy


def measure_recovery(args, agent_cfg, device):
    env, stand_z = build_env(args.task, args.num_envs, args.seed, device,
                             prone_prob=1.0, push=None)
    vec_env, policy = load_policy(env, args.task, agent_cfg,
                                 Path(args.checkpoint).resolve(), device)
    obs, _ = vec_env.reset()

    n, steps = args.num_envs, args.recovery_steps
    zs = torch.zeros(steps, n, device=device)
    cos_ts = torch.zeros(steps, n, device=device)
    # `cos(tilt)` cannot separate a face-down spawn from a side spawn (both are
    # 90 deg), so keep the quaternion: a pitch-dominated one is face-down/up.
    robot = env.scene["robot"]
    q0 = robot.data.data.qpos[:, robot.data.indexing.free_joint_q_adr[3:7]].clone()

    for t in range(steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)
        zs[t], cos_ts[t] = trunk_state(env)

    tilt = torch.rad2deg(torch.acos(cos_ts.clamp(-1, 1)))
    up = (tilt < UPRIGHT_TILT_DEG) & (zs > UPRIGHT_Z_FRAC * stand_z)

    # First step at which the env is upright AND stays upright to the end.
    # A single crossing followed by a collapse is not a recovery.
    never_again_down = torch.flip(torch.cumprod(torch.flip(up, [0]).int(), 0).bool(), [0])
    recovered = never_again_down.any(dim=0)
    first = torch.where(
        recovered, never_again_down.float().argmax(dim=0),
        torch.full((n,), steps, device=device),
    ).float()

    initial_tilt = tilt[0]
    labels = {0: "face down", 1: "face up", 2: "side", 3: "upright"}
    pitch_dom = q0[:, 2].abs() > q0[:, 1].abs()
    kinds = torch.full((n,), 2, dtype=torch.long, device=device)
    kinds[pitch_dom & (q0[:, 2] > 0)] = 0
    kinds[pitch_dom & (q0[:, 2] < 0)] = 1
    kinds[initial_tilt < 45] = 3

    print()
    print("--- recovery (all envs start prone) ---")
    print(f"spawn      {n} envs   initial tilt "
          f"{initial_tilt.mean():.0f} deg (min {initial_tilt.min():.0f},"
          f" max {initial_tilt.max():.0f})")
    ok = int(recovered.sum())
    if ok:
        print(f"recovered  {ok}/{n}  ({ok / n:.0%})   "
              f"time {first[recovered].mean() / 50:.2f} s mean, "
              f"{first[recovered].max() / 50:.2f} s worst")
    else:
        print(f"recovered  {ok}/{n}  (0%)")
    print(f"final      height {zs[-1].mean():.4f} m"
          f" ({zs[-1].mean() / stand_z:.0%} of {stand_z:.4f})"
          f"   tilt {tilt[-1].mean():.1f} deg"
          f"   still upright {100.0 * up[-1].float().mean():.0f}%")
    for idx, label in labels.items():
        m = kinds == idx
        if m.sum() == 0:
            continue
        print(f"  {label:<9} n={int(m.sum()):3d}   recovered"
              f" {100.0 * recovered[m].float().mean():3.0f}%"
              f"   mean time {first[m].mean() / 50:.2f} s")

    vec_env.close()
    return ok / n


def main():
    args = parse_args()
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    print(f"[verify] checkpoint {ckpt}")
    agent_cfg = load_rl_cfg(args.task)

    gain = None if args.skip_walk else measure_walk(args, agent_cfg, device)
    rate = None if args.skip_recovery else measure_recovery(args, agent_cfg, device)

    print()
    if gain is not None and rate is not None:
        walk_ok = gain >= 0.35
        rec_ok = rate >= 0.80
        print(f"VERDICT  walk {'PASS' if walk_ok else 'FAIL'} ({gain:+.0%} over"
              f" baseline, need >=+35%)   recovery {'PASS' if rec_ok else 'FAIL'}"
              f" ({rate:.0%}, need >=80%)")
        return 0 if (walk_ok and rec_ok) else 1
    if gain is not None:
        return 0 if gain >= 0.35 else 1
    return 0 if rate >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())

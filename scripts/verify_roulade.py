"""Measure a roulade policy on the ONE thing the training log cannot tell you.

The training log prints `Episode_Reward/roulade_progress: 0.93`, and that
number is nearly unreadable for three reasons:

  * It is a per-step AVERAGE of `delta / (step_dt * 2pi)` — i.e. paid radians
    per second divided by 2pi — not a fraction of a roll. To convert:
    radians_per_second = reward / weight * 2pi, and the total rotation is that
    times the episode length. A converged run reads ~0.93, which is ~0.73 rad/s
    of paid rotation, which is ~294 deg over a 7 s episode. "0.93" is not
    "93% of a roll"; it is ~82%.
  * `Episode_Reward/roulade_landing_*` are gated on the accumulated angle
    crossing 260-330 deg, so a high value proves the gate opened, but says
    nothing about WHEN, how many times, or whether the robot was already
    standing when the episode ended.
  * `Metrics/twist/error_vel_xy` is meaningless here: the roulade task pays no
    velocity tracking at all.

So this script drives the checkpoint directly and reads the accumulator the
rewards read (`env._roulade_max`, radians, forward-only, support-gated,
flatness-gated), plus height / tilt / head-latch / displacement, and reports
them as physical quantities.

Two comparisons are built in, because a roll number alone proves nothing:

  --baseline <ckpt>   run a second checkpoint (normally `model_0.pt`, the
                      untrained net) through the identical setup. If it scores
                      the same, the reward shaped nothing.
  --null              also run a zero-action policy — the "does doing nothing
                      already fall over the gate" sanity check.

Usage:
  .venv/Scripts/python.exe scripts/verify_roulade.py \
      --checkpoint logs/rsl_rl/xgoduck_roulade/<run>/model_5999.pt \
      --baseline logs/rsl_rl/xgoduck_roulade/<run>/model_0.pt --null
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

# Mirrors the env cfg's own completion window. Imported values would be nicer,
# but the roulade cfg builds them at module scope for MicroDuck and XgoDuck
# alike; duplicating two constants is cheaper than importing a private name.
GATE_LO_DEG = 260.0
GATE_HI_DEG = 330.0
UPRIGHT_TILT_DEG = 25.0
UPRIGHT_Z_FRAC = 0.90


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-Roulade-Flat-XgoDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline", default=None,
                   help="second checkpoint through the identical setup "
                        "(use model_0.pt to prove the reward shaped something)")
    p.add_argument("--null", action="store_true",
                   help="also run a zero-action policy")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=400, help="8 s at 50 Hz")
    p.add_argument("--elan", type=float, default=0.0,
                   help="forward base velocity at spawn (m/s). 0 = standstill "
                        "start, the strict test.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_env(task, num_envs, seed, device, elan):
    """Roulade env configured for measurement, standing spawn only.

    Terminations are removed: every one of them ends the episode with a reset,
    and the reset re-runs `reset_roulade_state`, which zeroes the accumulator
    and teleports the robot back to a fresh standing pose. With a 12 s window
    and an episode cap of ~7 s that would erase the very thing being measured
    (the same trap `measure_prone_z.py` and `verify_velstand.py` both hit).
    """
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.terminations = {}
    cfg.episode_length_s = 3600.0

    # The spawn mix is written by a curriculum term on EVERY step, so setting
    # standing_prob alone is silently undone at the first reset.
    cfg.curriculum.pop("roulade_spawn_mix", None)

    spawn = cfg.events["set_roulade_state"].params
    spawn["standing_prob"] = 1.0
    spawn["midroll_prob"] = 0.0
    spawn["joint_noise_std"] = 0.0          # clean, repeatable standing pose
    spawn["tuck_factor_range"] = (1.0, 1.0)  # no tuck pre-loading either
    spawn["forward_vel_range"] = (elan, elan)

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.extras.setdefault("log", {})
    stand_z = env.cfg.rewards["roulade_landing_composite"].params["target_height"]
    return env, stand_z


def load_policy(env, task, agent_cfg, checkpoint, device):
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=device)
    return vec_env, runner.get_inference_policy(device=device)


def trunk(env):
    """Trunk height above the terrain origin, cos(tilt), and world xy."""
    robot = env.scene["robot"]
    idx = robot.data.indexing
    qpos = robot.data.data.qpos
    origin = env.scene.terrain.env_origins
    z = qpos[:, idx.free_joint_q_adr[2]] - origin[:, 2]
    quat = qpos[:, idx.free_joint_q_adr[3:7]]
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    xy = qpos[:, idx.free_joint_q_adr[0:2]] - origin[:, 0:2]
    return (torch.nan_to_num(z, nan=0.0),
            torch.nan_to_num(cos_tilt, nan=1.0),
            torch.nan_to_num(xy, nan=0.0))


def run(args, agent_cfg, device, checkpoint, label, null=False):
    env, stand_z = build_env(args.task, args.num_envs, args.seed, device,
                             args.elan)
    if null:
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policy = None
    else:
        vec_env, policy = load_policy(env, args.task, agent_cfg,
                                      Path(checkpoint).resolve(), device)
    obs, _ = vec_env.reset()

    n, steps = args.num_envs, args.steps
    ang = torch.zeros(steps, n, device=device)   # frontier, radians
    zs = torch.zeros(steps, n, device=device)
    cos_ts = torch.zeros(steps, n, device=device)
    xys = torch.zeros(steps, n, 2, device=device)

    for t in range(steps):
        if policy is None:
            actions = torch.zeros(n, env.action_manager.total_action_dim,
                                  device=device)
        else:
            with torch.no_grad():
                actions = policy(obs)
        obs, _, _, _ = vec_env.step(actions)
        # The reward terms update the accumulator as a side effect of being
        # evaluated, so by the time step() returns it is already current.
        ang[t] = env._roulade_max
        zs[t], cos_ts[t], xys[t] = trunk(env)

    deg = torch.rad2deg(ang)
    tilt = torch.rad2deg(torch.acos(cos_ts.clamp(-1, 1)))
    latch = env._roulade_head_latch

    reached = deg[-1] >= GATE_LO_DEG
    in_gate = (deg[-1] >= GATE_LO_DEG) & (deg[-1] <= GATE_HI_DEG)
    over = deg[-1] > GATE_HI_DEG
    # First step at which the frontier crossed the gate, per env.
    crossed = deg >= GATE_LO_DEG
    first = torch.where(
        crossed.any(dim=0), crossed.float().argmax(dim=0),
        torch.full((n,), steps, device=device),
    ).float()
    final_up = (tilt[-1] < UPRIGHT_TILT_DEG) & (zs[-1] > UPRIGHT_Z_FRAC * stand_z)
    disp = torch.norm(xys[-1] - xys[0], dim=-1)

    print()
    print(f"=== {label} ===")
    print(f"spawn        standing, elan {args.elan:.2f} m/s, "
          f"{n} envs x {steps} steps ({steps / 50:.0f} s)")
    print(f"rotation     frontier {deg[-1].mean():5.0f} deg mean"
          f"   (min {deg[-1].min():.0f}, max {deg[-1].max():.0f})")
    print(f"             past {GATE_LO_DEG:.0f} deg: "
          f"{100.0 * reached.float().mean():3.0f}%"
          f"   past {GATE_HI_DEG:.0f} deg (full roll): "
          f"{100.0 * over.float().mean():3.0f}%"
          f"   parked in the {GATE_LO_DEG:.0f}-{GATE_HI_DEG:.0f} deg window: "
          f"{100.0 * in_gate.float().mean():3.0f}%")
    if int(reached.sum()):
        print(f"             time to gate "
              f"{first[reached].mean() / 50:.2f} s mean, "
              f"{first[reached].max() / 50:.2f} s worst")
    print(f"head latch   {100.0 * latch.float().mean():3.0f}% of envs "
          f"(head-top touched ground inside 20-170 deg)")
    print(f"final        height {zs[-1].mean():.4f} m"
          f" ({zs[-1].mean() / stand_z:.0%} of {stand_z:.4f})"
          f"   tilt {tilt[-1].mean():.1f} deg"
          f"   standing {100.0 * final_up.float().mean():3.0f}%")
    print(f"displacement {disp.mean():.2f} m mean, {disp.max():.2f} m max")
    print(f"stability    max tilt {tilt.max(dim=0).values.mean():.0f} deg mean"
          f"   airborne-free height min {zs.min(dim=0).values.mean():.4f} m")

    # Per-env trajectory of the env closest to the median frontier — a single
    # trace is the only way to see whether the rotation was one continuous
    # roll or a stumble that happens to sum past 260 deg.
    med = int((deg[-1] - deg[-1].median()).abs().argmin())
    trace = deg[:, med]
    print(f"trace        env {med} frontier every 25 steps: "
          + " ".join(f"{trace[i]:.0f}" for i in range(0, steps, 25)))
    print(f"             env {med} tilt: "
          + " ".join(f"{tilt[i, med]:.0f}" for i in range(0, steps, 25)))

    vec_env.close()
    return deg[-1].mean().item(), reached.float().mean().item()


def main():
    args = parse_args()
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    agent_cfg = load_rl_cfg(args.task)

    print(f"[verify] checkpoint {ckpt}")
    mean_deg, reached = run(args, agent_cfg, device, ckpt, ckpt.parent.name)

    base = None
    if args.baseline:
        b = Path(args.baseline).resolve()
        base = run(args, agent_cfg, device, b, f"baseline {b.name}")
    if args.null:
        run(args, agent_cfg, device, None, "null (zero actions)", null=True)

    print()
    ok = reached >= 0.80
    print(f"VERDICT  roulade {'PASS' if ok else 'FAIL'}"
          f" ({reached:.0%} of envs past {GATE_LO_DEG:.0f} deg, need >=80%"
          f"   mean frontier {mean_deg:.0f} deg)")
    if base is not None:
        delta = mean_deg - base[0]
        print(f"         vs baseline: {delta:+.0f} deg frontier,"
              f" {reached - base[1]:+.0%} reaching the gate"
              + ("" if abs(delta) > 30 else "  <- WARNING: no separation,"
                 " the reward did not shape a roll"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

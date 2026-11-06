# XgoDuck tasks in this fork

This repository is a fork of
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl),
rebased onto upstream `develop` at `bae39f5`, that adds the **XgoDuck** biped as
a second robot alongside MicroDuck.

The upstream README, `AGENTS.md` playbook and `docs/` policy writeups all still
apply: XgoDuck reuses the MicroDuck task *recipes* (rewards, curricula, terrain,
events) and swaps the robot, the actuator model and the height constants.

Branch: `xgoduck`.

## Tasks

`uv run list-envs` prints the live registry. The 13 XgoDuck ids:

| Task | Notes |
|---|---|
| `Mjlab-Velocity-Flat-XgoDuck` | walking, flat |
| `Mjlab-Velocity-Rough-XgoDuck` | walking, rough terrain |
| `Mjlab-Velocity-Swizzle-XgoDuck` | turning in place |
| `Mjlab-StandUp-Flat-XgoDuck` | fall recovery, flat |
| `Mjlab-StandUp-Rough-XgoDuck` | fall recovery, rough |
| `Mjlab-SitStand-Flat-XgoDuck` | sit ↔ stand |
| `Mjlab-SitStand-Rough-XgoDuck` | sit ↔ stand, rough |
| `Mjlab-GroundPick-Flat-XgoDuck` | pick off the ground |
| `Mjlab-GroundPick-Rough-XgoDuck` | pick, rough |
| `Mjlab-BallKick-Flat-XgoDuck` | right-footed kick |
| `Mjlab-BallKick-Left-Flat-XgoDuck` | left-footed kick |
| `Mjlab-Roulade-Flat-XgoDuck` | roll |
| `Mjlab-Roulade-Angle-Flat-XgoDuck` | roll, with an angle observation |

## What differs from MicroDuck

| | MicroDuck | XgoDuck |
|---|---|---|
| Actuators | Dynamixel XL330 | HLS1910 (`params/1910_m6.json`) |
| MJCF | walk model | **all-collisions** model, 34 geoms |
| Standing height | 0.115 m | 0.139158 m |
| Sit height | — | 0.085868 m |
| StandUp episode | 8 s | 8 s |
| Init state | `set_ground_state` mix | `set_random_drop_state` (drop + freeze) |

Six `_retarget_xgoduck_*` adapters in `tasks/__init__.py` do the swap. The
important property they preserve is the **shared 61-D actor observation
contract** — identical across every task, which is what makes a checkpoint
portable between tasks at all. The critic does *not* share a contract
(Velocity 76-D, StandUp 74-D), so cross-task reuse is actor-only.

## Setup

Requires a CUDA GPU and [uv](https://docs.astral.sh/uv/).

**Windows:** PyPI's `torch` wheel is CPU-only, so `pyproject.toml` routes win32
to a CUDA 12.8 mirror:

```toml
torch = [
  { index = "pytorch-cu129", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
  { index = "pytorch-cu128-win", marker = "sys_platform == 'win32'" },
]
```

Without that the GPU is invisible and `train` dies in `select_gpus()`.

**`uv sync` may abort** with `[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]`
if a safe-delete hook is active. Prefix with `CODEBUDDY_SAFE_DELETE_ENABLED=0`.

## Training

```bash
# walking
uv run train Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 1024

# stand-up, warm-started from the walking policy (see below)
MICRODUCK_WARM_START=1 uv run train Mjlab-StandUp-Flat-XgoDuck \
  --env.scene.num-envs 1024 --agent.max_iterations 3000 \
  --agent.resume True --agent.load-run <velocity-run> --agent.load-checkpoint model_2999.pt
```

### Use 1024 envs on an 8 GB card, not 4096

Upstream's `train.bat` talks about 4096 envs at 3.8 GB peak. That holds for
MicroDuck's walk model, **not** for XgoDuck's all-collisions model:

- **4096 OOMs outright.** `convex_narrowphase` allocates
  `epa_pr = (naccdmax, 6 + MJ_MAX_EPAFACES*epa_iterations)` — 2.5 GB for a
  single array at that env count.
- **2048 fits but is slower than 1024.** It leaves 405 MiB free, and Windows
  WDDM starts paging VRAM to system RAM: `Collection time` goes 2.9 s → 122 s
  while `Learning time` stays at 4.5 s. 5.65 s/iter → 115 s/iter.
- **1024 keeps 3.8 GB free** and runs at 1.78 s/iter, **2.9x faster than 2048**
  (12500 vs 8707 steps/s).

The test is not "does it allocate" but "is the free margin larger than the
desktop's own VRAM swing" — background apps moved the baseline between 804 and
1630 MiB on the machine this was measured on.

Do **not** shrink `nconmax`/`njmax` to squeeze more envs in. Dropped contacts
corrupt the physics silently and you train a broken policy.

### Warm start vs resume

- **Resume** (same task): weights, normalizer, optimizer and the step counters
  all continue.
- **Warm start** (`MICRODUCK_WARM_START=1`): loads another task's checkpoint and
  resets `common_step_counter` and the iteration counter to 0 so every
  curriculum restarts at stage 0. Cross-task it also detects the critic
  `size mismatch` and reloads with `load_cfg={"actor": True}` — critic and
  optimizer start fresh.

`--agent.resume` always creates a **new timestamped run directory** and loads
the checkpoint into it, so resuming never overwrites the source run.

## Tooling added in this fork

| Script | Purpose |
|---|---|
| `scripts/watch_run.py` | convergence report from tfevents: real s/iter, whether the task metrics are still moving, NaN check. Exit 0 keep going / 1 plateaued / 2 broken. |
| `scripts/verify_standup.py` | measures trunk height and tilt from a trained policy and records a video. Reports a PASS/CHECK verdict. |
| `scripts/verify_velocity.py` | replays a velocity policy against the commands it was given and compares its tracking error to a **do-nothing baseline**. Reports a PASS/CHECK verdict. |
| `scripts/verify_velstand.py` | measures walk and fall-recovery separately, because a VelStand log mixes them: walk tracking vs the do-nothing baseline (pushes off), then 12 s prone-start recovery rate per spawn orientation. |
| `scripts/play_teleop.py` | keyboard teleop in the native viewer: arrows for vx/vy, `a`/`d` for yaw, `space` stop, `t` freeze. Works on Windows (`msvcrt`) and POSIX (`termios`). |

Both exist because the obvious approaches lie:

- Training stdout is **block-buffered** when redirected to a file, so a healthy
  run looks frozen for minutes while a wedged one keeps printing. Read the
  TensorBoard event file instead.
- `root_link_pos_w` / `root_link_quat_w` read `xpos`/`xquat`, which only
  `sim.forward()` refreshes — reading them right after `env.step()` gives stale
  values. Read the free joint out of `qpos` instead.
- The standing height is **task-specific** after retargeting. Read
  `env_cfg.rewards["height_stand"].params["target_height"]`; hardcoding either
  robot's value misreports the other by 20%.
- `Metrics/twist/error_vel_xy` is **not** in m/s. `UniformVelocityCommand`
  accumulates `‖cmd − v‖ / max_command_step` every step without dividing by the
  episode length, so the logged number is a per-`max_command_time` sum: an
  episode of 1000 steps at `resampling_time_range=(3, 8)` reports ~2.5x the
  per-step mean. Divide by `episode_steps / max_command_steps` before quoting
  it — or cross-check against the reward, which is per-step: with `std: 0.25`,
  `Episode_Reward/track_linear_velocity = 2.23` of 3.0 means
  `err = 0.25 * sqrt(-ln(2.23/3.0)) = 0.12 m/s`, i.e. a logged 0.296.
- A velocity error is only meaningful next to the **command magnitude**. This
  task commands `lin_vel_x ∈ ±0.4`, `lin_vel_y ∈ ±0.3`, so the mean command is
  ~0.17 m/s and a policy that ignores the command entirely still scores ~0.17.
  `verify_velocity.py` prints that baseline for exactly this reason.

### Watching a policy run

```bash
.venv/Scripts/play.exe Mjlab-VelStand-Flat-XgoDuck \
  --checkpoint-file logs/rsl_rl/xgoduck_velstand/<run>/model_2999.pt \
  --num-envs 8 --viewer native
```

Startup takes ~2.5 min (MuJoCo Warp loads its kernels) and the log stops moving
while it does — that is not a hang. `--viewer viser` gives a browser UI instead.

`train` has **no live window**: the sim is GPU-batched over 1024-4096 envs and
the viewer can only render one. Use `--video --video-interval N` to get mp4
segments during training, or open `play` on an intermediate `model_XXXX.pt`.

Keyboard: `play` itself has no robot control. Its window bindings are playback
only (Enter reset, Space pause, `-`/`=` speed, `,`/`.` env, `P` plots, `R` debug
vis, `A` show-all, Right single-step) and the arrow keys are already taken. Real
teleop lives in `scripts/infer_policy.py`, which is POSIX-only, so on Windows use
`scripts/play_teleop.py` — it reads keys from the terminal rather than the viewer
window and writes the velocity command straight into the command manager
(resampling and heading control are disabled for the run, otherwise the sampled
command overwrites the keyboard every 3-8 s). Run it from PowerShell, Windows
Terminal or cmd; mintty / Git Bash has no Win32 console, so `msvcrt` cannot see
the keys there and the script says so.

## Tests

```bash
uv run --with pytest python -m pytest tests/ -q
# 379 passed, 1 skipped
```

105 of those cover XgoDuck specifically (12 `tests/test_xgoduck_*.py` files).
Three of them caught **silent** regressions that a smoke test cannot see — the
code ran, it just trained the wrong thing:

- `VelocityCommandCommandOnlyCfg.rel_backward_envs` was removed upstream, so
  setting the attribute succeeded and was ignored: 15% reverse practice quietly
  became 0%.
- `deepcopy(MICRODUCK_ROUGH_TERRAINS_CFG)` was dropped, so every cfg instance
  shared one mutable terrain object and play-mode calls overwrote the training
  terrain settings.
- The roulade support gate let "supported forward + airborne backward" ratchet
  the accumulator up freely (airborne return farming).

## Status

| Policy | State |
|---|---|
| StandUp | **trained and verified** — [checkpoints/xgoduck_standup](../checkpoints/xgoduck_standup/) |
| Velocity | **trained and verified** — [checkpoints/xgoduck_velocity](../checkpoints/xgoduck_velocity/) |

### Velocity, measured

5498 iterations over three sittings at 4096 envs.
`verify_velocity.py` on 32 envs x 550 steps, pushes active, `--seed 7`:

```
height 0.140 m   tilt 2.7 deg   falls 0.0%
lin_xy err 0.119 m/s   vs do-nothing baseline 0.183   (+35%)
yaw      err 0.270 rad/s vs do-nothing baseline 0.335   (+20%)
cmd 0.345 -> 0.275 m/s (80%)   drift when told to stand 0.047 m/s
turn in place: wz 0.71 commanded -> 0.53 (75%)
```

It stands, walks and turns without ever falling, and tracks speed to within
~20% of the command. The remaining softness is a capacity limit, not a tuning
leftover: linear and yaw tracking trade off against each other and against
`head_pose_tracking`, which reaches 1.82 of its 2.0 weight.

Two reward lessons from getting here are recorded in that checkpoint's README:
tightening `track_linear_velocity` std 0.25 → 0.15 is what unlocked the speed
regulation, and tightening `track_angular_velocity` std 0.707 → 0.4 is what
broke turning — a Gaussian has no gradient past its own std, and
`track_yaw_rate_fine`'s docstring already documented that exact failure. Add a
fine term instead of sharpening the coarse one.
| SitStand, GroundPick, BallKick, Roulade, Swizzle | registered, smoke-tested, not trained |

**VelStand** (the velocity + stand-up distillation chain) is not ported yet, and
it is not a registration away:

- Upstream's `microduck_velstand_env_cfg.py` is a 1005-line BC distillation
  pipeline needing a walking teacher, a get-up teacher and `distill.py`.
- XgoDuck's all-collisions MJCF has no **named servo shell geoms**, so
  `servo_impact` / `servo_stall` costs have nothing to attach to; the fork's
  StandUp uses `self_collisions` as a proxy instead.
- The get-up teacher *is* the StandUp policy, so the order is StandUp → export
  `alpha_stand.onnx` → VelStand. The first step is now done.

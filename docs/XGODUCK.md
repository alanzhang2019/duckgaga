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
| `scripts/verify_roulade.py` | measures a roll in **degrees**: reads the same rotation accumulator the rewards read, from a standstill standing spawn, and compares against `model_0.pt` and a zero-action policy so a stumble cannot pass as a roll. |
| `scripts/play_teleop.py` | keyboard teleop in the native viewer: arrows for vx/vy, `a`/`d` for yaw, `space` stop, `r` roll (with `--roulade`), `t` freeze. Works on Windows (`msvcrt`) and POSIX (`termios`). |

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

The teleop writes the command to **every** env, while the viewer renders one at
a time (the `1/4` in its corner; `,`/`.` switches). Use `--num-envs 1` unless you
want a group.

It also defaults `--push none`, and that matters: `play` leaves `topple_push` at
the curriculum's **final** value (+-1.2 m/s of trunk velocity every 0.5-1 s), so
what it shows is a push-recovery demo, not a walk demo — the robot spends its
time being knocked over and getting back up, which reads as random tumbling.
Same checkpoint, 1500 steps, `--cmd 0 0 0`:

| `--push` | height | tilt | upright | travelled |
|---|---|---|---|---|
| `topple` (what `play` does) | 0.1174 m | 37.1 deg | 33% | 0.77 m |
| `none` (default) | 0.1392 m | 2.8 deg | 98% | 0.53 m |

and walking at `--cmd 0.3 0 0` with pushes off: 0.27 m/s achieved, 98% upright.
`--prone` starts every env on the floor so the get-up is visible (93% upright
over 10 s including the rise); it drops `so3_prob`, because a uniform random
orientation is past upside-down half the time and those take up to 7.6 s.

`--roulade <ckpt>` loads a **second** actor and binds `r` to it:

```bash
play_teleop.cmd "" "" --roulade checkpoints\xgoduck_roulade\model_5999.pt
```

Pressing `r` hands the action stream to the roll actor for
`--roulade-duration` seconds (default 2.5) with the velocity command forced to
zero, then hands it back. It is a policy switch, not a command: the roulade
task is episodic and has no phase input, so "start rolling" can only mean
"let the other actor drive for a while". That is legal because every XgoDuck
task shares the same 61-D actor observation, and because the roll was trained
with the twist command pinned to ±0.01 m/s — i.e. exactly the zero the switch
writes. Measured end to end: tilt 1° → 171° → 2° in 1.2 s, then standing at
0.7° for the rest of the window, then back to walking.

## Tests

```bash
PATH="$PWD/.venv/Scripts:$PATH" .venv/Scripts/python.exe -m pytest tests/ -q
# 397 passed, 1 skipped
```

`PATH` matters for one test: `test_hf_jobs_flag.py` asserts that `train` is
resolvable and is mjlab's, and an unactivated venv has no `train` on PATH, so
running the interpreter by absolute path alone reports a false failure.

161 of those cover XgoDuck and the teleop specifically (13
`tests/test_xgoduck_*.py` files plus `test_velstand_cfg.py` and
`test_play_teleop.py`). Three of them caught **silent** regressions that a
smoke test cannot see — the code ran, it just trained the wrong thing:

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
| VelStand | **trained and verified** — [checkpoints/xgoduck_velstand](../checkpoints/xgoduck_velstand/) |
| Roulade | **trained and verified** — [checkpoints/xgoduck_roulade](../checkpoints/xgoduck_roulade/) |
| SitStand, GroundPick, BallKick, Swizzle | registered, smoke-tested, not trained |

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

### VelStand, measured

The velocity recipe plus fall recovery, warm-started from the Velocity teacher,
3000 iterations at 1024 envs. `verify_velstand.py` splits the two skills and
measures each in a clean environment, 48 envs, `--seed 7`:

| | Velocity teacher | VelStand |
|---|---|---|
| lin_xy error | 0.116 m/s | 0.116 m/s |
| yaw error | 0.236 rad/s | 0.243 rad/s |
| upright steps | 98% | 100% |
| recovery from prone | 21% | **90%** (0.63 s mean) |

The rise skill is nearly free: identical linear tracking, 0.007 rad/s of yaw,
and it stands more cleanly than the teacher. Face-down recovers 100%, face-up
88%, side 83% — side is slowest because the first move is a quarter roll onto
the front. Full detail in that checkpoint's README.

Two things this task taught, both about measurement rather than training:

- `Metrics/twist/error_vel_xy` accumulates over the episode and is not divided
  by its length, so with `prone_init_prob` at 0.45 it measures floor time, not
  tracking. It read 0.617 for a policy that tracks to 0.116 m/s.
- `play` leaves `topple_push` at its curriculum final — 1.2 m/s of trunk
  velocity every 0.5–1 s. The Velocity teacher itself reads 0.080 m / 68.7° in
  that environment, so it is a push-tolerance demo, not a walk demo.
  `play_teleop.py` therefore defaults to `--push none`.

### Roulade, measured

One sitting, 6000 iterations at 2048 envs (2.7 s/iter, 4.1 GB), converged at
~2750. `verify_roulade.py` drives it from a standing spawn at standstill and
reads the rotation accumulator the rewards read, in degrees:

```
rotation     frontier 349 deg mean (min 333, max 365)   100% past 260 deg
head latch   100%   (head-top touched ground inside 20-170 deg)
final        height 0.1389 m (100% of nominal)   tilt 0.4 deg   standing 100%
time to gate 0.78 s mean        displacement 0.41 m
```

Against `model_0.pt` (9°) and zero actions (3°), so the reward shaped the roll
rather than a fall producing it. Pressing `r` in `play_teleop.py` hands the
action stream to this actor for 2.5 s: tilt 1° → 171° → 2° in 1.2 s, then
standing at 0.7° for the rest of the window, then back to the walk actor.

The `roulade_progress: 0.93` in the training log is the metric most likely to
be misread — it is a per-step average of paid radians over 2π, so 0.93 means
~0.73 rad/s × 7 s ≈ 294° per episode, not "93% of a roll". The conversion is
written out in that checkpoint's README.


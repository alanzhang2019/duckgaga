# XgoDuck Velocity policy

Trained walking policy for the XgoDuck biped, task
`Mjlab-Velocity-Flat-XgoDuck`.

## Files

| File | What it is |
|---|---|
| `model_2999.pt` | rsl_rl checkpoint after iteration 2999 (actor + critic + optimizer + normalizers) |
| `xgoduck_velocity.onnx` | actor exported for deployment — input `obs [1, 61]`, output `actions [1, 14]` |
| `params/agent.yaml` | the exact PPO hyperparameters used |
| `params/env.yaml` | the fully-resolved env config (rewards, events, curricula, terrain) |
| `train.diff` | code state at training time: base commit + the working-tree diff |

## How it was trained

```
task        Mjlab-Velocity-Flat-XgoDuck
envs        4096            (3.5 GB VRAM — Velocity's flat scene fits where StandUp's does not)
iterations  3000  @ 24 steps/env
wall clock  ~2.6 h          (3.1 s/iter, 31000 steps/s)
seed        42
GPU         RTX 4060 8 GB
warm start  XgoDuck StandUp policy, actor only
```

```bash
MICRODUCK_WARM_START=1 .venv/Scripts/python.exe -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096 \
  --agent.max_iterations 3000 --agent.save-interval 100 \
  --agent.resume True --agent.load-run <standup-run> \
  --agent.load-checkpoint model_2999.pt
```

The run was interrupted at iteration 2274 by the hourly monitor (it judged the
task metrics plateaued) and finished in a second sitting from `model_2200.pt`:

```bash
# NOTE: no MICRODUCK_WARM_START here — this is a same-task resume, and setting
# it would zero common_step_counter and restart every curriculum at stage 0.
.venv/Scripts/python.exe -u -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096 \
  --agent.max_iterations 800 --agent.save-interval 100 \
  --agent.resume True --agent.load-run <this-run> \
  --agent.load-checkpoint model_2200.pt
```

`--agent.max_iterations` is a **delta** on a resume: 800 here, not 3000, or the
run would have gone to 5200.

## Verification

`scripts/verify_velocity.py` replays the policy against the commands it was
given and reports the error next to a **do-nothing baseline** — the error a
policy that always outputs zero velocity would score on the same commands. 32
envs x 550 steps, pushes active:

```
height 0.141 m   tilt 2.4 deg   falls 0.0%

commands        lin_xy 0.173 m/s mean (max 0.444)   yaw 0.374 rad/s
tracking        lin_xy err 0.123 m/s                yaw err 0.330 rad/s
do-nothing      0.173                               0.374
gain            +29%                                +12%

cmd 0.00 -> 0.06 m/s (drift floor)   err 0.060
cmd 0.11 -> 0.14                     err 0.132
cmd 0.23 -> 0.18                     err 0.161
cmd 0.34 -> 0.27                     err 0.169
turn in place: wz 0.72 commanded -> 0.38 achieved, 0.12 m/s linear drift
```

Read: it stands, walks and heads in the commanded direction without ever
falling, but **regulates speed magnitude and yaw rate poorly** — 22% undershoot
at 0.34 m/s, 0.06 m/s of drift when told to stand, and half the commanded yaw
rate when turning in place.

### Do not quote `Metrics/twist/error_vel_xy` as m/s

The final logged value was 0.296, which looks like a 0.3 m/s error. It is not:
`UniformVelocityCommand._update_metrics` accumulates `‖cmd − v‖ / max_command_step`
every step **without** dividing by the episode length, so the number is a sum
over one `max_command_time` (8 s = 400 steps) while the episode is 1000 steps.
Divide by 2.5 and you get 0.12 m/s, matching the measurement above. The
per-step reward confirms it independently: with `std: 0.25`,
`Episode_Reward/track_linear_velocity = 2.23` of 3.0 gives
`err = 0.25 * sqrt(-ln(2.23/3.0)) = 0.12 m/s`.

### Why it plateaued there

The reward is satisfied long before the tracking is good:

- `track_linear_velocity` `std: 0.25` still pays **79%** at a 0.12 m/s error.
- `track_angular_velocity` `std: 0.707` still pays **49%** at a 0.6 rad/s error.
- `head_pose_tracking` (`std: 0.5`, weight 2.0) reached 1.82 of 2.0 — the policy
  is far better at pointing its head at a random target than at holding a
  commanded speed, and the head is a heavy payload on a 14-servo biped.

Tightening the two tracking stds to 0.15 / 0.4 and resuming ~1500 iterations
from this checkpoint is the cheap way to make it a better distillation teacher.

## Run it

```bash
# viewer
uv run play Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file checkpoints/xgoduck_velocity/model_2999.pt

# re-measure
.venv/Scripts/python.exe scripts/verify_velocity.py \
  --checkpoint checkpoints/xgoduck_velocity/model_2999.pt \
  --num-envs 32 --steps 600 --video
```

## Provenance and licensing

Weights trained by the author of this fork on this codebase. The code is
Apache-2.0; the XgoDuck meshes and MJCF are the author's. Note that the
MicroDuck STL assets that ship in this repo are **CC BY-NC-SA 4.0** (see
[LICENSE-MODELS.md](../../LICENSE-MODELS.md)) — this policy was not trained
against them, but if you redistribute this repository as a whole, that licence
travels with those files.

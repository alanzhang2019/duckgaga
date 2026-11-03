# XgoDuck Velocity policy

Trained walking policy for the XgoDuck biped, task
`Mjlab-Velocity-Flat-XgoDuck`.

## Files

| File | What it is |
|---|---|
| `model_5497.pt` | rsl_rl checkpoint after iteration 5497 (actor + critic + optimizer + normalizers) |
| `xgoduck_velocity.onnx` | actor exported for deployment — input `obs [1, 61]`, output `actions [1, 14]` |
| `params/agent.yaml` | the exact PPO hyperparameters used |
| `params/env.yaml` | the fully-resolved env config (rewards, events, curricula, terrain) |
| `train.diff` | code state at training time: base commit + the working-tree diff |

## How it was trained

Three sittings, 5498 iterations in total, all at 4096 envs (3.1 s/iter, 3.5 GB
VRAM — Velocity's flat scene fits where StandUp's does not):

| Sitting | Iters | Reward change | Wall clock |
|---|---|---|---|
| 1 | 0 → 2999 | stock recipe, warm start from StandUp (actor only) | ~2.6 h |
| 2 | 2999 → 4498 | `track_linear_velocity` std 0.25 → 0.15, `track_angular_velocity` std 0.707 → 0.4 | 1 h 19 min |
| 3 | 4498 → 5497 | angular std back to 0.707, added `track_yaw_fine` (std 0.35, w 2.0) | 51 min |

```bash
# sitting 1: cross-task warm start, so MICRODUCK_WARM_START is what you want
MICRODUCK_WARM_START=1 .venv/Scripts/python.exe -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096 \
  --agent.max_iterations 3000 --agent.save-interval 100 \
  --agent.resume True --agent.load-run <standup-run> \
  --agent.load-checkpoint model_2999.pt

# sittings 2 and 3: same-task resume, so do NOT set it — it would zero
# common_step_counter and restart every curriculum at stage 0.
.venv/Scripts/python.exe -u -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-XgoDuck --env.scene.num-envs 4096 \
  --agent.max_iterations <N> --agent.save-interval 100 \
  --agent.resume True --agent.load-run <previous-run> \
  --agent.load-checkpoint <previous-checkpoint>
```

`--agent.max_iterations` is a **delta** on a resume: 1000 for sitting 3, not
5497, or the run would have gone to 6497.

## Verification

`scripts/verify_velocity.py` replays the policy against the commands it was
given and reports the error next to a **do-nothing baseline** — the error a
policy that always outputs zero velocity would score on the same commands.
32 envs x 550 steps, pushes active, `--seed 7` so the three checkpoints see
identical commands and identical startup randomization:

| | sitting 1 | sitting 2 | **sitting 3 (shipped)** |
|---|---|---|---|
| lin_xy error | 0.124 m/s | 0.103 | **0.119** |
| lin_xy gain over do-nothing (0.183) | +32% | +44% | **+35%** |
| yaw error | 0.311 rad/s | 0.413 | **0.270** |
| yaw gain over do-nothing (0.335) | +7% | −23% | **+20%** |
| turn in place, wz 0.71 commanded | 0.38 (54%) | 0.20 (28%) | **0.53 (75%)** |
| drift when told to stand | 0.055 m/s | 0.039 | **0.047** |
| cmd 0.345 m/s -> achieved | 0.273 (79%) | 0.298 (86%) | **0.275 (80%)** |
| height / tilt / falls | 0.1407 m / 2.1° / 0% | 0.1394 / 2.8° / 0% | **0.1400 / 2.7° / 0%** |

Sitting 3 dominates sitting 1 on every axis, so it is the one shipped. Sitting 2
had the best linear tracking but broke turning.

### Why the linear std fix worked and the yaw one backfired

The stock recipe's `track_linear_velocity` `std: 0.25` still pays **79%** at the
0.12 m/s error sitting 1 plateaued on, so the gradient died there. Tightening it
to 0.15 puts that same error at 53% and the policy kept pushing — that is where
sitting 2's +44% came from.

The same move on yaw does the opposite, because a Gaussian has **no gradient
past its own std** and this policy's yaw error sits around 0.5 rad/s. At
`std: 0.4` the term simply stopped paying for turning, and turn-in-place
collapsed from 54% to 28%. `mdp.track_yaw_rate_fine`'s docstring records the
identical failure from 2026-10-01 (a std-0.35 replacement took 1.0 rad/s
tracking from 0.42 to 0.16) — **check that docstring before touching a std**.
The fix is the documented one: keep the loose 0.707 for the far-error gradient
and add `track_yaw_fine` to price small errors, which is also what
`microduck_velstand_env_cfg.py` does under `ENABLE_YAW_FIX`.

Linear and yaw compete for the same capacity on this robot, so the two gains
trade off: sitting 3 gave back most of the linear gain to buy 75% turn tracking.
The residual softness is the policy's capacity limit under this recipe (14
servos, a heavy head that must also point at a random target, a push every
0.5-1 s), not a tuning mistake left on the table.

### Do not quote `Metrics/twist/error_vel_xy` as m/s

The final logged value was 0.2885, which looks like a 0.29 m/s error. It is not:
`UniformVelocityCommand._update_metrics` accumulates `‖cmd − v‖ / max_command_step`
every step **without** dividing by the episode length, so the number is a sum
over one `max_command_time` (8 s = 400 steps) while the episode is 1000 steps.
Divide by 2.5 and you get 0.115 m/s, matching the measurement above. The
per-step reward confirms it independently: with `std: 0.15`,
`Episode_Reward/track_linear_velocity = 1.745` of 3.0 gives
`err = 0.15 * sqrt(-ln(1.745/3.0)) = 0.12 m/s`.

## Run it

```bash
# viewer
uv run play Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file checkpoints/xgoduck_velocity/model_5497.pt

# re-measure
.venv/Scripts/python.exe scripts/verify_velocity.py \
  --checkpoint checkpoints/xgoduck_velocity/model_5497.pt \
  --num-envs 32 --steps 600 --seed 7 --video
```

`--seed` fixes both the command stream and the startup domain randomization. Use
it whenever two checkpoints are being compared, or ~5% sampling noise will
swamp the difference.

## Provenance and licensing

Weights trained by the author of this fork on this codebase. The code is
Apache-2.0; the XgoDuck meshes and MJCF are the author's. Note that the
MicroDuck STL assets that ship in this repo are **CC BY-NC-SA 4.0** (see
[LICENSE-MODELS.md](../../LICENSE-MODELS.md)) — this policy was not trained
against them, but if you redistribute this repository as a whole, that licence
travels with those files.

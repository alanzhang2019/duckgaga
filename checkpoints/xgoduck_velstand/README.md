# XgoDuck VelStand policy

Walk **and** get up, in one actor. Task `Mjlab-VelStand-Flat-XgoDuck`.

This is the walking policy that also knows how to recover: it walks and turns
about as well as the Velocity teacher, and on top of that it gets itself off the
floor from any resting orientation. It is the default checkpoint for
`play_teleop.cmd`, and the walk actor on the other side of the roulade
keyboard switch.

## Files

| File | What it is |
|---|---|
| `model_2999.pt` | rsl_rl checkpoint after iteration 2999 (actor + critic + optimizer + normalizers) |
| `xgoduck_velstand.onnx` | actor exported for deployment — input `obs [1, 61]`, output `actions [1, 14]` |
| `params/agent.yaml` | the exact PPO hyperparameters used |
| `params/env.yaml` | the fully-resolved env config (rewards, events, curricula) |
| `train.diff` | code state at training time: base commit + the command line |

Same 61-D actor observation contract as every other XgoDuck task, so this
checkpoint and the Velocity / Roulade actors are interchangeable in a runner.

## How it was trained

One sitting, 3000 iterations at 1024 envs, 1 h 25 min, **same-task warm start**
from the Velocity teacher (`model_5497.pt`) with `MICRODUCK_WARM_START` unset —
so weights, obs normalizer and optimizer carry over and the step counter keeps
running, which leaves the inherited velocity curricula at their final stage
instead of re-running them from zero.

1024 envs is not a preference. XgoDuck's all-collisions MJCF needs 2.5 GB of
EPA convex buffers at 4096, and 2048 on an 8 GB card trips WDDM paging
(7719/8188 MiB, 40 s/iter). 1024 sits at 4479 MiB / 1.6 s/iter.

## The training log cannot be read directly

Recorded here because the numbers in `logs/train_velstand_1024.log` look like a
collapse and are not one:

| iter | mean episode length | mean reward | error_vel_xy | error_vel_yaw |
|---|---|---|---|---|
| 300 | 996 | +71.2 | 0.463 | 2.169 |
| 1500 | 1000 | −25.6 | 0.641 | 3.594 |
| 2999 | 1000 | −27.3 | 0.617 | 3.750 |

Mean reward falls by 98 over the run and `error_vel_xy` rises. Both are
artifacts:

- The reward sum is not comparable across iterations, because the curricula
  keep changing the objective — protection costs, prone spawns and topple
  pushes all arrive partway through, and each one is a cost that did not exist
  at iteration 300.
- `Metrics/twist/error_vel_xy` accumulates over the **whole** episode and is not
  divided by episode length, so it mostly measures "how long did it spend on
  the floor". With `prone_init_prob` up to 0.45 and a 20 s episode cap that is
  a floor-time metric wearing a tracking metric's name. See
  `docs/XGODUCK.md` and the `verify_velstand.py` docstring.
- `Episode_Termination/fallen_too_long == 0` does not mean "never falls"; it
  means "never stays down for 8 s".

## Verification

`scripts/verify_velstand.py` splits the checkpoint into its two skills and
measures each in a clean environment: the walk phase removes `topple_push`
(which `play` leaves at its curriculum final, 1.2 m/s of trunk velocity every
0.5–1 s) and starts standing; the recovery phase sets `prone_prob = 1.0`,
removes `push_robot`, and measures how long each env takes to become upright
**and stay upright**. 48 envs, `--seed 7`, 550-step walk / 600-step recovery.

| | Velocity teacher `model_5497` | **VelStand `model_2999`** |
|---|---|---|
| lin_xy error | 0.116 m/s | **0.116 m/s** |
| lin_xy gain over do-nothing (0.172) | +32% | **+32%** |
| yaw error | 0.236 rad/s | **0.243 rad/s** |
| yaw gain over do-nothing (0.263) | +10% | **+8%** |
| height / tilt / upright steps | 0.1380 m / 4.3° / 98% | **0.1393 m / 3.0° / 100%** |
| **recovery from prone** | 21% | **90%** |

So the rise skill is close to free for walking — identical linear tracking, and
0.007 rad/s of yaw (two points of gain) — while buying 69 points of recovery.
It also stands *more* cleanly than the teacher it came from (100% upright steps
vs 98%, 3.0° vs 4.3°). Run-to-run jitter on these gains is ±1 point, so read
the errors, not the percentages.

Recovery detail, 43/48 envs, 0.63 s mean, 1.66 s worst:

| spawn orientation | n | recovered | mean time |
|---|---|---|---|
| face down | 16 | 100% | 0.64 s |
| face up | 8 | 88% | 2.14 s |
| side | 24 | 83% | 2.49 s |

Face-down is the easy case (it can just push up); side is the slow one because
the first move is a quarter roll onto the front. The teacher scores 38% / 0% /
17% on the same battery, which is what "the walking policy has no recovery
skill at all" looks like in numbers.

`play_teleop.py --prone` is the interactive version of this test: every env
starts on the floor and you watch it get up.

# XgoDuck Roulade policy

Forward roll (前滚翻) policy for the XgoDuck biped, task
`Mjlab-Roulade-Flat-XgoDuck`. From a standstill it rolls over its own head and
comes back up standing — one full 360° rotation in about 1.2 s.

## Files

| File | What it is |
|---|---|
| `model_5999.pt` | rsl_rl checkpoint after iteration 5999 (actor + critic + optimizer + normalizers) |
| `xgoduck_roulade.onnx` | actor exported for deployment — input `obs [1, 61]`, output `actions [1, 14]` |
| `params/agent.yaml` | the exact PPO hyperparameters used |
| `params/env.yaml` | the fully-resolved env config (rewards, events, curricula) |
| `train.diff` | code state at training time: base commit + the command line |

The observation layout is the shared 61-D actor contract, so this actor drops
into the same slot as the walking one — which is what makes the keyboard
trigger in `scripts/play_teleop.py` a policy *switch* rather than a task swap.

## How it was trained

One sitting, 6000 iterations at 2048 envs, 2.72 s/iter, 4121 MiB, 4 h 23 min,
no warm start (fresh task — there is no teacher to inherit from).

It converged at **~2750 iterations** and then sat flat for the remaining 3250:

| iter | progress | landing_composite | landing_sharp | upright_after_roll |
|---|---|---|---|---|
| 0 | 0.021 | 0.000 | 0.000 | 0.000 |
| 1000 | 1.038 | 0.000 | 0.000 | 0.035 |
| 2000 | 0.969 | 0.387 | 0.290 | 1.119 |
| 2750 | 0.990 | 2.730 | 1.507 | 1.280 |
| 4000 | 0.928 | 3.268 | 1.648 | 1.301 |
| **5999** | **0.934** | **3.315** | **1.672** | **1.305** |

The `landing_*` and `upright_after_roll` numbers are the ones that mattered:
they are gated on the rotation accumulator crossing 260°, so they only move
once the robot is genuinely going over the top and standing back up.

## Reading `roulade_progress: 0.93` correctly

This is the one metric that is easy to misread by a factor of six, and it is
worth writing down because it looks like "93% of a roll" and is not.

`roulade_progress` returns `Δaccum / (step_dt · 2π)`, and `Episode_Reward/*`
is a per-step average of that. So:

```
paid rotation rate = reward / weight · 2π = 0.934 / 8 · 2π ≈ 0.73 rad/s
total rotation     ≈ 0.73 rad/s · 7 s episode ≈ 5.1 rad ≈ 294°
```

294° is inside the 260–330° landing window, i.e. the policy was completing the
roll — not 93% of one. The honest reading is "≈82% of a full turn per episode,
measured over the whole episode including the standing-up tail", which is why
the number never reaches 1.0 even for a perfect policy.

## Verification

`scripts/verify_roulade.py` drives the checkpoint from a **standing spawn at
standstill** and reads the same accumulator the rewards read
(`env._roulade_max`: forward-only, support-gated, flatness-gated), then reports
it in degrees. 64 envs × 400 steps (8 s), `--seed 7`:

```
rotation     frontier 349 deg mean   (min 333, max 365)
             past 260 deg: 100%   past 330 deg (full roll): 100%
             time to gate 0.78 s mean, 0.88 s worst
head latch   100% of envs (head-top touched ground inside 20-170 deg)
final        height 0.1389 m (100% of 0.1392)   tilt 0.4 deg   standing 100%
displacement 0.41 m mean
trace        env 18 frontier every 25 steps: 3 168 318 350 350 ...
             env 18 tilt:                    6 169  32   0   0 ...
```

Three separate comparisons, because a roll number on its own proves nothing:

| | frontier | past 260° |
|---|---|---|
| **this policy** | **349°** | **100%** |
| `model_0.pt` (untrained net) | 9° | 0% |
| zero actions | 3° | 0% |

The untrained net does not roll, so the reward shaped the behaviour rather
than a fall doing it. A different seed and a 0.2 m/s running start gives 344°
/ 98%, so it is not seed-luck.

The `trace` line is the part that rules out a stumble: tilt goes 6° → 169°
(inverted) → 32° → 0° and stays there. One continuous rotation, then standing.
`head latch 100%` is the second half of that argument — the latch requires the
**flat top** of the head to touch the ground while the accumulator is inside
20–170°, which a shoulder roll or a face-plant cannot produce.

## Triggering it by keyboard

`r` in `scripts/play_teleop.py` hands the action stream to this actor for
`--roulade-duration` seconds (default 2.5 s) with the velocity command forced to
zero, then hands it back to the walk actor:

```bash
play_teleop.cmd "" "" --roulade checkpoints/xgoduck_roulade/model_5999.pt
```

Measured end to end, standing spawn, walk actor = VelStand `model_2999`:

```
before   z=0.1401  tilt=  0.9 deg
step  10 z=0.1392  tilt= 44.8
step  30 z=0.1209  tilt=171.2   <- inverted
step  60 z=0.1278  tilt=  2.2   <- back on its feet
step 125 z=0.1390  tilt=  0.7   <- roll handoff, walk actor takes over
after    z=0.1394  tilt=  1.0   <- and stays standing
```

2.5 s is deliberately longer than the 1.2 s the roll needs: the roulade actor
is episodic and never learned to *stop* standing, so the extra time is spent
standing still inside the roll policy, which is harmless. Cut it to 1.5 s if
you want the handoff to look tighter.

# XgoDuck StandUp policy

Trained stand-up (fall recovery) policy for the XgoDuck biped, task
`Mjlab-StandUp-Flat-XgoDuck`.

## Files

| File | What it is |
|---|---|
| `model_2999.pt` | rsl_rl checkpoint after iteration 2999 (actor + critic + optimizer + normalizers) |
| `xgoduck_standup.onnx` | actor exported for deployment — input `obs [1, 61]`, output `actions [1, 14]` |
| `params/agent.yaml` | the exact PPO hyperparameters used |
| `params/env.yaml` | the fully-resolved env config (rewards, events, curricula, terrain) |
| `train.diff` | code state at training time: base commit + the working-tree diff |

The ONNX is the artifact the runtime consumes. `passive_*` joints are filtered
out of the export (mdp Patch 4), which is why the output is 14 wide and not 21.

## How it was trained

```
task        Mjlab-StandUp-Flat-XgoDuck
envs        1024            (2048 and 4096 both fail on an 8 GB card, see below)
iterations  3000  @ 24 steps/env
wall clock  1 h 28 min      (1.78 s/iter steady, 12500 steps/s)
seed        42
GPU         RTX 4060 8 GB, 4.4 GB peak
warm start  XgoDuck Velocity policy, actor only
```

```bash
MICRODUCK_WARM_START=1 .venv/Scripts/python.exe -m mjlab.scripts.train \
  Mjlab-StandUp-Flat-XgoDuck --env.scene.num-envs 1024 \
  --agent.max_iterations 3000 --agent.save-interval 100 \
  --agent.resume True --agent.load-run <velocity-run> \
  --agent.load-checkpoint model_2999.pt
```

**Warm start is cross-task, so the critic does not fit.** The actor's 61-D
observation contract is shared by every task, but the critic sees
task-specific privileged observations (Velocity 76-D vs StandUp 74-D), so
`MICRODUCK_WARM_START=1` catches the `size mismatch`, reloads with
`load_cfg={"actor": True}`, and resets `common_step_counter` to 0 so the
curricula restart at stage 0. The critic and optimizer start fresh — which is
what a cross-task warm start wants anyway.

**1024 envs, not 4096.** XgoDuck uses the all-collisions MJCF (34 geoms), so
`naccdmax` explodes and `convex_narrowphase` asks for a 2.5 GB `epa_pr` array:
4096 OOMs outright, and 2048 fits in name only — it leaves 405 MiB free, and
Windows WDDM then pages VRAM to system RAM, collapsing the sim from 5.65 s/iter
to 115 s/iter. At 1024 the run keeps 3.8 GB of headroom and is **2.9x faster**
than 2048 (12500 vs 8707 steps/s). Do not shrink `nconmax`/`njmax` to make more
envs fit: dropped contacts corrupt the physics silently.

## Verification

`scripts/verify_standup.py` steps the policy in play mode and measures the pose
directly, because a rising `standing_composite` is not proof of standing — it is
a product of sub-terms and jumps superlinearly once the last one lands.

```
target height  0.1392 m   (XGODUCK_STAND_Z, read from the reward cfg)
settled z      0.1385 m = 99.5% of target   (worst env 99.0%)
settled tilt   2.3 deg    [0.5 .. 5.3]
standing       16/16 envs, 100% of settled steps
height swing   0.01 cm/step
get-up         ~110 steps = 2.2 s, from any face-up / face-down start
```

Final training metrics: `standing_composite` 2.39, `upright_sharp` 1.03,
`height_stand_sharp` 0.71, `Episode_Termination/time_out` 2.92, no NaN.

## Run it

```bash
# viewer
uv run play Mjlab-StandUp-Flat-XgoDuck \
  --checkpoint-file checkpoints/xgoduck_standup/model_2999.pt

# re-measure
.venv/Scripts/python.exe scripts/verify_standup.py \
  --checkpoint checkpoints/xgoduck_standup/model_2999.pt \
  --num-envs 16 --steps 380 --tail 120 --video
```

Episode length is 8 s = 400 steps. Keep `--steps` under that or every env resets
mid-rollout and the settled window straddles the reset.

## Provenance and licensing

Weights trained by the author of this fork on this codebase. The code is
Apache-2.0; the XgoDuck meshes and MJCF are the author's. Note that the
MicroDuck STL assets that ship in this repo are **CC BY-NC-SA 4.0** (see
[LICENSE-MODELS.md](../../LICENSE-MODELS.md)) — this policy was not trained
against them, but if you redistribute this repository as a whole, that licence
travels with those files.

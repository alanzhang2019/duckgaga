# checkpoints

Trained policies that ship with the repo, so a clone can run `play` and
`scripts/verify_standup.py` without retraining.

These are the **final** checkpoints only. Training writes one every 100
iterations into `logs/rsl_rl/`, which is gitignored — a full 3000-iteration run
leaves 31 files and ~150 MB, so keeping the whole ladder in git is not worth it.
If you need the intermediate ladder, retrain or use a release asset.

| Policy | Task | Status |
|---|---|---|
| [xgoduck_standup](xgoduck_standup/) | `Mjlab-StandUp-Flat-XgoDuck` | verified — stands 99.5% of target height, 2.3 deg tilt |
| [xgoduck_velocity](xgoduck_velocity/) | `Mjlab-Velocity-Flat-XgoDuck` | verified — +35% over do-nothing on lin_xy, +20% on yaw |
| [xgoduck_velstand](xgoduck_velstand/) | `Mjlab-VelStand-Flat-XgoDuck` | verified — walks like the teacher, recovers from prone 90% |
| [xgoduck_roulade](xgoduck_roulade/) | `Mjlab-Roulade-Flat-XgoDuck` | verified — 349 deg mean rotation, lands standing 100% |

## Why plain git and not Git LFS

`model_2999.pt` is 4.8 MB and the ONNX 0.8 MB, both far under GitHub's 50 MB
warning threshold. LFS would add a client-side dependency and a bandwidth quota
for no benefit at this size.

Revisit that if a single file crosses ~40 MB. At that point:

```bash
git lfs install
git lfs track "checkpoints/**/*.pt"
```

## Layout convention

```
checkpoints/<policy>/
  model_<N>.pt        rsl_rl checkpoint (actor + critic + optimizer)
  <policy>.onnx       actor only, exported for deployment
  params/agent.yaml   PPO hyperparameters
  params/env.yaml     fully-resolved env config
  train.diff          code state at training time
  README.md           how it was trained, and how it was verified
```

Keep the `.onnx` and `params/` with every checkpoint: a `.pt` alone cannot be
reproduced, and the env config is the only record of what the reward actually
was after the XgoDuck retarget rewrote it.

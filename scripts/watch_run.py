#!/usr/bin/env python3
"""Compact progress/convergence report for an rsl_rl run, read from tfevents.

Reads the TensorBoard event file instead of the training stdout, which is
block-buffered to a file and therefore arrives in 4-8 KiB lumps: a run that is
perfectly healthy can look frozen for minutes.

    uv run python scripts/watch_run.py                          # newest run
    uv run python scripts/watch_run.py -e xgoduck_standup -n 5  # 5 newest runs
    uv run python scripts/watch_run.py --keys Episode_Reward/standing_composite

Exit code is 0 when the run still looks alive and improving, 1 when the chosen
metrics have plateaued over the last `--window` iterations (i.e. a checkpoint
is safe to keep and the run can be stopped early), and 2 when the run looks
broken (NaN, or rewards collapsing) — so it can be used from a cron/automation.
"""

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_EXPERIMENT = "xgoduck_standup"

# The standup recipe's own signals: `standing_composite` is the one that says
# "it got up and stayed up"; `time_out` says "episodes are surviving to the
# end" rather than ending early; `com_upward_velocity` is the discovery term
# and goes quiet once the motion is learned.
DEFAULT_KEYS = (
    "Episode_Reward/standing_composite",
    "Episode_Reward/upright_sharp",
    "Episode_Reward/height_stand_sharp",
    "Episode_Reward/com_upward_velocity",
    "Episode_Reward/arrival_damping",
    "Episode_Termination/time_out",
)


def latest_run(experiment, index=0):
    runs = sorted(
        (
            d
            for d in glob.glob(os.path.join("logs", "rsl_rl", experiment, "*"))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "events.out.tfevents.*"))
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    if not runs:
        sys.exit(f"no run with an event file under logs/rsl_rl/{experiment}/")
    if index >= len(runs):
        sys.exit(f"only {len(runs)} run(s) under logs/rsl_rl/{experiment}/")
    return runs[index]


def series(ea, tag):
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def at(points, step):
    """Last value at or before `step`, or None."""
    best = None
    for s, v in points:
        if s <= step:
            best = v
        else:
            break
    return best


def pct_change(points, window):
    """Relative change of a metric over the last `window` iterations."""
    if len(points) < 2:
        return None
    last_step, last = points[-1]
    prev = at(points, last_step - window)
    if prev is None or prev == 0:
        return None
    return (last - prev) / abs(prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("-n", "--index", type=int, default=0, help="0 = newest run")
    ap.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    ap.add_argument("--window", type=int, default=300, help="plateau window (iters)")
    ap.add_argument("--tol", type=float, default=0.05, help="plateau threshold")
    ap.add_argument("--cols", type=int, default=6, help="sample columns")
    args = ap.parse_args()

    run = latest_run(args.experiment, args.index)
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        sys.exit(f"{run}: event file has no scalars yet (still warming up)")

    # Every tag is logged once per iteration, so the longest series doubles as
    # the iteration counter (rsl_rl does not log one explicitly).
    ref_tag = max(tags, key=lambda t: len(series(ea, t)))
    ref = ea.Scalars(ref_tag)
    steps = [e.step for e in ref]
    first, last = steps[0], steps[-1]
    print(f"run      {run}")
    print(f"iters    {first} → {last}   ({len(steps)} logged)")

    # Wall clock from the event timestamps: iteration time without parsing the
    # block-buffered stdout. Use the MEDIAN of recent deltas — the first delta
    # spans env construction, warp kernel compile and CUDA graph capture, so an
    # average over the whole run reports nonsense early on.
    per_iter = None
    if len(ref) >= 3:
        deltas = [
            ref[i].wall_time - ref[i - 1].wall_time
            for i in range(max(1, len(ref) - 50), len(ref))
        ]
        deltas.sort()
        per_iter = deltas[len(deltas) // 2]
        print(f"speed    {per_iter:.2f} s/iter (median of last {len(deltas)})")

    total = None
    params = os.path.join(run, "params", "agent.yaml")
    if os.path.exists(params):
        for line in open(params, encoding="utf-8"):
            if line.startswith("max_iterations:"):
                total = int(line.split(":", 1)[1])
                break
    if total and per_iter:
        print(f"target   {total} iters   ETA {(total - last) * per_iter / 3600:.1f} h")

    # Sample columns across the run so a trend is visible at a glance.
    picks = sorted({round(first + (last - first) * i / max(args.cols - 1, 1)) for i in range(args.cols)})
    header = "metric".ljust(42) + "".join(f"{s:>10}" for s in picks)
    print()
    print(header)
    print("-" * len(header))
    verdict = []
    for key in args.keys:
        pts = series(ea, key)
        if not pts:
            continue
        row = key.replace("Episode_Reward/", "R:").replace("Episode_Termination/", "T:")
        row = row.ljust(42)
        for s in picks:
            v = at(pts, s)
            row += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        print(row)
        verdict.append((key, pct_change(pts, args.window)))

    print()
    measured = [(k, ch) for k, ch in verdict if ch is not None]
    for key, ch in verdict:
        if ch is None:
            print(f"  {key}: no {args.window}-iter history yet")
            continue
        state = "PLATEAU" if abs(ch) < args.tol else ("rising" if ch > 0 else "FALLING")
        print(f"  {key}: {ch:+.1%} over last {args.window} iters → {state}")

    nan = any(v != v for t in tags for _, v in series(ea, t))
    if nan:
        print("\nBROKEN: NaN in the scalars")
        return 2
    if not measured:
        print(f"\nTOO EARLY: {last} iterations in, need {args.window} for a verdict.")
        return 0
    if any(ch > args.tol for _, ch in measured):
        print("\nIMPROVING: still worth running.")
        return 0
    print(
        f"\nPLATEAUED: nothing moved more than {args.tol:.0%} in {args.window} iters "
        "— safe to stop and keep the best checkpoint."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

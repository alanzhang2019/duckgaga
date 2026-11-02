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

# Per-task signal preferences, tried in order; the first tuple whose leading
# entry exists in the run wins. Keeping them here (rather than defaulting to one
# task's list) is what stops the report from showing a single `time_out` column
# after you switch tasks.
KEY_PREFERENCES = (
    # standup: `standing_composite` is the one that says "it got up and stayed
    # up"; `time_out` says episodes survive to the end rather than ending early.
    (
        "Episode_Reward/standing_composite",
        "Episode_Reward/upright_sharp",
        "Episode_Reward/height_stand_sharp",
        "Episode_Reward/com_upward_velocity",
        "Episode_Termination/time_out",
    ),
    # velocity / velstand: tracking rewards up, tracking error down, and falls
    # are the thing that must go to zero.
    (
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity",
        "Metrics/twist/error_vel_xy",
        "Episode_Termination/fell_over",
    ),
)

# Metrics where DOWN is better. Without this the verdict reads a falling
# tracking error as "not improving" and calls a healthy run plateaued.
LOWER_IS_BETTER = ("Metrics/", "Episode_Termination/")


def latest_run(experiment=None, index=0):
    """Newest run by EVENT-FILE mtime, across all experiments when unspecified.

    Two things this deliberately does not do:
      - default to one hardcoded experiment. Whichever task is training right
        now is the one you want, and after switching tasks the old default
        silently reported a finished run as if it were the live one.
      - key on the run directory's mtime. A directory's mtime only moves when
        an entry is added or removed, so a run that is actively appending to
        its event file but not yet checkpointing looks stale.
    """
    pattern = os.path.join("logs", "rsl_rl", experiment or "*", "*")
    runs = []
    for d in glob.glob(pattern):
        if not os.path.isdir(d):
            continue
        events = glob.glob(os.path.join(d, "events.out.tfevents.*"))
        if events:
            runs.append((max(os.path.getmtime(e) for e in events), d))
    runs.sort(reverse=True)
    if not runs:
        where = f"logs/rsl_rl/{experiment}/" if experiment else "logs/rsl_rl/"
        sys.exit(f"no run with an event file under {where}")
    if index >= len(runs):
        sys.exit(f"only {len(runs)} run(s) under {pattern}")
    return runs[index][1]


def series(ea, tag):
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def pick_keys(ea):
    """Task-appropriate signals, without being told which task this is.

    Prefers a known recipe; for an unknown task falls back to the reward terms
    that actually moved, which is usually where the learning is happening.
    """
    tags = set(ea.Tags()["scalars"])
    for pref in KEY_PREFERENCES:
        if pref[0] in tags:
            return [k for k in pref if k in tags]
    scored = []
    for t in tags:
        if not t.startswith("Episode_Reward/"):
            continue
        pts = series(ea, t)
        if len(pts) < 2:
            continue
        a, b = pts[0][1], pts[-1][1]
        scored.append((abs(b - a) / max(abs(a), abs(b), 1e-6), t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:5]]


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
    ap.add_argument(
        "-e", "--experiment", default=None,
        help="experiment name; omit to pick the newest run across ALL experiments",
    )
    ap.add_argument("-n", "--index", type=int, default=0, help="0 = newest run")
    ap.add_argument(
        "--keys", nargs="*", default=None,
        help="explicit metric names; omit to auto-pick for the task",
    )
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

    keys = args.keys or pick_keys(ea)

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
        # On a resume, rsl_rl treats `max_iterations` as "N MORE iterations", so the
        # real end is first + N (a run started at 2200 with max_iterations 800 ends
        # at 3000). A fresh run has first == 0 and this reduces to `total`.
        end = first + total if first else total
        if end > last:
            print(f"target   {end} iters   ETA {(end - last) * per_iter / 3600:.1f} h")
        else:
            print(f"target   {end} iters   reached ({last})")

    # Sample columns across the run so a trend is visible at a glance.
    picks = sorted({round(first + (last - first) * i / max(args.cols - 1, 1)) for i in range(args.cols)})
    header = "metric".ljust(42) + "".join(f"{s:>10}" for s in picks)
    print()
    print(header)
    print("-" * len(header))
    verdict = []
    for key in keys:
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
    # Signed so "up" always means "better": a falling tracking error is
    # progress, not a stall, and `fell_over` going down is the whole point.
    measured = [
        (k, -ch if k.startswith(LOWER_IS_BETTER) else ch)
        for k, ch in verdict
        if ch is not None
    ]
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

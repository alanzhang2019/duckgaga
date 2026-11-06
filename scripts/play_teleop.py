"""Drive a trained policy with the keyboard, in the native MuJoCo viewer.

Why this exists rather than `play --viewer native` plus something:

  * The native viewer already consumes the arrow keys. Its built-in bindings
    (Enter reset, Space pause, `-`/`=` speed, `,`/`.` env, `P` plots, `R` debug
    vis, `A` show-all, Right single-step) are wired inside
    ``mjlab.viewer.native.viewer``, and the arrow keys are spoken for.
  * ``scripts/infer_policy.py`` does have a full keyboard teleop, but its reader
    is POSIX-only (``termios`` + ``tty`` cbreak) and is explicitly disabled on
    Windows; it is also a CPU/MicroDuck rehearsal path with its own XML and ONNX
    wiring.

So this script reads keys from the TERMINAL (not the viewer window) and writes
the velocity command straight into the command manager, one step ahead of the
policy. It works on Windows via ``msvcrt`` and on POSIX via ``termios``.

Command resampling is turned off for the run (otherwise the sampled command
would fight the keyboard every 3-8 s), and the head / body pose commands are
held at zero so the robot does not wander while you are steering.

Usage:
  .venv/Scripts/python.exe scripts/play_teleop.py \
      --task Mjlab-VelStand-Flat-XgoDuck \
      --checkpoint logs/rsl_rl/xgoduck_velstand/<run>/model_2999.pt \
      --num-envs 4

  # no keyboard (CI / smoke test): run a fixed number of steps then exit
  ... --max-steps 120 --no-keys

Keyboard (type in the terminal you launched from, NOT the viewer window):
  up / down      lin_vel_x  +- 0.1 m/s   (clamped to +-0.4)
  left / right   lin_vel_y  +- 0.1 m/s   (clamped to +-0.3)
  a / d          ang_vel_z  -+ 0.2 rad/s (clamped to +-1.0)
  space          zero the command (stand)
  t              toggle the policy (freeze / resume the motors)
  q              quit

On Windows the reader needs a real console: run it from PowerShell, Windows
Terminal or cmd. mintty / Git Bash does not provide one, so ``msvcrt`` cannot
see the keys there and the script says so instead of pretending.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

VX_STEP, VX_MAX = 0.10, 0.40
VY_STEP, VY_MAX = 0.10, 0.30
WZ_STEP, WZ_MAX = 0.20, 1.00


class KeyReader:
    """Single-keypress reader: ``msvcrt`` on Windows, ``termios`` on POSIX.

    Keys are reported as ``"up"``/``"down"``/``"left"``/``"right"``, a single
    lowercase letter, or ``" "``. Windows arrow keys arrive as a ``\\x00`` /
    ``\\xe0`` prefix followed by a scan code; POSIX ones as ``ESC [ A/B/C/D``.
    """

    _WIN_SCAN = {"H": "up", "P": "down", "K": "left", "M": "right"}
    _POSIX_ARROW = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self.backend: str | None = None
        self.error: str | None = None

    def __enter__(self) -> "KeyReader":
        try:
            import msvcrt  # noqa: F401
        except ImportError:
            self._start_posix()
        else:
            self._start_windows()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        restore = getattr(self, "_restore", None)
        if restore is not None:
            restore()

    # -- Windows ---------------------------------------------------------
    def _start_windows(self) -> None:
        import msvcrt

        try:
            # kbhit() raises on a handle that is not a Win32 console (mintty).
            msvcrt.kbhit()
        except Exception as e:  # noqa: BLE001
            self.error = (
                "no Win32 console on stdin (mintty / Git Bash?). Run this from "
                f"PowerShell, Windows Terminal or cmd. [{e}]"
            )
            return
        self.backend = "msvcrt"
        threading.Thread(target=self._loop_windows, daemon=True).start()

    def _loop_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            try:
                ch = msvcrt.getwch()
            except Exception:  # noqa: BLE001
                return
            if ch in ("\x00", "\xe0"):
                name = self._WIN_SCAN.get(msvcrt.getwch())
                if name:
                    self._queue.put(name)
                continue
            if ch == "\x03":  # Ctrl+C arrives as a character, not a signal
                self._queue.put("q")
                continue
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    # -- POSIX -----------------------------------------------------------
    def _start_posix(self) -> None:
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            self.error = "stdin is not a TTY"
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # cbreak, not raw: ISIG stays on so Ctrl+C works

        def restore() -> None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        self._restore = restore
        self.backend = "termios"
        threading.Thread(target=self._loop_posix, args=(fd, select, os), daemon=True).start()

    def _loop_posix(self, fd: int, select, os_mod) -> None:
        def read1(timeout: float) -> str | None:
            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return None
            data = os_mod.read(fd, 1)  # unbuffered: sys.stdin would eat the
            return data.decode(errors="ignore") if data else None  # escape tail

        while not self._stop.is_set():
            ch = read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":
                if read1(0.05) == "[":
                    final = read1(0.05)
                    name = self._POSIX_ARROW.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def drain(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                return out


class Teleop:
    """Policy wrapper: writes the keyboard command, then runs the policy.

    The command is written before the policy call, so it lands on the
    observation of the NEXT step — one step (20 ms) of lag, which is not
    perceptible and keeps the env/command API untouched.
    """

    def __init__(self, policy, env, num_envs: int, device):
        self.policy = policy
        self.cmd = env.command_manager.get_term("twist")
        self.hold = [env.command_manager.get_term(n)
                     for n in ("head_pose", "body_pose")
                     if n in env.command_manager.active_terms]
        self.num_envs = num_envs
        self.device = device
        self.action_dim = env.action_manager.total_action_dim
        self._last = torch.zeros((num_envs, self.action_dim), device=device)
        self.target = [0.0, 0.0, 0.0]
        self.enabled = True
        self._tensor = torch.zeros((num_envs, 3), device=device)

    def handle(self, key: str) -> bool:
        """Apply one key. Returns False when the run should end."""
        if key == "up":
            self.target[0] = min(self.target[0] + VX_STEP, VX_MAX)
        elif key == "down":
            self.target[0] = max(self.target[0] - VX_STEP, -VX_MAX)
        elif key == "left":
            self.target[1] = min(self.target[1] + VY_STEP, VY_MAX)
        elif key == "right":
            self.target[1] = max(self.target[1] - VY_STEP, -VY_MAX)
        elif key == "a":
            self.target[2] = min(self.target[2] + WZ_STEP, WZ_MAX)
        elif key == "d":
            self.target[2] = max(self.target[2] - WZ_STEP, -WZ_MAX)
        elif key == " ":
            self.target[:] = [0.0, 0.0, 0.0]
        elif key == "t":
            self.enabled = not self.enabled
            print(f"[teleop] policy {'ON' if self.enabled else 'OFF (frozen)'}")
            return True
        elif key == "q":
            return False
        else:
            return True
        print(f"[teleop] vx {self.target[0]:+.2f}  vy {self.target[1]:+.2f}"
              f"  wz {self.target[2]:+.2f}")
        return True

    def __call__(self, obs):
        if not self.enabled:
            # Frozen: keep replaying the last action. Returning zeros would be
            # a joint-position offset of zero, i.e. the robot folding up, not
            # "stop moving".
            return self._last
        self._tensor[:] = torch.tensor(self.target, device=self.device)
        self.cmd.command[:] = self._tensor
        for term in self.hold:
            term.command[:] = 0.0
        self._last = self.policy(obs)
        return self._last


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Mjlab-VelStand-Flat-XgoDuck")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after N env steps (smoke test)")
    p.add_argument("--no-keys", action="store_true",
                   help="do not read the keyboard at all")
    p.add_argument("--frame-rate", type=float, default=60.0)
    p.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("VX", "VY", "WZ"),
                   help="command to start from, so a headless run can prove the "
                        "injection path actually reaches the robot")
    return p.parse_args()


def main():
    args = parse_args()
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    # The keyboard owns the command: stop resampling it, and stop spawning
    # "stand still" envs that would ignore it.
    twist = env_cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    # Every knob that could overwrite the keyboard's command mid-run. The
    # heading one is a hard requirement, not a preference: `_update_command`
    # recomputes ang_vel_z from the heading error EVERY step when it is on, and
    # the validator refuses heading_command=False while ranges.heading is set.
    for attr in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs",
                 "rel_forward_envs", "init_velocity_prob"):
        if hasattr(twist, attr):
            setattr(twist, attr, 0.0)
    if hasattr(twist, "heading_command"):
        twist.heading_command = False
        twist.ranges.heading = None

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[teleop] checkpoint {ckpt.name}   {args.num_envs} envs   {device}")

    teleop = Teleop(policy, env, args.num_envs, device)
    teleop.target[:] = list(args.cmd)

    reader = None
    if not args.no_keys:
        reader = KeyReader().__enter__()
        if reader.error:
            print(f"[teleop] keyboard disabled: {reader.error}")
        else:
            print(f"[teleop] keyboard ready ({reader.backend}) — type HERE, "
                  "not in the viewer window")
            print("[teleop] up/down=vx  left/right=vy  a/d=yaw  space=stop"
                  "  t=freeze  q=quit")

    # The key thread only touches `teleop.target`; the viewer's own loop is the
    # single writer of sim state, so no locking is needed for a 3-float write.
    stop = threading.Event()

    def pump():
        while not stop.is_set() and reader is not None:
            for key in reader.drain():
                if not teleop.handle(key):
                    stop.set()
                    break
            time.sleep(0.02)

    if reader is not None and not reader.error:
        threading.Thread(target=pump, daemon=True).start()

    # The viewer wants the WRAPPED env: it calls env.get_observations() and
    # env.step(), which are the rsl_rl VecEnv API, not ManagerBasedRlEnv's.
    viewer = NativeMujocoViewer(vec_env, teleop, frame_rate=args.frame_rate)
    robot = env.scene["robot"]
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    try:
        viewer.run(num_steps=args.max_steps)
    finally:
        stop.set()
        if reader is not None:
            reader.__exit__(None, None, None)
        moved = (robot.data.root_link_pos_w[:, :2] - start_xy).norm(dim=-1)
        print(f"[teleop] cmd {teleop.target}  ->  trunk moved"
              f" {moved.mean():.3f} m mean ({moved.max():.3f} m max)"
              f" over {viewer._step_count} steps")
        vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lock the keyboard mapping in scripts/play_teleop.py.

The script itself needs a GPU env and a window, so nothing here constructs one:
the key arithmetic, the clamps and the scan-code tables are plain functions on
the Teleop object, and those are the parts that silently rot. The failure mode
this guards against is a teleop that looks alive (keys print, numbers move) but
walks the robot into a wall because a clamp was dropped or an arrow key was
mapped to the wrong axis.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pt():
    spec = importlib.util.spec_from_file_location(
        "play_teleop", REPO / "scripts" / "play_teleop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _teleop(pt):
    """A Teleop with only the fields handle() touches — no env, no tensors."""
    t = object.__new__(pt.Teleop)
    t.target = [0.0, 0.0, 0.0]
    t.enabled = True
    return t


def test_arrows_drive_the_expected_axes(pt):
    t = _teleop(pt)
    t.handle("up")
    assert t.target[0] == pytest.approx(pt.VX_STEP)
    t.handle("down")
    t.handle("down")
    assert t.target[0] == pytest.approx(-pt.VX_STEP)
    t.handle("left")
    assert t.target[1] == pytest.approx(pt.VY_STEP)
    t.handle("right")
    t.handle("right")
    assert t.target[1] == pytest.approx(-pt.VY_STEP)
    t.handle("a")
    assert t.target[2] == pytest.approx(pt.WZ_STEP)
    t.handle("d")
    t.handle("d")
    assert t.target[2] == pytest.approx(-pt.WZ_STEP)


def test_every_axis_clamps(pt):
    for key, index, limit in (
        ("up", 0, pt.VX_MAX),
        ("down", 0, -pt.VX_MAX),
        ("left", 1, pt.VY_MAX),
        ("right", 1, -pt.VY_MAX),
        ("a", 2, pt.WZ_MAX),
        ("d", 2, -pt.WZ_MAX),
    ):
        t = _teleop(pt)
        for _ in range(50):  # far more than enough to saturate
            t.handle(key)
        assert t.target[index] == pytest.approx(limit), key


def test_space_zeroes_and_quit_reports_stop(pt):
    t = _teleop(pt)
    t.handle("up")
    t.handle("a")
    t.handle(" ")
    assert t.target == [0.0, 0.0, 0.0]
    assert t.handle("q") is False
    assert t.handle("up") is True


def test_freeze_toggles_without_moving_the_command(pt):
    t = _teleop(pt)
    t.handle("up")
    before = list(t.target)
    assert t.handle("t") is True
    assert t.enabled is False
    assert t.target == before


def test_unknown_keys_are_ignored(pt):
    t = _teleop(pt)
    assert t.handle("x") is True
    assert t.target == [0.0, 0.0, 0.0]


def test_windows_arrow_scan_codes(pt):
    # msvcrt returns these after a \x00 / \xe0 prefix; getting them wrong makes
    # the arrows dead with no error anywhere.
    assert pt.KeyReader._WIN_SCAN == {"H": "up", "P": "down", "K": "left", "M": "right"}
    assert pt.KeyReader._POSIX_ARROW == {"A": "up", "B": "down", "C": "right", "D": "left"}

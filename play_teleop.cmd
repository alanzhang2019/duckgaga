@echo off
rem Keyboard teleop for the native MuJoCo viewer. Double-click or run from cmd.
rem
rem   play_teleop.cmd                       default task + the trained VelStand ckpt
rem   play_teleop.cmd <TASK> <CKPT>         explicit
rem   play_teleop.cmd "" "" --max-steps 20  extra args are forwarded
rem   play_teleop.cmd "" "" --num-envs 4    a group instead of a single robot
rem   play_teleop.cmd "" "" --prone         start on the floor, watch the get-up
rem   play_teleop.cmd "" "" --push topple   put the 1.2 m/s shoves back on
rem
rem Keys go to THIS window, not the viewer: up/down=vx, left/right=vy,
rem a/d=yaw, space=stop, t=freeze, q=quit.
setlocal
cd /d "%~dp0"

set "TASK=%~1"
if "%TASK%"=="" set "TASK=Mjlab-VelStand-Flat-XgoDuck"

set "CKPT=%~2"
if "%CKPT%"=="" set "CKPT=logs\rsl_rl\xgoduck_velstand\2026-11-03_16-18-33_xgoduck\model_2999.pt"

if not exist "%CKPT%" (
  echo [teleop] checkpoint not found: %CKPT%
  echo [teleop] pass one explicitly: play_teleop.cmd ^<TASK^> ^<CKPT^>
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [teleop] no .venv here: %CD%
  exit /b 1
)

echo [teleop] task %TASK%
echo [teleop] ckpt %CKPT%
echo [teleop] startup takes ~2.5 min; after that the keys go in THIS window.
echo.

.venv\Scripts\python.exe scripts\play_teleop.py --task %TASK% --checkpoint "%CKPT%" --num-envs 1 %3 %4 %5 %6

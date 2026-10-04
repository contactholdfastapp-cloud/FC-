@echo off
REM Usage: scripts\replay.bat recording.mp4 [calib.json]
call .venv\Scripts\activate.bat
if "%~2"=="" (
  python -m fctac.replay.viewer --video "%~1" --config configs\fc27_1440p.json --debug
) else (
  python -m fctac.replay.viewer --video "%~1" --config configs\fc27_1440p.json --calib "%~2" --debug
)

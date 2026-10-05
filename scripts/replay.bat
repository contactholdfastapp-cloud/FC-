@echo off
REM Works when double-clicked: always run from the project folder
cd /d "%~dp0.."
REM Usage: drag a gameplay video file onto this file (or: scripts\replay.bat recording.mp4 [calib.json])
if "%~1"=="" (
  echo Drag a video file onto replay.bat to open it.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if "%~2"=="" (
  python -m fctac.replay.viewer --video "%~1" --config configs\fc27_1440p.json --debug
) else (
  python -m fctac.replay.viewer --video "%~1" --config configs\fc27_1440p.json --calib "%~2" --debug
)
pause

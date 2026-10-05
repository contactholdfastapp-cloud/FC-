@echo off
REM Works when double-clicked: always run from the project folder
cd /d "%~dp0.."
REM Live overlay over FC 27 (borderless/windowed). F8 overlay, F9 debug, F10 quit.
REM Usage: scripts\live.bat [calib.json]
call .venv\Scripts\activate.bat
if "%~1"=="" (
  python -m fctac.live --config configs\fc27_1440p.json --report benchmarks\live_last.json
) else (
  python -m fctac.live --config configs\fc27_1440p.json --calib "%~1" --report benchmarks\live_last.json
)
pause

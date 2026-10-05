@echo off
REM Shows the assistant on a computer-made practice match (no game needed).
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
if not exist data\synthetic\demo.mp4 python tools\make_synthetic.py --seconds 60 --out data\synthetic\demo
python -m fctac.replay.viewer --video data\synthetic\demo.mp4
pause

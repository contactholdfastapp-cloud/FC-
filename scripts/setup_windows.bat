@echo off
REM One-time setup on the gaming PC (run from the repository root).
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .[dev] || goto :err
REM Inference provider: CUDA build for NVIDIA (falls back to DirectML/CPU automatically if absent)
pip install onnxruntime-gpu || pip install onnxruntime-directml || pip install onnxruntime
REM Live capture backends (Windows Graphics Capture preferred, DXGI fallback) + system stats
pip install windows-capture dxcam psutil pynvml
python tools\inspect_machine.py --dxdiag
python -m pytest -q
echo.
echo Setup done. Next: scripts\replay.bat path\to\recording.mp4
goto :eof
:err
echo Setup failed - see the messages above.
exit /b 1

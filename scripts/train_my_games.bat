@echo off
rem Improve the assistant with YOUR OWN FC 27 recordings.
rem Double-click this file and choose your recordings, or drag them onto it, or run:
rem    scripts\train_my_games.bat D:\Videos\match1.mp4 D:\Videos\match2.mp4
rem The last video by file name (e.g. 5_...) is kept aside to check that the new models are really better.
cd /d "%~dp0.."
if not exist .venv\Scripts\activate.bat (
  echo Please run scripts\setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import torch" 2>nul
if errorlevel 1 goto :install_torch
goto :train

:install_torch
echo Installing the training library PyTorch - one time only, about 3 GB, please wait...
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :torch_cpu
goto :train
:torch_cpu
pip install torch
if errorlevel 1 (
  echo Could not install PyTorch - see the messages above.
  pause
  exit /b 1
)

:train
python -c "import onnx" 2>nul
if errorlevel 1 pip install onnx
python -c "import torch; print('Training on', 'GRAPHICS CARD: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'processor (no CUDA graphics card found - this will be slower)')"
python tools\train_on_my_games.py %*
if errorlevel 1 goto :failed
echo.
echo Finished. New models are only switched on if they measured better.
echo Send these two files to Claude: data\my_games\summary.json and data\my_games\radar_qa.jpg
pause
exit /b 0

:failed
echo.
echo ===== TRAINING FAILED - nothing was changed, the assistant still uses its current models =====
echo Send these two files to Claude: data\my_games\summary.json and data\my_games\train_log.txt
pause
exit /b 1

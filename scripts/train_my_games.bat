@echo off
rem Improve the assistant with YOUR OWN FC 27 recordings.
rem Drag 2 or more recordings (mp4) onto this file, or run:
rem    scripts\train_my_games.bat data\recordings\game1.mp4 data\recordings\game2.mp4
rem The last video is kept aside to check that the new models are really better.
cd /d "%~dp0.."
if "%~2"=="" (
  echo Please give at least TWO recordings ^(drag them onto this file together^).
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python tools\train_on_my_games.py %*
echo.
echo Finished. New models are only switched on if they measured better.
pause

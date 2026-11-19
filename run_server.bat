@echo off
title Space-Time Volume Web Studio
echo ========================================================
echo   Space-Time Volume 4D-to-3D Web Studio Server
echo   Running at: http://127.0.0.1:8000
echo ========================================================
echo.
C:\miniconda3\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
pause

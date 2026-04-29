@echo off
setlocal

REM Kill any stale Python process holding :8052 before launching
echo Checking :8052...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr LISTENING ^| findstr :8052') do (
    echo Found stale process %%p on :8052, killing...
    taskkill /F /PID %%p >nul 2>&1
)

echo === Portfolio-Manager (tim) launching ===

REM Try the four common conda install locations
set CONDA_OK=0
for %%d in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\Miniconda3" "%USERPROFILE%\Anaconda3") do (
    if exist "%%~d\condabin\conda.bat" (
        call "%%~d\condabin\conda.bat" activate portfolio-manager 2>nul
        if not errorlevel 1 set CONDA_OK=1
    )
)
if "%CONDA_OK%"=="0" (
    where conda >nul 2>nul
    if not errorlevel 1 (
        call conda activate portfolio-manager 2>nul
        if not errorlevel 1 set CONDA_OK=1
    )
)
if "%CONDA_OK%"=="0" (
    echo ERROR: Could not activate conda env 'portfolio-manager'.
    echo Setup commands:
    echo     conda create -n portfolio-manager python=3.12 -y
    echo     conda activate portfolio-manager
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

python --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in env.
    pause
    exit /b 1
)

REM Browser before server (intentional — server start races the browser load)
start "" http://127.0.0.1:8052/

REM Run from the project root so 'tim' is importable as a package
cd /D "%~dp0\.."

python -m tim.app

endlocal

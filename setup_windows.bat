@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 (
    echo Could not create virtual environment. Install Python 3.11 or 3.12 and try again.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA 12.8 support. Requires a compatible NVIDIA driver.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo PyTorch CUDA install failed. Check your NVIDIA driver/Python version.
    pause
    exit /b 1
)

echo.
echo Installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo One or more dependencies failed to install. Review the error above.
    pause
    exit /b 1
)

echo.
echo Setup finished. Start the app with:
echo   .venv\Scripts\activate
echo   python main.py
pause

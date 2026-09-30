@echo off
cd /d "%~dp0"
echo === TimesFM app setup ===

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer is required.
  pause
  exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
  echo Creating virtual environment...
  python -m venv venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Package installation failed.
  pause
  exit /b 1
)

echo.
echo Installing covariate (XReg) support...
pip install jax scikit-learn
if errorlevel 1 (
  echo [WARN] XReg install failed. Covariates will be unavailable; the rest still works.
)

echo.
echo Setup complete. Run run.bat to start the app.
pause

@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
  echo venv not found. Running setup first...
  call setup.bat
)
call venv\Scripts\activate.bat
streamlit run app.py
pause

@echo off
TITLE Apex Banking - Term Deposit AI Intelligence Dashboard
echo ======================================================================
echo    APEX BANKING - TERM DEPOSIT CONVERSION INTELLIGENCE PLATFORM
echo ======================================================================
echo.

cd /d "%~dp0"

IF EXIST ".venv\Scripts\streamlit.exe" (
    echo [INFO] Detected dedicated project virtual environment (.venv).
    echo [INFO] Starting Streamlit server...
    echo.
    ".venv\Scripts\streamlit.exe" run app.py
) ELSE (
    echo [WARNING] .venv not detected. Launching with system streamlit...
    streamlit run app.py
)

pause

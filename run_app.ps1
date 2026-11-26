Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   APEX BANKING - TERM DEPOSIT CONVERSION INTELLIGENCE PLATFORM" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path $PSScriptRoot

if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    Write-Host "[INFO] Detected dedicated project virtual environment (.venv)." -ForegroundColor Green
    Write-Host "[INFO] Starting Streamlit server on http://localhost:8501 ..." -ForegroundColor Green
    Write-Host ""
    & ".\.venv\Scripts\streamlit.exe" run app.py
} else {
    Write-Host "[WARNING] .venv not detected. Launching with system streamlit..." -ForegroundColor Yellow
    streamlit run app.py
}

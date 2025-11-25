<#
PowerShell helper to create a venv, install requirements and run the drowsiness script.
Usage: Open PowerShell in this folder and run `.
un_script.ps1`.
#>

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path .\venv)) {
    python -m venv .\venv
}

Write-Host "Activating venv and installing requirements..."
# Activate venv for PowerShell
. .\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt

Write-Host "Starting drowsiness script (press 'q' in the window to quit)..."
python .\realtime_drowsiness_mediapipe.py

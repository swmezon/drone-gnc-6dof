$ErrorActionPreference = "Stop"

Write-Host "Applying drone-gnc-6dof control optimization update..." -ForegroundColor Cyan

if (-not (Test-Path ".\src")) {
    throw "Run this script from the root of the drone-gnc-6dof repository."
}

Copy-Item ".\update_files\src\*" ".\src" -Recurse -Force
Copy-Item ".\update_files\scripts\*" ".\scripts" -Recurse -Force
Copy-Item ".\update_files\tests\*" ".\tests" -Recurse -Force
Copy-Item ".\update_files\docs\*" ".\docs" -Recurse -Force

Write-Host "Update applied." -ForegroundColor Green
Write-Host "Next run: python -m pytest -v"
Write-Host "Then:     python -m scripts.run_pid_autotune"

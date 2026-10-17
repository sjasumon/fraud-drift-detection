$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required" }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip setup failed" }
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
$env:USE_GPU = "false"
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Validation failed; inspect pytest output" }
Write-Host "Setup complete. Run: .\.venv\Scripts\python.exe -m src.demo"

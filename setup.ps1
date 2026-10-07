$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
if (!(Test-Path .venv/Scripts/python.exe)) { python -m venv .venv }
& ./.venv/Scripts/python.exe -m ensurepip
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt

$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot '.tmp'
$env:TMP=$env:TEMP
$env:YOLO_CONFIG_DIR=Join-Path $PSScriptRoot '.yolo'
$env:MPLCONFIGDIR=Join-Path $PSScriptRoot '.mpl'
$taskPython=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
# Main experiment: all runs use the same budget and validation; test remains sealed until selection.
& $taskPython train.py --models models/yolo11n.pt models/yolov8n.pt --fractions 10 30 100 --epochs 50 --batch 4
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect training.log and artifacts/experiments.json' }
& $taskPython analyze_features.py --images 1000
if ($LASTEXITCODE -ne 0) { throw 'Auxiliary experiment failed' }
& $taskPython train.py --final-test --batch 4
if ($LASTEXITCODE -ne 0) { throw 'Final evaluation failed' }
& $taskPython summarize.py

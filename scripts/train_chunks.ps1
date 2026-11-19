# Phase 5: train DistilBERT in SMALL, resumable chunks so the GPU is never
# busy for long. Each chunk trains a few hundred steps, saves a checkpoint,
# exits, and waits before the next one. Stop any time and resume later.
#
# Usage (PowerShell, from D:\VoxLens):
#   .\scripts\train_chunks.ps1                     # defaults: 200k budget, 250 steps/chunk
#   .\scripts\train_chunks.ps1 -TotalSteps 1250    # small run
#   .\scripts\train_chunks.ps1 -Model bert-base-uncased -BatchSize 2 -GradAccum 16
#
# Every chunk is a separate process: the GPU is released between chunks.

param(
    [string]$Model = "distilbert-base-uncased",
    [int]$Sample = 200000,
    [int]$ValSample = 20000,
    [int]$BatchSize = 4,
    [int]$GradAccum = 8,
    [int]$ChunkSteps = 250,
    [int]$TotalSteps = 12500,
    [int]$PauseSeconds = 30,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = "D:\VoxLens\.venv\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "models/" + ($Model -split "/")[-1]
}
New-Item -ItemType Directory -Force -Path "models" | Out-Null

$steps = @()
for ($s = $ChunkSteps; $s -lt $TotalSteps; $s += $ChunkSteps) { $steps += $s }
$steps += $TotalSteps

$i = 0
foreach ($target in $steps) {
    $i++
    $free = (nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)
    Write-Host "=== chunk $i/$($steps.Count): max_steps=$target | GPU used $free ==="
    $args = @(
        "-m", "voc.models.transformer", "train",
        "--model-name", $Model,
        "--sample", "$Sample",
        "--val-sample", "$ValSample",
        "--epochs", "1",
        "--batch-size", "$BatchSize",
        "--grad-accum", "$GradAccum",
        "--max-steps", "$target",
        "--output-dir", $OutputDir
    )
    if ($i -gt 1) { $args += "--resume" }
    & $py @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "chunk failed (exit $LASTEXITCODE) - stopping. Fix cause before resuming."
        exit $LASTEXITCODE
    }
    if ($i -lt $steps.Count) {
        Write-Host "pausing $PauseSeconds s to let the GPU rest..."
        Start-Sleep -Seconds $PauseSeconds
    }
}
Write-Host "all chunks done -> $OutputDir"

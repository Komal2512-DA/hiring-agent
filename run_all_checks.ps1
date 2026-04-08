param(
    [string]$VenvPath = ".venv_clean",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-ExitCodeZero {
    param(
        [int]$ExitCode,
        [string]$Context
    )
    if ($ExitCode -ne 0) {
        throw "$Context failed with exit code $ExitCode"
    }
}

$python = Join-Path (Get-Location) "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable not found at: $python"
}

if (-not $SkipInstall) {
    Write-Step "Installing dependencies"
    & $python -m pip install --upgrade pip
    Require-ExitCodeZero $LASTEXITCODE "pip upgrade"

    & $python -m pip install -r requirements.txt
    Require-ExitCodeZero $LASTEXITCODE "pip install -r requirements.txt"
}

Write-Step "Dependency import check"
& $python -c "import fastapi,uvicorn,openai,dotenv; print('deps_ok')"
Require-ExitCodeZero $LASTEXITCODE "dependency import check"

$env:USE_LLM_JUSTIFICATION = "0"
$env:OPENENV_LLM_PROXY_PROBE = "0"

$server = $null
$serverLog = Join-Path (Get-Location) "server_smoke.log"
$serverErrLog = Join-Path (Get-Location) "server_smoke.err.log"
if (Test-Path $serverLog) {
    Remove-Item -Force $serverLog
}
if (Test-Path $serverErrLog) {
    Remove-Item -Force $serverErrLog
}

try {
    Write-Step "Starting API server on 127.0.0.1:7860"
    $server = Start-Process `
        -FilePath $python `
        -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","7860" `
        -WorkingDirectory (Get-Location) `
        -PassThru `
        -RedirectStandardOutput $serverLog `
        -RedirectStandardError $serverErrLog

    $healthOk = $false
    for ($i = 1; $i -le 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:7860/health" -Method Get
            if ($health.status -eq "ok") {
                $healthOk = $true
                break
            }
        } catch {
            # keep retrying while server boots
        }
    }

    if (-not $healthOk) {
        $tail = ""
        if (Test-Path $serverLog) {
            $tail = (Get-Content $serverLog -ErrorAction SilentlyContinue | Select-Object -Last 40) -join "`n"
        }
        if (Test-Path $serverErrLog) {
            $tailErr = (Get-Content $serverErrLog -ErrorAction SilentlyContinue | Select-Object -Last 40) -join "`n"
            $tail = "$tail`n$tailErr"
        }
        throw "Server did not become healthy on :7860.`n$tail"
    }

    Write-Step "Endpoint smoke tests"
    $tasks = Invoke-RestMethod -Uri "http://127.0.0.1:7860/tasks" -Method Get
    if (-not $tasks -or $tasks.Count -lt 3) {
        throw "Expected at least 3 tasks from /tasks"
    }

    $resetBody = @{ task_id = "task_easy_screen_backend" } | ConvertTo-Json -Compress
    $reset = Invoke-RestMethod -Uri "http://127.0.0.1:7860/reset" -Method Post -ContentType "application/json" -Body $resetBody
    if ($reset.observation.task_id -ne "task_easy_screen_backend") {
        throw "Unexpected reset task_id: $($reset.observation.task_id)"
    }

    $state = Invoke-RestMethod -Uri "http://127.0.0.1:7860/state" -Method Get
    if (-not $state.task.task_id) {
        throw "/state response missing task metadata"
    }

    $stepBody = @{
        action_type = "shortlist_candidates"
        payload = @{
            candidate_ids = @("C001", "C002")
        }
    } | ConvertTo-Json -Compress
    $step = Invoke-RestMethod -Uri "http://127.0.0.1:7860/step" -Method Post -ContentType "application/json" -Body $stepBody
    if (-not $step.observation.task_id) {
        throw "/step response missing observation"
    }

    Write-Host "API smoke ok: tasks=$($tasks.Count) reset_task=$($reset.observation.task_id) step_done=$($step.observation.done)" -ForegroundColor Green
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}

Write-Step "Running pytest"
& $python -m pytest -q tests
Require-ExitCodeZero $LASTEXITCODE "pytest"

Write-Step "Running validator"
& $python validator.py
Require-ExitCodeZero $LASTEXITCODE "validator"

Write-Step "Running baseline inference (output also saved to inference_output.log)"
& $python inference.py | Tee-Object -FilePath "inference_output.log"
Require-ExitCodeZero $LASTEXITCODE "inference"

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green
Write-Host "Server log: $serverLog"
Write-Host "Server err log: $serverErrLog"
Write-Host "Inference log: $(Join-Path (Get-Location) 'inference_output.log')"

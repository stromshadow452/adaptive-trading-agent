#Requires -Version 5.1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

param (
    [string]$CsvDir,
    [string]$MetaModel,
    [string]$Universe,
    [double]$PrimaryThresh,
    [double]$FinrlThresh,
    [string]$ExpectTimeframe,
    [switch]$StrictMeta
)

if (-not $PSBoundParameters.ContainsKey('CsvDir')         -or [string]::IsNullOrWhiteSpace($CsvDir))         { $CsvDir = "data/raw/M15_only" }
if (-not $PSBoundParameters.ContainsKey('MetaModel')      -or [string]::IsNullOrWhiteSpace($MetaModel))      { $MetaModel = "models/fx_bin_19f.joblib" }
if (-not $PSBoundParameters.ContainsKey('Universe')       -or [string]::IsNullOrWhiteSpace($Universe))       { $Universe = "forex" }
if (-not $PSBoundParameters.ContainsKey('PrimaryThresh')  -or $null -eq $PrimaryThresh)                       { $PrimaryThresh = 0.35 }
if (-not $PSBoundParameters.ContainsKey('FinrlThresh')    -or $null -eq $FinrlThresh)                         { $FinrlThresh = 0.35 }
if (-not $PSBoundParameters.ContainsKey('ExpectTimeframe')-or [string]::IsNullOrWhiteSpace($ExpectTimeframe)) { $ExpectTimeframe = "M15" }

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
$python     = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

function _Path([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $p }
    if ([IO.Path]::IsPathRooted($p))      { return $p }
    return (Join-Path $RepoRoot $p)
}

$CsvDir    = _Path $CsvDir
$MetaModel = _Path $MetaModel

$ApprovedJson = Join-Path $RepoRoot "reports\daily\approved_top1.json"
$FixPlansPy   = Join-Path $RepoRoot "tools\fix_plans_price.py"
$TopPlansPy   = Join-Path $RepoRoot "tools\select_top_plans.py"
$ExecutorPy   = Join-Path $RepoRoot "tools\executor.py"
$ExecDir      = Join-Path $RepoRoot "reports\executions"

New-Item -ItemType Directory -Force -Path $ExecDir | Out-Null
$ts      = (Get-Date).ToString('yyyyMMdd_HHmmss')
$execOut = Join-Path $ExecDir "executions_$ts.csv"

function Banner([string]$text) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor DarkGray
    Write-Host ("== {0} ==" -f $text) -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor DarkGray
    Write-Host ""
}

function Run-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )
    Banner $Title
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Action
        $sw.Stop()
        Write-Host ("[OK] {0} in {1:N2}s" -f $Title, ($sw.Elapsed.TotalSeconds)) -ForegroundColor Green
    } catch {
        $sw.Stop()
        Write-Host ("[FAIL] {0} after {1:N2}s" -f $Title, ($sw.Elapsed.TotalSeconds)) -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        if ($_.ScriptStackTrace) { Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed }
        throw
    }
}

Run-Step -Title "1/3 Fixing plans (price/size backfill)" -Action {
    & $python $FixPlansPy
}

Run-Step -Title "2/3 Selecting top plan per symbol" -Action {
    & $python $TopPlansPy
}

Run-Step -Title "3/3 Executing" -Action {
    $argsList = @(
        "--approved",            $ApprovedJson,
        "--executions",          $execOut,
        "--model_primary_path",  $MetaModel,
        "--csv_price_dir",       $CsvDir,
        "--universe",            $Universe,
        "--primary_thresh",      $PrimaryThresh,
        "--finrl_thresh",        $FinrlThresh,
        "--verbose"
    )
    if ($ExpectTimeframe -and $ExpectTimeframe.Trim()) { $argsList += @("--expect_timeframe", $ExpectTimeframe) }
    if ($StrictMeta.IsPresent) { $argsList += @("--strict_meta") }
    & $python $ExecutorPy @argsList
}

Write-Host ""
Write-Host ("Executions written to: {0}" -f $execOut) -ForegroundColor Green

if (Test-Path $execOut) {
    Get-Content $execOut | Select-Object -Last 10
} else {
    Write-Host "Expected executions file not found at $execOut" -ForegroundColor Yellow
}

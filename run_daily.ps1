param
(
  [string]$CsvDir = "data/raw/forex_backup_2020_2025",
  [string]$MetaModel = "models/fx_bin_19f.joblib",
  [string]$Universe = "forex",
  [double]$PrimaryThresh = 0.35,
  [double]$FinrlThresh = 0.35,
  [string]$ExpectTimeframe = "M15",
   [switch]$StrictMeta = $true,
  [switch]$RequireCsvPrice = $false,
  [int]$MaxPosition = 1,
  [switch]$Verbose = $true
)

# ---------- helpers ----------
function Banner($text, $color = "Cyan") {
  $line = "=" * ($text.Length + 12)
  Write-Host ""
  Write-Host $line -ForegroundColor $color
  Write-Host ("[ " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " ]  " + $text) -ForegroundColor $color
  Write-Host $line -ForegroundColor $color
}
function FailIfErr($msg) {
  if ($LASTEXITCODE -ne 0) {
    Write-Host ("❌ " + $msg) -ForegroundColor Red
    exit 1
  }
}

$start = Get-Date
$ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
$execOut = "reports/executions/executions_$ts.csv"

# ---------- validations ----------
if (-not (Test-Path $MetaModel)) { Write-Host "❌ Model not found: $MetaModel" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $CsvDir))    { Write-Host "❌ CSV dir not found: $CsvDir" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path (Split-Path $execOut) | Out-Null

Write-Host ""
Write-Host "Parameters:" -ForegroundColor DarkGray
Write-Host ("  CsvDir         : {0}" -f $CsvDir)
Write-Host ("  MetaModel      : {0}" -f $MetaModel)
Write-Host ("  Universe       : {0}" -f $Universe)
Write-Host ("  PrimaryThresh  : {0}" -f $PrimaryThresh)
Write-Host ("  FinrlThresh    : {0}" -f $FinrlThresh)
Write-Host ("  ExpectTimeframe: {0}" -f $ExpectTimeframe)
Write-Host ("  StrictMeta     : {0}" -f ([bool]$StrictMeta))
Write-Host ("  RequireCsvPrice: {0}" -f ([bool]$RequireCsvPrice))
Write-Host ("  MaxPosition    : {0}" -f $MaxPosition)
Write-Host ("  Verbose        : {0}" -f ([bool]$Verbose))
Write-Host ""

# ---------- step 1 ----------
Banner "STEP 1/3 — Fixing plans (price/size backfill)"
python tools\fix_plans_price.py
FailIfErr "Error during price/size backfill."

# ---------- step 2 ----------
Banner "STEP 2/3 — Selecting top plan per symbol"
python tools\select_top_plans.py
FailIfErr "Error selecting top plans."

# ---------- step 3 ----------
Banner "STEP 3/3 — Executing with model + metadata checks"
# Build executor args cleanly
$executorArgs = @(
  "--approved", "reports\daily\approved_top1.json",
  "--executions", $execOut,
  "--model_primary_path", $MetaModel,
  "--csv_price_dir", $CsvDir,
  "--universe", $Universe,
  "--primary_thresh", $PrimaryThresh,
  "--finrl_thresh", $FinrlThresh,
  "--max_position", $MaxPosition,
  "--mode", "paper"
)

if ($ExpectTimeframe -and $ExpectTimeframe.Trim()) {
  # Note: `tools/executor.py` does not accept --expect_timeframe in this version; skip adding it.
}
# `--strict_meta` flag not supported by current executor; skipping addition to args.
}
if ($RequireCsvPrice) {
  $executorArgs += @("--require_csv_price")
}
if ($Verbose) {
  $executorArgs += @("--verbose")
}

# Run executor
python tools\executor.py @executorArgs
FailIfErr "Execution failed."

# ---------- summary ----------
Banner "✅ EXECUTION COMPLETE — Report Generated" "Green"
Write-Host ("Saved to: {0}" -f $execOut) -ForegroundColor Green

if (Test-Path $execOut) {
  Banner "Tail of Executions (last 10 rows)" "DarkCyan"
  Get-Content $execOut | Select-Object -Last 10 | ForEach-Object { Write-Host $_ }
} else {
  Write-Host "⚠️ Execution file not found (unexpected)." -ForegroundColor Yellow
}

$elapsed = (Get-Date) - $start
Banner ("🎯 DAILY PIPELINE FINISHED in {0:mm\:ss} (mm:ss)" -f $elapsed) "Green"

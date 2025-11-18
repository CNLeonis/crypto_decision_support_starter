param(
    [switch]$Install = $false,
    [switch]$NoDownload = $false,
    [switch]$SkipEDA = $false,
    [switch]$SkipBacktest = $false,
    [switch]$SkipTrain = $false,
    [switch]$SkipInference = $false,
    [string]$Symbols = "ALL"
)

$ErrorActionPreference = "Stop"

# Move to project root (this file lives in scripts/)
Set-Location -Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

# Prepare logging
$reportsDir = "reports"
$logDir = Join-Path $reportsDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir ("run-all_{0}.log" -f $timestamp)

Start-Transcript -Path $logPath -Append | Out-Null

function Get-PythonCmd {
    if (Test-Path ".\.venv\Scripts\python.exe") { return ".\.venv\Scripts\python.exe" }
    return "python"
}
$PY = Get-PythonCmd

function Get-SymbolList {
    param([string]$SymbolsArg)
    $files = @(Get-ChildItem -Path "data/raw" -Filter "*_1h.parquet" -ErrorAction SilentlyContinue)
    if (-not $files) { return @() }
    $names = $files | Sort-Object Name | ForEach-Object { $_.BaseName -replace "_1h$","" }
    if ($SymbolsArg -eq "ALL") { return $names }
    $wanted = $SymbolsArg.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if (-not $wanted) { return $names }
    $wanted = $wanted | ForEach-Object { $_.ToUpper() }
    return $names | Where-Object { $wanted -contains $_.ToUpper() }
}

Write-Host "==[ run-all ]===================================================" -ForegroundColor Cyan
Write-Host "Install=$Install NoDownload=$NoDownload SkipEDA=$SkipEDA SkipBacktest=$SkipBacktest SkipTrain=$SkipTrain SkipInference=$SkipInference Symbols=$Symbols"
Write-Host "Using Python: $($PY)"

# Ensure required dirs exist
New-Item -ItemType Directory -Force -Path "reports\eda" | Out-Null
New-Item -ItemType Directory -Force -Path "reports\backtest" | Out-Null
New-Item -ItemType Directory -Force -Path "reports\models\lgbm" | Out-Null

# Optional install/update
if ($Install) {
    Write-Host "[1/10] Updating pip and installing project dependencies..." -ForegroundColor Yellow
    & $PY -m pip install -U pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $PY -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "project install failed" }

    # Optional: pre-commit hooks if git repo and pre-commit present
    if (Test-Path ".\.git") {
        Write-Host "[pre-commit] Installing hooks (if available)..." -ForegroundColor DarkYellow
        try { pre-commit install | Out-Null } catch {}
    }
}
else {
    Write-Host "[1/10] Skipping install (use -Install to enable)" -ForegroundColor DarkGray
}

# Download data (unless skipped)
if (-not $NoDownload) {
    Write-Host "[2/10] Downloading data via src.data.download ..." -ForegroundColor Yellow
    & $PY -m src.data.download
    if ($LASTEXITCODE -ne 0) { throw "data download failed" }
}
else {
    Write-Host "[2/10] Skipping data download (NoDownload)" -ForegroundColor DarkGray
}

# EDA
if (-not $SkipEDA) {
    Write-Host "[3/10] Running EDA notebooks/eda_altcoins.py ..." -ForegroundColor Yellow
    & $PY notebooks/eda_altcoins.py
    if ($LASTEXITCODE -ne 0) { throw "EDA failed" }
}
else {
    Write-Host "[3/10] Skipping EDA" -ForegroundColor DarkGray
}

# Backtest
if (-not $SkipBacktest) {
    Write-Host "[4/10] Backtest: src.backtest.run_altcoins ..." -ForegroundColor Yellow
    & $PY -m src.backtest.run_altcoins
    if ($LASTEXITCODE -ne 0) { throw "backtest failed" }
}
else {
    Write-Host "[4/10] Skipping backtest" -ForegroundColor DarkGray
}

# Train LightGBM
if (-not $SkipTrain) {
    Write-Host "[5/10] Training LightGBM per symbol ..." -ForegroundColor Yellow
    if ($Symbols -ne "ALL") {
        & $PY -m src.models.train_lgbm_altcoins --symbols $Symbols
    }
    else {
        & $PY -m src.models.train_lgbm_altcoins
    }
    if ($LASTEXITCODE -ne 0) { throw "training failed" }
}
else {
    Write-Host "[5/10] Skipping training" -ForegroundColor DarkGray
}

$calibCmd = {
    if ($Symbols -ne "ALL") {
        & $PY -m src.models.calibrate_thresholds --symbols $Symbols
    }
    else {
        & $PY -m src.models.calibrate_thresholds
    }
}

Write-Host "[6/10] Calibrating probability thresholds ..." -ForegroundColor Yellow
& $calibCmd
if ($LASTEXITCODE -ne 0) { throw "probability calibration failed" }

$symbolList = @()
if (-not $SkipInference) {
    $symbolList = Get-SymbolList -SymbolsArg $Symbols
    if ($symbolList.Count -eq 0) {
        Write-Host "[7/10] Skipping inference & visualizations (no symbols found)" -ForegroundColor DarkGray
    }
    else {
        Write-Host "[7/10] Building inference predictions & plots ..." -ForegroundColor Yellow
        foreach ($sym in $symbolList) {
            Write-Host ("  -> {0}: inference" -f $sym) -ForegroundColor DarkCyan
            & $PY scripts/predict_future_lgbm.py --symbol $sym
            if ($LASTEXITCODE -ne 0) { throw "inference failed for $sym" }

            & $PY scripts/build_predictions_vs_price.py --symbol $sym
            if ($LASTEXITCODE -ne 0) { throw "build_predictions_vs_price failed for $sym" }

            $oofCsv = "reports/models/lgbm/$sym/predictions_vs_price.csv"
            if (Test-Path $oofCsv) {
                & $PY scripts/plot_predictions_vs_price.py --csv $oofCsv --symbol $sym
                if ($LASTEXITCODE -ne 0) { throw "plot predictions_vs_price failed for $sym" }
            }
            else {
                Write-Host ("    missing {0}, skipping OOF plot" -f $oofCsv) -ForegroundColor DarkGray
            }

            $inferCsv = "reports/models/lgbm/$sym/inference_predictions.csv"
            if (Test-Path $inferCsv) {
                $inferOut = Join-Path (Split-Path $inferCsv -Parent) ("{0}_inference_pred_vs_price.png" -f $sym)
                & $PY scripts/plot_predictions_vs_price.py --csv $inferCsv --symbol $sym --out $inferOut
                if ($LASTEXITCODE -ne 0) { throw "plot inference predictions failed for $sym" }
            }
            else {
                Write-Host ("    missing {0}, skipping inference plot" -f $inferCsv) -ForegroundColor DarkGray
            }
        }
    }
}
else {
    Write-Host "[7/10] Skipping inference & visualizations" -ForegroundColor DarkGray
}

# Summaries
Write-Host "[8/10] Summaries:" -ForegroundColor Green
$eda = "reports\eda\eda_summary.csv"
$bt = "reports\backtest\metrics_altcoins.csv"
$sum = "reports\models\lgbm\summary.csv"
if (Test-Path $eda) { Write-Host ("  EDA:     {0}" -f (Resolve-Path $eda)) }
if (Test-Path $bt) { Write-Host ("  Backtest:{0}" -f (Resolve-Path $bt)) }
if (Test-Path $sum) { Write-Host ("  LGBM:    {0}" -f (Resolve-Path $sum)) }

Write-Host "[9/10] Calibration Backtest ..." -ForegroundColor Cyan
& $PY -m src.backtest.run_calibrated
if ($LASTEXITCODE -ne 0) { throw "calibration backtest failed" }

Write-Host "[10/10] Compare & Report ..." -ForegroundColor Cyan
& $PY -m src.backtest.compare_baselines_vs_calibrated
# (optional) report generator if present:
# & $PY -m src.reports.build_calibration_report

Write-Host "Done. Full log at: $logPath" -ForegroundColor Cyan
Stop-Transcript | Out-Null

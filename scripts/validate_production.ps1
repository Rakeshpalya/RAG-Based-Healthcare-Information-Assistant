# ==============================================================================
# AI-Healthcare-Agent: Windows Production Validation PowerShell Script
# Validates FAISS vector invariants, startup pre-flight checks, and health endpoints.
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "AI-HEALTHCARE-AGENT: WINDOWS PRODUCTION DEPLOYMENT VALIDATION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Locate Python Virtual Environment
$PythonExe = Join-Path $RootDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[1/4] Running Python Production Validation Harness..." -ForegroundColor Yellow
& $PythonExe (Join-Path $RootDir "scripts\validate_production.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Production validation harness failed!" -ForegroundColor Red
    exit 1
}

if (-not $SkipTests) {
    Write-Host "[2/4] Running Phase 6 Production Readiness Tests..." -ForegroundColor Yellow
    & $PythonExe -m pytest (Join-Path $RootDir "tests\test_phase6_production_readiness.py") -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Phase 6 tests failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "[3/4] Running Evaluation Runner Benchmark..." -ForegroundColor Yellow
    & $PythonExe -m backend.evaluation.evaluation_runner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Evaluation runner failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[4/4] Final Vector Store Invariant Check..." -ForegroundColor Yellow
& $PythonExe -c "import json, faiss; idx = faiss.read_index('data/vector_store/index.faiss'); meta = json.load(open('data/vector_store/metadata.json', 'r', encoding='utf-8')); recs = meta.get('records', []); print(f'FAISS: {idx.ntotal} | Meta: {len(recs)}'); assert idx.ntotal == 744 and len(recs) == 744, 'Invariant failure!'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Vector store invariant check failed!" -ForegroundColor Red
    exit 1
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] All Production Validation & Invariant Checks PASSED (100%)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
exit 0

# Runs the same steps as the CI pipeline.   Usage:  .\run_all.ps1
Write-Host "== 1. Lint ==" -ForegroundColor Yellow
python -m flake8 src tests --max-line-length=100
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: lint failed" -ForegroundColor Red; exit 1 }

Write-Host "== 2. Unit tests ==" -ForegroundColor Yellow
python -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: tests failed" -ForegroundColor Red; exit 1 }

Write-Host "== 3. Data quality gate ==" -ForegroundColor Yellow
python -m src.data_validation
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: data gate failed" -ForegroundColor Red; exit 1 }

Write-Host "== 4. Train model ==" -ForegroundColor Yellow
python -m src.train
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: training failed" -ForegroundColor Red; exit 1 }

Write-Host "== 5. Model quality gate ==" -ForegroundColor Yellow
python -m src.quality_gate
if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: model gate failed" -ForegroundColor Red; exit 1 }

Write-Host "Pipeline finished OK - model is allowed to be published" -ForegroundColor Green

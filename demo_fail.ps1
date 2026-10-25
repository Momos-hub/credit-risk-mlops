# DEMO 1: model gate fails (stricter AUC rule).  Nothing is edited.
python -m src.quality_gate --min-auc 0.99
Write-Host "Exit code:" $LASTEXITCODE

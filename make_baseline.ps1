# Re-create baseline_metrics.json from the older v1 data (only needed if you change the model).
python -m src.train --data data/credit_card_customers_v1.csv --out models/baseline
Copy-Item models/baseline/metrics.json baseline_metrics.json -Force
Write-Host "baseline_metrics.json updated"

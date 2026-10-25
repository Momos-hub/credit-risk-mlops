# DEMO 2: data gate fails (corrupted copy of the data).  The real data is not touched.
python -m src.demo_bad_data
python -m src.data_validation --data data/corrupted_demo.csv
Write-Host "Exit code:" $LASTEXITCODE

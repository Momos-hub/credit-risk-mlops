# Credit Risk Model with a CI/CD Quality Gate

Data: `data/credit_card_customers_v1.csv` (800 customers) and `..._v2.csv` (1,100 customers; v1 + 300 new).
The model trained on v1 is the **baseline** (`baseline_metrics.json`); the model trained on v2 must beat the gates.

Pipeline: lint -> unit tests -> **data gate** -> train -> **model gate** -> publish artifact.
Either gate exits with code 1, which turns the CI build red and stops the release.

## Run on Windows (PowerShell)
    python -m venv .venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    .\run_all.ps1            # whole pipeline, ends with "QUALITY GATE PASSED"
    .\demo_fail.ps1          # demo 1: model gate fails (min AUC 0.99)
    .\demo_bad_data.ps1      # demo 2: data gate rejects a corrupted copy of the data
    python -m src.make_plots # regenerate figures in docs/figures

## Individual commands
    python -m src.data_validation [--data file.csv]
    python -m src.train [--data file.csv] [--out folder]
    python -m src.quality_gate [--min-auc 0.99]
    .\make_baseline.ps1      # rebuild baseline_metrics.json from v1 (only if you change the model)

## Mac / Linux
    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
    flake8 src tests --max-line-length=100 && pytest -q && python -m src.data_validation \
      && python -m src.train && python -m src.quality_gate

## Put it on GitHub (this is what makes the CI/CD real)
    git init
    git add .
    git commit -m "Credit risk model with quality gates"
    git branch -M main
    git remote add origin https://github.com/<your-username>/credit-risk-mlops.git
    git push -u origin main
Then open the repo -> Actions tab; the run starts automatically.

## Notes
- Docker is NOT needed. Nothing here requires it.
- Open the folder that directly contains `config.yaml` in VS Code; all paths are relative to it.
- Step-by-step presentation script: `docs/DEMO_GUIDE.md`.

## Settings
Everything tunable (data path, validation rules, thresholds) is in `config.yaml`.

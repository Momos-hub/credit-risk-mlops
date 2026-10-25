# Demo guide (Mukesh Gond, B063)

## Before the presentation (do once, at home)
1. VS Code: File > Open Folder > choose the folder that CONTAINS config.yaml.
2. Terminal > New Terminal, then:
       python --version                      (should be 3.10 - 3.13)
       python -m venv .venv
       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
       .\.venv\Scripts\Activate.ps1          (prompt now starts with (.venv))
       pip install -r requirements.txt
3. Run   .\run_all.ps1   once. It must end with: Pipeline finished OK
4. Push to GitHub (see README) and wait for a green tick in the Actions tab.

## Live demo order (about 5 minutes)
| # | What you say | Command |
|---|---|---|
| 1 | "Whole pipeline, locally" | .\run_all.ps1 |
| 2 | "Model gate can fail" | .\demo_fail.ps1 |
| 3 | "Data gate can fail" | .\demo_bad_data.ps1 |
| 4 | "On GitHub it runs by itself" | Show Actions tab (green run) |
| 5 | "A bad change is blocked" | Edit config.yaml: min_auc 0.90 -> 0.99, commit, push -> red cross. Change back, push -> green |

## If .\run_all.ps1 is blocked or fails, type the steps by hand
    python -m flake8 src tests --max-line-length=100
    python -m pytest -q
    python -m src.data_validation
    python -m src.train
    python -m src.quality_gate
Demo 2 by hand:  python -m src.quality_gate --min-auc 0.99
Demo 3 by hand:  python -m src.demo_bad_data
                 python -m src.data_validation --data data/corrupted_demo.csv
Check the exit code of the last command:   echo $LASTEXITCODE   (0 = passed, 1 = failed)

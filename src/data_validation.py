"""Data quality gate: run before training. Exit code 1 if the data is not fit for use."""
import argparse
import sys

import pandas as pd
import yaml

NUMERIC_RANGES = {            # column: (minimum allowed, maximum allowed or None)
    "Age": (18, 100),
    "Credit_Limit": (1, None),
    "Current_Balance": (0, None),
    "Monthly_Spend": (0, None),
    "Num_Transactions_Monthly": (0, None),
    "Payment_Delay_Days": (0, None),
    "Tenure_Months": (0, None),
}
REQUIRED = ["Customer_ID", "Income_Bracket"] + list(NUMERIC_RANGES)


def validate(df, cfg):
    """Return a list of problems found (empty list = data is fine)."""
    v, data = cfg["validation"], cfg["data"]
    target = data["target"]
    problems = []
    missing = [c for c in REQUIRED + [target] if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if len(df) < v["min_rows"]:
        problems.append(f"Only {len(df)} rows, need at least {v['min_rows']}")
    if df[data["id_column"]].duplicated().any():
        problems.append(f"{int(df[data['id_column']].duplicated().sum())} duplicate customer IDs")
    for col, frac in df.isnull().mean().items():
        if frac > v["max_null_fraction"]:
            problems.append(f"{col}: {frac:.1%} missing (limit {v['max_null_fraction']:.0%})")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        s = df[col].dropna()
        if (s < lo).any():
            problems.append(f"{col}: {int((s < lo).sum())} values below {lo}")
        if hi is not None and (s > hi).any():
            problems.append(f"{col}: {int((s > hi).sum())} values above {hi}")
    bad = set(df["Income_Bracket"].dropna().unique()) - set(v["income_levels"])
    if bad:
        problems.append(f"Unknown Income_Bracket values: {sorted(bad)}")
    if not set(df[target].dropna().unique()) <= {0, 1}:
        problems.append(f"{target} must contain only 0 and 1")
    else:
        rate = df[target].mean()
        lo, hi = v["target_rate_range"]
        if not lo <= rate <= hi:
            problems.append(f"High-risk rate {rate:.1%} outside expected {lo:.0%}-{hi:.0%}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV to check (default: data.path in config.yaml)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    path = args.data or cfg["data"]["path"]
    df = pd.read_csv(path)
    target = cfg["data"]["target"]
    print(f"File     : {path}")
    print(f"Rows     : {len(df)}   Columns: {df.shape[1]}")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print("Missing  :", ", ".join(f"{c}={n}" for c, n in nulls.items()) or "none")
    if target in df.columns:
        print(f"High risk: {df[target].mean():.1%}")
    problems = validate(df, cfg)
    if problems:
        print("DATA VALIDATION FAILED:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("DATA VALIDATION PASSED")


if __name__ == "__main__":
    main()

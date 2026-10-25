"""Model quality gate: exit code 1 (fails the CI build) if the model is not good enough."""
import argparse
import json
import os
import sys

import yaml

CHECKS = [("AUC", "auc", "min_auc"), ("Recall", "recall", "min_recall"),
          ("Precision", "precision", "min_precision"), ("KS", "ks", "min_ks")]


def check(metrics, gate, baseline=None):
    """Return a list of failure messages (empty list = gate passed)."""
    failures = []
    for label, key, thr in CHECKS:
        if thr in gate and metrics[key] < gate[thr]:
            failures.append(f"{label} {metrics[key]} < {gate[thr]}")
    if baseline:
        drop = baseline["auc"] - metrics["auc"]
        if drop > gate["max_auc_drop_vs_baseline"]:
            failures.append(f"AUC dropped {drop:.4f} vs baseline {baseline['auc']}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="models/metrics.json")
    ap.add_argument("--baseline", default="baseline_metrics.json")
    ap.add_argument("--min-auc", type=float, help="override min_auc (handy for demos)")
    ap.add_argument("--min-recall", type=float, help="override min_recall")
    ap.add_argument("--min-precision", type=float, help="override min_precision")
    ap.add_argument("--min-ks", type=float, help="override min_ks")
    args = ap.parse_args()

    gate = yaml.safe_load(open("config.yaml"))["gate"]
    for key in ("min_auc", "min_recall", "min_precision", "min_ks"):
        val = getattr(args, key)
        if val is not None:
            gate[key] = val
    metrics = json.load(open(args.metrics))
    baseline = json.load(open(args.baseline)) if os.path.exists(args.baseline) else None

    print(f"{'Metric':<11}{'Value':>8}  {'Required':>10}  Status")
    for label, key, thr in CHECKS:
        ok = metrics[key] >= gate[thr]
        print(f"{label:<11}{metrics[key]:>8.4f}  {'>= ' + str(gate[thr]):>10}  "
              f"{'PASS' if ok else 'FAIL'}")
    if baseline:
        print(f"Baseline AUC {baseline['auc']} -> new AUC {metrics['auc']} "
              f"(change {metrics['auc'] - baseline['auc']:+.4f})")
    failures = check(metrics, gate, baseline)
    if failures:
        print("QUALITY GATE FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()

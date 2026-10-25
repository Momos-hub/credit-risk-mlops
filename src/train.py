"""Train the credit-risk model, evaluate it, and save model + metrics."""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_xy


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def make_pipeline(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def train_model(df, cfg):
    """Fit on 80% of df, score on the other 20%. Returns (model, metrics, predictions)."""
    seed = cfg["model"]["random_state"]
    X, y = build_xy(df, cfg["data"]["target"], cfg["data"]["id_column"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["model"]["test_size"], random_state=seed, stratify=y)
    model = make_pipeline(seed).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= cfg["model"]["threshold"]).astype(int)
    cv = cross_val_score(make_pipeline(seed), X, y, scoring="roc_auc",
                         cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    metrics = {
        "auc": round(float(roc_auc_score(yte, proba)), 4),
        "recall": round(float(recall_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred)), 4),
        "ks": round(ks_statistic(yte, proba), 4),
        "cv_auc": round(float(cv.mean()), 4),
        "rows": int(len(df)),
    }
    preds = pd.DataFrame({"y_true": yte.values, "proba": proba, "pred": pred})
    return model, metrics, preds, list(X.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV to train on (default: data.path in config.yaml)")
    ap.add_argument("--out", default="models", help="output folder")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    path = args.data or cfg["data"]["path"]
    df = pd.read_csv(path)
    model, metrics, preds, cols = train_model(df, cfg)
    metrics["data_file"] = os.path.basename(path)

    os.makedirs(args.out, exist_ok=True)
    coefs = model.named_steps["clf"].coef_[0]
    json.dump({c: round(float(k), 4) for c, k in zip(cols, coefs)},
              open(os.path.join(args.out, "coefficients.json"), "w"), indent=2)
    preds.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    joblib.dump(model, os.path.join(args.out, "model.pkl"))
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"Trained on {metrics['data_file']} ({metrics['rows']} rows)")
    print("Metrics:", {k: v for k, v in metrics.items() if k not in ("rows", "data_file")})


if __name__ == "__main__":
    main()

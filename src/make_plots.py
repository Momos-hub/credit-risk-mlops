"""Create the report figures (run after src.train)."""
import json

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

GREEN, RUST, MUSTARD, CHAR, CREAM, SAGE = (
    "#2F5D50", "#C65D3B", "#D9A441", "#2B2B2B", "#F5EFE6", "#9DB5A0")
OUT = "docs/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": CHAR,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(OUT + name, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close()


def pipeline_diagram():
    steps = [("Push code", SAGE), ("Lint", SAGE), ("Unit\ntests", SAGE),
             ("DATA\nGATE", MUSTARD), ("Train\nmodel", SAGE), ("MODEL\nGATE", RUST),
             ("Publish", GREEN)]
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (txt, col) in enumerate(steps):
        x = 0.2 + i * 2.05
        ax.add_patch(FancyBboxPatch((x, 0.5), 1.6, 1.0, boxstyle="round,pad=0.05",
                                    fc=col, ec=CHAR, lw=1.2))
        ax.text(x + 0.8, 1.0, txt, ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if col in (RUST, GREEN) else CHAR)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.0, 1.0), xytext=(x + 1.68, 1.0),
                        arrowprops=dict(arrowstyle="-|>", color=CHAR, lw=1.5))
    ax.text(0.2 + 3 * 2.05 + 0.8, 0.15, "bad data stops here", ha="center", fontsize=8.5,
            color=CHAR, style="italic")
    ax.text(0.2 + 5 * 2.05 + 0.8, 0.15, "weak model stops here", ha="center", fontsize=8.5,
            color=RUST, style="italic")
    save("pipeline.png")


def data_versions(v1, v2, target):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].bar(["v1", "v2"], [len(v1), len(v2)], color=[SAGE, GREEN], width=0.5)
    for i, n in enumerate([len(v1), len(v2)]):
        axes[0].text(i, n + 20, str(n), ha="center")
    axes[0].set_title("Customers (rows)", color=CHAR, fontsize=10)
    axes[0].set_ylim(0, 1300)
    rates = [v1[target].mean() * 100, v2[target].mean() * 100]
    axes[1].bar(["v1", "v2"], rates, color=[SAGE, GREEN], width=0.5)
    for i, r in enumerate(rates):
        axes[1].text(i, r + 0.5, f"{r:.1f}%", ha="center")
    axes[1].set_title("High-risk customers", color=CHAR, fontsize=10)
    axes[1].set_ylim(0, 30)
    save("data_versions.png")


def missing_plot(df):
    nulls = df.isnull().mean().mul(100)
    nulls = nulls[nulls > 0].sort_values()
    plt.figure(figsize=(5.2, 2.6))
    plt.barh(nulls.index, nulls.values, color=RUST, height=0.5)
    for i, v in enumerate(nulls.values):
        plt.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.axvline(10, color=CHAR, ls="--", lw=1.5)
    plt.text(10.15, 1.6, "gate limit\n10%", fontsize=8)
    plt.xlim(0, 13)
    plt.xlabel("Missing values (%)")
    plt.title("Columns with missing data (v2)", color=CHAR, fontsize=10)
    save("missing.png")


def utilization_plot(df):
    u = (df.Current_Balance / df.Credit_Limit).dropna()
    hr = df.loc[u.index, "High_Risk_Customer"]
    plt.figure(figsize=(5.4, 3.5))
    bins = np.linspace(0, 1.4, 29)
    plt.hist(u[hr == 0].clip(upper=1.4), bins, color=GREEN, alpha=0.85, label="Not high risk")
    plt.hist(u[hr == 1].clip(upper=1.4), bins, color=RUST, alpha=0.85, label="High risk")
    plt.xlabel("Credit utilisation (balance / limit)")
    plt.ylabel("Customers")
    plt.title("High-risk customers use far more of their limit", color=CHAR, fontsize=10)
    plt.legend(frameon=False)
    save("utilization.png")


def roc_plot(pr):
    fpr, tpr, _ = roc_curve(pr.y_true, pr.proba)
    ks = np.argmax(tpr - fpr)
    plt.figure(figsize=(4.8, 4.2))
    plt.plot(fpr, tpr, color=GREEN, lw=2.4, label="Model")
    plt.plot([0, 1], [0, 1], "--", color="#888", label="Random guess")
    plt.vlines(fpr[ks], fpr[ks], tpr[ks], color=RUST, lw=2, label="Max gap (KS)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)", color=CHAR)
    plt.legend(frameon=False, loc="lower right")
    save("roc.png")


def confusion(pr, thr):
    cm = confusion_matrix(pr.y_true, pr.pred)
    cmap = LinearSegmentedColormap.from_list("cr", [CREAM, MUSTARD, RUST])
    plt.figure(figsize=(4.4, 3.9))
    plt.imshow(cm, cmap=cmap)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14, color=CHAR)
    plt.xticks([0, 1], ["Pred: safe", "Pred: high risk"])
    plt.yticks([0, 1], ["Actual: safe", "Actual: high risk"])
    plt.title(f"Confusion matrix (threshold {thr})", color=CHAR, fontsize=10)
    for s in plt.gca().spines.values():
        s.set_visible(False)
    save("confusion.png")


def coef_plot():
    coefs = json.load(open("models/coefficients.json"))
    items = sorted(coefs.items(), key=lambda kv: kv[1])
    plt.figure(figsize=(5.6, 3.9))
    plt.barh([k for k, _ in items], [v for _, v in items],
             color=[RUST if v < 0 else GREEN for _, v in items])
    plt.axvline(0, color=CHAR, lw=1)
    plt.xlabel("Coefficient (right raises risk, left lowers it)")
    plt.title("What drives the prediction", color=CHAR)
    save("coefficients.png")


def gate_bars(metrics, gate, name, title, override=None):
    keys = [("AUC", "auc", "min_auc"), ("Recall", "recall", "min_recall"),
            ("Precision", "precision", "min_precision"), ("KS", "ks", "min_ks")]
    vals = [metrics[k] for _, k, _ in keys]
    mins = [gate[t] if not (override and t in override) else override[t] for _, _, t in keys]
    x = np.arange(len(keys))
    plt.figure(figsize=(5.8, 3.8))
    plt.bar(x, vals, 0.5, color=[GREEN if v >= m else RUST for v, m in zip(vals, mins)])
    for i, m in enumerate(mins):
        plt.hlines(m, i - 0.38, i + 0.38, color=CHAR, lw=2.2, linestyles="--")
        plt.text(i, vals[i] + 0.02, f"{vals[i]:.2f}", ha="center", fontsize=10)
    plt.xticks(x, [k[0] for k in keys])
    plt.ylim(0, 1.12)
    plt.title(title, color=RUST if override else CHAR, fontsize=10)
    save(name)


def v1_vs_v2(base, new):
    names, keys = ["AUC", "Recall", "Precision", "KS"], ["auc", "recall", "precision", "ks"]
    x = np.arange(4)
    plt.figure(figsize=(5.8, 3.7))
    plt.bar(x - 0.19, [base[k] for k in keys], 0.38, color=SAGE, label="Baseline (v1 data)")
    plt.bar(x + 0.19, [new[k] for k in keys], 0.38, color=GREEN, label="New model (v2 data)")
    for i, k in enumerate(keys):
        plt.text(i - 0.19, base[k] + 0.015, f"{base[k]:.2f}", ha="center", fontsize=8)
        plt.text(i + 0.19, new[k] + 0.015, f"{new[k]:.2f}", ha="center", fontsize=8)
    plt.xticks(x, names)
    plt.ylim(0, 1.2)
    plt.legend(frameon=False, ncol=2, loc="upper center")
    plt.title("New model compared with the baseline", color=CHAR, fontsize=10)
    save("v1_vs_v2.png")


def tradeoff(pr, thr):
    ths = np.linspace(0.1, 0.9, 33)
    y = pr.y_true.values
    rec = [((pr.proba >= t) & (y == 1)).sum() / (y == 1).sum() for t in ths]
    prec = [((pr.proba >= t) & (y == 1)).sum() / max((pr.proba >= t).sum(), 1) for t in ths]
    plt.figure(figsize=(5.4, 3.7))
    plt.plot(ths, rec, color=GREEN, lw=2.4, label="Recall")
    plt.plot(ths, prec, color=RUST, lw=2.4, label="Precision")
    plt.axvline(thr, color=MUSTARD, lw=2, linestyle="--")
    plt.text(thr + 0.01, 0.45, f"chosen: {thr}", color=CHAR, fontsize=9)
    plt.xlabel("Decision threshold")
    plt.title("Recall vs precision trade-off", color=CHAR)
    plt.legend(frameon=False)
    save("tradeoff.png")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    v1 = pd.read_csv("data/credit_card_customers_v1.csv")
    v2 = pd.read_csv("data/credit_card_customers_v2.csv")
    pr = pd.read_csv("models/predictions.csv")
    metrics = json.load(open("models/metrics.json"))
    base = json.load(open("baseline_metrics.json"))
    gate, thr = cfg["gate"], cfg["model"]["threshold"]
    pipeline_diagram()
    data_versions(v1, v2, cfg["data"]["target"])
    missing_plot(v2)
    utilization_plot(v2)
    roc_plot(pr)
    confusion(pr, thr)
    coef_plot()
    gate_bars(metrics, gate, "gate_pass.png", "Model metrics vs gate thresholds")
    gate_bars(metrics, gate, "gate_fail.png", "Demo: min_auc raised to 0.99 -> gate fails",
              override={"min_auc": 0.99})
    v1_vs_v2(base, metrics)
    tradeoff(pr, thr)
    print("figures saved in", OUT)

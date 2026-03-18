"""
Evaluation helpers, cost-matrix analysis, and submission export.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)


# ── Cost parameters (adjust to match real business values) ──────────────────
# Cost of missing a churner (false negative): lost revenue + future LTV
COST_FN = 500   # $ per missed churner
# Cost of a wasted retention offer (false positive): discount + agent time
COST_FP = 50    # $ per wrongly flagged customer
# Value of successfully retaining a churner (true positive)
VALUE_TP = 300  # $ net saved after retention spend


def evaluate(y_true, y_pred_proba, threshold: float = 0.5, model_name: str = "Model"):
    """Print classification report, ROC-AUC, and cost analysis."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Threshold: {threshold}")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))

    total_cost = (fn * COST_FN) + (fp * COST_FP) - (tp * VALUE_TP)
    print(f"-- Cost Analysis (threshold={threshold}) --")
    print(f"  True Positives  (retained churners) : {tp:>6,}  → saves ${tp * VALUE_TP:>10,.0f}")
    print(f"  False Positives (wasted offers)      : {fp:>6,}  → costs ${fp * COST_FP:>10,.0f}")
    print(f"  False Negatives (missed churners)    : {fn:>6,}  → costs ${fn * COST_FN:>10,.0f}")
    print(f"  Net business impact                  :         ${-total_cost:>+10,.0f}")

    return auc


def plot_roc_curves(models: dict, y_true):
    """
    Plot ROC curves for multiple models on one axis.
    models: {name: y_pred_proba}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in models.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
    """Annotated heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    return fig


def find_optimal_threshold(y_true, y_pred_proba, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """
    Sweep thresholds and return the one that minimises total business cost.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    best_t = thresholds[np.argmin(costs)]
    return float(best_t)


def make_submission(test_ids: pd.Series, y_pred_proba: np.ndarray, path: str):
    """Save Kaggle-format submission CSV."""
    sub = pd.DataFrame({"id": test_ids, "Churn": y_pred_proba})
    sub.to_csv(path, index=False)
    print(f"Submission saved → {path}  ({len(sub):,} rows)")


def plot_threshold_analysis(
    y_true,
    y_pred_proba,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    value_tp: float = VALUE_TP,
    model_name: str = "Model",
):
    """
    Dual-panel figure:
      Left:  Precision & Recall vs decision threshold
      Right: Net business value ($) vs decision threshold, annotated at the optimum
    Returns the figure.
    """
    from sklearn.metrics import precision_score, recall_score

    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, net_values = [], [], []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        net_values.append(tp * value_tp - fp * cost_fp - fn * cost_fn)

    optimal_idx = int(np.argmax(net_values))
    optimal_t = float(thresholds[optimal_idx])
    max_val = net_values[optimal_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: Precision & Recall ────────────────────────────────────────────
    ax1.plot(thresholds, precisions, label="Precision", color="steelblue", lw=2)
    ax1.plot(thresholds, recalls, label="Recall", color="tomato", lw=2)
    ax1.axvline(optimal_t, color="gray", linestyle="--", lw=1.5,
                label=f"Optimal t = {optimal_t:.2f}")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{model_name} — Precision & Recall vs Threshold")
    ax1.legend()
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0, 1.05)

    # ── Right: Net business value ────────────────────────────────────────────
    net_k = [v / 1_000 for v in net_values]
    ax2.plot(thresholds, net_k, color="seagreen", lw=2)
    ax2.axvline(optimal_t, color="gray", linestyle="--", lw=1.5)
    ax2.fill_between(thresholds, net_k, alpha=0.12, color="seagreen")
    # annotation offset: push text right if near left edge
    text_x = optimal_t + 0.08 if optimal_t < 0.7 else optimal_t - 0.25
    ax2.annotate(
        f"Max: ${max_val:,.0f}\nt = {optimal_t:.2f}",
        xy=(optimal_t, max_val / 1_000),
        xytext=(text_x, max_val / 1_000 * 0.88),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Net Business Value ($K, validation set)")
    ax2.set_title(f"{model_name} — Business Value vs Threshold")
    ax2.set_xlim(0.05, 0.95)

    plt.tight_layout()
    return fig


def business_cost_table(
    models_probas: dict,
    y_true,
    threshold: float = 0.5,
    per_n: int = 10_000,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    value_tp: float = VALUE_TP,
) -> pd.DataFrame:
    """
    Compare models by business economics, normalised to `per_n` customers.
    Returns a DataFrame with one row per model.
    """
    n = len(y_true)
    scale = per_n / n
    rows = []
    baseline_net = None

    for name, proba in models_probas.items():
        y_pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net = (tp * value_tp - fp * cost_fp - fn * cost_fn) * scale

        if baseline_net is None:
            baseline_net = net

        rows.append({
            "Model": name,
            f"Retained / {per_n:,}": int(round(tp * scale)),
            f"Wasted offers / {per_n:,}": int(round(fp * scale)),
            f"Missed churners / {per_n:,}": int(round(fn * scale)),
            f"Net value / {per_n:,}": f"${net:+,.0f}",
            "vs. Baseline": f"${net - baseline_net:+,.0f}",
        })

    return pd.DataFrame(rows)
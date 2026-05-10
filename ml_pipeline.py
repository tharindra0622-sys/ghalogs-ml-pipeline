"""
ml_pipeline.py
==============
Full ML pipeline for GHALogs dataset (GitHub Actions runs).
Target  : metadata_conclusion  (success / failure)
Features: all log_* columns (22 features)

Outputs written to outputs/
  - predictions.csv         : per-row predictions + probabilities
  - metrics.json            : accuracy, precision, recall, F1, ROC-AUC
  - feature_importance.csv  : ranked feature importances
  - summary_report.md       : human-readable daily summary
  - confusion_matrix.png    : visualisation
  - roc_curve.png           : visualisation
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "data/runs_200_2.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

LOG_FEATURES = [
    "log_num_jobs", "log_total_steps", "log_total_duration_sec",
    "log_shell_steps", "log_action_steps", "log_error_steps",
    "log_total_lines", "log_has_linux", "log_has_macos", "log_has_windows",
    "log_num_os_types", "log_early3_total_dur", "log_early3_max_dur",
    "log_early3_min_dur", "log_early3_shell_count", "log_early3_action_count",
    "log_early3_error_count", "log_early3_avg_dur", "log_error_rate",
    "log_shell_ratio", "log_avg_step_dur", "log_max_step_dur"
]
TARGET_COL = "metadata_conclusion"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/6] Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame):
    print("[2/6] Preprocessing ...")
    df = df.copy()

    # Keep only rows with a known conclusion
    df = df[df[TARGET_COL].isin(["success", "failure"])].reset_index(drop=True)

    # Binary target: 1 = failure
    df["target"] = (df[TARGET_COL] == "failure").astype(int)

    X = df[LOG_FEATURES].fillna(0)
    y = df["target"]

    # Basic feature engineering
    X = X.copy()
    X["log_error_x_duration"] = X["log_error_rate"] * X["log_total_duration_sec"]
    X["log_steps_per_job"]    = X["log_total_steps"] / (X["log_num_jobs"].replace(0, 1))
    X["log_early_ratio"]      = X["log_early3_total_dur"] / (X["log_total_duration_sec"].replace(0, 1))

    print(f"      X shape: {X.shape}  |  failure rate: {y.mean()*100:.1f}%")
    return df, X, y


def balance_classes(X_train, y_train):
    """Upsample minority class (failure) to balance training set."""
    df_train = X_train.copy()
    df_train["_y"] = y_train.values

    majority = df_train[df_train["_y"] == 0]
    minority = df_train[df_train["_y"] == 1]

    if len(minority) == 0:
        return X_train, y_train

    minority_upsampled = resample(
        minority, replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE
    )
    df_balanced = pd.concat([majority, minority_upsampled]).sample(
        frac=1, random_state=RANDOM_STATE
    )
    return df_balanced.drop("_y", axis=1), df_balanced["_y"]


def train_and_evaluate(X, y):
    print("[3/6] Training models ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=RANDOM_STATE
            ))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    results = {}
    best_f1, best_name, best_model = -1, None, None

    for name, pipe in models.items():
        pipe.fit(X_train_bal, y_train_bal)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(
            pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1", n_jobs=-1
        )

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.0, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std":  round(cv_scores.std(),  4),
        }
        results[name] = metrics
        print(f"      {name:20s}  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}  CV-F1={metrics['cv_f1_mean']:.3f}±{metrics['cv_f1_std']:.3f}")

        if metrics["f1"] > best_f1:
            best_f1, best_name, best_model = metrics["f1"], name, pipe

    print(f"      Best model: {best_name} (F1={best_f1:.3f})")
    return best_model, best_name, results, X_train, X_test, y_train, y_test


def save_predictions(model, X_test, y_test, df, X):
    print("[4/6] Saving predictions ...")

    # Full dataset predictions
    y_all_pred  = model.predict(X)
    y_all_proba = model.predict_proba(X)[:, 1]

    pred_df = df[[
        "_id", "repository_name", "workflow_path",
        "run_number", TARGET_COL
    ]].copy()
    pred_df["predicted_conclusion"] = ["failure" if p == 1 else "success" for p in y_all_pred]
    pred_df["failure_probability"]  = np.round(y_all_proba, 4)
    pred_df["correct"]              = (pred_df[TARGET_COL] == pred_df["predicted_conclusion"])

    pred_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"      Saved {len(pred_df)} predictions → {OUTPUT_DIR}/predictions.csv")
    return pred_df


def save_feature_importance(model, feature_names):
    print("[5/6] Saving feature importance ...")
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": np.round(importances, 6)
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1

    fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    print(f"      Saved → {OUTPUT_DIR}/feature_importance.csv")
    return fi_df


def save_plots(model, X_test, y_test, fi_df):
    # -- Confusion matrix --
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["success", "failure"])
    ax.set_yticklabels(["success", "failure"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    # -- ROC Curve --
    ax2 = axes[1]
    if len(y_test.unique()) > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax2.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax2.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -- Feature importance bar chart --
    top_n = min(15, len(fi_df))
    fig, ax = plt.subplots(figsize=(10, 6))
    fi_top = fi_df.head(top_n).iloc[::-1]
    bars = ax.barh(fi_top["feature"], fi_top["importance"], color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, fi_top["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved plots → {OUTPUT_DIR}/")


def save_metrics(all_results, best_name, df, y, pred_df):
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(df)
    failures = int((df[TARGET_COL] == "failure").sum())
    successes = total - failures

    output = {
        "run_date":     run_date,
        "dataset_rows": total,
        "failure_count": failures,
        "success_count": successes,
        "failure_rate":  round(failures / total * 100, 2),
        "best_model":   best_name,
        "models":       all_results
    }

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"      Saved → {OUTPUT_DIR}/metrics.json")
    return output


def save_summary_report(metrics_data, fi_df, best_name):
    run_date = metrics_data["run_date"]
    best     = metrics_data["models"][best_name]
    fi_top5  = fi_df.head(5)[["rank", "feature", "importance"]].to_string(index=False)

    md = f"""# GitHub Actions ML Pipeline — Daily Report

**Run date:** {run_date}
**Best model:** {best_name}

---

## Dataset Overview

| Metric | Value |
|---|---|
| Total runs | {metrics_data['dataset_rows']:,} |
| Successful runs | {metrics_data['success_count']:,} |
| Failed runs | {metrics_data['failure_count']:,} |
| Failure rate | {metrics_data['failure_rate']}% |

---

## Best Model Performance ({best_name})

| Metric | Score |
|---|---|
| Accuracy | {best['accuracy']} |
| Precision | {best['precision']} |
| Recall | {best['recall']} |
| F1 Score | {best['f1']} |
| ROC-AUC | {best['roc_auc']} |
| CV F1 (5-fold) | {best['cv_f1_mean']} ± {best['cv_f1_std']} |

---

## All Models Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
"""
    for mname, m in metrics_data["models"].items():
        md += f"| {mname} | {m['accuracy']} | {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} |\n"

    md += f"""
---

## Top 5 Predictive Features

```
{fi_top5}
```

---

## Output Files

| File | Description |
|---|---|
| `predictions.csv` | Per-run predictions and failure probabilities |
| `metrics.json` | Full metrics for all models |
| `feature_importance.csv` | All features ranked by importance |
| `model_evaluation.png` | Confusion matrix + ROC curve |
| `feature_importance.png` | Feature importance bar chart |
| `summary_report.md` | This report |

---
*Generated automatically by ml_pipeline.py via GitHub Actions*
"""
    with open(f"{OUTPUT_DIR}/summary_report.md", "w") as f:
        f.write(md)
    print(f"      Saved → {OUTPUT_DIR}/summary_report.md")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  GHALogs ML Pipeline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    df             = load_data(INPUT_CSV)
    df, X, y       = preprocess(df)
    model, best_name, all_results, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    pred_df        = save_predictions(model, X_test, y_test, df, X)
    fi_df          = save_feature_importance(model, list(X.columns))

    print("[5b/6] Saving plots ...")
    save_plots(model, X_test, y_test, fi_df)

    print("[6/6] Saving metrics and summary report ...")
    metrics_data = save_metrics(all_results, best_name, df, y, pred_df)
    save_summary_report(metrics_data, fi_df, best_name)

    print("=" * 60)
    print("  Pipeline complete. All outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()

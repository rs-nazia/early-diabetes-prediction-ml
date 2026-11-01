"""
Model Evaluation, Benchmark Execution, and Visualization Generator.
Reproduces and expands upon evaluation results of Alzboon et al. (2023).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

from dataset.data_loader import load_dataset, preprocess_data, get_scaled_splits
from diabetes_pipeline.feature_importance import compute_feature_rankings
from diabetes_pipeline.models import get_classifier_suite


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Calculate clinical classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
    }


def plot_model_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Generate high-resolution grouped comparison bar chart."""
    plt.figure(figsize=(13, 6))
    sns.set_theme(style="whitegrid")

    metrics = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    melted = results_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    palette = sns.color_palette("mako", n_colors=len(metrics))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette)

    plt.title("Comparative Performance of Machine Learning Classifiers (Early Diabetes)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Classification Algorithm", fontsize=12, labelpad=10)
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.legend(title="Evaluation Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(conf_matrices: Dict[str, np.ndarray], output_path: Path) -> None:
    """Plot confusion matrices in a clean grid layout."""
    n_models = len(conf_matrices)
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    labels = ["Healthy (0)", "Diabetic (1)"]

    for idx, (name, cm) in enumerate(conf_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name, fontsize=11, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Confusion Matrix Grid across Evaluated Models", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(roc_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]], output_path: Path) -> None:
    """Plot multi-model ROC Curves."""
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")

    for name, (fpr, tpr, auc_val) in roc_dict.items():
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Benchmark Curves", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_full_benchmark(
    dataset_path: str = "dataset/diabetes.csv",
    output_dir: str = "visualizations",
    random_seed: int = 42
) -> pd.DataFrame:
    """Runs data loading, feature ranking, model tuning, evaluation, and chart generation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    raw_df = load_dataset(dataset_path)

    # 2. Compute and Save Feature Rankings (Table I reproduction)
    rankings_df = compute_feature_rankings(raw_df)
    rankings_df.to_csv(out / "feature_importance_rankings.csv", index=False)

    # 3. Clean and Scale Data
    X, y, clean_df = preprocess_data(raw_df, impute_missing_zeros=True)
    X_train, X_test, y_train, y_test, scaler = get_scaled_splits(X, y, test_size=0.30, random_state=random_seed)

    suite = get_classifier_suite(random_seed)
    results = []
    cms = {}
    roc_dict = {}

    print(f"[*] Commencing benchmark evaluation across {len(suite)} algorithms...")

    for name, (base_model, param_grid) in suite.items():
        t0 = time.time()
        print(f"  -> Training & tuning: {name}")

        if param_grid:
            grid = GridSearchCV(base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model = base_model.fit(X_train, y_train)
            best_params = {}

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-decision))
        else:
            y_proba = y_pred

        metrics = compute_metrics(y_test, y_pred, y_proba)
        elapsed = round(time.time() - t0, 3)

        results.append({
            "Model": name,
            "AUC": round(metrics["AUC"], 4),
            "Accuracy": round(metrics["Accuracy"], 4),
            "F1": round(metrics["F1"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "Specificity": round(metrics["Specificity"], 4),
            "Fit_Time_s": elapsed,
            "Best_Parameters": json.dumps(best_params)
        })

        cms[name] = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_dict[name] = (fpr, tpr, metrics["AUC"])

    summary_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False).reset_index(drop=True)
    summary_df.to_csv(out / "benchmark_metrics.csv", index=False)

    # 4. Generate Visualizations
    plot_model_comparison(summary_df, out / "model_comparison_bar.png")
    plot_confusion_matrices(cms, out / "confusion_matrices.png")
    plot_roc_curves(roc_dict, out / "roc_curves.png")

    # 5. Save run metadata
    with open(dataset_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "dataset_file": dataset_path,
        "dataset_sha256": file_hash,
        "total_instances": len(raw_df),
        "features": len(raw_df.columns) - 1,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "seed": random_seed,
        "models_evaluated": len(suite)
    }

    with open(out / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[+] Benchmark completed successfully. Artifacts saved to:", out.resolve())
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diabetes prediction algorithms.")
    parser.add_argument("--data", default="dataset/diabetes.csv", help="Path to diabetes CSV dataset.")
    parser.add_argument("--out", default="visualizations", help="Output directory for results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    run_full_benchmark(args.data, args.out, args.seed)

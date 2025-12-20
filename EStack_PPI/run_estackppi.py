#!/usr/bin/env python3
"""
EStack-PPI: ESM2-based Protein-Protein Interaction Prediction

A simplified version of HybridStack-PPI using only ESM2 embeddings.
This implementation INHERITS from HybridStackPPI modules.

Usage:
    python run_estackppi.py --dataset yeast   # Run on DIP-Yeast
    python run_estackppi.py --dataset human   # Run on DIP-Human
    python run_estackppi.py --dataset all     # Run on both datasets
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports from HybridStackPPI
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# INHERIT FROM HYBRIDSTACKPPI MODULES
# ============================================================================
from pipelines.builders import create_esm_stacking_pipeline
from pipelines.metrics import display_full_metrics, print_paper_style_results


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute all classification metrics."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }


def save_combined_roc(all_fold_data: list, save_path: str, dataset_name: str):
    """Save single image with ROC curves for all folds."""
    plt.figure(figsize=(8, 7), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(all_fold_data)))
    
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for i, data in enumerate(all_fold_data):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.5, 
                 label=f"Fold {i+1} (AUC = {roc_auc:.4f})")
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color="#1f77b4", lw=3,
             label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.7)
    
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves (All Folds) - {dataset_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_combined_pr(all_fold_data: list, save_path: str, dataset_name: str):
    """Save single image with PR curves for all folds."""
    plt.figure(figsize=(8, 7), dpi=150)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(all_fold_data)))
    
    pr_aucs = []
    for i, data in enumerate(all_fold_data):
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        pr_auc = average_precision_score(data["y_true"], data["y_proba"])
        pr_aucs.append(pr_auc)
        plt.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.5,
                 label=f"Fold {i+1} (AUPRC = {pr_auc:.4f})")
    
    mean_pr_auc = np.mean(pr_aucs)
    std_pr_auc = np.std(pr_aucs)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves (All Folds) - {dataset_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.text(0.98, 0.02, f"Mean AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f}",
             transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_cross_validation(X: np.ndarray, y: np.ndarray, dataset_name: str, 
                         results_dir: str, n_splits: int = 5, n_jobs: int = -1):
    """Run 5-fold cross-validation using INHERITED HybridStackPPI pipeline."""
    print(f"\n{'='*70}")
    print(f"🚀 EStack-PPI: {dataset_name} Dataset")
    print(f"{'='*70}")
    print(f"📊 Data shape: X={X.shape}, y={y.shape}")
    
    X_df = pd.DataFrame(X, columns=[f"esm_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_metrics = []
    all_fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y_series), start=1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]
        
        # ================================================
        # USE PIPELINE FROM HybridStackPPI (pipelines.builders)
        # ================================================
        model = create_esm_stacking_pipeline(n_jobs=n_jobs)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics["Fold"] = fold_idx
        all_metrics.append(metrics)
        
        # Print ALL metrics for this fold
        print(f"  ✅ Fold {fold_idx} Metrics:")
        for k, v in metrics.items():
            if k != "Fold":
                print(f"    - {k:20s}: {v*100:.2f}%" if v <= 1 else f"    - {k:20s}: {v}")
        
        # Save fold data for combined plot
        all_fold_data.append({
            "y_true": y_val.values,
            "y_proba": y_proba,
        })
    
    # Save combined curves
    roc_all_path = os.path.join(results_dir, "roc_all_folds.png")
    pr_all_path = os.path.join(results_dir, "pr_all_folds.png")
    
    print(f"\n🎨 Generating combined visualization curves...")
    save_combined_roc(all_fold_data, roc_all_path, dataset_name)
    save_combined_pr(all_fold_data, pr_all_path, dataset_name)
    print(f"  💾 Saved: {roc_all_path}")
    print(f"  💾 Saved: {pr_all_path}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(results_dir, "cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"💾 Saved full metric table: {metrics_path}")
    
    # Print summary using HybridStackPPI's print_paper_style_results
    print(f"\n{'='*70}")
    print(f"📊 MEAN PERFORMANCE SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    
    mean_metrics = metrics_df.drop(columns=["Fold"]).mean()
    std_metrics = metrics_df.drop(columns=["Fold"]).std()
    
    for metric in mean_metrics.index:
        print(f"  {metric:12s}: {mean_metrics[metric]*100:.2f}% ± {std_metrics[metric]*100:.2f}%")
    
    return metrics_df


def main():
    parser = argparse.ArgumentParser(description="EStack-PPI: ESM2-based PPI Prediction (Inherits from HybridStackPPI)")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["yeast", "human", "all"],
                       help="Dataset to run: yeast, human, or all")
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 for all cores)")
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    results_base = script_dir / "results"
    
    # Dataset configurations
    datasets = {
        "yeast": {
            "X_path": data_dir / "ppis" / "X_esm2.npy",
            "y_path": data_dir / "ppis" / "y.npy",
            "results_dir": results_base / "yeast",
            "name": "DIP-Yeast",
        },
        "human": {
            "X_path": data_dir / "feats" / "X_esm2.npy",
            "y_path": data_dir / "feats" / "y_esm2.npy",
            "results_dir": results_base / "human",
            "name": "DIP-Human",
        },
    }
    
    # Select datasets to run
    if args.dataset == "all":
        selected = ["yeast", "human"]
    else:
        selected = [args.dataset]
    
    all_results = {}
    
    for dataset_key in selected:
        config = datasets[dataset_key]
        
        # Ensure results directory exists
        config["results_dir"].mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"\n📂 Loading {config['name']} data...")
        X = np.load(config["X_path"])
        y = np.load(config["y_path"])
        
        # Run cross-validation
        metrics_df = run_cross_validation(
            X=X, y=y,
            dataset_name=config["name"],
            results_dir=str(config["results_dir"]),
            n_splits=5,
            n_jobs=args.n_jobs,
        )
        all_results[dataset_key] = metrics_df
    
    print(f"\n{'='*70}")
    print("🎉 EStack-PPI EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    main()

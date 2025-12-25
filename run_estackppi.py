#!/usr/bin/env python3
"""
E-StackPPI: ESM2-based Protein-Protein Interaction Prediction

Khung dự đoán tương tác Protein-Protein dựa trên mô hình ngôn ngữ protein 
và kiến trúc học máy xếp tầng tích hợp chọn lọc đặc trưng.

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
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
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.selectors import CumulativeFeatureSelector


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_fasta(fasta_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA file."""
    sequences = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line.split()[0][1:]
                sequences[header] = ""
            elif header:
                sequences[header] += line
    return sequences


def load_pairs(pairs_path: str) -> pd.DataFrame:
    """Load interaction pairs from TSV file."""
    pairs_df = pd.read_csv(
        pairs_path, sep="\t", header=None, 
        names=["protein1", "protein2", "label"]
    )
    return pairs_df


def canonicalize_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort protein IDs within each pair and drop duplicates to prevent inflated metrics.
    """
    if pairs_df.empty:
        return pairs_df
    
    df = pairs_df.copy()
    sorted_pairs = df[["protein1", "protein2"]].apply(
        lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1
    )
    df["protein1"], df["protein2"] = zip(*sorted_pairs)
    df["pair_key"] = df["protein1"] + "||" + df["protein2"]
    
    dup_count = df.duplicated("pair_key").sum()
    if dup_count > 0:
        print(f"  ⚠️ Removed {dup_count} duplicate pair rows (unordered).")
    
    df = df.drop_duplicates(subset="pair_key", keep="first")
    df = df.drop(columns=["pair_key"]).reset_index(drop=True)
    return df


def get_protein_based_splits(
    pairs_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Protein-level splits to avoid data leakage.
    A protein appears in only one fold's validation set.
    
    This is CRITICAL for fair evaluation in PPI prediction.
    """
    unique_proteins = list(set(pairs_df["protein1"]) | set(pairs_df["protein2"]))
    unique_proteins.sort()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    print(f"  📊 Generating {n_splits}-fold PROTEIN-LEVEL splits ({len(unique_proteins)} unique proteins)")
    
    for fold_idx, (train_prot_idx, val_prot_idx) in enumerate(kf.split(unique_proteins)):
        train_prots = set(unique_proteins[i] for i in train_prot_idx)
        val_prots = set(unique_proteins[i] for i in val_prot_idx)
        
        # Pairs where BOTH proteins are in train set
        train_mask = pairs_df.apply(
            lambda x: (x["protein1"] in train_prots) and (x["protein2"] in train_prots), axis=1
        )
        # Pairs where BOTH proteins are in validation set
        val_mask = pairs_df.apply(
            lambda x: (x["protein1"] in val_prots) and (x["protein2"] in val_prots), axis=1
        )
        
        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))
        
        print(
            f"    Fold {fold_idx+1}: Train={len(train_indices):,}, Val={len(val_indices):,} pairs"
        )
    
    return splits


# ============================================================================
# MODEL BUILDING
# ============================================================================

def create_estackppi_pipeline(n_jobs: int = -1) -> Pipeline:
    """
    E-StackPPI Pipeline: ESM2-only with 3-Stage Feature Selection + 2x LGBM Stacking.
    
    Architecture:
    1. StandardScaler
    2. 3-Stage Feature Selector (Variance → LGBM Importance → Correlation)
    3. 2x LGBM Base Estimators (Stacking)
    4. Logistic Regression Meta-Learner
    """
    # 3-stage feature selector
    selector = CumulativeFeatureSelector(
        variance_threshold=0.0,
        importance_quantile=0.90,
        corr_threshold=0.98,
        verbose=True
    )
    
    # LGBM parameters
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": n_jobs,
        "verbose": -1,
        "class_weight": "balanced",
    }
    
    # Create stacking classifier with 2x LGBM (different colsample_bytree for diversity)
    base_estimator_1 = LGBMClassifier(**lgbm_params, colsample_bytree=0.8)
    base_estimator_2 = LGBMClassifier(**lgbm_params, colsample_bytree=0.7)
    
    stacking = StackingClassifier(
        estimators=[
            ("lgbm_1", base_estimator_1),
            ("lgbm_2", base_estimator_2),
        ],
        final_estimator=LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=2000
        ),
        cv=3,
        n_jobs=n_jobs,
        verbose=0,
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector),
        ("stacking", stacking),
    ])
    
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass
    
    print("  ✅ E-StackPPI Pipeline created: Scaler → Selector → 2xLGBM Stacking → LR")
    return pipeline


# ============================================================================
# METRICS & VISUALIZATION
# ============================================================================

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
    plt.title(f"ROC Curves - {dataset_name} (Protein-Level CV)", fontsize=14, fontweight="bold")
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
    plt.title(f"Precision-Recall Curves - {dataset_name} (Protein-Level CV)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.text(0.98, 0.02, f"Mean AUPRC = {mean_pr_auc:.4f} ± {std_pr_auc:.4f}",
             transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_cross_validation(
    X: np.ndarray, 
    y: np.ndarray, 
    pairs_df: pd.DataFrame,
    dataset_name: str, 
    results_dir: str, 
    n_splits: int = 5, 
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Run 5-fold PROTEIN-LEVEL cross-validation (NO DATA LEAKAGE).
    """
    print(f"\n{'='*70}")
    print(f"🚀 E-StackPPI: {dataset_name} Dataset")
    print(f"{'='*70}")
    print(f"📊 Data shape: X={X.shape}, y={y.shape}")
    print(f"📊 Positive: {int(y.sum()):,}, Negative: {len(y) - int(y.sum()):,}")
    
    X_df = pd.DataFrame(X, columns=[f"esm_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")
    
    # ================================================
    # PROTEIN-LEVEL CV (NO DATA LEAKAGE)
    # ================================================
    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    
    all_metrics = []
    all_fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]
        
        # Create and train pipeline
        model = create_estackppi_pipeline(n_jobs=n_jobs)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics["Fold"] = fold_idx
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"  ✅ Fold {fold_idx} Results:")
        for k, v in metrics.items():
            if k != "Fold":
                print(f"    - {k:15s}: {v*100:.2f}%")
        
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
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 MEAN PERFORMANCE SUMMARY: {dataset_name} (Protein-Level CV)")
    print(f"{'='*70}")
    
    mean_metrics = metrics_df.drop(columns=["Fold"]).mean()
    std_metrics = metrics_df.drop(columns=["Fold"]).std()
    
    for metric in mean_metrics.index:
        print(f"  {metric:12s}: {mean_metrics[metric]*100:.2f}% ± {std_metrics[metric]*100:.2f}%")
    
    return metrics_df


def main():
    parser = argparse.ArgumentParser(
        description="E-StackPPI: ESM2-based PPI Prediction with Protein-Level CV"
    )
    parser.add_argument(
        "--dataset", type=str, default="all", 
        choices=["yeast", "human", "all"],
        help="Dataset to run: yeast, human, or all"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 for all cores)"
    )
    args = parser.parse_args()
    
    # Paths - corrected to match repository structure
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    results_base = script_dir / "results"
    
    # Dataset configurations - using CORRECT paths
    datasets = {
        "yeast": {
            "fasta_path": data_dir / "yeast" / "sequences.fasta",
            "pairs_path": data_dir / "yeast" / "pairs.tsv",
            "X_path": data_dir / "yeast" / "X_esm2.npy",
            "y_path": data_dir / "yeast" / "y.npy",
            "results_dir": results_base / "yeast",
            "name": "DIP-Yeast",
        },
        "human": {
            "fasta_path": data_dir / "human" / "sequences.fasta",
            "pairs_path": data_dir / "human" / "pairs.tsv",
            "X_path": data_dir / "human" / "X_esm2.npy",
            "y_path": data_dir / "human" / "y.npy",
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
        
        # Check if pre-computed features exist
        if not config["X_path"].exists():
            print(f"\n❌ Pre-computed features not found: {config['X_path']}")
            print(f"   Please run: python extract_esm2.py --dataset {dataset_key}")
            continue
        
        # Load data
        print(f"\n📂 Loading {config['name']} data...")
        X = np.load(config["X_path"])
        y = np.load(config["y_path"])
        
        # Load pairs for protein-level split
        pairs_df = load_pairs(config["pairs_path"])
        pairs_df = canonicalize_pairs(pairs_df)
        
        # Run cross-validation
        metrics_df = run_cross_validation(
            X=X, y=y,
            pairs_df=pairs_df,
            dataset_name=config["name"],
            results_dir=str(config["results_dir"]),
            n_splits=5,
            n_jobs=args.n_jobs,
        )
        all_results[dataset_key] = metrics_df
    
    print(f"\n{'='*70}")
    print("🎉 E-StackPPI EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    main()

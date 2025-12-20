#!/usr/bin/env python3
"""
E-StackPPI Ablation Study

This script runs ablation experiments to validate each component of E-StackPPI:
1. ESM-only + Logistic Regression (baseline)
2. ESM-only + LGBM (no selector)
3. ESM-only + LGBM + 3-stage selector
4. ESM-only + 2x LGBM Stacking (full E-StackPPI)

Usage:
    python run_ablation.py --dataset yeast
    python run_ablation.py --dataset human
"""

import os
import sys
import argparse
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
    confusion_matrix,
)
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.selectors import CumulativeFeatureSelector


# ============================================================================
# DATA UTILITIES (same as run_estackppi.py)
# ============================================================================

def load_pairs(pairs_path: str) -> pd.DataFrame:
    """Load interaction pairs from TSV file."""
    pairs_df = pd.read_csv(
        pairs_path, sep="\t", header=None, 
        names=["protein1", "protein2", "label"]
    )
    return pairs_df


def canonicalize_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Sort protein IDs within each pair and drop duplicates."""
    if pairs_df.empty:
        return pairs_df
    
    df = pairs_df.copy()
    sorted_pairs = df[["protein1", "protein2"]].apply(
        lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1
    )
    df["protein1"], df["protein2"] = zip(*sorted_pairs)
    df["pair_key"] = df["protein1"] + "||" + df["protein2"]
    df = df.drop_duplicates(subset="pair_key", keep="first")
    df = df.drop(columns=["pair_key"]).reset_index(drop=True)
    return df


def get_protein_based_splits(
    pairs_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Protein-level splits to avoid data leakage."""
    unique_proteins = list(set(pairs_df["protein1"]) | set(pairs_df["protein2"]))
    unique_proteins.sort()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for fold_idx, (train_prot_idx, val_prot_idx) in enumerate(kf.split(unique_proteins)):
        train_prots = set(unique_proteins[i] for i in train_prot_idx)
        val_prots = set(unique_proteins[i] for i in val_prot_idx)
        
        train_mask = pairs_df.apply(
            lambda x: (x["protein1"] in train_prots) and (x["protein2"] in train_prots), axis=1
        )
        val_mask = pairs_df.apply(
            lambda x: (x["protein1"] in val_prots) and (x["protein2"] in val_prots), axis=1
        )
        
        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))
    
    return splits


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def create_lr_pipeline() -> Pipeline:
    """Model 1: ESM + Logistic Regression (baseline)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs"
        ))
    ])


def create_lgbm_pipeline(n_jobs: int = -1) -> Pipeline:
    """Model 2: ESM + LGBM (no selector)."""
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
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(**lgbm_params))
    ])


def create_lgbm_selector_pipeline(n_jobs: int = -1) -> Pipeline:
    """Model 3: ESM + 3-stage selector + LGBM."""
    selector = CumulativeFeatureSelector(
        variance_threshold=0.0,
        importance_quantile=0.90,
        corr_threshold=0.98,
        verbose=False
    )
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
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector),
        ("model", LGBMClassifier(**lgbm_params))
    ])
    try:
        pipeline.set_output(transform="pandas")
    except:
        pass
    return pipeline


def create_stacking_pipeline(n_jobs: int = -1) -> Pipeline:
    """Model 4: ESM + 3-stage selector + 2x LGBM Stacking (Full E-StackPPI)."""
    selector = CumulativeFeatureSelector(
        variance_threshold=0.0,
        importance_quantile=0.90,
        corr_threshold=0.98,
        verbose=False
    )
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
    
    stacking = StackingClassifier(
        estimators=[
            ("lgbm_1", LGBMClassifier(**lgbm_params, colsample_bytree=0.8)),
            ("lgbm_2", LGBMClassifier(**lgbm_params, colsample_bytree=0.7)),
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
    except:
        pass
    return pipeline


# ============================================================================
# METRICS
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


# ============================================================================
# ABLATION STUDY
# ============================================================================

def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    pairs_df: pd.DataFrame,
    dataset_name: str,
    results_dir: str,
    n_splits: int = 5,
    n_jobs: int = -1
) -> pd.DataFrame:
    """Run ablation study with 4 model configurations."""
    
    print(f"\n{'='*70}")
    print(f"🔬 E-StackPPI ABLATION STUDY: {dataset_name}")
    print(f"{'='*70}")
    print(f"📊 Data: X={X.shape}, y={y.shape}")
    
    X_df = pd.DataFrame(X, columns=[f"esm_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="label")
    
    # Get protein-level splits
    splits = get_protein_based_splits(pairs_df, n_splits=n_splits)
    
    # Model configurations
    models = [
        ("1. LR (baseline)", create_lr_pipeline),
        ("2. LGBM", lambda: create_lgbm_pipeline(n_jobs)),
        ("3. LGBM + Selector", lambda: create_lgbm_selector_pipeline(n_jobs)),
        ("4. E-StackPPI (full)", lambda: create_stacking_pipeline(n_jobs)),
    ]
    
    all_results = []
    
    for model_name, model_builder in models:
        print(f"\n{'─'*50}")
        print(f"📌 Model: {model_name}")
        print(f"{'─'*50}")
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]
            
            # Train
            model = model_builder()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            metrics = compute_metrics(y_val, y_pred, y_proba)
            fold_metrics.append(metrics)
            
            print(f"  Fold {fold_idx}: ROC-AUC={metrics['ROC-AUC']:.4f}, Acc={metrics['Accuracy']:.4f}")
        
        # Average metrics
        fold_df = pd.DataFrame(fold_metrics)
        mean_metrics = fold_df.mean().to_dict()
        std_metrics = fold_df.std().to_dict()
        
        result = {"Model": model_name}
        for k in mean_metrics:
            result[k] = mean_metrics[k]
            result[f"{k}_std"] = std_metrics[k]
        
        all_results.append(result)
        
        print(f"  📊 Mean: ROC-AUC={mean_metrics['ROC-AUC']:.4f}±{std_metrics['ROC-AUC']:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Saved: {results_path}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"📊 ABLATION STUDY RESULTS: {dataset_name}")
    print(f"{'='*70}")
    
    display_cols = ["Model", "Accuracy", "ROC-AUC", "PR-AUC", "F1", "MCC"]
    display_df = results_df[display_cols].copy()
    
    for col in display_cols[1:]:
        std_col = f"{col}_std"
        if std_col in results_df.columns:
            display_df[col] = results_df.apply(
                lambda r: f"{r[col]*100:.2f}±{r[std_col]*100:.2f}", axis=1
            )
    
    print(display_df.to_string(index=False))
    
    # Generate LaTeX table
    latex_path = os.path.join(results_dir, "ablation_results.tex")
    with open(latex_path, "w") as f:
        f.write("% Ablation Study Results\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study results on " + dataset_name + "}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Model & Accuracy & ROC-AUC & PR-AUC & MCC \\\\\n")
        f.write("\\hline\n")
        
        for _, row in results_df.iterrows():
            line = f"{row['Model']} & "
            line += f"{row['Accuracy']*100:.2f}$\\pm${row['Accuracy_std']*100:.2f} & "
            line += f"{row['ROC-AUC']*100:.2f}$\\pm${row['ROC-AUC_std']*100:.2f} & "
            line += f"{row['PR-AUC']*100:.2f}$\\pm${row['PR-AUC_std']*100:.2f} & "
            line += f"{row['MCC']*100:.2f}$\\pm${row['MCC_std']*100:.2f} \\\\\n"
            f.write(line)
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"💾 Saved LaTeX table: {latex_path}")
    
    # Plot comparison
    plot_ablation_comparison(results_df, results_dir, dataset_name)
    
    return results_df


def plot_ablation_comparison(results_df: pd.DataFrame, results_dir: str, dataset_name: str):
    """Generate ablation comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    metrics = ["Accuracy", "ROC-AUC", "PR-AUC", "F1"]
    x = np.arange(len(metrics))
    width = 0.2
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    
    for i, row in results_df.iterrows():
        values = [row[m] for m in metrics]
        errors = [row[f"{m}_std"] for m in metrics]
        
        bars = ax.bar(
            x + i * width, values, width, 
            label=row["Model"],
            color=colors[i],
            alpha=0.8,
            yerr=errors,
            capsize=3
        )
    
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Ablation Study: {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "ablation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"💾 Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="E-StackPPI Ablation Study")
    parser.add_argument(
        "--dataset", type=str, default="yeast",
        choices=["yeast", "human"],
        help="Dataset to run ablation on"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs"
    )
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    results_base = script_dir / "results"
    
    # Dataset config
    datasets = {
        "yeast": {
            "pairs_path": data_dir / "yeast" / "pairs.tsv",
            "X_path": data_dir / "yeast" / "X_esm2.npy",
            "y_path": data_dir / "yeast" / "y.npy",
            "results_dir": results_base / "yeast" / "ablation",
            "name": "DIP-Yeast",
        },
        "human": {
            "pairs_path": data_dir / "human" / "pairs.tsv",
            "X_path": data_dir / "human" / "X_esm2.npy",
            "y_path": data_dir / "human" / "y.npy",
            "results_dir": results_base / "human" / "ablation",
            "name": "DIP-Human",
        },
    }
    
    config = datasets[args.dataset]
    
    # Check files
    if not config["X_path"].exists():
        print(f"❌ Pre-computed features not found: {config['X_path']}")
        print(f"   Please run: python extract_esm2.py --dataset {args.dataset}")
        return
    
    # Load data
    print(f"📂 Loading {config['name']} data...")
    X = np.load(config["X_path"])
    y = np.load(config["y_path"])
    pairs_df = load_pairs(config["pairs_path"])
    pairs_df = canonicalize_pairs(pairs_df)
    
    # Run ablation
    results = run_ablation(
        X=X, y=y,
        pairs_df=pairs_df,
        dataset_name=config["name"],
        results_dir=str(config["results_dir"]),
        n_splits=5,
        n_jobs=args.n_jobs
    )
    
    print(f"\n{'='*70}")
    print("🎉 ABLATION STUDY COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

"""
EStack-PPI Ablation Study Runner.

Runs comprehensive ablation experiments comparing:
- Baseline (ESM2 + LR/LGBM)
- Various feature selection configurations
- Single LGBM vs Stacking ensemble

Author: EStack-PPI Team
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

from .pipeline import get_ablation_pipelines
from .utils import print_metrics_table, plot_ablation_results


def run_single_experiment(
    pipeline_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict, List[np.ndarray], List[np.ndarray]]:
    """
    Run single experiment with cross-validation.
    
    Parameters
    ----------
    pipeline_factory : callable
        Function that returns a fresh pipeline instance.
    X : ndarray
        Feature matrix.
    y : ndarray
        Labels.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.
    
    Returns
    -------
    avg_metrics : dict
        Average metrics across folds.
    all_y_test : list
        True labels for each fold.
    all_y_proba : list
        Predicted probabilities for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_y_test = []
    all_y_proba = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"  Fold {fold}/{n_splits}...", end=" ", flush=True)
        
        fold_start = time.time()
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create fresh pipeline and train
        pipeline = pipeline_factory()
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Store for curves
        all_y_test.append(y_test)
        all_y_proba.append(y_proba)
        
        # Calculate metrics
        metrics = {
            'Fold': fold,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba),
            'AUC-PR': average_precision_score(y_test, y_proba),
            'Time': time.time() - fold_start
        }
        fold_results.append(metrics)
        
        if verbose:
            print(f"AUC={metrics['AUC-ROC']:.4f}, F1={metrics['F1']:.4f} ({metrics['Time']:.1f}s)")
    
    # Aggregate results
    df = pd.DataFrame(fold_results)
    avg_metrics = {
        'Accuracy': df['Accuracy'].mean(),
        'Accuracy_std': df['Accuracy'].std(),
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1': df['F1'].mean(),
        'F1_std': df['F1'].std(),
        'AUC-ROC': df['AUC-ROC'].mean(),
        'AUC-ROC_std': df['AUC-ROC'].std(),
        'AUC-PR': df['AUC-PR'].mean(),
        'AUC-PR_std': df['AUC-PR'].std(),
        'Time_total': df['Time'].sum()
    }
    
    return avg_metrics, all_y_test, all_y_proba


def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    use_gpu: bool = None,
    n_jobs: int = -1,
    random_state: int = 42,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run full ablation study with all variants.
    
    Parameters
    ----------
    X : ndarray
        Feature matrix (N, 2560 for ESM2).
    y : ndarray
        Binary labels.
    n_splits : int
        Number of CV folds.
    use_gpu : bool, optional
        Force GPU/CPU. Auto-detect if None.
    n_jobs : int
        Number of parallel jobs.
    random_state : int
        Random seed.
    save_path : str, optional
        Path to save results CSV.
    verbose : bool
        Print progress.
    
    Returns
    -------
    pd.DataFrame
        Results table with all variants and metrics.
    """
    print("=" * 80)
    print("🔬 E-STACKPPI ABLATION STUDY")
    print("=" * 80)
    print(f"   Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Positive ratio: {y.mean():.2%}")
    print(f"   CV Folds: {n_splits}")
    print("=" * 80)
    
    # Get all pipeline variants
    pipelines = get_ablation_pipelines(use_gpu=use_gpu, n_jobs=n_jobs, verbose=False)
    
    results = []
    all_predictions = {}
    
    total_start = time.time()
    
    for name, (factory, description) in pipelines.items():
        print(f"\n📦 {name}")
        print(f"   {description}")
        
        try:
            avg_metrics, y_tests, y_probas = run_single_experiment(
                pipeline_factory=factory,
                X=X, y=y,
                n_splits=n_splits,
                random_state=random_state,
                verbose=verbose
            )
            
            row = {
                'Variant': name,
                'Accuracy': avg_metrics['Accuracy'],
                'Acc_std': avg_metrics['Accuracy_std'],
                'F1': avg_metrics['F1'],
                'F1_std': avg_metrics['F1_std'],
                'AUC-ROC': avg_metrics['AUC-ROC'],
                'ROC_std': avg_metrics['AUC-ROC_std'],
                'AUC-PR': avg_metrics['AUC-PR'],
                'PR_std': avg_metrics['AUC-PR_std'],
                'Time': avg_metrics['Time_total']
            }
            results.append(row)
            all_predictions[name] = (y_tests, y_probas)
            
            print(f"   ✅ Acc={row['Accuracy']:.4f}±{row['Acc_std']:.4f}, "
                  f"AUC={row['AUC-ROC']:.4f}±{row['ROC_std']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'Variant': name,
                'Accuracy': np.nan,
                'F1': np.nan,
                'AUC-ROC': np.nan,
                'AUC-PR': np.nan,
                'Error': str(e)
            })
    
    total_time = time.time() - total_start
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 80)
    
    display_cols = ['Variant', 'Accuracy', 'F1', 'AUC-ROC', 'AUC-PR', 'Time']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    
    print(f"\n⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Highlight best
    if not df['AUC-ROC'].isna().all():
        best_idx = df['AUC-ROC'].idxmax()
        best_variant = df.loc[best_idx, 'Variant']
        best_auc = df.loc[best_idx, 'AUC-ROC']
        print(f"🏆 Best: {best_variant} (AUC-ROC = {best_auc:.4f})")
    
    # Save results
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n💾 Results saved to: {save_path}")
    
    return df, all_predictions


def format_results_latex(df: pd.DataFrame, caption: str = "Ablation Study Results") -> str:
    """
    Format results as LaTeX table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from run_ablation_study.
    caption : str
        Table caption.
    
    Returns
    -------
    str
        LaTeX table code.
    """
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Variant} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{AUC-ROC} & \\textbf{AUC-PR} \\\\\n"
    latex += "\\hline\n"
    
    for _, row in df.iterrows():
        variant = row['Variant'].replace('_', '\\_')
        
        # Format with std if available
        if 'Acc_std' in df.columns and not pd.isna(row.get('Acc_std')):
            acc = f"{row['Accuracy']:.4f}$\\pm${row['Acc_std']:.4f}"
        else:
            acc = f"{row['Accuracy']:.4f}" if not pd.isna(row['Accuracy']) else "-"
        
        if 'F1_std' in df.columns and not pd.isna(row.get('F1_std')):
            f1 = f"{row['F1']:.4f}$\\pm${row['F1_std']:.4f}"
        else:
            f1 = f"{row['F1']:.4f}" if not pd.isna(row['F1']) else "-"
        
        if 'ROC_std' in df.columns and not pd.isna(row.get('ROC_std')):
            roc = f"{row['AUC-ROC']:.4f}$\\pm${row['ROC_std']:.4f}"
        else:
            roc = f"{row['AUC-ROC']:.4f}" if not pd.isna(row['AUC-ROC']) else "-"
        
        if 'PR_std' in df.columns and not pd.isna(row.get('PR_std')):
            pr = f"{row['AUC-PR']:.4f}$\\pm${row['PR_std']:.4f}"
        else:
            pr = f"{row['AUC-PR']:.4f}" if not pd.isna(row['AUC-PR']) else "-"
        
        latex += f"{variant} & {acc} & {f1} & {roc} & {pr} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:ablation}\n"
    latex += "\\end{table}\n"
    
    return latex

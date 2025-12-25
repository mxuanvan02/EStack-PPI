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
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)

from .pipeline import get_ablation_pipelines
from .utils import plot_ablation_results, calculate_specificity


def run_single_experiment(
    pipeline_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict, List[np.ndarray], List[np.ndarray]]:
    """Run single experiment with cross-validation."""
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
        
        pipeline = pipeline_factory()
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        all_y_test.append(y_test)
        all_y_proba.append(y_proba)
        
        metrics = {
            'Fold': fold,
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred) * 100,
            'Recall': recall_score(y_test, y_pred) * 100,
            'F1': f1_score(y_test, y_pred) * 100,
            'Specificity': calculate_specificity(y_test, y_pred) * 100,
            'MCC': matthews_corrcoef(y_test, y_pred) * 100,
            'ROC-AUC': roc_auc_score(y_test, y_proba) * 100,
            'PR-AUC': average_precision_score(y_test, y_proba) * 100,
            'Time': time.time() - fold_start
        }
        fold_results.append(metrics)
        
        if verbose:
            print(f"AUC={metrics['ROC-AUC']:.2f}%, F1={metrics['F1']:.2f}%")
    
    df = pd.DataFrame(fold_results)
    avg_metrics = {}
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC', 'ROC-AUC', 'PR-AUC']:
        avg_metrics[col] = df[col].mean()
        avg_metrics[f'{col}_std'] = df[col].std()
    avg_metrics['Time_total'] = df['Time'].sum()
    
    return avg_metrics, all_y_test, all_y_proba


def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    variance_threshold: float = 0.002,
    importance_quantile: float = 0.90,
    corr_threshold: float = 0.90,
    n_splits: int = 5,
    use_gpu: bool = None,
    n_jobs: int = -1,
    random_state: int = 42,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full ablation study with all variants.
    
    Returns DataFrame with all 8 metrics and visualization.
    """
    print("=" * 120)
    print("🔬 E-STACKPPI ABLATION STUDY")
    print("=" * 120)
    print(f"   Data: {X.shape[0]} samples, {X.shape[1]} features | Positive: {y.mean():.2%} | Folds: {n_splits}")
    print(f"   Params: var={variance_threshold}, imp={importance_quantile}, corr={corr_threshold}")
    print("=" * 120)
    
    pipelines = get_ablation_pipelines(
        variance_threshold=variance_threshold,
        importance_quantile=importance_quantile,
        corr_threshold=corr_threshold,
        use_gpu=use_gpu, 
        n_jobs=n_jobs, 
        verbose=False
    )

    
    results = []
    all_predictions = {}
    
    total_start = time.time()
    
    for name, (factory, description) in pipelines.items():
        print(f"\n📦 {name}: {description}")
        
        try:
            avg_metrics, y_tests, y_probas = run_single_experiment(
                pipeline_factory=factory,
                X=X, y=y,
                n_splits=n_splits,
                random_state=random_state,
                verbose=verbose
            )
            
            row = {'Variant': name}
            for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC', 'ROC-AUC', 'PR-AUC']:
                row[f'{col} (%)'] = avg_metrics[col]
                row[f'{col}_std'] = avg_metrics[f'{col}_std']
            row['Time'] = avg_metrics['Time_total']
            
            results.append(row)
            all_predictions[name] = (y_tests, y_probas)
            
            print(f"   ✅ Acc={avg_metrics['Accuracy']:.2f}±{avg_metrics['Accuracy_std']:.2f}% | "
                  f"AUC={avg_metrics['ROC-AUC']:.2f}±{avg_metrics['ROC-AUC_std']:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            row = {'Variant': name, 'Error': str(e)}
            for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC', 'ROC-AUC', 'PR-AUC']:
                row[f'{col} (%)'] = np.nan
            results.append(row)
    
    total_time = time.time() - total_start
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Create display table with mean±std format
    display_data = []
    for _, row in df.iterrows():
        display_row = {'Variant': row['Variant']}
        for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'MCC', 'ROC-AUC', 'PR-AUC']:
            val, std = row.get(f'{col} (%)', np.nan), row.get(f'{col}_std', np.nan)
            if pd.notna(val):
                display_row[f'{col} (%)'] = f"{val:.2f}±{std:.2f}"
            else:
                display_row[f'{col} (%)'] = "-"
        display_data.append(display_row)
    
    df_display = pd.DataFrame(display_data)
    
    # Print summary table
    print("\n" + "=" * 120)
    print("📊 ABLATION STUDY RESULTS")
    print("=" * 120)
    print(df_display.to_string(index=False))
    
    print(f"\n⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Highlight best
    metric_cols = [c for c in df.columns if c.endswith(' (%)') and 'std' not in c]
    if 'ROC-AUC (%)' in df.columns and not df['ROC-AUC (%)'].isna().all():
        best_idx = df['ROC-AUC (%)'].idxmax()
        best_variant = df.loc[best_idx, 'Variant']
        best_auc = df.loc[best_idx, 'ROC-AUC (%)']
        print(f"🏆 Best: {best_variant} (ROC-AUC = {best_auc:.2f}%)")
    
    # Save results
    if save_path:
        df_display.to_csv(save_path, index=False)
        print(f"\n💾 Results saved to: {save_path}")
    
    # Visualization
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            # Bar chart for key metrics
            fig, ax = plt.subplots(figsize=(14, 6))
            
            variants = df['Variant'].tolist()
            x = np.arange(len(variants))
            width = 0.15
            
            metrics_to_plot = ['Accuracy (%)', 'F1 (%)', 'MCC (%)', 'ROC-AUC (%)', 'PR-AUC (%)']
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
            
            for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
                if metric in df.columns:
                    values = df[metric].fillna(0).tolist()
                    ax.bar(x + i * width, values, width, label=metric.replace(' (%)', ''), color=color)
            
            ax.set_xlabel('Variant', fontsize=12)
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title('Ablation Study Results', fontsize=14)
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(variants, rotation=30, ha='right')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 105)
            
            plt.tight_layout()
            plt.savefig('ablation_chart.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("✅ Saved: ablation_chart.png")
        except Exception as e:
            print(f"⚠️ Could not plot: {e}")
    
    return df, all_predictions


def format_results_latex(df: pd.DataFrame, caption: str = "Ablation Study Results") -> str:
    """Format results as LaTeX table."""
    latex = "\\begin{table}[htbp]\n\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\begin{tabular}{l" + "c" * 8 + "}\n\\hline\n"
    latex += "\\textbf{Variant} & \\textbf{Acc} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{F1} & \\textbf{Spec} & \\textbf{MCC} & \\textbf{ROC} & \\textbf{PR} \\\\\n"
    latex += "\\hline\n"
    
    for _, row in df.iterrows():
        variant = row['Variant'].replace('_', '\\_')
        values = []
        for col in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 (%)', 'Specificity (%)', 'MCC (%)', 'ROC-AUC (%)', 'PR-AUC (%)']:
            val = row.get(col, '-')
            values.append(str(val) if val != '-' else '-')
        latex += f"{variant} & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\label{tab:ablation}\n\\end{table}\n"
    return latex

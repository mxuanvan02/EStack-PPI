"""
EStack-PPI Utility Functions.

Provides:
- GPU/CPU detection
- Data loading utilities
- Visualization functions

Author: EStack-PPI Team
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Device Detection
# ============================================================================

def detect_device() -> Dict[str, any]:
    """
    Detect available compute device (GPU/CPU).
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'device': 'gpu' or 'cpu'
        - 'name': Device name string
        - 'cuda_available': bool
        - 'lgbm_gpu': bool (whether LightGBM can use GPU)
    """
    result = {
        'device': 'cpu',
        'name': 'CPU',
        'cuda_available': False,
        'lgbm_gpu': False
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            result['cuda_available'] = True
            result['device'] = 'gpu'
            result['name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Check LightGBM GPU support
    try:
        from lightgbm import LGBMClassifier
        # Try to create a GPU classifier
        clf = LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, verbose=-1)
        # Fit with tiny data to test
        X_test = np.random.randn(10, 5)
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        clf.fit(X_test, y_test)
        result['lgbm_gpu'] = True
    except Exception:
        result['lgbm_gpu'] = False
    
    return result


def print_device_info():
    """Print device information."""
    info = detect_device()
    print(f"🖥️  Device: {info['name']}")
    print(f"   CUDA available: {info['cuda_available']}")
    print(f"   LightGBM GPU: {info['lgbm_gpu']}")
    return info


# ============================================================================
# Data Loading
# ============================================================================

def load_sequences(fasta_path: str) -> Dict[str, str]:
    """
    Load protein sequences from FASTA file.
    
    Parameters
    ----------
    fasta_path : str
        Path to FASTA file.
    
    Returns
    -------
    dict
        Dictionary mapping protein ID to sequence.
    """
    try:
        from Bio import SeqIO
    except ImportError:
        raise ImportError("BioPython required. Install with: pip install biopython")
    
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq).upper()
    
    return sequences


def load_pairs(tsv_path: str, has_header: bool = False) -> pd.DataFrame:
    """
    Load protein pairs from TSV file.
    
    Parameters
    ----------
    tsv_path : str
        Path to TSV file with columns: prot1, prot2, label
    has_header : bool
        Whether the file has a header row.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: prot1, prot2, label
    """
    if has_header:
        df = pd.read_csv(tsv_path, sep="\t")
    else:
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["prot1", "prot2", "label"])
    
    return df


def load_embeddings_from_h5(
    h5_path: str,
    pairs_df: pd.DataFrame,
    sequences: Dict[str, str],
    key_format: str = "sequence"
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load and concatenate embeddings from H5 cache.
    
    Parameters
    ----------
    h5_path : str
        Path to H5 cache file.
    pairs_df : pd.DataFrame
        DataFrame with prot1, prot2, label columns.
    sequences : dict
        Dictionary mapping protein ID to sequence.
    key_format : str
        Key format in H5: "sequence" or "protein_id"
    
    Returns
    -------
    X : ndarray
        Feature matrix (N, 2*embedding_dim)
    y : ndarray
        Labels
    valid_indices : list
        Indices of valid pairs
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")
    
    from tqdm import tqdm
    
    X_list = []
    y_list = []
    valid_indices = []
    
    with h5py.File(h5_path, "r") as h5f:
        h5_keys = set(h5f.keys())
        
        for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Loading embeddings"):
            prot1, prot2, label = row['prot1'], row['prot2'], row['label']
            
            if key_format == "sequence":
                seq1 = sequences.get(prot1, "")
                seq2 = sequences.get(prot2, "")
                key1 = f"{seq1}_global_v2"
                key2 = f"{seq2}_global_v2"
            else:
                key1 = f"{prot1}_global_v2"
                key2 = f"{prot2}_global_v2"
            
            if key1 in h5_keys and key2 in h5_keys:
                emb1 = h5f[key1][:]
                emb2 = h5f[key2][:]
                X_list.append(np.concatenate([emb1, emb2]))
                y_list.append(int(label))
                valid_indices.append(idx)
    
    if len(X_list) == 0:
        raise ValueError("No valid pairs found! Check H5 key format.")
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    return X, y, valid_indices


# ============================================================================
# Visualization
# ============================================================================

def plot_roc_pr_curves(
    all_y_test: List[np.ndarray],
    all_y_proba: List[np.ndarray],
    title: str = "E-StackPPI",
    save_path: Optional[str] = None
):
    """
    Plot ROC and Precision-Recall curves for all folds.
    
    Parameters
    ----------
    all_y_test : list
        List of true labels for each fold.
    all_y_proba : list
        List of predicted probabilities for each fold.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for folds
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_y_test)))
    
    # ===== ROC Curve =====
    ax1 = axes[0]
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for i, (y_test, y_proba) in enumerate(zip(all_y_test, all_y_proba)):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate for mean curve
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        ax1.plot(fpr, tpr, color=colors[i], alpha=0.6, lw=1.5,
                label=f'Fold {i+1} (AUC={roc_auc:.4f})')
    
    # Mean curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax1.plot(mean_fpr, mean_tpr, color='navy', lw=2,
            label=f'Mean (AUC={mean_auc:.4f}±{std_auc:.4f})')
    
    # Fill std area
    std_tpr = np.std(tprs, axis=0)
    ax1.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='grey', alpha=0.2)
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'ROC Curve - {title}', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== PR Curve =====
    ax2 = axes[1]
    mean_recall = np.linspace(0, 1, 100)
    precisions_interp = []
    pr_aucs = []
    
    for i, (y_test, y_proba) in enumerate(zip(all_y_test, all_y_proba)):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        
        # Interpolate for mean curve (reverse because recall is decreasing)
        precision_interp = np.interp(mean_recall[::-1], recall[::-1], precision[::-1])[::-1]
        precisions_interp.append(precision_interp)
        
        ax2.plot(recall, precision, color=colors[i], alpha=0.6, lw=1.5,
                label=f'Fold {i+1} (AUC={pr_auc:.4f})')
    
    # Mean curve
    mean_precision = np.mean(precisions_interp, axis=0)
    mean_pr_auc = np.mean(pr_aucs)
    std_pr_auc = np.std(pr_aucs)
    ax2.plot(mean_recall, mean_precision, color='darkgreen', lw=2,
            label=f'Mean (AUC={mean_pr_auc:.4f}±{std_pr_auc:.4f})')
    
    # Fill std area
    std_precision = np.std(precisions_interp, axis=0)
    ax2.fill_between(mean_recall, 
                     np.maximum(mean_precision - std_precision, 0),
                     np.minimum(mean_precision + std_precision, 1),
                     color='grey', alpha=0.2)
    
    # Baseline (at the end of legend)
    baseline = np.mean([y.mean() for y in all_y_test])
    ax2.axhline(y=baseline, color='k', linestyle='--', lw=1, 
               label=f'Baseline ({baseline:.3f})')
    
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'Precision-Recall Curve - {title}', fontsize=14)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    
    return {
        'roc_auc_mean': mean_auc,
        'roc_auc_std': std_auc,
        'pr_auc_mean': mean_pr_auc,
        'pr_auc_std': std_pr_auc
    }


def plot_ablation_results(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot ablation study results as a bar chart.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: Variant, Accuracy, F1, AUC-ROC, AUC-PR
    save_path : str, optional
        Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required. Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'F1', 'AUC-ROC', 'AUC-PR']
    x = np.arange(len(results_df))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i])
    
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Variant'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ============================================================================
# Metrics
# ============================================================================

def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (True Negative Rate).
    
    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    
    Returns
    -------
    float
        Specificity score.
    """
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, 
                      as_percentage: bool = False) -> Dict:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    y_proba : ndarray
        Predicted probabilities for positive class.
    as_percentage : bool
        If True, return values as percentages (0-100).
    
    Returns
    -------
    dict
        Dictionary with: Accuracy, Precision, Recall, F1, Specificity, MCC, ROC-AUC, PR-AUC
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, average_precision_score,
        matthews_corrcoef
    )
    
    multiplier = 100 if as_percentage else 1
    suffix = " (%)" if as_percentage else ""
    
    return {
        f'Accuracy{suffix}': accuracy_score(y_true, y_pred) * multiplier,
        f'Precision{suffix}': precision_score(y_true, y_pred) * multiplier,
        f'Recall{suffix}': recall_score(y_true, y_pred) * multiplier,
        f'F1{suffix}': f1_score(y_true, y_pred) * multiplier,
        f'Specificity{suffix}': calculate_specificity(y_true, y_pred) * multiplier,
        f'MCC{suffix}': matthews_corrcoef(y_true, y_pred) * multiplier,
        f'ROC-AUC{suffix}': roc_auc_score(y_true, y_proba) * multiplier,
        f'PR-AUC{suffix}': average_precision_score(y_true, y_proba) * multiplier
    }


def print_metrics_table(results: List[Dict], title: str = "Results"):
    """Print formatted metrics table."""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print('='*80)
    print(df.to_string(index=False))
    
    # Averages
    print(f"\n📈 Average (Mean ± Std):")
    for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AUC-PR']:
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            print(f"   {col:12s}: {m:.4f} ± {s:.4f}")
    
    return df

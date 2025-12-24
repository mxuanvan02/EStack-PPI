"""
EStack-PPI: Embedding-based Stacking for Protein-Protein Interaction Prediction.

A high-performance PPI prediction pipeline using:
- ESM-2 protein language model embeddings
- 3-stage cumulative feature selection (Variance → Importance → Correlation)
- Stacking ensemble (2×LightGBM + Logistic Regression meta-learner)

Usage
-----
>>> from EStackPPI import create_estack_pipeline
>>> pipeline = create_estack_pipeline(use_gpu=True)
>>> pipeline.fit(X_train, y_train)
>>> y_pred = pipeline.predict(X_test)

For ablation study:
>>> from EStackPPI import run_ablation_study
>>> results = run_ablation_study(X, y, n_splits=5)

Author: EStack-PPI Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "EStack-PPI Team"

# Core components
from .selectors import CumulativeFeatureSelector
from .pipeline import (
    create_estack_pipeline,
    create_baseline_lr,
    create_baseline_lgbm,
    create_var_only_pipeline,
    create_var_imp_pipeline,
    create_full_pipeline,
    create_single_lgbm_pipeline,
    get_ablation_pipelines
)
from .utils import (
    detect_device,
    print_device_info,
    load_sequences,
    load_pairs,
    load_embeddings_from_h5,
    plot_roc_pr_curves,
    plot_ablation_results,
    calculate_metrics,
    print_metrics_table
)
from .ablation import (
    run_single_experiment,
    run_ablation_study,
    format_results_latex
)

__all__ = [
    # Version
    "__version__",
    
    # Selectors
    "CumulativeFeatureSelector",
    
    # Pipelines
    "create_estack_pipeline",
    "create_baseline_lr",
    "create_baseline_lgbm",
    "create_var_only_pipeline",
    "create_var_imp_pipeline",
    "create_full_pipeline",
    "create_single_lgbm_pipeline",
    "get_ablation_pipelines",
    
    # Utils
    "detect_device",
    "print_device_info",
    "load_sequences",
    "load_pairs",
    "load_embeddings_from_h5",
    "plot_roc_pr_curves",
    "plot_ablation_results",
    "calculate_metrics",
    "print_metrics_table",
    
    # Ablation
    "run_single_experiment",
    "run_ablation_study",
    "format_results_latex",
]

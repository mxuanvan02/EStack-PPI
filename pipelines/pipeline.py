"""
EStack-PPI Pipeline Factory.

Provides functions to create various pipeline configurations:
- Full E-StackPPI (3-stage selector + stacking)
- Ablation variants (baseline, var-only, etc.)

Architecture:
    [Variance Filter] → [StandardScaler] → [Importance Filter] → 
    [Correlation Filter] → [Stacking: 2×LGBM + LR Meta]

Author: EStack-PPI Team
"""

from typing import List, Optional
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingClassifier

from .selectors import CumulativeFeatureSelector
from .utils import detect_device


# ============================================================================
# Pipeline Factory
# ============================================================================

def create_estack_pipeline(
    use_selector: bool = True,
    use_variance: bool = True,
    use_importance: bool = True,
    use_corr: bool = True,
    variance_threshold: float = 0.01,
    importance_quantile: float = 0.90,
    corr_threshold: float = 0.98,
    use_stacking: bool = True,
    n_estimators: int = 500,
    use_gpu: bool = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Pipeline:
    """
    Create E-StackPPI pipeline with configurable components.
    
    Architecture:
        Variance → Scaler → [Importance → Correlation] → Stacking
    
    Parameters
    ----------
    use_selector : bool
        Whether to use feature selector at all.
    use_variance : bool
        Enable variance filter (Stage 1).
    use_importance : bool
        Enable importance filter (Stage 2).
    use_corr : bool
        Enable correlation filter (Stage 3).
    variance_threshold : float
        Variance threshold for Stage 1.
    importance_quantile : float
        Cumulative importance quantile for Stage 2.
    corr_threshold : float
        Correlation threshold for Stage 3.
    use_stacking : bool
        Use stacking (2×LGBM + LR) vs single LGBM.
    n_estimators : int
        Number of trees for LGBM.
    use_gpu : bool, optional
        Force GPU (True) or CPU (False). Auto-detect if None.
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Print progress messages.
    
    Returns
    -------
    Pipeline
        Configured sklearn Pipeline.
    """
    # Auto-detect GPU
    if use_gpu is None:
        device_info = detect_device()
        use_gpu = device_info['lgbm_gpu']
        if verbose:
            print(f"🖥️  Device: {device_info['name']} (GPU={use_gpu})")
    
    # Build pipeline steps
    steps = []
    
    # Step 1: Variance filter (BEFORE scaling)
    if use_selector and use_variance and variance_threshold > 0:
        from sklearn.feature_selection import VarianceThreshold
        steps.append(("variance_filter", VarianceThreshold(threshold=variance_threshold)))
        if verbose:
            print(f"  ├─ VarianceThreshold(threshold={variance_threshold})")
    
    # Step 2: Standard Scaler
    steps.append(("scaler", StandardScaler()))
    if verbose:
        print(f"  ├─ StandardScaler()")
    
    # Step 3: Importance + Correlation filter
    if use_selector and (use_importance or use_corr):
        selector = CumulativeFeatureSelector(
            variance_threshold=0.0,  # Already done above
            importance_quantile=importance_quantile,
            corr_threshold=corr_threshold,
            use_variance=False,  # Already done
            use_importance=use_importance,
            use_corr=use_corr,
            use_gpu=use_gpu,
            verbose=verbose
        )
        steps.append(("selector", selector))
        if verbose:
            stages = []
            if use_importance:
                stages.append(f"Importance(q={importance_quantile})")
            if use_corr:
                stages.append(f"Correlation(t={corr_threshold})")
            print(f"  ├─ CumulativeFeatureSelector({', '.join(stages)})")
    
    # Step 4: Classifier
    lgbm_params = _get_lgbm_params(n_estimators, use_gpu, n_jobs)
    
    if use_stacking:
        classifier = _create_stacking_classifier(lgbm_params, n_jobs, use_gpu)
        if verbose:
            print(f"  └─ StackingClassifier(2×LGBM + LR)")
    else:
        if HAS_LGBM:
            classifier = LGBMClassifier(**lgbm_params, random_state=42)
        else:
            classifier = GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=42
            )
        if verbose:
            print(f"  └─ LGBMClassifier(n_estimators={n_estimators})")
    
    steps.append(("classifier", classifier))
    
    pipeline = Pipeline(steps)
    
    if verbose:
        print(f"✅ E-StackPPI pipeline created")
    
    return pipeline


def _get_lgbm_params(n_estimators: int, use_gpu: bool, n_jobs: int) -> dict:
    """Get LightGBM parameters."""
    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": n_jobs,
        "verbose": -1,
        "class_weight": "balanced",
    }
    
    if use_gpu and HAS_LGBM:
        params.update({
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0
        })
    
    return params


def _create_stacking_classifier(lgbm_params: dict, n_jobs: int, use_gpu: bool):
    """Create stacking classifier with 2×LGBM + LR."""
    if HAS_LGBM:
        lgbm_1 = LGBMClassifier(**lgbm_params, random_state=42)
        lgbm_2 = LGBMClassifier(**lgbm_params, random_state=123)
    else:
        lgbm_1 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lgbm_2 = GradientBoostingClassifier(n_estimators=100, random_state=123)
    
    return StackingClassifier(
        estimators=[
            ("lgbm_1", lgbm_1),
            ("lgbm_2", lgbm_2)
        ],
        final_estimator=LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=1000
        ),
        cv=3,
        n_jobs=1 if use_gpu else n_jobs,
        verbose=0,
    )


# ============================================================================
# Ablation Variants
# ============================================================================

def create_baseline_lr(n_jobs: int = -1, verbose: bool = True) -> Pipeline:
    """
    Baseline: ESM2 + Logistic Regression (no feature selection).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42, 
            class_weight="balanced", 
            max_iter=2000,
            n_jobs=n_jobs
        ))
    ])
    
    if verbose:
        print("✅ Baseline (ESM2 + LR) created")
    
    return pipeline


def create_baseline_lgbm(use_gpu: bool = None, n_jobs: int = -1, verbose: bool = True) -> Pipeline:
    """
    Baseline: ESM2 + Single LGBM (no feature selection).
    """
    if use_gpu is None:
        use_gpu = detect_device()['lgbm_gpu']
    
    lgbm_params = _get_lgbm_params(500, use_gpu, n_jobs)
    
    if HAS_LGBM:
        classifier = LGBMClassifier(**lgbm_params, random_state=42)
    else:
        classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])
    
    if verbose:
        print("✅ Baseline (ESM2 + LGBM) created")
    
    return pipeline


def create_var_only_pipeline(
    variance_threshold: float = 0.01,
    use_gpu: bool = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Pipeline:
    """
    Variant: Variance filter only (no importance, no correlation).
    """
    return create_estack_pipeline(
        use_selector=True,
        use_variance=True,
        use_importance=False,
        use_corr=False,
        variance_threshold=variance_threshold,
        use_stacking=True,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        verbose=verbose
    )


def create_var_imp_pipeline(
    variance_threshold: float = 0.01,
    importance_quantile: float = 0.90,
    use_gpu: bool = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Pipeline:
    """
    Variant: Variance + Importance (no correlation filter).
    """
    return create_estack_pipeline(
        use_selector=True,
        use_variance=True,
        use_importance=True,
        use_corr=False,
        variance_threshold=variance_threshold,
        importance_quantile=importance_quantile,
        use_stacking=True,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        verbose=verbose
    )


def create_full_pipeline(
    variance_threshold: float = 0.01,
    importance_quantile: float = 0.90,
    corr_threshold: float = 0.98,
    use_gpu: bool = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Pipeline:
    """
    Full E-StackPPI: Variance + Importance + Correlation + Stacking.
    """
    return create_estack_pipeline(
        use_selector=True,
        use_variance=True,
        use_importance=True,
        use_corr=True,
        variance_threshold=variance_threshold,
        importance_quantile=importance_quantile,
        corr_threshold=corr_threshold,
        use_stacking=True,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        verbose=verbose
    )


def create_single_lgbm_pipeline(
    variance_threshold: float = 0.01,
    importance_quantile: float = 0.90,
    corr_threshold: float = 0.98,
    use_gpu: bool = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Pipeline:
    """
    Full selector + Single LGBM (no stacking, for comparison).
    """
    return create_estack_pipeline(
        use_selector=True,
        use_variance=True,
        use_importance=True,
        use_corr=True,
        variance_threshold=variance_threshold,
        importance_quantile=importance_quantile,
        corr_threshold=corr_threshold,
        use_stacking=False,  # Single LGBM
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        verbose=verbose
    )


# ============================================================================
# Pipeline Collection
# ============================================================================

def get_ablation_pipelines(
    variance_threshold: float = 0.002,
    importance_quantile: float = 0.90,
    corr_threshold: float = 0.90,
    use_gpu: bool = None, 
    n_jobs: int = -1, 
    verbose: bool = False
):
    """
    Get all ablation study pipelines.
    All variants use Logistic Regression as meta-learner.
    
    Parameters
    ----------
    variance_threshold : float
        Variance threshold for feature selection.
    importance_quantile : float
        Cumulative importance quantile.
    corr_threshold : float
        Correlation threshold.
    use_gpu : bool
        Use GPU for LightGBM.
    n_jobs : int
        Number of parallel jobs.
    verbose : bool
        Print progress.
    
    Returns
    -------
    dict
        Dictionary mapping variant name to (pipeline_factory, description).
    """
    return {
        "Baseline (LR)": (
            lambda: create_baseline_lr(n_jobs, verbose),
            "ESM2 + Logistic Regression"
        ),
        "Var-Only": (
            lambda: create_var_only_pipeline(variance_threshold, use_gpu, n_jobs, verbose),
            f"Variance({variance_threshold}) + Stacking"
        ),
        "Var + Imp": (
            lambda: create_var_imp_pipeline(variance_threshold, importance_quantile, use_gpu, n_jobs, verbose),
            f"Var + Imp(q={importance_quantile}) + Stacking"
        ),
        "E-StackPPI": (
            lambda: create_full_pipeline(variance_threshold, importance_quantile, corr_threshold, use_gpu, n_jobs, verbose),
            f"Var + Imp + Corr({corr_threshold}) + Stacking"
        ),
    }


from typing import List
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipelines.feature_engine import FeatureEngine
from pipelines.selectors import CumulativeFeatureSelector


def create_lgbm_pipeline(
    n_jobs: int = -1,
    selector_quantile: float = 0.8,
    use_selector: bool = True,
    lgbm_params: dict | None = None,
) -> Pipeline:
    pipeline_steps = [("scaler", StandardScaler())]
    if use_selector:
        selector = CumulativeFeatureSelector(
            importance_quantile=selector_quantile, corr_threshold=0.95, verbose=True
        )
        pipeline_steps.append(("selector", selector))

    default_params = {
        "n_estimators": 300,
        "num_leaves": 20,
        "max_depth": 10,
        "learning_rate": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_child_samples": 30,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    if lgbm_params:
        default_params.update(lgbm_params)

    model = LGBMClassifier(**default_params)
    pipeline_steps.append(("model", model))

    pipeline = Pipeline(pipeline_steps)
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass

    print(f"✅ LGBM (Selector={use_selector}) pipeline created.")
    return pipeline


def create_stacking_pipeline(interp_cols: List[str], embed_cols: List[str], n_jobs: int = -1, use_selector: bool = True):
    if use_selector:
        interp_preprocessor = CumulativeFeatureSelector(
            importance_quantile=0.95, corr_threshold=0.97, variance_threshold=0.01, verbose=True
        )
    else:
        interp_preprocessor = "passthrough"

    embed_steps = [("scaler", StandardScaler())]
    if use_selector:
        embed_steps.append(
            (
                "selector",
                CumulativeFeatureSelector(
                    importance_quantile=0.90, corr_threshold=0.98, variance_threshold=0.0, verbose=True
                ),
            )
        )
    embed_preprocessor = Pipeline(embed_steps)

    try:
        if hasattr(interp_preprocessor, "set_output"):
            interp_preprocessor.set_output(transform="pandas")
        embed_preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    common_lgbm_params = {
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

    interp_base_estimator = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("interp_transformer", interp_preprocessor, interp_cols)], remainder="drop", n_jobs=n_jobs)),
            ("model", LGBMClassifier(**common_lgbm_params)),
        ]
    )

    embed_base_estimator = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("embed_transformer", embed_preprocessor, embed_cols)], remainder="drop", n_jobs=n_jobs)),
            ("model", LGBMClassifier(**common_lgbm_params)),
        ]
    )

    stacking_model = StackingClassifier(
        estimators=[("interp", interp_base_estimator), ("embed", embed_base_estimator)],
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=3,
        n_jobs=n_jobs,
        verbose=0,
    )
    print(f"✅ Stacking (Selector={use_selector}) pipeline created (using *permissive* thresholds).")
    return stacking_model


def create_svm_pipeline(n_jobs: int = -1, selector_quantile: float = 0.5) -> Pipeline:
    selector = CumulativeFeatureSelector(importance_quantile=selector_quantile, corr_threshold=0.95, verbose=True)
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42, class_weight="balanced")
    pipeline = Pipeline([("scaler", StandardScaler()), ("selector", selector), ("model", model)])
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass
    print(f"✅ SVM (Scaler -> Selector(q={selector_quantile}) -> SVC) pipeline created.")
    return pipeline


def create_esm_lr_pipeline(n_jobs: int = -1) -> Pipeline:
    lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=2000, solver="lbfgs")
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", lr_model)])
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass
    print("✅ [E-StackPPI] Pipeline: ESM-only + LR (no selector) created.")
    return pipeline


def create_esm_lgbm_raw_pipeline(n_jobs: int = -1) -> Pipeline:
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
    model = LGBMClassifier(**lgbm_params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass
    print("✅ [E-StackPPI] Pipeline: ESM-only + LGBM (no selector) created.")
    return pipeline


def create_esm_lgbm_selector_pipeline(n_jobs: int = -1) -> Pipeline:
    selector = CumulativeFeatureSelector(
        variance_threshold=0.0, importance_quantile=0.90, corr_threshold=0.98, verbose=True
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
    model = LGBMClassifier(**lgbm_params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("selector", selector), ("model", model)])
    try:
        pipeline.set_output(transform="pandas")
    except Exception:
        pass
    print("✅ [E-StackPPI] Pipeline: ESM-only + Selector + LGBM created.")
    return pipeline


def create_esm_stacking_pipeline(n_jobs: int = -1):
    """
    EStack-PPI Pipeline: ESM2-only with 2x LGBM Stacking + Logistic Regression Meta-Learner.
    
    Architecture:
    1. StandardScaler
    2. 3-Stage Feature Selector (Variance -> LGBM Importance -> Correlation)
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
    
    print("✅ [E-StackPPI] Pipeline: ESM-only + Selector + 2xLGBM Stacking + LR created.")
    return pipeline


def create_embed_only_pipeline(embed_cols: List[str], n_jobs: int = -1, use_selector: bool = True):
    embed_steps = [("scaler", StandardScaler())]
    if use_selector:
        embed_steps.append(
            (
                "selector",
                CumulativeFeatureSelector(
                    importance_quantile=0.90, corr_threshold=0.98, variance_threshold=0.0, verbose=True
                ),
            )
        )
    embed_preprocessor = Pipeline(embed_steps)

    try:
        embed_preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    model_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": n_jobs,
        "verbose": -1,
    }

    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("embed_transformer", embed_preprocessor, embed_cols)], remainder="drop", n_jobs=n_jobs)),
            ("model", LGBMClassifier(**model_params, class_weight="balanced")),
        ]
    )
    print("✅ Embed-Only pipeline created.")
    return pipeline


def create_interp_only_pipeline(interp_cols: List[str], n_jobs: int = -1, use_selector: bool = True):
    if use_selector:
        interp_preprocessor = CumulativeFeatureSelector(
            importance_quantile=0.95, corr_threshold=0.97, variance_threshold=0.01, verbose=True
        )
    else:
        interp_preprocessor = "passthrough"

    try:
        if hasattr(interp_preprocessor, "set_output"):
            interp_preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    model_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "max_depth": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": n_jobs,
        "verbose": -1,
    }

    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("interp_transformer", interp_preprocessor, interp_cols)], remainder="drop", n_jobs=n_jobs)),
            ("model", LGBMClassifier(**model_params, class_weight="balanced")),
        ]
    )
    print("✅ Interp-Only pipeline (updated) created.")
    return pipeline


def define_stacking_columns(feature_engine: FeatureEngine, pairing_strategy: str = "concat") -> tuple[List[str], List[str]]:
    handcraft_names = feature_engine.handcraft_extractor.get_feature_names()
    motif_names = feature_engine.motif_names
    global_emb_names = feature_engine.global_emb_names
    local_emb_names = feature_engine.local_emb_names

    interp_names = handcraft_names + motif_names
    embed_names = global_emb_names + local_emb_names

    if pairing_strategy == "concat":
        prefix1, prefix2 = "P1_", "P2_"
    elif pairing_strategy == "avgdiff":
        prefix1, prefix2 = "Avg_", "Diff_"
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")

    interp_cols = [f"{prefix1}{name}" for name in interp_names] + [f"{prefix2}{name}" for name in interp_names]
    embed_cols = [f"{prefix1}{name}" for name in embed_names] + [f"{prefix2}{name}" for name in embed_names]
    return interp_cols, embed_cols

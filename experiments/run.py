import gc
import os
import warnings

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipelines.feature_engine import EmbeddingComputer, FeatureEngine
from experiments.logger import PipelineLogger
from experiments.metrics import (
    display_full_metrics,
    plot_feature_importance_for_paper,
    print_paper_style_results,
    plot_hybrid_feature_importance,
    save_feature_importance_table,
    plot_roc_pr_curves,
)
from experiments.data_utils import (
    load_data,
    canonicalize_pairs,
    create_feature_matrix,
    get_cache_filename,
    save_feature_matrix_h5,
    load_feature_matrix_h5,
    build_esm_only_pair_matrix,
    get_protein_based_splits,
    get_cluster_based_splits,
    load_cluster_map,
)
from pipelines.builders import (
    create_embed_only_pipeline,
    create_esm_lgbm_raw_pipeline,
    create_esm_lgbm_selector_pipeline,
    create_esm_lr_pipeline,
    create_interp_only_pipeline,
    create_stacking_pipeline,
    define_stacking_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _log_selector_state(logger: PipelineLogger, model_pipeline, prefix: str = ""):
    """
    Log how many features remain after selector and list a short preview.
    Supports simple pipelines and stacking branches.
    """
    def _print_selected(name, selector_obj):
        if not selector_obj or not getattr(selector_obj, "selected_features_", None):
            return
        feats = selector_obj.selected_features_
        preview = ", ".join(feats[:15])
        suffix = " ..." if len(feats) > 15 else ""
        logger.info(f"{prefix}{name}: kept {len(feats)} features")
        logger.info(f"{prefix}{name} preview: {preview}{suffix}")

    if hasattr(model_pipeline, "named_steps"):
        _print_selected("Selector", model_pipeline.named_steps.get("selector"))

    if hasattr(model_pipeline, "estimators_"):
        for branch_name, est in model_pipeline.estimators_:
            if hasattr(est, "named_steps"):
                pre = est.named_steps.get("preprocessor")
                if pre and getattr(pre, "transformers_", None):
                    sel = getattr(pre.transformers_[0][1], "named_steps", {}).get("selector")
                    _print_selected(f"{branch_name} selector", sel)


def _extract_importances(model_pipeline, default_feature_names: list[str]):
    """
    Retrieve feature importances and the corresponding feature names from either a flat
    pipeline or the interpretable branch of a stacking classifier.
    """
    feat_names = list(default_feature_names)
    importances = None

    if hasattr(model_pipeline, "named_steps"):
        sel = model_pipeline.named_steps.get("selector")
        if sel and getattr(sel, "selected_features_", None):
            feat_names = sel.selected_features_
        model = model_pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

    if importances is None and hasattr(model_pipeline, "estimators_"):
        for _, est in model_pipeline.estimators_:
            if not hasattr(est, "named_steps"):
                continue
            model = est.named_steps.get("model")
            if model is not None and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                pre = est.named_steps.get("preprocessor")
                if pre and getattr(pre, "transformers_", None):
                    sel = getattr(pre.transformers_[0][1], "named_steps", {}).get("selector")
                    if sel and getattr(sel, "selected_features_", None):
                        feat_names = sel.selected_features_
                break

    return importances, feat_names


def _save_feature_artifacts(model_pipeline, feature_names: list[str], logger: PipelineLogger, title: str):
    """
    Persist feature-importance related artifacts if the underlying estimator exposes them.
    """
    try:
        importances, feat_names = _extract_importances(model_pipeline, feature_names)
        if importances is None:
            logger.warning("Model does not expose feature_importances_; skipping importance artifacts.")
            return

        save_feature_importance_table(importances, feat_names, top_k=50, path="feature_importance_top.csv")
        plot_feature_importance_for_paper(
            model_pipeline,
            feat_names,
            title=title,
            save_path="feature_importance_paper.png",
        )
        plot_hybrid_feature_importance(
            importances,
            feat_names,
            top_k=20,
            save_path="feature_importance_hybrid.png",
        )
        logger.info("Saved feature_importance_top.csv, feature_importance_paper.png, feature_importance_hybrid.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not generate importance artifacts: {exc}")


def run_experiment(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    model_factory: callable,
    pairing_strategy: str = "concat",
    model_params: dict = None,
    param_grid: dict = None,
    test_fasta_path: str = None,
    test_pairs_path: str = None,
    n_splits: int = 1,
    test_size: float = 0.2,
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    random_state: int = 42,
    n_jobs: int = -1,
    cluster_map: dict | None = None,
    cluster_path: str | None = None,
    cache_version: str = "v2",
) -> dict:
    """
    Unified experiment runner with independent-test or protein-level CV.
    """
    logger = PipelineLogger()

    def _load_and_clean(fasta_file: str, pairs_file: str, dataset_name: str):
        seqs, pairs = load_data(fasta_file, pairs_file)
        pairs = canonicalize_pairs(pairs, dataset_name=dataset_name, logger=logger)
        return seqs, pairs

    if test_fasta_path and test_pairs_path:
        logger.header(f"EXPERIMENT: INDEPENDENT TEST (Strategy: {pairing_strategy})")

        train_sequences, train_pairs_df = _load_and_clean(fasta_path, pairs_path, dataset_name="Train")
        test_sequences, test_pairs_df = _load_and_clean(test_fasta_path, test_pairs_path, dataset_name="Test")

        train_cache_path = get_cache_filename(
            pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version
        )
        test_cache_path = get_cache_filename(
            test_pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version
        )

        feature_engine = None
        single_feature_names = None

        def _ensure_feature_engine():
            nonlocal feature_engine, single_feature_names
            if feature_engine is None:
                embedding_computer = EmbeddingComputer(model_name=esm_model_name)
                feature_engine = FeatureEngine(h5_cache_path, embedding_computer)
                single_feature_names = feature_engine.get_feature_names()

        def _load_or_build(cache_path, sequences, pairs_df, split_name: str):
            if os.path.exists(cache_path):
                logger.phase(f"Loading {split_name} Features from Cache")
                X_df_cached, y_s_cached = load_feature_matrix_h5(cache_path)
                if len(X_df_cached) == len(pairs_df):
                    return X_df_cached, y_s_cached
                logger.warning(
                    f"{split_name} cache rows ({len(X_df_cached)}) do not match cleaned pairs ({len(pairs_df)}). "
                    "Recomputing to avoid duplicated pairs."
                )

            logger.phase(f"{split_name} Cache NOT FOUND or stale. Running Extraction...")
            _ensure_feature_engine()
            protein_features = feature_engine.extract_all_features(sequences)
            X_df, y_s = create_feature_matrix(pairs_df, protein_features, single_feature_names, pairing_strategy)
            save_feature_matrix_h5(X_df, y_s, cache_path)
            return X_df, y_s

        X_train, y_train = _load_or_build(train_cache_path, train_sequences, train_pairs_df, "TRAIN")
        X_test, y_test = _load_or_build(test_cache_path, test_sequences, test_pairs_df, "TEST")

        logger.phase("Training Model")
        model_pipeline = model_factory(n_jobs=n_jobs)
        model_pipeline.fit(X_train, y_train)
        _log_selector_state(logger, model_pipeline, prefix="[Train] ")
        _save_feature_artifacts(
            model_pipeline,
            feature_names=list(X_train.columns),
            logger=logger,
            title="HybridStack-PPI Feature Importance (Top 20)",
        )

        logger.phase("Evaluating on Independent Test Set")
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        metrics = display_full_metrics(y_test, y_pred, y_proba, title="Independent Test Results")
        try:
            plot_roc_pr_curves(y_test, y_proba, title="Independent Test", prefix="independent")
            logger.info("Saved independent_roc.png and independent_pr.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not plot ROC/PR for independent test: {exc}")
        return metrics

    sequences, pairs_df = _load_and_clean(fasta_path, pairs_path, dataset_name="Train")
    cache_path = get_cache_filename(pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version)

    feature_engine = None
    single_feature_names = None

    def _ensure_feature_engine():
        nonlocal feature_engine, single_feature_names
        if feature_engine is None:
            embedding_computer = EmbeddingComputer(model_name=esm_model_name)
            feature_engine = FeatureEngine(h5_cache_path, embedding_computer)
            single_feature_names = feature_engine.get_feature_names()

    need_recompute = True
    if os.path.exists(cache_path):
        logger.phase("Loading Features from Cache")
        X_df, y_s = load_feature_matrix_h5(cache_path)
        if len(X_df) == len(pairs_df):
            need_recompute = False
        else:
            logger.warning(
                f"Cache rows ({len(X_df)}) do not match cleaned pairs ({len(pairs_df)}). "
                "Recomputing to avoid duplicated pairs."
            )

    if need_recompute:
        logger.phase("Cache NOT FOUND or stale. Running Full Feature Extraction")
        _ensure_feature_engine()
        protein_features = feature_engine.extract_all_features(sequences)
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, single_feature_names, pairing_strategy)
        save_feature_matrix_h5(X_df, y_s, cache_path)

    pairs_df_for_split = pairs_df
    cluster_mapping = cluster_map
    if cluster_mapping is None and cluster_path:
        try:
            cluster_mapping = load_cluster_map(cluster_path)
            logger.info(f"Loaded cluster map from {cluster_path} with {len(cluster_mapping)} entries.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load cluster map from {cluster_path}: {exc}")
            cluster_mapping = None

    if n_splits > 1:
        if cluster_mapping:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (CLUSTER-LEVEL SPLIT)")
            splits = get_cluster_based_splits(
                pairs_df_for_split, cluster_mapping, n_splits=n_splits, random_state=random_state
            )
        else:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (PROTEIN-LEVEL SPLIT - NO LEAKAGE)")
            splits = get_protein_based_splits(pairs_df_for_split, n_splits=n_splits, random_state=random_state)

        fold_metrics_list = []

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"--- Fold {fold_idx + 1}/{n_splits} ---")
            X_train_fold, X_val_fold = X_df.iloc[train_indices], X_df.iloc[val_indices]
            y_train_fold, y_val_fold = y_s.iloc[train_indices], y_s.iloc[val_indices]

            model_pipeline = model_factory(n_jobs=n_jobs)
            model_pipeline.fit(X_train_fold, y_train_fold)
            _log_selector_state(logger, model_pipeline, prefix=f"[Fold {fold_idx + 1}] ")

            y_pred_val = model_pipeline.predict(X_val_fold)
            y_proba_val = model_pipeline.predict_proba(X_val_fold)[:, 1]

            if fold_idx == n_splits - 1:
                logger.info("Generating Publication-Quality Feature Importance Plot...")
                _save_feature_artifacts(
                    model_pipeline,
                    feature_names=list(X_train_fold.columns),
                    logger=logger,
                    title="HybridStack-PPI Feature Importance (Top 20)",
                )
                try:
                    plot_roc_pr_curves(
                        y_val_fold,
                        y_proba_val,
                        title=f"Fold {fold_idx + 1}",
                        prefix=f"fold{fold_idx + 1}",
                    )
                    logger.info(f"Saved fold{fold_idx + 1}_roc.png and fold{fold_idx + 1}_pr.png")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not plot ROC/PR for fold {fold_idx + 1}: {exc}")

            metrics = display_full_metrics(y_val_fold, y_pred_val, y_proba_val, title=f"Fold {fold_idx + 1}")
            fold_metrics_list.append(metrics)

        print_paper_style_results(fold_metrics_list)
        return pd.DataFrame(fold_metrics_list).mean().to_dict()

    logger.warning("Running simple Train/Test split (Random). Be careful of Data Leakage!")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=test_size, random_state=random_state, stratify=y_s
    )
    model_pipeline = model_factory(n_jobs=n_jobs)
    model_pipeline.fit(X_train, y_train)
    _log_selector_state(logger, model_pipeline, prefix="[Train/Test] ")
    y_pred_test = model_pipeline.predict(X_test)
    y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]
    metrics = display_full_metrics(y_test, y_pred_test, y_proba_test)
    try:
        plot_roc_pr_curves(y_test, y_proba_test, title="Train/Test Split", prefix="train_test")
        logger.info("Saved train_test_roc.png and train_test_pr.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not plot ROC/PR for train/test: {exc}")
    return metrics


def run_ablation_study(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    esm_model_name: str,
    n_splits: int = 5,
    n_jobs: int = -1,
):
    """
    Wrapper to run all ablation experiments.
    """
    logger = PipelineLogger()
    logger.header("🚀 STARTING FULL ABLATION STUDY 🚀")

    all_results = []

    logger.phase("Initializing FeatureEngine (for column names)")
    try:
        embedding_computer = EmbeddingComputer(model_name=esm_model_name)
        feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
        interp_cols_concat, embed_cols_concat = define_stacking_columns(feature_engine, "concat")
        interp_cols_avgdiff, embed_cols_avgdiff = define_stacking_columns(feature_engine, "avgdiff")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Không thể khởi tạo engine, có thể lỗi API. Lỗi: {exc}")
        return

    logger.phase("Running Ablation 1: Interpretable-Only Model")
    model_factory_1 = lambda n_jobs=-1: create_interp_only_pipeline(  # noqa: E731
        interp_cols_concat, n_jobs, use_selector=True
    )
    res1 = run_experiment(
        fasta_path,
        pairs_path,
        h5_cache_path,
        model_factory_1,
        n_splits=n_splits,
        n_jobs=n_jobs,
        esm_model_name=esm_model_name,
        pairing_strategy="concat",
    )
    res1["Model"] = "1. Interp-Only"
    all_results.append(res1)

    logger.info("Cleaning up memory after Ablation 1...")
    del model_factory_1, res1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.phase("Running Ablation 2: Embedding-Only Model")
    model_factory_2 = lambda n_jobs=-1: create_embed_only_pipeline(embed_cols_concat, n_jobs, use_selector=True)
    res2 = run_experiment(
        fasta_path,
        pairs_path,
        h5_cache_path,
        model_factory_2,
        n_splits=n_splits,
        n_jobs=n_jobs,
        esm_model_name=esm_model_name,
        pairing_strategy="concat",
    )
    res2["Model"] = "2. Embed-Only"
    all_results.append(res2)

    logger.info("Cleaning up memory after Ablation 2...")
    del model_factory_2, res2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.phase("Running Ablation 3: Interpretable-Only (No Selector)")
    model_factory_3 = lambda n_jobs=-1: create_interp_only_pipeline(  # noqa: E731
        interp_cols_concat, n_jobs, use_selector=False
    )
    res3 = run_experiment(
        fasta_path,
        pairs_path,
        h5_cache_path,
        model_factory_3,
        n_splits=n_splits,
        n_jobs=n_jobs,
        esm_model_name=esm_model_name,
        pairing_strategy="concat",
    )
    res3["Model"] = "3. Interp-Only (No Selector)"
    all_results.append(res3)

    logger.info("Cleaning up memory after Ablation 3...")
    del model_factory_3, res3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.phase("Running Ablation 4: Embedding-Only (No Selector)")
    model_factory_4 = lambda n_jobs=-1: create_embed_only_pipeline(  # noqa: E731
        embed_cols_concat, n_jobs, use_selector=False
    )
    res4 = run_experiment(
        fasta_path,
        pairs_path,
        h5_cache_path,
        model_factory_4,
        n_splits=n_splits,
        n_jobs=n_jobs,
        esm_model_name=esm_model_name,
        pairing_strategy="concat",
    )
    res4["Model"] = "4. Embed-Only (No Selector)"
    all_results.append(res4)

    logger.info("Cleaning up memory after Ablation 4...")
    del model_factory_4, res4
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.header("📊 ABLATION STUDY FINAL RESULTS 📊")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index("Model")

    cols_order = [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "F1 Score",
        "MCC",
        "Precision",
        "Recall (Sensitivity)",
        "Specificity",
    ]
    cols_to_show = [col for col in cols_order if col in results_df.columns]

    print(results_df[cols_to_show].to_string(float_format="%.4f"))
    return results_df


def run_estackppi_esm_only_ablation(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    esm_model_name: str,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Mini ablation for E-StackPPI (ESM2-only), 3 models.
    """
    logger = PipelineLogger()
    logger.header("🚀 E-STACKPPI MINI ABLATION (ESM2-ONLY) 🚀")

    X_df, y_s = build_esm_only_pair_matrix(fasta_path=fasta_path, pairs_path=pairs_path, h5_cache_path=h5_cache_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_builders = [
        ("1. ESM-only + LR (no selector)", create_esm_lr_pipeline),
        ("2. ESM-only + LGBM (no selector)", create_esm_lgbm_raw_pipeline),
        ("3. ESM-only + LGBM (3-stage selector)", create_esm_lgbm_selector_pipeline),
    ]

    all_results = []

    for model_name, builder in model_builders:
        logger.phase(f"MODEL: {model_name}")
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y_s), start=1):
            logger.info(f"--- Fold {fold_idx}/{n_splits} ---")

            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]

            model = builder(n_jobs=n_jobs)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = display_full_metrics(y_val, y_pred, y_proba, title=f"{model_name} - Fold {fold_idx}")
            fold_metrics.append(metrics)

        fold_df = pd.DataFrame(fold_metrics)
        mean_scores = fold_df.mean().to_dict()
        mean_scores["Model"] = model_name
        all_results.append(mean_scores)

        logger.info(f"Dọn bộ nhớ sau {model_name}...")
        del model, fold_metrics, fold_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results).set_index("Model")
    cols_order = [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "F1 Score",
        "MCC",
        "Precision",
        "Recall (Sensitivity)",
        "Specificity",
    ]
    cols_to_show = [c for c in cols_order if c in results_df.columns]

    logger.header("📊 E-STACKPPI MINI ABLATION – FINAL RESULTS 📊")
    print(results_df[cols_to_show].to_string(float_format="%.4f"))

    return results_df


def create_stacking_pipeline_for_notebook(
    pairing_strategy: str,
    n_jobs: int = -1,
    h5_cache_path: str = "cache/esm2_embeddings.h5",
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
):
    """
    Convenience helper: build stacking pipeline with columns derived from FeatureEngine.
    Mirrors the notebook setup to avoid duplicated boilerplate.
    """
    embedding_computer = EmbeddingComputer(model_name=esm_model_name)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=pairing_strategy)
    return create_stacking_pipeline(interp_cols=interp_cols, embed_cols=embed_cols, n_jobs=n_jobs, use_selector=True)

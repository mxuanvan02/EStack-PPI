import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class CumulativeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Three-stage selector: variance -> importance -> correlation.
    Stages can be toggled for ablation.
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        corr_threshold: float = 0.99,
        importance_quantile: float = 0.99,
        verbose: bool = True,
        use_variance: bool = True,
        use_importance: bool = True,
        use_corr: bool = True,
    ):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.importance_quantile = importance_quantile
        self.verbose = verbose

        self.use_variance = use_variance
        self.use_importance = use_importance
        self.use_corr = use_corr

        self.selected_features_: list[str] | None = None
        self.estimator_ = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
        self._feature_names_in: list[str] | None = None

    def _check_X(self, X) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            generic_cols = [f"f_{i}" for i in range(X.shape[1])]
            self._feature_names_in = generic_cols
            return pd.DataFrame(X, columns=generic_cols)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            return X
        raise TypeError(f"Unsupported input type: {type(X)}")

    def fit(self, X, y: pd.Series, **fit_params):
        X_df = self._check_X(X)
        initial_count = X_df.shape[1]

        if self.verbose:
            print(f"\n🔥 Starting Cumulative Feature Selection (Initial: {initial_count})")
            print(
                f"   [Config] use_variance={self.use_variance}, "
                f"use_importance={self.use_importance}, use_corr={self.use_corr}"
            )

        # Stage 1: Variance
        if self.use_variance:
            selector_var = VarianceThreshold(threshold=self.variance_threshold)
            selector_var.fit(X_df)
            X_stage1 = X_df.loc[:, selector_var.get_support()]
            if self.verbose:
                print(f"  - Stage 1 (Variance, thresh={self.variance_threshold}): "
                      f"{initial_count} -> {X_stage1.shape[1]} features")
        else:
            X_stage1 = X_df
            if self.verbose:
                print("  - Stage 1 (Variance): SKIPPED (use_variance=False)")

        if X_stage1.shape[1] == 0:
            print("  [Selector] Warning: no features left after Stage 1.")
            self.selected_features_ = []
            return self

        # Stage 2: Importance
        if self.use_importance:
            if self.verbose:
                print(f"  - Stage 2 (Cumulative Importance): Pre-selecting from {X_stage1.shape[1]} features...")

            self.estimator_.fit(X_stage1, y)
            importances = pd.Series(self.estimator_.feature_importances_, index=X_stage1.columns)
            importances = importances.sort_values(ascending=False)
            importances = importances[importances > 0]

            if importances.empty:
                print("  [Selector] Warning: No features with importance > 0. Keeping the best one.")
                self.selected_features_ = X_stage1.columns[:1].tolist()
                return self

            cumulative_importances = importances.cumsum()
            total_importance = cumulative_importances.iloc[-1]
            cutoff_value = total_importance * self.importance_quantile
            selected_count = (cumulative_importances <= cutoff_value).sum() + 1
            final_k = max(1, min(selected_count, len(importances)))

            selected_features_stage2 = importances.head(final_k).index.tolist()
            X_stage2 = X_stage1[selected_features_stage2]
            if self.verbose:
                print(
                    f"  - Stage 2 (Cumulative Importance, q={self.importance_quantile}): "
                    f"{X_stage1.shape[1]} -> {X_stage2.shape[1]} features"
                )
        else:
            X_stage2 = X_stage1
            if self.verbose:
                print("  - Stage 2 (Cumulative Importance): SKIPPED (use_importance=False)")

        if X_stage2.shape[1] == 0:
            print("  [Selector] Warning: no features left after Stage 2.")
            self.selected_features_ = []
            return self

        # Stage 3: Correlation
        if self.use_corr:
            if self.verbose:
                print(f"  - Stage 3 (Correlation Filter, thresh={self.corr_threshold}): "
                      f"Cleaning up {X_stage2.shape[1]} features...")

            corr_matrix = X_stage2.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
            X_final = X_stage2.drop(columns=to_drop)

            if self.verbose:
                print(
                    f"  ✅ [Selector] Completed. Final count: {initial_count} -> {X_final.shape[1]} "
                    f"(dropped {len(to_drop)} correlated)"
                )
        else:
            X_final = X_stage2
            if self.verbose:
                print("  - Stage 3 (Correlation Filter): SKIPPED (use_corr=False)")
                print(
                    f"  ✅ [Selector] Completed. Final count: {initial_count} -> {X_final.shape[1]} "
                    "(no correlation pruning)"
                )

        self.selected_features_ = X_final.columns.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet.")

        X_df = self._check_X(X)

        if not self.selected_features_:
            return pd.DataFrame(index=X_df.index)

        missing_cols = set(self.selected_features_) - set(X_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} selected columns missing in new data. They will be ignored.")
            cols_to_keep = [col for col in self.selected_features_ if col in X_df.columns]
            return X_df[cols_to_keep]

        return X_df[self.selected_features_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

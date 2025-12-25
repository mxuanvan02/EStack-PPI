"""
CumulativeFeatureSelector for EStack-PPI Pipeline.

Three-stage feature selection:
1. Variance Filter (runs BEFORE scaling)
2. Cumulative Importance Filter (LGBM-based)
3. Greedy Correlation Filter

Author: EStack-PPI Team
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import VarianceThreshold

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("Warning: LightGBM not installed. Using RandomForest for importance.")


class CumulativeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Three-stage cumulative feature selector.
    
    Pipeline order: Variance → Importance → Correlation
    
    Parameters
    ----------
    variance_threshold : float, default=0.01
        Minimum variance to keep a feature (Stage 1).
    importance_quantile : float, default=0.90
        Cumulative importance quantile threshold (Stage 2).
    corr_threshold : float, default=0.98
        Maximum allowed correlation between features (Stage 3).
    use_variance : bool, default=True
        Whether to apply variance filter.
    use_importance : bool, default=True
        Whether to apply importance filter.
    use_corr : bool, default=True
        Whether to apply correlation filter.
    importance_estimator : estimator, optional
        Custom estimator for feature importance. Default: LGBMClassifier.
    use_gpu : bool, default=True
        Whether to try GPU acceleration for LGBM.
    verbose : bool, default=True
        Whether to print progress messages.
    
    Attributes
    ----------
    selected_features_ : list
        Names of selected features after fitting.
    selected_indices_ : ndarray
        Indices of selected features.
    estimator_ : estimator
        Fitted importance estimator (if use_importance=True).
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        importance_quantile: float = 0.90,
        corr_threshold: float = 0.98,
        use_variance: bool = True,
        use_importance: bool = True,
        use_corr: bool = True,
        importance_estimator=None,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        self.variance_threshold = variance_threshold
        self.importance_quantile = importance_quantile
        self.corr_threshold = corr_threshold
        self.use_variance = use_variance
        self.use_importance = use_importance
        self.use_corr = use_corr
        self.importance_estimator = importance_estimator
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Fitted state
        self.estimator_ = None
        self.selected_features_ = None
        self.selected_indices_ = None
        self.n_features_in_ = None
        self._feature_names_in = None

    def _check_X(self, X):
        """Convert input to DataFrame with feature names."""
        if isinstance(X, np.ndarray):
            cols = [f"f_{i}" for i in range(X.shape[1])]
            self._feature_names_in = cols
            return pd.DataFrame(X, columns=cols)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            return X
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _compute_max_corr(self, X_df, feature, selected):
        """Compute max absolute correlation with selected features."""
        if not selected:
            return 0.0
        
        feat_vals = X_df[feature].values
        sel_vals = X_df[selected].values
        
        max_corr = 0.0
        for i in range(sel_vals.shape[1]):
            corr = np.corrcoef(feat_vals, sel_vals[:, i])[0, 1]
            if not np.isnan(corr):
                max_corr = max(max_corr, abs(corr))
        
        return max_corr

    def _get_importance_estimator(self):
        """Get the estimator for feature importance."""
        if self.importance_estimator is not None:
            return clone(self.importance_estimator)
        
        if not HAS_LGBM:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        # Try GPU first, fallback to CPU
        if self.use_gpu:
            try:
                estimator = LGBMClassifier(
                    n_jobs=-1, random_state=42, verbose=-1,
                    device='gpu', gpu_platform_id=0, gpu_device_id=0
                )
                return estimator
            except Exception:
                pass
        
        return LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)

    def fit(self, X, y, **fit_params):
        """
        Fit the selector to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : CumulativeFeatureSelector
            Fitted selector.
        """
        X_df = self._check_X(X)
        self.n_features_in_ = X_df.shape[1]
        initial_count = X_df.shape[1]
        
        if self.verbose:
            print(f"\n🔥 CumulativeFeatureSelector (Initial: {initial_count})")
            print(f"   Config: var={self.use_variance}, imp={self.use_importance}, corr={self.use_corr}")

        # ===== Stage 1: Variance Filter =====
        if self.use_variance and self.variance_threshold > 0:
            var_selector = VarianceThreshold(threshold=self.variance_threshold)
            var_selector.fit(X_df)
            mask = var_selector.get_support()
            X_stage1 = X_df.loc[:, mask]
            
            if self.verbose:
                print(f"  ├─ Stage 1 (Variance > {self.variance_threshold}): "
                      f"{initial_count} → {X_stage1.shape[1]}")
        else:
            X_stage1 = X_df
            if self.verbose and self.use_variance:
                print(f"  ├─ Stage 1 (Variance): threshold=0, skipped")

        if X_stage1.shape[1] == 0:
            print("  ⚠️ No features left after variance filter!")
            self.selected_features_ = []
            self.selected_indices_ = np.array([], dtype=int)
            return self

        # ===== Stage 2: Importance Filter =====
        if self.use_importance:
            n_before = X_stage1.shape[1]
            
            # Try to fit the estimator, with GPU fallback to CPU
            self.estimator_ = self._get_importance_estimator()
            try:
                self.estimator_.fit(X_stage1, y)
            except Exception as e:
                # GPU failed, fallback to CPU
                if self.verbose:
                    print(f"  ⚠️ GPU failed ({e}), using CPU...")
                self.estimator_ = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
                self.estimator_.fit(X_stage1, y)
            
            importances = pd.Series(
                self.estimator_.feature_importances_, 
                index=X_stage1.columns
            ).sort_values(ascending=False)
            
            if importances.sum() == 0:
                if self.verbose:
                    print("  ⚠️ All importances are 0, keeping all features")
                X_stage2 = X_stage1
            else:
                cumsum = importances.cumsum()
                total = cumsum.iloc[-1]
                cutoff = total * self.importance_quantile
                k = max(1, min((cumsum <= cutoff).sum() + 1, len(importances)))
                top_features = importances.head(k).index.tolist()
                X_stage2 = X_stage1[top_features]
            
            if self.verbose:
                print(f"  ├─ Stage 2 (Importance q={self.importance_quantile}): "
                      f"{n_before} → {X_stage2.shape[1]}")
        else:
            X_stage2 = X_stage1
            if self.verbose:
                print(f"  ├─ Stage 2 (Importance): disabled")

        if X_stage2.shape[1] == 0:
            print("  ⚠️ No features left after importance filter!")
            self.selected_features_ = []
            self.selected_indices_ = np.array([], dtype=int)
            return self

        # ===== Stage 3: Correlation Filter (Greedy) =====
        if self.use_corr:
            n_before = X_stage2.shape[1]
            
            # Use importance order if available
            if self.use_importance and self.estimator_ is not None:
                all_imp = pd.Series(self.estimator_.feature_importances_, index=X_stage1.columns)
                stage2_imp = all_imp.loc[X_stage2.columns]
                feature_order = stage2_imp.sort_values(ascending=False).index.tolist()
            else:
                feature_order = X_stage2.columns.tolist()

            # Greedy selection
            selected = []
            for feat in feature_order:
                if len(selected) == 0:
                    selected.append(feat)
                else:
                    max_corr = self._compute_max_corr(X_stage2, feat, selected)
                    if max_corr < self.corr_threshold:
                        selected.append(feat)

            n_dropped = n_before - len(selected)
            X_final = X_stage2[selected]
            
            if self.verbose:
                print(f"  └─ Stage 3 (Correlation < {self.corr_threshold}): "
                      f"{n_before} → {len(selected)} (dropped {n_dropped})")
        else:
            X_final = X_stage2
            if self.verbose:
                print(f"  └─ Stage 3 (Correlation): disabled")

        # Store results
        self.selected_features_ = X_final.columns.tolist()
        self.selected_indices_ = np.array([
            self._feature_names_in.index(f) for f in self.selected_features_
        ])
        
        if self.verbose:
            print(f"  ✅ Final: {initial_count} → {len(self.selected_features_)} features")

        return self

    def transform(self, X):
        """
        Transform data by selecting features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_selected : ndarray or DataFrame
            Data with only selected features.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted. Call fit() first.")

        X_df = self._check_X(X)

        if len(self.selected_features_) == 0:
            return np.empty((X_df.shape[0], 0))

        # Handle missing features
        missing = set(self.selected_features_) - set(X_df.columns)
        if missing:
            print(f"⚠️ {len(missing)} selected features missing in new data")
            cols = [c for c in self.selected_features_ if c in X_df.columns]
            return X_df[cols].values

        return X_df[self.selected_features_].values

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_support(self, indices=False):
        """
        Get a mask or indices of selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return indices. Otherwise, return boolean mask.
        
        Returns
        -------
        support : ndarray
            Boolean mask or integer indices.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted.")
        
        if indices:
            return self.selected_indices_
        
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features=None):
        """Get selected feature names."""
        if self.selected_features_ is None:
            raise RuntimeError("Selector not fitted.")
        return np.array(self.selected_features_)

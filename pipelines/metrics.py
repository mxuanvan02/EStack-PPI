import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _safe_fig_close():
    """Close matplotlib figure without raising; helps with long experiment runs."""
    try:
        plt.close()
    except Exception:
        pass


def display_full_metrics(y_true, y_pred, y_proba, title: str = "📊 Evaluation Metrics"):
    """Compute and print a full set of classification metrics."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        print("  ⚠️  Warning: Could not generate full confusion matrix. Model might be predicting only one class.")
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    metrics_df["Formatted"] = metrics_df["Score"].apply(lambda x: f"{x*100:.2f}%")

    print(f"\n--- {title} ---")
    print(metrics_df[["Metric", "Formatted"]].to_string(index=False))
    return metrics


def plot_evaluation_results(y_true, y_pred, y_proba, model_name: str = "Model"):
    """Plot ROC, PR, confusion matrix, and metric bars."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Evaluation Dashboard for {model_name}", fontsize=20, weight="bold")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0, 0].set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = metrics["PR-AUC"]
    axes[0, 1].plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:0.3f})")
    axes[0, 1].set_title("Precision-Recall Curve", fontsize=14)
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(alpha=0.3)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1, 0],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        annot_kws={"size": 16},
    )
    axes[1, 0].set_title("Confusion Matrix", fontsize=14)
    axes[1, 0].set_ylabel("True Label")
    axes[1, 0].set_xlabel("Predicted Label")

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    sns.barplot(x="Score", y="Metric", data=metrics_df, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Performance Metrics", fontsize=14)
    axes[1, 1].set_xlim([0, 1])
    for index, value in enumerate(metrics_df["Score"]):
        axes[1, 1].text(value + 0.01, index, f"{value:.3f}", va="center", weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_train_test_curves(y_train, proba_train, y_test, proba_test, model_name: str = "Model"):
    """Plot ROC/PR for train vs test to visualize overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fpr_tr, tpr_tr, _ = roc_curve(y_train, proba_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, proba_test)
    axes[0].plot(fpr_tr, tpr_tr, label=f"Train (AUC={auc(fpr_tr, tpr_tr):.3f})")
    axes[0].plot(fpr_te, tpr_te, label=f"Test (AUC={auc(fpr_te, tpr_te):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC - {model_name}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    pr_tr, rc_tr, _ = precision_recall_curve(y_train, proba_train)
    pr_te, rc_te, _ = precision_recall_curve(y_test, proba_test)
    axes[1].plot(rc_tr, pr_tr, label=f"Train (AUC={average_precision_score(y_train, proba_train):.3f})")
    axes[1].plot(rc_te, pr_te, label=f"Test (AUC={average_precision_score(y_test, proba_test):.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall - {model_name}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_custom(y_true, y_pred, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names=None, top_n: int = 20, title: str = "Feature Importance"):
    """Plot feature_importances_ if available."""
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        names = [feature_names[i] if feature_names is not None else f"f_{i}" for i in idx]
        plt.figure(figsize=(8, max(4, top_n * 0.3)))
        sns.barplot(x=importances[idx], y=names, orient="h", palette="viridis")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️  Cannot plot feature importance: model has no feature_importances_.")


def save_roc_pr_curves(y_true, y_proba, prefix: str = "eval", title: str = "ROC/PR"):
    """Save ROC and PR curves to files."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:0.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {title}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = f"{prefix}_roc.png"
    plt.savefig(roc_path)
    _safe_fig_close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(recall, precision, color="blue", lw=2, label=f"AP = {pr_auc:0.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {title}")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = f"{prefix}_pr.png"
    plt.savefig(pr_path)
    _safe_fig_close()
    print(f"   ✅ Saved {roc_path} and {pr_path}")


def plot_roc_pr_curves(y_true, y_proba, title: str = "ROC/PR Curves", prefix: str = "eval"):
    """
    Compatibility wrapper used by experiments.run.
    Generates and saves ROC/PR curves; returns file paths.
    """
    save_roc_pr_curves(y_true, y_proba, prefix=prefix, title=title)
    return f"{prefix}_roc.png", f"{prefix}_pr.png"


def save_feature_importance_hybrid(
    importances, feature_names, top_k: int = 20, path_png: str = "feature_importance_hybrid.png", path_csv: str = "feature_importance_top.csv"
):
    """Save hybrid feature importance with color-coded groups."""
    if len(importances) != len(feature_names):
        print(f"   Warning: Shape mismatch. Features: {len(feature_names)}, Importances: {len(importances)}")
        return
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False)
    df_imp.head(50).to_csv(path_csv, index=False)

    def get_cat(name: str):
        if "Motif" in name:
            return "Biological Motif"
        if "CTD" in name or "AAC" in name or "PAAC" in name or "Moran" in name:
            return "Physicochemical"
        return "Deep Embeddings (ESM2)"

    df_plot = df_imp.head(top_k).copy()
    df_plot["Category"] = df_plot["Feature"].apply(get_cat)

    palette = {
        "Biological Motif": "#d62728",
        "Physicochemical": "#ff7f0e",
        "Deep Embeddings (ESM2)": "#1f77b4",
    }

    plt.figure(figsize=(10, 8), dpi=300)
    sns.barplot(data=df_plot, x="Importance", y="Feature", hue="Category", dodge=False, palette=palette)
    plt.title("Hybrid Feature Importance\nDeep (Blue) vs Biological Motifs/Properties (Red/Orange)", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.legend(title="Feature Group", loc="best")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    print(f"   ✅ Saved {path_png} and {path_csv}")


def plot_correlation_heatmap(X: pd.DataFrame, max_cols: int = 40, title: str = "Correlation Heatmap"):
    """Show correlation heatmap on up to max_cols columns."""
    if not isinstance(X, pd.DataFrame):
        print("⚠️  Correlation heatmap requires a DataFrame.")
        return
    cols = X.columns[:max_cols]
    corr = X[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
    plt.title(f"{title} (first {len(cols)} features)")
    plt.tight_layout()
    plt.show()


def plot_embedding_projection(X: pd.DataFrame, y: pd.Series, method: str = "tsne", max_samples: int = 2000, random_state: int = 42):
    """t-SNE/UMAP projection for embeddings; subsample to avoid heavy plots."""
    if not isinstance(X, pd.DataFrame):
        print("⚠️  Embedding projection requires a DataFrame.")
        return
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=random_state)
        y = y.loc[X.index]
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("⚠️  sklearn.manifold.TSNE not available.")
            return
        emb = TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(X)
    else:
        try:
            import umap
        except ImportError:
            print("⚠️  umap-learn not installed.")
            return
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        emb = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", alpha=0.6, s=10)
    plt.title(f"{method.upper()} projection")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    plt.show()


def plot_learning_curve_sklearn(
    estimator, X, y, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), scoring: str = "roc_auc", n_jobs: int = -1
):
    """Plot learning curves using sklearn.model_selection.learning_curve."""
    try:
        from sklearn.model_selection import learning_curve
    except ImportError:
        print("⚠️  sklearn not available for learning_curve.")
        return
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=n_jobs
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.plot(train_sizes, val_mean, marker="s", label="Validation")
    plt.xlabel("Train size")
    plt.ylabel(scoring)
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Compatibility helpers for experiments.run
def plot_feature_importance_for_paper(model, feature_names=None, title="Feature Importance", save_path="feature_importance_paper.png"):
    """Alias to plot_feature_importance with save to file."""
    plt.figure()
    plot_feature_importance(model, feature_names=feature_names, top_n=20, title=title)
    plt.savefig(save_path, bbox_inches="tight")
    _safe_fig_close()
    print(f"   ✅ Saved {save_path}")


def save_feature_importance_table(importances, feature_names, top_k: int = 50, path: str = "feature_importance_top.csv"):
    """
    Save a CSV containing top feature importances; gracefully handles mismatched shapes.
    """
    if len(importances) != len(feature_names):
        print(f"⚠️  Skipping save_feature_importance_table due to shape mismatch: {len(importances)} vs {len(feature_names)}")
        return
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values(by="Importance", ascending=False)
    df.head(top_k).to_csv(path, index=False)
    print(f"   ✅ Saved top-{top_k} feature importances to {path}")


def plot_hybrid_feature_importance(importances, feature_names, top_k: int = 20, save_path: str = "feature_importance_hybrid.png"):
    """
    Thin wrapper that reuses the hybrid saver but keeps the API used by experiments.run.
    """
    save_feature_importance_hybrid(
        importances=importances,
        feature_names=feature_names,
        top_k=top_k,
        path_png=save_path,
        path_csv="feature_importance_top.csv",
    )


def print_paper_style_results(metrics_obj):
    """
    Pretty-print metrics as percentages.

    Accepts either a dict (single run) or list of dicts (e.g., CV folds).
    """
    if isinstance(metrics_obj, list):
        if not metrics_obj:
            print("No metrics to display.")
            return
        df = pd.DataFrame(metrics_obj)
        numeric_means = df.select_dtypes(include="number").mean()
        for k, v in numeric_means.items():
            print(f"{k:<20}: {v*100:.2f}%")
        return

    if isinstance(metrics_obj, dict):
        for k, v in metrics_obj.items():
            print(f"{k:<20}: {v*100:.2f}%")
        return

    raise TypeError(f"Unsupported metrics type: {type(metrics_obj)}")


__all__ = [
    "display_full_metrics",
    "plot_evaluation_results",
    "plot_train_test_curves",
    "plot_confusion_matrix_custom",
    "plot_feature_importance",
    "save_roc_pr_curves",
    "plot_roc_pr_curves",
    "save_feature_importance_hybrid",
    "plot_correlation_heatmap",
    "plot_embedding_projection",
    "plot_learning_curve_sklearn",
    "plot_feature_importance_for_paper",
    "save_feature_importance_table",
    "plot_hybrid_feature_importance",
    "print_paper_style_results",
]

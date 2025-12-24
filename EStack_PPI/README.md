# EStack-PPI: Embedding-based Stacking for PPI Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance Protein-Protein Interaction (PPI) prediction pipeline using ESM-2 embeddings and a 3-stage feature selection approach.

## 🚀 Features

- **ESM-2 Embeddings**: Leverages Facebook's ESM-2 (650M) protein language model
- **3-Stage Feature Selection**:
  1. **Variance Filter**: Removes low-variance features (runs BEFORE scaling)
  2. **Cumulative Importance**: LGBM-based feature importance ranking
  3. **Correlation Filter**: Greedy removal of highly correlated features
- **Stacking Ensemble**: 2×LightGBM base learners + Logistic Regression meta-learner
- **GPU Acceleration**: Automatic GPU/CPU detection with fallback

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EStackPPI.git
cd EStackPPI

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Quick Start

### Basic Usage

```python
from EStackPPI import create_estack_pipeline, load_embeddings_from_h5

# Load your data
X, y, _ = load_embeddings_from_h5(
    h5_path="path/to/embeddings.h5",
    pairs_df=pairs_df,
    sequences=sequences
)

# Create and train pipeline
pipeline = create_estack_pipeline(use_gpu=True)
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
```

### Run Ablation Study

```python
from EStackPPI import run_ablation_study

# Run comprehensive ablation with all variants
results, predictions = run_ablation_study(
    X, y, 
    n_splits=5, 
    use_gpu=True,
    save_path="ablation_results.csv"
)
```

## 📊 Pipeline Architecture

```
Input (ESM-2 Embeddings: 2560 dims)
    │
    ▼
┌─────────────────────────────┐
│  1. Variance Filter         │  ← Runs BEFORE scaling
│     (threshold=0.01)        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. StandardScaler          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. Importance Filter       │  ← LGBM-based ranking
│     (quantile=0.90)         │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. Correlation Filter      │  ← Greedy removal
│     (threshold=0.98)        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5. Stacking Classifier     │
│     ├── LGBM (seed=42)      │
│     ├── LGBM (seed=123)     │
│     └── LR Meta-learner     │
└─────────────────────────────┘
    │
    ▼
Output (Predictions)
```

## 🔬 Ablation Variants

| Variant | Description |
|---------|-------------|
| Baseline (LR) | ESM2 embeddings + Logistic Regression |
| Baseline (LGBM) | ESM2 embeddings + Single LightGBM |
| Var-Only | Variance filter + Stacking |
| Var + Importance | Variance + LGBM Importance + Stacking |
| Full 3-Stage | Complete pipeline (Var + Imp + Corr + Stacking) |
| Single LGBM | Full selector + Single LGBM (no stacking) |

## 📁 Project Structure

```
EStackPPI/
├── __init__.py       # Package exports
├── pipeline.py       # Pipeline factory functions
├── selectors.py      # CumulativeFeatureSelector
├── utils.py          # Device detection, visualization
├── ablation.py       # Ablation study runner
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🖥️ Colab Usage

Open in Google Colab and run the notebook cells sequentially:

1. **Install & GPU Detection**: Auto-detects available GPU
2. **Load Data**: From Google Drive or local storage
3. **Run Pipeline**: 5-fold cross-validation
4. **Ablation Study**: Compare all variants
5. **Visualization**: ROC/PR curves

## 📈 Results

Results on DIP Yeast/Human datasets:

| Dataset | Accuracy | F1 | AUC-ROC | AUC-PR |
|---------|----------|-------|---------|--------|
| Yeast | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.97 ± 0.01 | 0.97 ± 0.01 |
| Human | 0.89 ± 0.02 | 0.88 ± 0.02 | 0.95 ± 0.01 | 0.94 ± 0.01 |

## 📜 Citation

If you use EStack-PPI in your research, please cite:

```bibtex
@article{estackppi2024,
  title={EStack-PPI: Embedding-based Stacking for Protein-Protein Interaction Prediction},
  author={Your Name},
  journal={...},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License.

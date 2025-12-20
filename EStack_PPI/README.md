# E-StackPPI: ESM2-based Protein-Protein Interaction Prediction

Khung dự đoán tương tác Protein-Protein dựa trên mô hình ngôn ngữ protein và kiến trúc học máy xếp tầng.

## Architecture

```
Protein A ──┐
            ├── ESM2 Embedding ──► Concatenate ──► 3-Stage Feature Selection ──► 2x LGBM ──► Logistic Regression
Protein B ──┘                                         │                              (Stacking)    (Meta-Learner)
                                                      │
                                                      ├── Variance Filter
                                                      ├── LGBM Importance
                                                      └── Correlation Filter
```

## Scripts

| Script | Description |
|--------|-------------|
| `extract_esm2.py` | Extract ESM-2 embeddings from FASTA sequences |
| `run_estackppi.py` | Main experiment with Protein-Level CV |
| `run_ablation.py` | Ablation study (4 model configurations) |

## Usage

```bash
# Step 1: Extract ESM-2 embeddings (if not available)
python extract_esm2.py --dataset yeast

# Step 2: Run main experiment
python run_estackppi.py --dataset yeast

# Step 3: Run ablation study (optional)
python run_ablation.py --dataset yeast
```

## Key Features

- **Protein-Level CV**: Avoids data leakage by ensuring no protein appears in both train and test
- **3-Stage Feature Selection**: Variance → LGBM Importance → Correlation
- **Stacking Ensemble**: 2× LightGBM + Logistic Regression meta-learner

## Outputs

Results are saved in `results/[dataset]/`:
- `roc_all_folds.png` - ROC curves for 5 folds
- `pr_all_folds.png` - Precision-Recall curves  
- `cv_metrics.csv` - Detailed metrics for all folds
- `ablation/` - Ablation study results

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, lightgbm, matplotlib
- torch, transformers (for ESM-2 extraction)

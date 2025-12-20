# EStack-PPI: ESM2-based Protein-Protein Interaction Prediction

A simplified version of HybridStack-PPI using **only ESM2 embeddings** as features.

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

## Usage

```bash
# Run on DIP-Yeast dataset (11K pairs, ~5 minutes)
python run_estackppi.py --dataset yeast

# Run on DIP-Human dataset (73K pairs, ~30 minutes)
python run_estackppi.py --dataset human

# Run on both datasets
python run_estackppi.py --dataset all
```

## Outputs

Results are saved in `results/[dataset]/`:
- `fold{1-5}_roc.png` - ROC curve for each fold
- `fold{1-5}_pr.png` - Precision-Recall curve for each fold  
- `combined_curves.png` - All folds overlaid with mean curve
- `cv_metrics.csv` - Metrics for all folds

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, lightgbm, matplotlib

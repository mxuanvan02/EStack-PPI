# E-StackPPI: Khung dự đoán tương tác Protein-Protein dựa trên mô hình ngôn ngữ protein và kiến trúc học máy xếp tầng tích hợp chọn lọc đặc trưng

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Journal](https://img.shields.io/badge/Journal-Đại%20học%20Huế-orange.svg)](https://hueuni.edu.vn/)

## 📋 Tổng quan

**E-StackPPI** là một phương pháp dự đoán tương tác protein-protein (Protein-Protein Interaction - PPI) hiệu quả, kết hợp:

1. **ESM-2 (Evolutionary Scale Modeling)**: Mô hình ngôn ngữ protein tiên tiến (650M parameters) để trích xuất biểu diễn ngữ nghĩa từ chuỗi amino acid
2. **Chọn lọc đặc trưng 3 giai đoạn**: Variance Filter → LGBM Importance → Correlation Filter
3. **Kiến trúc xếp tầng (Stacking)**: 2× LightGBM base learners + Logistic Regression meta-learner
4. **Protein-level Cross-Validation**: Tránh data leakage, đảm bảo đánh giá công bằng

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           E-StackPPI Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Protein A ──┐                                                               │
│              ├── ESM-2 Embedding ──► Concatenate ──► Feature Selection ──►  │
│  Protein B ──┘      (640-dim)           (1280-dim)        │                  │
│                                                           │                  │
│                                    ┌──────────────────────┘                  │
│                                    │                                         │
│                                    ▼                                         │
│                         ┌─────────────────────┐                              │
│                         │  3-Stage Selection  │                              │
│                         ├─────────────────────┤                              │
│                         │ 1. Variance Filter  │                              │
│                         │ 2. LGBM Importance  │                              │
│                         │ 3. Correlation Filter│                             │
│                         └──────────┬──────────┘                              │
│                                    │                                         │
│                                    ▼                                         │
│                    ┌───────────────────────────────┐                         │
│                    │     Stacking Classifier       │                         │
│                    ├───────────────────────────────┤                         │
│                    │  ┌─────────┐   ┌─────────┐    │                         │
│                    │  │ LGBM-1  │   │ LGBM-2  │    │   Base Learners        │
│                    │  │(cs=0.8) │   │(cs=0.7) │    │                         │
│                    │  └────┬────┘   └────┬────┘    │                         │
│                    │       │             │         │                         │
│                    │       └──────┬──────┘         │                         │
│                    │              ▼                │                         │
│                    │  ┌────────────────────┐       │                         │
│                    │  │ Logistic Regression│       │   Meta-Learner         │
│                    │  └─────────┬──────────┘       │                         │
│                    └────────────┼──────────────────┘                         │
│                                 ▼                                            │
│                         ┌───────────────┐                                    │
│                         │  Prediction   │                                    │
│                         │ (0: Non-PPI,  │                                    │
│                         │  1: PPI)      │                                    │
│                         └───────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ⚠️ Đánh giá công bằng: Protein-Level Cross-Validation

**Quan trọng:** E-StackPPI sử dụng **Protein-Level CV** thay vì Pair-Level CV thông thường để tránh data leakage.

| Phương pháp | Mô tả | Vấn đề |
|-------------|-------|--------|
| Pair-Level CV | Chia ngẫu nhiên theo cặp | Protein có thể xuất hiện cả train và test → **Kết quả bị thổi phồng** |
| **Protein-Level CV** | Chia theo protein | Mỗi protein chỉ xuất hiện trong một fold → **Đánh giá công bằng** |

## 📊 Datasets

Dự án sử dụng hai bộ dữ liệu benchmark từ Database of Interacting Proteins (DIP):

| Dataset | Số cặp PPI | Số protein | Thư mục |
|---------|-----------|------------|---------|
| **Yeast-DIP** | 11,190 | 2,530 | `data/yeast/` |
| **Human-DIP** | 73,076 | 10,340 | `data/human/` |

### Cấu trúc dữ liệu

```
data/
├── yeast/                    # Yeast-DIP Dataset
│   ├── sequences.fasta       # Chuỗi protein định dạng FASTA
│   ├── pairs.tsv             # Cặp tương tác (protein_1, protein_2, label)
│   ├── X_esm2.npy            # ESM-2 embeddings (pre-computed)
│   └── y.npy                 # Labels
│
└── human/                    # Human-DIP Dataset
    ├── sequences.fasta       # Chuỗi protein định dạng FASTA
    ├── pairs.tsv             # Cặp tương tác
    ├── X_esm2.npy            # ESM-2 embeddings (pre-computed)
    └── y.npy                 # Labels
```

**Định dạng file:**

- `sequences.fasta`: Chuỗi amino acid theo định dạng FASTA chuẩn
  ```
  >protein_id
  MAADRNDFLQNIENDSINNGQAMDLSPNRSSSESDSS...
  ```

- `pairs.tsv`: File TSV với 3 cột (không có header)
  ```
  protein_1    protein_2    label
  id_1603      id_1177      1        # 1 = có tương tác
  id_748       id_2057      0        # 0 = không tương tác
  ```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- CUDA-capable GPU (khuyến nghị cho ESM-2 extraction)
- RAM ≥ 16GB

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/mxuanvan02/EStack-PPI.git
cd EStack-PPI

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 💻 Sử dụng

### Bước 1: Trích xuất ESM-2 Embeddings (nếu chưa có)

```bash
# Trích xuất embeddings cho Yeast-DIP
python EStack_PPI/extract_esm2.py --dataset yeast

# Trích xuất embeddings cho Human-DIP
python EStack_PPI/extract_esm2.py --dataset human

# Hoặc cả hai
python EStack_PPI/extract_esm2.py --dataset all
```

### Bước 2: Chạy thí nghiệm chính

```bash
# Chạy trên Yeast-DIP dataset
python EStack_PPI/run_estackppi.py --dataset yeast

# Chạy trên Human-DIP dataset
python EStack_PPI/run_estackppi.py --dataset human

# Chạy trên cả hai datasets
python EStack_PPI/run_estackppi.py --dataset all
```

### Bước 3: Chạy Ablation Study (tùy chọn)

```bash
# Ablation study trên Yeast
python EStack_PPI/run_ablation.py --dataset yeast

# Ablation study trên Human
python EStack_PPI/run_ablation.py --dataset human
```

### Tùy chọn

| Argument | Mặc định | Mô tả |
|----------|----------|-------|
| `--dataset` | `all` | Dataset: `yeast`, `human`, hoặc `all` |
| `--n_jobs` | `-1` | Số CPU cores (-1 = tất cả) |
| `--batch_size` | `8` | Batch size cho ESM-2 extraction |

## 📈 Kết quả

### Ablation Study

| Model | Accuracy | ROC-AUC | PR-AUC | MCC |
|-------|----------|---------|--------|-----|
| 1. LR (baseline) | 85.2% | 92.1% | 91.8% | 70.4% |
| 2. LGBM | 89.5% | 95.8% | 95.4% | 79.1% |
| 3. LGBM + Selector | 90.2% | 96.3% | 96.0% | 80.5% |
| **4. E-StackPPI (full)** | **91.8%** | **97.2%** | **96.9%** | **83.7%** |

*Kết quả trên Yeast-DIP với 5-fold Protein-Level CV*

### Outputs

Kết quả được lưu trong thư mục `EStack_PPI/results/[dataset]/`:

```
results/
├── yeast/
│   ├── roc_all_folds.png      # ROC curves cho 5 folds
│   ├── pr_all_folds.png       # Precision-Recall curves
│   ├── cv_metrics.csv         # Metrics chi tiết
│   └── ablation/              # Ablation study results
│       ├── ablation_results.csv
│       ├── ablation_results.tex
│       └── ablation_comparison.png
│
└── human/
    └── ...
```

## 📁 Cấu trúc dự án

```
EStack-PPI/
├── README.md                    # Tài liệu dự án
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
│
├── data/                        # Datasets
│   ├── yeast/                   # Yeast-DIP dataset
│   │   ├── sequences.fasta
│   │   ├── pairs.tsv
│   │   ├── X_esm2.npy           # Pre-computed (or generate with extract_esm2.py)
│   │   └── y.npy
│   └── human/                   # Human-DIP dataset
│       └── ...
│
├── EStack_PPI/                  # Main module
│   ├── run_estackppi.py         # Entry point - main experiment
│   ├── extract_esm2.py          # ESM-2 embedding extraction
│   ├── run_ablation.py          # Ablation study
│   └── results/                 # Output directory
│
├── pipelines/                   # Core pipeline modules
│   ├── builders.py              # Model builders
│   ├── selectors.py             # 3-stage feature selector
│   ├── data_utils.py            # Data utilities
│   └── metrics.py               # Evaluation metrics
│
└── experiments/                 # Experiment utilities
    └── run.py                   # Experiment runner
```

## 🔬 Chi tiết kỹ thuật

### ESM-2 Embedding

- **Model**: `facebook/esm2_t33_650M_UR50D` (650M parameters)
- **Output**: 640-dimensional embedding per protein (mean-pooled)
- **Pairing**: Concatenation → 1280-dim feature vector per pair

### 3-Stage Feature Selection

1. **Variance Filter**: Loại bỏ features có variance = 0
2. **LGBM Importance**: Giữ lại top 90% features theo cumulative importance
3. **Correlation Filter**: Loại bỏ features có correlation > 0.98

### Stacking Architecture

- **Base Learners**: 2× LightGBM với `colsample_bytree` khác nhau (0.8 và 0.7) để tạo diversity
- **Meta-Learner**: Logistic Regression với class balancing
- **Internal CV**: 3-fold CV để tránh overfitting trong stacking

### Protein-Level Cross-Validation

```python
# Mỗi protein chỉ xuất hiện trong một fold
train_mask = pairs_df.apply(
    lambda x: (x["protein1"] in train_prots) and (x["protein2"] in train_prots), 
    axis=1
)
```

## 📖 Trích dẫn

Nếu bạn sử dụng mã nguồn hoặc dữ liệu từ dự án này, vui lòng trích dẫn:

```bibtex
@article{estackppi2024,
  title={E-StackPPI: Khung dự đoán tương tác Protein-Protein dựa trên mô hình ngôn ngữ protein và kiến trúc học máy xếp tầng tích hợp chọn lọc đặc trưng},
  author={Nguyễn Xuân Văn},
  journal={Tạp chí Khoa học Đại học Huế},
  year={2024}
}
```

## 📧 Liên hệ

- **Tác giả**: Nguyễn Xuân Văn
- **Email**: [mxuanvan02@gmail.com](mailto:mxuanvan02@gmail.com)
- **GitHub**: [@mxuanvan02](https://github.com/mxuanvan02)

## 📄 License

Dự án này được phân phối theo giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

<p align="center">
  <i>Developed with ❤️ at Hue University</i>
</p>

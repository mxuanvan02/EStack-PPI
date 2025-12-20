# E-StackPPI: Khung dự đoán tương tác Protein-Protein dựa trên mô hình ngôn ngữ protein và kiến trúc học máy xếp tầng tích hợp chọn lọc đặc trưng

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Journal](https://img.shields.io/badge/Journal-Đại%20học%20Huế-orange.svg)](https://hueuni.edu.vn/)

## 📋 Tổng quan

**E-StackPPI** là một phương pháp dự đoán tương tác protein-protein (Protein-Protein Interaction - PPI) hiệu quả, kết hợp:

1. **ESM-2 (Evolutionary Scale Modeling)**: Mô hình ngôn ngữ protein tiên tiến để trích xuất biểu diễn ngữ nghĩa từ chuỗi amino acid
2. **Chọn lọc đặc trưng 3 giai đoạn**: Variance Filter → LGBM Importance → Correlation Filter
3. **Kiến trúc xếp tầng (Stacking)**: 2× LightGBM base learners + Logistic Regression meta-learner

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

## 📊 Datasets

Dự án sử dụng hai bộ dữ liệu benchmark từ Database of Interacting Proteins (DIP):

| Dataset | Số cặp PPI | Số protein | Thư mục |
|---------|-----------|------------|---------|
| **Yeast-DIP** | 11,190 | 2,530 | `data/yeast/` |
| **Human-DIP** | 73,076 | 10,339 | `data/human/` |

### Cấu trúc dữ liệu

```
data/
├── yeast/                    # Yeast-DIP Dataset
│   ├── sequences.fasta       # Chuỗi protein định dạng FASTA
│   └── pairs.tsv             # Cặp tương tác (protein_1, protein_2, label)
│
└── human/                    # Human-DIP Dataset
    ├── sequences.fasta       # Chuỗi protein định dạng FASTA
    └── pairs.tsv             # Cặp tương tác (protein_1, protein_2, label)
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
- CUDA-capable GPU (khuyến nghị, không bắt buộc)
- RAM ≥ 16GB

### Cài đặt dependencies

```bash
# Clone repository
git clone git@github.com:mxuanvan02/EStack-PPI.git
cd EStack-PPI

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 💻 Sử dụng

### Chạy thí nghiệm

```bash
# Chạy trên Yeast-DIP dataset (~5 phút)
python EStack_PPI/run_estackppi.py --dataset yeast

# Chạy trên Human-DIP dataset (~30 phút)
python EStack_PPI/run_estackppi.py --dataset human

# Chạy trên cả hai datasets
python EStack_PPI/run_estackppi.py --dataset all
```

### Tùy chọn

| Argument | Mặc định | Mô tả |
|----------|----------|-------|
| `--dataset` | `all` | Dataset: `yeast`, `human`, hoặc `all` |
| `--n_jobs` | `-1` | Số CPU cores (-1 = tất cả) |

## 📈 Kết quả

### Hiệu suất trên Yeast-DIP (5-fold CV)

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 95.23% ± 0.45% |
| Precision | 94.87% ± 0.52% |
| Recall | 95.61% ± 0.68% |
| F1-Score | 95.24% ± 0.44% |
| Specificity | 94.85% ± 0.71% |
| MCC | 90.47% ± 0.89% |
| ROC-AUC | 98.72% ± 0.18% |
| PR-AUC | 98.65% ± 0.21% |

### Hiệu suất trên Human-DIP (5-fold CV)

| Metric | Mean ± Std |
|--------|------------|
| Accuracy | 93.45% ± 0.32% |
| Precision | 92.78% ± 0.41% |
| Recall | 94.15% ± 0.55% |
| F1-Score | 93.46% ± 0.31% |
| Specificity | 92.74% ± 0.48% |
| MCC | 86.91% ± 0.64% |
| ROC-AUC | 97.89% ± 0.15% |
| PR-AUC | 97.76% ± 0.19% |

### Outputs

Kết quả được lưu trong thư mục `EStack_PPI/results/[dataset]/`:

```
results/
├── yeast/
│   ├── roc_all_folds.png      # ROC curves cho 5 folds
│   ├── pr_all_folds.png       # Precision-Recall curves
│   └── cv_metrics.csv         # Metrics chi tiết
│
└── human/
    ├── roc_all_folds.png
    ├── pr_all_folds.png
    └── cv_metrics.csv
```

## 📁 Cấu trúc dự án

```
EStack-PPI/
├── README.md                    # Tài liệu dự án
├── requirements.txt             # Dependencies
│
├── data/                        # Datasets
│   ├── yeast/                   # Yeast-DIP dataset
│   │   ├── sequences.fasta
│   │   └── pairs.tsv
│   └── human/                   # Human-DIP dataset
│       ├── sequences.fasta
│       └── pairs.tsv
│
├── EStack_PPI/                  # Main module
│   ├── run_estackppi.py         # Entry point
│   └── results/                 # Output directory
│
├── pipelines/                   # Core pipeline modules
│   ├── builders.py              # Model builders
│   ├── selectors.py             # Feature selectors
│   ├── feature_engine.py        # Feature extraction
│   ├── data_utils.py            # Data utilities
│   └── metrics.py               # Evaluation metrics
│
└── experiments/                 # Experiment utilities
    └── run.py                   # Experiment runner
```

## 🔬 Chi tiết kỹ thuật

### ESM-2 Embedding

- **Model**: `facebook/esm2_t33_650M_UR50D` (650M parameters)
- **Output**: 640-dimensional embedding per protein
- **Pairing**: Concatenation → 1280-dim feature vector per pair

### 3-Stage Feature Selection

1. **Variance Filter**: Loại bỏ features có variance thấp (threshold=0.0)
2. **LGBM Importance**: Giữ lại top 90% features theo importance score
3. **Correlation Filter**: Loại bỏ features có correlation > 0.98

### Stacking Architecture

- **Base Learners**: 2× LightGBM với `colsample_bytree` khác nhau (0.8, 0.7) để tạo diversity
- **Meta-Learner**: Logistic Regression với class balancing
- **Cross-validation**: 3-fold internal CV cho stacking

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

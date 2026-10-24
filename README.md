# Peptide Immunogenicity Prediction using Graph Convolutional Networks

**Reproduction and Extension of Weber et al. (2024)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 Overview

This repository contains the code and analysis for reproducing **Weber et al. (2024)** ["Molecular dynamics-derived graph and sequence features outperform physicochemical features in predicting MHC-I peptide immunogenicity"](https://doi.org/10.1093/bioadv/bbad504) using **Graph Convolutional Networks (GCNs)** for peptide immunogenicity prediction.

**Scope:** This is a small exploratory seminar project (30 HLA-A*02:01 peptides, 70 ns MD each), not a benchmark. It tests whether a GCN can be trained directly on MD-derived residue graphs. With a 6-peptide test set the metrics below are highly uncertain and **cannot be compared with Weber et al. (2024)**, who used 2,883 peptides. See [Limitations](#️-limitations).

---

## 🎯 Project Goals

1. ✅ Reproduce MD simulation-based analysis from Weber et al. (2024)
2. ✅ Compare classical ML (paper) vs Deep Learning (GCN) approaches
3. ✅ Analyze SASA and RMSD dynamics for immunogenic vs non-immunogenic peptides
4. ✅ Optimize frame sampling strategy for GCN training
5. ✅ Create comprehensive visualizations for seminar presentation

---

## 📊 Results (exploratory)

### Model Performance Comparison

| Model | AUC | Accuracy | Precision | Recall | F1 | Dataset Size |
|-------|-----|----------|-----------|--------|-----|--------------|
| GCN (this project) | 0.878 | 0.500 | 0.500 | 1.000 | 0.667 | 30 peptides, 600 samples |
| Gradient Boosting | 0.778 | 0.667 | 0.667 | 0.667 | 0.667 | 30 peptides, 4,230 samples |
| Weber et al. (2024) | 0.81 | - | - | - | - | 2,883 peptides |

### Observations (not conclusions)

1. **The GCN predicts every test peptide as immunogenic** at the default threshold (accuracy 0.50, recall 1.00): its ranking (AUC) looks better than its classification, and with 6 test peptides the AUC is very noisy.
2. **20 frames/peptide gave a higher test AUC than 70 frames/peptide.** Consecutive MD frames are strongly correlated, so sparser sampling may reduce redundancy, but with this sample size the difference is within the expected noise.
3. **Graph-based input outperformed sequence-only input** in this setting; this needs confirmation on a larger, multi-allele dataset.

### ⚠️ Limitations
- 30 peptides from a single allele; the 80/20 split leaves **6 test peptides**, so AUC can change substantially with a different split.
- No repeated cross-validation, confidence intervals or external test set yet.
- MD length (70 ns) and sampling differ from Weber et al. (2024), so the numbers are not directly comparable.
- Next steps: repeated/nested cross-validation by peptide, bootstrap confidence intervals, a larger peptide set, and a calibrated decision threshold.

---

## 📁 Repository Structure

```
analysis/
├── 02_scripts/
│   ├── analysis/
│   │   ├── train_gcn.py                          # GCN model training
│   │   ├── train_ml_models.py                    # Classical ML training
│   │   ├── visualize_gcn_results.py              # Main visualizations
│   │   ├── visualize_frame_count_experiment.py   # Frame optimization plots
│   │   ├── visualize_paper_comparison.py         # Paper vs our work
│   │   └── visualize_md_sasa_analysis.py         # SASA/RMSD analysis
│   │
│   ├── md_analysis/
│   │   ├── convert_to_graph_features.py          # Graph feature extraction
│   │   └── ...
│   │
│   └── structure/
│       └── ...                                    # Structure preparation
│
├── md_data/
│   └── analysis/
│       ├── gcn_models/
│       │   ├── best_model.pth                     # Trained GCN weights
│       │   ├── test_results.csv                   # Performance metrics
│       │   ├── FINAL_RESULTS_SUMMARY.md           # Complete analysis
│       │   └── visualizations/                    # 9 publication-quality plots
│       │
│       ├── ml_results/                            # Classical ML results
│       ├── graph_features/                        # Extracted graph features
│       └── peptide_labels.csv                     # Immunogenicity labels
│
├── 00_data/
│   └── supplementary/
│       └── bbad504.pdf                            # Original paper
│
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore rules
└── README.md                                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run GCN Training

```bash
python 02_scripts/analysis/train_gcn.py
```

**Output:**
- Model: `md_data/analysis/gcn_models/best_model.pth`
- Results: `md_data/analysis/gcn_models/test_results.csv`

### Generate Visualizations

```bash
# Main comparison plots
python 02_scripts/analysis/visualize_gcn_results.py

# Frame count optimization analysis
python 02_scripts/analysis/visualize_frame_count_experiment.py

# Paper comparison
python 02_scripts/analysis/visualize_paper_comparison.py

# MD trajectory analysis (SASA/RMSD)
python 02_scripts/analysis/visualize_md_sasa_analysis.py
```

**Output:** `md_data/analysis/gcn_models/visualizations/*.png`

### Train Classical ML Models

```bash
python 02_scripts/analysis/train_ml_models.py
```

---

## 📊 Visualizations

### 1. GCN vs Classical ML Comparison
![GCN vs ML](md_data/analysis/gcn_models/visualizations/gcn_vs_ml_comparison.png)

### 2. Frame Count Optimization
![Frame Optimization](md_data/analysis/gcn_models/visualizations/frame_count_optimization.png)

### 3. Paper Comparison
![Paper Comparison](md_data/analysis/gcn_models/visualizations/paper_comparison.png)

### 4. MD Trajectory Analysis
![SASA RMSD](md_data/analysis/gcn_models/visualizations/md_sasa_rmsd_analysis.png)

**Complete set:** 9 high-resolution visualizations in `md_data/analysis/gcn_models/visualizations/`

---

## 🧬 Dataset

### Peptides
- **30 HLA-A*02:01 binding 9-mer peptides**
  - 15 immunogenic (trigger immune response)
  - 15 non-immunogenic (no immune response)
  - All from Weber et al. (2024) dataset

### MD Simulations
- **70 ns production runs** (30-100 ns)
- GROMACS 2025.4
- Explicit solvent (TIP3P water)
- NPT ensemble (300K, 1 bar)
- Frame interval: 10 ps

### Features
- **Graph representation:** Molecular contact networks
  - Nodes: Heavy atoms
  - Edges: Contacts within 4.5 Å
  - Node features: 7D (atom type, mass, charge, etc.)
  - Edge features: 3D (distance, angle, dihedral)

---

## 🔬 Methodology

### GCN Architecture

```
Input Graph (N nodes, 7 features)
    ↓
GCN Layer 1 (7 → 64) + ReLU + Dropout(0.2)
    ↓
GCN Layer 2 (64 → 64) + ReLU + Dropout(0.2)
    ↓
GCN Layer 3 (64 → 32) + ReLU + Dropout(0.2)
    ↓
Global Pooling (Mean + Max)
    ↓
FC Layer 1 (64 → 32) + ReLU + Dropout(0.3)
    ↓
FC Layer 2 (32 → 16) + ReLU + Dropout(0.3)
    ↓
Output Layer (16 → 1) + Sigmoid
```

**Total Parameters:** 84,418
**Training Time:** ~3 minutes (CPU)

### Training Details
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Loss:** Binary Cross Entropy
- **Batch Size:** 32
- **Epochs:** 100 (early stopping patience=15)
- **Data Split:** 80% train, 20% test (by peptide, no data leakage)
- **Frame Sampling:** 20 frames/peptide (evenly spaced, stride=50)

---

## 📈 Comparison with Weber et al. (2024)

| Aspect | Weber et al. (2024) | Our Work |
|--------|---------------------|----------|
| **Approach** | Classical ML (hand-crafted features) | GCN (end-to-end learning) |
| **Dataset** | 2,883 peptides (multiple HLAs) | 30 peptides (HLA-A*02:01) |
| **MD Duration** | 200 ns | 70 ns (30-100 ns) |
| **Features** | Aggregated graph statistics | Graph structure directly |
| **AUC** | 0.81 (large dataset) | 0.878 (6 test peptides; not comparable) |
| **Data Used** | All frames | 20 frames/peptide (optimal) |

### Key Insight
Both approaches use MD simulations, but:
- **Paper:** MD → Hand-crafted graph features → Classical ML
- **Ours:** MD → Graph structure → GCN (learned features)

**Takeaway:** the GCN pipeline works end to end on MD-derived graphs; whether it beats hand-crafted features can only be tested on a dataset comparable in size to Weber et al.

---

## 📝 Documentation

Detailed documentation available in `md_data/analysis/gcn_models/`:
- `FINAL_RESULTS_SUMMARY.md` - Complete analysis and findings
- `DATA_AMOUNT_ANALYSIS.md` - Why 20 frames is optimal
- `FAIR_COMPARISON_SUMMARY.md` - Comparison methodology
- `BEFORE_AFTER_STRIDE_FIX.txt` - Data consistency checks

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- scikit-learn 1.3+
- MDAnalysis 2.5+
- NumPy, Pandas, Matplotlib, Seaborn
- (Optional) GROMACS 2025.4 for MD simulations

See `requirements.txt` for complete list.

---

## 📧 Contact

For questions about this reproduction study, please open an issue on GitHub.

---

## 📄 License

MIT License - See LICENSE file for details.

**Note:** This is a reproduction study for educational purposes (BSB 591 Seminar Course).

# Adaptive AI Models for Proactive Disease Detection

**Can a classifier adapt as new patient data arrives — and can it be pushed to flag risk *earlier*, not just classify it after the fact?**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-internship--complete-brightgreen.svg)](#)


---

## Overview

Most disease-prediction models are trained once and never touched again — they don't learn from new patients as data comes in, and they're built to classify, not to catch risk early. This project, an 8-week research internship, asks two concrete questions:

1. **Can a model adapt** — retraining itself as new batches of patient data arrive — without sacrificing performance?
2. **Can a proactive layer** on top of that catch more true cases earlier, even if it means trading away some precision?

We built five standard baseline classifiers, then an adaptive version of the strongest one, then a proactive detection layer on top of that — and evaluated all three configurations honestly, including where the adaptive approach *doesn't* win, because that's part of the finding.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team**

| Name | Roll Number |
|---|---|
| Venkata Ajay Odugu | AP24110011016 |
| Mohanasritha Eerla | AP24110011024 |
| Vijay Perla | AP24110011059 |
| Neelima Bojanapu | AP24110011111 |

---

## Table of contents

- [Datasets](#datasets)
- [Method](#method)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Datasets

| | Primary — Pima Indians Diabetes | Validation — UCI Chronic Kidney Disease |
|---|---|---|
| **Size** | 768 patients, 8 clinical features | 400 patients, 24 features after preprocessing |
| **Task** | Binary diabetes outcome | Binary CKD outcome |
| **Role** | Main benchmark for baseline, adaptive, and proactive experiments | Independent cross-dataset check — does the framework generalize beyond one disease? |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | [UCI ML Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) |

## Method

```
                 ┌────────────────────┐
                 │   Raw patient data │
                 └─────────┬──────────┘
                            │  split BEFORE any fitting
                 ┌─────────▼──────────┐
                 │ Train / test split │  (stratified, seed=42)
                 └─────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌─────────────┐  ┌───────────────┐  ┌──────────────┐
   │  Baselines  │  │ Adaptive MLP  │  │  Proactive   │
   │ LR·DT·RF·   │─▶│ retrains on   │─▶│  layer:      │
   │ SVM·MLP     │  │ sequential    │  │ rolling-window│
   └─────────────┘  │ batches       │  │ risk flagging │
                     └───────────────┘  └──────────────┘
                            │                   │
                            ▼                   ▼
                 ┌────────────────────────────────────┐
                 │  Evaluated on: Pima  +  CKD (x-val) │
                 └────────────────────────────────────┘
```

1. **Baselines** — Logistic Regression, Decision Tree, Random Forest, SVM, and a static MLP, each tuned with `GridSearchCV`.
2. **Adaptive MLP** — the strongest baseline architecture, retrained incrementally as new batches of patient data arrive, instead of being trained once and frozen.
3. **Adaptive + Proactive** — a rolling-window layer on top of the adaptive model that flags risk earlier, deliberately trading precision for recall.
4. **Cross-dataset validation** — the same framework re-run natively on CKD, plus a generalization test where a diabetes-trained model is evaluated on CKD without retraining.

All preprocessing follows the same rule throughout: **split first, fit second.** Imputation medians and scalers are fit on the training fold only and applied to the test fold — never the reverse. See [`pipeline.py`](pipeline.py) for the canonical implementation.

---

## Results

### Baseline models (Pima Diabetes)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| Decision Tree | 0.7692 | 0.6842 | 0.6111 | 0.6452 | 0.8056 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| SVM | 0.7208 | 0.6341 | 0.4815 | 0.5474 | 0.7970 |
| MLP | 0.7532 | 0.6818 | 0.5556 | 0.6122 | **0.8467** |

*Source: [`results/canonical_baseline_results.csv`](results/canonical_baseline_results.csv). Decision Tree leads on raw accuracy; MLP separates the classes best overall (highest AUC).*

### Adaptive & proactive framework — ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP (baseline) | 0.7532 | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7318 | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.6500 | 0.4853 | **0.8046** | 0.5979 |

*Source: [`results/week5/canonical_ablation_results.csv`](results/week5/canonical_ablation_results.csv), mean across all evaluation batches.*

The Adaptive MLP alone does **not** beat the static baseline on accuracy or F1 — reported as-is rather than hidden. The full Adaptive + Proactive system trades a substantial amount of precision for a large recall gain, catching far more true positive cases (0.80 vs 0.56 recall) at the cost of more false alarms — a defensible trade-off in a clinical screening context, where a missed case is typically more costly than an unnecessary follow-up.

A McNemar's test comparing baseline MLP vs Random Forest on accuracy was **not statistically significant** (p = 0.2632, [`results/statistical_significance.csv`](results/statistical_significance.csv)). This has not yet been extended to Static vs Adaptive vs Proactive — see [Limitations](#limitations).

### Cross-dataset validation (CKD)

Three separate experiments, kept separate deliberately since they answer different questions:

| Experiment | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| RF trained & tested on CKD | 0.9750 | 1.0000 | 0.9333 | 0.9655 | 0.9993 |
| Adaptive MLP trained on diabetes, tested on CKD (no retraining) | 0.9750 | 0.9667 | 0.9667 | 0.9667 | — |
| Adaptive MLP trained natively on CKD | **0.9875** | 1.0000 | 0.9800 | **0.9899** | 1.0000 |

**Robustness checks** (included so the near-perfect scores above aren't taken at face value):

- **3-seed validation** (seeds 42 / 7 / 123): mean accuracy **0.9792 ± 0.0072**
- **5-fold cross-validation**: mean accuracy **0.9875** (2 of 5 folds hit a perfect 1.0 — the honest headline is the mean, not the best fold)

CKD's near-perfect separability is a property of the dataset (it includes features close to clinical diagnostic markers, e.g. hemoglobin, specific gravity, packed cell volume) and is consistent with prior published results on this dataset (97–100% accuracy is typical across RF/SVM/KNN/DT) — not an artifact of leakage. No leakage was found: split precedes scaling/imputation, the target column is excluded from features, and all 400 rows are retained.

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                    Original datasets (diabetes.csv, chronic_kidney_disease.csv)
│   └── processed/              Cleaned, imputed, encoded CKD data (train/test)
├── notebooks/
│   ├── diabetes/                EDA, 5 baselines, adaptive MLP, proactive detection,
│   │                             ablation study, error analysis, statistical significance
│   └── ckd/                     Preprocessing, EDA, baseline RF, native adaptive MLP,
│                                 multi-seed validation, generalization test
├── md/                          Related work, contribution statement, error analysis, limitations
├── paper/                       Paper drafts (paper1–3), gap analysis, replication notes
├── results/                     All logged metrics, figures, and per-week outputs
├── pipeline.py                  Canonical preprocessing — leakage-safe split, seed=42
├── requirements.txt
└── LICENSE
```

---

## Getting started

```bash
git clone https://github.com/ap24110011024/Disease_detection.git
cd Disease_detection
pip install -r requirements.txt
jupyter notebook
```

Notebooks are run from their own folder (paths are relative to `notebooks/diabetes/` or `notebooks/ckd/`). Run `ckd/preprocessing.ipynb` before any CKD model notebook — it produces the cleaned dataset they depend on.

**To reproduce the canonical run exactly:**

```
Diabetes:  eda → logistic_regression → decision_tree → random_forest → svm_model
           → week4_mlp → adaptive_mlp → proactive_detection → ablation_study
           → statistical_significance → final_results

CKD:       ckd/preprocessing → ckd/random_forest → ckd/adaptive_mlp
           → ckd/generalization → ckd/multiseed_validation
```

All seeds are fixed at 42 unless a notebook is deliberately testing multiple seeds.

---

## Reproducibility

Every metric in this README comes from a single canonical run — notebooks executed in order, output CSVs in `results/` used as-is. **If a number in a paper draft or slide ever disagrees with the CSVs in `results/`, the CSV is correct.**

| Category | Notebooks | Seed status |
|---|---|---|
| Sklearn baselines | `logistic_regression`, `decision_tree`, `random_forest`, `svm_model`, `ckd/random_forest`, `ckd/generalization`, `statistical_significance` | `random_state=42` |
| Neural nets (Keras/TF) | `week4_mlp`, `adaptive_mlp`, `proactive_detection`, `ckd/adaptive_mlp` | `random.seed`, `np.random.seed`, `tf.random.set_seed` all fixed to 42 |
| Deliberately multi-seed | `ckd/multiseed_validation` | Loops seed ∈ {42, 7, 123} — a robustness check, not the headline number |
| Pure EDA / aggregation | `eda.ipynb` (both), `ckd/preprocessing`, `final_results`, `ablation_study`, `roc_all_models`, `error_analysis` | No training, no seed needed |
| **Open item** | `linear_regression_scratch.ipynb` | No seed found yet — add one before treating any of its output as canonical |

TensorFlow/Keras results are seeded but not guaranteed bit-for-bit reproducible across machines — expect small metric variation on retraining, which is why the multiseed and cross-validation tables exist above.

---

## Limitations

Full detail in [`md/limitations.md`](md/limitations.md); in brief:

- The Adaptive MLP alone underperforms the static baseline on this dataset — the framework's benefit is in recall, not raw accuracy.
- McNemar's significance test has only been run for MLP vs Random Forest (not significant, p = 0.263); it has not yet been extended to Static vs Adaptive vs Proactive, which is the project's actual central claim.
- Evaluation is batch/window-based, not true real-time streaming.
- Validated on two disease domains (diabetes, CKD) — broader generalization across more datasets is future work.

---

## Documentation

| File | Contents |
|---|---|
| [`md/related_work.md`](md/related_work.md) | Prior work this project builds on |
| [`md/contribution.md`](md/contribution.md) | Explicit list of contributions |
| [`md/error_analysis.md`](md/error_analysis.md) | Where and why the models get it wrong |
| [`md/limitations.md`](md/limitations.md) | Full limitations discussion |
| [`md/report_eda.md`](md/report_eda.md) | Exploratory data analysis writeup |
| [`paper/`](paper/) | Paper drafts and replication notes |

## Citation

If you build on this work, please cite it as:

```bibtex
@misc{disease_detection_2026,
  title  = {Adaptive AI Models for Proactive Disease Detection},
  author = {Odugu, Venkata Ajay and Eerla, Mohanasritha and Perla, Vijay and Bojanapu, Neelima},
  year   = {2026},
  note   = {Research internship supervised by Dr. Ch. Anil Carie, SRM University--AP},
  url    = {https://github.com/ap24110011024/Disease_detection}
}
```

## License

Released under the [MIT License](LICENSE).
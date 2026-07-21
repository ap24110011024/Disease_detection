# Adaptive AI for Proactive Disease Detection

An 8-week research internship project investigating whether a machine learning model can adapt to new patient data over time, and whether pushing it toward earlier risk-flagging — rather than static after-the-fact classification — is worth the trade-offs it introduces.

Evaluated on the **Pima Indians Diabetes Dataset** (primary) and independently validated on the **UCI Chronic Kidney Disease (CKD) Dataset**, to test generalization beyond a single disease.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team (Group 1):**
- Venkata Ajay Odugu (AP24110011016)
- Mohanasritha Eerla (AP24110011024) — canonical/primary repo
- Vijay Perla (AP24110011059)
- Neelima Bojanapu (AP24110011111)

**Target venue:** IEEE CBMS 2025 (first choice) / IEEE EMBC 2025 (second choice) — IEEE two-column format, 6 pages max, 10+ references.

---

## What this project does

Most disease-prediction models are trained once and never touched again — they don't learn from new patients as data arrives, and they classify rather than anticipate. This project asks:

1. **Can a model adapt** — retraining itself as new patient batches arrive — without sacrificing performance?
2. **Can a proactive layer** on top of that catch more true cases earlier, even at the cost of some precision?

Five standard baseline classifiers were built first, then an adaptive version of the strongest one, then a proactive detection layer on top. All results are reported honestly, including where the adaptive approach falls short — that's part of the finding, not a gap in it.

---

## Datasets

**Primary — Pima Indians Diabetes**
768 patients, 8 clinical features, binary diabetes outcome.
[Source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Validation — UCI Chronic Kidney Disease**
400 patients, 24 features after preprocessing, used to test whether the framework generalizes to a different disease and dataset without retuning.
[Source](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

---

## Canonical results

All numbers below come from a single locked run (`random_state=42`, GridSearchCV scored on `f1`) — no cherry-picking across runs. Source files: `results/canonical_baseline_results.csv`, `results/week5/canonical_ablation_results.csv`, `results/week6/canonical_ckd_multiseed_results.csv`.

### Baseline models (Pima Diabetes)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| Decision Tree (GridSearchCV, `max_depth` 3–10, scored on F1) | 0.7987 | 0.7447 | 0.6481 | 0.6931 | 0.7921 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| SVM (GridSearchCV, `C`∈{0.1,1,10}, `γ`∈{auto,scale,0.01}, scored on F1) | 0.7662 | 0.6863 | 0.6364 | 0.6604 | 0.7374 |
| MLP | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8467 |

Switching Decision Tree's grid search from `scoring="accuracy"` to `scoring="f1"` raised its F1 from 0.373 to 0.693 — the earlier low score reflected a metric mismatch (the tree was tuned for accuracy, then judged on F1), not a weaker model. All tuned models are now optimized on the same metric the paper reports.

MLP has the highest AUC of the group despite a middling F1 — it separates the classes well, but its default decision threshold isn't tuned for this task. That gap motivated the adaptive/proactive work below.

### Adaptive and proactive framework — ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP (baseline) | 0.7532 | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7318 | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.6500 | 0.4853 | 0.8046 | 0.5979 |

Values are means across all evaluation batches, not a single best-case run. **The Adaptive MLP alone does not outperform the static baseline** — reported plainly rather than hidden. The full Adaptive + Proactive system trades a substantial amount of precision for a large recall gain (0.80 vs 0.56), catching more true positive cases at the cost of more false alarms. In a clinical screening context, that trade-off is often worthwhile: a missed case is typically costlier than an unnecessary follow-up.

A McNemar's test comparing baseline MLP against Random Forest on accuracy was **not statistically significant** (p = 0.2632) — see `md/limitations.md` for the full discussion.

**Paper framing:** the contribution is a recall/sensitivity trade-off for earlier detection, not a raw accuracy or F1 improvement over static models. This table is the basis for the Discussion section, not an "adaptive wins" narrative.

### Cross-dataset validation (CKD)

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CKD | Random Forest | 0.9750 | 1.0000 | 0.9333 | 0.9655 |
| CKD | Adaptive MLP | 0.9875 | 1.0000 | 0.9800 | 0.9899 |

**3-seed validation** (seeds 42, 7, 123): mean accuracy 0.979 ± 0.007, mean F1 0.972 ± 0.010.

Verified no data leakage: clean train/test split, evaluation strictly on held-out data, target column excluded from features, full 400-row dataset retained after preprocessing.

Near-perfect CKD scores are consistent with prior published work on this dataset (97–100% accuracy reported using RF, SVM, KNN, and Decision Tree elsewhere). CKD includes features like hemoglobin, specific gravity, and packed cell volume that are themselves close to clinical diagnostic markers — high separability here is a property of the dataset, not a pipeline artifact.

> **Scope note:** the paper reports only Adaptive MLP + Random Forest on CKD, not the full multi-model CKD pipeline. This keeps the CKD section a focused generalization check rather than a second full benchmark — within a 6-page IEEE limit, a second 5-model comparison would crowd out the primary Pima analysis and ablation study without adding much: the point of the CKD run is to show the framework transfers, not to re-rank models on a second dataset.

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                     Original datasets (diabetes.csv, chronic_kidney_disease.csv)
│   └── processed/                Cleaned, imputed, split CKD data
├── notebooks/
│   ├── diabetes/                 EDA, 5 baselines, adaptive MLP, proactive detection, ablation study,
│   │                             ROC curves, error analysis, statistical significance
│   └── ckd/                      Preprocessing, EDA, random forest, adaptive MLP, multi-seed validation,
│                                  generalization test
├── md/                           related_work.md, contribution.md, error_analysis.md, limitations.md,
│                                  comparison_dt_rf.md, report_eda.md
├── paper/                        Paper drafts, gap analysis, replication notes
├── results/                      All logged metrics and figures, per-week outputs (week1–week6),
│                                  plus canonical_*.csv — the locked source of truth for paper numbers
├── pipeline.py                   Shared preprocessing (train/test split, scaling — leakage-safe)
├── explore.py
├── requirements.txt
└── LICENSE                       MIT
```

---

## Running this yourself

```bash
git clone https://github.com/ap24110011024/Disease_detection.git
cd Disease_detection
pip install -r requirements.txt
jupyter notebook
```

Notebooks assume they're run from their own folder (e.g. `notebooks/diabetes/` or `notebooks/ckd/`), since file paths are relative to that location. Run `preprocessing.ipynb` in `notebooks/ckd/` before any CKD model notebooks — it produces the cleaned dataset they depend on.

**Reproducibility note:** the MLP baseline is a small Keras/TensorFlow network. Seeds are set (`random`, `numpy`, `tensorflow`), but neural network training isn't perfectly bit-for-bit reproducible across runs even with seeding — expect small variation if retrained.

**On the canonical results:** every number in the paper traces back to `results/canonical_baseline_results.csv`, `results/week5/canonical_ablation_results.csv`, and `results/week6/canonical_ckd_multiseed_results.csv`. These came from one locked pass and should not be regenerated mid-writing — a new run with different batch sampling or dropout will produce different numbers and break consistency across the paper.

---

## Honest limitations

Full discussion in `md/limitations.md`:

- Adaptive MLP alone underperforms the static baseline on this dataset — the framework's benefit is in recall, not raw accuracy
- The McNemar significance test did not reach p < 0.05
- Evaluation is batch/window-based, not true real-time streaming
- Novelty is in the combination and clinical application of adaptive + proactive techniques on tabular data, not in a new underlying algorithm

Reported here plainly rather than left for a reviewer to find.

---
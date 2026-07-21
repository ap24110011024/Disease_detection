# Adaptive AI for Proactive Disease Detection

An 8-week research internship project exploring whether a machine learning model can adapt to new patient data over time, and whether it can be pushed toward flagging disease risk earlier rather than just classifying it after the fact.

Built and evaluated on the **Pima Indians Diabetes Dataset**, with a second, independent evaluation on the **UCI Chronic Kidney Disease (CKD) Dataset** to check whether the approach generalizes beyond a single disease or dataset.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team (Group 1):**
- Venkata Ajay Odugu (AP24110011016)
- Mohanasritha Eerla (AP24110011024) — canonical/primary repo
- Vijay Perla (AP24110011059)
- Neelima Bojanapu (AP24110011111)

**Target venue:** IEEE CBMS 2025 (first choice) / IEEE EMBC 2025 (second choice) — IEEE two-column format, 6 pages max, 10+ references.

---

## What this project actually does

Most disease-prediction models are trained once and never touched again — they don't learn from new patients as data comes in, and they're built to classify, not to catch risk early. This project asks two questions:

1. **Can a model adapt** — retraining itself as new batches of patient data arrive — without sacrificing performance?
2. **Can a "proactive" layer** on top of that catch more true cases earlier, even if it means trading away some precision?

We built five standard baseline classifiers first, then an adaptive version of the strongest one, then a proactive detection layer on top of that. Everything is evaluated honestly — including where the adaptive approach doesn't win, because that's part of the finding.

---

## Datasets

**Primary — Pima Indians Diabetes**
768 patients, 8 clinical features, binary diabetes outcome.
[Source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

**Validation — UCI Chronic Kidney Disease**
400 patients, 24 features after preprocessing, used purely to test whether the framework generalizes to a different disease and a different dataset without any retuning.
[Source](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

---

## Results

### Baseline models (Pima Diabetes)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| Decision Tree (GridSearchCV, max_depth 3–10) | 0.6948 | 0.6667 | 0.2593 | 0.3733 | 0.7881 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| SVM (GridSearchCV, C∈{0.1,1,10}, γ∈{auto,scale,0.01}) | 0.7662 | 0.6863 | 0.6364 | 0.6604 | 0.7374 |
| MLP | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8467 |

All splits use `train_test_split(..., stratify=y, random_state=42)`; Decision Tree and SVM are tuned via 5-fold `GridSearchCV` (currently scored on accuracy — switching to `scoring="f1"` is a pending refinement, see **Open items** below).

MLP has the highest AUC of the group despite a middling F1 — it separates the classes well but its default decision threshold isn't tuned for this. That gap is part of what motivated the adaptive/proactive work below rather than just picking the best baseline and stopping.

### Adaptive and proactive framework — ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP (baseline) | 0.7532 | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7318 | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.6500 | 0.4853 | 0.8046 | 0.5979 |

These are mean values averaged across all evaluation batches, not a single best-case run. Reported honestly: **the Adaptive MLP alone does not outperform the static baseline.** The full Adaptive + Proactive system trades a substantial amount of precision for a large recall gain — it catches far more true positive cases (0.80 vs 0.56 recall) at the cost of more false alarms. Whether that trade-off is worth it depends on the application: in a clinical screening context, catching more real cases early is often worth some extra false alarms, since a missed case is more costly than an unnecessary follow-up.

A McNemar's test comparing the baseline MLP against Random Forest on accuracy came back **not statistically significant** (p = 0.2632) — noted here rather than left out, and discussed further in `md/limitations.md`.

**This is the framing the paper should use:** the contribution is a recall/sensitivity trade-off for earlier detection, not a raw accuracy or F1 improvement over static models. Any abstract/discussion sentence that claims the adaptive model "outperforms" the static baseline needs to be rewritten before submission — it contradicts this table and `md/limitations.md`.

### Cross-dataset validation (CKD)

Random Forest and the adaptive framework both achieve **near-perfect scores** on the CKD dataset:

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CKD | Random Forest | 0.9750 | 1.0000 | 0.9333 | 0.9655 |
| CKD | Adaptive MLP | 0.9875 | 1.0000 | 0.9800 | 0.9899 |

- **3-seed validation** (seeds 42, 7, 123): mean accuracy 0.979 ± 0.007, mean F1 0.972 ± 0.010
- Verified no data leakage — clean train/test split, correct evaluation on held-out data, target column properly excluded, dataset size intact after preprocessing (all 400 rows retained)

This result is consistent with prior published work on this same dataset, which similarly reports 97–100% accuracy using Random Forest, SVM, KNN, and Decision Tree. The CKD dataset is small (400 patients) and includes features like hemoglobin, specific gravity, and packed cell volume that are themselves close to clinical diagnostic markers for the condition — so near-perfect separability here is a property of the dataset, not a bug in the pipeline.

> Note: the paper only reports Adaptive MLP + Random Forest on CKD (not the full 6-model CKD pipeline) — that's the agreed scope for this paper.

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                    Original datasets (diabetes.csv, chronic_kidney_disease.csv)
│   └── processed/               Cleaned, imputed, split CKD data
├── notebooks/
│   ├── diabetes/                EDA, 5 baselines, adaptive MLP, proactive detection, ablation study,
│   │                            ROC curves, error analysis, statistical significance
│   └── ckd/                     Preprocessing, EDA, random forest, adaptive MLP, multi-seed validation,
│                                 generalization test
├── md/                          related_work.md, contribution.md, error_analysis.md, limitations.md,
│                                 comparison_dt_rf.md, report_eda.md
├── paper/                       Paper drafts, gap analysis, replication notes
├── results/                     All logged metrics, figures, and per-week outputs (week1–week6)
├── pipeline.py                  Shared preprocessing (train/test split, scaling — leakage-safe)
├── explore.py
├── requirements.txt
└── LICENSE                      MIT
```

**Pending, per the team submission plan** — to be pulled in from `AP24110011016/disease-ai-intern` and committed here:
- `ckd_preprocessing.py` — ARFF-format loader for the UCI CKD dataset
- `ckd_adaptive_mlp.py` — adaptive MLP implementation applied to CKD
- `architecture.drawio` — system architecture diagram for the paper figure

---

## Running this yourself

```bash
git clone https://github.com/ap24110011024/Disease_detection.git
cd Disease_detection
pip install -r requirements.txt
jupyter notebook
```

Notebooks assume they're run from their own folder (e.g. `notebooks/diabetes/` or `notebooks/ckd/`), since file paths are relative to that location. Run `preprocessing.ipynb` in `notebooks/ckd/` before any of the CKD model notebooks, since it produces the cleaned dataset they depend on.

**Note on reproducibility:** the MLP baseline is a small Keras/TensorFlow network. Seeds are set (`random`, `numpy`, `tensorflow`), but neural network training isn't perfectly bit-for-bit reproducible across runs even with seeding — expect metrics to vary by a small amount if you retrain it yourself.

---

## Honest limitations

Documented in full in `md/limitations.md`, briefly:

- Adaptive MLP alone underperforms the static baseline on this dataset — the benefit of this framework is in recall, not raw accuracy
- The McNemar significance test did not reach p < 0.05
- Evaluation is batch/window-based, not true real-time streaming
- Novelty is in the combination and application of adaptive + proactive techniques on clinical tabular data, not in a new underlying algorithm

We'd rather report these plainly than leave them for a reviewer to find.

---

## Open items before the canonical numbers are locked

- [ ] Pull the 3 files listed above from `AP24110011016`
- [ ] Re-score Decision Tree / SVM `GridSearchCV` on `scoring="f1"` instead of `scoring="accuracy"` for consistency with the paper's primary metric
- [ ] Align CKD multi-seed set to whichever seed list the team agrees on (currently `[42, 7, 123]`)
- [ ] Run the single canonical, locked-seed pass and commit `results/canonical_pima.csv`, `results/canonical_ablation.csv`, `results/canonical_ckd.csv` — every number in the paper should trace back to one of these three files
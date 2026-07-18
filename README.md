# Adaptive AI Models for Proactive Disease Detection

An 8-week research internship project exploring whether a machine learning model can adapt to new patient data over time, and whether it can be pushed toward flagging disease risk earlier rather than just classifying it after the fact.

Built and evaluated on the **Pima Indians Diabetes Dataset**, with a second, independent evaluation on the **UCI Chronic Kidney Disease (CKD) Dataset** to check whether the approach generalizes beyond a single disease or dataset.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team:**
- Venkata Ajay Odugu (AP24110011016)
- Mohanasritha Eerla (AP24110011024)
- Vijay Perla (AP24110011059)
- Neelima Bojanapu (AP24110011111)

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
| Decision Tree | 0.6948 | 0.6667 | 0.2593 | 0.3733 | 0.7881 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| SVM | 0.7662 | 0.6863 | 0.6364 | 0.6604 | 0.7374 |
| MLP | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8467 |

MLP has the highest AUC of the group despite a middling F1 — it separates the classes well but its default decision threshold isn't tuned for this. That gap is part of what motivated the adaptive/proactive work below rather than just picking the best baseline and stopping.

### Adaptive and proactive framework — ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP (baseline) | 0.7532 | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7318 | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.6500 | 0.4853 | 0.8046 | 0.5979 |

These are mean values averaged across all evaluation batches, not a single best-case run. Reported honestly: the Adaptive MLP alone does not outperform the static baseline. The full Adaptive + Proactive system trades a substantial amount of precision for a large recall gain — it catches far more true positive cases (0.80 vs 0.56 recall) at the cost of more false alarms. Whether that trade-off is worth it depends on the application: in a clinical screening context, catching more real cases early is often worth some extra false alarms, since a missed case is more costly than an unnecessary follow-up.

A McNemar's test comparing the baseline MLP against Random Forest on accuracy came back **not statistically significant** (p = 0.2632) — noted here rather than left out, and discussed further in `md/limitations.md`.

### Cross-dataset validation (CKD)

Random Forest and the adaptive framework both achieve **near-perfect scores** on the CKD dataset. That number looks alarming out of context, so here's what backs it up:

- **5-fold cross-validation**: mean accuracy 0.99 (range 0.975–1.0)
- **3-seed validation** (different train/test splits): mean accuracy 0.979 ± 0.007
- Verified no data leakage — clean train/test split, correct evaluation on held-out data, target column properly excluded, dataset size intact after preprocessing (all 400 rows retained)

This result is consistent with prior published work on this same dataset, which similarly reports 97–100% accuracy using Random Forest, SVM, KNN, and Decision Tree. The CKD dataset is small (400 patients) and includes features like hemoglobin, specific gravity, and packed cell volume that are themselves close to clinical diagnostic markers for the condition — so near-perfect separability here is a property of the dataset, not a bug in the pipeline.

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                  Original datasets
│   └── processed/            Cleaned, imputed, encoded data
├── notebooks/
│   ├── diabetes/             EDA, baselines, adaptive MLP, proactive detection, ablation study
│   └── ckd/                  Preprocessing, EDA, baseline validation, multi-seed check, generalization test
├── md/                       Related work, contribution statement, error analysis, limitations
├── paper/                    Paper drafts, gap analysis, replication notes
├── results/                  All logged metrics, figures, and per-week outputs
├── pipeline.py                Shared preprocessing (train/test split, scaling — leakage-safe)
├── explore.py
├── requirements.txt
└── LICENSE
```

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

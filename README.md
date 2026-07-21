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
400 patients, 24 features after preprocessing, used to test whether the framework generalizes to a different disease and a different dataset.
[Source](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)

---

## Reproducibility: how every number in this README was produced

Every metric below comes from a **single canonical run**: `pipeline.py` with `RANDOM_SEED = 42`, notebooks executed in order, output CSVs in `results/` copied here without hand-editing. If a number in a paper draft or slide ever disagrees with the CSVs in `results/`, **the CSV is correct — regenerate the write-up, not the other way around.**

**Seed audit** — confirmed across every notebook in this repo:

| Category | Notebooks | Seed status |
|---|---|---|
| Sklearn baselines | `logistic_regression`, `decision_tree`, `random_forest`, `svm_model`, `ckd/random_forest`, `ckd/generalization`, `statistical_significance` | `random_state=42` ✅ |
| Neural nets (Keras/TF) | `week4_mlp`, `adaptive_mlp`, `proactive_detection`, `ckd/adaptive_mlp` | Full triple seed — `random.seed(42)`, `np.random.seed(42)`, `tf.random.set_seed(42)` ✅ |
| Deliberately multi-seed | `ckd/multiseed_validation` | Loops `seed ∈ {42, 7, 123}` on purpose — a robustness check, not the headline number |
| No seed needed | `eda.ipynb` (both), `ckd/preprocessing`, `final_results`, `ablation_study`, `roc_all_models`, `error_analysis` | Pure EDA/aggregation/plotting — no training happens |
| Needs attention | `linear_regression_scratch.ipynb` | No seed found — add one if it performs any random split or initialization |
| Removed | `hello_world.ipynb` | Scratch/test notebook, deleted — not part of the analysis |

### Changelog — what changed in the latest canonical run and why

Two real methodology fixes were made after the first canonical run, which moved some numbers below. Documented here so nobody has to reverse-engineer a diff to find out why:

- **Decision Tree & SVM: `GridSearchCV` scoring changed from `accuracy` to `f1`.** The dataset is imbalanced (~65/35); optimizing for raw accuracy let Decision Tree collapse toward the majority class (previously **0.26 recall** — missing 3 of every 4 real positive cases). Switching to F1-scoring fixed this: Decision Tree recall improved to 0.61.
- **SVM: added `stratify=y` to `train_test_split`**, which was missing before. This is the methodologically correct choice (guarantees matching class balance in train/test) and changes which exact rows fall into the test set, which is why SVM's accuracy moved from an earlier, unstratified run.
- **New experiment added: `ckd/adaptive_mlp.ipynb`** — trains the Adaptive MLP natively on CKD data. This is a *different* experiment from `ckd/generalization.ipynb` (which trains only on diabetes and tests on CKD without ever seeing CKD during training). Both are reported separately below — neither replaces the other.

---

## Results

### Baseline models (Pima Diabetes)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| Decision Tree | 0.7692 | 0.6842 | 0.6111 | 0.6452 | 0.8056 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| SVM | 0.7208 | 0.6341 | 0.4815 | 0.5474 | 0.7970 |
| MLP | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8467 |

*(Source: `results/canonical_baseline_results.csv`. Logistic Regression, Random Forest, and MLP are unchanged from the previous run — same seed, same code, bit-for-bit reproducible. Decision Tree and SVM changed for the documented reasons above, not due to randomness.)*

Decision Tree now edges out the group on raw accuracy, but MLP still has the highest AUC — it separates the classes best even where its default threshold isn't optimal. SVM's drop after adding `stratify=y` is reported as-is rather than reverted, since the stratified split is the more correct evaluation.

### Adaptive and proactive framework — ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP (baseline) | 0.7532 | 0.6818 | 0.5556 | 0.6122 |
| Adaptive MLP | 0.7318 | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.6500 | 0.4853 | 0.8046 | 0.5979 |

*(Source: `results/week5/canonical_ablation_results.csv`, means across all evaluation batches — see `adaptive_summary.csv` / `proactive_summary.csv` for the per-batch breakdown.)*

Reported honestly: Adaptive MLP alone does **not** outperform the static baseline on accuracy or F1. The full Adaptive + Proactive system trades a substantial amount of precision for a large recall gain — it catches far more true positive cases (0.80 vs 0.56 recall) at the cost of more false alarms. Whether that trade-off is worth it depends on the application: in a clinical screening context, catching more real cases early is often worth some extra false alarms, since a missed case is more costly than an unnecessary follow-up.

A McNemar's test comparing the baseline MLP against Random Forest on accuracy came back **not statistically significant** (p = 0.2632, `results/statistical_significance.csv`). **Not yet extended** to Static vs Adaptive vs Proactive — that's the project's actual central claim and is the next thing to test, not just the RF-vs-MLP baseline comparison.

### Cross-dataset validation (CKD)

Two separate experiments — kept separate deliberately, since they answer different questions.

**Experiment 1 — Random Forest trained and tested on CKD** (`results/week6/ckd_rf_result.csv`):

| Metric | Value |
|---|---|
| Accuracy | 0.9750 |
| Precision | 1.0000 |
| Recall | 0.9333 |
| F1 | 0.9655 |
| AUC | 0.9993 |

**Experiment 2 — Adaptive MLP trained on diabetes, tested on CKD without retraining** (generalization test, `results/week6/generalization_results.csv`):

| Metric | Value |
|---|---|
| Accuracy | 0.9750 |
| Precision | 0.9667 |
| Recall | 0.9667 |
| F1 | 0.9667 |

**Experiment 3 — Adaptive MLP trained natively on CKD** (new, `results/week6/ckd_adaptive_mlp_result.csv`):

| Metric | Value |
|---|---|
| Accuracy | 0.9875 |
| Precision | 1.0000 |
| Recall | 0.9800 |
| F1 | 0.9899 |
| AUC | 1.0000 |

**Robustness checks, not headline numbers — included so the near-perfect scores above aren't taken at face value:**

3-seed validation (`results/week6/canonical_ckd_multiseed_results.csv`):

| Seed | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 42 | 0.9750 | 1.0000 | 0.9333 | 0.9655 |
| 7 | 0.9875 | 0.9677 | 1.0000 | 0.9836 |
| 123 | 0.9750 | 1.0000 | 0.9333 | 0.9655 |
| **Mean ± Std** | **0.9792 ± 0.0072** | 0.9892 ± 0.0186 | 0.9556 ± 0.0385 | 0.9715 ± 0.0104 |

5-fold cross-validation (`results/week6/cross_validation_results.csv`):

| Fold | 1 | 2 | 3 | 4 | 5 | **Mean** |
|---|---|---|---|---|---|---|
| Accuracy | 1.000 | 1.000 | 0.975 | 0.975 | 0.9875 | **0.9875** |

Two of five CV folds do hit a perfect 1.0 — worth knowing, but the honest headline is the mean (~0.98), not a cherry-picked fold.

**Why the score is this high in the first place:** verified no data leakage — split before scaling/imputation, target column excluded, all 400 rows retained. Also consistent with prior published work on this dataset (97–100% accuracy is typical across RF, SVM, KNN, DT). CKD includes features like hemoglobin, specific gravity, and packed cell volume that are close to clinical diagnostic markers — near-perfect separability here is a property of the dataset, not a bug in the pipeline.

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                  Original datasets
│   └── processed/            Cleaned, imputed, encoded data
├── notebooks/
│   ├── diabetes/             EDA, baselines, adaptive MLP, proactive detection, ablation study
│   └── ckd/                  Preprocessing, EDA, baseline validation, native adaptive MLP,
│                              multi-seed check, generalization test
├── md/                        Related work, contribution statement, error analysis, limitations
├── paper/                     Paper drafts, gap analysis, replication notes
├── results/                   All logged metrics, figures, and per-week outputs
├── pipeline.py                 Shared preprocessing (train/test split, scaling — leakage-safe, canonical seed=42)
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

**To reproduce the canonical run exactly:** `eda → baselines (logistic_regression, decision_tree, random_forest, svm_model) → week4_mlp → adaptive_mlp → proactive_detection → ablation_study → statistical_significance → final_results`, then separately `ckd/preprocessing → ckd/random_forest → ckd/adaptive_mlp → ckd/generalization → ckd/multiseed_validation`. All seeds are fixed at 42.

**Note on reproducibility:** the Keras/TensorFlow models are seeded (`random`, `numpy`, `tensorflow`), but neural network training isn't perfectly bit-for-bit reproducible across runs even with seeding — expect small metric variation on retraining, which is why the multiseed table exists.

**If a number ever changes after a re-run:** check `git diff` on the notebook source before assuming it's noise — in this project's history, every observed change so far has traced back to an intentional code fix (scoring metric, stratification), not randomness. Document the reason in this changelog section rather than silently overwriting the old number.

---

## Honest limitations

Documented in full in `md/limitations.md`, briefly:

- Adaptive MLP alone underperforms the static baseline on this dataset — the benefit of this framework is in recall, not raw accuracy
- The McNemar significance test has only been run for MLP vs Random Forest (not significant, p=0.2632); it has **not yet** been extended to Static vs Adaptive vs Proactive, which is the project's actual central claim
- Evaluation is batch/window-based, not true real-time streaming
- The `Glucose_Insulin_Ratio` feature in early EDA was computed before zero-value cleanup in some notebooks — flagged for consistency review, does not affect final reported model results
- Novelty is in the combination and application of adaptive + proactive techniques on clinical tabular data, not in a new underlying algorithm

We'd rather report these plainly than leave them for a reviewer to find.
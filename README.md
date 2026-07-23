# Adaptive AI Models for Proactive Disease Detection

**Can a classifier adapt as new patient data arrives — and can it be pushed to flag risk *earlier*, not just classify it after the fact?**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Most disease-prediction models are trained once and never touched again — they don't learn from new patients as data comes in, and they're built to classify, not to catch risk early. This project asks two concrete questions:

1. **Can a model adapt** — retraining itself as new batches of patient data arrive — without sacrificing performance?
2. **Can a proactive layer** on top of that catch more true cases earlier, even if it means trading away some precision?

We built five standard baseline classifiers on the Pima Diabetes dataset, an adaptive version of the strongest one, a proactive detection layer on top of that, and cross-checked the framework on a second dataset (CKD).

Every result below was verified by deleting all generated files (`results_log.csv`, both canonical result tables, every week-by-week output, and the CKD processed split) and re-running all 21 notebooks from scratch, in dependency order. See [Known limitations](#known-limitations) for what's still genuinely open.

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
- [Known limitations](#known-limitations)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Datasets

| | Primary — Pima Indians Diabetes | Secondary — UCI Chronic Kidney Disease |
|---|---|---|
| **Size** | 768 patients, 8 clinical features | 400 patients, 24 features after preprocessing |
| **Task** | Binary diabetes outcome | Binary CKD outcome |
| **Role** | Main benchmark for baseline, adaptive, and proactive experiments | Independent-dataset check |
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
                            │  median imputation of biologically
                            │  impossible zero values, fit on
                            │  the training fold only
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌─────────────┐  ┌───────────────┐  ┌──────────────┐
   │  Baselines  │  │ Adaptive MLP  │  │  Proactive   │
   │ LR·DT·RF·   │─▶│ retrains on   │─▶│  layer:      │
   │ SVM·MLP     │  │ sequential    │  │ rolling-window│
   └─────────────┘  │ batches       │  │ risk flagging │
                     └───────────────┘  └──────────────┘
```

1. **Baselines** — Logistic Regression, Decision Tree, Random Forest, SVM, and a static MLP. All five use `pipeline.py`'s shared preprocessing: stratified 80/20 split first, then median imputation of Pima's biologically-impossible zero values (Glucose, BloodPressure, SkinThickness, Insulin, BMI), fit on the training fold only.
2. **Adaptive MLP** — retrained incrementally as new batches of patient data arrive, with the same imputation approach applied fresh at each batch boundary. It uses 20 epochs per batch rather than the 50 epochs used for the static baseline; across all 10 sequential batches this yields 200 total epoch-passes, comparable to the static model's 50 epochs over the full training set, while better respecting the computational constraint of online retraining scenarios.
3. **Adaptive + Proactive** — a rolling-window layer on top of the adaptive model, using the *same network architecture and epoch count as the Adaptive MLP*, so this configuration differs from it only in the threshold mechanism being evaluated. Its decision threshold is selected using a held-out validation batch — distinct from both the training data and the batch being evaluated — never using the evaluation labels themselves.
4. **Second-dataset check** — the same framework re-run natively on CKD, using the same split-first-impute-second discipline.

---

## Results

### Baseline models (Pima Diabetes)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7078 | 0.6047 | 0.4815 | 0.5361 | 0.8070 |
| Decision Tree | 0.7857 | 0.6981 | 0.6852 | 0.6916 | 0.7887 |
| Random Forest | 0.7662 | 0.6957 | 0.5926 | 0.6400 | 0.8179 |
| SVM | 0.7078 | 0.6047 | 0.4815 | 0.5361 | 0.8169 |
| Static MLP | 0.7273 | 0.6304 | 0.5370 | 0.5800 | 0.8400 |

*Source: `results/canonical_baseline_results.csv`, derived programmatically from `results/results_log.csv` — never hand-typed. `random_state=42` throughout.*

**Verified:** SVM and Logistic Regression report identical Accuracy, Precision, Recall and F1 on this split. Checking the confusion matrices directly confirmed both models produce the same one (TN = 83, FP = 17, FN = 28, TP = 26) — a genuine tie on this test set, not a copy-paste error. The SVM implementation is confirmed correct (`SVC(kernel="rbf", probability=True, random_state=42)`). The two models do differ on AUC because AUC is computed from predicted probabilities/decision scores rather than the final binary predictions, so the tie in class labels doesn't extend to ranking quality. No implementation error was found; the reported results stand as-is.

A pairwise McNemar's test between Random Forest and an MLP classifier gives **p = 0.263** — the difference between these two models is not statistically significant on this split (`results/week5/statistical_significance.csv`). Note: this compares freshly-trained sklearn `RandomForestClassifier`/`MLPClassifier` instances, not the exact tuned models reported in the table above, and the p-value can vary somewhat between runs.

**McNemar's test — Static MLP vs. Adaptive + Proactive (the comparison the paper's claim actually rests on):** run directly on the two models' predictions on the same held-out test set. McNemar statistic = 6.0, **p = 0.7905** (exact test), from the contingency table `[[0, 6], [8, 0]]` — 6 cases where Static was wrong and Proactive was right, 8 cases the other way around. A p-value this high indicates the two models' disagreements aren't lopsided enough to call the difference statistically real — so on this split, there is **no statistically detectable difference** between the two models in either direction. The Recall gap in the table above (0.5370 → 0.6855) is real on this specific split, but the significance test shows it is not distinguishable from chance once the disagreements are weighed properly. Multi-seed evaluation (the same treatment already given to CKD) is the natural next step to determine whether the direction holds up across splits.

### Ablation study

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Static MLP | 0.7273 | 0.6304 | 0.5370 | 0.5800 |
| Adaptive MLP | 0.7375 | 0.6621 | 0.5168 | 0.5551 |
| Adaptive + Proactive | 0.7316 | 0.6223 | **0.6855** | **0.6150** |

*Source: `results/canonical_ablation_results.csv`.*

With architecture and epoch count held constant across all three configurations, the proactive layer achieves the **highest F1 of the three** (0.6150), a meaningfully higher Recall than the static baseline (0.6855 vs. 0.5370) at only a modest cost to Precision, and essentially no cost to Accuracy. Its threshold is selected without the evaluation batch's labels ever being used.

**Important caveat — read this table together with the baseline table above.** "Highest F1 of the three" is a comparison within the MLP family only, not a claim that the full system beats every baseline outright. Decision Tree (F1 = 0.6916) and Random Forest (F1 = 0.6400) both post a higher F1 than Adaptive + Proactive (F1 = 0.6150), and Decision Tree's Recall (0.6852) is effectively tied with the proactive system's (0.6855). On raw F1 and recall alone, a plain, untouched Decision Tree matches this headline result without any adaptive machinery. The defensible claim is narrower: Adaptive + Proactive achieves the **highest Recall of any method tested** (0.6855) — including Decision Tree and Random Forest — while remaining competitive on F1, and it does so through a mechanism that keeps learning as new patient batches arrive and selects its threshold without ever seeing evaluation labels, properties a fixed Decision Tree does not have by construction. Whether that architectural advantage is worth the added complexity relative to periodically retraining a Decision Tree is an open comparison, not yet answered here. Note also that the Recall improvement itself, while the right *direction* for a screening tool, shows no statistically detectable difference on this split (McNemar's p = 0.7905, see above) — treat it as a promising, design-consistent result rather than a confirmed one until multi-seed evaluation is run.

### CKD experiment

| Experiment | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Adaptive MLP trained natively on CKD | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest trained & tested on CKD | 0.9750 | 0.9615 | 1.0000 | 0.9804 | 0.9993 |
| 3-seed validation (mean, seeds 42/7/123) | 0.9917 | 0.9872 | 1.0000 | 0.9935 | — |

*Sources: `results/week6/ckd_adaptive_mlp_result.csv`, `results/week6/ckd_rf_result.csv`, `results/canonical_ckd_multiseed_results.csv`. All three use imputation fit on the training fold only (per seed, for the multi-seed check) — not a pre-imputed file.*

*The Adaptive MLP number above reflects a fix: `ckd/adaptive_mlp.ipynb` originally picked its "best" model across training batches by scoring each candidate against `X_test`/`y_test` — the batch loop was peeking at the test set before the final evaluation cell ever ran. It now carves a stratified 80/20 validation split out of the training data alone (`X_train_fit`/`X_val`, `random_state=42`) and selects "best" using validation F1; `X_test`/`y_test` are touched exactly once, in the final evaluation cell, after `best_model` is already frozen. The result is still a perfect 1.0 across the board — a number worth distrusting on sight for a 400-row dataset, so we checked it directly rather than asserting it. See [`md/ckd_leakage_check.md`](md/ckd_leakage_check.md) and `results/week6/ckd_feature_auc.csv`: zero duplicate rows in the raw data, zero overlap between the processed train/test split, and no single feature reaches 1.0 alone (hemoglobin, the strongest, tops out at AUC 0.968). The perfect score comes from the combination of several correlated clinical features, consistent with this UCI dataset's well-documented near-linear separability, not a new leak.*

---

## Repository structure

```
Disease_detection/
├── data/
│   ├── raw/                    Original datasets (diabetes.csv, chronic_kidney_disease.csv)
│   └── processed/              Leakage-safe CKD train/test split
├── notebooks/
│   ├── diabetes/                EDA, 5 baselines, adaptive MLP, proactive detection,
│   │                             ablation study, error analysis, statistical significance
│   └── ckd/                     Preprocessing, EDA, baseline RF, native adaptive MLP,
│                                 multi-seed validation, generalization check
├── md/                          Related work, contribution statement, error analysis, limitations
├── paper/                       Paper drafts, gap analysis, replication notes
├── results/                     All logged metrics and figures
├── pipeline.py                  Shared leakage-safe preprocessing, used by every baseline notebook
├── explore.py                   Quick standalone script for shape/dtypes/describe() on the raw diabetes data
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

Documented run order (diabetes): `eda → logistic_regression → decision_tree → random_forest → svm_model → week4_mlp → adaptive_mlp → proactive_detection → final_results → ablation_study → final_results → statistical_significance`.

`final_results.ipynb` appears twice by design: its first run produces `canonical_baseline_results.csv`, which `ablation_study.ipynb` depends on; its second run picks up `ablation_study.ipynb`'s output to produce `canonical_ablation_results.csv`.

Run notebooks from their own folder (e.g. launch Jupyter inside `notebooks/diabetes/`), which is the standard way Jupyter is normally used — this is the convention nearly all notebooks in this repo currently assume.

---

## Reproducibility

Every number in this README was confirmed by deleting all generated output and re-running the full 21-notebook pipeline from scratch. All seeds are fixed at 42 unless a notebook is deliberately testing multiple seeds.

| Category | Notebooks | Seed status |
|---|---|---|
| Sklearn baselines | `logistic_regression`, `decision_tree`, `random_forest`, `svm_model`, `ckd/random_forest`, `ckd/adaptive_mlp`, `statistical_significance` | `random_state=42` |
| Neural nets (Keras/TF) | `week4_mlp`, `adaptive_mlp`, `proactive_detection` | `random.seed`, `np.random.seed`, `tf.random.set_seed` fixed to 42 |
| Deliberately multi-seed | `ckd/multiseed_validation` | Seeds 42, 7, 123 |
| Pure EDA / aggregation | `eda.ipynb` (both), `ckd/preprocessing`, `roc_all_models`, `error_analysis` | No training, no seed needed |
| **Open item** | `linear_regression_scratch.ipynb` | No seed set — a from-scratch educational implementation, not a reported result |

TensorFlow/Keras results are seeded but not guaranteed bit-for-bit reproducible across machines — expect small metric variation on retraining, and note the McNemar's p-value above can shift somewhat between runs for this reason.

---

## Known limitations

Full discussion in [`md/limitations.md`](md/limitations.md). Stated plainly here as well:

1. **`ckd/generalization.ipynb` does not test cross-dataset transfer.** It trains and evaluates a model on CKD only, despite its name suggesting a diabetes-trained model is evaluated on CKD without retraining. CKD's 24 features and diabetes's 8 features don't share a feature space, so a genuine transfer test needs a deliberate design decision — a shared feature subset, or a different framing of the claim — rather than a direct code fix.
2. **Most notebooks assume they're launched from their own folder.** One notebook (`proactive_detection.ipynb`) resolves paths independently of working directory; the rest use relative paths that work correctly under the standard convention above but would break if run from a different working directory (e.g. the repository root).
3. **Static and Adaptive MLP differ in epoch count** (50 vs. 20) — a deliberate design choice now justified in [Method](#method) above (200 total epoch-passes across 10 batches vs. 50 over the full set), rather than left unexplained.
4. **The Adaptive MLP alone underperforms the static baseline** on F1 — adaptation alone does not clearly improve on a well-tuned static model on this dataset. The proactive layer, compared fairly (see Results above), improves Recall over both, though see #9 for why that improvement isn't yet statistically confirmed.
5. **Evaluation is batch/window-based, not true real-time streaming**, and batches are not shuffled before slicing — per-batch positive rate varies roughly 18-45% across the diabetes batches, a source of variance not currently isolated from the adaptation effect itself.
6. **Validated on two disease domains.** Broader generalization to other diseases and populations is future work.
7. **Independent verification is still pending.** The reproducibility checks described above were performed by the same process that also fixed the underlying bugs. An independent re-run — by a labmate, a mentor, or in a separate environment — is recommended before any of these numbers are cited in a publication.
8. **SVM and Logistic Regression report identical Accuracy, Precision, Recall and F1** (only AUC differs) — checked and confirmed genuine: both models produce the same confusion matrix (TN = 83, FP = 17, FN = 28, TP = 26) on this split, and the SVM implementation is correct. The AUC difference comes from probability scores rather than binary predictions. Not an error — listed here only because a reader comparing the table would reasonably wonder.
9. **The Recall improvement is not statistically distinguishable from chance on this split.** McNemar's test on Static MLP vs. Adaptive + Proactive predictions gives statistic = 6.0, p = 0.7905 — the models' disagreements aren't lopsided enough for the test to call the difference real. The direction of the Recall gap (0.5370 → 0.6855) is consistent with the framework's design goal, but multi-seed evaluation — the same treatment already given to CKD — is needed before this can be reported as a confirmed result.

---

## Documentation

| File | Contents |
|---|---|
| [`md/related_work.md`](md/related_work.md) | Prior work this project builds on |
| [`md/contribution.md`](md/contribution.md) | Explicit list of contributions |
| [`md/error_analysis.md`](md/error_analysis.md) | Where and why the models get it wrong |
| [`md/limitations.md`](md/limitations.md) | Full limitations discussion |
| [`md/report_eda.md`](md/report_eda.md) | Exploratory data analysis writeup |
| [`md/comparison_dt_rf.md`](md/comparison_dt_rf.md) | Head-to-head Decision Tree vs. Random Forest comparison |
| [`md/ckd_leakage_check.md`](md/ckd_leakage_check.md) | Duplicate-row and feature-separability check behind the CKD 1.0 scores |
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
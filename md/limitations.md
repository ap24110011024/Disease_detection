# Limitations

This project demonstrates the feasibility of an adaptive machine learning framework for proactive disease prediction. However, several limitations remain.

## 1. Dataset Size

The experiments were conducted using publicly available benchmark datasets (Pima Indians Diabetes and Chronic Kidney Disease), both of which are relatively small compared to real-world clinical datasets. Larger and more diverse datasets would improve the robustness of the evaluation.

---

## 2. Limited Disease Coverage

The current framework has been evaluated on only two disease prediction datasets. Although the methodology was successfully applied to both, further validation on additional diseases is necessary before broader clinical adoption.

---

## 3. Static Clinical Features

The models rely primarily on structured tabular clinical attributes. Temporal patient histories, laboratory trends, imaging data, genomic information, and electronic health records were not incorporated.

---

## 4. Threshold Selection

For the publication-ready implementation, a fixed decision threshold was used during evaluation to avoid tuning on the evaluation set. Future work may determine the optimal threshold using an independent validation set.

---

## 5. Limited Hyperparameter Search

Hyperparameter tuning was intentionally limited to maintain reproducibility and computational efficiency. More extensive optimization techniques such as Bayesian Optimization or Optuna may further improve performance.

---

## 6. Generalization Scope

The proposed methodology was independently evaluated on both Diabetes and CKD datasets using the same experimental protocol. This demonstrates generalization of the adaptive framework across disease prediction tasks. It does not represent cross-dataset transfer learning, where a model trained on one disease is directly evaluated on another without retraining.

---

## 7. Clinical Validation

The proposed framework has been evaluated using retrospective benchmark datasets only. Prospective clinical validation involving real patient populations remains future work.

---

# Future Work

Future research will focus on:

- Evaluation on larger multi-center clinical datasets.
- Incorporating longitudinal patient records.
- Integration with Explainable AI (SHAP, LIME).
- Validation using external hospital datasets.
- Federated learning for privacy-preserving healthcare.
- Transformer-based adaptive models for sequential medical data.
- Clinical deployment using real-time electronic health records.
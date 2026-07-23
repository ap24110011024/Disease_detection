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

### 8. Model Behaviour and Evaluation Limitations

- The notebook `ckd/generalization.ipynb` does **not** perform cross-dataset transfer learning. It trains and evaluates a model directly on the CKD dataset, despite its name suggesting evaluation of a diabetes-trained model on CKD. Since the Pima Diabetes dataset contains 8 features and the CKD dataset contains 24 features, they do not share a common feature space. A genuine transfer-learning experiment would require either a shared feature subset or a different experimental formulation rather than a direct code modification.

- Most notebooks assume they are executed from their respective notebook directories. While `proactive_detection.ipynb` resolves repository paths independently of the working directory, the remaining notebooks rely on relative paths that function correctly only under the standard project structure.

- The Static MLP and Adaptive MLP use different training durations (50 epochs versus 20 epochs). This is an intentional design decision rather than an inconsistency. The Static MLP is trained once on the complete training set, whereas the Adaptive MLP is retrained repeatedly after each incremental update, resulting in a substantially higher cumulative computational cost if identical epoch counts were used.

- The Adaptive MLP alone does not outperform the Static MLP on F1-score. The adaptive learning strategy by itself provides no clear improvement over a well-trained static model on the current dataset. The additional proactive threshold-selection mechanism primarily improves Recall, although this improvement is not statistically significant on the current evaluation split.

- The proposed framework is evaluated using sequential batch-based windows rather than a true real-time streaming environment. Furthermore, the batches are generated without shuffling, causing the positive-class proportion to vary between batches. This introduces additional variability that is not isolated from the adaptation mechanism.

- The framework has been evaluated on two disease datasets (Pima Diabetes and Chronic Kidney Disease). Although this demonstrates applicability across multiple disease domains, broader validation on additional diseases and patient populations remains future work.

- The reproducibility checks and verification procedures were performed by the same development process that implemented the fixes. An independent verification by another researcher, laboratory member, or computing environment is recommended before reporting these results in a publication.


### 9. Statistical Observations

- Logistic Regression and Support Vector Machine produce identical Accuracy, Precision, Recall, and F1-score values on the current test split. Verification confirmed that both models generate the same confusion matrix (**TN = 83, FP = 17, FN = 28, TP = 26**). The SVM implementation (`SVC(kernel="rbf", probability=True, random_state=42)`) is correct, and the difference in ROC-AUC arises solely from differences in probability estimates rather than binary class predictions.

- McNemar's test comparing the Static MLP and the proposed Adaptive + Proactive framework produced a **p-value of 0.7905**, which exceeds the conventional significance threshold of 0.05. Consequently, although the proposed framework exhibits a different operating point with higher Recall, the observed improvement is **not statistically significant** on the current test split. Future work will include multi-seed evaluation and larger datasets to provide stronger statistical evidence.


# Future Work

Future research will focus on:

- Evaluation on larger multi-center clinical datasets.
- Incorporating longitudinal patient records.
- Integration with Explainable AI (SHAP, LIME).
- Validation using external hospital datasets.
- Federated learning for privacy-preserving healthcare.
- Transformer-based adaptive models for sequential medical data.
- Clinical deployment using real-time electronic health records.
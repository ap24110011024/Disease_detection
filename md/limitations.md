# Limitations

Although the proposed adaptive disease detection framework demonstrated promising performance, several limitations should be considered.

## 1. Dataset Scope

The proposed framework was evaluated using the Pima Indians Diabetes dataset as the primary benchmark and the Chronic Kidney Disease (CKD) dataset for cross-dataset validation. Although these datasets are widely used in medical machine learning research, they represent only two disease domains. Future work should evaluate the framework on additional public healthcare datasets to assess its generalizability across different diseases and populations.

---

## 2. Statistical Significance

The Adaptive MLP framework was compared with the Random Forest baseline using McNemar's statistical significance test.

The obtained p-value was: p = 0.263

Since the p-value is greater than the conventional significance threshold (0.05), the observed performance difference between the Adaptive MLP and Random Forest models is **not statistically significant** on the Pima Diabetes dataset.

This result suggests that additional experiments using larger and more diverse datasets are required before concluding that adaptive learning consistently outperforms conventional machine learning models.

---

## 3. Adaptive Learning Performance

The corrected ablation study reports the **mean performance across all adaptive batches** rather than the final batch alone.

Under this evaluation protocol, the Adaptive MLP did not outperform the Static MLP baseline in terms of overall Accuracy or F1 Score. However, the Adaptive + Proactive framework achieved substantially higher Recall, indicating improved sensitivity for identifying positive disease cases.

This trade-off is consistent with the objective of proactive disease detection, where reducing false negatives is often more important than maximizing overall classification accuracy.

---

## 4. Batch-Based Evaluation

The adaptive framework currently performs model updates using fixed sequential batches rather than a continuous real-time data stream.

Future work can extend the framework to support true online learning and streaming clinical data environments.

---

## 5. Future Work

Several directions can further improve the proposed framework:

- Evaluation on additional medical datasets.
- Integration with deep learning architectures.
- Real-time online adaptive learning.
- Multi-disease prediction within a unified framework.
- Clinical validation using hospital data.
# Research Contributions

This repository presents an adaptive machine learning framework for proactive disease prediction.

## Primary Contributions

### 1. Canonical Machine Learning Baselines

Implemented and evaluated five supervised learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron

using a unified experimental protocol.

---

### 2. Reproducible Experimental Pipeline

Developed a reproducible workflow including:

- Dataset preprocessing
- Train/test splitting
- Feature scaling
- Model training
- Performance evaluation
- Result logging

ensuring consistent experimental results.

---

### 3. Adaptive Learning Framework

Proposed an adaptive retraining strategy capable of updating the predictive model as new patient batches become available, enabling continuous learning.

---

### 4. Proactive Disease Detection

Introduced a proactive evaluation framework that simulates future patient batches and evaluates the ability of the adaptive model to identify patients at elevated disease risk earlier than conventional static models.

---

### 5. Comparative Evaluation

Performed systematic comparison between:

- Static Baseline
- Adaptive Learning
- Proactive Adaptive Learning

using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

---

### 6. Statistical Validation

Performed statistical comparison of competing models using McNemar's Test to evaluate whether observed performance differences were statistically significant.

---

### 7. Cross-Dataset Evaluation

Applied the adaptive methodology independently to both:

- Pima Indians Diabetes Dataset
- Chronic Kidney Disease Dataset

demonstrating that the proposed framework can be successfully applied across multiple disease prediction tasks.

---

### 8. Publication-Ready Repository

The repository includes:

- Well-organized notebooks
- Reproducible result tables
- Performance visualizations
- Markdown documentation
- Canonical result files
- Research paper assets

to facilitate reproducibility and future research.

---

# Overall Contribution

This work demonstrates that adaptive machine learning strategies can improve disease prediction workflows by continuously incorporating newly available patient data while maintaining reproducible experimental practices. The framework serves as a foundation for future research on continual learning and proactive clinical decision support systems.
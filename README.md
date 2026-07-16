# Adaptive AI Models for Proactive Disease Detection

## Overview

This repository contains the implementation of our research project, **Adaptive AI Models for Proactive Disease Detection**, developed during an 8-week research internship under the guidance of 
**Dr. Ch Anil Carie**.

The project aims to develop an adaptive machine learning framework capable of improving disease prediction through adaptive learning and proactive detection. The framework is evaluated using the **Pima Indians Diabetes Dataset** as the primary dataset and validated on the **Chronic Kidney Disease (CKD) Dataset** to assess cross-dataset generalization.

---

# Research Objectives

- Develop machine learning models for disease prediction.
- Compare multiple supervised learning algorithms.
- Design an Adaptive Multi-Layer Perceptron (Adaptive MLP).
- Implement proactive disease detection using sliding-window analysis.
- Validate model generalization using the CKD dataset.
- Evaluate performance using multiple classification metrics.

---

# Team Members

| Name | Roll Number |
|------|-------------|
| Venkata Ajay Odugu | AP24110011016 |
| Mohanasritha Eerla | AP24110011024 |
| Vijay Perla | AP24110011059 |
| Neelima Bojanapu | AP24110011111 |

---

# Mentor

**Dr. Ch Anil Carie**

---

# Datasets

## Primary Dataset

**Pima Indians Diabetes Dataset**

- Source: UCI Machine Learning Repository
- Samples: 768
- Features: 8
- Target: Diabetes Prediction

Dataset Link

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

---

## Cross-Dataset Validation

**Chronic Kidney Disease Dataset**

- Source: UCI Machine Learning Repository
- Samples: 400
- Features: 24
- Target: CKD Prediction

Dataset Link

https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

---

# Repository Structure

```text
Disease_detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── diabetes/
│   ├── ckd/
│
├── paper/
│
├── results/
│   ├── accuracy/
│   ├── roc/
│   ├── matrix/
│   ├── week6/
│
├── explore.py
├── pipeline.py
├── requirements.txt
└── README.md
```

---

# Methodology

The project follows the workflow below:

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Baseline Model Development
5. Adaptive Learning
6. Proactive Disease Detection
7. Cross-Dataset Generalization
8. Performance Evaluation

---

# Machine Learning Models

The following baseline models were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

### Proposed Models

- Adaptive MLP
- Adaptive + Proactive Disease Detection Framework

---

# Performance Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

# Baseline Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|-------:|---------:|--------:|
| Logistic Regression | 0.7143 | 0.6087 | 0.5185 | 0.5600 | 0.8230 |
| Decision Tree | 0.6948 | 0.6667 | 0.2593 | 0.3733 | 0.7881 |
| Random Forest | 0.7468 | 0.6531 | 0.5926 | 0.6214 | 0.8143 |
| Support Vector Machine | 0.7662 | 0.6863 | 0.6364 | 0.6604 | 0.7374 |
| Multi-Layer Perceptron | 0.7468 | 0.6415 | 0.6296 | 0.6355 | 0.8439 |

---

# Proposed Framework Results

| Configuration | Precision | Recall | F1 Score |
|---------------|----------:|-------:|---------:|
| Static MLP | 0.6415 | 0.6296 | 0.6355 |
| Adaptive MLP | 0.6635 | 0.4655 | 0.5232 |
| Adaptive + Proactive | 0.4853 | 0.8046 | 0.5979 |

---

# Cross-Dataset Validation

The Adaptive MLP framework was validated on the **Chronic Kidney Disease (CKD)** dataset to evaluate its ability to generalize across different clinical domains.

| Dataset | Model | Accuracy | Precision | Recall | F1 Score |
|----------|--------|---------:|----------:|-------:|---------:|
| CKD | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CKD | Adaptive MLP | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

# Repository Contents

The repository includes:

- Data preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Baseline machine learning models
- Adaptive learning implementation
- Proactive disease detection experiments
- Cross-dataset validation
- Performance evaluation
- ROC curves
- Confusion matrices
- Feature importance analysis
- Ablation study
- Error analysis
- Research paper resources

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
- Git
- GitHub

---

# How to Run

Clone the repository:

```bash
git clone https://github.com/ap24110011024/Disease_detection.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute them in sequence.

---

# Future Work

- Validate the framework on additional clinical datasets.
- Incorporate Explainable AI techniques such as SHAP and LIME.
- Explore continual and online learning strategies.
- Investigate deployment for real-time clinical decision support.
- Extend the framework to multi-disease prediction.

---

# Acknowledgement

This work was carried out as part of an academic research internship under the guidance of **Dr. Ch Anil Carie**.

---

# License

This repository is intended for academic and research purposes.
# Related Work

## Overview

Machine learning has become an effective tool for disease prediction by enabling early diagnosis from clinical data. Numerous studies have applied supervised learning algorithms to medical datasets, particularly for diabetes and chronic kidney disease prediction.

---

## Traditional Machine Learning Methods

Logistic Regression is one of the most widely used baseline classifiers for medical diagnosis because of its simplicity and interpretability. However, its linear decision boundary often limits prediction performance for complex clinical datasets.

Decision Trees provide interpretable decision rules but are susceptible to overfitting, especially when trained on relatively small medical datasets.

Random Forest reduces overfitting through ensemble learning and has consistently demonstrated strong performance for disease prediction tasks.

Support Vector Machine (SVM) is effective for nonlinear classification using kernel functions and has been widely applied in healthcare analytics.

---

## Neural Network Approaches

Multi-Layer Perceptrons (MLPs) have been successfully used for disease prediction because they can model nonlinear relationships among clinical features.

Most existing MLP-based disease prediction systems employ static training, where the model remains unchanged after the initial training process. Such approaches may struggle to adapt when new patient data become available.

---

## Adaptive Learning

Adaptive learning techniques enable machine learning models to update their behavior as new information is introduced. These approaches improve robustness and allow models to remain effective under changing data distributions.

Adaptive learning has been investigated in several domains; however, its application to proactive disease prediction remains relatively limited.

---

## Proactive Disease Detection

Traditional disease prediction systems generally focus on identifying the current disease status of a patient.

Proactive disease detection extends this concept by identifying patients who may become high-risk based on recent clinical observations. Sliding-window analysis provides a practical mechanism for incorporating temporal information into predictive models.

---

## Research Gap

Although previous studies have demonstrated the effectiveness of traditional machine learning models for disease prediction, relatively few have investigated adaptive learning together with proactive detection strategies.

Furthermore, most published studies evaluate models using a single dataset, making it difficult to assess generalization across different diseases.

---

## Contribution of This Work

The proposed research addresses these limitations by:

- Comparing multiple baseline machine learning models.
- Developing an Adaptive Multi-Layer Perceptron (Adaptive MLP).
- Incorporating proactive disease detection using a sliding-window approach.
- Validating the proposed framework on both Diabetes and Chronic Kidney Disease datasets.
- Evaluating performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

This work demonstrates that adaptive learning can improve predictive performance while proactive detection enhances sensitivity for identifying disease-positive patients.
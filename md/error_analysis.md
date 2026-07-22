# Error Analysis

This document analyzes the prediction errors made by the proposed adaptive disease detection framework.

## 1. False Positives

False positives correspond to healthy patients incorrectly classified as having the disease.

Possible reasons include:

- Overlapping clinical feature distributions.
- High glucose or BMI values without actual disease.
- Conservative threshold selection favoring early detection.

Although false positives increase unnecessary follow-up testing, they are generally less harmful than missed diagnoses in disease screening applications.

---

## 2. False Negatives

False negatives correspond to diseased patients incorrectly classified as healthy.

Possible causes include:

- Mild or early-stage disease presentation.
- Limited discriminative information in available features.
- Small dataset size.

Reducing false negatives was one of the primary objectives of the adaptive framework.

---

## 3. Precision–Recall Trade-off

The Adaptive + Proactive framework intentionally prioritizes Recall over Precision.

Higher Recall enables earlier identification of high-risk patients, while accepting a moderate increase in false positives. This trade-off is appropriate for preventive healthcare applications, where missed diagnoses are typically more costly than additional diagnostic testing.

---

## 4. Dataset Characteristics

### Pima Diabetes Dataset

The dataset contains overlapping feature distributions between diabetic and non-diabetic patients, making perfect classification impossible.

### CKD Dataset

The CKD dataset exhibits stronger feature separability, leading to substantially higher classification performance across multiple machine learning models.

---

## 5. Sources of Error

Observed prediction errors may result from:

- Measurement noise.
- Missing clinical information.
- Small sample size.
- Class imbalance.
- Patient heterogeneity.
- Biological variability.

---

## 6. Lessons Learned

The experiments indicate that adaptive retraining and proactive prediction improve the detection of high-risk patients without substantially reducing overall model stability.

Future improvements should focus on larger datasets, richer feature sets, and prospective clinical validation.
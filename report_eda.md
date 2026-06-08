# Exploratory Data Analysis Report

## Dataset Source

The dataset used in this project is the Pima Indians Diabetes Dataset obtained from the UCI Machine Learning Repository and Kaggle. The dataset contains 768 patient records and is widely used for diabetes prediction research and machine learning studies.

---

## Class Balance

The target variable in the dataset is **Outcome**.

* Non-Diabetic (0): 500 patients
* Diabetic (1): 268 patients

The dataset shows a moderate class imbalance, with non-diabetic patients representing the majority class.

---

## Missing Value Strategy

Several features such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI contain zero values that are medically unrealistic and are treated as missing values.

To address this issue:

* Zero values were replaced using the median of the respective feature.
* Median imputation was selected because it is less sensitive to outliers and preserves the overall data distribution.

---

## Feature Distributions

Histograms were generated for all numerical features using Matplotlib.

Key observations include:

* Glucose and BMI show relatively wide distributions across patients.
* Insulin contains several extreme values indicating possible outliers.
* Age distribution is right-skewed, indicating a larger number of younger patients.
* Diabetes Pedigree Function shows significant variability among individuals.

---

## Key Insights

### Insight 1

Glucose appears to be one of the strongest indicators of diabetes, with diabetic patients generally showing higher glucose levels.

### Insight 2

The dataset contains class imbalance, which should be considered when selecting evaluation metrics and machine learning models.

### Insight 3

Several features contain unrealistic zero values, making data preprocessing and missing value handling essential before model training.

---

## Conclusion

The exploratory data analysis helped us better understand the structure and quality of the dataset. The findings from this analysis will guide future preprocessing, feature engineering, and machine learning model development tasks in the project.

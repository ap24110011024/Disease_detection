# Machine Learning-Based Models for the Prediction of Breast Cancer Recurrence Risk

## Problem

Nearly 30% of breast cancer patients experience recurrence after treatment, yet clinicians have no reliable, low-cost tool to predict who is at risk. Existing methods depend on expensive imaging or pathological data and typically catch recurrence only after symptoms appear, missing the early intervention window entirely.

## Method

The authors collected 25 routine clinical and laboratory features from electronic medical records of 342 breast cancer patients. Eleven machine learning algorithms were trained on 70% of the data and tested on the remaining 30%, with 3-fold cross-validation applied during training. AdaBoost was selected as the best model, and SHAP values were used to explain which features drove each prediction.

## Key Metrics

AdaBoost achieved an AUC of 0.987, accuracy of 97.1%, sensitivity of 94.7%, specificity of 97.6%, and an NPV of 98.8%. SHAP analysis identified CA125, CEA, fibrinogen, and tumor diameter as the four most important predictors. Decision curve analysis confirmed the model provided greater net clinical benefit than treating all or no patients.

## Gap Identified

The study is single-center with only 342 patients and no external validation cohort, limiting generalizability. Genomic features such as gene mutations were not included despite their known prognostic value. The retrospective design also introduces selection bias, and prospective validation across diverse populations and hospital settings is still needed before clinical deployment.

# Decision Tree vs Random Forest

| Metric    | Decision Tree | Random Forest |
| ----------|--------------:|--------------:|
| Accuracy  | 0.7692 | 0.7468 |
| Precision | 0.6842 | 0.6531 |
| Recall    | 0.6111 | 0.5926 |
| F1 Score  | 0.6452 | 0.6214 |
| AUC ROC   | 0.8056 | 0.8143 |

*Numbers taken from `results/canonical_baseline_results.csv`. Both models use `random_state=42`. The Decision Tree is tuned using 5-fold GridSearchCV with F1-score as the optimization metric.*

---

## Observations

1. **Decision Tree achieved a slightly higher Accuracy (76.92%) than Random Forest (74.68%)** on the Pima Diabetes dataset.

2. **Decision Tree obtained higher Precision (68.42%) than Random Forest (65.31%)**, indicating fewer false positive predictions.

3. **Decision Tree achieved a slightly higher Recall (61.11%) compared to Random Forest (59.26%)**, identifying marginally more diabetic patients in this experimental split.

4. **Decision Tree achieved the highest F1 Score (0.6452) among the two models**, demonstrating a slightly better balance between Precision and Recall.

5. **Random Forest achieved the highest AUC-ROC (0.8143 vs 0.8056)**, indicating stronger overall discrimination across all classification thresholds, even though its default-threshold classification metrics were slightly lower.

6. Overall, both models performed competitively on the Pima Diabetes dataset. The optimized Decision Tree achieved better Accuracy, Precision, Recall, and F1 Score on this train-test split, whereas Random Forest demonstrated stronger ranking capability through its higher AUC-ROC. Depending on the application, either model could be preferred: Decision Tree for its simpler interpretability and threshold-based performance, or Random Forest for its greater robustness and overall discriminative ability.

---

## Conclusion

The optimized Decision Tree slightly outperformed Random Forest in Accuracy, Precision, Recall, and F1 Score after hyperparameter tuning using 5-fold GridSearchCV with F1-score optimization. However, Random Forest continued to achieve the highest AUC-ROC, suggesting better overall class separation across varying decision thresholds. These results indicate that while the tuned Decision Tree performed better on the selected evaluation split, Random Forest remains a strong and reliable ensemble baseline for diabetes prediction.
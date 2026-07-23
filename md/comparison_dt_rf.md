# Decision Tree vs Random Forest

| Metric    | Decision Tree | Random Forest |
| ----------|--------------:|--------------:|
| Accuracy  | 0.7857 | 0.7662 |
| Precision | 0.6981 | 0.6957 |
| Recall    | 0.6852 | 0.5926 |
| F1 Score  | 0.6916 | 0.6400 |
| AUC ROC   | 0.7887 | 0.8179 |

*Numbers taken from `results/canonical_baseline_results.csv`. Both models use `random_state=42`. The Decision Tree is tuned using 5-fold GridSearchCV with F1-score as the optimization metric.*

---

## Observations

1. **Decision Tree achieved a higher Accuracy (78.57%) than Random Forest (76.62%)** on the Pima Diabetes dataset.

2. **Decision Tree obtained marginally higher Precision (69.81%) than Random Forest (69.57%)**, essentially a tie on false-positive rate.

3. **Decision Tree achieved a meaningfully higher Recall (68.52% vs. 59.26%)**, identifying more diabetic patients in this experimental split.

4. **Decision Tree achieved the highest F1 Score (0.6916) among the two models**, demonstrating a better balance between Precision and Recall.

5. **Random Forest achieved the highest AUC-ROC (0.8179 vs 0.7887)**, indicating stronger overall discrimination across all classification thresholds, even though its default-threshold classification metrics were lower.

6. Overall, both models performed competitively on the Pima Diabetes dataset. The optimized Decision Tree achieved better Accuracy, Precision, Recall, and F1 Score on this train-test split, whereas Random Forest demonstrated stronger ranking capability through its higher AUC-ROC. Depending on the application, either model could be preferred: Decision Tree for its simpler interpretability and threshold-based performance, or Random Forest for its greater robustness and overall discriminative ability.

---

## Conclusion

The optimized Decision Tree outperformed Random Forest in Accuracy, Precision, Recall, and F1 Score after hyperparameter tuning using 5-fold GridSearchCV with F1-score optimization. However, Random Forest continued to achieve the highest AUC-ROC, suggesting better overall class separation across varying decision thresholds. These results indicate that while the tuned Decision Tree performed better on the selected evaluation split, Random Forest remains a strong and reliable ensemble baseline for diabetes prediction.
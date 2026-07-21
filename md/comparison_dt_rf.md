# Decision Tree vs Random Forest

| Metric    | DT     | RF     |
| --------- | ------ | ------ |
| Accuracy  | 0.7987 | 0.7468 |
| Precision | 0.7447 | 0.6531 |
| Recall    | 0.6481 | 0.5926 |
| F1 Score  | 0.6931 | 0.6214 |
| AUC ROC   | 0.7921 | 0.8143 |

*Numbers from `results/canonical_baseline_results.csv`, both models trained with `random_state=42`. Decision Tree is tuned via 5-fold `GridSearchCV` over `max_depth` (3–10), scored on F1.*

## Observations

1. Decision Tree achieved higher Accuracy (79.87%) compared to Random Forest (74.68%).
2. Decision Tree achieved higher Recall (64.81%) compared to Random Forest (59.26%), indicating better identification of diabetic patients on this split.
3. Decision Tree obtained a higher F1 Score (0.6931) than Random Forest (0.6214), showing a better balance between Precision and Recall.
4. Random Forest still achieves a higher AUC-ROC (0.8143 vs 0.7921), meaning it separates the two classes better across all thresholds — the Decision Tree's edge is specific to its default 0.5 decision threshold, not overall discriminative power.
5. Based on Accuracy, Precision, Recall, and F1, Decision Tree performed better than Random Forest on this Pima Diabetes split. Random Forest remains the stronger model by AUC-ROC, and is generally the more robust choice across different thresholds and resampling, since a single decision tree is more sensitive to the specific train/test split than an ensemble.
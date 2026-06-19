# Decision Tree vs Random Forest

| Metric    | DT     | RF     |
| --------- | ------ | ------ |
| Accuracy  | 0.6948 | 0.7468 |
| Precision | 0.6667 | 0.6531 |
| Recall    | 0.2593 | 0.5926 |
| F1 Score  | 0.3733 | 0.6214 |
| AUC ROC   | 0.7881 | 0.8143 |

## Observations

1. Random Forest achieved higher Accuracy (74.68%) compared to Decision Tree (69.48%).
2. Random Forest achieved significantly higher Recall (59.26%) compared to Decision Tree (25.93%), indicating better identification of diabetic patients.
3. Random Forest obtained a higher F1 Score (0.6214) than Decision Tree (0.3733), showing a better balance between Precision and Recall.
4. Random Forest also achieved a higher AUC-ROC score (0.8143), indicating stronger overall classification performance.
5. Based on all evaluation metrics, Random Forest performed better than Decision Tree on the Pima Diabetes Dataset.

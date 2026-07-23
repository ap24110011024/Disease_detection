# CKD Leakage Verification

A leakage verification script was executed before the final CKD experiments.

## Duplicate Record Check

| Check | Result |
|-------|-------:|
| Exact duplicate rows | 0 |
| Duplicate feature rows | 0 |
| Shared train/test feature rows | 0 |

These checks indicate that the training and testing sets do not contain duplicate patient records or overlapping feature vectors.

## Feature Separability

Single-feature ROC-AUC values show that several laboratory measurements are highly predictive.

| Feature | ROC-AUC |
|---------|--------:|
| Hemoglobin | 0.9679 |
| Packed Cell Volume | 0.9326 |
| Serum Creatinine | 0.9179 |
| Specific Gravity | 0.8884 |
| Red Blood Cell Count | 0.8563 |

These findings suggest that the CKD dataset is naturally highly separable, explaining the strong classification performance achieved by multiple machine learning models without indicating data leakage.

## Conclusion

No evidence of data leakage was observed.

The high performance of CKD models is primarily explained by the strong discriminative power of several clinical biomarkers rather than overlap between training and testing data.
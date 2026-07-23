"""
CKD leakage / separability check.
Run once, keep the output as documentation of due diligence.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

raw = pd.read_csv("data/raw/chronic_kidney_disease.csv")
train = pd.read_csv("data/processed/chronic_kidney_disease_train.csv")
test = pd.read_csv("data/processed/chronic_kidney_disease_test.csv")

feat_cols_raw = [c for c in raw.columns if c != "class"]
feat_cols_proc = [c for c in train.columns if c != "class"]

print("1) Exact duplicate rows in raw data (any label):", raw.duplicated().sum())
print("2) Duplicate feature-rows in raw data ignoring label:", raw.duplicated(subset=feat_cols_raw).sum())
print("3) Feature-rows shared between processed train and test:",
      len(train[feat_cols_proc].merge(test[feat_cols_proc], how="inner")))

full = pd.concat([train, test], ignore_index=True)
y = full["class"]
print("\n4) Single-feature AUC (numeric columns only), full train+test:")
aucs = []
for c in feat_cols_proc:
    col = full[c]
    if col.dtype == object:
        continue
    auc = roc_auc_score(y, col)
    auc = max(auc, 1 - auc)
    aucs.append((c, auc))
aucs.sort(key=lambda x: -x[1])
for c, a in aucs:
    print(f"   {c:10s} {a:.4f}")

import os

auc_df = pd.DataFrame(
    aucs,
    columns=["Feature", "ROC_AUC"]
)
os.makedirs("results/week6", exist_ok=True)

auc_df.to_csv(
    "results/week6/ckd_feature_auc.csv",
    index=False
)

print("\nSaved results/week6/ckd_feature_auc.csv")
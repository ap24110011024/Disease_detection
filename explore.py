import pandas as pd

df = pd.read_csv("data/raw/diabetes.csv")

print("===== SHAPE =====")
print(df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DESCRIBE =====")
print(df.describe())
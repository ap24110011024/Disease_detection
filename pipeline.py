import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Columns where 0 means missing
cols_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace 0 with median
for col in cols_with_missing:

    median_value = df.loc[df[col] != 0, col].median()

    df[col] = df[col].replace(
        0,
        median_value
    )

# Features and Target
X = df.drop("Outcome", axis=1)

y = df["Outcome"]

# Min-Max Scaling
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

print("Pipeline Completed")
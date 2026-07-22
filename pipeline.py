import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42

cols_with_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def preprocess(csv_path):
    """
    Loads the diabetes dataset and performs
    leakage-free preprocessing.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Features and Target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-Test Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # Median Imputation (Training only)
    for col in cols_with_missing:

        median_value = X_train.loc[
            X_train[col] != 0,
            col
        ].median()

        X_train[col] = X_train[col].replace(
            0,
            median_value
        )

        X_test[col] = X_test[col].replace(
            0,
            median_value
        )

    # Scaling (Training only)
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess(
        "data/raw/diabetes.csv"
    )

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    print("Pipeline Completed Successfully")
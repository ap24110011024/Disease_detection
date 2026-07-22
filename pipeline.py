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


def impute_zeros_median(X_train, X_test, cols=cols_with_missing):
    """
    Median-imputes Pima's biologically-impossible zero values, fitting
    the median on X_train only and applying it to both X_train and X_test.

    Reusable building block: also used by notebooks that do their own
    scaling and/or their own batch/window splitting (week4_mlp,
    adaptive_mlp, proactive_detection), where a fresh median needs to be
    fit at each batch/split boundary rather than a single global split.
    """
    for col in cols:

        median_value = X_train.loc[
            X_train[col] != 0,
            col
        ].median()

        X_train[col] = X_train[col].replace(0, median_value)
        X_test[col] = X_test[col].replace(0, median_value)

    return X_train, X_test


def preprocess(csv_path, scale=True):
    """
    Loads the diabetes dataset and performs
    leakage-free preprocessing: split first, then median-impute
    (fit on train only), then optionally MinMax-scale (fit on train only).

    Parameters
    ----------
    scale : bool
        If True (default), returns MinMax-scaled numpy arrays. If False,
        returns imputed-but-unscaled DataFrames, for notebooks that
        already do their own MinMaxScaler fit_transform/transform
        downstream and would otherwise double-scale the data.

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
    X_train, X_test = impute_zeros_median(X_train, X_test)

    if not scale:
        return X_train, X_test, y_train, y_test

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
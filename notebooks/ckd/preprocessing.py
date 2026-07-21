import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_ckd_data():

    # ----------------------------
    # Load ARFF Dataset
    # ----------------------------

    file_path = "data/chronic_kidney_disease_full.arff"

    with open(file_path, "r") as f:
        lines = f.readlines()

    columns = [
        "age","bp","sg","al","su","rbc","pc","pcc","ba",
        "bgr","bu","sc","sod","pot","hemo","pcv",
        "wc","rc","htn","dm","cad","appet","pe","ane","class"
    ]

    data = []

    start = False

    for line in lines:

        line = line.strip()

        if not start:
            if line.lower() == "@data":
                start = True
            continue

        if line == "" or line.startswith("%"):
            continue

        if line.endswith(","):
            line = line[:-1]

        values = [x.strip() for x in line.split(",")]

        # Skip malformed row
        if len(values) != 25:
            continue

        data.append(values)

    df = pd.DataFrame(data, columns=columns)

    # ----------------------------
    # Replace Missing Values
    # ----------------------------

    df.replace("?", np.nan, inplace=True)
    df.replace("\t?", np.nan, inplace=True)
    df.replace(" ?", np.nan, inplace=True)

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df.replace("nan", np.nan, inplace=True)

    # ----------------------------
    # Numerical Columns
    # ----------------------------

    numeric_cols = [
        "age","bp","sg","al","su",
        "bgr","bu","sc","sod","pot",
        "hemo","pcv","wc","rc"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Imputation
    # ----------------------------

    num_cols = df.select_dtypes(include=["float64","int64"]).columns

    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # ----------------------------
    # Label Encoding
    # ----------------------------

    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    # ----------------------------
    # Features and Target
    # ----------------------------

    X = df.drop("class", axis=1)

    y = df["class"]

    # ----------------------------
    # Scaling
    # ----------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # ----------------------------
    # Train Test Split
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    feature_names = X.columns.tolist()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

return X_train, X_test, y_train, y_test, feature_names

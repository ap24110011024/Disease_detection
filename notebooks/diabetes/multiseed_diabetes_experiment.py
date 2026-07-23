"""
Multi-seed robustness check for the Pima Diabetes ablation study
(Static MLP vs Adaptive MLP vs Adaptive + Proactive).

This is a NEW analysis, not part of the original repository. It replicates
the exact methodology of notebooks/diabetes/week4_mlp.ipynb,
notebooks/diabetes/adaptive_mlp.ipynb, and
notebooks/diabetes/proactive_detection.ipynb line-for-line (same
architecture, same epochs, same batch construction, same threshold search),
and runs it across seeds 42, 7, 123 -- the same three seeds already used for
the CKD multi-seed check in notebooks/ckd/multiseed_validation.ipynb.

What varies per seed, and why:
- Static MLP: the train/test split's random_state (42/7/123), exactly like
  the CKD multi-seed check varies its split seed. This gives a genuinely
  different train/test partition per seed.
- Adaptive MLP / Adaptive + Proactive: in the original notebooks, batch
  construction is a fixed slice of the dataframe in file order and does NOT
  depend on any seed at all -- batches[i] is the same 76-or-84-row slice no
  matter what random_state is set. So for these two, the only thing a
  "seed" can vary, without changing the batching methodology the repo
  itself uses, is TensorFlow/NumPy/Python's own training randomness (weight
  initialization, minibatch shuffling order during .fit). That is what is
  varied here for these two configurations. This is stated explicitly in
  the report -- it is a real methodological limitation of testing
  robustness this way, not hidden.
"""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "raw" / "diabetes.csv"

COLS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def impute_zeros_median(X_train, X_test, cols=COLS_MISSING):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in cols:
        median_value = X_train.loc[X_train[col] != 0, col].median()
        X_train[col] = X_train[col].replace(0, median_value)
        X_test[col] = X_test[col].replace(0, median_value)
    return X_train, X_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_static(seed):
    set_seed(seed)
    df = pd.read_csv(DATA)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_test = impute_zeros_median(X_train, X_test)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_mlp(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    return dict(
        Accuracy=accuracy_score(y_test, y_pred),
        Precision=precision_score(y_test, y_pred, zero_division=0),
        Recall=recall_score(y_test, y_pred, zero_division=0),
        F1=f1_score(y_test, y_pred, zero_division=0),
    )


def make_batches(df):
    batch_size = len(df) // 10
    batches = []
    for i in range(10):
        start = i * batch_size
        batch = df.iloc[start:] if i == 9 else df.iloc[start:start + batch_size]
        batches.append(batch)
    return batches


def run_adaptive(seed):
    set_seed(seed)
    df = pd.read_csv(DATA)
    batches = make_batches(df)

    rows = []
    for i in range(1, len(batches)):
        train_data = pd.concat(batches[:i], ignore_index=True)
        test_data = batches[i]

        X_train = train_data.drop("Outcome", axis=1)
        y_train = train_data["Outcome"]
        X_test = test_data.drop("Outcome", axis=1)
        y_test = test_data["Outcome"]

        X_train, X_test = impute_zeros_median(X_train, X_test)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        y_prob = model.predict(X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int)

        rows.append(dict(
            Accuracy=accuracy_score(y_test, y_pred),
            Precision=precision_score(y_test, y_pred, zero_division=0),
            Recall=recall_score(y_test, y_pred, zero_division=0),
            F1=f1_score(y_test, y_pred, zero_division=0),
        ))
    dfres = pd.DataFrame(rows)
    return dict(dfres.mean())


def run_proactive(seed):
    set_seed(seed)
    df = pd.read_csv(DATA)
    batches = make_batches(df)
    thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7]

    rows = []
    for i in range(3, 8):
        train_batches = batches[:i - 1]
        val_batch = batches[i - 1]
        future_batch = batches[i]

        train_df = pd.concat(train_batches)
        X_train = train_df.drop("Outcome", axis=1)
        y_train = train_df["Outcome"]
        X_val = val_batch.drop("Outcome", axis=1)
        y_val = val_batch["Outcome"]
        X_future = future_batch.drop("Outcome", axis=1)
        y_future = future_batch["Outcome"]

        for col in COLS_MISSING:
            median_value = X_train.loc[X_train[col] != 0, col].median()
            X_train[col] = X_train[col].replace(0, median_value)
            X_val[col] = X_val[col].replace(0, median_value)
            X_future[col] = X_future[col].replace(0, median_value)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_future = scaler.transform(X_future)

        model = build_mlp(X_train.shape[1])
        model.fit(X_train, y_train, epochs=20, verbose=0)

        y_val_prob = model.predict(X_val, verbose=0).flatten()
        best_f1, threshold = -1, 0.5
        for t in thresholds_to_try:
            y_val_pred = (y_val_prob >= t).astype(int)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            if val_f1 > best_f1:
                best_f1, threshold = val_f1, t

        y_prob = model.predict(X_future, verbose=0).flatten()
        y_pred = (y_prob >= threshold).astype(int)

        rows.append(dict(
            Accuracy=accuracy_score(y_future, y_pred),
            Precision=precision_score(y_future, y_pred, zero_division=0),
            Recall=recall_score(y_future, y_pred, zero_division=0),
            F1=f1_score(y_future, y_pred, zero_division=0),
        ))
    dfres = pd.DataFrame(rows)
    return dict(dfres.mean())


if __name__ == "__main__":
    seeds = [42, 7, 123]
    all_rows = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        s = run_static(seed)
        a = run_adaptive(seed)
        p = run_proactive(seed)
        print("static:", s)
        print("adaptive:", a)
        print("proactive:", p)
        all_rows.append(dict(Seed=seed, Config="Static MLP", **s))
        all_rows.append(dict(Seed=seed, Config="Adaptive MLP", **a))
        all_rows.append(dict(Seed=seed, Config="Adaptive + Proactive", **p))

    out = pd.DataFrame(all_rows)
    output_dir = REPO / "results" / "week6"
    output_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        output_dir / "multiseed_diabetes_ablation.csv",
        index=False
    )
    print(out.round(4))

    # Summary: mean/std per configuration across seeds
    summary = out.groupby("Config")[["Accuracy", "Precision", "Recall", "F1"]].agg(["mean", "std"])
    summary.to_csv(
    output_dir / "multiseed_diabetes_ablation_summary.csv"
)    
    print(summary.round(4))
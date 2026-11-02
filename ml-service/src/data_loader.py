import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    RANDOM_STATE,
    FEATURE_COLS,
    RAW_NUMERICAL_COLS,
    PCA_COLS,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def load_and_validate_data(filepath=RAW_DATA_PATH):
    """
    Loads creditcard.csv and executes strict validation:
    - File existence check with onboarding instructions if absent.
    - Column name check (must contain Time, Amount, Class, V1..V28).
    - Null / NaN check.
    - Duplicate detection and removal (logs and drops ~1,081 duplicates).
    Returns (cleaned_df, validation_report).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Please download the Kaggle Credit Card Fraud Detection dataset from:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place 'creditcard.csv' at '{filepath.resolve()}'."
        )

    df = pd.read_csv(filepath)
    raw_row_count = len(df)

    # Validate required columns
    expected_cols = set(FEATURE_COLS + [TARGET_COL])
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(list(missing_cols))}")

    # Check nulls
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())

    # Check duplicates
    duplicate_count = int(df.duplicated().sum())
    
    # Drop duplicates
    if duplicate_count > 0:
        df_clean = df.drop_duplicates().reset_index(drop=True)
    else:
        df_clean = df.copy()

    cleaned_row_count = len(df_clean)
    fraud_count = int((df_clean[TARGET_COL] == 1).sum())
    legit_count = int((df_clean[TARGET_COL] == 0).sum())
    fraud_pct = float((fraud_count / cleaned_row_count) * 100)

    validation_report = {
        "status": "valid",
        "file_path": str(filepath.resolve()),
        "raw_rows": raw_row_count,
        "duplicate_rows_dropped": duplicate_count,
        "duplicate_pct": round((duplicate_count / raw_row_count) * 100, 4),
        "clean_rows": cleaned_row_count,
        "total_nulls": total_nulls,
        "columns_count": len(df_clean.columns),
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "data_types": {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
    }

    return df_clean, validation_report


def split_data(df, random_state=RANDOM_STATE):
    """
    Performs stratified 70% train / 15% validation / 15% test split.
    Ensures zero data leakage between splits.
    Returns ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    # First split: 70% train, 30% temp
    test_and_val_ratio = VAL_RATIO + TEST_RATIO  # 0.30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_and_val_ratio,
        stratify=y,
        random_state=random_state
    )

    # Second split: 50/50 of temp -> 15% val, 15% test
    val_share_of_temp = VAL_RATIO / test_and_val_ratio  # 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_share_of_temp),
        stratify=y_temp,
        random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _compute_histogram_bins(series, bins=35, log_scale=False):
    """
    Computes histogram bin intervals and counts for charting.
    """
    data = series.values.copy()
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return []

    if log_scale:
        min_val = max(0.01, float(np.min(data)))
        max_val = max(min_val * 1.1, float(np.max(data)))
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
    else:
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        if min_val == max_val:
            max_val += 1.0
        bin_edges = np.linspace(min_val, max_val, bins + 1)

    counts, edges = np.histogram(data, bins=bin_edges)
    histogram = []
    for i in range(len(counts)):
        histogram.append({
            "bin_start": round(float(edges[i]), 2),
            "bin_end": round(float(edges[i + 1]), 2),
            "bin_label": f"${edges[i]:.1f} - ${edges[i+1]:.1f}",
            "count": int(counts[i])
        })
    return histogram


def compute_dataset_summary(df, validation_report, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Constructs a comprehensive dataset summary dictionary ready for JSON serialization.
    """
    def get_stats(series):
        s = series.astype(float)
        return {
            "count": int(len(s)),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "median": round(float(s.median()), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p99": round(float(s.quantile(0.99)), 2),
            "p99_9": round(float(s.quantile(0.999)), 2),
            "max": round(float(s.max()), 2),
        }

    amount_all = df["Amount"]
    amount_legit = df[df[TARGET_COL] == 0]["Amount"]
    amount_fraud = df[df[TARGET_COL] == 1]["Amount"]

    summary = {
        "validation": validation_report,
        "splits": {
            "train": {
                "total": len(X_train),
                "legit": int((y_train == 0).sum()),
                "fraud": int((y_train == 1).sum()),
                "fraud_rate": round(float((y_train == 1).mean() * 100), 4)
            },
            "validation": {
                "total": len(X_val),
                "legit": int((y_val == 0).sum()),
                "fraud": int((y_val == 1).sum()),
                "fraud_rate": round(float((y_val == 1).mean() * 100), 4)
            },
            "test": {
                "total": len(X_test),
                "legit": int((y_test == 0).sum()),
                "fraud": int((y_test == 1).sum()),
                "fraud_rate": round(float((y_test == 1).mean() * 100), 4)
            }
        },
        "amount_statistics": {
            "overall": get_stats(amount_all),
            "legit": get_stats(amount_legit),
            "fraud": get_stats(amount_fraud)
        },
        "time_statistics": {
            "overall": get_stats(df["Time"]),
            "span_hours": round(float((df["Time"].max() - df["Time"].min()) / 3600), 1)
        },
        "amount_distributions": {
            "log_bins_overall": _compute_histogram_bins(amount_all, bins=30, log_scale=True),
            "linear_bins_under_500": _compute_histogram_bins(amount_all[amount_all <= 500], bins=25, log_scale=False),
            "fraud_amounts": _compute_histogram_bins(amount_fraud, bins=20, log_scale=False)
        }
    }
    return summary

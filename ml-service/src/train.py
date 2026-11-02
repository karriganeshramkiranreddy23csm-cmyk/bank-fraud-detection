import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_PATH,
    ARTIFACTS_DIR,
    RANDOM_STATE,
    RAW_NUMERICAL_COLS,
    PCA_COLS,
    FEATURE_COLS,
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
)
from src.data_loader import load_and_validate_data, split_data, compute_dataset_summary
from src.transformers import OutlierCapper
from src.pipeline import build_fraud_pipeline
from src.evaluate import evaluate_predictions, compute_pr_curve_data, compute_roc_curve_data
from src.ablation import run_ablation_study


def train_and_evaluate_all():
    """
    Executes the entire end-to-end ML pipeline:
    1. Loads, validates, and deduplicates Kaggle creditcard.csv.
    2. Performs stratified 70/15/15 split (Train/Val/Test).
    3. Computes and saves dataset_summary.json.
    4. Computes outlier reports across strategies (percentile, IQR) and saves outlier_report.json.
    5. Runs ablation study across capping, scalers, and resampling; saves ablation_results.json.
    6. Trains candidate models (Dummy, Logistic Regression, Random Forest, LightGBM, XGBoost).
    7. Evaluates each model on Validation (threshold selection) and Test (unbiased final evaluation).
    8. Selects the champion model, extracts feature importances, and saves best_model.joblib & metrics.json.
    """
    total_start = time.time()
    print("=" * 70)
    print("BANKING FRAUD PIPELINE: TRAINING & EVALUATION ENGINE")
    print("=" * 70)

    # 1. Load and validate data
    print("\n[Step 1/6] Loading and validating creditcard.csv...")
    df, validation_report = load_and_validate_data(RAW_DATA_PATH)
    print(f"Loaded {validation_report['clean_rows']:,} rows after dropping {validation_report['duplicate_rows_dropped']} duplicates.")
    print(f"Frauds: {validation_report['fraud_count']:,} ({validation_report['fraud_percentage']:.3f}%) | Legitimate: {validation_report['legit_count']:,}")

    # 2. Stratified Split (No Leakage)
    print("\n[Step 2/6] Executing stratified 70/15/15 split (random_state=42)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df, random_state=RANDOM_STATE)
    print(f"Train set: {len(X_train):,} rows ({y_train.sum()} frauds, {y_train.mean()*100:.3f}%)")
    print(f"Val set:   {len(X_val):,} rows ({y_val.sum()} frauds, {y_val.mean()*100:.3f}%)")
    print(f"Test set:  {len(X_test):,} rows ({y_test.sum()} frauds, {y_test.mean()*100:.3f}%)")

    # 3. Generate dataset summary artifact
    print("\n[Step 3/6] Generating dataset summary and distribution statistics...")
    dataset_summary = compute_dataset_summary(
        df, validation_report, X_train, y_train, X_val, y_val, X_test, y_test
    )
    with open(ARTIFACTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)
    print("Saved dataset_summary.json")

    # 4. Outlier Analysis & Report
    print("\n[Step 4/6] Fitting OutlierCapper on training set and auditing clipping...")
    outlier_report = {}
    for strat in ["percentile", "iqr", "none"]:
        capper = OutlierCapper(strategy=strat, columns=RAW_NUMERICAL_COLS)
        capper.fit(X_train)
        
        # Test clipping audit on test set
        _ = capper.transform(X_test)
        outlier_report[strat] = capper.get_report()

    # Add distribution comparison before/after capping on Amount for UI visualization
    p_capper = OutlierCapper(strategy="percentile", columns=["Amount"]).fit(X_train)
    amt_test_raw = X_test["Amount"].values
    amt_test_capped = p_capper.transform(X_test)["Amount"].values

    outlier_report["amount_comparison"] = {
        "raw_max": round(float(np.max(amt_test_raw)), 2),
        "raw_p99": round(float(np.percentile(amt_test_raw, 99)), 2),
        "capped_max": round(float(np.max(amt_test_capped)), 2),
        "lower_bound": round(float(p_capper.bounds_["Amount"]["lower"]), 2),
        "upper_bound": round(float(p_capper.bounds_["Amount"]["upper"]), 2),
    }

    with open(ARTIFACTS_DIR / "outlier_report.json", "w") as f:
        json.dump(outlier_report, f, indent=2)
    print("Saved outlier_report.json")

    # 5. Ablation Study
    print("\n[Step 5/6] Running systematic ablation study...")
    ablation_results = run_ablation_study(X_train, y_train, X_val, y_val, model_type="lgb", verbose=True)
    with open(ARTIFACTS_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("Saved ablation_results.json")

    # 6. Train and Evaluate Candidate Models
    print("\n[Step 6/6] Training and evaluating candidate models across splits...")
    models_to_train = [
        {"key": "dummy", "name": "Dummy Baseline (Most Frequent)", "type": "dummy", "capper": "none", "scaler": "none", "imbalance": "none"},
        {"key": "lr", "name": "Logistic Regression (Balanced)", "type": "lr", "capper": "percentile", "scaler": "robust", "imbalance": "balanced"},
        {"key": "rf", "name": "Random Forest", "type": "rf", "capper": "percentile", "scaler": "robust", "imbalance": "balanced"},
        {"key": "xgb", "name": "XGBoost (Primary)", "type": "xgb", "capper": "percentile", "scaler": "robust", "imbalance": "balanced"},
        {"key": "lgb", "name": "LightGBM", "type": "lgb", "capper": "percentile", "scaler": "robust", "imbalance": "balanced"},
    ]

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    trained_pipelines = {}
    model_evaluations = {}

    for m in models_to_train:
        m_key = m["key"]
        print(f"\n--- Training {m['name']} ---")
        t0 = time.time()
        pipeline = build_fraud_pipeline(
            model_type=m["type"],
            capper_strategy=m["capper"],
            scaler_type=m["scaler"],
            imbalance_strategy=m["imbalance"],
            capper_cols=RAW_NUMERICAL_COLS,
            pos_weight=pos_weight,
            random_state=RANDOM_STATE
        )
        pipeline.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)
        trained_pipelines[m_key] = pipeline

        # Predictions
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        if m["type"] == "dummy":
            # Dummy represents predicting the majority class (not fraud) for all rows
            # This yields the classic ~99.83% accuracy trap with 0 caught frauds
            best_th = 0.5
            val_eval = evaluate_predictions(y_val, y_val_prob, chosen_threshold=0.5, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
            test_eval = evaluate_predictions(y_test, y_test_prob, chosen_threshold=0.5, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
        else:
            # Evaluate on validation (to select optimal thresholds)
            val_eval = evaluate_predictions(y_val, y_val_prob, chosen_threshold=0.5, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)
            
            # Pick best F1 threshold from validation
            best_th = val_eval["optimal_thresholds"].get("max_f1", {}).get("threshold", 0.5)

            # Evaluate on test with the validation-selected optimal threshold
            test_eval = evaluate_predictions(y_test, y_test_prob, chosen_threshold=best_th, fn_cost=DEFAULT_FN_COST, fp_cost=DEFAULT_FP_COST)

        print(f"Validation PR-AUC: {val_eval['summary']['pr_auc']:.4f} | Test PR-AUC: {test_eval['summary']['pr_auc']:.4f} | Test F1: {test_eval['metrics_at_threshold']['f1']:.4f}")
        print(f"Test Accuracy: {test_eval['summary']['accuracy']*100:.2f}% | Caught Frauds: {test_eval['metrics_at_threshold']['confusion_matrix']['tp']}/{int(y_test.sum())}")

        model_evaluations[m_key] = {
            "key": m_key,
            "name": m["name"],
            "fit_time_sec": fit_time,
            "validation": val_eval,
            "test": test_eval,
            "optimal_threshold_val": best_th
        }

    # Select champion model based on validation PR-AUC (excluding dummy)
    candidate_keys = [k for k in trained_pipelines.keys() if k != "dummy"]
    best_key = max(candidate_keys, key=lambda k: model_evaluations[k]["validation"]["summary"]["pr_auc"])
    champion_pipeline = trained_pipelines[best_key]

    print(f"\n==================================================")
    print(f"CHAMPION MODEL SELECTED: {model_evaluations[best_key]['name']} (PR-AUC: {model_evaluations[best_key]['test']['summary']['pr_auc']:.4f})")
    print(f"==================================================")

    # Feature Importance for Champion model
    feature_importances = []
    clf_step = champion_pipeline.named_steps.get("classifier")
    if hasattr(clf_step, "feature_importances_"):
        raw_importances = clf_step.feature_importances_
        # Feature order: Time, Amount, V1..V28 (after ColumnTransformer)
        feature_names = RAW_NUMERICAL_COLS + PCA_COLS
        for name, score in zip(feature_names, raw_importances):
            feature_importances.append({
                "feature": name,
                "importance": round(float(score), 5),
                "is_pca": name.startswith("V")
            })
        feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

    # Save artifacts
    metrics_payload = {
        "models": model_evaluations,
        "champion_key": best_key,
        "champion_name": model_evaluations[best_key]["name"],
        "feature_importances": feature_importances,
        "prevalence": {
            "train": round(float(y_train.mean()), 6),
            "val": round(float(y_val.mean()), 6),
            "test": round(float(y_test.mean()), 6),
        },
        "default_costs": {
            "fn_cost": DEFAULT_FN_COST,
            "fp_cost": DEFAULT_FP_COST
        }
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("Saved metrics.json")

    # Save champion pipeline
    joblib.dump(champion_pipeline, ARTIFACTS_DIR / "best_model.joblib")
    print("Saved best_model.joblib")

    elapsed = round(time.time() - total_start, 1)
    print(f"\nAll artifacts generated successfully in {elapsed}s.")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_all()

import os
import io
import json
import uuid
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLS,
    RAW_NUMERICAL_COLS,
    PCA_COLS,
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
)
from src.train import train_and_evaluate_all
from src.evaluate import compute_threshold_sweep, find_optimal_thresholds

app = FastAPI(
    title="Banking Fraud ML Engine",
    description="High-performance fraud detection pipeline with outlier capping & PR diagnostics",
    version="1.0.0"
)

# Enable CORS for local gateway & frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory training job tracker
training_jobs: Dict[str, Dict[str, Any]] = {}

# Cached artifacts in memory
cached_artifacts = {
    "model": None,
    "metrics": None,
    "dataset_summary": None,
    "outlier_report": None,
    "ablation_results": None,
}


def load_artifacts_from_disk():
    """Loads precomputed JSON artifacts and joblib pipeline into memory cache."""
    try:
        model_path = ARTIFACTS_DIR / "best_model.joblib"
        if model_path.exists():
            cached_artifacts["model"] = joblib.load(model_path)

        for name in ["metrics", "dataset_summary", "outlier_report", "ablation_results"]:
            json_path = ARTIFACTS_DIR / f"{name}.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    cached_artifacts[name] = json.load(f)
    except Exception as e:
        print(f"Warning loading artifacts: {e}")


# Load immediately on module load
load_artifacts_from_disk()


# Pydantic schemas
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction amount in currency units")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom classification threshold")


def run_training_worker(job_id: str):
    """Worker executing training in background thread."""
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["message"] = "Training models, running ablation study, and computing PR diagnostics..."
        training_jobs[job_id]["progress"] = 20

        # Execute training
        metrics = train_and_evaluate_all()

        # Reload cache
        load_artifacts_from_disk()

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = 100
        training_jobs[job_id]["message"] = "Training completed successfully."
        training_jobs[job_id]["completed_at"] = time.time()
        training_jobs[job_id]["champion_model"] = metrics.get("champion_name", "Random Forest")
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["progress"] = 0
        training_jobs[job_id]["error"] = str(e)
        training_jobs[job_id]["message"] = f"Training failed: {e}"


# API Endpoints
@app.get("/health")
def health_check():
    has_model = cached_artifacts["model"] is not None
    return {
        "status": "healthy",
        "model_loaded": has_model,
        "artifacts_ready": all(cached_artifacts[k] is not None for k in ["metrics", "dataset_summary", "outlier_report", "ablation_results"])
    }


@app.post("/train")
def start_training(background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Training job queued...",
        "created_at": time.time()
    }
    background_tasks.add_task(run_training_worker, job_id)
    return {"job_id": job_id, "status": "queued", "message": "Training job initiated in background."}


@app.get("/train/status/{job_id}")
def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return training_jobs[job_id]


@app.get("/dataset/summary")
def get_dataset_summary():
    if not cached_artifacts["dataset_summary"]:
        load_artifacts_from_disk()
    if not cached_artifacts["dataset_summary"]:
        raise HTTPException(status_code=404, detail="Dataset summary artifact not found. Please train first.")
    return cached_artifacts["dataset_summary"]


@app.get("/outliers/report")
def get_outliers_report(strategy: Optional[str] = Query(None, description="percentile, iqr, none")):
    if not cached_artifacts["outlier_report"]:
        load_artifacts_from_disk()
    report = cached_artifacts["outlier_report"]
    if not report:
        raise HTTPException(status_code=404, detail="Outlier report artifact not found.")

    if strategy and strategy in report:
        return {
            "strategy": strategy,
            "report": report[strategy],
            "amount_comparison": report.get("amount_comparison")
        }
    return report


@app.get("/metrics")
def get_metrics():
    if not cached_artifacts["metrics"]:
        load_artifacts_from_disk()
    metrics = cached_artifacts["metrics"]
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics artifact not found. Please train first.")
    return metrics


@app.get("/pr-curve")
def get_pr_curve(
    model: str = Query("xgb", description="Model key: dummy, lr, rf, xgb, lgb"),
    split: str = Query("test", description="val or test")
):
    if not cached_artifacts["metrics"]:
        load_artifacts_from_disk()
    metrics = cached_artifacts["metrics"]
    if not metrics or "models" not in metrics:
        raise HTTPException(status_code=404, detail="Metrics not ready.")

    if model not in metrics["models"]:
        model = metrics.get("champion_key", "rf")

    split_key = "test" if split == "test" else "validation"
    model_data = metrics["models"][model][split_key]
    pr_curve_data = model_data.get("pr_curve", {})

    return {
        "model_key": model,
        "model_name": metrics["models"][model]["name"],
        "split": split,
        "pr_auc": pr_curve_data.get("pr_auc", 0.0),
        "baseline_no_skill": pr_curve_data.get("baseline_no_skill", 0.0017),
        "points": pr_curve_data.get("points", []),
    }


@app.get("/thresholds")
def get_thresholds(
    model: str = Query("xgb", description="Model key: dummy, lr, rf, xgb, lgb"),
    strategy: str = Query("cost", description="cost, max_f1, max_f2, precision_floor_90"),
    fn_cost: float = Query(DEFAULT_FN_COST, description="Cost of missed fraud (FN)"),
    fp_cost: float = Query(DEFAULT_FP_COST, description="Cost of false alarm (FP)"),
    split: str = Query("test", description="val or test")
):
    if not cached_artifacts["metrics"]:
        load_artifacts_from_disk()
    metrics = cached_artifacts["metrics"]
    if not metrics or "models" not in metrics:
        raise HTTPException(status_code=404, detail="Metrics not ready.")

    if model not in metrics["models"]:
        model = metrics.get("champion_key", "rf")

    split_key = "test" if split == "test" else "validation"
    model_eval = metrics["models"][model][split_key]

    # Recompute sweep with custom FN/FP costs dynamically
    raw_sweep = model_eval.get("threshold_sweep", [])
    recalculated_sweep = []
    total_samples = 42559  # standard test/val split size

    for pt in raw_sweep:
        tp, fp, fn, tn = pt["tp"], pt["fp"], pt["fn"], pt["tn"]
        exp_cost = float(fn * fn_cost + fp * fp_cost)
        recalculated_sweep.append({
            **pt,
            "expected_cost": round(exp_cost, 2),
            "cost_per_transaction": round(exp_cost / total_samples, 4)
        })

    optimal = find_optimal_thresholds(recalculated_sweep)

    # Strategy resolution
    strat_key_map = {
        "cost": "cost_optimal",
        "max_f1": "max_f1",
        "max_f2": "max_f2",
        "precision_floor_90": "precision_floor_90"
    }
    chosen_opt_key = strat_key_map.get(strategy, "cost_optimal")
    chosen_threshold_point = optimal.get(chosen_opt_key, recalculated_sweep[50])

    return {
        "model_key": model,
        "model_name": metrics["models"][model]["name"],
        "strategy": strategy,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "optimal_point": chosen_threshold_point,
        "all_optimal": optimal,
        "sweep_table": recalculated_sweep
    }


@app.get("/ablation")
def get_ablation():
    if not cached_artifacts["ablation_results"]:
        load_artifacts_from_disk()
    ablation = cached_artifacts["ablation_results"]
    if not ablation:
        raise HTTPException(status_code=404, detail="Ablation results artifact not found.")
    return ablation


@app.post("/predict")
def predict_single_transaction(txn: TransactionInput):
    if not cached_artifacts["model"]:
        load_artifacts_from_disk()
    model = cached_artifacts["model"]
    if not model:
        raise HTTPException(status_code=503, detail="Model is not trained yet.")

    data_dict = txn.dict(exclude={"threshold"})
    df_row = pd.DataFrame([data_dict])[FEATURE_COLS]

    # Check capping behavior for the audit trail
    capper_step = model.named_steps.get("capper")
    capping_audit = {}
    if capper_step and hasattr(capper_step, "bounds_"):
        for col in ["Amount", "Time"]:
            if col in capper_step.bounds_:
                val = float(df_row[col].iloc[0])
                lb = capper_step.bounds_[col]["lower"]
                ub = capper_step.bounds_[col]["upper"]
                is_clipped = bool(val < lb or val > ub)
                capped_val = float(np.clip(val, lb, ub))
                capping_audit[col] = {
                    "original_value": val,
                    "lower_bound": round(lb, 2) if not np.isneginf(lb) else None,
                    "upper_bound": round(ub, 2) if not np.isposinf(ub) else None,
                    "was_clipped": is_clipped,
                    "capped_value": round(capped_val, 2)
                }

    # Model inference
    proba = float(model.predict_proba(df_row)[0, 1])

    # Determine threshold
    default_th = 0.5
    if cached_artifacts["metrics"] and "champion_key" in cached_artifacts["metrics"]:
        champ_key = cached_artifacts["metrics"]["champion_key"]
        default_th = cached_artifacts["metrics"]["models"][champ_key].get("optimal_threshold_val", 0.5)

    threshold = txn.threshold if txn.threshold is not None else default_th
    is_fraud = int(proba >= threshold)

    # Risk categorical level
    if proba < 0.15:
        risk_level = "LOW"
    elif proba < 0.50:
        risk_level = "MEDIUM"
    elif proba < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "fraud_probability": round(proba, 5),
        "is_fraud": is_fraud,
        "prediction_label": "FRAUD" if is_fraud == 1 else "LEGITIMATE",
        "threshold_applied": round(threshold, 4),
        "risk_level": risk_level,
        "capping_audit": capping_audit,
    }


@app.post("/predict/batch")
async def predict_batch_transactions(file: UploadFile = File(...)):
    if not cached_artifacts["model"]:
        load_artifacts_from_disk()
    model = cached_artifacts["model"]
    if not model:
        raise HTTPException(status_code=503, detail="Model is not trained yet.")

    content = await file.read()
    try:
        df_batch = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    missing_cols = set(FEATURE_COLS) - set(df_batch.columns)
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded CSV is missing columns: {sorted(list(missing_cols))[:5]}..."
        )

    # Limit batch to at most 10,000 for responsive API performance
    if len(df_batch) > 10000:
        df_batch = df_batch.iloc[:10000]

    probabilities = model.predict_proba(df_batch[FEATURE_COLS])[:, 1]
    
    threshold = 0.5
    if cached_artifacts["metrics"] and "champion_key" in cached_artifacts["metrics"]:
        champ_key = cached_artifacts["metrics"]["champion_key"]
        threshold = cached_artifacts["metrics"]["models"][champ_key].get("optimal_threshold_val", 0.5)

    predictions = (probabilities >= threshold).astype(int)

    # Compile preview of scored rows
    scored_rows = []
    # Return top 50 highest risk rows and general sample
    df_batch["fraud_prob"] = np.round(probabilities, 4)
    df_batch["predicted_fraud"] = predictions

    # Sort by risk descending
    sorted_df = df_batch.sort_values(by="fraud_prob", ascending=False)
    sample_preview = sorted_df.head(50).to_dict(orient="records")

    total = len(df_batch)
    fraud_count = int(np.sum(predictions))

    return {
        "filename": file.filename,
        "total_rows_evaluated": total,
        "predicted_fraud_count": fraud_count,
        "predicted_fraud_rate": round(float((fraud_count / total) * 100), 3) if total > 0 else 0.0,
        "threshold_used": round(threshold, 4),
        "top_flagged_transactions": sample_preview,
    }

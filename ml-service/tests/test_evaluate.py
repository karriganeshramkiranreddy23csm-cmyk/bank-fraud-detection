import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluate import (
    compute_pr_curve_data,
    compute_roc_curve_data,
    compute_threshold_sweep,
    find_optimal_thresholds,
    evaluate_predictions
)


def test_average_precision_matches_sklearn():
    np.random.seed(42)
    # 1000 samples, 20 positives (imbalanced)
    y_true = np.zeros(1000, dtype=int)
    y_true[:20] = 1
    # Generate scores
    y_prob = np.random.uniform(0.0, 0.4, 1000)
    y_prob[:20] += np.random.uniform(0.3, 0.6, 20)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    expected_ap = average_precision_score(y_true, y_prob)
    pr_data = compute_pr_curve_data(y_true, y_prob, max_points=100)

    assert pr_data["pr_auc"] == pytest.approx(round(expected_ap, 4), abs=1e-4)
    assert len(pr_data["points"]) <= 100
    assert pr_data["baseline_no_skill"] == pytest.approx(20 / 1000, abs=1e-5)


def test_roc_auc_matches_sklearn():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.4, 0.75, 0.3, 0.2])

    expected_roc = roc_auc_score(y_true, y_prob)
    roc_data = compute_roc_curve_data(y_true, y_prob, max_points=50)

    assert roc_data["roc_auc"] == pytest.approx(round(expected_roc, 4), abs=1e-4)
    assert len(roc_data["points"]) <= 50


def test_threshold_sweep_cost_and_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.8])
    # Total samples = 4, Total Positives = 2, Total Negatives = 2
    # At threshold 0.5:
    # Pred: [1, 0, 0, 1]
    # TP: y=1, pred=1 -> sample 0 (1)
    # FN: y=1, pred=0 -> sample 1 (1)
    # FP: y=0, pred=1 -> sample 3 (1)
    # TN: y=0, pred=0 -> sample 2 (1)
    # FN_cost = 100, FP_cost = 10 -> Expected cost = 1*100 + 1*10 = 110
    sweep = compute_threshold_sweep(y_true, y_prob, fn_cost=100.0, fp_cost=10.0, steps=11)
    pt_half = next(p for p in sweep if abs(p["threshold"] - 0.5) < 1e-4)

    assert pt_half["tp"] == 1
    assert pt_half["fn"] == 1
    assert pt_half["fp"] == 1
    assert pt_half["tn"] == 1
    assert pt_half["expected_cost"] == 110.0
    assert pt_half["precision"] == 0.5
    assert pt_half["recall"] == 0.5


def test_optimal_thresholds_selection():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.05, 0.01, 0.01])
    sweep = compute_threshold_sweep(y_true, y_prob, fn_cost=100.0, fp_cost=5.0)
    optimal = find_optimal_thresholds(sweep)

    assert "max_f1" in optimal
    assert "max_f2" in optimal
    assert "precision_floor_90" in optimal
    assert "cost_optimal" in optimal
    assert optimal["max_f1"]["f1"] > 0

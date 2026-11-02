import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def downsample_curve(x_vals, y_vals, thresholds=None, max_points=200):
    """
    Downsamples curve points down to at most max_points, preserving endpoints.
    """
    n = len(x_vals)
    if n <= max_points:
        indices = np.arange(n)
    else:
        # Uniformly spaced indices including 0 and n-1
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        indices = np.unique(indices)

    points = []
    has_thresholds = thresholds is not None and len(thresholds) > 0

    for idx in indices:
        pt = {
            "x": round(float(x_vals[idx]), 4),
            "y": round(float(y_vals[idx]), 4),
        }
        if has_thresholds:
            # Note: precision_recall_curve thresholds has length n - 1
            th_idx = min(idx, len(thresholds) - 1)
            pt["threshold"] = round(float(thresholds[th_idx]), 4)
        points.append(pt)

    return points


def compute_pr_curve_data(y_true, y_prob, max_points=200):
    """
    Computes Precision-Recall curve and AP (PR-AUC).
    Returns downsampled points and baseline fraud prevalence.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    ap = float(average_precision_score(y_true_arr, y_prob_arr))
    baseline = float(np.mean(y_true_arr))

    # Sklearn precision_recall_curve returns precision, recall, thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    # Downsample points for efficient API payloads
    curve_points = []
    n = len(recalls)
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        indices = np.unique(indices)

    for idx in indices:
        th_val = float(thresholds[min(idx, len(thresholds) - 1)]) if len(thresholds) > 0 else 0.5
        curve_points.append({
            "recall": round(float(recalls[idx]), 4),
            "precision": round(float(precisions[idx]), 4),
            "threshold": round(th_val, 4)
        })

    return {
        "pr_auc": round(ap, 4),
        "baseline_no_skill": round(baseline, 6),
        "points": curve_points
    }


def compute_roc_curve_data(y_true, y_prob, max_points=200):
    """
    Computes ROC curve and ROC-AUC.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        fprs, tprs, thresholds = roc_curve(y_true_arr, y_prob_arr)
    except Exception:
        return {"roc_auc": 0.5, "points": []}

    curve_points = []
    n = len(fprs)
    if n <= max_points:
        indices = np.arange(n)
    else:
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        indices = np.unique(indices)

    for idx in indices:
        curve_points.append({
            "fpr": round(float(fprs[idx]), 4),
            "tpr": round(float(tprs[idx]), 4),
            "threshold": round(float(thresholds[idx]), 4)
        })

    return {
        "roc_auc": round(roc_auc, 4),
        "points": curve_points
    }


def compute_threshold_sweep(y_true, y_prob, fn_cost=100.0, fp_cost=5.0, steps=101):
    """
    Sweeps thresholds from 0.00 to 1.00 calculating confusion matrix, metrics, and cost.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    total_samples = len(y_true_arr)
    total_positives = int(np.sum(y_true_arr))
    total_negatives = total_samples - total_positives

    thresholds = np.linspace(0.00, 1.00, steps)
    sweep = []

    for th in thresholds:
        th = round(float(th), 3)
        y_pred = (y_prob_arr >= th).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        rec = float(tp / total_positives) if total_positives > 0 else 0.0
        acc = float((tp + tn) / total_samples)

        if (prec + rec) > 0:
            f1 = float(2 * prec * rec / (prec + rec))
            f2 = float((1 + 4) * prec * rec / (4 * prec + rec))
        else:
            f1 = 0.0
            f2 = 0.0

        expected_cost = float(fn * fn_cost + fp * fp_cost)

        sweep.append({
            "threshold": th,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "accuracy": round(acc, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "expected_cost": round(expected_cost, 2),
            "cost_per_transaction": round(expected_cost / total_samples, 4)
        })

    return sweep


def find_optimal_thresholds(sweep_table):
    """
    Identifies optimal thresholds on validation data across multiple strategies:
    - max_f1
    - max_f2 (prioritizes recall)
    - precision_floor_90 (highest recall with precision >= 0.90)
    - cost_optimal (minimizes FN_cost*FN + FP_cost*FP)
    """
    if not sweep_table:
        return {}

    best_f1_entry = max(sweep_table, key=lambda x: x["f1"])
    best_f2_entry = max(sweep_table, key=lambda x: x["f2"])
    min_cost_entry = min(sweep_table, key=lambda x: x["expected_cost"])

    # Precision floor: precision >= 0.90, maximize recall
    eligible_prec_90 = [x for x in sweep_table if x["precision"] >= 0.90 and x["recall"] > 0]
    if eligible_prec_90:
        prec_90_entry = max(eligible_prec_90, key=lambda x: x["recall"])
    else:
        # Fallback to the highest precision entry with non-zero recall
        with_rec = [x for x in sweep_table if x["recall"] > 0]
        prec_90_entry = max(with_rec, key=lambda x: x["precision"]) if with_rec else sweep_table[-1]

    return {
        "max_f1": best_f1_entry,
        "max_f2": best_f2_entry,
        "precision_floor_90": prec_90_entry,
        "cost_optimal": min_cost_entry,
    }


def compute_targets_at_operating_points(y_true, y_prob):
    """
    Computes precision at target recall (80%, 90%) and recall at target precision (90%).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)

    # Precision at Recall >= 0.80: find highest precision where recall >= 0.80
    mask_rec_80 = recalls >= 0.80
    prec_at_rec_80 = float(np.max(precisions[mask_rec_80])) if np.any(mask_rec_80) else 0.0

    # Precision at Recall >= 0.90: find highest precision where recall >= 0.90
    mask_rec_90 = recalls >= 0.90
    prec_at_rec_90 = float(np.max(precisions[mask_rec_90])) if np.any(mask_rec_90) else 0.0

    # Recall at Precision >= 0.90: find highest recall where precision >= 0.90
    mask_prec_90 = precisions >= 0.90
    rec_at_prec_90 = float(np.max(recalls[mask_prec_90])) if np.any(mask_prec_90) else 0.0

    return {
        "precision_at_recall_80": round(prec_at_rec_80, 4),
        "precision_at_recall_90": round(prec_at_rec_90, 4),
        "recall_at_precision_90": round(rec_at_prec_90, 4),
    }


def evaluate_predictions(y_true, y_prob, chosen_threshold=0.5, fn_cost=100.0, fp_cost=5.0):
    """
    Full comprehensive evaluation package for a given model predictions.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob_arr >= chosen_threshold).astype(int)

    pr_curve = compute_pr_curve_data(y_true_arr, y_prob_arr)
    roc_curve_data = compute_roc_curve_data(y_true_arr, y_prob_arr)
    sweep = compute_threshold_sweep(y_true_arr, y_prob_arr, fn_cost=fn_cost, fp_cost=fp_cost)
    optimal_thresholds = find_optimal_thresholds(sweep)
    targets = compute_targets_at_operating_points(y_true_arr, y_prob_arr)

    cm = confusion_matrix(y_true_arr, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true_arr, y_pred, beta=2, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))
    expected_cost = float(fn * fn_cost + fp * fp_cost)

    return {
        "metrics_at_threshold": {
            "threshold": round(float(chosen_threshold), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "expected_cost": round(expected_cost, 2),
        },
        "summary": {
            "pr_auc": pr_curve["pr_auc"],
            "roc_auc": roc_curve_data["roc_auc"],
            "accuracy": round(acc, 4),
            "baseline_prevalence": pr_curve["baseline_no_skill"],
            "precision_at_recall_80": targets["precision_at_recall_80"],
            "precision_at_recall_90": targets["precision_at_recall_90"],
            "recall_at_precision_90": targets["recall_at_precision_90"],
        },
        "optimal_thresholds": optimal_thresholds,
        "pr_curve": pr_curve,
        "roc_curve": roc_curve_data,
        "threshold_sweep": sweep
    }

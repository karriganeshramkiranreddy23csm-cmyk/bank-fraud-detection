import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["artifacts_ready"] is True


def test_dataset_summary():
    response = client.get("/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert "splits" in data
    assert "amount_statistics" in data
    assert data["splits"]["train"]["total"] > 100000


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "dummy" in data["models"]
    assert "rf" in data["models"]
    assert "champion_name" in data


def test_pr_curve():
    response = client.get("/pr-curve?model=rf&split=test")
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "rf"
    assert data["pr_auc"] > 0.70
    assert len(data["points"]) > 10


def test_thresholds():
    response = client.get("/thresholds?model=rf&strategy=cost&fn_cost=100&fp_cost=5")
    assert response.status_code == 200
    data = response.json()
    assert "optimal_point" in data
    assert "sweep_table" in data
    assert len(data["sweep_table"]) > 50


def test_ablation():
    response = client.get("/ablation")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "best_configuration" in data
    assert len(data["results"]) >= 10


def test_single_predict_and_capping():
    # Construct a sample legit transaction
    sample = {
        "Time": 406.0,
        "Amount": 15000.0,  # Extreme outlier to trigger capping
    }
    for i in range(1, 29):
        sample[f"V{i}"] = 0.0

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "prediction_label" in data
    assert "capping_audit" in data
    assert "Amount" in data["capping_audit"]
    # 15000 is higher than 99th percentile (~$1000), so it must be flagged as clipped!
    assert data["capping_audit"]["Amount"]["was_clipped"] is True
    assert data["capping_audit"]["Amount"]["capped_value"] < 15000.0

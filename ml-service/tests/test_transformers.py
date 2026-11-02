import numpy as np
import pandas as pd
import pytest
from src.transformers import OutlierCapper, Log1pTransformer


def test_percentile_capping_no_leakage():
    # Synthetic train and test sets with deliberate extreme values
    train_data = pd.DataFrame({"Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]})
    test_data = pd.DataFrame({"Amount": [5.0, 25.0, 50.0, 200.0, 500.0]})

    # Fit ONLY on train
    capper = OutlierCapper(strategy="percentile", lower_percentile=10.0, upper_percentile=90.0, columns=["Amount"])
    capper.fit(train_data)

    # Train 10th percentile = 19.0, 90th percentile = 91.0
    bounds = capper.bounds_["Amount"]
    assert bounds["lower"] == pytest.approx(19.0)
    assert bounds["upper"] == pytest.approx(91.0)

    # Transform test set using TRAIN bounds
    transformed_test = capper.transform(test_data)
    amounts = transformed_test["Amount"].tolist()

    # 5.0 -> capped to 19.0
    # 25.0 -> 25.0
    # 50.0 -> 50.0
    # 200.0 -> capped to 91.0
    # 500.0 -> capped to 91.0
    assert amounts[0] == pytest.approx(19.0)
    assert amounts[1] == pytest.approx(25.0)
    assert amounts[2] == pytest.approx(50.0)
    assert amounts[3] == pytest.approx(91.0)
    assert amounts[4] == pytest.approx(91.0)

    # Check reported stats on test
    stats = capper.last_clipped_stats_["Amount"]
    assert stats["lower_clipped_count"] == 1
    assert stats["upper_clipped_count"] == 2
    assert stats["total_clipped_count"] == 3
    assert stats["clipped_percentage"] == 60.0


def test_iqr_capping():
    train_data = pd.DataFrame({"Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]})
    # Q1 = 25th percentile = 27.5, Q3 = 75th percentile = 62.5, IQR = 35.0
    # Lower bound = 27.5 - 1.5 * 35.0 = -25.0
    # Upper bound = 62.5 + 1.5 * 35.0 = 115.0
    capper = OutlierCapper(strategy="iqr", iqr_multiplier=1.5, columns=["Amount"])
    capper.fit(train_data)

    bounds = capper.bounds_["Amount"]
    assert bounds["lower"] == pytest.approx(-25.0)
    assert bounds["upper"] == pytest.approx(115.0)

    test_data = pd.DataFrame({"Amount": [-50.0, 30.0, 150.0]})
    transformed = capper.transform(test_data)
    assert transformed["Amount"].iloc[0] == pytest.approx(-25.0)
    assert transformed["Amount"].iloc[1] == pytest.approx(30.0)
    assert transformed["Amount"].iloc[2] == pytest.approx(115.0)


def test_none_strategy_passthrough():
    data = pd.DataFrame({"Amount": [-1000.0, 50.0, 100000.0]})
    capper = OutlierCapper(strategy="none", columns=["Amount"])
    capper.fit(data)
    transformed = capper.transform(data)
    assert transformed["Amount"].tolist() == [-1000.0, 50.0, 100000.0]
    assert capper.last_clipped_stats_["Amount"]["total_clipped_count"] == 0


def test_log1p_transformer():
    data = pd.DataFrame({"Amount": [0.0, 10.0, 100.0]})
    transformer = Log1pTransformer(columns=["Amount"])
    res = transformer.transform(data)
    assert res["Amount"].iloc[0] == pytest.approx(0.0)
    assert res["Amount"].iloc[1] == pytest.approx(np.log1p(10.0))
    assert res["Amount"].iloc[2] == pytest.approx(np.log1p(100.0))

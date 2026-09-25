"""MSE, MAPE and flux relative L2 summaries added to src.metrics."""

import math

import numpy as np
import pytest

from src.metrics import eigenvalue_report, flux_report, regression_report


def test_mse_is_rmse_squared():
    rep = regression_report([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert rep["mse"] == pytest.approx(4.0 / 3.0)
    assert rep["rmse"] == pytest.approx(math.sqrt(rep["mse"]))


def test_mape_skips_near_zero_truth():
    # entries below 1e-3 of the largest |truth| (here 4) are left out: only 2 and 4 count, with errors 50% and 0%
    rep = regression_report([0.0, 1e-9, 2.0, 4.0], [5.0, 5.0, 3.0, 4.0])
    assert rep["mape"] == pytest.approx(25.0)


def test_mape_is_nan_when_every_truth_is_zero():
    rep = regression_report([0.0, 0.0], [1.0, 2.0])
    assert math.isnan(rep["mape"])


def test_perfect_prediction_has_zero_mse_and_mape():
    y = np.array([0.9, 1.0, 1.1])
    rep = eigenvalue_report(y, y)
    assert rep["mse"] == 0.0 and rep["mape"] == 0.0


def test_flux_relative_l2_mean_and_p99():
    truth = np.ones((100, 8))
    pred = truth.copy()
    pred[:, 0] += np.linspace(0.0, 1.0, 100)  # per-profile relative L2 = shift / sqrt(8)
    rep = flux_report(truth, pred)
    rel = np.linspace(0.0, 1.0, 100) / math.sqrt(8)
    assert rep["rel_l2_mean"] == pytest.approx(rel.mean())
    assert rep["rel_l2_p99"] == pytest.approx(np.quantile(rel, 0.99))
    assert rep["rel_l2_median"] <= rep["rel_l2_p99"] <= rep["rel_l2_max"]

"""Solver physics invariants and metric definitions. numpy only, no training."""

import numpy as np
import pytest

from src.metrics import eigenvalue_report, flux_report, regression_report
from src.sampling import latin_hypercube
from src.verify import SOLVER_TOLERANCE, check_solver, mirror_positions


def test_solver_satisfies_every_invariant():
    res = check_solver(latin_hypercube(12, seed=5, n_knots=5))
    for key, tol in SOLVER_TOLERANCE.items():
        assert res["worst"][key] <= tol, f"{key}: {res['worst'][key]:.2e} > {tol:.0e}"
    assert not any(res["violations"].values())
    assert res["passed"]


def test_mirror_is_an_involution():
    P = latin_hypercube(5, seed=1, n_knots=5)
    assert np.array_equal(mirror_positions(mirror_positions(P)), P)
    assert not np.array_equal(mirror_positions(P), P)


def test_perfect_prediction_metrics():
    y = np.linspace(0.9, 1.1, 50)
    r = regression_report(y, y)
    assert r["r2"] == pytest.approx(1.0) and r["rmse"] == 0.0 and r["slope"] == pytest.approx(1.0)
    assert r["intercept"] == pytest.approx(0.0, abs=1e-12) and r["bias"] == 0.0


def test_known_metric_values():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    r = regression_report(y, y + 0.5)
    assert r["rmse"] == pytest.approx(0.5) and r["mae"] == pytest.approx(0.5) and r["bias"] == pytest.approx(0.5)
    assert r["r2"] == pytest.approx(1 - 4 * 0.25 / 5.0)  # ss_res 1.0, ss_tot 5.0
    e = eigenvalue_report(np.array([1.0, 1.0]), np.array([1.001, 0.999]))
    assert e["rmse_pcm"] == pytest.approx(100.0) and e["max_abs_pcm"] == pytest.approx(100.0)


def test_flux_report_flags_negative_and_shifted_peaks():
    x = np.linspace(0, np.pi, 40)
    truth = np.sin(x)[None, :].repeat(3, axis=0)
    pred = truth.copy()
    pred[1, 5] = -0.1
    pred[2] = np.roll(truth[2], 3)
    rep = flux_report(truth, pred)
    assert rep["negative_fraction"] == pytest.approx(1 / 120)
    assert rep["peak_cell_err_p95"] >= 1
    assert flux_report(truth, truth)["profile_r2_min"] == pytest.approx(1.0)

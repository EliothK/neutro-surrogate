"""Cross-validation with hybrid Ritz metrics, and a one-trial smoke run of the tuner with --objective ritz. Outputs go to a temporary directory."""

import json
import math
import sys

import pytest

from src import tune
from src.crossval import DEFAULT_CONFIG, METRICS, RITZ_METRICS, cross_validate, refit_and_test

TINY_CFG = {**DEFAULT_CONFIG, "epochs": 2, "batch_size": 32, "width": 32, "depth": 2}


def test_cross_validate_adds_ritz_metrics_only_when_asked(tiny_dataset):
    plain = cross_validate(tiny_dataset, TINY_CFG, folds=2, device="cpu")
    assert set(plain["mean"]) == set(METRICS)
    with_ritz = cross_validate(tiny_dataset, TINY_CFG, folds=2, device="cpu", ritz_m=17)
    assert set(with_ritz["mean"]) == set(METRICS) | set(RITZ_METRICS)
    assert all(math.isfinite(with_ritz["mean"][m]) and with_ritz["mean"][m] >= 0 for m in RITZ_METRICS)
    assert len(with_ritz["folds"]) == 2


def test_refit_and_test_reports_ritz_metrics(tiny_dataset):
    test = refit_and_test(tiny_dataset, TINY_CFG, device="cpu", ritz_m=17)
    assert set(RITZ_METRICS) <= set(test)


def test_tuner_runs_with_ritz_objective(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tune, "FIGURE_DIR", tmp_path)
    monkeypatch.setattr(tune, "DEFAULT_CONFIG", TINY_CFG)
    monkeypatch.setattr(tune, "suggest_config", lambda trial, physics=False, flux_options=False: {
        **TINY_CFG, "flux_loss": trial.suggest_categorical("flux_loss", ["mse", "blend"]),
        "zone_head": trial.suggest_categorical("zone_head", [False, True])})
    monkeypatch.setattr(sys, "argv", ["tune", "--data", tiny_dataset, "--trials", "1", "--folds", "2", "--objective", "ritz"])
    tune.main()
    out = json.loads((tmp_path / "tune_tiny_ritz.json").read_text())
    assert out["objective"] == "ritz" and out["ritz"] == 17
    assert set(RITZ_METRICS) <= set(out["default_cv"])
    assert set(RITZ_METRICS) <= set(out["best"]["test"])
    assert (tmp_path / "optuna_tiny_ritz.db").exists()

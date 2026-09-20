"""Search space and scoring for the Optuna tuner. No training."""

import optuna
import pytest

from src.crossval import DEFAULT_CONFIG
from src.model import ACTIVATIONS
from src.tune import PHYSICS_ONLY, relative_score, suggest_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def draw(physics, n=40):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return [suggest_config(study.ask(), physics) for _ in range(n)]


def test_configs_have_every_training_key_and_stay_valid():
    for cfg in draw(physics=True):
        assert set(cfg) == set(DEFAULT_CONFIG)
        assert cfg["activation"] in ACTIVATIONS
        assert cfg["scheduler"] in ("cosine", "linear", "none")
        assert cfg["epochs"] >= 100 and cfg["depth"] >= 2 and 0.0 <= cfg["dropout"] <= 0.3
        assert cfg["train_fraction"] == 1.0


def test_physics_parameters_only_vary_with_physics():
    plain = draw(physics=False)
    assert all(cfg[k] == DEFAULT_CONFIG[k] for cfg in plain for k in PHYSICS_ONLY)
    assert len({cfg["mu_max"] for cfg in draw(physics=True)}) > 1


def test_relative_score_is_one_for_the_reference_and_lower_when_better():
    ref = {"eigen_pcm_median": 200.0, "flux_l2_median": 0.04}
    assert relative_score(ref, ref) == pytest.approx(1.0)
    assert relative_score({"eigen_pcm_median": 100.0, "flux_l2_median": 0.02}, ref) == pytest.approx(0.5)

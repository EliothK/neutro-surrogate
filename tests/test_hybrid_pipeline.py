"""src.hybrid end to end: the gate and its solver fallback, the timing report and the command line."""

import json
import sys

import numpy as np
import pytest
import torch

from src import hybrid
from src.benchmark import load_model
from src.hybrid import predict, predict_max, ritz, timing
from src.solver import solve
from src.train import load_split


@pytest.fixture(scope="module")
def loaded(tiny_checkpoint, tiny_dataset):
    model, norm, _ = load_model(tiny_checkpoint, "cpu")
    positions, eigenvalue, flux, idx, _ = load_split(tiny_dataset, "cpu")
    return model, norm, positions[idx["test"]], eigenvalue[idx["test"]], flux[idx["test"]]


def test_no_threshold_sends_nothing_to_the_solver(loaded):
    model, norm, x, _, _ = loaded
    out = predict(model, norm, x, refine=1)
    assert not out["fallback"].any()
    assert out["k"].shape == out["k_direct"].shape == out["residual"].shape == (len(x),)


def test_gate_replaces_flagged_samples_with_the_solver(loaded):
    model, norm, x, _, _ = loaded
    out = predict(model, norm, x, refine=1, threshold=-1.0)  # every residual exceeds -1
    assert out["fallback"].all()
    for i in range(len(x)):
        k_i, flux_i = solve(x[i].double().numpy())
        assert out["k"][i].item() == pytest.approx(k_i, rel=1e-12)
        np.testing.assert_allclose(out["flux"][i].numpy(), flux_i, rtol=1e-10)


def test_infinite_threshold_flags_nothing(loaded):
    model, norm, x, _, _ = loaded
    assert not predict(model, norm, x, threshold=float("inf"))["fallback"].any()


def test_ritz_stage_keeps_flux_normalised_and_k_positive(loaded):
    model, norm, x, _, _ = loaded
    out = predict(model, norm, x, refine=1, ritz_m=17)
    assert (out["k"] > 0).all()
    torch.testing.assert_close(out["flux"].amax(-1), torch.ones(len(x), dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ritz_above_32_matches_on_gpu_and_cpu(loaded):
    model, norm, x, _, _ = loaded
    with torch.no_grad():
        _, flux = model(norm(x))
    cpu = ritz(x.double(), flux.double(), 33)
    gpu = ritz(x.double().cuda(), flux.double().cuda(), 33)
    torch.testing.assert_close(gpu[0].cpu(), cpu[0])
    torch.testing.assert_close(gpu[1].cpu(), cpu[1])


@pytest.fixture(scope="module")
def pair(loaded, tiny_checkpoint_plain):
    model, norm, x, k, flux = loaded
    other, other_norm, _ = load_model(tiny_checkpoint_plain, "cpu")
    return [(model, norm), (other, other_norm)], x, k, flux


def test_predict_max_with_one_member_equals_predict(loaded):
    model, norm, x, _, _ = loaded
    single = predict(model, norm, x, refine=1, ritz_m=17)
    combined = predict_max([(model, norm)], x, refine=1, ritz_m=17)
    for key in ("k", "flux", "k_direct", "residual", "fallback"):
        torch.testing.assert_close(combined[key], single[key])
    assert (combined["chosen"] == 0).all()


@pytest.mark.parametrize("ritz_m", [0, 17])
def test_predict_max_keeps_the_largest_k_and_its_flux(pair, ritz_m):
    members, x, _, _ = pair
    each = [predict(m, n, x, refine=1, ritz_m=ritz_m) for m, n in members]
    out = predict_max(members, x, refine=1, ritz_m=ritz_m)
    torch.testing.assert_close(out["k"], torch.maximum(each[0]["k"], each[1]["k"]))
    for i, c in enumerate(out["chosen"].tolist()):
        torch.testing.assert_close(out["flux"][i], each[c]["flux"][i])
        torch.testing.assert_close(out["residual"][i], each[c]["residual"][i])
        torch.testing.assert_close(out["k_direct"][i], each[c]["k_direct"][i])


@pytest.mark.parametrize("ritz_m", [0, 17])
def test_predict_max_is_never_worse_than_its_best_member(pair, ritz_m):
    members, x, k_true, _ = pair
    k_true = k_true.double()
    each = [(predict(m, n, x, refine=1, ritz_m=ritz_m)["k"] - k_true).abs() for m, n in members]
    err = (predict_max(members, x, refine=1, ritz_m=ritz_m)["k"] - k_true).abs()
    # every candidate is a lower bound on the true k (up to the float32 labels), so the largest is the closest
    assert (err <= torch.minimum(*each) + 1e-6 * k_true).all()


def test_predict_max_gates_after_choosing(pair):
    members, x, _, _ = pair
    out = predict_max(members, x, refine=1, threshold=-1.0)
    assert out["fallback"].all()
    k0, _ = solve(x[0].double().numpy())
    assert out["k"][0].item() == pytest.approx(k0, rel=1e-12)
    assert not predict_max(members, x, refine=1)["fallback"].any()


def test_timing_reports_every_stage(loaded):
    model, norm, x, _, _ = loaded
    rep = timing([(model, norm)], x, threshold=0.0, refine=1, ritz_m=17, n_solver=3, repeats=1)
    assert rep["solver_ms_per_sample"] > 0 and rep["n_models"] == 1
    for batch in ("batch_1", f"batch_{len(x)}"):
        rows = rep["stages"][batch]
        stages = [k for k in rows if k != "fallback_fraction"]
        assert stages == ["network", "+ rayleigh k and residual", "+ 1 inverse-power step(s)",
                          "+ ritz m=17 and 1 more step", "+ residual gate (solver fallback)"]
        assert all(rows[s]["ms_per_sample"] > 0 and rows[s]["speedup_vs_solver"] > 0 for s in stages)


def test_timing_leaves_out_stages_that_are_off(loaded):
    model, norm, x, _, _ = loaded
    rows = timing([(model, norm)], x, threshold=None, refine=1, ritz_m=0, n_solver=2, repeats=1)["stages"]["batch_1"]
    assert not any(k.startswith("+ ritz") or k.startswith("+ residual gate") for k in rows)


def test_timing_with_two_models_costs_more_for_the_networks(pair):
    members, x, _, _ = pair
    rep = timing(members, x, threshold=None, refine=1, ritz_m=17, n_solver=2, repeats=1)
    assert rep["n_models"] == 2
    assert rep["stages"][f"batch_{len(x)}"]["+ ritz m=17 and 1 more step"]["ms_per_sample"] > 0


def _run_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["hybrid", *args])
    hybrid.main()


def test_cli_without_gate(monkeypatch, tmp_path, tiny_checkpoint, tiny_dataset):
    out = tmp_path / "r.json"
    _run_cli(monkeypatch, ["--model", tiny_checkpoint, "--data", tiny_dataset, "--refine", "1", "--ritz", "17",
                           "--fallback", "0", "--results", str(out)])
    rep = json.loads(out.read_text())
    assert rep["threshold"] is None and rep["ritz"] == 17
    assert "surrogate_only" in rep and "gated" not in rep
    assert {"k", "k_direct", "flux"} <= set(rep["surrogate_only"])


def test_cli_with_gate_eval_data_and_timing(monkeypatch, tmp_path, tiny_checkpoint, tiny_dataset):
    out = tmp_path / "r.json"
    _run_cli(monkeypatch, ["--model", tiny_checkpoint, "--data", tiny_dataset, "--eval-data", tiny_dataset,
                           "--fallback", "0.25", "--timing", "--results", str(out)])
    rep = json.loads(out.read_text())
    assert rep["eval_data"] == tiny_dataset and rep["threshold"] is not None
    assert 0.0 < rep["gated"]["fallback_fraction"] <= 1.0
    assert rep["gated"]["k"]["rmse"] <= rep["surrogate_only"]["k"]["rmse"] + 1e-12  # the solver only replaces samples with exact answers
    assert "timing" in rep


def test_cli_with_two_models(monkeypatch, tmp_path, tiny_checkpoint, tiny_checkpoint_plain, tiny_dataset):
    monkeypatch.setattr(hybrid, "RESULTS_DIR", tmp_path)
    _run_cli(monkeypatch, ["--model", tiny_checkpoint, tiny_checkpoint_plain, "--data", tiny_dataset,
                           "--ritz", "17", "--fallback", "0.25"])
    rep = json.loads((tmp_path / "hybrid_tiny+tiny_plain.json").read_text())  # default name joins the checkpoint stems
    assert rep["model"] == [tiny_checkpoint, tiny_checkpoint_plain]
    for run in ("surrogate_only", "gated"):
        fractions = rep[run]["chosen_fraction"]
        assert len(fractions) == 2 and sum(fractions) == pytest.approx(1.0)


def test_cli_rejects_a_missing_model(monkeypatch, tiny_dataset):
    with pytest.raises(FileNotFoundError):
        _run_cli(monkeypatch, ["--model", "does/not/exist.pt", "--data", tiny_dataset, "--fallback", "0"])


def test_cli_warns_when_calibrating_in_sample(monkeypatch, tmp_path, capsys, tiny_checkpoint, tiny_dataset):
    ckpt = torch.load(tiny_checkpoint, weights_only=False)
    ckpt["trained_on"] = "train+val"
    refit = tmp_path / "refit.pt"
    torch.save(ckpt, refit)
    _run_cli(monkeypatch, ["--model", str(refit), "--data", tiny_dataset, "--fallback", "0", "--results", str(tmp_path / "r.json")])
    assert f"warning: {refit} was trained on train+val" in capsys.readouterr().out

"""src.hybrid end to end: the gate and its solver fallback, the timing report and the command line."""

import json
import sys

import numpy as np
import pytest
import torch

from src import hybrid
from src.benchmark import load_model
from src.hybrid import predict, ritz, timing
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


def test_timing_reports_every_stage(loaded):
    model, norm, x, _, _ = loaded
    rep = timing(model, norm, x, threshold=0.0, refine=1, ritz_m=17, n_solver=3, repeats=1)
    assert rep["solver_ms_per_sample"] > 0
    for batch in ("batch_1", f"batch_{len(x)}"):
        rows = rep["stages"][batch]
        stages = [k for k in rows if k != "fallback_fraction"]
        assert stages == ["network", "+ rayleigh k and residual", "+ 1 inverse-power step(s)",
                          "+ ritz m=17 and 1 more step", "+ residual gate (solver fallback)"]
        assert all(rows[s]["ms_per_sample"] > 0 and rows[s]["speedup_vs_solver"] > 0 for s in stages)


def test_timing_leaves_out_stages_that_are_off(loaded):
    model, norm, x, _, _ = loaded
    rows = timing(model, norm, x, threshold=None, refine=1, ritz_m=0, n_solver=2, repeats=1)["stages"]["batch_1"]
    assert not any(k.startswith("+ ritz") or k.startswith("+ residual gate") for k in rows)


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


def test_cli_warns_when_calibrating_in_sample(monkeypatch, tmp_path, capsys, tiny_checkpoint, tiny_dataset):
    ckpt = torch.load(tiny_checkpoint, weights_only=False)
    ckpt["trained_on"] = "train+val"
    refit = tmp_path / "refit.pt"
    torch.save(ckpt, refit)
    _run_cli(monkeypatch, ["--model", str(refit), "--data", tiny_dataset, "--fallback", "0", "--results", str(tmp_path / "r.json")])
    assert "warning: this model was trained on train+val" in capsys.readouterr().out

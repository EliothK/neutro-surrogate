"""train() options: returned metrics, best-checkpoint restore, early stopping, flux losses and the zone head."""

import math

import pytest
import torch

from src.benchmark import load_model
from src.model import SurrogateMLP
from src.train import evaluate, load_split, train

TINY = {"epochs": 4, "batch_size": 32, "width": 32, "depth": 2}


def _val_metrics(data, model, norm):
    positions, eigenvalue, flux, idx, _ = load_split(data, "cpu")
    va = idx["val"]
    return evaluate(model, norm, positions[va], eigenvalue[va], flux[va])


@pytest.mark.parametrize("keep_best", [False, True])
def test_returned_metrics_belong_to_returned_weights(tiny_dataset, keep_best):
    model, norm, _, metrics = train(tiny_dataset, device="cpu", verbose=False, keep_best=keep_best, **TINY)
    assert metrics is not None
    assert metrics == pytest.approx(_val_metrics(tiny_dataset, model, norm))


def test_keep_best_is_never_worse_than_last_epoch(tiny_dataset):
    _, _, _, last = train(tiny_dataset, device="cpu", verbose=False, seed=1, **TINY)
    _, _, _, best = train(tiny_dataset, device="cpu", verbose=False, seed=1, keep_best=True, **TINY)
    assert best["eigen_pcm_median"] <= last["eigen_pcm_median"] + 1e-6


def test_patience_stops_when_nothing_improves(tiny_dataset, capsys):
    # lr = 0 leaves the weights fixed, so the first epoch stays best and patience 2 stops at epoch 3
    train(tiny_dataset, device="cpu", verbose=True, keep_best=True, patience=2, lr=0.0, **{**TINY, "epochs": 20})
    out = capsys.readouterr().out
    assert "early stop at epoch 3, best epoch 1" in out


def test_patience_is_ignored_without_keep_best(tiny_dataset, capsys):
    train(tiny_dataset, device="cpu", verbose=True, patience=1, lr=0.0, **TINY)
    assert "early stop" not in capsys.readouterr().out


@pytest.mark.parametrize("flux_loss", ["mse", "shape", "blend"])
def test_every_flux_loss_trains(tiny_dataset, flux_loss):
    _, _, _, metrics = train(tiny_dataset, device="cpu", verbose=False, flux_loss=flux_loss, **TINY)
    assert all(math.isfinite(v) for v in metrics.values())


def test_unknown_flux_loss_is_rejected(tiny_dataset):
    with pytest.raises(KeyError):
        train(tiny_dataset, device="cpu", verbose=False, flux_loss="bogus", **TINY)


def test_zone_head_starts_as_the_plain_model():
    torch.manual_seed(0)
    zoned = SurrogateMLP(width=32, depth=2, zone_head=True)
    plain = SurrogateMLP(width=32, depth=2)
    plain.load_state_dict({k: v for k, v in zoned.state_dict().items() if not k.startswith("zone_head")})
    x = torch.randn(5, zoned.trunk[0].in_features)
    for a, b in zip(zoned(x), plain(x)):
        torch.testing.assert_close(a, b)


def test_zone_head_output_is_non_negative_and_peak_normalised():
    model = SurrogateMLP(width=32, depth=2, zone_head=True)
    torch.nn.init.normal_(model.zone_head.weight)  # move away from the identity start
    _, flux = model(torch.randn(6, model.trunk[0].in_features))
    assert (flux >= 0).all()
    torch.testing.assert_close(flux.amax(-1), torch.ones(6))


def test_zone_head_checkpoint_round_trip(tiny_checkpoint, tiny_dataset):
    model, norm, ckpt = load_model(tiny_checkpoint, "cpu")
    assert ckpt["zone_head"] and model.zone_head is not None
    positions, _, _, idx, _ = load_split(tiny_dataset, "cpu")
    with torch.no_grad():
        a = model(norm(positions[idx["test"]]))
        b = model(norm(positions[idx["test"]]))
    torch.testing.assert_close(a, b)
    # the flag matters: the zone-head weights do not fit a plain model
    with pytest.raises(RuntimeError):
        SurrogateMLP(width=32, depth=2).load_state_dict(ckpt["state_dict"])

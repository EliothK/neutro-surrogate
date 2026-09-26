"""Shared fixtures: a tiny solved dataset and a briefly trained checkpoint, so training, cross-validation and hybrid paths run in seconds."""

import numpy as np
import pytest
import torch

from src.make_dataset import generate
from src.sampling import make_splits

TINY_SPLITS = (0.6, 0.2, 0.2)
TINY_TRAIN = {"epochs": 3, "batch_size": 32, "width": 32, "depth": 2}


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """120 solved rescaled samples written in the same .npz layout as src.make_dataset."""
    positions, eigenvalue, flux, reflector = generate(120, seed=5, mode="rescaled", verbose=False)
    train_idx, val_idx, test_idx = make_splits(len(positions), TINY_SPLITS, 5)
    path = tmp_path_factory.mktemp("data") / "tiny.npz"
    np.savez(path, positions=positions, eigenvalue=eigenvalue, flux=flux, reflector=reflector,
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return str(path)


def _save_tiny(tiny_dataset, tmp_path_factory, name, zone_head, seed):
    from src.train import train

    model, norm, _, _ = train(tiny_dataset, device="cpu", verbose=False, zone_head=zone_head, seed=seed, **TINY_TRAIN)
    path = tmp_path_factory.mktemp("models") / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "normalizer": norm.state_dict(), "physics": False,
                "width": TINY_TRAIN["width"], "depth": TINY_TRAIN["depth"], "activation": "silu", "dropout": 0.0,
                "zone_head": zone_head, "trained_on": "train"}, path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_checkpoint(tiny_dataset, tmp_path_factory):
    """A zone-head model trained for a few epochs on tiny_dataset, saved the way src.train saves checkpoints."""
    return _save_tiny(tiny_dataset, tmp_path_factory, "tiny", zone_head=True, seed=0)


@pytest.fixture(scope="session")
def tiny_checkpoint_plain(tiny_dataset, tmp_path_factory):
    """A second, different model (no zone head, another seed) for the multi-model paths."""
    return _save_tiny(tiny_dataset, tmp_path_factory, "tiny_plain", zone_head=False, seed=1)

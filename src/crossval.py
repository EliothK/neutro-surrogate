"""K-fold cross-validation over the train + validation pool. The test split is never touched here.

Each fold trains a fresh model, fits its own input normaliser and loss scales on that fold's training part only, and is scored on the held-out part. Folds are fixed by FOLD_SEED so every configuration is compared on identical folds.

`refit_and_test` trains once on the whole pool and scores the untouched test split; call it once, at the very end.

Run with `python -m src.crossval --data data/dataset_rescaled.npz` (add --from-tuning <json> to use a tuning run's best config).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import (DATA_DIR, DEFAULT_ACTIVATION, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_DROPOUT, DEFAULT_EPOCHS,
                     DEFAULT_GRAD_CLIP, DEFAULT_LAM, DEFAULT_LR, DEFAULT_MU_MAX, DEFAULT_RAMP_FRAC,
                     DEFAULT_SCHEDULER, DEFAULT_TRAIN_FRACTION, DEFAULT_WEIGHT_DECAY, DEFAULT_WIDTH)
from .train import evaluate, load_split, train

DEFAULT_FOLDS = 5
FOLD_SEED = 0
METRICS = ("eigen_pcm_median", "eigen_pcm_p95", "flux_l2_median", "flux_l2_max")

DEFAULT_CONFIG = {"epochs": DEFAULT_EPOCHS, "lr": DEFAULT_LR, "batch_size": DEFAULT_BATCH_SIZE, "width": DEFAULT_WIDTH,
                  "depth": DEFAULT_DEPTH, "lam": DEFAULT_LAM, "mu_max": DEFAULT_MU_MAX, "ramp_frac": DEFAULT_RAMP_FRAC,
                  "weight_decay": DEFAULT_WEIGHT_DECAY, "dropout": DEFAULT_DROPOUT, "activation": DEFAULT_ACTIVATION,
                  "scheduler": DEFAULT_SCHEDULER, "train_fraction": DEFAULT_TRAIN_FRACTION, "grad_clip": DEFAULT_GRAD_CLIP,
                  "flux_loss": "mse", "zone_head": False}
RITZ_METRICS = ("ritz_k_pcm_rmse", "ritz_k_pcm_median", "ritz_flux_l2_mean")


def ritz_metrics(model, norm, positions, eigenvalue, flux, ritz_m):
    """Errors of the hybrid pipeline (1 inverse-power step, Ritz re-fit with ritz_m hats, 1 more step): k RMSE and median in pcm, mean flux relative L2."""
    from .hybrid import predict  # local import: hybrid imports train, which crossval also uses

    out = predict(model, norm, positions, refine=1, ritz_m=ritz_m)
    k_err = (out["k"] - eigenvalue.double()).abs() * 1e5
    ref = flux.double()
    l2 = (out["flux"] - ref).norm(dim=-1) / ref.norm(dim=-1)
    return {"ritz_k_pcm_rmse": float(k_err.pow(2).mean().sqrt()), "ritz_k_pcm_median": float(k_err.median()),
            "ritz_flux_l2_mean": float(l2.mean())}


def score_split(model, norm, positions, eigenvalue, flux, ritz_m=0):
    """`evaluate` metrics, plus the hybrid-pipeline metrics when ritz_m > 0."""
    metrics = evaluate(model, norm, positions, eigenvalue, flux)
    if ritz_m > 0:
        metrics.update(ritz_metrics(model, norm, positions, eigenvalue, flux, ritz_m))
    return metrics


def make_folds(pool, k, seed=FOLD_SEED):
    """Shuffle the pool and split it into k (train, val) index pairs. Every pool index is validated exactly once."""
    if k < 2:
        raise ValueError("need at least 2 folds")
    pool = np.asarray(pool)
    parts = np.array_split(np.random.default_rng(seed).permutation(pool), k)
    return [(np.concatenate(parts[:i] + parts[i + 1:]), parts[i]) for i in range(k)]


def _pool(idx):
    return np.concatenate([idx["train"].cpu().numpy(), idx["val"].cpu().numpy()])


def cross_validate(data, config, use_physics=False, folds=DEFAULT_FOLDS, seed=0, device=None, loaded=None, after_fold=None, ritz_m=0):
    """Train `folds` models and return per-fold metrics plus their mean and standard deviation.

    `after_fold(i, per_fold)` is called after each fold; it may raise to abandon the run (used for pruning).
    With ritz_m > 0 every fold is also scored through the hybrid pipeline (see `ritz_metrics`).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positions, eigenvalue, flux, idx, _ = loaded or load_split(data, device)

    per_fold = []
    for tr, va in make_folds(_pool(idx), folds):
        tr, va = (torch.tensor(a, dtype=torch.long, device=device) for a in (tr, va))
        model, norm, _, _ = train(data, use_physics=use_physics, seed=seed, device=device, verbose=False, split={"train": tr, "val": va}, **config)
        per_fold.append(score_split(model, norm, positions[va], eigenvalue[va], flux[va], ritz_m))
        if after_fold is not None:
            after_fold(len(per_fold) - 1, per_fold)

    keys = METRICS + (RITZ_METRICS if ritz_m > 0 else ())
    return {"folds": per_fold,
            "mean": {m: float(np.mean([f[m] for f in per_fold])) for m in keys},
            "std": {m: float(np.std([f[m] for f in per_fold], ddof=1)) for m in keys}}


def refit_and_test(data, config, use_physics=False, seed=0, device=None, loaded=None, ritz_m=0):
    """Train once on the whole train + val pool and score the held-out test split."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positions, eigenvalue, flux, idx, _ = loaded or load_split(data, device)
    pool = torch.tensor(_pool(idx), dtype=torch.long, device=device)
    model, norm, _, _ = train(data, use_physics=use_physics, seed=seed, device=device, verbose=False, split={"train": pool, "val": idx["val"]}, **config)
    te = idx["test"]
    return score_split(model, norm, positions[te], eigenvalue[te], flux[te], ritz_m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_DIR / "dataset_rescaled.npz"))
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--physics", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-tuning", default=None, help="use the best config recorded in a results/tune_*.json")
    p.add_argument("--test", action="store_true", help="also refit on the whole pool and score the test split")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.from_tuning:
        config.update(json.loads(Path(args.from_tuning).read_text())["best"]["config"])

    result = {"data": args.data, "physics": args.physics, "folds": args.folds, "config": config, "cv": cross_validate(args.data, config, args.physics, args.folds, args.seed)}
    for m in METRICS:
        print(f"{m:18s} {result['cv']['mean'][m]:10.4f} +/- {result['cv']['std'][m]:.4f}   ({args.folds} folds)")
    if args.test:
        result["test"] = refit_and_test(args.data, config, args.physics, args.seed)
        print("test (refit on train+val): " + ", ".join(f"{m} {result['test'][m]:.4f}" for m in METRICS))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

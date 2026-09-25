"""Learning curve: validation error as a function of how much training data is used.

Trains the same config on 10% ... 100% of the training split and scores the fixed validation split, then fits a power law error ~ n^-alpha.
A clearly positive alpha with no flattening means more data would still help.
The test split is not used.
Add --tuning to use a tuned config, otherwise the defaults.

Run with `python -m src.learning_curve --data data/dataset_rescaled.npz`.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DATA_DIR, FIGURE_DIR, RESULTS_DIR, ensure_dirs
from .crossval import DEFAULT_CONFIG
from .metrics import full_report
from .train import load_split, train
from .verify import predict

DEFAULT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def fit_power_law(n, err):
    """Least-squares fit of log(err) = log(c) - alpha log(n). Returns (alpha, c)."""
    slope, intercept = np.polyfit(np.log(n), np.log(err), 1)
    return -float(slope), float(np.exp(intercept))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_DIR / "dataset_rescaled.npz"))
    p.add_argument("--tuning", default=None, help="results/tune_*.json to take the config (and physics flag) from")
    p.add_argument("--physics", action="store_true")
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ensure_dirs()

    config, physics = dict(DEFAULT_CONFIG), args.physics
    if args.tuning:
        t = json.loads(Path(args.tuning).read_text())
        config.update(t["best"]["config"])
        physics = t["physics"]
    tag = Path(args.data).stem + ("_physics" if physics else "") + ("_tuned" if args.tuning else "")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions, eigenvalue, flux, idx, _ = load_split(args.data, device)
    va = idx["val"]
    rows = []
    for frac in args.fractions:
        cfg = {**config, "train_fraction": frac}
        model, norm, _, _ = train(args.data, use_physics=physics, seed=args.seed, device=device, verbose=False, **cfg)
        kp, fp = predict(model, norm, positions[va].cpu().numpy(), device)
        rep = full_report(eigenvalue[va].cpu().numpy(), kp, flux[va].cpu().numpy(), fp)
        n = int(frac * len(idx["train"]))
        rows.append({"fraction": frac, "n_train": n, "report": rep})
        print(f"{frac:5.0%}  n={n:6d}  k: R2 {rep['eigenvalue']['r2']:.5f} RMSE {rep['eigenvalue']['rmse_pcm']:8.1f} pcm   "
              f"flux: R2 {rep['flux']['r2']:.5f} rel-L2 {rep['flux']['rel_l2_median']:.4f}", flush=True)

    n = np.array([r["n_train"] for r in rows], dtype=float)
    fits = {}
    for label, err in (("eigenvalue_rmse_pcm", [r["report"]["eigenvalue"]["rmse_pcm"] for r in rows]),
                       ("flux_rel_l2_median", [r["report"]["flux"]["rel_l2_median"] for r in rows])):
        alpha, c = fit_power_law(n, err)
        fits[label] = {"alpha": alpha, "c": c}
        print(f"power-law fit {label}: error ~ n^-{alpha:.2f}")

    out = RESULTS_DIR / f"learning_curve_{tag}.json"
    out.write_text(json.dumps({"data": args.data, "config": config, "physics": physics, "rows": rows, "fits": fits}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], "eigenvalue_rmse_pcm", "eigenvalue RMSE (pcm)"),
                           (axes[1], "flux_rel_l2_median", "flux relative L2 (median)")):
        err = [r["report"]["eigenvalue"]["rmse_pcm"] if key.startswith("eig") else r["report"]["flux"]["rel_l2_median"]
               for r in rows]
        ax.loglog(n, err, "o-", label="validation")
        ax.loglog(n, fits[key]["c"] * n ** -fits[key]["alpha"], "--", label=f"n^-{fits[key]['alpha']:.2f}")
        ax.set_xlabel("training samples")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    (FIGURE_DIR / "tuning").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "tuning" / f"learning_curve_{tag}.png", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

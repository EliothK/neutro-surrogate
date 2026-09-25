"""Train the tuned model for real and keep it.

`src.tune` only reports a refit's test score; it does not save the model.
This reads the winning config from a results/tune_*.json, trains it on the whole train + validation pool, saves models/tuned_<name>.pt (loadable by `src.benchmark.load_model`), and reports R^2 / RMSE / MAE on the pool (in-sample) and on the untouched test split.

Run with `python -m src.final --tuning results/tune_dataset_rescaled_physics.json`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import MODEL_DIR, RESULTS_DIR, ensure_dirs
from .metrics import format_report, full_report
from .train import load_split, train
from .verify import predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tuning", required=True, help="results/tune_*.json from src.tune")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="checkpoint path (default models/tuned_<name>.pt)")
    args = p.parse_args()
    ensure_dirs()

    tuning = json.loads(Path(args.tuning).read_text())
    data, physics, config = tuning["data"], tuning["physics"], tuning["best"]["config"]
    name = Path(args.tuning).stem.replace("tune_", "")
    out = Path(args.out or MODEL_DIR / f"tuned_{name}.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions, eigenvalue, flux, idx, _ = load_split(data, device)
    pool = torch.cat([idx["train"], idx["val"]])
    model, norm, _, _ = train(data, use_physics=physics, seed=args.seed, device=device, verbose=False,
                              split={"train": pool, "val": idx["val"]}, **config)

    torch.save({"state_dict": model.state_dict(), "normalizer": norm.state_dict(), "physics": physics,
                "epochs": config["epochs"], "width": config["width"], "depth": config["depth"],
                "activation": config["activation"], "dropout": config["dropout"], "zone_head": config.get("zone_head", False), "config": config, "data": data,
                "trained_on": "train+val"}, out)
    print(f"wrote {out}")

    report = {"data": data, "physics": physics, "config": config, "model": str(out)}
    for label, rows in (("pool (in-sample)", pool), ("test (held out)", idx["test"])):
        r = rows.cpu().numpy()
        kp, fp = predict(model, norm, positions[rows].cpu().numpy(), device)
        report[label] = full_report(eigenvalue[rows].cpu().numpy(), kp, flux[rows].cpu().numpy(), fp)
        print(format_report(label, report[label]))
    gap = report["test (held out)"]["eigenvalue"]["rmse_pcm"] / report["pool (in-sample)"]["eigenvalue"]["rmse_pcm"]
    report["eigenvalue_rmse_test_over_pool"] = gap
    print(f"eigenvalue RMSE test / in-sample = {gap:.2f}  (near 1 means no overfitting gap)")
    (RESULTS_DIR / f"final_{name}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

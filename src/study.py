"""Full study: cross-validated Optuna tuning for every dataset variant, with and without the physics loss.

For each combination this runs `src.tune` (k-fold CV scoring, then one refit + test score for the chosen config).
Each study is stored in SQLite, so rerunning this command resumes where it stopped.

Run with `python -m src.study`.
"""

import argparse
import subprocess
import sys

from .config import DATA_DIR
from .crossval import DEFAULT_FOLDS
from .tune import DEFAULT_TRIALS

VARIANTS = ("rescaled", "natural")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--models", nargs="+", choices=("data_only", "physics"), default=["data_only", "physics"])
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--timeout-hours", type=float, default=None, help="per-study wall-clock limit")
    p.add_argument("--fresh", action="store_true", help="delete saved studies and start every one over")
    args = p.parse_args()

    for variant in args.variants:
        for model in args.models:
            cmd = [sys.executable, "-m", "src.tune", "--data", str(DATA_DIR / f"dataset_{variant}.npz"),
                   "--folds", str(args.folds), "--trials", str(args.trials)]
            if model == "physics":
                cmd.append("--physics")
            if args.timeout_hours:
                cmd += ["--timeout-hours", str(args.timeout_hours)]
            if args.fresh:
                cmd.append("--fresh")
            print(f"==> [{variant} / {model}] $ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

    print("==> study done. See results/tune_*.json, results/tune_*_trials.csv and figures/tuning/.")


if __name__ == "__main__":
    main()

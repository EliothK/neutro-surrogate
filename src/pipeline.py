"""End-to-end pipeline: tests -> dataset -> train (data-only + physics) -> benchmark -> figures.

Run with `python -m src.pipeline`.
"""

import argparse
import subprocess
import sys

from .config import DATA_DIR, DEFAULT_N_SAMPLES, DEFAULT_SEED


def run(cmd, label):
    print(f"==> {label}")
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--data", default=str(DATA_DIR / "dataset.npz"))
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-dataset", action="store_true", help="reuse the existing --data file")
    args = p.parse_args()

    python = sys.executable

    if not args.skip_tests:
        run([python, "-m", "pytest", "-q"], "[1/5] tests")

    if not args.skip_dataset:
        run([python, "-m", "src.make_dataset", "--n-samples", str(args.n_samples), "--seed", str(args.seed), "--out", args.data], "[2/5] dataset")

    run([python, "-m", "src.train", "--data", args.data, "--epochs", str(args.epochs)],"[3/5] train: data-only")

    run([python, "-m", "src.train", "--data", args.data, "--epochs", str(args.epochs), "--physics"], "[3/5] train: physics-informed")

    run([python, "-m", "src.benchmark", "--data", args.data], "[4/5] benchmark")

    run([python, "-m", "src.plots", "--data", args.data], "[5/5] figures")

    print("==> done. models/, results/metrics.json, and figures/ are up to date.")


if __name__ == "__main__":
    main()

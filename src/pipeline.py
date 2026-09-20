"""End-to-end pipeline: tests -> per variant (rescaled, natural): dataset -> train (data-only + physics) -> benchmark -> figures.

Run with `python -m src.pipeline`.
"""

import argparse
import subprocess
import sys

from .config import (DATA_DIR, DEFAULT_EPOCHS, DEFAULT_N_SAMPLES, DEFAULT_SEED, FIGURE_DIR, MODEL_DIR,
                     NATURAL_K_RANGE, PROFILE_KNOTS, RESULTS_DIR, ensure_dirs)


def run(cmd, label):
    print(f"==> {label}")
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


VARIANTS = ("rescaled", "natural")


def run_variant(python, name, args):
    """Dataset -> train (data-only + physics) -> benchmark -> figures, all files suffixed with the variant name."""
    data = DATA_DIR / f"dataset_{name}.npz"
    plain, phys = MODEL_DIR / f"surrogate_{name}_data_only.pt", MODEL_DIR / f"surrogate_{name}_physics.pt"

    if not args.skip_dataset:
        run([python, "-m", "src.make_dataset", "--n-samples", str(args.n_samples), "--seed", str(args.seed),
             "--mode", name, "--n-knots", str(args.n_knots), "--k-min", str(args.k_min), "--k-max", str(args.k_max),
             "--out", str(data)], f"[{name}] dataset")

    common = ["--data", str(data), "--epochs", str(args.epochs)]
    run([python, "-m", "src.train", *common, "--out", str(plain)], f"[{name}] train: data-only")
    run([python, "-m", "src.train", *common, "--physics", "--out", str(phys)], f"[{name}] train: physics-informed")
    run([python, "-m", "src.benchmark", "--data", str(data), "--model", str(plain), "--physics-model", str(phys),
         "--results", str(RESULTS_DIR / f"metrics_{name}.json")], f"[{name}] benchmark")
    run([python, "-m", "src.plots", "--data", str(data), "--model", str(plain),
         "--fig-dir", str(FIGURE_DIR / name)], f"[{name}] figures")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--variant", choices=(*VARIANTS, "both"), default="both",
                   help="rescaled = exact-k near-critical band; natural = un-rescaled k kept within --k-min/--k-max")
    p.add_argument("--n-knots", type=int, default=PROFILE_KNOTS, help="knots per smooth profile; 0 = independent zones")
    p.add_argument("--k-min", type=float, default=NATURAL_K_RANGE[0])
    p.add_argument("--k-max", type=float, default=NATURAL_K_RANGE[1])
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-dataset", action="store_true", help="reuse the existing dataset_<variant>.npz files")
    args = p.parse_args()

    ensure_dirs()
    python = sys.executable

    if not args.skip_tests:
        run([python, "-m", "pytest", "-q"], "tests")

    for name in (VARIANTS if args.variant == "both" else (args.variant,)):
        run_variant(python, name, args)

    print("==> done. See models/, results/metrics_<variant>.json and figures/<variant>/.")


if __name__ == "__main__":
    main()

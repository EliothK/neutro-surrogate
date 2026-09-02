"""Build data/dataset.npz: LHS design -> solved & criticality-rescaled samples -> splits."""

import argparse
import time

import numpy as np

from .config import DATA_DIR, DEFAULT_N_SAMPLES, DEFAULT_SEED, SPLIT_FRACTIONS, ensure_dirs
from .sampling import (apply_reflectors, draw_eigenvalue_targets, is_degenerate, latin_hypercube, make_splits, rescale_to_target_eigenvalue)
from .solver import solve_arpack, unpack


def generate(n_samples, seed, chunk_factor=1.3, batch=4096, verbose=True):
    """Rejection-sample LHS draws until n_samples solved, rescaled rows are collected."""
    rng = np.random.default_rng(seed)
    positions, eigenvalues, fluxes, reflector = [], [], [], []
    drawn = 0

    while len(positions) < n_samples:
        remaining = n_samples - len(positions)
        n_draw = max(batch, int(remaining * chunk_factor))
        candidates = latin_hypercube(n_draw, seed=seed + drawn)
        candidates, refl_mask = apply_reflectors(candidates, rng)
        targets = draw_eigenvalue_targets(n_draw, rng)
        drawn += n_draw

        for pos, target, is_refl in zip(candidates, targets, refl_mask):
            if len(positions) >= n_samples:
                break
            if is_degenerate(pos):
                continue
            slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(pos)
            try:
                eig_val_raw, neu_flux = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
            except (ValueError, ArithmeticError):
                continue
            if eig_val_raw <= 0:
                continue

            rescaled = rescale_to_target_eigenvalue(pos, eig_val_raw, target)
            positions.append(rescaled)
            eigenvalues.append(target)
            fluxes.append(neu_flux)
            reflector.append(bool(is_refl))

        if verbose:
            print(f"  {len(positions):6d} / {n_samples} accepted ({drawn} drawn)")

    return (np.asarray(positions[:n_samples]),
            np.asarray(eigenvalues[:n_samples]),
            np.asarray(fluxes[:n_samples]),
            np.asarray(reflector[:n_samples]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=str(DATA_DIR / "dataset.npz"))
    args = p.parse_args()

    ensure_dirs()
    t0 = time.perf_counter()

    positions, eigenvalue, flux, reflector = generate(args.n_samples, args.seed)
    train_idx, val_idx, test_idx = make_splits(len(positions), SPLIT_FRACTIONS, args.seed)

    np.savez(args.out,
              positions=positions,
              eigenvalue=eigenvalue,
              flux=flux,
              reflector=reflector,
              train_idx=train_idx,
              val_idx=val_idx,
              test_idx=test_idx)

    dt = time.perf_counter() - t0
    print(f"wrote {args.out} ({len(positions)} samples, {dt:.1f} s)")


if __name__ == "__main__":
    main()

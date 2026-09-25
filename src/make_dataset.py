"""Build data/dataset.npz: LHS design -> solved & criticality-rescaled samples -> splits."""

import argparse
import time

import numpy as np

from .config import (DATA_DIR, DEFAULT_N_SAMPLES, DEFAULT_SEED, NATURAL_K_RANGE, PROFILE_KNOTS, SPLIT_FRACTIONS, ensure_dirs, widened_bounds)
from .sampling import (apply_reflectors, draw_eigenvalue_targets, is_degenerate, latin_hypercube, make_splits, rescale_to_target_eigenvalue)
from .solver import solve_arpack, unpack


def generate(n_samples, seed, mode="rescaled", n_knots=PROFILE_KNOTS, k_range=NATURAL_K_RANGE,
             batch=4096, verbose=True, bounds=None):
    """Rejection-sample LHS draws until n_samples solved rows are collected.

    mode="rescaled": fission is rescaled so k lands exactly on a random target (near-critical band).
    mode="natural":  parameters are kept as drawn and rows are accepted if k falls inside k_range.
    """
    rng = np.random.default_rng(seed)
    positions, eigenvalues, fluxes, reflector = [], [], [], []
    drawn, seen = 0, 0

    while len(positions) < n_samples:
        remaining = n_samples - len(positions)
        accept_rate = max(0.02, len(positions) / seen) if seen else 1.0
        n_draw = max(batch, int(1.2 * remaining / accept_rate))
        candidates = latin_hypercube(n_draw, seed=seed + drawn, bounds=bounds, n_knots=n_knots)
        candidates, refl_mask = apply_reflectors(candidates, rng)
        targets = draw_eigenvalue_targets(n_draw, rng)
        drawn += n_draw

        for pos, target, is_refl in zip(candidates, targets, refl_mask):
            if len(positions) >= n_samples:
                break
            seen += 1
            if is_degenerate(pos):
                continue
            slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(pos)
            try:
                eig_val_raw, neu_flux = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
            except (ValueError, ArithmeticError):
                continue
            if eig_val_raw <= 0:
                continue

            if mode == "rescaled":
                pos, eig_val_raw = rescale_to_target_eigenvalue(pos, eig_val_raw, target), target
            elif not k_range[0] <= eig_val_raw <= k_range[1]:
                continue
            positions.append(pos)
            eigenvalues.append(eig_val_raw)
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
    p.add_argument("--out", default=None)
    p.add_argument("--mode", choices=("rescaled", "natural"), default="rescaled")
    p.add_argument("--n-knots", type=int, default=PROFILE_KNOTS, help="knots per profile; 0 = independent zones")
    p.add_argument("--ood", action="store_true",
                   help="draw from the widened parameter box instead (out-of-distribution test set, all rows are test)")
    p.add_argument("--k-min", type=float, default=NATURAL_K_RANGE[0])
    p.add_argument("--k-max", type=float, default=NATURAL_K_RANGE[1])
    args = p.parse_args()

    ensure_dirs()
    t0 = time.perf_counter()

    out = args.out or str(DATA_DIR / (f"ood_{args.mode}.npz" if args.ood else f"dataset_{args.mode}.npz"))
    positions, eigenvalue, flux, reflector = generate(args.n_samples, args.seed, args.mode, args.n_knots,
                                                       (args.k_min, args.k_max),
                                                       bounds=widened_bounds() if args.ood else None)
    if args.ood:  # nothing to train on: every row is a held-out test row
        train_idx = val_idx = np.array([], dtype=int)
        test_idx = np.arange(len(positions))
    else:
        train_idx, val_idx, test_idx = make_splits(len(positions), SPLIT_FRACTIONS, args.seed)

    np.savez(out,
              positions=positions,
              eigenvalue=eigenvalue,
              flux=flux,
              reflector=reflector,
              train_idx=train_idx,
              val_idx=val_idx,
              test_idx=test_idx)

    dt = time.perf_counter() - t0
    print(f"wrote {out} ({len(positions)} samples, {dt:.1f} s)")


if __name__ == "__main__":
    main()

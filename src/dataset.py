"""Dataset generator driver"""

import argparse
import time

import numpy as np

from .config import(DATA_DIR, DEFAULT_N_SAMPLES, DEFAULT_SEED, N_CELLS, SPLIT_FRACTIONS,
                    ensure_dirs, input_bounds, widened_bounds)
from .sampling import(apply_reflectors, draw_eigenvalue_targets, is_degenerate, latin_hypercube,
                      make_splits, rescale_to_target_eigenvalue)
from .solver import(solve_arpack, unpack)

def generate(n_samples, seed, ood=False, verbose=True):
    """Return (pos_on_slab, eigenvalue, neu_flux, is_reflector, n_rejected)"""
    bounds = widened_bounds() if ood else input_bounds()
    rng = np.random.default_rng(seed)

    positions_on_slab = latin_hypercube(n_samples, seed, bounds=bounds)
    positions_on_slab, reflector = apply_reflectors(positions_on_slab, rng)
    eig_val_tar = draw_eigenvalue_targets(n_samples, rng)

    rows, eig_vals, neu_fluxes, refl = [], [], [], []
    rejected = 0
    t0=time.perf_counter()

    for i in range(n_samples):
        pos_on_slab = positions_on_slab[i]

        if is_degenerate(pos_on_slab):
            rejected += 1
            continue

        slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(pos_on_slab)
        eig_val_raw, _ = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)

        pos_on_slab_scaled = rescale_to_target_eigenvalue(pos_on_slab, eig_val_raw, eig_val_tar[i])
        slab_width_2, diff_coeff_2, macro_absorp_cross_section_2, fis_neu_prod_2 = unpack(pos_on_slab_scaled)
        eigenvalue, neu_flux = solve_arpack(diff_coeff_2, macro_absorp_cross_section_2, fis_neu_prod_2, slab_width_2)

        rows.append(pos_on_slab_scaled)
        eig_vals.append(eigenvalue)
        neu_fluxes.append(neu_flux)
        refl.append(reflector[i])

        if verbose and (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"{i + 1}/{n_samples} ({rate:.0f} samples)")

    return (np.array(rows), np.array(eig_vals), np.array(neu_fluxes), np.array(refl, dtype=bool), rejected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--ood", action="store_true", help="sample from the widened box for out-of-distribution evaluation")
    args = p.parse_args()

    ensure_dirs()
    out = args.out or (DATA_DIR / ("ood.npz" if args.ood else "dataset.npz"))

    t0 = time.perf_counter()
    positions_on_slab, eigenvalue, neu_flux, reflector, rejected = generate(args.n, args.seed, ood=args.ood)
    dt = time.perf_counter() - t0

    train, val, test = make_splits(len(positions_on_slab), SPLIT_FRACTIONS, args.seed)
    np.savez_compressed(out, positions_on_slab=positions_on_slab, eigenvalue=eigenvalue,
                        neu_flux=neu_flux, reflector=reflector, train_idx=train, val_idx=val, test_idx=test,
                        seed=args.seed, n_cells=N_CELLS)

    print(f"wrote {out}")
    print(f" {len(positions_on_slab)} samples, {rejected} rejected as degenerate, {dt:.1f} s")
    print(f" eigenvalue range {eigenvalue.min():.4f} to {eigenvalue.max():.4f}, median {np.median(eigenvalue):.4f}")
    print(f" reflector cases: {reflector.sum()} ({100 * reflector.mean():.1f}%)")
    print(f" split sizes: {len(train)} / {len(val)}, {len(test)}")


if __name__ == "__main__":
    main()
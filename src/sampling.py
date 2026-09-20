"""Design of experiments for the parameter study"""

import numpy as np
from scipy.stats import qmc

from .config import (TARGET_EIGENVALUE_RANGE, MIN_TOTAL_FISSION, N_INPUTS, N_ZONES, REFLECTOR_FRACTION, SLICE_FIS_NEU_PROD, input_bounds)

N_PROPERTIES = 3  # diffusion coefficient, absorption, fission production

def latin_hypercube(n_samples, seed, bounds=None, n_knots=0):
    """Latin hypercube sample of shape (n_samples, N_INPUTS), scaled into the box.

    With n_knots >= 2 each property is drawn at n_knots points and linearly interpolated onto the N_ZONES zones, giving smooth spatial profiles instead of independent zones.
    """
    low, high = bounds if bounds is not None else input_bounds()
    if n_knots < 2:
        unit = qmc.LatinHypercube(d=N_INPUTS, seed=seed).random(n_samples)
        return qmc.scale(unit, low, high)

    unit = qmc.LatinHypercube(d=1 + N_PROPERTIES * n_knots, seed=seed).random(n_samples)
    zone_pos = np.linspace(0.0, 1.0, N_ZONES)
    knot_pos = np.linspace(0.0, 1.0, n_knots)
    out = np.empty((n_samples, N_INPUTS))
    out[:, 0] = low[0] + unit[:, 0] * (high[0] - low[0])
    for p in range(N_PROPERTIES):
        first = 1 + p * N_ZONES  # bounds are constant within a property block
        knots = low[first] + unit[:, 1 + p * n_knots:1 + (p + 1) * n_knots] * (high[first] - low[first])
        for i in range(n_samples):
            out[i, first:first + N_ZONES] = np.interp(zone_pos, knot_pos, knots[i])
    return out

def apply_reflectors(X, rng, fraction=REFLECTOR_FRACTION):
    """Zero the fission cross section in the outermost zones for a fraction of rows"""
    mask = rng.random(len(X)) < fraction
    fis_neu_prod = X[:, SLICE_FIS_NEU_PROD]
    fis_neu_prod[mask, 0] = 0.0
    fis_neu_prod[mask, -1] = 0.0
    X[:, SLICE_FIS_NEU_PROD] = fis_neu_prod
    return X, mask

def is_degenerate(x, min_total=MIN_TOTAL_FISSION):
    """True if there is effectively no fission source anywhere"""
    return float(np.sum(x[SLICE_FIS_NEU_PROD])) < min_total

def rescale_to_target_eigenvalue(pos_on_slab, eig_val_raw, eig_val_tar):
    """Return a copy of pos_on_slab with fis_neu_prod scaled so the solved eigenvalue becomes exactly eig_val_tar"""
    fis_rescale_multi = eig_val_tar / eig_val_raw
    out = np.array(pos_on_slab, dtype=float, copy=True)
    out[SLICE_FIS_NEU_PROD] = out[SLICE_FIS_NEU_PROD] * fis_rescale_multi
    return out

def draw_eigenvalue_targets(n, rng, eigenvalue_range=TARGET_EIGENVALUE_RANGE):
    return rng.uniform(eigenvalue_range[0], eigenvalue_range[1], size=n)

def make_splits(n, fractions, seed):
    """Shuffled train/val/test index arrays. Save these; never recompute a split"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
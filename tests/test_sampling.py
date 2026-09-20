"""Design of experiments invariants. numpy/scipy only"""

import numpy as np

from src.config import N_INPUTS, N_ZONES, SLICE_FIS_NEU_PROD, input_bounds, widened_bounds
from src.sampling import (apply_reflectors, is_degenerate, latin_hypercube, make_splits, rescale_to_target_eigenvalue)
from src.solver import solve_arpack, unpack

def test_lhs_stays_in_bounds():
    low, high = input_bounds()
    positions = latin_hypercube(500, seed=0)
    assert positions.shape == (500, N_INPUTS)
    assert np.all(positions >= low) and np.all(positions <= high)

def test_widened_bounded_stay_positive():
    low, _ = widened_bounds()
    assert np.all(low >= 0), "widening must never produce a negative cross section"

def test_reflectors_zero_outer_zones():
    rng = np.random.default_rng(1)
    positions = latin_hypercube(400, seed=1)
    positions, mask = apply_reflectors(positions, rng, fraction=0.5)
    fis_neu_prod = positions[:, SLICE_FIS_NEU_PROD]
    assert np.all(fis_neu_prod[mask,0] == 0.0) and np.all(fis_neu_prod[mask, 0-1] == 0.0)
    assert np.all(fis_neu_prod[~mask] > 0.0)

def test_degenerate_detection():
    postions = latin_hypercube(10, seed=2)
    assert not is_degenerate(postions[0])
    dead = postions[0].copy()
    dead[SLICE_FIS_NEU_PROD] = 0.0
    assert is_degenerate(dead)

def test_rescale_hits_target_exactly():
    positions = latin_hypercube(20, seed=3)
    for pos in positions:
        slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(pos)
        eig_val_raw, neu_flux_prod_raw = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
        target = 1.05
        pos_2 = rescale_to_target_eigenvalue(pos, eig_val_raw, target)
        slab_width_2, diff_coeff_2, macro_absorp_cross_section_2, fis_neu_prod_2 = unpack(pos_2)
        eig_val_2, neu_flux_prod_2 = solve_arpack(diff_coeff_2, macro_absorp_cross_section_2, fis_neu_prod_2, slab_width_2)
        assert abs(eig_val_2 - target) < 1e-8
        assert np.abs(neu_flux_prod_raw - neu_flux_prod_2).max() < 1e-10

def test_splits_are_disjoint_and_complete():
    train, val, test = make_splits(1000, (0.8, 0.1, 0.1), seed=0)
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == 1000
    assert len(np.unique(all_idx)) == 1000

def test_smooth_profiles_are_in_bounds_and_piecewise_linear():
    low, high = input_bounds()
    positions = latin_hypercube(100, seed=3, n_knots=4)
    assert positions.shape == (100, N_INPUTS)
    assert np.all(positions >= low) and np.all(positions <= high)
    diff = positions[:, 1:1 + N_ZONES]
    assert np.abs(np.diff(diff, axis=1)).max() < 0.5 * (diff.max() - diff.min())

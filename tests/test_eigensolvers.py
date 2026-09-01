"""Two independent eigensolvers must agree. numpy/scipy only"""

import numpy as np

from src.solver import solve_arpack, solve_power, zone_to_cells

slab_width = 200.0

def test_arpack_matches_power_iteration():
    rng = np.random.default_rng(62922)
    for _ in range(5):
        diff_coeff = zone_to_cells(rng.uniform(0.7, 1.4, 5))
        macro_absorp_cross_section = zone_to_cells(rng.uniform(0.02, 0.15, 5))
        fis_neu_prod = zone_to_cells(rng.uniform(0.02, 0.16, 5))

        eigen_arpack, peak_neu_arpack = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
        eigen_power, peak_neu_power, _ = solve_power(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width, total_eigenvalue=1e-13, total_neu_flux=1e-11)

        assert abs(eigen_arpack - eigen_power) * 1e5 < 1e-6, f"{abs(eigen_arpack - eigen_power) * 1e5:2e} pcm apart"
        assert np.abs(peak_neu_arpack - peak_neu_power).max() < 1e-8
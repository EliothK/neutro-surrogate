"""Two independent eigensolvers must agree. numpy/scipy only"""

import numpy as np

from src.config import DIFF_COEFFICIENT_RANGE, FIS_NEU_PROD_RANGE, MACRO_ABSORP_CROSS_SECTION_RANGE
from src.solver import solve_arpack, solve_power, zone_to_cells

N_ZONES_TEST = 5

slab_width = 200.0

def test_arpack_matches_power_iteration():
    rng = np.random.default_rng(62922)
    for _ in range(5):
        diff_coeff = zone_to_cells(rng.uniform(*DIFF_COEFFICIENT_RANGE, N_ZONES_TEST))
        macro_absorp_cross_section = zone_to_cells(rng.uniform(*MACRO_ABSORP_CROSS_SECTION_RANGE, N_ZONES_TEST))
        fis_neu_prod = zone_to_cells(rng.uniform(*FIS_NEU_PROD_RANGE, N_ZONES_TEST))

        eigen_arpack, peak_neu_arpack = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
        eigen_power, peak_neu_power, _ = solve_power(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width, total_eigenvalue=1e-13, total_neu_flux=1e-11)

        assert abs(eigen_arpack - eigen_power) * 1e5 < 1e-6, f"{abs(eigen_arpack - eigen_power) * 1e5:2e} pcm apart"
        assert np.abs(peak_neu_arpack - peak_neu_power).max() < 1e-8
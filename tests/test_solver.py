import numpy as np, pytest
from src.config import DIFF_COEFFICIENT_RANGE, FIS_NEU_PROD_RANGE, MACRO_ABSORP_CROSS_SECTION_RANGE
from src.solver import build_operator, solve_power, eigenvalue_analytic

N_ZONES_TEST = 5

from src.config import N_CELLS
from src.solver import (build_operator, cell_centers, dense_operator, eigenvalue_analytic, solve_arpack, solve_power, tridiag_matvec, zone_to_cells)

standard_diff_coeff, standard_macro_absorp_cross_seciton, standard_fis_neu_prod, standard_slab_width = 1.0, 0.10, 0.1050, 200.0

def uniform(n=N_CELLS):
    return np.full(n, standard_diff_coeff), np.full(n, standard_macro_absorp_cross_seciton), np.full(n, standard_fis_neu_prod)

def test_banded_layout_matches_dense():
    """The (3,N) banded loss must represent the matrix we think it does"""
    diff_coeff, macro_absorp_cross_section, _ = uniform(6)
    band_loss = build_operator(diff_coeff, macro_absorp_cross_section, 1.0)
    dense = dense_operator(diff_coeff, macro_absorp_cross_section, 1.0)
    x = np.arange(1.0,7.0)
    assert np.allclose(tridiag_matvec(band_loss, x), dense @ x)

def test_uniform_matches_analytic():
    """Uniform material collapses to the bare slab, which theory solves exactly."""
    diff_coeff, macro_absorp_cross_section, fis_neu_prod = uniform()
    eigenvalue, _, _ = solve_power(diff_coeff, macro_absorp_cross_section, fis_neu_prod, standard_slab_width)
    err_pcm = abs(eigenvalue - eigenvalue_analytic(standard_diff_coeff, standard_macro_absorp_cross_seciton, standard_fis_neu_prod, standard_slab_width)) *1e5
    assert err_pcm < 1.0, f"{err_pcm:.4f} pcm from analytic"

def test_second_order_convergence():
    """Error must fall as 1/N**2. A slope near 1 means a first order bug"""
    exact = eigenvalue_analytic(standard_diff_coeff, standard_macro_absorp_cross_seciton, standard_fis_neu_prod, standard_slab_width)
    ns = np.array([25, 50, 100, 200, 400])
    errs = []

    for n in ns:
        diff_coeff, macro_absorp_cross_section, fis_neu_prod = uniform(n)
        eigenvalue, _ = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, standard_slab_width)
        errs.append(abs(eigenvalue - exact))

    errs = np.array(errs)
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert slope < -1.9, f"convergence slop {slope:.3f}, expected about -2"
    ratios = errs[:-1] / errs[1:]
    assert np.all(ratios > 3.5), f"error ratios {ratios} should be near 4"

def test_flux_matches_sine():
    """Exact flux for the uniform case is sin(pi x / a) at CELL CENTERS"""
    n = 400
    diff_coeff, macro_absorp_cross_section, fis_neu_prod = uniform(n)
    _, neutron_flux = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, standard_slab_width)
    exact = np.sin(np.pi * cell_centers(standard_slab_width, n) / standard_slab_width)
    assert np.abs(neutron_flux - exact).max() / exact.max() < 1e-4

def test_fission_scaling_is_exact():
    """Eigenvalue is linear in a uniform fission multiplier and the eigenvector is unchanged"""
    rng = np.random.default_rng(0)
    diff_coeff = zone_to_cells(rng.uniform(*DIFF_COEFFICIENT_RANGE, N_ZONES_TEST))
    macro_absorp_cross_section = zone_to_cells(rng.uniform(*MACRO_ABSORP_CROSS_SECTION_RANGE, N_ZONES_TEST))
    fis_neu_prod = zone_to_cells(rng.uniform(*FIS_NEU_PROD_RANGE, N_ZONES_TEST))

    eigenvalue1, neutron_flux_1 = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, standard_slab_width)
    target = 1.0
    eigenvalue2, neutron_flux_2 = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod * (target/eigenvalue1), standard_slab_width)

    assert abs(eigenvalue2 - target) < 1e-8, f"rescaled eigenvalue off by {abs(eigenvalue2 - target) * 1e5:.2e} pcm"
    assert np.abs(neutron_flux_1 - neutron_flux_2).max() < 1e-10, "flux shape must be untouched"

def test_zoned_flux_is_not_cosine():
    """Guard against the model degenerating back into a solvable bare slab"""
    diff_coeff_Z = np.array([1.0, 0.9, 0.9, 0.9, 1.0])
    macro_absorp_cross_section_Z = np.array([0.03, 0.10, 0.12, 0.10, 0.03])
    fis_neu_prod_Z = np.array([0.00, 0.11, 0.13, 0.11, 0.00])
    _, neutron_flux = solve_arpack(zone_to_cells(diff_coeff_Z), zone_to_cells(macro_absorp_cross_section_Z), zone_to_cells(fis_neu_prod_Z), standard_slab_width)
    cosine = np.sin(np.pi * cell_centers(standard_slab_width, len(neutron_flux)) / standard_slab_width)
    assert np.abs(neutron_flux - cosine).max() > 0.1

def test_zero_fission_raises():
    diff_coeff, macro_absorp_cross_section, _ = uniform()
    with pytest.raises(ValueError):
        solve_arpack(diff_coeff, macro_absorp_cross_section, np.zeros(N_CELLS), standard_slab_width)
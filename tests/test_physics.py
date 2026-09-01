"""The residual must be machine_zero for the exact solution and large for a wrong one."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import N_CELLS
from src.physics import bands_from_input, physics_weight, residual_loss
from src.sampling import apply_reflectors, latin_hypercube
from src.solver import cell_centers, solve_arpack, unpack

def _sample_batch(n=8, seed=62922):
    rng = np.random.default_rng(seed)
    positions = latin_hypercube(n, seed=seed)
    positions, _ = apply_reflectors(positions, rng, fraction=0.5)
    eigenvalues, fluxes = [], []
    for pos_on_slab in positions:
        slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(pos_on_slab)
        eigenvalue, flux = solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
        eigenvalues.append(eigenvalue)
        fluxes.append(flux)
    return (torch.tensor(positions, dtype=torch.float64),
            torch.tensor(eigenvalues, dtype=torch.float64),
            torch.tensor(np.stack(fluxes), dtype=torch.float64))

def test_residual_is_zer_for_exact_solution():
    positions, eigenvalue, flux = _sample_batch()
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    loss = residual_loss(flux, eigenvalue, lower, diag, upper, fis_neu_prod)
    assert loss.item() < 1e-20, f"residual {loss.item():.3e} should be machine-zero"

def test_residual_is_large_for_wrong_flux():
    """If this does not fire, the test cannot discriminate and proves nothing"""
    positions, eigenvalue, _ = _sample_batch()
    slab_width = positions[:, 0].numpy()
    cosine = np.stack([np.sin(np.pi * cell_centers(pos) / pos) for pos in slab_width])
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    loss = residual_loss(torch.tensor(cosine, dtype=torch.float64),
                         eigenvalue, lower, diag, upper, fis_neu_prod)
    assert loss.item() >0.01

def test_residual_is_scale_invariant_in_flux():
    """The eigenvector has no natural amplitude, so the loss must not care"""
    positions, eigenvalue, flux = _sample_batch()
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    a = residual_loss(flux, eigenvalue, lower, diag, upper, fis_neu_prod)
    b = residual_loss(flux * 7.3, eigenvalue, lower, diag, upper, fis_neu_prod)
    assert torch.allclose(a,b, atol=1e-24)

def test_bands_match_numpy_operator():
    """The torch band convention must describe the same matrix as the scipy one"""
    from src.solver import build_operator, tridiag_matvec as np_matvec
    positions, _, _ = _sample_batch(n=3)
    lower, diag, upper, _ = bands_from_input(positions)
    for i in range(len(positions)):
        slab_width, diff_coeff, macro_absorp_cross_section, _ = unpack(positions[i].numpy())
        band_loss = build_operator(diff_coeff, macro_absorp_cross_section, slab_width/N_CELLS)
        v = np.linspace(1.0, 2.0, N_CELLS)
        got = (diag[i].numpy() * v + np.pad(upper[i].numpy()[:-1] * v[1:], (0,1))
                                            + np.pad(upper[i].numpy()[:-1] * v[:-1], (1,0)))
        assert np.allclose(got, np_matvec(band_loss, v))

def test_physics_weigth_ramps():
    assert physics_weight(0, 200) == 0.0
    assert physics_weight(50, 200) == pytest.approx(0.1)
    assert physics_weight(199, 200) == pytest.approx(0.1)

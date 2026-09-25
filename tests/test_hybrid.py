import numpy as np
import torch
from scipy.linalg import solve_banded

from src.config import N_CELLS
from src.hybrid import hat_functions, inverse_power_steps, rayleigh_residual, ritz, thomas
from src.physics import bands_from_input
from src.sampling import apply_reflectors, latin_hypercube
from src.solver import build_operator, solve, unpack


def _inputs(n=4, seed=3):
    X = latin_hypercube(n, seed, n_knots=5)
    return apply_reflectors(X, np.random.default_rng(seed))[0]


def test_thomas_matches_scipy():
    X = _inputs()
    positions = torch.tensor(X, dtype=torch.float64)
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    rhs = torch.rand(len(X), N_CELLS, dtype=torch.float64)
    x = thomas(lower, diag, upper, rhs).numpy()
    for i, xi in enumerate(X):
        a, d, sa, _ = unpack(xi)
        np.testing.assert_allclose(x[i], solve_banded((1, 1), build_operator(d, sa, a / N_CELLS), rhs[i].numpy()), rtol=1e-10)


def test_rayleigh_of_exact_flux_recovers_k():
    X = _inputs()
    solved = [solve(x) for x in X]
    positions = torch.tensor(X, dtype=torch.float64)
    flux = torch.tensor(np.stack([f for _, f in solved]), dtype=torch.float64)
    k, resid = rayleigh_residual(positions, flux)
    np.testing.assert_allclose(k.numpy(), [k_ for k_, _ in solved], rtol=1e-9)
    assert resid.max() < 1e-6


def test_inverse_power_steps_reduce_error():
    X = _inputs()
    positions = torch.tensor(X, dtype=torch.float64)
    exact = torch.tensor(np.stack([solve(x)[1] for x in X]), dtype=torch.float64)
    noisy = (exact * (1 + 0.05 * torch.randn_like(exact))).clamp_min(0)
    err = lambda f: ((f - exact).norm(dim=-1) / exact.norm(dim=-1)).max()
    assert err(inverse_power_steps(positions, noisy, 3)) < err(noisy)


def test_hat_functions_partition_unity():
    h = hat_functions(9, N_CELLS)
    assert h.shape == (9, N_CELLS)
    np.testing.assert_allclose(h.sum(0).numpy(), 1.0, atol=1e-12)
    assert (h >= 0).all()


def test_ritz_is_variational_and_fixes_amplitudes():
    X = _inputs(8)
    solved = [solve(x) for x in X]
    positions = torch.tensor(X, dtype=torch.float64)
    k_true = np.array([k_ for k_, _ in solved])
    exact = torch.tensor(np.stack([f for _, f in solved]), dtype=torch.float64)
    # right local shape, wrong split of power along the slab: a smooth tilt of a factor of 10 end to end
    x = (torch.arange(N_CELLS, dtype=torch.float64) + 0.5) / N_CELLS
    wrong = exact * torch.exp(np.log(10.0) * (x - 0.5))
    wrong = wrong / wrong.amax(-1, keepdim=True)
    k_ray, _ = rayleigh_residual(positions, wrong)
    k_ritz, flux_ritz = ritz(positions, wrong, 17)
    assert (k_ritz.numpy() <= k_true * (1 + 1e-10)).all()  # variational upper bound
    assert (k_ritz >= k_ray * (1 - 1e-12)).all()  # never worse than the plain Rayleigh quotient
    err = lambda f: ((f - exact).norm(dim=-1) / exact.norm(dim=-1)).max()
    assert err(flux_ritz) < 0.05 * err(wrong)


def test_small_batch_path_matches_batched():
    X = _inputs()
    positions = torch.tensor(X, dtype=torch.float64)
    flux = torch.rand(len(X), N_CELLS, dtype=torch.float64)
    batched = inverse_power_steps(positions, flux, 2, small_batch=0)
    per_sample = inverse_power_steps(positions, flux, 2, small_batch=len(X) + 1)
    np.testing.assert_allclose(per_sample.numpy(), batched.numpy(), rtol=1e-9)

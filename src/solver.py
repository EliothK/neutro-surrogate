"""One group neutron diffusion solver for a multi zone 1D slab.

Governing equation, eigenvalue form:
    -d/dx [ D(x) dphi/dx ] + Sa(x) phi(x) = (1/k) nSf(x) phi(x)
    in english

with zero flux at x = 0 and x = a.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, eigs

from .config import IDX_SLAB, N_CELLS, SLICE_DIFF_COEFFICIENT, SLICE_FIS_NEU_PROD, SLICE_MACRO_ABSORP_CROSS_SECTION


def zone_to_cells(zone_values, n_cells=N_CELLS):
    """Expand (n_zones,) material values to (n_cells,)."""
    zone_values = np.asarray(zone_values, dtype=float)
    return np.repeat(zone_values, n_cells // len(zone_values))

def unpack(x, n_cells=N_CELLS):
    """Split an input vector into (slab_width, D, Sa, nSf) with material values expanded to cells"""
    x = np.asarray(x, dtype=float)

    return (float(x[IDX_SLAB]),
            zone_to_cells(x[SLICE_DIFF_COEFFICIENT], n_cells),
            zone_to_cells(x[SLICE_MACRO_ABSORP_CROSS_SECTION], n_cells),
            zone_to_cells(x[SLICE_FIS_NEU_PROD], n_cells))

def cell_centers(slab_width, n_cells=N_CELLS):
    """Cell centres, at (i + 0.5) cell width"""
    cell_width = slab_width / n_cells
    return (np.arange(n_cells) + 0.5) * cell_width

def build_operator (diff_coeff, macro_absorp_cross_section, cell_width):
    """Banded (3,N) representation of the loss operator."""
    N = len(diff_coeff)
    harm_mean_diff_co = 2.0 * diff_coeff[:-1] * diff_coeff[1:] / (diff_coeff[:-1] + diff_coeff[1:])    #Harmonic mean at interfaces

    diag = macro_absorp_cross_section.astype(float).copy()
    diag[:-1] += harm_mean_diff_co / cell_width**2
    diag[1:] += harm_mean_diff_co / cell_width**2
    diag[0] += 2.0 * diff_coeff[0] / cell_width**2    #zero flux at x = 0
    diag[-1] += 2.0 * diff_coeff[-1] / cell_width**2  #zero flux at x = a

    band_loss = np.zeros((3,N))
    band_loss[0,1:] = -harm_mean_diff_co / cell_width**2   #superdiagonal
    band_loss[1,:] = diag
    band_loss[2, :-1] = -harm_mean_diff_co / cell_width**2 #subdiagonal
    return band_loss

def dense_operator(diff_coeff, macro_absorp_cross_section, cell_width):
    """Dense form of the same operator. Used only to check the banded layout."""
    band_loss = build_operator(diff_coeff,macro_absorp_cross_section, cell_width)
    return (np.diag(band_loss[1,:]) + np.diag(band_loss[0,1:], k = 1) + np.diag(band_loss[2, :-1], k = -1))

def tridiag_matvec(band_loss, pos_on_slab):
    """Multiply the banded operator by a vector without forming the matrix."""
    y = band_loss[1] * pos_on_slab
    y[:-1] += band_loss[0,1:] * pos_on_slab[1:]
    y[1:] += band_loss[2, :-1] * pos_on_slab[:-1]
    return y

def eigenvalue_analytic(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width):
    geometric_buckling = (np.pi / slab_width) ** 2
    return fis_neu_prod / (macro_absorp_cross_section + diff_coeff * geometric_buckling)

def solve_power(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slabWidth, total_eigenvalue=1e-10, total_neu_flux=1e-8, power_iter_cap=20000):
    """Power iteration. Returns (eigenvalue, peakNeutronFlux, iterations)."""

    N = len(diff_coeff)
    cell_width = slabWidth / N
    band_loss = build_operator(diff_coeff, macro_absorp_cross_section, cell_width)

    neu_flux = np.ones(N)
    eigenvalue = 1.0
    src = fis_neu_prod * neu_flux

    for it in range(power_iter_cap):
        new_neu_flux = solve_banded((1,1), band_loss, src / eigenvalue)
        src_new = fis_neu_prod * new_neu_flux
        new_eigenvalue = eigenvalue * src_new.sum() / src.sum()
        pn, po = new_neu_flux / new_neu_flux.max(), neu_flux / neu_flux.max()
        coverged = abs(new_eigenvalue - eigenvalue) / eigenvalue < total_eigenvalue and np.linalg.norm(pn - po) / np.linalg.norm(po) < total_neu_flux
        scale = new_neu_flux.max()
        neu_flux, eigenvalue, src = new_neu_flux / scale, new_eigenvalue, src_new / scale
        if coverged:
            return eigenvalue, neu_flux, it +1
    raise RuntimeError("power iteration did not converge")

def solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width):
    """Same eigenproblem via ARPACK on A^-1 F. Retruns (eigenvalue, peakNeutronFlux)"""

    if not np.any(fis_neu_prod):
        raise ValueError("fission source is zero; eigenvalue problem is degenerate")

    N = len(diff_coeff)
    band_loss = build_operator(diff_coeff, macro_absorp_cross_section, slab_width/N)

    op = LinearOperator((N,N), matvec=lambda v: solve_banded((1,1), band_loss, v * fis_neu_prod), dtype=float)

    vals, vecs = eigs(op, k=1, which="LM", tol=1e-11, v0=np.ones(N))  # fixed start: ARPACK's random one made solves irreproducible
    neu_flux = np.abs(vecs[:, 0].real)
    return float(vals[0].real), neu_flux/neu_flux.max()

def solve(x, method="arpack"):
    """Solve directly from an input vector"""
    slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(x)
    if method == "arpack":
        return solve_arpack(diff_coeff,macro_absorp_cross_section,fis_neu_prod,slab_width)
    eigenvalue, neu_flux, _ = solve_power(diff_coeff,macro_absorp_cross_section,fis_neu_prod,slab_width)
    return eigenvalue, neu_flux
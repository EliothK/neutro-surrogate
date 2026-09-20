"""Batched, differentiable form of the diffusion operator, for the residual loss"""

import torch
import torch.nn.functional as F

from .config import CELLS_PER_ZONE, IDX_SLAB, N_CELLS, SLICE_DIFF_COEFFICIENT, SLICE_FIS_NEU_PROD, SLICE_MACRO_ABSORP_CROSS_SECTION

def expand_zones(zones_values, cells_per_zone=CELLS_PER_ZONE):
    """(B, n_zones) > (B, n_cells), matching solver.zone_to_cells"""
    return zones_values.repeat_interleave(int(cells_per_zone), dim=1)

def bands_from_input(positions, n_cells=N_CELLS):
    """Build (lower, diag, upper, fis_neu_prod) from a batch of raw (unnormalsed inputs)"""
    slab_width = positions[:, IDX_SLAB:IDX_SLAB + 1]
    cell_width2 = (slab_width / n_cells) ** 2

    diff_coeff = expand_zones(positions[:, SLICE_DIFF_COEFFICIENT])
    macro_absorp_cross_section = expand_zones(positions[:, SLICE_MACRO_ABSORP_CROSS_SECTION])
    fis_neu_prod = expand_zones(positions[:, SLICE_FIS_NEU_PROD])

    harm_mean_diff_co = 2.0 * diff_coeff[:, :-1] * diff_coeff[:,1:]/ (diff_coeff[:, :-1] + diff_coeff[:, 1:])
    coup = harm_mean_diff_co / cell_width2

    diag = (macro_absorp_cross_section
            + F.pad(coup, (0,1))
            + F.pad(coup, (1,0))
            + F.pad(2.0 * diff_coeff[:, :1] / cell_width2, (0, n_cells-1))
            + F.pad(2.0 * diff_coeff[:, -1:] / cell_width2, (n_cells -1, 0)))

    lower = F.pad(-coup, (1,0))
    upper = F.pad(-coup, (0,1))
    return lower, diag, upper, fis_neu_prod

def tridiag_matvec(lower, diag, upper, x):
    """Batched tridiagonal product. All tensors (B, N)"""
    y = diag * x
    y = y + F.pad(upper[:, :-1] * x[:, 1:], (0,1))
    y = y + F.pad(lower[:, 1:] * x[:, :-1], (1,0))
    return y

def residual(flux_hat, eigen_hat, lower, diag, upper, fis_neu_prod):
    flux = fis_neu_prod * flux_hat
    return tridiag_matvec(lower, diag, upper, flux_hat) - flux/eigen_hat.unsqueeze(-1)

def residual_loss(flux_hat, eigen_hat, lower, diag, upper, fis_neu_prod, reduce=True):
    """Scale-free squared residual"""
    flux = fis_neu_prod * flux_hat
    resid = residual(flux_hat, eigen_hat, lower, diag, upper, fis_neu_prod)
    per_sample = (resid ** 2).sum(-1) / (flux ** 2).sum(-1).clamp_min(1e-12)
    return per_sample.mean() if reduce else per_sample

def physics_weight(epoch, n_epochs, mu_max=0.1, ramp_frac=0.25):
    """Linear ramp of mu from 0 to mu_max over the first ramp_frac of training"""

    ramp_epochs = max(1, int(ramp_frac * n_epochs))
    return mu_max * min(1.0, epoch / ramp_epochs)
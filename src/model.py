"""Surrogate network: N_INPUTS parameters > eigenvalue eff and an N_CELLS point flux profile"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_ACTIVATION, DEFAULT_DEPTH, DEFAULT_DROPOUT, DEFAULT_WIDTH, N_CELLS, N_INPUTS

ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

class SurrogateMLP(nn.Module):
    def __init__(self, n_inputs=N_INPUTS, n_cells=N_CELLS, width=DEFAULT_WIDTH, depth=DEFAULT_DEPTH,
                 activation=DEFAULT_ACTIVATION, dropout=DEFAULT_DROPOUT):
        super().__init__()
        layers = []
        d_in = n_inputs
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.eigenvalue_head = nn.Linear(width, 1)
        self.flux_head = nn.Linear(width, n_cells)

    def forward(self, pos_on_slab_normal):
        """pos_on_slab_normal is standardised. Returns (eigenvalue, flux) in physical terms"""
        z = self.trunk(pos_on_slab_normal)
        eigenvalue = torch.exp(self.eigenvalue_head(z)).squeeze(-1)
        flux = F.softplus(self.flux_head(z))  # flux is physically non-negative
        flux = flux / flux.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        return eigenvalue, flux

class Normalizer:
    """Input standardisation, fitted on the Training split only."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, X_train):
        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0).clamp_min(1e-8)
        return cls(mean, std)

    def __call__(self, X):
        return (X - self.mean.to(X.device)) / self.std.to(X.device)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])
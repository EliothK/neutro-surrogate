"""Accuracy metrics, baseline comparison, and honest timing"""

import argparse
import json
import time

import numpy as np
import torch

from.baseline import fit_baselines
from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs
from .model import Normalizer, SurrogateMLP
from .physics import bands_from_input, residual_loss
from .solver import solve_arpack, unpack

def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SurrogateMLP().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = Normalizer.from_state_dict(ckpt["normalizer"])
    return model, norm, ckpt

def accuracy(model, norm, positions, eigenvalue, flux):
    """Percentiles, not just means. A mean hides the tail, and the tail is what
    a reactor engineer cares about"""
    with torch.no_grad():
        eigen_hat, flux_hat = model(norm(positions))

    pcm = ((eigen_hat - eigenvalue).abs() * 1e5).numpy()
    l2 = ((flux_hat - flux).pow(2).sum(-1).sqrt() / flux.pow(2).sum(-1).sqrt()).numpy()

    return {
        "eigen_pcm_median": float(np.median(pcm)),
        "eigen_pcm_p95": float(np.quantile(pcm, 0.95)),
        "eigen_pcm_max": float(pcm.max()),
        "flux_l2_median": float(np.median(l2)),
        "flux_l2_p95": float(np.quantile(l2, 0.95)),
        "flux_l2_max": float(l2.max()),
        "worst_flux_index": int(np.argmax(l2)),
    }

def physics_residual(model, norm, positions):
    """Residual magnitude of the model's own predictions, trained on it or not"""
    with torch.no_grad():
        eigen_hat, flux_hat = model(norm(positions))
        lower, diag, upper, fis_neu_prod = bands_from_input(positions.double())
        loss = residual_loss(flux_hat.double(), eigen_hat.double(), lower, diag, upper, fis_neu_prod)
    return float(loss)

def timing(model, norm, positions, n_solver=200, n_warm=20, batch=1000):
    """Time the solver and the surrogate honestly"""

    dev = next(model.parameters()).device

    poses = positions[:n_solver].cpu().numpy()
    t0 = time.perf_counter()

    for x in poses:
        slab_width, diff_coeff, macro_absorp_cross_section, fis_neu_prod = unpack(x)
        solve_arpack(diff_coeff, macro_absorp_cross_section, fis_neu_prod, slab_width)
    solver_ms = (time.perf_counter() - t0) / len(poses) * 1e3

    one = positions[:1]
    with torch.no_grad():
        for _ in range(n_warm):
            model(norm(one))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            model(norm(one))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        single_ms = (time.perf_counter() - t0) / 200 * 1e3

        xb = positions[:batch]
        for _ in range(3):
            model(norm(xb))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(norm(xb))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        batch_ms = (time.perf_counter() - t0) / 10 * 1e3


    return {
        "device": str(dev),
        "solver_ms_per_solve": solver_ms,
        "surrogate_ms_single": single_ms,
        "speedup_single": solver_ms / single_ms,
        "batch_size": batch,
        "surrogate_ms_per_sample_batched": batch_ms/batch,
        "speedup_batched": solver_ms / (batch_ms / batch),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR/"dataset.npz"))
    p.add_argument("--ood", default=None)
    p.add_argument("--model", default=str(MODEL_DIR/"surrogate_data_only.pt"))
    p.add_argument("==physics-model", default=str(MODEL_DIR/"surrogate_physics.pt"))
    args = p.parse_args()

    ensure_dirs()
    diff_coeff = np.load(args.data)
    test = diff_coeff["test_idx"]
    positions = torch.tensor(diff_coeff["positions"][test], dtype=torch.float32)
    eigenvalue = torch.tensor(diff_coeff["eigenvalue"][test], dtype=torch.float32)
    flux = torch.tensor(diff_coeff["flux"][test], dtype=torch.float32)

    out = {"n_test": len(test)}
    model, norm, ckpt = load_model(args.model)
    out["surrogate"] = accuracy(model, norm, positions, eigenvalue, flux)
    out["surrogate"]["physics_residual"] = physics_residual(model, norm, positions)
    out["timing"] = timing(model, norm, positions)

    tr = diff_coeff["train_idx"]
    out["baselines"] = fit_baselines(diff_coeff["positions"][tr], diff_coeff["eigenvalue"],
                                     diff_coeff["positions"][test], diff_coeff["eigenvalue"][test],
                                     diff_coeff["flux"][tr], diff_coeff["flux"][test])

    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dupms(out, indent=2))
    print(f"\nwrote {path}")

if __name__ == "__main__":
    main()
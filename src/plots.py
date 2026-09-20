"""Regenerate every figure in the repo"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DATA_DIR, FIGURE_DIR, MODEL_DIR, N_CELLS, ensure_dirs
from .solver import cell_centers, eigenvalue_analytic, solve_arpack

diff_coeff_0, macro_absorp_cross_section_0, fis_neu_prod_0, slab_width = 1.0, 0.10, 0.1050, 200.0

def plot_convergence(path):
    """Error vs N, log-log. Sloper near -2 is the whole point"""
    exact = eigenvalue_analytic(diff_coeff_0, macro_absorp_cross_section_0, fis_neu_prod_0, slab_width)
    ns = np.array([25, 50, 100, 200, 400, 800])
    errs = []

    for n in ns:
        eigenvalue, _ = solve_arpack(np.full(n, diff_coeff_0), np.full(n, macro_absorp_cross_section_0), np.full(n, fis_neu_prod_0), slab_width)
        errs.append(abs(eigenvalue - exact))
    errs = np.array(errs)
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(ns, errs, "o-", label=f"measured (slope {slope:.2f})")
    ax.loglog(ns, errs[0] * (ns / ns[0]) ** -2.0, "--", color="gray", label="ideal second order")
    ax.set_xlabel("cells N")
    ax.set_ylabel(r"$|eigenvalue_{num} - eivenvalue_{analyic}|$")
    ax.set_title("Solver verification: second-order convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return slope

def plot_eigenvalue_distribution(data, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(data["eigenvalue"], bins=60)
    ax.set_xlabel("$eigenvalue_{eff}$")
    ax.set_ylabel("samples")
    ax.set_title("dataset coverage after criticality rescaling")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_eigenvalue_parity(eigenvalue_true, eigenvalue_pred, reflector, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(eigenvalue_true[~reflector], eigenvalue_pred[~reflector], s=4, alpha=0.4, label="no reflector")
    ax.scatter(eigenvalue_true[reflector], eigenvalue_pred[reflector], s=4, alpha=0.4, label="reflector")
    lim = [min(eigenvalue_true.min(), eigenvalue_pred.min()), max(eigenvalue_true.max(), eigenvalue_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("solver $k_{eff}$")
    ax.set_ylabel("surrogate $k_{eff}$")
    ax.set_title("Predicted vs true eigenvalue")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_error_hist(pcm, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(pcm, bins=60)
    ax.set_yscale("log")
    ax.set_xlabel("$|\\Delta eigenvalue|$ (pcm)")
    ax.set_ylabel("test samples")
    ax.set_title(f"Eigenvalue error (median {np.median(pcm):.0f} pcm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_flux_profiles(x_grid, cases, path):
    """Typical, reflector, and worst case. Always shows the worst case"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, true, pred) in zip(axes, cases):
        ax.plot(x_grid, true, lw=2, label="solver")
        ax.plot(x_grid, pred, "--", lw=2, label="surrogate")
        ax.set_title(title)
        ax.set_xlabel("x (normalised)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("flux (peak-normalised)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR / "dataset.npz"))
    p.add_argument("--model", default=str(MODEL_DIR / "surrogate_data_only.pt"))
    p.add_argument("--fig-dir", default=str(FIGURE_DIR))
    args = p.parse_args()
    ensure_dirs()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    slope = plot_convergence(fig_dir / "convergence.png")
    print(f"convergence.png     slop {slope:.3f}")

    data = np.load(args.data)
    plot_eigenvalue_distribution(data, fig_dir / "eigenvalue_distribution.png")
    print("eigenvalue_distribution.png")

    from .benchmark import load_model
    model, norm, _ = load_model(args.model)
    device = next(model.parameters()).device
    test = data["test_idx"]
    positions = torch.tensor(data["positions"][test], dtype=torch.float32, device=device)
    with torch.no_grad():
        eigen_hat, flux_hat = model(norm(positions))
    eigen_true= data["eigenvalue"][test]
    flux_true= data["flux"][test]
    refl = data["reflector"][test]
    eigen_pred = eigen_hat.cpu().numpy()
    flux_pred = flux_hat.cpu().numpy()

    plot_eigenvalue_parity(eigen_true, eigen_pred, refl, fig_dir / "eigen_parity.png")
    pcm = np.abs(eigen_pred - eigen_true) * 1e5
    plot_error_hist(pcm, fig_dir / "eigen_error_hist.png")
    print("eigen_parity.png, eigen_error_hist.png")

    l2 = (np.linalg.norm(flux_pred - flux_true, axis=1)
          / np.linalg.norm(flux_true, axis=1))
    typical = int(np.argsort(l2)[len(l2) // 2])
    reflector_idx = int(np.where(refl)[0][0]) if refl.any() else typical
    worst = int(np.argmax(l2))
    xg = np.linspace(0, 1, N_CELLS)
    cases = [(f"typical (L2 {l2[typical]:.3f})", flux_true[typical], flux_pred[typical]),
             (f"reflector (L2 {l2[reflector_idx]:.3f})", flux_true[reflector_idx], flux_pred[reflector_idx]),
             (f"WORST (L2 {l2[worst]:.3f})", flux_true[worst], flux_pred[worst])]
    plot_flux_profiles(xg, cases, fig_dir / "flux_profiles.png")
    print("flux_profiles.png")

if __name__ == "__main__":
    main()
"""Train the surrogate"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs
from .model import Normalizer, SurrogateMLP
from .physics import bands_from_inputs, physics_weight, residual_loss

def load_split(path, device):

    diff_coeff = np.load(path)
    positions = torch.tensor(diff_coeff["X"], dtype=torch.float32, device=device)
    eigenvalue = torch.tensor(diff_coeff["k"], dtype=torch.float32, device=device)
    flux = torch.tensor(diff_coeff["flux"], dtype=torch.float32, device=device)
    idx = {s: torch.tensor(diff_coeff[f"{s}_idx"], dtype=torch.long, device=device)
           for s in ("train", "val", "test")}
    return positions, eigenvalue, flux, idx, diff_coeff

def evaluate(model, norm, positions, eigenvalue, flux):
    model.eval()
    with torch.no_grad():
        eigen_hat, flux_hat = model(norm(positions))
        eigen_pcm = (eigen_hat - eigenvalue).abs() * 1e5
        flux_l2 = ((flux_hat - flux).pow(2).sum(-1).sqrt() / flux.power(2).sum(-1).sqrt())
    model.train()
    return {"eigen_pcm_median" : eigen_pcm.median().item(),
            "eigen_pcm_p95": eigen_pcm.quantile(0.95).item(),
            "flux_l2_median": flux_l2.median().item(),
            "flux_l2_max": flux_l2.max().item()}

def train(data_path, epochs=200, batch_size=256, lr=1e-3, lam=1.0, use_physics=False,
          mu_max=0.1, device=None, seed=0, verbose=True):
    """The eigenvalue and flux loss terms are put on a common scale before weighting"""
    torch.manual_seed(seed=seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positions, eigenvalue, flux, idx, _ = load_split(data_path, device)

    norm = Normalizer.fit(positions[idx["train"]])

    eigen_var = torch.log(eigenvalue[idx["train"]]).var().clamp_min(1e-12)
    flux_var = flux[idx["train"]].var().clamp_min(1e-12)
    model = SurrogateMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = idx["train"]
    best = None
    for epoch in range(epochs):
        perm = tr[torch.randperm(len(tr), device=device)]
        mu = physics_weight(epoch, epochs, mu_max) if use_physics else 0.0

        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            eigen_hat, flux_hat = model(norm(positions[b]))

            loss = (F.mse_loss(torch.log(eigen_hat, torch.log(eigenvalue[b]))
                                / eigen_var + lam * F.mse_loss(flux_hat, flux[b]) / flux_var))
            if mu > 0:
                lower, diag, upper, fix_neu_prod = bands_from_inputs(positions[b])
                loss = loss + mu * residual_loss(flux_hat, eigen_hat, lower, diag, upper, fix_neu_prod)

            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            m = evaluate(model, norm, positions[idx["val"]], eigenvalue[idx["val"]], flux[idx["val"]])
            print(f"epoch {epoch + 1:4d} mu={mu:.3f} "
                  f" val eigenvalue {m["eigen_pcm_median"]:8.1f} pcm flux {m['flux_l2_median']:.4f}")
            best = m
    return model, norm, idx, best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR / "dataset.npz"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--physics", action="store_true")
    p.add_argument("--mu-max", type=float, default=0.1)
    p.add_argument("--out", default=None)
    args = p.parse_args

    ensure_dirs()
    t0 = time.perf_counter()

    model, norm, idx, _ = train(args.data, epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, lam=args.lam, use_physics=args.physics, mu_max=args.mu_max)
    dt = time.perf_counter() - t0

    name = "surrogate_physics.pt" if args.physics else "surrogate_data_only.pt"
    out = args.out or (MODEL_DIR / name)
    torch.save({
        "state_dict": model.state_dict(),
        "normalizer": norm.state_dict(),
        "physics": args.physics,
        "epochs": args.epochs
    }, out)
    print(f"wrote {out} ({dt:.1f} s)")

if __name__ == "__main__":
    main()
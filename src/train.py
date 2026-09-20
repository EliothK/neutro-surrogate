"""Train the surrogate"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import (DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_EPOCHS, DEFAULT_LAM, DEFAULT_LR,
                     DEFAULT_MU_MAX, DEFAULT_WIDTH, DEFAULT_RAMP_FRAC, DEFAULT_WEIGHT_DECAY,
                     DEFAULT_DROPOUT, DEFAULT_ACTIVATION, DEFAULT_SCHEDULER, DEFAULT_TRAIN_FRACTION, DEFAULT_GRAD_CLIP, MODEL_DIR, RESULTS_DIR, ensure_dirs)
from .model import Normalizer, SurrogateMLP
from .physics import bands_from_input, physics_weight, residual_loss

def load_split(path, device):

    dataset = np.load(path)
    positions = torch.tensor(dataset["positions"], dtype=torch.float32, device=device)
    eigenvalue = torch.tensor(dataset["eigenvalue"], dtype=torch.float32, device=device)
    flux = torch.tensor(dataset["flux"], dtype=torch.float32, device=device)
    idx = {s: torch.tensor(dataset[f"{s}_idx"], dtype=torch.long, device=device)
           for s in ("train", "val", "test")}
    return positions, eigenvalue, flux, idx, dataset

def evaluate(model, norm, positions, eigenvalue, flux):
    model.eval()
    with torch.no_grad():
        eigen_hat, flux_hat = model(norm(positions))
        eigen_pcm = (eigen_hat - eigenvalue).abs() * 1e5
        flux_l2 = ((flux_hat - flux).pow(2).sum(-1).sqrt() / flux.pow(2).sum(-1).sqrt())
    model.train()
    return {"eigen_pcm_median" : eigen_pcm.median().item(),
            "eigen_pcm_p95": eigen_pcm.quantile(0.95).item(),
            "flux_l2_median": flux_l2.median().item(),
            "flux_l2_max": flux_l2.max().item()}

def train(data_path, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR, lam=DEFAULT_LAM,
          use_physics=False, mu_max=DEFAULT_MU_MAX, width=DEFAULT_WIDTH, depth=DEFAULT_DEPTH,
          ramp_frac=DEFAULT_RAMP_FRAC, weight_decay=DEFAULT_WEIGHT_DECAY, dropout=DEFAULT_DROPOUT,
          activation=DEFAULT_ACTIVATION, scheduler=DEFAULT_SCHEDULER, train_fraction=DEFAULT_TRAIN_FRACTION,
          grad_clip=DEFAULT_GRAD_CLIP, split=None, device=None, seed=0, verbose=True):
    """The eigenvalue and flux loss terms are put on a common scale before weighting"""
    torch.manual_seed(seed=seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positions, eigenvalue, flux, idx, _ = load_split(data_path, device)
    if split is not None:  # e.g. a cross-validation fold: {"train": LongTensor, "val": LongTensor}
        idx = {**idx, **split}

    norm = Normalizer.fit(positions[idx["train"]])

    eigen_var = torch.log(eigenvalue[idx["train"]]).var().clamp_min(1e-12)
    flux_var = flux[idx["train"]].var().clamp_min(1e-12)
    model = SurrogateMLP(width=width, depth=depth, activation=activation, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = {"cosine": lambda: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs),
             "linear": lambda: torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=epochs),
             "none": lambda: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)}[scheduler]()

    tr = idx["train"][:max(1, int(train_fraction * len(idx["train"])))]
    best = None
    for epoch in range(epochs):
        perm = tr[torch.randperm(len(tr), device=device)]
        mu = physics_weight(epoch, epochs, mu_max, ramp_frac) if use_physics else 0.0

        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            eigen_hat, flux_hat = model(norm(positions[b]))

            loss = (F.mse_loss(torch.log(eigen_hat), torch.log(eigenvalue[b])) / eigen_var
                    + lam * F.mse_loss(flux_hat, flux[b]) / flux_var)
            if mu > 0:
                lower, diag, upper, fis_neu_prod = bands_from_input(positions[b])
                loss = loss + mu * residual_loss(flux_hat, eigen_hat, lower, diag, upper, fis_neu_prod)

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lam", type=float, default=DEFAULT_LAM)
    p.add_argument("--physics", action="store_true")
    p.add_argument("--mu-max", type=float, default=DEFAULT_MU_MAX)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--ramp-frac", type=float, default=DEFAULT_RAMP_FRAC)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--activation", choices=("silu", "relu", "gelu", "tanh"), default=DEFAULT_ACTIVATION)
    p.add_argument("--scheduler", choices=("cosine", "linear", "none"), default=DEFAULT_SCHEDULER)
    p.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    p.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ensure_dirs()
    t0 = time.perf_counter()

    model, norm, idx, _ = train(args.data, epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.lr, lam=args.lam, use_physics=args.physics, mu_max=args.mu_max,
                                width=args.width, depth=args.depth, ramp_frac=args.ramp_frac,
                                weight_decay=args.weight_decay, dropout=args.dropout, activation=args.activation,
                                scheduler=args.scheduler, train_fraction=args.train_fraction, grad_clip=args.grad_clip)
    dt = time.perf_counter() - t0

    name = "surrogate_physics.pt" if args.physics else "surrogate_data_only.pt"
    out = args.out or (MODEL_DIR / name)
    torch.save({
        "state_dict": model.state_dict(),
        "normalizer": norm.state_dict(),
        "physics": args.physics,
        "epochs": args.epochs,
        "width": args.width,
        "depth": args.depth,
        "activation": args.activation,
        "dropout": args.dropout,
    }, out)
    print(f"wrote {out} ({dt:.1f} s)")

if __name__ == "__main__":
    main()
"""Hybrid prediction: surrogate flux, k from its Rayleigh quotient, and a residual gate that sends doubtful samples to the solver.

The operator is symmetric, so the Rayleigh quotient's k error is quadratic in the flux error: an accurate flux gives a much better k than the network's own eigenvalue head.
Optional inverse-power steps (one tridiagonal solve each) polish the flux further; they converge as k2/k1, so they cannot rescue near-degenerate cores.
In those cores the network usually gets the shape within each region right but puts the power in the wrong region.
A Rayleigh-Ritz step over the predicted flux times m hat functions re-fits those amplitudes; it is variational, so it can only move k towards the true value.
A residual gate can still send doubtful samples to the solver: a large residual ||A phi - F phi / k|| / ||F phi / k|| flags a flux that does not satisfy the equation.

Run with `python -m src.hybrid --model models/tuned_dataset_rescaled_physics.pt --data data/dataset_rescaled.npz --refine 1 --ritz 17 --fallback 0`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve_banded

from .benchmark import load_model
from .config import DATA_DIR, RESULTS_DIR, ensure_dirs
from .metrics import eigenvalue_report, flux_report
from .physics import bands_from_input, tridiag_matvec
from .solver import solve
from .train import load_split

SMALL_BATCH = 64  # below this, refinement runs per sample in scipy rather than as a batched Thomas loop


def rayleigh_residual(positions, flux):
    """Rayleigh-quotient k and relative residual for a batch of fluxes; positions are raw (unnormalised) inputs."""
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    a_flux = tridiag_matvec(lower, diag, upper, flux)
    f_flux = fis_neu_prod * flux
    k = (flux * f_flux).sum(-1) / (flux * a_flux).sum(-1)
    source = f_flux / k.unsqueeze(-1)
    resid = (a_flux - source).norm(dim=-1) / source.norm(dim=-1).clamp_min(1e-30)
    return k, resid


def thomas(lower, diag, upper, rhs):
    """Batched tridiagonal solve A x = rhs, all tensors (B, N); A is the diffusion operator, which is diagonally dominant, so no pivoting is needed."""
    n = diag.shape[-1]
    c, d = torch.empty_like(diag), torch.empty_like(rhs)
    c[:, 0] = upper[:, 0] / diag[:, 0]
    d[:, 0] = rhs[:, 0] / diag[:, 0]
    for i in range(1, n):
        m = diag[:, i] - lower[:, i] * c[:, i - 1]
        c[:, i] = upper[:, i] / m
        d[:, i] = (rhs[:, i] - lower[:, i] * d[:, i - 1]) / m
    x = torch.empty_like(rhs)
    x[:, -1] = d[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d[:, i] - c[:, i] * x[:, i + 1]
    return x


def inverse_power_steps(positions, flux, steps, small_batch=SMALL_BATCH):
    """Apply `steps` inverse-power iterations phi <- A^-1 F phi, renormalised to a peak of 1.
    Batches below small_batch use scipy's compiled banded solve per sample, since the batched Thomas loop costs milliseconds in per-cell overhead however few samples it carries."""
    if steps == 0:
        return flux
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    if len(flux) < small_batch:
        ab = torch.stack([F.pad(upper[:, :-1], (1, 0)), diag, F.pad(lower[:, 1:], (0, 1))], dim=1).cpu().numpy()
        nsf, out = fis_neu_prod.cpu().numpy(), flux.cpu().numpy().copy()
        for i in range(len(out)):
            for _ in range(steps):
                out[i] = solve_banded((1, 1), ab[i], nsf[i] * out[i])
                out[i] /= out[i].max()
        return torch.as_tensor(out, dtype=flux.dtype, device=flux.device)
    for _ in range(steps):
        flux = thomas(lower, diag, upper, fis_neu_prod * flux)
        flux = flux / flux.amax(dim=-1, keepdim=True)
    return flux


def hat_functions(m, n_cells, dtype=torch.float64, device=None):
    """m piecewise-linear hat functions on the cell centres, forming a partition of unity; shape (m, n_cells)."""
    x = (torch.arange(n_cells, dtype=dtype, device=device) + 0.5) / n_cells
    nodes = torch.linspace(0, 1, m, dtype=dtype, device=device)
    return (1 - (x[None, :] - nodes[:, None]).abs() * (m - 1)).clamp_min(0)


def ritz(positions, flux, m):
    """Rayleigh-Ritz on span{flux * hat_j, j = 1..m}: keeps the predicted shape within each hat and re-fits the amplitudes between them.
    Returns (k, flux) for the top Ritz pair; k is variational, so it never exceeds the true k and is never below the Rayleigh quotient of `flux` itself."""
    lower, diag, upper, fis_neu_prod = bands_from_input(positions)
    batch, n = flux.shape
    V = flux[:, None, :] * hat_functions(m, n, flux.dtype, flux.device)[None]
    rep = lambda t: t.repeat_interleave(m, 0)
    AV = tridiag_matvec(rep(lower), rep(diag), rep(upper), V.reshape(batch * m, n)).reshape(batch, m, n)
    A_m = V @ AV.transpose(1, 2)
    F_m = V @ (fis_neu_prod[:, None, :] * V).transpose(1, 2)
    # reduce the generalized problem F c = k A c to a standard one with the Cholesky factor of A_m (symmetric positive definite)
    jitter = 1e-14 * A_m.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(m, dtype=A_m.dtype, device=A_m.device)
    L_inv = torch.linalg.inv(torch.linalg.cholesky(A_m + jitter))
    S_m = L_inv @ F_m @ L_inv.transpose(1, 2)
    if S_m.is_cuda and m > 32:  # batched GPU eigh falls off a cliff above 32 x 32 (about 25x slower than CPU at m = 33)
        w, U = (t.to(S_m.device) for t in torch.linalg.eigh(S_m.cpu()))
    else:
        w, U = torch.linalg.eigh(S_m)
    coef = (L_inv.transpose(1, 2) @ U[:, :, -1:]).squeeze(-1)
    out = (coef[:, :, None] * V).sum(1)
    out = out * torch.sign(out.sum(-1, keepdim=True))
    return w[:, -1], out / out.amax(-1, keepdim=True)


def predict(model, norm, positions, refine=0, ritz_m=0, threshold=None):
    """Hybrid prediction for raw inputs `positions` (B, N_INPUTS).
    The flux goes through `refine` inverse-power steps; with ritz_m > 0 it then gets a Rayleigh-Ritz amplitude re-fit and one more inverse-power step.
    Returns a dict of tensors: k (Rayleigh, or solver where gated), flux, k_direct (network head), residual and fallback (bool mask).
    With threshold=None nothing is sent to the solver."""
    model.eval()
    with torch.no_grad():
        k_direct, flux = model(norm(positions))
        positions, flux = positions.double(), flux.double()
        flux = inverse_power_steps(positions, flux, refine)
        if ritz_m > 0:
            flux = inverse_power_steps(positions, ritz(positions, flux, ritz_m)[1], 1)
        k, resid = rayleigh_residual(positions, flux)
    out = {"k": k, "flux": flux, "k_direct": k_direct.double(), "residual": resid}
    return apply_gate(positions, out, threshold)


def predict_max(members, positions, refine=0, ritz_m=0, threshold=None):
    """Run every (model, norm) in `members` through `predict` and keep, per sample, the one with the largest k.
    Rayleigh and Ritz values are lower bounds on the true k, so the largest is always the closest: the combination is never worse than its best member.
    Adds `chosen`, the index of the member used for each sample; the gate is applied after the choice."""
    outs = [predict(model, norm, positions, refine=refine, ritz_m=ritz_m) for model, norm in members]
    chosen = torch.stack([o["k"] for o in outs]).argmax(0)
    rows = torch.arange(len(chosen), device=chosen.device)
    pick = lambda key: torch.stack([o[key] for o in outs])[chosen, rows]
    out = {key: pick(key) for key in ("k", "flux", "k_direct", "residual")}
    out["chosen"] = chosen
    return apply_gate(positions.double(), out, threshold)


def apply_gate(positions, out, threshold):
    """Replace samples whose residual exceeds `threshold` with the solver's answer and record them in out["fallback"]; threshold=None sends nothing."""
    resid = out["residual"]
    fallback = resid > threshold if threshold is not None else torch.zeros_like(resid, dtype=torch.bool)
    for i in torch.nonzero(fallback).flatten().tolist():
        k_i, flux_i = solve(positions[i].cpu().numpy())
        out["k"][i] = k_i
        out["flux"][i] = torch.as_tensor(flux_i, dtype=out["flux"].dtype, device=out["flux"].device)
    out["fallback"] = fallback
    return out


def score(out, eigenvalue, flux):
    """Full metric reports (R^2, MSE, RMSE, MAE, MAPE, error percentiles) for a `predict` result."""
    k_true, flux_true = eigenvalue.cpu().numpy(), flux.cpu().numpy()
    return {"fallback_fraction": float(out["fallback"].float().mean()),
            "k": eigenvalue_report(k_true, out["k"].cpu().numpy()),
            "k_direct": eigenvalue_report(k_true, out["k_direct"].cpu().numpy()),
            "flux": flux_report(flux_true, out["flux"].cpu().numpy())}


def timing(members, positions, threshold, refine=1, ritz_m=0, n_solver=200, repeats=5):
    """Wall-clock cost per sample of each hybrid stage for the (model, norm) pairs in `members`, against the ARPACK solver on the same inputs.
    Stages are cumulative: networks, then Rayleigh k and residual, then refinement, then Ritz (if ritz_m), then the gate with its solver calls (if threshold)."""
    dev = positions.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    xs = positions[:n_solver].cpu().numpy()
    t0 = time.perf_counter()
    for x in xs:
        solve(x)
    solver_ms = (time.perf_counter() - t0) / len(xs) * 1e3

    def per_sample_ms(fn, xb):
        fn(xb)  # warm-up
        sync()
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(xb)
        sync()
        return (time.perf_counter() - t0) / repeats / len(xb) * 1e3

    def net(xb):
        with torch.no_grad():
            return [model(norm(xb)) for model, norm in members]

    stages = {"network": net,
              "+ rayleigh k and residual": lambda xb: predict_max(members, xb, refine=0),
              f"+ {refine} inverse-power step(s)": lambda xb: predict_max(members, xb, refine=refine)}
    if ritz_m:
        stages[f"+ ritz m={ritz_m} and 1 more step"] = lambda xb: predict_max(members, xb, refine=refine, ritz_m=ritz_m)
    if threshold is not None:
        stages["+ residual gate (solver fallback)"] = lambda xb: predict_max(members, xb, refine=refine, ritz_m=ritz_m, threshold=threshold)
    report = {"device": str(dev), "n_models": len(members), "solver_ms_per_sample": solver_ms, "stages": {}}
    for batch in (1, len(positions)):
        xb = positions[:batch]
        fallback = float(predict_max(members, xb, refine=refine, ritz_m=ritz_m, threshold=threshold)["fallback"].float().mean())
        rows = {}
        for name, fn in stages.items():
            ms = per_sample_ms(fn, xb) if batch > 1 else min(per_sample_ms(fn, positions[i:i + 1]) for i in range(20))
            rows[name] = {"ms_per_sample": ms, "speedup_vs_solver": solver_ms / ms}
        report["stages"][f"batch_{batch}"] = {"fallback_fraction": fallback, **rows}
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, nargs="+", help="one or more checkpoints; with several, each sample keeps the largest k")
    p.add_argument("--data", default=str(DATA_DIR / "dataset_rescaled.npz"))
    p.add_argument("--refine", type=int, default=1, help="inverse-power steps applied to the surrogate flux")
    p.add_argument("--ritz", type=int, default=0, help="hat functions for the Rayleigh-Ritz amplitude re-fit (0 = off; 17 or 33 work well)")
    p.add_argument("--fallback", type=float, default=0.05, help="fraction of val sent to the solver; sets the residual threshold (0 = no gate)")
    p.add_argument("--calibrate-on", default="val", help="split used to set the threshold; use a split the model did not train on")
    p.add_argument("--eval-data", default=None, help="score the test split of this dataset instead (e.g. an --ood set); the threshold is still set on --data")
    p.add_argument("--timing", action="store_true", help="also time each stage against the solver on the test split")
    p.add_argument("--results", default=None)
    args = p.parse_args()

    ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = [load_model(path, device) for path in args.model]
    members = [(model, norm) for model, norm, _ in loaded]
    positions, eigenvalue, flux, idx, _ = load_split(args.data, device)
    for path, (_, _, ckpt) in zip(args.model, loaded):
        if ckpt.get("trained_on") == "train+val" and args.calibrate_on == "val":
            print(f"warning: {path} was trained on train+val, so a val-calibrated threshold is set on in-sample data")

    cal = idx[args.calibrate_on]
    resid_cal = predict_max(members, positions[cal], refine=args.refine, ritz_m=args.ritz)["residual"].cpu().numpy()
    threshold = float(np.quantile(resid_cal, 1.0 - args.fallback)) if args.fallback > 0 else None

    if args.eval_data:
        positions, eigenvalue, flux, idx, _ = load_split(args.eval_data, device)
    te = idx["test"]
    report = {"model": args.model, "data": args.data, "eval_data": args.eval_data or args.data, "refine": args.refine,
              "ritz": args.ritz, "threshold": threshold, "calibrated_on": args.calibrate_on, "target_fallback": args.fallback}
    runs = (("surrogate_only", None),) + ((("gated", threshold),) if threshold is not None else ())
    for name, thr in runs:
        out = predict_max(members, positions[te], refine=args.refine, ritz_m=args.ritz, threshold=thr)
        report[name] = score(out, eigenvalue[te].double(), flux[te].double())
        report[name]["chosen_fraction"] = [float((out["chosen"] == i).float().mean()) for i in range(len(members))]
        r = report[name]
        print(f"{name:15s} fallback {r['fallback_fraction']:.1%} | flux R2 {r['flux']['r2']:.5f} RMSE {r['flux']['rmse']:.5f} MAE {r['flux']['mae']:.5f} "
              f"MAPE {r['flux']['mape']:.2f}% rel-L2 mean {r['flux']['rel_l2_mean']:.4f} p99 {r['flux']['rel_l2_p99']:.4f} | "
              f"k R2 {r['k']['r2']:.6f} RMSE {r['k']['rmse_pcm']:.1f} MAE {r['k']['mae_pcm']:.1f} median {r['k']['median_abs_pcm']:.2f} "
              f"p99 {r['k']['p99_abs_pcm']:.1f} pcm MAPE {r['k']['mape']:.4f}%")
        if len(members) > 1:
            print(f"{'':15s} chosen per model: " + ", ".join(f"{frac:.1%}" for frac in r["chosen_fraction"]))
    if args.timing:
        report["timing"] = timing(members, positions[te], threshold, refine=args.refine, ritz_m=args.ritz)
        print(f"solver {report['timing']['solver_ms_per_sample']:.3f} ms/sample on {report['timing']['device']}")
        for batch, rows in report["timing"]["stages"].items():
            print(f"{batch} (fallback {rows['fallback_fraction']:.1%})")
            for name, r in rows.items():
                if name != "fallback_fraction":
                    print(f"  {name:36s} {r['ms_per_sample']:.5f} ms/sample  speedup {r['speedup_vs_solver']:8.1f}x")
    out = args.results or RESULTS_DIR / f"hybrid_{'+'.join(Path(m).stem for m in args.model)}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

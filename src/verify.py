"""Does the solver obey the physics, and does the surrogate?

A. Solver invariants (numpy only): properties any correct diffusion eigen-solver must satisfy:
   mirror symmetry, exact linearity in the fission multiplier, monotonicity in absorption and fission, neutron balance (production = k x (absorption + leakage)), the k <= max(nSf/Sa) bound, and the homogeneous-slab analytic answer.
B. Surrogate consistency: the same properties measured on the model's predictions.
C. Out-of-distribution accuracy: R^2 / RMSE / MAE against fresh solver labels drawn outside the training box.

Run with `python -m src.verify --model models/tuned_dataset_rescaled_physics.pt --data data/dataset_rescaled.npz --ood data/ood_rescaled.npz`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .config import (DATA_DIR, N_CELLS, RESULTS_DIR, SLICE_DIFF_COEFFICIENT, SLICE_FIS_NEU_PROD,
                     SLICE_MACRO_ABSORP_CROSS_SECTION, ensure_dirs)
from .metrics import format_report, full_report
from .solver import eigenvalue_analytic, solve_arpack, unpack, zone_to_cells

ZONE_SLICES = (SLICE_DIFF_COEFFICIENT, SLICE_MACRO_ABSORP_CROSS_SECTION, SLICE_FIS_NEU_PROD)
BUMP = 1.05  # 5% perturbation for the monotonicity and scaling probes

SOLVER_TOLERANCE = {"mirror_k_rel": 1e-8, "mirror_flux_max": 1e-6, "fission_scaling_k_rel": 1e-8,
                    "fission_scaling_flux_max": 1e-8, "balance_rel": 1e-8, "homogeneous_k_rel": 1e-4,
                    "homogeneous_flux_max": 1e-3}


def mirror_positions(P):
    """Reverse every per-zone profile (slab flipped left to right). P is (N, N_INPUTS) or (N_INPUTS,)."""
    out = np.array(P, dtype=float, copy=True)
    for sl in ZONE_SLICES:
        out[..., sl] = out[..., sl][..., ::-1]
    return out


def scale_block(P, sl, factor):
    out = np.array(P, dtype=float, copy=True)
    out[..., sl] = out[..., sl] * factor
    return out


def solve_row(x):
    slab_width, D, Sa, nSf = unpack(x)
    k, flux = solve_arpack(D, Sa, nSf, slab_width)
    return k, flux, (slab_width, D, Sa, nSf)


def solver_invariants(P):
    """Worst-case deviation of every solver invariant over the rows of P. Small numbers are good."""
    worst = {key: 0.0 for key in ("mirror_k_rel", "mirror_flux_max", "fission_scaling_k_rel", "fission_scaling_flux_max",
                                  "balance_rel")}
    violations = {"absorption_monotone": 0, "fission_monotone": 0, "k_bound": 0}
    for x in P:
        k, flux, (w, D, Sa, nSf) = solve_row(x)

        km, fm, _ = solve_row(mirror_positions(x))
        worst["mirror_k_rel"] = max(worst["mirror_k_rel"], abs(km - k) / k)
        worst["mirror_flux_max"] = max(worst["mirror_flux_max"], float(np.abs(fm - flux[::-1]).max()))

        ks, fs, _ = solve_row(scale_block(x, SLICE_FIS_NEU_PROD, BUMP))
        worst["fission_scaling_k_rel"] = max(worst["fission_scaling_k_rel"], abs(ks / BUMP - k) / k)
        worst["fission_scaling_flux_max"] = max(worst["fission_scaling_flux_max"], float(np.abs(fs - flux).max()))

        # Balance: summing the discrete equations over all cells, interior leakage cancels, leaving sum(nSf phi) = k * (sum(Sa phi) + boundary leakage 2 D phi / h^2 at both faces).
        h = w / len(D)
        absorption_plus_leak = np.sum(Sa * flux) + 2 * D[0] * flux[0] / h ** 2 + 2 * D[-1] * flux[-1] / h ** 2
        production = np.sum(nSf * flux)
        worst["balance_rel"] = max(worst["balance_rel"], abs(production - k * absorption_plus_leak) / production)

        if solve_row(scale_block(x, SLICE_MACRO_ABSORP_CROSS_SECTION, BUMP))[0] >= k:
            violations["absorption_monotone"] += 1
        if solve_row(scale_block(x, SLICE_FIS_NEU_PROD, BUMP))[0] <= k:
            violations["fission_monotone"] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            if k > np.nanmax(np.where(Sa > 0, nSf / Sa, 0.0)) * (1 + 1e-9):
                violations["k_bound"] += 1
    return {"worst": worst, "violations": violations, "n": len(P)}


def homogeneous_check(width=200.0, D=1.0, Sa=0.10, nSf=0.105, n_cells=N_CELLS):
    """Uniform slab: the solver must reproduce the analytic k and the cosine flux."""
    k, flux = solve_arpack(np.full(n_cells, D), np.full(n_cells, Sa), np.full(n_cells, nSf), width)
    x = (np.arange(n_cells) + 0.5) * width / n_cells
    return {"homogeneous_k_rel": abs(k - eigenvalue_analytic(D, Sa, nSf, width)) / k,
            "homogeneous_flux_max": float(np.abs(flux - np.sin(np.pi * x / width)).max())}


def check_solver(P):
    res = solver_invariants(P)
    res["worst"].update(homogeneous_check())
    res["tolerance"] = SOLVER_TOLERANCE
    res["passed"] = (all(res["worst"][k] <= SOLVER_TOLERANCE[k] for k in SOLVER_TOLERANCE)
                     and not any(res["violations"].values()))
    return res


# ---- surrogate ------------------------------------------------------------------------------------------------------

def predict(model, norm, P, device, batch=4096):
    import torch
    ks, fs = [], []
    with torch.no_grad():
        for i in range(0, len(P), batch):
            x = torch.tensor(np.asarray(P[i:i + batch]), dtype=torch.float32, device=device)
            k, f = model(norm(x))
            ks.append(k.cpu().numpy())
            fs.append(f.cpu().numpy())
    return np.concatenate(ks).astype(float), np.concatenate(fs).astype(float)


def surrogate_consistency(model, norm, P, device):
    """The same physics properties measured on the model's predictions."""
    k0, f0 = predict(model, norm, P, device)
    km, fm = predict(model, norm, mirror_positions(P), device)
    k_up_fis, _ = predict(model, norm, scale_block(P, SLICE_FIS_NEU_PROD, BUMP), device)
    k_up_abs, _ = predict(model, norm, scale_block(P, SLICE_MACRO_ABSORP_CROSS_SECTION, BUMP), device)
    Sa = P[:, SLICE_MACRO_ABSORP_CROSS_SECTION]
    nSf = P[:, SLICE_FIS_NEU_PROD]
    bound = np.max(np.where(Sa > 0, nSf / np.where(Sa > 0, Sa, 1.0), 0.0), axis=1)

    mirror_flux_l2 = np.linalg.norm(fm - f0[:, ::-1], axis=1) / np.linalg.norm(f0, axis=1)
    log_sens = np.log(k_up_fis / k0) / np.log(BUMP)  # the exact solver gives 1.0
    return {
        "n": len(P),
        "mirror_k_median_pcm": float(np.median(np.abs(km - k0)) * 1e5),
        "mirror_k_p95_pcm": float(np.quantile(np.abs(km - k0), 0.95) * 1e5),
        "mirror_flux_rel_l2_median": float(np.median(mirror_flux_l2)),
        "mirror_flux_rel_l2_p95": float(np.quantile(mirror_flux_l2, 0.95)),
        "fission_sensitivity_median": float(np.median(log_sens)),
        "fission_sensitivity_abs_err_p95": float(np.quantile(np.abs(log_sens - 1.0), 0.95)),
        "fission_monotone_violation_rate": float(np.mean(k_up_fis <= k0)),
        "absorption_monotone_violation_rate": float(np.mean(k_up_abs >= k0)),
        "k_bound_violation_rate": float(np.mean(k0 > bound * (1 + 1e-9))),
        "flux_negative_fraction": float(np.mean(f0 < 0)),
        "flux_peak_is_one": bool(np.allclose(f0.max(axis=1), 1.0, atol=1e-5)),
    }


def load_rows(path, split="test", limit=None, seed=0):
    d = np.load(path)
    idx = d[f"{split}_idx"]
    if limit and len(idx) > limit:
        idx = np.random.default_rng(seed).choice(idx, size=limit, replace=False)
    return d["positions"][idx], d["eigenvalue"][idx], d["flux"][idx]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="checkpoint (.pt); omit to check only the solver")
    p.add_argument("--data", default=str(DATA_DIR / "dataset_rescaled.npz"))
    p.add_argument("--ood", default=None, help="out-of-distribution dataset from `make_dataset --ood`")
    p.add_argument("--n", type=int, default=200, help="rows used for the solver and consistency probes")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ensure_dirs()

    P, _, _ = load_rows(args.data, "test", args.n)
    result = {"data": args.data, "model": args.model}

    print(f"A. solver invariants on {len(P)} test rows")
    result["solver"] = check_solver(P)
    for key, tol in SOLVER_TOLERANCE.items():
        v = result["solver"]["worst"][key]
        print(f"   {key:28s} {v:10.2e}  (tol {tol:.0e})  {'ok' if v <= tol else 'FAIL'}")
    print(f"   monotonicity / bound violations: {result['solver']['violations']}")
    print(f"   solver passed: {result['solver']['passed']}")

    if args.model:
        import torch
        from .benchmark import load_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, norm, ckpt = load_model(args.model, device=device)
        in_sample = {"train", "val"} if (ckpt.get("trained_on") == "train+val" or "config" in ckpt) else {"train"}

        print(f"\nB. surrogate consistency on the same {len(P)} rows (physics the model was never told)")
        result["surrogate_consistency"] = surrogate_consistency(model, norm, P, device)
        for key, v in result["surrogate_consistency"].items():
            print(f"   {key:36s} {v}")

        print("\nC. accuracy (R^2, RMSE, MAE, bias, fit slope) against solver labels")
        result["accuracy"] = {}
        for split in ("train", "val", "test"):
            Ps, ks, fs = load_rows(args.data, split, 3000)
            if len(Ps):
                kp, fp = predict(model, norm, Ps, device)
                result["accuracy"][split] = full_report(ks, kp, fs, fp)
                print(format_report(f"{split} (in-sample)" if split in in_sample else f"{split} (held out)",
                                    result["accuracy"][split]))
        if args.ood:
            Po, ko, fo = load_rows(args.ood, "test", 3000)
            kp, fp = predict(model, norm, Po, device)
            result["accuracy"]["out_of_distribution"] = full_report(ko, kp, fo, fp)
            print(format_report("OUT-of-distribution", result["accuracy"]["out_of_distribution"]))

    out = Path(args.out or RESULTS_DIR / f"verify_{Path(args.model).stem if args.model else 'solver'}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

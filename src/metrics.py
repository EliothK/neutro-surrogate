"""Regression metrics used everywhere a model is compared with the solver: R^2, RMSE, MAE, bias, fit slope, ...

`eigenvalue_report` and `flux_report` return plain floats so results drop straight into json.
"""

import numpy as np

PCM = 1e5  # 1 pcm = 1e-5 in k


def regression_report(y_true, y_pred):
    """Pooled metrics over flat arrays. A perfect fit has r2 = 1, slope = 1, intercept = 0, bias = 0.
    MAPE (in %) skips entries whose true value is below 1e-3 of the largest one, since the flux goes to zero at the slab edges and would make it blow up."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    slope, intercept = np.polyfit(y_true, y_pred, 1) if np.ptp(y_true) > 0 else (float("nan"), float("nan"))
    nonzero = np.abs(y_true) > 1e-3 * np.abs(y_true).max()
    return {"n": int(y_true.size),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "mse": float(np.mean(err ** 2)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mape": float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan"),
            "max_abs_err": float(np.max(np.abs(err))),
            "bias": float(np.mean(err)),
            "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.ptp(y_true) > 0 and np.ptp(y_pred) > 0 else float("nan"),
            "slope": float(slope), "intercept": float(intercept)}


def eigenvalue_report(k_true, k_pred):
    """k_eff metrics: the pooled regression report plus the error distribution in pcm and relative %."""
    k_true, k_pred = np.asarray(k_true, float), np.asarray(k_pred, float)
    rep = regression_report(k_true, k_pred)
    abs_pcm = np.abs(k_pred - k_true) * PCM
    rep.update({"rmse_pcm": rep["rmse"] * PCM, "mae_pcm": rep["mae"] * PCM, "bias_pcm": rep["bias"] * PCM,
                "median_abs_pcm": float(np.median(abs_pcm)), "p95_abs_pcm": float(np.quantile(abs_pcm, 0.95)),
                "p99_abs_pcm": float(np.quantile(abs_pcm, 0.99)), "max_abs_pcm": float(abs_pcm.max()),
                "median_rel_err_pct": float(np.median(np.abs(k_pred - k_true) / k_true) * 100)})
    return rep


def flux_report(flux_true, flux_pred):
    """Flux metrics: pooled over every cell, plus per-profile relative L2, per-profile R^2 and peak position error."""
    flux_true, flux_pred = np.asarray(flux_true, float), np.asarray(flux_pred, float)
    rep = regression_report(flux_true, flux_pred)
    rel_l2 = np.linalg.norm(flux_pred - flux_true, axis=1) / np.linalg.norm(flux_true, axis=1)
    ss_res = np.sum((flux_pred - flux_true) ** 2, axis=1)
    ss_tot = np.sum((flux_true - flux_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2_each = 1.0 - ss_res / np.maximum(ss_tot, 1e-30)
    peak_err = np.abs(np.argmax(flux_pred, axis=1) - np.argmax(flux_true, axis=1))
    rep.update({"rel_l2_mean": float(rel_l2.mean()), "rel_l2_median": float(np.median(rel_l2)),
                "rel_l2_p95": float(np.quantile(rel_l2, 0.95)), "rel_l2_p99": float(np.quantile(rel_l2, 0.99)),
                "rel_l2_max": float(rel_l2.max()),
                "profile_r2_median": float(np.median(r2_each)), "profile_r2_p05": float(np.quantile(r2_each, 0.05)),
                "profile_r2_min": float(r2_each.min()),
                "peak_cell_err_median": float(np.median(peak_err)), "peak_cell_err_p95": float(np.quantile(peak_err, 0.95)),
                "negative_fraction": float(np.mean(flux_pred < 0))})
    return rep


def full_report(k_true, k_pred, flux_true, flux_pred):
    return {"eigenvalue": eigenvalue_report(k_true, k_pred), "flux": flux_report(flux_true, flux_pred)}


def format_report(name, rep):
    e, f = rep["eigenvalue"], rep["flux"]
    return (f"{name:22s} k: R2 {e['r2']:.5f}  RMSE {e['rmse_pcm']:8.1f} pcm  MAE {e['mae_pcm']:8.1f} pcm  "
            f"bias {e['bias_pcm']:+8.1f} pcm  slope {e['slope']:.4f}\n"
            f"{'':22s} flux: R2 {f['r2']:.5f}  RMSE {f['rmse']:.5f}  MAE {f['mae']:.5f}  "
            f"rel-L2 median {f['rel_l2_median']:.4f}  p95 {f['rel_l2_p95']:.4f}  profile-R2 median {f['profile_r2_median']:.5f}")

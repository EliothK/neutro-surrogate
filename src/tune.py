"""Cross-validated hyperparameter search with Optuna (TPE sampler, median pruner, SQLite storage).

All 13 tunable variables are searched jointly over continuous / categorical ranges (`train_fraction` is a data-size diagnostic, not a tuning knob, so it stays at 1.0); `mu_max` and `ramp_frac` join the space only with --physics.
Every trial is scored by k-fold cross-validation on the train + validation pool: the mean over folds of the eigenvalue and flux errors, each divided by the same errors of the default config (so 1.0 = the defaults, lower is better).
Trials that fall behind the median after a fold are pruned, which saves most of their cost.

The default config is re-run with two extra seeds to measure noise. The winner only replaces the defaults if it beats them by more than that noise. The test split is used once: the chosen config is refit on the whole pool and scored on it.

Trials live in results/optuna_<dataset>[_physics].db, so a crash or restart resumes where it stopped. Also writes results/tune_<...>.json, a trials CSV, and figures/tuning/<...>_{history,importance,folds}.png.

Run with `python -m src.tune --data data/dataset_rescaled.npz`.
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from optuna.storages import RDBStorage

from .config import DATA_DIR, FIGURE_DIR, RESULTS_DIR, ensure_dirs
from .crossval import DEFAULT_CONFIG, DEFAULT_FOLDS, cross_validate, refit_and_test
from .train import load_split

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

DEFAULT_TRIALS = 50
MIN_MARGIN = 0.02
PHYSICS_ONLY = ("mu_max", "ramp_frac")


def suggest_config(trial, physics=False):
    """Draw one full configuration from the search space."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        epochs=trial.suggest_int("epochs", 100, 800, step=50),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        width=trial.suggest_categorical("width", [128, 256, 384, 512, 768]),
        depth=trial.suggest_int("depth", 2, 6),
        lam=trial.suggest_float("lam", 0.1, 10.0, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        activation=trial.suggest_categorical("activation", ["silu", "relu", "gelu", "tanh"]),
        scheduler=trial.suggest_categorical("scheduler", ["cosine", "linear", "none"]),
        grad_clip=trial.suggest_categorical("grad_clip", [0.0, 0.5, 1.0, 5.0]),
    )
    if physics:
        cfg.update(mu_max=trial.suggest_float("mu_max", 1e-3, 1.0, log=True), ramp_frac=trial.suggest_float("ramp_frac", 0.05, 0.5))
    return cfg


def relative_score(metrics, ref):
    """Eigenvalue and flux median errors relative to the reference (default-config) errors, averaged."""
    return 0.5 * (metrics["eigen_pcm_median"] / ref["eigen_pcm_median"] + metrics["flux_l2_median"] / ref["flux_l2_median"])


def fold_scores(per_fold, ref):
    return [relative_score(f, ref) for f in per_fold]


def write_trials_csv(study, path):
    """One row per trial: number, state, score, seconds, then every parameter. No pandas needed."""
    keys = sorted({k for t in study.trials for k in t.params})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "score", "seconds", *keys])
        for t in study.trials:
            secs = "" if t.duration is None else round(t.duration.total_seconds(), 1)
            w.writerow([t.number, t.state.name, "" if t.value is None else t.value, secs, *[t.params.get(k, "") for k in keys]])


def plot_history(study, path):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return
    vals = [t.value for t in done]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([t.number for t in done], vals, s=14, alpha=0.6, label="trial")
    ax.plot([t.number for t in done], np.minimum.accumulate(vals), color="C1", label="best so far")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="defaults")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV score (lower is better)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(study, path):
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as exc:  # too few trials, or constant objective
        print(f"  (skipped importance plot: {exc})")
        return
    names, vals = list(imp)[::-1], list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1.2))
    ax.barh(names, vals)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_folds(base_folds, best_folds, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, label in ((axes[0], "eigen_pcm_median", "eigenvalue error (pcm)"), (axes[1], "flux_l2_median", "flux L2 error")):
        x = np.arange(len(base_folds))
        ax.bar(x - 0.2, [f[key] for f in base_folds], 0.4, label="defaults")
        ax.bar(x + 0.2, [f[key] for f in best_folds], 0.4, label="tuned")
        ax.set_xticks(x, [f"fold {i + 1}" for i in x])
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_DIR / "dataset_rescaled.npz"))
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="total finished trials wanted (resumed runs count)")
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--physics", action="store_true", help="tune the physics-informed model (adds mu_max, ramp_frac)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-hours", type=float, default=None, help="stop starting new trials after this long")
    p.add_argument("--fresh", action="store_true", help="delete any saved study and start over")
    args = p.parse_args()

    ensure_dirs()
    tag = f"{Path(args.data).stem}{'_physics' if args.physics else ''}"
    db = RESULTS_DIR / f"optuna_{tag}.db"
    if args.fresh and db.exists():
        db.unlink()
    fig_dir = FIGURE_DIR / "tuning"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_split(args.data, device)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = RDBStorage(f"sqlite:///{db}", heartbeat_interval=60, grace_period=300, heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=2))
    study = optuna.create_study(
        study_name=tag, storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))

    def cv(config, seed=args.seed, after_fold=None):
        return cross_validate(args.data, config, args.physics, args.folds, seed, device, loaded, after_fold)

    # Reference: the default config under CV (seed 0), plus two more seeds to measure run-to-run noise.
    if "ref" not in study.user_attrs:
        print("evaluating the default config (3 seeds) for the reference and noise floor ...", flush=True)
        runs = [cv(DEFAULT_CONFIG, seed=args.seed + s) for s in range(3)]
        ref = runs[0]["mean"]
        noise = max(abs(relative_score(r["mean"], ref) - 1.0) for r in runs[1:])
        study.set_user_attr("ref", ref)
        study.set_user_attr("noise", noise)
        study.set_user_attr("base_folds", runs[0]["folds"])
        study.set_user_attr("base_std", runs[0]["std"])
        print(f"  default CV: {ref['eigen_pcm_median']:.1f} pcm, flux L2 {ref['flux_l2_median']:.4f}; "
              f"noise {noise:.3f}", flush=True)
    ref, noise = study.user_attrs["ref"], study.user_attrs["noise"]
    margin = max(MIN_MARGIN, noise)

    def objective(trial):
        config = suggest_config(trial, args.physics)

        def after_fold(i, per_fold):
            trial.report(float(np.mean(fold_scores(per_fold, ref))), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = cv(config, after_fold=after_fold)
        trial.set_user_attr("cv_mean", result["mean"])
        trial.set_user_attr("cv_std", result["std"])
        trial.set_user_attr("folds", result["folds"])
        return relative_score(result["mean"], ref)

    def report(study_, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = " *best*" if study_.best_trial.number == trial.number else ""
            print(f"trial {trial.number:3d}  score {trial.value:6.3f}  ({trial.user_attrs['cv_mean']['eigen_pcm_median']:.0f} pcm)"
                  f"{best}", flush=True)
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"trial {trial.number:3d}  pruned", flush=True)

    finished = lambda: len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    remaining = args.trials - finished()
    print(f"{finished()} trials already finished, running {max(0, remaining)} more "
          f"({args.folds}-fold CV each, noise margin {margin:.3f})", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[report], timeout=None if args.timeout_hours is None else args.timeout_hours * 3600)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("no completed trials; nothing to report")
        return
    best_trial = study.best_trial
    beats_defaults = best_trial.value < 1.0 - margin
    best_config = suggest_config(optuna.trial.FixedTrial(best_trial.params), args.physics) if beats_defaults \
        else dict(DEFAULT_CONFIG)
    best_cv = best_trial.user_attrs if beats_defaults else {"cv_mean": ref, "cv_std": study.user_attrs["base_std"], "folds": study.user_attrs["base_folds"]}

    print("refitting the chosen config on the whole train + val pool and scoring the test split (once)", flush=True)
    test = refit_and_test(args.data, best_config, args.physics, args.seed, device, loaded)

    result = {"data": args.data, "physics": args.physics, "folds": args.folds, "trials": len(study.trials),
              "noise_margin": margin, "default_cv": ref, "beats_defaults": beats_defaults,
              "best": {"config": best_config, "score": best_trial.value if beats_defaults else 1.0,
                       "cv_mean": best_cv["cv_mean"], "cv_std": best_cv["cv_std"], "test": test}}
    out = RESULTS_DIR / f"tune_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    write_trials_csv(study, RESULTS_DIR / f"tune_{tag}_trials.csv")
    plot_history(study, fig_dir / f"{tag}_history.png")
    plot_importance(study, fig_dir / f"{tag}_importance.png")
    plot_folds(study.user_attrs["base_folds"], best_cv["folds"], fig_dir / f"{tag}_folds.png")

    cm, cs = best_cv["cv_mean"], best_cv["cv_std"]
    print(f"\nbest: score {result['best']['score']:.3f} vs defaults 1.000 "
          f"({'beats' if beats_defaults else 'does NOT beat'} the defaults by more than the {margin:.3f} noise margin)")
    print(f"  cv:   {cm['eigen_pcm_median']:.1f} +/- {cs['eigen_pcm_median']:.1f} pcm, "
          f"flux L2 {cm['flux_l2_median']:.4f} +/- {cs['flux_l2_median']:.4f}  ({args.folds} folds)")
    print(f"  test: {test['eigen_pcm_median']:.1f} pcm, flux L2 {test['flux_l2_median']:.4f}")
    print("  config: " + ", ".join(f"{k}={v}" for k, v in best_config.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

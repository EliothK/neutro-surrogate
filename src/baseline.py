"""Non-neural baselines"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

def fit_baselines(X_train, y_train, X_test, y_test, flux_train=None, flux_test=None):
    """Fit ridge on gradient-boosted trees on eigenvalue, plus ridge on the flux profile"""

    out = {}

    ridge = Ridge(alpha=1.0).fit(X_train, np.log(y_train))
    err = np.abs(np.exp(ridge.predict(X_test)) - y_test) * 1e5
    out["ridge_eigen_pcm_meidan"] = float(np.median(err))
    out["ridge_eigen_pcm_p95"] = float(np.quantile(err, 0.95))

    gbt = HistGradientBoostingRegressor(max_iter=300).fit(X_train, np.log(y_train))
    err = np.abs(np.exp(gbt.predict(X_test)) - y_test) * 1e5
    out["gbt_eigen_pcm_median"] = float(np.median(err))
    out["gbt_eigen_pcm_p95"] = float(np.quantile(err, 0.95))

    if flux_train is not None:
        rf = Ridge(alpha=1.0).fit(X_train, flux_train)
        pred = rf.predict(X_test)
        l2 = (np.linalg.norm(pred - flux_test, axis=1) / np.linalg.norm(flux_test, axis=1))
        out["ridge_flux_l2_median"] = float(np.median(l2))
        out["ridge_flux_l2_max"] = float(np.max(l2))

    out["gbt_flux_note"] = "trees produce no flux profile; network required for shape"
    return out
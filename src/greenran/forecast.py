"""XGBoost point (squared error) and quantile forecasters on site-normalised load."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURES

BASE = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, tree_method="hist", early_stopping_rounds=50)


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, quantiles, seed: int):
    """Return dict name -> predictions (in load units) for val and test, plus best iterations."""
    out = {"val": {}, "test": {}, "best_iter": {}}
    Xtr, ytr = train[FEATURES], train["y"]
    Xva, yva = val[FEATURES], val["y"]
    m = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, **BASE)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    out["best_iter"]["point"] = int(m.best_iteration)
    out["val"]["point"] = np.clip(m.predict(Xva), 0, None) * val["scale"].values
    out["test"]["point"] = np.clip(m.predict(test[FEATURES]), 0, None) * test["scale"].values
    qm = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(quantiles),
                          random_state=seed, **BASE)
    qm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    out["best_iter"]["quantile"] = int(qm.best_iteration)
    pv, pt = qm.predict(Xva), qm.predict(test[FEATURES])
    pv, pt = np.sort(pv, axis=1), np.sort(pt, axis=1)  # avoid quantile crossing
    for i, q in enumerate(quantiles):
        out["val"][f"q{int(round(q*100))}"] = np.clip(pv[:, i], 0, None) * val["scale"].values
        out["test"][f"q{int(round(q*100))}"] = np.clip(pt[:, i], 0, None) * test["scale"].values
    return out


def pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))

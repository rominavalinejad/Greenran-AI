"""Day-ahead forecasting features (all lags >= 24 h, so a full day can be planned at midnight)."""
from __future__ import annotations

import pandas as pd

LAGS = [24, 48, 72, 168]


def make_features(load: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    df = load.sort_values(["site", "hour"]).copy()
    df["scale"] = df["site"].map(scale)
    df["y"] = df["load"] / df["scale"]
    g = df.groupby("site")["y"]
    for l in LAGS:
        df[f"lag{l}"] = g.shift(l)
    # mean of the previous day (known at planning time for every target hour of the day)
    df["date"] = df["hour"].dt.normalize()
    daymean = df.groupby(["site", "date"])["y"].mean().rename("prevday_mean").reset_index()
    daymean["date"] = daymean["date"] + pd.Timedelta(days=1)
    df = df.merge(daymean, on=["site", "date"], how="left")
    df["hod"] = df["hour"].dt.hour
    df["dow"] = df["hour"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["site_row"] = df["site"] // 1000
    df["site_col"] = df["site"] % 1000
    df["log_scale"] = (df["scale"]).apply(lambda v: __import__("math").log1p(v))
    return df


FEATURES = [f"lag{l}" for l in LAGS] + ["prevday_mean", "hod", "dow", "weekend", "site_row", "site_col", "log_scale"]

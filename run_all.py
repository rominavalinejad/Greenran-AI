"""End-to-end GreenRAN AI pipeline: data -> forecasts -> sleep optimisation -> metrics -> report.

Usage: python run_all.py [--quick]
All numbers reported in README/results are written by this script into results/.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from greenran.data import build_hourly, day_list, download, select_squares  # noqa: E402
from greenran.evaluate import aggregate, bootstrap, metrics_from_units, plan, simulate  # noqa: E402
from greenran.features import make_features  # noqa: E402
from greenran.forecast import fit_predict, pinball  # noqa: E402
from greenran.power import PowerModel  # noqa: E402
from greenran.topology import site_load, size_cells  # noqa: E402


def strategies(cfg):
    s = {"always_on": ("always_on",)}
    for thr in cfg["thresholds"]:
        s[f"threshold_{int(thr*100)}"] = ("threshold", thr)
    s["cpsat_point"] = ("cpsat", "point", 1.0)
    for m in cfg["point_margins"]:
        s[f"cpsat_point_x{m:.2f}"] = ("cpsat", "point", m)
    for q in cfg["quantiles"]:
        s[f"cpsat_q{int(round(q*100))}"] = ("cpsat", f"q{int(round(q*100))}", 1.0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, fewer bootstrap samples, no sensitivity")
    a = ap.parse_args()
    cfg = json.load(open(ROOT / "config.json"))
    dcfg, ecfg = cfg["data"], cfg["experiment"]
    res = ROOT / "results"; res.mkdir(exist_ok=True)
    t_start = time.time()

    # 1) data
    days = day_list(dcfg["start"], dcfg["n_days"])
    raw = ROOT / dcfg["raw_dir"]
    if not all((raw / f"{d}.parquet").exists() for d in days):
        download(days, select_squares(**dcfg["block"]), raw, dcfg["guestbook"], dcfg["workers"])
    hourly = build_hourly(days, raw)
    load = site_load(hourly, ecfg["site_block"], ecfg["signal"])
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    load.to_parquet(ROOT / "data/processed/site_load.parquet", index=False)

    sp = ecfg["split"]
    tr0, tr1 = pd.Timestamp(sp["train"][0]), pd.Timestamp(sp["train"][1]) + pd.Timedelta(hours=23)
    va0, va1 = pd.Timestamp(sp["val"][0]), pd.Timestamp(sp["val"][1]) + pd.Timedelta(hours=23)
    te0, te1 = pd.Timestamp(sp["test"][0]), pd.Timestamp(sp["test"][1]) + pd.Timedelta(hours=23)
    cells = size_cells(load, tr1, ecfg["n_capacity_cells"], ecfg["headroom"])
    cells.to_csv(res / "cells.csv", index=False)
    scale = load[load.hour <= tr1].groupby("site")["load"].mean()
    feat = make_features(load, scale)
    feat["load_prev"] = feat.groupby("site")["load"].shift(1)
    train = feat[(feat.hour >= tr0) & (feat.hour <= tr1)].dropna()
    val = feat[(feat.hour >= va0) & (feat.hour <= va1)].dropna()
    test = feat[(feat.hour >= te0) & (feat.hour <= te1)].copy()
    assert test[["lag168", "load_prev"]].notna().all().all()

    pm = PowerModel(**cfg["power"])
    strat = strategies(ecfg)
    names = list(strat)
    seeds = ecfg["seeds"][:1] if a.quick else ecfg["seeds"]
    n_boot = 300 if a.quick else ecfg["n_bootstrap"]
    qs = ecfg["quantiles"]
    all_rows, fc_rows, rt_rows, boot_store = [], [], [], {}
    for seed in seeds:
        fc = fit_predict(train, val, test, qs, seed)
        te = test[["site", "hour", "load", "load_prev"]].copy()
        for k, v in fc["test"].items():
            te[k] = v
        y = te["load"].values
        fr = dict(seed=seed, mae_point=float(np.mean(np.abs(y - te["point"]))),
                  mape_point_pct=float(100 * np.sum(np.abs(y - te["point"])) / np.sum(y)),
                  best_iter_point=fc["best_iter"]["point"], best_iter_quantile=fc["best_iter"]["quantile"])
        for q in qs:
            c = f"q{int(round(q*100))}"
            fr[f"coverage_{c}_pct"] = float(100 * np.mean(y <= te[c]))
            fr[f"pinball_{c}"] = pinball(y, te[c].values, q)
        fc_rows.append(fr)
        dec, rts = plan(te, cells, strat, pm, ecfg["time_limit_s"])
        rts["seed"] = seed; rt_rows.append(rts)
        dec.to_parquet(res / f"decisions_seed{seed}.parquet", index=False)
        sim = simulate(dec, cells, names, pm)
        units = aggregate(sim)
        base = units[units.strategy == "always_on"].energy_wh.sum()
        for nm in names:
            r = dict(seed=seed, strategy=nm, **metrics_from_units(units[units.strategy == nm], base))
            all_rows.append(r)
        ci, stats = bootstrap(units, names, n_boot, seed)
        ci["seed"] = seed
        boot_store[seed] = (ci, stats)
        print(f"seed {seed} done ({time.time()-t_start:.0f}s)", flush=True)

    per_seed = pd.DataFrame(all_rows); per_seed.to_csv(res / "metrics_per_seed.csv", index=False)
    fcm = pd.DataFrame(fc_rows); fcm.to_csv(res / "forecast_metrics.csv", index=False)
    rt = pd.concat(rt_rows); rt.to_csv(res / "optimizer_runtime.csv", index=False)
    ci_all = pd.concat([b[0] for b in boot_store.values()]); ci_all.to_csv(res / "bootstrap_ci_per_seed.csv", index=False)

    # summary: mean over seeds; CI = seed-0 bootstrap (site-day cluster resampling)
    summ = per_seed.groupby("strategy", sort=False).agg(
        energy_kwh=("energy_kwh", "mean"), saving_pct=("saving_pct", "mean"), saving_pct_sd=("saving_pct", "std"),
        violation_rate_pct=("violation_rate_pct", "mean"), violation_rate_pct_sd=("violation_rate_pct", "std"),
        unserved_pct=("unserved_pct", "mean")).reset_index()
    ci0 = boot_store[seeds[0]][0]
    summ = summ.merge(ci0, on="strategy")
    rtm = rt.groupby("strategy").runtime_s.agg(["mean", "max", "sum"]).rename(
        columns={"mean": "solve_mean_s", "max": "solve_max_s", "sum": "solve_total_s"}).reset_index()
    summ = summ.merge(rtm, on="strategy", how="left")
    summ.to_csv(res / "summary.csv", index=False)

    # paired comparisons (seed 0 bootstrap draws)
    stats = boot_store[seeds[0]][1]
    pairs = []
    for a_, b_ in ecfg["comparisons"]:
        for k in ["saving_pct", "violation_rate_pct", "unserved_pct"]:
            d = np.array(stats[a_][k]) - np.array(stats[b_][k])
            pairs.append(dict(a=a_, b=b_, metric=k, diff_mean=float(d.mean()),
                              lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    pd.DataFrame(pairs).to_csv(res / "paired_differences.csv", index=False)

    # sensitivity on power model parameters (seed 0 forecasts, re-optimised)
    if not a.quick:
        dec0 = pd.read_parquet(res / f"decisions_seed{seeds[0]}.parquet")
        base_cols = ["site", "hour", "load", "load_prev", "point"] + [f"q{int(round(q*100))}" for q in qs]
        sens = []
        for var in ecfg["sensitivity"]:
            p = dict(cfg["power"]); p.update(var)
            pmv = PowerModel(**p)
            d2, _ = plan(dec0[base_cols], cells, strat, pmv, ecfg["time_limit_s"])
            u2 = aggregate(simulate(d2, cells, names, pmv))
            b2 = u2[u2.strategy == "always_on"].energy_wh.sum()
            for nm in names:
                sens.append(dict(**{k: p[k] for k in p}, strategy=nm,
                                 **metrics_from_units(u2[u2.strategy == nm], b2)))
        pd.DataFrame(sens).to_csv(res / "sensitivity_power.csv", index=False)

    meta = dict(python=platform.python_version(), platform=platform.platform(), seeds=seeds,
                n_sites=int(load.site.nunique()), n_squares=int(hourly.square.nunique()),
                days=[days[0], days[-1]], split=sp, power=cfg["power"], quick=a.quick,
                runtime_total_s=round(time.time() - t_start, 1),
                test_site_hours=int(len(test)))
    json.dump(meta, open(res / "run_meta.json", "w"), indent=2)
    from make_report import write_report
    write_report(ROOT)
    from make_report import plot_frontier
    plot_frontier(ROOT)
    print("done", meta["runtime_total_s"], "s")


if __name__ == "__main__":
    main()

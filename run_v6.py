"""v6 experiments: conformal recalibration, matched-energy comparison, rolling day-by-day mode.

Usage: python run_v6.py [--quick]
Protocol: results_v6/protocol_v6.md (written before the run). Outputs go to results_v6/ only.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from greenran.conformal import apply_offset, calibrate_quantiles, conformal_offset  # noqa: E402
from greenran.data import build_hourly, day_list  # noqa: E402
from greenran.evaluate import aggregate, bootstrap, metrics_from_units, plan, simulate  # noqa: E402
from greenran.features import make_features  # noqa: E402
from greenran.forecast import fit_predict  # noqa: E402
from greenran.power import PowerModel  # noqa: E402
from greenran.topology import site_load, size_cells  # noqa: E402

OUT = ROOT / "results_v6"
QNAMES = {"q50": 0.5, "q80": 0.8, "q90": 0.9, "q95": 0.95}
GRIDS = {
    "point": [round(1.00 + 0.01 * i, 2) for i in range(31)],
    "rawq90": [round(0.80 + 0.01 * i, 2) for i in range(41)],
    "confq90": [round(0.70 + 0.01 * i, 2) for i in range(30)],
    "threshold": [round(0.60 + 0.02 * i, 2) for i in range(20)],
}


def git_state():
    h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--", "src", "run_v6.py", "config.json"],
                       cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return h, bool(d)


def fam_spec(fam, p):
    if fam == "point":
        return ("cpsat", "point", p)
    if fam == "rawq90":
        return ("cpsat", "q90", p)
    if fam == "confq90":
        return ("cpsat", f"cq90_t{int(round(p * 100))}", 1.0)
    return ("threshold", p)


def fam_name(fam, p):
    return f"{fam}_{p:.2f}"


def add_forecasts(df, fc_part):
    for k, v in fc_part.items():
        df[k] = v
    return df


def pooled_units(units_by_seed):
    """Sum each site-day over seeds so that a bootstrap cluster holds that site-day for every seed."""
    u = pd.concat(units_by_seed)
    return u.groupby(["strategy", "site", "day"], as_index=False)[
        ["energy_wh", "served", "unserved", "demand", "violations", "hours"]].sum()


def ci_table(units, names, n_boot, seed=0):
    ci, stats = bootstrap(units, names, n_boot, seed)
    base = units[units.strategy == "always_on"].energy_wh.sum()
    rows = []
    for nm in names:
        r = dict(strategy=nm, **metrics_from_units(units[units.strategy == nm], base))
        rows.append(r)
    return pd.DataFrame(rows).merge(ci, on="strategy"), stats


def paired(stats, pairs):
    out = []
    for a, b in pairs:
        for k in ["saving_pct", "violation_rate_pct", "unserved_pct"]:
            d = np.array(stats[a][k]) - np.array(stats[b][k])
            out.append(dict(a=a, b=b, metric=k, diff_mean=float(d.mean()),
                            lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, 300 bootstrap draws")
    a = ap.parse_args()
    t0 = time.time()
    commit, dirty = git_state()
    cfg = json.load(open(ROOT / "config.json"))
    dcfg, ecfg = cfg["data"], cfg["experiment"]
    OUT.mkdir(exist_ok=True)
    days = day_list(dcfg["start"], dcfg["n_days"])
    hourly = build_hourly(days, ROOT / dcfg["raw_dir"])
    load = site_load(hourly, ecfg["site_block"], ecfg["signal"])
    sp = ecfg["split"]
    D = lambda s: pd.Timestamp(s)  # noqa: E731
    H23 = pd.Timedelta(hours=23)
    tr0, tr1 = D(sp["train"][0]), D(sp["train"][1]) + H23
    va0, va1 = D(sp["val"][0]), D(sp["val"][1]) + H23
    te0, te1 = D(sp["test"][0]), D(sp["test"][1]) + H23
    cells = size_cells(load, tr1, ecfg["n_capacity_cells"], ecfg["headroom"])
    scale = load[load.hour <= tr1].groupby("site")["load"].mean()
    feat = make_features(load, scale)
    feat["load_prev"] = feat.groupby("site")["load"].shift(1)
    win = lambda lo, hi: feat[(feat.hour >= lo) & (feat.hour <= hi)]  # noqa: E731
    train, val, test = win(tr0, tr1).dropna(), win(va0, va1).dropna(), win(te0, te1).copy()
    pm = PowerModel(**cfg["power"])
    seeds = ecfg["seeds"][:1] if a.quick else ecfg["seeds"]
    n_boot = 300 if a.quick else ecfg["n_bootstrap"]
    qs = ecfg["quantiles"]
    tl = ecfg["time_limit_s"]
    cols = ["site", "hour", "load", "load_prev", "scale"]

    stratA = {"always_on": ("always_on",), "cpsat_point": ("cpsat", "point", 1.0),
              "cpsat_point_x1.10": ("cpsat", "point", 1.1), "cpsat_q90": ("cpsat", "q90", 1.0),
              "cpsat_cq90": ("cpsat", "cq90", 1.0)}
    stratB = {"always_on": ("always_on",)}
    for fam, grid in GRIDS.items():
        for p in grid:
            stratB[fam_name(fam, p)] = fam_spec(fam, p)

    cov_rows, offs_rows, unitsA, unitsBval, frames_test = [], [], [], [], {}
    for seed in seeds:
        fc = fit_predict(train, val, test, qs, seed)
        va = add_forecasts(val[cols].copy(), fc["val"])
        te = add_forecasts(test[cols].copy(), fc["test"])
        cal, offs = calibrate_quantiles(fc["val"], fc["test"], va["load"].values, va["scale"].values,
                                        te["scale"].values, QNAMES)
        for n in QNAMES:
            te[f"c{n}"] = cal[n]
            offs_rows.append(dict(seed=seed, quantile=n, offset_norm=offs[n]))
        # conformal level grid on the q90 base (offset from validation residuals only)
        for p in GRIDS["confq90"]:
            o = conformal_offset(va["load"], va["q90"], va["scale"], p)
            te[f"cq90_t{int(round(p*100))}"] = apply_offset(te["q90"], te["scale"], o)
            va[f"cq90_t{int(round(p*100))}"] = apply_offset(va["q90"], va["scale"], o)
        y = te["load"].values
        for n, lvl in QNAMES.items():
            cov_rows.append(dict(seed=seed, quantile=n, nominal_pct=100 * lvl,
                                 raw_val_coverage_pct=float(100 * np.mean(va["load"] <= va[n])),
                                 raw_test_coverage_pct=float(100 * np.mean(y <= te[n])),
                                 conformal_test_coverage_pct=float(100 * np.mean(y <= te[f"c{n}"]))))
        frames_test[seed] = te
        decA, _ = plan(te, cells, stratA, pm, tl)
        decA.to_parquet(OUT / f"decisions_oneshot_seed{seed}.parquet", index=False)
        u = aggregate(simulate(decA, cells, list(stratA), pm)); u["seed"] = seed; unitsA.append(u)
        decBv, _ = plan(va, cells, stratB, pm, tl)
        u = aggregate(simulate(decBv, cells, list(stratB), pm)); u["seed"] = seed; unitsBval.append(u)
        print(f"seed {seed}: A + B-val done ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(cov_rows).to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(offs_rows).to_csv(OUT / "conformal_offsets.csv", index=False)
    pd.concat(unitsA).to_csv(OUT / "units_oneshot.csv", index=False)
    namesA = list(stratA)
    perseedA = []
    for u in unitsA:
        base = u[u.strategy == "always_on"].energy_wh.sum()
        for nm in namesA:
            perseedA.append(dict(seed=int(u.seed.iloc[0]), strategy=nm, **metrics_from_units(u[u.strategy == nm], base)))
    pd.DataFrame(perseedA).to_csv(OUT / "conformal_per_seed.csv", index=False)
    tabA, statsA = ci_table(pooled_units(unitsA), namesA, n_boot)
    tabA.to_csv(OUT / "conformal_summary.csv", index=False)
    paired(statsA, [("cpsat_cq90", "cpsat_q90"), ("cpsat_cq90", "cpsat_point"),
                    ("cpsat_cq90", "cpsat_point_x1.10"), ("cpsat_q90", "cpsat_point_x1.10")]
           ).to_csv(OUT / "conformal_paired.csv", index=False)

    # ---- B: selection on validation
    pv = pooled_units(unitsBval)
    base_v = pv[pv.strategy == "always_on"].energy_wh.sum()
    val_rows = []
    for fam, grid in GRIDS.items():
        for p in grid:
            nm = fam_name(fam, p)
            val_rows.append(dict(family=fam, param=p, strategy=nm, **metrics_from_units(pv[pv.strategy == nm], base_v)))
    vt = pd.DataFrame(val_rows); vt.to_csv(OUT / "matched_validation_grid.csv", index=False)
    target = float(vt[(vt.family == "point") & (np.isclose(vt.param, 1.10))].saving_pct.iloc[0])
    chosen = {}
    for fam in GRIDS:
        g = vt[vt.family == fam].assign(gap=lambda x: (x.saving_pct - target).abs())
        g = g.sort_values(["gap", "saving_pct"])
        chosen[fam] = float(g.param.iloc[0])
    stratB_sel = {"always_on": ("always_on",)}
    for fam, p in chosen.items():
        stratB_sel[fam_name(fam, p)] = fam_spec(fam, p)
    unitsB = []
    for seed in seeds:
        dec, _ = plan(frames_test[seed], cells, stratB_sel, pm, tl)
        u = aggregate(simulate(dec, cells, list(stratB_sel), pm)); u["seed"] = seed; unitsB.append(u)
    tabB, statsB = ci_table(pooled_units(unitsB), list(stratB_sel), n_boot)
    sel = pd.DataFrame([dict(family=f, param=p, strategy=fam_name(f, p),
                             val_saving_pct=float(vt[vt.strategy == fam_name(f, p)].saving_pct.iloc[0]),
                             val_violation_rate_pct=float(vt[vt.strategy == fam_name(f, p)].violation_rate_pct.iloc[0]))
                        for f, p in chosen.items()])
    sel["target_val_saving_pct"] = target
    tabB = tabB.merge(sel, on="strategy", how="left")
    tabB.to_csv(OUT / "matched_energy_test.csv", index=False)
    pref = fam_name("point", chosen["point"])
    paired(statsB, [(fam_name(f, chosen[f]), pref) for f in GRIDS if f != "point"]).to_csv(
        OUT / "matched_energy_paired.csv", index=False)
    print(f"B done ({time.time()-t0:.0f}s)", flush=True)

    # ---- C: rolling day by day
    stratC = {k: v for k, v in stratA.items()}
    test_days = pd.date_range(D(sp["test"][0]), D(sp["test"][1]), freq="D")
    unitsC, roll_cov = [], []
    for seed in seeds:
        state = {}  # strategy -> {site: n_on at end of previous applied day}
        parts = []
        for d in test_days:
            trw = feat[(feat.hour >= tr0) & (feat.hour < d - pd.Timedelta(days=3))].dropna()
            caw = feat[(feat.hour >= d - pd.Timedelta(days=3)) & (feat.hour < d)].dropna()
            tdw = feat[(feat.hour >= d) & (feat.hour <= d + H23)].copy()
            fc = fit_predict(trw, caw, tdw, qs, seed)
            ca = add_forecasts(caw[cols].copy(), fc["val"])
            td = add_forecasts(tdw[cols].copy(), fc["test"])
            o = conformal_offset(ca["load"], ca["q90"], ca["scale"], 0.9)
            td["cq90"] = apply_offset(td["q90"], td["scale"], o)
            roll_cov.append(dict(seed=seed, day=str(d.date()), offset_norm=o,
                                 raw_q90_coverage_pct=float(100 * np.mean(td.load <= td.q90)),
                                 conformal_q90_coverage_pct=float(100 * np.mean(td.load <= td.cq90))))
            td = td.sort_values(["site", "hour"]).reset_index(drop=True)
            dec = plan_with_state(td, cells, stratC, pm, tl, state)
            parts.append(dec)
        dec = pd.concat(parts, ignore_index=True)
        dec.to_parquet(OUT / f"decisions_rolling_seed{seed}.parquet", index=False)
        u = aggregate(simulate(dec, cells, list(stratC), pm)); u["seed"] = seed; unitsC.append(u)
        print(f"seed {seed}: rolling done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(roll_cov).to_csv(OUT / "rolling_coverage.csv", index=False)
    pd.concat(unitsC).to_csv(OUT / "units_rolling.csv", index=False)
    uC = pooled_units(unitsC); uC["strategy"] = "rolling:" + uC["strategy"]
    uA = pooled_units(unitsA); uA["strategy"] = "oneshot:" + uA["strategy"]
    both = pd.concat([uA, uC])
    both.loc[both.strategy == "oneshot:always_on", "strategy"] = "always_on"
    both = both[both.strategy != "rolling:always_on"]
    namesC = ["always_on"] + [f"{m}:{s}" for s in namesA if s != "always_on" for m in ("oneshot", "rolling")]
    tabC, statsC = ci_table(both, namesC, n_boot)
    tabC.to_csv(OUT / "rolling_vs_oneshot.csv", index=False)
    paired(statsC, [(f"rolling:{s}", f"oneshot:{s}") for s in namesA if s != "always_on"]).to_csv(
        OUT / "rolling_vs_oneshot_paired.csv", index=False)

    meta = dict(git_commit=commit, git_dirty=dirty, command="python run_v6.py" + (" --quick" if a.quick else ""),
                seeds=seeds, n_bootstrap=n_boot, bootstrap="site-day clusters pooled over seeds, rng seed 0",
                python=platform.python_version(), platform=platform.platform(),
                chosen_matched_params=chosen, matched_target_val_saving_pct=target,
                runtime_total_s=round(time.time() - t0, 1))
    json.dump(meta, open(OUT / "run_meta_v6.json", "w"), indent=2)
    from make_report import write_report_v6
    write_report_v6(ROOT)
    print("done", meta["runtime_total_s"], "s")


def plan_with_state(td, cells, strategies, pm, tl, state):
    """Plan one day per site and strategy, starting from the applied cell state of the previous day."""
    from greenran.optimizer import solve_day
    cap = cells.set_index("site")
    for name, spec in strategies.items():
        col = np.zeros(len(td), dtype=int)
        st = state.setdefault(name, {})
        for site, g in td.groupby("site", sort=False):
            cc, K = float(cap.at[site, "cell_cap"]), int(cap.at[site, "n_cap"])
            pos = g.index.to_numpy()
            if spec[0] == "always_on":
                n_on = np.full(len(g), K)
            else:
                n_on, _, _ = solve_day(g[spec[1]].values * spec[2], cc, K, pm, st.get(site), tl)
            col[pos] = n_on
            st[site] = int(n_on[-1])
        td[f"n_on_{name}"] = col
    return td


if __name__ == "__main__":
    main()

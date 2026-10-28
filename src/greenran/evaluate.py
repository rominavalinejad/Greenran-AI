"""Strategy simulation on actual test traffic, metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import solve_day, threshold_policy
from .power import PowerModel, site_hour_energy, switch_energy


def plan(test: pd.DataFrame, cells: pd.DataFrame, strategies: dict, pm: PowerModel,
         time_limit: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """test: rows site,hour,load,load_prev(observed t-1), forecast columns.
    strategies: name -> ("always_on",) | ("threshold", thr) | ("cpsat", column, multiplier)
    Returns decisions frame (n_on_<name>) and runtime frame."""
    test = test.sort_values(["site", "hour"]).copy()
    cap = cells.set_index("site")
    rts = []
    for name, spec in strategies.items():
        col = np.zeros(len(test), dtype=int)
        for site, g in test.groupby("site", sort=False):
            cc, K = float(cap.at[site, "cell_cap"]), int(cap.at[site, "n_cap"])
            idx = g.index
            pos = test.index.get_indexer(idx)
            if spec[0] == "always_on":
                col[pos] = K
            elif spec[0] == "threshold":
                col[pos] = threshold_policy(g["load_prev"].values, cc, K, spec[1])
            elif spec[0] == "cpsat":
                n_prev = None
                for day, gd in g.groupby(g["hour"].dt.normalize(), sort=True):
                    n_on, rt, opt = solve_day(gd[spec[1]].values * spec[2], cc, K, pm, n_prev, time_limit)
                    col[test.index.get_indexer(gd.index)] = n_on
                    n_prev = int(n_on[-1])
                    rts.append(dict(strategy=name, site=site, day=day, runtime_s=rt, optimal=opt))
            else:
                raise ValueError(spec)
        test[f"n_on_{name}"] = col
    return test, pd.DataFrame(rts)


def simulate(dec: pd.DataFrame, cells: pd.DataFrame, names, pm: PowerModel) -> pd.DataFrame:
    """Per site-hour energy / served / unserved / violation for each strategy."""
    cap = cells.set_index("site")
    cc = dec["site"].map(cap["cell_cap"]).values
    K = dec["site"].map(cap["n_cap"]).values
    rows = []
    for name in names:
        n = dec[f"n_on_{name}"].values
        e, served, unserved, viol = site_hour_energy(n, dec["load"].values, cc, K, pm)
        sw = np.zeros(len(dec))
        for site, g in dec.groupby("site", sort=False):
            p = dec.index.get_indexer(g.index)
            sw[p] = switch_energy(n[p], pm)
        rows.append(pd.DataFrame(dict(strategy=name, site=dec["site"].values, hour=dec["hour"].values,
                                      energy_wh=e + sw, served=served, unserved=unserved,
                                      demand=dec["load"].values, violation=viol.astype(int))))
    return pd.concat(rows, ignore_index=True)


def aggregate(sim: pd.DataFrame) -> pd.DataFrame:
    sim = sim.assign(day=pd.to_datetime(sim["hour"]).dt.normalize())
    return sim.groupby(["strategy", "site", "day"], as_index=False).agg(
        energy_wh=("energy_wh", "sum"), served=("served", "sum"), unserved=("unserved", "sum"),
        demand=("demand", "sum"), violations=("violation", "sum"), hours=("violation", "size"))


def metrics_from_units(u: pd.DataFrame, base_energy: float) -> dict:
    return dict(energy_kwh=u.energy_wh.sum() / 1000,
                saving_pct=100 * (1 - u.energy_wh.sum() / base_energy),
                violation_rate_pct=100 * u.violations.sum() / u.hours.sum(),
                unserved_pct=100 * u.unserved.sum() / u.demand.sum(),
                served_traffic=u.served.sum())


def bootstrap(units: pd.DataFrame, names, n_boot: int = 2000, seed: int = 0, baseline="always_on"):
    """Cluster bootstrap over site-days (same resample for all strategies => paired)."""
    rng = np.random.default_rng(seed)
    piv = {k: units.pivot_table(index=["site", "day"], columns="strategy", values=k)
           for k in ["energy_wh", "unserved", "demand", "violations", "hours"]}
    n = len(piv["energy_wh"])
    stats = {nm: {"saving_pct": [], "violation_rate_pct": [], "unserved_pct": []} for nm in names}
    arr = {k: v.to_numpy() for k, v in piv.items()}
    cols = list(piv["energy_wh"].columns)
    b = cols.index(baseline)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        E = arr["energy_wh"][i].sum(0)
        V = arr["violations"][i].sum(0); H = arr["hours"][i].sum(0)
        U = arr["unserved"][i].sum(0); D = arr["demand"][i].sum(0)
        for nm in names:
            j = cols.index(nm)
            stats[nm]["saving_pct"].append(100 * (1 - E[j] / E[b]))
            stats[nm]["violation_rate_pct"].append(100 * V[j] / H[j])
            stats[nm]["unserved_pct"].append(100 * U[j] / D[j])
    out = []
    for nm in names:
        r = {"strategy": nm}
        for k, v in stats[nm].items():
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        out.append(r)
    return pd.DataFrame(out), stats

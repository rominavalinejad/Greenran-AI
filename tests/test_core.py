import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from greenran.optimizer import solve_day, threshold_policy  # noqa: E402
from greenran.power import PowerModel, site_hour_energy  # noqa: E402
from greenran.topology import square_to_site  # noqa: E402
from greenran.evaluate import bootstrap  # noqa: E402

PM = PowerModel()


def test_power_model_endpoints():
    e, served, uns, v = site_hour_energy([3, 0], [0.0, 0.0], 10.0, 3, PM)
    assert e[0] == 4 * PM.p0
    assert e[1] == PM.p0 + 3 * PM.p_sleep
    e, served, uns, v = site_hour_energy([3], [40.0], 10.0, 3, PM)
    assert np.isclose(e[0], 4 * (PM.p0 + PM.delta_p * PM.p_max))
    assert not v[0]


def test_violation_when_demand_exceeds_capacity():
    e, served, uns, v = site_hour_energy([1], [25.0], 10.0, 3, PM)
    assert served[0] == 20 and uns[0] == 5 and v[0]


def test_optimizer_meets_forecast_and_is_minimal():
    demand = np.array([5, 15, 25, 35, 45, 12])
    n, rt, opt = solve_day(demand, 10.0, 3, PM)
    cap = 10 * (1 + n)
    assert opt
    assert np.all(cap >= np.minimum(demand, 40))
    assert n[0] == 0 and n[3] == 3


def test_optimizer_switching_cost_keeps_cells_on():
    pm = PowerModel(e_switch=1000)
    n, _, _ = solve_day(np.array([25, 5, 25]), 10.0, 3, pm, n_prev=1)
    assert list(n) == [2, 2, 2]


def test_threshold_policy():
    assert list(threshold_policy([0, 8, 16, 100], 10.0, 3, 0.8)) == [0, 0, 1, 3]


def test_square_to_site_2x2():
    sq = np.array([4541, 4542, 4641, 4642, 4543])
    s = square_to_site(sq)
    assert len(set(s[:4])) == 1 and s[4] != s[0]


def test_bootstrap_paired_shapes():
    rows = []
    for st, e in [("always_on", 100.0), ("x", 90.0)]:
        for i in range(10):
            rows.append(dict(strategy=st, site=i, day=pd.Timestamp("2013-11-28"), energy_wh=e,
                             unserved=0.0, demand=1.0, violations=0, hours=24))
    ci, _ = bootstrap(pd.DataFrame(rows), ["x"], n_boot=50)
    assert np.isclose(ci.saving_pct_lo[0], 10.0) and np.isclose(ci.saving_pct_hi[0], 10.0)

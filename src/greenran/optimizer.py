"""CP-SAT sleep scheduling per site and day.

Decision x[k,t] in {0,1}: capacity cell k active in hour t (coverage cell always on).
min  sum_t sum_k (P0 - P_sleep) x[k,t] + E_switch * w[k,t]
s.t. cell_cap * (1 + sum_k x[k,t]) >= min(D_t, total capacity)    (D_t = point or quantile forecast)
     w[k,t] >= x[k,t] - x[k,t-1]                                    (wake-up indicator)
     x[k,t] >= x[k+1,t]                                             (symmetry breaking)
The load-dependent term is constant w.r.t. x under even load sharing and so is omitted.
"""
from __future__ import annotations

import math
import time

import numpy as np
from ortools.sat.python import cp_model


def solve_day(demand, cell_cap, n_cap, pm, n_prev: int | None = None, time_limit: float = 5.0):
    T = len(demand)
    need = np.minimum(np.asarray(demand, float), cell_cap * (1 + n_cap))
    # required number of capacity cells; tiny tolerance for floating point
    req = np.clip(np.ceil(need / cell_cap - 1 - 1e-9), 0, n_cap).astype(int)
    m = cp_model.CpModel()
    x = {(k, t): m.NewBoolVar(f"x{k}_{t}") for k in range(n_cap) for t in range(T)}
    w = {(k, t): m.NewBoolVar(f"w{k}_{t}") for k in range(n_cap) for t in range(T)}
    for t in range(T):
        m.Add(sum(x[k, t] for k in range(n_cap)) >= int(req[t]))
        for k in range(n_cap - 1):
            m.Add(x[k, t] >= x[k + 1, t])
        for k in range(n_cap):
            if t > 0:
                m.Add(w[k, t] >= x[k, t] - x[k, t - 1])
            elif n_prev is not None:
                m.Add(w[k, t] >= x[k, t] - (1 if k < n_prev else 0))
            else:
                m.Add(w[k, t] == 0)
    c_on = int(round(pm.p0 - pm.p_sleep))
    c_sw = int(round(pm.e_switch))
    m.Minimize(sum(c_on * x[k, t] + c_sw * w[k, t] for k in range(n_cap) for t in range(T)))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 1
    t0 = time.perf_counter()
    st = s.Solve(m)
    rt = time.perf_counter() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed")
    n_on = np.array([sum(s.Value(x[k, t]) for k in range(n_cap)) for t in range(T)])
    return n_on, rt, st == cp_model.OPTIMAL


def threshold_policy(load_prev, cell_cap, n_cap, thr: float = 0.8):
    """Reactive fixed-threshold baseline: size active cells so that last hour's observed
    load would use at most `thr` of active capacity."""
    need = np.asarray(load_prev, float) / thr
    return np.clip(np.ceil(need / cell_cap - 1 - 1e-9), 0, n_cap).astype(int)

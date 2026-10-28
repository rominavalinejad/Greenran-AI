"""Load-dependent base-station power model.

Linear model of Auer et al., "How much energy is needed to run a wireless network?",
IEEE Wireless Communications 18(5), 2011 (EARTH project):
    P_in = P0 + delta_p * P_max * rho      (active, 0 <= rho <= 1)
    P_in = P_sleep                         (sleep)
Default per-cell (one transceiver chain, macro) values from the EARTH macro table:
P0 = 130 W, delta_p = 4.7, P_max = 20 W, P_sleep = 75 W. Treating one cell as one
transceiver chain and the switching energy are ASSUMPTIONS of this MVP; sensitivity
runs vary them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class PowerModel:
    p0: float = 130.0        # W, active no-load
    delta_p: float = 4.7     # load slope
    p_max: float = 20.0      # W RF output at full load
    p_sleep: float = 75.0    # W in sleep
    e_switch: float = 20.0   # Wh per wake-up (ASSUMPTION)

    def as_dict(self):
        return asdict(self)


def site_hour_energy(n_active_cap, load, cell_cap, n_cap, pm: PowerModel):
    """Energy (Wh) per site-hour; served load spread evenly over active cells.

    n_active_cap: number of active capacity cells (coverage cell always active).
    Returns (energy_wh, served, unserved, violation_flag).
    """
    n_active_cap = np.asarray(n_active_cap, dtype=float)
    load = np.asarray(load, dtype=float)
    n_on = 1.0 + n_active_cap
    cap_on = n_on * cell_cap
    served = np.minimum(load, cap_on)
    unserved = load - served
    # sum over active cells of delta_p * p_max * rho_c, with rho_c = served / cap_on
    e = n_on * pm.p0 + pm.delta_p * pm.p_max * served / cell_cap + (n_cap - n_active_cap) * pm.p_sleep
    return e, served, unserved, (unserved > 1e-9)


def switch_energy(n_active_cap_seq, pm: PowerModel, n_prev: float = None):
    n = np.asarray(n_active_cap_seq, dtype=float)
    prev = np.concatenate([[n[0] if n_prev is None else n_prev], n[:-1]])
    return pm.e_switch * np.clip(n - prev, 0, None)

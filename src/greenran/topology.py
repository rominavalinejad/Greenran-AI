"""Map grid squares to simulated base-station sites with capacity-layer cells.

ASSUMPTION (no real cell layout is published with the dataset): each 2x2 block of
grid squares (~470 m x 470 m) is served by one site. A site has one coverage cell
(always on) and K capacity cells that may sleep. Cells at a site have equal capacity,
sized so that total site capacity = headroom x peak hourly load seen in the training period.
Units of capacity/load are the dataset's anonymised "internet activity" units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def square_to_site(square: np.ndarray, block: int = 2) -> np.ndarray:
    """Site id = site_row * 1000 + site_col, with rows/cols counted from the selected block's corner."""
    s = np.asarray(square) - 1
    row, col = s // 100, s % 100
    row, col = row - row.min(), col - col.min()
    return (row // block) * 1000 + (col // block)


def site_load(hourly: pd.DataFrame, block: int = 2, signal: str = "internet") -> pd.DataFrame:
    df = hourly.assign(site=square_to_site(hourly["square"].values, block))
    out = df.groupby(["site", "hour"], as_index=False)[signal].sum().rename(columns={signal: "load"})
    return out


def size_cells(load: pd.DataFrame, train_end: pd.Timestamp, n_capacity_cells: int = 3,
               headroom: float = 1.1) -> pd.DataFrame:
    """Return per-site cell capacity (equal for all 1 + K cells)."""
    tr = load[load.hour <= train_end]
    peak = tr.groupby("site")["load"].max()
    cap = headroom * peak / (1 + n_capacity_cells)
    return pd.DataFrame({"site": peak.index, "cell_cap": cap.values, "n_cap": n_capacity_cells})

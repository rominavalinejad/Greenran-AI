"""Streamlit dashboard: streamlit run app.py  (needs results/ from `python run_all.py`)."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from greenran.evaluate import simulate  # noqa: E402
from greenran.power import PowerModel  # noqa: E402

st.set_page_config(page_title="GreenRAN AI", layout="wide")
R = ROOT / "results"
if not (R / "summary.csv").exists():
    st.error("No results found. Run `python run_all.py` first."); st.stop()

cfg = json.load(open(ROOT / "config.json"))
meta = json.load(open(R / "run_meta.json"))
cells = pd.read_csv(R / "cells.csv")
seed = st.sidebar.selectbox("Forecast seed", meta["seeds"])
dec = pd.read_parquet(R / f"decisions_seed{seed}.parquet")
names = [c[5:] for c in dec.columns if c.startswith("n_on_")]
st.title("GreenRAN AI: uncertainty-aware cell sleeping (Milan data)")
st.caption("Load = Telecom Italia grid-square internet activity aggregated to simulated sites (proxy for cell load). "
           "Energy is model-estimated with an assumed EARTH-style power model.")

sites = sorted(dec.site.unique())
site = st.sidebar.selectbox("Site", ["All sites (sum)"] + sites)
days = sorted(dec.hour.dt.normalize().unique())
day = st.sidebar.selectbox("Day", days, format_func=lambda d: pd.Timestamp(d).strftime("%a %Y-%m-%d"))
sel = st.sidebar.multiselect("Strategies", names, default=[n for n in ["always_on", "threshold_80", "cpsat_point", "cpsat_q90"] if n in names])

pm = PowerModel(**cfg["power"])
sim = simulate(dec, cells, names, pm)
sim["day"] = pd.to_datetime(sim.hour).dt.normalize()
d = dec[dec.hour.dt.normalize() == day]
s = sim[sim.day == day]
if site != "All sites (sum)":
    d = d[d.site == site]; s = s[s.site == site]
cap = d.merge(cells, on="site")
ts = d.groupby("hour")[["load", "point", "q50", "q90"] + [c for c in ["q95"] if c in d]].sum().reset_index()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                    subplot_titles=("Traffic, forecast band and active capacity", "Active capacity cells"))
fig.add_trace(go.Scatter(x=ts.hour, y=ts.q90, line=dict(width=0), showlegend=False), 1, 1)
fig.add_trace(go.Scatter(x=ts.hour, y=ts.q50, fill="tonexty", line=dict(width=0), name="P50–P90 band",
                         fillcolor="rgba(0,120,200,0.2)"), 1, 1)
fig.add_trace(go.Scatter(x=ts.hour, y=ts.point, name="point forecast", line=dict(dash="dot")), 1, 1)
fig.add_trace(go.Scatter(x=ts.hour, y=ts.load, name="actual load", line=dict(color="black", width=2)), 1, 1)
for n in sel:
    capn = (cap.assign(c=(1 + cap[f"n_on_{n}"]) * cap.cell_cap).groupby("hour").c.sum())
    fig.add_trace(go.Scatter(x=capn.index, y=capn.values, name=f"capacity {n}", line=dict(shape="hv")), 1, 1)
    non = d.groupby("hour")[f"n_on_{n}"].sum()
    fig.add_trace(go.Scatter(x=non.index, y=non.values, name=f"cells on {n}", line=dict(shape="hv"), showlegend=False), 2, 1)
fig.update_layout(height=650)
st.plotly_chart(fig, width="stretch")

agg = s[s.strategy.isin(sel)].groupby("strategy").agg(
    energy_kwh=("energy_wh", lambda x: x.sum() / 1000), violation_site_hours=("violation", "sum"),
    unserved=("unserved", "sum"), served=("served", "sum")).reset_index()
base = s[s.strategy == "always_on"].energy_wh.sum()
agg["saving_vs_always_on_pct"] = 100 * (1 - 1000 * agg.energy_kwh / base)
st.subheader("Selected day: energy and QoS violations per strategy")
st.dataframe(agg, width="stretch")
st.subheader("Whole test period (from results/summary.csv)")
st.dataframe(pd.read_csv(R / "summary.csv"), width="stretch")
if (R / "frontier.png").exists():
    st.image(str(R / "frontier.png"))

# ---- v6: rolling day-by-day loop per site (needs results_v6/ from `python run_v6.py`)
R6 = ROOT / "results_v6"
st.header("Rolling loop: forecast -> plan -> apply -> measure (v6)")
if not (R6 / f"decisions_rolling_seed{seed}.parquet").exists():
    st.info("No v6 results. Run `python run_v6.py` to see the rolling loop.")
else:
    rd = pd.read_parquet(R6 / f"decisions_rolling_seed{seed}.parquet")
    rnames = [c[5:] for c in rd.columns if c.startswith("n_on_")]
    rsite = st.selectbox("Site (rolling loop)", sorted(rd.site.unique()), key="rsite")
    rstrat = st.selectbox("Strategy", [n for n in rnames if n != "always_on"], key="rstrat",
                          index=max(0, [n for n in rnames if n != "always_on"].index("cpsat_cq90")
                                    if "cpsat_cq90" in rnames else 0))
    g = rd[rd.site == rsite].sort_values("hour")
    rsim = simulate(g.reset_index(drop=True), cells, rnames, pm)
    cc = float(cells.set_index("site").at[rsite, "cell_cap"])
    f2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=("Load, re-fitted daily forecast band and planned capacity",
                                       "Active capacity cells (plan applied)", "Energy per hour (Wh)"))
    f2.add_trace(go.Scatter(x=g.hour, y=g.q90, line=dict(width=0), showlegend=False), 1, 1)
    f2.add_trace(go.Scatter(x=g.hour, y=g.q50, fill="tonexty", line=dict(width=0), name="P50–P90 (raw)",
                            fillcolor="rgba(0,120,200,0.2)"), 1, 1)
    f2.add_trace(go.Scatter(x=g.hour, y=g.cq90, name="conformal P90", line=dict(dash="dash")), 1, 1)
    f2.add_trace(go.Scatter(x=g.hour, y=g.load, name="actual load", line=dict(color="black", width=2)), 1, 1)
    f2.add_trace(go.Scatter(x=g.hour, y=(1 + g[f"n_on_{rstrat}"]) * cc, name=f"capacity {rstrat}",
                            line=dict(shape="hv")), 1, 1)
    sv = rsim[rsim.strategy == rstrat]
    viol = sv[sv.violation == 1]
    f2.add_trace(go.Scatter(x=viol.hour, y=g.set_index("hour").loc[viol.hour, "load"], mode="markers",
                            marker=dict(color="red", size=9, symbol="x"), name="QoS violation"), 1, 1)
    f2.add_trace(go.Scatter(x=g.hour, y=g[f"n_on_{rstrat}"], line=dict(shape="hv"), showlegend=False), 2, 1)
    for n in ["always_on", rstrat]:
        e = rsim[rsim.strategy == n]
        f2.add_trace(go.Scatter(x=e.hour, y=e.energy_wh, name=f"energy {n}"), 3, 1)
    f2.update_layout(height=750)
    st.plotly_chart(f2, width="stretch")
    tot = rsim.groupby("strategy").agg(energy_kwh=("energy_wh", lambda x: x.sum() / 1000),
                                       violation_hours=("violation", "sum")).reset_index()
    b = tot.set_index("strategy").at["always_on", "energy_kwh"]
    tot["saving_vs_always_on_pct"] = 100 * (1 - tot.energy_kwh / b)
    st.dataframe(tot, width="stretch")
    if (R6 / "rolling_vs_oneshot.csv").exists():
        st.subheader("Rolling vs one-shot, all sites, pooled over seeds (results_v6/rolling_vs_oneshot.csv)")
        st.dataframe(pd.read_csv(R6 / "rolling_vs_oneshot.csv"), width="stretch")

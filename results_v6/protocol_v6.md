# GreenRAN AI v6 protocol (written before any v6 run)

Data, sites, cell sizing, power model, seeds (0, 1, 2) and the train / validation / test split are
unchanged from `results/` (train 11-11..11-24, validation 11-25..11-27, test 11-28..12-01).
`results/` is not touched; every v6 output goes to `results_v6/`. Run with `python run_v6.py`.

## A. Split conformal recalibration
- For each seed, the quantile model is fitted exactly as before. One additive offset per quantile level
  (q50, q80, q90, q95) is computed from validation residuals in site-normalised units,
  finite-sample level ceil((n+1)·tau)/n. Levels are re-sorted (running maximum) to stay non-crossing.
- Known caveat: the validation window is also used for early stopping of the forecaster (as in v5).
- Target: test coverage of conformal P90 within 88–92 %. Reported whatever it is.
- Strategies compared on test: `cpsat_q90` (raw), `cpsat_cq90` (conformal), `cpsat_point`, `cpsat_point_x1.10`.
- CIs: paired site-day cluster bootstrap (2000 draws), each cluster holds that site-day for all 3 seeds.

## B. Matched-energy comparison (selection rule fixed here)
- Target saving S* = validation energy saving of `cpsat_point_x1.10`, pooled over the 3 seeds.
- Families and grids (validation only):
  - point × m, m = 1.00 … 1.30 step 0.01
  - raw q90 × m, m = 0.80 … 1.20 step 0.01
  - conformal q90 at level tau, tau = 0.70 … 0.99 step 0.01 (offset from validation)
  - reactive threshold thr, thr = 0.60 … 0.98 step 0.02
- For each family, pick the one parameter (shared by all seeds) whose pooled validation saving is
  closest to S*; ties go to the lower saving. Parameters are then frozen and evaluated once on test.
- Report test saving and QoS violations with the pooled cluster bootstrap and paired differences
  against the point family. Matching is on validation, so test savings will not be exactly equal;
  the test saving gap is reported next to the violation difference.

## C. Rolling day-by-day mode
- For each test day d (4 days) and seed: refit point and quantile models on 11-11 .. d−4 days,
  use d−3 .. d−1 for early stopping and conformal calibration, forecast day d, plan with CP-SAT
  starting from the cell state actually applied at the end of day d−1, apply to observed traffic, measure.
- Cell sizing and site scales stay those fitted on the original training window.
- Compared against the one-shot plan of section A for the same strategies.

"""`python run_v6.py --quick` (1 seed, 300 bootstrap draws): fast checks always, the real run opt-in.

The real quick run takes about 3.5 minutes (measured: 204 s on an 8-core x86_64 box, Python 3.11), so it is
skipped in the default `pytest` run and executed by its own CI job:

    GREENRAN_QUICK_TEST=1 python -m pytest -q tests/test_quick_mode.py

It writes to a temporary --out-dir, never to the committed results_v6/.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from make_report import write_report_v6  # noqa: E402

REPORT_INPUTS = ["coverage.csv", "conformal_summary.csv", "conformal_paired.csv", "matched_energy_test.csv",
                 "matched_energy_paired.csv", "rolling_vs_oneshot.csv", "rolling_vs_oneshot_paired.csv",
                 "rolling_coverage.csv", "run_meta_v6.json"]


def test_run_v6_cli_documents_quick_and_out_dir():
    out = subprocess.run([sys.executable, str(ROOT / "run_v6.py"), "--help"], capture_output=True, text=True,
                         check=True, timeout=120).stdout
    assert "--quick" in out and "--out-dir" in out


def test_v6_report_rebuilds_from_committed_csvs_into_a_temp_dir(tmp_path):
    for name in REPORT_INPUTS:
        shutil.copy(ROOT / "results_v6" / name, tmp_path / name)
    write_report_v6(ROOT, tmp_path)
    assert (tmp_path / "results_v6.md").read_text() == (ROOT / "results_v6" / "results_v6.md").read_text()


@pytest.mark.skipif(os.environ.get("GREENRAN_QUICK_TEST") != "1",
                    reason="about 3.5 min; set GREENRAN_QUICK_TEST=1 (the CI 'quick' job does)")
def test_run_v6_quick_end_to_end(tmp_path):
    out = tmp_path / "quick"
    committed = (ROOT / "results_v6" / "run_meta_v6.json").read_bytes()
    subprocess.run([sys.executable, str(ROOT / "run_v6.py"), "--quick", "--out-dir", str(out)],
                   check=True, cwd=str(ROOT), capture_output=True, text=True, timeout=1500)
    meta = json.loads((out / "run_meta_v6.json").read_text())
    assert meta["command"] == "python run_v6.py --quick" and meta["seeds"] == [0] and meta["n_bootstrap"] == 300
    for name in REPORT_INPUTS + ["results_v6.md", "decisions_oneshot_seed0.parquet", "decisions_rolling_seed0.parquet"]:
        assert (out / name).exists(), name
    assert not (out / "decisions_oneshot_seed1.parquet").exists()  # only one seed
    s = pd.read_csv(out / "conformal_summary.csv").set_index("strategy")
    assert set(s.index) == {"always_on", "cpsat_point", "cpsat_point_x1.10", "cpsat_q90", "cpsat_cq90"}
    assert s.loc["always_on", "saving_pct"] == 0
    assert (s.saving_pct.drop("always_on") > 0).all() and s.violation_rate_pct.between(0, 100).all()
    assert s.loc["cpsat_q90", "violation_rate_pct"] < s.loc["cpsat_point", "violation_rate_pct"]
    assert (ROOT / "results_v6" / "run_meta_v6.json").read_bytes() == committed  # committed results untouched

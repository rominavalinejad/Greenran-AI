"""Download and preprocess a subset of the Telecom Italia Milan dataset.

Source: Telecom Italia, "Telecommunications - SMS, Call, Internet - MI",
Harvard Dataverse, doi:10.7910/DVN/EGZHFV (ODbL licence).
Each daily file has tab-separated rows:
  square_id, time_interval_ms, country_code, sms_in, sms_out, call_in, call_out, internet
in 10-minute slots. We keep a block of grid squares, sum over country codes and
aggregate to hourly totals. Grid activity is used as a *proxy* for cell load.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DOI = "doi:10.7910/DVN/EGZHFV"
API = "https://dataverse.harvard.edu/api"
UA = {"User-Agent": "curl/8.0"}
COLS = ["square", "ts", "cc", "sms_in", "sms_out", "call_in", "call_out", "internet"]


def select_squares(row0: int = 45, col0: int = 40, size: int = 20) -> list[int]:
    """Square ids of a size x size block of the 100x100 Milan grid (id = row*100 + col + 1)."""
    return [r * 100 + c + 1 for r in range(row0, row0 + size) for c in range(col0, col0 + size)]


def list_files() -> dict[str, int]:
    url = f"{API}/datasets/:persistentId/?persistentId={DATASET_DOI}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        d = json.load(r)
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in d["data"]["latestVersion"]["files"]}


def signed_url(file_id: int, guestbook: dict) -> str:
    """The dataset is guestbook-gated; the Dataverse API returns a signed URL after a guestbook response."""
    req = urllib.request.Request(
        f"{API}/access/datafile/{file_id}",
        data=json.dumps({"guestbookResponse": guestbook}).encode(),
        headers={"Content-Type": "application/json", **UA}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]["signedUrl"]


def download_day(day: str, file_id: int, squares: list[int], out_dir: Path, guestbook: dict) -> Path:
    out = out_dir / f"{day}.parquet"
    if out.exists():
        return out
    url = signed_url(file_id, guestbook)
    ids = ",".join(str(s) for s in squares)
    # stream through awk so the ~350 MB daily file is never stored in full
    awk = f'BEGIN{{n=split("{ids}",a,",");for(i=1;i<=n;i++)k[a[i]]=1;FS="\\t"}} ($1 in k)'
    cmd = f"curl -sSL --retry 5 '{url}' | awk '{awk}'"
    txt = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
    df = pd.read_csv(io.StringIO(txt), sep="\t", names=COLS, header=None)
    df = df.fillna(0.0)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("Europe/Rome").dt.tz_localize(None)
    df["hour"] = df["time"].dt.floor("h")
    df["calls"] = df["call_in"] + df["call_out"]
    df["sms"] = df["sms_in"] + df["sms_out"]
    h = df.groupby(["square", "hour"], as_index=False)[["internet", "calls", "sms"]].sum()
    tmp = out.with_suffix(".tmp")
    h.to_parquet(tmp, index=False)
    tmp.rename(out)
    return out


EMAIL_PLACEHOLDER = "<EMAIL_FROM_ENV>"
EMAIL_ENV = "GREENRAN_GUESTBOOK_EMAIL"


def resolve_guestbook(guestbook: dict, env=None) -> dict:
    """Return a copy of the guestbook with the email placeholder replaced from the environment.

    The address is never stored in the repository; set GREENRAN_GUESTBOOK_EMAIL before downloading.
    """
    env = os.environ if env is None else env
    gb = dict(guestbook)
    email = str(gb.get("email", "")).strip()
    if email in ("", EMAIL_PLACEHOLDER):
        email = str(env.get(EMAIL_ENV, "")).strip()
        if not email:
            raise RuntimeError(
                f"Guestbook email missing: set the {EMAIL_ENV} environment variable to the address "
                f"to submit with the Dataverse guestbook response (config.json keeps the placeholder {EMAIL_PLACEHOLDER}).")
    if "@" not in email or email.startswith("<"):
        raise RuntimeError(f"Guestbook email '{email}' is not a valid address; check {EMAIL_ENV}.")
    if email.lower().rsplit("@", 1)[-1] in ("example.com", "example.org", "example.net", "example.invalid"):
        raise RuntimeError(f"Guestbook email '{email}' is a placeholder domain; set {EMAIL_ENV} to a real address.")
    gb["email"] = email
    return gb


def download(days: list[str], squares: list[int], raw_dir: Path, guestbook: dict, workers: int = 4) -> None:
    guestbook = resolve_guestbook(guestbook)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = list_files()
    jobs = []
    with ThreadPoolExecutor(workers) as ex:
        for d in days:
            fn = f"sms-call-internet-mi-{d}.txt"
            if fn not in files:
                raise FileNotFoundError(fn)
            jobs.append(ex.submit(download_day, d, files[fn], squares, raw_dir, guestbook))
        for j in jobs:
            print("downloaded", j.result(), flush=True)


def build_hourly(days: list[str], raw_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(raw_dir / f"{d}.parquet") for d in days]
    df = pd.concat(parts).groupby(["square", "hour"], as_index=False).sum()
    # timestamps near midnight can spill across files; keep only requested days, full grid of hours
    start, end = pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(hours=23)
    hours = pd.date_range(start, end, freq="h")
    idx = pd.MultiIndex.from_product([sorted(df.square.unique()), hours], names=["square", "hour"])
    df = df.set_index(["square", "hour"]).reindex(idx).reset_index()
    # DST change not in Nov/Dec window; missing hours are filled by interpolation per square
    df[["internet", "calls", "sms"]] = df.groupby("square")[["internet", "calls", "sms"]].transform(
        lambda s: s.interpolate(limit_direction="both"))
    return df


def day_list(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n, freq="D")]

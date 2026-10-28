"""Download the Milan subset defined in config.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from greenran.data import day_list, download, select_squares

cfg = json.load(open(Path(__file__).parent / "config.json"))
d = cfg["data"]
download(day_list(d["start"], d["n_days"]), select_squares(**d["block"]), Path(d["raw_dir"]), d["guestbook"], d["workers"])

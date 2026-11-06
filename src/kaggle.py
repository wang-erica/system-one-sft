"""Kaggle source normalization and leaderboard evidence (no solution code execution)."""
from __future__ import annotations

import csv
import io
import re
import subprocess
from urllib.parse import urlparse


def parse_source(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"kaggle.com", "www.kaggle.com"}:
        raise ValueError("expected an https://www.kaggle.com competition or dataset URL")
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"c", "competitions"}:
        kind, ref = "competition", parts[1]
    elif len(parts) >= 3 and parts[0] == "datasets":
        kind, ref = "dataset", "/".join(parts[1:3])
    else:
        raise ValueError("use /competitions/<slug> or /datasets/<owner>/<slug>")
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", part) for part in ref.split("/")):
        raise ValueError("invalid Kaggle reference")
    return {"kind": kind, "ref": ref, "url": f"https://www.kaggle.com/{'competitions' if kind == 'competition' else 'datasets'}/{ref}"}


def discover(url: str, runner=subprocess.run) -> dict:
    source = parse_source(url)
    if source["kind"] == "dataset":
        return {"source": source, "status": "no_leaderboard", "leader": None}
    command = ["kaggle", "competitions", "leaderboard", source["ref"], "--show", "--csv"]
    result = runner(command, capture_output=True, text=True, check=True, timeout=120)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if not rows:
        raise ValueError("Kaggle returned an empty leaderboard")
    # Preserve server order: scores may be higher-is-better or lower-is-better.
    return {"source": source, "status": "leader_identified", "leader": rows[0],
            "leaderboard_csv": result.stdout, "command": command,
            "solution_status": "requires published solution and runnable adapter",
            "note": "Server standings: final for concluded competitions, otherwise current; verify eligibility and competition status before calling this a winner."}

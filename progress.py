"""
progress.py — Progress history persistence and chart data preparation.

Stores daily snapshots of projected and actual FCHL standings in a JSON file,
and converts them to DataFrames for Altair charting.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

HISTORY_FILE = Path(__file__).parent / "data" / "progress_history.json"


def load_history() -> dict:
    """Load progress history from JSON. Returns dict with 'snapshots' list."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "snapshots" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"snapshots": []}


def save_history(history: dict) -> None:
    """Write progress history to JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def record_snapshot(
    history: dict,
    snapshot_date: str,
    projected: dict[str, float],
    actual: dict[str, int],
) -> tuple[dict, bool]:
    """
    Add or overwrite a snapshot for the given date.
    Returns (updated history, was_overwrite).
    """
    snapshots = history["snapshots"]
    was_overwrite = any(s["date"] == snapshot_date for s in snapshots)
    snapshots = [s for s in snapshots if s["date"] != snapshot_date]
    snapshots.append({
        "date": snapshot_date,
        "projected": projected,
        "actual": actual,
    })
    snapshots.sort(key=lambda s: s["date"])
    history["snapshots"] = snapshots
    return history, was_overwrite


def history_to_dataframe(history: dict) -> pd.DataFrame:
    """
    Convert history into a long-form DataFrame suitable for Altair.
    Columns: date, team, projected, actual
    """
    rows = []
    for snap in history["snapshots"]:
        d = snap["date"]
        for team in snap["projected"]:
            rows.append({
                "date": d,
                "team": team,
                "projected": snap["projected"].get(team),
                "actual": snap["actual"].get(team),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "team", "projected", "actual"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

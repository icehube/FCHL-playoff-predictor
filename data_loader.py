"""
data_loader.py — CSV parsing, name matching, and schedule stat derivation.
"""
import csv
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
from thefuzz import process as fuzz_process, fuzz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NHL_TEAM_MAP: dict[str, str] = {
    "Anaheim Ducks":        "ANA",
    "Boston Bruins":        "BOS",
    "Buffalo Sabres":       "BUF",
    "Calgary Flames":       "CGY",
    "Carolina Hurricanes":  "CAR",
    "Chicago Blackhawks":   "CHI",
    "Colorado Avalanche":   "COL",
    "Columbus Blue Jackets":"CBJ",
    "Dallas Stars":         "DAL",
    "Detroit Red Wings":    "DET",
    "Edmonton Oilers":      "EDM",
    "Florida Panthers":     "FLA",
    "Los Angeles Kings":    "LAK",
    "Minnesota Wild":       "MIN",
    "Montreal Canadiens":   "MTL",
    "Nashville Predators":  "NSH",
    "New Jersey Devils":    "NJD",
    "New York Islanders":   "NYI",
    "New York Rangers":     "NYR",
    "Ottawa Senators":      "OTT",
    "Philadelphia Flyers":  "PHI",
    "Pittsburgh Penguins":  "PIT",
    "San Jose Sharks":      "SJS",
    "Seattle Kraken":       "SEA",
    "St. Louis Blues":      "STL",
    "Tampa Bay Lightning":  "TBL",
    "Toronto Maple Leafs":  "TOR",
    "Utah Mammoth":         "UTA",
    "Vancouver Canucks":    "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals":  "WSH",
    "Winnipeg Jets":        "WPG",
}

DEFAULT_FCHL_POINTS: dict[str, int] = {
    "GVR": 1172,
    "LPT": 1157,
    "ZSK": 1117,
    "BOT": 1087,
    "SRL": 1075,
    "MAC": 1041,
}

FCHL_TEAMS = ["BOT", "GVR", "LPT", "MAC", "SRL", "ZSK"]

# ---------------------------------------------------------------------------
# FCHL Roster
# ---------------------------------------------------------------------------

def load_fchl_roster(path: str) -> list[dict]:
    """
    Load FCHL roster from players.csv (exported from FCHL Online).
    Filters to our 6 FCHL teams and status=='Lineup'.
    Returns list of dicts: {name, position, fchl_team}.
    """
    df = pd.read_csv(path, low_memory=False)
    active = df[df["abb"].isin(FCHL_TEAMS) & (df["status"] == "Lineup")]

    players = []
    for _, row in active.iterrows():
        players.append({
            "name": str(row["name"]).strip(),
            "position": str(row["position"]).strip(),
            "fchl_team": str(row["abb"]).strip(),
        })
    return players



# ---------------------------------------------------------------------------
# Skater stats
# ---------------------------------------------------------------------------

def load_skater_stats(path: str) -> dict[str, dict]:
    """
    Load skaters.csv, filter to situation=='all'.
    Returns dict keyed by player name.
    """
    df = pd.read_csv(path)
    df = df[df["situation"] == "all"].copy()

    stats: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        stats[name] = {
            "name": name,
            "nhl_team": str(row["team"]).strip(),
            "position": str(row.get("position", "")).strip(),
            "games_played": float(row["games_played"]),
            "goals": float(row.get("I_F_goals", 0) or 0),
            "primary_assists": float(row.get("I_F_primaryAssists", 0) or 0),
            "secondary_assists": float(row.get("I_F_secondaryAssists", 0) or 0),
        }
    return stats


# ---------------------------------------------------------------------------
# Goalie stats
# ---------------------------------------------------------------------------

def load_goalie_stats(path: str) -> dict[str, dict]:
    """
    Load goalies.csv, filter to situation=='all'.
    Returns dict keyed by player name.
    Wins/shutouts come from the schedule, not this file.
    """
    df = pd.read_csv(path)
    df = df[df["situation"] == "all"].copy()

    stats: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        stats[name] = {
            "name": name,
            "nhl_team": str(row["team"]).strip(),
            "games_played": float(row["games_played"]),
        }
    return stats


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def load_schedule(path: str) -> dict:
    """
    Load nhl-202526-asplayed.csv using csv.reader (positional indexing)
    because the file has two columns both named 'Score'.

    Column indices (0-based, after header row):
      0: Date, 1: Start Time (Sask), 2: Start Time (ET),
      3: Visitor, 4: Visitor Score, 5: Home, 6: Home Score,
      7: Status, 8: Visitor Goalie, 9: Home Goalie, ...

    Returns a dict with:
      team_completed: {nhl_abbr: int}  — completed games per team
      team_remaining: {nhl_abbr: int}  — scheduled games per team
      goalie_schedule_stats: {goalie_name: {starts, wins, shutouts}}
    """
    team_completed: dict[str, int] = {}
    team_remaining: dict[str, int] = {}
    goalie_stats: dict[str, dict] = {}

    def _goalie_entry(name: str) -> dict:
        if name not in goalie_stats:
            goalie_stats[name] = {"starts": 0, "wins": 0, "shutouts": 0}
        return goalie_stats[name]

    today = date.today().isoformat()  # "YYYY-MM-DD"

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if len(row) < 8:
                continue

            game_date = row[0].strip()
            visitor_full = row[3].strip()
            home_full = row[5].strip()
            status = row[7].strip()

            visitor_abbr = NHL_TEAM_MAP.get(visitor_full)
            home_abbr = NHL_TEAM_MAP.get(home_full)

            if game_date >= today:
                # Today or future — count as remaining
                if visitor_abbr:
                    team_remaining[visitor_abbr] = team_remaining.get(visitor_abbr, 0) + 1
                if home_abbr:
                    team_remaining[home_abbr] = team_remaining.get(home_abbr, 0) + 1
            else:
                # Past game — count as completed
                if visitor_abbr:
                    team_completed[visitor_abbr] = team_completed.get(visitor_abbr, 0) + 1
                if home_abbr:
                    team_completed[home_abbr] = team_completed.get(home_abbr, 0) + 1

                # Only extract goalie stats if scores are present
                if status == "Scheduled":
                    continue

                visitor_goalie = row[8].strip() if len(row) > 8 else ""
                home_goalie = row[9].strip() if len(row) > 9 else ""

                # Parse scores
                try:
                    v_score = int(row[4].strip())
                    h_score = int(row[6].strip())
                except (ValueError, IndexError):
                    continue

                # Goalie starts
                if visitor_goalie:
                    _goalie_entry(visitor_goalie)["starts"] += 1
                if home_goalie:
                    _goalie_entry(home_goalie)["starts"] += 1

                # Wins (one winner always — no ties in NHL)
                if visitor_goalie and home_goalie:
                    if v_score > h_score:
                        _goalie_entry(visitor_goalie)["wins"] += 1
                    else:
                        _goalie_entry(home_goalie)["wins"] += 1

                # Shutouts
                if home_goalie and v_score == 0:
                    _goalie_entry(home_goalie)["shutouts"] += 1
                if visitor_goalie and h_score == 0:
                    _goalie_entry(visitor_goalie)["shutouts"] += 1

    return {
        "team_completed": team_completed,
        "team_remaining": team_remaining,
        "goalie_schedule_stats": goalie_stats,
    }


# ---------------------------------------------------------------------------
# NHL API — live goalie stats & standings
# ---------------------------------------------------------------------------

NHL_SEASON_ID = "20252026"
STALE_SECONDS = 24 * 60 * 60  # 1 day

_NHL_GOALIE_URL = (
    "https://api.nhle.com/stats/rest/en/goalie/summary"
    "?isAggregate=false&isGame=false"
    "&sort=%5B%7B%22property%22%3A%22wins%22%2C%22direction%22%3A%22DESC%22%7D%5D"
    f"&cayenneExp=seasonId%3D{NHL_SEASON_ID}%20and%20gameTypeId%3D2"
    "&limit=-1"
)

_NHL_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/{date}"


def _fetch_json(url: str) -> dict:
    """Fetch JSON from a URL, returning parsed dict."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_nhl_goalie_stats(cache_path: Path) -> dict[str, dict]:
    """
    Fetch goalie stats (wins, shutouts, gamesStarted) from the NHL API.
    Caches to a local JSON file; re-fetches if older than 24h.
    Returns {goalie_name: {"starts", "wins", "shutouts", "nhl_team"}}.
    """
    # Check cache freshness
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < STALE_SECONDS:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

    try:
        data = _fetch_json(_NHL_GOALIE_URL)
    except Exception:
        # Fall back to cache if API fails
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    result: dict[str, dict] = {}
    for g in data.get("data", []):
        name = g.get("goalieFullName", "").strip()
        if not name:
            continue
        # teamAbbrevs may be "OTT, COL" for traded goalies — use last
        team_raw = g.get("teamAbbrevs", "")
        team = team_raw.split(",")[-1].strip() if team_raw else ""
        result[name] = {
            "starts": g.get("gamesStarted", 0),
            "wins": g.get("wins", 0),
            "shutouts": g.get("shutouts", 0),
            "nhl_team": team,
        }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def fetch_nhl_standings(cache_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """
    Fetch team standings from the NHL API to get games played.
    Returns (team_completed, team_remaining) dicts keyed by NHL abbreviation.
    """
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < STALE_SECONDS:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
                return cached["team_completed"], cached["team_remaining"]

    today = date.today().isoformat()
    try:
        data = _fetch_json(_NHL_STANDINGS_URL.format(date=today))
    except Exception:
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
                return cached["team_completed"], cached["team_remaining"]
        return {}, {}

    team_completed: dict[str, int] = {}
    team_remaining: dict[str, int] = {}
    for t in data.get("standings", []):
        abbrev = t.get("teamAbbrev", {}).get("default", "")
        gp = t.get("gamesPlayed", 0)
        if abbrev:
            team_completed[abbrev] = gp
            team_remaining[abbrev] = 82 - gp

    cached = {"team_completed": team_completed, "team_remaining": team_remaining}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, indent=2)

    return team_completed, team_remaining


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def fuzzy_match_name(query: str, candidates: list[str], score_cutoff: int = 80) -> str | None:
    """
    Use thefuzz to find the best matching name above the cutoff.
    Returns the matched string or None.
    """
    if not candidates:
        return None
    result = fuzz_process.extractOne(query, candidates, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= score_cutoff:
        return result[0]
    return None


def build_player_lookup(
    fchl_roster: list[dict],
    skater_stats: dict[str, dict],
    goalie_stats: dict[str, dict],
) -> dict[str, str | None]:
    """
    For each FCHL player, find their matching key in the appropriate stats dict.
    Returns {fchl_player_name: matched_stats_key | None}.
    Run once at startup.
    """
    skater_names = list(skater_stats.keys())
    goalie_names = list(goalie_stats.keys())

    lookup: dict[str, str | None] = {}

    for player in fchl_roster:
        name = player["name"]
        if name in lookup:
            continue

        if player["position"] == "G":
            candidates = goalie_names
            pool = goalie_stats
        else:
            candidates = skater_names
            pool = skater_stats

        if name in pool:
            lookup[name] = name
        else:
            matched = fuzzy_match_name(name, candidates)
            lookup[name] = matched  # None if no match

    return lookup

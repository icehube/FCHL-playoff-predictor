"""
projections.py — Player projection calculations.

Scoring:
  Goals:    1 pt
  Assists:  1 pt
  Wins:     2 pts
  Shutouts: 3 pts
"""

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

GOAL_PTS = 1
ASSIST_PTS = 1
WIN_PTS = 2
SHUTOUT_PTS = 3


# ---------------------------------------------------------------------------
# Per-player projections
# ---------------------------------------------------------------------------

def project_skater(
    player: dict,
    stats_key: str | None,
    skater_stats: dict[str, dict],
    team_remaining: dict[str, int],
) -> dict:
    """
    Project remaining season fantasy points for a skater.
    Returns a PlayerProjection dict.
    """
    base = {
        "name": player["name"],
        "position": player["position"],
        "fchl_team": player["fchl_team"],
        "nhl_team": "",
        "remaining_games": 0,
        "proj_goals": 0.0,
        "proj_assists": 0.0,
        "proj_wins": 0.0,
        "proj_shutouts": 0.0,
        "proj_pts": 0.0,
        "found_in_stats": False,
    }

    if stats_key is None or stats_key not in skater_stats:
        return base

    s = skater_stats[stats_key]
    gp = s["games_played"]
    if gp <= 0:
        base["nhl_team"] = s["nhl_team"]
        base["found_in_stats"] = True
        base["remaining_games"] = team_remaining.get(s["nhl_team"], 0)
        return base

    remaining = team_remaining.get(s["nhl_team"], 0)
    gpg = s["goals"] / gp
    apg = (s["primary_assists"] + s["secondary_assists"]) / gp

    proj_goals = gpg * remaining
    proj_assists = apg * remaining
    proj_pts = (proj_goals * GOAL_PTS) + (proj_assists * ASSIST_PTS)

    return {
        **base,
        "nhl_team": s["nhl_team"],
        "remaining_games": remaining,
        "proj_goals": proj_goals,
        "proj_assists": proj_assists,
        "proj_pts": proj_pts,
        "found_in_stats": True,
    }


def project_goalie(
    player: dict,
    stats_key: str | None,
    goalie_stats: dict[str, dict],
    goalie_schedule_stats: dict[str, dict],
    team_completed: dict[str, int],
    team_remaining: dict[str, int],
) -> dict:
    """
    Project remaining season fantasy points for a goalie.

    Remaining starts are scaled proportionally to completed starts vs. team games.
    Returns a PlayerProjection dict.
    """
    base = {
        "name": player["name"],
        "position": player["position"],
        "fchl_team": player["fchl_team"],
        "nhl_team": "",
        "remaining_games": 0,
        "proj_goals": 0.0,
        "proj_assists": 0.0,
        "proj_wins": 0.0,
        "proj_shutouts": 0.0,
        "proj_pts": 0.0,
        "found_in_stats": False,
    }

    if stats_key is None or stats_key not in goalie_stats:
        return base

    g = goalie_stats[stats_key]
    nhl_team = g["nhl_team"]

    # Schedule-derived stats — key is the name as it appears in the schedule CSV.
    # The stats_key from fuzzy matching is the goalie_stats key (from goalies.csv),
    # which may differ from the schedule CSV name. Try both.
    sched = goalie_schedule_stats.get(stats_key) or goalie_schedule_stats.get(player["name"], {})
    starts = sched.get("starts", 0)
    wins = sched.get("wins", 0)
    shutouts = sched.get("shutouts", 0)

    team_comp = team_completed.get(nhl_team, 0)
    team_rem = team_remaining.get(nhl_team, 0)

    base["nhl_team"] = nhl_team
    base["remaining_games"] = team_rem
    base["found_in_stats"] = True

    if starts == 0 or team_comp == 0:
        return base

    win_rate = wins / starts
    shutout_rate = shutouts / starts
    remaining_starts = (starts / team_comp) * team_rem

    proj_wins = win_rate * remaining_starts
    proj_shutouts = shutout_rate * remaining_starts
    proj_pts = (proj_wins * WIN_PTS) + (proj_shutouts * SHUTOUT_PTS)

    return {
        **base,
        "proj_wins": proj_wins,
        "proj_shutouts": proj_shutouts,
        "proj_pts": proj_pts,
    }


def project_all_players(
    fchl_roster: list[dict],
    player_lookup: dict[str, str | None],
    skater_stats: dict[str, dict],
    goalie_stats: dict[str, dict],
    schedule_data: dict,
) -> list[dict]:
    """
    Project all FCHL players. Returns a list of PlayerProjection dicts.
    """
    team_remaining = schedule_data["team_remaining"]
    team_completed = schedule_data["team_completed"]
    goalie_schedule_stats = schedule_data["goalie_schedule_stats"]

    results = []
    for player in fchl_roster:
        stats_key = player_lookup.get(player["name"])
        if player["position"] == "G":
            proj = project_goalie(
                player, stats_key, goalie_stats,
                goalie_schedule_stats, team_completed, team_remaining,
            )
        else:
            proj = project_skater(player, stats_key, skater_stats, team_remaining)
        results.append(proj)

    return results


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------

def compute_standings(
    projections: list[dict],
    current_pts: dict[str, int],
) -> list[dict]:
    """
    Aggregate projections by FCHL team, add current_pts baseline.
    Returns a list of TeamStanding dicts sorted desc by proj_total.
    """
    team_proj: dict[str, float] = {}
    for proj in projections:
        team = proj["fchl_team"]
        team_proj[team] = team_proj.get(team, 0.0) + proj["proj_pts"]

    standings = []
    all_teams = set(list(team_proj.keys()) + list(current_pts.keys()))
    for team in all_teams:
        cur = current_pts.get(team, 0)
        rem = team_proj.get(team, 0.0)
        standings.append({
            "fchl_team": team,
            "current_pts": cur,
            "proj_remaining": rem,
            "proj_total": cur + rem,
        })

    standings.sort(key=lambda x: x["proj_total"], reverse=True)
    return standings

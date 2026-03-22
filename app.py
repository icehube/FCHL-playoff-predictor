"""
app.py — FCHL Season Predictor (Streamlit)

Simulates remaining NHL games and projects fantasy points for 6 FCHL teams.

Scoring: G=1pt, A=1pt, W=2pts, SO=3pts
Lineup:  12F, 6D, 2G per team
"""

import json
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import (
    DEFAULT_FCHL_POINTS,
    FCHL_TEAMS,
    build_player_lookup,
    fetch_nhl_goalie_stats,
    fetch_nhl_standings,
    load_fchl_roster,
    load_goalie_stats,
    load_schedule,
    load_skater_stats,
)
from progress import history_to_dataframe, load_history, record_snapshot, save_history
from projections import compute_standings, project_all_players

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
FCHL_CSV = str(BASE_DIR / "data" / "FCHL Players - Sheet1.csv")
ROSTER_JSON = BASE_DIR / "data" / "fchl_roster.json"
SCHEDULE_CSV = str(BASE_DIR / "data" / "nhl-202526-asplayed.csv")
SKATERS_CSV = BASE_DIR / "data" / "skaters.csv"
GOALIES_CSV = BASE_DIR / "data" / "goalies.csv"

NHL_GOALIE_CACHE = BASE_DIR / "data" / "nhl_goalie_stats.json"
NHL_STANDINGS_CACHE = BASE_DIR / "data" / "nhl_team_standings.json"

MONEYPUCK_URLS = {
    SKATERS_CSV: "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv",
    GOALIES_CSV: "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv",
}

STALE_SECONDS = 24 * 60 * 60  # 1 day


def refresh_stats_csvs():
    """Download skaters/goalies CSVs from MoneyPuck if older than 24 hours."""
    for local_path, url in MONEYPUCK_URLS.items():
        if local_path.exists():
            age = time.time() - local_path.stat().st_mtime
            if age < STALE_SECONDS:
                continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(local_path, "wb") as out:
                out.write(resp.read())
        except Exception as e:
            st.sidebar.warning(f"Could not update {local_path.name}: {e}")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def get_skater_stats():
    return load_skater_stats(str(SKATERS_CSV))

@st.cache_data
def get_goalie_stats():
    return load_goalie_stats(str(GOALIES_CSV))

@st.cache_data
def get_schedule(_today: str):
    """Build schedule_data from NHL API (goalie stats + standings)."""
    nhl_goalie_stats = fetch_nhl_goalie_stats(NHL_GOALIE_CACHE)
    team_completed, team_remaining = fetch_nhl_standings(NHL_STANDINGS_CACHE)

    # If API data is available, use it
    if nhl_goalie_stats and team_completed:
        return {
            "team_completed": team_completed,
            "team_remaining": team_remaining,
            "goalie_schedule_stats": nhl_goalie_stats,
        }

    # Fall back to schedule CSV if API failed
    return load_schedule(SCHEDULE_CSV)

@st.cache_data
def get_original_roster():
    return load_fchl_roster(FCHL_CSV)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slot_label_df(df: pd.DataFrame, pos_col: str = "Pos") -> pd.DataFrame:
    """
    Insert a '#' column like 'F1', 'F2', 'D3', 'G1' based on each row's
    position and its order in the DataFrame.  Call this after sorting.
    """
    counts: dict[str, int] = {}
    slots = []
    for pos in df[pos_col]:
        counts[pos] = counts.get(pos, 0) + 1
        slots.append(f"{pos}{counts[pos]}")
    df = df.copy()
    df.insert(0, "#", slots)
    return df


def position_counts(roster: list[dict], fchl_team: str) -> str:
    """Return a quick count string like '12F  6D  2G' for a team."""
    team = [p for p in roster if p["fchl_team"] == fchl_team]
    f = sum(1 for p in team if p["position"] == "F")
    d = sum(1 for p in team if p["position"] == "D")
    g = sum(1 for p in team if p["position"] == "G")
    return f"{f}F  {d}D  {g}G"

# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FCHL Season Predictor",
    page_icon="🏒",
    layout="wide",
)

st.title("🏒 FCHL Season Predictor")
st.caption(
    "Projects remaining 2025-26 NHL season fantasy points for each FCHL team. "
    "Scoring: G=1, A=1, W=2, SO=3"
)

# ---------------------------------------------------------------------------
# Refresh stats CSVs from MoneyPuck (daily)
# ---------------------------------------------------------------------------

# Track mtimes so we can bust caches when files are refreshed.
_pre_mtimes = {p: p.stat().st_mtime if p.exists() else 0 for p in MONEYPUCK_URLS}
refresh_stats_csvs()
for _path in MONEYPUCK_URLS:
    if _path.exists() and _path.stat().st_mtime != _pre_mtimes[_path]:
        get_skater_stats.clear()
        get_goalie_stats.clear()
        break

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

skater_stats = get_skater_stats()
goalie_stats = get_goalie_stats()
schedule_data = get_schedule(date.today().isoformat())
progress_history = load_history()

# ---------------------------------------------------------------------------
# Session state — mutable roster & lookup
# ---------------------------------------------------------------------------

if "roster" not in st.session_state:
    if ROSTER_JSON.exists():
        with open(ROSTER_JSON, encoding="utf-8") as _f:
            _saved = json.load(_f)
        _required = {"name", "position", "fchl_team"}
        if isinstance(_saved, list) and all(_required <= set(p.keys()) for p in _saved):
            st.session_state.roster = [
                {
                    "raw": f"{p['position']} {p['name']} (saved)",
                    "name": p["name"],
                    "position": p["position"],
                    "fchl_team": p["fchl_team"],
                }
                for p in _saved
            ]
        else:
            st.session_state.roster = list(get_original_roster())
    else:
        st.session_state.roster = list(get_original_roster())  # mutable copy

if "player_lookup" not in st.session_state:
    with st.spinner("Building player name index…"):
        st.session_state.player_lookup = build_player_lookup(
            st.session_state.roster, skater_stats, goalie_stats
        )

# ---------------------------------------------------------------------------
# Sidebar — editable current points
# ---------------------------------------------------------------------------

st.sidebar.header("Current FCHL Points")
st.sidebar.caption("Update these to reflect your league's current standings.")

SIDEBAR_TEAM_ORDER = ["LPT", "GVR", "ZSK", "SRL", "BOT", "MAC"]

current_pts: dict[str, int] = {}
for team in SIDEBAR_TEAM_ORDER:
    current_pts[team] = st.sidebar.number_input(
        label=team,
        value=DEFAULT_FCHL_POINTS.get(team, 0),
        step=1,
        min_value=0,
        key=f"sidebar_pts_{team}",
    )

st.sidebar.divider()
remaining_games = schedule_data["team_remaining"]
total_remaining = sum(remaining_games.values()) // 2  # each game counted twice
st.sidebar.metric("Remaining NHL Games", total_remaining)

# Stats CSV freshness
st.sidebar.divider()
st.sidebar.caption("**Player Stats (MoneyPuck)**")
for _csv_path, _label in [(SKATERS_CSV, "Skaters"), (GOALIES_CSV, "Goalies")]:
    if _csv_path.exists():
        _mtime = datetime.fromtimestamp(_csv_path.stat().st_mtime, tz=timezone.utc).astimezone()
        st.sidebar.text(f"{_label}: {_mtime.strftime('%b %d, %Y %I:%M %p')}")
    else:
        st.sidebar.text(f"{_label}: not found")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Standings", "👤 Player Projections", "🔧 Roster Builder", "📈 Progress Tracker", "🔄 Trade Simulator"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Standings
# ---------------------------------------------------------------------------

with tab1:
    projections = project_all_players(
        st.session_state.roster,
        st.session_state.player_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )
    standings = compute_standings(projections, current_pts)

    st.subheader("Projected Final Standings")
    st.caption("Current points + projected remaining fantasy points for each team.")

    rows = []
    for i, s in enumerate(standings):
        rows.append({
            "Rank": i + 1,
            "Team": s["fchl_team"],
            "Roster": position_counts(st.session_state.roster, s["fchl_team"]),
            "Current Pts": s["current_pts"],
            "Proj Remaining": round(s["proj_remaining"], 1),
            "Proj Total": round(s["proj_total"], 1),
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Proj Total": st.column_config.NumberColumn(
                help="Current Pts + Projected Remaining"
            ),
        },
    )

    # Per-team breakdown
    st.subheader("Team Breakdown")
    selected_team = st.selectbox(
        "Select team to inspect", sorted(FCHL_TEAMS), key="standings_team_select"
    )
    team_projs = [p for p in projections if p["fchl_team"] == selected_team]

    if team_projs:
        df_team = pd.DataFrame(team_projs)

        skater_df = df_team[df_team["position"] != "G"][
            ["name", "position", "nhl_team", "remaining_games", "proj_goals", "proj_assists", "proj_pts", "found_in_stats"]
        ].copy()
        goalie_df = df_team[df_team["position"] == "G"][
            ["name", "position", "nhl_team", "remaining_games", "proj_wins", "proj_shutouts", "proj_pts", "found_in_stats"]
        ].copy()

        for col in ["proj_goals", "proj_assists", "proj_pts"]:
            skater_df[col] = skater_df[col].round(1)
        for col in ["proj_wins", "proj_shutouts", "proj_pts"]:
            goalie_df[col] = goalie_df[col].round(1)

        skater_df = skater_df.sort_values("proj_pts", ascending=False)
        skater_df = skater_df.rename(columns={
            "name": "Player", "position": "Pos", "nhl_team": "NHL Team",
            "remaining_games": "Rem GP",
            "proj_goals": "Proj G", "proj_assists": "Proj A",
            "proj_pts": "Proj Pts", "found_in_stats": "Found",
        })
        skater_df = slot_label_df(skater_df)

        goalie_df = goalie_df.sort_values("proj_pts", ascending=False)
        goalie_df = goalie_df.rename(columns={
            "name": "Player", "position": "Pos", "nhl_team": "NHL Team",
            "remaining_games": "Rem GP",
            "proj_wins": "Proj W", "proj_shutouts": "Proj SO",
            "proj_pts": "Proj Pts", "found_in_stats": "Found",
        })
        goalie_df = slot_label_df(goalie_df)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Skaters**")
            st.dataframe(skater_df, hide_index=True, width="stretch")
        with col2:
            st.markdown("**Goalies**")
            st.dataframe(goalie_df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2 — Player Projections
# ---------------------------------------------------------------------------

with tab2:
    projections2 = project_all_players(
        st.session_state.roster,
        st.session_state.player_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )

    st.subheader("All Player Projections")

    # Filters
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        team_filter = st.selectbox(
            "Filter by FCHL Team", ["All"] + sorted(FCHL_TEAMS), key="proj_team_filter"
        )
    with fcol2:
        pos_filter = st.selectbox(
            "Filter by Position", ["All", "F", "D", "G"], key="proj_pos_filter"
        )

    df_all = pd.DataFrame(projections2)

    if team_filter != "All":
        df_all = df_all[df_all["fchl_team"] == team_filter]
    if pos_filter != "All":
        df_all = df_all[df_all["position"] == pos_filter]

    # Round and rename
    for col in ["proj_goals", "proj_assists", "proj_wins", "proj_shutouts", "proj_pts"]:
        df_all[col] = df_all[col].round(1)

    df_all = df_all.sort_values("proj_pts", ascending=False)

    display = df_all[[
        "name", "position", "fchl_team", "nhl_team", "remaining_games",
        "proj_goals", "proj_assists", "proj_wins", "proj_shutouts", "proj_pts",
        "found_in_stats",
    ]].rename(columns={
        "name": "Player",
        "position": "Pos",
        "fchl_team": "FCHL Team",
        "nhl_team": "NHL Team",
        "remaining_games": "Rem GP",
        "proj_goals": "Proj G",
        "proj_assists": "Proj A",
        "proj_wins": "Proj W",
        "proj_shutouts": "Proj SO",
        "proj_pts": "Proj Pts",
        "found_in_stats": "Found",
    })

    # Add slot numbers only when filtered to a single team (numbers are per-team)
    if team_filter != "All":
        display = slot_label_df(display)

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Found": st.column_config.CheckboxColumn(help="Player found in stats CSV"),
            "Proj Pts": st.column_config.NumberColumn(help="Projected remaining fantasy points"),
        },
    )

    # Warn about unmatched players
    unmatched = [p for p in projections2 if not p["found_in_stats"]]
    if unmatched:
        names = ", ".join(p["name"] for p in unmatched)
        st.warning(f"⚠️ {len(unmatched)} player(s) not found in stats (will project 0 pts): {names}")

    # Remaining games reference
    with st.expander("Remaining games per NHL team"):
        rg = schedule_data["team_remaining"]
        rg_df = pd.DataFrame(
            sorted(rg.items(), key=lambda x: -x[1]),
            columns=["NHL Team", "Remaining Games"],
        )
        st.dataframe(rg_df, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 — Roster Builder
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Roster Builder")
    st.caption(
        "Modify rosters to simulate trade scenarios or lineup changes. "
        "Changes here update the Standings and Player Projections tabs."
    )

    # --- Export / Import ---
    with st.expander("💾 Export / Import Rosters", expanded=False):
        exp_col, imp_col = st.columns(2)

        with exp_col:
            st.markdown("**Export current rosters**")
            roster_export = [
                {"name": p["name"], "position": p["position"], "fchl_team": p["fchl_team"]}
                for p in st.session_state.roster
            ]
            st.download_button(
                label="⬇️ Download roster JSON",
                data=json.dumps(roster_export, indent=2),
                file_name="fchl_roster.json",
                mime="application/json",
            )

        with imp_col:
            st.markdown("**Import saved rosters**")
            uploaded = st.file_uploader(
                "Upload roster JSON", type=["json"], key="roster_import",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                # Use (name, size) as a stable identity for this upload so we
                # don't re-process the same file on every subsequent rerun,
                # which would cause an infinite flicker loop.
                upload_key = (uploaded.name, uploaded.size)
                if st.session_state.get("_last_import_key") != upload_key:
                    try:
                        data = json.load(uploaded)
                        required = {"name", "position", "fchl_team"}
                        if isinstance(data, list) and all(required <= set(p.keys()) for p in data):
                            st.session_state.roster = [
                                {
                                    "raw": f"{p['position']} {p['name']} (imported)",
                                    "name": p["name"],
                                    "position": p["position"],
                                    "fchl_team": p["fchl_team"],
                                }
                                for p in data
                            ]
                            if "player_lookup" in st.session_state:
                                del st.session_state.player_lookup
                            st.session_state._last_import_key = upload_key
                            st.rerun()
                        else:
                            st.error("Invalid format — expected a list with name, position, fchl_team fields.")
                    except Exception as e:
                        st.error(f"Failed to load file: {e}")

    # --- Add a player ---
    with st.expander("➕ Add a player to a roster", expanded=False):
        all_stat_players = sorted(
            set(list(skater_stats.keys()) + list(goalie_stats.keys()))
        )
        # BUG FIX: derive exclusions from the live roster, not the lookup dict.
        # The lookup retains entries for removed players, so using it caused
        # removed players to stay hidden from the available list.
        current_stats_keys = {
            st.session_state.player_lookup.get(p["name"])
            for p in st.session_state.roster
            if st.session_state.player_lookup.get(p["name"])
        }
        available = sorted(set(all_stat_players) - current_stats_keys)

        acol1, acol2, acol3 = st.columns(3)
        with acol1:
            add_name = st.selectbox("Player", available, key="add_player_name")
        with acol2:
            add_pos = st.selectbox("Position", ["F", "D", "G"], key="add_player_pos")
        with acol3:
            add_team = st.selectbox("FCHL Team", sorted(FCHL_TEAMS), key="add_player_team")

        if st.button("Add Player", key="btn_add_player"):
            new_player = {
                "raw": f"{add_pos} {add_name} (added)",
                "name": add_name,
                "position": add_pos,
                "fchl_team": add_team,
            }
            st.session_state.roster.append(new_player)
            # Exact match since the name came directly from the stats dict keys
            st.session_state.player_lookup[add_name] = add_name
            # Jump the edit-team selector to the team the player was added to
            st.session_state.edit_team_select = add_team
            st.rerun()

    # --- Edit existing team ---
    st.markdown("#### Edit Team Roster")

    edit_team = st.selectbox(
        "Select team to edit", sorted(FCHL_TEAMS), key="edit_team_select"
    )

    team_players = [p for p in st.session_state.roster if p["fchl_team"] == edit_team]

    if not team_players:
        st.info("No players on this team.")
    else:
        # Group by position, numbered per group
        for pos_label, pos_code in [("Forwards", "F"), ("Defensemen", "D"), ("Goalies", "G")]:
            pos_players = [p for p in team_players if p["position"] == pos_code]
            if not pos_players:
                continue
            st.markdown(f"**{pos_label} ({len(pos_players)})**")
            for idx, player in enumerate(pos_players, start=1):
                slot = f"{pos_code}{idx}"
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"`{slot}` {player['name']}")
                with col2:
                    team_opts = sorted(FCHL_TEAMS)
                    cur_idx = team_opts.index(player["fchl_team"])
                    new_team = st.selectbox(
                        "Move to",
                        team_opts,
                        index=cur_idx,
                        key=f"move_{player['name']}_{player['fchl_team']}",
                        label_visibility="collapsed",
                    )
                    if new_team != player["fchl_team"]:
                        for p in st.session_state.roster:
                            if p["name"] == player["name"] and p["fchl_team"] == player["fchl_team"]:
                                p["fchl_team"] = new_team
                                break
                        st.rerun()
                with col3:
                    if st.button("Remove", key=f"remove_{player['name']}_{player['fchl_team']}"):
                        st.session_state.roster = [
                            p for p in st.session_state.roster
                            if not (p["name"] == player["name"] and p["fchl_team"] == player["fchl_team"])
                        ]
                        st.rerun()

    st.divider()

    # --- Projected standings with current (possibly modified) roster ---
    st.markdown("#### Projected Standings with Current Rosters")
    proj3 = project_all_players(
        st.session_state.roster,
        st.session_state.player_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )
    standings3 = compute_standings(proj3, current_pts)
    rows3 = []
    for i, s in enumerate(standings3):
        rows3.append({
            "Rank": i + 1,
            "Team": s["fchl_team"],
            "Roster": position_counts(st.session_state.roster, s["fchl_team"]),
            "Current Pts": s["current_pts"],
            "Proj Remaining": round(s["proj_remaining"], 1),
            "Proj Total": round(s["proj_total"], 1),
        })
    st.dataframe(pd.DataFrame(rows3), hide_index=True, width="stretch")

    st.divider()
    if st.button("🔄 Reset All Rosters to Original"):
        del st.session_state.roster
        del st.session_state.player_lookup
        st.rerun()

# ---------------------------------------------------------------------------
# Tab 4 — Progress Tracker
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Progress Tracker")
    st.caption(
        "Record daily snapshots of projected and actual standings to track trends over the season."
    )

    # Compute current standings (reuse same pattern as Tab 1)
    proj4 = project_all_players(
        st.session_state.roster,
        st.session_state.player_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )
    standings4 = compute_standings(proj4, current_pts)
    projected_totals = {s["fchl_team"]: round(s["proj_total"], 1) for s in standings4}

    # --- Record Snapshot ---
    st.markdown("#### Record Snapshot")

    snap_date = st.date_input("Snapshot date", value=date.today(), key="snap_date")

    st.markdown("**Actual FCHL points** (pre-filled from sidebar)")
    actual_cols = st.columns(len(SIDEBAR_TEAM_ORDER))
    actual_pts: dict[str, int] = {}
    for col, team in zip(actual_cols, SIDEBAR_TEAM_ORDER):
        with col:
            actual_pts[team] = st.number_input(
                team, value=current_pts.get(team, 0), step=1, min_value=0,
                key=f"actual_{team}",
            )

    st.markdown("**Projected totals** (auto-computed from current rosters & stats)")
    proj_cols = st.columns(len(SIDEBAR_TEAM_ORDER))
    for col, team in zip(proj_cols, SIDEBAR_TEAM_ORDER):
        with col:
            st.metric(team, projected_totals.get(team, 0))

    if st.button("💾 Record Snapshot", key="btn_record_snapshot"):
        progress_history, was_overwrite = record_snapshot(
            progress_history,
            snap_date.isoformat(),
            projected_totals,
            actual_pts,
        )
        save_history(progress_history)
        if was_overwrite:
            st.info(f"Snapshot for {snap_date} updated (overwritten).")
        else:
            st.success(f"Snapshot for {snap_date} recorded.")

    # --- Charts ---
    df_hist = history_to_dataframe(progress_history)

    if len(progress_history["snapshots"]) == 0:
        st.info("Record your first snapshot above to start tracking progress.")
    else:
        st.divider()

        # Chart 1: Projected Finals Over Time
        st.markdown("#### Projected Final Totals Over Time")
        chart_proj = (
            alt.Chart(df_hist)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("projected:Q", title="Projected Total Points",
                         scale=alt.Scale(zero=False)),
                color=alt.Color("team:N", title="FCHL Team"),
                tooltip=["date:T", "team:N",
                          alt.Tooltip("projected:Q", title="Projected", format=".1f")],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_proj, width='stretch')

        # Chart 2: Projected vs Actual
        st.markdown("#### Projected vs Actual")
        tracker_team_filter = st.selectbox(
            "Filter by team", ["All"] + SIDEBAR_TEAM_ORDER, key="tracker_team_filter"
        )

        df_chart2 = df_hist.copy()
        if tracker_team_filter != "All":
            df_chart2 = df_chart2[df_chart2["team"] == tracker_team_filter]

        df_long = df_chart2.melt(
            id_vars=["date", "team"],
            value_vars=["projected", "actual"],
            var_name="metric",
            value_name="points",
        )
        df_long["metric"] = df_long["metric"].str.capitalize()

        chart_compare = (
            alt.Chart(df_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("points:Q", title="Points", scale=alt.Scale(zero=False)),
                color=alt.Color("team:N", title="FCHL Team"),
                strokeDash=alt.StrokeDash("metric:N", title="Type"),
                tooltip=["date:T", "team:N", "metric:N",
                          alt.Tooltip("points:Q", format=".1f")],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_compare, width='stretch')

        # --- History Table ---
        with st.expander("View Snapshot History", expanded=False):
            history_rows = []
            for snap in progress_history["snapshots"]:
                row = {"Date": snap["date"]}
                for team in SIDEBAR_TEAM_ORDER:
                    row[f"{team} Proj"] = snap["projected"].get(team)
                    row[f"{team} Actual"] = snap["actual"].get(team)
                history_rows.append(row)
            st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")

            if st.button("🗑️ Clear All History", key="btn_clear_history"):
                progress_history = {"snapshots": []}
                save_history(progress_history)
                st.rerun()

# ---------------------------------------------------------------------------
# Tab 5 — Trade Simulator
# ---------------------------------------------------------------------------

with tab5:
    st.subheader("Trade Simulator — BOT")
    st.caption(
        "Evaluate trades for BOT without changing the actual roster. "
        "Select players to send away and bring in, then see the projected impact."
    )

    # --- Baseline: BOT projected remaining points ---
    trade_proj_base = project_all_players(
        st.session_state.roster,
        st.session_state.player_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )
    bot_base_pts = sum(
        p["proj_pts"] for p in trade_proj_base if p["fchl_team"] == "BOT"
    )

    # --- Player selection ---
    bot_roster = [p for p in st.session_state.roster if p["fchl_team"] == "BOT"]
    bot_names = [f"{p['name']} ({p['position']})" for p in bot_roster]

    out_col, in_col = st.columns(2)

    with out_col:
        st.markdown("**Players Out** (leaving BOT)")
        players_out = st.multiselect(
            "Select players to trade away",
            bot_names,
            key="trade_players_out",
            label_visibility="collapsed",
        )

    with in_col:
        st.markdown("**Players In** (coming to BOT)")
        # Build available list: all stats players not currently on BOT
        all_stat_skaters = {}
        for name, stats in skater_stats.items():
            all_stat_skaters[name] = "D" if stats.get("position") == "D" else "F"
        all_stat_goalies = {name: "G" for name in goalie_stats}
        all_stat_pool = {**all_stat_skaters, **all_stat_goalies}

        bot_stat_keys = {
            st.session_state.player_lookup.get(p["name"])
            for p in bot_roster
            if st.session_state.player_lookup.get(p["name"])
        }
        available_in = sorted(
            name for name in all_stat_pool if name not in bot_stat_keys
        )
        available_in_labels = [
            f"{name} ({all_stat_pool[name]})" for name in available_in
        ]

        players_in = st.multiselect(
            "Select players to acquire",
            available_in_labels,
            key="trade_players_in",
            label_visibility="collapsed",
        )

    # --- Compute post-trade projection ---
    # Parse selected names back out
    out_names = {s.rsplit(" (", 1)[0] for s in players_out}
    in_entries = []
    for label in players_in:
        name, pos_part = label.rsplit(" (", 1)
        pos = pos_part.rstrip(")")
        in_entries.append({"name": name, "position": pos})

    # Build modified roster
    trade_roster = [
        p for p in st.session_state.roster
        if not (p["fchl_team"] == "BOT" and p["name"] in out_names)
    ]
    trade_lookup = dict(st.session_state.player_lookup)
    for entry in in_entries:
        trade_roster.append({
            "raw": f"{entry['position']} {entry['name']} (trade-sim)",
            "name": entry["name"],
            "position": entry["position"],
            "fchl_team": "BOT",
        })
        trade_lookup[entry["name"]] = entry["name"]

    trade_proj_after = project_all_players(
        trade_roster,
        trade_lookup,
        skater_stats,
        goalie_stats,
        schedule_data,
    )
    bot_after_pts = sum(
        p["proj_pts"] for p in trade_proj_after if p["fchl_team"] == "BOT"
    )

    # --- Results ---
    st.divider()

    delta = bot_after_pts - bot_base_pts
    has_trade = bool(players_out or players_in)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Current Proj Remaining", f"{bot_base_pts:.1f}")
    with r2:
        st.metric(
            "After Trade",
            f"{bot_after_pts:.1f}" if has_trade else "—",
            delta=f"{delta:+.1f} pts" if has_trade else None,
            delta_color="normal",
        )
    with r3:
        standings_base = compute_standings(trade_proj_base, current_pts)
        bot_total_base = next(s["proj_total"] for s in standings_base if s["fchl_team"] == "BOT")
        if has_trade:
            standings_after = compute_standings(trade_proj_after, current_pts)
            bot_total_after = next(s["proj_total"] for s in standings_after if s["fchl_team"] == "BOT")
            st.metric(
                "Proj Season Total",
                f"{bot_total_after:.1f}",
                delta=f"{bot_total_after - bot_total_base:+.1f} pts",
                delta_color="normal",
            )
        else:
            st.metric("Proj Season Total", f"{bot_total_base:.1f}")

    # --- Breakdown tables ---
    if has_trade:
        st.divider()
        b1, b2 = st.columns(2)

        if players_out:
            with b1:
                st.markdown("**Losing**")
                out_rows = []
                for p in trade_proj_base:
                    if p["fchl_team"] == "BOT" and p["name"] in out_names:
                        out_rows.append({
                            "Player": p["name"],
                            "Pos": p["position"],
                            "Proj Pts": round(p["proj_pts"], 1),
                        })
                out_rows.sort(key=lambda r: -r["Proj Pts"])
                st.dataframe(pd.DataFrame(out_rows), hide_index=True, width="stretch")

        if players_in:
            with b2:
                st.markdown("**Gaining**")
                in_names = {e["name"] for e in in_entries}
                in_rows = []
                for p in trade_proj_after:
                    if p["fchl_team"] == "BOT" and p["name"] in in_names:
                        in_rows.append({
                            "Player": p["name"],
                            "Pos": p["position"],
                            "Proj Pts": round(p["proj_pts"], 1),
                        })
                in_rows.sort(key=lambda r: -r["Proj Pts"])
                st.dataframe(pd.DataFrame(in_rows), hide_index=True, width="stretch")

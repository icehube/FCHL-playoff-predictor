# FCHL Predictor

A personal Streamlit app that simulates the remaining 2025-26 NHL season and projects fantasy points for 6 FCHL teams. Feed it different roster configurations to compare projected final standings.

## Features

- **Schedule-accurate projections** — remaining games counted from the actual 2025-26 NHL schedule
- **Per-game rate model** — skater goals/assists projected from season-to-date pace
- **Goalie tracking** — wins and shutouts derived from completed schedule results
- **Roster Builder** — swap players between teams to evaluate trades or lineup changes
- **Progress Tracker** — record daily snapshots, chart projected vs actual standings over time
- **Editable standings** — update each team's current point total in the sidebar

## Scoring

| Stat    | Points |
|---------|--------|
| Goal    | 1      |
| Assist  | 1      |
| Win     | 2      |
| Shutout | 3      |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
FCHL-predictor/
├── app.py             # Streamlit UI (4 tabs: Standings, Players, Roster Builder, Progress Tracker)
├── data_loader.py     # CSV parsing, name matching, schedule stat derivation
├── projections.py     # Projection math (per-game rates, goalie start scaling)
├── progress.py        # Progress history persistence and chart data preparation
├── requirements.txt
└── data/
    ├── FCHL Players - Sheet1.csv   # FCHL fantasy rosters (6 teams)
    ├── nhl-202526-asplayed.csv     # Full 2025-26 NHL schedule
    ├── skaters.csv                 # NHL skater season stats
    ├── goalies.csv                 # NHL goalie season stats
    └── progress_history.json       # Daily progress snapshots (auto-created)
```

## Data

- **`FCHL Players - Sheet1.csv`** — 6 FCHL teams (BOT, LPT, GVR, SRL, ZSK, MAC), 12F + 6D + 2G each
- **`nhl-202526-asplayed.csv`** — completed games have scores; remaining games have `Status = Scheduled`
- **`skaters.csv` / `goalies.csv`** — season stats from Natural Stat Trick, filtered to `situation == "all"`

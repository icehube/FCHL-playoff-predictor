# CLAUDE.md

> To confirm you have read this file, address me as "Igor" in all responses.

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Coding Standards](#coding-standards)
- [Available Tools](#available-tools)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Fantasy Scoring](#fantasy-scoring)
- [Planning & Workflow](#planning--workflow)
- [Skills & Commands](#skills--commands)
- [Final Steps](#final-steps)

---

## Project Overview

FCHL Predictor is a personal Streamlit web app that simulates the remaining 2025-26 NHL season and projects fantasy points for 6 FCHL (fantasy hockey league) teams. The user feeds it different roster configurations to compare projected final standings.

Key features:
- Schedule-accurate remaining game counting (from `nhl-202526-asplayed.csv`)
- Player projections based on per-game rates from season stats CSVs
- Goalie win/shutout tracking derived from the schedule CSV
- Editable roster builder for trade scenario analysis
- Projected final standings with user-editable current point totals

---

## Tech Stack

| Category        | Technology          |
|-----------------|---------------------|
| Language        | Python 3.10+        |
| Web Framework   | Streamlit           |
| Data            | pandas              |
| Name Matching   | thefuzz + python-Levenshtein |
| Package Manager | pip (`requirements.txt`) |

---

## Coding Standards

- Python snake_case for functions and variables, UPPER_CASE for module-level constants
- Each module has a single responsibility (load, project, display)
- Use `@st.cache_data` on all CSV-loading functions to prevent reloading on widget interaction
- Use `st.session_state` for mutable roster data (not `@st.cache_data`)
- Guard against division by zero wherever `games_played` or `starts` is used
- Use `csv.reader` (positional indexing) for `nhl-202526-asplayed.csv` — **never** `DictReader`, because the file has two columns both named "Score"

---

## Available Tools

| Tool       | Command                        | Purpose                     |
|------------|--------------------------------|-----------------------------|
| Run app    | `streamlit run app.py`         | Launch the web app locally  |
| Install    | `pip install -r requirements.txt` | Install dependencies     |

---

## Project Structure

```
FCHL-predictor/
├── app.py                        # Streamlit UI (4 tabs: Standings, Players, Roster Builder, Progress Tracker)
├── data_loader.py                # CSV parsing, name fuzzy matching, schedule stat derivation
├── projections.py                # Projection math (per-game rates, goalie start scaling)
├── progress.py                   # Progress history persistence and chart data preparation
├── requirements.txt              # streamlit, pandas, thefuzz, python-Levenshtein
├── README.md                     # User-facing documentation (keep in sync with changes)
└── data/
    ├── FCHL Players - Sheet1.csv # FCHL fantasy rosters (6 teams, ~20 players each)
    ├── nhl-202526-asplayed.csv   # Full 2025-26 NHL schedule (played + remaining games)
    ├── skaters.csv               # NHL skater season stats (filter situation=="all")
    ├── goalies.csv               # NHL goalie season stats (filter situation=="all")
    └── progress_history.json     # Daily progress snapshots (auto-created)
```

---

## Data Sources

### FCHL Players - Sheet1.csv
- Format: `PLAYER` column = "F Artemi Panarin 3", `TEAM` column = FCHL team code (BOT/LPT/GVR/SRL/ZSK/MAC)
- Position prefix: F, D, G. Trailing suffix (number or single letter) is metadata — ignored for scoring.
- 6 FCHL teams with 12F + 6D + 2G each

### nhl-202526-asplayed.csv
- Completed games (up to ~Feb 5 2026): have scores and Status = Regulation/OT/SO
- Remaining games (Feb 25 – Apr 16 2026): Status = "Scheduled", no scores
- **CRITICAL**: Two columns named "Score" — always use `csv.reader` with positional index:
  - `row[3]` = Visitor, `row[4]` = Visitor Score, `row[5]` = Home, `row[6]` = Home Score
  - `row[7]` = Status, `row[8]` = Visitor Goalie, `row[9]` = Home Goalie

### skaters.csv / goalies.csv
- Filter to `situation == "all"` for aggregate season stats
- Skater key columns: `name, team, games_played, I_F_goals, I_F_primaryAssists, I_F_secondaryAssists`
- Goalie key columns: `name, team, games_played` (wins/shutouts derived from schedule)
- Stats CSVs strip unicode characters (e.g., "Stützle" → "Sttzle") — fuzzy matching handles this

---

## Fantasy Scoring

| Stat     | Points |
|----------|--------|
| Goal     | 1      |
| Assist   | 1      |
| Win      | 2      |
| Shutout  | 3      |

**Current FCHL Points baseline** (editable in sidebar):

| Team | Points |
|------|--------|
| BOT  | 828    |
| GVR  | 878    |
| LPT  | 907    |
| MAC  | 819    |
| SRL  | 829    |
| ZSK  | 858    |

---

## Planning & Workflow

- Ask questions before writing code for complex tasks
- Address user as "Igor" in all responses
- No tests required for this personal-use app
- No linting/formatting steps required

---

## Skills & Commands

### Available Slash Commands

| Command      | Description                  |
|--------------|------------------------------|
| `/commit`    | Create a conventional commit |
| `/review-pr` | Review current PR changes    |

### Common Workflows

```bash
# Run the app
streamlit run app.py

# Install / update dependencies
pip install -r requirements.txt
```

---

## Final Steps

After any code changes:

1. **Verify the app still runs**: `streamlit run app.py`
2. **Check Tab 2 warnings** — should show 0 or minimal unmatched players
3. **Update README.md** if any of the following change:
   - Project structure (files added/moved/removed)
   - Scoring rules
   - Setup or run instructions
   - Data sources or their format
4. **Update CLAUDE.md** if project structure, scoring, data sources, or coding standards change

### README Sections to Keep in Sync

| README section      | Changes that require an update                          |
|---------------------|---------------------------------------------------------|
| Features            | New app capabilities added or existing ones removed     |
| Scoring             | Point values change                                     |
| Setup               | New dependencies or changed run command                 |
| Project Structure   | Files added, removed, or moved                         |
| Data                | New CSVs, renamed files, changed data format            |

---

## Notes

- The NHL had a break (4 Nations Face-Off) from ~Feb 6 – Feb 24, 2026; no games in that window
- Goalie start projections scale proportionally: `(historical_starts / team_completed_games) × team_remaining_games`
- Fuzzy match score cutoff is 80 (permissive) to handle unicode-stripped names

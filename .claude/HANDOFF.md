# Session Handoff — Read This First

**Path:** /Users/dylanjaynes/Augusta National Model/.claude/HANDOFF.md
**Previous coordination:** COORDINATION.md, PHASE4.md (long, historical — skim only)
**Date:** 2026-04-08

---

## What's been built
- `run_production.py` — single canonical pipeline (replaces 9 old scripts)
- `augusta_model/calibration.py` — Platt scaling, unified MC, debutant shrinkage
- `augusta_model/features/new_features.py` — 11 Augusta-profile-aware features
- 148K round-level SG rows from DataGolf golden endpoint (2015-2026)
- Historical closing odds from The Odds API (2021-2025 Masters)
- Streamlit app deployed at github.com/dylanjaynes/augusta-national-model

## Current model output (NOT GOOD ENOUGH)
Backtest Cal AUC: 0.626 avg. Market Spearman: 0.339, Model Spearman: 0.191.
Market is 1.8x better at ranking the winner than our model.

2026 predictions have issues:
- DeChambeau #1 at 12% (market has him #3 at 6.2%)
- Schauffele #12 at 1.7% (market has him #5 at 4.8%) — KNOWN BUG
- Patrick Reed #5 at 4.7% (market has him ~15th) — old Augusta history overweighted
- Elderly champions (Vijay Singh, Fred Couples) still showing as value plays

## UNFIXED BUGS — MUST ADDRESS

### 1. Date ordering still broken for 2025-2026 events
Session B fixed dates for most events but many 2025-2026 events still have placeholder
dates (all on same day like "2026-06-15"). This means the rolling feature builder
processes rounds in arbitrary order within those years. AFFECTS ALL PLAYERS but
especially those whose most recent form matters (like Schauffele).

**To verify:** Run: `python3 -c "import pandas as pd; ext=pd.read_parquet('data/processed/historical_rounds_extended.parquet'); print(ext[ext.season==2026].groupby('date').event_name.nunique().sort_values(ascending=False).head(5))"`
Multiple events on the same date = broken.

**To fix:** Need real dates for all 2025-2026 events. Session B used The Odds API
event list or DG event list to get dates for some events but not all.

### 2. Model overconfident vs market (especially at the top)
The model's win probabilities for its top picks are 1.7-2.4x higher than the market.
When backtested against closing odds:
- Model claims 13% avg probability on value bets, actual hit rate is 1.1%
- Market is 1.8x better at ranking winners (Spearman 0.339 vs 0.191)

Root cause: S2 binary classifier produces raw probabilities that are too extreme,
and the Platt scaling + MC pipeline doesn't sufficiently compress them.

### 3. Augusta history over-weighted vs current form
Players with strong Augusta records but poor current form (Reed, elderly champions)
rank too high. Players with elite current form but thin Augusta history (Schauffele,
Aberg) rank too low. The S2 regularization increase (Session C) helped but not enough.

### 4. SG scale mismatch between eras
The golden endpoint returns real per-round SG values. The pipeline divides by
rounds_played after aggregating to tournament level (to match original CSV scale).
This is a LOSSY workaround. Between-player std still differs: original 1.025 vs
golden 1.066. Close but not perfect.

## WHAT NEEDS TO HAPPEN (priority order)

### P0: Fix the date ordering — PARTIALLY DONE
Used DG schedule API (`get-schedule`) to fix dates for 2024-2026 (23,685 rows fixed).
2024-2026 now have 0 date collisions. BUT: 2015-2023 schedule API returns 400 (only
supports recent seasons), so those years still have some placeholder dates.
Also: some 2025 events still have "2025-06-15" where fuzzy name matching failed.

**Remaining issue:** Scheffler's model_score = 0.4488 (45th percentile!) despite being
world #1. The rolling feature builder is STILL computing wrong values, likely because
2015-2022 data wasn't rescrapped with round-level format and dates may be inconsistent
with 2023+ data during the rolling window computation.

### P1: Calibrate model against closing odds
The model should be calibrated so its win probabilities match observed hit rates
against the market. Use the historical_closing_odds.csv (2021-2025) as the
calibration target instead of just backtest outcomes.

### P2: Pick 3-5 outright winners
User wants the model to identify 3-5 outright winner bets for the 2026 Masters.
Winner trend filters (from user-provided images):
- 24/26 winners top 30 world ranking (dg_rank <= 30)
- 27/28 made cut the year before (augusta_made_cut_prev_year == 1)
- 22/27 played 3+ previous Masters (augusta_starts >= 3)
- 15/16 had top-8 in last 7 events
- 16/16 top 30 SG:T2G for the year
- 14/14 gained 18+ strokes T2G in last 4 events
Apply these as filters, then rank by model+market blend.

### P3: Fix Schauffele specifically
His model_score (S1) is 0.169 (17th percentile) despite being world #4.
His S2 raw is 0.579 vs Scheffler's 0.948. Something is fundamentally wrong
with his rolling feature computation — likely the date ordering issue.

## KEY FILES
- `run_production.py` — main pipeline, run with `python3 run_production.py`
- `augusta_model/calibration.py` — Platt scaling, MC, debutant adjustment
- `augusta_model/features/new_features.py` — new features including aging decay
- `data/processed/historical_rounds_extended.parquet` — 148K round-level rows
- `data/processed/historical_closing_odds.csv` — real closing odds 2021-2026
- `data/processed/predictions_2026.parquet` — current predictions (with market odds)
- `.env` — API keys (DATAGOLF_API_KEY, ODDS_API_KEY)

## API ENDPOINTS
- DataGolf golden endpoint: `PUT https://datagolf.com/get-historical-stat-file`
  Body: `{"base":"no","t_num":T,"year":Y}`. Requires session cookie.
- The Odds API: `GET https://api.the-odds-api.com/v4/sports/golf_masters_tournament_winner/odds`
  Historical: `/v4/historical/sports/golf_masters_tournament_winner/odds?date=YYYY-MM-DDT12:00:00Z`
  ~17,700 requests remaining.
- DG event list: `GET https://datagolf.com/get-event-list?tour=pga`
- DG field: `GET https://feeds.datagolf.com/field-updates?tour=pga&file_format=json&key={KEY}`

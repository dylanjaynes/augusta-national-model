# Session Coordination

**Path:** /Users/dylanjaynes/Augusta National Model/.claude/COORDINATION.md
Both sessions: READ this file before starting work. UPDATE it when claiming or completing tasks.

---

## Active Sessions

| Session | Branch | Working Directory |
|---------|--------|-------------------|
| A | claude/cool-brown | .claude/worktrees/cool-brown/ |
| B | main | /Users/dylanjaynes/Augusta National Model/ |

## Task Claims

Mark tasks here before starting to avoid conflicts. Format: `- [ ] Task — claimed by [A/B]`

- [x] Data scraping, model training, backtest, predictions, Streamlit — **Session B (done)**
- [ ] **Probability calibration + debutant handling — Session A (in progress)**
  - Diagnose why S2 top-10 probs compressed to 55-71% band
  - Diagnose why win% is compressed AND noisy for debutants
  - Fix calibration (Platt scaling, isotonic regression, temperature tuning)
  - Add debutant discount so unknown players don't get phantom signal
  - Working on branch: claude/cool-brown
- [x] **LIV SG category breakdown imputation — Session B (DONE)**
  - LIV rows: sg_ott/app/arg/putt imputed using scaled PGA Tour means
  - Method: sg_cat = pga_mean_cat × (liv_total / pga_mean_total) using last 20 PGA events
  - Rahm: ott=0.503 app=1.021 for total=2.016 (approach-heavy, matches real profile)
  - DeChambeau: ott=1.692 for total=2.349 (driving-dominant, matches real profile)
  - Also cleaned 3,205 duplicate 2024 rows (past-results overlap with golden endpoint)
  - Dataset: 42,122 rows, 100% sg_ott coverage across all seasons

## Completed

- [x] **Session B** — All data scraping, model training, backtesting, prediction generation complete
- [x] Real SG breakdown scraped via `PUT /get-historical-stat-file` endpoint (the golden endpoint)
- [x] 45,327 training rows, 2015-2026, 100% sg_ott/app/arg/putt coverage
- [x] Backtest: Blend AUC 0.650, T10 Precision 44%, profitable 4/5 years
- [x] 2026 predictions generated and saved
- [x] Streamlit app built (4 pages), pushed to github.com/dylanjaynes/augusta-national-model

## Notes

### Session B → Session A: Critical context

**THE DATA BUG WE FOUND AND FIXED:**
2023/2024 data was originally scraped from `/past-results/` pages which only have `sg_total` (no ott/app/arg/putt breakdown). This meant 12 of 16 rolling SG features were zeroed out for 67/81 players. Aberg had 0% win probability because the model thought he had zero driving/approach/putting ability.

**THE FIX:** Discovered `PUT https://datagolf.com/get-historical-stat-file` endpoint. Payload: `{"base":"no","t_num":T,"year":Y}` where `t_num` is the internal DG event number (1-54 for PGA Tour). Returns full JSON with leaderboard + stats including sg_ott/app/arg/putt per player. Requires session cookie auth. This is the ONLY way to get historical SG breakdown — the stats page HTML always loads the latest year regardless of URL params.

**CURRENT STATE OF DATA:**
- `data/processed/historical_rounds_extended.parquet`: 45,327 rows, 2015-2026
- 100% sg_ott coverage for PGA Tour events
- LIV events (2024-2026): only have sg_total from `/past-results/liv-golf/` pages (no breakdown available via stats endpoint — LIV events aren't on the historical-tournament-stats page)
- PGA t_nums: [2,3,4,5,6,7,9,10,11,12,13,14,16,19,20,21,23,26,27,28,30,32,33,34,41,47,54]

**SCRAPER ENDPOINTS:**
1. `PUT /get-historical-stat-file` — full SG breakdown for any PGA event+year. Auth: session cookie.
2. `GET /get-event-list?tour=pga` — all PGA event IDs by year (no auth needed)
3. `GET /get-event-list?tour=alt` — all LIV event IDs by year
4. `/past-results/pga-tour/{t_num}/{year}` — leaderboard + per-round sg_total only
5. `/past-results/liv-golf/{tournament_num}/{year}` — LIV leaderboard + per-round sg_total only

**KEY FILES:**
- `scrape_real_sg.py` — uses the golden endpoint for 2023-2024 PGA
- `scrape_all_missing.py` — uses past-results for LIV
- `run_final_v7.py` — full retrain + backtest + 2026 predictions
- `run_field_strength.py` — field-strength weighted rolling features

**BACKTEST V7 (real SG data):**
| Year | Blend AUC | T10 Prec | Spearman |
|------|----------|---------|----------|
| 2021 | 0.514 | 30% | 0.204 |
| 2022 | 0.645 | 50% | 0.006 |
| 2023 | 0.719 | 40% | -0.048 |
| 2024 | 0.729 | 50% | 0.224 |
| 2025 | 0.645 | 50% | 0.020 |
| AVG  | 0.650 | 44% | 0.081 |

**2026 TOP 5:**
1. Scottie Scheffler — 3.8% win, 71.0% T10
2. Sungjae Im — 4.2% win, 68.4% T10
3. Xander Schauffele — 2.1% win, 66.6% T10
4. Ludvig Aberg — 0.4% win, 66.1% T10
5. Corey Conners — 2.3% win, 63.6% T10

**KNOWN REMAINING ISSUES:**
- LIV events only have sg_total (no category breakdown) — the `/get-historical-stat-file` endpoint doesn't serve LIV events
- Event names from the golden endpoint come as "t_num_X" instead of real names (course_info.event_name is empty in the response) — cosmetic issue
- S2 top-10% still compressed (55-71% range for top players) — rankings are reliable, absolute probs are not
- Win% noisy for debutants (Marco Penge 4.7%, Johnny Keefer 1.0%) — no Augusta features to discount

# Augusta National Model

## Project purpose
Predict Masters Tournament outcomes. Outputs: win/top 5/10/20 probabilities, H2H matchup edges, DFS lineups. Benchmarked against closing odds lines.

## FIRST STEP 
check /Users/dylanjaynes/Augusta National Model/.claude/COORDINATION.md to communicate with other claude sessions, sign off with your chosen username, try to communicate as effectively as possible for maximum efficiency
## Stack
- Python 3.11+
- XGBoost (primary model)
- Monte Carlo simulation (10k runs)
- Streamlit (frontend)
- Data Golf API (historical SG data)
- The Odds API (closing lines benchmark)
- Open-Meteo (historical weather, free)

## Project structure
augusta_model/        # core Python package
  data/               # ingestion + cleaning scripts
  features/           # feature engineering
  model/              # XGBoost training + evaluation
  simulation/         # Monte Carlo engine
  output/             # odds conversion, H2H, DFS
data/
  raw/                # never committed (in .gitignore)
  processed/          # clean parquet files, committed
notebooks/            # EDA only, not production code
streamlit_app/        # Streamlit pages
tests/

## Key conventions
- All API keys via .env / environment variables, never hardcoded
- Data Golf base URL: https://feeds.datagolf.com
- Processed data stored as parquet via pandas
- All feature engineering must be reproducible from raw data
- Monte Carlo simulation seeded for reproducibility (seed=42)

## API keys needed (set in .env)
DATAGOLF_API_KEY=91328c2c8e115c9e9b13461a8c3f
ODDS_API_KEY=78fcb125d1413e1e832f024e1335ff3a

## Existing golf_model context
The broader golf model lives at /Users/dylanjaynes/golf_model.
Key patterns to carry over:

### API
- DataGolf base URL: https://feeds.datagolf.com
- dg_get() helper pattern: params always include key + file_format=json
- Endpoints in use: historical-raw-data/rounds, field-updates, 
  preds/skill-ratings, preds/pre-tournament, betting-tools/outrights
- SubscriptionError class pattern for graceful 403 handling
- fetch_historical_rounds() uses event_id=all per year — reuse this exactly

### Data schema
historical_rounds.csv columns:
  dg_id, player_name, season, event_name, course, date,
  field_size, finish_pos, sg_ott, sg_app, sg_arg, sg_putt, sg_t2g, sg_total

### Feature engineering conventions
- Rolling windows: 3-event (w3) and 8-event (w8) exponential decay weights
- Features: sg_ott_3w, sg_app_3w, sg_arg_3w, sg_putt_3w, sg_t2g_3w,
  sg_total_3w, sg_ott_8w, sg_app_8w, sg_arg_8w, sg_putt_8w, sg_t2g_8w,
  sg_total_8w, sg_total_std_8, sg_total_momentum, sg_ball_strike_ratio,
  log_field_size
- Course fit applied as SG multipliers before feature construction
- finish_pct = (finish_num - 1) / (field_size - 1), clipped 0-1, target var

### Augusta-specific profile (starting point, to be verified by regression)
{"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}
NOTE: Only 2022 Masters data (78 rows, 78 players) exists in historical_rounds.csv.
Must pull 2014-2024 Masters history via API to build proper Augusta course weights.

### XGBoost config (carry over exactly)
n_estimators=600, learning_rate=0.02, max_depth=4,
subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
reg_alpha=0.1, reg_lambda=1.0, objective=reg:squarederror, random_state=42

### Monte Carlo config
N_SIMULATIONS=50_000, noise_std=0.16, TARGET_PRED_STD=0.10, seed=42
Feature normalization: z-score within field → rescale to training distribution

### Simulation output columns
model_win_prob, model_top5_prob, model_top10_prob, model_top20_prob,
model_make_cut, model_score

### Odds handling
American → decimal: +800 = 9.0, -110 = 1.909
_american_to_decimal() and odds_df_for_book() patterns from model.py

## V3 Backtest Results (2026-04-06, Session 3)

### Architectural decision: Augusta-only Stage 2
Stage 2 trains on masters_unified.parquet ONLY (no other majors, no tour-wide data).
Other majors (US Open, Open, PGA) have non-transferable course dynamics.
Augusta is uniquely itself. Small sample is handled by XGBoost's native NaN support.

### Two-stage model architecture
- Stage 1: XGBoost regression on finish_pct (16 rolling SG features, tour-wide training)
- Stage 2: XGBoost binary classifier for top-10 (34 features = rolling + Augusta-specific)
  - Trained on Augusta data only (271-624 rows depending on backtest year)
  - Config: n_estimators=300, lr=0.03, max_depth=3, reg_alpha=0.5, reg_lambda=2.0
  - scale_pos_weight = dynamic (neg/pos ratio per year)
- Optimal blend: 0.9 * Stage2 + 0.1 * MonteCarlo (optimized via AUC grid search)

### V3 Metrics
| Year | Brier(w) | Spearman | T10 Prec | S2 AUC10 | Blend AUC | MC AUC | ROI(t10) |
|------|---------|----------|---------|---------|----------|--------|---------|
| 2021 | 0.0134  | 0.227    | 30%     | 0.500   | 0.534    | 0.534  | 0%      |
| 2022 | 0.0141  | -0.009   | 50%     | 0.674   | 0.676    | 0.469  | +55%    |
| 2023 | 0.0145  | -0.015   | 40%     | 0.669   | 0.657    | 0.391  | +104%   |
| 2024 | 0.0134  | 0.177    | 60%     | 0.738   | 0.741    | 0.777  | +36%    |
| 2025 | 0.0144  | 0.173    | 30%     | 0.600   | 0.592    | 0.569  | +78%    |
| AVG  | 0.0140  | 0.111    | **42%** | **0.636**| **0.640**| 0.548  | **+55%**|

Key improvements over V2:
- Top-10 precision: 42% (was 26%) — +16pp
- S2 AUC: 0.636 (was 0.617) — trained on Augusta only
- Top-10 ROI: +55% avg (was +40%) — profitable in 4/5 years
- 2021 S2 AUC=0.500 because no top-10 finishers in pre-2021 training data (no SG era)

### 6 new experience features (Session 3)
1. augusta_competitive_rounds: total rounds played under tournament pressure (4 if cut, 2 if MC)
2. augusta_made_cut_prev_year: binary — did player make cut at Augusta last year? (9/10 recent winners = 1)
3. augusta_experience_tier: ordinal 0-4 (debutant/learning/established/veteran/deep_veteran)
4. augusta_scoring_trajectory: career_avg minus recent_2_avg score_vs_field (positive = improving)
5. augusta_rounds_last_2yrs: competitive rounds in Y-1 and Y-2 (max 8)
6. augusta_best_finish_recent: best finish in last 3 prior appearances

Plus: tour_vs_augusta_divergence = sg_total_8w percentile minus scoring_avg percentile
(positive = tour form overpredicts Augusta performance, catches JT-type players)

### JT diagnosis — FIXED
| Year | JT Rank (V2) | JT Rank (V3) | S2 Prob (V3) | Actual |
|------|-------------|-------------|-------------|--------|
| 2023 | #1          | **#5**      | 0.408       | CUT    |
| 2024 | #1          | **#18**     | 0.535       | CUT    |
| 2025 | #1          | **#28**     | 0.022       | T36    |

Root cause confirmed: tour_vs_augusta_divergence for JT = 0.16→0.43→0.51 (rising),
meaning his tour form increasingly overpredicts his Augusta performance.
His scoring_trajectory is -0.25 (declining). The Augusta-only Stage 2 model
picks up these signals because it was trained on Augusta data where they matter.

### Blend weight optimization results
| Weight | Mean AUC | T10 Prec@13 | Spearman |
|--------|---------|------------|----------|
| 0.50   | 0.615   | 30.8%      | 0.212    |
| 0.60   | 0.625   | 33.8%      | 0.229    |
| 0.70   | 0.631   | 35.4%      | 0.243    |
| 0.80   | 0.633   | 35.4%      | 0.249    |
| **0.90**| **0.640**| **40.0%** | **0.251**|
| 1.00   | 0.636   | 36.9%      | N/A      |

Optimal = 0.90 (Stage 2 dominates, small MC correction helps)

### Weather conditions (2021-2025)
| Year | Wind Avg | Wind Max | Rain  | Conditions |
|------|---------|---------|-------|-----------|
| 2021 | 14.8mph | 17.3mph | 8.1mm | wet       |
| 2022 | 15.9mph | 20.3mph | 4.1mm | windy     |
| 2023 | 14.2mph | 17.4mph | 58.6mm| wet       |
| 2024 | 16.3mph | 22.5mph | 35.0mm| wet       |
| 2025 | 9.8mph  | 13.3mph | 4.6mm | moderate  |

### Data sources ingested
- Local Masters round CSVs (2021-2025): 1,792 round rows, 448 tournament rows
- Wikipedia Masters leaderboards (2004-2020): 1,397 rows, all 17 years
- golf_model/historical_rounds.csv: 29K tour-wide rows (2015-2022)
- Open-Meteo weather: 5 tournament weeks (2021-2025)

### Event Tier Weighting (tested, rejected)
Tested event-tier weights on rolling SG features: Masters=3.0x, Players/WGC=2.0x,
Majors=1.8x, Elevated=1.4x, Standard=1.0x, Weak-field=0.5x, LIV=0.4x.
Backtest result: AVG AUC dropped from 0.636 to 0.624 (-0.012).
Root cause: training data is PGA Tour only (2015-2022), no LIV events to downweight.
Tier weighting adds noise to exponential decay without enough signal.
Decision: keep unweighted rolling features. Tier module available at
augusta_model/features/event_tiers.py if LIV data is added later.

### MC Spread Fix (Session 5)
noise_std: 0.16 → 0.10, target_pred_std: 0.10 → 0.14
Result: Scheffler win% 5.5% → 10.8%, #1/median ratio 8.6x → 152x
Temperature scaling T=2.5 for S2 calibration (optimal Brier on backtest)
DraftKings odds fetched live via The Odds API for edge calculation

### Data gaps remaining
- No SG data for 2019-2020 Masters
- No real historical closing odds for top-10/top-20 markets
- 2021 S2 has no prior top-10 examples (first SG year) — cold start problem
- DG approach-skill/par-5 SG requires paid tier
- Training data is PGA Tour 2015-2022 only — no LIV events, limits event-tier weighting
- S2 top-10% still inflated for longshots (Vijay Singh at 31% etc.)

## Briefing protocol
After every pipeline run or model change, update output_briefing_vX.md
with full results. Dylan will share this with Claude at claude.ai
for analysis and next-step planning. Claude at claude.ai will write
complete Python scripts (not prompts) for the next fix — drop them
directly into the project folder and run them.
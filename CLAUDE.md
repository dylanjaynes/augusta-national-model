# Augusta National Model

## Project purpose
Predict Masters Tournament outcomes. Outputs: win/top 5/10/20 probabilities, H2H matchup edges, DFS lineups. Benchmarked against closing odds lines.

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

## V2 Backtest Results (2026-04-06, Session 2)

### Two-stage model architecture
- Stage 1: XGBoost regression on finish_pct (16 rolling SG features, tour-wide training)
- Stage 2: XGBoost binary classifier for top-10 (29 features = rolling + Augusta-specific)
- Blended output: 0.6 * Stage2 + 0.4 * MonteCarlo for top-10/top-20 probs
- Stage 2 config: same hyperparams, objective=binary:logistic, scale_pos_weight=8.0

### V2 Metrics (with Stage 2 model)
| Year | Brier(w) | Spearman | T10 Prec | S2 AUC10 | Blend AUC10 | MC AUC10 |
|------|---------|----------|---------|---------|------------|---------|
| 2021 | 0.0134  | 0.227    | 40%     | 0.649   | 0.639      | 0.534   |
| 2022 | 0.0141  | -0.009   | 20%     | 0.604   | 0.520      | 0.469   |
| 2023 | 0.0145  | -0.015   | 10%     | 0.433   | 0.424      | 0.391   |
| 2024 | 0.0134  | 0.177    | 50%     | 0.803   | 0.800      | 0.777   |
| 2025 | 0.0144  | 0.173    | 10%     | 0.597   | 0.608      | 0.569   |
| AVG  | 0.0140  | 0.111    | 26%     | 0.617   | 0.598      | 0.548   |

Key finding: Stage 2 AUC (0.617) consistently beats MC-only AUC (0.548).
2024 was the standout year (AUC=0.803).

### Top-10/20 betting edge by market (avg 5-year ROI)
| Market | Avg Bets/yr | Avg Hit Rate | Avg ROI |
|--------|------------|-------------|---------|
| Top-5  | 25         | 7%          | +1.0%   |
| Top-10 | 33         | 20%         | +40.0%  |
| Top-20 | 38         | 31%         | +11.9%  |

Top-10 market shows strongest persistent edge (+40% avg ROI across all 5 years).
Top-10 was profitable in every backtest year (2021-2025).

### H2H matchup accuracy
53.9% (3,146/5,839 correct) vs 50% baseline.

### New features added (Session 2)
Scoring pattern features (from round-level data):
- augusta_birdie_rate: estimated birdies/hole from round scores
- augusta_bogey_avoidance: rate of clean (at/under par) rounds
- augusta_round_variance_score: std dev of round scores
- augusta_back9_scoring: R3+R4 performance vs field median

Weather features (Open-Meteo):
- tournament_wind_avg: avg daily max wind across tournament week
- conditions_bucket: calm/moderate/windy/wet

### Weather conditions (2021-2025)
| Year | Wind Avg | Wind Max | Rain  | Conditions |
|------|---------|---------|-------|-----------|
| 2021 | 14.8mph | 17.3mph | 8.1mm | wet       |
| 2022 | 15.9mph | 20.3mph | 4.1mm | windy     |
| 2023 | 14.2mph | 17.4mph | 58.6mm| wet       |
| 2024 | 16.3mph | 22.5mph | 35.0mm| wet       |
| 2025 | 9.8mph  | 13.3mph | 4.6mm | moderate  |

### SHAP / Feature importance
Top 5 by SHAP: sg_total_3w, sg_total_momentum, log_field_size, sg_ott_8w, sg_total_std_8
Augusta-specific features not in top importance — model still driven by
recent tour form. Augusta features have signal (AUC improvement) but low
individual SHAP magnitude because training data is mostly non-Masters events.

### JT diagnosis
Justin Thomas is over-ranked (top 1-2 prediction in 2023-2025, actual: CUT/CUT/T36).
Root cause: sg_total_3w and sg_total_momentum are his dominant SHAP contributors.
His tour form is genuinely elite (sg_total_8w ~0.59) but it doesn't translate
at Augusta. His augusta_round_variance_score is 3.5-4.7 (very high volatility)
and his augusta_scoring_avg turned positive by 2025 (+0.75, meaning he scores
worse than field median). The model over-weights tour form vs Augusta-specific
underperformance. Fix: increase Stage 2 blend weight or add a tour_vs_augusta
divergence penalty feature.

### Data sources ingested
- Local Masters round CSVs (2021-2025): 1,792 round rows, 448 tournament rows
- Wikipedia Masters leaderboards (2004-2020): 1,397 rows, all 17 years
- golf_model/historical_rounds.csv: 29K tour-wide rows (2015-2022)
- Open-Meteo weather: 5 tournament weeks (2021-2025)
- DG skill-ratings: available (sg breakdown per player), no shot-shape data
- DG field-updates: available (field list + ranks)
- DG approach-skill: 403 (paid tier)
- PGA Tour proximity stats: requires JS rendering (not scraped)

### Data gaps remaining
- No SG data for 2019-2020 Masters
- PGA Tour proximity data requires Selenium/Playwright
- DG approach-skill/par-5 specific SG requires paid tier
- No real historical closing odds for top-10/top-20 markets
- Augusta features dominated by tour form in SHAP — need more Augusta-specific training data
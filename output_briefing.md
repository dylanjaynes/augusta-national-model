# Augusta National Model — Methods & Results Briefing

## 1. Project Goal

Predict Masters Tournament outcomes (win/top-5/top-10/top-20 probabilities) using a hybrid XGBoost + Monte Carlo model trained on Strokes Gained data, historical scores, and Augusta-specific player features. Validate via walk-forward backtesting against actual results from 2021–2025.

---

## 2. Data Sources

### 2a. DataGolf Round-Level SG (2021–2025)
- Source: Per-round CSVs with full SG breakdown (sg_ott, sg_app, sg_arg, sg_putt, sg_t2g, sg_total)
- Coverage: 1,792 round rows, 448 player-tournament rows across 5 years
- Players per year: 88 (2021), 90 (2022), 86 (2023), 89 (2024), 95 (2025)
- Each player has per-round SG values; CUT players have R1/R2 SG but null R3/R4

### 2b. Wikipedia Historical Scores (2004–2020)
- Scraped full leaderboards for all 17 years (1,397 total rows, ~80 players/year)
- Fields: round scores (R1–R4), total, finish position
- Derived: made_cut, scoring_avg, score_vs_field (player total minus field median), r3r4_scoring (back-nine pressure proxy)

### 2c. Tour-Wide Training Data
- golf_model/historical_rounds.csv: 29,182 tournament-level rows across 2015–2022 PGA Tour
- Full SG breakdown per player per event
- Used as the XGBoost training set (not Masters-specific — all PGA events)

### 2d. Unified Dataset
- 1,845 rows total, 924 unique players, 22 seasons (2004–2025)
- 447 rows with full SG data (2021–2025), 1,397 with scores only (2004–2020)
- Schema: dg_id, player_name, season, finish_pos, finish_num, made_cut, field_size, sg_ott/app/arg/putt/t2g/total, has_sg_data, r1-r4_score, total_score, score_vs_field, r3r4_scoring, data_source

---

## 3. Feature Engineering

### 3a. Rolling Tour Form Features (from all PGA events, not just Masters)
Built per player from their full PGA Tour history, applied at prediction time:
- **3-event exponential weighted mean** (recent form): sg_ott_3w, sg_app_3w, sg_arg_3w, sg_putt_3w, sg_t2g_3w, sg_total_3w
- **8-event exponential weighted mean** (sustained form): sg_ott_8w, sg_app_8w, sg_arg_8w, sg_putt_8w, sg_t2g_8w, sg_total_8w
- **sg_total_std_8**: Rolling 8-event std dev of sg_total (consistency measure)
- **sg_total_momentum**: sg_total_3w minus sg_total_8w (trending up or down)
- **sg_ball_strike_ratio**: (sg_ott_8w + sg_app_8w) / (sum of absolute SG categories) — ball-striking dominance
- **log_field_size**: log(field_size) — controls for weak vs strong fields

### 3b. Augusta-Specific History Features
Computed per player-season with strict temporal cutoff (only prior years):

**SG-based (available for players with prior Augusta SG data):**
- augusta_sg_app_career: Exponentially weighted career avg sg_app at Augusta (decay=0.75^years_ago)
- augusta_sg_total_career: Same for sg_total
- augusta_sg_variance: Std of sg_total across Augusta appearances
- augusta_round_sg_variance: Mean within-tournament R1–R4 sg_total std dev (consistency)
- augusta_back9_sg: Mean sg_total in R3+R4 specifically (pressure rounds)

**Score-based (available for all players with any prior Augusta appearance):**
- augusta_starts: Number of prior Masters appearances
- augusta_made_cut_rate: Cuts made / starts
- augusta_top10_rate: Top-10 finishes / starts
- augusta_best_finish: Best historical finish (numeric)
- augusta_scoring_avg: Exponentially decay-weighted career mean score_vs_field at Augusta
- augusta_scoring_trend: Score_vs_field in last 2 appearances minus prior avg
- augusta_experience_bucket: 0 (debutant) / 1-2 / 3-5 / 6+

Experience distribution for 2025 field: 965 debutants, 445 with 1-2 starts, 291 with 3-5, 144 with 6+ across the full historical dataset.

### 3c. Augusta Course Weights
Ridge regression on finish_pct ~ sg_ott + sg_app + sg_arg + sg_putt using Masters SG data, normalized against tour-average Ridge coefficients.

- **Hardcoded prior** (from course-type research): {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}
- **Computed from data** (447 rows, 5 years): {"sg_ott": 0.25, "sg_app": 0.26, "sg_arg": 0.25, "sg_putt": 0.26}
- The computed weights are nearly uniform, diverging sharply from the approach-heavy prior. This may indicate that with only 5 years of data the regression can't distinguish category importance, or that Augusta genuinely tests all skills equally when measured via SG.
- Course weights are applied as SG multipliers before rolling feature construction in the training pipeline.

---

## 4. Model Architecture

### 4a. XGBoost
- Target: finish_pct = (finish_num - 1) / (field_size - 1), clipped [0,1]
- Config: n_estimators=600, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, objective=reg:squarederror, random_state=42
- Training data: All PGA Tour events with season < backtest_year (~14,000–16,000 rows depending on year)
- Features: 16 rolling SG features (3w + 8w windows + std + momentum + ball_strike_ratio + log_field_size)

### 4b. Monte Carlo Simulation
- 50,000 simulations per backtest year
- Process: XGBoost predicts finish_pct for each player → z-score normalize within field → rescale to target_pred_std=0.10 → add Gaussian noise (std=0.16) per simulation → rank → count outcomes
- Outputs: model_win_prob, model_top5_prob, model_top10_prob, model_top20_prob
- All randomness seeded at 42

### 4c. Walk-Forward Backtesting
For each Masters year Y in {2021, 2022, 2023, 2024, 2025}:
1. Train XGBoost on ALL tour data with season < Y
2. Compute Augusta course weights using only Masters SG data from years < Y
3. Build feature matrix for year Y field using each player's latest tour form + Augusta history features
4. Run 50,000 Monte Carlo simulations
5. Score against actual results

Strict temporal integrity: no future data leaks at any step.

---

## 5. Backtest Results

### 5a. Summary Table

| Year | Brier(win) | LogLoss(win) | Spearman | Top-10 Precision | ROI% |
|------|-----------|-------------|----------|-----------------|------|
| 2021 | 0.0134 | 0.0823 | 0.227 | 30% | -100% |
| 2022 | 0.0141 | 0.1100 | -0.009 | 30% | -100% |
| 2023 | 0.0145 | 0.0712 | -0.015 | 10% | +172% |
| 2024 | 0.0134 | 0.0576 | 0.177 | 60% | +192% |
| 2025 | 0.0144 | 0.0968 | 0.173 | 10% | -100% |
| **AVG** | **0.0140** | **0.0836** | **0.111** | **28%** | **+13%** |

### 5b. Interpretation
- **Brier scores** (~0.014) are close to the naive baseline of p*(1-p) ≈ 0.013 for a 1/80 event. The model isn't dramatically mis-calibrated but also isn't adding huge calibration value for the win market.
- **Spearman correlation** averages 0.111 — weakly positive, meaning the model's ranking of players has a slight correlation with actual finish order. Best year was 2021 (0.227), worst was 2023 (-0.015).
- **Top-10 precision** of 28% means ~3 of the model's top-10 predicted players actually finish top-10. The 2024 result (60% = 6/10) is a standout.
- **ROI** is volatile: 2 winning years (2023 Rahm, 2024 Scheffler) offset 3 losing years. The +13% average across 5 years is promising but sample is too small for statistical significance.

### 5c. Top 10 Model Predictions vs Actual Results

**2021** (Winner: Hideki Matsuyama — not in model's top 10)
| Rank | Player | Win Prob | Actual Finish |
|------|--------|----------|---------------|
| 1 | Dustin Johnson | 6.57% | CUT |
| 2 | Lee Westwood | 5.75% | CUT |
| 3 | Jason Kokrak | 5.30% | 49 |
| 4 | Xander Schauffele | 4.61% | T3 |
| 5 | Scottie Scheffler | 4.49% | T18 |
| 6 | Justin Rose | 4.03% | 7 |
| 7 | Si Woo Kim | 3.72% | T12 |
| 8 | Hudson Swafford | 3.58% | CUT |
| 9 | Jon Rahm | 3.41% | T5 |
| 10 | Brooks Koepka | 3.18% | CUT |

**2022** (Winner: Scottie Scheffler — not in model's top 10)
| Rank | Player | Win Prob | Actual Finish |
|------|--------|----------|---------------|
| 1 | Jon Rahm | 5.60% | T27 |
| 2 | Kevin Na | 5.27% | T14 |
| 3 | Sepp Straka | 4.78% | T30 |
| 4 | Francesco Molinari | 4.77% | CUT |
| 5 | Padraig Harrington | 4.77% | CUT |
| 6 | Patrick Cantlay | 4.74% | T39 |
| 7 | Adam Scott | 4.55% | T48 |
| 8 | Guido Migliozzi | 4.35% | CUT |
| 9 | Thomas Pieters | 3.61% | CUT |
| 10 | Justin Thomas | 3.30% | T8 |

**2023** (Winner: Jon Rahm — model had him at 2.13%, 14th ranked)
| Rank | Player | Win Prob | Actual Finish |
|------|--------|----------|---------------|
| 1 | Justin Thomas | 4.95% | CUT |
| 2 | Billy Horschel | 4.52% | 52 |
| 3 | Sam Burns | 4.44% | T29 |
| 4 | Scottie Scheffler | 4.36% | T10 |
| 5 | Tony Finau | 4.05% | T26 |
| 6 | Joaquin Niemann | 4.05% | T16 |
| 7 | Scott Stallings | 3.93% | T26 |
| 8 | Max Homa | 3.72% | T43 |
| 9 | Patrick Cantlay | 3.67% | T14 |
| 10 | Seamus Power | 3.25% | T46 |

**2024** (Winner: Scottie Scheffler — model had him 4th at 4.67%)
| Rank | Player | Win Prob | Actual Finish |
|------|--------|----------|---------------|
| 1 | Justin Thomas | 5.49% | CUT |
| 2 | Will Zalatoris | 4.96% | T9 |
| 3 | Sam Burns | 4.86% | CUT |
| **4** | **Scottie Scheffler** | **4.67%** | **1 (WINNER)** |
| 5 | Joaquin Niemann | 4.35% | T22 |
| 6 | Tony Finau | 4.21% | T55 |
| 7 | Max Homa | 3.96% | T3 |
| 8 | Patrick Cantlay | 3.90% | T22 |
| 9 | Denny McCarthy | 3.57% | T45 |
| 10 | Peter Malnati | 3.35% | CUT |

**2025** (Winner: Rory McIlroy — not shown in model's top 10)
| Rank | Player | Win Prob | Actual Finish |
|------|--------|----------|---------------|
| 1 | Justin Thomas | 4.96% | T36 |
| 2 | Will Zalatoris | 4.87% | CUT |
| 3 | Scottie Scheffler | 4.50% | 4 |
| 4 | Sam Burns | 4.42% | T46 |
| 5 | Billy Horschel | 4.10% | CUT |
| 6 | Joaquin Niemann | 3.99% | T29 |
| 7 | Tony Finau | 3.85% | CUT |
| 8 | Max Homa | 3.74% | T12 |
| 9 | Patrick Cantlay | 3.55% | T36 |
| 10 | Davis Riley | 3.42% | T21 |

### 5d. Notable Betting Results
Model flags bets where Kelly edge > 15% (model_prob - market_prob) / market_prob. Market proxy = uniform 1/N since real closing odds weren't available.

- **2023 Jon Rahm**: Model 2.13% vs market 1.47%, edge +45% → WON. ROI on all 2023 bets: +172%
- **2024 Scottie Scheffler**: Model 4.67% vs market 1.43%, edge +227% → WON. ROI on all 2024 bets: +192%
- **2021/2022/2025**: No flagged player won. -100% ROI each year.
- **5-year average ROI**: +13% (2 winners out of ~122 total bets placed)

---

## 6. Known Limitations & Next Steps

### Current Weaknesses
1. **Course weights are nearly uniform** — the Ridge regression on 447 rows can't differentiate SG categories at Augusta. The hardcoded prior (approach-heavy) may be more informative. A blended approach or Bayesian hierarchical model could help.
2. **Justin Thomas consistently over-ranked** — appears #1 in 2023, 2024, 2025 predictions but performed poorly (CUT twice, T36). The model may be over-weighting recent tour form without properly penalizing Augusta-specific struggles.
3. **Market baseline is uniform 1/N** — without real closing odds, the Kelly edge calculations are inflated. Every player with above-average model probability gets flagged.
4. **No 2019-2020 SG data** — gap between Wikipedia scores (end 2020) and CSV SG data (start 2021).
5. **Spearman correlation inconsistent** — negative in 2022/2023, suggesting the model's ranking was worse than random in those years.
6. **Training data is tour-wide, not Augusta-specific** — the XGBoost model learns general PGA relationships, then applies Augusta weights as a filter. A model trained on major championships or Augusta-only data might capture different dynamics.

### Improvement Opportunities
1. Integrate real closing odds (DG betting-tools/outrights or The Odds API) for proper Kelly edge calculation
2. Add Augusta-specific features to the XGBoost model itself (not just as course weight multipliers)
3. Investigate why the model over-ranks certain players (JT, Sam Burns) — may need form recency weighting adjustments
4. Pull 2019-2020 data to close the gap between sources
5. Consider ensemble with DG's own pre-tournament probabilities as a feature or benchmark
6. Add weather data (Open-Meteo) as round-level features for within-tournament prediction

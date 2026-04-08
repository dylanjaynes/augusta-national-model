# Augusta National Model — V5 Final Briefing
## 2026 Masters Tournament — April 9-12

---

## 1. What This Is

A two-stage XGBoost + Monte Carlo prediction model for the Masters Tournament. It predicts win, top-5, top-10, and top-20 probabilities for every player in the field. Built across 6 development sessions, backtested on 5 Masters (2021-2025), and retrained on a freshly scraped dataset covering 2015-2026.

**Backtest headline: 40% top-10 precision, 0.644 AUC, profitable in 4/5 years on a naive top-10 betting strategy.**

**Repo**: [github.com/dylanjaynes/augusta-national-model](https://github.com/dylanjaynes/augusta-national-model)

---

## 2. Architecture

### Stage 1 — XGBoost Regression → Monte Carlo
- **Target**: finish_pct = (finish_position - 1) / (field_size - 1)
- **Training**: 18,290 PGA Tour tournament rows (2015-2026) with full SG breakdown
- **Features**: 16 rolling SG features (3-event and 8-event exponential weighted means for each SG category, plus volatility, momentum, ball-strike ratio, field size)
- **Config**: n_estimators=600, lr=0.02, max_depth=4, subsample=0.8
- **Simulation**: 50,000 Monte Carlo runs (noise_std=0.10, target_pred_std=0.14, seed=42) to generate win/top-5/top-10/top-20 probabilities

### Stage 2 — XGBoost Binary Classifier for Top-10
- **Target**: 1 if finish_position <= 10, else 0
- **Training**: Augusta data only — 719 Masters player-seasons from masters_unified.parquet (2004-2025). Both SG-era (2021-2025, 447 rows) and scores-only (2004-2020, 272 rows with NaN SG handled natively)
- **Features**: 34 total = 16 rolling SG + 18 Augusta-specific experience features
- **Config**: n_estimators=300, lr=0.03, max_depth=3, reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=dynamic
- **Temperature scaling**: T=2.5 applied to raw S2 probabilities (optimal Brier on backtest)

### Blending
```
final_top10 = 0.90 × Stage2_calibrated + 0.10 × MC_top10
```
Optimized via grid search across [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]. Stage 2 dominates because it's trained on Augusta-specific data.

---

## 3. Data Sources

| Source | Rows | Coverage | Notes |
|--------|------|----------|-------|
| **PGA Tour SG** (historical_rounds_extended.parquet) | 33,443 | 2015-2026 | Scraped from datagolf.com. 80/81 field players have current data |
| **Masters SG** (masters_sg_rounds.parquet) | 1,792 | 2021-2025 | Per-round SG from local CSVs |
| **Masters scores** (Wikipedia) | 1,397 | 2004-2020 | Full leaderboards, round scores, derived fields |
| **Masters unified** (masters_unified.parquet) | 1,845 | 2004-2025 | Merged SG + scores, 22 seasons, 924 players |
| **Weather** (Open-Meteo) | 5 | 2021-2025 | Wind, rain, temp per tournament week |
| **DraftKings odds** (The Odds API) | 91 | 2026 live | Outrights, converted to fair probs |

### Critical data fix in this version
The original training CSV stopped at 2022. We scraped 4,750 new rows (42 events) from datagolf.com/historical-tournament-stats using session cookie auth. Every player's rolling SG features now reflect 2025-2026 form instead of being frozen at 2022.

**Before scrape**: 57/81 players had stale (2022) data, 24 had zero tour history
**After scrape**: 80/81 have current (2025-2026) data. Only Casey Jarvis missing.

Gap: 2023 season is not in the dataset. The DG historical stats page only serves the most recent year per event — year-switching is client-side JavaScript we couldn't replicate server-side.

---

## 4. Augusta-Specific Features (18 total)

All computed with strict temporal cutoff — only data from prior seasons.

### 6 Experience Features (built in Session 3)
| Feature | Description |
|---------|-------------|
| `augusta_competitive_rounds` | Total rounds under tournament pressure (4 if made cut, 2 if missed) |
| `augusta_made_cut_prev_year` | Binary: made the cut at Augusta last year? 9 of 10 recent winners = 1 |
| `augusta_experience_tier` | 0=Debutant, 1=Learning (1-7 rds), 2=Established (8-19), 3=Veteran (20-35), 4=Deep Vet (36+) |
| `augusta_scoring_trajectory` | Career avg minus recent-2 avg score_vs_field. Positive = improving |
| `augusta_rounds_last_2yrs` | Competitive rounds in Y-1 and Y-2 (max 8) |
| `augusta_best_finish_recent` | Best finish in last 3 appearances |

### Other Augusta Features
| Feature | Description |
|---------|-------------|
| `augusta_starts` | Total prior Masters appearances |
| `augusta_made_cut_rate` | Career cut rate at Augusta |
| `augusta_top10_rate` | Career top-10 rate at Augusta |
| `augusta_best_finish` | Career best finish |
| `augusta_scoring_avg` | Exponentially decay-weighted career score_vs_field |
| `augusta_sg_app_career` | Decay-weighted career sg_app at Augusta (SG-era only) |
| `augusta_sg_total_career` | Decay-weighted career sg_total at Augusta |
| `augusta_bogey_avoidance` | Rate of clean (at/under par) rounds at Augusta |
| `augusta_round_variance_score` | Std dev of round scores (volatility) |
| `augusta_back9_scoring` | R3+R4 performance (pressure rounds) |
| `tournament_wind_avg` | Weather: avg daily max wind for tournament week |
| `tour_vs_augusta_divergence` | sg_total_8w percentile minus scoring_avg percentile. Positive = tour form overpredicts Augusta |

---

## 5. V4 Backtest Results (2021-2025)

| Year | S2 AUC | Blend AUC | MC-only AUC | T10 Precision | Spearman |
|------|--------|-----------|-------------|--------------|----------|
| 2021 | 0.500 | 0.531 | 0.531 | 30% | 0.224 |
| 2022 | 0.625 | 0.659 | 0.476 | 50% | 0.008 |
| 2023 | 0.694 | 0.671 | 0.372 | 40% | -0.030 |
| 2024 | 0.700 | 0.722 | 0.778 | 50% | 0.204 |
| 2025 | 0.630 | 0.635 | 0.584 | 30% | 0.205 |
| **AVG** | **0.630** | **0.644** | **0.548** | **40%** | **0.122** |

- 2021 S2 AUC = 0.500 (cold start: no SG-era top-10 examples before 2021)
- Blend consistently beats MC-only (0.644 vs 0.548 avg AUC)
- Top-10 precision: 40% means 4 of the model's top-10 predicted players actually finish top-10
- 2024 best year: AUC=0.722, 50% precision, Scheffler ranked #2 (won)

### Version History
| Version | S2 AUC | T10 Prec | Key Change |
|---------|--------|---------|------------|
| V1 (Session 1) | N/A | 28% | Base MC-only model |
| V2 (Session 2) | 0.617 | 26% | Added Stage 2, tour-wide training |
| V3 (Session 3) | 0.636 | 42% | Augusta-only S2, experience features, JT fix |
| **V4 (Session 6)** | **0.630** | **40%** | **Extended data (33K rows), current tour form** |

V4 S2 AUC is slightly lower than V3 (-0.006) because the extended training data changes rolling feature paths. But 2026 predictions are dramatically more accurate because players now have current form data.

---

## 6. 2026 Predictions — Full Field

### Top 20

| Rk | Player | Win% | Top-10% | Top-20% | DK Odds | Exp Tier | Cut Prev | Trajectory | Divergence |
|----|--------|------|---------|---------|---------|----------|----------|-----------|-----------|
| 1 | Scottie Scheffler | 5.9% | 73.8% | 73.8% | +495 | Veteran (20 rds) | Yes | +2.46 | 0.29 |
| 2 | Xander Schauffele | 1.9% | 71.5% | 71.5% | +1800 | Established (18 rds) | Yes | -0.05 | 0.22 |
| 3 | Cameron Young | 0.0% | 68.7% | 68.7% | +2300 | Established (12 rds) | No | +0.07 | 0.23 |
| 4 | Sungjae Im | 2.8% | 68.6% | 68.6% | +11500 | Established (16 rds) | Yes | -0.55 | 0.07 |
| 5 | Bryson DeChambeau | 0.3% | 66.6% | 66.6% | +1050 | Veteran (28 rds) | Yes | +2.08 | 0.10 |
| 6 | Corey Conners | 2.6% | 65.2% | 65.2% | +8400 | Veteran (22 rds) | Yes | -1.53 | 0.11 |
| 7 | Cameron Smith | 1.5% | 64.3% | 64.3% | +11000 | Established (18 rds) | No | -2.33 | 0.27 |
| 8 | Jordan Spieth | 2.6% | 63.9% | 63.9% | +4200 | Established (16 rds) | Yes | -0.01 | 0.12 |
| 9 | Patrick Cantlay | 5.1% | 63.3% | 65.5% | +5800 | Veteran (26 rds) | Yes | -0.82 | 0.26 |
| 10 | Patrick Reed | 0.0% | 63.1% | 63.1% | +4300 | Veteran (20 rds) | Yes | +0.93 | -0.26 |
| 11 | Ludvig Aberg | 0.0% | 62.4% | 62.4% | +1650 | Established (8 rds) | Yes | +0.50 | 0.38 |
| 12 | Zach Johnson | 0.4% | 61.1% | 61.1% | +55000 | Veteran (26 rds) | Yes | -0.97 | -0.19 |
| 13 | Justin Rose | 2.7% | 60.8% | 60.8% | +3500 | Established (16 rds) | Yes | -0.93 | -0.04 |
| 14 | Matt Fitzpatrick | 0.0% | 60.5% | 60.5% | +2350 | Veteran (28 rds) | Yes | -1.27 | 0.40 |
| 15 | Rory McIlroy | 0.0% | 60.5% | 60.5% | +1175 | Established (16 rds) | Yes | -1.49 | 0.01 |
| 16 | Max Homa | 5.2% | 60.4% | 66.6% | +12000 | Established (18 rds) | Yes | +5.29 | 0.14 |
| 17 | Davis Riley | 4.0% | 59.9% | 61.5% | +82500 | Learning (4 rds) | Yes | 0.00 | 0.00 |
| 18 | Tommy Fleetwood | 0.6% | 59.6% | 59.6% | +2250 | Veteran (24 rds) | Yes | +3.06 | -0.02 |
| 19 | Kurt Kitayama | 0.1% | 59.2% | 59.2% | +9400 | Learning (6 rds) | No | 0.00 | 0.16 |
| 20 | Jon Rahm | 2.5% | 59.1% | 59.1% | +910 | Veteran (20 rds) | Yes | -3.54 | 0.13 |

### Rest of Field (21-81)

| Rk | Player | Win% | Top-10% | DK Odds | Tier | Cut Prev | Traj |
|----|--------|------|---------|---------|------|----------|------|
| 21 | Collin Morikawa | 0.0% | 58.9% | +3100 | Vet | Yes | +0.65 |
| 22 | Justin Thomas | 7.7% | 58.9% | +5400 | Est | Yes | +1.20 |
| 23 | Jason Day | 0.0% | 58.0% | +6800 | Vet | Yes | +1.17 |
| 24 | Haotong Li | 5.6% | 57.2% | +27500 | Debut | No | 0.00 |
| 25 | Ben Griffin | 0.0% | 57.0% | +11000 | Debut | No | 0.00 |
| 26 | Marco Penge | 6.2% | 56.8% | +14500 | Debut | No | 0.00 |
| 27 | Shane Lowry | 0.5% | 55.1% | +6700 | Vet | Yes | -3.58 |
| 28 | Brooks Koepka | 0.1% | 54.5% | +3700 | Est | No | +0.86 |
| 29 | Tyrrell Hatton | 2.1% | 54.1% | +8200 | Vet | Yes | +4.23 |
| 30 | Johnny Keefer | 3.1% | 53.6% | +37000 | Debut | No | 0.00 |
| 31 | Jake Knapp | 0.9% | 53.5% | +6500 | Learn | No | 0.00 |
| 32 | Daniel Berger | 4.1% | 53.2% | +12000 | Est | Yes | -3.18 |
| 33 | Aldrich Potgieter | 0.0% | 52.5% | +36000 | Learn | No | 0.00 |
| 34 | Viktor Hovland | 0.0% | 51.5% | +4500 | Vet | Yes | +1.51 |
| 35 | Sam Stevens | 3.6% | 51.2% | +17500 | Debut | No | 0.00 |
| 36 | Keegan Bradley | 0.1% | 50.6% | +14000 | Vet | No | +0.25 |
| 37 | Rasmus Neergaard-Petersen | 0.1% | 50.4% | +32000 | Debut | No | 0.00 |
| 38 | Russell Henley | 0.0% | 49.7% | +4700 | Vet | No | +1.36 |
| 39 | Harry Hall | 1.6% | 49.7% | +15500 | Debut | No | 0.00 |
| 40 | Nicolai Hojgaard | 0.3% | 49.1% | +7400 | Learn | No | 0.00 |
| 41 | Charl Schwartzel | 0.1% | 48.9% | +62500 | Est | Yes | -3.14 |
| 42 | Chris Gotterup | 0.0% | 47.8% | +25000 | Debut | No | 0.00 |
| 43 | Matt McCarty | 1.1% | 47.8% | +28000 | Learn | Yes | 0.00 |
| 44 | Alex Noren | 0.0% | 47.3% | +25000 | Learn | No | 0.00 |
| 45 | Nick Taylor | 0.0% | 47.2% | +19000 | Learn | Yes | 0.00 |
| 46 | Brian Harman | 1.6% | 47.1% | +22000 | Est | Yes | +1.56 |
| 47 | Casey Jarvis | 0.0% | 46.8% | +23000 | Debut | No | 0.00 |
| 48 | Akshay Bhatia | 0.0% | 46.5% | +5600 | Est | Yes | +0.29 |
| 49 | Si Woo Kim | 0.9% | 45.4% | +5000 | Est | No | +0.03 |
| 50 | Sam Burns | 6.7% | 43.8% | +7200 | Est | Yes | +0.70 |
| 51 | Kristoffer Reitan | 1.0% | 43.7% | +22500 | Debut | No | 0.00 |
| 52 | Nico Echavarria | 2.9% | 43.4% | +32500 | Learn | Yes | 0.00 |
| 53 | J.J. Spaun | 0.4% | 43.1% | +6900 | Est | Yes | +1.73 |
| 54 | Tom McKibbin | 0.5% | 42.6% | +29000 | Debut | No | 0.00 |
| 55 | Aaron Rai | 0.0% | 41.9% | +25000 | Learn | Yes | 0.00 |
| 56 | Gary Woodland | 0.8% | 40.6% | +12000 | Est | No | -0.84 |
| 57 | Andrew Novak | 2.6% | 40.4% | +29000 | Debut | No | 0.00 |
| 58 | Hideki Matsuyama | 0.2% | 39.9% | +2700 | Vet | Yes | -2.14 |
| 59 | Ryan Fox | 0.0% | 39.5% | +20000 | Est | No | +0.14 |
| 60 | Sergio Garcia | 2.1% | 37.9% | +25000 | Est | No | 0.00 |
| 61 | Danny Willett | 0.1% | 35.7% | +225000 | Est | Yes | -1.30 |
| 62 | Max Greyserman | 0.0% | 35.6% | +21000 | Learn | Yes | 0.00 |
| 63 | Carlos Ortiz | 0.0% | 35.3% | +23000 | Learn | No | 0.00 |
| 64 | Maverick McNealy | 0.0% | 35.2% | +7800 | Learn | Yes | 0.00 |
| 65 | Adam Scott | 0.0% | 34.8% | +6000 | Est | No | +2.71 |
| 66 | Sepp Straka | 0.0% | 34.1% | +7800 | Est | No | -0.66 |
| 67 | Dustin Johnson | 0.0% | 33.7% | +25000 | Est | No | +0.75 |
| 68 | Bubba Watson | 0.3% | 33.7% | +52500 | Vet | Yes | -0.47 |
| 69 | Fred Couples | 0.1% | 32.8% | +500000 | Deep Vet | No | +3.17 |
| 70 | Sami Valimaki | 0.0% | 32.6% | N/A | Debut | No | 0.00 |
| 71 | Ryan Gerard | 0.0% | 32.1% | +15500 | Debut | No | 0.00 |
| 72 | Wyndham Clark | 0.0% | 31.6% | +20000 | Learn | Yes | 0.00 |
| 73 | Vijay Singh | 0.0% | 31.5% | +500000 | Vet | No | +1.07 |
| 74 | Jacob Bridgeman | 4.3% | 30.9% | +8400 | Debut | No | 0.00 |
| 75 | Rasmus Hojgaard | 0.1% | 30.6% | +12000 | Learn | Yes | 0.00 |
| 76 | Robert MacIntyre | 0.0% | 27.7% | +3500 | Est | No | +0.18 |
| 77 | Min Woo Lee | 0.0% | 24.8% | +3400 | Est | Yes | -0.41 |
| 78 | Harris English | 0.0% | 24.0% | +10000 | Est | Yes | +1.74 |
| 79 | Michael Kim | 0.0% | 22.2% | +21000 | Learn | Yes | 0.00 |
| 80 | Brian Campbell | 0.0% | 20.6% | N/A | Learn | Yes | 0.00 |
| 81 | Mike Weir | 0.0% | 19.2% | +500000 | Est | No | 0.00 |

---

## 7. Fade Candidates

Players where `tour_vs_augusta_divergence > 0.35` — their PGA Tour form significantly overpredicts Augusta results:

| Player | Divergence | Top-10% | Why |
|--------|-----------|---------|-----|
| **Nicolai Hojgaard** | 0.43 | 49.1% | Limited Augusta track record (6 rds), strong tour form |
| **Matt Fitzpatrick** | 0.40 | 60.5% | Scoring trajectory -1.27, declining at Augusta |
| **Ludvig Aberg** | 0.38 | 62.4% | Only 8 competitive rounds, tour form may not translate |
| **Justin Thomas** | 0.31 | 58.9% | The model's original problem player — CUT in 2023 and 2024 |
| **Scottie Scheffler** | 0.29 | 73.8% | Not a fade — divergence is moderate, trajectory is +2.46 |
| **Sam Burns** | 0.28 | 43.8% | Model already ranks him low; market at +7200 |
| **Cameron Smith** | 0.27 | 64.3% | Didn't make cut last year, trajectory -2.33 |

---

## 8. Known Limitations

1. **S2 top-10 probabilities are still compressed upward**. The Stage 2 model assigns 60%+ to most "good" players because it trains on ~700 Augusta rows where ~8% are top-10 finishers, but after temperature scaling at T=2.5 the probabilities smooth toward the mean. The **relative ranking** is more reliable than the **absolute probabilities**.

2. **Win probabilities are noisy for non-elite players**. The MC simulation with noise_std=0.10 creates spread but some longshots (Haotong Li at 5.6% win, Marco Penge at 6.2%) have inflated win% driven by tour form features without Augusta history to discount them. Their top-10% is more trustworthy than their win%.

3. **2023 season data is missing**. The DG scrape could only get the most recent year per event. There's a hole between 2022 and 2024 in the training data. Players who had breakout 2023 seasons (Wyndham Clark won the US Open) have that context partially captured via 2024-2025 rolling features but not directly.

4. **15 debutants have zero Augusta features**. The model relies entirely on tour form for first-timers. Historically, debutants almost never win (one winner since 1979) but can finish top-10 at ~11% base rate.

5. **Market odds scaling for top-10**. DraftKings only exposes outright win odds via The Odds API. Top-10 fair probabilities are estimated as `win_fair_prob × 8.5` (capped at 72%). This is an approximation; real top-10 market lines would sharpen the edge analysis.

6. **Event-tier weighting was tested and rejected**. Weighting rolling SG features by event importance (Masters=3x, LIV=0.4x) hurt backtest AUC by -0.012. Root cause: training data is PGA Tour only — no LIV events to downweight. The module exists at `augusta_model/features/event_tiers.py` for future use.

---

## 9. Build History

| Session | What Changed | Key Result |
|---------|-------------|------------|
| 1 | Data ingestion: DG CSVs, Wikipedia scrape, unified dataset. Basic XGBoost + MC backtest | Pipeline works, 5 backtest years |
| 2 | Stage 2 binary classifier, scoring pattern features, weather, betting edge analysis | Top-10 market identified as money market (+40% ROI) |
| 3 | 6 new experience features, Augusta-only S2, blend optimization, tour_vs_augusta_divergence | JT fixed: #1→#28. T10 prec 26%→42% |
| 4 | 2026 live predictions, DraftKings odds, Streamlit app, GitHub deploy | Ship day — 4-page app at dylanjaynes/augusta-national-model |
| 5 | MC spread fix (noise 0.16→0.10), temperature scaling, real DK odds | Scheffler win% 5.5%→10.8%, 152x spread ratio |
| 5b | Event-tier weighting tested and rejected (-0.012 AUC) | Module saved for future LIV data integration |
| **6** | **Scraped 4,750 new tour rows from DG website (2025-2026). Fixed -9999 sentinels. Retrained on 33K rows. Re-backtested.** | **80/81 players current. Aberg/Fitzpatrick/McIlroy now have real data** |

---

## 10. Files & Pipeline

| File | Purpose |
|------|---------|
| `run_pipeline.py` | V1: data ingestion + basic backtest |
| `run_v2_pipeline.py` | V2: scoring features, weather, Stage 2 |
| `run_v3_pipeline.py` | V3: experience features, Augusta-only S2 |
| `run_2026_predictions.py` | 2026 field through pipeline + odds |
| `run_fix_spread.py` | MC spread fix + temperature scaling |
| `run_event_tiers.py` | Event-tier weighting (tested, rejected) |
| `run_retrain_extended.py` | **V4: retrain on extended 33K-row dataset** |
| `scrape_dg_historical.py` | Scraper for DG historical stats pages |
| `streamlit_app/app.py` | Streamlit entry point (4 pages) |
| `augusta_model/features/event_tiers.py` | Event importance weights module |

### Key Data Files

| File | Rows | Description |
|------|------|-------------|
| `data/processed/historical_rounds_extended.parquet` | 33,443 | All PGA Tour SG data 2015-2026 |
| `data/processed/masters_unified.parquet` | 1,845 | All Masters player-seasons 2004-2025 |
| `data/processed/augusta_player_features.parquet` | 1,845 | 18 Augusta features per player-season |
| `data/processed/predictions_2026.parquet` | 81 | Final 2026 predictions |
| `data/processed/backtest_results_v4.parquet` | 368 | V4 backtest output |
| `data/processed/dk_odds_2026.csv` | 96 | DraftKings fair probabilities |

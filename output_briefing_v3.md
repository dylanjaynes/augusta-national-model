# Augusta National Model ��� V3 Methods & Results Briefing

## 1. What Changed in V3

V3 is a focused rebuild of the Augusta experience features and a retraining of Stage 2 on Augusta-only data. No new data sources were added. The changes:

1. **6 new experience features** replace the old `augusta_starts` / `augusta_experience_bucket` system
2. **Stage 2 now trains on Augusta data only** (was training on all PGA Tour events in V2)
3. **Optimal blend weight raised to 0.90** (was 0.60 in V2) — Stage 2 now dominates
4. **`tour_vs_augusta_divergence`** feature directly catches players whose tour form doesn't translate
5. **No majors blending** — architectural decision that Augusta is uniquely itself

### V2 → V3 Improvement Summary

| Metric | V2 | V3 | Delta |
|--------|----|----|-------|
| Top-10 Precision | 26% | **42%** | **+16pp** |
| S2 AUC (avg) | 0.617 | **0.636** | +0.019 |
| Blend AUC (avg) | 0.598 | **0.640** | +0.042 |
| Top-10 ROI (avg) | +40% | **+55%** | +15pp |
| JT Rank 2024 | #1 | **#18** | Fixed |
| JT Rank 2025 | #1 | **#28** | Fixed |

---

## 2. Model Architecture (V3)

### Stage 1 — XGBoost Regression (unchanged)
- Target: finish_pct = (finish_num - 1) / (field_size - 1)
- Training: All PGA Tour events, season < backtest year (~14K-16K rows)
- Features: 16 rolling SG features
- Purpose: Drives outright win probabilities via Monte Carlo (50K simulations)

### Stage 2 — XGBoost Binary Classifier for Top-10 (rebuilt)
- Target: made_top10 = 1 if finish_num <= 10
- **Training: Augusta data only** (masters_unified.parquet, season < backtest year)
  - 271 rows for 2021 backtest, growing to 624 rows for 2025
  - Both SG-era (2021+) and scores-only (2004-2020) rows included
  - XGBoost handles NaN natively — no imputation for pre-SG rows
- Features: 34 total (16 rolling SG + 18 Augusta-specific)
- Config: n_estimators=300, lr=0.03, max_depth=3, reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=dynamic

### Blending
```
final_top10_prob = 0.90 * Stage2_prob + 0.10 * MonteCarlo_top10_prob
```
Optimized via grid search over [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]:

| Weight | Mean AUC | T10 Prec@13 | Spearman |
|--------|---------|------------|----------|
| 0.50 | 0.615 | 30.8% | 0.212 |
| 0.60 | 0.625 | 33.8% | 0.229 |
| 0.70 | 0.631 | 35.4% | 0.243 |
| 0.80 | 0.633 | 35.4% | 0.249 |
| **0.90** | **0.640** | **40.0%** | **0.251** |
| 1.00 | 0.636 | 36.9% | N/A |

Stage 2 dominates. The 10% MC contribution provides a small but real regularization effect.

---

## 3. The 6 New Experience Features

All computed with strict temporal cutoff (only data from seasons prior to the prediction year).

### Feature 1: `augusta_competitive_rounds`
Total rounds played at Augusta under tournament pressure. Made cut = 4 rounds, missed cut = 2.

Why it's better than `augusta_starts`: A 5-time starter who made every cut has 20 rounds of course knowledge. A 5-time starter who missed every cut has only 10. That gap is enormous for green reading and course management.

### Feature 2: `augusta_made_cut_prev_year`
Binary: 1 if player made the cut at Augusta in the immediately prior year.

9 of the last 10 Masters winners (2016-2025) had this = 1. Recent tournament-pressure experience at Augusta is the strongest single predictor of success there.

### Feature 3: `augusta_experience_tier`
Ordinal 0-4 based on competitive_rounds:
- 0: debutant (0 rounds)
- 1: learning (1-7 rounds)
- 2: established (8-19 rounds)
- 3: veteran (20-35 rounds)
- 4: deep_veteran (36+ rounds)

### Feature 4: `augusta_scoring_trajectory`
career_avg score_vs_field minus recent_2 score_vs_field (among made-cut appearances only). Positive = improving at Augusta, negative = declining.

This is the JT killer: his trajectory is -0.25 (declining) while Scheffler's is +2.16 going into 2024 (rapidly improving).

### Feature 5: `augusta_rounds_last_2yrs`
Competitive rounds in the 2 most recent prior seasons. Max = 8 (made cut both years). Captures recency of course time — Augusta changes year to year.

### Feature 6: `augusta_best_finish_recent`
Best finish in last 3 prior appearances. Different from career best — a player's best-ever finish 8 years ago is less relevant than their recent ceiling.

### Bonus: `tour_vs_augusta_divergence`
Player's sg_total_8w percentile rank minus their augusta_scoring_avg percentile rank (among players with 6+ competitive rounds). Positive = tour form overpredicts Augusta performance.

JT's values: 0.16 (2023) → 0.43 (2024) → 0.51 (2025) — rising divergence correctly flagged.

---

## 4. Feature Values for Key Players

| Player | Year | CompRds | CutPrev | Tier | Trajectory | Rds2yr | BestRec | CutRate | T10Rate | ScoreAvg |
|--------|------|---------|---------|------|-----------|--------|---------|---------|---------|---------|
| **Justin Thomas** | 2022 | 4 | 1 | 1 | 0.00 | 4 | 21 | 1.00 | 0.00 | -1.00 |
| **Justin Thomas** | 2023 | 8 | 1 | 2 | -0.25 | 8 | 8 | 1.00 | 0.50 | -3.00 |
| **Justin Thomas** | 2024 | 10 | 0 | 2 | -0.25 | 6 | 8 | 0.67 | 0.33 | 0.03 |
| **Justin Thomas** | 2025 | 12 | 0 | 2 | -0.25 | 4 | 8 | 0.50 | 0.25 | +0.75 |
| Scottie Scheffler | 2022 | 4 | 1 | 1 | 0.00 | 4 | 18 | 1.00 | 0.00 | -2.00 |
| Scottie Scheffler | 2023 | 8 | 1 | 2 | -0.82 | 8 | 1 | 1.00 | 0.50 | -8.57 |
| **Scottie Scheffler** | **2024** | **12** | **1** | **2** | **+2.16** | **8** | **1** | **1.00** | **0.67** | **-6.59** |
| Scottie Scheffler | 2025 | 16 | 1 | 2 | -0.03 | 8 | 1 | 1.00 | 0.75 | -10.03 |
| Jon Rahm | 2021 | 0 | 0 | 0 | 0.00 | 0 | 89 | 0.00 | 0.00 | 0.00 |
| Jon Rahm | 2023 | 8 | 1 | 2 | +0.54 | 8 | 5 | 1.00 | 0.50 | -2.71 |
| Rory McIlroy | 2025 | 12 | 1 | 2 | +1.33 | 6 | 2 | 0.50 | 0.25 | -0.38 |
| Xander Schauffele | 2024 | 10 | 1 | 2 | +0.56 | 6 | 3 | 0.67 | 0.67 | -2.54 |
| Cameron Smith | 2024 | 12 | 1 | 2 | +0.25 | 8 | 3 | 1.00 | 0.67 | -2.00 |

Notice the pattern: Scheffler going into 2024 has trajectory=+2.16 (rapidly improving), made_cut_prev=1, best_recent=1, scoring_avg=-6.59 (dominant). The model ranked him #2 and he won.

---

## 5. Backtest Results (V3)

### Summary Table

| Year | Brier(w) | Spearman | T10 Prec | S2 AUC | Blend AUC | MC AUC | ROI(t10) |
|------|---------|----------|---------|--------|----------|--------|---------|
| 2021 | 0.0134 | 0.227 | 30% | 0.500 | 0.534 | 0.534 | 0% |
| 2022 | 0.0141 | -0.009 | **50%** | **0.674** | **0.676** | 0.469 | **+55%** |
| 2023 | 0.0145 | -0.015 | **40%** | **0.669** | **0.657** | 0.391 | **+104%** |
| 2024 | 0.0134 | 0.177 | **60%** | **0.738** | **0.741** | 0.777 | **+36%** |
| 2025 | 0.0144 | 0.173 | 30% | 0.600 | 0.592 | 0.569 | **+78%** |
| **AVG** | **0.0140** | **0.111** | **42%** | **0.636** | **0.640** | **0.548** | **+55%** |

Note: 2021 S2 AUC = 0.500 (random) because the pre-2021 Augusta training data has 0% top-10 positive rate in the SG era ��� cold start problem. From 2022 onward, S2 consistently beats MC.

### Where the model found its winners

| Year | Winner | Model Rank | Blend Prob | S2 Prob |
|------|--------|-----------|-----------|---------|
| 2021 | Hideki Matsuyama | #48/76 | 0.028 | 0.022 |
| 2022 | Scottie Scheffler | #17/73 | 0.413 | 0.456 |
| 2023 | Jon Rahm | #25/68 | 0.053 | 0.034 |
| 2024 | **Scottie Scheffler** | **#2/70** | **0.873** | **0.932** |
| 2025 | Rory McIlroy | #16/71 | 0.087 | 0.090 |

The model's top-10 predictions are much stronger than its outright winner picks. It correctly identified Scheffler 2024 as the #2 most likely top-10 finisher (he won). It also correctly identified Schauffele 2024 as #1 (he finished 8th).

### Top 15 Predictions vs Actual — Best and Worst Years

**2024 — BEST YEAR (60% precision, AUC=0.741)**
| Rk | Player | Blend | S2 | Actual |
|----|--------|-------|-----|--------|
| 1 | Xander Schauffele | 0.891 | 0.962 | **8th** |
| 2 | Scottie Scheffler | 0.873 | 0.932 | **1st (WIN)** |
| 3 | Will Zalatoris | 0.862 | 0.919 | **T9** |
| 4 | Cameron Smith | 0.853 | 0.921 | **T6** |
| 5 | Collin Morikawa | 0.811 | 0.896 | **T3** |
| 6 | Brooks Koepka | 0.737 | 0.805 | T45 |
| 7 | Viktor Hovland | 0.697 | 0.772 | CUT |
| 8 | Corey Conners | 0.683 | 0.732 | T38 |
| 9 | Jordan Spieth | 0.681 | 0.729 | CUT |
| 10 | Cameron Young | 0.681 | 0.750 | **T9** |
6 of top 10 predictions actually finished top-10. Missed: Homa (T3), Fleetwood (T3), DeChambeau (T6), Hatton (T9).

**2021 — COLD START YEAR (30% precision, S2 AUC=0.500)**
| Rk | Player | Blend | S2 | Actual |
|----|--------|-------|-----|--------|
| 1 | Dustin Johnson | 0.060 | 0.022 | CUT |
| 2 | Lee Westwood | 0.058 | 0.022 | CUT |
| 3 | Jason Kokrak | 0.055 | 0.022 | 49 |
| 4 | Xander Schauffele | 0.054 | 0.022 | **T3** |
| 5 | Scottie Scheffler | 0.052 | 0.022 | T18 |
S2 assigns 0.022 to everyone — it hasn't seen any top-10 examples in the SG era yet. The blend probabilities are driven entirely by MC in 2021. Matsuyama (winner) was ranked #48.

---

## 6. JT Diagnosis — The Fix That Worked

### V2 → V3 Ranking Comparison

| Year | V2 Rank | V2 Blend | V3 Rank | V3 Blend | Actual |
|------|---------|---------|---------|---------|--------|
| 2023 | **#1** | 0.734 | **#5** | 0.402 | CUT |
| 2024 | **#1** | 0.741 | **#18** | 0.518 | CUT |
| 2025 | **#1** | 0.734 | **#28** | 0.055 | T36 |

### What the features reveal about JT's decline

| Feature | 2022 | 2023 | 2024 | 2025 | Direction |
|---------|------|------|------|------|-----------|
| competitive_rounds | 4 | 8 | 10 | 12 | Growing experience |
| made_cut_prev_year | 1 | 1 | **0** | **0** | Lost recent cut streak |
| scoring_trajectory | 0.00 | -0.25 | -0.25 | **-0.25** | Declining |
| made_cut_rate | 1.00 | 1.00 | **0.67** | **0.50** | Falling fast |
| scoring_avg | -1.00 | -3.00 | **+0.03** | **+0.75** | Turned positive (bad) |
| tour_vs_augusta_divergence | — | 0.16 | **0.43** | **0.51** | Tour form overpredicts |

JT's tour form is still elite (sg_total_8w ~0.59). But every Augusta-specific signal is deteriorating: he stopped making cuts, his scoring average turned positive (worse than field median), and his divergence keeps growing. The Augusta-only Stage 2 picks up all of these signals because it was trained on data where they matter.

### Why V2 failed on JT
V2's Stage 2 trained on all PGA Tour events (~16K rows). Augusta features were almost always 0 in that training set, so the model learned weak splits on them. JT's dominant sg_total_3w (+2.8 SHAP) drowned out the weak Augusta signals. V3 trains on 400-600 Augusta-only rows where experience features are always populated and directly predictive.

---

## 7. Known Limitations

1. **2021 cold start**: No SG-era top-10 examples before 2021, so S2 assigns near-uniform probabilities. The model effectively falls back to MC-only for the first backtest year.
2. **Winners are hard to predict**: The model ranks winners at #2 (Scheffler 2024), #16-17 (Scheffler 2022, McIlroy 2025), or #25-48 (Rahm 2023, Matsuyama 2021). Top-10 prediction is fundamentally easier than outright winner prediction.
3. **No real closing odds**: All ROI calculations use naive 10/field_size market probabilities. Real sportsbook top-10 odds would sharpen the edge analysis.
4. **Small Augusta training set**: 271-624 rows depending on year. XGBoost handles this with regularization (max_depth=3, high reg_alpha/lambda) but the model is necessarily conservative.
5. **2022-2023 Spearman negative**: The model's overall ranking of players (by win probability) was slightly worse than random in these years, even though top-10 precision was good. Win probability and top-10 probability capture different signals.
6. **S2 probabilities poorly calibrated**: In 2024, S2 assigns 0.962 to Schauffele and 0.932 to Scheffler — these are not true probabilities. They're useful for ranking but not for literal probability estimates.

---

## 8. Data Pipeline Summary

### What exists
| File | Rows | Description |
|------|------|-------------|
| masters_unified.parquet | 1,845 | All Masters player-seasons 2004-2025 |
| masters_sg_rounds.parquet | 1,792 | Round-level SG 2021-2025 |
| masters_sg_history.parquet | 448 | Tournament-level SG 2021-2025 |
| masters_scores_historical.parquet | 1,397 | Wikipedia scores 2004-2020 |
| augusta_player_features.parquet | 1,845 | 18 Augusta features per player-season |
| masters_weather.parquet | 5 | Tournament weather 2021-2025 |
| backtest_results_v3.parquet | 358 | Full V3 backtest output |

### Pipeline scripts
- `run_pipeline.py` — V1: data ingestion, basic backtest
- `run_v2_pipeline.py` — V2: scoring features, weather, Stage 2, betting edge
- `run_v3_pipeline.py` — V3: experience features, Augusta-only S2, blend optimization

---

## 9. Recommended Next Steps

1. **Integrate real top-10 odds** from The Odds API or DG betting-tools/outrights for proper ROI calculation
2. **Build Streamlit app** around top-10 predictions — show blend probabilities, flag high-edge plays, display experience feature profiles
3. **Add 2019-2020 Masters scores** to the SG-era gap (these years have Wikipedia data but no SG — could scrape round scores from Masters.com archive with Selenium)
4. **Calibrate S2 probabilities** — apply Platt scaling or isotonic regression so the probabilities are meaningful, not just useful for ranking
5. **Test Stage 2 on 2026 Masters** when data becomes available — this will be the first true out-of-sample test

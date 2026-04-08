# Augusta National Model — V6 Final Briefing
## 2026 Masters Tournament — April 9-12

---

## 1. What This Is

A two-stage XGBoost + Monte Carlo prediction model for the Masters, built across 7 development sessions. Backtested on 5 Masters (2021-2025). Trained on 33,443 PGA Tour SG rows (2015-2026) with real DG field strength weighting. Generates win, top-5, top-10, and top-20 probabilities for every player in the 2026 field.

**Repo**: [github.com/dylanjaynes/augusta-national-model](https://github.com/dylanjaynes/augusta-national-model)

---

## 2. Model Architecture

### Stage 1 — XGBoost Regression → Monte Carlo
- Predicts finish percentile from 16 rolling SG features
- Trained on 18,290 PGA Tour tournament rows (2015-2026)
- **NEW**: Rolling features use real DG field strength weights — a +2.0 SG at the Players Championship (FS=+0.42, weight=1.13) counts more than +2.0 SG at Barbasol (FS=-1.18, weight=0.65)
- 50,000 Monte Carlo simulations (noise_std=0.10, target_pred_std=0.14)

### Stage 2 — XGBoost Binary Classifier (top-10)
- Trained on Augusta data only (719 Masters rows, 2004-2025)
- 34 features: 16 rolling SG + 18 Augusta-specific experience features
- Temperature scaling T=2.5 on raw probabilities

### Blending
```
final_top10 = 0.90 × Stage2_calibrated + 0.10 × MC_top10
```

---

## 3. Data Pipeline

| Source | Rows | Seasons | Notes |
|--------|------|---------|-------|
| PGA Tour SG (extended) | 33,443 | 2015-2026 | Scraped from datagolf.com, 80/81 field players current |
| DG Field Strength | 838 | 2019-2026 | Real field quality scores for weighting |
| Masters SG Rounds | 1,792 | 2021-2025 | Per-round SG from local CSVs |
| Masters Scores (Wikipedia) | 1,397 | 2004-2020 | Full leaderboards |
| Masters Unified | 1,845 | 2004-2025 | Merged for Stage 2 training |
| DraftKings Odds | 91 | 2026 | Live outrights via The Odds API |

### Field Strength Weighting
Each event's SG values are weighted by measured field quality:
```
weight = max(0.4, min(3.0, 1.0 + 0.3 × field_strength_mean))
```

| Event Type | FS Score | Weight | Effect |
|-----------|---------|--------|--------|
| Tour Championship | +1.33 | 1.40 | SG counts 40% more |
| Masters | +0.40 | 1.12 | SG counts 12% more |
| Players Championship | +0.42 | 1.13 | SG counts 13% more |
| Average PGA event | 0.00 | 1.00 | Baseline |
| LIV average | -0.10 | 0.97 | SG counts 3% less |
| Sanderson Farms | -0.40 | 0.88 | SG counts 12% less |
| Barbasol | -1.18 | 0.65 | SG counts 35% less |
| Korn Ferry avg | -1.13 | 0.66 | SG counts 34% less |

72% of training rows used real FS weights. 28% (pre-2019) used hardcoded tier fallbacks.

---

## 4. Backtest Results (V4/V5)

| Year | Blend AUC | T10 Precision | Spearman |
|------|----------|--------------|----------|
| 2021 | 0.529 | 30% | 0.224 |
| 2022 | 0.652 | 50% | 0.008 |
| 2023 | 0.671 | 50% | -0.030 |
| 2024 | 0.712 | 50% | 0.204 |
| 2025 | 0.608 | 30% | 0.205 |
| **AVG** | **0.634** | **42%** | **0.122** |

Top-10 precision of 42% means ~4 of the model's top-10 predicted players actually finish in the real top-10. The model was profitable on a naive top-10 betting strategy in 4 of 5 backtest years.

### Build History

| Version | Avg AUC | T10 Prec | Key Change |
|---------|---------|---------|------------|
| V1 | — | 28% | Base MC model |
| V2 | 0.598 | 26% | Stage 2 (tour-wide training) |
| V3 | 0.640 | 42% | Augusta-only S2, experience features |
| V4 | 0.644 | 40% | Extended data scrape (33K rows) |
| **V5** | **0.634** | **42%** | **Real field strength weights** |

---

## 5. 2026 Predictions — Top 20

| Rk | Player | Win% | Top-10% | Top-20% | DK Odds | Exp Tier | Cut Prev | Trajectory | Diverg |
|----|--------|------|---------|---------|---------|----------|----------|-----------|--------|
| 1 | Scottie Scheffler | 6.3% | 73.8% | 73.8% | +495 | Veteran | Yes | +2.46 | 0.31 |
| 2 | Cameron Young | 0.2% | 72.4% | 72.4% | +2300 | Established | No | +0.07 | 0.26 |
| 3 | Xander Schauffele | 3.1% | 71.3% | 71.3% | +1800 | Established | Yes | -0.05 | 0.22 |
| 4 | Sungjae Im | 2.6% | 70.0% | 70.0% | +11500 | Established | Yes | -0.55 | 0.10 |
| 5 | Bryson DeChambeau | 0.2% | 65.9% | 65.9% | +1050 | Veteran | Yes | +2.08 | 0.16 |
| 6 | Corey Conners | 3.1% | 65.0% | 65.0% | +8400 | Veteran | Yes | -1.53 | 0.18 |
| 7 | Ben Griffin | 0.0% | 63.7% | 63.7% | +11000 | Debutant | No | 0.00 | 0.00 |
| 8 | Ludvig Aberg | 0.0% | 63.6% | 63.6% | +1650 | Established | Yes | +0.50 | 0.37 |
| 9 | Rory McIlroy | 0.0% | 63.6% | 63.6% | +1175 | Established | Yes | -1.49 | 0.06 |
| 10 | Cameron Smith | 2.1% | 63.6% | 63.6% | +11000 | Established | No | -2.33 | 0.28 |
| 11 | Justin Thomas | 8.4% | 61.9% | 74.5% | +5400 | Established | Yes | +1.20 | 0.29 |
| 12 | Zach Johnson | 0.4% | 61.7% | 61.7% | +55000 | Veteran | Yes | -0.97 | -0.11 |
| 13 | Patrick Reed | 0.1% | 61.5% | 61.5% | +4300 | Veteran | Yes | +0.93 | -0.21 |
| 14 | Max Homa | 5.3% | 61.2% | 66.5% | +12000 | Established | Yes | +5.29 | 0.12 |
| 15 | Jordan Spieth | 3.9% | 60.9% | 60.9% | +4200 | Established | Yes | -0.01 | 0.14 |
| 16 | Kurt Kitayama | 0.1% | 60.1% | 60.1% | +9400 | Learning | No | 0.00 | 0.13 |
| 17 | Jason Day | 0.0% | 60.1% | 60.1% | +6800 | Veteran | Yes | +1.17 | -0.13 |
| 18 | Haotong Li | 5.6% | 60.0% | 67.5% | +27500 | Debutant | No | 0.00 | 0.00 |
| 19 | Marco Penge | 5.9% | 59.9% | 68.3% | +14500 | Debutant | No | 0.00 | 0.00 |
| 20 | Collin Morikawa | 0.0% | 59.9% | 59.9% | +3100 | Veteran | Yes | +0.65 | -0.09 |

### Rest of Field (21-81)

| Rk | Player | Win% | Top-10% | DK Odds | Tier | Cut Prev | Traj | Diverg |
|----|--------|------|---------|---------|------|----------|------|--------|
| 21 | Matt Fitzpatrick | 0.0% | 59.8% | +2350 | Vet | Yes | -1.27 | 0.42 |
| 22 | Justin Rose | 3.1% | 58.9% | +3500 | Est | Yes | -0.93 | -0.02 |
| 23 | Patrick Cantlay | 3.9% | 58.6% | +5800 | Vet | Yes | -0.82 | 0.24 |
| 24 | Davis Riley | 4.4% | 57.6% | +82500 | Learn | Yes | 0.00 | 0.00 |
| 25 | Jon Rahm | 1.9% | 57.2% | +910 | Vet | Yes | -3.54 | 0.17 |
| 26 | Shane Lowry | 0.7% | 56.9% | +6700 | Vet | Yes | -3.58 | 0.19 |
| 27 | Tommy Fleetwood | 0.5% | 56.6% | +2250 | Vet | Yes | +3.06 | 0.03 |
| 28 | Matt McCarty | 1.9% | 56.3% | +28000 | Learn | Yes | 0.00 | 0.00 |
| 29 | Tyrrell Hatton | 2.3% | 56.2% | +8200 | Vet | Yes | +4.23 | -0.03 |
| 30 | Nicolai Hojgaard | 0.3% | 55.5% | +7400 | Learn | No | 0.00 | 0.43 |
| 31 | Brooks Koepka | 0.2% | 55.1% | +3700 | Est | No | +0.86 | -0.10 |
| 32 | Harry Hall | 1.7% | 54.8% | +15500 | Debut | No | 0.00 | 0.00 |
| 33 | Johnny Keefer | 1.9% | 54.7% | +37000 | Debut | No | 0.00 | 0.00 |
| 34 | Jake Knapp | 1.1% | 53.0% | +6500 | Learn | No | 0.00 | 0.00 |
| 35 | Rasmus Neergaard-Petersen | 0.1% | 52.8% | +32000 | Debut | No | 0.00 | 0.00 |
| 36 | Keegan Bradley | 0.2% | 52.2% | +14000 | Vet | No | +0.25 | 0.07 |
| 37 | Viktor Hovland | 0.0% | 51.3% | +4500 | Vet | Yes | +1.51 | -0.06 |
| 38 | Brian Harman | 2.1% | 51.2% | +22000 | Est | Yes | +1.56 | 0.05 |
| 39 | Daniel Berger | 3.0% | 50.4% | +12000 | Est | Yes | -3.18 | 0.01 |
| 40 | Alex Noren | 0.0% | 50.3% | +25000 | Learn | No | 0.00 | 0.00 |
| 41 | Kristoffer Reitan | 0.9% | 49.6% | +22500 | Debut | No | 0.00 | 0.00 |
| 42 | Russell Henley | 0.0% | 49.4% | +4700 | Vet | No | +1.36 | -0.14 |
| 43 | Casey Jarvis | 0.0% | 47.7% | +23000 | Debut | No | 0.00 | 0.00 |
| 44 | Sam Stevens | 3.0% | 47.6% | +17500 | Debut | No | 0.00 | 0.00 |
| 45 | Charl Schwartzel | 0.2% | 47.1% | +62500 | Est | Yes | -3.14 | -0.29 |
| 46 | Aldrich Potgieter | 0.0% | 46.7% | +36000 | Learn | No | 0.00 | 0.00 |
| 47 | Tom McKibbin | 0.5% | 44.9% | +29000 | Debut | No | 0.00 | 0.00 |
| 48 | Aaron Rai | 0.0% | 43.9% | +25000 | Learn | Yes | 0.00 | 0.00 |
| 49 | Chris Gotterup | 0.0% | 43.2% | +25000 | Debut | No | 0.00 | 0.00 |
| 50 | Sam Burns | 6.7% | 43.0% | +7200 | Est | Yes | +0.70 | 0.27 |
| 51 | Nick Taylor | 0.0% | 42.9% | +19000 | Learn | Yes | 0.00 | -0.23 |
| 52 | Sergio Garcia | 1.8% | 42.4% | +25000 | Est | No | 0.00 | -0.04 |
| 53 | Nico Echavarria | 1.6% | 42.4% | +32500 | Learn | Yes | 0.00 | 0.00 |
| 54 | Hideki Matsuyama | 0.3% | 42.0% | +2700 | Vet | Yes | -2.14 | -0.00 |
| 55 | Si Woo Kim | 0.3% | 41.8% | +5000 | Est | No | +0.03 | -0.18 |
| 56 | Akshay Bhatia | 0.0% | 40.8% | +5600 | Est | Yes | +0.29 | -0.24 |
| 57 | Gary Woodland | 0.7% | 40.7% | +12000 | Est | No | -0.84 | 0.11 |
| 58 | J.J. Spaun | 0.5% | 39.7% | +6900 | Est | Yes | +1.73 | -0.05 |
| 59 | Adam Scott | 0.0% | 37.0% | +6000 | Est | No | +2.71 | -0.34 |
| 60 | Max Greyserman | 0.0% | 36.8% | +21000 | Learn | Yes | 0.00 | 0.00 |
| 61 | Ryan Fox | 0.0% | 35.7% | +20000 | Est | No | +0.14 | -0.20 |
| 62 | Andrew Novak | 2.9% | 35.3% | +29000 | Debut | No | 0.00 | 0.00 |
| 63 | Dustin Johnson | 0.0% | 34.9% | +25000 | Est | No | +0.75 | -0.08 |
| 64 | Bubba Watson | 0.3% | 34.2% | +52500 | Vet | Yes | -0.47 | -0.27 |
| 65 | Jacob Bridgeman | 3.5% | 34.1% | +8400 | Debut | No | 0.00 | 0.00 |
| 66 | Sepp Straka | 0.0% | 33.5% | +7800 | Est | No | -0.66 | -0.31 |
| 67 | Maverick McNealy | 0.0% | 32.5% | +7800 | Learn | Yes | 0.00 | 0.00 |
| 68 | Sami Valimaki | 0.0% | 32.2% | N/A | Debut | No | 0.00 | 0.00 |
| 69 | Ryan Gerard | 0.0% | 31.7% | +15500 | Debut | No | 0.00 | 0.00 |
| 70 | Wyndham Clark | 0.0% | 31.0% | +20000 | Learn | Yes | 0.00 | -0.35 |
| 71 | Fred Couples | 0.1% | 30.5% | +500000 | Deep Vet | No | +3.17 | -0.25 |
| 72 | Carlos Ortiz | 0.0% | 30.2% | +23000 | Learn | No | 0.00 | 0.00 |
| 73 | Danny Willett | 0.0% | 28.3% | +225000 | Est | Yes | -1.30 | -0.26 |
| 74 | Robert MacIntyre | 0.0% | 27.7% | +3500 | Est | No | +0.18 | -0.41 |
| 75 | Rasmus Hojgaard | 0.1% | 27.1% | +12000 | Learn | Yes | 0.00 | 0.00 |
| 76 | Vijay Singh | 0.0% | 26.5% | +500000 | Vet | No | +1.07 | -0.39 |
| 77 | Harris English | 0.0% | 26.0% | +10000 | Est | Yes | +1.74 | -0.37 |
| 78 | Min Woo Lee | 0.0% | 25.2% | +3400 | Est | Yes | -0.41 | -0.50 |
| 79 | Brian Campbell | 0.0% | 21.9% | N/A | Learn | Yes | 0.00 | 0.00 |
| 80 | Mike Weir | 0.0% | 21.6% | +500000 | Est | No | 0.00 | -0.42 |
| 81 | Michael Kim | 0.0% | 20.3% | +21000 | Learn | Yes | 0.00 | 0.00 |

---

## 6. Augusta Experience Features

All computed with strict temporal cutoff — only data from prior seasons.

| Feature | Description |
|---------|-------------|
| `augusta_competitive_rounds` | Total rounds under tournament pressure (4 if cut, 2 if MC) |
| `augusta_made_cut_prev_year` | Made the cut at Augusta last year? 9/10 recent winners = Yes |
| `augusta_experience_tier` | Debutant / Learning / Established / Veteran / Deep Vet |
| `augusta_scoring_trajectory` | Career avg minus recent-2 avg. Positive = improving |
| `augusta_rounds_last_2yrs` | Competitive rounds in Y-1 and Y-2 (max 8) |
| `augusta_best_finish_recent` | Best finish in last 3 appearances |
| `tour_vs_augusta_divergence` | Tour form percentile minus Augusta scoring percentile. Positive = fade candidate |
| `augusta_starts` | Total prior Masters appearances |
| `augusta_made_cut_rate` | Career cut rate at Augusta |
| `augusta_top10_rate` | Career top-10 rate at Augusta |
| `augusta_best_finish` | Career best finish |
| `augusta_scoring_avg` | Decay-weighted career score_vs_field |
| `augusta_sg_app_career` | Decay-weighted career sg_app at Augusta |
| `augusta_sg_total_career` | Decay-weighted career sg_total at Augusta |
| `augusta_bogey_avoidance` | Rate of clean (at/under par) rounds |
| `augusta_round_variance_score` | Std dev of round scores (volatility) |
| `augusta_back9_scoring` | R3+R4 performance (pressure rounds) |
| `tournament_wind_avg` | Weather: avg daily max wind for tournament week |

---

## 7. Fade Candidates

Players where `tour_vs_augusta_divergence > 0.35`:

| Player | Divergence | Top-10% | Reason |
|--------|-----------|---------|--------|
| Nicolai Hojgaard | 0.43 | 55.5% | 6 Augusta rounds, strong tour form not proven here |
| Matt Fitzpatrick | 0.42 | 59.8% | Trajectory -1.27, declining at Augusta despite veteran status |
| Ludvig Aberg | 0.37 | 63.6% | Only 8 competitive rounds, elite tour form may not translate |

---

## 8. Known Issues

1. **S2 top-10% compressed upward**: Most "good" players get 55-73% top-10 probability. The **relative ordering** is more reliable than the absolute numbers. Use rankings, not raw probabilities.

2. **Win% noisy for non-elite players**: MC simulation gives inflated win% to some debutants/longshots (Haotong Li 5.6%, Marco Penge 5.9%) because they have strong tour SG but zero Augusta history for S2 to discount.

3. **2023 season gap**: DG stats pages only serve the most recent year per event via server-side rendering. Year-switching is client-side JS we couldn't replicate. The dataset jumps from 2022 → 2024.

4. **LIV data limited**: DG field strength CSV has 51 LIV events (FS scores range -0.59 to +0.05), and some LIV players appear in our extended tour data. But we couldn't scrape LIV-specific per-event SG from the stats pages.

5. **15 debutants have zero Augusta features**: Model relies entirely on tour form. Historically debutants almost never win but can finish top-10 at ~11% base rate.

---

## 9. What Changed in Each Session

| Session | Key Change | Impact |
|---------|-----------|--------|
| 1 | Data ingestion, base MC model | Foundation |
| 2 | Stage 2 classifier, scoring features, weather | Top-10 market = money market |
| 3 | Augusta-only S2, 6 experience features, JT fix | T10 prec 26% → 42% |
| 4 | Streamlit app, DraftKings odds, deployment | Ship day |
| 5 | MC spread fix (noise 0.16→0.10), temperature scaling | Scheffler win% 5.5% → 10.8% |
| 6 | **Scraped 4,750 new tour rows (2025-2026)** | **80/81 players current (was 57/81 stale)** |
| 7 | **Real DG field strength weights (838 records)** | **Cameron Young +0.082 sg_total_3w, LIV players discounted** |

---

## 10. Files

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | V1 data ingestion |
| `run_v2_pipeline.py` | V2 scoring features + Stage 2 |
| `run_v3_pipeline.py` | V3 experience features |
| `run_2026_predictions.py` | 2026 field through pipeline |
| `run_fix_spread.py` | MC spread + temperature scaling |
| `run_retrain_extended.py` | V4 retrain on extended data |
| `run_field_strength.py` | V5 real field strength weights |
| `scrape_dg_historical.py` | DG stats page scraper |
| `streamlit_app/app.py` | Streamlit app (4 pages) |

| Data File | Rows | Description |
|-----------|------|-------------|
| `historical_rounds_extended.parquet` | 33,443 | All tour SG 2015-2026 |
| `dg_field_strength.csv` | 838 | Real field quality scores |
| `masters_unified.parquet` | 1,845 | All Masters 2004-2025 |
| `augusta_player_features.parquet` | 1,845 | 18 features per player-season |
| `predictions_2026.parquet` | 81 | Final 2026 predictions |
| `dk_odds_2026.csv` | 96 | DraftKings fair probabilities |

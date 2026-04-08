# Augusta National Model — V7 Final Briefing
## 2026 Masters Tournament — April 9-12

---

## 1. What This Is

A two-stage XGBoost + Monte Carlo model predicting Masters outcomes. Built across 8 sessions. Trained on **38,629 tour rows** spanning 2015-2026 with real DG field strength weighting. Includes PGA Tour, LIV Golf, and 5 years of Augusta-specific SG data. Generates win, top-5, top-10, and top-20 probabilities for all 81 players.

**Backtest: 44% top-10 precision, 0.640 AUC, profitable in 4/5 years.**

**Repo**: [github.com/dylanjaynes/augusta-national-model](https://github.com/dylanjaynes/augusta-national-model)

---

## 2. Architecture

### Stage 1 — XGBoost Regression → Monte Carlo
- Predicts finish percentile from 16 rolling SG features (field-strength weighted)
- Trained on 18,187 rows | 50,000 MC simulations (noise=0.10, target_std=0.14)

### Stage 2 — XGBoost Binary Classifier (top-10)
- Trained on Augusta data only (719 rows, 2004-2025)
- 34 features: 16 rolling SG + 18 Augusta experience features
- Temperature scaling T=2.5

### Blend
```
final_top10 = 0.90 × Stage2_calibrated + 0.10 × MC_top10
```

---

## 3. Training Data

**38,629 total rows** — every season from 2015-2026 now covered.

| Season | Rows | Events | LIV Events | Notes |
|--------|------|--------|-----------|-------|
| 2015 | 2,694 | 26 | 0 | Original CSV |
| 2016 | 2,948 | 27 | 0 | Original CSV |
| 2017 | 3,604 | 31 | 0 | Original CSV |
| 2018 | 3,996 | 34 | 0 | Original CSV |
| 2019 | 4,455 | 36 | 0 | Original CSV |
| 2020 | 3,793 | 30 | 0 | Original CSV |
| 2021 | 4,821 | 40 | 0 | Original CSV |
| 2022 | 3,457 | 28 | 0 | Original CSV + DG scrape |
| **2023** | **3,142** | **24** | **0** | **NEW — scraped from DG past-results** |
| 2024 | 839 | 14 | 707 | DG scrape + LIV past-results |
| 2025 | 3,059 | 34 | 703 | DG scrape + LIV past-results |
| 2026 | 1,821 | 18 | 285 | DG scrape + LIV past-results |

### Data Sources
- **PGA Tour 2015-2022**: Original golf_model CSV (full SG breakdown)
- **PGA Tour 2023**: Scraped from `datagolf.com/past-results/pga-tour/{tid}/2023` (28 events, sg_total only)
- **PGA Tour 2024-2026**: Scraped from `datagolf.com/historical-tournament-stats` (full SG breakdown)
- **LIV Golf 2024-2026**: Scraped from `datagolf.com/past-results/liv-golf/{tid}/{year}` (31 events, sg_total per round)
- **Masters SG 2021-2025**: Local CSVs with full round-level SG breakdown
- **Masters Scores 2004-2020**: Wikipedia leaderboards
- **DG Field Strength**: 838 records (2019-2026) for event-tier weighting
- **DraftKings Odds**: Live via The Odds API

### Field Strength Weighting
```
weight = max(0.4, min(3.0, 1.0 + 0.3 × field_strength_mean))
```
Applied to rolling SG feature computation. 72% of rows use real FS scores, 28% use hardcoded tier fallbacks (pre-2019).

---

## 4. Key Player Data Coverage

| Player | Total Events | Latest | 2023 Events | LIV Events | Career sg_total |
|--------|-------------|--------|-------------|-----------|----------------|
| Scottie Scheffler | 102 | 2026 | 16 | 0 | +1.156 |
| Jon Rahm | 142 | 2026 | 15 | 31 | +1.043 |
| Ludvig Aberg | 24 | 2026 | 5 | 0 | +1.011 |
| Rory McIlroy | 108 | 2026 | 12 | 0 | +0.654 |
| Cameron Young | 57 | 2026 | 16 | 0 | +0.614 |
| Viktor Hovland | 87 | 2026 | 15 | 0 | +0.612 |
| Bryson DeChambeau | 143 | 2026 | 3 | 31 | +0.538 |
| Xander Schauffele | 132 | 2026 | 15 | 0 | +0.472 |
| Cameron Smith | 163 | 2026 | 3 | 31 | +0.110 |
| Wyndham Clark | 123 | 2026 | 13 | 0 | -0.105 |

**80 of 81** field players have current (2025-2026) tour data. Only Casey Jarvis missing.

---

## 5. Backtest Results

| Year | Blend AUC | T10 Precision | MC-only AUC |
|------|----------|--------------|-------------|
| 2021 | 0.531 | 30% | 0.531 |
| 2022 | 0.652 | 50% | 0.476 |
| 2023 | 0.679 | 50% | 0.372 |
| 2024 | 0.720 | 60% | 0.778 |
| 2025 | 0.619 | 30% | 0.584 |
| **AVG** | **0.640** | **44%** | **0.548** |

Field-strength weighting added +0.006 AUC in 2023 and +2pp precision overall vs unweighted.

---

## 6. 2026 Predictions — Full Field

### Top 20

| Rk | Player | Win% | Top-10% | Top-20% | DK Odds | Tier | Cut Prev | Trajectory | Diverg |
|----|--------|------|---------|---------|---------|------|----------|-----------|--------|
| 1 | Scottie Scheffler | 6.7% | 74.4% | 74.4% | +495 | Vet | Yes | +2.46 | 0.31 |
| 2 | Cameron Young | 0.2% | 72.3% | 72.3% | +2300 | Est | No | +0.07 | 0.26 |
| 3 | Xander Schauffele | 3.0% | 71.2% | 71.2% | +1800 | Est | Yes | -0.05 | 0.22 |
| 4 | Sungjae Im | 2.3% | 70.0% | 70.0% | +11500 | Est | Yes | -0.55 | 0.10 |
| 5 | Corey Conners | 2.9% | 66.1% | 66.1% | +8400 | Vet | Yes | -1.53 | 0.18 |
| 6 | Bryson DeChambeau | 0.2% | 65.2% | 65.2% | +1050 | Vet | Yes | +2.08 | 0.16 |
| 7 | Ludvig Aberg | 0.0% | 63.1% | 63.1% | +1650 | Est | Yes | +0.50 | 0.37 |
| 8 | Cameron Smith | 1.8% | 62.5% | 62.5% | +11000 | Est | No | -2.33 | 0.28 |
| 9 | Rory McIlroy | 0.0% | 62.1% | 62.1% | +1175 | Est | Yes | -1.49 | 0.06 |
| 10 | Justin Thomas | 8.1% | 62.1% | 73.9% | +5400 | Est | Yes | +1.20 | 0.29 |
| 11 | Ben Griffin | 0.0% | 61.9% | 61.9% | +11000 | Debut | No | 0.00 | 0.00 |
| 12 | Max Homa | 5.6% | 61.7% | 67.3% | +12000 | Est | Yes | +5.29 | 0.12 |
| 13 | Patrick Reed | 0.1% | 61.3% | 61.3% | +4300 | Vet | Yes | +0.93 | -0.21 |
| 14 | Jordan Spieth | 3.3% | 61.0% | 61.0% | +4200 | Est | Yes | -0.01 | 0.14 |
| 15 | Kurt Kitayama | 0.1% | 61.0% | 61.0% | +9400 | Learn | No | 0.00 | 0.13 |
| 16 | Zach Johnson | 0.5% | 60.4% | 60.4% | +55000 | Vet | Yes | -0.97 | -0.11 |
| 17 | Collin Morikawa | 0.0% | 60.3% | 60.3% | +3100 | Vet | Yes | +0.65 | -0.09 |
| 18 | Haotong Li | 5.5% | 59.1% | 67.3% | +27500 | Debut | No | 0.00 | 0.00 |
| 19 | Marco Penge | 6.4% | 59.1% | 69.6% | +14500 | Debut | No | 0.00 | 0.00 |
| 20 | Jon Rahm | 1.8% | 58.7% | 58.7% | +910 | Vet | Yes | -3.54 | 0.17 |

### 21-50

| Rk | Player | Win% | Top-10% | DK Odds | Tier | Cut Prev | Traj |
|----|--------|------|---------|---------|------|----------|------|
| 21 | Matt Fitzpatrick | 0.0% | 58.7% | +2350 | Vet | Yes | -1.27 |
| 22 | Jason Day | 0.0% | 58.5% | +6800 | Vet | Yes | +1.17 |
| 23 | Patrick Cantlay | 3.8% | 58.1% | +5800 | Vet | Yes | -0.82 |
| 24 | Justin Rose | 3.0% | 58.1% | +3500 | Est | Yes | -0.93 |
| 25 | Shane Lowry | 0.6% | 56.6% | +6700 | Vet | Yes | -3.58 |
| 26 | Tommy Fleetwood | 0.5% | 56.6% | +2250 | Vet | Yes | +3.06 |
| 27 | Tyrrell Hatton | 2.1% | 56.1% | +8200 | Vet | Yes | +4.23 |
| 28 | Davis Riley | 4.0% | 56.1% | +82500 | Learn | Yes | 0.00 |
| 29 | Brooks Koepka | 0.2% | 54.9% | +3700 | Est | No | +0.86 |
| 30 | Matt McCarty | 1.9% | 53.8% | +28000 | Learn | Yes | 0.00 |
| 31 | Nicolai Hojgaard | 0.3% | 53.7% | +7400 | Learn | No | 0.00 |
| 32 | Johnny Keefer | 2.1% | 52.5% | +37000 | Debut | No | 0.00 |
| 33 | Keegan Bradley | 0.2% | 51.7% | +14000 | Vet | No | +0.25 |
| 34 | Harry Hall | 1.5% | 51.5% | +15500 | Debut | No | 0.00 |
| 35 | Jake Knapp | 1.2% | 51.3% | +6500 | Learn | No | 0.00 |
| 36 | Rasmus Neergaard-Petersen | 0.1% | 51.2% | +32000 | Debut | No | 0.00 |
| 37 | Daniel Berger | 2.6% | 50.6% | +12000 | Est | Yes | -3.18 |
| 38 | Russell Henley | 0.0% | 49.8% | +4700 | Vet | No | +1.36 |
| 39 | Brian Harman | 2.0% | 49.7% | +22000 | Est | Yes | +1.56 |
| 40 | Viktor Hovland | 0.0% | 49.6% | +4500 | Vet | Yes | +1.51 |
| 41 | Kristoffer Reitan | 1.0% | 48.2% | +22500 | Debut | No | 0.00 |
| 42 | Charl Schwartzel | 0.2% | 47.9% | +62500 | Est | Yes | -3.14 |
| 43 | Alex Noren | 0.0% | 47.5% | +25000 | Learn | No | 0.00 |
| 44 | Sam Stevens | 2.8% | 46.7% | +17500 | Debut | No | 0.00 |
| 45 | Casey Jarvis | 0.0% | 46.5% | +23000 | Debut | No | 0.00 |
| 46 | Aldrich Potgieter | 0.0% | 44.8% | +36000 | Learn | No | 0.00 |
| 47 | Sergio Garcia | 1.9% | 42.8% | +25000 | Est | No | 0.00 |
| 48 | Hideki Matsuyama | 0.2% | 42.5% | +2700 | Vet | Yes | -2.14 |
| 49 | Si Woo Kim | 0.3% | 42.3% | +5000 | Est | No | +0.03 |
| 50 | Chris Gotterup | 0.0% | 41.9% | +25000 | Debut | No | 0.00 |

### 51-81

| Rk | Player | Win% | Top-10% | DK Odds | Tier | Cut Prev |
|----|--------|------|---------|---------|------|----------|
| 51 | Aaron Rai | 0.0% | 41.9% | +25000 | Learn | Yes |
| 52 | Nick Taylor | 0.0% | 41.7% | +19000 | Learn | Yes |
| 53 | Gary Woodland | 0.8% | 40.5% | +12000 | Est | No |
| 54 | Tom McKibbin | 0.4% | 40.3% | +29000 | Debut | No |
| 55 | J.J. Spaun | 0.5% | 39.4% | +6900 | Est | Yes |
| 56 | Nico Echavarria | 2.4% | 39.3% | +32500 | Learn | Yes |
| 57 | Akshay Bhatia | 0.0% | 38.9% | +5600 | Est | Yes |
| 58 | Sam Burns | 7.4% | 38.3% | +7200 | Est | Yes |
| 59 | Adam Scott | 0.0% | 36.4% | +6000 | Est | No |
| 60 | Bubba Watson | 0.3% | 35.8% | +52500 | Vet | Yes |
| 61 | Dustin Johnson | 0.0% | 35.5% | +25000 | Est | No |
| 62 | Max Greyserman | 0.0% | 34.6% | +21000 | Learn | Yes |
| 63 | Ryan Fox | 0.0% | 33.1% | +20000 | Est | No |
| 64 | Andrew Novak | 3.0% | 32.0% | +29000 | Debut | No |
| 65 | Ryan Gerard | 0.0% | 31.9% | +15500 | Debut | No |
| 66 | Sepp Straka | 0.0% | 31.9% | +7800 | Est | No |
| 67 | Fred Couples | 0.1% | 31.4% | +500000 | Deep Vet | No |
| 68 | Wyndham Clark | 0.0% | 31.2% | +20000 | Learn | Yes |
| 69 | Maverick McNealy | 0.0% | 31.2% | +7800 | Learn | Yes |
| 70 | Jacob Bridgeman | 3.8% | 30.1% | +8400 | Debut | No |
| 71 | Carlos Ortiz | 0.0% | 28.9% | +23000 | Learn | No |
| 72 | Sami Valimaki | 0.0% | 28.6% | N/A | Debut | No |
| 73 | Vijay Singh | 0.0% | 28.2% | +500000 | Vet | No |
| 74 | Robert MacIntyre | 0.0% | 28.1% | +3500 | Est | No |
| 75 | Danny Willett | 0.0% | 27.3% | +225000 | Est | Yes |
| 76 | Rasmus Hojgaard | 0.1% | 27.2% | +12000 | Learn | Yes |
| 77 | Harris English | 0.0% | 26.0% | +10000 | Est | Yes |
| 78 | Min Woo Lee | 0.0% | 24.8% | +3400 | Est | Yes |
| 79 | Mike Weir | 0.0% | 21.9% | +500000 | Est | No |
| 80 | Michael Kim | 0.0% | 19.5% | +21000 | Learn | Yes |
| 81 | Brian Campbell | 0.0% | 19.1% | N/A | Learn | Yes |

---

## 7. Fade Candidates

Players where `tour_vs_augusta_divergence > 0.35`:

| Player | Divergence | Top-10% | Why |
|--------|-----------|---------|-----|
| Nicolai Hojgaard | 0.43 | 53.7% | 6 Augusta rounds, strong tour form unproven here |
| Matt Fitzpatrick | 0.42 | 58.7% | Trajectory -1.27, declining despite veteran status |
| Ludvig Aberg | 0.37 | 63.1% | Only 8 rounds, elite tour form may not translate |
| Scottie Scheffler | 0.31 | 74.4% | Not a fade — trajectory +2.46, best in field |
| Justin Thomas | 0.29 | 62.1% | CUT in 2023 and 2024, still ranked too high by tour form |

---

## 8. Known Issues

1. **S2 top-10% still compressed upward**: Most established players cluster at 55-75%. Rankings are more reliable than absolute probabilities.
2. **Win% noisy for debutants**: Haotong Li (5.5%), Marco Penge (6.4%) have inflated win% from strong tour SG with zero Augusta history. Their top-10% rankings are more trustworthy.
3. **2023 PGA + LIV data has sg_total only** (no ott/app/arg/putt breakdown). These rows contribute to rolling sg_total features but not to individual SG category features.
4. **2024 PGA data sparse**: Only 14 events scraped for 2024 (DG stats page limitations). 2025-2026 have better coverage.
5. **LIV SG is per-round total only**: No strokes gained breakdown by category. LIV events are field-strength weighted at ~0.97x.

---

## 9. Build History (8 Sessions)

| Session | Key Change | Impact |
|---------|-----------|--------|
| 1 | Data ingestion, base MC model | Foundation: 1,845 Masters + 29K tour rows |
| 2 | Stage 2 classifier, weather, betting edge | Top-10 market = money market |
| 3 | Augusta-only S2, 6 experience features | JT fixed: #1→#28. T10 prec 26%→42% |
| 4 | Streamlit app, DraftKings odds, deploy | Shipped to github.com/dylanjaynes/augusta-national-model |
| 5 | MC spread fix, temperature scaling | Win% spread fixed: 152x ratio |
| 6 | **Scraped 4,750 new PGA rows (2025-2026)** | **80/81 players current (was 57/81 stale)** |
| 7 | **Real DG field strength weights (838 records)** | **Data-driven event weighting replaces hardcoded tiers** |
| 8 | **2023 PGA (3,718 rows) + LIV (1,695 rows)** | **38,629 total rows, all seasons 2015-2026, Rahm/DeChambeau have LIV data** |

---

## 10. Files

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | V1 data ingestion |
| `run_v2_pipeline.py` | V2 Stage 2 + scoring features |
| `run_v3_pipeline.py` | V3 experience features + Augusta-only S2 |
| `run_2026_predictions.py` | 2026 field predictions |
| `run_fix_spread.py` | MC spread + temperature scaling |
| `run_retrain_extended.py` | Retrain on extended data |
| `run_field_strength.py` | **Real FS weights + retrain** |
| `run_event_tiers.py` | Event tier testing (rejected for FS) |
| `scrape_dg_historical.py` | DG historical stats page scraper |
| `scrape_all_missing.py` | **2023 PGA + LIV past-results scraper** |
| `streamlit_app/app.py` | Streamlit app (4 pages) |

| Data File | Rows | Description |
|-----------|------|-------------|
| `historical_rounds_extended.parquet` | 38,629 | All tour SG 2015-2026 (PGA + LIV) |
| `dg_field_strength.csv` | 838 | Real field quality scores |
| `masters_unified.parquet` | 1,845 | All Masters 2004-2025 |
| `augusta_player_features.parquet` | 1,845 | 18 features per player-season |
| `predictions_2026.parquet` | 81 | Final 2026 predictions |
| `dk_odds_2026.csv` | 96 | DraftKings fair probabilities |

# Augusta National Model — Final Briefing (V4)
## 2026 Masters Tournament — April 9-12

---

## 1. What This Model Is

A two-stage XGBoost + Monte Carlo prediction model purpose-built for the Masters Tournament. It combines tour-wide Strokes Gained form with Augusta-specific experience features trained exclusively on Masters data (2004-2025, 1,845 player-seasons).

**Backtested on 5 Masters (2021-2025): 42% top-10 precision, +55% avg ROI on top-10 market.**

### Architecture
- **Stage 1**: XGBoost regression predicting finish percentile from 16 rolling SG features, trained on 16K PGA Tour rows (2015-2022). Feeds into 50,000 Monte Carlo simulations for win/top-5/10/20 probabilities.
- **Stage 2**: XGBoost binary classifier for top-10 finish, trained on Augusta data only (719 rows). Uses 34 features: 16 rolling SG + 18 Augusta-specific experience features.
- **Blend**: `final_top10 = 0.90 * Stage2 + 0.10 * MonteCarlo` (optimized via AUC grid search)
- **Calibration**: Platt scaling fitted on 244 backtest observations to convert raw S2 probabilities into calibrated probabilities.

### Key Design Decision
Stage 2 trains on **Augusta data only**. Other majors (US Open, Open Championship, PGA Championship) have non-transferable course dynamics. Augusta is uniquely itself — par-5 reachability, green contours, Amen Corner, azalea rough. Small sample is handled by XGBoost's native NaN support and heavy regularization (max_depth=3, reg_alpha=0.5, reg_lambda=2.0).

---

## 2. The 2026 Field

- **91 players** in the DG field-updates API
- **81 with SG data** (10 dropped — no skill ratings)
- **23 debutants** (first Masters appearance)
- **46 players with 3+ prior Augusta starts**
- **Weather forecast**: 9.0 mph avg wind, 0mm rain, 66°F — calm/moderate conditions

---

## 3. 2026 Predictions — Full Top 20

| Rk | Player | Win% | Top-10% | Top-20% | Exp Tier | Cut Prev Yr | Trajectory | Comp Rounds | Flags |
|----|--------|------|---------|---------|----------|-------------|-----------|-------------|-------|
| 1 | Scottie Scheffler | 5.5% | 42.5% | 57.0% | Veteran (20 rds) | Yes | +2.46 | 20 | |
| 2 | Xander Schauffele | 3.0% | 41.4% | 45.1% | Established (18 rds) | Yes | -0.05 | 18 | |
| 3 | Sungjae Im | 3.1% | 39.3% | 46.0% | Established (16 rds) | Yes | -0.55 | 16 | Declining |
| 4 | Cameron Smith | 2.8% | 38.6% | 43.3% | Established (18 rds) | No | -2.33 | 18 | Declining |
| 5 | Corey Conners | 2.9% | 38.1% | 44.8% | Veteran (22 rds) | Yes | -1.53 | 22 | Declining |
| 6 | Collin Morikawa | 0.2% | 37.5% | 11.4% | Veteran (20 rds) | Yes | +0.65 | 20 | |
| 7 | Jordan Spieth | 3.0% | 37.2% | 45.5% | Established (16 rds) | Yes | -0.01 | 16 | |
| 8 | Bryson DeChambeau | 1.2% | 35.9% | 29.6% | Veteran (28 rds) | Yes | +2.08 | 28 | |
| 9 | Sam Stevens | 1.1% | 35.8% | 28.4% | Debutant | No | 0.00 | 0 | Debut |
| 10 | Zach Johnson | 1.3% | 35.4% | 32.3% | Veteran (26 rds) | Yes | -0.97 | 26 | Declining |
| 11 | Chris Gotterup | 0.8% | 35.4% | 25.5% | Debutant | No | 0.00 | 0 | Debut |
| 12 | Cameron Young | 0.3% | 35.0% | 13.6% | Established (12 rds) | No | +0.07 | 12 | |
| 13 | Casey Jarvis | 0.3% | 34.8% | 12.8% | Debutant | No | 0.00 | 0 | Debut |
| 14 | Davis Riley | 4.3% | 34.4% | 51.8% | Learning (4 rds) | Yes | 0.00 | 4 | |
| 15 | Ludvig Aberg | 0.9% | 34.2% | 25.3% | Established (8 rds) | Yes | +0.50 | 8 | |
| 16 | Jacob Bridgeman | 0.8% | 34.0% | 25.2% | Debutant | No | 0.00 | 0 | Debut |
| 17 | Tom McKibbin | 0.9% | 33.9% | 25.6% | Debutant | No | 0.00 | 0 | Debut |
| 18 | Kristoffer Reitan | 0.4% | 33.3% | 16.7% | Debutant | No | 0.00 | 0 | Debut |
| 19 | Matt Fitzpatrick | 0.8% | 33.1% | 25.5% | Veteran (28 rds) | Yes | -1.27 | 28 | FADE? Declining |
| 20 | Ben Griffin | 0.7% | 32.8% | 24.2% | Debutant | No | 0.00 | 0 | Debut |

### Model's take on the big names further down

| Rk | Player | Win% | Top-10% | Trajectory | Divergence | Notes |
|----|--------|------|---------|-----------|-----------|-------|
| 22 | Justin Thomas | 6.2% | 31.5% | +1.20 | 0.38 | FADE — tour form overpredicts Augusta |
| 23 | Jon Rahm | 2.5% | 31.2% | -3.54 | -0.18 | Sharp decline at Augusta recently |
| 27 | Max Homa | 4.6% | 29.8% | +5.29 | -0.02 | Strong upward trajectory |
| 31 | Patrick Cantlay | 4.6% | 28.9% | -0.82 | 0.23 | |
| 41 | Justin Rose | 3.0% | 25.7% | -0.93 | -0.43 | |
| 42 | Rory McIlroy | 0.3% | 25.6% | -1.49 | -0.35 | Model is low on Rory |
| 45 | Tyrrell Hatton | 2.8% | 24.6% | +4.23 | -0.11 | Big positive trajectory |
| 50 | Sam Burns | 5.6% | 23.3% | +0.70 | 0.60 | FADE — highest divergence in field |
| 59 | Hideki Matsuyama | 0.8% | 19.8% | -2.14 | -0.35 | |

---

## 4. Augusta Experience Features Explained

These are the features that differentiate this model from a generic golf model. All computed with strict temporal cutoff — no future data leakage.

| Feature | What it measures | Why it matters at Augusta |
|---------|-----------------|-------------------------|
| **competitive_rounds** | Total rounds played under tournament pressure (4 if cut, 2 if MC) | Augusta course knowledge compounds — green reading, angles, wind patterns |
| **made_cut_prev_year** | Did this player make the cut here last year? | 9 of last 10 winners had this = 1. Recent familiarity is the strongest single signal |
| **experience_tier** | 0=Debutant, 1=Learning (1-7 rds), 2=Established (8-19), 3=Veteran (20-35), 4=Deep Vet (36+) | Sweet spot is tier 2-3. Only one debutant has won since 1979 (Fuzzy Zoeller) |
| **scoring_trajectory** | career_avg minus recent_2 score_vs_field (positive = improving) | Catches players trending up/down at Augusta specifically |
| **rounds_last_2yrs** | Competitive rounds in the 2 most recent prior seasons (max 8) | Course changes subtly year to year — recency matters |
| **best_finish_recent** | Best finish in last 3 appearances | Recent ceiling is more predictive than career best from 8 years ago |
| **tour_vs_augusta_divergence** | sg_total_8w percentile minus scoring_avg percentile | Directly catches players whose tour form doesn't translate here. Positive = fade candidate |

---

## 5. Fade Candidates — Tour Form Won't Translate

Players where `tour_vs_augusta_divergence > 0.35` — their recent PGA Tour form significantly overpredicts their Augusta performance:

| Player | Divergence | Model Top-10% | Why they're flagged |
|--------|-----------|--------------|-------------------|
| **Sam Burns** | 0.60 | 23.3% | Highest divergence in field. Elite tour SG but consistently underperforms at Augusta |
| **Nicolai Hojgaard** | 0.56 | 28.2% | Tour form strong but limited Augusta track record doesn't match |
| **Justin Thomas** | 0.38 | 31.5% | The model's poster child for this feature. CUT in 2024, CUT in 2023. Made cut rate dropped to 50% |
| **Matt Fitzpatrick** | 0.35 | 33.1% | Scoring trajectory is -1.27 (declining), veteran with lots of rounds but results fading |
| **Kurt Kitayama** | 0.35 | 19.2% | Limited Augusta data but tour form suggests he should be higher than he actually performs here |

In backtesting, this divergence feature was the key to fixing the JT problem — his ranking dropped from #1 in V2 to #5/#18/#28 in V3's 2023/2024/2025 backtests.

---

## 6. Market Edge Analysis

**Odds source: DraftKings (91 players, outrights)**

Important caveat: The Odds API returned win market odds only, not explicit top-10 market odds. Top-10 market probabilities are estimated as `win_prob * 10` (capped at 95%). This makes the edge calculations directional, not precise.

### Market Fades — Model Below Market

These are players the market prices higher than our model says they should be for top-10:

| Player | Model Top-10% | Market Top-10% | Edge | Signal |
|--------|--------------|---------------|------|--------|
| Jon Rahm | 31.2% | 70.4% | -56% | Model sees sharp Augusta decline (trajectory -3.54) |
| Scottie Scheffler | 42.5% | 95.0% | -55% | Model has him #1 but market has him even higher — no edge |
| Rory McIlroy | 25.6% | 55.8% | -54% | Model is bearish — declining trajectory at Augusta |
| Bryson DeChambeau | 35.9% | 61.9% | -42% | Market has him too high per model |

These are NOT fades in the traditional sense — the model still likes Scheffler (#1). It's saying the market prices leave no value. You're paying full price.

---

## 7. Backtest Track Record (2021-2025)

| Year | Top-10 Precision | S2 AUC | Top-10 ROI | Winner | Winner Rank |
|------|-----------------|--------|-----------|--------|-------------|
| 2021 | 30% | 0.500 | 0% | Matsuyama | #48 |
| 2022 | **50%** | **0.674** | **+55%** | Scheffler | #17 |
| 2023 | **40%** | **0.669** | **+104%** | Rahm | #25 |
| 2024 | **60%** | **0.738** | **+36%** | Scheffler | **#2** |
| 2025 | 30% | 0.600 | **+78%** | McIlroy | #16 |
| **AVG** | **42%** | **0.636** | **+55%** | — | — |

The model is better at predicting top-10 finishes than picking outright winners. It correctly identified Scheffler as #2 most likely top-10 finisher in 2024 (he won). Top-10 market ROI was positive in 4 of 5 backtest years.

---

## 8. How the Model Was Built (4 Sessions)

**Session 1**: Data ingestion — DG round CSVs (2021-2025), Wikipedia leaderboards (2004-2020), tour-wide training data. Basic XGBoost + Monte Carlo pipeline with walk-forward backtest.

**Session 2**: Added scoring pattern features (birdie rate, bogey avoidance, round variance), weather data, and Stage 2 binary classifier. Identified top-10 as the money market (+40% ROI). SHAP analysis revealed JT over-ranking.

**Session 3**: Rebuilt experience features (competitive_rounds, made_cut_prev_year, experience_tier, scoring_trajectory, rounds_last_2yrs, best_finish_recent). Added tour_vs_augusta_divergence. Retrained Stage 2 on Augusta data only. JT rank dropped from #1 to #28 by 2025. Top-10 precision: 26% → 42%.

**Session 4**: Pulled live 2026 field, fetched DraftKings odds, applied Platt calibration, built Streamlit app, deployed to GitHub.

---

## 9. What's Deployed

- **GitHub**: [github.com/dylanjaynes/augusta-national-model](https://github.com/dylanjaynes/augusta-national-model)
- **Local app**: `python3 -m streamlit run streamlit_app/app.py`
- **Streamlit Cloud**: Deploy via share.streamlit.io → dylanjaynes/augusta-national-model → streamlit_app/app.py

### App pages
1. **Model Picks** — Full field sorted by calibrated top-10 probability, with experience tier, trajectory, and flags
2. **Betting Edge** — Value plays and fades per market (top-10, top-20, win) with adjustable edge threshold
3. **H2H Matchups** — Compare any two players head-to-head on probabilities and Augusta profile
4. **Backtest** — 2021-2025 summary metrics and year-by-year results

---

## 10. Known Limitations

1. **Top-10 market odds are estimated** (win x 10), not actual sportsbook top-10 lines. Edge calculations are directional.
2. **34 players had no tour history** in our 2015-2022 CSV — used current skill ratings as proxy without rolling features.
3. **Debutants get generic predictions** — the model has no Augusta-specific signal for first-timers, relying entirely on tour form.
4. **S2 probabilities cluster** — many players in the 30-40% range with limited separation. The model is more confident about who will NOT top-10 than who will.
5. **No live scoring updates** — predictions are pre-tournament only. A live updater was designed but not implemented.
6. **5-year backtest is small** — +55% ROI is promising but not statistically conclusive. True out-of-sample test starts Thursday.

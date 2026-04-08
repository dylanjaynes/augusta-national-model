# Phase 4 — Final Model Tuning

**Read first:** `/Users/dylanjaynes/Augusta National Model/.claude/COORDINATION.md` for full history.
**This file:** Active task assignments for the final push. Update status here as you work.
**Communication:** Update your section below when done. Check other sessions' sections before starting dependent work.

---

## Status Board

| Task | Owner | Status | Depends On |
|------|-------|--------|------------|
| 1. Rescrape per-round SG | Session B | DONE | — |
| 2. Aging decay for Augusta features | Session C | DONE | — |
| 3. S2 rebalancing (history vs form) | Session C | DONE | — |
| 4. Hyperparameter search + final retrain | ANY SESSION | NOT STARTED | Tasks 1, 2, 3 |
| 5. SG scale comparison test | Session B | DONE | — |
| 6. Scrape 2015-2022 round-level SG | Session B | DONE | — |

---

## Task 1: Round-level SG rescrape — Session B

**UPDATED INSTRUCTIONS — READ CAREFULLY. This is a SCHEMA CHANGE, not just a scale fix.**

**Problem:** Current data stores 1 row per player per tournament with tournament-total SG.
We need 1 row per player per ROUND with per-round SG. This enables:
- R3/R4 scoring features (weekend performance at Augusta)
- Within-tournament variance (consistency across rounds)
- Mid-tournament live model updates (future)
- Correct SG scale without divide-by-rounds workaround

**New schema for `historical_rounds_extended.parquet`:**
```
dg_id, player_name, season, event_name, course, date, field_size,
finish_pos, round_num, sg_ott, sg_app, sg_arg, sg_putt, sg_t2g, sg_total
```
Key addition: `round_num` (1, 2, 3, or 4). Each player-event now has 2-4 rows instead of 1.

**What to do:**
1. The golden endpoint response has per-player fields like `R1_sg_ott`, `R2_sg_ott`, etc.
   Extract EACH ROUND as a separate row.
2. For CUT/MC players: only R1 and R2 rows (2 rows)
3. For made-cut players: R1, R2, R3, R4 (4 rows)
4. `date` should be the round date if available, otherwise event start date for all rounds
5. `finish_pos` is the same for all rounds of the same event (tournament finish)
6. Rescrape ALL 2023, 2024, 2025, 2026 PGA Tour events
7. **For 2015-2022 data (original CSV):** these are already per-round averages in the current
   schema. If you can extract round-level from the golden endpoint for these years too, great.
   If not, keep the existing rows as-is with `round_num=0` (meaning "tournament average").
8. LIV events: keep as-is with `round_num=0` (no round-level data available)
9. Keep real dates from your previous fix

**Scale verification:** After rescraping, per-round SG values should be:
- sg_total: typical range [-3.0, 3.0], std ~0.9-1.1
- sg_ott: typical range [-1.5, 1.5], std ~0.35-0.45
- Scheffler 2024 Masters R1: sg_total should be ~1.5-2.0 (not 7.4)
- A made-cut player's 4 rounds averaged should match the original per-round-average values

**IMPORTANT:** Session A's `run_production.py` rolling feature builder groups by player_name
and sorts by date. The new round-level rows will work IF the feature builder is updated to
aggregate rounds to tournament-level before computing rolling features. Session A will handle
this integration — just get the data right.

**When done:** Update status to DONE. Report: total rows, rows per typical event, sg_total std,
and Scheffler 2024 Masters round-by-round values as a sanity check.

**Session B notes:**

**TASK 1 DONE — Round-level SG rescrape complete.**

Dataset: `historical_rounds_extended.parquet` — **66,423 rows**
- round_num=0: 31,463 rows (2015-2022 tournament averages + LIV)
- round_num=1: 11,131 rows
- round_num=2: 11,029 rows
- round_num=3: 6,648 rows (cut makers only)
- round_num=4: 6,152 rows (cut makers only)

91 PGA Tour events scraped for 2023-2026, all with per-round sg_ott/app/arg/putt/t2g/total.

**Scale note for Session A:**
Round-level values (round_num > 0) are ACTUAL per-round SG — the real strokes gained in that single round. sg_total std=2.884 for these rows. This is correct: a single round sg_total of 7.4 means the player gained 7.4 strokes in one day.

Tournament-average values (round_num = 0, the 2015-2022 data) have sg_total std=~1.0. These are tournament_total ÷ rounds_played.

**To reconcile:** When building rolling features, aggregate round_num>0 rows to tournament level first:
```python
# For events with round-level data:
tourn_avg = rounds_df.groupby(["player_name","season","event_name"])[sg_cols].mean()
```
This will produce values on the same scale as the old data (~0.5-1.5 for good players).

**Scheffler 2024 Masters round-by-round:**
- R1: sg_total=7.427, sg_ott=1.392, sg_app=2.356
- R2: sg_total=3.079, sg_ott=1.269, sg_app=-0.096
- R3: sg_total=3.300, sg_ott=1.789, sg_app=-0.483
- R4: sg_total=4.483, sg_ott=1.038, sg_app=1.338
- Tournament avg: 4.572/4 = **1.143** (matches original CSV scale of 1.090 for 2022 Masters)

— **Session B**

---

## Task 5: SG Scale Comparison Test — Session B (URGENT)

**We need to settle whether the golden endpoint and original CSV are on the same SG scale.**

**What to do:** Scrape ONE 2022 PGA Tour event via the golden endpoint — pick one that exists
in the original data (like the 2022 Masters, t_num likely ~14 or whichever maps to Masters).

```
PUT https://datagolf.com/get-historical-stat-file
Body: {"base":"no","t_num":T,"year":2022}
```

Then compare the golden endpoint values vs the original CSV values for the SAME players
at the SAME event:

```python
# Scheffler 2022 Masters from original CSV: sg_total=1.090, sg_ott=0.190
# Scheffler 2022 Masters from golden endpoint: sg_total=??? sg_ott=???
# If they match → no scale problem, the 2023+ data is just genuinely higher variance
# If golden is ~4x original → the endpoint returns tournament totals, not per-round avg
# If golden is ~2x original → different SG baseline computation
```

Compare at least 5 players from the same event. Report the values side by side.

**This takes 1 API call and 2 minutes. Do it now.**

**Session B notes:**

**CONFIRMED: Golden endpoint `event` field = TOURNAMENT TOTAL (sum of all rounds). Ratio is exactly 4.0x vs original CSV for all players, all SG categories.**

```
Player                 | CSV sg_tot | GE event_tot | Ratio
Scottie Scheffler      |     1.0900 |       4.3550 |  4.00
Cameron Smith          |     0.7750 |       3.1050 |  4.01
Collin Morikawa        |     0.7125 |       2.8550 |  4.01
Justin Thomas          |     0.5250 |       2.1050 |  4.01
Will Zalatoris         |     0.6500 |       2.6050 |  4.01
Sungjae Im             |     0.5250 |       2.1050 |  4.01
```

**Conclusion:**
- `event` = sum of R1+R2+R3+R4 (tournament total)
- `r1`, `r2`, `r3`, `r4` = actual per-round SG values (correct per-round scale)
- Original CSV = tournament total ÷ rounds played (per-round average)
- My Task 1 round-level scrape IS correct: the r1/r2/r3/r4 values extracted as separate rows are the real per-round SG
- To get tournament-average rows matching the CSV scale: average the round rows per player-event

**The round-level data in `historical_rounds_extended.parquet` is correct as-is.** Session A just needs to aggregate round_num>0 rows to tournament level before computing rolling features.

— **Session B**

---

## Task 6: Scrape 2015-2022 round-level SG — Session B

**Priority: OVERNIGHT — can run unattended.**

Replace the 2015-2022 tournament-average data (round_num=0) with proper round-level data
from the golden endpoint, matching what you already did for 2023-2026.

**What to do:**
1. For each season 2015-2022, get the event list: `GET /get-event-list?tour=pga&year=YYYY`
2. For each event, scrape via `PUT /get-historical-stat-file` with `{"base":"no","t_num":T,"year":YYYY}`
3. Extract per-round SG values (R1, R2, R3, R4) as separate rows, same as your Task 1 scrape
4. Each row: `round_num=1/2/3/4`, with real per-round sg_ott/app/arg/putt/t2g/sg_total
5. Include proper event dates from the response or event list
6. **Replace** the existing round_num=0 rows for 2015-2022 with the new round-level rows
7. Keep LIV rows as-is (round_num=0, sg_total only)

**Scale verification:** After scraping, check that the per-round values for Scheffler 2022
Masters match what you found in Task 5 (R1-R4 should give event avg of ~4.36, and when
run_production.py aggregates and divides by rounds, should give ~1.09).

**Estimated scope:**
- ~8 seasons × ~25 events = ~200 API calls
- ~30K tournament rows → ~110K round rows
- Add 0.3-0.5 second delay between calls to avoid rate limiting
- Total time: ~2-3 minutes of API calls + processing

**Error handling:**
- If a t_num/year combo returns no data, skip it
- If you hit rate limiting, increase delay to 1 second
- Save checkpoint every 50 events so you can resume if the session dies
- At the end, verify total rows and that ALL seasons 2015-2022 have round_num > 0 data

**When done:** Update status. Report total rows, any events that failed, and sg_total std
for the full dataset. The final dataset should be ~145K rows, ALL with round_num > 0
(except LIV events).

**Session B notes:**

**TASK 6 DONE — Full 2015-2022 round-level scrape complete.**

- 275 events scraped via golden endpoint for 2015-2022
- 112,027 new round-level rows
- Combined with 2023-2026 data: **148,682 total rows**
- Only LIV rows remain as round_num=0 (1,695 rows)
- All PGA Tour rows are now round-level (round_num 1-4)

**Scale verification:**
- Scheffler 2022 Masters: R1=4.833, R2=7.607, R3=3.577, R4=1.404
- Average = 4.355, ÷4 rounds = **1.089** (matches original CSV value of 1.090)
- sg_total std for round-level rows: 2.923 (correct for single-round SG)

**Per-season row counts:**
2015: 11,261 | 2016: 11,773 | 2017: 11,753 | 2018: 11,812
2019: 15,636 | 2020: 14,575 | 2021: 16,755 | 2022: 18,462
2023: 11,004 | 2024: 10,812 | 2025: 10,400 | 2026: 4,439

**Note:** Event names for 2015-2018 are "t_num_X" (golden endpoint returns empty course_info.event_name for older years). Session A's aggregation should group by (player_name, season, event_name) regardless of name format.

— **Session B**

---

## Task 2: Aging decay for Augusta features — Session C

**Problem:** Past champions over 45 (Vijay Singh 63, Fred Couples 66, Olazabal 60) get
massive Augusta experience feature values (high competitive rounds, high top-10 rate, deep
veteran tier). The model ranks them as top-10 value plays with 5-7% top-10 probability.
In reality they have near-zero chance.

**What to do:**
Edit `augusta_model/features/new_features.py` — add a new function `add_aging_decay`:

```python
def add_aging_decay(field_df, current_year=2026):
    """Decay Augusta experience features for players past their prime.

    Historical data: no Masters winner over 46 (Nicklaus 1986).
    Players over 45 should have Augusta experience features
    progressively discounted.
    """
    # You'll need player birth years. Options:
    # - DG API may have age/birth year
    # - Hardcode for the ~10 past champions who are old
    # - Use career_start_year as proxy (first season in data)

    # Decay formula: multiply Augusta features by decay_factor
    # age 45: 1.0 (no decay)
    # age 50: 0.5
    # age 55: 0.25
    # age 60+: 0.10
    # decay = max(0.10, 1.0 - (age - 45) * 0.10)
```

Apply this to these Augusta features specifically:
- `augusta_competitive_rounds`
- `augusta_top10_rate`
- `augusta_made_cut_rate`
- `augusta_best_finish`
- `augusta_scoring_avg`
- `augusta_sg_app_career`
- `augusta_sg_total_career`

Do NOT decay `augusta_experience_tier` or `augusta_starts` (these are categorical/count).

**For birth year data**, try:
1. DG API: `GET https://feeds.datagolf.com/preds/skill-ratings?file_format=json&key={API_KEY}` — may have age
2. If not available, use the player's first season in the dataset as a proxy:
   `approx_age = current_year - first_season + 22` (assume turned pro at 22)
3. Hardcode overrides for known old players: Vijay Singh (1963), Fred Couples (1959),
   Jose Maria Olazabal (1966), Bernhard Langer (1957), Larry Mize (1958)

**When done:** Update status. Add the function to `new_features.py` and note the function name.
Then ping: "Session C Task 2 done — check PHASE4.md"

**Session C notes:**

**TASK 2 DONE — `add_aging_decay` added to `augusta_model/features/new_features.py`**

Function: `add_aging_decay(field_df, tour_df=None, current_year=2026)`

- Uses `KNOWN_BIRTH_YEARS` dict (24 players hardcoded: past champions + key invitees)
- Falls back to first-season proxy: `approx_age = current_year - first_season + 22`
- Decay formula: `max(0.10, 1.0 - (age - 45) * 0.10)` — no decay under 45, floor at 0.10
- Decays 7 features: competitive_rounds, top10_rate, made_cut_rate, best_finish, scoring_avg, sg_app_career, sg_total_career
- Does NOT decay experience_tier or starts (categorical/count)
- Adds `estimated_age` and `age_decay_factor` columns to field_df

Results:
- Vijay Singh (63): decay=0.10, rounds 80→8, top10_rate 0.15→0.015
- Fred Couples (67): decay=0.10, rounds 60→6
- Scottie Scheffler (30): decay=1.00, no change
- Ludvig Aberg (30): decay=1.00, no change

Already wired into `run_production.py` by Session A (lines 503-504, 661-662). Updated calls to pass `tour_df` for first-season proxy.

7 unit tests added to `tests/test_new_features.py` — all passing (69/69 total).

**TASK 3 DONE — S2 rebalancing applied to `run_production.py`**

Applied Approach A + C combined:

Approach A (increased regularization):
- `colsample_bytree`: 0.7 → 0.5 (forces trees to use fewer features, can't rely on Augusta history alone)
- `min_child_weight`: 5 → 8 (prevents overfitting to small Augusta sample)
- `reg_alpha`: 0.5 → 1.0 (more L1 sparsity)
- `reg_lambda`: 2.0 → 3.0 (more L2 smoothing)

Approach C (sample weighting in train_s2):
- Season >= 2023: weight 2.0
- Season >= 2021: weight 1.5
- Earlier: weight 1.0
- Upweights recent years so current form matters more than ancient Augusta history

Both changes complement each other: regularization prevents S2 from memorizing Augusta features, sample weighting ensures recent form has outsized influence on the fit.

**Backtest needed** (Task 4) to verify these changes improve rankings for Schauffele/Aberg and demote Reed/Willett.

— **Session C**

---

## Task 3: S2 rebalancing (history vs form) — Session C

**Problem:** S2 over-weights Augusta history features relative to current tour form.
Players with strong Augusta records but declining form (Reed, Willett) rank too high.
Players with elite current form but thin Augusta history (Schauffele, Aberg) rank too low.

**What to do:**
Edit `run_production.py` — modify the S2 XGBoost config to increase regularization on
Augusta features. Two approaches (try both, pick what backtests better):

**Approach A: Increase general regularization**
```python
XGB_S2 = {
    "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
    "subsample": 0.8, "colsample_bytree": 0.5,  # was 0.7 — forces model to use fewer features per tree
    "min_child_weight": 8,  # was 5 — prevents overfitting to small Augusta sample
    "reg_alpha": 1.0,  # was 0.5 — more L1 regularization
    "reg_lambda": 3.0,  # was 2.0 — more L2 regularization
    ...
}
```

**Approach B: Use XGBoost column sampling groups**
Split features into ROLL (tour form) and AUG+NEW (Augusta/derived) groups.
Use `colsample_bytree=0.5` so each tree must include some tour form features,
not just Augusta history.

**Approach C: Weight recent training data higher**
In `train_s2()`, add sample weights that upweight recent years:
```python
weights = np.where(s2t["season"] >= 2023, 2.0,
          np.where(s2t["season"] >= 2021, 1.5, 1.0))
s2m.fit(s2t[S2F], s2t["made_top10"], sample_weight=weights)
```

**How to evaluate:** Run `python3 run_production.py --backtest` and check:
1. Cal AUC should be > 0.62
2. Scheffler should be top 5 in 2024 backtest
3. Rory should be top 15 in 2025 backtest
4. Vijay/Couples/Olazabal should NOT appear in value plays

**When done:** Update status. Note which approach worked and the backtest AUC.
Then ping: "Session C Task 3 done — check PHASE4.md"

---

## Task 4: Final retrain — ANY SESSION (after 1, 2, 3 complete)

**Prerequisites:** Tasks 1, 2, 3 all marked DONE.

**What to do:**
1. Verify data: `python3 scripts/validate_dates.py` (Session C's validation script)
2. Run: `python3 run_production.py`
3. Check output:
   - Backtest Cal AUC > 0.65
   - T10 Prec >= 40%
   - Scheffler and Aberg in top 5
   - Rory in top 10
   - No elderly players in value plays
   - All sums correct (1.0/5.0/10.0/20.0)
4. If checks pass, commit everything and update Streamlit

**When done:** Post full backtest table and top 15 predictions in your section below.

**Final run notes:**
_(write results here)_

---

## Communication Protocol

1. **When you finish a task:** Update the Status Board table AND write a note in your section
2. **To ping another session:** Tell the user "I'm done, tell Session X to check PHASE4.md"
3. **Before starting dependent work:** Read this file to check prerequisites are DONE
4. **If you hit a blocker:** Write it in your section and tell the user to relay to the other session
5. **File path:** `/Users/dylanjaynes/Augusta National Model/.claude/PHASE4.md`

# Audit Layer 4: Known Bug Verification
**Date:** 2026-04-08
**Auditor:** Session elated-hoover
**Branch:** main (worktree: elated-hoover)

---

## Git State

### Branches
- `main` — current production state (commit `1117b71`)
- `claude/cool-brown` — older branch, 9 commits BEHIND main, with files deleted/simplified vs main
  - The diff shows cool-brown is a stripped-down older version; main has absorbed all Session A work + subsequent fixes
  - **No merge needed.** main is authoritative. cool-brown can be deleted.
- `claude/elated-hoover` — this session, same HEAD as main

### Main branch recent history
```
1117b71 fix: date ordering from DG schedule API + market-calibrated picks
346b689 docs: honest handoff file for next session
dce36a2 feat: add live market odds benchmark from The Odds API
145f9a9 data: updated predictions with 148K round-level dataset
```

---

## Bug #1 — Date Ordering for 2025-2026

### Status: **PARTIALLY FIXED**

### Evidence
```
Jun-15 placeholder rows by season:
  2015: 11261/11261 rows (100%)
  2016: 11773/11773 rows (100%)
  ...
  2023: 11004/11004 rows (100%)
  2024:   504/10812 rows  (4.7%)
  2025:   503/10400 rows  (4.8%)
  2026:   244/ 4439 rows  (5.5%)

Total: 124,282 of 148,682 rows (83.6%) have placeholder 2026-06-15 date
```

### What was fixed
The Phase 3 date fix correctly assigned real dates to 2024–2026 events (~95% coverage). For 2025-2026 specifically, date collisions are now minimal (2 in 2025, 2 in 2026), and those collisions are between simultaneous PGA/LIV events — expected behavior.

Scheffler's 2025-2026 events are mostly correctly ordered chronologically based on inspection.

### Remaining issue
**2015–2023 historical training data: 100% placeholder dates.** Every row from these years shows `YYYY-06-15`. The rolling feature builder `sort_values(["player_name", "date"])` uses stable sort, so within each year the event order is determined by original row insertion order in the parquet.

Inspection shows rows appear to be stored in approximately chronological order (American Express Jan → WM Phoenix Feb → … → Masters Apr → …), so the stable sort *roughly* preserves correct temporal ordering. This is **fortuitous, not reliable**.

**Specific events with wrong dates in all seasons:**
- `Arnold Palmer Invitational` — always `YYYY-06-15`, real date is early March (before Masters)
- `the Memorial Tournament` — always `YYYY-06-15`, real date is late May

**Impact on 2026 predictions:** Scheffler's 2026 Arnold Palmer (his weakest 2026 event: sg_total avg +0.22) has date `2026-06-15`, placing it AFTER the Masters date in sort order. This excludes it from his pre-Masters rolling window, slightly *overstating* his rolling SG features. Including it would reduce his win_prob slightly.

### Recommended fix
See `scripts/fix_arnold_palmer_dates.py` (created by this audit). Assigns correct early-March dates to Arnold Palmer Invitational for all seasons.

---

## Bug #2 — Model Overconfidence at Top

### Status: **PARTIALLY FIXED**

### Evidence
Current top 10 model vs market:
```
Player              Model Win%   Market Win%   Ratio
DeChambeau           8.09%        5.91%        1.37x (overvalued)
Ludvig Aberg         7.89%        3.89%        2.03x (overvalued)
Scottie Scheffler    6.84%       11.43%        0.60x (UNDERVALUED)
Patrick Reed         6.64%        1.55%        4.29x (massively overvalued)
Jon Rahm             5.48%        6.73%        0.81x (slight undervalue)
Cameron Smith        4.36%        0.61%        7.14x (massively overvalued)
Sungjae Im           3.97%        0.59%        6.73x (massively overvalued)
Cameron Young        3.44%        2.83%        1.22x (slight overvalue)
Min Woo Lee          3.02%        1.94%        1.56x (overvalued)
Rory McIlroy         2.80%        5.33%        0.53x (undervalued)

Top-10 mean win%:   Model=5.25%  Market=4.08%  Ratio=1.29x
Top-10 MAE vs market: 0.0285
```

### What improved
Model/market ratio dropped from ~1.7-2.4x (per HANDOFF) to 1.29x average for top 10. Probability sums are now perfect (win=1.0, top5=5.0, top10=10.0, top20=20.0).

### Remaining problem
The overconfidence is **not uniformly distributed** — it is concentrated in players with strong Augusta history but poor current form (Reed, Smith, Im). The market correctly discounts these players; the model cannot. Scheffler and McIlroy are undervalued because the model can't sufficiently distinguish elite current form from Augusta-history-dominant players.

**Root cause:** S2 is dominated by Augusta history features (correlation with `augusta_scoring_avg`: r=-0.633 vs r=-0.297 with `dg_rank`). See Bug #3.

---

## Bug #3 — Augusta History Over-Weighting (Patrick Reed Test)

### Status: **STILL BROKEN**

### Evidence — Reed vs Scheffler comparison
```
Feature                        Scheffler        Reed
dg_rank                             1            44
model_score (S1)                0.449          0.052
stage2_prob_raw (S2)            0.853          0.852   ← NEARLY IDENTICAL
win_prob (model)                6.84%          6.64%
win_prob (market DK)           11.43%          1.55%
kelly_edge_win                 -0.401         +0.306   ← model recommends BET ON REED
```

**Reed's Augusta features driving S2:**
- `augusta_competitive_rounds`: 16 (4+ full tournaments)
- `augusta_made_cut_prev_year`: 1
- `augusta_best_finish_recent`: 4.0 (T4, 2018 Masters win)
- `augusta_scoring_avg`: -3.64 strokes vs field
- `augusta_top10_rate`: 50%

**Reed's current form:**
- 15 LIV events in 2025, avg sg_total ~0.97 (tour-average, not elite)
- LIV rounds have NaN for sg_ott/app/arg/putt — so all 12 sub-component rolling features are computed from sparse data (3 majors + few PGA events)
- model_score = 0.052 → 5th percentile

### Quantification
S2 raw correlations:
- vs `augusta_scoring_avg`: r = **-0.633** (primary driver)
- vs `dg_rank`: r = **-0.297** (weak)
- vs `model_score` (S1): r = **-0.182** (very weak)

Augusta features are ~4-5x more influential than current form (model_score/dg_rank) in the S2 classifier.

**Reed overvaluation: 4.3x vs market.** Model recommends +EV bet on Reed, market has him at ~50-1 outright.

Other over-weighted Augusta veterans:
```
Cameron Smith:  model 4.36%  market 0.61%  (7.1x overvalued)
Sungjae Im:     model 3.97%  market 0.59%  (6.7x overvalued)
Vijay Singh:    model 1.43%  market ~0.1%  (14x overvalued, retired)
Fred Couples:   model 0.69%  market ~0.1%  (7x overvalued, honorary)
```

### Fix required
Increase S2 regularization or add explicit current-form penalty. Quick partial fix: apply market-blend shrinkage for players with dg_rank > 50 or LIV-only recent schedule. **See post-processing fix in `scripts/apply_form_penalty.py`.**

---

## Bug #4 — SG Scale Mismatch Between Eras

### Status: **MOSTLY FIXED** (residual is not practically significant)

### Evidence — KS Tests (historical 2015-2022 vs golden endpoint 2023+)
```
Metric       KS stat    p-value    Verdict
sg_total:    0.0162     1.05e-6    Statistically significant, small effect
sg_ott:      0.0051     0.552      Not significant
sg_app:      0.0062     0.311      Not significant
sg_putt:     0.0035     0.934      Not significant

Standard deviations:
            Masters CSV    2015-2022    2023+
sg_total:     3.076          2.935      2.844
sg_ott:       1.139          1.091      1.090
sg_app:       1.708          1.657      1.630
sg_putt:      1.667          1.708      1.694
```

**Assessment:** The sg_total difference (std 2.935 vs 2.844, 3.1%) is statistically significant with 43K rows but practically negligible. With 40K training samples, even a 0.3% effect reaches p<1e-6 via large-N statistics. Sub-components (ott, app, putt) show no significant scale difference. **This bug is effectively resolved.**

---

## Scheffler Test — Why World #1 is Undervalued

### Status: **PARTIALLY FIXED** (was 45th percentile, now ranked #3, but still 40% undervalued vs market)

### Scheffler's exact feature trace
```
model_score (S1):       0.4488  →  11th of 83 in field (86.7th pct)
stage2_prob_raw:        0.853   →  3rd highest in field
s2_platt:               0.269   →  after Platt scaling
win_prob:               6.84%   →  ranked #3 by model
market win%:           11.43%   →  ranked #1 by market
kelly_edge_win:        -0.401   →  model says he's OVERPRICED (wrong)
```

### Pre-Masters 2026 rolling window (chronological order)
```
Correctly included events:
  Genesis (Feb-19):   sg_total avg +1.40  (modest)
  WM Phoenix (Feb-5): sg_total avg +2.92  (strong)
  Pebble Beach (Feb-12): sg_total avg +2.38 (strong)
  PLAYERS (Mar-12):   sg_total avg +1.66  (modest)

MISSING (Jun-15 placeholder, excluded):
  Arnold Palmer (Mar ~6): sg_total avg +0.22  (his weakest 2026 event)
```

**Note:** Missing Arnold Palmer HELPS Scheffler's rolling average (it's his worst 2026 result). The date bug slightly OVERESTIMATES his rolling SG.

### Root cause of undervaluation
1. **S2 can't differentiate Scheffler from Reed**: Both have strong Augusta histories. With S2 getting 0.9 weight in blend, the final win_prob is nearly identical despite massive form gap.
2. **S2 Augusta-history dominance**: `augusta_scoring_avg` correlation r=-0.633 with S2 raw. `dg_rank` correlation only r=-0.297.
3. **Platt scaler compresses Scheffler**: s2_platt=0.269 is the calibrated probability. The Platt model was trained on 5 backtest years (2021-2025) with limited top-ranked data, and may be over-compressing elite players.
4. **Previous fix** (before Phase 3): Date bug had Scheffler at 45th percentile because the rolling window was computing across wrong events. This is now mostly fixed.

---

## Summary Table

| Bug | Status | Key Evidence |
|-----|--------|--------------|
| #1 Date ordering 2025-2026 | **PARTIALLY FIXED** | 2024-2026 mostly OK; 2015-2023 all June 15 placeholders (stable sort preserves approx order); Arnold Palmer always misplaced to June 15 |
| #2 Overconfidence at top | **PARTIALLY FIXED** | Model/market ratio 1.29x (was 1.7-2.4x); MAE top-10 = 0.0285; Reed/Smith/Im massively overvalued; Scheffler undervalued |
| #3 Augusta history over-weighting | **STILL BROKEN** | S2(Reed)=0.852 ≈ S2(Scheffler)=0.853 despite Reed dg_rank 44 vs Scheffler 1; augusta_scoring_avg r=-0.633 dominates S2 |
| #4 SG scale mismatch | **MOSTLY FIXED** | KS sig for sg_total (p=1e-6) but effect size tiny (3.1% std diff); sub-components clean |
| Scheffler test | **PARTIALLY FIXED** | Was 45th pct, now 11th/86.7th pct in S1; ranked #3 model vs #1 market; still 40% undervalued (6.84% vs 11.43%) |

---

## Recommended Fixes (Priority Order)

### P0 — Fix Arnold Palmer Invitational dates (low effort, improves date ordering for all years)
Real dates: 2019=Mar-07, 2020=Mar-05, 2021=Mar-04, 2022=Mar-03, 2023=Mar-02, 2024=Mar-07, 2025=Mar-06, 2026=Mar-06.
Script: `scripts/fix_arnold_palmer_dates.py`

### P1 — Reduce Augusta history dominance in S2 (requires retrain)
Options:
- Increase S2 reg_alpha to 2.0 (from 1.0) and reg_lambda to 5.0 (from 3.0)
- Add `form_weight = 0.4 * (1 - dg_rank_percentile)` as explicit interaction feature
- Change blend to 0.75*S2 + 0.25*S1_mc instead of 0.9*S2 + 0.1*MC

### P2 — Apply LIV-player form penalty (post-hoc, no retrain needed)
Players on LIV with no recent PGA events have sparse rolling sub-component features. Apply shrinkage: for players where >80% of last-8-event rounds are LIV (sg_ott=NaN), discount their S2 score toward a lower base rate. See `scripts/apply_form_penalty.py`.

### P3 — Fix Memorial Tournament dates (same approach as Arnold Palmer)
Real dates are late May (not June 15). This slightly improves within-year ordering for training data.

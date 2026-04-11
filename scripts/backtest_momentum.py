"""
Backtest: does momentum adjustment improve mid-round MC predictions?

Tests 4 weight configurations at mid-round snapshots (holes 6, 9, 12, 15):
  none         — no momentum (pure skill-based baseline)
  conservative — 0.15 / 0.05 / 0.02
  moderate     — 0.25 / 0.10 / 0.03  (current production)
  aggressive   — 0.40 / 0.20 / 0.08

Metrics vs actual final tournament results:
  Spearman     — rank correlation (predicted rank vs actual finish)
  Top-10 Prec  — % of model's top-10 picks who actually finished top-10
  Brier        — MSE of top-10 probability (lower = better calibrated)

Data: masters_hole_by_hole.parquet (2015-2025, 11 years × up to 4 rounds × 72 holes)

Usage:
    python3 scripts/backtest_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ── Constants (match production values) ──────────────────────────────────────
ROUND_MEAN  = 0.50   # Augusta historical avg score vs par per round
ROUND_STD   = 3.05   # per-round standard deviation
FIELD_CORR  = 0.28   # field-wide correlation (weather/pins)
N_SIMS      = 5_000  # sims per snapshot — fast but stable
SEED        = 42

SNAPSHOTS   = [6, 9, 12, 15, 18]   # holes completed in current round

WEIGHT_CONFIGS = {
    "none":         (0.00, 0.00, 0.00),
    "conservative": (0.15, 0.05, 0.02),
    "moderate":     (0.25, 0.10, 0.03),   # current production
    "aggressive":   (0.40, 0.20, 0.08),
}

ROUND_PARAMS = {
    1: {"mean": 0.25, "std": 2.83},
    2: {"mean": 0.27, "std": 2.74},
    3: {"mean": 0.83, "std": 3.25},
    4: {"mean": 0.01, "std": 3.01},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_finish(pos: str) -> int | None:
    """'TT12' → 12, '1' → 1, '-' / MC → None."""
    if pd.isna(pos):
        return None
    s = str(pos).strip().lstrip("T").upper()
    if s in ("-", "", "MC", "CUT", "WD", "DQ", "T-"):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def momentum_adjust(
    baseline: float,
    current_pace: float | None,
    thru: float,
    weight: float,
) -> float:
    """
    Single-tier momentum blend.
    weight is already the full tier weight (e.g. 0.25 for current_rest at full fraction).
    """
    if current_pace is None or thru < 3:
        return baseline
    fraction = min(thru / 18.0, 1.0)
    w = weight * fraction
    return baseline * (1.0 - w) + current_pace * w


def simulate_field(
    player_names:   list[str],
    current_scores: np.ndarray,      # cumulative score to par
    baselines:      np.ndarray,      # expected score per round (skill estimate)
    current_round:  int,
    thru:           int,             # holes completed in current round
    today_pace:     np.ndarray | None,  # projected 18-hole score for today (or None)
    weight_config:  tuple[float, float, float],
    n_sims:         int = N_SIMS,
    seed:           int = SEED,
) -> dict[str, dict]:
    """
    MC simulation from mid-round snapshot.
    Returns {player_name: {win_prob, top10_prob, projected_median}}.
    """
    rng = np.random.default_rng(seed)
    n = len(player_names)
    w_rest, w_next, w_later = weight_config

    # ── Build momentum-adjusted expected scores per segment ──────────────────
    exp_rest  = np.zeros(n)
    exp_next  = np.zeros(n)
    exp_later = np.zeros(n)

    for i in range(n):
        pace = today_pace[i] if today_pace is not None else None
        exp_rest[i]  = momentum_adjust(baselines[i], pace, thru, w_rest)
        exp_next[i]  = momentum_adjust(baselines[i], pace, thru, w_next)
        exp_later[i] = momentum_adjust(baselines[i], pace, thru, w_later)

    # ── Remaining rounds structure ────────────────────────────────────────────
    current_round_remaining = max(0.0, (18 - thru) / 18.0)
    full_rounds_after = max(0, 4 - current_round)
    rounds_left = current_round_remaining + full_rounds_after

    if rounds_left <= 0:
        final_scores = np.tile(current_scores, (n_sims, 1))
    else:
        # ── Expected remaining score per player ───────────────────────────────
        remaining_means = np.zeros(n)
        rounds_accounted = 0.0

        for r_offset in range(5):
            if rounds_accounted >= rounds_left - 1e-9:
                break
            abs_round = current_round + r_offset
            if abs_round > 4:
                break

            rp = ROUND_PARAMS.get(abs_round, {"mean": ROUND_MEAN, "std": ROUND_STD})
            round_mean_shift = rp["mean"] - ROUND_MEAN

            if r_offset == 0:
                frac = current_round_remaining
                exp_seg = exp_rest
            elif r_offset == 1:
                frac = min(1.0, rounds_left - rounds_accounted)
                exp_seg = exp_next
            else:
                frac = min(1.0, rounds_left - rounds_accounted)
                exp_seg = exp_later

            remaining_means += exp_seg * frac + round_mean_shift * frac
            rounds_accounted += frac

        # ── Simulate remaining rounds ─────────────────────────────────────────
        field_std  = ROUND_STD * np.sqrt(FIELD_CORR * rounds_left)
        indiv_std  = ROUND_STD * np.sqrt((1 - FIELD_CORR) * rounds_left)

        field_shocks  = rng.normal(0, field_std, n_sims)                  # (n_sims,)
        indiv_scores  = rng.normal(remaining_means, indiv_std, (n_sims, n))

        remaining_total = indiv_scores + field_shocks[:, np.newaxis]
        final_scores = current_scores + remaining_total

    # ── Count outcomes ────────────────────────────────────────────────────────
    sorted_idx   = np.argsort(final_scores, axis=1)
    winners      = np.argmin(final_scores, axis=1)
    win_counts   = np.bincount(winners, minlength=n)
    top10_counts = np.zeros(n, dtype=int)
    for i in range(n):
        top10_counts[i] = int((sorted_idx[:, :10] == i).sum())

    proj_median = np.median(final_scores, axis=0)

    results = {}
    for i, name in enumerate(player_names):
        results[name] = {
            "win_prob":    win_counts[i]   / n_sims,
            "top10_prob":  top10_counts[i] / n_sims,
            "proj_median": float(proj_median[i]),
        }
    return results


# ── Build player skill baselines (no look-ahead) ─────────────────────────────

def build_skill_baselines(
    hbh: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """
    For each (player, year) pair: expected score per round at Augusta,
    estimated from all their prior Augusta rounds (years < current year).
    Returns expected score vs par per round (negative = better than field avg).
    Falls back to ROUND_MEAN for players with no prior history.
    """
    # Per-player, per-year, per-round total score
    round_totals = (
        hbh.groupby(["year", "player_name", "round"])["score_to_par"]
        .sum()
        .reset_index()
        .rename(columns={"score_to_par": "round_stp"})
    )

    baselines: dict[tuple[str, int], float] = {}
    years = sorted(hbh["year"].unique())

    for year in years:
        prior = round_totals[round_totals["year"] < year]
        players_this_year = hbh[hbh["year"] == year]["player_name"].unique()
        for player in players_this_year:
            prior_rounds = prior[prior["player_name"] == player]["round_stp"]
            if len(prior_rounds) >= 2:
                baselines[(player, year)] = float(prior_rounds.mean())
            else:
                baselines[(player, year)] = ROUND_MEAN

    return baselines


# ── Build actual tournament outcomes ─────────────────────────────────────────

def build_actual_outcomes(hbh: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (year, player): finish_num, top10, made_cut, total_score_to_par.
    """
    tourney = (
        hbh.groupby(["year", "player_name"])
        .agg(
            made_cut=("made_cut", "first"),
            finish_pos=("finish_pos", "first"),
            total_stp=("score_to_par", "sum"),
        )
        .reset_index()
    )
    tourney["finish_num"] = tourney["finish_pos"].apply(parse_finish)
    # Players who missed cut: assign 999 (last place proxy)
    tourney["finish_num"] = tourney.apply(
        lambda r: r["finish_num"] if r["finish_num"] is not None
        else (999 if not r["made_cut"] else 75),
        axis=1,
    )
    tourney["top10"] = tourney["finish_num"].apply(lambda x: int(x <= 10))
    return tourney.set_index(["year", "player_name"])


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[dict],   # [{player_name, year, round, snapshot, top10_prob, win_prob, proj_median}]
    actuals: pd.DataFrame,
) -> dict:
    """
    Compute Spearman, Top-10 precision, Brier score across all snapshots.

    Spearman is computed per (year, round, snapshot) group — each group is an
    independent field at one point in time — then averaged. Pooling all rows
    together is wrong because the same player appears in multiple snapshots.
    """
    from scipy.stats import spearmanr

    rows = []
    for pred in predictions:
        key = (pred["year"], pred["player_name"])
        if key not in actuals.index:
            continue
        actual = actuals.loc[key]
        rows.append({
            "year":         pred["year"],
            "round":        pred["round"],
            "snapshot":     pred["snapshot"],
            "top10_prob":   pred["top10_prob"],
            "win_prob":     pred["win_prob"],
            "proj_median":  pred["proj_median"],
            "actual_top10": actual["top10"],
            "actual_finish": actual["finish_num"],
            "actual_total":  actual["total_stp"],
        })

    if not rows:
        return {"spearman": float("nan"), "top10_prec": float("nan"), "brier": float("nan")}

    df = pd.DataFrame(rows)

    # Spearman: per-snapshot group, correlate win_prob rank with actual finish
    # Only among players who made the cut (finish < 999) — MC players don't rank
    spearman_vals = []
    for (y, r, snap), grp in df.groupby(["year", "round", "snapshot"]):
        grp_cut = grp[grp["actual_finish"] < 999].copy()
        if len(grp_cut) < 5:
            continue
        try:
            corr, _ = spearmanr(grp_cut["win_prob"], -grp_cut["actual_finish"])
            if not np.isnan(corr):
                spearman_vals.append(corr)
        except Exception:
            pass
    spearman = float(np.mean(spearman_vals)) if spearman_vals else float("nan")

    # Top-10 precision: per snapshot, pick top-10 by predicted prob, check actual
    # Average precision across all snapshots
    prec_vals = []
    for (y, r, snap), grp in df.groupby(["year", "round", "snapshot"]):
        n_actual_top10 = int(grp["actual_top10"].sum())
        if n_actual_top10 == 0:
            continue
        k = min(max(n_actual_top10, 10), len(grp))
        top_k = grp.nlargest(k, "top10_prob")
        prec_vals.append(float(top_k["actual_top10"].mean()))
    prec = float(np.mean(prec_vals)) if prec_vals else float("nan")

    # Brier: MSE of top10 probability (pooled — calibration metric, pooling is fine)
    brier = float(((df["top10_prob"] - df["actual_top10"]) ** 2).mean())

    return {"spearman": spearman, "top10_prec": prec, "brier": brier}


# ── Main backtest loop ────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    hbh = pd.read_parquet(ROOT / "data" / "processed" / "masters_hole_by_hole.parquet")

    print(f"  {len(hbh)} hole-level rows, years {hbh['year'].min()}-{hbh['year'].max()}")

    print("Building player baselines (no look-ahead)...")
    baselines = build_skill_baselines(hbh)

    print("Building actual tournament outcomes...")
    actuals = build_actual_outcomes(hbh)

    # Per-player per-round total scores for snapshot building
    round_totals = (
        hbh.groupby(["year", "player_name", "round"])["score_to_par"]
        .sum()
        .reset_index()
        .rename(columns={"score_to_par": "round_stp"})
    )

    # ── Collect predictions for each weight config ────────────────────────────
    # predictions[config_name] = list of prediction dicts
    all_predictions: dict[str, list[dict]] = {k: [] for k in WEIGHT_CONFIGS}

    years = sorted(hbh["year"].unique())

    total_snapshots = 0
    for year in years:
        year_hbh = hbh[hbh["year"] == year].copy()
        year_players = year_hbh["player_name"].unique().tolist()

        # Players who made the cut (only they play R3/R4)
        made_cut_players = set(
            year_hbh[year_hbh["made_cut"] == True]["player_name"].unique()
        )

        # Get all actual outcomes for this year for score normalization
        year_actuals = actuals[actuals.index.get_level_values("year") == year]

        for current_round in [1, 2, 3, 4]:
            # Only players eligible for this round
            if current_round >= 3:
                eligible = [p for p in year_players if p in made_cut_players]
            else:
                eligible = year_players

            if len(eligible) < 5:
                continue

            for snap_hole in SNAPSHOTS:
                # Build state for each player at this snapshot
                player_names = []
                current_scores = []
                today_paces = []
                player_baselines = []

                for player in eligible:
                    # Cumulative score through all prior rounds
                    prior_rounds_score = 0
                    for r in range(1, current_round):
                        rt = round_totals[
                            (round_totals["year"] == year) &
                            (round_totals["player_name"] == player) &
                            (round_totals["round"] == r)
                        ]
                        if not rt.empty:
                            prior_rounds_score += int(rt.iloc[0]["round_stp"])

                    # Current round: score through snap_hole
                    curr_round_data = year_hbh[
                        (year_hbh["player_name"] == player) &
                        (year_hbh["round"] == current_round) &
                        (year_hbh["hole_number"] <= snap_hole)
                    ].sort_values("hole_number")

                    holes_played = len(curr_round_data)
                    if holes_played < snap_hole:
                        # Player hasn't played enough holes in this round — skip
                        continue

                    today_stp = int(curr_round_data["score_to_par"].sum())
                    cumulative = prior_rounds_score + today_stp

                    # Current pace (project to full 18 holes)
                    if holes_played >= 3:
                        pace = today_stp / holes_played * 18.0
                    else:
                        pace = None

                    player_names.append(player)
                    current_scores.append(cumulative)
                    today_paces.append(pace)
                    player_baselines.append(baselines.get((player, year), ROUND_MEAN))

                if len(player_names) < 5:
                    continue

                current_scores_arr = np.array(current_scores, dtype=float)
                baselines_arr      = np.array(player_baselines, dtype=float)
                today_pace_arr     = np.array(
                    [p if p is not None else baselines_arr[i]
                     for i, p in enumerate(today_paces)],
                    dtype=float,
                )

                # Run MC for each weight config
                for config_name, weights in WEIGHT_CONFIGS.items():
                    results = simulate_field(
                        player_names   = player_names,
                        current_scores = current_scores_arr,
                        baselines      = baselines_arr,
                        current_round  = current_round,
                        thru           = snap_hole,
                        today_pace     = today_pace_arr,
                        weight_config  = weights,
                        n_sims         = N_SIMS,
                        seed           = SEED,
                    )

                    for name, probs in results.items():
                        all_predictions[config_name].append({
                            "year":       year,
                            "round":      current_round,
                            "snapshot":   snap_hole,
                            "player_name": name,
                            "top10_prob": probs["top10_prob"],
                            "win_prob":   probs["win_prob"],
                            "proj_median": probs["proj_median"],
                        })

                total_snapshots += 1

        print(f"  {year}: done ({len(made_cut_players)} cut players)")

    print(f"\nTotal snapshots processed: {total_snapshots}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")

    results_rows = []
    for config_name, preds in all_predictions.items():
        w = WEIGHT_CONFIGS[config_name]

        # Overall metrics across all rounds/snapshots
        overall = compute_metrics(preds, actuals)

        # Per-snapshot breakdown
        snap_rows = []
        for snap_hole in SNAPSHOTS:
            snap_preds = [p for p in preds if p["snapshot"] == snap_hole]
            metrics = compute_metrics(snap_preds, actuals)
            snap_rows.append({
                "config": config_name,
                "snapshot_hole": snap_hole,
                **metrics,
            })

        # Per-round breakdown (round 3 most interesting — mid-round momentum)
        round_rows = []
        for r in [1, 2, 3, 4]:
            r_preds = [p for p in preds if p["round"] == r]
            metrics = compute_metrics(r_preds, actuals)
            round_rows.append({
                "config": config_name,
                "round": r,
                **metrics,
            })

        results_rows.append({
            "config":       config_name,
            "weights":      f"{w[0]}/{w[1]}/{w[2]}",
            "n_preds":      len(preds),
            "spearman":     overall["spearman"],
            "top10_prec":   overall["top10_prec"],
            "brier":        overall["brier"],
            "snap_detail":  snap_rows,
            "round_detail": round_rows,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MOMENTUM BACKTEST RESULTS — Augusta National 2015-2025")
    print("=" * 72)
    print(f"{'Config':<14} {'Weights':>12}  {'Spearman':>10}  {'T10 Prec':>10}  {'Brier':>8}")
    print("-" * 72)

    best = {}
    for row in results_rows:
        sp = row["spearman"]
        pr = row["top10_prec"]
        br = row["brier"]
        print(
            f"{row['config']:<14} {row['weights']:>12}  "
            f"{sp:>10.4f}  {pr:>9.1%}  {br:>8.5f}"
        )
        best["spearman"]   = row if "spearman" not in best or sp > best["spearman"]["spearman"] else best["spearman"]
        best["top10_prec"] = row if "top10_prec" not in best or pr > best["top10_prec"]["top10_prec"] else best["top10_prec"]
        best["brier"]      = row if "brier" not in best or br < best["brier"]["brier"] else best["brier"]

    print("-" * 72)
    print(f"  Best Spearman:    {best['spearman']['config']}")
    print(f"  Best T10 Prec:    {best['top10_prec']['config']}")
    print(f"  Best Brier:       {best['brier']['config']}")

    # ── Per-round breakdown for R3 mid-round (most diagnostic) ───────────────
    print("\n--- Round 3 Mid-Round Breakdown (holes 6-15) ---")
    print(f"{'Config':<14} {'Hole':>5}  {'Spearman':>10}  {'T10 Prec':>10}  {'Brier':>8}")
    print("-" * 56)
    for row in results_rows:
        for sd in row["snap_detail"]:
            if sd["snapshot_hole"] in [6, 9, 12, 15]:
                # Only show R3 snaps for brevity — filter from round_detail instead
                pass
    # Show per-snapshot (all rounds combined) for cleaner comparison
    for snap_hole in [6, 9, 12, 15]:
        print(f"\n  Hole {snap_hole}:")
        for row in results_rows:
            sd = next((s for s in row["snap_detail"] if s["snapshot_hole"] == snap_hole), None)
            if sd:
                print(
                    f"    {row['config']:<12}  "
                    f"Spearman {sd['spearman']:>6.4f}  "
                    f"T10 {sd['top10_prec']:>5.1%}  "
                    f"Brier {sd['brier']:>7.5f}"
                )

    # ── Save detailed results ─────────────────────────────────────────────────
    save_rows = []
    for row in results_rows:
        for sd in row["snap_detail"]:
            save_rows.append({
                "config":       row["config"],
                "weights":      row["weights"],
                "snapshot_hole": sd["snapshot_hole"],
                "spearman":     sd["spearman"],
                "top10_prec":   sd["top10_prec"],
                "brier":        sd["brier"],
            })
        for rd in row["round_detail"]:
            save_rows.append({
                "config":       row["config"],
                "weights":      row["weights"],
                "round":        rd["round"],
                "spearman":     rd["spearman"],
                "top10_prec":   rd["top10_prec"],
                "brier":        rd["brier"],
            })

    out_df = pd.DataFrame(save_rows)
    out_path = ROOT / "data" / "processed" / "backtest_momentum_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    none_row = next(r for r in results_rows if r["config"] == "none")
    print("\n--- Delta vs No-Momentum Baseline ---")
    print(f"{'Config':<14}  {'ΔSpearman':>10}  {'ΔT10 Prec':>10}  {'ΔBrier':>9}")
    print("-" * 52)
    for row in results_rows:
        if row["config"] == "none":
            continue
        ds = row["spearman"]   - none_row["spearman"]
        dp = row["top10_prec"] - none_row["top10_prec"]
        db = row["brier"]      - none_row["brier"]   # negative = better
        print(
            f"{row['config']:<14}  "
            f"{ds:>+10.4f}  {dp:>+9.1%}  {db:>+9.5f}"
        )

    # Overall best
    overall_winners = [best["spearman"]["config"], best["top10_prec"]["config"], best["brier"]["config"]]
    from collections import Counter
    winner = Counter(overall_winners).most_common(1)[0][0]
    print(f"\n{'=' * 52}")
    print(f"  RECOMMENDED: {winner.upper()} weights")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()

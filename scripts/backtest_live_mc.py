#!/usr/bin/env python3
"""
Backtest the remaining-rounds Monte Carlo simulator against historical Masters data.

For each year 2021-2025, reconstructs leaderboard snapshots at:
  - R1 after 4, 9, 13, 18 holes
  - R2 after 4, 9, 13, 18 holes
  - R3 after 4, 9, 13, 18 holes
  - R4 after 4, 9, 13 holes

At each snapshot, runs simulate_remaining_rounds() and records predictions.
Compares against actual final outcomes.

Metrics reported per checkpoint group:
  - Brier score (win, top5, top10)
  - Top-10 precision (our top-10 predicted = actual top-10?)
  - Spearman rho (projected finish order vs actual)
  - Leader win rate (did our #1 probability player win?)
  - Win% calibration buckets

Usage:
    python3 scripts/backtest_live_mc.py
    python3 scripts/backtest_live_mc.py --years 2024 2025
    python3 scripts/backtest_live_mc.py --n-sims 5000 --quick
"""

from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def _norm_name(name: str) -> str:
    """Normalise diacritics → ASCII for name matching (Åberg → Aberg)."""
    if not isinstance(name, str):
        return str(name)
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from augusta_model.simulation.remaining_rounds_mc import (
    simulate_remaining_rounds,
    DEFAULT_ROUND_MEAN,
)

PROCESSED = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "data" / "processed"

# ── Config ────────────────────────────────────────────────────────────────────

BACKTEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# (current_round, holes_thru) — holes_thru=18 means round is complete
CHECKPOINTS = [
    (1,  4), (1,  9), (1, 13), (1, 18),
    (2,  4), (2,  9), (2, 13), (2, 18),
    (3,  4), (3,  9), (3, 13), (3, 18),
    (4,  4), (4,  9), (4, 13),           # R4 hole 18 = tournament over
]

N_SIMS   = 10_000   # per checkpoint simulation (fast but robust)
SEED     = 42


# ── Data loading & validation ─────────────────────────────────────────────────

def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # ── Hole-by-hole ──────────────────────────────────────────────
    hbh_path = PROCESSED / "masters_hole_by_hole.parquet"
    assert hbh_path.exists(), f"Missing: {hbh_path}"
    hbh = pd.read_parquet(hbh_path)
    hbh = hbh[hbh["round"].isin([1, 2, 3, 4])].copy()   # drop playoff rounds
    hbh["player_name"] = hbh["player_name"].apply(_norm_name)

    print(f"\nmasters_hole_by_hole: {hbh.shape[0]:,} rows")
    print(f"  Years:   {sorted(hbh['year'].unique())}")
    print(f"  Rounds:  {sorted(hbh['round'].unique())}")
    for col in ["year", "player_name", "round", "hole_number", "score_to_par"]:
        null_n = hbh[col].isna().sum()
        flag   = "✓" if null_n == 0 else f"⚠ {null_n} nulls"
        print(f"  {col}: {flag}")

    # Check completeness — every player should have 18 holes per round
    round_counts = (
        hbh.groupby(["year", "player_name", "round"])["hole_number"].count()
    )
    incomplete = (round_counts < 18).sum()
    print(f"  Incomplete rounds (<18 holes): {incomplete} / {len(round_counts)}")

    # ── SG rounds ────────────────────────────────────────────────
    sg_path = PROCESSED / "masters_sg_rounds.parquet"
    assert sg_path.exists(), f"Missing: {sg_path}"
    sg = pd.read_parquet(sg_path)
    sg = sg.rename(columns={"season": "year", "round_num": "round"})
    sg["player_name"] = sg["player_name"].apply(_norm_name)

    print(f"\nmasters_sg_rounds: {sg.shape[0]:,} rows")
    print(f"  Years: {sorted(sg['year'].unique())}")
    sg_cols = [c for c in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
               if c in sg.columns]
    for col in sg_cols:
        null_pct = sg[col].isna().mean()
        flag = "✓" if null_pct < 0.05 else f"⚠ {null_pct:.1%} null"
        print(f"  {col}: {flag}")

    # ── Unified (final results) ────────────────────────────────────
    uni_path = PROCESSED / "masters_unified.parquet"
    assert uni_path.exists(), f"Missing: {uni_path}"
    uni = pd.read_parquet(uni_path)
    uni = uni.rename(columns={"season": "year"})
    uni["player_name"] = uni["player_name"].apply(_norm_name)

    print(f"\nmasters_unified: {uni.shape[0]:,} rows")
    print(f"  Years: {sorted(uni['year'].unique())}")
    for col in ["finish_num", "total_score", "made_cut"]:
        if col in uni.columns:
            null_pct = uni[col].isna().mean()
            flag = "✓" if null_pct < 0.05 else f"⚠ {null_pct:.1%} null"
            print(f"  {col}: {flag}")

    # ── Name consistency (2024 spot-check) ───────────────────────
    year_check = 2024
    hbh_names = set(hbh[hbh["year"] == year_check]["player_name"].unique())
    sg_names  = set(sg[sg["year"]   == year_check]["player_name"].unique())
    uni_names = set(uni[uni["year"] == year_check]["player_name"].unique())

    overlap_hbh_sg  = hbh_names & sg_names
    only_in_hbh     = hbh_names - sg_names
    only_in_sg      = sg_names  - hbh_names

    print(f"\nName match 2024 — hbh∩sg: {len(overlap_hbh_sg)}, "
          f"hbh-only: {len(only_in_hbh)}, sg-only: {len(only_in_sg)}")
    if only_in_hbh:
        print(f"  In hole-by-hole but not sg: {sorted(only_in_hbh)[:8]}")
    if only_in_sg:
        print(f"  In sg but not hole-by-hole: {sorted(only_in_sg)[:8]}")

    overlap_hbh_uni = hbh_names & uni_names
    print(f"  hbh∩unified: {len(overlap_hbh_uni)} of {len(hbh_names)} hbh players")

    print("\n✓ Data validation complete\n")
    return hbh, sg, uni


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_checkpoint_leaderboard(
    hbh: pd.DataFrame,
    year: int,
    current_round: int,
    holes_thru: int,
) -> pd.DataFrame:
    """
    Reconstruct the leaderboard at (year, current_round, holes_thru).

    Returns player_name | current_score (cumul to-par) | holes_completed | made_cut_actual
    """
    yr = hbh[hbh["year"] == year].copy()

    # Sum all completed rounds
    completed_rounds = list(range(1, current_round))
    if completed_rounds:
        comp_scores = (
            yr[yr["round"].isin(completed_rounds)]
            .groupby("player_name")["score_to_par"]
            .sum()
        )
    else:
        comp_scores = pd.Series(dtype=float)

    # Add partial current round
    if holes_thru > 0:
        partial = yr[
            (yr["round"] == current_round) &
            (yr["hole_number"] <= holes_thru)
        ]
        partial_scores = partial.groupby("player_name")["score_to_par"].sum()
    else:
        partial_scores = pd.Series(dtype=float)

    # All players who teed off in current_round
    active_players = set(yr[yr["round"] == current_round]["player_name"].unique())
    if not active_players:
        # R4 might be smaller field — use whoever survived to that round
        for r in range(current_round - 1, 0, -1):
            active_players = set(yr[yr["round"] == r]["player_name"].unique())
            if active_players:
                break

    rows = []
    for p in active_players:
        comp  = float(comp_scores.get(p, 0.0))
        part  = float(partial_scores.get(p, 0.0)) if p in partial_scores.index else 0.0
        # made_cut_actual: did this player play rounds 3 & 4?
        played_r3 = not yr[(yr["player_name"] == p) & (yr["round"] == 3)].empty
        rows.append({
            "player_name":    p,
            "current_score":  comp + part,
            "holes_completed": (current_round - 1) * 18 + (
                holes_thru if p in partial_scores.index else 0
            ),
            "made_cut_actual": int(played_r3 or current_round >= 3),
        })

    return pd.DataFrame(rows)


def get_cumulative_sg(
    sg: pd.DataFrame,
    year: int,
    completed_rounds: list[int],
) -> pd.DataFrame:
    """
    Sum SG across completed_rounds for each player.
    Returns player_name | sg_ott | sg_app | sg_arg | sg_putt | sg_t2g | sg_total
    """
    sg_cols = [c for c in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
               if c in sg.columns]

    if not completed_rounds:
        # No completed rounds — return empty SG (all zeros)
        players = sg[sg["year"] == year]["player_name"].unique()
        df = pd.DataFrame({"player_name": players})
        for col in sg_cols:
            df[col] = 0.0
        return df

    yr_sg = sg[(sg["year"] == year) & (sg["round"].isin(completed_rounds))]
    if yr_sg.empty:
        return pd.DataFrame(columns=["player_name"] + sg_cols)

    cumul = yr_sg.groupby("player_name")[sg_cols].sum().reset_index()
    return cumul


def build_pre_df(uni: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Pre-tournament player profile based on prior 3 years of Augusta performance.

    Uses score_vs_field (strokes vs field average) from masters_unified.
    Normalises to model_score in [0,1] where 0=best, 0.5=field-average, 1=worst.
    Debutants get model_score=0.5 (field average prior).
    """
    prior_years = [y for y in [year - 1, year - 2, year - 3] if y >= 2004]
    prior = uni[uni["year"].isin(prior_years)].copy()

    # Weight more recent years higher (1.0, 0.7, 0.4)
    weights = {prior_years[i]: [1.0, 0.7, 0.4][i] for i in range(len(prior_years))}
    prior["w"] = prior["year"].map(weights)

    if "score_vs_field" in prior.columns and prior["score_vs_field"].notna().any():
        # Weighted average score vs field (negative = better)
        prior["wsvf"] = prior["score_vs_field"] * prior["w"]
        agg = prior.groupby("player_name").agg(
            wsvf_sum=("wsvf", "sum"),
            w_sum=("w", "sum"),
        )
        agg["avg_svf"] = agg["wsvf_sum"] / agg["w_sum"]

        # Normalise: scale so that ±10 strokes → 0/1 range
        # model_score = 0 → elite Augusta performer, 0.5 → average, 1 → poor
        agg["model_score"] = ((agg["avg_svf"] + 10) / 20).clip(0, 1)
        pre_df = agg[["model_score"]].reset_index()
    else:
        pre_df = pd.DataFrame(columns=["player_name", "model_score"])

    # Add SG rolling features for 2021-2025 (if year has prior SG data)
    sg_prior_years = [y for y in prior_years if y >= 2021]
    if sg_prior_years and "sg_ott" in uni.columns:
        sg_prior = uni[uni["year"].isin(sg_prior_years)].copy()
        sg_prior["w"] = sg_prior["year"].map(weights)
        sg_cols = [c for c in ["sg_ott", "sg_app", "sg_arg", "sg_putt"] if c in sg_prior.columns]
        for col in sg_cols:
            sg_prior[f"w_{col}"] = sg_prior[col] * sg_prior["w"]
        agg_sg = sg_prior.groupby("player_name").agg(
            **{f"{col}_wsum": (f"w_{col}", "sum") for col in sg_cols},
            w_sum=("w", "sum"),
        )
        for col in sg_cols:
            agg_sg[f"{col}_8w"] = agg_sg[f"{col}_wsum"] / agg_sg["w_sum"]

        sg_features = agg_sg[[f"{col}_8w" for col in sg_cols]].reset_index()
        pre_df = pre_df.merge(sg_features, on="player_name", how="left")

    # Fill debutants (players in current year not in prior data)
    current_players = uni[uni["year"] == year]["player_name"].unique()
    existing = set(pre_df["player_name"].values) if not pre_df.empty else set()
    new_rows = [{"player_name": p, "model_score": 0.5}
                for p in current_players if p not in existing]
    if new_rows:
        pre_df = pd.concat([pre_df, pd.DataFrame(new_rows)], ignore_index=True)

    # Ensure model_score exists and is filled
    if "model_score" not in pre_df.columns:
        pre_df["model_score"] = 0.5
    pre_df["model_score"] = pre_df["model_score"].fillna(0.5)

    return pre_df


def get_final_results(uni: pd.DataFrame, year: int) -> pd.DataFrame:
    """Actual final tournament results for a year."""
    yr = uni[uni["year"] == year].copy()

    yr["final_rank"] = yr["finish_num"].fillna(999).astype(float)
    yr["is_winner"]  = yr["final_rank"] == 1
    yr["is_top5"]    = yr["final_rank"] <= 5
    yr["is_top10"]   = yr["final_rank"] <= 10
    yr["made_cut"]   = yr["made_cut"].fillna(0).astype(int)

    return yr[["player_name", "final_rank", "is_winner", "is_top5", "is_top10",
               "total_score", "made_cut"]].copy()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, checkpoint_label: str) -> dict:
    """
    Compute accuracy metrics for a single checkpoint's predictions vs actual.

    df must have: mc_win_prob, mc_top5_prob, mc_top10_prob,
                  is_winner, is_top5, is_top10, final_rank
    """
    m = {"checkpoint": checkpoint_label, "n_players": len(df)}

    valid = df.dropna(subset=["mc_win_prob", "final_rank"])
    valid = valid[valid["final_rank"] < 900]   # exclude MC/WD from rank metrics

    if valid.empty:
        return m

    # Brier scores
    for prob_col, target_col in [
        ("mc_win_prob",   "is_winner"),
        ("mc_top5_prob",  "is_top5"),
        ("mc_top10_prob", "is_top10"),
    ]:
        if prob_col in df.columns and target_col in df.columns:
            brier = np.mean(
                (df[prob_col].fillna(0) - df[target_col].astype(float)) ** 2
            )
            key = prob_col.replace("mc_", "brier_").replace("_prob", "")
            m[key] = round(brier, 5)

    # Top-10 precision: of our top-10 by mc_top10_prob, how many were actual top-10?
    if "mc_top10_prob" in df.columns and "is_top10" in df.columns:
        predicted_top10 = df.nlargest(10, "mc_top10_prob")
        m["top10_precision"] = round(predicted_top10["is_top10"].mean(), 3)

    # Spearman rank correlation (higher mc_win_prob = predicted lower finish)
    if len(valid) >= 5:
        rho, _ = spearmanr(-valid["mc_win_prob"], valid["final_rank"])
        m["spearman_rho"] = round(float(rho), 3)

    # Did our predicted leader win?
    if "mc_win_prob" in df.columns and "is_winner" in df.columns:
        leader_idx = df["mc_win_prob"].idxmax()
        m["leader_won"] = int(df.loc[leader_idx, "is_winner"])
        m["leader_name"] = df.loc[leader_idx, "player_name"]

    # Winner's predicted rank (where did the actual winner appear in our list?)
    winners = df[df["is_winner"] == True]
    if not winners.empty and "mc_win_prob" in df.columns:
        df_ranked = df.sort_values("mc_win_prob", ascending=False).reset_index(drop=True)
        df_ranked["pred_rank"] = df_ranked.index + 1
        winner_name = winners.iloc[0]["player_name"]
        winner_row  = df_ranked[df_ranked["player_name"] == winner_name]
        if not winner_row.empty:
            m["winner_pred_rank"] = int(winner_row.iloc[0]["pred_rank"])

    return m


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    years: list[int],
    checkpoints: list[tuple],
    n_sims: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full backtest. Returns (all_predictions_df, metrics_df).
    """
    hbh, sg, uni = load_and_validate()

    all_preds  = []
    all_metrics = []

    for year in years:
        print(f"\n{'─'*60}")
        print(f"Year: {year}")
        print(f"{'─'*60}")

        # Pre-tournament profile (built from prior years)
        pre_df = build_pre_df(uni, year)
        print(f"  pre_df built: {len(pre_df)} players, model_score range "
              f"[{pre_df['model_score'].min():.2f}, {pre_df['model_score'].max():.2f}]")

        # Final actual results for this year
        final = get_final_results(uni, year)
        if final.empty:
            print(f"  ⚠ No final results for {year} — skipping")
            continue
        print(f"  Final results: {len(final)} players, "
              f"winner = {final[final['is_winner']]['player_name'].values}")

        for (current_round, holes_thru) in checkpoints:
            label = f"R{current_round}h{holes_thru:02d}"

            # Skip R4 hole 18 — tournament over, nothing to simulate
            if current_round == 4 and holes_thru >= 18:
                continue

            # Reconstruct leaderboard at this checkpoint
            board = build_checkpoint_leaderboard(hbh, year, current_round, holes_thru)
            if board.empty or len(board) < 5:
                if verbose:
                    print(f"  {label}: skipped (too few players: {len(board)})")
                continue

            # Completed SG rounds (for mid-round checkpoints, exclude current round)
            if holes_thru < 18:
                sg_complete_rounds = list(range(1, current_round))
            else:
                sg_complete_rounds = list(range(1, current_round + 1))

            cumul_sg = get_cumulative_sg(sg, year, sg_complete_rounds)

            # Build live_df (format expected by simulate_remaining_rounds)
            live_df = board.merge(cumul_sg, on="player_name", how="left")

            # Fill missing SG with 0 (no data = no adjustment)
            for col in ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]:
                if col not in live_df.columns:
                    live_df[col] = 0.0
                else:
                    live_df[col] = live_df[col].fillna(0.0)

            # Make-cut probability: before R2 complete, use 0.75 prior;
            # after R2 complete, use actual cut result
            if current_round <= 2 and holes_thru < 18:
                live_df["dg_make_cut"] = 0.75
            elif current_round == 2 and holes_thru == 18:
                # Merge actual cut result
                live_df = live_df.merge(
                    final[["player_name", "made_cut"]],
                    on="player_name", how="left"
                )
                live_df["dg_make_cut"] = live_df["made_cut"].fillna(0.5)
                live_df = live_df.drop(columns=["made_cut"], errors="ignore")
            else:
                # R3+ — everyone here made the cut
                live_df["dg_make_cut"] = 1.0

            # Run MC simulation
            try:
                mc = simulate_remaining_rounds(
                    live_df=live_df,
                    pre_df=pre_df,
                    current_round=current_round,
                    median_thru=float(holes_thru),
                    n_sims=n_sims,
                    seed=SEED,
                )
            except Exception as e:
                print(f"  {label}: MC failed — {e}")
                continue

            if mc.empty:
                if verbose:
                    print(f"  {label}: MC returned empty")
                continue

            # Merge actual results
            mc = mc.merge(
                final[["player_name", "final_rank", "is_winner", "is_top5",
                        "is_top10", "total_score"]],
                on="player_name", how="left"
            )
            mc["year"]          = year
            mc["current_round"] = current_round
            mc["holes_thru"]    = holes_thru
            mc["checkpoint"]    = label

            all_preds.append(mc)

            # Metrics for this single checkpoint
            m = compute_metrics(mc, label)
            m["year"] = year
            all_metrics.append(m)

            if verbose:
                top3 = mc.nlargest(3, "mc_win_prob")[
                    ["player_name", "current_score", "mc_win_prob"]
                ].values
                top3_str = ", ".join(
                    f"{r[0]} ({r[2]:.0%})" for r in top3
                )
                leader_won = "✓" if m.get("leader_won") else "✗"
                print(
                    f"  {label}: {len(mc)} players | "
                    f"Top3: {top3_str} | "
                    f"Leader won: {leader_won} | "
                    f"T10 prec: {m.get('top10_precision', 0):.0%} | "
                    f"ρ: {m.get('spearman_rho', 0):.3f}"
                )

    preds_df   = pd.concat(all_preds, ignore_index=True)  if all_preds   else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics)                 if all_metrics else pd.DataFrame()

    return preds_df, metrics_df


# ── Summary reporting ─────────────────────────────────────────────────────────

def print_summary(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        print("No metrics to summarise.")
        return

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY — Remaining-Rounds MC")
    print("=" * 70)

    # Group by checkpoint (across all years)
    key_cols = [c for c in
                ["brier_win", "brier_top5", "brier_top10",
                 "top10_precision", "spearman_rho", "leader_won",
                 "winner_pred_rank"]
                if c in metrics_df.columns]

    # By round stage
    metrics_df["round_stage"] = "R" + metrics_df["checkpoint"].str[1]

    stage_summary = (
        metrics_df
        .groupby("round_stage")[key_cols]
        .mean()
        .round(3)
    )
    print("\n— By Round Stage (avg across years & hole checkpoints) —")
    print(stage_summary.to_string())

    # By holes-thru bucket
    metrics_df["holes_bucket"] = pd.cut(
        metrics_df["checkpoint"].str[3:].astype(int),
        bins=[0, 4, 9, 13, 18],
        labels=["Q1 (≤4h)", "Half (5-9h)", "Q3 (10-13h)", "Full round"],
        right=True,
    )
    bucket_summary = (
        metrics_df
        .groupby("holes_bucket")[key_cols]
        .mean()
        .round(3)
    )
    print("\n— By Holes-Thru Bucket (avg across years & rounds) —")
    print(bucket_summary.to_string())

    # By year
    year_summary = (
        metrics_df
        .groupby("year")[key_cols]
        .mean()
        .round(3)
    )
    print("\n— By Year (avg across all checkpoints) —")
    print(year_summary.to_string())

    # Overall
    overall = metrics_df[key_cols].mean().round(3)
    print("\n— OVERALL AVERAGES —")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # Win% calibration: did 10% win-prob players win ~10% of the time?
    print("\n— Win% Calibration (actual win rate by predicted win-prob bucket) —")
    # Handled in saved predictions CSV


def print_calibration(preds_df: pd.DataFrame) -> None:
    if preds_df.empty or "mc_win_prob" not in preds_df.columns:
        return
    preds_df = preds_df.copy()
    preds_df["prob_bucket"] = pd.cut(
        preds_df["mc_win_prob"],
        bins=[0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 1.0],
        labels=["0-1%", "1-3%", "3-6%", "6-10%", "10-20%", "20-40%", ">40%"],
        right=True,
    )
    cal = (
        preds_df[preds_df["is_winner"].notna()]
        .groupby("prob_bucket", observed=True)
        .agg(
            n=("mc_win_prob", "count"),
            avg_pred_win_pct=("mc_win_prob", "mean"),
            actual_win_rate=("is_winner", "mean"),
        )
        .round(3)
    )
    print(cal.to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest remaining-rounds MC simulator")
    parser.add_argument("--years", nargs="+", type=int, default=BACKTEST_YEARS)
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--quick", action="store_true",
                        help="Only run half-round and full-round checkpoints")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    checkpoints = CHECKPOINTS
    if args.quick:
        checkpoints = [(r, h) for r, h in CHECKPOINTS if h in (9, 18)]

    print(f"Backtesting years: {args.years}")
    print(f"Checkpoints: {len(checkpoints)} per year")
    print(f"Simulations per checkpoint: {args.n_sims:,}")

    preds_df, metrics_df = run_backtest(
        years=args.years,
        checkpoints=checkpoints,
        n_sims=args.n_sims,
        verbose=not args.quiet,
    )

    if preds_df.empty:
        print("\n⚠ No predictions generated — check data paths")
        sys.exit(1)

    # Save outputs
    preds_path   = OUT_DIR / "backtest_live_mc_predictions.csv"
    metrics_path = OUT_DIR / "backtest_live_mc_metrics.csv"
    preds_df.to_csv(preds_path,   index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Predictions saved → {preds_path}")
    print(f"✓ Metrics saved     → {metrics_path}")

    print_summary(metrics_df)
    print("\n— Win% Calibration —")
    print_calibration(preds_df)


if __name__ == "__main__":
    main()

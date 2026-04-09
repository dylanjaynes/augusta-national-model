#!/usr/bin/env python3
"""
Post-hoc form penalty for LIV-primary players with strong Augusta history but poor current form.

Root cause of Bug #3: S2 classifier is dominated by Augusta history features (r=-0.633 with
augusta_scoring_avg vs r=-0.297 with dg_rank). Players like Patrick Reed and Cameron Smith
get nearly identical S2 scores to Scottie Scheffler despite massively inferior current form.

This script applies a shrinkage penalty to win_prob and top10_prob for players where:
  1. dg_rank > 40 (not among elite current players)
  2. model_score (S1) < 0.15 (bottom quartile of current-form rating)

The penalty blends their model probability toward a market-adjusted base rate.

Usage:
  python3 scripts/apply_form_penalty.py
  # Reads predictions_2026.parquet, writes predictions_2026_form_adjusted.parquet/.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRED_IN = Path("data/processed/predictions_2026.parquet")
PRED_OUT_PQ = Path("data/processed/predictions_2026_form_adjusted.parquet")
PRED_OUT_CSV = Path("data/processed/predictions_2026_form_adjusted.csv")

# Field size for base rates
FIELD_SIZE = 82
BASE_WIN = 1 / FIELD_SIZE
BASE_TOP10 = 10 / FIELD_SIZE

# Penalty parameters
DG_RANK_THRESHOLD = 40    # players ranked outside top 40 are subject to review
S1_THRESHOLD = 0.15       # S1 score below this = poor current form
SHRINKAGE = 0.45          # blend 45% toward base rate for flagged players


def apply_form_penalty(df):
    df = df.copy()

    # Identify players with poor current form relative to Augusta-history-boosted S2
    poor_form = (
        (df["dg_rank"].fillna(999) > DG_RANK_THRESHOLD) &
        (df["model_score"] < S1_THRESHOLD)
    )
    n_flagged = poor_form.sum()
    print(f"Players flagged for form penalty: {n_flagged}")
    print(df[poor_form][["player_name", "dg_rank", "model_score", "win_prob",
                           "top10_prob", "dk_fair_prob_win"]].to_string())

    # Apply shrinkage toward base rate
    for col, base in [("win_prob", BASE_WIN), ("top5_prob", 5/FIELD_SIZE),
                      ("top10_prob", BASE_TOP10), ("top20_prob", 20/FIELD_SIZE)]:
        if col in df.columns:
            df.loc[poor_form, col] = (
                (1 - SHRINKAGE) * df.loc[poor_form, col] + SHRINKAGE * base
            )

    # Re-normalize so sums remain correct
    for col, target in [("win_prob", 1.0), ("top5_prob", 5.0),
                        ("top10_prob", 10.0), ("top20_prob", 20.0)]:
        if col in df.columns:
            current_sum = df[col].sum()
            df[col] = df[col] * (target / current_sum)

    return df


def main():
    print(f"Loading {PRED_IN}...")
    pred = pd.read_parquet(PRED_IN)
    print(f"  {len(pred)} players")

    print("\nApplying form penalty...")
    pred_adj = apply_form_penalty(pred)

    print("\n=== Before vs After (top 15 by win_prob after adjustment) ===")
    compare = pred_adj[["player_name", "win_prob", "top10_prob",
                          "model_score", "dg_rank", "dk_fair_prob_win"]].copy()
    compare["win_prob_before"] = pred["win_prob"]
    compare["win_delta"] = compare["win_prob"] - compare["win_prob_before"]
    print(compare.sort_values("win_prob", ascending=False).head(15).to_string())

    print("\nSum checks (should be 1.0, 5.0, 10.0, 20.0):")
    for col, target in [("win_prob", 1.0), ("top5_prob", 5.0),
                        ("top10_prob", 10.0), ("top20_prob", 20.0)]:
        if col in pred_adj.columns:
            print(f"  {col}: {pred_adj[col].sum():.4f} (target {target})")

    print(f"\nSaving to {PRED_OUT_PQ} and {PRED_OUT_CSV}...")
    pred_adj.to_parquet(PRED_OUT_PQ, index=False)
    pred_adj.to_csv(PRED_OUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

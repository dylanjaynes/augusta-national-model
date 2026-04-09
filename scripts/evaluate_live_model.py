"""
Backtest the live in-tournament model.

Evaluates at each snapshot point (holes 3, 6, 9, 12, 15, 18):
  - AUC for top-10 predictions
  - Spearman correlation with final finish
  - Top-10 precision @ top 13 picks
  - Compares against: (a) pre-tournament baseline, (b) leaderboard position

Usage:
    python3 scripts/evaluate_live_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from augusta_model.model.live_model import (
    _prepare_features,
    load_live_model,
    TRAIN_YEARS,
    VAL_YEAR,
    TEST_YEAR,
)

DATA_DIR = ROOT / "data" / "processed"
SNAPSHOT_HOLES = [3, 6, 9, 12, 15, 18]


def leaderboard_rank_at_snapshot(data: pd.DataFrame, snap: int) -> pd.Series:
    """
    Given snapshot data for one year, compute leaderboard rank (by cumulative score)
    at the given snapshot hole.
    """
    snap_data = data[data["snapshot_hole"] == snap].copy()
    snap_data["leaderboard_rank"] = snap_data.groupby("year")["cumulative_score_to_par"].rank(
        method="min", ascending=True  # lower score = better
    )
    return snap_data.set_index(["year", "player_name"])["leaderboard_rank"]


def evaluate_at_snapshot(
    data: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str],
    snap: int,
    baseline_col: str = "top10_prob",
) -> dict:
    """
    Evaluate all three approaches at a given snapshot hole:
    1. Live model
    2. Pre-tournament baseline
    3. Current leaderboard position
    """
    snap_data = data[data["snapshot_hole"] == snap].copy()
    if snap_data.empty:
        return {}

    y_top10 = snap_data["top10"].fillna(0).astype(int)
    y_finish = snap_data["finish_pct"].fillna(0.5)

    # --- Live model ---
    X = _prepare_features(snap_data, feature_cols)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    live_probs = clf.predict_proba(X)[:, 1]
    live_finish = reg.predict(X)

    # --- Baseline: pre-tournament top10_prob ---
    if baseline_col in snap_data.columns:
        base_probs = snap_data[baseline_col].fillna(snap_data[baseline_col].mean())
    else:
        base_probs = pd.Series(np.full(len(snap_data), 10.0 / 88), index=snap_data.index)

    # --- Leaderboard: rank by current score (lower = better) ---
    # Convert rank to a probability-like score (inverse rank)
    score_rank = snap_data["cumulative_score_to_par"].rank(ascending=True)
    n = len(snap_data)
    leaderboard_score = 1.0 - (score_rank - 1) / max(n - 1, 1)  # 1=leader, 0=last

    def safe_auc(y, probs):
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        return float(roc_auc_score(y, probs))

    def top_n_precision(y, probs, n_top=13):
        n_top = min(n_top, len(probs))
        # Use per-player best probability (take max across snapshot holes for this player)
        idx = np.argsort(-np.array(probs))[:n_top]
        return float(np.array(y)[idx].mean())

    result = {
        "snapshot_hole": snap,
        "n_rows": len(snap_data),
        "n_top10": int(y_top10.sum()),
        # Live model
        "live_auc": safe_auc(y_top10, live_probs),
        "live_spearman": float(spearmanr(y_finish, live_finish)[0]),
        "live_t10_prec": top_n_precision(y_top10, live_probs),
        # Baseline
        "base_auc": safe_auc(y_top10, base_probs),
        "base_spearman": float(spearmanr(y_finish, base_probs)[0]),
        "base_t10_prec": top_n_precision(y_top10, base_probs),
        # Leaderboard
        "board_auc": safe_auc(y_top10, leaderboard_score),
        "board_spearman": float(spearmanr(y_finish, leaderboard_score)[0]),
        "board_t10_prec": top_n_precision(y_top10, leaderboard_score),
    }

    return result


def run_backtest(years: list[int] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Full backtest across years and snapshot points.
    """
    if years is None:
        years = [VAL_YEAR, TEST_YEAR]

    print("Loading live model...")
    clf, reg, feature_cols, metadata = load_live_model()
    print(f"  Model trained on: {metadata.get('train_years')}")
    print(f"  Val AUC: {metadata.get('val_auc_top10')}  |  Spearman: {metadata.get('val_spearman')}")

    print("\nLoading training data...")
    data = pd.read_parquet(DATA_DIR / "live_training_data.parquet")

    # Deduplicate to one row per player/year/round/snapshot
    data = data.drop_duplicates(subset=["year", "player_name", "round_num", "snapshot_hole"])

    all_results = []

    for year in years:
        year_data = data[data["year"] == year]
        if year_data.empty:
            print(f"  No data for {year}")
            continue

        print(f"\n{'='*60}")
        print(f"Year {year}  (n_players={year_data['player_name'].nunique()})")
        print(f"{'='*60}")
        print(f"{'Snap':>6} | {'LiveAUC':>8} {'BaseAUC':>8} {'BoardAUC':>9} | {'LiveSpear':>10} {'BaseSpear':>10} | {'LivePrec':>9} {'BasePrec':>9}")
        print("-" * 85)

        for snap in SNAPSHOT_HOLES:
            metrics = evaluate_at_snapshot(
                data=year_data,
                clf=clf,
                reg=reg,
                feature_cols=feature_cols,
                snap=snap,
            )
            if not metrics:
                continue

            metrics["year"] = year
            all_results.append(metrics)

            if verbose:
                print(
                    f"  {snap:4d} | "
                    f"{metrics['live_auc']:8.3f} {metrics['base_auc']:8.3f} {metrics['board_auc']:9.3f} | "
                    f"{metrics['live_spearman']:10.3f} {metrics['base_spearman']:10.3f} | "
                    f"{metrics['live_t10_prec']:9.1%} {metrics['base_t10_prec']:9.1%}"
                )

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("\nNo results generated.")
        return df

    print(f"\n{'='*60}")
    print("Average across all years:")
    print(f"{'='*60}")
    avg_cols = ["live_auc", "base_auc", "board_auc", "live_spearman", "base_spearman", "live_t10_prec", "base_t10_prec"]
    avg_cols = [c for c in avg_cols if c in df.columns]
    grouped = df.groupby("snapshot_hole")[avg_cols].mean()
    print(grouped.to_string(float_format=lambda x: f"{x:.3f}"))

    # Final summary
    print(f"\nKey finding: vs pre-tournament baseline, live model improvement by snapshot:")
    if "live_auc" in df.columns and "base_auc" in df.columns:
        df["auc_lift"] = df["live_auc"] - df["base_auc"]
        lift_by_snap = df.groupby("snapshot_hole")["auc_lift"].mean()
        for snap, lift in lift_by_snap.items():
            print(f"  Hole {snap:2d}: AUC lift = {lift:+.3f}")

    # Save results
    out_path = DATA_DIR / "live_backtest_results.parquet"
    df.to_parquet(out_path, index=False)
    df.to_csv(str(out_path).replace(".parquet", ".csv"), index=False)
    print(f"\nResults saved to {out_path}")

    return df


def main():
    # Backtest on validation year + test year
    years = [VAL_YEAR, TEST_YEAR]
    results = run_backtest(years=years)

    if len(results) == 0:
        print("\nCheck that live model is trained and training data exists.")
        return

    # Print final summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Live Model vs Baselines — Average Across 2024+2025")
    print("=" * 70)
    avg = results[["snapshot_hole", "live_auc", "base_auc", "board_auc",
                   "live_t10_prec", "base_t10_prec"]].groupby("snapshot_hole").mean()
    print(avg.round(3).to_string())


if __name__ == "__main__":
    main()

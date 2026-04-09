"""
Build live-model training dataset from historical hole-by-hole data.

For each player/year/round, creates snapshots at holes 3, 6, 9, 12, 15, 18.
Joins weather, course difficulty, and pre-tournament baseline features.

Output: data/processed/live_training_data.parquet

Usage:
    python3 scripts/build_live_training_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from augusta_model.features.live_features import (
    _load_course_stats,
    _get_weather_for_round,
    build_player_round_snapshot,
    add_pretournament_baseline,
)

DATA_DIR = ROOT / "data" / "processed"

SNAPSHOT_HOLES = [3, 6, 9, 12, 15, 18]

# Years to use. Train=2015-2023, val=2024, test=2025.
ALL_YEARS = list(range(2015, 2026))


def _build_historical_pretournament(hbh_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    For a given target year, build pseudo-pre-tournament predictions from
    prior years' hole-by-hole data. This proxies what the production model
    would have predicted before the tournament.

    Returns a DataFrame with player_name + approximate pre-tournament columns.
    """
    # Use all prior years' data
    prior = hbh_df[hbh_df["year"] < year].copy()
    if prior.empty:
        # No history — return empty frame with expected columns
        return pd.DataFrame(
            columns=["player_name", "win_prob", "top10_prob", "top20_prob",
                     "dg_rank", "model_score", "stage2_prob_raw",
                     "augusta_competitive_rounds", "augusta_experience_tier",
                     "augusta_made_cut_prev_year", "augusta_scoring_trajectory",
                     "tour_vs_augusta_divergence"]
        )

    # Compute simple Augusta-history features from prior data
    # (This is the "pre-tournament" proxy for training — in live use, we
    #  load actual predictions_2026.parquet)

    player_stats = []
    players = prior["player_name"].unique()
    for player in players:
        p = prior[prior["player_name"] == player]
        years_played = sorted(p["year"].unique())
        starts = len(years_played)
        cuts_made = p.groupby("year")["made_cut"].first().sum()
        cut_rate = float(cuts_made / starts) if starts > 0 else 0.0

        # Avg score to par across all rounds
        avg_score = float(p["score_to_par"].mean())

        # Finish positions
        finish_rows = (
            p.groupby("year")
            .apply(lambda g: g["finish_pos"].iloc[0] if "finish_pos" in g.columns else 80)
            .reset_index()
        )
        finish_rows.columns = ["year", "finish"]
        def parse_fp(x):
            try:
                return int(str(x).replace("T", "").strip())
            except Exception:
                return 80
        finish_rows["finish"] = finish_rows["finish"].apply(parse_fp)

        top10_years = (finish_rows["finish"] <= 10).sum()
        top10_rate = float(top10_years / starts) if starts > 0 else 0.0

        # Scoring trajectory: recent 2yr avg vs career avg
        recent_yrs = [y for y in years_played[-2:]]
        recent_avg = float(prior[(prior["player_name"] == player) & (prior["year"].isin(recent_yrs))]["score_to_par"].mean()) if recent_yrs else avg_score
        scoring_trajectory = avg_score - recent_avg  # positive = improving recently

        # Experience tier
        if starts == 0:
            tier = 0
        elif starts <= 1:
            tier = 1
        elif starts <= 3:
            tier = 2
        elif starts <= 6:
            tier = 3
        else:
            tier = 4

        # Made cut prev year
        prev_year = year - 1
        if prev_year in p["year"].values:
            prev_cut = p[p["year"] == prev_year]["made_cut"].iloc[0]
            made_cut_prev = int(bool(prev_cut))
        else:
            made_cut_prev = 0

        # best finish in last 3 years
        last3 = finish_rows[finish_rows["year"] >= year - 3]["finish"]
        best_recent = int(last3.min()) if len(last3) > 0 else 80

        # Proxy model_score: combination of cut rate and avg score
        # Use a simple heuristic (lower avg score = better)
        model_score_proxy = float(cut_rate * 0.3 + (1 - min(avg_score + 5, 10) / 10) * 0.7)

        player_stats.append({
            "player_name": player,
            "win_prob": float(top10_rate * 0.15 / max(1, starts)),  # rough proxy
            "top10_prob": float(top10_rate * 0.6 + cut_rate * 0.2),
            "top20_prob": float(top10_rate * 0.7 + cut_rate * 0.3),
            "dg_rank": 100.0,  # unknown in backtesting context
            "model_score": model_score_proxy,
            "stage2_prob_raw": float(top10_rate * 0.5 + 0.1),
            "augusta_competitive_rounds": int(sum([
                4 if c else 2
                for c in p.groupby("year")["made_cut"].first()
            ])),
            "augusta_experience_tier": tier,
            "augusta_made_cut_prev_year": made_cut_prev,
            "augusta_scoring_trajectory": float(scoring_trajectory),
            "tour_vs_augusta_divergence": 0.0,  # unknown for training proxy
        })

    return pd.DataFrame(player_stats)


def build_training_data(
    hbh_df: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    course_stats: pd.DataFrame,
    years: list[int] = None,
    snapshot_holes: list[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build snapshot-based training dataset.

    Each row = one player at one snapshot point in one round in one year.
    Target: finish_pos (final tournament finish), made_cut.
    """
    if years is None:
        years = ALL_YEARS
    if snapshot_holes is None:
        snapshot_holes = SNAPSHOT_HOLES

    all_rows = []

    for year in years:
        year_df = hbh_df[hbh_df["year"] == year]
        if year_df.empty:
            if verbose:
                print(f"  No data for {year}, skipping")
            continue

        # Build pre-tournament proxy for this year
        pretournament = _build_historical_pretournament(hbh_df, year)

        players = year_df["player_name"].unique()
        rounds = sorted(year_df["round"].dropna().unique())

        year_rows = []
        for player in players:
            for rnd in rounds:
                snap_rows = build_player_round_snapshot(
                    hbh_df=year_df,
                    year=year,
                    player_name=player,
                    round_num=rnd,
                    snapshot_holes=snapshot_holes,
                    course_stats=course_stats,
                    weather_hourly=weather_hourly,
                )
                year_rows.extend(snap_rows)

        if not year_rows:
            continue

        year_snap_df = pd.DataFrame(year_rows)

        # Join pre-tournament features
        year_snap_df = add_pretournament_baseline(
            year_snap_df, pretournament
        )

        all_rows.append(year_snap_df)
        if verbose:
            print(f"  {year}: {len(year_snap_df):,} rows ({len(players)} players)")

    if not all_rows:
        raise ValueError("No training data built — check input files")

    result = pd.concat(all_rows, ignore_index=True)

    # Clip finish_pos to reasonable range
    result["finish_pos"] = result["finish_pos"].clip(1, 100)

    # Binary targets
    result["top10"] = (result["finish_pos"] <= 10).astype(int)
    result["top20"] = (result["finish_pos"] <= 20).astype(int)

    # finish_pct: 0=winner, 1=last (same scale as main model)
    field_size = result.groupby("year")["finish_pos"].transform("max").clip(lower=40)
    result["finish_pct"] = ((result["finish_pos"] - 1) / (field_size - 1)).clip(0, 1)

    return result


def main():
    print("Loading data...")
    hbh = pd.read_parquet(DATA_DIR / "masters_hole_by_hole.parquet")
    weather = pd.read_parquet(DATA_DIR / "masters_weather_hourly.parquet")

    print(f"  Hole-by-hole: {len(hbh):,} rows, years {sorted(hbh['year'].unique())}")
    print(f"  Weather: {len(weather):,} hourly rows")

    print("Computing course stats from training years (2015-2023)...")
    train_hbh = hbh[hbh["year"] <= 2023]
    course_stats = _load_course_stats(train_hbh)
    print("  Course difficulty per hole:")
    print(course_stats.sort_values("avg_score_to_par", ascending=False).head(5).to_string(index=False))

    print("\nBuilding training snapshots...")
    data = build_training_data(
        hbh_df=hbh,
        weather_hourly=weather,
        course_stats=course_stats,
        years=ALL_YEARS,
        snapshot_holes=SNAPSHOT_HOLES,
        verbose=True,
    )

    print(f"\nTotal rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Year breakdown:")
    print(data.groupby("year")[["top10", "top20"]].agg(["sum", "count"]).to_string())

    out_path = DATA_DIR / "live_training_data.parquet"
    data.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Also save course stats
    course_out = DATA_DIR / "course_hole_stats.parquet"
    all_course_stats = _load_course_stats(hbh)
    all_course_stats.to_parquet(course_out, index=False)
    print(f"Course stats saved to {course_out}")

    return data


if __name__ == "__main__":
    main()

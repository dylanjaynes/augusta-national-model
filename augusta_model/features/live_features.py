"""
Live in-tournament feature engineering for the Augusta National Model.

Takes hole-by-hole scoring data and computes progressive features at any
snapshot point during a round. Also joins pre-tournament baseline features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Augusta hole definitions
# Par-5 birdie opportunities: 2, 8, 13, 15
# Par-3 danger holes: 4, 6, 12 (Golden Bell), 16
# Amen Corner: 11, 12, 13

PAR5_HOLES = {2, 8, 13, 15}
PAR3_HOLES = {4, 6, 12, 16}
AMEN_CORNER_HOLES = {11, 12, 13}

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"


def _load_course_stats(hole_df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical scoring averages per hole from training data."""
    stats = (
        hole_df.groupby("hole_number")["score_to_par"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "avg_score_to_par", "std": "std_score_to_par"})
        .reset_index()
    )
    return stats


def _remaining_difficulty(
    holes_completed: int,
    course_stats: pd.DataFrame,
    starting_hole: int = 1,
) -> tuple[float, int, int]:
    """
    Compute difficulty metrics for holes not yet played.

    Returns:
        remaining_difficulty: weighted avg score_to_par of remaining holes
        hard_holes_remaining: count of holes with avg > 0.2 (bogey-prone)
        birdie_holes_remaining: count of holes with avg < -0.1 (birdie opportunities)
    """
    if holes_completed >= 18:
        return 0.0, 0, 0

    # Build sequence of holes in play order
    hole_seq = [(((starting_hole - 1 + i) % 18) + 1) for i in range(18)]
    remaining = hole_seq[holes_completed:]

    rem_stats = course_stats[course_stats["hole_number"].isin(remaining)]
    if rem_stats.empty:
        return 0.0, 0, 0

    avg_diff = rem_stats["avg_score_to_par"].mean()
    hard = int((rem_stats["avg_score_to_par"] > 0.20).sum())
    birdie = int((rem_stats["avg_score_to_par"] < -0.10).sum())
    return float(avg_diff), hard, birdie


def compute_snapshot_features(
    player_holes: pd.DataFrame,
    snapshot_hole: int,
    course_stats: pd.DataFrame,
    weather_row: Optional[pd.Series] = None,
    starting_hole: int = 1,
) -> dict:
    """
    Compute live features for a single player at a given snapshot point.

    Args:
        player_holes: All holes for this player in this round (columns:
                      hole_number, par, score, score_to_par)
        snapshot_hole: Number of holes completed so far (1-18)
        course_stats: Historical avg per hole from _load_course_stats()
        weather_row: Optional row from weather DataFrame
        starting_hole: 1 or 10 (tee assignment)

    Returns:
        dict of live features
    """
    feat = {}

    # Build ordered hole sequence
    hole_seq = [(((starting_hole - 1 + i) % 18) + 1) for i in range(18)]
    played_holes = hole_seq[:snapshot_hole]

    completed = player_holes[player_holes["hole_number"].isin(played_holes)].copy()
    completed = completed.sort_values("hole_number")

    holes_done = len(completed)
    feat["holes_completed"] = holes_done
    feat["holes_remaining"] = 18 - holes_done
    feat["holes_completed_pct"] = holes_done / 18.0
    feat["confidence_weight"] = np.sqrt(holes_done / 18.0)

    if holes_done == 0:
        # No data yet — all live features are 0/NaN
        feat.update({
            "cumulative_score_to_par": 0,
            "cumulative_birdie_rate": 0.0,
            "cumulative_bogey_rate": 0.0,
            "par5_scoring": 0.0,
            "par3_scoring": 0.0,
            "front_nine_score": 0,
            "back_nine_score": 0,
            "amen_corner_score": 0,
            "scoring_trend": 0.0,
            "remaining_difficulty": float(course_stats["avg_score_to_par"].mean()),
            "hard_holes_remaining": int((course_stats["avg_score_to_par"] > 0.20).sum()),
            "birdie_holes_remaining": int((course_stats["avg_score_to_par"] < -0.10).sum()),
        })
        _add_weather_features(feat, weather_row)
        return feat

    # Cumulative score
    feat["cumulative_score_to_par"] = int(completed["score_to_par"].sum())

    # Birdie / bogey rates
    is_birdie = completed["score_to_par"] <= -1
    is_bogey = completed["score_to_par"] >= 1
    feat["cumulative_birdie_rate"] = float(is_birdie.sum() / holes_done)
    feat["cumulative_bogey_rate"] = float(is_bogey.sum() / holes_done)

    # Par 5 scoring (birdie opportunities)
    par5_done = completed[completed["hole_number"].isin(PAR5_HOLES)]
    feat["par5_scoring"] = float(par5_done["score_to_par"].mean()) if len(par5_done) > 0 else 0.0

    # Par 3 scoring (danger holes — 12 in particular)
    par3_done = completed[completed["hole_number"].isin(PAR3_HOLES)]
    feat["par3_scoring"] = float(par3_done["score_to_par"].mean()) if len(par3_done) > 0 else 0.0

    # Front/back nine
    front = completed[completed["hole_number"].isin(range(1, 10))]
    back = completed[completed["hole_number"].isin(range(10, 19))]
    feat["front_nine_score"] = int(front["score_to_par"].sum()) if len(front) > 0 else 0
    feat["back_nine_score"] = int(back["score_to_par"].sum()) if len(back) > 0 else 0

    # Amen Corner (11-13)
    amen = completed[completed["hole_number"].isin(AMEN_CORNER_HOLES)]
    feat["amen_corner_score"] = int(amen["score_to_par"].sum()) if len(amen) > 0 else 0

    # Scoring trend: linear slope of cumulative score over last 6 holes
    if holes_done >= 3:
        recent = completed.tail(min(6, holes_done))
        x = np.arange(len(recent))
        y = recent["score_to_par"].values.astype(float)
        if len(x) >= 2:
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
    else:
        slope = 0.0
    feat["scoring_trend"] = slope

    # Remaining difficulty
    rem_diff, hard, birdie_rem = _remaining_difficulty(holes_done, course_stats, starting_hole)
    feat["remaining_difficulty"] = rem_diff
    feat["hard_holes_remaining"] = hard
    feat["birdie_holes_remaining"] = birdie_rem

    # Weather features
    _add_weather_features(feat, weather_row)

    return feat


def _add_weather_features(feat: dict, weather_row: Optional[pd.Series]) -> None:
    """Add weather features from an hourly row."""
    if weather_row is None:
        feat["current_wind_speed"] = 0.0
        feat["wind_direction"] = 0.0
        feat["temperature"] = 65.0
        feat["precipitation"] = 0.0
        feat["weather_severity_score"] = 0.0
        feat["morning_afternoon"] = 0
        return

    wind = float(weather_row.get("wind_speed_10m", 0) or 0)
    rain = float(weather_row.get("precipitation", 0) or 0)
    temp = float(weather_row.get("temperature_2m", 65) or 65)
    direction = float(weather_row.get("wind_direction_10m", 0) or 0)

    feat["current_wind_speed"] = wind
    feat["wind_direction"] = direction
    feat["temperature"] = temp
    feat["precipitation"] = rain

    # Severity: 0-1 scale (wind > 20 mph = severe; rain > 5mm/hr = severe)
    wind_severity = min(wind / 20.0, 1.0)
    rain_severity = min(rain / 5.0, 1.0)
    feat["weather_severity_score"] = float(0.7 * wind_severity + 0.3 * rain_severity)

    # Approximate morning wave: tee times before ~11am local = 1
    t = weather_row.get("time")
    if t is not None:
        hour = pd.Timestamp(t).hour
        feat["morning_afternoon"] = 1 if hour < 12 else 0
    else:
        feat["morning_afternoon"] = 0


def build_player_round_snapshot(
    hbh_df: pd.DataFrame,
    year: int,
    player_name: str,
    round_num: int,
    snapshot_holes: list[int],
    course_stats: pd.DataFrame,
    weather_hourly: pd.DataFrame,
) -> list[dict]:
    """
    Build all snapshot feature rows for one player/year/round combination.

    Returns list of dicts (one per snapshot_holes entry).
    """
    player_round = hbh_df[
        (hbh_df["year"] == year)
        & (hbh_df["player_name"] == player_name)
        & (hbh_df["round"] == round_num)
    ].copy()

    if player_round.empty:
        return []

    # Get starting hole (default 1 if not available)
    starting_hole = 1
    if "starting_hole" in player_round.columns:
        sh = player_round["starting_hole"].dropna()
        if len(sh) > 0:
            val = int(sh.iloc[0])
            if val in (1, 10):
                starting_hole = val

    made_cut = bool(player_round["made_cut"].iloc[0])
    finish_pos_raw = player_round["finish_pos"].iloc[0]

    # Parse finish position
    try:
        finish_pos = int(str(finish_pos_raw).replace("T", "").strip())
    except (ValueError, TypeError):
        finish_pos = 80  # MC or unknown → bad finish

    rows = []
    for snap in snapshot_holes:
        feat = compute_snapshot_features(
            player_holes=player_round,
            snapshot_hole=snap,
            course_stats=course_stats,
            weather_row=_get_weather_for_round(weather_hourly, year, round_num),
            starting_hole=starting_hole,
        )
        feat["year"] = year
        feat["player_name"] = player_name
        feat["round_num"] = round_num
        feat["snapshot_hole"] = snap
        feat["made_cut"] = int(made_cut)
        feat["finish_pos"] = finish_pos
        rows.append(feat)

    return rows


def _get_weather_for_round(
    weather_hourly: pd.DataFrame, year: int, round_num: int
) -> Optional[pd.Series]:
    """Get representative afternoon weather for a given year/round."""
    if weather_hourly.empty:
        return None

    day_rows = weather_hourly[
        (weather_hourly["season"] == year)
        & (weather_hourly["tournament_day"] == round_num)
    ]
    if day_rows.empty:
        return None

    # Use ~2pm (peak play) as representative hour
    day_rows = day_rows.copy()
    day_rows["hour"] = pd.to_datetime(day_rows["time"]).dt.hour
    afternoon = day_rows[day_rows["hour"].between(13, 16)]
    if afternoon.empty:
        return day_rows.iloc[0]
    return afternoon.iloc[0]


def add_pretournament_baseline(
    snapshot_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    historical_sg: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join pre-tournament predictions onto snapshot rows.

    predictions_df should have columns:
        player_name, win_prob, top10_prob, dg_rank, stage2_prob_raw, etc.
    """
    pred_cols = [
        "player_name",
        "win_prob",
        "top5_prob",
        "top10_prob",
        "top20_prob",
        "dg_rank",
        "model_score",
        "stage2_prob_raw",
        "augusta_competitive_rounds",
        "augusta_experience_tier",
        "augusta_made_cut_prev_year",
        "augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
    ]
    available = [c for c in pred_cols if c in predictions_df.columns]
    pred_sub = predictions_df[available].copy()

    merged = snapshot_df.merge(pred_sub, on="player_name", how="left")

    # Fill missing pre-tournament features with neutral values
    if "win_prob" in merged.columns:
        merged["win_prob"] = merged["win_prob"].fillna(1.0 / 88)
    if "top10_prob" in merged.columns:
        merged["top10_prob"] = merged["top10_prob"].fillna(10.0 / 88)
    if "dg_rank" in merged.columns:
        merged["dg_rank"] = merged["dg_rank"].fillna(200.0)
    if "model_score" in merged.columns:
        merged["model_score"] = merged["model_score"].fillna(0.1)

    # Interaction features
    if "current_wind_speed" in merged.columns and "remaining_difficulty" in merged.columns:
        merged["wind_x_difficulty"] = (
            merged["current_wind_speed"] * merged["remaining_difficulty"]
        )
    if "confidence_weight" in merged.columns and "cumulative_score_to_par" in merged.columns:
        # Pace: current cumulative / pct completed (projected final score)
        pct = merged["holes_completed_pct"].clip(lower=0.01)
        merged["projected_final_score"] = merged["cumulative_score_to_par"] / pct

    return merged


def get_live_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used by the live model."""
    return [
        # Live scoring
        "cumulative_score_to_par",
        "cumulative_birdie_rate",
        "cumulative_bogey_rate",
        "par5_scoring",
        "par3_scoring",
        "front_nine_score",
        "back_nine_score",
        "amen_corner_score",
        "scoring_trend",
        "holes_completed_pct",
        "confidence_weight",
        "holes_remaining",
        # Course difficulty ahead
        "remaining_difficulty",
        "hard_holes_remaining",
        "birdie_holes_remaining",
        # Weather
        "current_wind_speed",
        "temperature",
        "precipitation",
        "weather_severity_score",
        "morning_afternoon",
        # Pre-tournament baseline
        "win_prob",
        "top10_prob",
        "dg_rank",
        "model_score",
        "stage2_prob_raw",
        "augusta_competitive_rounds",
        "augusta_experience_tier",
        "augusta_made_cut_prev_year",
        "augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
        # Interactions
        "wind_x_difficulty",
        "projected_final_score",
    ]

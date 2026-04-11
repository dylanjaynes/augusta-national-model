"""
Live inference script for the Augusta National Model.

Takes current tournament state (player scores through N holes) and produces
updated win/top10/top20 probabilities.

Can be run every 5-10 minutes during the tournament.

Usage:
    # From a CSV/JSON of current scores:
    python3 scripts/run_live_inference.py --scores path/to/live_scores.csv

    # From DataGolf live tournament stats API:
    python3 scripts/run_live_inference.py --dg-live

    # Demo mode with test data:
    python3 scripts/run_live_inference.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from augusta_model.features.live_features import (
    _load_course_stats,
    _get_weather_for_round,
    compute_snapshot_features,
    add_pretournament_baseline,
)
from augusta_model.model.live_model import (
    load_live_model,
    _prepare_features,
    predict_live,
)
from augusta_model.simulation.remaining_rounds_mc import simulate_remaining_rounds

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "live"


def _find_data_file(filename: str) -> Path:
    """Return path to data file, checking worktree fallback paths."""
    local = DATA_DIR / filename
    if local.exists():
        return local
    # Worktree fallback: walk up to find the actual repo root
    for parent in ROOT.parents:
        candidate = parent / "data" / "processed" / filename
        if candidate.exists():
            return candidate
    return local  # Return local path so error message is clear

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATAGOLF_API_KEY = os.getenv("DATAGOLF_API_KEY", "91328c2c8e115c9e9b13461a8c3f")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "78fcb125d1413e1e832f024e1335ff3a")

# DataGolf live stats endpoint
DG_LIVE_URL = "https://feeds.datagolf.com/preds/live-tournament-stats"


def prob_to_american(p: float) -> str:
    """Convert win probability to American odds string (+150, -200, etc.)."""
    if pd.isna(p) or p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        return f"{round(-p / (1 - p) * 100):+d}"
    else:
        return f"+{round((1 - p) / p * 100)}"


def fetch_live_book_odds() -> pd.DataFrame:
    """
    Fetch current live outright odds from The Odds API.
    Returns DataFrame: player_name, book_american_win (best available across DK/FD/MGM).
    """
    url = "https://api.the-odds-api.com/v4/sports/golf_masters_tournament_winner/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "outrights",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    # Collect best (highest) American odds per player across all books
    best_odds: dict[str, int] = {}
    book_used: dict[str, str] = {}
    for event in data:
        for book in event.get("bookmakers", []):
            book_name = book["key"]
            for market in book.get("markets", []):
                if market["key"] != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = _normalize_name(outcome["name"])
                    price = outcome["price"]
                    # Higher American odds = better value for bettor; track best
                    if name not in best_odds or price > best_odds[name]:
                        best_odds[name] = price
                        book_used[name] = book_name

    if not best_odds:
        return pd.DataFrame()

    rows = [
        {"player_name": name, "book_american_win": odds, "best_book": book_used[name]}
        for name, odds in best_odds.items()
    ]
    df = pd.DataFrame(rows)
    # Convert American to implied probability
    df["book_implied_win"] = df["book_american_win"].apply(
        lambda x: 100 / (x + 100) if x >= 0 else (-x) / (-x + 100)
    )
    print(f"  Live book odds: {len(df)} players from Odds API")
    return df


def _normalize_name(name: str) -> str:
    """Convert 'Last, First' → 'First Last'. Leaves 'First Last' unchanged."""
    name = name.strip()
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name


_DG_SKIP_KEYS = {"dg_id", "player_name", "datagolf"}
_PREFERRED_BOOKS = {"draftkings", "fanduel", "betmgm", "bet365", "caesars",
                    "bovada", "betonline", "pointsbet"}


def fetch_dg_placement_odds(market: str) -> pd.DataFrame:
    """
    Fetch real sportsbook T5 / T10 / T20 odds from DataGolf betting-tools outrights.

    market: 'top_5' | 'top_10' | 'top_20'

    Returns DataFrame: player_name, book_{market}_american (int), book_{market}_implied (float),
    best_book_{market} (str with book name).

    Best odds = highest American value (least negative for favourites, most positive for dogs).
    Preferred books (DK/FD/MGM/365/Caesars/Bovada/Betonline/PointsBet) are tried first;
    falls back to any book if preferred books have no line.
    """
    url = "https://feeds.datagolf.com/betting-tools/outrights"
    params = {
        "market": market,
        "tour": "pga",
        "odds_format": "american",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  DG {market} odds error: {e}")
        return pd.DataFrame()

    players = data.get("odds", [])
    if not players:
        return pd.DataFrame()

    def _parse_american(val) -> int | None:
        if val is None or str(val).strip().lower() in ("n/a", "", "null"):
            return None
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return None

    rows = []
    for p in players:
        name = _normalize_name(p.get("player_name", ""))
        if not name:
            continue

        # Collect all valid (book, odds) pairs
        book_odds: list[tuple[str, int]] = []
        for key, val in p.items():
            if key in _DG_SKIP_KEYS:
                continue
            parsed = _parse_american(val)
            if parsed is not None:
                book_odds.append((key, parsed))

        if not book_odds:
            continue

        # Prefer regulated US books; fall back to any
        preferred = [(b, o) for b, o in book_odds if b in _PREFERRED_BOOKS]
        candidates = preferred if preferred else book_odds
        best_book, best_odds = max(candidates, key=lambda x: x[1])

        # Implied probability from American odds
        if best_odds >= 0:
            implied = 100.0 / (best_odds + 100.0)
        else:
            implied = (-best_odds) / (-best_odds + 100.0)

        rows.append({
            "player_name":               name,
            f"book_{market}_american":   best_odds,
            f"book_{market}_implied":    round(implied, 6),
            f"best_book_{market}":       best_book,
        })

    df = pd.DataFrame(rows)
    print(f"  DG {market} odds: {len(df)} players (books: {set(r[2] for r in [(r['player_name'], r.get(f'book_{market}_american'), r.get(f'best_book_{market}')) for _, r in df.iterrows()])})")
    return df


def fetch_dg_live_scores() -> pd.DataFrame:
    """
    Fetch live tournament stats from DataGolf API (live-tournament-stats endpoint).

    Returns DataFrame with columns:
        player_name, dg_id, round, holes_completed, current_score_to_par
        plus SG stats: sg_ott, sg_app, sg_arg, sg_putt, sg_t2g, sg_total
    """
    params = {
        "stats": "sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_bs,sg_total",
        "round": "event_cumulative",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    try:
        resp = requests.get(DG_LIVE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  DG live API error: {e}")
        return pd.DataFrame()

    # API returns live_stats key (not data)
    if not data or "live_stats" not in data:
        print(f"  DG live API: unexpected response format, keys={list(data.keys()) if data else []}")
        return pd.DataFrame()

    print(f"  Live data as of: {data.get('last_updated', 'unknown')}")
    print(f"  Event: {data.get('event_name', 'unknown')}")

    rows = []
    for player in data.get("live_stats", []):
        rows.append({
            "player_name": _normalize_name(player.get("player_name", "")),
            "dg_id": player.get("dg_id"),
            # 'total' = cumulative score to par, 'round' = today's score
            "round": 1,  # current_round not in this endpoint; use thru to infer
            "holes_completed": player.get("thru", 0) or 0,
            "current_score_to_par": player.get("total", 0) or 0,
            "today_score": player.get("round", 0) or 0,
            "position": player.get("position", ""),
            # SG stats (event cumulative)
            "sg_ott": player.get("sg_ott"),
            "sg_app": player.get("sg_app"),
            "sg_arg": player.get("sg_arg"),
            "sg_putt": player.get("sg_putt"),
            "sg_t2g": player.get("sg_t2g"),
            "sg_total": player.get("sg_total"),
        })

    return pd.DataFrame(rows)


def fetch_dg_inplay_predictions() -> pd.DataFrame:
    """
    Fetch DataGolf's own in-play model predictions.
    Returns win/top5/top10/top20/make_cut probabilities per player.
    """
    url = "https://feeds.datagolf.com/preds/in-play"
    params = {
        "tour": "pga",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  DG in-play API error: {e}")
        return pd.DataFrame()

    rows = []
    for player in data.get("data", []):
        rows.append({
            "player_name": _normalize_name(player.get("player_name", "")),
            "dg_id": player.get("dg_id"),
            "current_round": player.get("round", 1),
            "current_pos": player.get("current_pos", ""),
            "current_score": player.get("current_score", 0) or 0,
            "today": player.get("today", 0) or 0,
            "thru": player.get("thru", 0) or 0,
            "dg_win_prob": player.get("win", 0) or 0,
            "dg_top5_prob": player.get("top_5", 0) or 0,
            "dg_top10_prob": player.get("top_10", 0) or 0,
            "dg_top20_prob": player.get("top_20", 0) or 0,
            "dg_make_cut": player.get("make_cut", 0) or 0,
        })

    return pd.DataFrame(rows)


def parse_live_scores_csv(path: str) -> pd.DataFrame:
    """
    Parse a CSV of live scores. Expected columns:
        player_name, round, holes_completed, score_to_par
    Optionally: hole_1...hole_18 (individual hole scores)
    """
    df = pd.read_csv(path)
    required = ["player_name", "round", "holes_completed", "score_to_par"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def build_snapshot_from_live_scores(
    live_df: pd.DataFrame,
    course_stats: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    current_round: int = 1,
    year: int = 2026,
) -> pd.DataFrame:
    """
    Convert live score data into a feature snapshot DataFrame.

    live_df: DataFrame with player_name, holes_completed, score_to_par,
             and optionally individual hole scores.
    """
    # Load hole-by-hole data as a proxy for detailed scoring
    hbh_path = _find_data_file("masters_hole_by_hole.parquet")
    has_hbh = hbh_path.exists()

    rows = []
    for _, player_row in live_df.iterrows():
        player_name = str(player_row["player_name"])
        holes_done = int(player_row.get("holes_completed", 0))
        # Support both field names (live API uses current_score_to_par)
        total_score = int(
            player_row.get("current_score_to_par")
            or player_row.get("score_to_par")
            or 0
        )

        # Build a synthetic hole-by-hole if we only have aggregate scores
        # Distribute the total score proportionally across known avg difficulty
        if has_hbh:
            # Try to use actual hole-level scores from live data if provided
            hole_cols = [c for c in player_row.index if str(c).startswith("hole_")]
            if hole_cols and len(hole_cols) >= holes_done:
                # We have individual hole scores — use them directly
                hole_records = []
                for i, hcol in enumerate(hole_cols[:holes_done]):
                    hole_num = i + 1
                    par_row = course_stats[course_stats["hole_number"] == hole_num]
                    par = int(par_row.iloc[0]["par"]) if len(par_row) > 0 else 4
                    raw_score = int(player_row[hcol]) if not pd.isna(player_row[hcol]) else par
                    hole_records.append({
                        "hole_number": hole_num,
                        "par": par,
                        "score": raw_score,
                        "score_to_par": raw_score - par,
                    })
                player_holes = pd.DataFrame(hole_records)
            else:
                # Synthetic: spread total score proportionally across avg difficulty
                player_holes = _synthesize_hole_scores(total_score, holes_done, course_stats)
        else:
            player_holes = _synthesize_hole_scores(total_score, holes_done, course_stats)

        weather_row = _get_weather_for_round(weather_hourly, year, current_round)
        feat = compute_snapshot_features(
            player_holes=player_holes,
            snapshot_hole=holes_done,
            course_stats=course_stats,
            weather_row=weather_row,
        )
        feat["player_name"] = player_name
        feat["round_num"] = current_round
        feat["year"] = year
        feat["snapshot_hole"] = holes_done

        # For live scoring: total to par across all rounds
        if "total_score_to_par" in player_row.index:
            feat["total_score_to_par"] = int(player_row["total_score_to_par"])
        else:
            feat["total_score_to_par"] = total_score

        rows.append(feat)

    return pd.DataFrame(rows)


def _synthesize_hole_scores(
    total_score_to_par: int,
    holes_done: int,
    course_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Synthesize hole-level scores from aggregate score + holes completed.
    Distributes total score proportionally to historical difficulty.
    """
    if holes_done == 0:
        return pd.DataFrame(columns=["hole_number", "par", "score", "score_to_par"])

    played_holes = list(range(1, holes_done + 1))
    played_stats = course_stats[course_stats["hole_number"].isin(played_holes)].copy()

    if played_stats.empty:
        return pd.DataFrame(columns=["hole_number", "par", "score", "score_to_par"])

    # Distribute total score proportionally by avg difficulty
    total_avg_diff = played_stats["avg_score_to_par"].sum()
    records = []
    distributed = 0
    for i, (_, hrow) in enumerate(played_stats.iterrows()):
        if i < len(played_stats) - 1:
            # Proportional allocation
            if total_avg_diff != 0:
                share = hrow["avg_score_to_par"] / total_avg_diff
                hole_stp = int(round(total_score_to_par * share))
            else:
                hole_stp = 0
            distributed += hole_stp
        else:
            # Last hole gets the remainder
            hole_stp = total_score_to_par - distributed

        par = 4  # default
        records.append({
            "hole_number": int(hrow["hole_number"]),
            "par": par,
            "score": par + hole_stp,
            "score_to_par": hole_stp,
        })

    return pd.DataFrame(records)


def generate_demo_scores(predictions_df: pd.DataFrame, round_num: int = 1) -> pd.DataFrame:
    """
    Generate synthetic 'live' scores for demo purposes.
    Uses historical scoring variance to create realistic snapshots.
    """
    np.random.seed(42 + round_num)
    players = predictions_df["player_name"].head(20).tolist()

    rows = []
    for player in players:
        # Simulate random holes completed (3-18)
        holes = np.random.choice([3, 6, 9, 12, 15, 18])
        # Simulate score: better players have slightly lower scores
        player_row = predictions_df[predictions_df["player_name"] == player]
        skill = float(player_row["model_score"].iloc[0]) if len(player_row) > 0 else 0.3
        avg_stp = (0.5 - skill) * 2  # better players average under par
        score = int(np.random.normal(avg_stp * holes / 18, 1.5))
        rows.append({
            "player_name": player,
            "round": round_num,
            "holes_completed": holes,
            "score_to_par": score,
        })

    return pd.DataFrame(rows)


def run_inference(
    live_scores: pd.DataFrame,
    predictions_df: pd.DataFrame,
    course_stats: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    current_round: int = 1,
    year: int = 2026,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main inference function. Returns updated probability DataFrame.
    """
    print(f"  Building feature snapshots for {len(live_scores)} players...")
    snapshot_df = build_snapshot_from_live_scores(
        live_df=live_scores,
        course_stats=course_stats,
        weather_hourly=weather_hourly,
        current_round=current_round,
        year=year,
    )

    print(f"  Loading live model...")
    clf, reg, feature_cols, metadata = load_live_model()

    print(f"  Running predictions...")
    results = predict_live(
        snapshot_df=snapshot_df,
        clf=clf,
        reg=reg,
        feature_cols=feature_cols,
        predictions_2026=predictions_df,
    )

    # Add current score info for display
    if "total_score_to_par" in snapshot_df.columns:
        score_map = snapshot_df.set_index("player_name")["total_score_to_par"]
        results["current_score_to_par"] = results["player_name"].map(score_map)
    elif "cumulative_score_to_par" in snapshot_df.columns:
        score_map = snapshot_df.set_index("player_name")["cumulative_score_to_par"]
        results["current_score_to_par"] = results["player_name"].map(score_map)

    # Join pre-tournament odds for edge comparison
    if "market_win" in predictions_df.columns:
        mkt_map = predictions_df.set_index("player_name")[["market_win", "market_top10"]].to_dict()
        results["market_win"] = results["player_name"].map(mkt_map["market_win"])
        results["market_top10"] = results["player_name"].map(mkt_map["market_top10"])
        results["win_edge_vs_market"] = results["blended_win_prob"] - results["market_win"].fillna(0)

    # American odds for our model predictions
    results["model_american_win"] = results["blended_win_prob"].apply(prob_to_american)
    results["model_american_top10"] = (results["blended_top10_prob"] / 10).apply(prob_to_american)

    # Sort by win probability descending, assign rank
    results = results.sort_values("blended_win_prob", ascending=False).reset_index(drop=True)
    results["live_rank"] = range(1, len(results) + 1)

    # Pre-tournament rank for comparison
    if "win_prob" in predictions_df.columns:
        pre_rank_map = (
            predictions_df.sort_values("win_prob", ascending=False)
            .reset_index(drop=True)
            .assign(pre_rank=lambda x: range(1, len(x) + 1))
            .set_index("player_name")["pre_rank"]
        )
        results["pre_tournament_rank"] = results["player_name"].map(pre_rank_map)
        results["rank_change"] = results["pre_tournament_rank"] - results["live_rank"]

    if verbose:
        print(f"\n{'Rank':>4} | {'Player':<25} | {'Score':>6} | {'Holes':>5} | {'T10%':>6} | {'Win%':>5} | {'Model Odds':>11} | {'Move':>5}")
        print("-" * 85)
        for _, row in results.head(20).iterrows():
            score_str = f"{int(row.get('current_score_to_par', 0)):+d}" if pd.notna(row.get('current_score_to_par')) else "  E"
            holes_str = f"{int(row.get('holes_completed', 0))}/18"
            move = row.get("rank_change", 0)
            move_str = f"{int(move):+d}" if pd.notna(move) else " --"
            print(
                f"{row['live_rank']:4d} | {row['player_name']:<25} | "
                f"{score_str:>6} | {holes_str:>5} | "
                f"{row['blended_top10_prob']:6.1%} | "
                f"{row['blended_win_prob']:5.1%} | "
                f"{row['model_american_win']:>11} | "
                f"{move_str:>5}"
            )

    return results


def apply_position_correction(
    results: pd.DataFrame,
    current_round: int = 1,
) -> pd.DataFrame:
    """
    Retired — position is now handled by the progressive DG blend in run_once().
    Kept as a no-op so call sites don't need updating.
    """
    return results


def save_results(results: pd.DataFrame, suffix: str = "") -> Path:
    """Save inference results to live data directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"live_predictions_{timestamp}{suffix}.csv"
    out_path = OUTPUT_DIR / fname
    results.to_csv(out_path, index=False)

    # Also save as 'latest' for Streamlit to read
    latest_path = OUTPUT_DIR / "live_predictions_latest.csv"
    results.to_csv(latest_path, index=False)

    print(f"\nResults saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run live in-tournament inference")
    parser.add_argument("--scores", help="Path to CSV of live scores")
    parser.add_argument("--dg-live", action="store_true", help="Fetch from DataGolf live API")
    parser.add_argument("--demo", action="store_true", help="Demo mode with synthetic scores")
    parser.add_argument("--round", type=int, default=1, help="Current round (1-4)")
    parser.add_argument("--year", type=int, default=2026, help="Tournament year")
    parser.add_argument("--loop", action="store_true", help="Run every 5 minutes continuously")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds")
    args = parser.parse_args()

    print("Loading base data...")
    predictions_df = pd.read_parquet(_find_data_file("predictions_2026.parquet"))
    try:
        weather_hourly = pd.read_parquet(_find_data_file("masters_weather_hourly.parquet"))
    except FileNotFoundError:
        print("  Warning: masters_weather_hourly.parquet not found — using empty weather")
        weather_hourly = pd.DataFrame()
    hbh = pd.read_parquet(_find_data_file("masters_hole_by_hole.parquet"))
    course_stats = _load_course_stats(hbh[hbh["year"] <= 2025])

    print(f"  {len(predictions_df)} players in pre-tournament predictions")
    print(f"  {len(weather_hourly)} hourly weather rows")

    def run_once():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running live inference...")

        if args.scores:
            live_scores = parse_live_scores_csv(args.scores)
            dg_inplay = pd.DataFrame()
        elif args.dg_live:
            live_scores = fetch_dg_live_scores()
            if live_scores.empty:
                print("  No live data returned from DG API")
                return
            print("  Fetching DG in-play predictions...")
            dg_inplay = fetch_dg_inplay_predictions()
        elif args.demo:
            live_scores = generate_demo_scores(predictions_df, round_num=args.round)
            dg_inplay = pd.DataFrame()
        else:
            print("Specify --scores, --dg-live, or --demo")
            parser.print_help()
            return

        # Infer current round from in-play data if available
        if not dg_inplay.empty and "current_round" in dg_inplay.columns:
            round_mode = dg_inplay["current_round"].mode()
            current_round = int(round_mode.iloc[0]) if len(round_mode) > 0 else args.round
            if current_round != args.round:
                print(f"  Auto-detected round {current_round} from DG in-play data")
        else:
            current_round = args.round

        results = run_inference(
            live_scores=live_scores,
            predictions_df=predictions_df,
            course_stats=course_stats,
            weather_hourly=weather_hourly,
            current_round=current_round,
            year=args.year,
        )

        # Merge live SG columns back (lost during snapshot feature building)
        sg_cols = ["player_name", "sg_ott", "sg_app", "sg_arg", "sg_putt",
                   "sg_t2g", "sg_total"]
        if all(c in live_scores.columns for c in ["sg_app", "sg_total"]):
            results = results.merge(
                live_scores[[c for c in sg_cols if c in live_scores.columns]],
                on="player_name", how="left",
            )

        # Merge DG in-play predictions for richer output
        if not dg_inplay.empty:
            results = results.merge(
                dg_inplay[["player_name", "current_pos", "current_score", "thru",
                            "today", "dg_win_prob", "dg_top5_prob", "dg_top10_prob",
                            "dg_top20_prob", "dg_make_cut"]],
                on="player_name",
                how="left",
            )

            # ── Use DG cumulative score as display score ──────────────────────
            if "current_score" in results.columns:
                results["current_score_to_par"] = results["current_score"]

            # ── Remaining-rounds Monte Carlo ──────────────────────────────────
            # Simulate rounds 3+4 for every player using:
            #   - Their Augusta-specific skill from pre-tournament model (sg, model_score)
            #   - Historical Augusta scoring distribution (mean +0.5, std 3.05/round)
            #   - Field-wide correlation for weather/pin conditions
            # Win% = fraction of 20k simulations where player finishes lowest.
            # This naturally handles margin of lead, what each player needs to shoot,
            # and the historical impossibility of -8 rounds at Augusta.
            median_thru = results["thru"].fillna(0).median() if "thru" in results.columns else 18.0
            total_holes = (current_round - 1) * 18 + median_thru
            print(f"  Running remaining-rounds MC ({total_holes:.0f} total holes complete)...")

            mc_results = simulate_remaining_rounds(
                live_df=results,
                pre_df=predictions_df,
                current_round=current_round,
                median_thru=median_thru,
                n_sims=20_000,
            )

            # Merge MC probabilities back onto results (including new percentile/scenario cols)
            mc_merge_cols = ["player_name", "mc_win_prob", "mc_top5_prob",
                             "mc_top10_prob", "mc_projected_total", "strokes_back"]
            for c in ["expected_score_per_round", "inweek_mean",
                      "mc_proj_p25", "mc_proj_p75", "mc_proj_p90",
                      "mc_collapse_prob", "mc_win_scenario_score"]:
                if c in mc_results.columns:
                    mc_merge_cols.append(c)
            results = results.merge(mc_results[mc_merge_cols], on="player_name", how="left")

            # MC is now our primary win signal; pre-tournament XGBoost adds Augusta
            # skill insight especially early in the tournament.
            # Blend: MC (position-aware) × pre-tournament Augusta skill weight
            # Weight schedule: after R1 50/50, after R2 70% MC, R3 85% MC, R4 95% MC
            if total_holes <= 18:
                mc_weight = 0.35 + 0.15 * (total_holes / 18)   # 0.35→0.50
            elif total_holes <= 36:
                mc_weight = 0.50 + 0.20 * ((total_holes - 18) / 18)  # 0.50→0.70
            elif total_holes <= 54:
                mc_weight = 0.70 + 0.15 * ((total_holes - 36) / 18)  # 0.70→0.85
            else:
                mc_weight = 0.85 + 0.10 * ((total_holes - 54) / 18)  # 0.85→0.95
            mc_weight = min(mc_weight, 0.95)
            xgb_weight = 1.0 - mc_weight
            print(f"  MC blend: {mc_weight:.0%} remaining-rounds MC + {xgb_weight:.0%} Augusta XGBoost")

            mc_win  = results["mc_win_prob"].fillna(0)
            mc_t10  = results["mc_top10_prob"].fillna(0) * 10
            xgb_win = results["blended_win_prob"].fillna(0)
            xgb_t10 = results["blended_top10_prob"].fillna(0)

            results["blended_win_prob"]   = mc_weight * mc_win  + xgb_weight * xgb_win
            results["blended_top10_prob"] = mc_weight * mc_t10  + xgb_weight * xgb_t10

            # Renormalise
            win_sum = results["blended_win_prob"].sum()
            t10_sum = results["blended_top10_prob"].sum()
            if win_sum > 0:
                results["blended_win_prob"] = results["blended_win_prob"] / win_sum
            if t10_sum > 0:
                results["blended_top10_prob"] = results["blended_top10_prob"] * (10.0 / t10_sum)

            # Edge vs DG
            results["edge_vs_dg_win"]   = results["blended_win_prob"] - results["dg_win_prob"].fillna(0)
            results["edge_vs_dg_top10"] = results["blended_top10_prob"] - results["dg_top10_prob"].fillna(0) * 10

            # Recompute American odds and re-sort
            results["model_american_win"]   = results["blended_win_prob"].apply(prob_to_american)
            results["model_american_top10"] = (results["blended_top10_prob"] / 10).apply(prob_to_american)
            results = results.sort_values("blended_win_prob", ascending=False).reset_index(drop=True)
            results["live_rank"] = range(1, len(results) + 1)

        # Fetch and merge live book odds (win + placement markets)
        print("  Fetching live book odds (win, top-5, top-10, top-20)...")
        book_odds = fetch_live_book_odds()
        if not book_odds.empty:
            results = results.merge(
                book_odds[["player_name", "book_american_win", "book_implied_win", "best_book"]],
                on="player_name",
                how="left",
            )

        for market in ("top_5", "top_10", "top_20"):
            placement_odds = fetch_dg_placement_odds(market)
            if not placement_odds.empty:
                results = results.merge(placement_odds, on="player_name", how="left")

        # Apply position correction AFTER all merges so scores are present
        print("  Applying leaderboard position correction...")
        results = apply_position_correction(results, current_round=current_round)

        # Recompute book edge using corrected win probs
        if "book_implied_win" in results.columns:
            results["live_edge_vs_book"] = (
                results["blended_win_prob"] - results["book_implied_win"].fillna(0)
            )

        results["current_round"] = current_round
        save_results(results)

    if args.loop:
        print(f"Polling every {args.interval} seconds. Ctrl+C to stop.")
        while True:
            run_once()
            print(f"  Waiting {args.interval}s...")
            time.sleep(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    main()

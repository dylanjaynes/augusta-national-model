#!/usr/bin/env python3
"""
Augusta National Model — V2 Pipeline
Session 2: Par-5/scoring features, weather, top-10 model, betting edge, SHAP
"""
import os
import sys
import json
import time
import requests
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, precision_recall_curve
import xgboost as xgb
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_ROUNDS_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
DG_BASE = "https://feeds.datagolf.com"
API_KEY = os.getenv("DATAGOLF_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Augusta par = 72
AUGUSTA_PAR = 72

XGB_PARAMS = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:squarederror",
    "random_state": 42,
}
XGB_CLF_PARAMS = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "binary:logistic",
    "eval_metric": "auc", "random_state": 42,
    "scale_pos_weight": 8.0,  # ~11% positive rate → ~9:1 imbalance
}
N_SIMULATIONS = 50_000
NOISE_STD = 0.16
TARGET_PRED_STD = 0.10
SEED = 42
HARDCODED_WEIGHTS = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

ROLLING_FEATURES = [
    "sg_ott_3w", "sg_app_3w", "sg_arg_3w", "sg_putt_3w", "sg_t2g_3w", "sg_total_3w",
    "sg_ott_8w", "sg_app_8w", "sg_arg_8w", "sg_putt_8w", "sg_t2g_8w", "sg_total_8w",
    "sg_total_std_8", "sg_total_momentum", "sg_ball_strike_ratio", "log_field_size",
]

MASTERS_THURSDAYS = {
    2021: "2021-04-08", 2022: "2022-04-07", 2023: "2023-04-06",
    2024: "2024-04-11", 2025: "2025-04-10",
}


def _parse_finish_num(pos):
    if pos is None or pd.isna(pos):
        return None
    pos = str(pos).strip().upper()
    if pos in ("CUT", "MC", "WD", "DQ", "MDF"):
        return 999
    pos = pos.replace("T", "").replace("=", "")
    try:
        return int(pos)
    except (ValueError, TypeError):
        return None


def _dg_get_safe(endpoint, params=None):
    if params is None:
        params = {}
    url = f"{DG_BASE}/{endpoint}"
    params = {"key": API_KEY, "file_format": "json", **params}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code in (403, 404):
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# TASK 1 — PAR-5 AND SCORING PATTERN FEATURES
# ═══════════════════════════════════════════════════════════

def task1_scoring_features():
    print("\n" + "=" * 60)
    print("TASK 1 — PAR-5 AND SCORING PATTERN FEATURES")
    print("=" * 60)

    rounds_df = pd.read_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet")
    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)

    # ── Compute round-level scoring features from masters_sg_rounds ──
    # score column is score relative to par for that round (e.g., -3 means 3 under)
    # Augusta par = 72, so actual score = 72 + score_relative_to_par
    # But our 'score' column IS the relative-to-par score already

    print("\n  Computing scoring pattern features from round-level data...")

    # Per player per season: aggregate round scores
    player_season_rounds = rounds_df.groupby(["player_name", "season"])

    scoring_rows = []
    for (player, season), grp in player_season_rounds:
        scores = grp["score"].dropna()
        sg_totals = grp["sg_total"].dropna()

        if len(scores) == 0:
            continue

        # augusta_birdie_rate: estimate birdies from round score
        # If round score is -3 (3 under par on 18 holes), estimate ~4 birdies
        # Heuristic: birdies_est = max(0, -score + 1) for under-par rounds
        # For over-par: assume 1-2 birdies offset by bogeys
        birdie_estimates = []
        for s in scores:
            if s <= -4:
                birdie_estimates.append(max(0, -s + 1))
            elif s <= -1:
                birdie_estimates.append(max(0, -s + 0.5))
            elif s <= 1:
                birdie_estimates.append(1.5)
            else:
                birdie_estimates.append(max(0, 2 - s * 0.3))
        avg_birdies = np.mean(birdie_estimates) / 18.0  # rate per hole

        # augusta_bogey_avoidance: rate of clean rounds (at or under par)
        clean_rounds = (scores <= 0).sum()
        bogey_avoidance = clean_rounds / len(scores)

        # augusta_round_variance: std of round scores
        round_variance = scores.std() if len(scores) > 1 else None

        # augusta_back9_scoring: R3+R4 scores vs field R3+R4 median
        r34 = grp[grp["round_num"].isin([3, 4])]["score"].dropna()
        # Get field R3+R4 median for this season
        field_r34 = rounds_df[
            (rounds_df["season"] == season) &
            (rounds_df["round_num"].isin([3, 4]))
        ]["score"].dropna()
        if len(r34) > 0 and len(field_r34) > 0:
            back9_scoring = r34.mean() - field_r34.median()
        else:
            back9_scoring = None

        scoring_rows.append({
            "player_name": player,
            "season": season,
            "augusta_birdie_rate": avg_birdies,
            "augusta_bogey_avoidance": bogey_avoidance,
            "augusta_round_variance": round_variance,
            "augusta_back9_scoring": back9_scoring,
        })

    scoring_df = pd.DataFrame(scoring_rows)
    print(f"  Computed scoring features for {len(scoring_df)} player-seasons")

    # ── Build temporal lookback features ──
    # For each player-season in unified, compute lookback using ONLY prior data
    print("  Building temporal lookback scoring features...")

    features_df = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")

    new_feat_rows = []
    for _, row in unified_df.iterrows():
        player = row["player_name"]
        season = row["season"]

        prior_scoring = scoring_df[
            (scoring_df["player_name"] == player) &
            (scoring_df["season"] < season)
        ]

        if len(prior_scoring) > 0:
            decay = 0.75 ** (season - prior_scoring["season"])

            br = prior_scoring["augusta_birdie_rate"].dropna()
            birdie_rate = np.average(br, weights=decay[br.index]) if len(br) > 0 else None

            ba = prior_scoring["augusta_bogey_avoidance"].dropna()
            bogey_avoid = np.average(ba, weights=decay[ba.index]) if len(ba) > 0 else None

            rv = prior_scoring["augusta_round_variance"].dropna()
            round_var = np.average(rv, weights=decay[rv.index]) if len(rv) > 0 else None

            b9 = prior_scoring["augusta_back9_scoring"].dropna()
            back9 = np.average(b9, weights=decay[b9.index]) if len(b9) > 0 else None
        else:
            birdie_rate = None
            bogey_avoid = None
            round_var = None
            back9 = None

        new_feat_rows.append({
            "player_name": player,
            "season": season,
            "augusta_birdie_rate": birdie_rate,
            "augusta_bogey_avoidance": bogey_avoid,
            "augusta_round_variance_score": round_var,
            "augusta_back9_scoring": back9,
        })

    new_feats = pd.DataFrame(new_feat_rows)

    # ── Probe DG approach-skill endpoint for par-5 data ─���
    print("\n  Probing DG approach-skill endpoint for par-5 data...")
    par5_data = _dg_get_safe("historical-raw-data/approach-skill",
                              {"event_id": "all", "year": 2023})
    if par5_data is not None:
        if isinstance(par5_data, list) and len(par5_data) > 0:
            sample = par5_data[0]
            print(f"  approach-skill response type: list, sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'not dict'}")
        elif isinstance(par5_data, dict):
            print(f"  approach-skill response keys: {list(par5_data.keys())}")
        else:
            print(f"  approach-skill returned: {type(par5_data)}")
    else:
        print("  approach-skill endpoint not available (403/404)")

    # Also try preds/approach-skill-by-distance
    approach_dist = _dg_get_safe("preds/approach-skill-by-distance")
    if approach_dist is not None:
        if isinstance(approach_dist, dict):
            print(f"  approach-skill-by-distance keys: {list(approach_dist.keys())[:10]}")
        else:
            print(f"  approach-skill-by-distance returned: {type(approach_dist)}")
    else:
        print("  approach-skill-by-distance not available")

    print("  Par-5 specific SG not available via API — using scoring proxies instead")

    # Merge new features into existing features
    features_df = features_df.merge(
        new_feats, on=["player_name", "season"], how="left", suffixes=("_old", "")
    )
    # Drop old back9 scoring if duplicated
    for col in features_df.columns:
        if col.endswith("_old"):
            features_df.drop(columns=[col], inplace=True)

    features_df.to_parquet(PROCESSED_DIR / "augusta_player_features.parquet", index=False)

    non_null = new_feats[["augusta_birdie_rate", "augusta_bogey_avoidance",
                           "augusta_round_variance_score", "augusta_back9_scoring"]].notna().sum()
    print(f"\n  Feature coverage:")
    for col, count in non_null.items():
        print(f"    {col}: {count}/{len(new_feats)} rows")

    print("\n  Task 1 complete.")
    return features_df


# ═══════════════════════════════════════════════════════════
# TASK 2 — ADDITIONAL DATA SOURCES
# ═══════════════════════════════════════════════════════════

def task2_additional_data():
    print("\n" + "=" * 60)
    print("TASK 2 — ADDITIONAL DATA SOURCES")
    print("=" * 60)

    # ── PART A: PGA Tour proximity data ──
    print("\n  PART A — PGA Tour proximity data")
    print("  Probing PGA Tour GraphQL API...")

    proximity_available = False
    try:
        # Try PGA Tour stats page
        r = requests.get(
            "https://www.pgatour.com/stats/detail/02330",
            timeout=15,
            headers={"User-Agent": "AugustaModel/1.0"}
        )
        if r.status_code == 200 and len(r.text) > 1000:
            # Check if it's JS-rendered or has data
            if "proximity" in r.text.lower() or "approach" in r.text.lower():
                print("  PGA Tour stats page returned HTML but likely JS-rendered")
            else:
                print("  PGA Tour stats page: no proximity data in raw HTML")
        else:
            print(f"  PGA Tour stats page returned {r.status_code}")
    except Exception as e:
        print(f"  PGA Tour stats page unreachable: {e}")

    print("  PGA Tour proximity data requires JS rendering — skipping for now")
    print("  (Recommend Selenium/Playwright for future integration)")

    # ── PART B: DG skill ratings schema ──
    print("\n  PART B — DataGolf skill decomposition")

    skill_data = _dg_get_safe("preds/skill-ratings", {"display": "detailed"})
    if skill_data is not None:
        if isinstance(skill_data, dict):
            print(f"  skill-ratings keys: {list(skill_data.keys())}")
            players = skill_data.get("players", [])
            if players:
                sample = players[0]
                print(f"  Player sample keys: {list(sample.keys())}")
                # Look for shot shape / approach fields
                for k in sample.keys():
                    if any(term in k.lower() for term in ["approach", "draw", "fade", "shape", "direction"]):
                        print(f"    Shot shape field found: {k} = {sample[k]}")
        else:
            print(f"  skill-ratings returned: {type(skill_data)}")
    else:
        print("  skill-ratings endpoint not available")

    field_data = _dg_get_safe("field-updates")
    if field_data is not None and isinstance(field_data, dict):
        print(f"  field-updates keys: {list(field_data.keys())}")
        field_list = field_data.get("field", [])
        if field_list:
            print(f"  field player sample keys: {list(field_list[0].keys())}")
    else:
        print("  field-updates not available")

    # ── PART C: Weather data ──
    print("\n  PART C — Weather data (Open-Meteo)")

    weather_rows = []
    for year, thursday in MASTERS_THURSDAYS.items():
        # Compute Sunday = Thursday + 3 days
        from datetime import datetime, timedelta
        thu = datetime.strptime(thursday, "%Y-%m-%d")
        sunday = (thu + timedelta(days=3)).strftime("%Y-%m-%d")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 33.5, "longitude": -82.02,
            "start_date": thursday, "end_date": sunday,
            "daily": "wind_speed_10m_max,precipitation_sum,temperature_2m_mean,wind_direction_10m_dominant",
            "timezone": "America/New_York",
            "wind_speed_unit": "mph",
        }

        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            daily = data.get("daily", {})

            winds = [w for w in (daily.get("wind_speed_10m_max") or []) if w is not None]
            precips = [p for p in (daily.get("precipitation_sum") or []) if p is not None]
            temps = [t for t in (daily.get("temperature_2m_mean") or []) if t is not None]
            wind_dirs = daily.get("wind_direction_10m_dominant") or []

            wind_avg = np.mean(winds) if winds else None
            wind_max = max(winds) if winds else None
            rain_total = sum(precips) if precips else None
            temp_avg = np.mean(temps) if temps else None

            # Conditions bucket
            if wind_avg is not None:
                if rain_total and rain_total > 5:
                    conditions = "wet"
                elif wind_avg > 14:
                    conditions = "windy"
                elif wind_avg > 8:
                    conditions = "moderate"
                else:
                    conditions = "calm"
            else:
                conditions = "unknown"

            weather_rows.append({
                "season": year,
                "wind_avg_mph": round(wind_avg, 1) if wind_avg else None,
                "wind_max_mph": round(wind_max, 1) if wind_max else None,
                "rain_total_mm": round(rain_total, 1) if rain_total else None,
                "temp_avg_f": round(temp_avg * 9/5 + 32, 1) if temp_avg else None,
                "conditions_bucket": conditions,
            })

            print(f"  {year}: wind_avg={wind_avg:.1f}mph, wind_max={wind_max:.1f}mph, "
                  f"rain={rain_total:.1f}mm, conditions={conditions}")

        except Exception as e:
            print(f"  {year}: weather fetch failed: {e}")
            weather_rows.append({
                "season": year, "wind_avg_mph": None, "wind_max_mph": None,
                "rain_total_mm": None, "temp_avg_f": None, "conditions_bucket": "unknown",
            })

    weather_df = pd.DataFrame(weather_rows)
    weather_df.to_parquet(PROCESSED_DIR / "masters_weather.parquet", index=False)
    print(f"\n  Saved masters_weather.parquet: {len(weather_df)} rows")

    print("\n  Task 2 complete.")
    return weather_df


# ═══════════════════════════════════════════════════════════
# TASK 3 — TOP-10 SPECIFIC MODEL
# ═══════════════════════════════════════════════════════════

def _build_rolling_features(df):
    df = df.sort_values(["player_name", "date"]).copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
    for col in sg_cols:
        df[f"{col}_3w"] = df.groupby("player_name")[col].transform(
            lambda x: x.ewm(span=3, min_periods=1).mean())
        df[f"{col}_8w"] = df.groupby("player_name")[col].transform(
            lambda x: x.ewm(span=8, min_periods=1).mean())
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(
        lambda x: x.rolling(8, min_periods=2).std())
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))
    return df


def _apply_course_weights(df, weights):
    df = df.copy()
    for col, w in weights.items():
        if col in df.columns:
            df[col] = df[col] * w
    return df


def _compute_course_weights_for_year(unified_df, target_year):
    sg_rows = unified_df[
        (unified_df["has_sg_data"] == True) & (unified_df["season"] < target_year)
    ].copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    sg_rows = sg_rows.dropna(subset=sg_cols)
    if len(sg_rows) < 10:
        return HARDCODED_WEIGHTS.copy()
    sg_rows["finish_num"] = sg_rows["finish_pos"].apply(_parse_finish_num)
    sg_rows = sg_rows[sg_rows["finish_num"] < 999]
    sg_rows["finish_pct"] = ((sg_rows["finish_num"] - 1) / (sg_rows["field_size"] - 1)).clip(0, 1)
    ridge = Ridge(alpha=1.0)
    ridge.fit(sg_rows[sg_cols].values, sg_rows["finish_pct"].values)
    raw_coefs = dict(zip(sg_cols, ridge.coef_))
    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["fn"] = tour_df["finish_pos"].apply(_parse_finish_num)
    tour_df = tour_df[(tour_df["fn"] < 999) & (tour_df["season"] < target_year)].dropna(subset=sg_cols)
    tour_df["fp"] = ((tour_df["fn"] - 1) / (tour_df["field_size"] - 1)).clip(0, 1)
    rt = Ridge(alpha=1.0)
    rt.fit(tour_df[sg_cols].values, tour_df["fp"].values)
    tc = dict(zip(sg_cols, rt.coef_))
    computed = {c: abs(raw_coefs[c]) / abs(tc[c]) if abs(tc[c]) > 1e-6 else 1.0 for c in sg_cols}
    if len(sg_rows) < 150:
        return {c: 0.70 * computed[c] + 0.30 * HARDCODED_WEIGHTS[c] for c in sg_cols}
    return computed


def _build_augusta_features_for_year(unified_df, rounds_df, features_df, player_name, target_year, weather_df=None):
    """Build Augusta-specific features for a player using only prior data."""
    prior = unified_df[
        (unified_df["player_name"] == player_name) & (unified_df["season"] < target_year)
    ]
    features = {}
    n_starts = len(prior)
    features["augusta_starts"] = n_starts

    if n_starts > 0:
        decay = 0.75 ** (target_year - prior["season"])
        prior_cuts = prior["made_cut"].dropna()
        features["augusta_made_cut_rate"] = prior_cuts.mean() if len(prior_cuts) > 0 else 0.0
        prior_finish = prior["finish_num"].dropna()
        features["augusta_top10_rate"] = (prior_finish <= 10).mean() if len(prior_finish) > 0 else 0.0
        good = prior_finish[prior_finish < 999]
        features["augusta_best_finish"] = good.min() if len(good) > 0 else 99
        svf = prior["score_vs_field"].dropna()
        features["augusta_scoring_avg"] = np.average(svf, weights=decay[svf.index]) if len(svf) > 0 else 0.0

        prior_sg = prior[prior["has_sg_data"] == True]
        if len(prior_sg) > 0:
            sg_decay = 0.75 ** (target_year - prior_sg["season"])
            for col in ["sg_app", "sg_total"]:
                vals = prior_sg[col].dropna()
                features[f"augusta_{col}_career"] = np.average(vals, weights=sg_decay[vals.index]) if len(vals) > 0 else 0.0
        else:
            features["augusta_sg_app_career"] = 0.0
            features["augusta_sg_total_career"] = 0.0

        features["augusta_experience_bucket"] = min(n_starts, 3) if n_starts <= 2 else (2 if n_starts <= 5 else 3)
    else:
        features.update({
            "augusta_made_cut_rate": 0.0, "augusta_top10_rate": 0.0,
            "augusta_best_finish": 99, "augusta_scoring_avg": 0.0,
            "augusta_sg_app_career": 0.0, "augusta_sg_total_career": 0.0,
            "augusta_experience_bucket": 0,
        })

    # New scoring features from features_df
    feat_row = features_df[
        (features_df["player_name"] == player_name) & (features_df["season"] == target_year)
    ]
    if len(feat_row) > 0:
        feat_row = feat_row.iloc[0]
        for col in ["augusta_birdie_rate", "augusta_bogey_avoidance",
                     "augusta_round_variance_score", "augusta_back9_scoring"]:
            val = feat_row.get(col)
            features[col] = val if pd.notna(val) else 0.0
    else:
        features.update({
            "augusta_birdie_rate": 0.0, "augusta_bogey_avoidance": 0.0,
            "augusta_round_variance_score": 0.0, "augusta_back9_scoring": 0.0,
        })

    # Weather feature
    if weather_df is not None:
        w_row = weather_df[weather_df["season"] == target_year]
        features["tournament_wind_avg"] = w_row["wind_avg_mph"].iloc[0] if len(w_row) > 0 and pd.notna(w_row["wind_avg_mph"].iloc[0]) else 10.0
    else:
        features["tournament_wind_avg"] = 10.0

    return features


def _run_monte_carlo(predictions, n_sims=N_SIMULATIONS, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(predictions)
    base = predictions.copy()
    mu, sigma = base.mean(), base.std()
    z = (base - mu) / sigma if sigma > 0 else np.zeros(n)
    z = z * TARGET_PRED_STD
    wins = np.zeros(n); t5 = np.zeros(n); t10 = np.zeros(n); t20 = np.zeros(n)
    for _ in range(n_sims):
        sim = z + rng.normal(0, NOISE_STD, n)
        ranks = sim.argsort().argsort() + 1
        wins += (ranks == 1); t5 += (ranks <= 5); t10 += (ranks <= 10); t20 += (ranks <= 20)
    return {"win_prob": wins/n_sims, "top5_prob": t5/n_sims,
            "top10_prob": t10/n_sims, "top20_prob": t20/n_sims}


# Augusta-specific features used in Stage 2
AUGUSTA_FEATURES = [
    "augusta_starts", "augusta_made_cut_rate", "augusta_top10_rate",
    "augusta_best_finish", "augusta_scoring_avg",
    "augusta_sg_app_career", "augusta_sg_total_career", "augusta_experience_bucket",
    "augusta_birdie_rate", "augusta_bogey_avoidance",
    "augusta_round_variance_score", "augusta_back9_scoring",
    "tournament_wind_avg",
]

STAGE2_FEATURES = ROLLING_FEATURES + AUGUSTA_FEATURES


def task3_top10_model():
    print("\n" + "=" * 60)
    print("TASK 3 — TOP-10/20 SPECIFIC MODEL + FULL BACKTEST")
    print("=" * 60)

    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)
    rounds_df = pd.read_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet")
    features_df = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)

    weather_path = PROCESSED_DIR / "masters_weather.parquet"
    weather_df = pd.read_parquet(weather_path) if weather_path.exists() else None

    sg_years = sorted(unified_df[unified_df["has_sg_data"] == True]["season"].unique())
    all_years = sorted(unified_df["season"].unique())
    backtest_years = [y for y in sg_years if y > min(all_years)]
    print(f"\n  Backtest years: {backtest_years}")

    all_results = []
    year_metrics = []
    all_bets = []

    for year in backtest_years:
        print(f"\n  {'─' * 55}")
        print(f"  BACKTESTING: {year}")
        print(f"  {'─' * 55}")

        # ── Stage 1: XGBoost regression ──
        train_tour = tour_df[tour_df["season"] < year].copy()
        train_tour = train_tour.dropna(subset=["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"])
        train_tour = train_tour[train_tour["finish_num"] < 999]

        course_weights = _compute_course_weights_for_year(unified_df, year)
        train_weighted = _apply_course_weights(train_tour, course_weights)
        train_featured = _build_rolling_features(train_weighted)
        train_featured["finish_pct"] = ((train_featured["finish_num"] - 1) / (train_featured["field_size"] - 1)).clip(0, 1)
        train_clean = train_featured.dropna(subset=ROLLING_FEATURES + ["finish_pct"])

        if len(train_clean) < 100:
            print(f"  Skipping {year}: insufficient training data ({len(train_clean)} rows)")
            continue

        # Train Stage 1
        model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
        model_s1.fit(train_clean[ROLLING_FEATURES].values, train_clean["finish_pct"].values)

        # ── Build field features ──
        field = unified_df[
            (unified_df["season"] == year) & (unified_df["has_sg_data"] == True)
        ].copy()

        if len(field) == 0:
            continue

        field_features = []
        for _, pr in field.iterrows():
            pname = pr["player_name"]
            player_tour = train_tour[train_tour["player_name"] == pname].copy()
            if len(player_tour) == 0:
                continue
            pw = _apply_course_weights(player_tour, course_weights)
            pf = _build_rolling_features(pw)
            if len(pf) == 0:
                continue
            latest = pf.iloc[-1]

            aug_feats = _build_augusta_features_for_year(
                unified_df, rounds_df, features_df, pname, year, weather_df)

            feat_dict = {"player_name": pname}
            for col in ROLLING_FEATURES:
                feat_dict[col] = latest.get(col)
            feat_dict.update(aug_feats)
            feat_dict["actual_finish_num"] = pr.get("finish_num")
            feat_dict["actual_finish_pos"] = pr.get("finish_pos")
            field_features.append(feat_dict)

        field_df = pd.DataFrame(field_features)
        field_df = field_df.dropna(subset=ROLLING_FEATURES)
        if len(field_df) < 5:
            continue

        print(f"  Field: {len(field_df)} players | Train: {len(train_clean)} rows")

        # Stage 1 predictions + Monte Carlo
        preds_s1 = model_s1.predict(field_df[ROLLING_FEATURES].values)
        mc = _run_monte_carlo(preds_s1)
        field_df["mc_win_prob"] = mc["win_prob"]
        field_df["mc_top5_prob"] = mc["top5_prob"]
        field_df["mc_top10_prob"] = mc["top10_prob"]
        field_df["mc_top20_prob"] = mc["top20_prob"]
        field_df["predicted_finish_pct"] = preds_s1

        # ── Stage 2: Binary classifier for top-10 ──
        # Training: use same tour data but with binary target
        train_clean["made_top10"] = (train_clean["finish_num"] <= 10).astype(int)

        # Add Augusta features to training data — most will be 0/NaN for non-Masters events
        for col in AUGUSTA_FEATURES:
            if col not in train_clean.columns:
                train_clean[col] = 0.0

        # Fill NaN in stage2 features
        train_s2 = train_clean[STAGE2_FEATURES + ["made_top10"]].fillna(0)

        model_s2 = xgb.XGBClassifier(**XGB_CLF_PARAMS)
        model_s2.fit(train_s2[STAGE2_FEATURES].values, train_s2["made_top10"].values)

        # Stage 2 prediction on field
        field_s2 = field_df[STAGE2_FEATURES].fillna(0)
        s2_probs = model_s2.predict_proba(field_s2.values)[:, 1]
        field_df["s2_top10_prob"] = s2_probs

        # Blended output: 0.6 * Stage2 + 0.4 * MonteCarlo
        field_df["blended_top10_prob"] = 0.6 * s2_probs + 0.4 * field_df["mc_top10_prob"]

        # Also compute blended top-20 (use MC for now, stage2 trained on top10)
        # Train separate top-20 classifier
        train_clean["made_top20"] = (train_clean["finish_num"] <= 20).astype(int)
        train_s2_20 = train_clean[STAGE2_FEATURES + ["made_top20"]].fillna(0)
        model_s2_20 = xgb.XGBClassifier(**{**XGB_CLF_PARAMS, "scale_pos_weight": 3.5})
        model_s2_20.fit(train_s2_20[STAGE2_FEATURES].values, train_s2_20["made_top20"].values)
        s2_top20_probs = model_s2_20.predict_proba(field_s2.values)[:, 1]
        field_df["s2_top20_prob"] = s2_top20_probs
        field_df["blended_top20_prob"] = 0.6 * s2_top20_probs + 0.4 * field_df["mc_top20_prob"]

        # Use blended probabilities as final
        field_df["model_win_prob"] = field_df["mc_win_prob"]
        field_df["model_top5_prob"] = field_df["mc_top5_prob"]
        field_df["model_top10_prob"] = field_df["blended_top10_prob"]
        field_df["model_top20_prob"] = field_df["blended_top20_prob"]
        field_df["season"] = year

        # ── Scoring ──
        actual = field_df["actual_finish_num"].values
        valid_mask = (actual < 999) & pd.notna(actual)
        metrics = {"year": year}

        for outcome, prob_col, threshold in [
            ("win", "model_win_prob", 1), ("top5", "model_top5_prob", 5),
            ("top10", "model_top10_prob", 10), ("top20", "model_top20_prob", 20),
        ]:
            actual_bin = (actual <= threshold).astype(float)
            pred_p = field_df[prob_col].values
            metrics[f"brier_{outcome}"] = np.mean((pred_p - actual_bin) ** 2)
            pc = np.clip(pred_p, 1e-6, 1 - 1e-6)
            metrics[f"logloss_{outcome}"] = -np.mean(actual_bin * np.log(pc) + (1 - actual_bin) * np.log(1 - pc))

        valid_df = field_df[valid_mask].copy()
        if len(valid_df) > 5:
            sp, _ = stats.spearmanr(valid_df["model_win_prob"].values, -valid_df["actual_finish_num"].values)
            metrics["spearman"] = sp
            t10p = valid_df.nlargest(10, "model_top10_prob")
            metrics["top10_precision"] = (t10p["actual_finish_num"] <= 10).sum() / 10
            t5p = valid_df.nlargest(5, "model_top5_prob")
            metrics["top5_precision"] = (t5p["actual_finish_num"] <= 5).sum() / 5

            # AUC for Stage 2
            actual_t10 = (valid_df["actual_finish_num"] <= 10).astype(int)
            if actual_t10.sum() > 0 and actual_t10.sum() < len(actual_t10):
                metrics["s2_auc_top10"] = roc_auc_score(actual_t10, valid_df["s2_top10_prob"])
                metrics["blended_auc_top10"] = roc_auc_score(actual_t10, valid_df["blended_top10_prob"])
                metrics["mc_auc_top10"] = roc_auc_score(actual_t10, valid_df["mc_top10_prob"])

            actual_t20 = (valid_df["actual_finish_num"] <= 20).astype(int)
            if actual_t20.sum() > 0 and actual_t20.sum() < len(actual_t20):
                metrics["s2_auc_top20"] = roc_auc_score(actual_t20, valid_df["s2_top20_prob"])
                metrics["blended_auc_top20"] = roc_auc_score(actual_t20, valid_df["blended_top20_prob"])

        year_metrics.append(metrics)
        all_results.append(field_df)

        auc_str = f"S2 AUC={metrics.get('s2_auc_top10', 'N/A'):.3f}" if metrics.get('s2_auc_top10') else "S2 AUC=N/A"
        blend_str = f"Blend AUC={metrics.get('blended_auc_top10', 'N/A'):.3f}" if metrics.get('blended_auc_top10') else ""
        print(f"  Brier(win)={metrics['brier_win']:.4f} | Spearman={metrics.get('spearman', 0):.3f} | "
              f"Top10 Prec={metrics.get('top10_precision', 0):.0%} | {auc_str} | {blend_str}")

    # Save
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_parquet(PROCESSED_DIR / "backtest_results_v2.parquet", index=False)

    # Print summary
    print(f"\n  {'=' * 80}")
    print(f"  BACKTEST SUMMARY (V2 — with Stage 2 top-10 model)")
    print(f"  {'=' * 80}")
    print(f"  {'Year':<6} | {'Brier(w)':<9} | {'Spearman':<9} | {'T10 Prec':<9} | {'S2 AUC10':<9} | {'Blend AUC10':<12} | {'MC AUC10':<9}")
    print(f"  {'─'*6}-+-{'─'*9}-+-{'─'*9}-+-{'─'*9}-+-{'─'*9}-+-{'─'*12}-+-{'─'*9}")
    for m in year_metrics:
        s = lambda k, fmt=".3f": f"{m[k]:{fmt}}" if m.get(k) is not None else "N/A"
        print(f"  {m['year']:<6} | {s('brier_win','.4f'):<9} | {s('spearman'):<9} | {s('top10_precision','.0%'):<9} | "
              f"{s('s2_auc_top10'):<9} | {s('blended_auc_top10'):<12} | {s('mc_auc_top10'):<9}")

    if year_metrics:
        def avg(k): vals = [m[k] for m in year_metrics if m.get(k) is not None]; return np.mean(vals) if vals else None
        a = {k: avg(k) for k in ["brier_win", "spearman", "top10_precision", "s2_auc_top10", "blended_auc_top10", "mc_auc_top10"]}
        sa = lambda v, fmt=".3f": f"{v:{fmt}}" if v is not None else "N/A"
        print(f"  {'AVG':<6} | {sa(a['brier_win'],'.4f'):<9} | {sa(a['spearman']):<9} | {sa(a['top10_precision'],'.0%'):<9} | "
              f"{sa(a['s2_auc_top10']):<9} | {sa(a['blended_auc_top10']):<12} | {sa(a['mc_auc_top10']):<9}")

    print("\n  Task 3 complete.")
    return year_metrics, all_results, model_s2 if all_results else None


# ═══════════════════════════════════════════════════════════
# TASK 4 — TOP-10/20 BETTING EDGE ANALYSIS
# ═══════════════════════════════════════════════════════════

def task4_betting_edge(year_metrics, all_results):
    print("\n" + "=" * 60)
    print("TASK 4 — TOP-10/20 BETTING EDGE ANALYSIS")
    print("=" * 60)

    if not all_results:
        print("  No backtest results available")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # ── Try DG pre-tournament for market probs ──
    print("\n  Fetching DG pre-tournament probs as market proxy...")
    dg_probs = _dg_get_safe("preds/pre-tournament", {"tour": "pga"})
    dg_market = {}
    if dg_probs and isinstance(dg_probs, dict):
        baseline = dg_probs.get("baseline", [])
        for p in baseline:
            name = p.get("player_name", "")
            dg_market[name] = {
                "win": p.get("win", 0), "top_5": p.get("top_5", 0),
                "top_10": p.get("top_10", 0), "top_20": p.get("top_20", 0),
                "make_cut": p.get("make_cut", 0),
            }
        print(f"  DG market probs loaded for {len(dg_market)} players (current event only)")
    else:
        print("  DG pre-tournament not available — using field-size-based estimates")

    # Market probability estimates per market
    MARKETS = [
        ("top5", "model_top5_prob", 5),
        ("top10", "model_top10_prob", 10),
        ("top20", "model_top20_prob", 20),
    ]

    all_market_bets = []
    market_summary = []

    for market_name, model_col, threshold in MARKETS:
        print(f"\n  ── {market_name.upper()} MARKET ──")

        for year in sorted(results_df["season"].unique()):
            yr_df = results_df[results_df["season"] == year].copy()
            n_field = len(yr_df)
            # Estimate market prob: top-N / field_size as naive baseline
            naive_prob = threshold / n_field

            bets = []
            for _, row in yr_df.iterrows():
                model_prob = row[model_col]
                # Use naive market prob (same for all players in a year without real odds)
                market_prob = naive_prob
                if market_prob > 0:
                    kelly_edge = (model_prob - market_prob) / market_prob
                else:
                    kelly_edge = 0

                if kelly_edge > 0.20:  # 20% edge threshold for top-N markets
                    actual_hit = 1 if row["actual_finish_num"] <= threshold else 0
                    payout = 1.0 / market_prob
                    profit = (payout - 1) * actual_hit - (1 - actual_hit)
                    bets.append({
                        "market": market_name, "season": year,
                        "player_name": row["player_name"],
                        "model_prob": round(model_prob, 4),
                        "market_prob": round(market_prob, 4),
                        "edge_pct": round(kelly_edge * 100, 1),
                        "actual_finish": row["actual_finish_pos"],
                        "hit": actual_hit, "profit": round(profit, 2),
                    })

            n_bets = len(bets)
            hits = sum(b["hit"] for b in bets)
            total_profit = sum(b["profit"] for b in bets)
            roi = (total_profit / n_bets * 100) if n_bets > 0 else 0
            hit_rate = hits / n_bets if n_bets > 0 else 0

            market_summary.append({
                "market": market_name, "year": year,
                "bets": n_bets, "hits": hits,
                "hit_rate": round(hit_rate, 3), "roi_pct": round(roi, 1),
            })
            all_market_bets.extend(bets)

            if n_bets > 0:
                print(f"    {year}: {n_bets} bets, {hits} hits ({hit_rate:.0%}), ROI: {roi:+.1f}%")

    # ── H2H matchup accuracy ��─
    print(f"\n  ── HEAD-TO-HEAD ACCURACY ──")
    h2h_correct = 0
    h2h_total = 0
    for year in sorted(results_df["season"].unique()):
        yr_df = results_df[results_df["season"] == year].copy()
        yr_valid = yr_df[yr_df["actual_finish_num"] < 999].copy()
        players = yr_valid.sort_values("model_win_prob", ascending=False)

        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                p1 = players.iloc[i]
                p2 = players.iloc[j]
                # p1 has higher model_win_prob
                if p1["actual_finish_num"] < p2["actual_finish_num"]:
                    h2h_correct += 1
                elif p1["actual_finish_num"] > p2["actual_finish_num"]:
                    pass  # incorrect
                else:
                    h2h_correct += 0.5  # tie = half credit
                h2h_total += 1

    h2h_acc = h2h_correct / h2h_total if h2h_total > 0 else 0
    print(f"  H2H accuracy: {h2h_acc:.1%} ({int(h2h_correct)}/{h2h_total} matchups correct)")
    print(f"  Baseline (random): 50.0%")

    # ── Summary table ──
    print(f"\n  {'=' * 70}")
    print(f"  BETTING EDGE BY MARKET")
    print(f"  {'=' * 70}")
    print(f"  {'Market':<8} | {'Year':<6} | {'Bets':<5} | {'Hits':<5} | {'Hit Rate':<9} | {'ROI%':<8}")
    print(f"  {'─'*8}-+-{'─'*6}-+-{'─'*5}-+-{'─'*5}-+-{'─'*9}-+-{'─'*8}")
    for s in market_summary:
        print(f"  {s['market']:<8} | {s['year']:<6} | {s['bets']:<5} | {s['hits']:<5} | {s['hit_rate']:.0%}       | {s['roi_pct']:+.1f}%")

    # Averages per market
    print(f"  {'─'*8}-+-{'─'*6}-+-{'─'*5}-+-{'─'*5}-+-{'─'*9}-+-{'─'*8}")
    for mkt in ["top5", "top10", "top20"]:
        rows = [s for s in market_summary if s["market"] == mkt]
        if rows:
            avg_bets = np.mean([r["bets"] for r in rows])
            avg_hits = np.mean([r["hit_rate"] for r in rows])
            avg_roi = np.mean([r["roi_pct"] for r in rows])
            print(f"  {mkt:<8} | {'AVG':<6} | {avg_bets:<5.0f} | {'':5} | {avg_hits:.0%}       | {avg_roi:+.1f}%")

    # Save
    bets_df = pd.DataFrame(all_market_bets)
    if len(bets_df) > 0:
        bets_df.to_parquet(PROCESSED_DIR / "betting_edge_by_market.parquet", index=False)

    print("\n  Task 4 complete.")
    return market_summary, h2h_acc


# ═══════════════════════════════════════════════════════════
# TASK 5 — FEATURE IMPORTANCE AUDIT
# ═══════════════════════════════════════════════════════════

def task5_feature_importance(all_results):
    print("\n" + "=" * 60)
    print("TASK 5 — FEATURE IMPORTANCE AUDIT")
    print("=" * 60)

    if not all_results:
        print("  No results to analyze")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Install shap if needed
    try:
        import shap
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
        import shap

    # Train a final Stage 2 model on all available data for analysis
    print("\n  Training final Stage 2 model for SHAP analysis...")
    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)
    train_tour = tour_df.dropna(subset=["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"])
    train_tour = train_tour[train_tour["finish_num"] < 999]

    train_weighted = _apply_course_weights(train_tour, HARDCODED_WEIGHTS)
    train_featured = _build_rolling_features(train_weighted)
    train_featured["finish_pct"] = ((train_featured["finish_num"] - 1) / (train_featured["field_size"] - 1)).clip(0, 1)
    train_featured["made_top10"] = (train_featured["finish_num"] <= 10).astype(int)

    for col in AUGUSTA_FEATURES:
        if col not in train_featured.columns:
            train_featured[col] = 0.0

    train_s2 = train_featured[STAGE2_FEATURES + ["made_top10"]].fillna(0)
    model_final = xgb.XGBClassifier(**XGB_CLF_PARAMS)
    model_final.fit(train_s2[STAGE2_FEATURES].values, train_s2["made_top10"].values)

    # XGBoost built-in feature importance
    print("\n  XGBoost Feature Importance (gain):")
    importance = model_final.get_booster().get_score(importance_type="gain")
    # Map f0, f1... back to feature names
    feat_imp = {}
    for fname, score in importance.items():
        idx = int(fname.replace("f", ""))
        if idx < len(STAGE2_FEATURES):
            feat_imp[STAGE2_FEATURES[idx]] = score

    sorted_imp = sorted(feat_imp.items(), key=lambda x: -x[1])
    for i, (feat, imp) in enumerate(sorted_imp[:20]):
        marker = " *AUGUSTA*" if feat in AUGUSTA_FEATURES else ""
        print(f"    {i+1:2d}. {feat:<35} {imp:>10.1f}{marker}")

    # Check specific questions
    aug_in_top10 = [f for f, _ in sorted_imp[:10] if f in AUGUSTA_FEATURES]
    print(f"\n  Augusta-specific features in top 10: {aug_in_top10 if aug_in_top10 else 'None'}")

    bogey_rank = next((i+1 for i, (f,_) in enumerate(sorted_imp) if f == "augusta_bogey_avoidance"), None)
    sg_total_rank = next((i+1 for i, (f,_) in enumerate(sorted_imp) if f == "augusta_sg_total_career"), None)
    print(f"  augusta_bogey_avoidance rank: {bogey_rank}")
    print(f"  augusta_sg_total_career rank: {sg_total_rank}")
    if bogey_rank and sg_total_rank:
        print(f"  bogey_avoidance {'>' if bogey_rank < sg_total_rank else '<'} sg_total_career")

    # ── SHAP on backtest field data ──
    print("\n  Computing SHAP values on backtest field data...")
    field_data = results_df[STAGE2_FEATURES].fillna(0)

    explainer = shap.TreeExplainer(model_final)
    shap_values = explainer.shap_values(field_data.values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(zip(STAGE2_FEATURES, mean_abs_shap))
    sorted_shap = sorted(shap_importance.items(), key=lambda x: -x[1])

    print("\n  SHAP Feature Importance (mean |SHAP|):")
    for i, (feat, imp) in enumerate(sorted_shap[:15]):
        marker = " *AUGUSTA*" if feat in AUGUSTA_FEATURES else ""
        print(f"    {i+1:2d}. {feat:<35} {imp:>8.4f}{marker}")

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=STAGE2_FEATURES)
    shap_df["player_name"] = results_df["player_name"].values
    shap_df["season"] = results_df["season"].values
    shap_df.to_parquet(PROCESSED_DIR / "shap_top10.parquet", index=False)

    # ── JT Diagnosis ──
    print(f"\n  {'─' * 55}")
    print(f"  JT DIAGNOSIS: Why is Justin Thomas over-ranked?")
    print(f"  {'─' * 55}")

    for year in [2023, 2024, 2025]:
        jt = results_df[
            (results_df["player_name"] == "Justin Thomas") &
            (results_df["season"] == year)
        ]
        if len(jt) == 0:
            print(f"\n  {year}: Justin Thomas not in backtest results")
            continue

        jt_row = jt.iloc[0]
        jt_idx = jt.index[0]
        print(f"\n  {year}: JT actual finish = {jt_row['actual_finish_pos']}")
        print(f"    model_win_prob={jt_row['model_win_prob']:.3%}, "
              f"model_top10_prob={jt_row['model_top10_prob']:.3%}")

        # Key feature values
        print(f"    Key features:")
        key_feats = ["sg_total_8w", "sg_total_3w", "sg_total_momentum",
                     "sg_app_8w", "sg_ball_strike_ratio",
                     "augusta_starts", "augusta_made_cut_rate", "augusta_top10_rate",
                     "augusta_bogey_avoidance", "augusta_round_variance_score",
                     "augusta_scoring_avg"]
        for f in key_feats:
            val = jt_row.get(f)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                print(f"      {f}: {val:.4f}")

        # SHAP contributions for JT
        if jt_idx < len(shap_values):
            jt_shap = dict(zip(STAGE2_FEATURES, shap_values[jt_idx]))
            sorted_jt_shap = sorted(jt_shap.items(), key=lambda x: -abs(x[1]))
            print(f"    Top SHAP contributors (pushing prediction up/down):")
            for feat, val in sorted_jt_shap[:8]:
                direction = "UP" if val > 0 else "DOWN"
                print(f"      {feat:<35} SHAP={val:+.4f} ({direction})")

    print("\n  Task 5 complete.")
    return sorted_shap


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AUGUSTA NATIONAL MODEL — V2 PIPELINE")
    print("=" * 60)

    # Task 1
    features_df = task1_scoring_features()

    # Task 2
    weather_df = task2_additional_data()

    # Task 3
    year_metrics, all_results, model_s2 = task3_top10_model()

    # Task 4
    market_summary, h2h_acc = task4_betting_edge(year_metrics, all_results)

    # Task 5
    shap_results = task5_feature_importance(all_results)

    print("\n" + "=" * 60)
    print("  ALL V2 TASKS COMPLETE")
    print("=" * 60)

    return year_metrics, market_summary, h2h_acc, shap_results


if __name__ == "__main__":
    main()

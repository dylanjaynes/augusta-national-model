"""
Task 6 — Walk-forward backtesting framework
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import xgboost as xgb
from sklearn.linear_model import Ridge
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = Path("data/processed")
HISTORICAL_ROUNDS_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
DG_BASE = "https://feeds.datagolf.com"
API_KEY = os.getenv("DATAGOLF_API_KEY")

# XGBoost config from CLAUDE.md
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
}

N_SIMULATIONS = 50_000
NOISE_STD = 0.16
TARGET_PRED_STD = 0.10
SEED = 42

HARDCODED_WEIGHTS = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

SG_FEATURES = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
ROLLING_FEATURES = [
    "sg_ott_3w", "sg_app_3w", "sg_arg_3w", "sg_putt_3w", "sg_t2g_3w", "sg_total_3w",
    "sg_ott_8w", "sg_app_8w", "sg_arg_8w", "sg_putt_8w", "sg_t2g_8w", "sg_total_8w",
    "sg_total_std_8", "sg_total_momentum", "sg_ball_strike_ratio", "log_field_size",
]


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


def _build_rolling_features(df):
    """Build rolling SG features per player, sorted by date."""
    df = df.sort_values(["player_name", "date"]).copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]

    for col in sg_cols:
        # 3-event exponential weighted mean
        df[f"{col}_3w"] = df.groupby("player_name")[col].transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()
        )
        # 8-event exponential weighted mean
        df[f"{col}_8w"] = df.groupby("player_name")[col].transform(
            lambda x: x.ewm(span=8, min_periods=1).mean()
        )

    # sg_total rolling std (8-event window)
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(
        lambda x: x.rolling(8, min_periods=2).std()
    )

    # Momentum: 3w - 8w
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]

    # Ball striking ratio
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6
    )

    df["log_field_size"] = np.log(df["field_size"].clip(1))

    return df


def _apply_course_weights(df, weights):
    """Apply course-fit SG multipliers before feature construction."""
    df = df.copy()
    for col, w in weights.items():
        if col in df.columns:
            df[col] = df[col] * w
    return df


def _compute_course_weights_for_year(unified_df, target_year):
    """Compute Augusta weights using only data prior to target_year."""
    sg_rows = unified_df[
        (unified_df["has_sg_data"] == True) &
        (unified_df["season"] < target_year)
    ].copy()

    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    sg_rows = sg_rows.dropna(subset=sg_cols)

    if len(sg_rows) < 10:
        return HARDCODED_WEIGHTS.copy()

    sg_rows["finish_num"] = sg_rows["finish_pos"].apply(_parse_finish_num)
    sg_rows = sg_rows[sg_rows["finish_num"] < 999]
    sg_rows["finish_pct"] = (sg_rows["finish_num"] - 1) / (sg_rows["field_size"] - 1)
    sg_rows["finish_pct"] = sg_rows["finish_pct"].clip(0, 1)

    ridge = Ridge(alpha=1.0)
    ridge.fit(sg_rows[sg_cols].values, sg_rows["finish_pct"].values)
    raw_coefs = dict(zip(sg_cols, ridge.coef_))

    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num_t"] = tour_df["finish_pos"].apply(_parse_finish_num)
    tour_df = tour_df[(tour_df["finish_num_t"] < 999) & (tour_df["season"] < target_year)]
    tour_df = tour_df.dropna(subset=sg_cols)
    tour_df["finish_pct_t"] = (tour_df["finish_num_t"] - 1) / (tour_df["field_size"] - 1)
    tour_df["finish_pct_t"] = tour_df["finish_pct_t"].clip(0, 1)

    ridge_tour = Ridge(alpha=1.0)
    ridge_tour.fit(tour_df[sg_cols].values, tour_df["finish_pct_t"].values)
    tour_coefs = dict(zip(sg_cols, ridge_tour.coef_))

    computed = {}
    for col in sg_cols:
        if abs(tour_coefs[col]) > 1e-6:
            computed[col] = abs(raw_coefs[col]) / abs(tour_coefs[col])
        else:
            computed[col] = 1.0

    # Shrinkage with thin data
    if len(sg_rows) < 150:
        final = {col: 0.70 * computed[col] + 0.30 * HARDCODED_WEIGHTS[col] for col in sg_cols}
    else:
        final = computed

    return final


def _build_augusta_features_for_year(unified_df, rounds_df, player_name, target_year):
    """Build Augusta-specific features for a player using only prior data."""
    prior = unified_df[
        (unified_df["player_name"] == player_name) &
        (unified_df["season"] < target_year)
    ]

    features = {}
    n_starts = len(prior)
    features["augusta_starts"] = n_starts

    if n_starts > 0:
        seasons_ago = target_year - prior["season"]
        decay = 0.75 ** seasons_ago

        prior_cuts = prior["made_cut"].dropna()
        features["augusta_made_cut_rate"] = prior_cuts.mean() if len(prior_cuts) > 0 else 0.0

        prior_finish = prior["finish_num"].dropna()
        features["augusta_top10_rate"] = (prior_finish <= 10).mean() if len(prior_finish) > 0 else 0.0
        good_finishes = prior_finish[prior_finish < 999]
        features["augusta_best_finish"] = good_finishes.min() if len(good_finishes) > 0 else 99

        svf = prior["score_vs_field"].dropna()
        if len(svf) > 0:
            features["augusta_scoring_avg"] = np.average(svf, weights=decay[svf.index])
        else:
            features["augusta_scoring_avg"] = 0.0

        # SG features from prior
        prior_sg = prior[prior["has_sg_data"] == True]
        if len(prior_sg) > 0:
            sg_decay = 0.75 ** (target_year - prior_sg["season"])
            for col in ["sg_app", "sg_total"]:
                vals = prior_sg[col].dropna()
                if len(vals) > 0:
                    features[f"augusta_{col}_career"] = np.average(vals, weights=sg_decay[vals.index])
                else:
                    features[f"augusta_{col}_career"] = 0.0
        else:
            features["augusta_sg_app_career"] = 0.0
            features["augusta_sg_total_career"] = 0.0

        if n_starts <= 2:
            features["augusta_experience_bucket"] = 1
        elif n_starts <= 5:
            features["augusta_experience_bucket"] = 2
        else:
            features["augusta_experience_bucket"] = 3
    else:
        features["augusta_made_cut_rate"] = 0.0
        features["augusta_top10_rate"] = 0.0
        features["augusta_best_finish"] = 99
        features["augusta_scoring_avg"] = 0.0
        features["augusta_sg_app_career"] = 0.0
        features["augusta_sg_total_career"] = 0.0
        features["augusta_experience_bucket"] = 0

    return features


def _dg_get_safe(endpoint, params=None):
    """Safe DG API call that returns None on error."""
    if params is None:
        params = {}
    url = f"{DG_BASE}/{endpoint}"
    params = {"key": API_KEY, "file_format": "json", **params}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 403:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _run_monte_carlo(predictions, n_sims=N_SIMULATIONS, seed=SEED):
    """Run Monte Carlo simulation from predicted finish_pct values."""
    rng = np.random.RandomState(seed)
    n_players = len(predictions)

    # predictions is array of predicted finish_pct (lower = better)
    base_scores = predictions.copy()

    # Normalize to z-scores
    mean_pred = base_scores.mean()
    std_pred = base_scores.std()
    if std_pred > 0:
        z_scores = (base_scores - mean_pred) / std_pred
    else:
        z_scores = np.zeros(n_players)

    # Rescale to target distribution
    z_scores = z_scores * TARGET_PRED_STD

    # Simulate
    wins = np.zeros(n_players)
    top5 = np.zeros(n_players)
    top10 = np.zeros(n_players)
    top20 = np.zeros(n_players)

    for _ in range(n_sims):
        noise = rng.normal(0, NOISE_STD, n_players)
        sim_scores = z_scores + noise
        ranks = sim_scores.argsort().argsort() + 1  # 1-indexed rank

        wins += (ranks == 1)
        top5 += (ranks <= 5)
        top10 += (ranks <= 10)
        top20 += (ranks <= 20)

    return {
        "win_prob": wins / n_sims,
        "top5_prob": top5 / n_sims,
        "top10_prob": top10 / n_sims,
        "top20_prob": top20 / n_sims,
    }


def task6_backtest():
    print("\n" + "=" * 60)
    print("TASK 6 — BACKTESTING FRAMEWORK")
    print("=" * 60)

    # Load data
    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)

    rounds_path = PROCESSED_DIR / "masters_sg_rounds.parquet"
    rounds_df = pd.read_parquet(rounds_path) if rounds_path.exists() else pd.DataFrame()

    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)

    # Determine backtest years: need SG data (from DG) + prior history
    sg_years = sorted(unified_df[unified_df["has_sg_data"] == True]["season"].unique())
    print(f"\n  Years with SG data: {sg_years}")

    # We need at least some prior Masters history, so first SG year with prior data
    all_years = sorted(unified_df["season"].unique())
    backtest_years = [y for y in sg_years if y > min(all_years)]
    print(f"  Backtest years: {backtest_years}")

    if not backtest_years:
        print("  No valid backtest years found!")
        return [], []

    all_results = []
    all_bets = []
    year_metrics = []

    for year in backtest_years:
        print(f"\n  {'─' * 50}")
        print(f"  BACKTESTING YEAR: {year}")
        print(f"  {'─' * 50}")

        # Step 1 — Build training set
        print(f"  Step 1: Building training set (season < {year})...")
        train_tour = tour_df[tour_df["season"] < year].copy()
        train_tour = train_tour.dropna(subset=["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"])
        train_tour = train_tour[train_tour["finish_num"] < 999]

        # Step 2 — Compute course weights for this year
        print(f"  Step 2: Computing Augusta weights from pre-{year} data...")
        course_weights = _compute_course_weights_for_year(unified_df, year)
        print(f"    Weights: {course_weights}")

        # Apply course weights to training data for Augusta-like events
        train_weighted = _apply_course_weights(train_tour, course_weights)

        # Build rolling features
        print(f"  Step 2b: Building rolling features on {len(train_weighted)} training rows...")
        train_featured = _build_rolling_features(train_weighted)
        train_featured["finish_pct"] = (train_featured["finish_num"] - 1) / (train_featured["field_size"] - 1)
        train_featured["finish_pct"] = train_featured["finish_pct"].clip(0, 1)

        feature_cols = ROLLING_FEATURES
        train_clean = train_featured.dropna(subset=feature_cols + ["finish_pct"])

        if len(train_clean) < 100:
            print(f"  ⚠ Only {len(train_clean)} clean training rows, skipping {year}")
            continue

        X_train = train_clean[feature_cols].values
        y_train = train_clean["finish_pct"].values

        # Train XGBoost
        print(f"  Step 2c: Training XGBoost on {len(X_train)} rows...")
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)

        # Step 3 — Build feature matrix for year Y field
        print(f"  Step 3: Building feature matrix for {year} Masters field...")
        field = unified_df[
            (unified_df["season"] == year) &
            (unified_df["has_sg_data"] == True)
        ].copy()

        if len(field) == 0:
            print(f"  ⚠ No field data for {year}, skipping")
            continue

        # For each player in the field, get their most recent tour data before the Masters
        field_features = []
        for _, player_row in field.iterrows():
            player_name = player_row["player_name"]
            dg_id = player_row.get("dg_id")

            # Get player's recent tour performance — try dg_id first, then name
            player_tour = pd.DataFrame()
            if dg_id is not None and not (isinstance(dg_id, float) and np.isnan(dg_id)):
                player_tour = train_tour[train_tour["dg_id"] == dg_id].copy()
            if len(player_tour) == 0:
                player_tour = train_tour[train_tour["player_name"] == player_name].copy()

            if len(player_tour) == 0:
                continue

            # Apply course weights and build features
            player_weighted = _apply_course_weights(player_tour, course_weights)
            player_featured = _build_rolling_features(player_weighted)

            if len(player_featured) == 0:
                continue

            # Take most recent row as current form
            latest = player_featured.iloc[-1]

            # Add Augusta-specific features
            aug_feats = _build_augusta_features_for_year(
                unified_df, rounds_df, player_name, year
            )

            feat_dict = {"player_name": player_name, "dg_id": dg_id}
            for col in feature_cols:
                feat_dict[col] = latest.get(col)
            feat_dict.update(aug_feats)

            # Actual results
            feat_dict["actual_finish_num"] = player_row.get("finish_num")
            feat_dict["actual_finish_pos"] = player_row.get("finish_pos")

            field_features.append(feat_dict)

        field_df = pd.DataFrame(field_features)
        field_df = field_df.dropna(subset=feature_cols)

        if len(field_df) < 5:
            print(f"  ⚠ Only {len(field_df)} players with features for {year}, skipping")
            continue

        print(f"    {len(field_df)} players with complete features")

        X_field = field_df[feature_cols].values
        predictions = model.predict(X_field)

        # Step 4 — Monte Carlo simulation
        print(f"  Step 4: Running {N_SIMULATIONS:,} Monte Carlo simulations...")
        sim_results = _run_monte_carlo(predictions, n_sims=N_SIMULATIONS, seed=SEED)

        field_df["model_win_prob"] = sim_results["win_prob"]
        field_df["model_top5_prob"] = sim_results["top5_prob"]
        field_df["model_top10_prob"] = sim_results["top10_prob"]
        field_df["model_top20_prob"] = sim_results["top20_prob"]
        field_df["predicted_finish_pct"] = predictions
        field_df["season"] = year

        # Step 5 — Score predictions
        print(f"  Step 5: Scoring predictions against actual results...")

        actual_finish = field_df["actual_finish_num"].values
        valid_mask = (actual_finish < 999) & pd.notna(actual_finish)

        # Calibration metrics
        metrics = {"year": year}

        for outcome, prob_col, threshold in [
            ("win", "model_win_prob", 1),
            ("top5", "model_top5_prob", 5),
            ("top10", "model_top10_prob", 10),
            ("top20", "model_top20_prob", 20),
        ]:
            actual_binary = (actual_finish <= threshold).astype(float)
            pred_prob = field_df[prob_col].values

            # Brier score
            brier = np.mean((pred_prob - actual_binary) ** 2)
            metrics[f"brier_{outcome}"] = brier

            # Log loss (with clipping to avoid log(0))
            pred_clipped = np.clip(pred_prob, 1e-6, 1 - 1e-6)
            logloss = -np.mean(
                actual_binary * np.log(pred_clipped) +
                (1 - actual_binary) * np.log(1 - pred_clipped)
            )
            metrics[f"logloss_{outcome}"] = logloss

        # Ranking quality
        valid_df = field_df[valid_mask].copy()
        if len(valid_df) > 5:
            spearman_corr, _ = stats.spearmanr(
                valid_df["model_win_prob"].values,
                -valid_df["actual_finish_num"].values  # negative because higher prob = lower finish
            )
            metrics["spearman"] = spearman_corr

            # Top-5 precision
            top5_predicted = valid_df.nlargest(5, "model_win_prob")
            top5_actual = (top5_predicted["actual_finish_num"] <= 5).sum()
            metrics["top5_precision"] = top5_actual / 5

            # Top-10 precision
            top10_predicted = valid_df.nlargest(10, "model_win_prob")
            top10_actual = (top10_predicted["actual_finish_num"] <= 10).sum()
            metrics["top10_precision"] = top10_actual / 10
        else:
            metrics["spearman"] = None
            metrics["top5_precision"] = None
            metrics["top10_precision"] = None

        # Betting edge — try DG pre-tournament as market proxy
        print(f"  Step 5b: Computing betting edge...")
        market_probs = {}
        try:
            pred_data = _dg_get_safe("preds/pre-tournament", {"tour": "pga"})
            if pred_data:
                baseline = pred_data.get("baseline", [])
                for p in baseline:
                    market_probs[p.get("dg_id")] = p.get("win", 0)
        except Exception:
            pass

        # If no live market, use simple 1/field_size as baseline
        if not market_probs:
            for _, row in field_df.iterrows():
                market_probs[row.get("dg_id", row["player_name"])] = 1.0 / len(field_df)

        bets = []
        for _, row in field_df.iterrows():
            model_prob = row["model_win_prob"]
            market_prob = market_probs.get(row.get("dg_id"), 1.0 / len(field_df))
            if market_prob > 0:
                kelly_edge = (model_prob - market_prob) / market_prob
            else:
                kelly_edge = 0

            if kelly_edge > 0.15:
                payout = 1.0 / market_prob if market_prob > 0 else 0
                actual_win = 1 if row["actual_finish_num"] == 1 else 0
                profit = (payout - 1) * actual_win - (1 - actual_win)

                bets.append({
                    "season": year,
                    "player_name": row["player_name"],
                    "model_prob": round(model_prob, 4),
                    "market_prob": round(market_prob, 4),
                    "edge_pct": round(kelly_edge * 100, 1),
                    "actual_finish": row.get("actual_finish_pos", row["actual_finish_num"]),
                    "outcome": "WIN" if actual_win else "LOSS",
                    "profit": round(profit, 2),
                })

        total_bets = len(bets)
        total_wins = sum(1 for b in bets if b["outcome"] == "WIN")
        total_profit = sum(b["profit"] for b in bets)
        roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

        metrics["n_bets"] = total_bets
        metrics["roi_pct"] = roi

        year_metrics.append(metrics)
        all_bets.extend(bets)
        all_results.append(field_df)

        print(f"    Brier(win): {metrics['brier_win']:.4f}")
        print(f"    LogLoss(win): {metrics['logloss_win']:.4f}")
        print(f"    Spearman: {metrics.get('spearman', 'N/A')}")
        print(f"    Top-10 Precision: {metrics.get('top10_precision', 'N/A')}")
        print(f"    Bets: {total_bets}, ROI: {roi:.1f}%")

    # Save results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_parquet(PROCESSED_DIR / "backtest_results.parquet", index=False)

    # Print summary table
    print(f"\n  {'=' * 75}")
    print(f"  BACKTEST SUMMARY")
    print(f"  {'=' * 75}")
    print(f"  {'Year':<6} | {'Brier(win)':<11} | {'LogLoss(win)':<13} | {'Spearman':<9} | {'Top10 Prec':<11} | {'ROI%':<6}")
    print(f"  {'─' * 6}-+-{'─' * 11}-+-{'─' * 13}-+-{'─' * 9}-+-{'─' * 11}-+-{'─' * 6}")

    for m in year_metrics:
        spear = f"{m['spearman']:.3f}" if m.get('spearman') is not None else "N/A"
        t10 = f"{m['top10_precision']:.1%}" if m.get('top10_precision') is not None else "N/A"
        print(f"  {m['year']:<6} | {m['brier_win']:.6f}   | {m['logloss_win']:.6f}     | {spear:<9} | {t10:<11} | {m['roi_pct']:.1f}%")

    if year_metrics:
        avg_brier = np.mean([m["brier_win"] for m in year_metrics])
        avg_ll = np.mean([m["logloss_win"] for m in year_metrics])
        avg_spear = np.mean([m["spearman"] for m in year_metrics if m.get("spearman") is not None])
        avg_t10 = np.mean([m["top10_precision"] for m in year_metrics if m.get("top10_precision") is not None])
        avg_roi = np.mean([m["roi_pct"] for m in year_metrics])
        print(f"  {'AVG':<6} | {avg_brier:.6f}   | {avg_ll:.6f}     | {avg_spear:.3f}     | {avg_t10:.1%}       | {avg_roi:.1f}%")

    # Print model bets
    if all_bets:
        print(f"\n  MODEL BETS (Kelly edge > 15%):")
        print(f"  {'Player':<25} | {'Model%':<7} | {'Market%':<8} | {'Edge%':<6} | {'Finish':<8} | {'Result'}")
        print(f"  {'─' * 25}-+-{'─' * 7}-+-{'─' * 8}-+-{'─' * 6}-+-{'─' * 8}-+-{'─' * 6}")
        for b in sorted(all_bets, key=lambda x: (x["season"], -x["edge_pct"])):
            print(f"  {b['player_name']:<25} | {b['model_prob']:.3%} | {b['market_prob']:.3%}  | {b['edge_pct']:>5.1f}% | {str(b['actual_finish']):<8} | {b['outcome']}")

    print("\n✓ Task 6 complete.")
    return year_metrics, all_bets

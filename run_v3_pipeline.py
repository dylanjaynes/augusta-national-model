#!/usr/bin/env python3
"""
Augusta National Model — V3 Pipeline
Session 3: Rebuilt experience features, Augusta-only Stage 2, blend optimization
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED_DIR = Path("data/processed")
HISTORICAL_ROUNDS_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
HARDCODED_WEIGHTS = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

XGB_PARAMS = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:squarederror",
    "random_state": 42,
}
N_SIMULATIONS = 50_000; NOISE_STD = 0.16; TARGET_PRED_STD = 0.10; SEED = 42

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


# ═══════════════════════════════════════════════════════════
# TASK 1 — REBUILD EXPERIENCE FEATURES
# ═══════════════════════════════════════════════════════════

def task1_experience_features():
    print("\n" + "=" * 60)
    print("TASK 1 — REBUILD EXPERIENCE FEATURES (THE RIGHT WAY)")
    print("=" * 60)

    unified = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified["finish_num"] = unified["finish_pos"].apply(_parse_finish_num)
    rounds_df = pd.read_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet")

    # Precompute rounds_played per player-season from round data and unified
    def _count_rounds(row):
        """Count competitive rounds from round score availability."""
        count = 0
        for col in ["r1_score", "r2_score", "r3_score", "r4_score"]:
            if pd.notna(row.get(col)):
                count += 1
        if count > 0:
            return count
        # Fallback: infer from made_cut
        if row.get("made_cut") == 1:
            return 4
        elif row.get("finish_num") == 999:
            return 2  # CUT = played R1+R2
        return 0

    unified["rounds_played"] = unified.apply(_count_rounds, axis=1)

    feature_rows = []
    for _, row in unified.iterrows():
        player = row["player_name"]
        season = row["season"]
        prior = unified[(unified["player_name"] == player) & (unified["season"] < season)]

        # ── Feature 1: augusta_competitive_rounds ──
        competitive_rounds = prior["rounds_played"].sum()

        # ── Feature 2: augusta_made_cut_prev_year ──
        prev_year = prior[prior["season"] == season - 1]
        if len(prev_year) > 0:
            made_cut_prev = int(prev_year.iloc[0]["made_cut"] == 1)
        else:
            made_cut_prev = 0

        # ── Feature 3: augusta_experience_tier ──
        if competitive_rounds == 0:
            tier = 0  # debutant
        elif competitive_rounds <= 7:
            tier = 1  # learning
        elif competitive_rounds <= 19:
            tier = 2  # established
        elif competitive_rounds <= 35:
            tier = 3  # veteran
        else:
            tier = 4  # deep_veteran

        # ── Feature 4: augusta_scoring_trajectory ──
        prior_cut_makers = prior[prior["made_cut"] == 1]
        if len(prior_cut_makers) >= 2:
            recent_2 = prior_cut_makers.nlargest(2, "season")
            recent_avg = recent_2["score_vs_field"].dropna().mean()
            # Career exponentially weighted avg
            all_svf = prior_cut_makers["score_vs_field"].dropna()
            if len(all_svf) > 0:
                decay = 0.75 ** (season - prior_cut_makers.loc[all_svf.index, "season"])
                career_avg = np.average(all_svf, weights=decay)
                trajectory = career_avg - recent_avg if pd.notna(recent_avg) else 0.0
            else:
                trajectory = 0.0
        else:
            trajectory = 0.0

        # ── Feature 5: augusta_rounds_last_2yrs ──
        recent_prior = prior[prior["season"].isin([season - 1, season - 2])]
        rounds_last_2 = recent_prior["rounds_played"].sum()

        # ── Feature 6: augusta_best_finish_recent ──
        recent_3 = prior.nlargest(3, "season") if len(prior) >= 1 else prior
        if len(recent_3) > 0:
            recent_finishes = recent_3["finish_num"].dropna()
            valid_finishes = recent_finishes[recent_finishes < 999]
            if len(valid_finishes) > 0:
                best_recent = valid_finishes.min()
            else:
                best_recent = row.get("field_size", 90) + 1
        else:
            best_recent = row.get("field_size", 90) + 1

        # Also carry forward existing features we still want
        n_starts = len(prior)
        if n_starts > 0:
            prior_cuts = prior["made_cut"].dropna()
            made_cut_rate = prior_cuts.mean() if len(prior_cuts) > 0 else 0.0
            prior_finish = prior["finish_num"].dropna()
            top10_rate = (prior_finish <= 10).mean() if len(prior_finish) > 0 else 0.0
            best_finish = prior_finish[prior_finish < 999].min() if (prior_finish < 999).any() else 99
            svf = prior["score_vs_field"].dropna()
            decay_all = 0.75 ** (season - prior.loc[svf.index, "season"]) if len(svf) > 0 else pd.Series()
            scoring_avg = np.average(svf, weights=decay_all) if len(svf) > 0 else 0.0
        else:
            made_cut_rate = 0.0; top10_rate = 0.0; best_finish = 99; scoring_avg = 0.0

        # SG career features
        prior_sg = prior[prior["has_sg_data"] == True]
        if len(prior_sg) > 0:
            sg_decay = 0.75 ** (season - prior_sg["season"])
            sg_app_vals = prior_sg["sg_app"].dropna()
            sg_app_career = np.average(sg_app_vals, weights=sg_decay[sg_app_vals.index]) if len(sg_app_vals) > 0 else None
            sg_total_vals = prior_sg["sg_total"].dropna()
            sg_total_career = np.average(sg_total_vals, weights=sg_decay[sg_total_vals.index]) if len(sg_total_vals) > 0 else None
        else:
            sg_app_career = None; sg_total_career = None

        # Scoring pattern features (from V2, recomputed with lookback)
        prior_rounds = rounds_df[
            (rounds_df["player_name"] == player) & (rounds_df["season"] < season)
        ] if len(rounds_df) > 0 else pd.DataFrame()

        if len(prior_rounds) > 0:
            scores = prior_rounds["score"].dropna()
            clean = (scores <= 0).sum()
            bogey_avoid = clean / len(scores) if len(scores) > 0 else 0.0

            round_var = scores.std() if len(scores) > 1 else None

            r34 = prior_rounds[prior_rounds["round_num"].isin([3, 4])]["score"].dropna()
            back9 = r34.mean() if len(r34) > 0 else None
        else:
            bogey_avoid = 0.0; round_var = None; back9 = None

        feature_rows.append({
            "player_name": player, "season": season,
            # 6 new experience features
            "augusta_competitive_rounds": competitive_rounds,
            "augusta_made_cut_prev_year": made_cut_prev,
            "augusta_experience_tier": tier,
            "augusta_scoring_trajectory": trajectory,
            "augusta_rounds_last_2yrs": rounds_last_2,
            "augusta_best_finish_recent": best_recent,
            # Carried forward
            "augusta_starts": n_starts,
            "augusta_made_cut_rate": made_cut_rate,
            "augusta_top10_rate": top10_rate,
            "augusta_best_finish": best_finish,
            "augusta_scoring_avg": scoring_avg,
            "augusta_sg_app_career": sg_app_career,
            "augusta_sg_total_career": sg_total_career,
            "augusta_bogey_avoidance": bogey_avoid,
            "augusta_round_variance_score": round_var,
            "augusta_back9_scoring": back9,
        })

    features_df = pd.DataFrame(feature_rows)
    features_df.to_parquet(PROCESSED_DIR / "augusta_player_features.parquet", index=False)

    print(f"\n  Shape: {features_df.shape}")
    print(f"  Columns: {list(features_df.columns)}")
    tier_dist = features_df["augusta_experience_tier"].value_counts().sort_index()
    print(f"  Experience tier distribution: {tier_dist.to_dict()}")

    # ── Validation table ──
    print(f"\n  {'=' * 100}")
    print(f"  VALIDATION TABLE")
    print(f"  {'=' * 100}")
    checks = [
        ("Justin Thomas", [2022, 2023, 2024, 2025]),
        ("Scottie Scheffler", [2022, 2023, 2024, 2025]),
        ("Jon Rahm", [2021, 2022, 2023]),
        ("Rory McIlroy", [2022, 2023, 2024, 2025]),
        ("Hideki Matsuyama", [2019, 2020, 2021]),
    ]
    cols = ["augusta_competitive_rounds", "augusta_made_cut_prev_year", "augusta_experience_tier",
            "augusta_scoring_trajectory", "augusta_rounds_last_2yrs", "augusta_best_finish_recent"]

    print(f"  {'Player':<22} {'Yr':>4} | {'CompRds':>7} {'CutPrev':>7} {'Tier':>4} {'Traj':>7} {'Rds2yr':>6} {'BestRec':>7}")
    print(f"  {'─'*22} {'─'*4}-+-{'─'*7}-{'─'*7}-{'─'*4}-{'─'*7}-{'─'*6}-{'─'*7}")
    for player, years in checks:
        for yr in years:
            r = features_df[(features_df["player_name"] == player) & (features_df["season"] == yr)]
            if len(r) == 0:
                print(f"  {player:<22} {yr:>4} | {'N/A':>7} {'':>7} {'':>4} {'':>7} {'':>6} {'':>7}")
                continue
            r = r.iloc[0]
            print(f"  {player:<22} {yr:>4} | {r['augusta_competitive_rounds']:>7.0f} "
                  f"{r['augusta_made_cut_prev_year']:>7.0f} {r['augusta_experience_tier']:>4.0f} "
                  f"{r['augusta_scoring_trajectory']:>7.2f} {r['augusta_rounds_last_2yrs']:>6.0f} "
                  f"{r['augusta_best_finish_recent']:>7.0f}")

    print("\n  Task 1 complete.")
    return features_df


# ═══════════════════════════════════════════════════════════
# HELPERS (Stage 1 / Monte Carlo — unchanged from V2)
# ═══════════════════════════════════════════════════════════

def _build_rolling_features(df):
    df = df.sort_values(["player_name", "date"]).copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
    for col in sg_cols:
        df[f"{col}_3w"] = df.groupby("player_name")[col].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        df[f"{col}_8w"] = df.groupby("player_name")[col].transform(lambda x: x.ewm(span=8, min_periods=1).mean())
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(lambda x: x.rolling(8, min_periods=2).std())
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))
    return df

def _apply_course_weights(df, weights):
    df = df.copy()
    for col, w in weights.items():
        if col in df.columns: df[col] = df[col] * w
    return df

def _compute_course_weights_for_year(unified_df, target_year):
    sg_rows = unified_df[(unified_df["has_sg_data"]==True)&(unified_df["season"]<target_year)].copy()
    sg_cols = ["sg_ott","sg_app","sg_arg","sg_putt"]
    sg_rows = sg_rows.dropna(subset=sg_cols)
    if len(sg_rows) < 10: return HARDCODED_WEIGHTS.copy()
    sg_rows["finish_num"] = sg_rows["finish_pos"].apply(_parse_finish_num)
    sg_rows = sg_rows[sg_rows["finish_num"]<999]
    sg_rows["finish_pct"] = ((sg_rows["finish_num"]-1)/(sg_rows["field_size"]-1)).clip(0,1)
    r = Ridge(alpha=1.0); r.fit(sg_rows[sg_cols].values, sg_rows["finish_pct"].values)
    raw = dict(zip(sg_cols, r.coef_))
    tdf = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tdf["fn"] = tdf["finish_pos"].apply(_parse_finish_num)
    tdf = tdf[(tdf["fn"]<999)&(tdf["season"]<target_year)].dropna(subset=sg_cols)
    tdf["fp"] = ((tdf["fn"]-1)/(tdf["field_size"]-1)).clip(0,1)
    rt = Ridge(alpha=1.0); rt.fit(tdf[sg_cols].values, tdf["fp"].values)
    tc = dict(zip(sg_cols, rt.coef_))
    comp = {c: abs(raw[c])/abs(tc[c]) if abs(tc[c])>1e-6 else 1.0 for c in sg_cols}
    if len(sg_rows)<150: return {c: 0.70*comp[c]+0.30*HARDCODED_WEIGHTS[c] for c in sg_cols}
    return comp

def _run_monte_carlo(preds, n_sims=N_SIMULATIONS, seed=SEED):
    rng = np.random.RandomState(seed); n = len(preds)
    mu, sigma = preds.mean(), preds.std()
    z = (preds - mu)/sigma*TARGET_PRED_STD if sigma > 0 else np.zeros(n)
    wins=np.zeros(n); t5=np.zeros(n); t10=np.zeros(n); t20=np.zeros(n)
    for _ in range(n_sims):
        sim = z + rng.normal(0, NOISE_STD, n)
        ranks = sim.argsort().argsort() + 1
        wins+=(ranks==1); t5+=(ranks<=5); t10+=(ranks<=10); t20+=(ranks<=20)
    return {"win_prob":wins/n_sims,"top5_prob":t5/n_sims,"top10_prob":t10/n_sims,"top20_prob":t20/n_sims}


# ═══════════════════════════════════════════════════════════
# AUGUSTA FEATURES FOR STAGE 2
# ═══════════════════════════════════════════════════════════

AUGUSTA_FEATURES = [
    # 6 new experience features
    "augusta_competitive_rounds", "augusta_made_cut_prev_year",
    "augusta_experience_tier", "augusta_scoring_trajectory",
    "augusta_rounds_last_2yrs", "augusta_best_finish_recent",
    # Carried from V1/V2
    "augusta_starts", "augusta_made_cut_rate", "augusta_top10_rate",
    "augusta_best_finish", "augusta_scoring_avg",
    "augusta_sg_app_career", "augusta_sg_total_career",
    "augusta_bogey_avoidance", "augusta_round_variance_score",
    "augusta_back9_scoring",
    # Context
    "tournament_wind_avg", "tour_vs_augusta_divergence",
]

STAGE2_FEATURES = ROLLING_FEATURES + AUGUSTA_FEATURES


def _build_field_features(player_name, train_tour, course_weights, unified_df, features_df, year, weather_df):
    """Build full feature vector for one player in a backtest year."""
    player_tour = train_tour[train_tour["player_name"] == player_name].copy()
    if len(player_tour) == 0:
        return None

    pw = _apply_course_weights(player_tour, course_weights)
    pf = _build_rolling_features(pw)
    if len(pf) == 0:
        return None
    latest = pf.iloc[-1]

    feat = {"player_name": player_name}
    for col in ROLLING_FEATURES:
        feat[col] = latest.get(col)

    # Augusta features from precomputed features_df
    fr = features_df[(features_df["player_name"]==player_name)&(features_df["season"]==year)]
    if len(fr) > 0:
        fr = fr.iloc[0]
        for col in AUGUSTA_FEATURES:
            if col in ("tournament_wind_avg", "tour_vs_augusta_divergence"):
                continue
            feat[col] = fr.get(col)
    else:
        for col in AUGUSTA_FEATURES:
            if col in ("tournament_wind_avg", "tour_vs_augusta_divergence"):
                continue
            feat[col] = 0.0 if "rate" in col or "tier" in col or "rounds" in col or "starts" in col else None

    # Weather
    if weather_df is not None:
        wr = weather_df[weather_df["season"]==year]
        feat["tournament_wind_avg"] = wr["wind_avg_mph"].iloc[0] if len(wr)>0 and pd.notna(wr["wind_avg_mph"].iloc[0]) else 10.0
    else:
        feat["tournament_wind_avg"] = 10.0

    # tour_vs_augusta_divergence
    # = player's sg_total_8w percentile in field MINUS their scoring_avg percentile
    # among players with 3+ competitive rounds. Computed later in batch.
    feat["tour_vs_augusta_divergence"] = np.nan  # placeholder, computed in batch

    return feat


# ═══════════════════════════════════════════════════════════
# TASKS 2-4 — FULL BACKTEST WITH BLEND OPTIMIZATION
# ═══════════════════════════════════════════════════════════

def task2_3_4_backtest():
    print("\n" + "=" * 60)
    print("TASK 2/3/4 — AUGUSTA-ONLY STAGE 2 + BLEND OPT + BACKTEST")
    print("=" * 60)

    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)
    features_df = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)
    weather_df = pd.read_parquet(PROCESSED_DIR / "masters_weather.parquet") if (PROCESSED_DIR/"masters_weather.parquet").exists() else None

    sg_years = sorted(unified_df[unified_df["has_sg_data"]==True]["season"].unique())
    all_years = sorted(unified_df["season"].unique())
    backtest_years = [y for y in sg_years if y > min(all_years)]

    print(f"\n  Stage 2 features ({len(STAGE2_FEATURES)}):")
    for i, f in enumerate(STAGE2_FEATURES):
        tag = " *NEW*" if f in ["augusta_competitive_rounds","augusta_made_cut_prev_year",
                                 "augusta_experience_tier","augusta_scoring_trajectory",
                                 "augusta_rounds_last_2yrs","augusta_best_finish_recent",
                                 "tour_vs_augusta_divergence"] else ""
        print(f"    {i+1:2d}. {f}{tag}")

    print(f"\n  Backtest years: {backtest_years}")

    all_results = []
    year_metrics = []

    for year in backtest_years:
        print(f"\n  {'─'*55}")
        print(f"  BACKTESTING: {year}")
        print(f"  {'─'*55}")

        # ── Stage 1 ──
        train_tour = tour_df[tour_df["season"]<year].copy()
        train_tour = train_tour.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
        train_tour = train_tour[train_tour["finish_num"]<999]
        cw = _compute_course_weights_for_year(unified_df, year)
        tw = _apply_course_weights(train_tour, cw)
        tf = _build_rolling_features(tw)
        tf["finish_pct"] = ((tf["finish_num"]-1)/(tf["field_size"]-1)).clip(0,1)
        tc = tf.dropna(subset=ROLLING_FEATURES+["finish_pct"])

        model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
        model_s1.fit(tc[ROLLING_FEATURES].values, tc["finish_pct"].values)

        # ── Build field ──
        field = unified_df[(unified_df["season"]==year)&(unified_df["has_sg_data"]==True)].copy()
        if len(field) == 0: continue

        field_feats = []
        for _, pr in field.iterrows():
            ff = _build_field_features(pr["player_name"], train_tour, cw, unified_df, features_df, year, weather_df)
            if ff is None: continue
            ff["actual_finish_num"] = pr.get("finish_num")
            ff["actual_finish_pos"] = pr.get("finish_pos")
            field_feats.append(ff)

        field_df = pd.DataFrame(field_feats)
        field_df = field_df.dropna(subset=ROLLING_FEATURES)
        if len(field_df) < 5: continue

        # Compute tour_vs_augusta_divergence in batch
        sg8w_pctile = field_df["sg_total_8w"].rank(pct=True)
        has_rounds = field_df["augusta_competitive_rounds"] >= 6  # ~2 made cuts
        sa_vals = field_df.loc[has_rounds, "augusta_scoring_avg"]
        if len(sa_vals) > 2:
            # Lower scoring_avg = better (negative = under field median)
            # Rank so that better score = higher percentile
            sa_pctile = (-sa_vals).rank(pct=True)
            field_df.loc[has_rounds, "tour_vs_augusta_divergence"] = sg8w_pctile[has_rounds] - sa_pctile
        field_df["tour_vs_augusta_divergence"] = field_df["tour_vs_augusta_divergence"].fillna(0)

        # Stage 1 predictions + MC
        preds_s1 = model_s1.predict(field_df[ROLLING_FEATURES].values)
        mc = _run_monte_carlo(preds_s1)
        field_df["mc_win_prob"] = mc["win_prob"]
        field_df["mc_top5_prob"] = mc["top5_prob"]
        field_df["mc_top10_prob"] = mc["top10_prob"]
        field_df["mc_top20_prob"] = mc["top20_prob"]
        field_df["predicted_finish_pct"] = preds_s1

        # ── Stage 2: AUGUSTA-ONLY training ──
        # Use masters_unified rows with season < year as training data
        s2_train = unified_df[unified_df["season"] < year].copy()
        s2_train["finish_num"] = s2_train["finish_pos"].apply(_parse_finish_num)
        s2_train["made_top10"] = (s2_train["finish_num"] <= 10).astype(int)

        # Merge features
        s2_train = s2_train.merge(features_df, on=["player_name", "season"], how="left")

        # Add rolling SG features — for SG-era rows, get from tour data
        # For pre-SG rows, these will be NaN (XGBoost handles natively)
        s2_feat_rows = []
        for _, row in s2_train.iterrows():
            pname = row["player_name"]
            pseason = row["season"]
            feat = {"idx": row.name}

            # Try to get rolling features from tour data
            pt = tour_df[(tour_df["player_name"]==pname)&(tour_df["season"]<=pseason)].copy()
            pt = pt.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
            pt = pt[pt["finish_num"]<999]
            if len(pt) > 0:
                pw = _apply_course_weights(pt, cw)
                pf = _build_rolling_features(pw)
                latest = pf.iloc[-1]
                for col in ROLLING_FEATURES:
                    feat[col] = latest.get(col)
            else:
                for col in ROLLING_FEATURES:
                    feat[col] = np.nan

            # Weather
            if weather_df is not None:
                wr = weather_df[weather_df["season"]==pseason]
                feat["tournament_wind_avg"] = wr["wind_avg_mph"].iloc[0] if len(wr)>0 and pd.notna(wr.iloc[0]["wind_avg_mph"]) else 10.0
            else:
                feat["tournament_wind_avg"] = 10.0

            feat["tour_vs_augusta_divergence"] = 0.0  # not enough context for historical
            s2_feat_rows.append(feat)

        s2_feats = pd.DataFrame(s2_feat_rows).set_index("idx")
        for col in STAGE2_FEATURES:
            if col not in s2_train.columns and col in s2_feats.columns:
                s2_train[col] = s2_feats[col]
            elif col not in s2_train.columns:
                s2_train[col] = np.nan

        # Filter to rows with valid target
        s2_train = s2_train[s2_train["finish_num"].notna()].copy()
        n_pos = s2_train["made_top10"].sum()
        n_neg = len(s2_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 8.0

        s2_clf_params = {
            "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
            "reg_alpha": 0.5, "reg_lambda": 2.0, "objective": "binary:logistic",
            "eval_metric": "auc", "random_state": 42, "scale_pos_weight": spw,
        }

        X_s2 = s2_train[STAGE2_FEATURES]
        y_s2 = s2_train["made_top10"]

        model_s2 = xgb.XGBClassifier(**s2_clf_params)
        model_s2.fit(X_s2, y_s2)

        # Stage 2 predict on field
        s2_probs = model_s2.predict_proba(field_df[STAGE2_FEATURES])[:, 1]
        field_df["s2_top10_prob"] = s2_probs

        # Also train top-20
        s2_train["made_top20"] = (s2_train["finish_num"] <= 20).astype(int)
        n_pos20 = s2_train["made_top20"].sum()
        spw20 = (len(s2_train)-n_pos20)/n_pos20 if n_pos20>0 else 3.5
        s2_clf20 = {**s2_clf_params, "scale_pos_weight": spw20}
        model_s2_20 = xgb.XGBClassifier(**s2_clf20)
        model_s2_20.fit(X_s2, s2_train["made_top20"])
        field_df["s2_top20_prob"] = model_s2_20.predict_proba(field_df[STAGE2_FEATURES])[:, 1]

        # Store all blend variants
        field_df["model_win_prob"] = field_df["mc_win_prob"]
        field_df["model_top5_prob"] = field_df["mc_top5_prob"]
        field_df["season"] = year
        all_results.append(field_df)

        print(f"  S2 train: {len(s2_train)} Augusta rows (pos_rate={n_pos/len(s2_train):.1%}) | Field: {len(field_df)} players")

        # Quick metrics
        actual = field_df["actual_finish_num"].values
        valid = (actual<999)&pd.notna(actual)
        vdf = field_df[valid]
        if len(vdf)>0:
            at10 = (vdf["actual_finish_num"]<=10).astype(int)
            if at10.sum()>0 and at10.sum()<len(at10):
                s2_auc = roc_auc_score(at10, vdf["s2_top10_prob"])
                mc_auc = roc_auc_score(at10, vdf["mc_top10_prob"])
                sp, _ = stats.spearmanr(vdf["mc_win_prob"], -vdf["actual_finish_num"])
                print(f"  S2 AUC={s2_auc:.3f} | MC AUC={mc_auc:.3f} | Spearman={sp:.3f}")

    if not all_results:
        print("  No backtest results!")
        return [], None

    results_df = pd.concat(all_results, ignore_index=True)

    # ═══════════════════════════════════════════════════════
    # TASK 3 — BLEND WEIGHT OPTIMIZATION
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'=' * 60}")
    print(f"  TASK 3 — BLEND WEIGHT OPTIMIZATION")
    print(f"  {'=' * 60}")

    blend_weights = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]
    blend_results = []

    for w in blend_weights:
        aucs = []; precs = []; spears = []
        for yr in sorted(results_df["season"].unique()):
            ydf = results_df[results_df["season"]==yr]
            actual = ydf["actual_finish_num"].values
            valid = (actual<999)&pd.notna(actual)
            vdf = ydf[valid].copy()
            if len(vdf) < 5: continue

            blended = w * vdf["s2_top10_prob"] + (1-w) * vdf["mc_top10_prob"]
            at10 = (vdf["actual_finish_num"]<=10).astype(int)
            if at10.sum()>0 and at10.sum()<len(at10):
                aucs.append(roc_auc_score(at10, blended))
            top13 = blended.nlargest(13).index
            precs.append((vdf.loc[top13, "actual_finish_num"]<=10).mean())
            sp, _ = stats.spearmanr(blended, -vdf["actual_finish_num"])
            spears.append(sp)

        blend_results.append({
            "weight": w,
            "mean_auc": np.mean(aucs) if aucs else 0,
            "mean_prec": np.mean(precs) if precs else 0,
            "mean_spearman": np.mean(spears) if spears else 0,
        })

    print(f"\n  {'Weight':<8} | {'Mean AUC':>9} | {'T10 Prec@13':>12} | {'Spearman':>9}")
    print(f"  {'─'*8}-+-{'─'*9}-+-{'─'*12}-+-{'─'*9}")
    best_w = 0.6; best_auc = 0
    for br in blend_results:
        marker = ""
        if br["mean_auc"] > best_auc:
            best_auc = br["mean_auc"]; best_w = br["weight"]
        print(f"  {br['weight']:<8.2f} | {br['mean_auc']:>9.4f} | {br['mean_prec']:>11.1%} | {br['mean_spearman']:>9.3f}")

    # Find best by AUC
    best_br = max(blend_results, key=lambda x: x["mean_auc"])
    best_w = best_br["weight"]
    print(f"\n  OPTIMAL BLEND WEIGHT: {best_w} (AUC={best_br['mean_auc']:.4f})")

    # Apply optimal blend
    results_df["blended_top10_prob"] = best_w * results_df["s2_top10_prob"] + (1-best_w) * results_df["mc_top10_prob"]
    results_df["blended_top20_prob"] = best_w * results_df["s2_top20_prob"] + (1-best_w) * results_df["mc_top20_prob"]
    results_df["model_top10_prob"] = results_df["blended_top10_prob"]
    results_df["model_top20_prob"] = results_df["blended_top20_prob"]

    # ═══════════════════════════════════════════════════════
    # TASK 4 — FULL SUMMARY + JT DIAGNOSTIC
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'=' * 80}")
    print(f"  TASK 4 — FULL BACKTEST RESULTS (V3)")
    print(f"  {'=' * 80}")

    print(f"\n  {'Year':<6} | {'Brier(w)':<9} | {'Spearman':<9} | {'T10 Prec':<9} | {'S2 AUC10':<9} | {'Blend AUC':<10} | {'MC AUC':<8} | {'ROI(t10)':<9}")
    print(f"  {'─'*6}-+-{'─'*9}-+-{'─'*9}-+-{'─'*9}-+-{'─'*9}-+-{'─'*10}-+-{'─'*8}-+-{'─'*9}")

    final_metrics = []
    for yr in sorted(results_df["season"].unique()):
        ydf = results_df[results_df["season"]==yr]
        actual = ydf["actual_finish_num"].values
        valid = (actual<999)&pd.notna(actual)
        vdf = ydf[valid].copy()

        m = {"year": yr}
        # Brier
        ab = (actual<=1).astype(float)
        m["brier_win"] = np.mean((ydf["mc_win_prob"].values - ab)**2)

        # Spearman
        sp, _ = stats.spearmanr(vdf["mc_win_prob"], -vdf["actual_finish_num"])
        m["spearman"] = sp

        # T10 precision
        t10p = vdf.nlargest(10, "model_top10_prob")
        m["top10_prec"] = (t10p["actual_finish_num"]<=10).sum()/10

        # AUC
        at10 = (vdf["actual_finish_num"]<=10).astype(int)
        if at10.sum()>0 and at10.sum()<len(at10):
            m["s2_auc"] = roc_auc_score(at10, vdf["s2_top10_prob"])
            m["blend_auc"] = roc_auc_score(at10, vdf["blended_top10_prob"])
            m["mc_auc"] = roc_auc_score(at10, vdf["mc_top10_prob"])

        # ROI (top-10 market)
        naive_prob = 10/len(ydf)
        bets = ydf[ydf["model_top10_prob"] > naive_prob * 1.2]
        if len(bets)>0:
            hits = (bets["actual_finish_num"]<=10).sum()
            payout = 1.0/naive_prob
            profit = hits*payout - len(bets)
            m["roi_t10"] = profit/len(bets)*100
        else:
            m["roi_t10"] = 0

        final_metrics.append(m)
        s = lambda k, fmt=".3f": f"{m[k]:{fmt}}" if m.get(k) is not None else "N/A"
        print(f"  {yr:<6} | {s('brier_win','.4f'):<9} | {s('spearman'):<9} | {s('top10_prec','.0%'):<9} | "
              f"{s('s2_auc'):<9} | {s('blend_auc'):<10} | {s('mc_auc'):<8} | {s('roi_t10','.1f')+'%':<9}")

    # Averages
    def avg(k): vals=[m[k] for m in final_metrics if m.get(k) is not None]; return np.mean(vals) if vals else None
    avgs = {k: avg(k) for k in ["brier_win","spearman","top10_prec","s2_auc","blend_auc","mc_auc","roi_t10"]}
    sa = lambda v,f=".3f": f"{v:{f}}" if v is not None else "N/A"
    print(f"  {'AVG':<6} | {sa(avgs['brier_win'],'.4f'):<9} | {sa(avgs['spearman']):<9} | {sa(avgs['top10_prec'],'.0%'):<9} | "
          f"{sa(avgs['s2_auc']):<9} | {sa(avgs['blend_auc']):<10} | {sa(avgs['mc_auc']):<8} | {sa(avgs['roi_t10'],'.1f')+'%':<9}")

    # ── JT DIAGNOSTIC ──
    print(f"\n  {'─'*80}")
    print(f"  JT DIAGNOSTIC + KEY PLAYERS")
    print(f"  {'─'*80}")
    diag_players = [
        ("Justin Thomas", [2023, 2024, 2025]),
        ("Scottie Scheffler", [2024]),
        ("Jon Rahm", [2023]),
        ("Rory McIlroy", [2025]),
    ]
    print(f"  {'Player':<22} {'Yr':>4} | {'CompRds':>7} {'Traj':>6} {'Diverg':>7} {'S2_t10':>7} {'Rank':>5} | {'Actual':<8}")
    print(f"  {'─'*22} {'─'*4}-+-{'─'*7}-{'─'*6}-{'─'*7}-{'─'*7}-{'─'*5}-+-{'─'*8}")

    for player, years in diag_players:
        for yr in years:
            r = results_df[(results_df["player_name"]==player)&(results_df["season"]==yr)]
            if len(r)==0:
                print(f"  {player:<22} {yr:>4} | {'N/A':>7}")
                continue
            r = r.iloc[0]
            # Compute rank within year
            yr_df = results_df[results_df["season"]==yr]
            rank = (yr_df["blended_top10_prob"] > r["blended_top10_prob"]).sum() + 1
            print(f"  {player:<22} {yr:>4} | {r.get('augusta_competitive_rounds',0):>7.0f} "
                  f"{r.get('augusta_scoring_trajectory',0):>6.2f} "
                  f"{r.get('tour_vs_augusta_divergence',0):>7.2f} "
                  f"{r['s2_top10_prob']:>7.3f} {rank:>5} | {r['actual_finish_pos']:<8}")

    # Check if JT is still #1
    for yr in [2023, 2024, 2025]:
        yr_df = results_df[results_df["season"]==yr]
        top1 = yr_df.nlargest(1, "blended_top10_prob").iloc[0]
        jt = yr_df[yr_df["player_name"]=="Justin Thomas"]
        if len(jt)>0:
            jt_rank = (yr_df["blended_top10_prob"] > jt.iloc[0]["blended_top10_prob"]).sum() + 1
            print(f"\n  {yr}: JT rank = #{jt_rank} (blend_t10={jt.iloc[0]['blended_top10_prob']:.3f}), "
                  f"#1 = {top1['player_name']} ({top1['blended_top10_prob']:.3f})")

    results_df.to_parquet(PROCESSED_DIR / "backtest_results_v3.parquet", index=False)
    print("\n  Saved backtest_results_v3.parquet")

    print("\n  Tasks 2-4 complete.")
    return final_metrics, best_w, results_df


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AUGUSTA NATIONAL MODEL — V3 PIPELINE")
    print("  Augusta-only Stage 2, rebuilt experience features")
    print("=" * 60)

    features_df = task1_experience_features()
    final_metrics, best_w, results_df = task2_3_4_backtest()

    print("\n" + "=" * 60)
    print("  ALL V3 TASKS COMPLETE")
    print("=" * 60)
    return final_metrics, best_w


if __name__ == "__main__":
    main()

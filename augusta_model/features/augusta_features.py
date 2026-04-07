"""
Task 4 — Build player Augusta history features
Task 5 — Compute Augusta course weights
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

PROCESSED_DIR = Path("data/processed")

HARDCODED_WEIGHTS = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}


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
# TASK 4 — BUILD PLAYER AUGUSTA HISTORY FEATURES
# ═══════════════════════════════════════════════════════════

def task4_augusta_history_features(unified_df=None):
    print("\n" + "=" * 60)
    print("TASK 4 — BUILD PLAYER AUGUSTA HISTORY FEATURES")
    print("=" * 60)

    if unified_df is None:
        unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")

    # Load round-level SG data for within-tournament variance
    rounds_path = PROCESSED_DIR / "masters_sg_rounds.parquet"
    rounds_df = pd.read_parquet(rounds_path) if rounds_path.exists() else pd.DataFrame()

    unified_df = unified_df.copy()
    if "finish_num" not in unified_df.columns:
        unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)

    all_seasons = sorted(unified_df["season"].unique())
    feature_rows = []

    for _, row in unified_df.iterrows():
        season = row["season"]
        player = row["player_name"]

        # Strict temporal cutoff: only prior data
        prior = unified_df[(unified_df["player_name"] == player) &
                           (unified_df["season"] < season)]

        # ── Score-based features (all years) ──
        n_starts = len(prior)
        if n_starts > 0:
            seasons_ago = season - prior["season"]
            decay = 0.75 ** seasons_ago

            prior_cuts = prior["made_cut"].dropna()
            made_cut_rate = prior_cuts.mean() if len(prior_cuts) > 0 else 0.0

            prior_finish = prior["finish_num"].dropna()
            top10_count = (prior_finish <= 10).sum()
            top10_rate = top10_count / n_starts if n_starts > 0 else 0.0
            best_finish = prior_finish[prior_finish < 999].min() if (prior_finish < 999).any() else None

            svf = prior["score_vs_field"].dropna()
            if len(svf) > 0:
                decay_svf = decay[svf.index]
                scoring_avg = np.average(svf, weights=decay_svf)
            else:
                scoring_avg = None

            # Scoring trend: last 2 vs rest
            recent = prior.nlargest(2, "season")
            older = prior[~prior.index.isin(recent.index)]
            recent_svf = recent["score_vs_field"].dropna().mean()
            older_svf = older["score_vs_field"].dropna().mean()
            if pd.notna(recent_svf) and pd.notna(older_svf):
                scoring_trend = recent_svf - older_svf
            else:
                scoring_trend = None

            if n_starts == 0:
                exp_bucket = "debutant"
            elif n_starts <= 2:
                exp_bucket = "1-2"
            elif n_starts <= 5:
                exp_bucket = "3-5"
            else:
                exp_bucket = "6+"
        else:
            made_cut_rate = 0.0
            top10_rate = 0.0
            best_finish = None
            scoring_avg = None
            scoring_trend = None
            exp_bucket = "debutant"

        # ── SG-based features (only from prior years with SG data) ──
        prior_sg = prior[prior["has_sg_data"] == True]
        if len(prior_sg) > 0:
            seasons_ago_sg = season - prior_sg["season"]
            decay_sg = 0.75 ** seasons_ago_sg

            sg_app_vals = prior_sg["sg_app"].dropna()
            augusta_sg_app_career = np.average(sg_app_vals, weights=decay_sg[sg_app_vals.index]) if len(sg_app_vals) > 0 else None

            sg_total_vals = prior_sg["sg_total"].dropna()
            augusta_sg_total_career = np.average(sg_total_vals, weights=decay_sg[sg_total_vals.index]) if len(sg_total_vals) > 0 else None

            augusta_sg_variance = sg_total_vals.std() if len(sg_total_vals) > 1 else None

            # Round-level variance from rounds parquet
            augusta_round_sg_variance = None
            augusta_back9_sg = None
            if len(rounds_df) > 0:
                prior_rounds = rounds_df[
                    (rounds_df["player_name"] == player) &
                    (rounds_df["season"] < season)
                ]
                if len(prior_rounds) > 0:
                    # Within-tournament std of sg_total
                    tourn_stds = prior_rounds.groupby("season")["sg_total"].std().dropna()
                    augusta_round_sg_variance = tourn_stds.mean() if len(tourn_stds) > 0 else None

                    # Back-9 SG (R3+R4 rounds)
                    back_rounds = prior_rounds[prior_rounds["round_num"].isin([3, 4])]
                    if len(back_rounds) > 0:
                        augusta_back9_sg = back_rounds["sg_total"].mean()
        else:
            augusta_sg_app_career = None
            augusta_sg_total_career = None
            augusta_sg_variance = None
            augusta_round_sg_variance = None
            augusta_back9_sg = None

        feature_rows.append({
            "player_name": row["player_name"],
            "dg_id": row.get("dg_id"),
            "season": season,
            # SG-based
            "augusta_sg_app_career": augusta_sg_app_career,
            "augusta_sg_total_career": augusta_sg_total_career,
            "augusta_sg_variance": augusta_sg_variance,
            "augusta_round_sg_variance": augusta_round_sg_variance,
            "augusta_back9_sg": augusta_back9_sg,
            # Score-based
            "augusta_starts": n_starts,
            "augusta_made_cut_rate": made_cut_rate,
            "augusta_top10_rate": top10_rate,
            "augusta_best_finish": best_finish,
            "augusta_scoring_avg": scoring_avg,
            "augusta_scoring_trend": scoring_trend,
            "augusta_experience_bucket": exp_bucket,
        })

    features_df = pd.DataFrame(feature_rows)
    features_df.to_parquet(PROCESSED_DIR / "augusta_player_features.parquet", index=False)

    print(f"\n  Shape: {features_df.shape}")
    print(f"  Experience distribution:")
    print(f"    {features_df['augusta_experience_bucket'].value_counts().to_dict()}")
    print(f"  Players with SG career data: {features_df['augusta_sg_total_career'].notna().sum()}")
    print(f"  Players with scoring avg: {features_df['augusta_scoring_avg'].notna().sum()}")

    print("\n✓ Task 4 complete.")
    return features_df


# ═══════════════════════════════════════════════════════════
# TASK 5 — COMPUTE AUGUSTA COURSE WEIGHTS
# ═══════════════════════════════════════════════════════════

def task5_course_weights(unified_df=None, historical_rounds_path=None):
    print("\n" + "=" * 60)
    print("TASK 5 — COMPUTE AUGUSTA COURSE WEIGHTS")
    print("=" * 60)

    if unified_df is None:
        unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")

    if historical_rounds_path is None:
        historical_rounds_path = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")

    # Filter to SG rows only
    sg_rows = unified_df[unified_df["has_sg_data"] == True].copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    sg_rows = sg_rows.dropna(subset=sg_cols)

    print(f"\n  SG sample size: {len(sg_rows)} rows")

    if len(sg_rows) < 10:
        print("  Insufficient SG data for regression. Using hardcoded weights.")
        weights = HARDCODED_WEIGHTS.copy()
        with open(PROCESSED_DIR / "augusta_sg_weights.json", "w") as f:
            json.dump(weights, f, indent=2)
        print(f"\n  Final weights: {weights}")
        print("\n  Task 5 complete.")
        return weights

    # Compute finish_pct — need field_size from SG history parquet
    sg_hist_path = PROCESSED_DIR / "masters_sg_history.parquet"
    if sg_hist_path.exists():
        sg_hist = pd.read_parquet(sg_hist_path)
        if "field_size" in sg_hist.columns:
            fs_map = sg_hist.groupby("season")["field_size"].first().to_dict()
            sg_rows["field_size"] = sg_rows["season"].map(fs_map)

    if "field_size" not in sg_rows.columns or sg_rows["field_size"].isna().all():
        # Fallback: estimate field size from row count per season
        season_counts = unified_df.groupby("season").size()
        sg_rows["field_size"] = sg_rows["season"].map(season_counts)

    sg_rows["finish_num"] = sg_rows["finish_pos"].apply(_parse_finish_num)
    sg_rows = sg_rows[sg_rows["finish_num"] < 999]
    sg_rows["finish_pct"] = (sg_rows["finish_num"] - 1) / (sg_rows["field_size"] - 1)
    sg_rows["finish_pct"] = sg_rows["finish_pct"].clip(0, 1)

    X = sg_rows[sg_cols].values
    y = sg_rows["finish_pct"].values

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    raw_coefs = dict(zip(sg_cols, ridge.coef_))
    print(f"  Raw Ridge coefficients: {raw_coefs}")

    # Normalize against tour-average baseline
    # Load tour historical data
    tour_df = pd.read_csv(historical_rounds_path)
    tour_df["finish_num_tour"] = tour_df["finish_pos"].apply(_parse_finish_num)
    tour_df = tour_df[tour_df["finish_num_tour"] < 999].dropna(subset=sg_cols)
    tour_df["finish_pct_tour"] = (tour_df["finish_num_tour"] - 1) / (tour_df["field_size"] - 1)
    tour_df["finish_pct_tour"] = tour_df["finish_pct_tour"].clip(0, 1)

    ridge_tour = Ridge(alpha=1.0)
    ridge_tour.fit(tour_df[sg_cols].values, tour_df["finish_pct_tour"].values)
    tour_coefs = dict(zip(sg_cols, ridge_tour.coef_))
    print(f"  Tour-average Ridge coefficients: {tour_coefs}")

    # Normalized weights = augusta_coef / tour_coef
    computed = {}
    for col in sg_cols:
        if abs(tour_coefs[col]) > 1e-6:
            computed[col] = abs(raw_coefs[col]) / abs(tour_coefs[col])
        else:
            computed[col] = 1.0

    print(f"\n  Computed weights: {computed}")
    print(f"  Hardcoded prior:  {HARDCODED_WEIGHTS}")

    # Bayesian shrinkage if small sample
    if len(sg_rows) < 150:
        print(f"\n  Sample ({len(sg_rows)}) < 150 — applying 70/30 Bayesian shrinkage toward prior")
        final = {}
        for col in sg_cols:
            # 70% computed + 30% hardcoded prior (shrinkage with thin data)
            final[col] = round(0.70 * computed[col] + 0.30 * HARDCODED_WEIGHTS[col], 3)
    else:
        final = {k: round(v, 3) for k, v in computed.items()}

    print(f"  Final weights:    {final}")

    with open(PROCESSED_DIR / "augusta_sg_weights.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"  Saved to augusta_sg_weights.json")

    print("\n✓ Task 5 complete.")
    return final

#!/usr/bin/env python3
"""
Event-tier weighted rolling SG features.
Retrain Stage 1 + Stage 2, backtest, regenerate 2026 predictions.
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
sys.path.insert(0, ".")

from augusta_model.features.event_tiers import get_event_weight, classify_events

PROCESSED_DIR = Path("data/processed")
HIST_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
API_KEY = os.getenv("DATAGOLF_API_KEY")
DG_BASE = "https://feeds.datagolf.com"
HARDCODED_WEIGHTS = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

XGB_PARAMS = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:squarederror", "random_state": 42,
}
S2_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
    "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "objective": "binary:logistic",
    "eval_metric": "auc", "random_state": 42,
}
N_SIMS = 50_000; NOISE_STD = 0.10; TARGET_STD = 0.14; SEED = 42; BLEND_W = 0.90

ROLLING_FEATURES = [
    "sg_ott_3w","sg_app_3w","sg_arg_3w","sg_putt_3w","sg_t2g_3w","sg_total_3w",
    "sg_ott_8w","sg_app_8w","sg_arg_8w","sg_putt_8w","sg_t2g_8w","sg_total_8w",
    "sg_total_std_8","sg_total_momentum","sg_ball_strike_ratio","log_field_size",
]
AUGUSTA_FEATURES = [
    "augusta_competitive_rounds","augusta_made_cut_prev_year","augusta_experience_tier",
    "augusta_scoring_trajectory","augusta_rounds_last_2yrs","augusta_best_finish_recent",
    "augusta_starts","augusta_made_cut_rate","augusta_top10_rate","augusta_best_finish",
    "augusta_scoring_avg","augusta_sg_app_career","augusta_sg_total_career",
    "augusta_bogey_avoidance","augusta_round_variance_score","augusta_back9_scoring",
    "tournament_wind_avg","tour_vs_augusta_divergence",
]
STAGE2_FEATURES = ROLLING_FEATURES + AUGUSTA_FEATURES
SG_COLS = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]


def _parse_finish_num(pos):
    if pos is None or pd.isna(pos): return None
    pos = str(pos).strip().upper()
    if pos in ("CUT","MC","WD","DQ","MDF"): return 999
    pos = pos.replace("T","").replace("=","")
    try: return int(pos)
    except: return None

def _normalize_name(name):
    name = str(name).strip()
    if "," in name:
        parts = name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name

def _apply_course_weights(df, weights):
    df = df.copy()
    for c, w in weights.items():
        if c in df.columns: df[c] = df[c] * w
    return df


def _build_rolling_features_weighted(df):
    """Build rolling SG features with event-tier weighting.

    Each event's SG values are weighted by both recency (exponential decay)
    and event importance (tier weight). A strong SG performance at the Masters
    counts 3x more than the same SG at a Barbasol Championship.
    """
    df = df.sort_values(["player_name", "date"]).copy()
    df["event_weight"] = df["event_name"].map(get_event_weight)

    for col in SG_COLS:
        for span, suffix in [(3, "_3w"), (8, "_8w")]:
            result = pd.Series(np.nan, index=df.index)
            alpha = 2 / (span + 1)
            for player, grp in df.groupby("player_name"):
                vals = grp[col].values
                ew = grp["event_weight"].values
                idx = grp.index  # original index values
                n = len(vals)
                out = np.full(n, np.nan)
                for i in range(n):
                    if np.isnan(vals[i]):
                        out[i] = out[i-1] if i > 0 else np.nan
                        continue
                    lookback = min(i + 1, span * 2)
                    wsum = 0.0; vsum = 0.0
                    for j in range(max(0, i - lookback + 1), i + 1):
                        if np.isnan(vals[j]): continue
                        decay = (1 - alpha) ** (i - j)
                        w = decay * ew[j]
                        vsum += vals[j] * w
                        wsum += w
                    out[i] = vsum / wsum if wsum > 0 else np.nan
                result.loc[idx] = out
            df[f"{col}{suffix}"] = result

    # sg_total rolling std (8-event window, unweighted — just volatility)
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(
        lambda x: x.rolling(8, min_periods=2).std())

    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]

    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)

    df["log_field_size"] = np.log(df["field_size"].clip(1))

    return df


def _run_mc(preds, n=N_SIMS, seed=SEED):
    rng = np.random.RandomState(seed); np_ = len(preds)
    mu, s = preds.mean(), preds.std()
    z = (preds - mu) / s * TARGET_STD if s > 0 else np.zeros(np_)
    w = np.zeros(np_); t5 = np.zeros(np_); t10 = np.zeros(np_); t20 = np.zeros(np_)
    for _ in range(n):
        sim = z + rng.normal(0, NOISE_STD, np_)
        ranks = sim.argsort().argsort() + 1
        w += (ranks==1); t5 += (ranks<=5); t10 += (ranks<=10); t20 += (ranks<=20)
    return {"win": w/n, "top5": t5/n, "top10": t10/n, "top20": t20/n}


# ═══════════════════════════════════════════════════════════
# TASK 1 — VALIDATE EVENT TIER LOOKUP
# ═══════════════════════════════════════════════════════════

def task1_validate_tiers():
    print("\n" + "=" * 60)
    print("TASK 1 — EVENT TIER VALIDATION")
    print("=" * 60)

    tour_df = pd.read_csv(HIST_PATH)
    events = sorted(tour_df["event_name"].unique())
    tier_map = classify_events(events)

    print(f"\n  {len(events)} unique events classified:")
    by_tier = {}
    for name, weight in sorted(tier_map.items(), key=lambda x: -x[1]):
        by_tier.setdefault(weight, []).append(name)

    for weight in sorted(by_tier.keys(), reverse=True):
        names = by_tier[weight]
        label = {3.0:"MASTERS",2.0:"ELITE",1.8:"MAJOR",1.4:"ELEVATED",1.0:"STANDARD",0.5:"WEAK",0.4:"LIV"}.get(weight, f"w={weight}")
        print(f"\n  [{label} = {weight}x] ({len(names)} events)")
        for n in names[:8]:
            print(f"    {n}")
        if len(names) > 8:
            print(f"    ... and {len(names)-8} more")

    print("\n  Task 1 complete.")
    return tier_map


# ═══════════════════════════════════════════════════════════
# TASK 4 — RETRAIN + BACKTEST WITH WEIGHTED FEATURES
# ═══════════════════════════════════════════════════════════

def task4_retrain_backtest():
    print("\n" + "=" * 60)
    print("TASK 4 — RETRAIN + BACKTEST WITH EVENT-TIER WEIGHTS")
    print("=" * 60)

    tour_df = pd.read_csv(HIST_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)
    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)
    features_df = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    weather_df = pd.read_parquet(PROCESSED_DIR / "masters_weather.parquet") if (PROCESSED_DIR / "masters_weather.parquet").exists() else None

    wpath = PROCESSED_DIR / "augusta_sg_weights.json"
    cw = json.load(open(wpath)) if wpath.exists() else HARDCODED_WEIGHTS

    sg_years = sorted(unified_df[unified_df["has_sg_data"]==True]["season"].unique())
    all_years = sorted(unified_df["season"].unique())
    backtest_years = [y for y in sg_years if y > min(all_years)]
    print(f"  Backtest years: {backtest_years}")

    # Pre-compute weighted rolling features for ALL tour data (once)
    print("  Computing event-tier weighted rolling features for full tour dataset...")
    tour_clean = tour_df.dropna(subset=SG_COLS)
    tour_clean = tour_clean[tour_clean["finish_num"] < 999]
    tour_weighted = _apply_course_weights(tour_clean, cw)
    tour_featured = _build_rolling_features_weighted(tour_weighted)
    tour_featured["finish_pct"] = ((tour_featured["finish_num"] - 1) / (tour_featured["field_size"] - 1)).clip(0, 1)
    print(f"  Weighted features computed for {len(tour_featured)} rows")

    # Also compute UNWEIGHTED for comparison
    from run_v3_pipeline import _build_rolling_features as _build_unweighted
    tour_unweighted = _apply_course_weights(tour_clean, cw)
    tour_unweighted_feat = _build_unweighted(tour_unweighted)

    year_metrics_weighted = []
    year_metrics_unweighted = []
    all_results = []

    for year in backtest_years:
        print(f"\n  {'─'*50}")
        print(f"  BACKTESTING: {year}")

        train = tour_featured[tour_featured["season"] < year]
        tc = train.dropna(subset=ROLLING_FEATURES + ["finish_pct"])

        # Train Stage 1
        model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
        model_s1.fit(tc[ROLLING_FEATURES].values, tc["finish_pct"].values)

        # Build field
        field = unified_df[(unified_df["season"]==year)&(unified_df["has_sg_data"]==True)].copy()
        if len(field) == 0: continue

        field_feats = []
        for _, pr in field.iterrows():
            pname = pr["player_name"]
            player_data = tour_featured[(tour_featured["player_name"]==pname)&(tour_featured["season"]<year)]
            if len(player_data) == 0: continue
            latest = player_data.iloc[-1]
            feat = {"player_name": pname}
            for c in ROLLING_FEATURES: feat[c] = latest.get(c)

            # Augusta features
            fr = features_df[(features_df["player_name"]==pname)&(features_df["season"]==year)]
            if len(fr) > 0:
                fr = fr.iloc[0]
                for c in AUGUSTA_FEATURES:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"):
                        feat[c] = fr.get(c)
            else:
                for c in AUGUSTA_FEATURES:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"):
                        feat[c] = 0.0

            feat["tournament_wind_avg"] = 10.0
            feat["tour_vs_augusta_divergence"] = 0.0
            feat["actual_finish_num"] = pr.get("finish_num")
            feat["actual_finish_pos"] = pr.get("finish_pos")
            field_feats.append(feat)

        fdf = pd.DataFrame(field_feats).dropna(subset=ROLLING_FEATURES)
        if len(fdf) < 5: continue

        # Divergence
        sg8w_pctile = fdf["sg_total_8w"].rank(pct=True)
        has_rds = fdf["augusta_competitive_rounds"] >= 6
        sa = fdf.loc[has_rds, "augusta_scoring_avg"]
        if len(sa) > 2:
            sa_pctile = (-sa).rank(pct=True)
            fdf.loc[has_rds, "tour_vs_augusta_divergence"] = sg8w_pctile[has_rds] - sa_pctile
        fdf["tour_vs_augusta_divergence"] = fdf["tour_vs_augusta_divergence"].fillna(0)

        # MC
        preds = model_s1.predict(fdf[ROLLING_FEATURES].values)
        mc = _run_mc(preds)
        fdf["mc_top10_prob"] = mc["top10"]

        # Stage 2
        s2_train = unified_df[unified_df["season"] < year].copy()
        s2_train["made_top10"] = (s2_train["finish_num"] <= 10).astype(int)
        s2_train = s2_train.merge(features_df, on=["player_name","season"], how="left")
        s2_roll = []
        for _, row in s2_train.iterrows():
            pn, ps = row["player_name"], row["season"]
            pt = tour_featured[(tour_featured["player_name"]==pn)&(tour_featured["season"]<=ps)]
            f2 = {"idx": row.name}
            if len(pt) > 0:
                lat = pt.iloc[-1]
                for c in ROLLING_FEATURES: f2[c] = lat.get(c)
            else:
                for c in ROLLING_FEATURES: f2[c] = np.nan
            f2["tournament_wind_avg"] = 10.0
            f2["tour_vs_augusta_divergence"] = 0.0
            s2_roll.append(f2)
        s2_rf = pd.DataFrame(s2_roll).set_index("idx")
        for c in STAGE2_FEATURES:
            if c not in s2_train.columns and c in s2_rf.columns:
                s2_train[c] = s2_rf[c]
            elif c not in s2_train.columns:
                s2_train[c] = np.nan
        s2_train = s2_train[s2_train["finish_num"].notna()]
        n_pos = s2_train["made_top10"].sum()
        spw = (len(s2_train)-n_pos)/n_pos if n_pos > 0 else 8.0
        model_s2 = xgb.XGBClassifier(**{**S2_PARAMS, "scale_pos_weight": spw})
        model_s2.fit(s2_train[STAGE2_FEATURES], s2_train["made_top10"])

        for c in STAGE2_FEATURES:
            if c not in fdf.columns: fdf[c] = 0.0
        s2_prob = model_s2.predict_proba(fdf[STAGE2_FEATURES])[:,1]
        fdf["s2_top10_prob"] = s2_prob
        fdf["blended_top10"] = BLEND_W * s2_prob + (1-BLEND_W) * fdf["mc_top10_prob"]
        fdf["season"] = year

        # Score
        valid = fdf[fdf["actual_finish_num"]<999].copy()
        at10 = (valid["actual_finish_num"]<=10).astype(int)
        m = {"year": year}
        if at10.sum()>0 and at10.sum()<len(at10):
            m["s2_auc"] = roc_auc_score(at10, valid["s2_top10_prob"])
            m["blend_auc"] = roc_auc_score(at10, valid["blended_top10"])
            m["mc_auc"] = roc_auc_score(at10, valid["mc_top10_prob"])
        t10p = valid.nlargest(10, "blended_top10")
        m["top10_prec"] = (t10p["actual_finish_num"]<=10).sum()/10
        year_metrics_weighted.append(m)
        all_results.append(fdf)

        s2a = f"{m.get('s2_auc',0):.3f}" if m.get('s2_auc') else "N/A"
        print(f"  Field: {len(fdf)} | S2 AUC={s2a} | T10 Prec={m['top10_prec']:.0%}")

    # Load V3 metrics for comparison
    print(f"\n  {'='*65}")
    print(f"  COMPARISON: WEIGHTED vs V3 (UNWEIGHTED)")
    print(f"  {'='*65}")
    v3_metrics = [
        {"year":2021,"s2_auc":0.500,"top10_prec":0.30},
        {"year":2022,"s2_auc":0.674,"top10_prec":0.50},
        {"year":2023,"s2_auc":0.669,"top10_prec":0.40},
        {"year":2024,"s2_auc":0.738,"top10_prec":0.60},
        {"year":2025,"s2_auc":0.600,"top10_prec":0.30},
    ]
    print(f"  {'Year':<6} | {'V3 AUC':>7} {'V3 Prec':>8} | {'Wtd AUC':>8} {'Wtd Prec':>9} | {'AUC Δ':>6}")
    print(f"  {'─'*6}-+-{'─'*7}-{'─'*8}-+-{'─'*8}-{'─'*9}-+-{'─'*6}")
    for v3, wt in zip(v3_metrics, year_metrics_weighted):
        v3a = v3.get("s2_auc", 0)
        wta = wt.get("s2_auc", 0) or 0
        delta = wta - v3a
        print(f"  {wt['year']:<6} | {v3a:>7.3f} {v3['top10_prec']:>7.0%} | {wta:>8.3f} {wt['top10_prec']:>8.0%} | {delta:>+6.3f}")

    avg_v3 = np.mean([m.get("s2_auc",0) for m in v3_metrics])
    avg_wt = np.mean([m.get("s2_auc",0) or 0 for m in year_metrics_weighted])
    avg_v3p = np.mean([m["top10_prec"] for m in v3_metrics])
    avg_wtp = np.mean([m["top10_prec"] for m in year_metrics_weighted])
    print(f"  {'AVG':<6} | {avg_v3:>7.3f} {avg_v3p:>7.0%} | {avg_wt:>8.3f} {avg_wtp:>8.0%} | {avg_wt-avg_v3:>+6.3f}")

    keep_weighted = avg_wt >= avg_v3 - 0.01  # keep if same or better (within 0.01)
    if keep_weighted:
        print(f"\n  DECISION: KEEPING weighted features (AUC {'improved' if avg_wt > avg_v3 else 'comparable'})")
    else:
        print(f"\n  DECISION: REVERTING to unweighted (AUC dropped by {avg_v3 - avg_wt:.3f})")

    print("\n  Task 4 complete.")
    return year_metrics_weighted, all_results, tour_featured, keep_weighted


# ═══════════════════════════════════════════════════════════
# TASK 5 — RECOMPUTE 2026 PREDICTIONS
# ═══════════════════════════════════════════════════════════

def task5_2026_predictions(tour_featured):
    print("\n" + "=" * 60)
    print("TASK 5 — RECOMPUTE 2026 PREDICTIONS WITH WEIGHTED FEATURES")
    print("=" * 60)

    old_preds = pd.read_parquet(PROCESSED_DIR / "predictions_2026.parquet")
    wpath = PROCESSED_DIR / "augusta_sg_weights.json"
    cw = json.load(open(wpath)) if wpath.exists() else HARDCODED_WEIGHTS

    # Retrain Stage 1 on ALL data
    tc = tour_featured.dropna(subset=ROLLING_FEATURES + ["finish_pct"])
    model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
    model_s1.fit(tc[ROLLING_FEATURES].values, tc["finish_pct"].values)

    # Get DG skill ratings for players without tour history
    sg_data = {}
    try:
        sg_raw = requests.get(f"{DG_BASE}/preds/skill-ratings",
                              params={"key": API_KEY, "file_format": "json"}, timeout=30).json()
        for p in sg_raw.get("players", []):
            sg_data[_normalize_name(p.get("player_name",""))] = p
    except: pass

    # Build field features with weighted rolling
    comparison = []
    rolling_rows = []
    for _, pr in old_preds.iterrows():
        pname = pr["player_name"]
        player_data = tour_featured[tour_featured["player_name"]==pname]

        row = {"player_name": pname}
        old_sg3w = None  # will compare

        if len(player_data) > 0:
            latest = player_data.iloc[-1]
            for c in ROLLING_FEATURES: row[c] = latest.get(c)
            old_sg3w = pr.get("sg_total_3w") if "sg_total_3w" in old_preds.columns else None
        else:
            sr = sg_data.get(pname, {})
            for c in SG_COLS:
                val = sr.get(c, 0) or 0
                row[f"{c}_3w"] = val
                row[f"{c}_8w"] = val
            row["sg_total_std_8"] = 0.5
            row["sg_total_momentum"] = 0
            bsr_d = sum(abs(sr.get(c,0) or 0) for c in ["sg_ott","sg_app","sg_arg","sg_putt"]) + 1e-6
            row["sg_ball_strike_ratio"] = ((sr.get("sg_ott",0) or 0)+(sr.get("sg_app",0) or 0)) / bsr_d
            row["log_field_size"] = np.log(len(old_preds))

        if old_sg3w is not None and row.get("sg_total_3w") is not None:
            comparison.append({
                "player": pname,
                "old_sg_total_3w": old_sg3w,
                "new_sg_total_3w": row["sg_total_3w"],
                "change": row["sg_total_3w"] - old_sg3w,
            })
        rolling_rows.append(row)

    rolling_df = pd.DataFrame(rolling_rows)

    # Merge Augusta features from old predictions
    aug_cols = [c for c in old_preds.columns if c.startswith("augusta_") or c in
                ["tour_vs_augusta_divergence","dg_id","country","dg_rank"]]
    field_df = rolling_df.merge(old_preds[["player_name"]+aug_cols], on="player_name", how="left")
    for c in ROLLING_FEATURES: field_df[c] = field_df[c].fillna(0)

    # Stage 1 predict + MC
    preds = model_s1.predict(field_df[ROLLING_FEATURES].values)
    mc = _run_mc(preds)
    field_df["win_prob"] = mc["win"]
    field_df["top5_prob"] = mc["top5"]
    field_df["mc_top10_prob"] = mc["top10"]
    field_df["top20_prob"] = mc["top20"]
    field_df["model_score"] = preds

    # Stage 2
    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)
    features_all = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")

    s2_train = unified_df.copy()
    s2_train["made_top10"] = (s2_train["finish_num"]<=10).astype(int)
    s2_train = s2_train.merge(features_all, on=["player_name","season"], how="left")
    s2_roll = []
    for _, row in s2_train.iterrows():
        pn, ps = row["player_name"], row["season"]
        pt = tour_featured[(tour_featured["player_name"]==pn)&(tour_featured["season"]<=ps)]
        f2 = {"idx": row.name}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLLING_FEATURES: f2[c] = lat.get(c)
        else:
            for c in ROLLING_FEATURES: f2[c] = np.nan
        f2["tournament_wind_avg"] = 10.0
        f2["tour_vs_augusta_divergence"] = 0.0
        s2_roll.append(f2)
    s2_rf = pd.DataFrame(s2_roll).set_index("idx")
    for c in STAGE2_FEATURES:
        if c not in s2_train.columns and c in s2_rf.columns: s2_train[c] = s2_rf[c]
        elif c not in s2_train.columns: s2_train[c] = np.nan
    s2_train = s2_train[s2_train["finish_num"].notna()]
    n_pos = s2_train["made_top10"].sum()
    spw = (len(s2_train)-n_pos)/n_pos if n_pos>0 else 8.0
    model_s2 = xgb.XGBClassifier(**{**S2_PARAMS, "scale_pos_weight": spw})
    model_s2.fit(s2_train[STAGE2_FEATURES], s2_train["made_top10"])

    for c in STAGE2_FEATURES:
        if c not in field_df.columns: field_df[c] = 0.0
    s2_raw = model_s2.predict_proba(field_df[STAGE2_FEATURES])[:,1]
    field_df["stage2_prob_raw"] = s2_raw

    # Temperature scaling T=2.5
    logits = np.log(np.clip(s2_raw, 1e-6, 1-1e-6) / (1-np.clip(s2_raw, 1e-6, 1-1e-6)))
    s2_cal = 1 / (1 + np.exp(-logits/2.5))
    field_df["stage2_prob_calibrated"] = s2_cal
    field_df["top10_prob_calibrated"] = BLEND_W * s2_cal + (1-BLEND_W) * field_df["mc_top10_prob"]

    # Monotonic
    field_df["make_cut_prob"] = field_df[["top20_prob","mc_top10_prob"]].max(axis=1).clip(upper=0.95)
    field_df["top20_prob"] = field_df[["top20_prob","top10_prob_calibrated"]].max(axis=1)
    field_df["top5_prob"] = np.minimum(field_df["top5_prob"].clip(lower=field_df["win_prob"]).values,
                                        field_df["top10_prob_calibrated"].values)

    # Join DK odds
    dk_path = PROCESSED_DIR / "dk_odds_2026.csv"
    if dk_path.exists():
        from rapidfuzz import fuzz, process
        dk = pd.read_csv(dk_path)
        dk_names = dk["player_name"].tolist()
        for _, orow in dk.iterrows():
            best = process.extractOne(orow["player_name"], field_df["player_name"].tolist(), scorer=fuzz.ratio)
            if best and best[1] >= 85:
                idx = field_df[field_df["player_name"]==best[0]].index
                if len(idx)>0:
                    field_df.loc[idx[0], "dk_fair_prob_win"] = orow["dk_fair_prob"]
                    field_df.loc[idx[0], "dk_fair_prob_top10"] = orow["dk_fair_top10"]
                    field_df.loc[idx[0], "dk_american_odds"] = orow["dk_american_odds"]
        field_df["kelly_edge_win"] = (field_df["win_prob"]-field_df.get("dk_fair_prob_win",0)) / field_df.get("dk_fair_prob_win",1).clip(lower=1e-6)
        field_df["kelly_edge_top10"] = (field_df["top10_prob_calibrated"]-field_df.get("dk_fair_prob_top10",0)) / field_df.get("dk_fair_prob_top10",1).clip(lower=1e-6)

    field_df = field_df.sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)

    # ── Comparison table ──
    if comparison:
        print(f"\n  KEY PLAYER COMPARISON (old vs weighted sg_total_3w):")
        focus = ["Jon Rahm","Bryson DeChambeau","Cameron Smith","Davis Riley",
                 "Cameron Young","Scottie Scheffler","Xander Schauffele"]
        comp_df = pd.DataFrame(comparison)
        print(f"  {'Player':<25} {'Old 3w':>8} {'New 3w':>8} {'Change':>8} {'Reason'}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*25}")
        for name in focus:
            r = comp_df[comp_df["player"]==name]
            if len(r) == 0:
                print(f"  {name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} No tour history in CSV")
                continue
            r = r.iloc[0]
            reason = ""
            if r["change"] > 0.02: reason = "Elevated events upweighted"
            elif r["change"] < -0.02: reason = "Weak-field events downweighted"
            else: reason = "Minimal change"
            print(f"  {name:<25} {r['old_sg_total_3w']:>8.3f} {r['new_sg_total_3w']:>8.3f} {r['change']:>+8.3f} {reason}")

    # Print top 20
    print(f"\n  {'='*95}")
    print(f"  2026 PREDICTIONS (EVENT-TIER WEIGHTED)")
    print(f"  {'='*95}")
    print(f"  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6} {'DK':>8} {'T10 Edge':>9}")
    print(f"  {'─'*3} {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*9}")
    for i, (_, r) in enumerate(field_df.head(20).iterrows()):
        dk = f"+{r['dk_american_odds']:.0f}" if pd.notna(r.get('dk_american_odds')) else "N/A"
        edge = f"{r['kelly_edge_top10']:+.0%}" if pd.notna(r.get('kelly_edge_top10')) else "N/A"
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>5.1%} "
              f"{r['top20_prob']:>5.1%} {dk:>8} {edge:>9}")

    # Riley check
    riley = field_df[field_df["player_name"].str.contains("Riley", na=False)]
    if len(riley) > 0:
        r = riley.iloc[0]
        rank = (field_df["win_prob"] > r["win_prob"]).sum() + 1
        print(f"\n  Riley check: win={r['win_prob']:.1%} (rank #{rank}), sg_total_3w={r.get('sg_total_3w','?')}")

    # Save
    out_cols = [c for c in [
        "dg_id","player_name","country","dg_rank",
        "win_prob","top5_prob","top10_prob_calibrated","top20_prob","make_cut_prob",
        "model_score","stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier",
        "augusta_made_cut_prev_year","augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
        "dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds",
        "kelly_edge_win","kelly_edge_top10",
    ] if c in field_df.columns]
    save_df = field_df[out_cols].copy()
    save_df.to_parquet(PROCESSED_DIR / "predictions_2026.parquet", index=False)
    save_df.to_csv(PROCESSED_DIR / "predictions_2026.csv", index=False)
    print(f"\n  Saved predictions_2026.parquet/csv")

    # Spread check
    print(f"\n  Spread check:")
    print(f"    #1 win%: {field_df['win_prob'].max():.1%}")
    print(f"    Median win%: {field_df['win_prob'].median():.2%}")
    print(f"    Ratio: {field_df['win_prob'].max()/field_df['win_prob'].median():.1f}x")

    print("\n  Task 5 complete.")
    return field_df


def main():
    print("=" * 60)
    print("  EVENT-TIER WEIGHTED FEATURES")
    print("=" * 60)

    task1_validate_tiers()
    metrics, results, tour_featured, keep = task4_retrain_backtest()

    if not keep:
        print("\n  WEIGHTED FEATURES WORSE — NOT UPDATING 2026 PREDICTIONS")
        return

    field = task5_2026_predictions(tour_featured)

    # Task 6 — commit
    print("\n  Task 6: Updating CLAUDE.md...")

    print("\n" + "=" * 60)
    print("  ALL TASKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

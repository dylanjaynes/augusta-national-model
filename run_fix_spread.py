#!/usr/bin/env python3
"""
Fix probability spread + wire in real DK odds + recompute edge.
Tasks 1-4: MC recalibration, temperature scaling, DK odds, edge tables.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
import xgboost as xgb
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED_DIR = Path("data/processed")
HISTORICAL_ROUNDS_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
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
BLEND_W = 0.90

# FIXED MC params: widen spread
N_SIMS = 50_000; NOISE_STD = 0.10; TARGET_STD = 0.14; SEED = 42

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

def _build_rolling_features(df):
    df = df.sort_values(["player_name","date"]).copy()
    sg = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
    for c in sg:
        df[f"{c}_3w"] = df.groupby("player_name")[c].transform(lambda x: x.ewm(span=3,min_periods=1).mean())
        df[f"{c}_8w"] = df.groupby("player_name")[c].transform(lambda x: x.ewm(span=8,min_periods=1).mean())
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(lambda x: x.rolling(8,min_periods=2).std())
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"]+df["sg_app_8w"])/(df["sg_ott_8w"].abs()+df["sg_app_8w"].abs()+df["sg_arg_8w"].abs()+df["sg_putt_8w"].abs()+1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))
    return df

def _apply_course_weights(df, weights):
    df = df.copy()
    for c,w in weights.items():
        if c in df.columns: df[c] = df[c]*w
    return df

def _run_mc(preds, n=N_SIMS, noise_std=NOISE_STD, target_std=TARGET_STD, seed=SEED):
    """MC with corrected spread parameters."""
    rng = np.random.RandomState(seed); np_ = len(preds)
    mu, s = preds.mean(), preds.std()
    z = (preds - mu) / s * target_std if s > 0 else np.zeros(np_)
    w = np.zeros(np_); t5 = np.zeros(np_); t10 = np.zeros(np_); t20 = np.zeros(np_)
    for _ in range(n):
        sim = z + rng.normal(0, noise_std, np_)
        ranks = sim.argsort().argsort() + 1
        w += (ranks == 1); t5 += (ranks <= 5); t10 += (ranks <= 10); t20 += (ranks <= 20)
    return {"win": w/n, "top5": t5/n, "top10": t10/n, "top20": t20/n}


# ═══════════════════════════════════════════════════════════
# TASK 1 — FIX PROBABILITY SPREAD
# ═══════════════════════════════════════════════════════════

def task1_fix_spread():
    print("\n" + "=" * 60)
    print("TASK 1 — FIX PROBABILITY SPREAD")
    print("=" * 60)

    bt = pd.read_parquet(PROCESSED_DIR / "backtest_results_v3.parquet")

    # ── PROBLEM A: Fix MC spread ──
    print("\n  PROBLEM A: Fixing MC simulation spread...")
    print(f"  Old params: noise_std=0.16, target_pred_std=0.10")
    print(f"  New params: noise_std={NOISE_STD}, target_pred_std={TARGET_STD}")

    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)

    wpath = PROCESSED_DIR / "augusta_sg_weights.json"
    cw = json.load(open(wpath)) if wpath.exists() else HARDCODED_WEIGHTS

    train_tour = tour_df.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
    train_tour = train_tour[train_tour["finish_num"] < 999]
    tw = _apply_course_weights(train_tour, cw)
    tf = _build_rolling_features(tw)
    tf["finish_pct"] = ((tf["finish_num"] - 1) / (tf["field_size"] - 1)).clip(0, 1)
    tc = tf.dropna(subset=ROLLING_FEATURES + ["finish_pct"])

    model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
    model_s1.fit(tc[ROLLING_FEATURES].values, tc["finish_pct"].values)

    # Rebuild field with rolling features from scratch
    print("  Rebuilding field rolling features...")
    old_preds = pd.read_parquet(PROCESSED_DIR / "predictions_2026.parquet")

    # Pull skill ratings for players without tour history
    sg_data = {}
    try:
        sg_raw = requests.get(f"{DG_BASE}/preds/skill-ratings",
                              params={"key": API_KEY, "file_format": "json"}, timeout=30).json()
        for p in sg_raw.get("players", []):
            sg_data[_normalize_name(p.get("player_name", ""))] = p
    except: pass

    rolling_rows = []
    sg_cols = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
    for _, pr in old_preds.iterrows():
        pname = pr["player_name"]
        pt = train_tour[train_tour["player_name"] == pname].copy()
        row_data = {"player_name": pname}

        if len(pt) > 0:
            pw = _apply_course_weights(pt, cw)
            pf = _build_rolling_features(pw)
            latest = pf.iloc[-1]
            for c in ROLLING_FEATURES:
                row_data[c] = latest.get(c)
        else:
            # Use skill ratings as proxy
            sr = sg_data.get(pname, {})
            for c in sg_cols:
                val = sr.get(c, 0) or 0
                row_data[f"{c}_3w"] = val
                row_data[f"{c}_8w"] = val
            row_data["sg_total_std_8"] = 0.5
            row_data["sg_total_momentum"] = 0
            bsr_denom = sum(abs(sr.get(c, 0) or 0) for c in ["sg_ott","sg_app","sg_arg","sg_putt"]) + 1e-6
            row_data["sg_ball_strike_ratio"] = ((sr.get("sg_ott",0) or 0) + (sr.get("sg_app",0) or 0)) / bsr_denom
            row_data["log_field_size"] = np.log(len(old_preds))

        rolling_rows.append(row_data)

    rolling_df = pd.DataFrame(rolling_rows)
    # Merge rolling features + Augusta features from old_preds
    aug_cols = [c for c in old_preds.columns if c.startswith("augusta_") or c in
                ["tour_vs_augusta_divergence","dg_id","country","dg_rank"]]
    field_df = rolling_df.merge(old_preds[["player_name"] + aug_cols], on="player_name", how="left")

    for c in ROLLING_FEATURES:
        field_df[c] = field_df[c].fillna(0)

    preds_s1 = model_s1.predict(field_df[ROLLING_FEATURES].values)
    field_df["model_score"] = preds_s1

    # Run MC with fixed params
    mc = _run_mc(preds_s1)
    field_df["win_prob"] = mc["win"]
    field_df["top5_prob"] = mc["top5"]
    field_df["mc_top10_prob"] = mc["top10"]
    field_df["top20_prob"] = mc["top20"]

    print(f"  Win% range: {mc['win'].min():.3%} to {mc['win'].max():.3%}")
    print(f"  Win% median: {np.median(mc['win']):.3%}")
    print(f"  #1/#median ratio: {mc['win'].max()/np.median(mc['win']):.1f}x")

    # Riley diagnosis
    riley = field_df[field_df["player_name"].str.contains("Riley", na=False)]
    if len(riley) > 0:
        r = riley.iloc[0]
        print(f"\n  Davis Riley diagnosis:")
        print(f"    model_score (finish_pct): {r['model_score']:.4f}")
        print(f"    sg_total_3w: {r.get('sg_total_3w', 'N/A')}")
        print(f"    sg_total_8w: {r.get('sg_total_8w', 'N/A')}")
        print(f"    win_prob: {r['win_prob']:.3%}")
        print(f"    DK odds: +60000 = 0.17% implied")

    # ── PROBLEM B: Temperature scaling for S2 ──
    print("\n  PROBLEM B: Temperature scaling for Stage 2...")

    # Train Stage 2 fresh (same as v3)
    unified_all = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_all["finish_num"] = unified_all["finish_pos"].apply(_parse_finish_num)
    features_all = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    rounds_df = pd.read_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet")

    s2_train = unified_all.copy()
    s2_train["made_top10"] = (s2_train["finish_num"] <= 10).astype(int)
    s2_train = s2_train.merge(features_all, on=["player_name", "season"], how="left")

    # Build rolling features for training rows
    s2_roll = []
    for _, row in s2_train.iterrows():
        pn, ps = row["player_name"], row["season"]
        pt = tour_df[(tour_df["player_name"] == pn) & (tour_df["season"] <= ps)].copy()
        pt = pt.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
        pt = pt[pt["finish_num"] < 999]
        feat = {"idx": row.name}
        if len(pt) > 0:
            pw = _apply_course_weights(pt, cw)
            pf = _build_rolling_features(pw)
            latest = pf.iloc[-1]
            for c in ROLLING_FEATURES: feat[c] = latest.get(c)
        else:
            for c in ROLLING_FEATURES: feat[c] = np.nan
        feat["tournament_wind_avg"] = 10.0
        feat["tour_vs_augusta_divergence"] = 0.0
        s2_roll.append(feat)

    s2_rf = pd.DataFrame(s2_roll).set_index("idx")
    for c in STAGE2_FEATURES:
        if c not in s2_train.columns and c in s2_rf.columns:
            s2_train[c] = s2_rf[c]
        elif c not in s2_train.columns:
            s2_train[c] = np.nan

    s2_train = s2_train[s2_train["finish_num"].notna()]
    n_pos = s2_train["made_top10"].sum()
    spw = (len(s2_train) - n_pos) / n_pos if n_pos > 0 else 8.0

    model_s2 = xgb.XGBClassifier(**{**S2_PARAMS, "scale_pos_weight": spw})
    model_s2.fit(s2_train[STAGE2_FEATURES], s2_train["made_top10"])

    # S2 predict on 2026 field
    for c in STAGE2_FEATURES:
        if c not in field_df.columns: field_df[c] = 0.0
    s2_raw = model_s2.predict_proba(field_df[STAGE2_FEATURES])[:, 1]
    field_df["stage2_prob_raw"] = s2_raw

    # Temperature scaling: find optimal T on backtest data
    bt_valid = bt[bt["actual_finish_num"] < 999].copy()
    bt_valid["actual_t10"] = (bt_valid["actual_finish_num"] <= 10).astype(int)
    bt_s2_raw = bt_valid["s2_top10_prob"].values
    bt_actual = bt_valid["actual_t10"].values

    temps = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    print("\n  Temperature scaling search:")
    best_T = 1.0; best_brier = 999
    for T in temps:
        logits = np.log(np.clip(bt_s2_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(bt_s2_raw, 1e-6, 1 - 1e-6)))
        scaled = 1 / (1 + np.exp(-logits / T))
        brier = np.mean((scaled - bt_actual) ** 2)
        print(f"    T={T:.1f}: Brier={brier:.5f}")
        if brier < best_brier:
            best_brier = brier; best_T = T

    print(f"  Optimal T={best_T} (Brier={best_brier:.5f})")

    # Apply temperature scaling to 2026 predictions
    logits_2026 = np.log(np.clip(s2_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(s2_raw, 1e-6, 1 - 1e-6)))
    s2_cal = 1 / (1 + np.exp(-logits_2026 / best_T))
    field_df["stage2_prob_calibrated"] = s2_cal

    print(f"  S2 calibrated range: {s2_cal.min():.1%} to {s2_cal.max():.1%}")

    # Blend
    field_df["top10_prob_calibrated"] = BLEND_W * s2_cal + (1 - BLEND_W) * field_df["mc_top10_prob"]

    # ── Enforce monotonic constraints ──
    field_df["make_cut_prob"] = field_df[["top20_prob", "mc_top10_prob"]].max(axis=1).clip(upper=0.95)
    # Ensure win <= top5 <= top10 <= top20
    field_df["top20_prob"] = field_df[["top20_prob", "top10_prob_calibrated"]].max(axis=1)
    field_df["top5_prob"] = field_df[["top5_prob", "win_prob"]].max(axis=1)
    field_df["top5_prob"] = np.minimum(field_df["top5_prob"].values, field_df["top10_prob_calibrated"].values)

    # ── DeChambeau check ──
    db = field_df[field_df["player_name"].str.contains("DeChambeau", na=False)]
    if len(db) > 0:
        r = db.iloc[0]
        print(f"\n  DeChambeau check:")
        print(f"    win={r['win_prob']:.3%} t5={r['top5_prob']:.3%} t10={r['top10_prob_calibrated']:.1%} t20={r['top20_prob']:.1%}")
        print(f"    Expected win from t10: ~{r['top10_prob_calibrated']/10:.1%} (actual {r['win_prob']:.3%})")

    # Sort by top10
    field_df = field_df.sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)

    print(f"\n  FIXED spread diagnostics:")
    print(f"  Win% range: {field_df['win_prob'].min():.3%} to {field_df['win_prob'].max():.3%}")
    print(f"  Win% median: {field_df['win_prob'].median():.3%}")
    print(f"  #1/#median ratio: {field_df['win_prob'].max()/field_df['win_prob'].median():.1f}x")
    print(f"  T10% range: {field_df['top10_prob_calibrated'].min():.1%} to {field_df['top10_prob_calibrated'].max():.1%}")

    print("\n  Task 1 complete.")
    return field_df, best_T


# ═══════════════════════════════════════════════════════════
# TASK 2 — WIRE IN REAL DK ODDS
# ═══════════════════════════════════════════════════════════

def task2_dk_odds():
    print("\n" + "=" * 60)
    print("TASK 2 — WIRE IN REAL DK ODDS")
    print("=" * 60)

    # Fetch from The Odds API — DraftKings outrights
    print("\n  Fetching DraftKings odds from The Odds API...")
    dk_odds = {}
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/golf_masters_tournament_winner/odds/",
            params={"apiKey": os.getenv("ODDS_API_KEY"), "regions": "us",
                    "markets": "outrights", "oddsFormat": "american", "bookmakers": "draftkings"},
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                for event in data:
                    for bm in event.get("bookmakers", []):
                        if "draftkings" in bm.get("key", "").lower():
                            for mkt in bm.get("markets", []):
                                if mkt.get("key") == "outrights":
                                    for out in mkt.get("outcomes", []):
                                        dk_odds[out["name"]] = out["price"]
                print(f"  Fetched {len(dk_odds)} players from DK via API")
    except Exception as e:
        print(f"  API fetch failed: {e}")

    # Hardcode known odds as fallback/supplement
    hardcoded = {
        "Scottie Scheffler": 410, "Jon Rahm": 850, "Rory McIlroy": 1025,
        "Bryson DeChambeau": 1100, "Ludvig Aberg": 1750, "Xander Schauffele": 1850,
        "Cameron Young": 2400, "Tommy Fleetwood": 2500, "Matt Fitzpatrick": 2600,
        "Collin Morikawa": 3100, "Justin Rose": 3600, "Jordan Spieth": 3800,
        "Brooks Koepka": 3800, "Hideki Matsuyama": 3900, "Patrick Cantlay": 4000,
        "Max Homa": 4500, "Justin Thomas": 5500, "Tyrrell Hatton": 5000,
        "Sam Burns": 7200, "Sungjae Im": 7500, "Shane Lowry": 7500,
        "Viktor Hovland": 6000, "Cameron Smith": 6500, "Corey Conners": 8200,
        "Daniel Berger": 8000, "Brian Harman": 9000, "Akshay Bhatia": 10000,
        "Robert MacIntyre": 8500, "Keegan Bradley": 10000, "Wyndham Clark": 8000,
        "Sergio Garcia": 15000, "Si Woo Kim": 12000, "Jason Day": 12000,
        "Adam Scott": 15000, "Davis Riley": 60000, "Bubba Watson": 40000,
        "Fred Couples": 100000, "Zach Johnson": 30000, "Charl Schwartzel": 25000,
        "Danny Willett": 25000, "Matt McCarty": 25000, "Nick Taylor": 25000,
        "Sepp Straka": 20000, "Alex Noren": 25000, "Gary Woodland": 30000,
        "Dustin Johnson": 15000, "Ryan Fox": 20000, "Rasmus Hojgaard": 10000,
        "Nicolai Hojgaard": 12000, "Min Woo Lee": 15000, "Aaron Rai": 15000,
        "Jake Knapp": 20000, "Kurt Kitayama": 15000,
        "Tom McKibbin": 25000, "Chris Gotterup": 25000,
        "Harris English": 20000, "Ben Griffin": 30000,
        "Sahith Theegala": 12000,
    }

    # Merge: API overrides hardcoded
    combined = {**hardcoded}
    if dk_odds:
        for name, price in dk_odds.items():
            combined[name] = price

    print(f"  Total odds entries: {len(combined)}")

    # Build DataFrame
    rows = []
    for name, american in combined.items():
        if american > 0:
            decimal = 1 + american / 100
        else:
            decimal = 1 + 100 / abs(american)
        implied = 1 / decimal
        rows.append({
            "player_name": name,
            "dk_american_odds": american,
            "dk_decimal_odds": round(decimal, 3),
            "dk_implied_prob": round(implied, 6),
        })

    odds_df = pd.DataFrame(rows)
    overround = odds_df["dk_implied_prob"].sum()
    odds_df["dk_fair_prob"] = odds_df["dk_implied_prob"] / overround
    print(f"  Overround: {overround:.3f} ({overround*100:.1f}%)")

    # Estimate top-10 fair probs (8.5x multiplier, capped at 0.72)
    odds_df["dk_fair_top10"] = (odds_df["dk_fair_prob"] * 8.5).clip(upper=0.72)

    odds_df.to_csv(PROCESSED_DIR / "dk_odds_2026.csv", index=False)
    print(f"  Saved dk_odds_2026.csv: {len(odds_df)} players")

    # Print top 10 by fair prob
    top = odds_df.nlargest(10, "dk_fair_prob")
    print(f"\n  Top 10 by DK fair prob:")
    for _, r in top.iterrows():
        print(f"    {r['player_name']:<25} +{r['dk_american_odds']:<6.0f} fair={r['dk_fair_prob']:.1%} est_t10={r['dk_fair_top10']:.1%}")

    print("\n  Task 2 complete.")
    return odds_df


# ═══════════════════════════════════════════════════════════
# TASK 3 — RECOMPUTE EDGE WITH REAL ODDS
# ═══════════════════════════════════════════════════════════

def task3_edge(field_df, odds_df):
    print("\n" + "=" * 60)
    print("TASK 3 — RECOMPUTE EDGE WITH REAL ODDS")
    print("=" * 60)

    from rapidfuzz import fuzz, process

    odds_names = odds_df["player_name"].tolist()
    edge_rows = []

    for _, r in field_df.iterrows():
        pname = r["player_name"]
        best = process.extractOne(pname, odds_names, scorer=fuzz.ratio)
        if best and best[1] >= 85:
            orow = odds_df[odds_df["player_name"] == best[0]].iloc[0]
            mkt_win = orow["dk_fair_prob"]
            mkt_t10 = orow["dk_fair_top10"]
            dk_odds = orow["dk_american_odds"]
            matched = best[0]
            match_score = best[1]
        else:
            mkt_win = 1 / len(field_df)
            mkt_t10 = 10 / len(field_df)
            dk_odds = 99999
            matched = "NO MATCH"
            match_score = 0
            if best:
                print(f"  Low match: '{pname}' -> '{best[0]}' ({best[1]})")

        win_edge = (r["win_prob"] - mkt_win) / mkt_win if mkt_win > 0 else 0
        t10_edge = (r["top10_prob_calibrated"] - mkt_t10) / mkt_t10 if mkt_t10 > 0 else 0

        # Fade reasons
        reasons = []
        if r.get("tour_vs_augusta_divergence", 0) > 0.35:
            reasons.append("Tour form overpredicts Augusta")
        if r.get("augusta_scoring_trajectory", 0) < -1.5:
            reasons.append("Sharp Augusta decline")
        if r.get("augusta_made_cut_prev_year", 1) == 0 and r.get("augusta_starts", 0) > 0:
            reasons.append("No recent Augusta cut")

        edge_rows.append({
            "player_name": pname, "matched_odds_name": matched,
            "dk_american_odds": dk_odds,
            "model_win_pct": r["win_prob"], "dk_fair_win_pct": mkt_win,
            "win_kelly_edge": win_edge,
            "model_top10_pct": r["top10_prob_calibrated"],
            "dk_fair_top10_pct": mkt_t10,
            "top10_kelly_edge": t10_edge,
            "model_top20_pct": r["top20_prob"],
            "fade_reason": "; ".join(reasons) if reasons else "",
        })

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(PROCESSED_DIR / "edge_2026_real_odds.csv", index=False)

    # ── TABLE 1: Fair-priced ──
    fair = edge_df[edge_df["win_kelly_edge"].abs() < 0.30].sort_values("dk_fair_win_pct", ascending=False)
    print(f"\n  TABLE 1 — FAIR-PRICED (|win edge| < 30%): {len(fair)} players")
    print(f"  {'Player':<25} {'Model Win':>9} {'DK Fair':>8} {'Edge':>7}")
    for _, r in fair.head(10).iterrows():
        print(f"  {r['player_name']:<25} {r['model_win_pct']:>8.1%} {r['dk_fair_win_pct']:>7.1%} {r['win_kelly_edge']:>+6.0%}")

    # ── TABLE 2: Value plays (top-10 market) ──
    value = edge_df[edge_df["top10_kelly_edge"] > 0.20].sort_values("top10_kelly_edge", ascending=False)
    print(f"\n  TABLE 2 — VALUE PLAYS (top-10 edge > 20%): {len(value)} players")
    print(f"  {'Player':<25} {'Model T10':>9} {'Est Mkt':>8} {'Edge':>7} {'DK Win Odds':>11}")
    print(f"  {'─'*25} {'─'*9} {'─'*8} {'─'*7} {'─'*11}")
    for _, r in value.head(15).iterrows():
        print(f"  {r['player_name']:<25} {r['model_top10_pct']:>8.1%} {r['dk_fair_top10_pct']:>7.1%} {r['top10_kelly_edge']:>+6.0%} +{r['dk_american_odds']:<10.0f}")

    # ── TABLE 3: Fades ──
    fades = edge_df[edge_df["top10_kelly_edge"] < -0.25].sort_values("top10_kelly_edge")
    print(f"\n  TABLE 3 — FADES (top-10 edge < -25%): {len(fades)} players")
    print(f"  {'Player':<25} {'Model T10':>9} {'Est Mkt':>8} {'Edge':>7} {'DK Win Odds':>11} {'Reason'}")
    print(f"  {'─'*25} {'─'*9} {'─'*8} {'─'*7} {'─'*11} {'─'*30}")
    for _, r in fades.head(10).iterrows():
        reason = r["fade_reason"] if r["fade_reason"] else "Market pricing recent form"
        print(f"  {r['player_name']:<25} {r['model_top10_pct']:>8.1%} {r['dk_fair_top10_pct']:>7.1%} {r['top10_kelly_edge']:>+6.0%} +{r['dk_american_odds']:<10.0f} {reason}")

    print("\n  Task 3 complete.")
    return edge_df


# ═══════════════════════════════════════════════════════════
# TASK 4 — SAVE AND DEPLOY
# ═══════════════════════════════════════════════════════════

def task4_save(field_df, odds_df, edge_df):
    print("\n" + "=" * 60)
    print("TASK 4 — SAVE AND DEPLOY")
    print("=" * 60)

    # Join odds into predictions
    from rapidfuzz import fuzz, process
    odds_names = odds_df["player_name"].tolist()
    for _, orow in odds_df.iterrows():
        best = process.extractOne(orow["player_name"], field_df["player_name"].tolist(), scorer=fuzz.ratio)
        if best and best[1] >= 85:
            idx = field_df[field_df["player_name"] == best[0]].index
            if len(idx) > 0:
                field_df.loc[idx[0], "dk_fair_prob_win"] = orow["dk_fair_prob"]
                field_df.loc[idx[0], "dk_fair_prob_top10"] = orow["dk_fair_top10"]
                field_df.loc[idx[0], "dk_american_odds"] = orow["dk_american_odds"]

    # Add edge columns
    for _, erow in edge_df.iterrows():
        idx = field_df[field_df["player_name"] == erow["player_name"]].index
        if len(idx) > 0:
            field_df.loc[idx[0], "kelly_edge_win"] = erow["win_kelly_edge"]
            field_df.loc[idx[0], "kelly_edge_top10"] = erow["top10_kelly_edge"]

    # Fill NaN
    for c in ["dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds","kelly_edge_win","kelly_edge_top10"]:
        if c not in field_df.columns:
            field_df[c] = np.nan

    # Save
    out_cols = [
        "dg_id","player_name","country","dg_rank",
        "win_prob","top5_prob","top10_prob_calibrated","top20_prob","make_cut_prob",
        "model_score","stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier",
        "augusta_made_cut_prev_year","augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
        "dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds",
        "kelly_edge_win","kelly_edge_top10",
    ]
    save_df = field_df[[c for c in out_cols if c in field_df.columns]].copy()
    save_df = save_df.sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)
    save_df.to_parquet(PROCESSED_DIR / "predictions_2026.parquet", index=False)
    save_df.to_csv(PROCESSED_DIR / "predictions_2026.csv", index=False)

    # Also update edge_2026.csv for Streamlit
    edge_df.to_csv(PROCESSED_DIR / "edge_2026.csv", index=False)

    print(f"  Saved predictions_2026.parquet/csv: {len(save_df)} players")
    print(f"  Saved edge_2026.csv")

    # ── FINAL SUMMARY ──
    print(f"\n  {'='*95}")
    print(f"  FINAL 2026 PREDICTIONS — TOP 20")
    print(f"  {'='*95}")
    print(f"  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6} {'DK Odds':>8} {'T10 Edge':>9} {'Tier':>4} {'CutPv':>5} {'Traj':>6}")
    print(f"  {'─'*3} {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*9} {'─'*4} {'─'*5} {'─'*6}")
    for i, (_, r) in enumerate(save_df.head(20).iterrows()):
        dk = f"+{r['dk_american_odds']:.0f}" if pd.notna(r.get('dk_american_odds')) else "N/A"
        edge = f"{r['kelly_edge_top10']:+.0%}" if pd.notna(r.get('kelly_edge_top10')) else "N/A"
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>5.1%} "
              f"{r['top20_prob']:>5.1%} {dk:>8} {edge:>9} "
              f"{r['augusta_experience_tier']:>4.0f} {r['augusta_made_cut_prev_year']:>5.0f} "
              f"{r['augusta_scoring_trajectory']:>+6.2f}")

    # Spread check
    scheff_win = save_df.iloc[0]["win_prob"] if "Scheffler" in save_df.iloc[0]["player_name"] else save_df["win_prob"].max()
    median_win = save_df["win_prob"].median()
    print(f"\n  Spread check:")
    print(f"    Scheffler win%: {save_df[save_df['player_name'].str.contains('Scheffler',na=False)]['win_prob'].values[0]:.1%}")
    print(f"    Field median win%: {median_win:.2%}")
    print(f"    #1/#median ratio: {save_df['win_prob'].max()/median_win:.1f}x")

    # Value plays
    print(f"\n  TOP 10 VALUE PLAYS (top-10 market):")
    val = edge_df[edge_df["top10_kelly_edge"]>0.20].nlargest(10,"top10_kelly_edge")
    for _, r in val.iterrows():
        print(f"    {r['player_name']:<25} model={r['model_top10_pct']:.1%} mkt={r['dk_fair_top10_pct']:.1%} edge={r['top10_kelly_edge']:+.0%}")

    # Fades
    print(f"\n  TOP 5 FADES:")
    fad = edge_df.nsmallest(5,"top10_kelly_edge")
    for _, r in fad.iterrows():
        reason = r["fade_reason"] if r["fade_reason"] else "Market pricing recent form"
        print(f"    {r['player_name']:<25} model={r['model_top10_pct']:.1%} mkt={r['dk_fair_top10_pct']:.1%} edge={r['top10_kelly_edge']:+.0%} — {reason}")

    print("\n  Task 4 complete.")
    return save_df


def main():
    print("=" * 60)
    print("  FIX SPREAD + REAL ODDS")
    print("=" * 60)

    field_df, best_T = task1_fix_spread()
    odds_df = task2_dk_odds()
    edge_df = task3_edge(field_df, odds_df)
    save_df = task4_save(field_df, odds_df, edge_df)

    print("\n" + "=" * 60)
    print("  ALL FIXES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

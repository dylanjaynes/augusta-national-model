#!/usr/bin/env python3
"""
2026 Masters Predictions — Live field through V3 pipeline
Tasks 1-2: Pull field, build features, run model, fetch odds, compute edge
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import requests
import joblib
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED_DIR = Path("data/processed")
HISTORICAL_ROUNDS_PATH = Path("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
DG_BASE = "https://feeds.datagolf.com"
API_KEY = os.getenv("DATAGOLF_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
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
N_SIMS = 50_000; NOISE_STD = 0.16; TARGET_STD = 0.10; SEED = 42; BLEND_W = 0.90

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

def _dg_get(endpoint, params=None):
    if params is None: params = {}
    url = f"{DG_BASE}/{endpoint}"
    params = {"key": API_KEY, "file_format": "json", **params}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _normalize_name(name):
    """Convert 'Last, First' to 'First Last'."""
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

def _run_mc(preds, n=N_SIMS, seed=SEED):
    rng = np.random.RandomState(seed); np_ = len(preds)
    mu,s = preds.mean(), preds.std()
    z = (preds-mu)/s*TARGET_STD if s>0 else np.zeros(np_)
    w=np.zeros(np_);t5=np.zeros(np_);t10=np.zeros(np_);t20=np.zeros(np_)
    for _ in range(n):
        sim = z + rng.normal(0, NOISE_STD, np_)
        ranks = sim.argsort().argsort()+1
        w+=(ranks==1);t5+=(ranks<=5);t10+=(ranks<=10);t20+=(ranks<=20)
    return {"win":w/n,"top5":t5/n,"top10":t10/n,"top20":t20/n}


def task1_2026_predictions():
    print("\n" + "="*60)
    print("TASK 1 — 2026 MASTERS PREDICTIONS")
    print("="*60)

    # ── STEP 1: Pull field ──
    print("\n  STEP 1: Pulling current field...")
    field_data = _dg_get("field-updates", {"tour": "pga"})
    event_name = field_data.get("event_name", "Unknown")
    print(f"  Event: {event_name}")
    players_raw = field_data.get("field", [])
    field_rows = []
    for p in players_raw:
        field_rows.append({
            "dg_id": p.get("dg_id"), "player_name": _normalize_name(p.get("player_name","")),
            "country": p.get("country",""), "dg_rank": p.get("dg_rank"),
        })
    field_df = pd.DataFrame(field_rows)
    print(f"  Field size: {len(field_df)}")
    print(f"  First 10: {field_df['player_name'].head(10).tolist()}")

    # ── STEP 2: Pull skill ratings ──
    print("\n  STEP 2: Pulling skill ratings...")
    sg_data = _dg_get("preds/skill-ratings")
    sg_players = sg_data.get("players", [])
    sg_lookup = {p["dg_id"]: p for p in sg_players}
    # Also build name lookup for players missing dg_id match
    sg_name_lookup = {_normalize_name(p.get("player_name","")): p for p in sg_players}
    sg_cols = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
    for col in sg_cols:
        field_df[col] = field_df.apply(
            lambda r: sg_lookup.get(r["dg_id"], sg_name_lookup.get(r["player_name"], {})).get(col),
            axis=1
        )
    matched = field_df[sg_cols[0]].notna().sum()
    print(f"  Matched skill ratings: {matched}/{len(field_df)}")
    nulls = field_df[sg_cols].isnull().any(axis=1).sum()
    print(f"  Players with any null SG: {nulls}")

    # ── STEP 3: Build rolling features ──
    print("\n  STEP 3: Building rolling features...")
    tour_df = pd.read_csv(HISTORICAL_ROUNDS_PATH)
    tour_df["finish_num"] = tour_df["finish_pos"].apply(_parse_finish_num)

    # Load course weights
    wpath = PROCESSED_DIR / "augusta_sg_weights.json"
    cw = json.load(open(wpath)) if wpath.exists() else HARDCODED_WEIGHTS

    rolling_rows = []
    no_tour_data = []
    for _, pr in field_df.iterrows():
        pname = pr["player_name"]
        pt = tour_df[tour_df["player_name"]==pname].copy()
        pt = pt.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
        pt = pt[pt["finish_num"]<999]
        if len(pt) == 0:
            # Use current skill ratings as a single-row proxy
            row_data = {"player_name": pname}
            for c in sg_cols:
                row_data[c] = pr.get(c)
            # Create a synthetic "latest" with the current SG as both 3w and 8w
            for c in sg_cols:
                val = pr.get(c, 0) or 0
                row_data[f"{c}_3w"] = val
                row_data[f"{c}_8w"] = val
            row_data["sg_total_std_8"] = 0.5  # assume average volatility
            row_data["sg_total_momentum"] = 0
            bsr_denom = sum(abs(pr.get(c,0) or 0) for c in ["sg_ott","sg_app","sg_arg","sg_putt"]) + 1e-6
            row_data["sg_ball_strike_ratio"] = ((pr.get("sg_ott",0) or 0)+(pr.get("sg_app",0) or 0)) / bsr_denom
            row_data["log_field_size"] = np.log(len(field_df))
            rolling_rows.append(row_data)
            no_tour_data.append(pname)
            continue

        pw = _apply_course_weights(pt, cw)
        pf = _build_rolling_features(pw)
        latest = pf.iloc[-1]
        row_data = {"player_name": pname}
        for c in ROLLING_FEATURES:
            row_data[c] = latest.get(c)
        rolling_rows.append(row_data)

    rolling_df = pd.DataFrame(rolling_rows)
    field_df = field_df.merge(rolling_df, on="player_name", how="left")
    if no_tour_data:
        print(f"  Players with no tour history (using current SG): {len(no_tour_data)}")

    # ── STEP 4: Augusta history features ──
    print("\n  STEP 4: Building Augusta history features...")
    features_df = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    unified_df = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_df["finish_num"] = unified_df["finish_pos"].apply(_parse_finish_num)
    rounds_df = pd.read_parquet(PROCESSED_DIR / "masters_sg_rounds.parquet")

    # For 2026, compute features fresh with cutoff at 2026
    debutants = []
    aug_rows = []
    for _, pr in field_df.iterrows():
        pname = pr["player_name"]
        prior = unified_df[(unified_df["player_name"]==pname)&(unified_df["season"]<2026)]

        feat = {}
        n_starts = len(prior)
        if n_starts == 0:
            debutants.append(pname)
            feat = {c: 0 for c in AUGUSTA_FEATURES}
            feat["augusta_best_finish"] = len(field_df)+1
            feat["augusta_best_finish_recent"] = len(field_df)+1
            feat["player_name"] = pname
            aug_rows.append(feat)
            continue

        # rounds_played
        def _cr(row):
            cnt = sum(1 for c in ["r1_score","r2_score","r3_score","r4_score"] if pd.notna(row.get(c)))
            if cnt > 0: return cnt
            return 4 if row.get("made_cut")==1 else 2
        comp_rounds = sum(_cr(r) for _,r in prior.iterrows())
        prev_yr = prior[prior["season"]==2025]
        cut_prev = int(len(prev_yr)>0 and prev_yr.iloc[0].get("made_cut")==1)
        tier = 0 if comp_rounds==0 else (1 if comp_rounds<=7 else (2 if comp_rounds<=19 else (3 if comp_rounds<=35 else 4)))
        prior_cut = prior[prior["made_cut"]==1]
        if len(prior_cut)>=2:
            rec2 = prior_cut.nlargest(2,"season")
            rec_avg = rec2["score_vs_field"].dropna().mean()
            all_svf = prior_cut["score_vs_field"].dropna()
            if len(all_svf)>0:
                decay = 0.75**(2026-prior_cut.loc[all_svf.index,"season"])
                car_avg = np.average(all_svf, weights=decay)
                traj = car_avg - rec_avg if pd.notna(rec_avg) else 0
            else: traj = 0
        else: traj = 0
        rec_prior = prior[prior["season"].isin([2025,2024])]
        rds_2yr = sum(_cr(r) for _,r in rec_prior.iterrows())
        rec3 = prior.nlargest(3,"season")
        vf = rec3["finish_num"].dropna()
        vf_good = vf[vf<999]
        best_rec = vf_good.min() if len(vf_good)>0 else len(field_df)+1
        cuts = prior["made_cut"].dropna()
        cut_rate = cuts.mean() if len(cuts)>0 else 0
        fn = prior["finish_num"].dropna()
        t10_rate = (fn<=10).mean() if len(fn)>0 else 0
        best_f = fn[fn<999].min() if (fn<999).any() else 99
        svf = prior["score_vs_field"].dropna()
        d = 0.75**(2026-prior.loc[svf.index,"season"]) if len(svf)>0 else pd.Series()
        sc_avg = np.average(svf, weights=d) if len(svf)>0 else 0
        psg = prior[prior["has_sg_data"]==True]
        if len(psg)>0:
            sd = 0.75**(2026-psg["season"])
            sa = psg["sg_app"].dropna()
            sg_app_c = np.average(sa, weights=sd[sa.index]) if len(sa)>0 else None
            st = psg["sg_total"].dropna()
            sg_tot_c = np.average(st, weights=sd[st.index]) if len(st)>0 else None
        else: sg_app_c = None; sg_tot_c = None

        pr_rds = rounds_df[(rounds_df["player_name"]==pname)&(rounds_df["season"]<2026)]
        if len(pr_rds)>0:
            scores = pr_rds["score"].dropna()
            ba = (scores<=0).sum()/len(scores) if len(scores)>0 else 0
            rv = scores.std() if len(scores)>1 else None
            r34 = pr_rds[pr_rds["round_num"].isin([3,4])]["score"].dropna()
            b9 = r34.mean() if len(r34)>0 else None
        else: ba=0; rv=None; b9=None

        feat = {
            "player_name": pname,
            "augusta_competitive_rounds": comp_rounds,
            "augusta_made_cut_prev_year": cut_prev,
            "augusta_experience_tier": tier,
            "augusta_scoring_trajectory": traj,
            "augusta_rounds_last_2yrs": rds_2yr,
            "augusta_best_finish_recent": best_rec,
            "augusta_starts": n_starts, "augusta_made_cut_rate": cut_rate,
            "augusta_top10_rate": t10_rate, "augusta_best_finish": best_f,
            "augusta_scoring_avg": sc_avg,
            "augusta_sg_app_career": sg_app_c, "augusta_sg_total_career": sg_tot_c,
            "augusta_bogey_avoidance": ba, "augusta_round_variance_score": rv,
            "augusta_back9_scoring": b9,
            "tournament_wind_avg": 10.0,  # placeholder, updated in step 5
            "tour_vs_augusta_divergence": 0.0,  # placeholder, computed in batch
        }
        aug_rows.append(feat)

    aug_df = pd.DataFrame(aug_rows)
    field_df = field_df.merge(aug_df, on="player_name", how="left")
    print(f"  Debutants: {len(debutants)}")
    has_3plus = (field_df["augusta_starts"]>=3).sum()
    print(f"  Players with 3+ Augusta starts: {has_3plus}")

    # ── STEP 5: Weather ──
    print("\n  STEP 5: Fetching 2026 weather forecast...")
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":33.5,"longitude":-82.02,
            "daily":"wind_speed_10m_max,precipitation_sum,temperature_2m_mean",
            "timezone":"America/New_York",
            "start_date":"2026-04-09","end_date":"2026-04-12",
            "wind_speed_unit":"mph",
        }, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily",{})
        winds = [w for w in (daily.get("wind_speed_10m_max") or []) if w is not None]
        precips = [p for p in (daily.get("precipitation_sum") or []) if p is not None]
        temps = [t for t in (daily.get("temperature_2m_mean") or []) if t is not None]
        wind_avg = np.mean(winds) if winds else 10.0
        rain = sum(precips) if precips else 0
        temp_avg = np.mean(temps)*9/5+32 if temps else 70
        cond = "wet" if rain>5 else ("windy" if wind_avg>14 else ("moderate" if wind_avg>8 else "calm"))
        print(f"  Forecast: wind_avg={wind_avg:.1f}mph, rain={rain:.1f}mm, temp={temp_avg:.0f}F, conditions={cond}")
        field_df["tournament_wind_avg"] = wind_avg
    except Exception as e:
        print(f"  Weather fetch failed: {e} — using 10mph default")
        field_df["tournament_wind_avg"] = 10.0

    # Compute tour_vs_augusta_divergence in batch
    sg8w_pctile = field_df["sg_total_8w"].rank(pct=True)
    has_rds = field_df["augusta_competitive_rounds"]>=6
    sa_vals = field_df.loc[has_rds, "augusta_scoring_avg"]
    if len(sa_vals)>2:
        sa_pctile = (-sa_vals).rank(pct=True)
        field_df.loc[has_rds, "tour_vs_augusta_divergence"] = sg8w_pctile[has_rds] - sa_pctile
    field_df["tour_vs_augusta_divergence"] = field_df["tour_vs_augusta_divergence"].fillna(0)

    # ── STEP 6: Run models ──
    print("\n  STEP 6: Running Stage 1 + MC + Stage 2 + Blend...")

    # Train Stage 1 on all historical data
    train_tour = tour_df.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
    train_tour = train_tour[train_tour["finish_num"]<999]
    tw = _apply_course_weights(train_tour, cw)
    tf = _build_rolling_features(tw)
    tf["finish_pct"] = ((tf["finish_num"]-1)/(tf["field_size"]-1)).clip(0,1)
    tc = tf.dropna(subset=ROLLING_FEATURES+["finish_pct"])
    model_s1 = xgb.XGBRegressor(**XGB_PARAMS)
    model_s1.fit(tc[ROLLING_FEATURES].values, tc["finish_pct"].values)
    print(f"  Stage 1 trained on {len(tc)} tour rows")

    # Predict
    field_clean = field_df.dropna(subset=ROLLING_FEATURES[:6])  # need at least some SG
    if len(field_clean) < len(field_df):
        print(f"  Dropped {len(field_df)-len(field_clean)} players with no SG data")
    # Fill remaining NaN rolling features with 0 for players with partial data
    for c in ROLLING_FEATURES:
        field_clean[c] = field_clean[c].fillna(0)

    preds = model_s1.predict(field_clean[ROLLING_FEATURES].values)
    mc = _run_mc(preds)
    field_clean["win_prob"] = mc["win"]
    field_clean["top5_prob"] = mc["top5"]
    field_clean["mc_top10_prob"] = mc["top10"]
    field_clean["top20_prob"] = mc["top20"]
    field_clean["make_cut_prob"] = 1 - mc["top20"]*0  # approximate
    field_clean["model_score"] = preds

    # Train Stage 2 on all Augusta data
    unified_all = pd.read_parquet(PROCESSED_DIR / "masters_unified.parquet")
    unified_all["finish_num"] = unified_all["finish_pos"].apply(_parse_finish_num)
    s2_train = unified_all.copy()
    s2_train["made_top10"] = (s2_train["finish_num"]<=10).astype(int)

    # Merge Augusta features
    feat_all = pd.read_parquet(PROCESSED_DIR / "augusta_player_features.parquet")
    s2_train = s2_train.merge(feat_all, on=["player_name","season"], how="left")

    # Build rolling features for S2 training rows
    s2_roll = []
    for _, row in s2_train.iterrows():
        pn = row["player_name"]
        ps = row["season"]
        pt = tour_df[(tour_df["player_name"]==pn)&(tour_df["season"]<=ps)].copy()
        pt = pt.dropna(subset=["sg_ott","sg_app","sg_arg","sg_putt","sg_total"])
        pt = pt[pt["finish_num"]<999]
        feat = {"idx": row.name}
        if len(pt)>0:
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
    spw = (len(s2_train)-n_pos)/n_pos if n_pos>0 else 8.0
    s2_params = {**S2_PARAMS, "scale_pos_weight": spw}
    model_s2 = xgb.XGBClassifier(**s2_params)
    model_s2.fit(s2_train[STAGE2_FEATURES], s2_train["made_top10"])
    print(f"  Stage 2 trained on {len(s2_train)} Augusta rows ({n_pos} top-10s)")

    # Stage 2 predict
    for c in STAGE2_FEATURES:
        if c not in field_clean.columns: field_clean[c] = 0.0
    s2_raw = model_s2.predict_proba(field_clean[STAGE2_FEATURES])[:,1]
    field_clean["stage2_prob_raw"] = s2_raw

    # Platt calibration using backtest data
    print("  Applying Platt calibration...")
    bt = pd.read_parquet(PROCESSED_DIR / "backtest_results_v3.parquet")
    bt_valid = bt[bt["actual_finish_num"]<999].copy()
    bt_valid["actual_top10"] = (bt_valid["actual_finish_num"]<=10).astype(int)
    if len(bt_valid)>20 and bt_valid["actual_top10"].sum()>3:
        cal = LogisticRegression(C=1.0)
        cal.fit(bt_valid[["s2_top10_prob"]].values, bt_valid["actual_top10"].values)
        s2_cal = cal.predict_proba(field_clean[["stage2_prob_raw"]].values)[:,1]
        field_clean["stage2_prob_calibrated"] = s2_cal
        joblib.dump(cal, PROCESSED_DIR / "platt_calibrator.pkl")
        print(f"  Platt calibration fitted on {len(bt_valid)} backtest rows")
    else:
        field_clean["stage2_prob_calibrated"] = s2_raw
        print("  Insufficient backtest data for Platt — using raw probs")

    # Blend
    field_clean["top10_prob_raw"] = BLEND_W * s2_raw + (1-BLEND_W) * field_clean["mc_top10_prob"]
    field_clean["top10_prob_calibrated"] = BLEND_W * field_clean["stage2_prob_calibrated"] + (1-BLEND_W) * field_clean["mc_top10_prob"]

    # ── STEP 7: Save and print ──
    print("\n  STEP 7: Saving predictions...")
    out_cols = [
        "dg_id","player_name","country","dg_rank",
        "win_prob","top5_prob","top10_prob_raw","top10_prob_calibrated",
        "top20_prob","make_cut_prob","model_score",
        "stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier",
        "augusta_made_cut_prev_year","augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
    ]
    for c in out_cols:
        if c not in field_clean.columns: field_clean[c] = 0

    preds_df = field_clean[out_cols].sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)
    preds_df.to_parquet(PROCESSED_DIR / "predictions_2026.parquet", index=False)
    preds_df.to_csv(PROCESSED_DIR / "predictions_2026.csv", index=False)
    print(f"  Saved predictions_2026.parquet/csv: {len(preds_df)} players")

    # Print top 20
    print(f"\n  {'='*95}")
    print(f"  2026 MASTERS — TOP 20 PREDICTIONS")
    print(f"  {'='*95}")
    print(f"  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%(cal)':>9} {'T20%':>6} {'Tier':>4} {'CutPv':>5} {'Traj':>6} {'Flags'}")
    print(f"  {'─'*3} {'─'*25} {'─'*6} {'─'*9} {'─'*6} {'─'*4} {'─'*5} {'─'*6} {'─'*20}")
    for i, (_, r) in enumerate(preds_df.head(20).iterrows()):
        flags = []
        if r["tour_vs_augusta_divergence"] > 0.35: flags.append("FADE?")
        if r["augusta_experience_tier"] == 0: flags.append("DEBUT")
        if r["augusta_scoring_trajectory"] < -0.5: flags.append("DECLINING")
        flag_str = " ".join(flags)
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>8.1%} "
              f"{r['top20_prob']:>5.1%} {r['augusta_experience_tier']:>4.0f} {r['augusta_made_cut_prev_year']:>5.0f} "
              f"{r['augusta_scoring_trajectory']:>6.2f} {flag_str}")

    # Flag divergence warnings
    div_players = preds_df[preds_df["tour_vs_augusta_divergence"]>0.35]
    if len(div_players)>0:
        print(f"\n  DIVERGENCE WARNINGS (tour form may not translate):")
        for _, r in div_players.iterrows():
            print(f"    {r['player_name']:<25} divergence={r['tour_vs_augusta_divergence']:.2f}")

    print("\n  Task 1 complete.")
    return preds_df


def task2_odds_edge(preds_df):
    print("\n" + "="*60)
    print("TASK 2 — REAL ODDS + EDGE CALCULATION")
    print("="*60)

    # ── PART A: Fetch odds ──
    print("\n  PART A: Fetching live odds...")
    odds_source = "none"
    market_probs = {}

    # Try The Odds API
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/golf_masters_tournament_winner/odds/",
            params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "outrights", "oddsFormat": "american"},
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            if data and len(data)>0:
                # Find best book (Pinnacle > DraftKings > first)
                bookmakers = data[0].get("bookmakers", [])
                book = None
                for pref in ["pinnacle","draftkings","fanduel","betmgm"]:
                    for bm in bookmakers:
                        if pref in bm.get("key","").lower():
                            book = bm; break
                    if book: break
                if not book and bookmakers: book = bookmakers[0]

                if book:
                    odds_source = book.get("title", book.get("key","Unknown"))
                    markets = book.get("markets", [])
                    for mkt in markets:
                        if mkt.get("key") == "outrights":
                            raw_probs = {}
                            for outcome in mkt.get("outcomes", []):
                                name = outcome.get("name","")
                                price = outcome.get("price", 0)
                                # American to implied prob
                                if price > 0:
                                    prob = 100 / (price + 100)
                                elif price < 0:
                                    prob = abs(price) / (abs(price) + 100)
                                else:
                                    prob = 0
                                raw_probs[name] = prob
                            # Remove vig
                            total = sum(raw_probs.values())
                            if total > 0:
                                market_probs = {k: v/total for k,v in raw_probs.items()}
                    print(f"  Odds source: {odds_source} ({len(market_probs)} players)")
        else:
            print(f"  Odds API returned {r.status_code}")
    except Exception as e:
        print(f"  Odds API failed: {e}")

    # Fallback: DG pre-tournament
    if not market_probs:
        try:
            dg = _dg_get("preds/pre-tournament", {"tour": "pga"})
            baseline = dg.get("baseline", [])
            for p in baseline:
                market_probs[p.get("player_name","")] = p.get("win", 0)
            odds_source = "DataGolf model (proxy)"
            print(f"  Fallback: DG pre-tournament ({len(market_probs)} players)")
        except:
            pass

    # Last fallback: uniform
    if not market_probs:
        for _, r in preds_df.iterrows():
            market_probs[r["player_name"]] = 1.0 / len(preds_df)
        odds_source = "Uniform 1/N (no odds available)"
        print(f"  Using uniform baseline")

    # ── PART B: Match and compute edge ──
    print(f"\n  PART B: Computing edge (source: {odds_source})...")

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        import subprocess; subprocess.check_call([sys.executable,"-m","pip","install","rapidfuzz"])
        from rapidfuzz import fuzz, process

    market_names = list(market_probs.keys())
    edge_rows = []
    for _, r in preds_df.iterrows():
        pname = r["player_name"]
        # Fuzzy match
        best = process.extractOne(pname, market_names, scorer=fuzz.ratio) if market_names else None
        if best and best[1] >= 85:
            mkt_win = market_probs.get(best[0], 1/len(preds_df))
            matched_name = best[0]
        else:
            mkt_win = 1/len(preds_df)
            matched_name = pname

        # Scale win odds to top-10/top-20 approximation
        mkt_top10 = min(mkt_win * 10, 0.95)
        mkt_top20 = min(mkt_win * 20, 0.95)

        # Edge calculations
        win_edge = (r["win_prob"] - mkt_win) / mkt_win if mkt_win > 0 else 0
        t10_edge = (r["top10_prob_calibrated"] - mkt_top10) / mkt_top10 if mkt_top10 > 0 else 0
        t20_edge = (r["top20_prob"] - mkt_top20) / mkt_top20 if mkt_top20 > 0 else 0

        # Recommendation
        if t10_edge > 0.40: rec = "Strong Value"
        elif t10_edge > 0.20: rec = "Value"
        elif t10_edge > 0: rec = "Slight lean"
        elif t10_edge > -0.20: rec = "Fair"
        elif t10_edge > -0.30: rec = "Avoid"
        else: rec = "Fade"

        edge_rows.append({
            "player_name": pname, "matched_odds_name": matched_name,
            "model_win_pct": r["win_prob"], "market_win_pct": mkt_win,
            "win_edge_pct": win_edge,
            "model_top10_pct": r["top10_prob_calibrated"],
            "market_top10_pct": mkt_top10, "top10_edge_pct": t10_edge,
            "model_top20_pct": r["top20_prob"],
            "market_top20_pct": mkt_top20, "top20_edge_pct": t20_edge,
            "recommendation": rec, "odds_source": odds_source,
        })

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(PROCESSED_DIR / "edge_2026.csv", index=False)
    print(f"  Saved edge_2026.csv: {len(edge_df)} players")

    # ── PART C: Print edge tables ──
    print(f"\n  {'='*90}")
    print(f"  TOP 15 VALUE PLAYS — TOP-10 MARKET (source: {odds_source})")
    print(f"  {'='*90}")
    value = edge_df.sort_values("top10_edge_pct", ascending=False).head(15)
    print(f"  {'Player':<25} {'Model T10%':>10} {'Market T10%':>11} {'Edge%':>7} {'Rec':<15}")
    print(f"  {'─'*25} {'─'*10} {'─'*11} {'─'*7} {'─'*15}")
    for _, r in value.iterrows():
        print(f"  {r['player_name']:<25} {r['model_top10_pct']:>9.1%} {r['market_top10_pct']:>10.1%} "
              f"{r['top10_edge_pct']:>+6.0%} {r['recommendation']:<15}")

    # Fades
    fades = edge_df[edge_df["top10_edge_pct"] < -0.30].sort_values("top10_edge_pct")
    if len(fades) > 0:
        print(f"\n  FADE CANDIDATES (model significantly below market):")
        for _, r in fades.head(10).iterrows():
            print(f"    {r['player_name']:<25} model={r['model_top10_pct']:.1%} market={r['market_top10_pct']:.1%} edge={r['top10_edge_pct']:+.0%}")

    print("\n  Task 2 complete.")
    return edge_df


def main():
    print("="*60)
    print("  2026 MASTERS — LIVE PREDICTIONS")
    print("="*60)
    preds = task1_2026_predictions()
    edge = task2_odds_edge(preds)
    print("\n" + "="*60)
    print("  PREDICTIONS AND ODDS COMPLETE")
    print("="*60)
    return preds, edge

if __name__ == "__main__":
    main()

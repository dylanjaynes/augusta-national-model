#!/usr/bin/env python3
"""
Integrate real DG field strength weights into rolling SG features.
Retrain, backtest, regenerate 2026 predictions.
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
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED = Path("data/processed")
HIST_PATH = PROCESSED / "historical_rounds_extended.parquet"
FS_PATH = PROCESSED / "dg_field_strength.csv"
API_KEY = os.getenv("DATAGOLF_API_KEY")
DG_BASE = "https://feeds.datagolf.com"
HARDCODED_CW = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

XGB_S1 = {"n_estimators":600,"learning_rate":0.02,"max_depth":4,"subsample":0.8,
           "colsample_bytree":0.8,"min_child_weight":10,"reg_alpha":0.1,"reg_lambda":1.0,
           "objective":"reg:squarederror","random_state":42}
XGB_S2 = {"n_estimators":300,"learning_rate":0.03,"max_depth":3,"subsample":0.8,
           "colsample_bytree":0.7,"min_child_weight":5,"reg_alpha":0.5,"reg_lambda":2.0,
           "objective":"binary:logistic","eval_metric":"auc","random_state":42}
N_SIMS=50000; NOISE=0.10; TPRED=0.14; SEED=42; BLEND=0.90; TEMP=2.5

SG = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
ROLL = ["sg_ott_3w","sg_app_3w","sg_arg_3w","sg_putt_3w","sg_t2g_3w","sg_total_3w",
        "sg_ott_8w","sg_app_8w","sg_arg_8w","sg_putt_8w","sg_t2g_8w","sg_total_8w",
        "sg_total_std_8","sg_total_momentum","sg_ball_strike_ratio","log_field_size"]
AUG = ["augusta_competitive_rounds","augusta_made_cut_prev_year","augusta_experience_tier",
       "augusta_scoring_trajectory","augusta_rounds_last_2yrs","augusta_best_finish_recent",
       "augusta_starts","augusta_made_cut_rate","augusta_top10_rate","augusta_best_finish",
       "augusta_scoring_avg","augusta_sg_app_career","augusta_sg_total_career",
       "augusta_bogey_avoidance","augusta_round_variance_score","augusta_back9_scoring",
       "tournament_wind_avg","tour_vs_augusta_divergence"]
S2F = ROLL + AUG

def pfn(p):
    if p is None or pd.isna(p): return None
    p=str(p).strip().upper()
    if p in ("CUT","MC","WD","DQ","MDF"): return 999
    p=p.replace("T","").replace("=","")
    try: return int(p)
    except: return None

def norm_name(n):
    n=str(n).strip()
    if "," in n: parts=n.split(",",1); return f"{parts[1].strip()} {parts[0].strip()}"
    return n

def apply_cw(df, w):
    df = df.copy()
    for c, wt in w.items():
        if c in df.columns: df[c] = df[c] * wt
    return df

def run_mc(p):
    rng=np.random.RandomState(SEED);n=len(p);mu,s=p.mean(),p.std()
    z=(p-mu)/s*TPRED if s>0 else np.zeros(n)
    w=np.zeros(n);t5=np.zeros(n);t10=np.zeros(n);t20=np.zeros(n)
    for _ in range(N_SIMS):
        sim=z+rng.normal(0,NOISE,n);r=sim.argsort().argsort()+1
        w+=(r==1);t5+=(r<=5);t10+=(r<=10);t20+=(r<=20)
    return {"win":w/N_SIMS,"top5":t5/N_SIMS,"top10":t10/N_SIMS,"top20":t20/N_SIMS}


# ═══════════════════════════════════════════════════════════
# TASK 1 + 2 — FIELD STRENGTH LOOKUP + WEIGHTED ROLLING
# ══════��════════════════════════════════════════════════════

# Hardcoded tier fallback for pre-2019 events
_TIER_RULES = [
    ("masters", 1.12), ("players championship", 1.12),
    ("wgc", 1.10), ("world golf championships", 1.10),
    ("tour championship", 1.40), ("bmw championship", 1.10),
    ("fedex", 1.10), ("u.s. open", 1.00), ("pga championship", 1.00),
    ("the open", 1.05),
    ("genesis", 1.05), ("arnold palmer", 1.05), ("memorial", 1.05),
    ("rbc heritage", 1.03), ("travelers", 1.03),
    ("barracuda", 0.70), ("barbasol", 0.70), ("puerto rico", 0.70),
    ("sanderson farms", 0.85), ("fortinet", 0.85), ("shriners", 0.85),
    ("korn ferry", 0.66), ("nationwide", 0.66),
    ("liv", 0.96),
]

def _tier_fallback(event_name):
    if not event_name or not isinstance(event_name, str): return 1.0
    lower = event_name.lower()
    best = 1.0
    for pattern, weight in _TIER_RULES:
        if pattern in lower: best = max(best, weight) if weight > 1.0 else min(best, weight)
    return best


class FieldStrengthLookup:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Build lookup: (year, event_name_lower) -> fs_mean
        self._exact = {}
        self._by_year = {}
        for _, r in self.df.iterrows():
            key = (int(r["year"]), r["event_name"].strip().lower())
            self._exact[key] = r["field_strength_mean"]
            yr = int(r["year"])
            if yr not in self._by_year:
                self._by_year[yr] = []
            self._by_year[yr].append((r["event_name"].strip(), r["field_strength_mean"]))

    def get_weight(self, event_name, year):
        """Get field strength weight. Real FS data for 2019+, fallback for earlier."""
        if not event_name or pd.isna(event_name):
            return 1.0

        event_lower = str(event_name).strip().lower()
        year = int(year) if not pd.isna(year) else 0

        # Try exact match
        key = (year, event_lower)
        if key in self._exact:
            fs = self._exact[key]
            return max(0.4, min(3.0, 1.0 + 0.3 * fs))

        # Try fuzzy match within year
        if year in self._by_year:
            names = [n for n, _ in self._by_year[year]]
            best = process.extractOne(event_lower, [n.lower() for n in names], scorer=fuzz.ratio)
            if best and best[1] >= 80:
                idx = [n.lower() for n in names].index(best[0])
                fs = self._by_year[year][idx][1]
                return max(0.4, min(3.0, 1.0 + 0.3 * fs))

        # Fallback to hardcoded tiers for pre-2019 or unmatched
        return _tier_fallback(event_name)


def build_roll_fs(df, fs_lookup):
    """Build rolling SG features with field-strength weighting."""
    df = df.sort_values(["player_name", "date"]).copy()

    # Get field strength weight for each row
    df["_fs_weight"] = df.apply(lambda r: fs_lookup.get_weight(r["event_name"], r["season"]), axis=1)

    for col in SG:
        for span, suffix in [(3, "_3w"), (8, "_8w")]:
            alpha = 2 / (span + 1)
            result = pd.Series(np.nan, index=df.index)
            for player, grp in df.groupby("player_name"):
                vals = grp[col].values
                ew = grp["_fs_weight"].values
                idx = grp.index
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

    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(
        lambda x: x.rolling(8, min_periods=2).std())
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))
    return df


def build_roll_plain(df):
    """Build rolling SG features WITHOUT field-strength weighting (baseline)."""
    df = df.sort_values(["player_name", "date"]).copy()
    for c in SG:
        df[f"{c}_3w"] = df.groupby("player_name")[c].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        df[f"{c}_8w"] = df.groupby("player_name")[c].transform(lambda x: x.ewm(span=8, min_periods=1).mean())
    df["sg_total_std_8"] = df.groupby("player_name")["sg_total"].transform(lambda x: x.rolling(8, min_periods=2).std())
    df["sg_total_momentum"] = df["sg_total_3w"] - df["sg_total_8w"]
    df["sg_ball_strike_ratio"] = (df["sg_ott_8w"] + df["sg_app_8w"]) / (
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() + df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))
    return df


def run_backtest(tour_feat, tag=""):
    """Run walk-forward backtest, return metrics list."""
    unified = pd.read_parquet(PROCESSED / "masters_unified.parquet")
    unified["finish_num"] = unified["finish_pos"].apply(pfn)
    features = pd.read_parquet(PROCESSED / "augusta_player_features.parquet")

    sg_years = sorted(unified[unified["has_sg_data"] == True]["season"].unique())
    bt_years = [y for y in sg_years if y > min(unified["season"].unique())]
    metrics = []

    for year in bt_years:
        train = tour_feat[tour_feat["season"] < year].dropna(subset=ROLL + ["finish_pct"])
        s1 = xgb.XGBRegressor(**XGB_S1)
        s1.fit(train[ROLL].values, train["finish_pct"].values)

        field = unified[(unified["season"] == year) & (unified["has_sg_data"] == True)].copy()
        if len(field) == 0: continue
        ff = []
        for _, pr in field.iterrows():
            pn = pr["player_name"]
            pt = tour_feat[(tour_feat["player_name"] == pn) & (tour_feat["season"] < year)]
            if len(pt) == 0: continue
            lat = pt.iloc[-1]
            d = {"player_name": pn}
            for c in ROLL: d[c] = lat.get(c)
            fr = features[(features["player_name"] == pn) & (features["season"] == year)]
            if len(fr) > 0:
                fr = fr.iloc[0]
                for c in AUG:
                    if c not in ("tournament_wind_avg", "tour_vs_augusta_divergence"): d[c] = fr.get(c)
            else:
                for c in AUG:
                    if c not in ("tournament_wind_avg", "tour_vs_augusta_divergence"): d[c] = 0.0
            d["tournament_wind_avg"] = 10.0; d["tour_vs_augusta_divergence"] = 0.0
            d["actual_finish_num"] = pr.get("finish_num")
            ff.append(d)

        fdf = pd.DataFrame(ff).dropna(subset=ROLL)
        if len(fdf) < 5: continue

        # Divergence
        sg8 = fdf["sg_total_8w"].rank(pct=True)
        hr = fdf["augusta_competitive_rounds"] >= 6
        sa = fdf.loc[hr, "augusta_scoring_avg"]
        if len(sa) > 2:
            sap = (-sa).rank(pct=True)
            fdf.loc[hr, "tour_vs_augusta_divergence"] = sg8[hr] - sap
        fdf["tour_vs_augusta_divergence"] = fdf["tour_vs_augusta_divergence"].fillna(0)

        preds = s1.predict(fdf[ROLL].values)
        mc = run_mc(preds); fdf["mc_top10"] = mc["top10"]

        # Stage 2
        s2t = unified[unified["season"] < year].copy()
        s2t["made_top10"] = (s2t["finish_num"] <= 10).astype(int)
        s2t = s2t.merge(features, on=["player_name", "season"], how="left")
        s2r = []
        for _, row in s2t.iterrows():
            pn, ps = row["player_name"], row["season"]
            pt = tour_feat[(tour_feat["player_name"] == pn) & (tour_feat["season"] <= ps)]
            f2 = {"idx": row.name}
            if len(pt) > 0:
                lat = pt.iloc[-1]
                for c in ROLL: f2[c] = lat.get(c)
            else:
                for c in ROLL: f2[c] = np.nan
            f2["tournament_wind_avg"] = 10.0; f2["tour_vs_augusta_divergence"] = 0.0
            s2r.append(f2)
        s2rf = pd.DataFrame(s2r).set_index("idx")
        for c in S2F:
            if c not in s2t.columns and c in s2rf.columns: s2t[c] = s2rf[c]
            elif c not in s2t.columns: s2t[c] = np.nan
        s2t = s2t[s2t["finish_num"].notna()]
        np_ = s2t["made_top10"].sum()
        spw = (len(s2t) - np_) / np_ if np_ > 0 else 8.0
        s2m = xgb.XGBClassifier(**{**XGB_S2, "scale_pos_weight": spw})
        s2m.fit(s2t[S2F], s2t["made_top10"])
        for c in S2F:
            if c not in fdf.columns: fdf[c] = 0.0
        s2p = s2m.predict_proba(fdf[S2F])[:, 1]
        fdf["blend_top10"] = BLEND * s2p + (1 - BLEND) * fdf["mc_top10"]

        v = fdf[fdf["actual_finish_num"] < 999].copy()
        at10 = (v["actual_finish_num"] <= 10).astype(int)
        m = {"year": year}
        if at10.sum() > 0 and at10.sum() < len(at10):
            m["s2_auc"] = roc_auc_score(at10, s2p[v.index - fdf.index[0]] if False else
                                          fdf.loc[v.index, "blend_top10"].values)  # use blend for consistency
            m["blend_auc"] = roc_auc_score(at10, v["blend_top10"])
            m["mc_auc"] = roc_auc_score(at10, v["mc_top10"])
        t10p = v.nlargest(10, "blend_top10")
        m["top10_prec"] = (t10p["actual_finish_num"] <= 10).sum() / 10
        metrics.append(m)

    return metrics


def main():
    print("=" * 60)
    print("  FIELD STRENGTH WEIGHTED FEATURES")
    print("=" * 60)

    # ── TASK 1: Load field strength ──
    print("\n" + "=" * 60)
    print("TASK 1 — FIELD STRENGTH DATA")
    print("=" * 60)

    fs_lookup = FieldStrengthLookup(FS_PATH)
    fs_df = pd.read_csv(FS_PATH)
    print(f"  Loaded {len(fs_df)} field strength records")
    print(f"  Years: {sorted(fs_df['year'].unique())}")
    print(f"  Tours: {fs_df['tour'].value_counts().to_dict()}")

    # Show weight distribution
    weights = fs_df["field_strength_mean"].apply(lambda x: max(0.4, min(3.0, 1.0 + 0.3 * x)))
    print(f"\n  Weight distribution: mean={weights.mean():.2f} std={weights.std():.2f} "
          f"min={weights.min():.2f} max={weights.max():.2f}")

    # ── TASK 2: Build weighted rolling features ──
    print("\n" + "=" * 60)
    print("TASK 2 — BUILD FS-WEIGHTED ROLLING FEATURES")
    print("=" * 60)

    tour = pd.read_parquet(HIST_PATH)
    tour["finish_num"] = tour["finish_pos"].apply(pfn)
    cw = json.load(open(PROCESSED / "augusta_sg_weights.json")) if (PROCESSED / "augusta_sg_weights.json").exists() else HARDCODED_CW

    tour_clean = tour.dropna(subset=SG)
    tour_clean = tour_clean[tour_clean["finish_num"] < 999]
    tour_cw = apply_cw(tour_clean, cw)

    print("  Computing FS-weighted rolling features...")
    tour_fs = build_roll_fs(tour_cw, fs_lookup)
    tour_fs["finish_pct"] = ((tour_fs["finish_num"] - 1) / (tour_fs["field_size"] - 1)).clip(0, 1)

    # Show weight stats
    fw = tour_fs["_fs_weight"]
    real_fs = (fw != 1.0).sum()  # rows where FS != default
    print(f"  Rows with real FS weight: {real_fs}/{len(tour_fs)} ({real_fs/len(tour_fs):.0%})")
    print(f"  FS weight stats: mean={fw.mean():.3f} std={fw.std():.3f} min={fw.min():.2f} max={fw.max():.2f}")

    # Also compute unweighted for comparison
    print("  Computing plain rolling features (baseline)...")
    tour_plain = build_roll_plain(tour_cw)
    tour_plain["finish_pct"] = ((tour_plain["finish_num"] - 1) / (tour_plain["field_size"] - 1)).clip(0, 1)

    # ── Player comparison ──
    print(f"\n  KEY PLAYER sg_total_3w: plain vs FS-weighted")
    print(f"  {'Player':<25} {'Plain':>8} {'FS-Wtd':>8} {'Delta':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for pn in ["Scottie Scheffler", "Jon Rahm", "Rory McIlroy", "Bryson DeChambeau",
                "Cameron Young", "Viktor Hovland", "Sam Burns"]:
        pp = tour_plain[tour_plain["player_name"] == pn]
        pf = tour_fs[tour_fs["player_name"] == pn]
        if len(pp) > 0 and len(pf) > 0:
            plain_v = pp.iloc[-1].get("sg_total_3w", np.nan)
            fs_v = pf.iloc[-1].get("sg_total_3w", np.nan)
            delta = fs_v - plain_v if pd.notna(plain_v) and pd.notna(fs_v) else np.nan
            print(f"  {pn:<25} {plain_v:>8.3f} {fs_v:>8.3f} {delta:>+8.3f}")
        else:
            print(f"  {pn:<25} {'N/A':>8} {'N/A':>8}")

    # ── TASK 5: Backtest comparison ──
    print(f"\n{'='*60}")
    print("TASK 5 — BACKTEST: FS-WEIGHTED vs PLAIN")
    print(f"{'='*60}")

    print("\n  Running FS-weighted backtest...")
    metrics_fs = run_backtest(tour_fs, "FS")
    print("  Running plain backtest...")
    metrics_plain = run_backtest(tour_plain, "Plain")

    print(f"\n  {'Year':<6} | {'Plain AUC':>10} {'Plain Prec':>11} | {'FS AUC':>8} {'FS Prec':>9} | {'AUC Δ':>7}")
    print(f"  {'─'*6}-+-{'─'*10}-{'─'*11}-+-{'─'*8}-{'─'*9}-+-{'─'*7}")
    for mp, mf in zip(metrics_plain, metrics_fs):
        pa = mp.get("blend_auc", 0) or 0
        pp_ = mp.get("top10_prec", 0)
        fa = mf.get("blend_auc", 0) or 0
        fp = mf.get("top10_prec", 0)
        delta = fa - pa
        print(f"  {mp['year']:<6} | {pa:>10.3f} {pp_:>10.0%} | {fa:>8.3f} {fp:>8.0%} | {delta:>+7.3f}")

    avg_plain = np.mean([m.get("blend_auc", 0) or 0 for m in metrics_plain])
    avg_fs = np.mean([m.get("blend_auc", 0) or 0 for m in metrics_fs])
    avg_pp = np.mean([m.get("top10_prec", 0) for m in metrics_plain])
    avg_fp = np.mean([m.get("top10_prec", 0) for m in metrics_fs])
    print(f"  {'AVG':<6} | {avg_plain:>10.3f} {avg_pp:>10.0%} | {avg_fs:>8.3f} {avg_fp:>8.0%} | {avg_fs-avg_plain:>+7.3f}")

    # Decision
    keep_fs = avg_fs >= avg_plain - 0.015
    if keep_fs:
        print(f"\n  DECISION: KEEPING FS-weighted features (AUC delta = {avg_fs-avg_plain:+.3f})")
        final_tour = tour_fs
    else:
        print(f"\n  DECISION: REVERTING to plain (AUC dropped {avg_plain-avg_fs:.3f})")
        final_tour = tour_plain

    # ── Regenerate 2026 predictions ──
    print(f"\n{'='*60}")
    print("REGENERATING 2026 PREDICTIONS")
    print(f"{'='*60}")

    tc = final_tour.dropna(subset=ROLL + ["finish_pct"])
    s1 = xgb.XGBRegressor(**XGB_S1)
    s1.fit(tc[ROLL].values, tc["finish_pct"].values)
    print(f"  S1 trained on {len(tc):,} rows")

    old_preds = pd.read_parquet(PROCESSED / "predictions_2026.parquet")
    aug_cols = [c for c in old_preds.columns if c.startswith("augusta_") or c in
                ["tour_vs_augusta_divergence", "dg_id", "country", "dg_rank"]]

    roll_rows = []
    for _, pr in old_preds.iterrows():
        pn = pr["player_name"]
        pt = final_tour[final_tour["player_name"] == pn]
        d = {"player_name": pn}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLL: d[c] = lat.get(c)
        else:
            for c in ROLL: d[c] = 0
        roll_rows.append(d)

    fdf = pd.DataFrame(roll_rows).merge(old_preds[["player_name"] + aug_cols], on="player_name", how="left")
    for c in ROLL: fdf[c] = fdf[c].fillna(0)

    preds = s1.predict(fdf[ROLL].values)
    mc = run_mc(preds)
    fdf["win_prob"] = mc["win"]; fdf["top5_prob"] = mc["top5"]
    fdf["mc_top10"] = mc["top10"]; fdf["top20_prob"] = mc["top20"]
    fdf["model_score"] = preds

    # S2
    unified = pd.read_parquet(PROCESSED / "masters_unified.parquet")
    unified["finish_num"] = unified["finish_pos"].apply(pfn)
    features = pd.read_parquet(PROCESSED / "augusta_player_features.parquet")
    s2t = unified.copy(); s2t["made_top10"] = (s2t["finish_num"] <= 10).astype(int)
    s2t = s2t.merge(features, on=["player_name", "season"], how="left")
    s2r = []
    for _, row in s2t.iterrows():
        pn, ps = row["player_name"], row["season"]
        pt = final_tour[(final_tour["player_name"] == pn) & (final_tour["season"] <= ps)]
        f2 = {"idx": row.name}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLL: f2[c] = lat.get(c)
        else:
            for c in ROLL: f2[c] = np.nan
        f2["tournament_wind_avg"] = 10.0; f2["tour_vs_augusta_divergence"] = 0.0
        s2r.append(f2)
    s2rf = pd.DataFrame(s2r).set_index("idx")
    for c in S2F:
        if c not in s2t.columns and c in s2rf.columns: s2t[c] = s2rf[c]
        elif c not in s2t.columns: s2t[c] = np.nan
    s2t = s2t[s2t["finish_num"].notna()]
    np_ = s2t["made_top10"].sum()
    spw = (len(s2t) - np_) / np_ if np_ > 0 else 8
    s2_final = xgb.XGBClassifier(**{**XGB_S2, "scale_pos_weight": spw})
    s2_final.fit(s2t[S2F], s2t["made_top10"])

    for c in S2F:
        if c not in fdf.columns: fdf[c] = 0.0
    sg8 = fdf["sg_total_8w"].rank(pct=True)
    hr = fdf["augusta_competitive_rounds"] >= 6
    sa = fdf.loc[hr, "augusta_scoring_avg"]
    if len(sa) > 2:
        sap = (-sa).rank(pct=True)
        fdf.loc[hr, "tour_vs_augusta_divergence"] = sg8[hr] - sap
    fdf["tour_vs_augusta_divergence"] = fdf["tour_vs_augusta_divergence"].fillna(0)

    s2raw = s2_final.predict_proba(fdf[S2F])[:, 1]
    logits = np.log(np.clip(s2raw, 1e-6, 1 - 1e-6) / (1 - np.clip(s2raw, 1e-6, 1 - 1e-6)))
    s2cal = 1 / (1 + np.exp(-logits / TEMP))
    fdf["stage2_prob_raw"] = s2raw; fdf["stage2_prob_calibrated"] = s2cal
    fdf["top10_prob_calibrated"] = BLEND * s2cal + (1 - BLEND) * fdf["mc_top10"]
    fdf["make_cut_prob"] = fdf[["top20_prob", "mc_top10"]].max(axis=1).clip(upper=0.95)
    fdf["top20_prob"] = fdf[["top20_prob", "top10_prob_calibrated"]].max(axis=1)
    fdf["top5_prob"] = np.minimum(fdf["top5_prob"].clip(lower=fdf["win_prob"]).values, fdf["top10_prob_calibrated"].values)

    # DK odds
    dk_path = PROCESSED / "dk_odds_2026.csv"
    if dk_path.exists():
        dk = pd.read_csv(dk_path)
        for _, orow in dk.iterrows():
            best = process.extractOne(orow["player_name"], fdf["player_name"].tolist(), scorer=fuzz.ratio)
            if best and best[1] >= 85:
                idx = fdf[fdf["player_name"] == best[0]].index
                if len(idx) > 0:
                    fdf.loc[idx[0], "dk_fair_prob_win"] = orow["dk_fair_prob"]
                    fdf.loc[idx[0], "dk_fair_prob_top10"] = orow["dk_fair_top10"]
                    fdf.loc[idx[0], "dk_american_odds"] = orow["dk_american_odds"]
        fdf["kelly_edge_win"] = (fdf["win_prob"] - fdf.get("dk_fair_prob_win", 0)) / fdf.get("dk_fair_prob_win", 1).clip(lower=1e-6)
        fdf["kelly_edge_top10"] = (fdf["top10_prob_calibrated"] - fdf.get("dk_fair_prob_top10", 0)) / fdf.get("dk_fair_prob_top10", 1).clip(lower=1e-6)

    fdf = fdf.sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)

    # Print top 20
    print(f"\n  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6} {'DK':>8}")
    print(f"  {'─'*3} {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    for i, (_, r) in enumerate(fdf.head(20).iterrows()):
        dk = f"+{r['dk_american_odds']:.0f}" if pd.notna(r.get('dk_american_odds')) else "N/A"
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>5.1%} "
              f"{r['top20_prob']:>5.1%} {dk:>8}")

    # Save
    out_cols = [c for c in [
        "dg_id","player_name","country","dg_rank","win_prob","top5_prob","top10_prob_calibrated",
        "top20_prob","make_cut_prob","model_score","stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier","augusta_made_cut_prev_year",
        "augusta_scoring_trajectory","tour_vs_augusta_divergence",
        "dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds","kelly_edge_win","kelly_edge_top10",
    ] if c in fdf.columns]
    fdf[out_cols].to_parquet(PROCESSED / "predictions_2026.parquet", index=False)
    fdf[out_cols].to_csv(PROCESSED / "predictions_2026.csv", index=False)
    print(f"\n  Saved predictions_2026.parquet/csv")

    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

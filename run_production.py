#!/usr/bin/env python3
"""
Production pipeline for Augusta National Model.

Single canonical script replacing all run_*.py variants.
Reads processed data, trains S1+S2, runs calibrated predictions, backtests.

Architecture:
  Stage 1: XGBoost regression on finish_pct (16 rolling SG features, tour-wide)
  Stage 2: XGBoost classifier for top-10 (34 features = rolling + Augusta, Augusta-only training)
  Calibration: Platt scaling → debutant adjustment → unified MC
  All markets (win/top5/top10/top20) flow from one consistent S2-based ranking.

Usage:
  python3 run_production.py              # full retrain + backtest + 2026 predictions
  python3 run_production.py --backtest   # backtest only (no 2026 predictions)
"""
import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

sys.path.insert(0, ".")
from augusta_model.calibration import (
    fit_platt_calibrator,
    calibrate_s2_platt,
    apply_debutant_adjustment,
    run_unified_mc,
    calibrate_full_pipeline,
)
from augusta_model.features.new_features import (
    add_approach_resilience,
    add_arg_resilience,
    add_putting_surface_features,
    add_scoring_profile,
    add_weather_features,
    add_form_momentum,
    add_sg_interactions,
    add_difficulty_scaling,
    add_winner_profile_features,
)

# Aging decay — will be available once Session C completes Task 2
try:
    from augusta_model.features.new_features import add_aging_decay
    HAS_AGING_DECAY = True
except ImportError:
    HAS_AGING_DECAY = False

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

PROCESSED = Path("data/processed")
API_KEY = os.getenv("DATAGOLF_API_KEY")
DG_BASE = "https://feeds.datagolf.com"

HARDCODED_CW = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

XGB_S1 = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 4,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "reg:squarederror", "random_state": 42,
}
XGB_S2 = {
    "n_estimators": 300, "learning_rate": 0.03, "max_depth": 3,
    "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 8,
    "reg_alpha": 2.0, "reg_lambda": 5.0,  # stronger regularisation — suppress Augusta history overfitting
    "objective": "binary:logistic", "eval_metric": "auc", "random_state": 42,
}

# S1/S2 blend weight for BACKTEST (S1 as current-form check, 15% weight).
# Lower weight because S1 has date-ordering artifacts for some players — use it
# as a gentle correction, not a dominant signal.
BLEND_S2_WEIGHT_BACKTEST = 0.85  # 85% S2, 15% S1 in backtest

# For 2026 PREDICTIONS we use dg_rank as the current-form signal instead of S1.
# dg_rank is fresh, derived from current-season performance, and immune to the
# date-ordering bug that corrupts rolling SG computation for some players.
# exp(-0.03 * (rank-1)) → rank1=1.0, rank44=0.27, rank100=0.05, rank200=0.002
BLEND_S2_WEIGHT_PRED = 0.70   # 70% S2, 30% rank-form signal
RANK_DECAY = 0.03              # exponential decay rate per rank position

# Stale player cap: hard ceiling on the BLENDED skill score before MC.
# Applied AFTER blend so the rank-form signal cannot lift stale players above the cap.
# Threshold raised to 200 so legitimate LIV/fringe players (rank 100-200) compete normally.
# Fred Couples/Vijay Singh type invitees are unranked (NaN) or rank 300+ → always stale.
STALE_RANK_THRESHOLD = 200
STALE_CAP = 0.04  # maps to ~1-2% T10 in 83-player field — right order of magnitude for past-champion honorary invitees

# MC parameters tuned for realistic golf variance
# Session 5 fix: noise 0.16→0.10, tpred 0.10→0.14 → Scheffler 5.5%→10.8%
# signal/noise ratio = 0.14/0.10 = 1.4 (good differentiation vs 0.06/0.16=0.375 which flattens field)
N_SIMS = 50000
MC_NOISE = 0.10      # per-round noise (lower = more skill-driven)
MC_TPRED = 0.14      # skill spread (higher = more differentiation between players)
SEED = 42

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_t2g", "sg_total"]
ROLL = [
    "sg_ott_3w", "sg_app_3w", "sg_arg_3w", "sg_putt_3w", "sg_t2g_3w", "sg_total_3w",
    "sg_ott_8w", "sg_app_8w", "sg_arg_8w", "sg_putt_8w", "sg_t2g_8w", "sg_total_8w",
    "sg_total_std_8", "sg_total_momentum", "sg_ball_strike_ratio", "log_field_size",
]
AUG = [
    "augusta_competitive_rounds", "augusta_made_cut_prev_year", "augusta_experience_tier",
    "augusta_scoring_trajectory", "augusta_rounds_last_2yrs", "augusta_best_finish_recent",
    "augusta_starts", "augusta_made_cut_rate", "augusta_top10_rate", "augusta_best_finish",
    "augusta_scoring_avg", "augusta_sg_app_career", "augusta_sg_total_career",
    "augusta_bogey_avoidance", "augusta_round_variance_score", "augusta_back9_scoring",
    "tournament_wind_avg", "tour_vs_augusta_divergence",
]
# New features from augusta_model/features/new_features.py
# Designed around Augusta's statistical profile (PGA Splits 2021-2025)
NEW = [
    "hard_course_sg_app",                                # approach resilience at tough courses
    "hard_course_sg_arg",                                # ARG at tough courses (short grass)
    "fast_green_sg_putt",                                # putting on fast greens (3-putt avoidance)
    "par5_scoring_proxy", "bogey_resistance", "driving_dominance",  # scoring profile
    "wind_experience_interaction", "wet_course",         # weather
    "events_since_top10", "recent_win", "recent_consistency",  # form momentum
    "augusta_fit_score", "power_precision", "short_game_package",  # SG interactions
    "difficulty_scaling",                                # plays up at hard courses?
    "dg_rank",                                           # world ranking (24/26 winners top 30)
    "career_wins",                                       # winning pedigree (15/17 winners had 4+)
    "recent_major_top6",                                 # top-6 in major within 2 yrs (14/16)
    "last_start_finish_pct",                             # 35th+ in previous start (16/16)
    "top8_in_last7",                                     # top-8 in last 7 events (15/16)
    "t2g_last4_total",                                   # 18+ SG T2G in last 4 (14/14)
]
S2F = ROLL + AUG + NEW


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def parse_finish(p):
    if p is None or pd.isna(p):
        return None
    p = str(p).strip().upper()
    if p in ("CUT", "MC", "WD", "DQ", "MDF"):
        return 999
    p = p.replace("T", "").replace("=", "")
    try:
        return int(p)
    except ValueError:
        return None


def fetch_dg_market_odds():
    """Fetch real sportsbook odds from DataGolf for win/top5/top10/top20.

    Strategy:
      1. Try betting-tools/outrights (live pre-event odds from 11 books).
      2. Fall back to preds/pre-tournament (market-calibrated snapshot) if
         outrights return 0 results (event in progress, markets closed).

    Returns DataFrame: player_name, market_win, market_top5, market_top10, market_top20
    """
    # ── Attempt 1: live outright markets ──────────────────────────────────────
    market_data = {}
    markets_found = 0
    for mkt_key, col in [("win", "market_win"), ("top_5", "market_top5"),
                         ("top_10", "market_top10"), ("top_20", "market_top20")]:
        try:
            r = requests.get(
                f"{DG_BASE}/betting-tools/outrights",
                params={"tour": "pga", "market": mkt_key, "odds_format": "percent",
                        "file_format": "json", "key": API_KEY},
                timeout=15,
            )
            if r.status_code == 200:
                players = r.json().get("results", [])
                if players:
                    markets_found += 1
                    for p in players:
                        name = normalize_name(p.get("player_name", ""))
                        market_data.setdefault(name, {})[col] = p.get("baseline")
        except Exception:
            pass

    if markets_found > 0:
        rows = [{"player_name": k, **v} for k, v in market_data.items()]
        df = pd.DataFrame(rows)
        print(f"  DG betting-tools: {len(df)} players ({markets_found}/4 markets live)")
        return df

    # ── Attempt 2: pre-tournament snapshot (market-calibrated) ───────────────
    try:
        r = requests.get(
            f"{DG_BASE}/preds/pre-tournament",
            params={"tour": "pga", "add_position": "1,5,10,20",
                    "file_format": "json", "key": API_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            players = data.get("baseline", [])
            if players:
                rows = [{
                    "player_name": normalize_name(p["player_name"]),
                    "market_win":   p.get("win"),
                    "market_top5":  p.get("top_5"),
                    "market_top10": p.get("top_10"),
                    "market_top20": p.get("top_20"),
                } for p in players]
                df = pd.DataFrame(rows)
                updated = data.get("last_updated", "unknown")
                print(f"  DG pre-tournament snapshot: {len(df)} players (as of {updated})")
                return df
    except Exception as e:
        print(f"  DG market odds fetch failed: {e}")

    return None


def normalize_name(n):
    n = str(n).strip()
    if "," in n:
        parts = n.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return n


def apply_course_weights(df, weights):
    df = df.copy()
    for col, wt in weights.items():
        if col in df.columns:
            df[col] = df[col] * wt
    return df


# ═══════════════════════════════════════════════════════════
# FIELD STRENGTH LOOKUP (from run_field_strength.py)
# ═══════════════════════════════════════════════════════════

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
    if not event_name or not isinstance(event_name, str):
        return 1.0
    lower = event_name.lower()
    best = 1.0
    for pattern, weight in _TIER_RULES:
        if pattern in lower:
            best = max(best, weight) if weight > 1.0 else min(best, weight)
    return best


class FieldStrengthLookup:
    def __init__(self, csv_path):
        if not Path(csv_path).exists():
            self._exact = {}
            self._by_year = {}
            return
        self.df = pd.read_csv(csv_path)
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
        if not event_name or pd.isna(event_name):
            return 1.0
        event_lower = str(event_name).strip().lower()
        year = int(year) if not pd.isna(year) else 0
        key = (year, event_lower)
        if key in self._exact:
            fs = self._exact[key]
            return max(0.4, min(3.0, 1.0 + 0.3 * fs))
        if year in self._by_year:
            names = [n for n, _ in self._by_year[year]]
            best = process.extractOne(event_lower, [n.lower() for n in names], scorer=fuzz.ratio)
            if best and best[1] >= 80:
                idx = [n.lower() for n in names].index(best[0])
                fs = self._by_year[year][idx][1]
                return max(0.4, min(3.0, 1.0 + 0.3 * fs))
        return _tier_fallback(event_name)


# ═══════════════════════════════════════════════════════════
# ROLLING FEATURE BUILDER
# ═══════════════════════════════════════════════════════════

def build_rolling_features(df, fs_lookup=None):
    """Build rolling SG features, optionally weighted by field strength."""
    df = df.sort_values(["player_name", "season", "date"]).copy()

    if fs_lookup is not None:
        df["_fs_weight"] = df.apply(
            lambda r: fs_lookup.get_weight(r.get("event_name"), r.get("season")), axis=1)
    else:
        df["_fs_weight"] = 1.0

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
                        out[i] = out[i - 1] if i > 0 else np.nan
                        continue
                    lookback = min(i + 1, span * 2)
                    wsum, vsum = 0.0, 0.0
                    for j in range(max(0, i - lookback + 1), i + 1):
                        if np.isnan(vals[j]):
                            continue
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
        df["sg_ott_8w"].abs() + df["sg_app_8w"].abs() +
        df["sg_arg_8w"].abs() + df["sg_putt_8w"].abs() + 1e-6)
    df["log_field_size"] = np.log(df["field_size"].clip(1))

    if "_fs_weight" in df.columns:
        df.drop(columns=["_fs_weight"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════
# AUGUSTA FEATURE BUILDER
# ═══════════════════════════════════════════════════════════

def build_augusta_features_for_player(pn, year, unified, rounds_df):
    """Build Augusta-specific features for one player-year. Strict temporal cutoff.

    Recency decay (0.80 per year) is applied to competitive rounds, cut rate,
    and top-10 rate so that 2019 Augusta history counts much less than 2024.
    Experience tier still uses raw counts (experience is binary, not decayed).
    """
    prior = unified[(unified["player_name"] == pn) & (unified["season"] < year)]
    f = {}
    n = len(prior)

    if n == 0:
        f = {c: 0 for c in AUG}
        f["augusta_best_finish"] = 91
        f["augusta_best_finish_recent"] = 91
        f["player_name"] = pn
        return f

    RECENCY_DECAY = 0.80  # 20% depreciation per year of Augusta history

    def _comp_rounds(row):
        cnt = sum(1 for c in ["r1_score", "r2_score", "r3_score", "r4_score"]
                  if pd.notna(row.get(c)))
        return cnt if cnt > 0 else (4 if row.get("made_cut") == 1 else 2)

    # Raw count drives experience tier (experience doesn't decay — you played or you didn't)
    raw_comp_rounds = sum(_comp_rounds(r) for _, r in prior.iterrows())
    # Decayed count is the actual feature value (recent play counts more)
    comp_rounds = sum(
        _comp_rounds(r) * (RECENCY_DECAY ** max(0, year - int(r["season"])))
        for _, r in prior.iterrows()
    )
    prev_yr = prior[prior["season"] == year - 1]
    cut_prev = int(len(prev_yr) > 0 and prev_yr.iloc[0].get("made_cut") == 1)
    tier = (0 if raw_comp_rounds == 0 else
            (1 if raw_comp_rounds <= 7 else
             (2 if raw_comp_rounds <= 19 else
              (3 if raw_comp_rounds <= 35 else 4))))

    # Scoring trajectory
    prior_cut = prior[prior["made_cut"] == 1]
    traj = 0
    if len(prior_cut) >= 2:
        rec2 = prior_cut.nlargest(2, "season")
        rec_avg = rec2["score_vs_field"].dropna().mean()
        all_svf = prior_cut["score_vs_field"].dropna()
        if len(all_svf) > 0:
            decay = 0.75 ** (year - prior_cut.loc[all_svf.index, "season"])
            car_avg = np.average(all_svf, weights=decay)
            traj = car_avg - rec_avg if pd.notna(rec_avg) else 0

    rec_prior = prior[prior["season"].isin([year - 1, year - 2])]
    rds_2yr = sum(_comp_rounds(r) for _, r in rec_prior.iterrows())
    rec3 = prior.nlargest(3, "season")
    vf_good = rec3["finish_num"].dropna()
    vf_good = vf_good[vf_good < 999]
    best_rec = vf_good.min() if len(vf_good) > 0 else 91

    # Recency-weighted cut rate and top-10 rate
    cuts = prior["made_cut"].dropna()
    if len(cuts) > 0:
        cut_decay = np.array([RECENCY_DECAY ** max(0, year - int(prior.loc[i, "season"]))
                              for i in cuts.index])
        cut_rate = float(np.average(cuts.values, weights=cut_decay))
    else:
        cut_rate = 0
    fn = prior["finish_num"].dropna()
    if len(fn) > 0:
        t10 = (fn <= 10).astype(float)
        t10_decay = np.array([RECENCY_DECAY ** max(0, year - int(prior.loc[i, "season"]))
                              for i in fn.index])
        t10_rate = float(np.average(t10.values, weights=t10_decay))
    else:
        t10_rate = 0
    best_f = fn[fn < 999].min() if (fn < 999).any() else 91

    svf = prior["score_vs_field"].dropna()
    if len(svf) > 0:
        d2 = 0.75 ** (year - prior.loc[svf.index, "season"])
        sc_avg = np.average(svf, weights=d2)
    else:
        sc_avg = 0

    psg = prior[prior["has_sg_data"] == True]
    sg_app_c, sg_tot_c = None, None
    if len(psg) > 0:
        sd = 0.75 ** (year - psg["season"])
        sa = psg["sg_app"].dropna()
        sg_app_c = np.average(sa, weights=sd[sa.index]) if len(sa) > 0 else None
        st = psg["sg_total"].dropna()
        sg_tot_c = np.average(st, weights=sd[st.index]) if len(st) > 0 else None

    # Round-level features
    pr_rds = rounds_df[(rounds_df["player_name"] == pn) & (rounds_df["season"] < year)]
    ba, rv, b9 = 0, None, None
    if len(pr_rds) > 0:
        scores = pr_rds["score"].dropna()
        ba = (scores <= 0).sum() / len(scores) if len(scores) > 0 else 0
        rv = scores.std() if len(scores) > 1 else None
        late = pr_rds[pr_rds["round_num"].isin([3, 4])]
        b9 = late["score"].dropna().mean() if len(late) > 0 else None

    f = {
        "player_name": pn,
        "augusta_competitive_rounds": comp_rounds,
        "augusta_made_cut_prev_year": cut_prev,
        "augusta_experience_tier": tier,
        "augusta_scoring_trajectory": traj,
        "augusta_rounds_last_2yrs": rds_2yr,
        "augusta_best_finish_recent": best_rec,
        "augusta_starts": n,
        "augusta_made_cut_rate": cut_rate,
        "augusta_top10_rate": t10_rate,
        "augusta_best_finish": best_f,
        "augusta_scoring_avg": sc_avg,
        "augusta_sg_app_career": sg_app_c,
        "augusta_sg_total_career": sg_tot_c,
        "augusta_bogey_avoidance": ba,
        "augusta_round_variance_score": rv,
        "augusta_back9_scoring": b9,
        "tournament_wind_avg": 10.0,
        "tour_vs_augusta_divergence": 0.0,
    }
    return f


# ═══════════════════════════════════════════════════════════
# STAGE 2 TRAINING
# ═══════════════════════════════════════════════════════════

def train_s2(unified, features, tour_feat, max_year):
    """Train Stage 2 classifier on Augusta data up to max_year."""
    s2t = unified[unified["season"] < max_year].copy()
    s2t["made_top10"] = (s2t["finish_num"] <= 10).astype(int)
    s2t = s2t.merge(features, on=["player_name", "season"], how="left")

    # Add rolling features from tour data
    s2r = []
    for _, row in s2t.iterrows():
        pn, ps = row["player_name"], row["season"]
        pt = tour_feat[(tour_feat["player_name"] == pn) & (tour_feat["season"] < ps)]
        f2 = {"idx": row.name}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLL:
                f2[c] = lat.get(c)
        else:
            for c in ROLL:
                f2[c] = np.nan
        f2["tournament_wind_avg"] = 10.0
        f2["tour_vs_augusta_divergence"] = 0.0
        s2r.append(f2)

    s2rf = pd.DataFrame(s2r).set_index("idx")
    for c in S2F:
        if c not in s2t.columns and c in s2rf.columns:
            s2t[c] = s2rf[c]
        elif c not in s2t.columns:
            s2t[c] = np.nan

    s2t = s2t[s2t["finish_num"].notna()]
    n_pos = s2t["made_top10"].sum()
    spw = (len(s2t) - n_pos) / n_pos if n_pos > 0 else 8.0

    # Weight recent years higher: current form matters more than 2015 Augusta history
    weights = np.where(s2t["season"] >= 2023, 2.0,
              np.where(s2t["season"] >= 2021, 1.5, 1.0))

    s2m = xgb.XGBClassifier(**{**XGB_S2, "scale_pos_weight": spw})
    s2m.fit(s2t[S2F], s2t["made_top10"], sample_weight=weights)
    return s2m


# ═══════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════

def run_backtest(tour_feat, unified, features, rounds_df):
    """Walk-forward backtest with Platt calibration."""
    print(f"\n{'=' * 60}")
    print("WALK-FORWARD BACKTEST")
    print(f"{'=' * 60}")

    sg_years = sorted(unified[unified["has_sg_data"] == True]["season"].unique())
    bt_years = [y for y in sg_years if y > min(unified["season"].unique())]
    metrics = []
    bt_results = []

    for year in bt_years:
        # Train S1 on tour data before this year
        train = tour_feat[tour_feat["season"] < year].dropna(subset=ROLL + ["finish_pct"])
        s1 = xgb.XGBRegressor(**XGB_S1)
        s1.fit(train[ROLL].values, train["finish_pct"].values)

        # Build field for this year
        field = unified[(unified["season"] == year) & (unified["has_sg_data"] == True)].copy()
        if len(field) == 0:
            continue

        ff = []
        for _, pr in field.iterrows():
            pn = pr["player_name"]
            pt = tour_feat[(tour_feat["player_name"] == pn) & (tour_feat["season"] < year)]
            if len(pt) == 0:
                continue
            lat = pt.iloc[-1]
            d = {"player_name": pn}
            for c in ROLL:
                d[c] = lat.get(c)

            # Augusta features
            aug = build_augusta_features_for_player(pn, year, unified, rounds_df)
            for c in AUG:
                if c not in ("tournament_wind_avg", "tour_vs_augusta_divergence"):
                    d[c] = aug.get(c, 0)
            d["tournament_wind_avg"] = 10.0
            d["tour_vs_augusta_divergence"] = 0.0
            d["actual_finish_num"] = pr.get("finish_num")
            d["actual_finish_pos"] = pr.get("finish_pos")
            ff.append(d)

        bdf = pd.DataFrame(ff).dropna(subset=ROLL)
        if len(bdf) < 5:
            continue

        # Compute divergence
        sg8 = bdf["sg_total_8w"].rank(pct=True)
        hr = bdf["augusta_competitive_rounds"] >= 6
        sa = bdf.loc[hr, "augusta_scoring_avg"]
        if len(sa) > 2:
            sap = (-sa).rank(pct=True)
            bdf.loc[hr, "tour_vs_augusta_divergence"] = sg8[hr] - sap
        bdf["tour_vs_augusta_divergence"] = bdf["tour_vs_augusta_divergence"].fillna(0)

        # Add new features (Augusta-profile-aware)
        prior_tour = tour_feat[tour_feat["season"] < year]
        bdf = add_scoring_profile(bdf)
        bdf = add_sg_interactions(bdf)
        bdf["season"] = year
        bdf = add_weather_features(bdf)
        bdf = add_approach_resilience(prior_tour, bdf)
        bdf = add_arg_resilience(prior_tour, bdf)
        bdf = add_putting_surface_features(prior_tour, bdf)
        bdf = add_form_momentum(prior_tour, bdf, target_year=year)
        bdf = add_difficulty_scaling(prior_tour, bdf)
        bdf = add_winner_profile_features(prior_tour, bdf, target_year=year)
        if HAS_AGING_DECAY:
            bdf = add_aging_decay(bdf, tour_df=prior_tour, current_year=year)

        # Train S2 on Augusta data before this year
        s2m = train_s2(unified, features, tour_feat, year)

        for c in S2F:
            if c not in bdf.columns:
                bdf[c] = 0.0
        s2_raw = s2m.predict_proba(bdf[S2F])[:, 1]
        bdf["s2_top10"] = s2_raw

        # Stage 1 predictions for S1/S2 blend (current-form anchor)
        s1_pred = s1.predict(bdf[ROLL].fillna(0).values)
        bdf["model_score"] = s1_pred

        # Calibrated metrics using unified MC with light S1 blend (15% weight)
        # Lower weight than prediction run because S1 has date-ordering artifacts
        # for some players — using it as a gentle correction, not dominant signal.
        tiers = bdf["augusta_experience_tier"].fillna(0).values
        cal = calibrate_full_pipeline(
            s2_raw, tiers,
            n_sims=N_SIMS, noise_std=MC_NOISE,
            target_pred_std=MC_TPRED, seed=SEED,
            s1_scores=s1_pred,
            blend_weight=BLEND_S2_WEIGHT_BACKTEST,  # 85% S2, 15% S1
            # No stale_mask in backtest — dg_rank not available for historical years
        )
        bdf["cal_top10"] = cal["top10_prob"].values
        bdf["cal_win"] = cal["win_prob"].values
        bdf["season"] = year

        # Metrics
        v = bdf[bdf["actual_finish_num"] < 999].copy()
        at10 = (v["actual_finish_num"] <= 10).astype(int)
        m = {"year": year, "field_size": len(bdf)}

        if at10.sum() > 0 and at10.sum() < len(at10):
            m["s2_auc"] = roc_auc_score(at10, v["s2_top10"])
            m["cal_auc"] = roc_auc_score(at10, v["cal_top10"])
            m["brier"] = brier_score_loss(at10, np.clip(v["cal_top10"], 0, 1))

        t10p = v.nlargest(10, "cal_top10")
        m["top10_prec"] = (t10p["actual_finish_num"] <= 10).sum() / 10
        sp, _ = sp_stats.spearmanr(v["cal_top10"], -v["actual_finish_num"])
        m["spearman"] = sp

        # Winner check
        winner = v[v["actual_finish_num"] == 1]
        if len(winner) > 0:
            wn = winner.iloc[0]["player_name"]
            wi = winner.index[0]
            w_win = bdf.loc[wi, "cal_win"]
            w_rank = (-bdf["cal_top10"]).rank().loc[wi]
            m["winner"] = wn
            m["winner_rank"] = int(w_rank)
            m["winner_win_pct"] = w_win

        metrics.append(m)
        bt_results.append(bdf)
        winner_str = ""
        if "winner" in m:
            wname = m["winner"]
            wrank = m["winner_rank"]
            wpct = m["winner_win_pct"]
            winner_str = f" | Winner: {wname} (#{wrank}, {wpct:.1%})"
        print(f"  {year}: field={len(bdf):>2} | S2 AUC={m.get('s2_auc', 0):.3f} "
              f"| Cal AUC={m.get('cal_auc', 0):.3f} | T10 Prec={m['top10_prec']:.0%} "
              f"| Brier={m.get('brier', 0):.4f} | Spearman={sp:.3f}{winner_str}")

    # Summary
    print(f"\n  {'Year':<6} {'S2 AUC':>8} {'Cal AUC':>9} {'Brier':>7} {'T10 Prec':>9} {'Spearman':>9}")
    for m in metrics:
        print(f"  {m['year']:<6} {m.get('s2_auc', 0):>8.3f} {m.get('cal_auc', 0):>9.3f} "
              f"{m.get('brier', 0):>7.4f} {m.get('top10_prec', 0):>8.0%} {m.get('spearman', 0):>9.3f}")

    def avg(k):
        vs = [m[k] for m in metrics if m.get(k)]
        return np.mean(vs) if vs else 0

    print(f"  {'AVG':<6} {avg('s2_auc'):>8.3f} {avg('cal_auc'):>9.3f} "
          f"{avg('brier'):>7.4f} {avg('top10_prec'):>8.0%} {avg('spearman'):>9.3f}")

    if bt_results:
        bt_all = pd.concat(bt_results, ignore_index=True)
        bt_all.to_parquet(PROCESSED / "backtest_results.parquet", index=False)
        print(f"\n  Saved backtest: {PROCESSED / 'backtest_results.parquet'}")

    return metrics, bt_results


# ═══════════════════════════════════════════════════════════
# 2026 PREDICTIONS
# ═══════════════════════════════════════════════════════════

def predict_2026(tour_feat, unified, features, rounds_df, s1_model):
    """Generate calibrated 2026 Masters predictions."""
    print(f"\n{'=' * 60}")
    print("2026 PREDICTIONS")
    print(f"{'=' * 60}")

    # Fetch 2026 field from DG API
    field_data = requests.get(
        f"{DG_BASE}/field-updates",
        params={"tour": "pga", "file_format": "json", "key": API_KEY},
        timeout=15
    ).json()
    field_players = [
        {"dg_id": p.get("dg_id"), "player_name": normalize_name(p.get("player_name", "")),
         "country": p.get("country", ""), "dg_rank": p.get("dg_rank")}
        for p in field_data.get("field", [])
    ]
    print(f"  Field: {len(field_players)} players")

    # Build rolling features for field
    roll_rows = []
    for fp in field_players:
        pn = fp["player_name"]
        pt = tour_feat[tour_feat["player_name"] == pn]
        d = {"player_name": pn, "dg_id": fp["dg_id"],
             "country": fp["country"], "dg_rank": fp["dg_rank"]}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLL:
                d[c] = lat.get(c)
        else:
            for c in ROLL:
                d[c] = 0
        roll_rows.append(d)

    fdf = pd.DataFrame(roll_rows)
    for c in ROLL:
        fdf[c] = fdf[c].fillna(0)
    fdf = fdf[fdf[ROLL[:6]].abs().sum(axis=1) > 0]  # drop players with all-zero SG
    print(f"  Field with SG data: {len(fdf)}")

    # Build Augusta features
    aug_rows = []
    for _, fp in fdf.iterrows():
        aug = build_augusta_features_for_player(fp["player_name"], 2026, unified, rounds_df)
        aug_rows.append(aug)
    aug_df = pd.DataFrame(aug_rows)
    fdf = fdf.merge(aug_df, on="player_name", how="left")
    for c in AUG:
        if c not in fdf.columns:
            fdf[c] = 0.0

    # Compute divergence
    sg8 = fdf["sg_total_8w"].rank(pct=True)
    hr = fdf["augusta_competitive_rounds"] >= 6
    sa = fdf.loc[hr, "augusta_scoring_avg"]
    if len(sa) > 2:
        sap = (-sa).rank(pct=True)
        fdf.loc[hr, "tour_vs_augusta_divergence"] = sg8[hr] - sap
    fdf["tour_vs_augusta_divergence"] = fdf["tour_vs_augusta_divergence"].fillna(0)

    # Add new features (Augusta-profile-aware)
    fdf = add_scoring_profile(fdf)
    fdf = add_sg_interactions(fdf)
    fdf["season"] = 2026
    fdf = add_weather_features(fdf)
    fdf = add_approach_resilience(tour_feat, fdf)
    fdf = add_arg_resilience(tour_feat, fdf)
    fdf = add_putting_surface_features(tour_feat, fdf)
    fdf = add_form_momentum(tour_feat, fdf, target_year=2026)
    fdf = add_difficulty_scaling(tour_feat, fdf)
    fdf = add_winner_profile_features(tour_feat, fdf, target_year=2026)
    if HAS_AGING_DECAY:
        fdf = add_aging_decay(fdf, tour_df=tour_feat, current_year=2026)

    # Train S2 on all Augusta data
    s2m = train_s2(unified, features, tour_feat, 2026)
    for c in S2F:
        if c not in fdf.columns:
            fdf[c] = 0.0
    s2_raw = s2m.predict_proba(fdf[S2F])[:, 1]
    fdf["stage2_prob_raw"] = s2_raw

    # Stage 1 predictions (stored for reference, NOT used as blend signal — S1 is
    # unreliable for some players due to date-ordering artifacts in rolling features).
    s1_pred = s1_model.predict(fdf[ROLL].fillna(0).values)
    fdf["model_score"] = s1_pred

    # ── Current-form signal: dg_rank (fresh, date-ordering immune) ──
    # S1 is broken for Scheffler (model_score=0.45, 45th pct despite being world #1)
    # due to rolling feature date-ordering. Use dg_rank instead.
    # exp(-RANK_DECAY * (rank-1)) → rank1=1.0, rank10=0.76, rank44=0.27, rank100=0.05
    dg_rank = pd.to_numeric(fdf.get("dg_rank", pd.Series(dtype=float)), errors="coerce")
    rank_form = np.exp(-RANK_DECAY * (dg_rank.fillna(300).values - 1))
    # Pass as (1 - rank_form) so that calibrate_full_pipeline's inversion restores rank_form
    rank_form_input = 1.0 - rank_form

    # Stale player mask: dg_rank > STALE_RANK_THRESHOLD or unranked
    stale_mask = (dg_rank > STALE_RANK_THRESHOLD) | dg_rank.isna()
    n_stale = stale_mask.sum()
    print(f"  Stale players (dg_rank > {STALE_RANK_THRESHOLD} or unranked): {n_stale}")

    # ── Platt calibration from backtest data ──
    bt_path = PROCESSED / "backtest_results.parquet"
    platt_params = None
    if bt_path.exists():
        bt = pd.read_parquet(bt_path)
        bt["actual_top10"] = (bt["actual_finish_num"] <= 10).astype(int)
        valid = bt.dropna(subset=["s2_top10", "actual_top10"])
        valid = valid[valid["s2_top10"] > 0.001]
        if len(valid) >= 30 and valid["actual_top10"].sum() >= 3:
            platt_params = fit_platt_calibrator(
                valid["s2_top10"].values, valid["actual_top10"].values)
            print(f"  Platt params: a={platt_params[0]:.4f}, b={platt_params[1]:.4f} "
                  f"(from {len(valid)} backtest rows)")

    # ── Full calibration pipeline: S2 + rank-form blend + stale cap ──
    tiers = fdf["augusta_experience_tier"].fillna(0).values
    cal = calibrate_full_pipeline(
        s2_raw, tiers,
        platt_params=platt_params,
        n_sims=N_SIMS, noise_std=MC_NOISE,
        target_pred_std=MC_TPRED, seed=SEED,
        field_size=len(fdf),
        s1_scores=rank_form_input,   # (1-rank_form) — inverted back inside pipeline
        blend_weight=BLEND_S2_WEIGHT_PRED,
        stale_mask=stale_mask.values,
        stale_cap=STALE_CAP,
    )

    fdf["win_prob"] = cal["win_prob"].values
    fdf["top5_prob"] = cal["top5_prob"].values
    fdf["top10_prob"] = cal["top10_prob"].values
    fdf["top20_prob"] = cal["top20_prob"].values
    fdf["make_cut_prob"] = cal["make_cut_prob"].values
    fdf["s2_platt"] = cal["s2_calibrated"].values

    # ── Join market odds from DataGolf (real sportsbook lines) ───────────────
    # Tries betting-tools/outrights first (live), falls back to pre-tournament snapshot.
    print("\n  Fetching market odds...")
    mkt = fetch_dg_market_odds()
    if mkt is not None:
        for _, mrow in mkt.iterrows():
            best = process.extractOne(mrow["player_name"], fdf["player_name"].tolist(), scorer=fuzz.ratio)
            if best and best[1] >= 80:
                idx = fdf[fdf["player_name"] == best[0]].index
                if len(idx) > 0:
                    for src_col, dst_col in [
                        ("market_win",   "market_win"),
                        ("market_top5",  "market_top5"),
                        ("market_top10", "market_top10"),
                        ("market_top20", "market_top20"),
                    ]:
                        if src_col in mrow and pd.notna(mrow[src_col]):
                            fdf.loc[idx[0], dst_col] = mrow[src_col]
        mkt_win  = fdf.get("market_win",   pd.Series(dtype=float)).reindex(fdf.index).fillna(0)
        mkt_t10  = fdf.get("market_top10", pd.Series(dtype=float)).reindex(fdf.index).fillna(0)
        mkt_t20  = fdf.get("market_top20", pd.Series(dtype=float)).reindex(fdf.index).fillna(0)
        fdf["kelly_edge_win"]   = np.where(mkt_win  > 0, (fdf["win_prob"]   - mkt_win)  / mkt_win,  np.nan)
        fdf["kelly_edge_top10"] = np.where(mkt_t10  > 0, (fdf["top10_prob"] - mkt_t10)  / mkt_t10,  np.nan)
        fdf["kelly_edge_top20"] = np.where(mkt_t20  > 0, (fdf["top20_prob"] - mkt_t20)  / mkt_t20,  np.nan)
        # Keep dk_fair_prob_win/top10 aliases for backward compatibility with Streamlit pages
        fdf["dk_fair_prob_win"]   = fdf.get("market_win",   pd.Series(dtype=float)).reindex(fdf.index)
        fdf["dk_fair_prob_top10"] = fdf.get("market_top10", pd.Series(dtype=float)).reindex(fdf.index)
    else:
        print("  WARNING: No market odds available — edges will be NaN")

    # ── Post-MC hard cap for honorary invitees (unranked past champions) ──
    # MC noise gives every player ~4-6% T10 even at zero skill due to golf variance.
    # Unranked players (Fred Couples, Vijay Singh, etc.) are not real contenders —
    # hard-cap them post-MC so they don't pollute the value plays table.
    unranked_mask = dg_rank.isna().values
    if unranked_mask.any():
        # Reindex mask to match fdf after potential row drops/resets
        unranked_idx = [i for i, m in zip(fdf.index, unranked_mask) if m]
        fdf.loc[unranked_idx, "win_prob"] = np.minimum(fdf.loc[unranked_idx, "win_prob"], 0.002)
        fdf.loc[unranked_idx, "top5_prob"] = np.minimum(fdf.loc[unranked_idx, "top5_prob"], 0.005)
        fdf.loc[unranked_idx, "top10_prob"] = np.minimum(fdf.loc[unranked_idx, "top10_prob"], 0.010)
        fdf.loc[unranked_idx, "top20_prob"] = np.minimum(fdf.loc[unranked_idx, "top20_prob"], 0.020)
        n_capped = int(unranked_mask.sum())
        print(f"  Honorary invitees (unranked) hard-capped: {n_capped} players → max T10=1%")

    fdf = fdf.sort_values("top10_prob", ascending=False).reset_index(drop=True)

    # ── Print results ──
    print(f"\n  {'#':<3} {'Player':<25} {'Win%':>6} {'T5%':>6} {'T10%':>6} {'T20%':>6} "
          f"{'Tier':>4} {'S2raw':>6}")
    print("  " + "-" * 80)
    for i, r in fdf.head(25).iterrows():
        tier = r.get("augusta_experience_tier", 0)
        print(f"  {i + 1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top5_prob']:>5.1%} "
              f"{r['top10_prob']:>5.1%} {r['top20_prob']:>5.1%} "
              f"{tier:>4.0f} {r['stage2_prob_raw']:>5.3f}")

    # Sum checks
    print(f"\n  Sums: win={fdf['win_prob'].sum():.3f} top5={fdf['top5_prob'].sum():.1f} "
          f"top10={fdf['top10_prob'].sum():.1f} top20={fdf['top20_prob'].sum():.1f}")

    # Value plays — exclude unranked/stale honorary invitees (not real betting targets)
    if "dk_fair_prob_top10" in fdf.columns:
        active = fdf[
            fdf.get("kelly_edge_top10", pd.Series(0)) > 0
        ].copy()
        # Filter to active/ranked players only (skip retired past-champion invitees)
        active_ranked = active[active["dg_rank"].notna() & (active["dg_rank"] <= STALE_RANK_THRESHOLD)]
        print(f"\n  TOP VALUE PLAYS (model > market for top-10, ranked players only):")
        for _, r in active_ranked.nlargest(8, "kelly_edge_top10").iterrows():
            print(f"    {r['player_name']:<25} model={r['top10_prob']:.1%} "
                  f"mkt={r['dk_fair_prob_top10']:.1%} edge={r['kelly_edge_top10']:+.0%}")

    # Save
    out_cols = [c for c in [
        "dg_id", "player_name", "country", "dg_rank",
        "win_prob", "top5_prob", "top10_prob", "top20_prob", "make_cut_prob",
        "model_score", "stage2_prob_raw", "s2_platt",
        "augusta_competitive_rounds", "augusta_experience_tier",
        "augusta_made_cut_prev_year", "augusta_scoring_trajectory",
        "tour_vs_augusta_divergence",
        "market_win", "market_top5", "market_top10", "market_top20",
        "dk_fair_prob_win", "dk_fair_prob_top10",
        "kelly_edge_win", "kelly_edge_top10", "kelly_edge_top20",
    ] if c in fdf.columns]

    save = fdf[out_cols]
    save.to_parquet(PROCESSED / "predictions_2026.parquet", index=False)
    save.to_csv(PROCESSED / "predictions_2026.csv", index=False)
    print(f"\n  Saved: predictions_2026.parquet + .csv ({len(save)} players)")

    return fdf


# ═════════════════════════════════════════════════════════���═
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="backtest only, no 2026 preds")
    args = parser.parse_args()

    print("=" * 60)
    print("  AUGUSTA NATIONAL MODEL — PRODUCTION PIPELINE")
    print("=" * 60)

    # Load data
    ext = pd.read_parquet(PROCESSED / "historical_rounds_extended.parquet")

    # Handle round-level schema: if round_num column exists, aggregate to tournament level
    # The golden endpoint returns real per-round SG values. The original CSV (2015-2022)
    # stores event_average / rounds_played (i.e., per-round average divided by rounds again).
    # To match scales: average round values, then divide by rounds played.
    if "round_num" in ext.columns:
        n_raw = len(ext)
        round_rows = ext[ext["round_num"] > 0]
        tourney_rows = ext[ext["round_num"] == 0]  # legacy rows without round-level data
        if len(round_rows) > 0:
            sg_agg = {c: "mean" for c in SG if c in round_rows.columns}
            sg_agg["round_num"] = "count"  # count rounds to divide by
            keep_cols = ["dg_id", "player_name", "season", "event_name", "course",
                         "date", "field_size", "finish_pos"]
            agg_round = (round_rows.groupby(keep_cols, dropna=False)
                         .agg(sg_agg).reset_index())
            # Divide by rounds played to match original CSV scale
            n_rounds = agg_round["round_num"]
            for c in SG:
                if c in agg_round.columns:
                    agg_round[c] = agg_round[c] / n_rounds
            agg_round.drop(columns=["round_num"], inplace=True)
            ext = pd.concat([tourney_rows.drop(columns=["round_num"], errors="ignore"),
                             agg_round], ignore_index=True)
            print(f"  Round-level data: {n_raw} rows → {len(ext)} tournaments (scaled to match CSV)")

    ext["finish_num"] = ext["finish_pos"].apply(parse_finish)
    unified = pd.read_parquet(PROCESSED / "masters_unified.parquet")
    unified["finish_num"] = unified["finish_pos"].apply(parse_finish)
    features = pd.read_parquet(PROCESSED / "augusta_player_features.parquet")
    rounds_df = pd.read_parquet(PROCESSED / "masters_sg_rounds.parquet")
    print(f"  Data loaded: {len(ext):,} tour rows, {len(unified)} Augusta rows")

    # Course weights
    cw_path = PROCESSED / "augusta_sg_weights.json"
    cw = json.load(open(cw_path)) if cw_path.exists() else HARDCODED_CW

    # Field strength
    fs_path = PROCESSED / "dg_field_strength.csv"
    fs_lookup = FieldStrengthLookup(fs_path)

    # Build rolling features
    print("  Computing rolling features...")
    tour_clean = ext.dropna(subset=SG)
    tour_clean = tour_clean[tour_clean["finish_num"] < 999]
    tour_cw = apply_course_weights(tour_clean, cw)
    tour_feat = build_rolling_features(tour_cw, fs_lookup)
    tour_feat["finish_pct"] = ((tour_feat["finish_num"] - 1) / (tour_feat["field_size"] - 1)).clip(0, 1)
    print(f"  Rolling features: {len(tour_feat):,} rows")

    # Train S1 on all data
    tc = tour_feat.dropna(subset=ROLL + ["finish_pct"])
    s1 = xgb.XGBRegressor(**XGB_S1)
    s1.fit(tc[ROLL].values, tc["finish_pct"].values)
    print(f"  Stage 1 trained: {len(tc):,} rows")

    # Backtest
    bt_metrics, bt_results = run_backtest(tour_feat, unified, features, rounds_df)

    # 2026 predictions
    if not args.backtest:
        preds = predict_2026(tour_feat, unified, features, rounds_df, s1)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    if not args.backtest:
        print(f"  Streamlit: python3 -m streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()

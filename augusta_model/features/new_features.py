"""
New features for Augusta National Model.

Feature design based on Augusta National's statistical profile (PGA Splits data 2021-2025):
- Approach: THE toughest course (-87.7% vs tour). Everyone loses → who loses LEAST = winner.
- ARG: THE most difficult (-91.1%). Short-grass chipping is -97.3%. Scrambling is brutal.
- Putting: THE toughest (-76.7%). 3-putt rate +50%. Long putts (15+ ft) especially hard.
- OTT: 13th easiest. Wide fairways (54 yds) BUT missing fairway = 38.3% GIR (worst on tour).
- Par 4s: Hardest by a large margin (+45.7%). Par 5 BoB rate 39.7% = must score there.
- Bogey%: +47.7% higher than tour average. Birdie/Bogey ratio drops from 1.44 to 0.80.

Key insight: Augusta amplifies weaknesses. The features should capture RESILIENCE UNDER
DIFFICULTY, not just raw skill.
"""
import numpy as np
import pandas as pd
from pathlib import Path


# Events known for difficulty profiles similar to Augusta
HARD_APPROACH_EVENTS = [
    "masters", "u.s. open", "players championship", "memorial",
    "arnold palmer", "rbc heritage", "pga championship",
]
FAST_GREEN_EVENTS = [
    "masters", "u.s. open", "tour championship", "players championship",
    "memorial", "east lake", "tpc sawgrass",
]
HARD_ARG_EVENTS = [
    "masters", "u.s. open", "the open", "players championship",
    "memorial", "tour championship",
]


def _event_matches(event_name, patterns):
    if not isinstance(event_name, str):
        return False
    lower = event_name.lower()
    return any(p in lower for p in patterns)


# ═══════════════════════════════════════════════════════════
# 1. APPROACH DIFFICULTY RESILIENCE
# ═══════════════════════════════════════════════════════════

def add_approach_resilience(tour_df, field_df):
    """How well a player maintains approach play at the hardest courses.

    Augusta is THE toughest approach course (-87.7% relative). Everyone loses
    strokes on approach — the question is WHO LOSES LEAST. This feature captures
    a player's sg_app specifically at difficult-approach courses.
    """
    field_df = field_df.copy()

    if "event_name" not in tour_df.columns or "sg_app" not in tour_df.columns:
        field_df["hard_course_sg_app"] = 0.0
        return field_df

    hard = tour_df[tour_df["event_name"].apply(lambda e: _event_matches(e, HARD_APPROACH_EVENTS))].copy()
    if len(hard) > 0:
        player_app = (hard.groupby("player_name")["sg_app"]
                      .agg(["mean", "count"])
                      .rename(columns={"mean": "hard_course_sg_app", "count": "_app_n"}))
        player_app.loc[player_app["_app_n"] < 3, "hard_course_sg_app"] = np.nan
        field_df = field_df.merge(
            player_app[["hard_course_sg_app"]],
            left_on="player_name", right_index=True, how="left"
        )
    else:
        field_df["hard_course_sg_app"] = np.nan

    field_df["hard_course_sg_app"] = field_df["hard_course_sg_app"].fillna(0)
    return field_df


# ═══════════════════════════════════════════════════════════
# 2. SHORT GAME UNDER PRESSURE (ARG at tough courses)
# ═══════════════════════════════════════════════════════════

def add_arg_resilience(tour_df, field_df):
    """ARG performance at courses with Augusta-like short game demands.

    Augusta ARG is -91.1% vs tour. Short-grass chipping is -97.3%.
    Scrambling drops to 52.6%. This captures who handles tight lies
    and fast green surrounds.
    """
    field_df = field_df.copy()

    if "event_name" not in tour_df.columns or "sg_arg" not in tour_df.columns:
        field_df["hard_course_sg_arg"] = 0.0
        return field_df

    hard = tour_df[tour_df["event_name"].apply(lambda e: _event_matches(e, HARD_ARG_EVENTS))].copy()
    if len(hard) > 0:
        player_arg = (hard.groupby("player_name")["sg_arg"]
                      .agg(["mean", "count"])
                      .rename(columns={"mean": "hard_course_sg_arg", "count": "_arg_n"}))
        player_arg.loc[player_arg["_arg_n"] < 3, "hard_course_sg_arg"] = np.nan
        field_df = field_df.merge(
            player_arg[["hard_course_sg_arg"]],
            left_on="player_name", right_index=True, how="left"
        )
    else:
        field_df["hard_course_sg_arg"] = np.nan

    field_df["hard_course_sg_arg"] = field_df["hard_course_sg_arg"].fillna(0)
    return field_df


# ═══════════════════════════════════════════════════════════
# 3. THREE-PUTT AVOIDANCE (putting on fast greens)
# ═══════════════════════════════════════════════════════════

def add_putting_surface_features(tour_df, field_df):
    """Putting at events with Augusta-like fast undulating greens.

    Augusta 3-putt rate is 4.34% vs 2.83% tour avg (+50.3%).
    SG: Putting 15+ feet is -62.2% relative. It's not about making putts —
    it's about NOT three-putting.
    """
    field_df = field_df.copy()

    if "event_name" not in tour_df.columns or "sg_putt" not in tour_df.columns:
        field_df["fast_green_sg_putt"] = 0.0
        return field_df

    fast = tour_df[tour_df["event_name"].apply(lambda e: _event_matches(e, FAST_GREEN_EVENTS))].copy()
    if len(fast) > 0:
        player_putt = (fast.groupby("player_name")["sg_putt"]
                       .agg(["mean", "count"])
                       .rename(columns={"mean": "fast_green_sg_putt", "count": "_putt_n"}))
        player_putt.loc[player_putt["_putt_n"] < 3, "fast_green_sg_putt"] = np.nan
        field_df = field_df.merge(
            player_putt[["fast_green_sg_putt"]],
            left_on="player_name", right_index=True, how="left"
        )
    else:
        field_df["fast_green_sg_putt"] = np.nan

    field_df["fast_green_sg_putt"] = field_df["fast_green_sg_putt"].fillna(0)
    return field_df


# ═══════════════════════════════════════════════════════════
# 4. PAR-5 CONVERSION + BOGEY AVOIDANCE
# ═══════════════════════════════════════════════════════════

def add_scoring_profile(df):
    """Scoring features that match Augusta's unique par-type difficulty.

    Par 4s: +45.7% harder, BoB% drops -44.6% → bogey avoidance is critical
    Par 5s: BoB rate 39.7% → must score, driven by driving + approach
    Bogey%: +47.7% higher → the player who avoids big numbers wins

    Par-5 proxy uses OTT (reach in two) + APP (approach from 200+).
    Bogey resistance uses the spread between sg_total and sg_total_std
    (high skill + low variance = fewer blowup holes).
    """
    df = df.copy()

    ott = df.get("sg_ott_8w", pd.Series(0, index=df.index)).fillna(0)
    app = df.get("sg_app_8w", pd.Series(0, index=df.index)).fillna(0)
    total = df.get("sg_total_8w", pd.Series(0, index=df.index)).fillna(0)
    std = df.get("sg_total_std_8", pd.Series(0.5, index=df.index)).fillna(0.5)

    # Par-5 conversion: driving + approach = eagle machine on reachable par 5s
    df["par5_scoring_proxy"] = ott * 0.45 + app * 0.40 + total * 0.15

    # Bogey resistance: high skill + low variance = consistent, avoids blowups
    # Augusta bogey% is +47.7% — this is the biggest separator
    df["bogey_resistance"] = total - std  # high total, low std = safe player

    # Driving dominance: OTT - APP balance
    # At Augusta, long hitters have par-5 edge (13th, 15th reachable)
    # Positive = OTT-dominant (drives well but approach average/poor)
    # Negative = APP-dominant (elite iron player)
    df["driving_dominance"] = ott - app

    return df


# ═══════════════════════════════════════════════════════════
# 5. REAL TOURNAMENT WEATHER
# ═══════════════════════════════════════════════════════════

def add_weather_features(df, weather_path=None):
    """Real weather for each Masters year, replacing the dead constant.

    Wind/rain affect all SG categories at Augusta differently:
    - Wind: harder approach + putting, benefits experienced players
    - Rain: softens greens (easier putting), firmer = harder approach
    """
    df = df.copy()

    if weather_path is None:
        weather_path = Path("data/processed/masters_weather.parquet")
    if not Path(weather_path).exists():
        return df

    weather = pd.read_parquet(weather_path)

    if "season" in df.columns:
        df = df.merge(
            weather[["season", "wind_avg_mph", "wind_max_mph", "rain_total_mm"]],
            on="season", how="left", suffixes=("_old", "")
        )
        if "tournament_wind_avg_old" in df.columns:
            df.drop(columns=["tournament_wind_avg_old"], inplace=True)
    else:
        df["wind_avg_mph"] = 12.0
        df["wind_max_mph"] = 18.0
        df["rain_total_mm"] = 10.0

    df["tournament_wind_avg"] = df.get("wind_avg_mph",
                                        pd.Series(12.0, index=df.index)).fillna(12.0)

    # Wind × experience: experienced players handle tough conditions better
    comp = df.get("augusta_competitive_rounds", pd.Series(0, index=df.index)).fillna(0)
    wind = df["tournament_wind_avg"]
    df["wind_experience_interaction"] = (wind - 12.0) * np.log1p(comp)

    # Wet course: softens greens → helps approach, changes putting speed
    rain = df.get("rain_total_mm", pd.Series(0, index=df.index)).fillna(0)
    df["wet_course"] = (rain > 15.0).astype(float)

    return df


# ═══════════════════════════════════════════════════════════
# 6. FORM MOMENTUM
# ═══════════════════════════════════════════════════════════

def add_form_momentum(tour_df, field_df, target_year=2026):
    """Recent competitive results beyond SG averages.

    Captures: how recently a player contended, recent wins,
    and consistency of finishes (low variance = reliable performer).
    """
    field_df = field_df.copy()

    if "player_name" not in tour_df.columns or "finish_num" not in tour_df.columns:
        field_df["events_since_top10"] = 20
        field_df["recent_win"] = 0
        field_df["recent_consistency"] = 0.5
        return field_df

    tour_sorted = tour_df.sort_values(["player_name", "date"]).copy()
    momentum_rows = []

    for pn in field_df["player_name"].unique():
        pt = tour_sorted[(tour_sorted["player_name"] == pn) &
                         (tour_sorted["season"] <= target_year)]
        recent = pt.tail(20)
        d = {"player_name": pn}

        if len(recent) == 0:
            d.update({"events_since_top10": 20, "recent_win": 0, "recent_consistency": 0.5})
            momentum_rows.append(d)
            continue

        fn = recent["finish_num"].dropna()
        valid = fn[fn < 999]

        top10_mask = valid <= 10
        if top10_mask.any():
            last_t10_idx = top10_mask[top10_mask].index[-1]
            d["events_since_top10"] = min(len(valid.loc[last_t10_idx:]) - 1, 20)
        else:
            d["events_since_top10"] = 20

        last_8 = valid.tail(8)
        d["recent_win"] = int((last_8 == 1).any())

        if len(valid) >= 3:
            field_sizes = recent.loc[valid.index, "field_size"].fillna(156)
            finish_pcts = (valid - 1) / (field_sizes - 1)
            d["recent_consistency"] = finish_pcts.std()
        else:
            d["recent_consistency"] = 0.5

        momentum_rows.append(d)

    mom_df = pd.DataFrame(momentum_rows)
    field_df = field_df.merge(mom_df, on="player_name", how="left")
    field_df["events_since_top10"] = field_df["events_since_top10"].fillna(20)
    field_df["recent_win"] = field_df["recent_win"].fillna(0)
    field_df["recent_consistency"] = field_df["recent_consistency"].fillna(0.5)
    return field_df


# ═══════════════════════════════════════════════════════════
# 7. AUGUSTA-SPECIFIC SG INTERACTIONS
# ═══════════════════════════════════════════════════════════

def add_sg_interactions(df):
    """SG interactions tuned to Augusta's statistical profile.

    Based on PGA Splits data (2021-2025):
    - APP is THE separator (-87.7% relative) → weight highest
    - ARG is the most difficult (-91.1%) → weight very high
    - PUTT is toughest course (-76.7%) → weight high
    - OTT is 13th easiest → weight lowest, but position matters for approach
    """
    df = df.copy()

    ott = df.get("sg_ott_8w", pd.Series(0, index=df.index)).fillna(0)
    app = df.get("sg_app_8w", pd.Series(0, index=df.index)).fillna(0)
    arg = df.get("sg_arg_8w", pd.Series(0, index=df.index)).fillna(0)
    putt = df.get("sg_putt_8w", pd.Series(0, index=df.index)).fillna(0)

    # Augusta fit score: weighted by how much each category SEPARATES at Augusta
    # APP (-87.7%) and ARG (-91.1%) are where the variance is
    df["augusta_fit_score"] = app * 1.60 + arg * 1.50 + putt * 1.20 + ott * 0.70

    # Power × precision: OTT × APP interaction
    # Long AND accurate approach = eagle machine on par 5s
    df["power_precision"] = ott * app

    # Short game package: ARG + PUTT combined
    # Augusta's short game is the hardest on tour — this captures the combo
    df["short_game_package"] = arg + putt

    return df


# ═══════════════════════════════════════════════════════════
# 8. DIFFICULTY SCALING
# ═══════════════════════════════════════════════════════════

def add_winner_profile_features(tour_df, field_df, target_year=2026):
    """Features derived from Masters Winner Trends analysis.

    Based on historical winner profiles (last 20-46 winners):
    - 24/26 ranked top 30 in world rankings
    - 15/17 had 4+ career wins
    - 14/16 had top-6 in a major within 2 years
    - 16/16 finished 35th+ in their previous start
    - 15/16 had top-8 finish in 7 events before Masters
    - 14/14 gained 18+ strokes T2G in 4 events before
    - 21/23 played in one of 2 weeks before Masters
    """
    field_df = field_df.copy()
    tour_sorted = tour_df.sort_values(["player_name", "date"]).copy()

    major_patterns = ["masters", "u.s. open", "pga championship", "the open", "open championship"]

    rows = []
    for pn in field_df["player_name"].unique():
        pt = tour_sorted[(tour_sorted["player_name"] == pn) &
                         (tour_sorted["season"] < target_year)]
        d = {"player_name": pn}

        if len(pt) == 0:
            d.update({
                "career_wins": 0, "recent_major_top6": 0,
                "last_start_finish_pct": 0.5, "top8_in_last7": 0,
                "t2g_last4_total": 0.0, "played_recently": 0,
            })
            rows.append(d)
            continue

        # Resolve finish column (tour data uses finish_pos, backtest uses finish_num)
        if "finish_num" in pt.columns:
            fn_col = "finish_num"
            fn = pt[fn_col].dropna()
        else:
            # Parse finish_pos to numeric
            def _pfn(p):
                if p is None or pd.isna(p): return np.nan
                p = str(p).strip().upper()
                if p in ("CUT","MC","WD","DQ","MDF"): return 999.0
                p = p.replace("T","").replace("=","")
                try: return float(p)
                except: return np.nan
            fn_col = "_fn"
            pt = pt.copy()
            pt["_fn"] = pt["finish_pos"].apply(_pfn)
            fn = pt["_fn"].dropna()

        # Career wins
        d["career_wins"] = int((fn == 1).sum())

        # Recent major top-6 (within 2 years)
        recent = pt[pt["season"] >= target_year - 2]
        majors = recent[recent["event_name"].apply(
            lambda e: any(p in str(e).lower() for p in major_patterns) if pd.notna(e) else False)]
        major_finishes = majors[fn_col].dropna() if fn_col in majors.columns else pd.Series()
        d["recent_major_top6"] = int((major_finishes <= 6).any()) if len(major_finishes) > 0 else 0

        # Last start finish (16/16 winners finished 35th+ in previous start)
        valid_finishes = pt[pt[fn_col] < 999].tail(1) if fn_col in pt.columns else pd.DataFrame()
        if len(valid_finishes) > 0:
            last_f = valid_finishes.iloc[0][fn_col]
            last_fs = valid_finishes.iloc[0].get("field_size", 156)
            d["last_start_finish_pct"] = (last_f - 1) / max(last_fs - 1, 1)
        else:
            d["last_start_finish_pct"] = 0.5

        # Top-8 in last 7 events
        last7 = pt.tail(7)
        last7_fn = last7[fn_col].dropna() if fn_col in last7.columns else pd.Series()
        d["top8_in_last7"] = int((last7_fn[last7_fn < 999] <= 8).any()) if len(last7_fn) > 0 else 0

        # T2G total in last 4 events
        last4 = pt.tail(4)
        t2g = last4["sg_t2g"].dropna() if "sg_t2g" in last4.columns else pd.Series()
        d["t2g_last4_total"] = t2g.sum() if len(t2g) > 0 else 0.0

        # Events since last start (0 = played in most recent event, higher = longer gap)
        # 21/23 Masters winners played in one of the 2 weeks before Masters
        last_date = pt.iloc[-1].get("date", None)
        try:
            from datetime import datetime
            masters_approx = pd.Timestamp(f"{target_year}-04-10")
            if last_date is not None and pd.notna(last_date):
                ld = pd.Timestamp(last_date)
                days_gap = (masters_approx - ld).days
                d["played_recently"] = 1 if days_gap <= 21 else 0
            else:
                d["played_recently"] = 0
        except Exception:
            d["played_recently"] = 0

        rows.append(d)

    wp_df = pd.DataFrame(rows)
    field_df = field_df.merge(wp_df, on="player_name", how="left")

    # Fill defaults
    field_df["career_wins"] = field_df["career_wins"].fillna(0)
    field_df["recent_major_top6"] = field_df["recent_major_top6"].fillna(0)
    field_df["last_start_finish_pct"] = field_df["last_start_finish_pct"].fillna(0.5)
    field_df["top8_in_last7"] = field_df["top8_in_last7"].fillna(0)
    field_df["t2g_last4_total"] = field_df["t2g_last4_total"].fillna(0)
    field_df["played_recently"] = field_df["played_recently"].fillna(0)

    return field_df


# ═══════════════════════════════════════════════════════════
# 9. AGING DECAY FOR AUGUSTA FEATURES
# ═══════════════════════════════════════════════════════════

# Known birth years for past champions / old invitees who inflate Augusta features
KNOWN_BIRTH_YEARS = {
    "Vijay Singh": 1963,
    "Fred Couples": 1959,
    "Jose Maria Olazabal": 1966,
    "Bernhard Langer": 1957,
    "Larry Mize": 1958,
    "Sandy Lyle": 1958,
    "Ian Woosnam": 1958,
    "Mark O'Meara": 1957,
    "Ben Crenshaw": 1952,
    "Tiger Woods": 1975,
    "Phil Mickelson": 1970,
    "Bubba Watson": 1978,
    "Zach Johnson": 1976,
    "Angel Cabrera": 1969,
    "Mike Weir": 1970,
    "Charl Schwartzel": 1984,
    "Adam Scott": 1980,
    "Danny Willett": 1987,
    "Patrick Reed": 1990,
    "Sergio Garcia": 1980,
    "Hideki Matsuyama": 1992,
    "Scottie Scheffler": 1996,
    "Jon Rahm": 1994,
    "Rory McIlroy": 1989,
}

# Augusta features to decay (continuous features that inflate with longevity)
DECAY_FEATURES = [
    "augusta_competitive_rounds",
    "augusta_top10_rate",
    "augusta_made_cut_rate",
    "augusta_best_finish",
    "augusta_scoring_avg",
    "augusta_sg_app_career",
    "augusta_sg_total_career",
]


def add_aging_decay(field_df, tour_df=None, current_year=2026):
    """Decay Augusta experience features for players past their prime.

    No Masters winner over 46 (Nicklaus 1986, age 46). Players over 45
    should have Augusta experience features progressively discounted.

    Uses known birth years for past champions, falls back to estimating
    from first season in tour data (assume turned pro at ~22).

    Decay formula: max(0.10, 1.0 - (age - 45) * 0.10)
      age 45: 1.00 (no decay)
      age 50: 0.50
      age 55: 0.25
      age 60+: 0.10 (floor)
    """
    field_df = field_df.copy()

    # Estimate ages
    ages = {}
    for _, row in field_df.iterrows():
        name = row["player_name"]
        if name in KNOWN_BIRTH_YEARS:
            ages[name] = current_year - KNOWN_BIRTH_YEARS[name]
        elif tour_df is not None and "player_name" in tour_df.columns:
            player_data = tour_df[tour_df["player_name"] == name]
            if len(player_data) > 0 and "season" in player_data.columns:
                first_season = player_data["season"].min()
                ages[name] = current_year - first_season + 22
            else:
                ages[name] = 30  # default: assume prime age
        else:
            ages[name] = 30

    field_df["estimated_age"] = field_df["player_name"].map(ages)

    # Compute decay factor
    field_df["age_decay_factor"] = field_df["estimated_age"].apply(
        lambda age: max(0.10, 1.0 - (age - 45) * 0.10) if age > 45 else 1.0
    )

    # Apply decay to Augusta features
    for feat in DECAY_FEATURES:
        if feat in field_df.columns:
            field_df[feat] = field_df[feat] * field_df["age_decay_factor"]

    return field_df


def add_difficulty_scaling(tour_df, field_df):
    """How a player's performance scales with course difficulty.

    Augusta amplifies weaknesses. Players who maintain their level at
    hard courses (US Open, Players, majors) outperform at Augusta vs
    players who pad stats at weak-field events.

    Computed as: sg_total at hard courses / sg_total overall.
    Ratio > 1.0 = plays UP at hard courses (Augusta fit).
    Ratio < 1.0 = plays DOWN (Augusta trap).
    """
    field_df = field_df.copy()

    hard_patterns = ["masters", "u.s. open", "pga championship", "the open",
                     "players championship", "memorial", "tour championship",
                     "arnold palmer", "genesis"]

    if "event_name" not in tour_df.columns or "sg_total" not in tour_df.columns:
        field_df["difficulty_scaling"] = 1.0
        return field_df

    rows = []
    for pn in field_df["player_name"].unique():
        pt = tour_df[tour_df["player_name"] == pn]
        if len(pt) < 5:
            rows.append({"player_name": pn, "difficulty_scaling": 1.0})
            continue

        overall_sg = pt["sg_total"].mean()
        hard = pt[pt["event_name"].apply(lambda e: _event_matches(e, hard_patterns))]

        if len(hard) >= 3:
            hard_sg = hard["sg_total"].mean()
            # Ratio: hard performance relative to overall
            # If overall is near 0, use difference instead
            if abs(overall_sg) > 0.05:
                scaling = hard_sg / overall_sg
            else:
                scaling = 1.0 + (hard_sg - overall_sg)
            scaling = np.clip(scaling, 0.3, 3.0)
        else:
            scaling = 1.0

        rows.append({"player_name": pn, "difficulty_scaling": scaling})

    scale_df = pd.DataFrame(rows)
    field_df = field_df.merge(scale_df, on="player_name", how="left")
    field_df["difficulty_scaling"] = field_df["difficulty_scaling"].fillna(1.0)
    return field_df

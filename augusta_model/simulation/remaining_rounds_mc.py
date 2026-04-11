"""
Remaining-rounds Monte Carlo simulator — player-specific scoring distributions.

Architecture:
  For each player, builds a personal expected score per remaining round from:
    1. In-tournament SG (this week's actual ball-striking/putting)
       — sg_app, sg_ott, sg_arg, sg_putt from DG live-tournament-stats
       — Converted to strokes via Augusta-specific regression coefficients
    2. Pre-tournament skill (season-long SG weighted by Augusta course fit)
       — Blended with in-tournament form (weight shifts toward in-tourney as
         more rounds are complete)
    3. Historical Augusta scoring variance (std ~3.0/round, field correlation 0.30)

Win probability = fraction of 20k simulations where player finishes lowest.

Key insight from data analysis:
  - Per-round SG components strongly predict that round's score:
      sg_app r=-0.598, sg_putt r=-0.521, sg_arg r=-0.474, sg_ott r=-0.414
  - Cumulative SG R1+R2 predicts R3+R4 score weakly (r=-0.095) — mean-reversion
  - Augusta historical round score: mean +0.5, std 3.05 (R3 slightly harder: +0.83)
  - No player in 11 years has shot better than -8 in any round
  - Field-wide correlation 0.30 (weather, pin positions affect everyone)

SG-to-strokes regression coefficients (from Augusta 2021-2025):
  score = intercept + b_ott*sg_ott + b_app*sg_app + b_arg*sg_arg + b_putt*sg_putt
  1 SG gained → approximately -0.85 strokes (slightly less than theoretical 1.0
  because SG is measured against the entire PGA Tour field, not just Masters field)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Augusta historical scoring parameters ─────────────────────────────────────
# From masters_sg_rounds.parquet + masters_hole_by_hole.parquet 2015-2025
ROUND_PARAMS = {
    1: {"mean": 0.25, "std": 2.83},
    2: {"mean": 0.27, "std": 2.74},
    3: {"mean": 0.83, "std": 3.25},
    4: {"mean": 0.01, "std": 3.01},
}
DEFAULT_ROUND_MEAN = 0.50
DEFAULT_ROUND_STD  = 3.05

# Field-wide correlation per round (weather, pin positions affect everyone)
FIELD_CORR = 0.28

# SG-to-score coefficients at Augusta (negative = better SG → lower score)
# Derived from per-round sg vs round_score correlation; sg_app most predictive
SG_TO_STROKES = {
    "sg_app":  -0.72,   # approach dominates at Augusta
    "sg_putt": -0.62,
    "sg_arg":  -0.55,
    "sg_ott":  -0.48,
}

# Augusta course-fit weights (from augusta_sg_weights.json in project)
# Multiplied into pre-tournament SG features
AUGUSTA_WEIGHTS = {
    "sg_ott":  0.247,
    "sg_app":  0.257,
    "sg_arg":  0.247,
    "sg_putt": 0.259,
}

N_SIMULATIONS = 20_000
SEED = 42


def _inweek_sg_to_strokes_per_round(
    sg_row: dict,
    rounds_played: float,
) -> float:
    """
    Convert cumulative in-week SG to expected strokes per remaining round.

    sg_row: dict with sg_app, sg_ott, sg_arg, sg_putt (event cumulative)
    rounds_played: how many rounds of data these SG numbers represent

    Returns expected strokes vs par per round (negative = better than field avg).
    """
    if rounds_played <= 0:
        return DEFAULT_ROUND_MEAN

    # Per-round average SG for each component
    strokes_adj = 0.0
    for sg_col, coef in SG_TO_STROKES.items():
        val = sg_row.get(sg_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            per_round = float(val) / rounds_played
            strokes_adj += coef * per_round

    return DEFAULT_ROUND_MEAN + strokes_adj


def _pretournament_expected_score(player_name: str, pre_df: pd.DataFrame) -> float:
    """
    Pre-tournament expected score per round at Augusta from season-long SG.
    Uses Augusta course-fit weights on rolling SG features.
    """
    row = pre_df[pre_df["player_name"] == player_name]
    if row.empty:
        return DEFAULT_ROUND_MEAN

    row = row.iloc[0]

    # Try SG rolling features if available
    sg_cols_8w = ["sg_ott_8w", "sg_app_8w", "sg_arg_8w", "sg_putt_8w"]
    sg_cols_3w = ["sg_ott_3w", "sg_app_3w", "sg_arg_3w", "sg_putt_3w"]
    sg_keys    = ["sg_ott",    "sg_app",    "sg_arg",    "sg_putt"]

    # Pick best available SG source
    sg_vals = {}
    for key, c8, c3 in zip(sg_keys, sg_cols_8w, sg_cols_3w):
        if c8 in row.index and pd.notna(row[c8]):
            sg_vals[key] = float(row[c8])
        elif c3 in row.index and pd.notna(row[c3]):
            sg_vals[key] = float(row[c3])

    if sg_vals:
        # Apply Augusta weights and SG-to-strokes conversion
        adj = sum(
            AUGUSTA_WEIGHTS.get(k, 0.25) * SG_TO_STROKES.get(k, -0.5) * v
            for k, v in sg_vals.items()
        )
        return DEFAULT_ROUND_MEAN + adj * 4  # scale: Augusta weights sum to ~1, multiply back

    # Fall back to model_score (0=best, 1=worst)
    if "model_score" in row.index and pd.notna(row["model_score"]):
        deviation = (float(row["model_score"]) - 0.5) * 5
        return DEFAULT_ROUND_MEAN + deviation

    return DEFAULT_ROUND_MEAN


def _momentum_adjust(
    baseline: float,
    current_pace: float | None,
    thru_holes: float,
    segment: str,
) -> float:
    """
    Bayesian blend of baseline expected score with current-round pace.

    Models the intuition that a player scorching through 12 holes is genuinely
    hot right now, but that heat decays rapidly across future rounds.

    Momentum weights (multiply by thru_holes/18 so early holes count less):
      "current_rest" — rest of current round:     0.25 × (thru/18)
      "next"         — next full round:           0.10 × (thru/18)
      "later"        — rounds 2+ out:             0.03 × (thru/18)

    For a player at thru=12 on a -9 pace:
      current_rest weight = 0.25 × (12/18) = 0.167
      next weight         = 0.10 × (12/18) = 0.067
      later weight        = 0.03 × (12/18) = 0.020

    Minimum 3 holes required to trust the pace signal.
    """
    if current_pace is None or thru_holes < 3:
        return baseline
    fraction = min(thru_holes / 18.0, 1.0)
    weights = {"current_rest": 0.25, "next": 0.10, "later": 0.03}
    w = weights.get(segment, 0.0) * fraction
    return baseline * (1.0 - w) + current_pace * w


def build_player_distributions(
    live_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    current_round: int,
    median_thru: float,
) -> pd.DataFrame:
    """
    Build per-player expected score and std for remaining rounds.

    Blends in-tournament SG (what they're doing THIS week) with
    pre-tournament skill (season-long Augusta-weighted profile).

    Blend weight toward in-tournament form:
      After R1:  40% in-week / 60% pre-tourney  (small sample)
      After R2:  65% in-week / 35% pre-tourney
      After R3:  85% in-week / 15% pre-tourney

    Additionally applies a momentum update based on current-round pace:
      A player scoring -6 through 12 holes is on a -9 pace. This signal
      decays: strong for the rest of the current round, moderate for the
      next round, minimal for rounds further out.
    """
    total_holes = (current_round - 1) * 18 + median_thru
    rounds_played = total_holes / 18.0

    if rounds_played <= 1.0:
        inweek_w = 0.25 + 0.15 * rounds_played   # 0.25→0.40 through R1
    elif rounds_played <= 2.0:
        inweek_w = 0.40 + 0.25 * (rounds_played - 1.0)  # 0.40→0.65 through R2
    elif rounds_played <= 3.0:
        inweek_w = 0.65 + 0.20 * (rounds_played - 2.0)  # 0.65→0.85 through R3
    else:
        inweek_w = 0.85 + 0.10 * (rounds_played - 3.0)  # 0.85→0.95 through R4
    inweek_w = min(inweek_w, 0.95)
    pre_w = 1.0 - inweek_w

    records = []
    for _, player in live_df.iterrows():
        name = player["player_name"]

        # In-tournament expected score (from this week's live SG)
        # Use player's individual rounds_played for SG division (not field median)
        player_thru = player.get("thru", 0)
        player_thru = float(player_thru) if pd.notna(player_thru) else 0.0
        player_rounds_played = ((current_round - 1) * 18 + player_thru) / 18.0
        sg_rounds_for_div = max(player_rounds_played, median_thru / 18.0, 0.5)

        sg_row = {
            "sg_app":  player.get("sg_app"),
            "sg_ott":  player.get("sg_ott"),
            "sg_arg":  player.get("sg_arg"),
            "sg_putt": player.get("sg_putt"),
        }
        inweek_mean = _inweek_sg_to_strokes_per_round(sg_row, sg_rounds_for_div)

        # Pre-tournament expected score (season-long Augusta profile)
        pre_mean = _pretournament_expected_score(name, pre_df)

        # Blended expected score per remaining round (SG-based, no momentum yet)
        blended_mean = inweek_w * inweek_mean + pre_w * pre_mean

        # ── Momentum update from current-round pace ───────────────────────────
        # today = score-to-par for the current round in progress (not cumulative)
        today_val = player.get("today")
        current_pace: float | None = None
        if today_val is not None and pd.notna(today_val) and player_thru >= 3:
            # Annualise to full-round equivalent (18 holes)
            current_pace = float(today_val) / player_thru * 18.0

        expected_current_rest = _momentum_adjust(blended_mean, current_pace, player_thru, "current_rest")
        expected_next_round   = _momentum_adjust(blended_mean, current_pace, player_thru, "next")
        expected_later_round  = _momentum_adjust(blended_mean, current_pace, player_thru, "later")

        # Std dev: slightly reduce for genuinely hot players (they're in control)
        sg_total_pw = player.get("sg_total")
        if sg_total_pw is not None and not np.isnan(float(sg_total_pw if sg_total_pw else 0)):
            sg_per_round = float(sg_total_pw) / max(sg_rounds_for_div, 0.5)
            std_adj = max(0.85, 1.0 - 0.03 * min(sg_per_round, 5))
        else:
            std_adj = 1.0

        records.append({
            "player_name":          name,
            "expected_score_per_round": blended_mean,  # SG-based only (for backward compat)
            "expected_current_rest": expected_current_rest,
            "expected_next_round":   expected_next_round,
            "expected_later_round":  expected_later_round,
            "round_std_adj":         std_adj,
            "inweek_mean":           blended_mean,      # momentum-unweighted, for display
            "pre_mean":              pre_mean,
            "inweek_weight":         inweek_w,
            "current_pace":          current_pace if current_pace is not None else float("nan"),
        })

    return pd.DataFrame(records)


def simulate_remaining_rounds(
    live_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    current_round: int,
    median_thru: float,
    n_sims: int = N_SIMULATIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of remaining rounds using player-specific distributions.

    live_df: must have player_name, current_score (cumulative to-par),
             dg_make_cut, and live SG columns (sg_app, sg_ott, sg_arg, sg_putt)
    pre_df:  pre-tournament predictions with model_score and rolling SG features
    current_round: 1-4
    median_thru: median holes completed in current round

    Returns DataFrame with mc_win_prob, mc_top5_prob, mc_top10_prob,
    mc_projected_total, strokes_back, expected_score_per_round.
    """
    rng = np.random.default_rng(seed)

    # Filter to active players
    players = live_df.copy()
    if "dg_make_cut" in players.columns:
        players = players[players["dg_make_cut"] > 0.05].copy()
    players = players.reset_index(drop=True)
    n_players = len(players)

    if n_players == 0:
        return pd.DataFrame(columns=[
            "player_name", "mc_win_prob", "mc_top5_prob", "mc_top10_prob",
            "mc_projected_total", "strokes_back", "expected_score_per_round"
        ])

    # Build player-specific scoring distributions
    player_dists = build_player_distributions(
        players, pre_df, current_round, median_thru
    )
    player_dists = player_dists.set_index("player_name")

    current_scores = players["current_score"].fillna(0).values.astype(float)

    # ── Per-player remaining round calculation ────────────────────────────────
    # Use individual 'thru' so a player through 12 holes (6 left) is not treated
    # the same as a player through 6 holes (12 left). current_score already
    # incorporates each player's actual holes played, so we only need to simulate
    # the correct remaining fraction per player.
    if "thru" in players.columns:
        player_thru_arr = players["thru"].fillna(median_thru).astype(float).values
    elif "holes_completed" in players.columns:
        player_thru_arr = players["holes_completed"].fillna(median_thru).astype(float).values
    else:
        player_thru_arr = np.full(n_players, float(median_thru))

    # Per-player fraction of current round still remaining (0 = done, 1 = not started)
    player_round_remaining = np.clip(1.0 - player_thru_arr / 18.0, 0.0, 1.0)  # (n_players,)
    full_rounds_remaining  = max(0, 4 - current_round)
    player_rounds_left     = player_round_remaining + full_rounds_remaining    # (n_players,)

    # Scalar rounds_left (field average) for field-shock variance scaling
    rounds_left = float(np.mean(player_round_remaining)) + full_rounds_remaining

    remaining_scores = None   # defined below when rounds_left > 0
    if rounds_left <= 0:
        final_scores = np.tile(current_scores, (n_sims, 1))
    else:
        def _get_dist_col(col, default):
            return np.array([
                float(player_dists.loc[name, col])
                if name in player_dists.index else default
                for name in players["player_name"]
            ])

        std_adj = _get_dist_col("round_std_adj", 1.0)

        # Per-player, per-segment momentum-adjusted expected scores
        # "current_rest" = rest of the round in progress
        # "next"         = first full round after the current one
        # "later"        = any rounds further out
        exp_current_rest = _get_dist_col("expected_current_rest", DEFAULT_ROUND_MEAN)
        exp_next_round   = _get_dist_col("expected_next_round",   DEFAULT_ROUND_MEAN)
        exp_later_round  = _get_dist_col("expected_later_round",  DEFAULT_ROUND_MEAN)

        # Build per-player remaining means, accounting for individual round fractions.
        # r_offset 0: current round remainder (per-player fraction)
        # r_offset 1: first full remaining round
        # r_offset 2+: further full rounds
        remaining_means = np.zeros(n_players)
        for r_offset in range(5):
            abs_round = current_round + r_offset
            if abs_round > 4:
                break
            rp = ROUND_PARAMS.get(abs_round, {"mean": DEFAULT_ROUND_MEAN, "std": DEFAULT_ROUND_STD})
            round_mean_shift = rp["mean"] - DEFAULT_ROUND_MEAN
            if r_offset == 0:
                frac = player_round_remaining        # (n_players,) — per-player
                exp_this_segment = exp_current_rest
            elif r_offset == 1:
                # Only applies if full_rounds_remaining >= 1
                if full_rounds_remaining < 1:
                    break
                frac = np.ones(n_players)
                exp_this_segment = exp_next_round
            else:
                if full_rounds_remaining < r_offset:
                    break
                frac = np.ones(n_players)
                exp_this_segment = exp_later_round
            remaining_means += exp_this_segment * frac + round_mean_shift * frac

        # Field-wide shock (same for all players — weather, pins)
        field_std = DEFAULT_ROUND_STD * np.sqrt(FIELD_CORR * max(rounds_left, 0.1))
        field_shocks = rng.normal(0, field_std, n_sims)  # (n_sims,)

        # Individual variance scales with per-player rounds remaining
        indiv_std = (
            DEFAULT_ROUND_STD * np.sqrt(np.maximum((1.0 - FIELD_CORR) * player_rounds_left, 0.01))
        )  # (n_players,)
        # (n_sims, n_players) — player-adjusted std
        individual_scores = rng.normal(
            remaining_means,                    # (n_players,) broadcast
            indiv_std * std_adj,                # (n_players,)
            size=(n_sims, n_players),
        )

        remaining_scores = individual_scores + field_shocks[:, np.newaxis]

        # Make-cut filter: players unlikely to make cut get +30 penalty
        if "dg_make_cut" in players.columns:
            cut_probs = players["dg_make_cut"].fillna(1.0).values
            cut_made  = rng.random((n_sims, n_players)) < cut_probs
            remaining_scores = np.where(cut_made, remaining_scores, remaining_scores + 30)

        final_scores = current_scores + remaining_scores

    # Count outcomes
    n = n_players
    winners    = np.argmin(final_scores, axis=1)
    sorted_idx = np.argsort(final_scores, axis=1)

    win_counts   = np.bincount(winners, minlength=n)
    top5_counts  = np.zeros(n, dtype=int)
    top10_counts = np.zeros(n, dtype=int)
    for i in range(n):
        top5_counts[i]  = (sorted_idx[:, :5]  == i).sum()
        top10_counts[i] = (sorted_idx[:, :10] == i).sum()

    # ── Percentile projected totals ───────────────────────────────────────────
    proj_p25 = np.percentile(final_scores, 25, axis=0)   # optimistic
    proj_p75 = np.percentile(final_scores, 75, axis=0)   # pessimistic
    proj_p90 = np.percentile(final_scores, 90, axis=0)   # worst case

    # ── Per-player rank in each simulation (0 = winner) ──────────────────────
    sim_ranks = np.argsort(np.argsort(final_scores, axis=1), axis=1)

    # ── Collapse probability: for the current leader, P(finish outside top 3)
    #    For all others: stored as their win probability (= comeback prob)  ───
    leader_idx = int(np.argmin(current_scores))
    collapse_prob = np.array([
        float((sim_ranks[:, i] >= 3).mean()) if i == leader_idx
        else float(win_counts[i] / n_sims)
        for i in range(n)
    ])

    # ── Win-scenario avg remaining score/round ────────────────────────────────
    # In simulations where player i wins, what do they avg per remaining round?
    win_scenarios_avg = np.full(n, np.nan)
    if rounds_left > 0 and remaining_scores is not None:
        for i in range(n):
            mask = (winners == i)
            if mask.sum() >= 20:
                pr_left = float(player_rounds_left[i]) if player_rounds_left[i] > 0 else 1.0
                win_scenarios_avg[i] = float(
                    remaining_scores[mask, i].mean() / pr_left
                )

    result = players[["player_name", "current_score"]].copy()
    result["mc_win_prob"]                = win_counts   / n_sims
    result["mc_top5_prob"]               = top5_counts  / n_sims
    result["mc_top10_prob"]              = top10_counts / n_sims
    result["mc_projected_total"]         = np.round(np.median(final_scores, axis=0), 1)
    result["mc_proj_p25"]                = np.round(proj_p25, 1)
    result["mc_proj_p75"]                = np.round(proj_p75, 1)
    result["mc_proj_p90"]                = np.round(proj_p90, 1)
    result["mc_collapse_prob"]           = collapse_prob
    result["mc_win_scenario_score"]      = win_scenarios_avg
    result["strokes_back"]               = result["current_score"] - result["current_score"].min()
    result["expected_score_per_round"]   = [
        float(player_dists.loc[n, "expected_score_per_round"])
        if n in player_dists.index else DEFAULT_ROUND_MEAN
        for n in result["player_name"]
    ]
    result["inweek_mean"] = [
        float(player_dists.loc[n, "expected_current_rest"])
        if n in player_dists.index else DEFAULT_ROUND_MEAN
        for n in result["player_name"]
    ]

    return result.sort_values("mc_win_prob", ascending=False).reset_index(drop=True)

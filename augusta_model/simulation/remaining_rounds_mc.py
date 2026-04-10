"""
Remaining-rounds Monte Carlo simulator for live tournament use.

For each player, simulates the remaining rounds using:
  1. Player-specific scoring distribution (skill-adjusted via SG and pre-tournament model)
  2. Historical Augusta field distribution (mean +0.5, std 3.0 per round)
  3. Field correlation (when conditions are hard, everyone scores higher)

Win probability = fraction of simulations where player finishes lowest.

This replaces the score-multiplier heuristic with a principled answer to:
"What does each player need to shoot to win, and how likely is that?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Historical Augusta scoring parameters (made-cut players, 2015-2025)
AUGUSTA_ROUND_MEAN = 0.50    # strokes over par per round (R3/R4 slightly harder)
AUGUSTA_ROUND_STD  = 3.05    # std dev per round
AUGUSTA_FIELD_CORR = 0.30    # field-wide correlation (weather / pin positions affect everyone)

N_SIMULATIONS = 20_000
SEED = 42


def _player_round_mean(
    player_name: str,
    pre_df: pd.DataFrame,
    sg_col: str = "sg_total",
) -> float:
    """
    Estimate a player's expected score vs field per round.
    Uses pre-tournament sg_total as skill signal, converted to strokes.
    Average SG total in the field ≈ 0, so sg_total directly adjusts mean.
    """
    row = pre_df[pre_df["player_name"] == player_name]
    if row.empty:
        return AUGUSTA_ROUND_MEAN  # unknown player: field average

    # model_score from predictions (0=best, 1=worst finish_pct)
    # Convert to strokes above/below field average
    # A player at model_score=0.3 (top 30%) typically shoots ~1 stroke better
    model_score = float(row["model_score"].iloc[0]) if "model_score" in row.columns else 0.5

    # sg_total_8w if available
    sg = None
    for c in ["sg_total_8w", "sg_total_3w", sg_col]:
        if c in row.columns and pd.notna(row[c].iloc[0]):
            sg = float(row[c].iloc[0])
            break

    if sg is not None:
        # SG total directly measures strokes gained vs field per round
        # Positive SG = shoots better than field (lower score)
        player_mean = AUGUSTA_ROUND_MEAN - sg
    else:
        # Fall back to model_score: 0.5 = field avg, 0.1 = top player
        deviation = (model_score - 0.5) * 6  # ±3 strokes across the range
        player_mean = AUGUSTA_ROUND_MEAN + deviation

    return player_mean


def simulate_remaining_rounds(
    live_df: pd.DataFrame,
    pre_df: pd.DataFrame,
    current_round: int,
    median_thru: float,
    n_sims: int = N_SIMULATIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of remaining rounds.

    live_df: must have player_name, current_score (cumulative to-par),
             dg_make_cut (probability of making cut, 0-1)
    pre_df:  pre-tournament predictions with model_score and sg features
    current_round: 1-4 (round currently being played)
    median_thru: median holes completed in current round

    Returns DataFrame with player_name and mc_win_prob, mc_top5_prob, mc_top10_prob.
    """
    rng = np.random.default_rng(seed)

    # Players who are effectively in the tournament
    players = live_df.copy()
    # Weight by make-cut probability (zero-out players who won't make it)
    if "dg_make_cut" in players.columns:
        players = players[players["dg_make_cut"] > 0.05].copy()

    n_players = len(players)
    if n_players == 0:
        return pd.DataFrame(columns=["player_name", "mc_win_prob", "mc_top5_prob", "mc_top10_prob"])

    # Current cumulative scores
    current_scores = players["current_score"].fillna(0).values.astype(float)

    # Rounds remaining (including partial current round)
    # Fraction of current round already played
    current_round_played = median_thru / 18.0
    current_round_remaining = 1.0 - current_round_played
    full_rounds_remaining = max(0, 4 - current_round)

    # Total remaining scoring opportunities in "round units"
    rounds_left = full_rounds_remaining + current_round_remaining

    # Player-specific mean per round
    player_means = np.array([
        _player_round_mean(name, pre_df)
        for name in players["player_name"]
    ])

    # Simulate: each sim generates remaining score for each player
    # Shape: (n_sims, n_players)
    if rounds_left <= 0:
        # Tournament over — just use current scores
        final_scores = np.tile(current_scores, (n_sims, 1))
    else:
        # Field-wide shock (weather, pin positions): same for all players each sim
        # Scaled by rounds remaining
        field_shocks = rng.normal(0, AUGUSTA_ROUND_STD * np.sqrt(AUGUSTA_FIELD_CORR), n_sims)
        field_component = np.outer(field_shocks, np.ones(n_players)) * np.sqrt(rounds_left)

        # Individual player variance (independent of field)
        indiv_std = AUGUSTA_ROUND_STD * np.sqrt(1 - AUGUSTA_FIELD_CORR) * np.sqrt(rounds_left)
        player_means_total = player_means * rounds_left  # scale mean by rounds left

        individual_component = rng.normal(
            player_means_total,     # (n_players,) broadcast across n_sims
            indiv_std,
            size=(n_sims, n_players)
        )

        remaining_scores = individual_component + field_component

        # Apply make-cut filter: players unlikely to make cut get a large penalty
        # so they almost never win in simulation
        if "dg_make_cut" in players.columns:
            cut_probs = players["dg_make_cut"].fillna(1.0).values
            # For each sim, each player either makes cut (no penalty) or gets +30 penalty
            cut_made = rng.random((n_sims, n_players)) < cut_probs
            remaining_scores = np.where(cut_made, remaining_scores, remaining_scores + 30)

        final_scores = current_scores + remaining_scores

    # Win = lowest final score; ties broken randomly (already handled by noise)
    winners = np.argmin(final_scores, axis=1)

    # Top 5 and top 10
    sorted_idx = np.argsort(final_scores, axis=1)
    top5_matrix  = sorted_idx[:, :5]
    top10_matrix = sorted_idx[:, :10]

    # Count wins per player
    win_counts   = np.bincount(winners, minlength=n_players)
    top5_counts  = np.zeros(n_players, dtype=int)
    top10_counts = np.zeros(n_players, dtype=int)
    for i in range(n_players):
        top5_counts[i]  = (top5_matrix  == i).sum()
        top10_counts[i] = (top10_matrix == i).sum()

    result = players[["player_name", "current_score"]].copy()
    result["mc_win_prob"]   = win_counts   / n_sims
    result["mc_top5_prob"]  = top5_counts  / n_sims
    result["mc_top10_prob"] = top10_counts / n_sims

    # Expected finishing score (median simulation)
    result["mc_projected_total"] = np.median(final_scores, axis=0)
    result["mc_projected_total"] = result["mc_projected_total"].round(1)

    # What score player needs to shoot over remaining rounds to have >50% win chance
    # = leader_score + enough to tie, roughly
    leader_score = current_scores.min()
    result["strokes_back"] = result["current_score"] - leader_score

    return result.sort_values("mc_win_prob", ascending=False).reset_index(drop=True)

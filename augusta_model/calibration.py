"""
Probability calibration module for Augusta National Model.

Fixes three critical problems:
1. S1 (tour regression) and S2 (Augusta classifier) produce different rankings,
   causing impossible results (e.g. Scheffler 0.002% win but 73.8% top-10).
2. Temperature scaling (T=2.5) compresses S2 toward 50%, inflating floor to ~17%.
3. Probabilities don't sum to physical constraints (top-10 sums to 34 instead of 10).

Solution: Platt scaling on S2, debutant adjustment, unified MC, sum normalization.

Why Platt over isotonic: isotonic regression on ~300 training rows creates a step
function with very few distinct output levels, collapsing 80+ players into 5-6 tiers.
Platt scaling (2-parameter logistic) preserves the full continuous ranking from S2
while properly shifting the distribution to match observed base rates.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Historical base rates by experience tier (from backtest_results_v7) ──
# Tier 0 (debutant): 9.7%, Tier 1 (learning): 15.1%, Tier 2 (established): 18.0%
# Tier 3 (veteran): 14.3%, Tier 4 (deep_veteran): 0% (n=1, unreliable)
# Overall base rate: 14.6%
TIER_BASE_RATES = {0: 0.097, 1: 0.151, 2: 0.180, 3: 0.143, 4: 0.143}
OVERALL_BASE_RATE = 0.146


def _to_logits(probs):
    """Convert probabilities to logits, clipping to avoid infinities."""
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def fit_platt_calibrator(s2_raw, actuals):
    """Fit Platt scaling: calibrated = sigmoid(a * logit(s2_raw) + b).

    Optimizes a, b to minimize log-loss on the training data.
    Returns (a, b) tuple.
    """
    logits = _to_logits(s2_raw)

    def neg_log_likelihood(params):
        a, b = params
        p = _sigmoid(a * logits + b)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p))

    result = minimize(neg_log_likelihood, x0=[0.5, -1.5],
                      method="Nelder-Mead", options={"maxiter": 5000})
    return tuple(result.x)


def calibrate_s2_platt(s2_raw, platt_params=None):
    """Apply Platt scaling to S2 raw probabilities.

    Args:
        s2_raw: array of S2 raw predicted probabilities
        platt_params: (a, b) tuple from fit_platt_calibrator, or None for defaults

    Returns:
        Calibrated probabilities (continuous, preserves full ordering)
    """
    logits = _to_logits(np.asarray(s2_raw, dtype=float))

    if platt_params is not None:
        a, b = platt_params
    else:
        # Defaults based on backtest analysis:
        # S2 raw [0.02, 0.98] → actual rates [~5%, ~29%]
        # Need to compress toward base rate (14.6%) while preserving spread
        a, b = 0.45, -1.75

    return _sigmoid(a * logits + b)


def apply_debutant_adjustment(probs, experience_tiers):
    """Shrink debutant probabilities toward the debutant base rate.

    For debutants (tier 0), we have no Augusta-specific signal, so the S2 model
    is relying entirely on tour form. Historical data shows debutants make top-10
    at 9.7% vs 16.5% for experienced players — a ~40% penalty.

    Uses Bayesian shrinkage: blend model prediction with tier-specific prior.
    Shrinkage is stronger for tier 0 (no Augusta data) and weaker for higher tiers.
    """
    tiers = np.asarray(experience_tiers, dtype=float)
    adjusted = np.array(probs, dtype=float)

    # Shrinkage weights: how much to trust the prior vs the model
    # Tier 0: 35% prior, 65% model (no Augusta features to learn from)
    # Tier 1: 12% prior, 88% model (some data)
    # Tier 2+: 0% prior, 100% model (enough data, trust model)
    shrinkage = np.where(tiers == 0, 0.35,
                np.where(tiers == 1, 0.12,
                         0.0))

    tier_priors = np.array([TIER_BASE_RATES.get(int(t), OVERALL_BASE_RATE) for t in tiers])
    adjusted = (1 - shrinkage) * probs + shrinkage * tier_priors

    return adjusted


def normalize_to_sum(probs, target_sum):
    """Scale probabilities so they sum to a target (e.g., 10 for top-10).

    Preserves relative ordering. Only used as a safety net when MC isn't
    used directly (MC already produces correct sums by construction).
    """
    probs = np.asarray(probs, dtype=float)
    current_sum = probs.sum()
    if current_sum <= 0:
        return np.full_like(probs, target_sum / len(probs))
    scaled = probs * (target_sum / current_sum)
    return np.clip(scaled, 0.001, 0.95)


def run_unified_mc(skill_scores, n_sims=50000, noise_std=0.10,
                   target_pred_std=0.14, seed=42):
    """Monte Carlo simulation using a single consistent skill ranking.

    Args:
        skill_scores: array where HIGHER = BETTER player (e.g., calibrated S2 top-10 prob).
                      Will be z-scored internally.
        n_sims: number of simulations
        noise_std: per-round noise (controls upset frequency)
        target_pred_std: rescaled z-score spread (controls skill differentiation)
        seed: random seed

    Returns:
        dict with win/top5/top10/top20/make_cut probabilities (sums to 1/5/10/20/~45)
    """
    rng = np.random.RandomState(seed)
    n = len(skill_scores)
    scores = np.asarray(skill_scores, dtype=float)

    mu, s = scores.mean(), scores.std()
    if s > 0:
        z = (scores - mu) / s * target_pred_std
    else:
        z = np.zeros(n)

    # Negate so lower z = better (rank 1 = best)
    z = -z

    wins = np.zeros(n)
    top5 = np.zeros(n)
    top10 = np.zeros(n)
    top20 = np.zeros(n)
    make_cut = np.zeros(n)
    cut_line = int(n * 0.55)  # ~top 55% make the cut at Augusta (incl ties)

    for _ in range(n_sims):
        sim = z + rng.normal(0, noise_std, n)
        ranks = sim.argsort().argsort() + 1
        wins += (ranks == 1)
        top5 += (ranks <= 5)
        top10 += (ranks <= 10)
        top20 += (ranks <= 20)
        make_cut += (ranks <= cut_line)

    return {
        "win": wins / n_sims,
        "top5": top5 / n_sims,
        "top10": top10 / n_sims,
        "top20": top20 / n_sims,
        "make_cut": make_cut / n_sims,
    }


def enforce_monotonic(df, win_col="win_prob", top5_col="top5_prob",
                      top10_col="top10_prob", top20_col="top20_prob",
                      cut_col="make_cut_prob"):
    """Enforce win <= top5 <= top10 <= top20 <= make_cut."""
    df = df.copy()
    df[top5_col] = np.maximum(df[top5_col], df[win_col])
    df[top10_col] = np.maximum(df[top10_col], df[top5_col])
    df[top20_col] = np.maximum(df[top20_col], df[top10_col])
    df[cut_col] = np.maximum(df[cut_col], df[top20_col])
    return df


def calibrate_full_pipeline(s2_raw, experience_tiers, platt_params=None,
                            n_sims=50000, noise_std=0.16,
                            target_pred_std=0.06, seed=42, field_size=None):
    """Full calibration pipeline: S2 → Platt → debutant adj → unified MC.

    All markets (win/top5/top10/top20) flow from the same S2-based skill ranking
    through a single MC simulation. This ensures:
    - Cross-market consistency (win <= top5 <= top10 <= top20 by construction)
    - Correct sums (MC produces exactly 1 winner, 5 top-5, etc. per sim)
    - No need for normalization or clipping (MC handles it naturally)

    MC parameters tuned for golf:
    - noise_std=0.16, target_pred_std=0.06 → ratio 0.375
    - Produces ~10-15% max win probability in 82-player field
    - Higher noise = more upsets, flatter distribution (realistic for golf)

    Args:
        s2_raw: array of S2 raw probabilities
        experience_tiers: array of experience tier values (0-4)
        platt_params: (a, b) from fit_platt_calibrator, or None for defaults
        n_sims, noise_std, target_pred_std, seed: MC parameters
        field_size: number of players (defaults to len(s2_raw))

    Returns:
        DataFrame with calibrated probabilities for all markets
    """
    n = len(s2_raw)
    if field_size is None:
        field_size = n

    # Step 1: Platt calibration (continuous, preserves full ordering)
    cal_probs = calibrate_s2_platt(np.asarray(s2_raw), platt_params)

    # Step 2: Debutant adjustment (shrink toward tier prior)
    adj_probs = apply_debutant_adjustment(cal_probs, experience_tiers)

    # Step 3: Unified MC — all markets from one simulation
    # MC naturally produces correct sums: win→1.0, top5→5.0, top10→10.0, top20→20.0
    mc = run_unified_mc(adj_probs, n_sims=n_sims, noise_std=noise_std,
                        target_pred_std=target_pred_std, seed=seed)

    result = pd.DataFrame({
        "win_prob": mc["win"],
        "top5_prob": mc["top5"],
        "top10_prob": mc["top10"],
        "top20_prob": mc["top20"],
        "make_cut_prob": mc["make_cut"],
        "s2_calibrated": cal_probs,
        "s2_adjusted": adj_probs,
    })

    # Monotonic is guaranteed by MC construction, but enforce as safety net
    result = enforce_monotonic(result)

    return result


def fit_and_calibrate_backtest(backtest_df, s2_col="s2_top10", actual_col="actual_top10",
                               tier_col="augusta_experience_tier", season_col="season"):
    """Walk-forward Platt calibration on backtest data.

    For each backtest year, fit Platt scaling on prior years and calibrate current year.
    Returns the backtest_df with new calibrated columns added, plus calibrator params.
    """
    df = backtest_df.copy()
    df["s2_platt"] = np.nan
    df["s2_adjusted"] = np.nan

    years = sorted(df[season_col].unique())
    platt_params_by_year = {}

    for year in years:
        train = df[df[season_col] < year]
        test_mask = df[season_col] == year

        # Need enough training data with variance in S2 outputs
        train_valid = train[train[s2_col] > 0.001]  # exclude cold-start constant preds
        if len(train_valid) < 30 or train_valid[actual_col].sum() < 3:
            # Not enough data — use raw S2 (better than nothing)
            df.loc[test_mask, "s2_platt"] = df.loc[test_mask, s2_col]
            df.loc[test_mask, "s2_adjusted"] = df.loc[test_mask, s2_col]
            continue

        # Fit Platt scaling on prior years
        params = fit_platt_calibrator(
            train_valid[s2_col].values,
            train_valid[actual_col].values
        )
        platt_params_by_year[year] = params

        # Calibrate current year
        cal = calibrate_s2_platt(df.loc[test_mask, s2_col].values, params)
        tiers = df.loc[test_mask, tier_col].values
        adj = apply_debutant_adjustment(cal, tiers)

        df.loc[test_mask, "s2_platt"] = cal
        df.loc[test_mask, "s2_adjusted"] = adj

    return df, platt_params_by_year

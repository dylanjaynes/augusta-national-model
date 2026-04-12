"""
PrizePicks Projections — Masters R4 2026
=========================================
Computes R4 projections for every player with a PrizePicks line across
all 10 prop types.  Writes output to data/live/prizepicks_projections.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
LIVE = ROOT / "data" / "live"
LIVE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PrizePicks total-strokes lines (R4 2026 Masters)
# ─────────────────────────────────────────────────────────────────────────────
STROKES_LINES: dict[str, float] = {
    "Rory McIlroy":       70.0,
    "Cameron Young":      70.5,
    "Shane Lowry":        71.5,
    "Sam Burns":          71.5,
    "Patrick Reed":       70.5,
    "Collin Morikawa":    70.5,
    "Jon Rahm":           69.5,
    "Ludvig Aberg":       70.0,
    "Tommy Fleetwood":    70.5,
    "Tyrrell Hatton":     71.0,
    "Brian Campbell":     73.0,
    "Hideki Matsuyama":   70.5,
    "Jordan Spieth":      70.5,
    "Brian Harman":       71.5,
    "Sungjae Im":         71.5,
    "Kurt Kitayama":      71.5,
    "Keegan Bradley":     71.5,
    "Justin Thomas":      71.0,
    "Ben Griffin":        71.5,
    "Jake Knapp":         71.0,
}

ALL_PP_PLAYERS = list(STROKES_LINES.keys())

# Augusta par layout
HOLE_PARS: dict[int, int] = {
    1: 4, 2: 5, 3: 4, 4: 3, 5: 4, 6: 3,  7: 4,
    8: 5, 9: 4, 10: 4, 11: 4, 12: 3, 13: 5,
    14: 4, 15: 5, 16: 3, 17: 4, 18: 4,
}
FRONT_HOLES  = list(range(1, 10))
BACK_HOLES   = list(range(10, 19))
PAR_3_HOLES  = [h for h, p in HOLE_PARS.items() if p == 3]
PAR_4_HOLES  = [h for h, p in HOLE_PARS.items() if p == 4]
PAR_5_HOLES  = [h for h, p in HOLE_PARS.items() if p == 5]
DRIVING_HOLES = PAR_4_HOLES + PAR_5_HOLES  # holes with fairways (14 total)
COURSE_PAR    = sum(HOLE_PARS.values())     # 72

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def year_weight(year: int, current_year: int = 2026) -> float:
    """Exponential recency weight: ~3× for last 2 years, decaying further back."""
    age = current_year - year
    return np.exp(-0.35 * age)


def over_prob(mu: float, sigma: float, line: float) -> float:
    """P(X > line) under Normal(mu, sigma)."""
    if sigma <= 0:
        return float(mu > line)
    return float(1 - stats.norm.cdf(line, loc=mu, scale=sigma))


def edge(mu: float, sigma: float, line: float) -> float:
    """Over edge vs 50/50 fair value."""
    return over_prob(mu, sigma, line) - 0.50


def round_to_half(x: float) -> float:
    return round(x * 2) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    hbh = pd.read_parquet(PROCESSED / "masters_hole_by_hole.parquet")

    live_p = LIVE / "live_predictions_latest.csv"
    if live_p.exists():
        live = pd.read_csv(live_p)
    else:
        # fallback: empty
        live = pd.DataFrame(columns=["player_name", "expected_score_per_round"])

    return hbh, live


# ─────────────────────────────────────────────────────────────────────────────
# Historical per-round summaries for each player
# ─────────────────────────────────────────────────────────────────────────────

def build_player_round_stats(hbh: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse hole-by-hole → one row per (player, year, round) with all
    the aggregated stats we need.
    """
    grp = hbh.groupby(["player_name", "year", "round"], as_index=False)

    def safe_count(series, values):
        if isinstance(values, str):
            return (series == values).sum()
        return series.isin(values).sum()

    agg_rows = []
    for (player, year, rnd), g in hbh.groupby(["player_name", "year", "round"]):
        if len(g) < 18:            # incomplete round — skip
            continue

        total_strokes = g["score"].sum()
        birdies  = safe_count(g["score_type"], ["BIRDIE", "EAGLE"])
        pars     = safe_count(g["score_type"], "PAR")
        bogeys   = safe_count(g["score_type"], ["BOGEY", "DOUBLE_BOGEY", "TRIPLE_BOGEY", "OTHER"])
        eagles   = safe_count(g["score_type"], "EAGLE")

        back9_score = g[g["hole_number"].isin(BACK_HOLES)]["score"].sum()

        # Specific hole scores
        def hole_score(h: int) -> float | None:
            s = g[g["hole_number"] == h]["score"]
            return float(s.iloc[0]) if len(s) > 0 else None

        h2  = hole_score(2)
        h8  = hole_score(8)
        h15 = hole_score(15)

        # Estimate GIR from scoring patterns
        gir_est = 0.0
        for _, row in g.iterrows():
            rel = row["score_to_par"]
            par = row["par"]
            if par == 3:
                if rel <= 0:
                    gir_est += 0.92
                elif rel == 1:
                    gir_est += 0.30
                else:
                    gir_est += 0.05
            elif par == 4:
                if rel <= -1:
                    gir_est += 0.95
                elif rel == 0:
                    gir_est += 0.80
                elif rel == 1:
                    gir_est += 0.35
                else:
                    gir_est += 0.05
            else:  # par 5
                if rel <= -1:
                    gir_est += 0.90
                elif rel == 0:
                    gir_est += 0.85
                elif rel == 1:
                    gir_est += 0.40
                else:
                    gir_est += 0.05

        # Estimate FIR on driving holes only (par 4+5)
        fir_est = 0.0
        drive_g = g[g["hole_number"].isin(DRIVING_HOLES)]
        for _, row in drive_g.iterrows():
            rel = row["score_to_par"]
            if rel <= -1:
                fir_est += 0.75
            elif rel == 0:
                fir_est += 0.55
            elif rel == 1:
                fir_est += 0.35
            else:
                fir_est += 0.15

        agg_rows.append({
            "player_name":   player,
            "year":          year,
            "round":         rnd,
            "weight":        year_weight(year),
            "total_strokes": total_strokes,
            "birdies":       birdies,
            "pars":          pars,
            "bogeys":        bogeys,
            "eagles":        eagles,
            "back9_score":   back9_score,
            "hole2_score":   h2,
            "hole8_score":   h8,
            "hole15_score":  h15,
            "gir_est":       gir_est,
            "fir_est":       fir_est,
        })

    return pd.DataFrame(agg_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted historical statistics for a single player
# ─────────────────────────────────────────────────────────────────────────────

def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    mu = float(np.dot(w, values))
    var = float(np.dot(w, (values - mu) ** 2))
    sigma = float(np.sqrt(var)) if var > 0 else 1e-4
    return mu, sigma


def player_hist_stats(round_stats: pd.DataFrame, player: str) -> dict:
    pdf = round_stats[round_stats["player_name"] == player].copy()
    n = len(pdf)

    def ws(col: str) -> tuple[float, float]:
        mask = pdf[col].notna()
        if mask.sum() == 0:
            return np.nan, np.nan
        return weighted_mean_std(pdf.loc[mask, col].values, pdf.loc[mask, "weight"].values)

    # Use sample std with a minimum floor to avoid collapse on tiny samples
    def floored_std(s: float, floor: float) -> float:
        return max(s, floor) if not np.isnan(s) else floor

    strokes_mu, strokes_sig = ws("total_strokes")
    birdies_mu, birdies_sig = ws("birdies")
    pars_mu,    pars_sig    = ws("pars")
    bogeys_mu,  bogeys_sig  = ws("bogeys")
    back9_mu,   back9_sig   = ws("back9_score")
    h2_mu,      h2_sig      = ws("hole2_score")
    h8_mu,      h8_sig      = ws("hole8_score")
    h15_mu,     h15_sig     = ws("hole15_score")
    gir_mu,     gir_sig     = ws("gir_est")
    fir_mu,     fir_sig     = ws("fir_est")

    return {
        "n_rounds":      n,
        "strokes_mu":    strokes_mu,   "strokes_sig":    floored_std(strokes_sig, 2.5),
        "birdies_mu":    birdies_mu,   "birdies_sig":    floored_std(birdies_sig, 1.2),
        "pars_mu":       pars_mu,      "pars_sig":       floored_std(pars_sig, 1.5),
        "bogeys_mu":     bogeys_mu,    "bogeys_sig":     floored_std(bogeys_sig, 1.3),
        "back9_mu":      back9_mu,     "back9_sig":      floored_std(back9_sig, 2.2),
        "h2_mu":         h2_mu,        "h2_sig":         floored_std(h2_sig, 0.7),
        "h8_mu":         h8_mu,        "h8_sig":         floored_std(h8_sig, 0.7),
        "h15_mu":        h15_mu,       "h15_sig":        floored_std(h15_sig, 0.7),
        "gir_mu":        gir_mu,       "gir_sig":        floored_std(gir_sig, 1.5),
        "fir_mu":        fir_mu,       "fir_sig":        floored_std(fir_sig, 1.8),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Field-wide fallback averages (debutants / no historical data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_field_averages(round_stats: pd.DataFrame) -> dict:
    """Compute field-wide averages as fallback for players with no history."""
    recent = round_stats[round_stats["year"] >= 2021]
    if len(recent) == 0:
        recent = round_stats

    def fa(col: str) -> float:
        return float(recent[col].mean()) if col in recent.columns else np.nan

    return {
        "strokes_mu": fa("total_strokes"), "strokes_sig": 3.5,
        "birdies_mu": fa("birdies"),       "birdies_sig": 1.5,
        "pars_mu":    fa("pars"),          "pars_sig":    1.8,
        "bogeys_mu":  fa("bogeys"),        "bogeys_sig":  1.5,
        "back9_mu":   fa("back9_score"),   "back9_sig":   2.5,
        "h2_mu":      fa("hole2_score"),   "h2_sig":      0.8,
        "h8_mu":      fa("hole8_score"),   "h8_sig":      0.8,
        "h15_mu":     fa("hole15_score"),  "h15_sig":     0.8,
        "gir_mu":     fa("gir_est"),       "gir_sig":     1.8,
        "fir_mu":     fa("fir_est"),       "fir_sig":     2.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MC adjustment for total strokes
# ─────────────────────────────────────────────────────────────────────────────

def get_mc_r4_strokes(live: pd.DataFrame, player: str, field_avg_strokes: float) -> tuple[float, float]:
    """
    Return (mc_mu, mc_sigma) for R4 total strokes from the live MC model.
    mc_mu  = 72 + expected_score_per_round
    mc_sigma estimated from mc_proj_p25/p75 if available, else 3.0.
    """
    row = live[live["player_name"] == player]
    if row.empty or "expected_score_per_round" not in row.columns:
        return field_avg_strokes, 3.2

    esr = float(row["expected_score_per_round"].iloc[0])
    if np.isnan(esr):
        return field_avg_strokes, 3.2

    mc_mu = COURSE_PAR + esr

    # Estimate sigma from percentile range if available
    sigma = 3.0
    if "mc_proj_p25" in row.columns and "mc_proj_p75" in row.columns:
        p25 = row["mc_proj_p25"].iloc[0]
        p75 = row["mc_proj_p75"].iloc[0]
        total_score = row["cumulative_score_to_par"].iloc[0] if "cumulative_score_to_par" in row.columns else 0
        if not (np.isnan(p25) or np.isnan(p75)):
            # These are projected totals. R4 sigma ≈ half the interquartile range / 0.674 (IQR / 1.35)
            iqr_total = float(p75 - p25)
            sigma = max(iqr_total / 1.35 / np.sqrt(2), 2.0)  # spread across remaining rounds

    return mc_mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Blend historical + MC for total strokes
# ─────────────────────────────────────────────────────────────────────────────

def blend_strokes(hist: dict, mc_mu: float, mc_sig: float) -> tuple[float, float]:
    """
    When we have ≥4 historical rounds, blend 60% MC + 40% historical.
    With fewer rounds, weight toward MC.
    """
    n = hist["n_rounds"]
    if np.isnan(hist["strokes_mu"]) or n == 0:
        return mc_mu, mc_sig

    # Precision-weighted blend
    hist_weight = min(n / 10.0, 1.0) * 0.4   # max 40% historical
    mc_weight   = 1 - hist_weight

    blended_mu  = mc_weight * mc_mu  + hist_weight * hist["strokes_mu"]
    blended_sig = mc_weight * mc_sig + hist_weight * hist["strokes_sig"]
    return blended_mu, blended_sig


# ─────────────────────────────────────────────────────────────────────────────
# Scale historical props relative to expected strokes
# ─────────────────────────────────────────────────────────────────────────────

def scale_prop(hist_mu: float, hist_strokes_mu: float, mc_strokes_mu: float,
               hist_val: float, hist_sig: float) -> tuple[float, float]:
    """
    Adjust a scoring-derived prop (birdies, pars, etc.) based on the ratio of
    MC expected score to historical average score. If MC says player will score
    better than their Augusta historical average, nudge birdies up, bogeys down.
    """
    if np.isnan(hist_mu) or np.isnan(hist_strokes_mu) or hist_strokes_mu == 0:
        return hist_val, hist_sig

    delta_strokes = mc_strokes_mu - hist_strokes_mu   # positive = worse than history
    # Each stroke difference ≈ 0.5 adjustment to birdies/bogeys, 0.3 for pars
    scale_factor = delta_strokes * 0.5
    adj = hist_mu - scale_factor     # better MC score → more birdies / fewer bogeys
    return adj, hist_sig


# ─────────────────────────────────────────────────────────────────────────────
# Main projection builder
# ─────────────────────────────────────────────────────────────────────────────

def build_projections(
    round_stats: pd.DataFrame,
    live: pd.DataFrame,
    field_avgs: dict,
) -> pd.DataFrame:

    rows = []
    for player, strokes_line in STROKES_LINES.items():
        hist = player_hist_stats(round_stats, player)

        # Fill NaN stats with field averages
        for key, fa_val in field_avgs.items():
            if np.isnan(hist.get(key, np.nan)):
                hist[key] = fa_val

        # MC total-strokes projection
        mc_mu, mc_sig = get_mc_r4_strokes(live, player, field_avgs["strokes_mu"])

        # Blended total strokes
        strokes_mu, strokes_sig = blend_strokes(hist, mc_mu, mc_sig)

        # Scale historical props using expected strokes
        birdies_mu, birdies_sig = scale_prop(
            hist["birdies_mu"], hist["strokes_mu"], strokes_mu,
            hist["birdies_mu"], hist["birdies_sig"]
        )
        pars_mu, pars_sig = scale_prop(
            hist["pars_mu"], hist["strokes_mu"], strokes_mu,
            hist["pars_mu"], hist["pars_sig"]
        )
        bogeys_mu, bogeys_sig = scale_prop(
            hist["bogeys_mu"], hist["strokes_mu"], strokes_mu,
            hist["bogeys_mu"], hist["bogeys_sig"]
        )
        back9_mu,  back9_sig  = hist["back9_mu"],  hist["back9_sig"]
        h2_mu,     h2_sig     = hist["h2_mu"],     hist["h2_sig"]
        h8_mu,     h8_sig     = hist["h8_mu"],     hist["h8_sig"]
        h15_mu,    h15_sig    = hist["h15_mu"],    hist["h15_sig"]
        gir_mu,    gir_sig    = hist["gir_mu"],    hist["gir_sig"]
        fir_mu,    fir_sig    = hist["fir_mu"],    hist["fir_sig"]

        # Adjust back nine by proportion of overall expected score improvement
        if not np.isnan(hist["strokes_mu"]):
            back9_adjust = (strokes_mu - hist["strokes_mu"]) * (9 / 18)
            back9_mu += back9_adjust

        # ── Estimated PP lines for non-strokes props ───────────────────────
        # Round to nearest 0.5 to mimic typical PP line setting
        birdies_line = round_to_half(birdies_mu)
        pars_line    = round_to_half(pars_mu)
        bogeys_line  = round_to_half(bogeys_mu)
        back9_line   = round_to_half(back9_mu)
        h2_line      = round_to_half(h2_mu)
        h8_line      = round_to_half(h8_mu)
        h15_line     = round_to_half(h15_mu)
        gir_line     = round_to_half(gir_mu)
        fir_line     = round_to_half(fir_mu)

        # ── Over probabilities & edges ─────────────────────────────────────
        def op(mu, sig, line): return over_prob(mu, sig, line)
        def ed(mu, sig, line): return edge(mu, sig, line)

        rows.append({
            "player_name":         player,
            "n_augusta_rounds":    hist["n_rounds"],
            "mc_r4_strokes":       round(mc_mu, 2),
            # ── Total Strokes ──────────────────────────────────────────────
            "strokes_proj":        round(strokes_mu, 2),
            "strokes_sig":         round(strokes_sig, 2),
            "strokes_line":        strokes_line,
            "strokes_over_pct":    round(op(strokes_mu, strokes_sig, strokes_line), 4),
            "strokes_under_pct":   round(1 - op(strokes_mu, strokes_sig, strokes_line), 4),
            "strokes_edge":        round(ed(strokes_mu, strokes_sig, strokes_line), 4),
            "strokes_rec":         "UNDER" if strokes_mu < strokes_line - 0.1 else ("OVER" if strokes_mu > strokes_line + 0.1 else "PASS"),
            # ── Birdies or Better ──────────────────────────────────────────
            "birdies_proj":        round(birdies_mu, 2),
            "birdies_sig":         round(birdies_sig, 2),
            "birdies_line":        birdies_line,
            "birdies_over_pct":    round(op(birdies_mu, birdies_sig, birdies_line), 4),
            "birdies_under_pct":   round(1 - op(birdies_mu, birdies_sig, birdies_line), 4),
            "birdies_edge":        round(ed(birdies_mu, birdies_sig, birdies_line), 4),
            # ── Pars ───────────────────────────────────────────────────────
            "pars_proj":           round(pars_mu, 2),
            "pars_sig":            round(pars_sig, 2),
            "pars_line":           pars_line,
            "pars_over_pct":       round(op(pars_mu, pars_sig, pars_line), 4),
            "pars_under_pct":      round(1 - op(pars_mu, pars_sig, pars_line), 4),
            "pars_edge":           round(ed(pars_mu, pars_sig, pars_line), 4),
            # ── Bogeys or Worse ────────────────────────────────────────────
            "bogeys_proj":         round(bogeys_mu, 2),
            "bogeys_sig":          round(bogeys_sig, 2),
            "bogeys_line":         bogeys_line,
            "bogeys_over_pct":     round(op(bogeys_mu, bogeys_sig, bogeys_line), 4),
            "bogeys_under_pct":    round(1 - op(bogeys_mu, bogeys_sig, bogeys_line), 4),
            "bogeys_edge":         round(ed(bogeys_mu, bogeys_sig, bogeys_line), 4),
            # ── Back 9 (Holes 10-18) ───────────────────────────────────────
            "back9_proj":          round(back9_mu, 2),
            "back9_sig":           round(back9_sig, 2),
            "back9_line":          back9_line,
            "back9_over_pct":      round(op(back9_mu, back9_sig, back9_line), 4),
            "back9_under_pct":     round(1 - op(back9_mu, back9_sig, back9_line), 4),
            "back9_edge":          round(ed(back9_mu, back9_sig, back9_line), 4),
            # ── Hole 2 (Par 5) ─────────────────────────────────────────────
            "h2_proj":             round(h2_mu, 2),
            "h2_sig":              round(h2_sig, 2),
            "h2_line":             h2_line,
            "h2_over_pct":         round(op(h2_mu, h2_sig, h2_line), 4),
            "h2_under_pct":        round(1 - op(h2_mu, h2_sig, h2_line), 4),
            "h2_edge":             round(ed(h2_mu, h2_sig, h2_line), 4),
            # ── Hole 8 (Par 5) ─────────────────────────────────────────────
            "h8_proj":             round(h8_mu, 2),
            "h8_sig":              round(h8_sig, 2),
            "h8_line":             h8_line,
            "h8_over_pct":         round(op(h8_mu, h8_sig, h8_line), 4),
            "h8_under_pct":        round(1 - op(h8_mu, h8_sig, h8_line), 4),
            "h8_edge":             round(ed(h8_mu, h8_sig, h8_line), 4),
            # ── Hole 15 (Par 5) ────────────────────────────────────────────
            "h15_proj":            round(h15_mu, 2),
            "h15_sig":             round(h15_sig, 2),
            "h15_line":            h15_line,
            "h15_over_pct":        round(op(h15_mu, h15_sig, h15_line), 4),
            "h15_under_pct":       round(1 - op(h15_mu, h15_sig, h15_line), 4),
            "h15_edge":            round(ed(h15_mu, h15_sig, h15_line), 4),
            # ── GIR (estimated) ────────────────────────────────────────────
            "gir_proj":            round(gir_mu, 2),
            "gir_sig":             round(gir_sig, 2),
            "gir_line":            gir_line,
            "gir_over_pct":        round(op(gir_mu, gir_sig, gir_line), 4),
            "gir_under_pct":       round(1 - op(gir_mu, gir_sig, gir_line), 4),
            "gir_edge":            round(ed(gir_mu, gir_sig, gir_line), 4),
            # ── FIR (estimated) ────────────────────────────────────────────
            "fir_proj":            round(fir_mu, 2),
            "fir_sig":             round(fir_sig, 2),
            "fir_line":            fir_line,
            "fir_over_pct":        round(op(fir_mu, fir_sig, fir_line), 4),
            "fir_under_pct":       round(1 - op(fir_mu, fir_sig, fir_line), 4),
            "fir_edge":            round(ed(fir_mu, fir_sig, fir_line), 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Best picks summary
# ─────────────────────────────────────────────────────────────────────────────

PROP_EDGE_COLS = {
    "Total Strokes":   ("strokes_edge",  "strokes_proj",  "strokes_line",  "strokes_over_pct"),
    "Birdies":         ("birdies_edge",  "birdies_proj",  "birdies_line",  "birdies_over_pct"),
    "Pars":            ("pars_edge",     "pars_proj",     "pars_line",     "pars_over_pct"),
    "Bogeys":          ("bogeys_edge",   "bogeys_proj",   "bogeys_line",   "bogeys_over_pct"),
    "Back 9":          ("back9_edge",    "back9_proj",    "back9_line",    "back9_over_pct"),
    "Hole 2":          ("h2_edge",       "h2_proj",       "h2_line",       "h2_over_pct"),
    "Hole 8":          ("h8_edge",       "h8_proj",       "h8_line",       "h8_over_pct"),
    "Hole 15":         ("h15_edge",      "h15_proj",      "h15_line",      "h15_over_pct"),
    "GIR (est)":       ("gir_edge",      "gir_proj",      "gir_line",      "gir_over_pct"),
    "Fairways (est)":  ("fir_edge",      "fir_proj",      "fir_line",      "fir_over_pct"),
}


def build_best_picks(df: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for prop, (edge_col, proj_col, line_col, over_col) in PROP_EDGE_COLS.items():
        for _, row in df.iterrows():
            e = row[edge_col]
            direction = "OVER" if e >= 0 else "UNDER"
            abs_edge = abs(e)
            # For under bets, edge is negative over; flip sign for ranking
            picks.append({
                "player_name": row["player_name"],
                "prop":        prop,
                "direction":   direction,
                "proj":        row[proj_col],
                "line":        row[line_col],
                "over_pct":    row[over_col],
                "under_pct":   1 - row[over_col],
                "edge":        e,
                "abs_edge":    abs_edge,
                "pick_pct":    row[over_col] if direction == "OVER" else 1 - row[over_col],
            })

    best = (pd.DataFrame(picks)
            .sort_values("abs_edge", ascending=False)
            .head(30)
            .reset_index(drop=True))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data…")
    hbh, live = load_data()

    print("Building per-round stats from hole-by-hole data…")
    round_stats = build_player_round_stats(hbh)
    print(f"  → {len(round_stats)} complete rounds across {round_stats['player_name'].nunique()} players")

    print("Computing field averages…")
    field_avgs = compute_field_averages(round_stats)

    print("Building projections…")
    df = build_projections(round_stats, live, field_avgs)

    out_path = LIVE / "prizepicks_projections.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    best_path = LIVE / "prizepicks_best_picks.csv"
    best = build_best_picks(df)
    best.to_csv(best_path, index=False)
    print(f"Saved → {best_path}")

    # Print summary
    print("\n── Top 15 picks by edge ─────────────────────────────────────────")
    print(best[["player_name", "prop", "direction", "line", "proj", "pick_pct", "abs_edge"]]
          .head(15)
          .to_string(index=False))

    print("\n── Total Strokes projections ────────────────────────────────────")
    strokes_summary = df[["player_name", "mc_r4_strokes", "strokes_proj", "strokes_line",
                           "strokes_over_pct", "strokes_edge", "strokes_rec"]].copy()
    strokes_summary = strokes_summary.sort_values("strokes_proj")
    print(strokes_summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""
PrizePicks Projections — Masters R4 2026
Distribution-based probability visualizer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="PrizePicks — Augusta Model",
    layout="wide",
    page_icon="🎯",
)

ROOT      = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
LIVE      = ROOT / "data" / "live"

COURSE_PAR = 72

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_hbh() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "masters_hole_by_hole.parquet")


@st.cache_data(ttl=300)
def load_live() -> pd.DataFrame | None:
    p = LIVE / "live_predictions_latest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(ttl=300)
def build_round_stats(hbh: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hole-by-hole → one row per (player, year, round)."""
    rows = []
    for (player, year, rnd), g in hbh.groupby(["player_name", "year", "round"]):
        if len(g) < 18:
            continue
        total  = int(g["score"].sum())
        bird   = int(g["score_type"].isin(["BIRDIE", "EAGLE"]).sum())
        pars   = int((g["score_type"] == "PAR").sum())
        bogs   = int(g["score_type"].isin(["BOGEY", "DOUBLE_BOGEY", "TRIPLE_BOGEY", "OTHER"]).sum())
        back9  = int(g[g["hole_number"] >= 10]["score"].sum())

        def hs(h):
            s = g[g["hole_number"] == h]["score"]
            return int(s.iloc[0]) if len(s) > 0 else None

        rows.append({
            "player_name": player, "year": year, "round": rnd,
            "weight":      float(np.exp(-0.35 * (2026 - year))),
            "total":       total, "birdies": bird, "pars": pars,
            "bogeys":      bogs, "back9": back9,
            "h2": hs(2), "h8": hs(8), "h15": hs(15),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution builders
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BINS  = list(range(63, 83))    # 63 … 82
BIRDIE_BINS = list(range(0, 11))     # 0 … 10+
PAR_BINS    = list(range(5, 17))     # 5 … 16+
BOGEY_BINS  = list(range(0, 12))     # 0 … 11+
BACK9_BINS  = list(range(30, 46))    # 30 … 45


def weighted_hist(values: list, weights: list, bins: list) -> np.ndarray:
    """
    Returns probability array aligned to `bins`.
    Values outside [min(bins), max(bins)] are clipped to the nearest edge bin.
    """
    bins_arr = np.array(bins)
    vals     = np.clip(np.array(values, dtype=float), bins_arr[0], bins_arr[-1])
    wts      = np.array(weights, dtype=float)
    wts     /= wts.sum() if wts.sum() > 0 else 1.0
    probs    = np.zeros(len(bins))
    for v, w in zip(vals, wts):
        idx = int(round(v)) - bins_arr[0]
        idx = max(0, min(len(bins) - 1, idx))
        probs[idx] += w
    return probs


def smooth_probs(probs: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Apply Gaussian kernel smoothing to a probability array."""
    n    = len(probs)
    out  = np.zeros(n)
    for i in range(n):
        for j in range(n):
            out[i] += probs[j] * stats.norm.pdf(i, loc=j, scale=sigma)
    s = out.sum()
    return out / s if s > 0 else out


def mc_gaussian_probs(bins: list, mu: float, sigma: float) -> np.ndarray:
    """Normal distribution centered on mu, evaluated at integer bins."""
    bins_arr = np.array(bins, dtype=float)
    # Use normal CDF to assign prob to each integer bin
    probs = np.diff(
        stats.norm.cdf(np.append(bins_arr - 0.5, bins_arr[-1] + 0.5), mu, sigma)
    )
    return probs / probs.sum()


def blend_distributions(
    hist_probs: np.ndarray,
    mc_probs: np.ndarray,
    n_hist: int,
    min_rounds_full_hist: int = 12,
) -> np.ndarray:
    """
    Weight empirical vs MC-driven distribution by number of rounds.
    With 0 rounds → 100% MC. With 12+ rounds → ~70% empirical.
    """
    hist_w = min(n_hist / min_rounds_full_hist, 1.0) * 0.70
    mc_w   = 1 - hist_w
    blended = hist_w * hist_probs + mc_w * mc_probs
    return blended / blended.sum()


def shift_probs(probs: np.ndarray, bins: list, delta: float) -> np.ndarray:
    """
    Shift a distribution by fractional `delta` bins via linear interpolation.
    Used to centre the empirical distribution on the MC projected mean.
    """
    if abs(delta) < 0.01:
        return probs
    bins_arr = np.arange(len(probs), dtype=float)
    shift_pos = bins_arr - delta
    out = np.interp(bins_arr, shift_pos, probs, left=0.0, right=0.0)
    s = out.sum()
    return out / s if s > 0 else out


def compute_over_under(probs: np.ndarray, bins: list, line: float) -> tuple[float, float]:
    """
    Returns (over_prob, under_prob) for P(X > line) vs P(X <= line).
    Uses bin probabilities.
    """
    bins_arr = np.array(bins)
    over  = float(probs[bins_arr > line].sum())
    under = float(probs[bins_arr <= line].sum())
    total = over + under
    if total <= 0:
        return 0.5, 0.5
    return over / total, under / total


# ─────────────────────────────────────────────────────────────────────────────
# Per-player distribution cache
# ─────────────────────────────────────────────────────────────────────────────

def get_player_distributions(
    player: str,
    rs: pd.DataFrame,
    live: pd.DataFrame | None,
) -> dict:
    """
    Returns dict with distribution arrays for each prop type.
    Also returns mc_mu (MC expected R4 strokes) and n_rounds.
    """
    pdf  = rs[rs["player_name"] == player].copy()
    n    = len(pdf)

    # MC expected score
    mc_mu = float(COURSE_PAR)     # default: par
    mc_sig = 3.5
    if live is not None:
        row = live[live["player_name"] == player]
        if not row.empty and "expected_score_per_round" in row.columns:
            esr = float(row["expected_score_per_round"].iloc[0])
            if not np.isnan(esr):
                mc_mu = COURSE_PAR + esr
        if not row.empty and "mc_proj_p25" in row.columns and "mc_proj_p75" in row.columns:
            p25 = row["mc_proj_p25"].iloc[0]
            p75 = row["mc_proj_p75"].iloc[0]
            if not (np.isnan(p25) or np.isnan(p75)):
                mc_sig = max(float(p75 - p25) / 1.35 / np.sqrt(2), 2.5)

    # Field averages (fallback for debutants)
    field = rs[rs["year"] >= 2021] if len(rs[rs["year"] >= 2021]) > 100 else rs
    fa_birdies_mu = float(field["birdies"].mean())
    fa_pars_mu    = float(field["pars"].mean())
    fa_bogeys_mu  = float(field["bogeys"].mean())
    fa_back9_mu   = float(field["back9"].mean())

    # ── Total strokes ─────────────────────────────────────────────────────────
    mc_score_probs = mc_gaussian_probs(SCORE_BINS, mc_mu, mc_sig)

    if n >= 4:
        total_hist = weighted_hist(pdf["total"].tolist(), pdf["weight"].tolist(), SCORE_BINS)
        total_hist = smooth_probs(total_hist, sigma=0.8)
        hist_mean  = float(np.dot(np.array(SCORE_BINS), total_hist))
        delta      = mc_mu - hist_mean
        total_hist = shift_probs(total_hist, SCORE_BINS, delta)
        score_probs = blend_distributions(total_hist, mc_score_probs, n)
    else:
        score_probs = mc_score_probs

    # ── Birdies ───────────────────────────────────────────────────────────────
    birdie_mu_adj = fa_birdies_mu + (COURSE_PAR - mc_mu) * 0.55
    mc_bird_probs = mc_gaussian_probs(BIRDIE_BINS, birdie_mu_adj, 1.6)
    if n >= 4:
        bird_hist  = weighted_hist(pdf["birdies"].tolist(), pdf["weight"].tolist(), BIRDIE_BINS)
        bird_hist  = smooth_probs(bird_hist, sigma=0.6)
        hist_bird  = float(np.dot(np.array(BIRDIE_BINS), bird_hist))
        adj_bird   = birdie_mu_adj - hist_bird
        bird_hist  = shift_probs(bird_hist, BIRDIE_BINS, adj_bird)
        birdie_probs = blend_distributions(bird_hist, mc_bird_probs, n)
    else:
        birdie_probs = mc_bird_probs

    # ── Pars ──────────────────────────────────────────────────────────────────
    pars_mu_adj = fa_pars_mu - (COURSE_PAR - mc_mu) * 0.10
    mc_pars_probs = mc_gaussian_probs(PAR_BINS, pars_mu_adj, 2.0)
    if n >= 4:
        pars_hist  = weighted_hist(pdf["pars"].tolist(), pdf["weight"].tolist(), PAR_BINS)
        pars_hist  = smooth_probs(pars_hist, sigma=0.8)
        hist_pars  = float(np.dot(np.array(PAR_BINS), pars_hist))
        pars_hist  = shift_probs(pars_hist, PAR_BINS, pars_mu_adj - hist_pars)
        pars_probs = blend_distributions(pars_hist, mc_pars_probs, n)
    else:
        pars_probs = mc_pars_probs

    # ── Bogeys ────────────────────────────────────────────────────────────────
    bogs_mu_adj = fa_bogeys_mu - (COURSE_PAR - mc_mu) * 0.35
    bogs_mu_adj = max(bogs_mu_adj, 0.3)
    mc_bogs_probs = mc_gaussian_probs(BOGEY_BINS, bogs_mu_adj, 1.6)
    if n >= 4:
        bogs_hist  = weighted_hist(pdf["bogeys"].tolist(), pdf["weight"].tolist(), BOGEY_BINS)
        bogs_hist  = smooth_probs(bogs_hist, sigma=0.6)
        hist_bogs  = float(np.dot(np.array(BOGEY_BINS), bogs_hist))
        bogs_hist  = shift_probs(bogs_hist, BOGEY_BINS, bogs_mu_adj - hist_bogs)
        bogs_probs = blend_distributions(bogs_hist, mc_bogs_probs, n)
    else:
        bogs_probs = mc_bogs_probs

    # ── Back 9 ────────────────────────────────────────────────────────────────
    back9_mu_adj = fa_back9_mu + (mc_mu - COURSE_PAR) * 0.50
    mc_back9_probs = mc_gaussian_probs(BACK9_BINS, back9_mu_adj, 2.2)
    if n >= 4:
        b9_hist  = weighted_hist(pdf["back9"].tolist(), pdf["weight"].tolist(), BACK9_BINS)
        b9_hist  = smooth_probs(b9_hist, sigma=0.8)
        hist_b9  = float(np.dot(np.array(BACK9_BINS), b9_hist))
        b9_hist  = shift_probs(b9_hist, BACK9_BINS, back9_mu_adj - hist_b9)
        back9_probs = blend_distributions(b9_hist, mc_back9_probs, n)
    else:
        back9_probs = mc_back9_probs

    # ── Hole scores (2, 8, 15) ────────────────────────────────────────────────
    hole_dists = {}
    for col, hole_num in [("h2", 2), ("h8", 8), ("h15", 15)]:
        hdata = pd.read_parquet(PROCESSED / "masters_hole_by_hole.parquet")
        h = hdata[(hdata["player_name"] == player) & (hdata["hole_number"] == hole_num)].copy()
        h["weight"] = h["year"].apply(lambda y: np.exp(-0.35 * (2026 - y)))

        # Bucket into: eagle(≤3), birdie(4), par(5), bogey(6), double+(≥7) for par 5
        cat_bins  = ["Eagle (≤3)", "Birdie (4)", "Par (5)", "Bogey (6)", "Double+ (7+)"]
        cat_vals  = [3, 4, 5, 6, 7]

        if len(h) >= 3:
            cat_probs = np.zeros(len(cat_bins))
            for _, row in h.iterrows():
                sc = int(row["score"])
                w  = float(row["weight"])
                if sc <= 3:    cat_probs[0] += w
                elif sc == 4:  cat_probs[1] += w
                elif sc == 5:  cat_probs[2] += w
                elif sc == 6:  cat_probs[3] += w
                else:          cat_probs[4] += w
            s = cat_probs.sum()
            if s > 0:
                cat_probs /= s
        else:
            # Field average for par 5s 2015-2025
            all_h = hdata[hdata["hole_number"] == hole_num].copy()
            all_h["weight"] = all_h["year"].apply(lambda y: np.exp(-0.35 * (2026 - y)))
            cat_probs = np.zeros(len(cat_bins))
            for _, row in all_h.iterrows():
                sc = int(row["score"])
                w  = float(row["weight"])
                if sc <= 3:    cat_probs[0] += w
                elif sc == 4:  cat_probs[1] += w
                elif sc == 5:  cat_probs[2] += w
                elif sc == 6:  cat_probs[3] += w
                else:          cat_probs[4] += w
            s = cat_probs.sum()
            if s > 0:
                cat_probs /= s

        hole_dists[hole_num] = {
            "labels": cat_bins,
            "probs":  cat_probs,
            "n": len(h),
        }

    return {
        "n_rounds":     n,
        "mc_mu":        round(mc_mu, 2),
        "score_probs":  score_probs,
        "birdie_probs": birdie_probs,
        "pars_probs":   pars_probs,
        "bogs_probs":   bogs_probs,
        "back9_probs":  back9_probs,
        "hole_dists":   hole_dists,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

CHART_HEIGHT = 320
COLORS = {
    "under": "#3B82F6",   # blue
    "over":  "#EF4444",   # red
    "line":  "#F59E0B",   # amber
    "bar":   "#6366F1",   # indigo
}


def make_dist_chart(
    bins: list,
    probs: np.ndarray,
    line: float | None,
    x_title: str,
    title: str,
) -> go.Figure:
    """
    Bar chart of probability distribution.
    Bars left of line are blue (under), right are red (over).
    """
    bins_arr = np.array(bins)
    bar_colors = []
    for b in bins_arr:
        if line is None:
            bar_colors.append(COLORS["bar"])
        elif b <= line:
            bar_colors.append(COLORS["under"])
        else:
            bar_colors.append(COLORS["over"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bins_arr, y=(probs * 100),
        marker_color=bar_colors,
        hovertemplate=f"{x_title}: %{{x}}<br>Prob: %{{y:.1f}}%<extra></extra>",
    ))

    if line is not None:
        fig.add_vline(
            x=line, line_width=2, line_dash="dash",
            line_color=COLORS["line"],
            annotation_text=f"Line: {line}",
            annotation_position="top right",
            annotation_font_color=COLORS["line"],
        )
        over_p, under_p = compute_over_under(probs, bins, line)
        fig.add_annotation(
            x=0.02, y=0.96, xref="paper", yref="paper",
            text=f"<b>UNDER {under_p*100:.1f}%</b>",
            showarrow=False, font=dict(size=13, color=COLORS["under"]),
            align="left",
        )
        fig.add_annotation(
            x=0.98, y=0.96, xref="paper", yref="paper",
            text=f"<b>OVER {over_p*100:.1f}%</b>",
            showarrow=False, font=dict(size=13, color=COLORS["over"]),
            align="right",
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title=x_title,
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=CHART_HEIGHT,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        bargap=0.04,
    )
    fig.update_xaxes(dtick=1)
    return fig


def make_hole_chart(labels: list, probs: np.ndarray, hole_name: str, n: int) -> go.Figure:
    """Horizontal bar showing outcome probabilities for a specific hole."""
    colors = ["#7C3AED", "#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
    fig = go.Figure(go.Bar(
        y=labels,
        x=(probs * 100),
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    note = f" (n={n} hist. rounds)" if n >= 3 else " (field avg — no player history)"
    fig.update_layout(
        title=dict(text=f"{hole_name}{note}", font=dict(size=15)),
        xaxis_title="Probability (%)",
        template="plotly_white",
        height=260,
        margin=dict(l=10, r=60, t=50, b=30),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("PrizePicks Projections — R4 Masters 2026")
st.caption(
    "Select a player to see their R4 probability distributions for each prop type. "
    "Enter the PrizePicks line to see Over/Under probabilities instantly."
)

# Load data
hbh  = load_hbh()
live = load_live()

if hbh is None or len(hbh) == 0:
    st.error("Hole-by-hole data not found.")
    st.stop()

rs = build_round_stats(hbh)

# Player list: live predictions players first, then all HBH players
live_players = live["player_name"].tolist() if live is not None else []
all_players  = live_players + [p for p in sorted(rs["player_name"].unique()) if p not in live_players]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Player")
    player = st.selectbox("Select player", all_players, index=0)

    st.markdown("---")
    st.markdown("### PrizePicks Lines")
    st.caption("Enter the line from the PrizePicks app for each prop.")

    # Total strokes — hardcoded defaults from screenshots
    DEFAULT_STROKES: dict[str, float] = {
        "Rory McIlroy": 70.0, "Cameron Young": 70.5, "Shane Lowry": 71.5,
        "Sam Burns": 71.5, "Patrick Reed": 70.5, "Collin Morikawa": 70.5,
        "Jon Rahm": 69.5, "Ludvig Aberg": 70.0, "Tommy Fleetwood": 70.5,
        "Tyrrell Hatton": 71.0, "Brian Campbell": 73.0, "Hideki Matsuyama": 70.5,
        "Jordan Spieth": 70.5, "Brian Harman": 71.5, "Sungjae Im": 71.5,
        "Kurt Kitayama": 71.5, "Keegan Bradley": 71.5, "Justin Thomas": 71.0,
        "Ben Griffin": 71.5, "Jake Knapp": 71.0,
    }
    default_strokes = DEFAULT_STROKES.get(player, 72.0)

    line_strokes = st.number_input("Total Strokes", value=default_strokes,
                                   min_value=63.0, max_value=85.0, step=0.5,
                                   format="%.1f", key="line_strokes")
    line_birdies = st.number_input("Birdies or Better", value=3.5,
                                   min_value=0.0, max_value=10.0, step=0.5,
                                   format="%.1f", key="line_birdies")
    line_pars    = st.number_input("Pars", value=10.5,
                                   min_value=4.0, max_value=17.0, step=0.5,
                                   format="%.1f", key="line_pars")
    line_bogeys  = st.number_input("Bogeys or Worse", value=3.5,
                                   min_value=0.0, max_value=12.0, step=0.5,
                                   format="%.1f", key="line_bogeys")
    line_back9   = st.number_input("Back 9 Strokes", value=36.5,
                                   min_value=28.0, max_value=48.0, step=0.5,
                                   format="%.1f", key="line_back9")

    st.markdown("---")
    st.caption("Lines pre-filled from PrizePicks screenshots for Total Strokes. "
               "All other lines are your estimates — adjust to match the app.")

# ── Compute distributions ─────────────────────────────────────────────────────
dist = get_player_distributions(player, rs, live)
n    = dist["n_rounds"]

# ── Header metrics ────────────────────────────────────────────────────────────
mc_mu = dist["mc_mu"]
col1, col2, col3 = st.columns(3)
col1.metric("MC Projected R4", f"{mc_mu:.1f} strokes", help="Monte Carlo expected score for R4")
col2.metric("Augusta Rounds (hist.)", n)
note = "✅ PP screenshot" if player in DEFAULT_STROKES else "⚠️ manual line"
col3.metric("Strokes Line", f"{line_strokes:.1f}", delta=f"{mc_mu - line_strokes:+.1f} vs line",
            delta_color="inverse", help=note)

if n < 4:
    st.info(f"Limited Augusta history ({n} complete rounds). Distribution leans on MC model + field averages.")

st.markdown("---")

# ── Strokes distribution ──────────────────────────────────────────────────────
st.subheader("Total Strokes Distribution")
note_src = "✅ Verified from PrizePicks screenshots" if player in DEFAULT_STROKES else "Line entered manually"
st.caption(note_src)

fig_strokes = make_dist_chart(
    SCORE_BINS, dist["score_probs"], line_strokes,
    "Strokes", f"{player} — R4 Total Strokes"
)
st.plotly_chart(fig_strokes, use_container_width=True, config={"displayModeBar": False})

# Over/Under callout
ov, un = compute_over_under(dist["score_probs"], SCORE_BINS, line_strokes)
rec = "OVER" if ov > un else "UNDER"
edge = abs(max(ov, un) - 0.50)
col_a, col_b, col_c = st.columns(3)
col_a.metric(f"Under {line_strokes:.1f}", f"{un*100:.1f}%")
col_b.metric(f"Over {line_strokes:.1f}", f"{ov*100:.1f}%")
col_c.metric("Model Edge", f"{edge*100:.1f}pp", delta=rec)

st.markdown("---")

# ── Two-column layout for counting stats ──────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Birdies or Better")
    fig_bird = make_dist_chart(
        BIRDIE_BINS, dist["birdie_probs"], line_birdies,
        "Birdies", f"{player} — Birdies or Better"
    )
    st.plotly_chart(fig_bird, use_container_width=True, config={"displayModeBar": False})
    ov_b, un_b = compute_over_under(dist["birdie_probs"], BIRDIE_BINS, line_birdies)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_birdies:.1f}", f"{un_b*100:.1f}%")
    c2.metric(f"Over {line_birdies:.1f}",  f"{ov_b*100:.1f}%")

with right:
    st.subheader("Bogeys or Worse")
    fig_bogs = make_dist_chart(
        BOGEY_BINS, dist["bogs_probs"], line_bogeys,
        "Bogeys", f"{player} — Bogeys or Worse"
    )
    st.plotly_chart(fig_bogs, use_container_width=True, config={"displayModeBar": False})
    ov_bg, un_bg = compute_over_under(dist["bogs_probs"], BOGEY_BINS, line_bogeys)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_bogeys:.1f}", f"{un_bg*100:.1f}%")
    c2.metric(f"Over {line_bogeys:.1f}",  f"{ov_bg*100:.1f}%")

# ── Pars & Back 9 ─────────────────────────────────────────────────────────────
left2, right2 = st.columns(2)

with left2:
    st.subheader("Pars")
    fig_pars = make_dist_chart(
        PAR_BINS, dist["pars_probs"], line_pars,
        "Pars", f"{player} — Pars in R4"
    )
    st.plotly_chart(fig_pars, use_container_width=True, config={"displayModeBar": False})
    ov_p, un_p = compute_over_under(dist["pars_probs"], PAR_BINS, line_pars)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_pars:.1f}", f"{un_p*100:.1f}%")
    c2.metric(f"Over {line_pars:.1f}",  f"{ov_p*100:.1f}%")

with right2:
    st.subheader("Back 9 Strokes (Holes 10–18)")
    fig_b9 = make_dist_chart(
        BACK9_BINS, dist["back9_probs"], line_back9,
        "Back 9 Score", f"{player} — Back Nine"
    )
    st.plotly_chart(fig_b9, use_container_width=True, config={"displayModeBar": False})
    ov_b9, un_b9 = compute_over_under(dist["back9_probs"], BACK9_BINS, line_back9)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_back9:.1f}", f"{un_b9*100:.1f}%")
    c2.metric(f"Over {line_back9:.1f}",  f"{ov_b9*100:.1f}%")

st.markdown("---")

# ── Par 5 hole charts ──────────────────────────────────────────────────────────
st.subheader("Par 5 Hole Scoring")
st.caption("Historical scoring distribution for this player on each par 5 (no O/U line — compare to PrizePicks hole score prop).")

hole_cols = st.columns(3)
hole_labels = {2: "Hole 2 — Pink Dogwood (575 yds)",
               8: "Hole 8 — Yellow Jasmine (570 yds)",
               15: "Hole 15 — Firethorn (530 yds)"}

for col_idx, (hole_num, label) in enumerate(hole_labels.items()):
    hd = dist["hole_dists"].get(hole_num, {})
    if not hd:
        continue
    fig_h = make_hole_chart(hd["labels"], hd["probs"], label, hd["n"])
    with hole_cols[col_idx]:
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ── Summary table ─────────────────────────────────────────────────────────────
st.subheader("Summary — All Props")

def exp_val(probs, bins):
    return float(np.dot(np.array(bins), probs))

def std_val(probs, bins):
    mu = exp_val(probs, bins)
    return float(np.sqrt(np.dot((np.array(bins) - mu)**2, probs)))

lines = {
    "Total Strokes":      (line_strokes, SCORE_BINS,  dist["score_probs"]),
    "Birdies or Better":  (line_birdies, BIRDIE_BINS, dist["birdie_probs"]),
    "Pars":               (line_pars,    PAR_BINS,    dist["pars_probs"]),
    "Bogeys or Worse":    (line_bogeys,  BOGEY_BINS,  dist["bogs_probs"]),
    "Back 9 Strokes":     (line_back9,   BACK9_BINS,  dist["back9_probs"]),
}

summary_rows = []
for prop_name, (line_val, bins, probs) in lines.items():
    mu  = exp_val(probs, bins)
    sig = std_val(probs, bins)
    ov, un = compute_over_under(probs, bins, line_val)
    best_dir = "OVER" if ov > un else "UNDER"
    best_pct = max(ov, un)
    edge_pp  = best_pct - 0.50
    summary_rows.append({
        "Prop":        prop_name,
        "Line":        line_val,
        "Model Proj":  round(mu, 2),
        "Std Dev":     round(sig, 2),
        "Over%":       f"{ov*100:.1f}%",
        "Under%":      f"{un*100:.1f}%",
        "Best Play":   f"{best_dir} {best_pct*100:.1f}%",
        "Edge":        f"{edge_pp*100:+.1f}pp",
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df, hide_index=True, use_container_width=True,
    column_config={
        "Prop":       st.column_config.TextColumn("Prop"),
        "Line":       st.column_config.NumberColumn("PP Line", format="%.1f"),
        "Model Proj": st.column_config.NumberColumn("Projection", format="%.2f"),
        "Std Dev":    st.column_config.NumberColumn("Std Dev", format="%.2f"),
        "Over%":      st.column_config.TextColumn("Over%"),
        "Under%":     st.column_config.TextColumn("Under%"),
        "Best Play":  st.column_config.TextColumn("Best Play"),
        "Edge":       st.column_config.TextColumn("Edge (pp)"),
    },
)

st.caption(
    "**Notes:** Distributions blend recency-weighted Augusta historical data with MC model projections. "
    "Players with fewer than 4 Augusta rounds rely more heavily on the MC model and field averages. "
    "GIR/Fairways not shown (no direct data available for Augusta). "
    "Hole 2/8/15 charts show historical scoring breakdown — compare to PrizePicks hole score prop."
)

"""
PrizePicks Projections — Masters R4 2026
Distribution-based probability visualizer + bird's-eye overview
"""
from __future__ import annotations

import urllib.request
import json
from datetime import datetime
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

# PrizePicks total-strokes lines — verified from screenshots
DEFAULT_STROKES: dict[str, float] = {
    "Rory McIlroy":    70.0, "Cameron Young":  70.5, "Shane Lowry":    71.5,
    "Sam Burns":       71.5, "Patrick Reed":   70.5, "Collin Morikawa":70.5,
    "Jon Rahm":        69.5, "Ludvig Aberg":   70.0, "Tommy Fleetwood":70.5,
    "Tyrrell Hatton":  71.0, "Brian Campbell": 73.0, "Hideki Matsuyama":70.5,
    "Jordan Spieth":   70.5, "Brian Harman":   71.5, "Sungjae Im":     71.5,
    "Kurt Kitayama":   71.5, "Keegan Bradley": 71.5, "Justin Thomas":  71.0,
    "Ben Griffin":     71.5, "Jake Knapp":     71.0,
}
ALL_PP_PLAYERS = list(DEFAULT_STROKES.keys())

# API stat type → internal key
STAT_MAP = {
    "Strokes":               "strokes",
    "Birdies Or Better":     "birdies",
    "Pars":                  "pars",
    "Bogeys or Worse":       "bogeys",
    "Hole 10 thru 18 Shots": "back9",
    "Hole 2 Shots":          "h2",
    "Hole 8 Shots":          "h8",
    "Hole 13 Shots":         "h13",
    "Hole 15 Shots":         "h15",
    "Greens In Regulation":  "gir",
    "Fairways Hit":          "fir",
}

# ─────────────────────────────────────────────────────────────────────────────
# Live PrizePicks lines
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_pp_lines() -> tuple[dict[str, dict[str, float]], str, bool]:
    """
    Fetch live lines from the PrizePicks partner API.
    Returns:
        lines:     {player_name: {stat_key: line_score}}
        timestamp: human-readable fetch time
        live:      True if fetch succeeded, False if using fallback
    """
    url = "https://partner-api.prizepicks.com/projections?per_page=1000"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        # Fallback to hardcoded strokes lines only
        fallback = {p: {"strokes": v} for p, v in DEFAULT_STROKES.items()}
        return fallback, "API unavailable — using hardcoded fallback", False

    included  = data.get("included", [])
    player_map = {
        x["id"]: (x["attributes"].get("display_name") or x["attributes"].get("name", ""))
        for x in included if x.get("type") == "new_player"
    }

    lines: dict[str, dict[str, float]] = {}
    for proj in data.get("data", []):
        attrs = proj.get("attributes", {})
        # Filter to PGA/Augusta projections with standard odds only
        if attrs.get("odds_type") != "standard":
            continue
        desc    = attrs.get("description", "")
        game_id = attrs.get("game_id", "")
        if "Augusta" not in desc and "PGA" not in game_id:
            continue
        stat_type  = attrs.get("stat_type", "")
        stat_key   = STAT_MAP.get(stat_type)
        if stat_key is None:
            continue
        line_score = attrs.get("line_score")
        if line_score is None:
            continue
        pid   = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id", "")
        pname = player_map.get(pid, "")
        if not pname:
            continue
        if pname not in lines:
            lines[pname] = {}
        lines[pname][stat_key] = float(line_score)

    ts = datetime.now().strftime("%H:%M:%S")
    return lines, f"Lines fetched at {ts}", True


def get_strokes_line(pp_lines: dict, player: str) -> float | None:
    """Return the live strokes line for a player, or None if unavailable."""
    return pp_lines.get(player, {}).get("strokes") or DEFAULT_STROKES.get(player)


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

        def hs(h: int) -> int | None:
            s = g[g["hole_number"] == h]["score"]
            return int(s.iloc[0]) if len(s) > 0 else None

        rows.append({
            "player_name": player, "year": year, "round": rnd,
            "weight":      float(np.exp(-0.35 * (2026 - year))),
            "total":       total, "birdies": bird, "pars": pars,
            "bogeys":      bogs,  "back9": back9,
            "h2": hs(2), "h8": hs(8), "h15": hs(15),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution primitives
# ─────────────────────────────────────────────────────────────────────────────

SCORE_BINS  = list(range(63, 83))
BIRDIE_BINS = list(range(0, 11))
PAR_BINS    = list(range(5, 17))
BOGEY_BINS  = list(range(0, 12))
BACK9_BINS  = list(range(30, 46))


def weighted_hist(values: list, weights: list, bins: list) -> np.ndarray:
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
    n    = len(probs)
    out  = np.zeros(n)
    for i in range(n):
        for j in range(n):
            out[i] += probs[j] * stats.norm.pdf(i, loc=j, scale=sigma)
    s = out.sum()
    return out / s if s > 0 else out


def mc_gaussian_probs(bins: list, mu: float, sigma: float) -> np.ndarray:
    bins_arr = np.array(bins, dtype=float)
    probs    = np.diff(stats.norm.cdf(
        np.append(bins_arr - 0.5, bins_arr[-1] + 0.5), mu, sigma
    ))
    return probs / probs.sum()


def blend_distributions(
    hist_probs: np.ndarray,
    mc_probs: np.ndarray,
    n_hist: int,
    min_rounds_full_hist: int = 12,
) -> np.ndarray:
    hist_w  = min(n_hist / min_rounds_full_hist, 1.0) * 0.70
    mc_w    = 1 - hist_w
    blended = hist_w * hist_probs + mc_w * mc_probs
    return blended / blended.sum()


def shift_probs(probs: np.ndarray, bins: list, delta: float) -> np.ndarray:
    """
    Shift distribution left/right by `delta` bins.
    delta > 0 → shift right (worse scores).
    delta < 0 → shift left (better scores).
    out[j] = probs[j - delta], so mean shifts by delta.
    """
    if abs(delta) < 0.01:
        return probs
    bins_arr = np.arange(len(probs), dtype=float)
    # Evaluate probs at (bins_arr - delta): pulls values from higher indices when delta<0
    out = np.interp(bins_arr - delta, bins_arr, probs, left=0.0, right=0.0)
    s   = out.sum()
    return out / s if s > 0 else out


def compute_over_under(probs: np.ndarray, bins: list, line: float) -> tuple[float, float]:
    bins_arr = np.array(bins)
    over     = float(probs[bins_arr > line].sum())
    under    = float(probs[bins_arr <= line].sum())
    total    = over + under
    if total <= 0:
        return 0.5, 0.5
    return over / total, under / total


def percentile_score(probs: np.ndarray, bins: list, pct: float) -> float:
    """Return the score at the given percentile (0-1)."""
    bins_arr = np.array(bins, dtype=float)
    cumsum   = np.cumsum(probs)
    idx      = np.searchsorted(cumsum, pct)
    idx      = max(0, min(len(bins) - 1, idx))
    return float(bins_arr[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Per-player score distribution
# ─────────────────────────────────────────────────────────────────────────────

def _mc_params(player: str, live: pd.DataFrame | None) -> tuple[float, float]:
    """Return (mc_mu, mc_sig) from live MC data, or (72, 3.5) fallback."""
    mc_mu, mc_sig = float(COURSE_PAR), 3.5
    if live is None:
        return mc_mu, mc_sig
    row = live[live["player_name"] == player]
    if row.empty:
        return mc_mu, mc_sig
    if "expected_score_per_round" in row.columns:
        esr = float(row["expected_score_per_round"].iloc[0])
        if not np.isnan(esr):
            mc_mu = COURSE_PAR + esr
    if "mc_proj_p25" in row.columns and "mc_proj_p75" in row.columns:
        p25 = row["mc_proj_p25"].iloc[0]
        p75 = row["mc_proj_p75"].iloc[0]
        if not (np.isnan(p25) or np.isnan(p75)):
            mc_sig = max(float(p75 - p25) / 1.35 / np.sqrt(2), 2.5)
    return mc_mu, mc_sig


def build_score_probs(
    player: str,
    rs: pd.DataFrame,
    live: pd.DataFrame | None,
) -> tuple[np.ndarray, float, int]:
    """Returns (score_probs, mc_mu, n_rounds)."""
    pdf   = rs[rs["player_name"] == player]
    n     = len(pdf)
    mc_mu, mc_sig = _mc_params(player, live)

    mc_probs = mc_gaussian_probs(SCORE_BINS, mc_mu, mc_sig)
    if n >= 4:
        hist     = weighted_hist(pdf["total"].tolist(), pdf["weight"].tolist(), SCORE_BINS)
        hist     = smooth_probs(hist, sigma=0.8)
        hist_mu  = float(np.dot(np.array(SCORE_BINS), hist))
        hist     = shift_probs(hist, SCORE_BINS, mc_mu - hist_mu)
        probs    = blend_distributions(hist, mc_probs, n)
    else:
        probs    = mc_probs

    return probs, mc_mu, n


# ─────────────────────────────────────────────────────────────────────────────
# Bird's-eye overview chart
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def build_overview_data(
    rs_pkl: bytes,   # serialised via pickle for caching
    live_csv: str | None,
) -> pd.DataFrame:
    """
    Pre-compute top-2 scores, mc_mu, p25/p75 for every PP player.
    rs_pkl / live_csv are pickle/csv strings so Streamlit can hash them.
    """
    import pickle, io
    rs   = pickle.loads(rs_pkl)
    live = pd.read_csv(io.StringIO(live_csv)) if live_csv else None

    rows = []
    for player in ALL_PP_PLAYERS:
        probs, mc_mu, n = build_score_probs(player, rs, live)
        bins_arr = np.array(SCORE_BINS)

        # Top-2 most probable integer scores
        sorted_idx = np.argsort(probs)[::-1]
        top1 = int(bins_arr[sorted_idx[0]])
        top2 = int(bins_arr[sorted_idx[1]])

        # 25th / 75th percentile range
        p25 = percentile_score(probs, SCORE_BINS, 0.25)
        p75 = percentile_score(probs, SCORE_BINS, 0.75)

        rows.append({
            "player":    player,
            "mc_mu":     round(mc_mu, 2),
            "top1":      top1,
            "top2":      top2,
            "p25":       p25,
            "p75":       p75,
            "pp_line":   DEFAULT_STROKES.get(player),
            "n_rounds":  n,
        })

    df = pd.DataFrame(rows).sort_values("mc_mu").reset_index(drop=True)
    return df


def make_birds_eye_chart(ov: pd.DataFrame) -> go.Figure:
    """
    Dumbbell chart: one row per player, sorted best → worst.
    • Thin horizontal bar = p25–p75 range
    • Large dot = top-1 most likely score
    • Smaller dot = top-2 most likely score
    • Diamond = PP strokes line
    • Vertical dashed line at par (72)
    """
    n_players = len(ov)
    row_h     = 24                 # px per row
    height    = max(420, n_players * row_h + 80)

    # Y positions: best player at top → highest y index
    ov = ov.copy().reset_index(drop=True)
    ov["y"] = list(range(n_players - 1, -1, -1))

    def player_color(mc_mu: float) -> str:
        if mc_mu < 71.5:   return "#22c55e"   # green
        if mc_mu > 72.5:   return "#ef4444"   # red
        return "#94a3b8"                       # slate-gray

    fig = go.Figure()

    # ── P25-P75 range bar ──────────────────────────────────────────────────────
    seg_x, seg_y = [], []
    for _, row in ov.iterrows():
        seg_x += [row["p25"], row["p75"], None]
        seg_y += [row["y"],   row["y"],   None]

    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y, mode="lines",
        line=dict(color="rgba(148,163,184,0.45)", width=6),
        hoverinfo="skip",
        showlegend=False,
        name="50% range",
    ))

    # ── Top-2 connector (thin line between the two dots) ──────────────────────
    conn_x, conn_y = [], []
    for _, row in ov.iterrows():
        lo, hi = sorted([row["top1"], row["top2"]])
        conn_x += [lo, hi, None]
        conn_y += [row["y"], row["y"], None]

    fig.add_trace(go.Scatter(
        x=conn_x, y=conn_y, mode="lines",
        line=dict(color="rgba(99,102,241,0.6)", width=2),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Top-1 dot (large) ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ov["top1"], y=ov["y"],
        mode="markers",
        marker=dict(
            size=12,
            color=[player_color(r["mc_mu"]) for _, r in ov.iterrows()],
            line=dict(color="white", width=1.5),
        ),
        text=[
            f"<b>{r['player']}</b><br>Mode: {r['top1']}<br>2nd: {r['top2']}<br>"
            f"MC proj: {r['mc_mu']:.1f}<br>PP line: {r['pp_line']}<br>"
            f"50% range: {int(r['p25'])}–{int(r['p75'])}"
            for _, r in ov.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
        name="Most likely score",
    ))

    # ── Top-2 dot (smaller) ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ov["top2"], y=ov["y"],
        mode="markers",
        marker=dict(
            size=7,
            color=[player_color(r["mc_mu"]) for _, r in ov.iterrows()],
            opacity=0.6,
            line=dict(color="white", width=1),
        ),
        text=[f"<b>{r['player']}</b> — 2nd most likely: {r['top2']}" for _, r in ov.iterrows()],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
        name="2nd most likely",
    ))

    # ── PP line diamonds ──────────────────────────────────────────────────────
    pp_x = [r["pp_line"] for _, r in ov.iterrows() if r["pp_line"] is not None]
    pp_y = [r["y"]       for _, r in ov.iterrows() if r["pp_line"] is not None]
    pp_t = [
        f"<b>{r['player']}</b><br>PP line: {r['pp_line']}"
        for _, r in ov.iterrows() if r["pp_line"] is not None
    ]

    fig.add_trace(go.Scatter(
        x=pp_x, y=pp_y,
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=9,
            color="#f59e0b",
            line=dict(color="white", width=1),
        ),
        text=pp_t,
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
        name="PP line",
    ))

    # ── Par line ──────────────────────────────────────────────────────────────
    fig.add_vline(
        x=72, line_width=1.5, line_dash="dot",
        line_color="rgba(100,100,100,0.4)",
        annotation_text="Par",
        annotation_position="top",
        annotation_font=dict(size=10, color="gray"),
    )

    # ── Expected score marker (small vertical tick) ───────────────────────────
    fig.add_trace(go.Scatter(
        x=ov["mc_mu"], y=ov["y"],
        mode="markers",
        marker=dict(
            symbol="line-ns",
            size=10,
            color=[player_color(r["mc_mu"]) for _, r in ov.iterrows()],
            line=dict(color=[player_color(r["mc_mu"]) for _, r in ov.iterrows()], width=2),
        ),
        text=[f"<b>{r['player']}</b><br>MC expected: {r['mc_mu']:.1f}" for _, r in ov.iterrows()],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
        name="MC expected",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    player_labels = [ov.loc[ov["y"] == i, "player"].iloc[0] for i in range(n_players)]
    mc_labels     = [f"{ov.loc[ov['y']==i, 'mc_mu'].iloc[0]:.1f}" for i in range(n_players)]

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=20, t=45, b=40),
        xaxis=dict(
            title="R4 Score",
            range=[65.5, 79.5],
            dtick=1,
            tickfont=dict(size=11),
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_players)),
            ticktext=[
                f"{player_labels[i]}  <span style='color:#94a3b8;font-size:10px'>{mc_labels[i]}</span>"
                for i in range(n_players)
            ],
            tickfont=dict(size=11),
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        hovermode="y unified",
        plot_bgcolor="white",
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Full per-player distributions
# ─────────────────────────────────────────────────────────────────────────────

def get_player_distributions(
    player: str,
    rs: pd.DataFrame,
    live: pd.DataFrame | None,
) -> dict:
    pdf  = rs[rs["player_name"] == player].copy()
    n    = len(pdf)
    mc_mu, mc_sig = _mc_params(player, live)

    # Field averages
    field         = rs[rs["year"] >= 2021] if len(rs[rs["year"] >= 2021]) > 100 else rs
    fa_birdies_mu = float(field["birdies"].mean())
    fa_pars_mu    = float(field["pars"].mean())
    fa_bogeys_mu  = float(field["bogeys"].mean())
    fa_back9_mu   = float(field["back9"].mean())

    def build(col: str, bins: list, fa_mu: float, sig: float, direction: float = 1.0) -> np.ndarray:
        """Build a distribution for a count/score stat. direction=+1 if higher=worse (bogeys), -1 if higher=better (birdies)."""
        adj_mu    = fa_mu + (COURSE_PAR - mc_mu) * 0.55 * (-direction)
        adj_mu    = max(adj_mu, 0.1) if direction > 0 else adj_mu
        mc_probs  = mc_gaussian_probs(bins, adj_mu, sig)
        if n >= 4:
            hist     = weighted_hist(pdf[col].tolist(), pdf["weight"].tolist(), bins)
            hist     = smooth_probs(hist, sigma=0.6)
            hist_mu  = float(np.dot(np.array(bins), hist))
            hist     = shift_probs(hist, bins, adj_mu - hist_mu)
            return blend_distributions(hist, mc_probs, n)
        return mc_probs

    score_probs, _, _ = build_score_probs(player, rs, live)

    birdie_probs = build("birdies", BIRDIE_BINS, fa_birdies_mu, 1.6, direction=-1.0)
    pars_probs   = build("pars",    PAR_BINS,    fa_pars_mu,    2.0, direction=0.0)
    bogs_probs   = build("bogeys",  BOGEY_BINS,  fa_bogeys_mu,  1.6, direction=1.0)

    back9_mu_adj = fa_back9_mu + (mc_mu - COURSE_PAR) * 0.50
    b9_mc        = mc_gaussian_probs(BACK9_BINS, back9_mu_adj, 2.2)
    if n >= 4:
        b9h    = weighted_hist(pdf["back9"].tolist(), pdf["weight"].tolist(), BACK9_BINS)
        b9h    = smooth_probs(b9h, sigma=0.8)
        b9_hist_mu = float(np.dot(np.array(BACK9_BINS), b9h))
        b9h    = shift_probs(b9h, BACK9_BINS, back9_mu_adj - b9_hist_mu)
        back9_probs = blend_distributions(b9h, b9_mc, n)
    else:
        back9_probs = b9_mc

    # Hole-specific distributions
    hbh_full = load_hbh()
    hole_dists = {}
    for hole_num in [2, 8, 15]:
        h  = hbh_full[(hbh_full["player_name"] == player) & (hbh_full["hole_number"] == hole_num)].copy()
        h["weight"] = h["year"].apply(lambda y: np.exp(-0.35 * (2026 - y)))
        cat_bins  = ["Eagle (≤3)", "Birdie (4)", "Par (5)", "Bogey (6)", "Double+ (7+)"]
        ref       = h if len(h) >= 3 else hbh_full[hbh_full["hole_number"] == hole_num].copy()
        if len(ref) > 0 and "year" in ref.columns:
            ref["weight"] = ref["year"].apply(lambda y: np.exp(-0.35 * (2026 - y)))
        cat_probs = np.zeros(5)
        for _, row in ref.iterrows():
            sc = int(row["score"]); w = float(row["weight"])
            if sc <= 3:    cat_probs[0] += w
            elif sc == 4:  cat_probs[1] += w
            elif sc == 5:  cat_probs[2] += w
            elif sc == 6:  cat_probs[3] += w
            else:          cat_probs[4] += w
        s = cat_probs.sum()
        if s > 0:
            cat_probs /= s
        hole_dists[hole_num] = {"labels": cat_bins, "probs": cat_probs, "n": len(h)}

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
# Per-player chart builders
# ─────────────────────────────────────────────────────────────────────────────

CHART_HEIGHT = 320
COLORS = {
    "under": "#3B82F6",
    "over":  "#EF4444",
    "line":  "#F59E0B",
    "bar":   "#6366F1",
}


def make_dist_chart(
    bins: list,
    probs: np.ndarray,
    line: float | None,
    x_title: str,
    title: str,
) -> go.Figure:
    bins_arr   = np.array(bins)
    bar_colors = [
        COLORS["bar"] if line is None
        else (COLORS["under"] if b <= line else COLORS["over"])
        for b in bins_arr
    ]

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
            showarrow=False, font=dict(size=13, color=COLORS["under"]), align="left",
        )
        fig.add_annotation(
            x=0.98, y=0.96, xref="paper", yref="paper",
            text=f"<b>OVER {over_p*100:.1f}%</b>",
            showarrow=False, font=dict(size=13, color=COLORS["over"]), align="right",
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
    colors = ["#7C3AED", "#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
    note   = f" (n={n} rounds)" if n >= 3 else " (field avg)"
    fig    = go.Figure(go.Bar(
        y=labels, x=(probs * 100), orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"{hole_name}{note}", font=dict(size=14)),
        xaxis_title="Probability (%)",
        template="plotly_white",
        height=240,
        margin=dict(l=10, r=60, t=45, b=30),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

st.title("PrizePicks Projections — R4 Masters 2026")
st.caption(
    "**Overview table:** all players sorted by model projection. "
    "**Detail view:** pick a player below, enter any line, see the full probability distribution."
)

hbh  = load_hbh()
live = load_live()

if hbh is None or len(hbh) == 0:
    st.error("Hole-by-hole data not found.")
    st.stop()

rs = build_round_stats(hbh)

# Player list for detail selector
live_players = live["player_name"].tolist() if live is not None else []
all_players  = live_players + [p for p in sorted(rs["player_name"].unique()) if p not in live_players]

# ── Fetch live PP lines ────────────────────────────────────────────────────────
pp_lines, pp_ts, pp_live = fetch_pp_lines()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detail: Select Player")
    player = st.selectbox("Player", all_players, index=0)

    st.markdown("---")
    # Line status indicator
    if pp_live:
        st.success(f"Live lines — {pp_ts}")
    else:
        st.warning(pp_ts)

    st.markdown("### PrizePicks Lines")
    st.caption("Pre-filled from live API. Adjust if needed.")

    player_lines = pp_lines.get(player, {})
    default_strokes = player_lines.get("strokes") or DEFAULT_STROKES.get(player, 72.0)
    default_birdies = player_lines.get("birdies", 3.5)
    default_pars    = player_lines.get("pars", 10.5)
    default_bogeys  = player_lines.get("bogeys", 3.5)
    default_back9   = player_lines.get("back9", 36.5)

    line_strokes = st.number_input("Total Strokes", value=float(default_strokes),
                                   min_value=63.0, max_value=85.0, step=0.5, format="%.1f")
    line_birdies = st.number_input("Birdies or Better", value=float(default_birdies),
                                   min_value=0.0, max_value=10.0, step=0.5, format="%.1f")
    line_pars    = st.number_input("Pars", value=float(default_pars),
                                   min_value=4.0, max_value=17.0, step=0.5, format="%.1f")
    line_bogeys  = st.number_input("Bogeys or Worse", value=float(default_bogeys),
                                   min_value=0.0, max_value=12.0, step=0.5, format="%.1f")
    line_back9   = st.number_input("Back 9 Strokes", value=float(default_back9),
                                   min_value=28.0, max_value=48.0, step=0.5, format="%.1f")

    st.markdown("---")
    if st.button("Refresh Lines"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW TABLE
# ═══════════════════════════════════════════════════════════════════════════════

live_src = "🟢 Live from PrizePicks API" if pp_live else "🟡 Hardcoded fallback"
st.subheader(f"R4 Score Projections — All Players  {live_src}")
st.caption(
    f"{pp_ts} · Sorted by Model Avg. "
    "**#1 / #2 Score** = two most likely R4 scores. "
    "**Under%** = probability of scoring ≤ PP line. Click any column to re-sort."
)

# Serialise for caching
import pickle
rs_pkl   = pickle.dumps(rs)
live_csv = live.to_csv(index=False) if live is not None else None

ov_data = build_overview_data(rs_pkl, live_csv)

# Collect all players with live lines (from API or fallback)
all_pp = sorted(
    set(ALL_PP_PLAYERS) | set(pp_lines.keys()),
    key=lambda p: pp_lines.get(p, {}).get("strokes", DEFAULT_STROKES.get(p, 99))
)

ov_rows = []
for p in all_pp:
    live_strokes = pp_lines.get(p, {}).get("strokes") or DEFAULT_STROKES.get(p)
    if live_strokes is None:
        continue
    probs, mc_mu, _ = build_score_probs(p, rs, live)
    sorted_idx = np.argsort(probs)[::-1]
    bins_arr   = np.array(SCORE_BINS)
    top1 = int(bins_arr[sorted_idx[0]])
    top2 = int(bins_arr[sorted_idx[1]])
    p1   = round(float(probs[sorted_idx[0]]) * 100, 1)
    p2   = round(float(probs[sorted_idx[1]]) * 100, 1)
    _, under = compute_over_under(probs, SCORE_BINS, float(live_strokes))
    ov_rows.append({
        "Player":    p,
        "#1 Score":  top1,
        "#1 Prob":   p1,
        "#2 Score":  top2,
        "#2 Prob":   p2,
        "Model Avg": round(mc_mu, 1),
        "PP Line":   float(live_strokes),
        "Under%":    round(under * 100, 1),
    })

ov_table = pd.DataFrame(ov_rows).sort_values("Model Avg").reset_index(drop=True)

st.dataframe(
    ov_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Player":    st.column_config.TextColumn("Player"),
        "#1 Score":  st.column_config.NumberColumn("#1 Score", format="%d"),
        "#1 Prob":   st.column_config.NumberColumn("#1 Prob %", format="%.1f%%"),
        "#2 Score":  st.column_config.NumberColumn("#2 Score", format="%d"),
        "#2 Prob":   st.column_config.NumberColumn("#2 Prob %", format="%.1f%%"),
        "Model Avg": st.column_config.NumberColumn("Model Avg", format="%.1f"),
        "PP Line":   st.column_config.NumberColumn("PP Line", format="%.1f"),
        "Under%":    st.column_config.NumberColumn("Under% at Line", format="%.1f%%"),
    },
)

with st.expander("Score range chart (visual)"):
    fig_ov = make_birds_eye_chart(ov_data)
    st.plotly_chart(fig_ov, use_container_width=True, config={"displayModeBar": False})

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PER-PLAYER DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader(f"Detail View — {player}")
dist = get_player_distributions(player, rs, live)
n    = dist["n_rounds"]
mc_mu = dist["mc_mu"]

col1, col2, col3 = st.columns(3)
col1.metric("MC Projected R4", f"{mc_mu:.1f}", help="MC expected R4 strokes")
col2.metric("Augusta Rounds (hist.)", n)
note = "🟢 Live API" if pp_live and player in pp_lines else ("🟡 Hardcoded" if player in DEFAULT_STROKES else "⚠️ manual")
col3.metric(
    "Strokes Line", f"{line_strokes:.1f}",
    delta=f"{mc_mu - line_strokes:+.1f} vs line",
    delta_color="inverse", help=note,
)

if n < 4:
    st.info(f"Limited Augusta history ({n} complete rounds). Distribution leans on MC model + field averages.")

# ── Strokes ────────────────────────────────────────────────────────────────────
st.markdown("#### Total Strokes Distribution")
note_src = f"🟢 Live from PrizePicks API ({pp_ts})" if pp_live and player in pp_lines else "🟡 Hardcoded / manual line"
st.caption(note_src)

fig_s = make_dist_chart(SCORE_BINS, dist["score_probs"], line_strokes,
                        "Strokes", f"{player} — R4 Total Strokes")
st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

ov_s, un_s = compute_over_under(dist["score_probs"], SCORE_BINS, line_strokes)
edge_s = abs(max(ov_s, un_s) - 0.50)
ca, cb, cc = st.columns(3)
ca.metric(f"Under {line_strokes:.1f}", f"{un_s*100:.1f}%")
cb.metric(f"Over {line_strokes:.1f}",  f"{ov_s*100:.1f}%")
cc.metric("Model Edge", f"{edge_s*100:.1f}pp",
          delta="UNDER" if un_s > ov_s else "OVER")

st.markdown("---")

# ── Counting stats ─────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown("#### Birdies or Better")
    fig_b = make_dist_chart(BIRDIE_BINS, dist["birdie_probs"], line_birdies,
                            "Birdies", f"{player} — Birdies or Better")
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
    ov_b, un_b = compute_over_under(dist["birdie_probs"], BIRDIE_BINS, line_birdies)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_birdies:.1f}", f"{un_b*100:.1f}%")
    c2.metric(f"Over {line_birdies:.1f}",  f"{ov_b*100:.1f}%")

with right:
    st.markdown("#### Bogeys or Worse")
    fig_bg = make_dist_chart(BOGEY_BINS, dist["bogs_probs"], line_bogeys,
                             "Bogeys", f"{player} — Bogeys or Worse")
    st.plotly_chart(fig_bg, use_container_width=True, config={"displayModeBar": False})
    ov_bg, un_bg = compute_over_under(dist["bogs_probs"], BOGEY_BINS, line_bogeys)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_bogeys:.1f}", f"{un_bg*100:.1f}%")
    c2.metric(f"Over {line_bogeys:.1f}",  f"{ov_bg*100:.1f}%")

left2, right2 = st.columns(2)

with left2:
    st.markdown("#### Pars")
    fig_p = make_dist_chart(PAR_BINS, dist["pars_probs"], line_pars,
                            "Pars", f"{player} — Pars in R4")
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})
    ov_p, un_p = compute_over_under(dist["pars_probs"], PAR_BINS, line_pars)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_pars:.1f}", f"{un_p*100:.1f}%")
    c2.metric(f"Over {line_pars:.1f}",  f"{ov_p*100:.1f}%")

with right2:
    st.markdown("#### Back 9 Strokes (Holes 10–18)")
    fig_b9 = make_dist_chart(BACK9_BINS, dist["back9_probs"], line_back9,
                             "Back 9", f"{player} — Back Nine")
    st.plotly_chart(fig_b9, use_container_width=True, config={"displayModeBar": False})
    ov_b9, un_b9 = compute_over_under(dist["back9_probs"], BACK9_BINS, line_back9)
    c1, c2 = st.columns(2)
    c1.metric(f"Under {line_back9:.1f}", f"{un_b9*100:.1f}%")
    c2.metric(f"Over {line_back9:.1f}",  f"{ov_b9*100:.1f}%")

st.markdown("---")

# ── Par 5 hole charts ──────────────────────────────────────────────────────────
st.markdown("#### Par 5 Hole Scoring")
st.caption("Historical scoring distribution — compare to PrizePicks hole score prop.")

hole_cols   = st.columns(3)
hole_labels = {
    2:  "Hole 2 — Pink Dogwood (575 yds)",
    8:  "Hole 8 — Yellow Jasmine (570 yds)",
    15: "Hole 15 — Firethorn (530 yds)",
}
for col_idx, (hole_num, label) in enumerate(hole_labels.items()):
    hd = dist["hole_dists"].get(hole_num, {})
    if not hd:
        continue
    with hole_cols[col_idx]:
        st.plotly_chart(
            make_hole_chart(hd["labels"], hd["probs"], label, hd["n"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

st.markdown("---")

# ── Summary table ──────────────────────────────────────────────────────────────
st.markdown("#### Summary — All Props")

def exp_val(probs, bins): return float(np.dot(np.array(bins), probs))
def std_val(probs, bins):
    mu = exp_val(probs, bins)
    return float(np.sqrt(np.dot((np.array(bins) - mu) ** 2, probs)))

prop_map = {
    "Total Strokes":    (line_strokes, SCORE_BINS,  dist["score_probs"]),
    "Birdies or Better":(line_birdies, BIRDIE_BINS, dist["birdie_probs"]),
    "Pars":             (line_pars,    PAR_BINS,    dist["pars_probs"]),
    "Bogeys or Worse":  (line_bogeys,  BOGEY_BINS,  dist["bogs_probs"]),
    "Back 9 Strokes":   (line_back9,   BACK9_BINS,  dist["back9_probs"]),
}

summary_rows = []
for prop_name, (line_val, bins, probs) in prop_map.items():
    mu   = exp_val(probs, bins)
    sig  = std_val(probs, bins)
    ov_, un_ = compute_over_under(probs, bins, line_val)
    best_dir = "OVER" if ov_ > un_ else "UNDER"
    best_pct = max(ov_, un_)
    summary_rows.append({
        "Prop":       prop_name,
        "Line":       line_val,
        "Projection": round(mu, 2),
        "Std Dev":    round(sig, 2),
        "Over%":      f"{ov_*100:.1f}%",
        "Under%":     f"{un_*100:.1f}%",
        "Best Play":  f"{best_dir} {best_pct*100:.1f}%",
        "Edge":       f"{(best_pct - 0.50)*100:+.1f}pp",
    })

st.dataframe(
    pd.DataFrame(summary_rows),
    hide_index=True, use_container_width=True,
    column_config={
        "Prop":       st.column_config.TextColumn("Prop"),
        "Line":       st.column_config.NumberColumn("PP Line", format="%.1f"),
        "Projection": st.column_config.NumberColumn("Model Proj", format="%.2f"),
        "Std Dev":    st.column_config.NumberColumn("Std Dev", format="%.2f"),
        "Over%":      st.column_config.TextColumn("Over%"),
        "Under%":     st.column_config.TextColumn("Under%"),
        "Best Play":  st.column_config.TextColumn("Best Play"),
        "Edge":       st.column_config.TextColumn("Edge"),
    },
)

st.caption(
    "Distributions blend recency-weighted Augusta hole-by-hole history with MC model projections. "
    "Players with <4 Augusta rounds rely on MC + field averages. "
    "GIR/Fairways omitted (no direct Augusta data available)."
)

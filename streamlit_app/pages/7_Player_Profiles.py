"""
Player Profiles page — Augusta National Model (mobile-first redesign)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Player Profiles — Augusta Model",
    layout="wide",
    page_icon="🏌️",
)

# ── Minimal CSS for clean card styling ────────────────────────────────────────
st.markdown("""
<style>
.player-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.big-score {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    color: #1a1a1a;
}
.big-pos {
    font-size: 2.2rem;
    font-weight: 600;
    color: #444;
}
.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 2px;
}
.badge {
    display: inline-block;
    background: #e8f4fd;
    color: #1565c0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-right: 8px;
}
.context-line {
    color: #666;
    font-size: 0.9rem;
    margin-top: 10px;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0e0e0;
}
.callout-text {
    background: #f8f9fa;
    border-left: 3px solid #4caf50;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    margin-bottom: 8px;
}
.callout-text-bad {
    background: #fff5f5;
    border-left: 3px solid #ef5350;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    margin-bottom: 8px;
}
.sg-summary {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.9rem;
    color: #444;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
LIVE_DIR = ROOT / "data" / "live"

CHART_CONFIG = {"displayModeBar": False}


# ── Data Loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def load_live_predictions() -> pd.DataFrame | None:
    p = LIVE_DIR / "live_predictions_latest.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=300)
def load_pretournament() -> pd.DataFrame | None:
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(ttl=3600)
def load_hole_by_hole() -> pd.DataFrame | None:
    p = PROCESSED / "masters_hole_by_hole.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(ttl=3600)
def load_course_stats() -> pd.DataFrame | None:
    p = PROCESSED / "course_hole_stats.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_get(row, *cols):
    """Return first non-null value from the given column names."""
    for col in cols:
        val = row.get(col, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
    return None


def score_fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    v = int(round(float(v)))
    if v == 0:
        return "E"
    return f"{v:+d}"


def parse_finish_num(f) -> int | None:
    if pd.isna(f):
        return None
    s = str(f).upper().replace("T", "").strip()
    for mc_str in ("CUT", "WD", "DQ", "MC", "MDF"):
        if mc_str in s:
            return 999
    try:
        return int(s)
    except ValueError:
        return None


def pct_fmt(v, decimals=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v) * 100:.{decimals}f}%"


# ── Load Data ─────────────────────────────────────────────────────────────────

live = load_live_predictions()
pretournament = load_pretournament()

if live is not None:
    field_df = live
elif pretournament is not None:
    field_df = pretournament
else:
    st.error("No prediction data found. Run: `python run_production.py`")
    st.stop()

players = sorted(field_df["player_name"].dropna().tolist())

# ── Player Selector ───────────────────────────────────────────────────────────

st.title("Player Profiles")
selected_player = st.selectbox("Select a player", players, index=0)
if not selected_player:
    st.stop()

player_live: pd.Series | None = None
player_pre: pd.Series | None = None

if live is not None and selected_player in live["player_name"].values:
    player_live = live[live["player_name"] == selected_player].iloc[0]
if pretournament is not None and selected_player in pretournament["player_name"].values:
    player_pre = pretournament[pretournament["player_name"] == selected_player].iloc[0]

form_src = player_live if player_live is not None else player_pre

# ── Historical data ───────────────────────────────────────────────────────────

hbh = load_hole_by_hole()
player_hist = pd.DataFrame()
times_played = made_cut_count = 0
made_cut_pct = best_finish = avg_position = None

if hbh is not None:
    player_hist = hbh[hbh["player_name"] == selected_player].copy()
    if len(player_hist) > 0:
        tourney_hist = (
            player_hist.groupby("year")
            .agg(finish_pos=("finish_pos", "first"), made_cut=("made_cut", "first"))
            .reset_index()
        )
        times_played = len(tourney_hist)
        made_cut_count = int(tourney_hist["made_cut"].sum())
        made_cut_pct = made_cut_count / times_played if times_played > 0 else 0.0
        tourney_hist["finish_num"] = tourney_hist["finish_pos"].apply(parse_finish_num)
        valid_finishes = tourney_hist["finish_num"].dropna()
        valid_finishes = valid_finishes[valid_finishes < 900]
        best_finish = int(valid_finishes.min()) if len(valid_finishes) > 0 else None
        avg_position = float(valid_finishes.mean()) if len(valid_finishes) > 0 else None

# ── MC Probabilities ──────────────────────────────────────────────────────────

if player_live is not None:
    mc_win   = float(safe_get(player_live, "mc_win_prob", "dg_win_prob") or 0)
    mc_top5  = float(safe_get(player_live, "mc_top5_prob", "dg_top5_prob") or 0)
    mc_top10 = float(safe_get(player_live, "mc_top10_prob", "dg_top10_prob") or 0)
    mc_top20 = float(safe_get(player_live, "mc_top20_prob", "dg_top20_prob") or 0)
    mc_cut   = float(safe_get(player_live, "dg_make_cut", "make_cut_prob") or 0.5)
elif player_pre is not None:
    mc_win   = float(safe_get(player_pre, "mc_win_prob", "win_prob") or 0)
    mc_top5  = float(safe_get(player_pre, "mc_top5_prob") or 0)
    mc_top10 = float(safe_get(player_pre, "mc_top10_prob", "top10_prob") or 0)
    mc_top20 = float(safe_get(player_pre, "mc_top20_prob", "top20_prob") or 0)
    mc_cut   = float(safe_get(player_pre, "make_cut_prob") or 0.5)
else:
    mc_win = mc_top5 = mc_top10 = mc_top20 = mc_cut = 0.0

# ── Section 1: Player Summary Card ───────────────────────────────────────────

# Score to par — use current_score_to_par (cumulative) not cumulative_score_to_par (pre-tourney baseline)
score_to_par = None
position_str = "—"
thru_str = ""
tournament_started = False

if player_live is not None:
    score_to_par = safe_get(player_live, "current_score_to_par", "current_score", "total")
    current_pos = safe_get(player_live, "current_pos")
    thru = safe_get(player_live, "thru")
    today = safe_get(player_live, "today")

    if score_to_par is not None and float(score_to_par) != 0:
        tournament_started = True
    elif score_to_par is None and current_pos is not None:
        tournament_started = True

    if current_pos is not None:
        pos_s = str(current_pos)
        if pos_s.replace(".", "").isdigit():
            position_str = f"T{int(float(pos_s))}"
        else:
            position_str = pos_s

    if thru is not None and not (isinstance(thru, float) and np.isnan(thru)) and int(float(thru)) > 0:
        thru_str = f"Thru {int(float(thru))}"

# Context line from historical data
context_parts = []
if times_played > 0:
    context_parts.append(f"{times_played}th Masters")
if best_finish == 1:
    context_parts.append("Won")
elif best_finish:
    year_data = tourney_hist[tourney_hist["finish_num"] == best_finish]["year"].values if times_played > 0 else []
    year_str = f" ({int(year_data[0])})" if len(year_data) else ""
    context_parts.append(f"Best: T{best_finish}{year_str}")
if form_src is not None:
    sg_total = safe_get(form_src, "sg_total")
    if sg_total is not None:
        context_parts.append(f"SG Total: {float(sg_total):+.2f}")

context_line = "  |  ".join(context_parts) if context_parts else ""

# Render card
score_color = "#1a7a45" if score_to_par is not None and float(score_to_par) < 0 else (
    "#c62828" if score_to_par is not None and float(score_to_par) > 0 else "#1a1a1a"
)
score_display = score_fmt(score_to_par) if tournament_started else "—"

st.markdown(f"""
<div class="player-card">
  <div style="font-size:1.5rem;font-weight:700;color:#1a1a1a;margin-bottom:14px">{selected_player}</div>
  <div style="display:flex;gap:32px;align-items:flex-end;margin-bottom:14px">
    <div>
      <div class="stat-label">Score to Par{(" · " + thru_str) if thru_str else ""}</div>
      <div class="big-score" style="color:{score_color}">{score_display}</div>
    </div>
    <div>
      <div class="stat-label">Position</div>
      <div class="big-pos">{position_str}</div>
    </div>
  </div>
  <div style="margin-bottom:12px">
    <span class="badge">Win {pct_fmt(mc_win)}</span>
    <span class="badge">Top 10 {pct_fmt(mc_top10)}</span>
    <span class="badge">Top 20 {pct_fmt(mc_top20)}</span>
  </div>
  {"<div class='context-line'>" + context_line + "</div>" if context_line else ""}
</div>
""", unsafe_allow_html=True)

# ── Section 2: Monte Carlo Histogram ─────────────────────────────────────────

st.markdown('<div class="section-header">Projected Finish Distribution</div>', unsafe_allow_html=True)

# 6 clean buckets: Win, T5, T10, T20, 30+, MC
p_win   = max(0.0, mc_win)
p_t5    = max(0.0, mc_top5 - mc_win)
p_t10   = max(0.0, mc_top10 - mc_top5)
p_t20   = max(0.0, mc_top20 - mc_top10)
p_30    = max(0.0, mc_cut - mc_top20)
p_mc    = max(0.0, 1.0 - mc_cut)

bucket_probs = [p_win, p_t5, p_t10, p_t20, p_30, p_mc]
total = sum(bucket_probs)
if total > 0:
    bucket_probs = [p / total for p in bucket_probs]

bucket_labels = ["Win", "T5", "T10", "T20", "30+", "MC"]
bucket_colors = ["#FFD700", "#2196f3", "#42a5f5", "#90caf9", "#cfd8dc", "#ef5350"]
bar_pcts = [round(p * 100, 1) for p in bucket_probs]

fig_mc = go.Figure(go.Bar(
    x=bucket_labels,
    y=bar_pcts,
    marker_color=bucket_colors,
    text=[f"{p:.0f}%" if p >= 1 else f"{p:.1f}%" for p in bar_pcts],
    textposition="outside",
    textfont=dict(size=14, color="#333"),
    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
))

fig_mc.update_layout(
    template="plotly_white",
    height=280,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(tickfont=dict(size=15), title=None),
    yaxis=dict(
        ticksuffix="%",
        tickfont=dict(size=13),
        title=None,
        range=[0, max(bar_pcts) * 1.25],
    ),
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
)

st.plotly_chart(fig_mc, use_container_width=True, config=CHART_CONFIG)

# ── Section 3: Hole-by-Hole ───────────────────────────────────────────────────

st.markdown('<div class="section-header">Hole-by-Hole Performance at Augusta</div>', unsafe_allow_html=True)

course_stats = load_course_stats()

if hbh is not None and len(player_hist) > 0:
    player_hole_avg = (
        player_hist.groupby("hole_number")["score_to_par"]
        .mean()
        .reset_index()
        .rename(columns={"score_to_par": "player_avg"})
    )

    hole_info = (
        hbh.groupby("hole_number")
        .agg(par=("par", "first"))
        .reset_index()
        .sort_values("hole_number")
    )

    hole_chart = (
        hole_info
        .merge(player_hole_avg, on="hole_number", how="left")
        .sort_values("hole_number")
    )

    # Round values to avoid floating point noise / scientific notation
    hole_chart["player_avg"] = hole_chart["player_avg"].apply(
        lambda v: round(float(v), 3) if not pd.isna(v) else v
    )

    x_holes = [str(int(h)) for h in hole_chart["hole_number"]]
    y_vals = hole_chart["player_avg"].tolist()

    bar_colors = []
    for v in y_vals:
        if pd.isna(v):
            bar_colors.append("#bdbdbd")
        elif v < -0.01:
            bar_colors.append("#43a047")
        elif v > 0.01:
            bar_colors.append("#e53935")
        else:
            bar_colors.append("#9e9e9e")

    fig_holes = go.Figure()

    fig_holes.add_trace(go.Bar(
        x=x_holes,
        y=y_vals,
        marker_color=bar_colors,
        hovertemplate="Hole %{x}: %{y:+.2f} avg vs par<extra></extra>",
    ))

    fig_holes.add_hline(y=0, line_color="#666", line_width=1.5)

    # Amen Corner annotation
    amen_x = ["11", "12", "13"]
    fig_holes.add_vrect(
        x0="10.5", x1="13.5",
        fillcolor="rgba(255,165,0,0.10)",
        line_width=0,
        annotation_text="Amen Corner",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#f57c00",
    )

    fig_holes.update_layout(
        template="plotly_white",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=None,
            tickvals=x_holes,
            ticktext=x_holes,
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=None,
            tickformat=".1f",
            tickfont=dict(size=13),
            zeroline=False,
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig_holes, use_container_width=True, config=CHART_CONFIG)

    # Callout text
    valid = hole_chart.dropna(subset=["player_avg"])
    if len(valid) > 0:
        best3 = valid.nsmallest(3, "player_avg")
        worst3 = valid.nlargest(3, "player_avg")

        best_str = ", ".join(
            f"#{int(r['hole_number'])} ({r['player_avg']:+.2f})"
            for _, r in best3.iterrows()
        )
        worst_str = ", ".join(
            f"#{int(r['hole_number'])} ({r['player_avg']:+.2f})"
            for _, r in worst3.iterrows()
        )

        st.markdown(f'<div class="callout-text">Best holes: {best_str}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="callout-text-bad">Toughest holes: {worst_str}</div>', unsafe_allow_html=True)

        # Amen Corner
        amen = valid[valid["hole_number"].isin([11, 12, 13])]
        if len(amen) > 0:
            amen_avg = amen["player_avg"].mean()
            amen_label = f"Amen Corner (11-13): {amen_avg:+.2f} avg vs par"
            css_class = "callout-text" if amen_avg < 0 else "callout-text-bad"
            st.markdown(f'<div class="{css_class}">{amen_label}</div>', unsafe_allow_html=True)

elif hbh is not None:
    st.info(f"No hole-by-hole history for {selected_player} at Augusta.")

# ── Section 4: SG Radar ───────────────────────────────────────────────────────

st.markdown('<div class="section-header">Strokes Gained Breakdown</div>', unsafe_allow_html=True)

sg_keys = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
sg_labels = ["Off-the-Tee", "Approach", "Short Game", "Putting"]

field_sg = []
for k in sg_keys:
    if k in field_df.columns:
        field_sg.append(float(field_df[k].mean()))
    else:
        field_sg.append(0.0)

player_sg_week = None
player_sg_season = None

if player_live is not None:
    vals = [safe_get(player_live, k) for k in sg_keys]
    if any(v is not None for v in vals):
        player_sg_week = [float(v) if v is not None else 0.0 for v in vals]

if player_pre is not None:
    vals = [safe_get(player_pre, k) for k in sg_keys]
    if any(v is not None for v in vals):
        player_sg_season = [float(v) if v is not None else 0.0 for v in vals]

if player_sg_week is None and player_sg_season is None:
    st.info("No SG data available for this player.")
else:
    cats_closed = sg_labels + [sg_labels[0]]

    primary_sg = player_sg_week or player_sg_season
    r_max = max(max(abs(v) for v in primary_sg), 0.5) * 1.35

    fig_radar = go.Figure()

    # Season avg — thin gray dashed
    if player_sg_season is not None:
        season_closed = player_sg_season + [player_sg_season[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=season_closed,
            theta=cats_closed,
            fill="toself",
            name="Season Avg",
            line=dict(color="#9e9e9e", width=1.5, dash="dash"),
            fillcolor="rgba(158,158,158,0.10)",
            hovertemplate="%{theta}: %{r:+.2f}<extra>Season</extra>",
        ))

    # This Week — bold colored
    if player_sg_week is not None:
        week_closed = player_sg_week + [player_sg_week[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=week_closed,
            theta=cats_closed,
            fill="toself",
            name="This Week",
            line=dict(color="#1565c0", width=3),
            fillcolor="rgba(21,101,192,0.18)",
            hovertemplate="%{theta}: %{r:+.2f}<extra>This Week</extra>",
        ))

    fig_radar.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-r_max, r_max],
                tickformat="+.1f",
                tickfont=dict(size=11),
                gridcolor="rgba(0,0,0,0.12)",
                linecolor="rgba(0,0,0,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                gridcolor="rgba(0,0,0,0.10)",
                linecolor="rgba(0,0,0,0.15)",
            ),
            bgcolor="white",
        ),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, font=dict(size=13)),
        paper_bgcolor="white",
    )

    st.plotly_chart(fig_radar, use_container_width=True, config=CHART_CONFIG)

    # SG summary text
    primary = player_sg_week if player_sg_week is not None else player_sg_season
    best_idx = int(np.argmax(primary))
    worst_idx = int(np.argmin(primary))
    field_best = field_sg[best_idx]
    field_worst = field_sg[worst_idx]
    vs_field_best = primary[best_idx] - field_best
    vs_field_worst = primary[worst_idx] - field_worst

    st.markdown(
        f'<div class="sg-summary">'
        f'<strong>Strength:</strong> {sg_labels[best_idx]} ({vs_field_best:+.2f} vs field)&nbsp;&nbsp;'
        f'<strong>Weakness:</strong> {sg_labels[worst_idx]} ({vs_field_worst:+.2f} vs field)'
        f'</div>',
        unsafe_allow_html=True,
    )

"""
Player Profiles page — Augusta National Model

Deep dive on a player's tournament performance:
- Current form vs historical Masters record
- Monte Carlo outcome histogram
- Hole-by-hole performance chart
- Strengths & weaknesses summary
- SG breakdown radar chart
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

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
LIVE_DIR = ROOT / "data" / "live"


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


def score_fmt(v) -> str:
    if pd.isna(v) or v is None:
        return "—"
    v = int(v)
    if v == 0:
        return "E"
    return f"{v:+d}"


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

# ── Page Header ───────────────────────────────────────────────────────────────

st.title("🏌️ Player Profiles")
st.markdown("Select a player to see their hole-by-hole history, Monte Carlo projections, and SG breakdown.")

selected_player = st.selectbox("Select a player", players, index=0)
if not selected_player:
    st.stop()

# Row lookups (pandas Series)
player_live: pd.Series | None = None
player_pre: pd.Series | None = None

if live is not None and selected_player in live["player_name"].values:
    player_live = live[live["player_name"] == selected_player].iloc[0]
if pretournament is not None and selected_player in pretournament["player_name"].values:
    player_pre = pretournament[pretournament["player_name"] == selected_player].iloc[0]

st.markdown("---")

# ── Section 1: Current Form vs Historical ────────────────────────────────────

hbh = load_hole_by_hole()

# Compute historical Masters stats for this player
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
    else:
        player_hist = pd.DataFrame()
        times_played = 0
        made_cut_count = 0
        made_cut_pct = 0.0
        best_finish = None
        avg_position = None
else:
    player_hist = pd.DataFrame()
    times_played = 0
    made_cut_count = 0
    made_cut_pct = 0.0
    best_finish = None
    avg_position = None

col_cur, col_hist_col = st.columns(2)

with col_cur:
    st.subheader("This Tournament")
    if player_live is not None:
        score_to_par = player_live.get("cumulative_score_to_par", None)
        current_pos = player_live.get("current_pos", None)
        thru = player_live.get("thru", None)
        today = player_live.get("today", None)
        current_round = player_live.get("current_round", None)

        m1, m2 = st.columns(2)
        with m1:
            if score_to_par is not None and not pd.isna(score_to_par):
                st.metric("Score to Par", score_fmt(score_to_par))
            else:
                st.metric("Score to Par", "—")
            if current_round is not None and not pd.isna(current_round):
                st.metric("Round", int(current_round))
        with m2:
            pos_str = f"T{int(current_pos)}" if current_pos is not None and str(current_pos).replace(".", "").isdigit() else (str(current_pos) if current_pos else "—")
            st.metric("Position", pos_str)
            if today is not None and not pd.isna(today):
                thru_note = f"Thru {int(thru)}" if thru is not None and not pd.isna(thru) else ""
                st.metric("Today", score_fmt(today), help=thru_note)
    else:
        st.info("Tournament not yet started — pre-tournament projections")
        if player_pre is not None:
            m1, m2 = st.columns(2)
            win_p = player_pre.get("win_prob", None)
            t10_p = player_pre.get("top10_prob", None)
            t20_p = player_pre.get("top20_prob", None)
            cut_p = player_pre.get("make_cut_prob", None)
            with m1:
                st.metric("Model Win %", f"{win_p:.1%}" if win_p is not None else "—")
                st.metric("Model Top-20 %", f"{t20_p:.1%}" if t20_p is not None else "—")
            with m2:
                st.metric("Model Top-10 %", f"{t10_p:.1%}" if t10_p is not None else "—")
                st.metric("Model Cut %", f"{cut_p:.1%}" if cut_p is not None else "—")

with col_hist_col:
    st.subheader("Masters History")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Times Played", times_played if times_played > 0 else "—")
        if best_finish == 1:
            st.metric("Best Finish", "Win! 🏆")
        elif best_finish:
            st.metric("Best Finish", f"T{best_finish}")
        else:
            st.metric("Best Finish", "—")
    with m2:
        st.metric("Made Cut", f"{made_cut_count}/{times_played}" if times_played > 0 else "—",
                  help=f"{made_cut_pct:.0%} cut rate" if times_played > 0 else None)
        st.metric("Avg Position", f"#{avg_position:.0f}" if avg_position else "—")

# Season Form (rolling SG from model)
st.markdown("---")
st.subheader("Season Form — Strokes Gained (Rolling)")

form_src = player_live if player_live is not None else player_pre
if form_src is not None:
    sg_display = {
        "Off-the-Tee": form_src.get("sg_ott", None),
        "Approach": form_src.get("sg_app", None),
        "Around Green": form_src.get("sg_arg", None),
        "Putting": form_src.get("sg_putt", None),
        "T2G": form_src.get("sg_t2g", None),
        "Total": form_src.get("sg_total", None),
    }
    sg_cols_ui = st.columns(6)
    for i, (label, val) in enumerate(sg_display.items()):
        with sg_cols_ui[i]:
            if val is not None and not pd.isna(val):
                v = float(val)
                delta_color = "normal" if abs(v) > 0.01 else "off"
                st.metric(f"SG: {label}", f"{v:+.2f}")
            else:
                st.metric(f"SG: {label}", "—")

st.markdown("---")

# ── Section 2: Monte Carlo Outcome Histogram ─────────────────────────────────

st.subheader("Monte Carlo Finish Distribution")

# Pull MC probabilities
if player_live is not None:
    mc_win    = float(player_live.get("mc_win_prob", 0) or 0)
    mc_top5   = float(player_live.get("mc_top5_prob", 0) or 0)
    mc_top10  = float(player_live.get("mc_top10_prob", 0) or 0)
    mc_top20  = float(player_live.get("mc_top20_prob", 0) or 0)
    mc_cut    = float(player_live.get("dg_make_cut", player_live.get("make_cut_prob", 0.5)) or 0.5)
elif player_pre is not None:
    mc_win    = float(player_pre.get("mc_win_prob", player_pre.get("win_prob", 0)) or 0)
    mc_top5   = float(player_pre.get("mc_top5_prob", 0) or 0)
    mc_top10  = float(player_pre.get("mc_top10_prob", player_pre.get("top10_prob", 0)) or 0)
    mc_top20  = float(player_pre.get("mc_top20_prob", player_pre.get("top20_prob", 0)) or 0)
    mc_cut    = float(player_pre.get("make_cut_prob", 0.5) or 0.5)
else:
    mc_win = mc_top5 = mc_top10 = mc_top20 = mc_cut = 0.0

FIELD_SIZE = max(len(field_df), 80)
N_SIMS = 20_000

# Decompose cumulative probs into bracket probabilities
p_win     = max(0.0, mc_win)
p_top5    = max(0.0, mc_top5 - mc_win)
p_top10   = max(0.0, mc_top10 - mc_top5)
p_top20   = max(0.0, mc_top20 - mc_top10)
p_cut     = max(0.0, mc_cut - mc_top20)
p_mc      = max(0.0, 1.0 - mc_cut)

bracket_probs = np.array([p_win, p_top5, p_top10, p_top20, p_cut, p_mc])
total = bracket_probs.sum()
if total > 0:
    bracket_probs /= total

rng = np.random.default_rng(seed=42)
bracket_counts = rng.multinomial(N_SIMS, bracket_probs)

# Simulate within-bracket finish positions
sim_positions = []
for i, count in enumerate(bracket_counts):
    if count == 0:
        continue
    if i == 0:    # Win
        sim_positions.extend([1] * count)
    elif i == 1:  # 2-5
        sim_positions.extend(rng.integers(2, 6, size=count).tolist())
    elif i == 2:  # 6-10
        sim_positions.extend(rng.integers(6, 11, size=count).tolist())
    elif i == 3:  # 11-20
        sim_positions.extend(rng.integers(11, 21, size=count).tolist())
    elif i == 4:  # 21-cut
        sim_positions.extend(rng.integers(21, max(22, FIELD_SIZE // 2 + 1), size=count).tolist())
    else:          # Missed cut
        sim_positions.extend([FIELD_SIZE + 1] * count)

sim_arr = np.array(sim_positions)

# Histogram bins
bin_edges  = [1, 2, 5, 10, 15, 20, 30, 50, 80]
bin_labels = ["Win", "2–4", "5–9", "10–14", "15–19", "20–29", "30–49", "50+", "MC"]
bin_counts = []
for j in range(len(bin_edges) - 1):
    lo, hi = bin_edges[j], bin_edges[j + 1]
    bin_counts.append(int(((sim_arr >= lo) & (sim_arr < hi)).sum()))
bin_counts.append(int((sim_arr > FIELD_SIZE).sum()))  # MC

ZONE_COLORS = {
    "Win":   "#FFD700",
    "2–4":   "#1a7a45",
    "5–9":   "#2ecc71",
    "10–14": "#5dade2",
    "15–19": "#85c1e9",
    "20–29": "#aed6f1",
    "30–49": "#d5dbdb",
    "50+":   "#e59866",
    "MC":    "#e74c3c",
}

fig_mc = go.Figure(go.Bar(
    x=bin_labels,
    y=[c / N_SIMS * 100 for c in bin_counts],
    marker_color=[ZONE_COLORS[l] for l in bin_labels],
    text=[f"{c / N_SIMS:.1%}" for c in bin_counts],
    textposition="outside",
    textfont=dict(size=11),
    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
))

# Zone annotations
fig_mc.add_vrect(x0=-0.5, x1=0.5,  fillcolor="rgba(255,215,0,0.08)",  line_width=0, annotation_text="Win zone",  annotation_position="top left",  annotation_font_size=10)
fig_mc.add_vrect(x0=-0.5, x1=2.5,  fillcolor="rgba(46,204,113,0.06)", line_width=0, annotation_text="Top 5",     annotation_position="top left",  annotation_font_size=10)
fig_mc.add_vrect(x0=-0.5, x1=3.5,  fillcolor="rgba(93,173,226,0.05)", line_width=0, annotation_text="Top 10",    annotation_position="top left",  annotation_font_size=10)

fig_mc.update_layout(
    title=dict(text=f"{selected_player} — Simulated Finish Distribution ({N_SIMS:,} sims)", font=dict(size=15)),
    xaxis_title="Finish Position",
    yaxis=dict(title="Probability (%)", ticksuffix="%"),
    height=420,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    showlegend=False,
    margin=dict(t=60, b=40),
)

st.plotly_chart(fig_mc, use_container_width=True)

# Quick MC summary metrics
if mc_win > 0 or mc_top10 > 0:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Win %", f"{mc_win:.1%}")
    mc2.metric("Top-5 %", f"{mc_top5:.1%}")
    mc3.metric("Top-10 %", f"{mc_top10:.1%}")
    mc4.metric("Top-20 %", f"{mc_top20:.1%}")

st.markdown("---")

# ── Section 3: Hole-by-Hole Performance ──────────────────────────────────────

st.subheader("Hole-by-Hole Performance at Augusta")

course_stats = load_course_stats()

if hbh is not None and len(player_hist) > 0:
    # Player average score-to-par per hole
    player_hole_avg = (
        player_hist.groupby("hole_number")["score_to_par"]
        .mean()
        .reset_index()
        .rename(columns={"score_to_par": "player_avg"})
    )

    # Hole info — par, name, yards from hbh data
    hole_info = (
        hbh.groupby("hole_number")
        .agg(par=("par", "first"), hole_name=("hole_name", "first"), yards=("yards", "first"))
        .reset_index()
        .sort_values("hole_number")
    )

    # Field average
    if course_stats is not None and "avg_score_to_par" in course_stats.columns:
        field_avg = course_stats[["hole_number", "avg_score_to_par"]].rename(
            columns={"avg_score_to_par": "field_avg"}
        )
    else:
        field_avg = (
            hbh.groupby("hole_number")["score_to_par"]
            .mean()
            .reset_index()
            .rename(columns={"score_to_par": "field_avg"})
        )

    hole_chart = (
        hole_info
        .merge(player_hole_avg, on="hole_number", how="left")
        .merge(field_avg[["hole_number", "field_avg"]], on="hole_number", how="left")
        .sort_values("hole_number")
    )

    # Label each bar
    x_labels = []
    for _, r in hole_chart.iterrows():
        hn = int(r["hole_number"])
        name = r.get("hole_name", "") or ""
        par = int(r["par"]) if not pd.isna(r.get("par", float("nan"))) else "?"
        x_labels.append(f"#{hn}<br>Par {par}<br><i>{name}</i>")

    # Colours: green = under par on average, red = over par
    bar_colors = [
        "#2ecc71" if (not pd.isna(v) and v < -0.01)
        else "#e74c3c" if (not pd.isna(v) and v > 0.01)
        else "#95a5a6"
        for v in hole_chart["player_avg"]
    ]

    # Identify top 3 strengths / weaknesses
    valid = hole_chart.dropna(subset=["player_avg"])
    strengths_hn  = valid.nsmallest(3, "player_avg")["hole_number"].tolist()
    weaknesses_hn = valid.nlargest(3, "player_avg")["hole_number"].tolist()

    # Add border highlight for strengths/weaknesses
    marker_line_colors = []
    marker_line_widths = []
    for hn in hole_chart["hole_number"]:
        if hn in strengths_hn:
            marker_line_colors.append("#00ff88")
            marker_line_widths.append(3)
        elif hn in weaknesses_hn:
            marker_line_colors.append("#ff4444")
            marker_line_widths.append(3)
        else:
            marker_line_colors.append("rgba(0,0,0,0)")
            marker_line_widths.append(0)

    fig_holes = go.Figure()

    fig_holes.add_trace(go.Bar(
        x=x_labels,
        y=hole_chart["player_avg"],
        name=f"{selected_player}",
        marker=dict(
            color=bar_colors,
            line=dict(color=marker_line_colors, width=marker_line_widths),
        ),
        opacity=0.85,
        text=[f"{v:+.2f}" if not pd.isna(v) else "" for v in hole_chart["player_avg"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="Hole %{x}: %{y:+.2f} avg<extra></extra>",
    ))

    fig_holes.add_trace(go.Scatter(
        x=x_labels,
        y=hole_chart["field_avg"],
        mode="lines+markers",
        name="Field Avg",
        line=dict(color="#f39c12", width=2, dash="dot"),
        marker=dict(size=5),
        hovertemplate="Field avg: %{y:+.2f}<extra></extra>",
    ))

    # Shade Amen Corner (holes 11-13)
    amen_indices = [i for i, r in hole_chart.reset_index(drop=True).iterrows()
                    if int(r["hole_number"]) in (11, 12, 13)]
    if len(amen_indices) >= 2:
        fig_holes.add_vrect(
            x0=amen_indices[0] - 0.5,
            x1=amen_indices[-1] + 0.5,
            fillcolor="rgba(255,165,0,0.10)",
            line_width=1,
            line_color="rgba(255,165,0,0.4)",
            annotation_text="Amen Corner",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="#f39c12",
        )

    fig_holes.add_hline(y=0, line_color="rgba(200,200,200,0.5)", line_width=1)

    fig_holes.update_layout(
        title=dict(text=f"{selected_player} — Scoring Average per Hole (Masters history)", font=dict(size=15)),
        xaxis=dict(title="Hole", tickangle=0),
        yaxis=dict(title="Avg Score vs Par", tickformat="+.2f", zeroline=True, zerolinecolor="gray"),
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        barmode="overlay",
        margin=dict(t=60, b=20),
    )

    st.plotly_chart(fig_holes, use_container_width=True)

    # ── Section 4: Strengths & Weaknesses ────────────────────────────────────
    st.markdown("---")
    st.subheader("Strengths & Weaknesses")

    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("**Best holes (strengths)**")
        for hn in strengths_hn:
            row = hole_chart[hole_chart["hole_number"] == hn]
            if len(row) > 0:
                r = row.iloc[0]
                name_str = f" — {r['hole_name']}" if r.get("hole_name") else ""
                par_str = f", Par {int(r['par'])}" if not pd.isna(r.get("par", float("nan"))) else ""
                st.markdown(f"✅ **Hole {int(hn)}{name_str}**{par_str}: {r['player_avg']:+.2f} avg vs par")

    with col_w:
        st.markdown("**Toughest holes (weaknesses)**")
        for hn in weaknesses_hn:
            row = hole_chart[hole_chart["hole_number"] == hn]
            if len(row) > 0:
                r = row.iloc[0]
                name_str = f" — {r['hole_name']}" if r.get("hole_name") else ""
                par_str = f", Par {int(r['par'])}" if not pd.isna(r.get("par", float("nan"))) else ""
                st.markdown(f"⚠️ **Hole {int(hn)}{name_str}**{par_str}: {r['player_avg']:+.2f} avg vs par")

    # Par type breakdown
    if "par" in hole_chart.columns:
        st.markdown("")
        par_breakdown = (
            hole_chart.dropna(subset=["player_avg", "par"])
            .groupby("par")["player_avg"]
            .mean()
        )
        par_items = [f"{'🟢' if avg < 0 else '🔴'} **Par {int(p)}**: {avg:+.2f} avg vs par"
                     for p, avg in par_breakdown.items()]
        st.markdown("  ·  ".join(par_items))

    # Amen Corner summary
    amen_rows = hole_chart[hole_chart["hole_number"].isin([11, 12, 13])].dropna(subset=["player_avg"])
    if len(amen_rows) > 0:
        amen_avg = amen_rows["player_avg"].mean()
        emoji = "🟢" if amen_avg < 0 else "🔴"
        st.markdown(f"{emoji} **Amen Corner (11–13)**: {amen_avg:+.2f} avg vs par")

elif hbh is not None:
    st.info(f"No historical hole-by-hole data for **{selected_player}** at Augusta.")

    # Fall back to SG-based strengths/weaknesses
    if form_src is not None:
        st.subheader("Strengths & Weaknesses")
        sg_map = {
            "Off-the-tee (driving distance/accuracy)": form_src.get("sg_ott", None),
            "Approach play (iron accuracy)":           form_src.get("sg_app", None),
            "Around the green (chipping/pitching)":    form_src.get("sg_arg", None),
            "Putting":                                  form_src.get("sg_putt", None),
        }
        strengths_list  = [(k, float(v)) for k, v in sg_map.items() if v is not None and not pd.isna(v) and float(v) > 0.2]
        weaknesses_list = [(k, float(v)) for k, v in sg_map.items() if v is not None and not pd.isna(v) and float(v) < -0.2]

        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for k, v in sorted(strengths_list, key=lambda x: -x[1]):
                st.markdown(f"✅ {k}: {v:+.2f} SG")
            if not strengths_list:
                st.markdown("_No clear strengths from SG data_")
        with col_w:
            st.markdown("**Weaknesses**")
            for k, v in sorted(weaknesses_list, key=lambda x: x[1]):
                st.markdown(f"⚠️ {k}: {v:+.2f} SG")
            if not weaknesses_list:
                st.markdown("_No clear weaknesses from SG data_")
else:
    st.info("Hole-by-hole data not available.")

st.markdown("---")

# ── Section 5: SG Radar Chart ─────────────────────────────────────────────────

st.subheader("Strokes Gained Breakdown")

sg_keys       = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
sg_cat_labels = ["Off-the-Tee", "Approach", "Around Green", "Putting"]

# Field average SG (from whichever df we have)
field_sg_avgs = {}
for k in sg_keys:
    if k in field_df.columns:
        field_sg_avgs[k] = float(field_df[k].mean())
    else:
        field_sg_avgs[k] = 0.0

# Player SG values — use live (current tournament) if available
player_sg_tournament = None
player_sg_season = None

if player_live is not None:
    vals = [player_live.get(k, None) for k in sg_keys]
    if any(v is not None and not pd.isna(v) for v in vals):
        player_sg_tournament = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in vals]

if player_pre is not None:
    vals = [player_pre.get(k, None) for k in sg_keys]
    if any(v is not None and not pd.isna(v) for v in vals):
        player_sg_season = [float(v) if v is not None and not pd.isna(v) else 0.0 for v in vals]

field_sg = [field_sg_avgs.get(k, 0.0) for k in sg_keys]

# Close the radar polygon
cats_closed  = sg_cat_labels + [sg_cat_labels[0]]
field_closed = field_sg + [field_sg[0]]

if player_sg_tournament is None and player_sg_season is None:
    st.info("No SG data available for this player.")
else:
    fig_radar = go.Figure()

    # Field average baseline
    fig_radar.add_trace(go.Scatterpolar(
        r=field_closed,
        theta=cats_closed,
        fill="toself",
        name="Field Average",
        line=dict(color="#85c1e9", width=1.5, dash="dot"),
        fillcolor="rgba(133,193,233,0.08)",
        hovertemplate="%{theta}: %{r:+.2f}<extra>Field Avg</extra>",
    ))

    # Season average (pre-tournament rolling)
    if player_sg_season is not None:
        season_closed = player_sg_season + [player_sg_season[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=season_closed,
            theta=cats_closed,
            fill="toself",
            name="Season Avg",
            line=dict(color="#2ecc71", width=2, dash="dash"),
            fillcolor="rgba(46,204,113,0.12)",
            hovertemplate="%{theta}: %{r:+.2f}<extra>Season</extra>",
        ))

    # Current tournament SG (live)
    if player_sg_tournament is not None:
        tournament_closed = player_sg_tournament + [player_sg_tournament[0]]
        label = "This Week" if player_live is not None else "Current"
        fig_radar.add_trace(go.Scatterpolar(
            r=tournament_closed,
            theta=cats_closed,
            fill="toself",
            name=label,
            line=dict(color="#FFD700", width=2.5),
            fillcolor="rgba(255,215,0,0.18)",
            hovertemplate="%{theta}: %{r:+.2f}<extra>This Week</extra>",
        ))

    # Determine the primary SG values to show in subtitle
    primary = player_sg_tournament if player_sg_tournament else player_sg_season
    best_cat  = sg_cat_labels[int(np.argmax(primary))]
    worst_cat = sg_cat_labels[int(np.argmin(primary))]

    r_max = max(max(abs(v) for v in (player_sg_tournament or [0])),
                max(abs(v) for v in (player_sg_season or [0])),
                0.5) * 1.3

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-r_max, r_max],
                tickformat="+.1f",
                gridcolor="rgba(128,128,128,0.25)",
                tickfont=dict(size=9),
            ),
            angularaxis=dict(gridcolor="rgba(128,128,128,0.25)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        title=dict(
            text=(f"{selected_player} — SG Breakdown<br>"
                  f"<sub>Strength: {best_cat}  ·  Weakness: {worst_cat}</sub>"),
            font=dict(size=14),
        ),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig_radar, use_container_width=True)

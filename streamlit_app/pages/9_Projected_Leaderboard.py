import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Projected Leaderboard — Augusta Model", layout="wide", page_icon="⛳")

ROOT = Path(__file__).parent.parent.parent
LIVE = ROOT / "data" / "live"
PROCESSED = ROOT / "data" / "processed"


@st.cache_data(ttl=300)
def load_live():
    p = LIVE / "live_predictions_latest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


live = load_live()
preds = load_predictions()

if live is None and preds is None:
    st.error("No prediction data found. Run the pipeline.")
    st.stop()

df = live if live is not None else preds

st.title("Projected Leaderboard")
st.caption("Monte Carlo simulation projections — full field finish distribution.")

# ── Column resolution ────────────────────────────────────────────────

def col(preferred, fallback=None):
    if preferred in df.columns:
        return preferred
    if fallback and fallback in df.columns:
        return fallback
    return None

proj_col   = col("mc_projected_total")
p25_col    = col("mc_proj_p25")
p90_col    = col("mc_proj_p90")
win_col    = col("mc_win_prob", "blended_win_prob")
t10_col    = col("mc_top10_prob", "blended_top10_prob")
# Prefer live DG current_score over model-internal cumulative_score_to_par
score_col  = col("current_score", "cumulative_score_to_par")
pos_col    = col("current_pos")
esr_col    = col("expected_score_per_round", "inweek_mean")
thru_col   = col("thru")

if proj_col is None:
    st.warning(
        "Monte Carlo projected scores not found in data. "
        "Showing pre-tournament model order."
    )

# ── Build projected leaderboard ──────────────────────────────────────

lb = df[["player_name"]].copy()

# Current score
if score_col:
    lb["current_score"] = pd.to_numeric(df[score_col], errors="coerce")
else:
    lb["current_score"] = np.nan

# Position
if pos_col:
    lb["position"] = df[pos_col].fillna("—")
else:
    lb["position"] = "—"

# Thru holes (for display)
if thru_col:
    lb["thru"] = df[thru_col]
else:
    lb["thru"] = np.nan

# Expected score per round (skill signal)
if esr_col:
    lb["expected_score_per_round"] = pd.to_numeric(df[esr_col], errors="coerce")
else:
    lb["expected_score_per_round"] = np.nan

# MC projected total
if proj_col:
    lb["projected_total"] = pd.to_numeric(df[proj_col], errors="coerce")
else:
    # Fallback: current_score + expected_score_per_round × remaining rounds
    # Better than win_prob * -100 which is nonsensical
    if score_col and esr_col:
        current = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        esr = pd.to_numeric(df[esr_col], errors="coerce").fillna(-0.5)
        lb["projected_total"] = current + esr * 4  # rough: 4 rounds remaining
    elif score_col:
        lb["projected_total"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        lb["projected_total"] = np.nan

if p25_col:
    lb["best_case_score"] = pd.to_numeric(df[p25_col], errors="coerce")
else:
    lb["best_case_score"] = lb["projected_total"] - 3

if p90_col:
    lb["worst_case_score"] = pd.to_numeric(df[p90_col], errors="coerce")
else:
    lb["worst_case_score"] = lb["projected_total"] + 4

if win_col:
    lb["win_pct"] = pd.to_numeric(df[win_col], errors="coerce")
else:
    lb["win_pct"] = np.nan

if t10_col:
    lb["t10_pct"] = pd.to_numeric(df[t10_col], errors="coerce")
else:
    lb["t10_pct"] = np.nan

# Drop rows without projected total
lb = lb.dropna(subset=["projected_total"]).copy()

# Sort by projected_total ascending (lower score = better)
lb = lb.sort_values("projected_total").reset_index(drop=True)
lb["expected_finish"] = lb.index + 1

# Best/worst case finishes
medians = lb["projected_total"].values
best_scores  = lb["best_case_score"].fillna(lb["projected_total"]).values
worst_scores = lb["worst_case_score"].fillna(lb["projected_total"]).values

lb["best_case_finish"] = [
    int(1 + np.sum(medians < best_scores[i])) for i in range(len(lb))
]
lb["worst_case_finish"] = [
    int(1 + np.sum(medians < worst_scores[i])) for i in range(len(lb))
]

# ── Format helpers ───────────────────────────────────────────────────

def fmt_score(val):
    try:
        v = int(round(float(val)))
        if v < 0:
            return str(v)
        elif v > 0:
            return f"+{v}"
        return "E"
    except Exception:
        return "—"


def fmt_finish(val):
    try:
        return f"T{int(val)}" if int(val) > 1 else "1"
    except Exception:
        return "—"


def fmt_esr(val):
    try:
        v = float(val)
        if v < 0:
            return f"{v:.2f}"
        elif v > 0:
            return f"+{v:.2f}"
        return "E"
    except Exception:
        return "—"


# ── Sortable table ───────────────────────────────────────────────────
st.subheader("Full Field Projections")

sort_options = {
    "Expected Finish": "expected_finish",
    "Current Score": "current_score",
    "Projected Final Score": "projected_total",
    "Win %": "win_pct",
    "Top-10 %": "t10_pct",
    "Best Case Finish": "best_case_finish",
    "Worst Case Finish": "worst_case_finish",
    "Skill (Exp Score/Rd)": "expected_score_per_round",
}
sort_label = st.selectbox("Sort by", list(sort_options.keys()), index=0)
sort_col = sort_options[sort_label]
sort_asc = sort_label not in ("Win %", "Top-10 %", "Skill (Exp Score/Rd)")

display = lb.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True).copy()
display.index = display.index + 1

table = pd.DataFrame({
    "Player": display["player_name"],
    "Pos": display["position"],
    "Current": display["current_score"].apply(fmt_score),
    "Proj Total": display["projected_total"].apply(fmt_score),
    "Exp Finish": display["expected_finish"].apply(fmt_finish),
    "Best": display["best_case_finish"].apply(fmt_finish),
    "Worst": display["worst_case_finish"].apply(fmt_finish),
    "Win %": display["win_pct"],
    "Top-10 %": display["t10_pct"],
    "Skill/Rd": display["expected_score_per_round"].apply(fmt_esr),
})

# Scale probabilities to percentage display
table["Win %"]   = table["Win %"] * 100
table["Top-10 %"] = table["Top-10 %"] * 100

pct_cfg = {
    "Win %":   st.column_config.NumberColumn("Win %",    format="%.1f%%"),
    "Top-10 %": st.column_config.NumberColumn("Top-10 %", format="%.1f%%"),
}

st.dataframe(table, use_container_width=True, column_config=pct_cfg, height=520)

# ── Plotly range chart ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Finish Range Chart")
st.caption(
    "Bar = Best Case (P25 score) to Worst Case (P90 score). "
    "Dot = Expected Finish. Players sorted by expected finish — tighter bars mean more consistent players."
)

max_players = st.slider("Players to show", min_value=10, max_value=len(lb), value=min(40, len(lb)), step=5)
chart_df = lb.head(max_players).copy()

max_win = chart_df["win_pct"].max() if chart_df["win_pct"].notna().any() else 1.0
if max_win == 0:
    max_win = 1.0


def win_color(win_p):
    if pd.isna(win_p):
        return "rgba(150,150,150,0.5)"
    frac = min(float(win_p) / float(max_win), 1.0)
    r  = int(46  + frac * (255 - 46))
    g  = int(204 - frac * (204 - 165))
    bl = int(113 - frac * (113 - 0))
    return f"rgba({r},{g},{bl},0.7)"


fig = go.Figure()

y_labels       = chart_df["player_name"].tolist()
best_finishes   = chart_df["best_case_finish"].tolist()
worst_finishes  = chart_df["worst_case_finish"].tolist()
expected_finishes = chart_df["expected_finish"].tolist()
win_probs       = chart_df["win_pct"].tolist()
proj_totals     = chart_df["projected_total"].tolist()
curr_scores     = chart_df["current_score"].tolist()

for i, (player, best, worst, exp, wp, proj, curr) in enumerate(
    zip(y_labels, best_finishes, worst_finishes, expected_finishes, win_probs, proj_totals, curr_scores)
):
    color = win_color(wp)
    curr_str = fmt_score(curr) if not pd.isna(curr) else "—"
    proj_str = fmt_score(proj)
    wp_str   = f"{wp:.1%}" if not pd.isna(wp) else "—"
    fig.add_trace(go.Scatter(
        x=[best, worst],
        y=[player, player],
        mode="lines",
        line=dict(color=color, width=10),
        showlegend=False,
        hovertemplate=(
            f"<b>{player}</b><br>"
            f"Current: {curr_str}<br>"
            f"Projected: {proj_str}<br>"
            f"Best case: T{best}<br>"
            f"Expected: T{exp}<br>"
            f"Worst case: T{worst}<br>"
            f"Win%: {wp_str}<extra></extra>"
        ),
    ))

fig.add_trace(go.Scatter(
    x=expected_finishes,
    y=y_labels,
    mode="markers",
    marker=dict(
        color="navy",
        size=9,
        symbol="circle",
        line=dict(color="white", width=1.5),
    ),
    name="Expected Finish",
    hovertemplate="<b>%{y}</b><br>Expected: T%{x}<extra></extra>",
))

fig.add_vline(x=10, line_dash="dot", line_color="green", opacity=0.4,
              annotation_text="Top 10", annotation_position="top")
fig.add_vline(x=5, line_dash="dot", line_color="gold", opacity=0.5,
              annotation_text="Top 5", annotation_position="top")

fig.update_layout(
    height=max(400, max_players * 22),
    xaxis=dict(
        title="Projected Finish Position",
        dtick=5,
    ),
    yaxis=dict(
        title="",
        categoryorder="array",
        categoryarray=y_labels[::-1],
        tickfont=dict(size=11),
    ),
    margin=dict(l=160, r=30, t=40, b=50),
    showlegend=True,
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    hovermode="closest",
)

fig.update_xaxes(range=[max(worst_finishes) + 2, 0])

st.plotly_chart(fig, use_container_width=True)

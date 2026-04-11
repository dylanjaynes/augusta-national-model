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

# ── Build projected leaderboard ─────────────────────────────────────

required_cols = {
    "mc_projected_total": "mc_projected_total",
    "mc_proj_p25": "mc_proj_p25",
    "mc_proj_p90": "mc_proj_p90",
    "mc_win_prob": "win_prob" if "win_prob" in df.columns and "mc_win_prob" not in df.columns else "mc_win_prob",
    "mc_top10_prob": "top10_prob" if "top10_prob" in df.columns and "mc_top10_prob" not in df.columns else "mc_top10_prob",
}

# Resolve column names flexibly
def col(preferred, fallback=None):
    if preferred in df.columns:
        return preferred
    if fallback and fallback in df.columns:
        return fallback
    return None

proj_col = col("mc_projected_total")
p25_col = col("mc_proj_p25")
p90_col = col("mc_proj_p90")
win_col = col("mc_win_prob", "blended_win_prob")
t10_col = col("mc_top10_prob", "blended_top10_prob")
score_col = col("cumulative_score_to_par", "current_score")
pos_col = col("current_pos")

if proj_col is None:
    st.warning(
        "Monte Carlo projected scores not found in data. "
        "Showing pre-tournament model order."
    )

# Work on a copy
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

# MC projected scores
if proj_col:
    lb["projected_total"] = pd.to_numeric(df[proj_col], errors="coerce")
else:
    # Fall back to pre-tournament predictions
    win_prob_col = col("win_prob")
    lb["projected_total"] = pd.to_numeric(df[win_prob_col], errors="coerce") * -100 if win_prob_col else np.nan

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

# Sort by projected_total ascending (lower score = better) for expected finish
lb = lb.sort_values("projected_total").reset_index(drop=True)
lb["expected_finish"] = lb.index + 1

# Best case finish: rank if this player shoots their best_case_score, others shoot median
medians = lb["projected_total"].values
best_scores = lb["best_case_score"].fillna(lb["projected_total"]).values
worst_scores = lb["worst_case_score"].fillna(lb["projected_total"]).values

lb["best_case_finish"] = [
    int(1 + np.sum(medians < best_scores[i])) for i in range(len(lb))
]
lb["worst_case_finish"] = [
    int(1 + np.sum(medians < worst_scores[i])) for i in range(len(lb))
]

# Format helpers
def fmt_score(val):
    try:
        v = int(float(val))
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


# ── Sortable table ──────────────────────────────────────────────────
st.subheader("Full Field Projections")

sort_options = {
    "Expected Finish": "expected_finish",
    "Current Score": "current_score",
    "Projected Final Score": "projected_total",
    "Win %": "win_pct",
    "Top-10 %": "t10_pct",
    "Best Case Finish": "best_case_finish",
    "Worst Case Finish": "worst_case_finish",
}
sort_label = st.selectbox("Sort by", list(sort_options.keys()), index=0)
sort_col = sort_options[sort_label]
sort_asc = sort_label not in ("Win %", "Top-10 %")

display = lb.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True).copy()
display.index = display.index + 1

table = pd.DataFrame({
    "Player": display["player_name"],
    "Pos": display["position"],
    "Current Score": display["current_score"].apply(fmt_score),
    "Projected Total": display["projected_total"].apply(fmt_score),
    "Expected Finish": display["expected_finish"].apply(fmt_finish),
    "Best Case": display["best_case_finish"].apply(fmt_finish),
    "Worst Case": display["worst_case_finish"].apply(fmt_finish),
    "Win %": display["win_pct"],
    "Top-10 %": display["t10_pct"],
})

pct_cfg = {
    "Win %": st.column_config.NumberColumn("Win %", format="%.1f%%"),
    "Top-10 %": st.column_config.NumberColumn("Top-10 %", format="%.1f%%"),
}
# Scale to percentage display
table["Win %"] = table["Win %"] * 100
table["Top-10 %"] = table["Top-10 %"] * 100

st.dataframe(table, use_container_width=True, column_config=pct_cfg, height=520)

# ── Plotly range chart ──────────────────────────────────────────────
st.markdown("---")
st.subheader("Finish Range Chart")
st.caption(
    "Bar = Best Case (P25 score) to Worst Case (P90 score). "
    "Dot = Expected Finish. Players sorted by expected finish — tighter bars mean more consistent players."
)

# Show top N for readability
max_players = st.slider("Players to show", min_value=10, max_value=len(lb), value=min(40, len(lb)), step=5)
chart_df = lb.head(max_players).copy()

# Color by win probability (gradient: gold for top contenders)
max_win = chart_df["win_pct"].max() if chart_df["win_pct"].notna().any() else 1.0
if max_win == 0:
    max_win = 1.0

def win_color(win_p):
    """Green for high win probability, gray for low."""
    if pd.isna(win_p):
        return "rgba(150,150,150,0.5)"
    frac = min(float(win_p) / float(max_win), 1.0)
    r = int(46 + frac * (255 - 46))
    g = int(204 - frac * (204 - 165))
    bl = int(113 - frac * (113 - 0))
    return f"rgba({r},{g},{bl},0.7)"


fig = go.Figure()

y_labels = chart_df["player_name"].tolist()
best_finishes = chart_df["best_case_finish"].tolist()
worst_finishes = chart_df["worst_case_finish"].tolist()
expected_finishes = chart_df["expected_finish"].tolist()
win_probs = chart_df["win_pct"].tolist()

# Draw bars (best to worst case)
for i, (player, best, worst, exp, wp) in enumerate(
    zip(y_labels, best_finishes, worst_finishes, expected_finishes, win_probs)
):
    color = win_color(wp)
    fig.add_trace(go.Scatter(
        x=[best, worst],
        y=[player, player],
        mode="lines",
        line=dict(color=color, width=10),
        showlegend=False,
        hovertemplate=(
            f"<b>{player}</b><br>"
            f"Best case: T{best}<br>"
            f"Expected: T{exp}<br>"
            f"Worst case: T{worst}<br>"
            f"Win%: {wp:.1%}<extra></extra>" if not pd.isna(wp) else
            f"<b>{player}</b><br>Best: T{best} | Exp: T{exp} | Worst: T{worst}<extra></extra>"
        ),
    ))

# Expected finish dots
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

# Vertical reference lines for cut / top-10
fig.add_vline(x=10, line_dash="dot", line_color="green", opacity=0.4,
              annotation_text="Top 10", annotation_position="top")
fig.add_vline(x=5, line_dash="dot", line_color="gold", opacity=0.5,
              annotation_text="Top 5", annotation_position="top")

fig.update_layout(
    height=max(400, max_players * 22),
    xaxis=dict(
        title="Projected Finish Position",
        autorange="reversed",  # 1 on the right (best) — flip so 1 is leftmost
        dtick=5,
    ),
    yaxis=dict(
        title="",
        categoryorder="array",
        categoryarray=y_labels[::-1],  # best at top
        tickfont=dict(size=11),
    ),
    margin=dict(l=160, r=30, t=40, b=50),
    showlegend=True,
    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    hovermode="closest",
)

# Fix x-axis so 1 is on left (lower = better)
fig.update_xaxes(autorange=True)
fig.update_xaxes(range=[max(worst_finishes) + 2, 0])

st.plotly_chart(fig, use_container_width=True)

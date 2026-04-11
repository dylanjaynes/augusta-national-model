"""
Projected Leaderboard — Augusta National Model (mobile-first redesign)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ROOT = Path(__file__).parent.parent.parent
LIVE = ROOT / "data" / "live"
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Projected Leaderboard — Augusta Model",
    layout="wide",
    page_icon="⛳",
)

st.markdown("""
<style>
.summary-banner {
    background: linear-gradient(135deg, #1a3a1a 0%, #2d5a2d 100%);
    color: white;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    font-size: 1.05rem;
    line-height: 1.7;
}
.summary-banner b { color: #f5c842; }
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0e0e0;
}
.cut-line-box {
    background: #fff8e1;
    border-left: 4px solid #f5a623;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.95rem;
    margin-bottom: 16px;
    color: #6d4c00;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_live():
    p = LIVE / "live_predictions_latest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(ttl=1800)
def load_predictions():
    for fname in ("predictions_2026.parquet", "predictions_2026.csv"):
        p = PROCESSED / fname
        if p.exists():
            return pd.read_parquet(p) if fname.endswith(".parquet") else pd.read_csv(p)
    return None


live = load_live()
preds = load_predictions()

if live is None and preds is None:
    st.error("No prediction data found. Run the pipeline first.")
    st.stop()

df = live if live is not None else preds


# ── Flexible column resolution ───────────────────────────────────────────────

def col(preferred, *fallbacks):
    for c in (preferred, *fallbacks):
        if c and c in df.columns:
            return c
    return None


proj_col  = col("mc_projected_total")
p25_col   = col("mc_proj_p25")
p90_col   = col("mc_proj_p90")
win_col   = col("mc_win_prob", "blended_win_prob", "win_prob")
t10_col   = col("mc_top10_prob", "blended_top10_prob", "top10_prob")
score_col = col("cumulative_score_to_par", "current_score")
pos_col   = col("current_pos")
expfin_col = col("mc_expected_finish", "expected_finish")


# ── Build leaderboard dataframe ─────────────────────────────────────────────

lb = df[["player_name"]].copy()

lb["current_score"]  = pd.to_numeric(df[score_col],  errors="coerce") if score_col  else np.nan
lb["position"]       = df[pos_col].fillna("—")                        if pos_col    else "—"
lb["projected_total"] = pd.to_numeric(df[proj_col],  errors="coerce") if proj_col   else np.nan
lb["best_case_score"]  = pd.to_numeric(df[p25_col],  errors="coerce") if p25_col    else np.nan
lb["worst_case_score"] = pd.to_numeric(df[p90_col],  errors="coerce") if p90_col    else np.nan
lb["win_pct"]  = pd.to_numeric(df[win_col],  errors="coerce") if win_col  else np.nan
lb["t10_pct"]  = pd.to_numeric(df[t10_col],  errors="coerce") if t10_col  else np.nan

# Use pre-computed expected finish if present, else compute from MC projected total
if expfin_col:
    lb["expected_finish"] = pd.to_numeric(df[expfin_col], errors="coerce")

lb = lb.dropna(subset=["projected_total"]).copy()
lb = lb.sort_values("projected_total").reset_index(drop=True)

if "expected_finish" not in lb.columns or lb["expected_finish"].isna().all():
    lb["expected_finish"] = lb.index + 1

# Compute finish ranges relative to field medians
medians = lb["projected_total"].values
best_scores  = lb["best_case_score"].fillna(lb["projected_total"]).values
worst_scores = lb["worst_case_score"].fillna(lb["projected_total"]).values

lb["best_case_finish"]  = [int(1 + np.sum(medians < best_scores[i]))  for i in range(len(lb))]
lb["worst_case_finish"] = [int(1 + np.sum(medians < worst_scores[i])) for i in range(len(lb))]


# ── Helper formatters ────────────────────────────────────────────────────────

def fmt_score(val):
    try:
        v = int(float(val))
        return str(v) if v < 0 else (f"+{v}" if v > 0 else "E")
    except Exception:
        return "—"


def fmt_finish(val):
    try:
        n = int(val)
        return "1st" if n == 1 else f"T{n}"
    except Exception:
        return "—"


# ── Context summary banner ───────────────────────────────────────────────────

st.title("Projected Leaderboard")

# Field leader (lowest current score)
has_live_scores = lb["current_score"].notna().any()
if has_live_scores:
    leader_row = lb.dropna(subset=["current_score"]).sort_values("current_score").iloc[0]
    leader_name  = leader_row["player_name"]
    leader_score = fmt_score(leader_row["current_score"])
    field_leader_txt = f"Field leader: <b>{leader_name}</b> ({leader_score})"
else:
    field_leader_txt = "Pre-tournament projections (tournament not yet started)"

# Projected winner (highest win prob)
if lb["win_pct"].notna().any():
    winner_row  = lb.dropna(subset=["win_pct"]).sort_values("win_pct", ascending=False).iloc[0]
    winner_name = winner_row["player_name"]
    winner_win  = winner_row["win_pct"]
    win_display = f"{winner_win * 100:.1f}%" if winner_win <= 1.0 else f"{winner_win:.1f}%"
    proj_winner_txt = f"Projected winner: <b>{winner_name}</b> ({win_display} MC win prob)"
else:
    proj_winner_txt = ""

# Cut line estimate: ~top 50 + ties project to make cut
cut_idx = min(50, len(lb) - 1)
cut_score = lb.iloc[cut_idx]["projected_total"] if len(lb) > cut_idx else None

banner_parts = [p for p in [field_leader_txt, proj_winner_txt] if p]
st.markdown(
    f'<div class="summary-banner">'
    + " &nbsp;|&nbsp; ".join(banner_parts)
    + "</div>",
    unsafe_allow_html=True,
)

if cut_score is not None:
    st.markdown(
        f'<div class="cut-line-box">Projected cut line: <b>{fmt_score(cut_score)}</b> '
        f'(~50th projected finish)</div>',
        unsafe_allow_html=True,
    )


# ── Sortable table ───────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Full Field Projections</div>', unsafe_allow_html=True)

sort_options = {
    "Expected Finish":        ("expected_finish",   True),
    "Current Score":          ("current_score",     True),
    "Projected Final Score":  ("projected_total",   True),
    "Win %":                  ("win_pct",           False),
    "Top-10 %":               ("t10_pct",           False),
    "Best Case Finish":       ("best_case_finish",  True),
    "Worst Case Finish":      ("worst_case_finish", True),
}

sort_label = st.selectbox("Sort by", list(sort_options.keys()), index=0)
sort_by, sort_asc = sort_options[sort_label]

display = lb.sort_values(sort_by, ascending=sort_asc).reset_index(drop=True).copy()

# Scale win probs to percentage
win_pct_display = display["win_pct"] * 100 if display["win_pct"].max(skipna=True) <= 1.0 else display["win_pct"]
t10_pct_display = display["t10_pct"] * 100 if display["t10_pct"].max(skipna=True) <= 1.0 else display["t10_pct"]

table = pd.DataFrame({
    "Player":           display["player_name"].values,
    "Pos":              display["position"].values,
    "Score":            display["current_score"].values.astype(float),
    "Projected":        display["projected_total"].values.astype(float),
    "Expected Finish":  display["expected_finish"].values.astype(float),
    "Best Case":        display["best_case_finish"].values.astype(float),
    "Worst Case":       display["worst_case_finish"].values.astype(float),
    "Win %":            win_pct_display.values.astype(float),
    "T10 %":            t10_pct_display.values.astype(float),
})

col_cfg = {
    "Score": st.column_config.NumberColumn(
        "Score", help="Current cumulative score to par", format="%+.0f"
    ),
    "Projected": st.column_config.NumberColumn(
        "Projected", help="Expected final total score to par (MC mean)", format="%+.0f"
    ),
    "Expected Finish": st.column_config.NumberColumn(
        "Expected Finish", help="Median finish position from Monte Carlo", format="%.0f"
    ),
    "Best Case": st.column_config.NumberColumn(
        "Best Case", help="25th percentile finish (optimistic)", format="%.0f"
    ),
    "Worst Case": st.column_config.NumberColumn(
        "Worst Case", help="90th percentile finish (pessimistic)", format="%.0f"
    ),
    "Win %": st.column_config.NumberColumn(
        "Win %", help="Win probability from Monte Carlo", format="%.1f%%"
    ),
    "T10 %": st.column_config.NumberColumn(
        "T10 %", help="Top-10 probability from Monte Carlo", format="%.1f%%"
    ),
}

st.dataframe(
    table,
    use_container_width=True,
    column_config=col_cfg,
    height=540,
    hide_index=True,
)


# ── Dumbbell / range chart ───────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-header">Finish Range Chart</div>', unsafe_allow_html=True)
st.caption(
    "Thin line = Best Case (P25 score) to Worst Case (P90 score).  "
    "Bold dot = Expected Finish.  "
    "Darker color = higher win probability."
)

n_players = st.slider(
    "Players to show", min_value=10, max_value=min(80, len(lb)), value=min(30, len(lb)), step=5
)
chart_df = lb.head(n_players).copy()

# Win-prob color scale (dark green → light gray)
raw_win = chart_df["win_pct"].fillna(0)
max_win = raw_win.max() if raw_win.max() > 0 else 1.0

def win_color(wp, alpha_line=0.75, alpha_dot=1.0):
    frac = float(np.clip(wp / max_win, 0, 1))
    # Dark green (#1a5c1a) for top, steel blue (#4a7ab5) for mid, silver (#bbb) for low
    if frac > 0.5:
        t = (frac - 0.5) * 2
        r = int(74  + t * (26  - 74))
        g = int(122 + t * (92  - 122))
        b = int(181 + t * (26  - 181))
    else:
        t = frac * 2
        r = int(187 + t * (74  - 187))
        g = int(187 + t * (122 - 187))
        b = int(187 + t * (181 - 187))
    return f"rgba({r},{g},{b},{alpha_line})", f"rgba({r},{g},{b},{alpha_dot})"

y_labels       = chart_df["player_name"].tolist()
best_finishes  = chart_df["best_case_finish"].tolist()
worst_finishes = chart_df["worst_case_finish"].tolist()
exp_finishes   = chart_df["expected_finish"].tolist()
win_probs      = raw_win.tolist()

fig = go.Figure()

# Range lines (best → worst case)
for player, best, worst, exp, wp in zip(
    y_labels, best_finishes, worst_finishes, exp_finishes, win_probs
):
    lc, dc = win_color(wp)
    hover = (
        f"<b>{player}</b><br>"
        f"Best case: {fmt_finish(best)}<br>"
        f"Expected: {fmt_finish(exp)}<br>"
        f"Worst case: {fmt_finish(worst)}<br>"
        f"Win%: {wp * 100:.1f}%<extra></extra>"
    )
    # Thin background line
    fig.add_trace(go.Scatter(
        x=[best, worst],
        y=[player, player],
        mode="lines",
        line=dict(color=lc, width=4),
        showlegend=False,
        hovertemplate=hover,
    ))

# Expected finish dots
dot_colors = [win_color(wp)[1] for wp in win_probs]
fig.add_trace(go.Scatter(
    x=exp_finishes,
    y=y_labels,
    mode="markers",
    marker=dict(
        color=dot_colors,
        size=11,
        symbol="circle",
        line=dict(color="white", width=2),
    ),
    name="Expected Finish",
    hovertemplate="<b>%{y}</b><br>Expected: %{x:.0f}<extra></extra>",
))

# Reference lines at key positions
ref_lines = {1: ("gold", "1st"), 5: ("#f5a623", "T5"), 10: ("#4caf50", "T10"), 20: ("#888", "T20")}
for pos, (color, label) in ref_lines.items():
    if pos <= max(worst_finishes):
        fig.add_vline(
            x=pos,
            line_dash="dot",
            line_color=color,
            opacity=0.6,
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=13,
            annotation_font_color=color,
        )

x_max = min(max(worst_finishes) + 3, len(lb) + 5)
chart_height = max(420, n_players * 26)

fig.update_layout(
    template="plotly_white",
    height=chart_height,
    xaxis=dict(
        title=dict(text="Finish Position (lower = better)", font=dict(size=14)),
        range=[x_max, 0],   # reversed: 1 on left
        dtick=5,
        tickfont=dict(size=13),
        gridcolor="#e8e8e8",
    ),
    yaxis=dict(
        title="",
        categoryorder="array",
        categoryarray=y_labels[::-1],  # best at top
        tickfont=dict(size=13),
        automargin=True,
    ),
    margin=dict(l=10, r=20, t=40, b=60),
    showlegend=True,
    legend=dict(
        yanchor="bottom", y=0.01,
        xanchor="right",  x=0.99,
        font=dict(size=13),
    ),
    hovermode="closest",
    font=dict(size=14),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

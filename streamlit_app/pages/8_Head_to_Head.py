import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
from pathlib import Path

st.set_page_config(page_title="Head to Head — Augusta Model", layout="wide", page_icon="⛳")

ROOT = Path(__file__).parent.parent.parent
LIVE = ROOT / "data" / "live"
PROCESSED = ROOT / "data" / "processed"


def norm_cdf(x):
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@st.cache_data(ttl=300)
def load_live():
    p = LIVE / "live_predictions_latest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(ttl=3600)
def load_hole_by_hole():
    p = PROCESSED / "masters_hole_by_hole.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=3600)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = PROCESSED / "predictions_2026.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


live = load_live()
hbh = load_hole_by_hole()
preds = load_predictions()

if live is None and preds is None:
    st.error("No prediction data found. Run the pipeline.")
    st.stop()

# Use live data if available, fall back to pre-tournament predictions
df = live if live is not None else preds

players = sorted(df["player_name"].dropna().tolist())

# ── Player selectors ────────────────────────────────────────────────
st.title("Head-to-Head Comparison")
st.caption("Live MC probabilities, Augusta historical scoring, and current season form.")

c1, c2 = st.columns(2)
with c1:
    player_a = st.selectbox("Player A", players, index=0)
with c2:
    default_b = next((i for i, p in enumerate(players) if p != player_a), 1)
    player_b = st.selectbox("Player B", players, index=default_b)

if player_a == player_b:
    st.warning("Select two different players.")
    st.stop()

a = df[df["player_name"] == player_a].iloc[0]
b = df[df["player_name"] == player_b].iloc[0]

# ── Helper functions ────────────────────────────────────────────────

def fmt_score(val):
    try:
        v = int(float(val))
        if v < 0:
            return str(v)
        elif v > 0:
            return f"+{v}"
        else:
            return "E"
    except Exception:
        return "—"


def fmt_pct(val):
    try:
        return f"{float(val):.1%}"
    except Exception:
        return "—"


def fmt_num(val, decimals=2):
    try:
        return f"{float(val):+.{decimals}f}"
    except Exception:
        return "—"


def parse_finish(pos_str):
    if pd.isna(pos_str):
        return np.nan
    s = str(pos_str).upper().strip()
    if s in {"MC", "CUT", "WD", "DQ", ""}:
        return np.nan
    s = s.lstrip("T")
    try:
        return int(float(s))
    except Exception:
        return np.nan


def get_masters_history(player_name, hbh_df):
    if hbh_df is None or hbh_df.empty:
        return {"times_played": "—", "avg_finish": "—", "best_finish": "—"}
    pdata = hbh_df[hbh_df["player_name"] == player_name]
    if pdata.empty:
        return {"times_played": 0, "avg_finish": "—", "best_finish": "—"}
    yearly = pdata.drop_duplicates("year")[["year", "finish_pos"]]
    times = len(yearly)
    finishes = yearly["finish_pos"].apply(parse_finish).dropna()
    avg_f = f"{finishes.mean():.1f}" if len(finishes) > 0 else "—"
    best_v = int(finishes.min()) if len(finishes) > 0 else None
    best_f = f"T{best_v}" if best_v is not None else "—"
    return {"times_played": times, "avg_finish": avg_f, "best_finish": best_f}


hist_a = get_masters_history(player_a, hbh)
hist_b = get_masters_history(player_b, hbh)

# ── Comparison table ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Side-by-Side Comparison")

rows = [
    # Section: Tournament (live)
    ("**TOURNAMENT**", "", ""),
    ("Current Score", fmt_score(a.get("cumulative_score_to_par", a.get("current_score"))),
                      fmt_score(b.get("cumulative_score_to_par", b.get("current_score")))),
    ("Position", str(a.get("current_pos", "—")), str(b.get("current_pos", "—"))),
    ("Thru (holes)", str(a.get("thru", a.get("holes_completed", "—"))),
                    str(b.get("thru", b.get("holes_completed", "—")))),
    ("MC Win %", fmt_pct(a.get("mc_win_prob")), fmt_pct(b.get("mc_win_prob"))),
    ("MC Top-5 %", fmt_pct(a.get("mc_top5_prob")), fmt_pct(b.get("mc_top5_prob"))),
    ("MC Top-10 %", fmt_pct(a.get("mc_top10_prob")), fmt_pct(b.get("mc_top10_prob"))),
    # Section: Augusta history
    ("**HISTORICAL MASTERS**", "", ""),
    ("Times at Augusta", str(hist_a["times_played"]), str(hist_b["times_played"])),
    ("Avg Finish (Augusta)", hist_a["avg_finish"], hist_b["avg_finish"]),
    ("Best Finish (Augusta)", hist_a["best_finish"], hist_b["best_finish"]),
    # Section: Season form
    ("**SEASON FORM**", "", ""),
    ("SG Total", fmt_num(a.get("sg_total")), fmt_num(b.get("sg_total"))),
    ("SG Off-Tee", fmt_num(a.get("sg_ott")), fmt_num(b.get("sg_ott"))),
    ("SG Approach", fmt_num(a.get("sg_app")), fmt_num(b.get("sg_app"))),
    ("SG Around Green", fmt_num(a.get("sg_arg")), fmt_num(b.get("sg_arg"))),
    ("SG Putting", fmt_num(a.get("sg_putt")), fmt_num(b.get("sg_putt"))),
]

comp_df = pd.DataFrame(rows, columns=["Metric", player_a, player_b])
st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ── H2H probability: who finishes ahead this tournament ────────────
st.markdown("---")
st.subheader("Tournament Edge")

mu_a = float(a.get("mc_projected_total", 0) or 0)
mu_b = float(b.get("mc_projected_total", 0) or 0)

def iqr_to_sigma(row):
    try:
        p25 = float(row.get("mc_proj_p25") or np.nan)
        p75 = float(row.get("mc_proj_p75") or np.nan)
        if not (np.isnan(p25) or np.isnan(p75)):
            return max((p75 - p25) / 1.349, 1.0)
    except Exception:
        pass
    return 4.0  # fallback: typical tournament scoring std

sigma_a = iqr_to_sigma(a)
sigma_b = iqr_to_sigma(b)
diff_mean = mu_a - mu_b  # positive → A projected worse (higher score)
diff_std = math.sqrt(sigma_a ** 2 + sigma_b ** 2)
if diff_std > 0:
    h2h_a_prob = norm_cdf(-diff_mean / diff_std)
else:
    h2h_a_prob = 0.5
h2h_b_prob = 1.0 - h2h_a_prob

if h2h_a_prob >= h2h_b_prob:
    edge_leader, edge_trailer = player_a, player_b
    edge_prob = h2h_a_prob
else:
    edge_leader, edge_trailer = player_b, player_a
    edge_prob = h2h_b_prob

st.info(
    f"Model gives **{edge_leader}** a **{edge_prob:.0%}** chance of finishing "
    f"ahead of **{edge_trailer}** in this tournament."
)

# ── Round winner probability ────────────────────────────────────────
sg_a = float(a.get("sg_total", 0) or 0)
sg_b = float(b.get("sg_total", 0) or 0)

round_std = 3.5  # typical PGA Tour round-to-round std
sg_diff = sg_a - sg_b  # positive → A gains more strokes (lower score)
if abs(round_std * math.sqrt(2)) > 0:
    round_h2h_a = norm_cdf(sg_diff / (round_std * math.sqrt(2)))
else:
    round_h2h_a = 0.5
round_h2h_b = 1.0 - round_h2h_a

if round_h2h_a >= round_h2h_b:
    rnd_leader, rnd_trailer = player_a, player_b
    rnd_prob = round_h2h_a
else:
    rnd_leader, rnd_trailer = player_b, player_a
    rnd_prob = round_h2h_b

st.info(
    f"For tomorrow's round, model expects **{rnd_leader}** to outscore "
    f"**{rnd_trailer}** **{rnd_prob:.0%}** of the time — based on current SG form."
)

# ── Hole-by-hole overlay chart ──────────────────────────────────────
st.markdown("---")
st.subheader("Hole-by-Hole: Historical Scoring at Augusta")

if hbh is None or hbh.empty:
    st.info("Hole-by-hole data not available.")
else:
    # Get hole metadata (par, name, yards) — pick most common values per hole
    hole_meta = (
        hbh.groupby("hole_number")
        .agg(par=("par", "median"), hole_name=("hole_name", "first"), yards=("yards", "median"))
        .reset_index()
        .sort_values("hole_number")
    )
    hole_labels = [
        f"{int(r.hole_number)}: {r.hole_name}" for _, r in hole_meta.iterrows()
    ]
    par_vals = hole_meta["par"].values

    def player_hole_avg(player_name, hbh_df):
        pdata = hbh_df[hbh_df["player_name"] == player_name]
        if pdata.empty:
            return None, 0
        rounds = pdata.groupby(["year", "round"])["score_to_par"].count().shape[0]
        avg = (
            pdata.groupby("hole_number")["score_to_par"]
            .mean()
            .reindex(range(1, 19))
        )
        return avg, pdata["year"].nunique()

    avg_a, years_a = player_hole_avg(player_a, hbh)
    avg_b, years_b = player_hole_avg(player_b, hbh)

    if avg_a is None and avg_b is None:
        st.info(f"No Augusta hole-by-hole data found for either player.")
    else:
        fig = go.Figure()

        # Par baseline
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

        color_a = "#1f77b4"
        color_b = "#d62728"

        if avg_a is not None:
            fig.add_trace(go.Scatter(
                x=hole_labels,
                y=avg_a.values,
                mode="lines+markers",
                name=f"{player_a} ({years_a} yr{'s' if years_a != 1 else ''})",
                line=dict(color=color_a, width=2.5),
                marker=dict(size=7),
                hovertemplate="%{x}<br>Avg: %{y:+.2f}<extra></extra>",
            ))

        if avg_b is not None:
            fig.add_trace(go.Scatter(
                x=hole_labels,
                y=avg_b.values,
                mode="lines+markers",
                name=f"{player_b} ({years_b} yr{'s' if years_b != 1 else ''})",
                line=dict(color=color_b, width=2.5),
                marker=dict(size=7),
                hovertemplate="%{x}<br>Avg: %{y:+.2f}<extra></extra>",
            ))

        # Shade regions where A has an advantage
        if avg_a is not None and avg_b is not None:
            diff = avg_a - avg_b  # negative → A scores lower (better) on that hole
            for i in range(18):
                if not np.isnan(diff.iloc[i]):
                    fill_color = (
                        "rgba(31,119,180,0.10)" if diff.iloc[i] < 0
                        else "rgba(214,39,40,0.10)"
                    )
                    x0 = hole_labels[max(i - 1, 0)]
                    x1 = hole_labels[min(i + 1, 17)]
                    # Use scatter fill between traces for per-hole shading
                    pass  # kept simple — colour-coded lines are clear enough

        fig.update_layout(
            title="Average Score to Par per Hole (lower = better)",
            xaxis_title="Hole",
            yaxis_title="Avg Score to Par",
            xaxis=dict(tickangle=-45),
            height=460,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified",
        )
        fig.update_yaxes(tickformat="+.2f")
        st.plotly_chart(fig, use_container_width=True)

        # Hole advantage summary
        if avg_a is not None and avg_b is not None:
            diff = avg_a - avg_b
            a_better = hole_meta.loc[diff.values < -0.05, "hole_name"].tolist()
            b_better = hole_meta.loc[diff.values > 0.05, "hole_name"].tolist()
            col1, col2 = st.columns(2)
            with col1:
                if a_better:
                    st.markdown(
                        f"**{player_a} advantage holes:** "
                        + ", ".join(a_better[:6])
                        + ("..." if len(a_better) > 6 else "")
                    )
            with col2:
                if b_better:
                    st.markdown(
                        f"**{player_b} advantage holes:** "
                        + ", ".join(b_better[:6])
                        + ("..." if len(b_better) > 6 else "")
                    )

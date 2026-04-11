import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Augusta Model — 2026 Masters", layout="wide", page_icon="⛳")

PROCESSED = Path(__file__).parent.parent / "data" / "processed"

@st.cache_data(ttl=1800)
def load_predictions():
    p = PROCESSED / "predictions_2026.parquet"
    if not p.exists():
        p = PROCESSED / "predictions_2026.csv"
    if not p.exists():
        return None
    return pd.read_parquet(PROCESSED / "predictions_2026.parquet") if (PROCESSED / "predictions_2026.parquet").exists() else pd.read_csv(p)

@st.cache_data(ttl=1800)
def load_edge():
    p = PROCESSED / "edge_2026.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

preds = load_predictions()
edge = load_edge()

if preds is None:
    st.error("Predictions not found. Run: `python run_production.py`")
    st.stop()

st.sidebar.title("⛳ Augusta Model")
st.sidebar.markdown("**2026 Masters — April 9-12**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%b %d, %H:%M')}")
st.sidebar.markdown(f"Field size: {len(preds)}")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/1_Model_Picks.py", label="Model Picks")
st.sidebar.page_link("pages/2_Betting_Edge.py", label="Betting Edge")
st.sidebar.page_link("pages/3_H2H_Matchups.py", label="H2H Matchups")
st.sidebar.page_link("pages/4_Backtest.py", label="Backtest")
st.sidebar.page_link("pages/5_Best_Bets.py", label="Best Bets")
st.sidebar.page_link("pages/6_Live_Tournament.py", label="Live Tournament")
st.sidebar.page_link("pages/7_Player_Profiles.py", label="Player Profiles")
st.sidebar.page_link("pages/8_Head_to_Head.py", label="Head to Head")
st.sidebar.page_link("pages/9_Projected_Leaderboard.py", label="Projected Leaderboard")

st.title("⛳ 2026 Masters — Augusta National Model")
st.markdown("Two-stage XGBoost + Monte Carlo model with Augusta-specific experience features. "
            "Backtested 2021-2025: **34% top-10 precision**, Cal AUC **0.626**.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Field Size", len(preds))
debutants = (preds.get("augusta_experience_tier", pd.Series(dtype=float)) == 0).sum()
col2.metric("Debutants", debutants)
top1 = preds.iloc[0]
t10_col = "top10_prob" if "top10_prob" in preds.columns else "top10_prob_calibrated"
col3.metric("Model #1 (Top-10)", top1["player_name"], f"{top1[t10_col]:.1%}")

# Odds favourite — handle different edge CSV column names
if edge is not None and len(edge) > 0:
    mkt_col = next((c for c in ["market_win_pct", "dk_fair_win_pct", "dk_implied_prob"] if c in edge.columns), None)
    if mkt_col:
        fav = edge.sort_values(mkt_col, ascending=False).iloc[0]
        col4.metric("Odds Favourite", fav["player_name"], f"{fav[mkt_col]:.1%} mkt")

st.markdown("---")
st.subheader("Quick Look — Top 10")
base_cols = ["player_name", "win_prob", t10_col, "top20_prob"]
if "make_cut_prob" in preds.columns:
    base_cols.append("make_cut_prob")
for c in ["augusta_experience_tier", "augusta_made_cut_prev_year", "augusta_scoring_trajectory"]:
    if c in preds.columns:
        base_cols.append(c)
top10 = preds.head(10)[[c for c in base_cols if c in preds.columns]].copy()

col_names = ["Player", "Win %", "Top-10 %", "Top-20 %"]
if "make_cut_prob" in top10.columns:
    col_names.append("Cut %")
if "augusta_experience_tier" in top10.columns:
    col_names.append("Exp Tier")
if "augusta_made_cut_prev_year" in top10.columns:
    col_names.append("Cut Prev Yr")
if "augusta_scoring_trajectory" in top10.columns:
    col_names.append("Trajectory")
top10.columns = col_names

for col in ["Win %", "Top-10 %", "Top-20 %", "Cut %"]:
    if col in top10.columns:
        top10[col] = top10[col] * 100
top10.index = range(1, len(top10) + 1)
_pct_cols = {c: st.column_config.NumberColumn(c, format="%.1f%%")
             for c in ["Win %", "Top-10 %", "Top-20 %", "Cut %"] if c in top10.columns}
st.dataframe(top10, width="stretch", column_config=_pct_cols)
